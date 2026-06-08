import os
import time
import logging
import threading
import concurrent.futures
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

API_BASE = "https://api.openaq.org/v3"
API_KEY = os.environ.get("OPENAQ_API_KEY")

COUNTRY_FILTERS = {
    "SG": {
        "cities": {"singapore"},
        "providers": {"airgradient"},
        "name_keywords": {"singapore", "sg"},
    },
    "TH": {
        "cities": {"bangkok"},
        "providers": {"airgradient", "air4thai"},
        "name_keywords": {"bangkok", "bkk", "thailand"},
    },
}

PARAMETER_MAP = {
    "pm25": "pm25",
    "pm10": "pm10",
    "pm1": "pm1",
    "temperature": "temperature_c",
    "humidity": "humidity_pct",
    "co2": "co2",
    "tvoc": "tvoc",
}


def normalize_header(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def pick_column(
    columns: list[str],
    must_contain: list[str],
    prefer: list[str] | None = None,
    exact_tokens: bool = False,
) -> str | None:
    normalized = {col: normalize_header(col) for col in columns}
    if exact_tokens:
        target = "".join(must_contain)
        matches = [col for col, norm in normalized.items() if norm == target]
    else:
        matches = [
            col for col, norm in normalized.items()
            if all(token in norm for token in must_contain)
        ]
    if not matches:
        return None
    if prefer:
        preferred = [
            col for col in matches
            if all(token in normalize_header(col) for token in prefer)
        ]
        if preferred:
            return preferred[0]
    return matches[0]


def resolve_pg_dsn() -> str:
    dsn = os.environ.get("PG_DSN")
    if dsn:
        return dsn
    host = os.environ.get("PG_HOST")
    port = os.environ.get("PG_PORT", "5432")
    dbname = os.environ.get("PG_DB")
    user = os.environ.get("PG_USER")
    password = os.environ.get("PG_PASSWORD")
    if not all([host, dbname, user, password]):
        raise ValueError("Missing PG_DSN or PG_HOST/PG_DB/PG_USER/PG_PASSWORD env vars.")
    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"


def get_session() -> requests.Session:
    if not API_KEY:
        raise ValueError("OPENAQ_API_KEY is required in the environment.")
    session = requests.Session()
    session.headers.update({"X-API-Key": API_KEY})
    return session


# One Session per thread — requests.Session is not safe for concurrent use across threads
_thread_local = threading.local()


def _get_thread_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        _thread_local.session = get_session()
    return _thread_local.session


def request_json(session: requests.Session, url: str, params: dict | None = None) -> dict:
    retryable = {408, 500, 502, 503, 504}
    for attempt in range(6):
        try:
            resp = session.get(url, params=params, timeout=90)
        except requests.exceptions.Timeout:
            wait = min(2 ** attempt, 60)
            logger.warning("Request timed out (attempt %s), sleeping %ss: %s", attempt + 1, wait, url)
            time.sleep(wait)
            continue
        except requests.exceptions.ConnectionError as exc:
            wait = min(2 ** attempt, 60)
            logger.warning("Connection error (attempt %s), sleeping %ss: %s", attempt + 1, wait, exc)
            time.sleep(wait)
            continue

        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", min(2 ** attempt, 60)))
            logger.warning(
                "Rate limited (attempt %s), sleeping %ss: %s",
                attempt + 1, retry_after, url,
            )
            time.sleep(retry_after)
            continue

        if resp.status_code in retryable:
            wait = min(2 ** attempt, 60)
            logger.warning("HTTP %s (attempt %s), sleeping %ss: %s", resp.status_code, attempt + 1, wait, url)
            time.sleep(wait)
            continue

        resp.raise_for_status()
        time.sleep(0.1)  # reduced from 0.3s — Retry-After handles actual rate limits
        return resp.json()

    raise RuntimeError(f"OpenAQ request failed after retries for {url}")


# ---------------------------------------------------------------------------
# Watermarks
# ---------------------------------------------------------------------------

def fetch_watermarks(conn) -> dict[str, datetime]:
    """Return the max ingested timestamp per location_key so we only fetch new data."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT location_key, MAX(timestamp_utc) FROM openaq.measurements GROUP BY location_key"
        )
        return {row[0]: row[1] for row in cur.fetchall()}


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------

def _fetch_country_locations(country: str, filters: dict) -> list[dict]:
    """Fetch all location pages for one country using a thread-local session."""
    session = _get_thread_session()
    locations: list[dict] = []
    page = 1
    while True:
        params = {"country": country, "isMonitor": "true", "limit": 100, "page": page}
        try:
            data = request_json(session, f"{API_BASE}/locations", params=params)
        except RuntimeError as exc:
            logger.warning("OpenAQ locations failed for %s page %s: %s", country, page, exc)
            break
        results = data.get("results", [])
        if not results:
            break
        for loc in results:
            provider = ((loc.get("provider") or {}).get("name") or "").lower()
            if provider not in filters["providers"]:
                continue
            locality_raw = loc.get("locality") or loc.get("city") or ""
            locality = locality_raw.lower()
            if locality and locality not in filters["cities"]:
                continue
            if not locality:
                name = (loc.get("name") or "").lower()
                keywords = filters.get("name_keywords", set())
                if keywords and not any(kw in name for kw in keywords):
                    continue
            locations.append(loc)
            logger.info(
                "Accepted location id=%s name=%r locality=%r provider=%r country=%s",
                loc.get("id"), loc.get("name"), locality or "(no locality)", provider, country,
            )
        page += 1
    return locations


def fetch_locations() -> list[dict]:
    """Fetch locations for all countries in parallel."""
    all_locations: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(COUNTRY_FILTERS)) as executor:
        futures = {
            executor.submit(_fetch_country_locations, country, filters): country
            for country, filters in COUNTRY_FILTERS.items()
        }
        for future in concurrent.futures.as_completed(futures):
            country = futures[future]
            try:
                locs = future.result()
                all_locations.extend(locs)
                logger.info("Country %s: %d locations accepted.", country, len(locs))
            except Exception as exc:
                logger.warning("Country %s location fetch failed: %s", country, exc)
    logger.info("Total accepted locations: %d", len(all_locations))
    return all_locations


def build_openaq_locations(locations: list[dict]) -> pd.DataFrame:
    rows = []
    for loc in locations:
        loc_id = loc.get("id")
        provider = (loc.get("provider") or {}).get("name")
        owner = (loc.get("owner") or {}).get("name")
        coords = loc.get("coordinates") or {}
        rows.append({
            "location_key": f"openaq:{loc_id}",
            "source": "openaq",
            "external_id": str(loc_id),
            "name": loc.get("name"),
            "locality": loc.get("locality"),
            "country_iso": (loc.get("country") or {}).get("code"),
            "owner_name": owner,
            "provider_name": provider,
            "is_monitor": loc.get("isMonitor"),
            "is_mobile": loc.get("isMobile"),
            "latitude": coords.get("latitude"),
            "longitude": coords.get("longitude"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sensors & Measurements
# ---------------------------------------------------------------------------

def fetch_sensors_for_location(session: requests.Session, location_id: int) -> list[dict]:
    try:
        data = request_json(session, f"{API_BASE}/locations/{location_id}/sensors")
        return data.get("results", [])
    except RuntimeError as exc:
        logger.warning("Failed to fetch sensors for location %s: %s", location_id, exc)
        return []


def fetch_sensor_measurements(
    session: requests.Session, sensor_id: int, start: datetime, end: datetime
) -> list[dict]:
    rows: list[dict] = []
    chunk_days = 90
    chunk_start = start

    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=chunk_days), end)
        page = 1
        params_base = {
            "date_from": chunk_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "date_to": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": 1000,
        }
        while True:
            params = {**params_base, "page": page}
            try:
                data = request_json(session, f"{API_BASE}/sensors/{sensor_id}/measurements", params=params)
            except RuntimeError as exc:
                logger.warning(
                    "Skipping sensor %s chunk %s-%s page %s after retries: %s",
                    sensor_id, chunk_start.date(), chunk_end.date(), page, exc,
                )
                break
            results = data.get("results", [])
            if not results:
                break
            rows.extend(results)
            page += 1
        chunk_start = chunk_end

    return rows


def normalize_openaq_measurements(rows: list[dict], location_id: int) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    records = []
    for r in rows:
        try:
            ts = r["period"]["datetimeTo"]["utc"]
            param_name = r["parameter"]["name"].lower()
            value = r["value"]
            records.append({
                "locationId": r.get("locationId", location_id),
                "datetimeUtc": ts,
                "parameter": param_name,
                "value": value,
            })
        except (KeyError, TypeError) as exc:
            logger.debug("Skipping malformed measurement row: %s - %s", r, exc)
            continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["datetimeUtc"] = pd.to_datetime(df["datetimeUtc"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetimeUtc", "value"])
    df["parameter"] = df["parameter"].map(PARAMETER_MAP)
    df = df.dropna(subset=["parameter"])

    pivot = df.pivot_table(
        index=["locationId", "datetimeUtc"],
        columns="parameter",
        values="value",
        aggfunc="mean",
    ).reset_index()

    pivot = pivot.rename(columns={"locationId": "location_id", "datetimeUtc": "timestamp_utc"})
    for col in ["pm25", "temperature_c", "humidity_pct", "pm1", "pm10", "co2", "tvoc"]:
        if col not in pivot.columns:
            pivot[col] = pd.NA

    pivot["has_pm1"] = pivot["pm1"].notna()
    pivot["has_pm10"] = pivot["pm10"].notna()
    pivot["has_co2"] = pivot["co2"].notna()
    pivot["has_tvoc"] = pivot["tvoc"].notna()

    return pivot


def _fetch_one_sensor(
    sensor_id: int, param_name: str, loc_id: int, start: datetime, end: datetime
) -> pd.DataFrame | None:
    """Worker: fetch + normalize one sensor's measurements. Uses a thread-local session."""
    session = _get_thread_session()
    logger.info("Fetching sensor %s (%s) for location %s", sensor_id, param_name, loc_id)
    rows = fetch_sensor_measurements(session, sensor_id, start, end)
    if not rows:
        return None
    df = normalize_openaq_measurements(rows, loc_id)
    if df.empty:
        return None
    df["location_key"] = df["location_id"].apply(lambda v: f"openaq:{v}")
    return df


def build_openaq_measurements(
    locations: list[dict],
    watermarks: dict[str, datetime],
    fallback_start: datetime,
    end: datetime,
    conn,
    max_workers: int = 8,
) -> None:
    """Fetch measurements for all sensors concurrently and stream-write to DB.

    Phase 1 (serial): collect sensor jobs — sensor-list calls are fast (~1 req/location).
    Phase 2 (concurrent): fetch measurements in parallel; write to DB from the main thread.
    """
    # Phase 1: discover sensors and build the job list
    session = get_session()
    jobs: list[tuple[int, str, int, datetime]] = []
    for loc in locations:
        loc_id = loc.get("id")
        if not loc_id:
            continue
        location_key = f"openaq:{loc_id}"
        start = watermarks.get(location_key, fallback_start)
        if start >= end:
            logger.info("Location %s is up-to-date, skipping.", location_key)
            continue
        sensors = fetch_sensors_for_location(session, loc_id)
        if not sensors:
            logger.warning("No sensors found for location %s", loc_id)
            continue
        for sensor in sensors:
            sensor_id = sensor.get("id")
            if not sensor_id:
                continue
            param_name = (sensor.get("parameter") or {}).get("name", "").lower()
            if param_name not in PARAMETER_MAP:
                logger.info(
                    "Skipping sensor %s (%s) for location %s — not in PARAMETER_MAP",
                    sensor_id, param_name, loc_id,
                )
                continue
            jobs.append((sensor_id, param_name, loc_id, start))

    logger.info("Submitting %d sensor fetch jobs (max_workers=%d).", len(jobs), max_workers)

    # Phase 2: concurrent fetch + stream writes to DB from the main thread
    # Worker threads only return DataFrames; DB writes stay on the main thread
    # because psycopg2 connections are not thread-safe.
    total_rows = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_one_sensor, sid, pname, lid, s, end): (sid, pname, lid)
            for sid, pname, lid, s in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            sid, pname, lid = futures[future]
            try:
                df = future.result()
                if df is not None:
                    upsert_measurements(conn, df, "openaq")
                    total_rows += len(df)
                    logger.info("Wrote %d rows for sensor %s (%s).", len(df), sid, pname)
            except Exception as exc:
                logger.warning("Sensor %s (%s) loc %s failed: %s", sid, pname, lid, exc)

    logger.info("build_openaq_measurements: total %d rows committed.", total_rows)


# ---------------------------------------------------------------------------
# Manila CSV
# ---------------------------------------------------------------------------

def normalize_manila_csv(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path, encoding="utf-8", encoding_errors="replace")
    columns = df.columns.tolist()

    ts_col = pick_column(columns, ["utc", "datetime"]) or pick_column(columns, ["utc", "date"])
    if not ts_col:
        raise ValueError(f"UTC timestamp column not found in {csv_path}.")

    col_pm25 = pick_column(columns, ["pm25"], ["corr"]) or pick_column(columns, ["pm25"])
    col_temp = pick_column(columns, ["temperature"], ["corr"]) or pick_column(columns, ["temperature"])
    col_humidity = pick_column(columns, ["humidity"], ["corr"]) or pick_column(columns, ["humidity"])
    col_pm1 = pick_column(columns, ["pm1"], exact_tokens=True)
    col_pm10 = pick_column(columns, ["pm10"], exact_tokens=True)
    col_co2 = pick_column(columns, ["co2"], ["corr"]) or pick_column(columns, ["co2"])
    col_tvoc = pick_column(columns, ["tvoc"], ["ppb"]) or pick_column(columns, ["tvoc"])

    location_id_col = pick_column(columns, ["location", "id"]) or pick_column(columns, ["sensor", "id"])
    location_name_col = pick_column(columns, ["location", "name"]) or pick_column(columns, ["location", "n"])

    measurements = pd.DataFrame()
    measurements["timestamp_utc"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    measurements["pm25"] = df[col_pm25] if col_pm25 else pd.NA
    measurements["temperature_c"] = df[col_temp] if col_temp else pd.NA
    measurements["humidity_pct"] = df[col_humidity] if col_humidity else pd.NA
    measurements["pm1"] = df[col_pm1] if col_pm1 else pd.NA
    measurements["pm10"] = df[col_pm10] if col_pm10 else pd.NA
    measurements["co2"] = df[col_co2] if col_co2 else pd.NA
    measurements["tvoc"] = df[col_tvoc] if col_tvoc else pd.NA

    measurements = measurements.dropna(subset=["timestamp_utc"]).copy()
    measurements["has_pm1"] = measurements["pm1"].notna()
    measurements["has_pm10"] = measurements["pm10"].notna()
    measurements["has_co2"] = measurements["co2"].notna()
    measurements["has_tvoc"] = measurements["tvoc"].notna()

    if location_id_col and location_id_col in df.columns:
        location_keys = df[location_id_col].astype(str).fillna("unknown").tolist()
    elif location_name_col and location_name_col in df.columns:
        location_keys = df[location_name_col].astype(str).fillna("unknown").tolist()
    else:
        location_keys = ["unknown"] * len(df)

    measurements = measurements.reset_index(drop=True)
    measurements["location_key"] = [f"manila:{k}" for k in location_keys[: len(measurements)]]

    locations = pd.DataFrame({
        "location_key": measurements["location_key"].unique(),
        "source": "manila",
        "external_id": None,
        "name": None,
        "locality": "Metro Manila",
        "country_iso": "PH",
        "owner_name": "AirGradient",
        "provider_name": "AirGradient",
        "is_monitor": True,
        "is_mobile": False,
        "latitude": None,
        "longitude": None,
    })

    return measurements, locations


# ---------------------------------------------------------------------------
# DB upserts
# ---------------------------------------------------------------------------

def upsert_locations(conn, locations: pd.DataFrame):
    if locations.empty:
        logger.info("upsert_locations: DataFrame is empty, skipping.")
        return

    records = []
    templates = []
    for row in locations.itertuples(index=False):
        lat, lon = row.latitude, row.longitude
        if lat is not None and lon is not None:
            geom_wkt = f"SRID=4326;POINT({lon} {lat})"
            tmpl = "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,ST_GeogFromText(%s))"
        else:
            geom_wkt = None
            tmpl = "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        records.append((
            row.location_key, row.source, row.external_id, row.name,
            row.locality, row.country_iso, row.owner_name, row.provider_name,
            row.is_monitor, row.is_mobile, lat, lon, geom_wkt,
        ))
        templates.append(tmpl)

    query_prefix = """
        INSERT INTO openaq.locations (
            location_key, source, external_id, name, locality,
            country_iso, owner_name, provider_name, is_monitor,
            is_mobile, latitude, longitude, geom
        ) VALUES
    """
    query_suffix = """
        ON CONFLICT (location_key)
        DO UPDATE SET
            name          = EXCLUDED.name,
            locality      = EXCLUDED.locality,
            country_iso   = EXCLUDED.country_iso,
            owner_name    = EXCLUDED.owner_name,
            provider_name = EXCLUDED.provider_name,
            is_monitor    = EXCLUDED.is_monitor,
            is_mobile     = EXCLUDED.is_mobile,
            latitude      = EXCLUDED.latitude,
            longitude     = EXCLUDED.longitude,
            geom          = EXCLUDED.geom,
            updated_at    = now()
    """

    chunk_size = 500
    total = 0
    with conn.cursor() as cur:
        for i in range(0, len(records), chunk_size):
            chunk_records = records[i: i + chunk_size]
            chunk_templates = templates[i: i + chunk_size]
            values_sql = ", ".join(chunk_templates)
            flat_params = [v for row in chunk_records for v in row]
            cur.execute(query_prefix + values_sql + query_suffix, flat_params)
            total += len(chunk_records)
    conn.commit()
    logger.info("upsert_locations: committed %d rows.", total)


def upsert_measurements(conn, measurements: pd.DataFrame, source: str):
    if measurements.empty:
        logger.info("upsert_measurements (%s): DataFrame is empty, skipping.", source)
        return

    required = {"location_key", "timestamp_utc"}
    if not required.issubset(measurements.columns):
        logger.warning(
            "upsert_measurements (%s): missing columns %s, skipping.",
            source, required - set(measurements.columns),
        )
        return

    values = [
        (
            row.location_key,
            row.timestamp_utc,
            getattr(row, "pm25", None),
            getattr(row, "temperature_c", None),
            getattr(row, "humidity_pct", None),
            getattr(row, "pm1", None),
            getattr(row, "pm10", None),
            getattr(row, "co2", None),
            getattr(row, "tvoc", None),
            bool(getattr(row, "has_pm1", False)),
            bool(getattr(row, "has_pm10", False)),
            bool(getattr(row, "has_co2", False)),
            bool(getattr(row, "has_tvoc", False)),
            source,
            None,
        )
        for row in measurements.itertuples(index=False)
    ]

    query = """
        INSERT INTO openaq.measurements (
            location_key, timestamp_utc,
            pm25, temperature_c, humidity_pct,
            pm1, pm10, co2, tvoc,
            has_pm1, has_pm10, has_co2, has_tvoc,
            source, raw_payload
        )
        VALUES %s
        ON CONFLICT (location_key, timestamp_utc)
        DO UPDATE SET
            pm25          = EXCLUDED.pm25,
            temperature_c = EXCLUDED.temperature_c,
            humidity_pct  = EXCLUDED.humidity_pct,
            pm1           = EXCLUDED.pm1,
            pm10          = EXCLUDED.pm10,
            co2           = EXCLUDED.co2,
            tvoc          = EXCLUDED.tvoc,
            has_pm1       = EXCLUDED.has_pm1,
            has_pm10      = EXCLUDED.has_pm10,
            has_co2       = EXCLUDED.has_co2,
            has_tvoc      = EXCLUDED.has_tvoc,
            source        = EXCLUDED.source
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, query, values, page_size=1000)
    conn.commit()
    logger.info("upsert_measurements (%s): committed %d rows.", source, len(values))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    dsn = resolve_pg_dsn()
    manila_dir = Path(os.environ.get("MANILA_DIR", "data/manila"))

    fallback_start = datetime.now(timezone.utc) - timedelta(days=365)
    end = datetime.now(timezone.utc)

    manila_measurements = pd.DataFrame()
    manila_locations = pd.DataFrame()
    if manila_dir.exists():
        manila_frames: list[pd.DataFrame] = []
        manila_loc_frames: list[pd.DataFrame] = []
        for csv_path in sorted(manila_dir.glob("*.csv")):
            try:
                meas, locs = normalize_manila_csv(csv_path)
                manila_frames.append(meas)
                manila_loc_frames.append(locs)
            except Exception as exc:
                logger.warning("Skipping %s - %s", csv_path, exc)
        if manila_frames:
            manila_measurements = pd.concat(manila_frames, ignore_index=True)
        if manila_loc_frames:
            manila_locations = pd.concat(manila_loc_frames, ignore_index=True)
            manila_locations = manila_locations.drop_duplicates(subset=["location_key"])

    with psycopg2.connect(dsn) as conn:
        watermarks = fetch_watermarks(conn)

        locations = fetch_locations()
        openaq_locations_df = build_openaq_locations(locations)

        logger.info("openaq_locations_df rows: %d", len(openaq_locations_df))
        logger.info("manila_locations rows: %d", len(manila_locations))
        logger.info("manila_measurements rows: %d", len(manila_measurements))

        upsert_locations(conn, openaq_locations_df)
        upsert_locations(conn, manila_locations)

        # Measurements are stream-written to DB inside build_openaq_measurements
        build_openaq_measurements(locations, watermarks, fallback_start, end, conn)
        upsert_measurements(conn, manila_measurements, "manila")

    print("ETL complete.")


if __name__ == "__main__":
    main()
