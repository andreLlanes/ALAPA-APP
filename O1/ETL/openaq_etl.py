"""Load OpenAQ ground-station PM2.5 into Postgres for the three study cities.

Source is the OpenAQ v3 API. Per city, locations inside a bounding box are
discovered, their sensors filtered to PM2.5, and each sensor's hourly series
(/sensors/{id}/hours) is fetched in date chunks and upserted. Fetching and
upserting happen per (city, sensor, date-chunk) and commit immediately; each
completed unit is appended to a text ledger, so a stopped run resumes by skipping
ledger entries rather than re-reading the database.

A sliding-window rate limiter enforces the registered-key caps. Tuning and table
names are constants below; only the OpenAQ API key and Postgres connection come
from the environment (OPENAQ_API_KEY, PG_DSN or PG_HOST/PG_DB/PG_USER/PG_PASSWORD).
"""

import os
import csv
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import psycopg2
import psycopg2.extras
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from dotenv import load_dotenv
load_dotenv()

API_BASE = "https://api.openaq.org/v3"
API_KEY = os.environ.get("OPENAQ_API_KEY")

CHUNK_DAYS = 365
MIN_SPLIT_SPAN = timedelta(days=1)
PAGE_LIMIT = 1000
MAX_PAGES_PER_CHUNK = 50
UPSERT_CHUNK_SIZE = 1000
MAX_RETRIES = 8
BACKOFF_FACTOR = 1
MAX_SLEEP_ON_RETRY_AFTER = 120
RATE_LIMIT_PER_MIN = 45
RATE_LIMIT_PER_HOUR = 1500
REQUEST_TIMEOUT = (15, 90)
SERIES_PATH = "hours"
LEDGER_PATH = "openaq_progress.txt"
MEASUREMENTS_TABLE = "openaq.gs_measurements"
LOCATIONS_TABLE = "openaq.gs_stations"

# bbox is (min_lon, min_lat, max_lon, max_lat). LA longitudes are West (negative).
CITY_BBOXES = {
    "Los Angeles":  (-118.67, 33.70, -118.15, 34.34),
    "Bangkok":      (100.32, 13.48, 100.94, 13.97),
    "Metro Manila": (120.93, 14.34, 121.15, 14.78),
}

CITY_PERIODS = {
    "Los Angeles":  (datetime(2017, 1, 1, tzinfo=timezone.utc),
                     datetime(2026, 6, 30, tzinfo=timezone.utc)),
    "Bangkok":      (datetime(2017, 1, 1, tzinfo=timezone.utc),
                     datetime(2026, 6, 30, tzinfo=timezone.utc)),
    "Metro Manila": (datetime(2023, 9, 6, tzinfo=timezone.utc),
                     datetime(2026, 6, 30, tzinfo=timezone.utc)),
}

PARAMETER_MAP = {
    "pm25": "pm25",
}

MEASUREMENT_COLUMNS = ["pm25"]


class OpenAQAuthError(Exception):
    """Raised on 401/403 (invalid or expired API key); not retried."""


class RateLimited(Exception):
    """Raised on a 429 whose Retry-After exceeds the wait cap; stops the run."""


class RequestTimeout(Exception):
    """Raised on a 408 so the caller can split the date window and retry."""


class RateLimiter:
    """Sliding-window limiter enforcing per-minute and per-hour request caps."""

    def __init__(self, per_minute, per_hour):
        self.windows = []
        if per_minute > 0:
            self.windows.append((60, per_minute, "minute"))
        if per_hour > 0:
            self.windows.append((3600, per_hour, "hour"))
        self.calls = []

    def acquire(self):
        """Block until a request may be sent without breaching any window."""
        if not self.windows:
            return
        max_span = max(s for s, _, _ in self.windows)
        while True:
            now = time.monotonic()
            self.calls = [t for t in self.calls if t > now - max_span]
            wait = 0.0
            label = None
            for span, cap, name in self.windows:
                recent = [t for t in self.calls if t > now - span]
                if len(recent) >= cap:
                    w = recent[0] + span - now
                    if w > wait:
                        wait, label = w, name
            if wait <= 0:
                self.calls.append(time.monotonic())
                return
            print(f"  rate limit ({label}) reached; waiting {wait:.0f}s")
            time.sleep(wait + 0.05)


LIMITER = RateLimiter(RATE_LIMIT_PER_MIN, RATE_LIMIT_PER_HOUR)

retry_strategy = Retry(
    total=MAX_RETRIES,
    backoff_factor=BACKOFF_FACTOR,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    raise_on_status=False,
)
SESSION = requests.Session()
_adapter = HTTPAdapter(max_retries=retry_strategy)
SESSION.mount("https://", _adapter)
SESSION.mount("http://", _adapter)
if API_KEY:
    SESSION.headers.update({"X-API-Key": API_KEY, "Connection": "close"})


def request_json(url, params=None):
    """GET a URL through the rate limiter and return the parsed JSON body.

    Retries transient transport and 5xx errors with backoff; treats 400/422 as
    empty results; raises OpenAQAuthError (401/403), RateLimited (429 with a long
    Retry-After), or RequestTimeout (408) for the caller to handle.
    """
    attempt = 0
    while True:
        LIMITER.acquire()
        try:
            resp = SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
        except requests.exceptions.RequestException as exc:
            attempt += 1
            if attempt >= MAX_RETRIES:
                raise
            wait = BACKOFF_FACTOR * (2 ** attempt)
            print(f"  transport error (attempt {attempt}/{MAX_RETRIES}): {exc}; "
                  f"retrying in {wait:.0f}s")
            time.sleep(wait)
            continue

        if resp.status_code in (401, 403):
            raise OpenAQAuthError(
                f"OpenAQ returned HTTP {resp.status_code} (invalid or expired API key). "
                f"Regenerate it at https://explore.openaq.org/ and update OPENAQ_API_KEY."
            )

        try:
            payload = resp.json()
        except ValueError:
            payload = None

        if resp.status_code == 200 and payload is not None:
            return payload

        if resp.status_code in (400, 422):
            print(f"  bad query (HTTP {resp.status_code}), skipping: params={params}")
            return {"results": []}

        detail = f"HTTP {resp.status_code}: {resp.text[:200]}"

        if resp.status_code == 429:
            ra = resp.headers.get("Retry-After")
            try:
                retry_after = int(ra) if ra is not None else None
            except ValueError:
                retry_after = None
            if retry_after is not None and retry_after <= MAX_SLEEP_ON_RETRY_AFTER:
                print(f"  rate limited; server asked to wait {retry_after}s")
                time.sleep(retry_after)
                continue
            raise RateLimited(detail)

        if resp.status_code == 408:
            raise RequestTimeout(detail)

        attempt += 1
        if attempt >= MAX_RETRIES:
            raise RuntimeError(f"OpenAQ request failed after {MAX_RETRIES} attempts: {detail}")
        wait = BACKOFF_FACTOR * (2 ** attempt)
        print(f"  server error {resp.status_code} (attempt {attempt}/{MAX_RETRIES}): "
              f"{detail}; retrying in {wait:.0f}s")
        time.sleep(wait)


def load_ledger():
    """Return the set of completed (city, sensor_id, chunk_start, chunk_end)."""
    done = set()
    if not os.path.exists(LEDGER_PATH):
        return done
    with open(LEDGER_PATH, newline="") as f:
        for row in csv.reader(f):
            if len(row) == 4:
                done.add(tuple(row))
    return done


def append_ledger(city, sensor_id, chunk_start, chunk_end):
    """Record one completed (city, sensor, date-chunk) in the ledger."""
    with open(LEDGER_PATH, "a", newline="") as f:
        csv.writer(f).writerow([city, sensor_id, chunk_start, chunk_end])


def fetch_city_locations(city, bbox):
    """Page through OpenAQ locations in a bbox, keeping those inside it."""
    locations = []
    page = 1
    while True:
        params = {"limit": 100, "page": page, "bbox": ",".join(str(x) for x in bbox)}
        data = request_json(f"{API_BASE}/locations", params=params)
        results = data.get("results", [])
        if not results:
            break
        min_lon, min_lat, max_lon, max_lat = bbox
        for loc in results:
            coords = loc.get("coordinates") or {}
            lat, lon = coords.get("latitude"), coords.get("longitude")
            if lat is None or lon is None:
                continue
            if not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
                continue
            locations.append(loc)
        page += 1
    return locations


def build_locations_frame(city, locations):
    """Flatten OpenAQ location objects into a station-metadata DataFrame."""
    rows = []
    for loc in locations:
        loc_id = loc.get("id")
        coords = loc.get("coordinates") or {}
        rows.append({
            "location_key": f"openaq:{loc_id}",
            "source": "openaq",
            "external_id": str(loc_id),
            "name": loc.get("name"),
            "locality": loc.get("locality"),
            "country_iso": (loc.get("country") or {}).get("code"),
            "owner_name": (loc.get("owner") or {}).get("name"),
            "provider_name": (loc.get("provider") or {}).get("name"),
            "is_monitor": loc.get("isMonitor"),
            "is_mobile": loc.get("isMobile"),
            "latitude": coords.get("latitude"),
            "longitude": coords.get("longitude"),
            "city": city,
        })
    return pd.DataFrame(rows)


def location_history_start(loc, fallback_start):
    """Clamp the fetch start to when the location first reported, if later."""
    dt_first = (loc.get("datetimeFirst") or {}).get("utc")
    if not dt_first:
        return fallback_start
    parsed = pd.to_datetime(dt_first, utc=True, errors="coerce")
    if pd.isna(parsed):
        return fallback_start
    return max(fallback_start, parsed.to_pydatetime())


def fetch_sensor_chunk(sensor_id, dt_from, dt_to):
    """Fetch a sensor's hourly rows for a date window, paging through results.

    On a 408 timeout the window is bisected and each half fetched recursively,
    down to MIN_SPLIT_SPAN; a window that still times out at the floor is skipped.
    """
    try:
        rows = []
        base = {"datetime_from": dt_from, "datetime_to": dt_to, "limit": PAGE_LIMIT}
        for page in range(1, MAX_PAGES_PER_CHUNK + 1):
            params = {**base, "page": page}
            data = request_json(f"{API_BASE}/sensors/{sensor_id}/{SERIES_PATH}", params=params)
            results = data.get("results", [])
            if not results:
                break
            rows.extend(results)
        return rows
    except RequestTimeout:
        start = datetime.strptime(dt_from, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        end = datetime.strptime(dt_to, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        if (end - start) <= MIN_SPLIT_SPAN:
            print(f"    timeout at minimum window {dt_from}..{dt_to}; skipping")
            return []
        mid = start + (end - start) / 2
        mid_str = mid.strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"    timeout; splitting {dt_from}..{dt_to} at {mid_str}")
        return (fetch_sensor_chunk(sensor_id, dt_from, mid_str)
                + fetch_sensor_chunk(sensor_id, mid_str, dt_to))


def _extract_utc(r):
    """Pull a UTC timestamp string from a measurement row across v3 shapes."""
    period = r.get("period") or {}
    for key in ("datetimeTo", "datetimeFrom"):
        d = period.get(key)
        if isinstance(d, dict) and d.get("utc"):
            return d["utc"]
    for key in ("datetime", "date"):
        d = r.get(key)
        if isinstance(d, dict) and d.get("utc"):
            return d["utc"]
        if isinstance(d, str) and d:
            return d
    return r.get("datetimeUtc")


def _extract_param(r):
    """Pull the lowercase parameter name from a measurement row."""
    p = r.get("parameter")
    if isinstance(p, dict):
        return (p.get("name") or "").lower()
    if isinstance(p, str):
        return p.lower()
    return ""


def normalize_measurements(rows, location_id, city):
    """Turn raw sensor rows into an hourly, wide, PM2.5-only measurement frame.

    Timestamps are floored to the hour and duplicate station-hours averaged;
    parameters outside PARAMETER_MAP are dropped. Returns an empty frame if
    nothing valid remains.
    """
    records = []
    for r in rows:
        ts = _extract_utc(r)
        param = _extract_param(r)
        value = r.get("value")
        if ts is None or not param or value is None:
            continue
        records.append({"locationId": r.get("locationId", location_id),
                        "datetimeUtc": ts, "parameter": param, "value": value})
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["datetimeUtc"] = pd.to_datetime(df["datetimeUtc"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetimeUtc"])
    df["datetimeUtc"] = df["datetimeUtc"].dt.floor("h")
    df["parameter"] = df["parameter"].map(PARAMETER_MAP)
    df = df.dropna(subset=["parameter"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    if df.empty:
        return pd.DataFrame()

    pivot = df.pivot_table(index=["locationId", "datetimeUtc"], columns="parameter",
                           values="value", aggfunc="mean").reset_index()
    pivot = pivot.rename(columns={"locationId": "location_id", "datetimeUtc": "timestamp_utc"})
    for col in MEASUREMENT_COLUMNS:
        if col not in pivot.columns:
            pivot[col] = pd.NA
    pivot["location_key"] = pivot["location_id"].apply(lambda v: f"openaq:{v}")
    pivot["city"] = city
    return pivot


def _date_chunks(start, end):
    """Yield (start, end) sub-windows of at most CHUNK_DAYS across [start, end)."""
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=CHUNK_DAYS), end)
        yield cur, nxt
        cur = nxt


def fetch_and_store_city(city, bbox, start_date, end_date, done):
    """Discover a city's PM2.5 sensors and stream each date-chunk to Postgres.

    Skips ledger-recorded chunks; a chunk whose fetch exhausts its retries is
    left unlogged so a later run retries it. Clamps each sensor's start to its
    first reporting time.
    """
    print(f"{city}: discovering locations in bbox {bbox}")
    locations = fetch_city_locations(city, bbox)
    print(f"{city}: {len(locations)} locations")
    if locations:
        upsert_locations(build_locations_frame(city, locations))

    city_rows = 0
    for loc in locations:
        loc_id = loc.get("id")
        if not loc_id:
            continue
        loc_start = location_history_start(loc, start_date)
        for sensor in (loc.get("sensors") or []):
            sensor_id = sensor.get("id")
            if not sensor_id:
                continue
            param = (sensor.get("parameter") or {}).get("name", "").lower()
            if param not in PARAMETER_MAP:
                continue
            for c_start, c_end in _date_chunks(loc_start, end_date):
                dt_from = c_start.strftime("%Y-%m-%dT%H:%M:%SZ")
                dt_to = c_end.strftime("%Y-%m-%dT%H:%M:%SZ")
                if dt_from >= dt_to:
                    continue
                key = (city, str(sensor_id), dt_from, dt_to)
                tag = f"  {city} sensor {sensor_id} ({param})  {c_start.date()}..{c_end.date()}"
                if key in done:
                    print(tag + "  [skip: already stored]")
                    continue
                print(tag)
                try:
                    rows = fetch_sensor_chunk(sensor_id, dt_from, dt_to)
                except RuntimeError as exc:
                    print(f"    server error persisted, leaving chunk for a later run: {exc}")
                    continue
                df = normalize_measurements(rows, loc_id, city) if rows else pd.DataFrame()
                store_chunk(df)
                append_ledger(city, sensor_id, dt_from, dt_to)
                done.add(key)
                city_rows += len(df)

    print(f"{city}: done ({city_rows} rows this run).")


def resolve_pg_dsn():
    """Build the Postgres DSN from PG_DSN, or from the PG_* component vars."""
    dsn = os.environ.get("PG_DSN")
    if dsn:
        return dsn
    host = os.environ.get("PG_HOST")
    port = os.environ.get("PG_PORT", "5432")
    dbname = os.environ.get("PG_DB")
    user = os.environ.get("PG_USER")
    password = os.environ.get("PG_PASSWORD")
    if not all([host, dbname, user, password]):
        raise ValueError("Missing PG_DSN or PG_HOST/PG_DB/PG_USER/PG_PASSWORD.")
    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"


_PG_DSN = None
_CONN = None


def get_conn():
    """Return a live connection, reconnecting if the previous one has dropped."""
    global _CONN, _PG_DSN
    if _PG_DSN is None:
        _PG_DSN = resolve_pg_dsn()
    if _CONN is not None and _CONN.closed == 0:
        try:
            with _CONN.cursor() as cur:
                cur.execute("SELECT 1")
            return _CONN
        except psycopg2.Error:
            try:
                _CONN.close()
            except psycopg2.Error:
                pass
            _CONN = None
    if _CONN is None or _CONN.closed != 0:
        _CONN = psycopg2.connect(_PG_DSN)
    return _CONN


def _safe_rollback(conn):
    """Roll back without raising if the connection is already broken."""
    try:
        conn.rollback()
    except psycopg2.Error:
        pass


def _na_to_none(v):
    """Map pandas NA/NaN to None so psycopg2 writes SQL NULL."""
    try:
        if v is None or pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def upsert_locations(df):
    """Upsert station metadata into the locations table, chunked, with PostGIS geom."""
    if df.empty:
        return
    conn = get_conn()
    records, templates = [], []
    for row in df.itertuples(index=False):
        lat, lon = row.latitude, row.longitude
        if lat is not None and lon is not None:
            geom_wkt = f"SRID=4326;POINT({lon} {lat})"
            tmpl = "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,ST_GeogFromText(%s))"
        else:
            geom_wkt = None
            tmpl = "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        records.append((
            row.location_key, row.source, row.external_id, row.name, row.locality,
            row.country_iso, row.owner_name, row.provider_name, row.is_monitor,
            row.is_mobile, lat, lon, geom_wkt,
        ))
        templates.append(tmpl)

    prefix = f"""
        INSERT INTO {LOCATIONS_TABLE} (
            location_key, source, external_id, name, locality,
            country_iso, owner_name, provider_name, is_monitor,
            is_mobile, latitude, longitude, geom
        ) VALUES
    """
    suffix = """
        ON CONFLICT (location_key) DO UPDATE SET
            name = EXCLUDED.name, locality = EXCLUDED.locality,
            country_iso = EXCLUDED.country_iso, owner_name = EXCLUDED.owner_name,
            provider_name = EXCLUDED.provider_name, is_monitor = EXCLUDED.is_monitor,
            is_mobile = EXCLUDED.is_mobile, latitude = EXCLUDED.latitude,
            longitude = EXCLUDED.longitude, geom = EXCLUDED.geom, updated_at = now()
    """
    try:
        with conn.cursor() as cur:
            for i in range(0, len(records), 500):
                chunk_r = records[i:i + 500]
                chunk_t = templates[i:i + 500]
                flat = [v for r in chunk_r for v in r]
                cur.execute(prefix + ", ".join(chunk_t) + suffix, flat)
        conn.commit()
    except Exception:
        _safe_rollback(conn)
        raise


def store_chunk(df):
    """Upsert one sensor-chunk's PM2.5 rows into the measurements table."""
    conn = get_conn()
    if df.empty:
        conn.commit()
        print("  committed 0 rows")
        return
    required = {"location_key", "timestamp_utc"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")

    values = [
        (
            row.location_key, row.timestamp_utc,
            _na_to_none(getattr(row, "pm25", None)),
            row.city, "openaq",
        )
        for row in df.itertuples(index=False)
    ]

    query = f"""
        INSERT INTO {MEASUREMENTS_TABLE} (
            location_key, timestamp_utc, pm25, city, source
        ) VALUES %s
        ON CONFLICT (location_key, timestamp_utc) DO UPDATE SET
            pm25 = EXCLUDED.pm25,
            city = EXCLUDED.city,
            source = EXCLUDED.source
    """
    try:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, query, values, page_size=UPSERT_CHUNK_SIZE)
        conn.commit()
    except Exception:
        _safe_rollback(conn)
        raise
    print(f"  committed {len(values)} rows")


def main():
    """Run the backfill for every city, resuming from the ledger."""
    if not API_KEY:
        raise ValueError("OPENAQ_API_KEY is required in the environment.")
    done = load_ledger()
    print(f"Loaded {len(done)} completed (city, sensor, chunk) entries from {LEDGER_PATH}.")
    try:
        for city, bbox in CITY_BBOXES.items():
            start_date, end_date = CITY_PERIODS[city]
            fetch_and_store_city(city, bbox, start_date, end_date, done)
    except RateLimited as e:
        print(f"\nStopped: OpenAQ rate limit hit ({e}).")
        print("Completed chunks are committed and logged; re-run to resume.")
        return
    finally:
        if _CONN is not None and _CONN.closed == 0:
            _CONN.close()
    print("Done.")


if __name__ == "__main__":
    main()