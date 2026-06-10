"""OpenAQ -> PostgreSQL ETL for the ALAPA air-quality thesis.

Pulls AirGradient/air4thai sensor measurements from Bangkok (TH) and Singapore (SG)
via the OpenAQ v3 API for transfer learning, plus optional Manila AirGradient CSVs,
and upserts them into the `openaq` schema.

Performance model (why a first 5-year backfill is slow and how we cope):
    OpenAQ rate limits a registered key to 60 req/min AND 2,000 req/hour. The wall
    clock of a full backfill is therefore bounded by the *number of requests*, not by
    CPU. We minimise both wasted requests and rate-limit penalties via:
      - a single global RateLimiter that paces every request just under the cap, so we
        never trigger the expensive 429 exponential back-off,
      - per-location watermarks (only fetch what is newer than what we already stored),
      - per-location history clamping using each location's datetimeFirst (AirGradient
        sensors are young; don't request years of empty range),
      - skipping sensors whose parameter we don't model.

Tunable via environment variables (all optional):
    OPENAQ_LOOKBACK_DAYS  default 730 (2 years) — fallback start when no watermark
    OPENAQ_RPM            default 55   — requests/minute budget (headroom under the 60 cap)
    OPENAQ_RPH            default 1800 — requests/hour budget (headroom under the 2000 cap)
    OPENAQ_MAX_WORKERS    default 5    — concurrent fetch threads
    OPENAQ_CHUNK_DAYS     default 90   — measurement date-range chunk size
"""

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

# --- Tunables (env-overridable) --------------------------------------------
LOOKBACK_DAYS = int(os.environ.get("OPENAQ_LOOKBACK_DAYS", "730"))  # 2 years
MAX_WORKERS = int(os.environ.get("OPENAQ_MAX_WORKERS", "5"))
CHUNK_DAYS = int(os.environ.get("OPENAQ_CHUNK_DAYS", "365"))
# Use the hourly-AGGREGATE series (/hours), NOT raw /measurements. High-frequency sensors
# (e.g. air4thai) return tens of thousands of raw points per chunk, and OpenAQ TIMES OUT
# (HTTP 408) on deep pagination of that. /hours returns one row per hour — a few pages per
# chunk — and is exactly the granularity the 72h forecast needs. The cap guards against any
# runaway pagination (hourly data needs ~ceil(chunk_days*24/1000) pages).
SENSOR_SERIES_PATH = os.environ.get("OPENAQ_SERIES_PATH", "hours")
MAX_PAGES_PER_CHUNK = 50
# Defaults sit under the 60/min & 2000/hr hard caps for safety (the prior account was
# suspended for bursting past them). Raise via env only if you have a higher tier.
RPM = int(os.environ.get("OPENAQ_RPM", "55"))
RPH = int(os.environ.get("OPENAQ_RPH", "1800"))
# Optional work-sharding: run N machines (each with its OWN OpenAQ key) over disjoint
# subsets of locations. ETL_SHARD_COUNT=1 (default) = no split / current behaviour.
SHARD_INDEX = int(os.environ.get("ETL_SHARD_INDEX", "0"))
SHARD_COUNT = int(os.environ.get("ETL_SHARD_COUNT", "1"))

# Region filters for OpenAQ /locations. Each region uses a server-side bbox (min_lon,
# min_lat, max_lon, max_lat) so we only page through the metro area, not the whole country
# — far fewer requests per run. isMonitor is deliberately NOT used: the low-cost AirGradient
# (SG/TH/Manila) and Clarity (Manila) sensors central to this study are isMonitor=false, so
# filtering isMonitor=true would exclude exactly what we want.
COUNTRY_FILTERS = {
    # Singapore (island-wide) — AirGradient network.
    "SG": {
        "bbox": (103.55, 1.13, 104.10, 1.50),
        "providers": {"airgradient"},
    },
    # Bangkok metropolitan region — AirGradient + air4thai.
    "TH": {
        "bbox": (100.25, 13.40, 101.00, 14.10),
        "providers": {"airgradient", "air4thai"},
    },
    # Manila / NCR — the transfer-learning TARGET domain. Accept ALL providers inside the
    # Metro Manila bbox (~69 sensors: Clarity, AirGradient, AirNow, Spartan; almost all PM2.5).
    "PH": {
        "bbox": (120.88, 14.30, 121.15, 14.80),
    },
}

# OpenAQ parameter name (lowercased) -> our DB column.
# NOTE: OpenAQ calls relative humidity "relativehumidity", NOT "humidity" — the old
# map silently dropped every humidity reading. We keep aliases for safety.
PARAMETER_MAP = {
    "pm25": "pm25",
    "pm10": "pm10",
    "pm1": "pm1",
    "temperature": "temperature_c",
    "temp": "temperature_c",
    "relativehumidity": "humidity_pct",
    "humidity": "humidity_pct",
    "rh": "humidity_pct",
    "co2": "co2",
    "tvoc": "tvoc",
}

# Columns the wide measurements frame must always expose (DB columns).
MEASUREMENT_COLUMNS = ["pm25", "temperature_c", "humidity_pct", "pm1", "pm10", "co2", "tvoc"]


class OpenAQAuthError(Exception):
    """Raised on 401/403 — an invalid/expired API key. Not a transient error, so it is
    deliberately NOT a RuntimeError (the discovery loops swallow RuntimeError)."""


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


def _pick_pm1(columns: list[str]) -> str | None:
    """Match a PM1 column without catching PM10/PM2.5. The unit suffix means headers
    normalize to e.g. "pm1g"/"pm10g", so we test substrings rather than exact tokens."""
    for col in columns:
        norm = normalize_header(col)
        if "pm1" in norm and "pm10" not in norm and "pm100" not in norm:
            return col
    return None


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
    # "Connection: close" disables HTTP keep-alive. Because the global RateLimiter spaces
    # requests ~2s apart across worker threads, pooled connections sit idle long enough for
    # OpenAQ to drop them, and the next reuse gets a TCP reset (WinError 10054). A fresh
    # connection per request avoids that; the extra handshake is free at this pacing.
    session.headers.update({"X-API-Key": API_KEY, "Connection": "close"})
    return session


# One Session per thread — requests.Session is not safe for concurrent use across threads.
_thread_local = threading.local()


def _get_thread_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        _thread_local.session = get_session()
    return _thread_local.session


# ---------------------------------------------------------------------------
# Global rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Process-wide minimum spacing between requests, shared across all threads.

    The binding OpenAQ constraint is the *hourly* budget: 2,000/hr == one request every
    1.8s. Pacing to that ceiling means we never hit 429 (whose back-off is what actually
    made the old code crawl), while still saturating throughput up to the cap.
    """

    def __init__(self, per_minute: int, per_hour: int):
        intervals = []
        if per_minute and per_minute > 0:
            intervals.append(60.0 / per_minute)
        if per_hour and per_hour > 0:
            intervals.append(3600.0 / per_hour)
        self.min_interval = max(intervals) if intervals else 0.0
        self._lock = threading.Lock()
        self._next_time = 0.0

    def acquire(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            scheduled = max(now, self._next_time)
            self._next_time = scheduled + self.min_interval
            wait = scheduled - now
        if wait > 0:
            time.sleep(wait)


RATE_LIMITER = RateLimiter(RPM, RPH)


def request_json(session: requests.Session, url: str, params: dict | None = None) -> dict:
    retryable = {408, 500, 502, 503, 504}
    # 4 attempts caps a persistently failing call at ~15s (1+2+4+8) instead of ~63s; the
    # server bugs we hit (e.g. 500s) don't clear within a single call's retries anyway.
    for attempt in range(4):
        RATE_LIMITER.acquire()  # global pacing — replaces the old per-request sleep
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
            session.close()  # drop any poisoned/reset connection so the retry opens a fresh one
            time.sleep(wait)
            continue

        if resp.status_code in (401, 403):
            # Invalid/expired key — fail loud and fast, do not waste a multi-hour run.
            raise OpenAQAuthError(
                f"OpenAQ returned HTTP {resp.status_code} (invalid or expired API key). "
                f"Regenerate the key at https://explore.openaq.org/ and update OPENAQ_API_KEY."
            )

        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", min(2 ** attempt, 60)))
            logger.warning(
                "Rate limited (attempt %s), sleeping %ss: %s", attempt + 1, retry_after, url,
            )
            time.sleep(retry_after)
            continue

        if resp.status_code in retryable:
            wait = min(2 ** attempt, 60)
            logger.warning("HTTP %s (attempt %s), sleeping %ss: %s", resp.status_code, attempt + 1, wait, url)
            time.sleep(wait)
            continue

        if resp.status_code in (400, 422):
            # Semantically-invalid query (e.g. an empty datetime window) — not retryable.
            # Skip it gracefully and return no results so one bad chunk can't propagate as an
            # exception that discards the sensor's other (good) chunks.
            logger.warning("HTTP %s (bad query, skipping): %s params=%s", resp.status_code, url, params)
            return {"results": []}

        resp.raise_for_status()
        return resp.json()

    raise RuntimeError(f"OpenAQ request failed after retries for {url}")


def verify_credentials(session: requests.Session) -> None:
    """Cheap preflight so a bad key aborts immediately with a clear message."""
    resp = session.get(f"{API_BASE}/locations", params={"limit": 1}, timeout=60)
    if resp.status_code in (401, 403):
        raise OpenAQAuthError(
            f"OpenAQ API key is invalid or expired (HTTP {resp.status_code}). "
            f"Regenerate it at https://explore.openaq.org/ and update OPENAQ_API_KEY in .env."
        )
    resp.raise_for_status()
    logger.info("OpenAQ credentials OK.")


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
    """Fetch + filter location pages for one region (thread-local session).

    All filters optional: providers (empty = accept any), a geographic bbox (sent
    server-side AND re-checked client-side), and locality/name keyword gates. We never
    filter by isMonitor (see COUNTRY_FILTERS note)."""
    session = _get_thread_session()
    providers = filters.get("providers") or set()
    bbox = filters.get("bbox")
    cities = filters.get("cities") or set()
    name_keywords = filters.get("name_keywords") or set()
    locations: list[dict] = []
    page = 1
    while True:
        params: dict = {"limit": 100, "page": page}
        if bbox:
            # bbox is authoritative for Manila; don't also send country (some sensors have
            # a null country and would be dropped by a country filter).
            params["bbox"] = ",".join(str(x) for x in bbox)
        else:
            params["country"] = country
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
            if providers and provider not in providers:
                continue
            if bbox:
                coords = loc.get("coordinates") or {}
                lat, lon = coords.get("latitude"), coords.get("longitude")
                if lat is None or lon is None:
                    continue
                min_lon, min_lat, max_lon, max_lat = bbox
                if not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
                    continue
            locality = (loc.get("locality") or loc.get("city") or "").lower()
            if cities or name_keywords:
                if locality and cities and locality not in cities:
                    continue
                if not locality and name_keywords and not any(
                    kw in (loc.get("name") or "").lower() for kw in name_keywords
                ):
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


def _apply_shard(locations: list[dict]) -> list[dict]:
    """Keep only this machine's slice of locations (id %% SHARD_COUNT == SHARD_INDEX) so
    multiple machines — each with its own OpenAQ key — can split the backfill without
    overlapping. SHARD_COUNT <= 1 is a no-op (single-machine default)."""
    if SHARD_COUNT <= 1:
        return locations
    sub = [
        loc for loc in locations
        if loc.get("id") is not None and int(loc["id"]) % SHARD_COUNT == SHARD_INDEX
    ]
    logger.info("Shard %d/%d -> %d of %d locations", SHARD_INDEX, SHARD_COUNT, len(sub), len(locations))
    return sub


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
    chunk_start = start

    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), end)
        dt_from = chunk_start.strftime("%Y-%m-%dT%H:%M:%SZ")
        dt_to = chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ")
        # When start is within the same second as end (a sensor that is already up to date),
        # the two format to an identical, empty window which OpenAQ rejects with HTTP 422.
        # Nothing to fetch in a zero-width window — stop.
        if dt_from >= dt_to:
            break
        params_base = {
            # OpenAQ /hours (and /measurements) filter on datetime_from/datetime_to — NOT
            # date_from/date_to, which the API silently IGNORES, returning the sensor's entire
            # history and forcing deep pagination that times out (HTTP 408).
            "datetime_from": dt_from,
            "datetime_to": dt_to,
            "limit": 1000,
        }
        for page in range(1, MAX_PAGES_PER_CHUNK + 1):
            params = {**params_base, "page": page}
            try:
                data = request_json(
                    session, f"{API_BASE}/sensors/{sensor_id}/{SENSOR_SERIES_PATH}", params=params
                )
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
        chunk_start = chunk_end

    return rows


def _extract_utc(r: dict) -> str | None:
    """Pull a UTC timestamp from a measurement row, tolerating the several shapes the
    v3 API uses across raw vs aggregated endpoints (this robustness is why measurements
    now actually land — the old code assumed a single nested shape and silently dropped
    every row when the shape differed)."""
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
    if r.get("datetimeUtc"):
        return r["datetimeUtc"]
    return None


def _extract_param(r: dict) -> str:
    p = r.get("parameter")
    if isinstance(p, dict):
        return (p.get("name") or "").lower()
    if isinstance(p, str):
        return p.lower()
    return ""


def normalize_openaq_measurements(rows: list[dict], location_id: int) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    records = []
    for r in rows:
        ts = _extract_utc(r)
        param = _extract_param(r)
        value = r.get("value")
        if ts is None or not param or value is None:
            continue
        records.append({
            "locationId": r.get("locationId", location_id),
            "datetimeUtc": ts,
            "parameter": param,
            "value": value,
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["datetimeUtc"] = pd.to_datetime(df["datetimeUtc"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetimeUtc"])
    # Hourly grid: floor to the hour and average within the bucket. This satisfies the
    # thesis "hourly aggregation" requirement and collapses any sub-hourly cadence.
    df["datetimeUtc"] = df["datetimeUtc"].dt.floor("h")
    df["parameter"] = df["parameter"].map(PARAMETER_MAP)
    df = df.dropna(subset=["parameter"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    if df.empty:
        return pd.DataFrame()

    pivot = df.pivot_table(
        index=["locationId", "datetimeUtc"],
        columns="parameter",
        values="value",
        aggfunc="mean",
    ).reset_index()

    pivot = pivot.rename(columns={"locationId": "location_id", "datetimeUtc": "timestamp_utc"})
    for col in MEASUREMENT_COLUMNS:
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


def _location_history_start(loc: dict, fallback_start: datetime) -> datetime:
    """Clamp the start to when the location actually began reporting (datetimeFirst).
    Avoids requesting years of empty range for young AirGradient sensors."""
    dt_first = (loc.get("datetimeFirst") or {}).get("utc")
    if not dt_first:
        return fallback_start
    parsed = pd.to_datetime(dt_first, utc=True, errors="coerce")
    if pd.isna(parsed):
        return fallback_start
    return max(fallback_start, parsed.to_pydatetime())


def build_openaq_measurements(
    locations: list[dict],
    watermarks: dict[str, datetime],
    fallback_start: datetime,
    end: datetime,
    conn,
    max_workers: int = MAX_WORKERS,
) -> None:
    """Fetch measurements for all sensors concurrently and stream-write to DB.

    Phase 1 (serial): collect sensor jobs — sensor-list calls are fast (~1 req/location).
    Phase 2 (concurrent): fetch measurements in parallel; write to DB from the main thread.
    """
    # Phase 1: discover sensors and build the job list.
    # The /v3/locations response already embeds each location's full `sensors` array, so we
    # read it from the location object instead of calling /locations/{id}/sensors per
    # location — that sub-endpoint returns persistent HTTP 500 for many air4thai stations
    # and was making discovery crawl (and contributes nothing we don't already have).
    jobs: list[tuple[int, str, int, datetime]] = []
    for loc in locations:
        loc_id = loc.get("id")
        if not loc_id:
            continue
        location_key = f"openaq:{loc_id}"
        # Watermark wins if present; otherwise clamp the 5-year fallback to real history.
        start = watermarks.get(location_key) or _location_history_start(loc, fallback_start)
        if start >= end:
            logger.info("Location %s is up-to-date, skipping.", location_key)
            continue
        sensors = loc.get("sensors") or []
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

    # Process not-yet-ingested locations (no watermark — e.g. Bangkok) FIRST. This sends the
    # run's budget to the outstanding work before any re-suspension; already-complete
    # locations (SG/PH) only get a quick "anything new?" check, and that happens last.
    jobs.sort(key=lambda j: f"openaq:{j[2]}" in watermarks)

    logger.info("Submitting %d sensor fetch jobs (max_workers=%d).", len(jobs), max_workers)

    # Phase 2: concurrent fetch + stream writes to DB from the main thread.
    # Worker threads only return DataFrames; DB writes stay on the main thread because
    # psycopg2 connections are not thread-safe.
    total_rows = 0
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_one_sensor, sid, pname, lid, s, end): (sid, pname, lid)
            for sid, pname, lid, s in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            sid, pname, lid = futures[future]
            done += 1
            try:
                df = future.result()
                if df is not None:
                    upsert_measurements(conn, df, "openaq")
                    total_rows += len(df)
                    logger.info(
                        "[%d/%d] Wrote %d rows for sensor %s (%s).",
                        done, len(jobs), len(df), sid, pname,
                    )
            except OpenAQAuthError:
                raise  # invalid key — abort the whole run, retrying every sensor is pointless
            except Exception as exc:
                # A failed upsert aborts the transaction; roll back so later sensors still write.
                conn.rollback()
                logger.warning("Sensor %s (%s) loc %s failed: %s", sid, pname, lid, exc)

    logger.info("build_openaq_measurements: total %d rows committed.", total_rows)


# ---------------------------------------------------------------------------
# Manila CSV
# ---------------------------------------------------------------------------

def normalize_manila_csv(
    csv_path: Path, location_hint: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path, encoding="utf-8", encoding_errors="replace")
    columns = df.columns.tolist()

    # Prefer an explicit UTC column (raw AirGradient export). Fall back to any timestamp/
    # datetime column — the cleaned files use a tz-aware "timestamp" (+08:00) which
    # to_datetime(utc=True) converts to UTC correctly.
    ts_col = (
        pick_column(columns, ["utc", "datetime"])
        or pick_column(columns, ["utc", "date"])
        or pick_column(columns, ["timestamp"])
        or pick_column(columns, ["datetime"])
        or pick_column(columns, ["date"])
        or pick_column(columns, ["time"])
    )
    if not ts_col:
        raise ValueError(f"No timestamp column found in {csv_path}.")

    col_pm25 = pick_column(columns, ["pm25"], ["corr"]) or pick_column(columns, ["pm25"])
    col_temp = pick_column(columns, ["temperature"], ["corr"]) or pick_column(columns, ["temperature"])
    col_humidity = pick_column(columns, ["humidity"], ["corr"]) or pick_column(columns, ["humidity"])
    # PM1/PM10 carry a unit suffix (e.g. "PM1 (µg/m³)" -> normalized "pm1g"), so the old
    # exact-token match found nothing — pm1/pm10 were always empty. Match by substring.
    col_pm10 = pick_column(columns, ["pm10"])
    col_pm1 = _pick_pm1(columns)
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
    elif location_hint:
        # Cleaned files carry no location column — derive identity from the filename.
        location_keys = [location_hint] * len(df)
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

def _na_to_none(value):
    """psycopg2 cannot adapt pandas NA/NaN to SQL — coerce them to NULL. Each sensor reports
    only one parameter, so most measurement columns are NA; without this the insert raises
    'can't adapt type NAType' and the whole sensor's write fails."""
    try:
        if value is None or pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


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
    try:
        with conn.cursor() as cur:
            for i in range(0, len(records), chunk_size):
                chunk_records = records[i: i + chunk_size]
                chunk_templates = templates[i: i + chunk_size]
                values_sql = ", ".join(chunk_templates)
                flat_params = [v for row in chunk_records for v in row]
                cur.execute(query_prefix + values_sql + query_suffix, flat_params)
                total += len(chunk_records)
        conn.commit()
    except Exception:
        conn.rollback()  # keep the connection usable for the next statement
        raise
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
            _na_to_none(getattr(row, "pm25", None)),
            _na_to_none(getattr(row, "temperature_c", None)),
            _na_to_none(getattr(row, "humidity_pct", None)),
            _na_to_none(getattr(row, "pm1", None)),
            _na_to_none(getattr(row, "pm10", None)),
            _na_to_none(getattr(row, "co2", None)),
            _na_to_none(getattr(row, "tvoc", None)),
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
    try:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, query, values, page_size=1000)
        conn.commit()
    except Exception:
        conn.rollback()  # abort cleanly so the next sensor's upsert can proceed
        raise
    logger.info("upsert_measurements (%s): committed %d rows.", source, len(values))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_manila(manila_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load + normalize all Manila AirGradient CSVs from manila_dir."""
    if not manila_dir.exists():
        logger.warning(
            "MANILA_DIR %s does not exist — skipping Manila ingest. "
            "Point MANILA_DIR at the folder holding your AirGradient CSV exports.",
            manila_dir.resolve(),
        )
        return pd.DataFrame(), pd.DataFrame()

    csv_paths = sorted(manila_dir.glob("*.csv"))
    if not csv_paths:
        logger.warning("MANILA_DIR %s has no *.csv files — skipping Manila ingest.", manila_dir.resolve())
        return pd.DataFrame(), pd.DataFrame()

    frames: list[pd.DataFrame] = []
    loc_frames: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        # e.g. "cubao-merged.csv" -> "cubao", "SDG-1HR-clean.csv" -> "SDG"
        location_hint = csv_path.stem.split("-")[0].split("_")[0]
        try:
            meas, locs = normalize_manila_csv(csv_path, location_hint=location_hint)
            frames.append(meas)
            loc_frames.append(locs)
            logger.info("Loaded Manila CSV %s (%d rows).", csv_path.name, len(meas))
        except Exception as exc:
            logger.warning("Skipping %s - %s", csv_path, exc)

    measurements = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    locations = pd.DataFrame()
    if loc_frames:
        locations = pd.concat(loc_frames, ignore_index=True).drop_duplicates(subset=["location_key"])
    return measurements, locations


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    dsn = resolve_pg_dsn()
    manila_dir = Path(os.environ.get("MANILA_DIR", "data/manila"))

    fallback_start = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    end = datetime.now(timezone.utc)
    logger.info(
        "Lookback %d days (from %s), rate budget %d/min & %d/hr, %d workers.",
        LOOKBACK_DAYS, fallback_start.date(), RPM, RPH, MAX_WORKERS,
    )

    session = get_session()
    verify_credentials(session)  # fail fast on a bad key before any long work

    manila_measurements, manila_locations = load_manila(manila_dir)

    with psycopg2.connect(dsn) as conn:
        watermarks = fetch_watermarks(conn)

        locations = _apply_shard(fetch_locations())
        openaq_locations_df = build_openaq_locations(locations)

        logger.info("openaq_locations_df rows: %d", len(openaq_locations_df))

        upsert_locations(conn, openaq_locations_df)

        # Measurements are stream-written to DB inside build_openaq_measurements.
        build_openaq_measurements(locations, watermarks, fallback_start, end, conn)

        # Manila CSVs are local + tiny — ingest only on shard 0 to avoid redundant
        # (idempotent) double-writes when multiple machines run.
        if SHARD_INDEX == 0:
            logger.info("manila_locations rows: %d, measurements rows: %d",
                        len(manila_locations), len(manila_measurements))
            upsert_locations(conn, manila_locations)
            upsert_measurements(conn, manila_measurements, "manila")

    print("ETL complete.")


if __name__ == "__main__":
    main()
