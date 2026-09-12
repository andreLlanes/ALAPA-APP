"""Load ECMWF IFS (9 km) meteorology into Postgres for the three study cities.

Source is the Open-Meteo Historical Weather API (/v1/archive, models=ecmwf_ifs):
the 9 km IFS archive (not ERA5, not ecmwf_ifs025), hourly, 2017 to present. Each
city bounding box is sampled on a coordinate lattice; Open-Meteo snaps each
request point to the nearest native cell and returns that cell's true center,
which is what gets stored (deduplicated, so points resolving to the same cell
collapse to one).

Fetching and storing happen per (city, coord-batch, date-chunk) and commit
immediately; each completed unit is appended to a text ledger, so a stopped run
resumes by skipping ledger entries. A sliding-window rate limiter enforces the
per-minute/hour/day call caps. Boundary-layer height is excluded and there is no
location_key (grid cells are keyed by city/latitude/longitude).

Tuning and table names are constants below; only the Postgres connection comes
from the environment (PG_DSN or PG_HOST/PG_DB/PG_USER/PG_PASSWORD).
"""

import os
import csv
import time
from datetime import date, timedelta

import pandas as pd
import psycopg2
import psycopg2.extras
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from dotenv import load_dotenv
load_dotenv()

ARCHIVE_API_BASE = "https://archive-api.open-meteo.com/v1/archive"
IFS_MODEL = "ecmwf_ifs"

GRID_STEP_DEG = 0.1
COORDS_PER_REQUEST = 8
CHUNK_DAYS = 365
UPSERT_CHUNK_SIZE = 1000
MAX_RETRIES = 8
BACKOFF_FACTOR = 1
MAX_SLEEP_ON_RETRY_AFTER = 120
RATE_LIMIT_PER_MIN = 600
RATE_LIMIT_PER_HOUR = 5000
RATE_LIMIT_PER_DAY = 10000
REQUEST_TIMEOUT = (15, 300)
TABLE = "openaq.ifs_met"
LEDGER_PATH = "ifs_progress.txt"

# bbox is (min_lat, min_lon, max_lat, max_lon). LA longitudes are West (negative).
CITY_BBOXES = {
    "Los Angeles":  (33.70,  -118.67, 34.34,  -118.15),
    "Bangkok":      (13.48,   100.32, 13.97,   100.94),
    "Metro Manila": (14.34,   120.93, 14.78,   121.15),
}

CITY_PERIODS = {
    "Los Angeles":  (date(2017, 1, 1), date(2026, 6, 30)),
    "Bangkok":      (date(2017, 1, 1), date(2026, 6, 30)),
    "Metro Manila": (date(2023, 9, 6), date(2026, 6, 30)),
}

# Open-Meteo variable name -> stored column name.
RENAMES = {
    "temperature_2m": "temperature_c",
    "relative_humidity_2m": "humidity_pct",
    "wind_speed_10m": "wind_speed_ms",
    "wind_gusts_10m": "wind_gusts_ms",
    "wind_direction_10m": "wind_dir_deg",
    "surface_pressure": "surface_pressure_hpa",
    "precipitation": "precipitation_mm",
}
HOURLY_VARIABLES = list(RENAMES.keys())

# Expected units per requested variable; a mismatch aborts the parse.
EXPECTED_UNITS = {
    "temperature_2m": "°C",
    "relative_humidity_2m": "%",
    "wind_speed_10m": "m/s",
    "wind_gusts_10m": "m/s",
    "wind_direction_10m": "°",
    "surface_pressure": "hPa",
    "precipitation": "mm",
}


class RateLimited(Exception):
    """Raised on a 429 whose Retry-After exceeds the wait cap; stops the run."""


class RateLimiter:
    """Sliding-window limiter enforcing per-minute, per-hour, and per-day caps."""

    WINDOWS = (
        (60, RATE_LIMIT_PER_MIN, "minute"),
        (3600, RATE_LIMIT_PER_HOUR, "hour"),
        (86400, RATE_LIMIT_PER_DAY, "day"),
    )

    def __init__(self):
        self.calls = []

    def acquire(self):
        """Block until a request may be sent without breaching any window."""
        while True:
            now = time.monotonic()
            horizon = now - self.WINDOWS[-1][0]
            self.calls = [t for t in self.calls if t > horizon]
            wait = 0.0
            label = None
            for span, cap, name in self.WINDOWS:
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


LIMITER = RateLimiter()

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


def generate_grid_centers(bbox):
    """Return the (lat, lon) lattice sampling a bbox at GRID_STEP_DEG spacing."""
    min_lat, min_lon, max_lat, max_lon = bbox
    pts = []
    lat = min_lat
    while lat <= max_lat + 1e-9:
        lon = min_lon
        while lon <= max_lon + 1e-9:
            pts.append((round(lat, 4), round(lon, 4)))
            lon += GRID_STEP_DEG
        lat += GRID_STEP_DEG
    return pts


def _build_params(lats, lons, start_date, end_date):
    """Build the archive-API query params for one multi-coordinate request."""
    return {
        "latitude": ",".join(f"{v:.4f}" for v in lats),
        "longitude": ",".join(f"{v:.4f}" for v in lons),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(HOURLY_VARIABLES),
        "models": IFS_MODEL,
        "timezone": "UTC",
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
    }


def _validate_units(hourly_units):
    """Raise if the response's units differ from EXPECTED_UNITS."""
    if not hourly_units:
        raise ValueError("Open-Meteo response missing hourly_units metadata.")
    mism = [f"{v}: expected {exp}, got {hourly_units.get(v)}"
            for v, exp in EXPECTED_UNITS.items() if hourly_units.get(v) != exp]
    if mism:
        raise ValueError("Unexpected IFS units: " + "; ".join(mism))


def _parse_location(obj):
    """Turn one location's response object into a renamed DataFrame.

    Stores the grid-cell center coordinates Open-Meteo returns, not the requested
    point. Returns an empty frame if the object carries no hourly series.
    """
    hourly = obj.get("hourly")
    if not hourly or "time" not in hourly:
        return pd.DataFrame()
    _validate_units(obj.get("hourly_units", {}))
    df = pd.DataFrame(hourly).rename(columns={"time": "timestamp_utc", **RENAMES})
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["latitude"] = float(obj["latitude"])
    df["longitude"] = float(obj["longitude"])
    return df


def fetch_batch(lats, lons, start_date, end_date):
    """Fetch one coord-batch/date-chunk and return per-location DataFrames.

    Retries transport and 5xx errors with backoff; a short 429 Retry-After is
    waited out, a long one raises RateLimited; any other 4xx raises immediately.
    """
    params = _build_params(lats, lons, start_date, end_date)
    attempt = 0
    while True:
        try:
            LIMITER.acquire()
            resp = SESSION.get(ARCHIVE_API_BASE, params=params, timeout=REQUEST_TIMEOUT)
        except requests.exceptions.RequestException as exc:
            attempt += 1
            if attempt >= MAX_RETRIES:
                raise
            wait = BACKOFF_FACTOR * (2 ** attempt)
            print(f"  transport error (attempt {attempt}/{MAX_RETRIES}): {exc}; "
                  f"retrying in {wait:.0f}s")
            time.sleep(wait)
            continue

        try:
            payload = resp.json()
        except ValueError:
            payload = None

        if resp.status_code == 200 and payload is not None:
            break

        reason = payload.get("reason") if isinstance(payload, dict) else None
        detail = reason or f"HTTP {resp.status_code}: {resp.text[:200]}"

        if 400 <= resp.status_code < 500 and resp.status_code != 429:
            raise ValueError(f"Open-Meteo rejected request ({resp.status_code}): {detail}")

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

        attempt += 1
        if attempt >= MAX_RETRIES:
            raise ValueError(f"Open-Meteo failed after {MAX_RETRIES} attempts: {detail}")
        wait = BACKOFF_FACTOR * (2 ** attempt)
        print(f"  server error {resp.status_code} (attempt {attempt}/{MAX_RETRIES}): "
              f"{detail}; retrying in {wait:.0f}s")
        time.sleep(wait)

    if isinstance(payload, dict) and payload.get("error"):
        raise ValueError(f"Open-Meteo error: {payload.get('reason')}")

    objs = payload if isinstance(payload, list) else [payload]
    return [f for f in (_parse_location(o) for o in objs) if not f.empty]


def _date_chunks(start_date, end_date):
    """Yield inclusive (start, end) sub-windows of at most CHUNK_DAYS."""
    if CHUNK_DAYS <= 0:
        yield start_date, end_date
        return
    cur = start_date
    while cur <= end_date:
        yield cur, min(cur + timedelta(days=CHUNK_DAYS - 1), end_date)
        cur = cur + timedelta(days=CHUNK_DAYS)


def load_ledger():
    """Return the set of completed (city, batch_index, chunk_start, chunk_end)."""
    done = set()
    if not os.path.exists(LEDGER_PATH):
        return done
    with open(LEDGER_PATH, newline="") as f:
        for row in csv.reader(f):
            if len(row) == 4:
                done.add(tuple(row))
    return done


def append_ledger(city, batch_index, chunk_start, chunk_end):
    """Record one completed (city, coord-batch, date-chunk) in the ledger."""
    with open(LEDGER_PATH, "a", newline="") as f:
        csv.writer(f).writerow([city, batch_index, chunk_start, chunk_end])


def fetch_and_store_city(conn, city, bbox, start_date, end_date, done):
    """Fetch a city's grid over its period per (coord-batch, date-chunk) and store.

    Skips ledger-recorded units and deduplicates points that snap to the same
    grid cell before upserting each chunk.
    """
    request_pts = generate_grid_centers(bbox)
    print(f"{city}: {len(request_pts)} request points "
          f"({start_date}..{end_date}) at {GRID_STEP_DEG} deg")

    city_rows = 0
    for i in range(0, len(request_pts), COORDS_PER_REQUEST):
        block = request_pts[i:i + COORDS_PER_REQUEST]
        lats = [p[0] for p in block]
        lons = [p[1] for p in block]
        for c_start, c_end in _date_chunks(start_date, end_date):
            key = (city, str(i), c_start.isoformat(), c_end.isoformat())
            tag = f"  coords {i}..{i+len(block)-1}  {c_start}..{c_end}"
            if key in done:
                print(tag + "  [skip: already stored]")
                continue
            print(tag)
            frames = fetch_batch(lats, lons, c_start, c_end)
            df = pd.DataFrame()
            if frames:
                df = pd.concat(frames, ignore_index=True)
                df["city"] = city
                df = df.drop_duplicates(["latitude", "longitude", "timestamp_utc"])
            store_chunk(conn, df)
            append_ledger(city, i, c_start.isoformat(), c_end.isoformat())
            done.add(key)
            city_rows += len(df)

    print(f"{city}: done ({city_rows} rows fetched/committed this run).")


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


def create_table_if_needed(conn):
    """Create the ifs_met table if it does not already exist."""
    ddl = f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            city text NOT NULL,
            latitude double precision NOT NULL,
            longitude double precision NOT NULL,
            timestamp_utc timestamptz NOT NULL,
            temperature_c double precision,
            humidity_pct double precision,
            wind_speed_ms double precision,
            wind_gusts_ms double precision,
            wind_dir_deg double precision,
            surface_pressure_hpa double precision,
            precipitation_mm double precision,
            source text NOT NULL DEFAULT 'open-meteo:ecmwf_ifs',
            inserted_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (city, latitude, longitude, timestamp_utc)
        )
    """
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()
    print(f"Ensured {TABLE} exists.")


def _na_to_none(v):
    """Map pandas NA/NaN to None so psycopg2 writes SQL NULL."""
    try:
        if v is None or pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def store_chunk(conn, df):
    """Upsert one chunk's grid-cell rows into the ifs_met table, chunked."""
    if df.empty:
        conn.commit()
        print("  committed 0 rows")
        return
    required = {"city", "latitude", "longitude", "timestamp_utc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rows = [
        (
            r.city, float(r.latitude), float(r.longitude), r.timestamp_utc,
            _na_to_none(getattr(r, "temperature_c", None)),
            _na_to_none(getattr(r, "humidity_pct", None)),
            _na_to_none(getattr(r, "wind_speed_ms", None)),
            _na_to_none(getattr(r, "wind_gusts_ms", None)),
            _na_to_none(getattr(r, "wind_dir_deg", None)),
            _na_to_none(getattr(r, "surface_pressure_hpa", None)),
            _na_to_none(getattr(r, "precipitation_mm", None)),
            "open-meteo:ecmwf_ifs",
        )
        for r in df.itertuples(index=False)
    ]

    query = f"""
        INSERT INTO {TABLE} (
            city, latitude, longitude, timestamp_utc,
            temperature_c, humidity_pct, wind_speed_ms, wind_gusts_ms,
            wind_dir_deg, surface_pressure_hpa, precipitation_mm, source
        ) VALUES %s
        ON CONFLICT (city, latitude, longitude, timestamp_utc)
        DO UPDATE SET
            temperature_c        = EXCLUDED.temperature_c,
            humidity_pct         = EXCLUDED.humidity_pct,
            wind_speed_ms        = EXCLUDED.wind_speed_ms,
            wind_gusts_ms        = EXCLUDED.wind_gusts_ms,
            wind_dir_deg         = EXCLUDED.wind_dir_deg,
            surface_pressure_hpa = EXCLUDED.surface_pressure_hpa,
            precipitation_mm     = EXCLUDED.precipitation_mm,
            source               = EXCLUDED.source,
            inserted_at          = now()
    """
    with conn.cursor() as cur:
        for i in range(0, len(rows), UPSERT_CHUNK_SIZE):
            psycopg2.extras.execute_values(
                cur, query, rows[i:i + UPSERT_CHUNK_SIZE], page_size=UPSERT_CHUNK_SIZE)
    conn.commit()
    print(f"  committed {len(rows)} rows")


def main():
    """Run the IFS backfill for every city, resuming from the ledger."""
    dsn = resolve_pg_dsn()
    done = load_ledger()
    print(f"Loaded {len(done)} completed (city, batch, chunk) entries from {LEDGER_PATH}.")
    with psycopg2.connect(dsn) as conn:
        create_table_if_needed(conn)
        try:
            for city, bbox in CITY_BBOXES.items():
                start_date, end_date = CITY_PERIODS[city]
                fetch_and_store_city(conn, city, bbox, start_date, end_date, done)
        except RateLimited as e:
            print(f"\nStopped: Open-Meteo rate limit hit ({e}).")
            print("Completed chunks are committed and logged; re-run to resume.")
            return
    print("Done.")


if __name__ == "__main__":
    main()