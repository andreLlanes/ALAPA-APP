import os
import time
import logging
import threading
from datetime import date

import pandas as pd
import psycopg2
import psycopg2.extras
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

AIR_QUALITY_API = "https://air-quality-api.open-meteo.com/v1/air-quality"


DOMAIN = os.environ.get("METEO_AQ_DOMAIN", "cams_global")
HOURLY_VARIABLES = ["pm10"]
SOURCE = "open-meteo"


BBOX = os.environ.get("METEO_BBOX", "14.316284,120.868835,14.781522,121.143494")
BBOX_PAD_DEG = float(os.environ.get("METEO_BBOX_PAD_DEG", "0.4"))


def _padded_bbox(bbox: str, pad: float) -> str:
    min_lat, min_lon, max_lat, max_lon = (float(x) for x in bbox.split(","))
    return f"{min_lat - pad},{min_lon - pad},{max_lat + pad},{max_lon + pad}"

DEFAULT_START_DATE = os.environ.get("METEO_START_DATE", "2021-06-30")
DEFAULT_END_DATE = os.environ.get("METEO_END_DATE", "2026-06-30")
UPSERT_CHUNK_SIZE = int(os.environ.get("METEO_UPSERT_CHUNK_SIZE", "1000"))
REQUEST_TIMEOUT = (15, 180)

MAX_RETRIES = int(os.environ.get("METEO_MAX_RETRIES", "3"))
BACKOFF_FACTOR = float(os.environ.get("METEO_BACKOFF_FACTOR", "1"))

_retry = Retry(
    total=MAX_RETRIES,
    backoff_factor=BACKOFF_FACTOR,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    raise_on_status=False,
)
SESSION = requests.Session()
_adapter = HTTPAdapter(max_retries=_retry)
SESSION.mount("https://", _adapter)
SESSION.mount("http://", _adapter)

RPM = int(os.environ.get("METEO_RPM", "550"))
RPH = int(os.environ.get("METEO_RPH", "4900"))


class RateLimiter:
    """Process-wide minimum spacing between requests."""

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


def create_table_if_needed(conn) -> None:
    """Standalone grid table. latitude/longitude identify the CAMS cell centre."""
    ddl = """
        CREATE TABLE IF NOT EXISTS openaq.openmeteo_pm10_grid (
            grid_key       text        NOT NULL,
            latitude       double precision NOT NULL,
            longitude      double precision NOT NULL,
            timestamp_utc  timestamptz NOT NULL,
            pm10           double precision,
            source         text        NOT NULL DEFAULT 'open-meteo',
            inserted_at    timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (grid_key, timestamp_utc)
        )
    """
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()
    logger.info("Ensured openaq.openmeteo_pm10_grid exists.")


def fetch_grid_pm10(start_date: date, end_date: date) -> list[dict]:
    """One bounding-box request. Returns a list of per-cell responses.

    With a bounding box, Open-Meteo returns a JSON ARRAY (one object per grid cell), each
    carrying its own latitude/longitude and hourly block.
    """
    query_bbox = _padded_bbox(BBOX, BBOX_PAD_DEG)
    params = {
        "bounding_box": query_bbox,
        "domains": DOMAIN,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "GMT",
    }
    logger.info("Querying padded bbox %s (true bbox %s, pad %.2f deg)", query_bbox, BBOX, BBOX_PAD_DEG)
    RATE_LIMITER.acquire()
    resp = SESSION.get(AIR_QUALITY_API, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    # Bounding-box responses are a list; a single-point response is a dict. Normalize to list.
    return data if isinstance(data, list) else [data]


def build_cell_df(cell: dict) -> pd.DataFrame:
    lat = cell.get("latitude")
    lon = cell.get("longitude")
    hourly = cell.get("hourly")
    if lat is None or lon is None or not hourly or "time" not in hourly or "pm10" not in hourly:
        return pd.DataFrame()

    df = pd.DataFrame({"timestamp_utc": hourly["time"], "pm10": hourly["pm10"]})
    if df.empty:
        return df
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["latitude"] = lat
    df["longitude"] = lon
    # Deterministic key from the cell centre so re-runs update the same cell.
    df["grid_key"] = f"openmeteo_grid:{lat:.4f}_{lon:.4f}"
    return df


def _na_to_none(value):
    try:
        if value is None or pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def upsert_grid_pm10(conn, df: pd.DataFrame) -> None:
    if df.empty:
        logger.info("No grid rows to upsert.")
        return

    rows = [
        (
            row.grid_key,
            float(row.latitude),
            float(row.longitude),
            row.timestamp_utc,
            _na_to_none(getattr(row, "pm10", None)),
            SOURCE,
        )
        for row in df.itertuples(index=False)
    ]

    query = """
        INSERT INTO openaq.openmeteo_pm10_grid
            (grid_key, latitude, longitude, timestamp_utc, pm10, source)
        VALUES %s
        ON CONFLICT (grid_key, timestamp_utc)
        DO UPDATE SET
            pm10        = EXCLUDED.pm10,
            latitude    = EXCLUDED.latitude,
            longitude   = EXCLUDED.longitude,
            source      = EXCLUDED.source,
            inserted_at = now()
    """
    with conn.cursor() as cur:
        for i in range(0, len(rows), UPSERT_CHUNK_SIZE):
            chunk = rows[i : i + UPSERT_CHUNK_SIZE]
            psycopg2.extras.execute_values(cur, query, chunk, page_size=UPSERT_CHUNK_SIZE)
    conn.commit()
    logger.info("Committed %d rows for %s.", len(rows), df["grid_key"].iloc[0])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    start_date = date.fromisoformat(DEFAULT_START_DATE)
    end_date = date.fromisoformat(DEFAULT_END_DATE)
    logger.info("Bounding box %s on domain %s, %s -> %s", BBOX, DOMAIN, start_date, end_date)

    dsn = resolve_pg_dsn()
    with psycopg2.connect(dsn) as conn:
        create_table_if_needed(conn)

        try:
            cells = fetch_grid_pm10(start_date, end_date)
        except requests.exceptions.RequestException as exc:
            logger.error("Bounding-box request failed: %s", exc)
            return

        logger.info("Open-Meteo returned %d grid cell(s) for the padded Manila bbox.", len(cells))

        captured = []
        for cell in cells:
            df = build_cell_df(cell)
            if df.empty:
                logger.warning("Empty/invalid cell skipped: %s", cell.get("latitude"))
                continue
            logger.info(
                "Cell %s (%.4f, %.4f): %d hours, %s -> %s",
                df["grid_key"].iloc[0], df["latitude"].iloc[0], df["longitude"].iloc[0],
                len(df), df["timestamp_utc"].min(), df["timestamp_utc"].max(),
            )
            upsert_grid_pm10(conn, df)
            captured.append((df["grid_key"].iloc[0], df["latitude"].iloc[0], df["longitude"].iloc[0]))

        logger.info("Captured %d distinct CAMS cell(s):", len(captured))
        for key, lat, lon in captured:
            logger.info("  %s at (%.4f, %.4f)", key, lat, lon)

    print("PM10 grid backfill complete.")


if __name__ == "__main__":
    main()