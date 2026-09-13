"""Build the merged_clean and merged_masked tables, one city at a time.

Per station, the pipeline runs:
    1. grade merge   - attach is_monitor and station lat/lon to raw PM2.5.
    2. clean         - grade-aware QC (reference monitors skip the flatline rule).
    3. met merge     - match the nearest IFS grid cell and attach its meteorology
                       to the cleaned timeline, so gap-filled hours still get met.
    4. build features - add the six cyclical time encodings.

Two tables are written from the same cleaned frame: merged_masked keeps every
row (long-gap NaNs included, for masking), merged_clean drops rows whose pm25 is
still NaN.

Reads only the raw tables (gs_measurements, gs_stations, ifs_met). Progress is
tracked per (city, station) in a text ledger, and the connection auto-reconnects,
so an interrupted run resumes. Only the Postgres connection comes from the
environment (PG_DSN or PG_HOST/PG_DB/PG_USER/PG_PASSWORD).
"""

import os
import csv
import sys

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

from dotenv import load_dotenv

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_COMMON_DIR = os.path.abspath(os.path.join(_HERE, "..", "common"))

load_dotenv(os.path.join(_REPO_ROOT, ".env"))
sys.path.insert(0, _COMMON_DIR)

from schema import MET_COLS, TIME_COLS, TIME_COL
from clean import clean_one_station
from build_features import build_features

RAW_MEASUREMENTS_TABLE = "openaq.gs_measurements"
RAW_MET_TABLE = "openaq.ifs_met"
STATIONS_TABLE = "openaq.gs_stations"
MERGED_TABLE = "openaq.merged_clean"
MASKED_TABLE = "openaq.merged_masked"
UPSERT_CHUNK_SIZE = 5000
LEDGER_PATH = "merged_progress.txt"
PROVENANCE_PATH = "provenance_log.csv"
PROVENANCE_FIELDS = [
    "city", "location_key", "raw_rows", "nonpositive_dropped", "duplicate_rows",
    "active_range_hours", "gap_hours_inserted", "is_reference",
    "flatline_hours_removed", "short_gap_filled_count", "long_gap_missing_count",
    "final_rows", "final_valid", "grid_latitude", "grid_longitude",
    "stored_rows", "stored_rows_masked",
]

CITIES = ["Los Angeles", "Bangkok", "Metro Manila"]
CITY_COUNTRY = {"Los Angeles": "US", "Bangkok": "TH", "Metro Manila": "PH"}


def load_ledger():
    """Return the set of (city, location_key) pairs already processed."""
    done = set()
    if not os.path.exists(LEDGER_PATH):
        return done
    with open(LEDGER_PATH, newline="") as f:
        for row in csv.reader(f):
            if len(row) == 2:
                done.add((row[0], row[1]))
    return done


def append_ledger(city, location_key):
    """Record one completed station in the ledger."""
    with open(LEDGER_PATH, "a", newline="") as f:
        csv.writer(f).writerow([city, location_key])


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


def create_table_if_needed():
    """Create merged_clean and merged_masked if they do not already exist."""
    conn = get_conn()
    met_ddl = ",\n            ".join(f"{c} double precision" for c in MET_COLS)
    time_ddl = ",\n            ".join(f"{c} double precision" for c in TIME_COLS)
    for table in (MERGED_TABLE, MASKED_TABLE):
        ddl = f"""
            CREATE TABLE IF NOT EXISTS {table} (
                location_key text NOT NULL,
                timestamp_utc timestamptz NOT NULL,
                city text,
                pm25 double precision,
                short_gap_filled boolean,
                long_gap_missing boolean,
                is_monitor boolean,
                latitude double precision,
                longitude double precision,
                grid_latitude double precision,
                grid_longitude double precision,
                {met_ddl},
                {time_ddl},
                inserted_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (location_key, timestamp_utc)
            )
        """
        with conn.cursor() as cur:
            cur.execute(ddl)
    conn.commit()
    print(f"Ensured {MERGED_TABLE} and {MASKED_TABLE} exist.")


def _read_sql(query, params=None):
    """Run a query on the managed connection and return a DataFrame.

    Reads via the raw cursor to avoid pandas' SQLAlchemy-connection warning.
    """
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(query, params)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


def fetch_station_keys(city):
    """Return the distinct station keys for a city, for per-station streaming."""
    query = f"""
        SELECT DISTINCT location_key
        FROM {RAW_MEASUREMENTS_TABLE}
        WHERE city = %s
        ORDER BY location_key
    """
    return [r[0] for r in _read_sql(query, (city,)).itertuples(index=False)]


def fetch_pm25_station(location_key):
    """Return one station's raw PM2.5 rows, ordered by time."""
    query = f"""
        SELECT location_key, timestamp_utc, pm25, city
        FROM {RAW_MEASUREMENTS_TABLE}
        WHERE location_key = %s
        ORDER BY timestamp_utc
    """
    return _read_sql(query, (location_key,))


def fetch_stations(city):
    """Return station metadata (coords, is_monitor) for a city's country."""
    query = f"""
        SELECT location_key, latitude, longitude, is_monitor
        FROM {STATIONS_TABLE}
        WHERE country_iso = %s AND latitude IS NOT NULL AND longitude IS NOT NULL
    """
    return _read_sql(query, (CITY_COUNTRY[city],))


def fetch_grid_cells(city):
    """Return the distinct IFS grid-cell coordinates for a city."""
    query = f"""
        SELECT DISTINCT latitude, longitude
        FROM {RAW_MET_TABLE}
        WHERE city = %s
    """
    return _read_sql(query, (city,))


def fetch_met_cell(city, grid_lat, grid_lon):
    """Return the meteorology time series for a single grid cell."""
    query = f"""
        SELECT latitude, longitude, timestamp_utc,
               temperature_c, humidity_pct, wind_speed_ms, wind_gusts_ms,
               wind_dir_deg, surface_pressure_hpa, precipitation_mm
        FROM {RAW_MET_TABLE}
        WHERE city = %s AND latitude = %s AND longitude = %s
        ORDER BY timestamp_utc
    """
    return _read_sql(query, (city, grid_lat, grid_lon))


def _transform_met_wind(met):
    """Convert wind speed/direction to orthogonal u/v components."""
    met = met.copy()
    theta = np.deg2rad(met["wind_dir_deg"])
    s = met["wind_speed_ms"]
    met["wind_u"] = -s * np.sin(theta)
    met["wind_v"] = -s * np.cos(theta)
    return met.drop(columns=["wind_speed_ms", "wind_dir_deg"])


def _na_to_none(v):
    """Map pandas NA/NaN to None so psycopg2 writes SQL NULL."""
    try:
        if v is None or pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def store_rows(table, df):
    """Upsert a station's rows into the given table, chunked."""
    conn = get_conn()
    if df.empty:
        conn.commit()
        print(f"  committed 0 rows to {table}")
        return
    cols = (["location_key", "timestamp_utc", "city", "pm25",
             "short_gap_filled", "long_gap_missing", "is_monitor",
             "latitude", "longitude", "grid_latitude", "grid_longitude"]
            + MET_COLS + TIME_COLS)
    values = [tuple(_na_to_none(getattr(r, c, None)) for c in cols)
              for r in df.itertuples(index=False)]
    col_sql = ", ".join(cols)
    update_sql = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols
                           if c not in ("location_key", "timestamp_utc"))
    query = f"""
        INSERT INTO {table} ({col_sql})
        VALUES %s
        ON CONFLICT (location_key, timestamp_utc) DO UPDATE SET
            {update_sql}
    """
    try:
        with conn.cursor() as cur:
            for i in range(0, len(values), UPSERT_CHUNK_SIZE):
                psycopg2.extras.execute_values(
                    cur, query, values[i:i + UPSERT_CHUNK_SIZE], page_size=UPSERT_CHUNK_SIZE)
        conn.commit()
    except Exception:
        _safe_rollback(conn)
        raise
    print(f"  committed {len(values)} rows to {table}")


def _match_one_station(lat, lon, cells):
    """Return the (lat, lon) of the grid cell nearest a station (haversine)."""
    cell_lat = np.radians(cells["latitude"].to_numpy())
    cell_lon = np.radians(cells["longitude"].to_numpy())
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    dlat = cell_lat - lat_r
    dlon = cell_lon - lon_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_r) * np.cos(cell_lat) * np.sin(dlon / 2) ** 2
    d = 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    j = int(np.argmin(d))
    return float(cells["latitude"].iloc[j]), float(cells["longitude"].iloc[j])


def process_city(city, done, prov):
    """Stream a city's stations, clean and merge each, and write both tables.

    Skips stations already in ``done``, appends one provenance row per station,
    and records each completed station in the ledger.
    """
    print(f"{city}: streaming stations")
    stations = fetch_stations(city)
    if stations.empty:
        print(f"  no stations; skipping")
        return
    st_lookup = stations.set_index("location_key")
    cells = fetch_grid_cells(city)
    met_cache = {}
    keys = fetch_station_keys(city)
    print(f"  {len(keys)} stations, {len(cells)} grid cells")

    for location_key in keys:
        if (city, location_key) in done:
            continue
        raw = fetch_pm25_station(location_key)
        if raw.empty:
            append_ledger(city, location_key)
            done.add((city, location_key))
            continue

        if location_key in st_lookup.index:
            srow = st_lookup.loc[location_key]
            raw["latitude"] = srow["latitude"]
            raw["longitude"] = srow["longitude"]
            raw["is_monitor"] = srow["is_monitor"]
        else:
            raw["latitude"] = np.nan
            raw["longitude"] = np.nan
            raw["is_monitor"] = np.nan

        cleaned, stats = clean_one_station(raw)
        rec = {"city": city, "location_key": location_key, **stats,
               "grid_latitude": None, "grid_longitude": None,
               "stored_rows": 0, "stored_rows_masked": 0}

        if not cleaned.empty and not cells.empty and pd.notna(cleaned["latitude"].iloc[0]):
            glat, glon = _match_one_station(
                cleaned["latitude"].iloc[0], cleaned["longitude"].iloc[0], cells)
            rec["grid_latitude"], rec["grid_longitude"] = glat, glon
            if (glat, glon) not in met_cache:
                mc = fetch_met_cell(city, glat, glon)
                met_cache[(glat, glon)] = _transform_met_wind(mc) if not mc.empty else mc
            met = met_cache[(glat, glon)]
            cleaned["grid_latitude"] = glat
            cleaned["grid_longitude"] = glon
            if not met.empty:
                met = met.rename(columns={"latitude": "grid_latitude",
                                          "longitude": "grid_longitude"})
                keep = ["grid_latitude", "grid_longitude", TIME_COL] + MET_COLS
                cleaned = cleaned.merge(met[keep],
                                        on=["grid_latitude", "grid_longitude", TIME_COL],
                                        how="left")
            else:
                for c in MET_COLS:
                    cleaned[c] = pd.NA
        else:
            cleaned["grid_latitude"] = pd.NA
            cleaned["grid_longitude"] = pd.NA
            for c in MET_COLS:
                cleaned[c] = pd.NA

        if not cleaned.empty:
            cleaned = build_features(cleaned)
            store_rows(MASKED_TABLE, cleaned)
            clean_rows = cleaned[cleaned["pm25"].notna()]
            store_rows(MERGED_TABLE, clean_rows)
            rec["stored_rows"] = len(clean_rows)
            rec["stored_rows_masked"] = len(cleaned)

        prov.writerow(rec)
        append_ledger(city, location_key)
        done.add((city, location_key))

    print(f"  {city} done")


def main():
    """Process every city, writing both tables and the provenance log."""
    done = load_ledger()
    print(f"Loaded {len(done)} completed (city, station) entries from {LEDGER_PATH}.")
    create_table_if_needed()
    new_log = not os.path.exists(PROVENANCE_PATH)
    with open(PROVENANCE_PATH, "a", newline="") as pf:
        prov = csv.DictWriter(pf, fieldnames=PROVENANCE_FIELDS)
        if new_log:
            prov.writeheader()
        for city in CITIES:
            try:
                process_city(city, done, prov)
            except Exception as exc:
                print(f"{city}: failed, left for a later run: {exc}")
    print(f"Done. Provenance written to {PROVENANCE_PATH}.")


if __name__ == "__main__":
    main()