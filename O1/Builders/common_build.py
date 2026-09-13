"""Shared helpers for the dataset builders.

Reads the merged tables (openaq.merged_clean / openaq.merged_masked) one station
at a time so no whole city is ever held in memory, which is required for building
LA and Bangkok on a small machine. Window arrays are written as float32 shards,
one file per station, under Outputs/<model>/<citySlug>/<source>/.

Normalization is not applied here: builders emit raw windows plus metadata so
train-only normalization can happen at training time without leakage. Only the
Postgres connection comes from the environment (PG_DSN or
PG_HOST/PG_DB/PG_USER/PG_PASSWORD).
"""

import os
import sys

import pandas as pd
import psycopg2

from dotenv import load_dotenv

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_PIPELINE_ROOT = os.path.dirname(_HERE)

load_dotenv(os.path.join(_REPO_ROOT, ".env"))

for _p in (os.path.join(_PIPELINE_ROOT, "common"), _HERE, os.path.join(_HERE, "Masking")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CITY_TABLES = {
    "clean": "openaq.merged_clean",
    "masked": "openaq.merged_masked",
}

# Full city name (used in SQL) -> short slug (used in output paths).
CITY_SLUG = {
    "Metro Manila": "MM",
    "Bangkok": "BK",
    "Los Angeles": "LA",
}
ALL_CITIES = list(CITY_SLUG)
ALL_SOURCES = list(CITY_TABLES)

OUTPUT_ROOT = os.path.join(_PIPELINE_ROOT, "Outputs")

_PG_DSN = None
_CONN = None


def safe_name(location_key):
    """Return a filesystem-safe form of a station key for shard filenames."""
    return location_key.replace(":", "-").replace("/", "-")


def shard_dir(model, city, source):
    """Return (and create) the output folder for one (model, city, source).

    Layout is Outputs/<model>/<citySlug>/<source>/.
    """
    slug = CITY_SLUG.get(city, city.replace(" ", "-"))
    d = os.path.join(OUTPUT_ROOT, model, slug, source)
    os.makedirs(d, exist_ok=True)
    return d


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


def get_conn():
    """Return one reused, self-healing connection for the whole build run."""
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
    _CONN = psycopg2.connect(_PG_DSN)
    return _CONN


def _table(source):
    """Map a source name to its merged table, raising on an unknown source."""
    if source not in CITY_TABLES:
        raise ValueError(f"source must be one of {list(CITY_TABLES)}; got {source!r}")
    return CITY_TABLES[source]


def station_keys(source, city):
    """Return the distinct station keys for a city, driving per-station streaming."""
    q = f"SELECT DISTINCT location_key FROM {_table(source)} WHERE city = %s ORDER BY location_key"
    with get_conn().cursor() as cur:
        cur.execute(q, (city,))
        return [r[0] for r in cur.fetchall()]


def load_station(source, location_key):
    """Return one station's merged rows, ordered by time and windowing-ready."""
    q = f"""
        SELECT * FROM {_table(source)}
        WHERE location_key = %s
        ORDER BY timestamp_utc
    """
    with get_conn().cursor() as cur:
        cur.execute(q, (location_key,))
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


def station_coords(source, city):
    """Return one row per station with its coordinates, for the GNN graph."""
    q = f"""
        SELECT DISTINCT location_key, latitude, longitude
        FROM {_table(source)}
        WHERE city = %s AND latitude IS NOT NULL AND longitude IS NOT NULL
    """
    with get_conn().cursor() as cur:
        cur.execute(q, (city,))
        cols = [d[0] for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)