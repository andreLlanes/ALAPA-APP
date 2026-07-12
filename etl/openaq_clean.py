"""
openaq_clean.py — reads openaq.measurements (raw), applies bounds + flatline
cleaning, then upserts into openaq.measurements_clean.

Run AFTER the SQL schema is applied:
    psql $PG_DSN -f sql/openaq_clean_schema.sql
    python etl/openaq_clean.py

Environment variables: same as openaq_etl.py
    PG_DSN  or  PG_HOST / PG_PORT / PG_DB / PG_USER / PG_PASSWORD
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PARAMS = ["pm25", "pm10", "temperature_c", "humidity_pct"]

BOUNDS: dict[str, tuple[float, float]] = {
    "pm25":         (0,    500),
    "pm10":         (0,    500),
    "temperature_c":(-10,   60),
    "humidity_pct": (0,    100),
}

FLATLINE_MAX_RUN = 3   # ≥3 identical consecutive hourly values → NULL

# ---------------------------------------------------------------------------
# Helpers (copied / adapted from openaq_etl.py)
# ---------------------------------------------------------------------------

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


def _na_to_none(value):
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


# ---------------------------------------------------------------------------
# Cleaning logic
# ---------------------------------------------------------------------------

def apply_bounds(df: pd.DataFrame) -> dict[str, int]:
    """NULL individual cells that fall outside physical bounds. Returns null counts per col."""
    nulled: dict[str, int] = {}
    for col, (lo, hi) in BOUNDS.items():
        if col not in df.columns:
            continue
        mask = df[col].notna() & ~df[col].between(lo, hi)
        count = int(mask.sum())
        if count:
            df.loc[mask, col] = np.nan
            nulled[col] = count
    return nulled


def null_flatlines(series: pd.Series, max_run: int = FLATLINE_MAX_RUN) -> tuple[pd.Series, int]:
    """
    Set runs of >= max_run identical consecutive values (rounded to 2 dp) to NaN.
    NaN positions are skipped — flatline detection only compares non-null values.
    Returns (cleaned series, count of newly nulled cells).
    """
    result = series.copy()
    notna_mask = series.notna()
    if notna_mask.sum() == 0:
        return result, 0

    s = series[notna_mask].round(2)
    run_id = (s != s.shift()).cumsum()
    run_len = s.groupby(run_id).transform("size")
    flatline_idx = s.index[run_len >= max_run]

    count = len(flatline_idx)
    result.loc[flatline_idx] = np.nan
    return result, count


def apply_flatlines(df: pd.DataFrame) -> dict[str, int]:
    """Apply flatline NULLing per parameter column. Returns null counts per col."""
    nulled: dict[str, int] = {}
    for col in PARAMS:
        if col not in df.columns:
            continue
        cleaned, count = null_flatlines(df[col])
        if count:
            df[col] = cleaned
            nulled[col] = count
    return nulled


# ---------------------------------------------------------------------------
# DB I/O
# ---------------------------------------------------------------------------

def fetch_locations(conn) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT location_key FROM openaq.locations ORDER BY location_key")
        return [row[0] for row in cur.fetchall()]


def fetch_raw(conn, location_key: str) -> pd.DataFrame:
    query = """
        SELECT location_key, timestamp_utc,
               pm25, pm10, temperature_c, humidity_pct
        FROM   openaq.measurements
        WHERE  location_key = %s
        ORDER  BY timestamp_utc ASC
    """
    with conn.cursor() as cur:
        cur.execute(query, (location_key,))
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=cols)


def upsert_clean(conn, df: pd.DataFrame) -> None:
    if df.empty:
        return

    values = [
        (
            row.location_key,
            row.timestamp_utc,
            _na_to_none(getattr(row, "pm25", None)),
            _na_to_none(getattr(row, "pm10", None)),
            _na_to_none(getattr(row, "temperature_c", None)),
            _na_to_none(getattr(row, "humidity_pct", None)),
        )
        for row in df.itertuples(index=False)
    ]

    # inserted_at is omitted here so the table default (now()) applies on insert.
    query = """
        INSERT INTO openaq.measurements_clean
            (location_key, timestamp_utc,
             pm25, pm10, temperature_c, humidity_pct)
        VALUES %s
        ON CONFLICT (location_key, timestamp_utc) DO UPDATE SET
            pm25          = EXCLUDED.pm25,
            pm10          = EXCLUDED.pm10,
            temperature_c = EXCLUDED.temperature_c,
            humidity_pct  = EXCLUDED.humidity_pct,
            inserted_at   = now()
    """
    try:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, query, values, page_size=1000)
        conn.commit()
    except Exception:
        conn.rollback()
        raise


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

@dataclass
class LocationStats:
    location_key: str
    raw_rows: int = 0
    bounds_nulled: dict[str, int] = field(default_factory=dict)
    flatline_nulled: dict[str, int] = field(default_factory=dict)

    def _fmt(self, d: dict[str, int]) -> str:
        if not d:
            return "none"
        return ", ".join(f"{k}={v}" for k, v in sorted(d.items()))

    def print(self):
        total_bounds   = sum(self.bounds_nulled.values())
        total_flatline = sum(self.flatline_nulled.values())
        print(f"\n[{self.location_key}]  {self.raw_rows:,} rows read")
        print(f"  bounds   nulled ({total_bounds:,} cells): {self._fmt(self.bounds_nulled)}")
        print(f"  flatline nulled ({total_flatline:,} cells): {self._fmt(self.flatline_nulled)}")


def print_summary(all_stats: list[LocationStats]):
    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)
    total_rows     = sum(s.raw_rows for s in all_stats)
    total_bounds   = sum(sum(s.bounds_nulled.values()) for s in all_stats)
    total_flatline = sum(sum(s.flatline_nulled.values()) for s in all_stats)

    # per-param totals
    param_bounds:   dict[str, int] = {}
    param_flatline: dict[str, int] = {}
    for s in all_stats:
        for k, v in s.bounds_nulled.items():
            param_bounds[k] = param_bounds.get(k, 0) + v
        for k, v in s.flatline_nulled.items():
            param_flatline[k] = param_flatline.get(k, 0) + v

    print(f"  Locations processed : {len(all_stats)}")
    print(f"  Total rows          : {total_rows:,}")
    print(f"  Bounds cells nulled : {total_bounds:,}")
    if param_bounds:
        for k, v in sorted(param_bounds.items()):
            print(f"    {k:20s} {v:,}")
    print(f"  Flatline cells nulled: {total_flatline:,}")
    if param_flatline:
        for k, v in sorted(param_flatline.items()):
            print(f"    {k:20s} {v:,}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def clean_location(conn, location_key: str) -> LocationStats:
    stats = LocationStats(location_key=location_key)

    df = fetch_raw(conn, location_key)
    stats.raw_rows = len(df)

    if df.empty:
        print(f"[{location_key}]  no data, skipping.")
        return stats

    # Work on a copy; leave the fetched df untouched
    clean = df.copy()

    stats.bounds_nulled   = apply_bounds(clean)
    stats.flatline_nulled = apply_flatlines(clean)

    upsert_clean(conn, clean)
    stats.print()
    return stats


_SCHEMA_SQL = """
-- Drop the dependent view first so the stale co2 / tvoc columns can be removed
-- (CREATE OR REPLACE VIEW cannot drop columns, and DROP COLUMN would fail while
-- the view still references them).
DROP VIEW IF EXISTS openaq.training_clean;

CREATE TABLE IF NOT EXISTS openaq.measurements_clean (
    location_key   text             NOT NULL REFERENCES openaq.locations(location_key),
    timestamp_utc  timestamptz      NOT NULL,
    pm25           double precision,
    pm10           double precision,
    temperature_c  double precision,
    humidity_pct   double precision,
    inserted_at    timestamptz      NOT NULL DEFAULT now(),
    UNIQUE (location_key, timestamp_utc)
);

-- Migrate tables created before pm1 / co2 / tvoc were dropped from scope.
ALTER TABLE openaq.measurements_clean DROP COLUMN IF EXISTS pm1;
ALTER TABLE openaq.measurements_clean DROP COLUMN IF EXISTS co2;
ALTER TABLE openaq.measurements_clean DROP COLUMN IF EXISTS tvoc;

CREATE INDEX IF NOT EXISTS idx_measurements_clean_loc_ts
    ON openaq.measurements_clean (location_key, timestamp_utc DESC);

CREATE VIEW openaq.training_clean AS
SELECT
    mc.location_key,
    mc.timestamp_utc,
    mc.pm25, mc.pm10,
    mc.temperature_c, mc.humidity_pct,
    CASE l.country_iso
        WHEN 'PH' THEN 'Manila'
        WHEN 'SG' THEN 'Singapore'
        WHEN 'TH' THEN 'Bangkok'
    END                                                    AS city,
    l.country_iso,
    l.source,
    l.latitude,
    l.longitude,
    EXTRACT(hour FROM mc.timestamp_utc AT TIME ZONE
        CASE l.country_iso
            WHEN 'PH' THEN 'Asia/Manila'
            WHEN 'SG' THEN 'Asia/Singapore'
            WHEN 'TH' THEN 'Asia/Bangkok'
        END
    )::int                                                 AS local_hour,
    EXTRACT(dow FROM mc.timestamp_utc AT TIME ZONE
        CASE l.country_iso
            WHEN 'PH' THEN 'Asia/Manila'
            WHEN 'SG' THEN 'Asia/Singapore'
            WHEN 'TH' THEN 'Asia/Bangkok'
        END
    )::int                                                 AS day_of_week
FROM openaq.measurements_clean mc
JOIN openaq.locations           l  USING (location_key);
"""


def ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(_SCHEMA_SQL)
    conn.commit()
    print("Schema ready (measurements_clean + training_clean view).")


def main():
    dsn = resolve_pg_dsn()
    conn = psycopg2.connect(dsn)

    try:
        ensure_schema(conn)
        locations = fetch_locations(conn)
        print(f"Found {len(locations)} location(s) to clean.\n")

        all_stats: list[LocationStats] = []
        for loc in locations:
            stats = clean_location(conn, loc)
            all_stats.append(stats)

        print_summary(all_stats)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
