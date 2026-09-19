#!/usr/bin/env python3
r"""Compute and persist a climatology baseline for PM2.5 forecasting.

The climatology baseline is the station-specific mean observed PM2.5 for each
hour-of-day across the historical training period. For any target timestamp,

yhat_{t+\ell} = \bar{y}_{h(t+\ell)}

where h(t+\ell) is the target hour of day and \bar{y}_h is the historical mean
at that hour-of-day for the station.

This script reads from the merged_clean table created by the O1 pipeline and
stores a compact per-station climatology lookup table under a PostgreSQL schema.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, Optional

import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(_REPO_ROOT, ".env"))

DEFAULT_SOURCE_TABLE = "openaq.merged_clean"
DEFAULT_OUTPUT_TABLE = "openaq.climatology_baseline"


def resolve_pg_dsn() -> str:
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
    """Return a live Postgres connection, reconnecting if needed."""
    conn = psycopg2.connect(resolve_pg_dsn())
    conn.autocommit = False
    return conn


def fetch_training_frame(source_table: str, city: Optional[str] = None) -> pd.DataFrame:
    """Read the historical PM2.5 series used to fit the climatology baseline."""
    query = f"""
        SELECT location_key, city, timestamp_utc, pm25
        FROM {source_table}
        WHERE pm25 IS NOT NULL
    """
    params: tuple = ()

    if city and city.lower() != "all":
        query += " AND city = %s"
        params = (city,)

    query += " ORDER BY city, location_key, timestamp_utc"

    with get_conn() as conn:
        frame = pd.read_sql_query(query, conn, params=params)

    if frame.empty:
        raise ValueError(f"No records found for city={city!r} in {source_table!r}.")

    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame["hour_of_day"] = frame["timestamp_utc"].dt.hour
    return frame


def compute_climatology(frame: pd.DataFrame) -> pd.DataFrame:
    """Fit the climatology lookup table by station and hour-of-day."""
    climatology = (
        frame.groupby(["city", "location_key", "hour_of_day"], as_index=False)["pm25"]
        .mean()
        .rename(columns={"pm25": "climatology_pm25"})
        .sort_values(["city", "location_key", "hour_of_day"])
        .reset_index(drop=True)
    )
    return climatology


def persist_climatology(climatology: pd.DataFrame, output_table: str) -> None:
    """Store the climatology table in Postgres."""
    table_parts = output_table.split(".")
    schema = table_parts[0]
    table_name = table_parts[1] if len(table_parts) > 1 else table_parts[0]

    rows = [
        (row.city, row.location_key, int(row.hour_of_day), float(row.climatology_pm25))
        for row in climatology.itertuples(index=False)
    ]

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"DROP TABLE IF EXISTS {schema}.{table_name};"
            )
            cur.execute(
                f"""
                CREATE TABLE {schema}.{table_name} (
                    city text NOT NULL,
                    location_key text NOT NULL,
                    hour_of_day integer NOT NULL,
                    climatology_pm25 double precision NOT NULL,
                    PRIMARY KEY (city, location_key, hour_of_day)
                );
                """
            )
            psycopg2.extras.execute_values(
                cur,
                f"INSERT INTO {schema}.{table_name} (city, location_key, hour_of_day, climatology_pm25) VALUES %s",
                rows,
            )
        conn.commit()


def apply_climatology(df: pd.DataFrame, climatology: pd.DataFrame) -> pd.DataFrame:
    """Join a forecasting frame to the climatology lookup by station and hour."""
    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["hour_of_day"] = df["timestamp_utc"].dt.hour
    out = df.merge(
        climatology,
        on=["city", "location_key", "hour_of_day"],
        how="left",
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute a climatology baseline for PM2.5 forecasts.")
    parser.add_argument("--city", default="all", help='City name or "all". Example: "Metro Manila"')
    parser.add_argument(
        "--source-table",
        default=DEFAULT_SOURCE_TABLE,
        help="Postgres source table with pm25 and timestamp_utc columns.",
    )
    parser.add_argument(
        "--output-table",
        default=DEFAULT_OUTPUT_TABLE,
        help="Postgres table to store the climatology lookup.",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Compute the climatology in-memory without writing it to Postgres.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    frame = fetch_training_frame(args.source_table, city=args.city)
    climatology = compute_climatology(frame)

    if not args.no_persist:
        persist_climatology(climatology, args.output_table)
        print(f"Saved climatology baseline to {args.output_table}.")

    print(f"Rows={len(climatology):,}; cities={sorted(climatology['city'].unique().tolist())}")
    print(climatology.head(10).to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI error handling
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
