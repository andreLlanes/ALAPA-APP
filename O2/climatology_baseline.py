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
from typing import Optional

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(_REPO_ROOT, ".env"))

DEFAULT_SOURCE_TABLE = "openaq.merged_clean"
DEFAULT_OUTPUT_TABLE = "openaq.climatology_baseline"
LOOKBACK_H = 72
HORIZON_H = 72
WINDOW_H = LOOKBACK_H + HORIZON_H
MIN_COMPLETENESS = 0.90


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


def filter_complete_stations(
    frame: pd.DataFrame, min_completeness: float = MIN_COMPLETENESS
) -> pd.DataFrame:
    """Keep stations with enough hourly observations over their full lifespan."""
    if not 0 < min_completeness <= 1:
        raise ValueError("min_completeness must be between 0 and 1.")

    station_cols = ["city", "location_key"]
    coverage = (
        frame.groupby(station_cols)
        .agg(
            first_timestamp=("timestamp_utc", "min"),
            last_timestamp=("timestamp_utc", "max"),
            observed_hours=("timestamp_utc", "nunique"),
        )
        .reset_index()
    )
    lifespan_hours = (
        (coverage["last_timestamp"] - coverage["first_timestamp"])
        / pd.Timedelta(hours=1)
    ) + 1
    coverage["completeness"] = coverage["observed_hours"] / lifespan_hours
    keep = coverage.loc[coverage["completeness"] >= min_completeness, station_cols]
    return frame.merge(keep, on=station_cols, how="inner")


def chronological_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict[str, pd.DataFrame]:
    """Split all stations at shared chronological boundaries without shuffling."""
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Split fractions must be positive.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must leave test data.")

    timestamps = np.sort(frame["timestamp_utc"].drop_duplicates().to_numpy())
    if len(timestamps) < 3:
        raise ValueError("At least three distinct timestamps are required to split data.")
    train_end = timestamps[min(int(len(timestamps) * train_fraction), len(timestamps) - 2)]
    validation_end = timestamps[
        min(int(len(timestamps) * (train_fraction + validation_fraction)), len(timestamps) - 1)
    ]
    return {
        "train": frame[frame["timestamp_utc"] < train_end].copy(),
        "validation": frame[
            (frame["timestamp_utc"] >= train_end)
            & (frame["timestamp_utc"] < validation_end)
        ].copy(),
        "test": frame[frame["timestamp_utc"] >= validation_end].copy(),
    }


def compute_climatology(frame: pd.DataFrame) -> pd.DataFrame:
    """Fit the climatology lookup table by station and hour-of-day."""
    frame = frame.copy()
    if "hour_of_day" not in frame.columns:
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
        frame["hour_of_day"] = frame["timestamp_utc"].dt.hour
    climatology = (
        frame.groupby(["city", "location_key", "hour_of_day"], as_index=False)["pm25"]
        .mean()
        .rename(columns={"pm25": "climatology_pm25"})
        .sort_values(["city", "location_key", "hour_of_day"])
        .reset_index(drop=True)
    )
    return climatology


def evaluate_climatology(
    frame: pd.DataFrame,
    climatology: pd.DataFrame,
    period_start: pd.Timestamp,
    period_end: Optional[pd.Timestamp] = None,
    lookback: int = LOOKBACK_H,
    horizon: int = HORIZON_H,
) -> dict[str, float]:
    """Evaluate complete 72-hour forecasts using the same 144-hour geometry."""
    if period_end is None:
        period_end = frame["timestamp_utc"].max() + pd.Timedelta(hours=1)

    y_true = []
    y_pred = []
    window_count = 0
    for (_, _), station in frame.groupby(["city", "location_key"], sort=False):
        station = station.sort_values("timestamp_utc").reset_index(drop=True)
        times = station["timestamp_utc"].to_numpy()
        values = station["pm25"].to_numpy(dtype=float)
        for start in range(0, len(station) - lookback - horizon + 1):
            target_start = start + lookback
            target_end = target_start + horizon
            target_times = times[target_start:target_end]
            if target_times[0] < period_start or target_times[-1] >= period_end:
                continue
            if not np.all(np.diff(times[start:target_end]) == np.timedelta64(1, "h")):
                continue
            observed = values[target_start:target_end]
            if np.isnan(observed).any():
                continue
            targets = pd.DataFrame({
                "city": station.loc[target_start:target_end - 1, "city"].to_numpy(),
                "location_key": station.loc[target_start:target_end - 1, "location_key"].to_numpy(),
                "timestamp_utc": pd.to_datetime(target_times, utc=True),
            })
            targets["hour_of_day"] = targets["timestamp_utc"].dt.hour
            predictions = targets.merge(
                climatology,
                on=["city", "location_key", "hour_of_day"],
                how="left",
            )["climatology_pm25"].to_numpy()
            if np.isnan(predictions).any():
                continue
            y_true.extend(observed)
            y_pred.extend(predictions)
            window_count += 1

    if not y_true:
        raise ValueError("No complete forecast windows found for the evaluation period.")
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    errors = actual - predicted
    denominator = np.sum(
        (np.abs(predicted - actual.mean()) + np.abs(actual - actual.mean())) ** 2
    )
    return {
        "windows": float(window_count),
        "observations": float(len(actual)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mae": float(np.mean(np.abs(errors))),
        "ioa": float(1 - np.sum(errors ** 2) / denominator) if denominator else float("nan"),
        "r2": float(1 - np.sum(errors ** 2) / np.sum((actual - actual.mean()) ** 2))
        if np.any(actual != actual.mean()) else float("nan"),
    }


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
    parser.add_argument("--start-date", help="Inclusive UTC study-period start, e.g. 2023-01-01.")
    parser.add_argument("--end-date", help="Exclusive UTC study-period end, e.g. 2026-01-01.")
    parser.add_argument("--min-completeness", type=float, default=MIN_COMPLETENESS)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--no-evaluate", action="store_true", help="Skip validation and test metrics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    frame = fetch_training_frame(args.source_table, city=args.city)
    if args.start_date:
        frame = frame[frame["timestamp_utc"] >= pd.Timestamp(args.start_date, tz="UTC")]
    if args.end_date:
        frame = frame[frame["timestamp_utc"] < pd.Timestamp(args.end_date, tz="UTC")]
    frame = filter_complete_stations(frame, args.min_completeness)
    if frame.empty:
        raise ValueError("No data remains after the study-period and completeness filters.")

    splits = chronological_split(frame, args.train_fraction, args.validation_fraction)
    climatology = compute_climatology(splits["train"])

    if not args.no_persist:
        persist_climatology(climatology, args.output_table)
        print(f"Saved climatology baseline to {args.output_table}.")

    print(f"Rows={len(climatology):,}; cities={sorted(climatology['city'].unique().tolist())}")
    print(climatology.head(10).to_string(index=False))
    if not args.no_evaluate:
        for name in ("validation", "test"):
            period = splits[name]
            metrics = evaluate_climatology(
                frame,
                climatology,
                period["timestamp_utc"].min(),
                period["timestamp_utc"].max() + pd.Timedelta(hours=1),
            )
            print(f"{name.title()} metrics: " + ", ".join(
                f"{key}={value:.4f}" for key, value in metrics.items()
            ))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI error handling
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
