""" Build a city's cache from the database: hourly rows, valid windows, lookback gaps, split, completeness.
    PYTHONPATH=.. python data/build.py --city mm bk
"""

import argparse
import io
import os
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

from Common.schema import (KEY_COL, TIME_COL, PM25_COL, MET_COLS, TIME_COLS,
                           LOOKBACK_H, WINDOW_H)
from Common.split import station_split

CITIES = {"mm": "Metro Manila", "bk": "Bangkok", "la": "Los Angeles"}
TABLE = "openaq.merged_clean"
FEATURES = MET_COLS + TIME_COLS
CACHE_DIR = Path(__file__).resolve().parent / "cache"
EPOCH = pd.Timestamp(0, tz="UTC")


def cache_path(city: str) -> Path:
    """ Return the path of a city's cache file (data/cache/<city>.npz).
    """
    return CACHE_DIR / f"{city}.npz"


def ensure_cache(city: str, rebuild: bool = False) -> Path:
    """ Return the city's cache path, building it from the database first if it is missing
        or if rebuild is set (e.g. after the database changed).
    """
    path = cache_path(city)
    if rebuild or not path.exists():
        print(f"    Building cache from database: {CITIES[city]}")
        CACHE_DIR.mkdir(exist_ok=True)
        np.savez(path, **build_cache(fetch_city_rows(city)))
    return path


def database_url() -> str:
    """ Build the Postgres connection string from .env: PG_DSN, or
        PG_HOST/PG_PORT/PG_DB/PG_USER/PG_PASSWORD.
    """
    env = os.environ
    if env.get("PG_DSN"):
        return env["PG_DSN"]
    return (f"postgresql://{env['PG_USER']}:{env['PG_PASSWORD']}"
            f"@{env['PG_HOST']}:{env.get('PG_PORT', '5432')}/{env['PG_DB']}")


def fetch_city_rows(city: str) -> pd.DataFrame:
    """ Fetch all hourly rows for one city, ordered by station then time.
    """
    load_dotenv()
    cols = [KEY_COL, TIME_COL, PM25_COL, "latitude", "longitude"] + FEATURES
    buf = io.StringIO()
    with closing(psycopg2.connect(database_url())) as conn, conn.cursor() as cur:
        query = cur.mogrify(
            f"SELECT {', '.join(cols)} FROM {TABLE} WHERE city = %s "
            f"ORDER BY {KEY_COL}, {TIME_COL}", (CITIES[city],)).decode()
        # Bulk COPY: much faster than reading row by row.
        cur.copy_expert(f"COPY ({query}) TO STDOUT WITH CSV HEADER", buf)
    buf.seek(0)
    return pd.read_csv(buf)


def nan_bounds(lookback: np.ndarray):
    """ Find the bounds of the gap each hour sits in: for every hour of every lookback row,
        the position of the nearest observed hour before it (prev) and after it (nxt).
    """
    width = lookback.shape[1]
    idx = np.arange(width, dtype=np.int16)
    seen = ~np.isnan(lookback)
    # An observed hour is its own bound; -1 / width mean nothing observed before / after.
    # Carry the last observed position forward, and the next observed position backward.
    prev = np.maximum.accumulate(np.where(seen, idx, np.int16(-1)), axis=1)
    nxt = np.minimum.accumulate(np.where(seen, idx, np.int16(width))[:, ::-1], axis=1)[:, ::-1]
    return prev, nxt


def running_count(flags: np.ndarray) -> np.ndarray:
    """ Count True flags cumulatively, so the count in any row range [a, b) is c[b] - c[a].
    """
    # Lets every window be checked at once instead of looping over windows.
    return np.concatenate([[0], np.cumsum(flags)])


def find_windows(station_of_row, hours, pm, feat) -> np.ndarray:
    """ Find the first row of every valid 144h window (72h lookback + 72h forecast).
    """
    breaks = np.ones(len(hours), bool)  # row r is not followed by the next hour of the same station
    breaks[:-1] = (station_of_row[1:] != station_of_row[:-1]) | (np.diff(hours) != 1)
    breaks_before = running_count(breaks)
    bad_feat_before = running_count(np.isnan(feat).any(1))
    missing_pm_before = running_count(np.isnan(pm))

    # Valid: one station, consecutive hours, complete met/time, every forecast hour observed.
    # Lookback gaps are allowed (interpolated at load time).
    first = np.arange(len(hours) - WINDOW_H + 1)  # every candidate first row
    valid = ((breaks_before[first + WINDOW_H - 1] == breaks_before[first])
             & (bad_feat_before[first + WINDOW_H] == bad_feat_before[first])
             & (missing_pm_before[first + WINDOW_H] == missing_pm_before[first + LOOKBACK_H]))
    return first[valid]


def longest_lookback_gaps(pm, start) -> np.ndarray:
    """ Measure the longest run of missing PM2.5 hours in each window's lookback (0 if none).
        Raise if a lookback is entirely missing.
    """
    gap = np.zeros(len(start), np.int16)
    # Only windows that have a gap are examined.
    missing_before = running_count(np.isnan(pm))
    with_gaps = np.flatnonzero(missing_before[start + LOOKBACK_H] > missing_before[start])
    incomplete = pm[start[with_gaps, None] + np.arange(LOOKBACK_H)]
    prev, nxt = nan_bounds(incomplete)
    if np.any((prev < 0) & (nxt == LOOKBACK_H)):
        raise ValueError("a window has an all-missing lookback")
    # A missing hour's gap length is the distance between its bounds, minus the bounds themselves.
    gap[with_gaps] = np.where(np.isnan(incomplete), nxt - prev - 1, 0).max(1)
    return gap


def station_completeness(station_of_row, hours, pm, n_stations) -> np.ndarray:
    """ Compute each station's completeness: observed PM2.5 hours divided by the hours
        from its first to its last observed hour.
    """
    seen = ~np.isnan(pm)
    span = pd.DataFrame({"s": station_of_row[seen], "h": hours[seen]}).groupby("s")["h"]
    ratio = span.size() / (span.max() - span.min() + 1)
    return ratio.reindex(range(n_stations), fill_value=0.0).to_numpy()  # no observations -> 0


def usable_stations(keys, station, split) -> np.ndarray:
    """ Mark the stations that have at least one train, val and test window after the purge,
        and report the rest as excluded.
    """
    counts = np.zeros((len(keys), 3), int)
    labelled = split >= 0
    np.add.at(counts, (station[labelled], split[labelled]), 1)
    usable = (counts > 0).all(1)
    for key in keys[~usable]:
        print(f"    Excluded Key: {key} - too short for a train/val/test split")
    return usable


def build_cache(df: pd.DataFrame) -> dict:
    """ Turn a city's database rows (ordered by station, then time) into the cache arrays:
        hourly PM2.5 and met/time, station coordinates and completeness, and per window its
        first row, station, origin hour, longest lookback gap and split label.
    """
    keys, station_of_row = np.unique(df[KEY_COL].to_numpy(str), return_inverse=True)
    if np.any(np.diff(station_of_row) < 0):
        raise ValueError("rows must be ordered by station")
    hours = ((pd.to_datetime(df[TIME_COL], utc=True) - EPOCH)
             // pd.Timedelta(hours=1)).to_numpy(np.int64)
    pm = df[PM25_COL].to_numpy(np.float32)
    feat = df[FEATURES].to_numpy(np.float32)

    start = find_windows(station_of_row, hours, pm, feat)
    station = station_of_row[start]
    origin = hours[start + LOOKBACK_H - 1]  # last lookback hour
    split = station_split(station, origin).astype(np.int8)
    first_row = np.unique(station_of_row, return_index=True)[1]

    return {
        "keys": keys,
        "lat": df["latitude"].to_numpy(float)[first_row],
        "lon": df["longitude"].to_numpy(float)[first_row],
        "completeness": station_completeness(station_of_row, hours, pm, len(keys)),
        "usable": usable_stations(keys, station, split),
        "pm": pm,
        "feat": feat,
        "start": start,
        "station": station,
        "origin": origin,
        "gap": longest_lookback_gaps(pm, start),
        "split": split,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", nargs="+", choices=CITIES, required=True)
    cities = ap.parse_args().city
    print(f"[DATA] Rebuilding Cache: {', '.join(cities)}")
    for c in cities:
        ensure_cache(c, rebuild=True)
