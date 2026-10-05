""" Turn a city's hourly rows into windows: valid windows, lookback gaps and station completeness.
"""

import numpy as np
import pandas as pd

from Common.schema import (KEY_COL, TIME_COL, PM25_COL, MET_COLS, TIME_COLS,
                           LOOKBACK_H, WINDOW_H)

FEATURES = MET_COLS + TIME_COLS
EPOCH = pd.Timestamp(0, tz="UTC")

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

def city_arrays(df: pd.DataFrame) -> dict:
    """ Turn a city's rows (ordered by station, then time) into arrays:
        hourly PM2.5, met/time and station, station coordinates and completeness, and per window its
        first row, station, origin hour and longest lookback gap.
    """
    keys, station_of_row = np.unique(df[KEY_COL].to_numpy(str), return_inverse=True)
    if np.any(np.diff(station_of_row) < 0):
        raise ValueError("rows must be ordered by station")
    hours = ((pd.to_datetime(df[TIME_COL], utc=True) - EPOCH)
             // pd.Timedelta(hours=1)).to_numpy(np.int64)
    pm = df[PM25_COL].to_numpy(np.float32)
    feat = df[FEATURES].to_numpy(np.float32)

    start = find_windows(station_of_row, hours, pm, feat)
    first_row = np.unique(station_of_row, return_index=True)[1]

    return {
        "keys": keys,
        "lat": df["latitude"].to_numpy(float)[first_row],
        "lon": df["longitude"].to_numpy(float)[first_row],
        "completeness": station_completeness(station_of_row, hours, pm, len(keys)),
        "pm": pm,
        "feat": feat,
        "row_station": station_of_row,
        "start": start,
        "station": station_of_row[start],
        "origin": hours[start + LOOKBACK_H - 1],  # last lookback hour, hours since epoch
        "gap": longest_lookback_gaps(pm, start),
    }
