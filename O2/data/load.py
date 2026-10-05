""" Load a city from the local database copy, build and filter its windows, normalize, and
    gather model inputs.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import config
from Common.schema import TIME_COL, LOOKBACK_H, HORIZON_H, LAG_OFFSETS, ROLL_WINDOWS, ROLL_STATS
from Common.splits import TRAIN, chronological_cuts, partition_labels, describe
from data.database import read_copy
from data.windows import FEATURES, city_arrays, nan_bounds
from utils import runlog

LOOKBACK = np.arange(LOOKBACK_H)
HORIZON = np.arange(LOOKBACK_H, LOOKBACK_H + HORIZON_H)
LOSO_PATH = Path(__file__).resolve().parents[1] / "loso_stations.json"
ROLL_FN = {"mean": np.mean, "max": np.max, "std": np.std}

@dataclass
class Stats:
    """ Hold the normalization fitted on the training split: PM2.5 (shared by input and target)
        and the met/time features.
    """
    pm_mean: float
    pm_std: float
    feat_mean: np.ndarray
    feat_std: np.ndarray

    def inverse_y(self, y: np.ndarray) -> np.ndarray:
        """ Convert normalized PM2.5 back to ug/m^3.
        """
        return y * self.pm_std + self.pm_mean

@dataclass
class Data:
    """ Hold a loaded city: hourly arrays indexed by row, window arrays indexed by window id.
    """
    keys: np.ndarray       # station keys
    lat: np.ndarray
    lon: np.ndarray
    pm: np.ndarray         # (rows,) normalized, NaN where missing
    feat: np.ndarray       # (rows, met+time) normalized
    row_station: np.ndarray  # (rows,) station of each hourly row
    lookback: np.ndarray   # (windows, 72) normalized, interpolated
    start: np.ndarray      # first row of each window
    station: np.ndarray
    origin: np.ndarray     # last lookback hour, datetime64[h] UTC
    split: np.ndarray      # TRAIN / VAL / TEST / PURGED
    active: np.ndarray     # not purged, and kept by ablation
    stats: Stats

    def split_window_ids(self, split: int) -> np.ndarray:
        """ Return the ids of the windows a split (TRAIN, VAL or TEST) trains or scores on:
            the windows labelled with that split that survive ablation.
        """
        return np.flatnonzero(self.active & (self.split == split))

    def gather(self, ids: np.ndarray):
        """ Gather model inputs for the given window ids:
            x_enc [n,72,1+F] (PM2.5 first, then met/time), x_dec [n,72,F], y [n,72].
        """
        first = self.start[ids][:, None]
        x_enc = np.concatenate([self.lookback[ids][..., None], self.feat[first + LOOKBACK]], -1)
        return x_enc, self.feat[first + HORIZON], self.pm[first + HORIZON]

def loso_keys() -> np.ndarray:
    """ Return the keys of the 20 LOSO stations, which the completeness filter never removes
        (none before the folds are selected).
    """
    if not LOSO_PATH.exists():
        return np.array([], dtype=str)
    with open(LOSO_PATH, encoding="utf-8") as f:
        return np.array([fold["location_key"] for fold in json.load(f)["folds"]])

def load_city(city: str, longest_gap: int = None, completeness: int = None,
              ablation: int = 100, block: int = 0, stats: Stats | None = None) -> Data:
    """ Load a city from the local database copy (copying the database first if needed), build its
        windows, filter them by completeness and longest gap, split the remaining windows 70/15/15,
        apply ablation, and normalize with its own training stats or the given (source) stats.
    """
    longest_gap = config.DEFAULT_LONGEST_GAP if longest_gap is None else longest_gap
    completeness = config.DEFAULT_COMPLETENESS if completeness is None else completeness
    abl = f"{ablation}%" + (f" (block {block})" if ablation < 100 else "")
    print(f"[DATA] Building Shards: longest gap = {longest_gap}, "
          f"station completeness = {completeness}%, ablation = {abl}")
    rows = read_copy(city)
    # With MATCH_LA_BKK, Los Angeles starts at Bangkok's first record.
    if city == "la" and config.MATCH_LA_BKK:
        rows = rows[rows[TIME_COL] >= read_copy("bk", columns=[TIME_COL])[TIME_COL].min()]
    arrays = city_arrays(rows)
    # Drop stations below the completeness threshold, except the LOSO stations.
    keep = (arrays["completeness"] * 100 >= completeness) | np.isin(arrays["keys"], loso_keys())
    if not keep.any():
        raise ValueError(f"{city}: no station passes completeness >= {completeness}%")

    # Drop windows with a longer lookback gap, in every split.
    settled = keep[arrays["station"]] & (arrays["gap"] <= longest_gap)
    if not settled.any():
        raise ValueError(f"{city}: no windows left after the filters")
    start, station = arrays["start"][settled], arrays["station"][settled]
    origin = arrays["origin"][settled].astype("datetime64[h]")

    # Split the settled windows 70/15/15 by pooled window count.
    cuts = chronological_cuts(origin)
    split = partition_labels(origin, cuts)
    parts = describe(origin, cuts)
    print(f"    Split: train = {parts['train']['n']}, val = {parts['val']['n']}, "
          f"test = {parts['test']['n']}, purged = {parts['purged']['n']} "
          f"(train end {cuts[0]}, val end {cuts[1]})")

    active = split >= 0
    # Ablation keeps one contiguous block of each station's train windows.
    if ablation < 100:
        active &= (split != TRAIN) | ablation_mask(station, split, ablation, block)
    if not (active & (split == TRAIN)).any():
        raise ValueError(f"{city}: no training windows left after the filters")
    train_before = int((split == TRAIN).sum())

    pm, feat = arrays["pm"], arrays["feat"]
    given_stats = stats
    # Normalize with this city's training stats, or the source city's stats in transfer.
    stats = stats or fit_stats(pm, feat, start[active & (split == TRAIN)])
    pm = (pm - stats.pm_mean) / stats.pm_std
    feat = (feat - stats.feat_mean) / stats.feat_std

    comp, protected = arrays["completeness"], np.isin(arrays["keys"], loso_keys())
    runlog.detail(f"DATA {city}", {
        "stations": f"{len(keep)} total, {int(keep.sum())} kept "
                    f"({int((keep & (comp * 100 < completeness)).sum())} kept only as LOSO stations)",
        "windows": f"{len(arrays['start'])} valid, {int(keep[arrays['station']].sum())} after "
                   f"completeness, {len(start)} after longest gap",
        "split": f"train {parts['train']['n']}, val {parts['val']['n']}, test {parts['test']['n']}, "
                 f"purged {parts['purged']['n']}; train end {cuts[0]}, val end {cuts[1]}",
        "ablation": f"{int((active & (split == TRAIN)).sum())} of {train_before} train windows kept",
        "normalization": "given (source city)" if given_stats is not None else "fitted on train rows",
        "pm25 mean / std": f"{stats.pm_mean:.4f} / {stats.pm_std:.4f}"},
        pd.DataFrame({"station": arrays["keys"], "completeness": comp.round(4), "kept": keep,
                      "loso": protected}))
    runlog.detail(f"FEATURE STATISTICS {city}", table=pd.DataFrame(
        {"feature": FEATURES, "mean": stats.feat_mean, "std": stats.feat_std}))

    return Data(arrays["keys"], arrays["lat"], arrays["lon"], pm, feat, arrays["row_station"],
                interpolate(pm[start[:, None] + LOOKBACK]),
                start, station, origin, split, active, stats)

def ablation_mask(station, split, fraction: int, block: int) -> np.ndarray:
    """ Select, per station, a contiguous `fraction`% block of its train windows:
        from the start (block 0), middle (block 1) or end (block 2) of its train period.
    """
    train = np.flatnonzero(split == TRAIN)
    _, first, inverse, count = np.unique(station[train], return_index=True,
                                         return_inverse=True, return_counts=True)
    pos = np.arange(len(train)) - first[inverse]  # position within the station's train windows
    size = count * fraction // 100
    lo = (count - size) * block // 2
    mask = np.zeros(len(split), bool)
    mask[train] = (pos >= lo[inverse]) & (pos < (lo + size)[inverse])
    return mask

def covered_rows(starts, n_rows) -> np.ndarray:
    """ Mark every hourly row covered by at least one of the windows (lookback + forecast).
    """
    # +1 where a window starts, -1 where it ends; the running sum counts covering windows.
    edges = (np.bincount(starts, minlength=n_rows + 1)
             - np.bincount(starts + LOOKBACK_H + HORIZON_H, minlength=n_rows + 1))
    return np.cumsum(edges)[:-1] > 0

def fit_stats(pm, feat, train_starts) -> Stats:
    """ Fit mean/std over every hourly row covered by a training window (lookback + forecast).
    """
    rows = covered_rows(train_starts, len(pm))
    # PM2.5 uses observed hours only; a near-zero std is set to 1.
    floor = lambda sd: np.where(sd < 1e-8, 1.0, sd).astype(np.float32)
    return Stats(float(np.nanmean(pm[rows])), float(floor(np.nanstd(pm[rows]))),
                 feat[rows].mean(0), floor(feat[rows].std(0)))

def interpolate(lookback: np.ndarray) -> np.ndarray:
    """ Fill missing hours in each lookback row linearly from the observed hours on either side,
        using only that row.
    """
    lookback = lookback.astype(np.float32)
    with_gaps = np.flatnonzero(np.isnan(lookback).any(1))
    incomplete = lookback[with_gaps]
    prev, nxt = nan_bounds(incomplete)
    width = lookback.shape[1]
    lo = np.take_along_axis(incomplete, np.clip(prev, 0, width - 1), 1)  # value before the gap
    hi = np.take_along_axis(incomplete, np.clip(nxt, 0, width - 1), 1)   # value after the gap
    t = ((np.arange(width) - prev) / np.maximum(nxt - prev, 1)).astype(np.float32)
    # A gap at the start or end of the row takes the nearest observed value.
    filled = np.where(prev < 0, hi, np.where(nxt >= width, lo, lo + t * (hi - lo)))
    lookback[with_gaps] = np.where(np.isnan(incomplete), filled, incomplete)
    return lookback

def gbt_features(lookback: np.ndarray) -> np.ndarray:
    """ Compute the GBT inputs from the lookback: PM2.5 lags (0 = origin hour) and rolling
        mean/max/std over the last 3/6/12/24 hours.
    """
    lags = lookback[:, LOOKBACK_H - 1 - np.array(LAG_OFFSETS)]
    rolls = [ROLL_FN[f](lookback[:, -w:], axis=1) for w in ROLL_WINDOWS for f in ROLL_STATS]
    return np.column_stack([lags, *rolls]).astype(np.float32)
