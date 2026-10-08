""" Load a city from the local database copy, build and filter its windows, normalize, and
    gather model inputs.
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

import config
from Common.schema import (DECODER_COLS, HORIZON_H, LAG_OFFSETS, LOOKBACK_H, ROLL_STATS, ROLL_WINDOWS,
                           TIME_COL)
from Common.splits import TRAIN, chronological_cuts, partition_labels, describe
from data.database import CITIES, read_copy
from data.windows import FEATURES, city_arrays, nan_bounds
from utils import runlog

LOOKBACK = np.arange(LOOKBACK_H)
HORIZON = np.arange(LOOKBACK_H, LOOKBACK_H + HORIZON_H)
LOSO_PATH = Path(__file__).resolve().parents[1] / "loso_stations.json"
ROLL_FN = {"mean": np.mean, "max": np.max, "std": np.std}
GBT_FEATURES = ([f"pm25_lag{k}" for k in LAG_OFFSETS]
                + [f"pm25_{f}{w}" for w in ROLL_WINDOWS for f in ROLL_STATS]
                + [f"{c}_target" for c in DECODER_COLS])

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
    city: np.ndarray       # city code of each station

    def split_window_ids(self, split: int) -> np.ndarray:
        """ Return the ids of the windows a split (TRAIN, VAL or TEST) trains or scores on:
            the windows labelled with that split that survive ablation.
        """
        return np.flatnonzero(self.active & (self.split == split))

    def split_by_city(self, split: int) -> list:
        """ Return a split's window ids grouped by city, one array per city present.
        """
        ids = self.split_window_ids(split)
        city = self.city[self.station[ids]]
        return [ids[city == c] for c in np.unique(city)]

    def only_scored_on(self, city: str) -> "Data":
        """ Return a copy whose validation and test windows are this city's alone, so a pooled
            dataset trains on every city but stops and scores on one.
        """
        mine = self.city[self.station] == city
        return replace(self, active=self.active & ((self.split == TRAIN) | mine))

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

def prepare_city(city: str, longest_gap: int, completeness: int) -> dict:
    """ Build one city's windows, drop stations below the completeness threshold (except the LOSO
        stations) and windows with a longer lookback gap, then split the rest 70/15/15 by pooled
        window count. Return the city's raw arrays plus the surviving windows and their split.
    """
    rows = read_copy(city)
    # With MATCH_LA_BKK, Los Angeles starts at Bangkok's first record.
    if city == "la" and config.MATCH_LA_BKK:
        rows = rows[rows[TIME_COL] >= read_copy("bk", columns=[TIME_COL])[TIME_COL].min()]
    arrays = city_arrays(rows)
    keep = (arrays["completeness"] * 100 >= completeness) | np.isin(arrays["keys"], loso_keys())
    if not keep.any():
        raise ValueError(f"{city}: no station passes completeness >= {completeness}%")

    # A fully missing lookback has gap 72, so the longest-gap filter drops it too.
    settled = keep[arrays["station"]] & (arrays["gap"] <= longest_gap)
    if not settled.any():
        raise ValueError(f"{city}: no windows left after the filters")
    start, station = arrays["start"][settled], arrays["station"][settled]
    origin = arrays["origin"][settled].astype("datetime64[h]")

    cuts = chronological_cuts(origin)
    split = partition_labels(origin, cuts)
    parts = describe(origin, cuts)
    print(f"    Split {CITIES[city]}: train = {parts['train']['n']}, val = {parts['val']['n']}, "
          f"test = {parts['test']['n']}, purged = {parts['purged']['n']} "
          f"(train end {cuts[0]}, val end {cuts[1]})")
    comp, protected = arrays["completeness"], np.isin(arrays["keys"], loso_keys())
    runlog.detail(f"DATA {CITIES[city]}", {
        "stations": f"{len(keep)} total, {int(keep.sum())} kept "
                    f"({int((keep & (comp * 100 < completeness)).sum())} kept only as LOSO stations)",
        "windows": f"{len(arrays['start'])} valid, {int(keep[arrays['station']].sum())} after "
                   f"completeness, {len(start)} after longest gap",
        "split": f"train {parts['train']['n']}, val {parts['val']['n']}, test {parts['test']['n']}, "
                 f"purged {parts['purged']['n']}; train end {cuts[0]}, val end {cuts[1]}"},
        pd.DataFrame({"station": arrays["keys"], "completeness": comp.round(4), "kept": keep,
                      "loso": protected}))
    return {**arrays, "city": city, "start": start, "station": station, "origin": origin,
            "split": split}

def load_city(cities, longest_gap: int = None, completeness: int = None,
              ablation: int = 100, block: int = 0, stats: Stats | None = None) -> Data:
    """ Load one city, or several pooled into one dataset: build and filter each city's windows,
        split each city on its own dates, trim the last city's training windows by `ablation`,
        and normalize with statistics fitted on every city's training rows, or the given
        (source) statistics in transfer.
    """
    cities = [cities] if isinstance(cities, str) else list(cities)
    longest_gap = config.DEFAULT_LONGEST_GAP if longest_gap is None else longest_gap
    completeness = config.DEFAULT_COMPLETENESS if completeness is None else completeness
    abl = f"{ablation}%" + (f" (block {block})" if ablation < 100 else "")
    print(f"[DATA] Building Shards: longest gap = {longest_gap}, "
          f"station completeness = {completeness}%, ablation = {abl}")
    built = [prepare_city(c, longest_gap, completeness) for c in cities]

    # Pool the cities: later cities' rows and stations come after the earlier ones.
    row_offset = np.cumsum([0] + [len(b["pm"]) for b in built[:-1]])
    station_offset = np.cumsum([0] + [len(b["keys"]) for b in built[:-1]])
    pooled = lambda name, offsets=None: np.concatenate(
        [b[name] if offsets is None else b[name] + offsets[i] for i, b in enumerate(built)])
    start, station = pooled("start", row_offset), pooled("station", station_offset)
    pm, feat = pooled("pm"), pooled("feat")
    split = pooled("split")

    active = split >= 0
    train_before = int((split == TRAIN).sum())
    # Ablation keeps one contiguous block of each station's train windows.
    if ablation < 100:
        last = np.arange(len(split)) >= len(split) - len(built[-1]["split"])
        active &= ~last | (split != TRAIN) | ablation_mask(station, split, ablation, block)
    if not (active & (split == TRAIN)).any():
        raise ValueError(f"no training windows left after the filters in {', '.join(cities)}")

    given_stats = stats
    stats = stats or fit_stats(pm, feat, start[active & (split == TRAIN)])
    pm = (pm - stats.pm_mean) / stats.pm_std
    feat = (feat - stats.feat_mean) / stats.feat_std
    runlog.detail(f"NORMALIZATION {', '.join(CITIES[c] for c in cities)}", {
        "ablation": f"{int((active & (split == TRAIN)).sum())} of {train_before} train windows kept",
        "normalization": "given (source cities)" if given_stats is not None else "fitted on train rows",
        "pm25 mean / std": f"{stats.pm_mean:.4f} / {stats.pm_std:.4f}"},
        pd.DataFrame({"feature": FEATURES, "mean": stats.feat_mean, "std": stats.feat_std}))

    keys = pooled("keys")
    city_of_station = np.concatenate([np.full(len(b["keys"]), b["city"]) for b in built])
    return Data(keys, pooled("lat"), pooled("lon"), pm, feat, pooled("row_station", station_offset),
                interpolate(pm[start[:, None] + LOOKBACK]), start, station,
                pooled("origin"), split, active, stats, city_of_station)

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
