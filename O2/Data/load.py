""" Load a city's cache, apply the filters, normalize, and gather model inputs.
"""

from dataclasses import dataclass

import numpy as np

from Common.schema import LOOKBACK_H, HORIZON_H, LAG_OFFSETS, ROLL_WINDOWS, ROLL_STATS
from Common.split import TRAIN, TEST
from data.build import ensure_cache, nan_bounds

LOOKBACK = np.arange(LOOKBACK_H)
HORIZON = np.arange(LOOKBACK_H, LOOKBACK_H + HORIZON_H)
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
    lookback: np.ndarray   # (windows, 72) normalized, interpolated
    start: np.ndarray      # first row of each window
    station: np.ndarray
    origin: np.ndarray     # last lookback hour, in hours since epoch (UTC)
    split: np.ndarray      # TRAIN / VAL / TEST / PURGED
    active: np.ndarray     # survives the longest-gap and ablation filters
    stats: Stats

    def split_window_ids(self, split: int) -> np.ndarray:
        """ Return the ids of the windows a split (TRAIN, VAL or TEST) trains or scores on:
            the windows labelled with that split that survive the longest-gap and ablation filters.
        """
        return np.flatnonzero(self.active & (self.split == split))

    def gather(self, ids: np.ndarray):
        """ Gather model inputs for the given window ids:
            x_enc [n,72,1+F] (PM2.5 first, then met/time), x_dec [n,72,F], y [n,72].
        """
        first = self.start[ids][:, None]
        x_enc = np.concatenate([self.lookback[ids][..., None], self.feat[first + LOOKBACK]], -1)
        return x_enc, self.feat[first + HORIZON], self.pm[first + HORIZON]


def load_city(city: str, longest_gap: int = 6, completeness: int = 90,
              ablation: int = 100, block: int = 0, stats: Stats | None = None) -> Data:
    """ Load a city (building its cache if missing), filter it by completeness, longest gap
        and ablation, and normalize it with its own training stats or the given (source) stats.
    """
    abl = f"{ablation}%" + (f" (block {block})" if ablation < 100 else "")
    print(f"[DATA] Building Shards: longest gap = {longest_gap}, "
          f"station completeness = {completeness}%, ablation = {abl}")
    cache = np.load(ensure_cache(city))
    # Drop stations below the completeness threshold.
    keep = cache["usable"] & (cache["completeness"] * 100 >= completeness)
    if not keep.any():
        raise ValueError(f"{city}: no station passes completeness >= {completeness}%")

    in_kept_station = keep[cache["station"]]
    start, station, origin, split, gap = (cache[k][in_kept_station] for k in
                                          ("start", "station", "origin", "split", "gap"))
    # Drop train/val windows with a longer lookback gap; test keeps every window.
    active = (split == TEST) | ((split >= 0) & (gap <= longest_gap))
    # Ablation keeps one contiguous block of each station's train windows.
    if ablation < 100:
        active &= (split != TRAIN) | ablation_mask(station, split, ablation, block)
    if not (active & (split == TRAIN)).any():
        raise ValueError(f"{city}: no training windows left after the filters")

    pm, feat = cache["pm"], cache["feat"]
    # Normalize with this city's training stats, or the source city's stats in transfer.
    stats = stats or fit_stats(pm, feat, start[active & (split == TRAIN)])
    pm = (pm - stats.pm_mean) / stats.pm_std
    feat = (feat - stats.feat_mean) / stats.feat_std

    return Data(cache["keys"], cache["lat"], cache["lon"], pm, feat,
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


def fit_stats(pm, feat, train_starts) -> Stats:
    """ Fit mean/std over every hourly row covered by a training window (lookback + forecast).
    """
    # +1 where a window starts, -1 where it ends; the running sum counts covering windows.
    edges = (np.bincount(train_starts, minlength=len(pm) + 1)
             - np.bincount(train_starts + LOOKBACK_H + HORIZON_H, minlength=len(pm) + 1))
    rows = np.cumsum(edges)[:-1] > 0
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
