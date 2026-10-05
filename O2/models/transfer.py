""" Transfer learning: freezing for fine-tuning, station MMD distances to the target city,
    and the GBT sample weights built from them (Section 4.5).
"""

import zlib

import numpy as np

import config
from Common.splits import TRAIN
from data.load import covered_rows
from models import mmd

VARIANTS = ("frozen", "full")            # LSTM / GNN fine-tuning
GBT_VARIANTS = ("pooled", "weighted")    # GBT: source weight 1, or exp(-d / tau)

def freeze_encoder(model):
    """ Prepare a pretrained model for the frozen variant: the encoder keeps its source weights,
        only the decoder and output layer adapt to the target (the full variant skips this).
    """
    for p in model.encoder.parameters():
        p.requires_grad = False

def training_rows(data):
    """ Return the raw [PM2.5, met, time] rows covered by the city's training windows that have an
        observed PM2.5, and the station of each row.
    """
    rows = covered_rows(data.start[data.split_window_ids(TRAIN)], len(data.pm)) & ~np.isnan(data.pm)
    s = data.stats
    pm = data.pm[rows] * s.pm_std + s.pm_mean
    feat = data.feat[rows] * s.feat_std + s.feat_mean
    return np.column_stack([pm, feat]).astype(np.float64), data.row_station[rows]

def _rng(*parts):
    """ Seeded generator per (repeat, name), so every draw is reproducible on its own.
    """
    return np.random.default_rng([config.DISTANCE_SEED] + [
        p if isinstance(p, int) else zlib.crc32(p.encode()) for p in parts])

def _draw(a, n, rng):
    """ Draw n rows without replacement (all rows if there are fewer).
    """
    return a if len(a) <= n else a[np.sort(rng.choice(len(a), n, replace=False))]

def station_distances(source, target, pool: dict = None) -> dict:
    """ Measure each source station's MMD^2 to the target city's training hours, standardized by
        the target's training statistics. Returns per-station arrays: station, n_rows, mmd2_pair
        (bandwidth per station pair) and mmd2_fixed (one bandwidth shared by every station), each
        with its sd over the repeats. Stations with too few complete hours are left out.
        pool = {city: data} of every source city: the fixed bandwidth is pooled from the target
        and all of them, so fixed distances are comparable across source cities.
    """
    x_target, _ = training_rows(target)
    x_source, row_station = training_rows(source)
    mu, sd = x_target.mean(0), x_target.std(0)
    sd[sd == 0] = 1.0
    x_target, x_source = (x_target - mu) / sd, (x_source - mu) / sd
    pool_rows = {city: (training_rows(data)[0] - mu) / sd
                 for city, data in sorted((pool or {"source": source}).items())}

    stations, counts = np.unique(row_station, return_counts=True)
    enough = counts >= config.DISTANCE_MIN_ROWS
    stations, counts = stations[enough], counts[enough]
    for key in source.keys[np.setdiff1d(np.unique(source.station), stations)]:
        print(f"    Excluded Key: {key} - fewer than {config.DISTANCE_MIN_ROWS} complete training hours")
    if not len(stations):
        raise ValueError("no source station has enough training hours for a distance")
    by_station = {s: x_source[row_station == s] for s in stations}

    pair = np.zeros((config.DISTANCE_REPEATS, len(stations)))
    fixed = np.zeros_like(pair)
    for r in range(config.DISTANCE_REPEATS):
        y = _draw(x_target, config.DISTANCE_SAMPLE, _rng(r, "target"))
        d_yy = mmd.sq_dists(y)
        yy_upper = mmd.upper_values(d_yy)
        # One bandwidth for every station in this repeat, pooled from the target and every source city.
        pooled = np.vstack([_draw(x_target, config.DISTANCE_GLOBAL, _rng(r, "global", "target"))]
                           + [_draw(rows, config.DISTANCE_GLOBAL, _rng(r, "global", city))
                              for city, rows in pool_rows.items()])
        s_fix = mmd.bandwidths(mmd.pooled_median(mmd.upper_values(mmd.sq_dists(pooled))))
        k_yy_fix = mmd.within_mean(d_yy, s_fix)

        for j, s in enumerate(stations):
            x = _draw(by_station[s], config.DISTANCE_SAMPLE, _rng(r, str(source.keys[s])))
            d_xx, d_xy = mmd.sq_dists(x), mmd.sq_dists(x, y)
            s_pair = mmd.bandwidths(mmd.pooled_median(mmd.upper_values(d_xx), yy_upper, d_xy))
            pair[r, j] = (mmd.within_mean(d_xx, s_pair) + mmd.within_mean(d_yy, s_pair)
                          - 2.0 * mmd.cross_mean(d_xy, s_pair))
            fixed[r, j] = (mmd.within_mean(d_xx, s_fix) + k_yy_fix
                           - 2.0 * mmd.cross_mean(d_xy, s_fix))

    ddof = 1 if config.DISTANCE_REPEATS > 1 else 0
    return {"station": stations, "n_rows": counts,
            "mmd2_pair": pair.mean(0), "mmd2_pair_sd": pair.std(0, ddof=ddof),
            "mmd2_fixed": fixed.mean(0), "mmd2_fixed_sd": fixed.std(0, ddof=ddof)}

def tau_grid(distances: dict, scales) -> np.ndarray:
    """ Return candidate taus relative to the median station distance, so the grid spans strong
        to weak downweighting whatever scale the distances have.
    """
    median = np.median(np.clip(distances[config.DISTANCE_COLUMN], 0, None))
    if median <= 0:
        raise ValueError("median station distance is not positive, so no tau can be derived")
    return median * np.asarray(scales)

def gbt_weights(source, ids, distances: dict, tau: float) -> np.ndarray:
    """ Return each source window's weight exp(-d_s / tau) from its station's distance
        (stations without a distance get 0). Target windows keep weight 1.
    """
    per_station = np.zeros(len(source.keys))
    per_station[distances["station"]] = mmd.sample_weights(distances[config.DISTANCE_COLUMN], tau)
    return per_station[source.station[ids]]
