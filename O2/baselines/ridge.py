""" Ridge baseline: one linear model per forecast hour on the GBT features (PM2.5 lags and rolling
    statistics, plus the met and time of the hour predicted), with the penalty chosen on the
    validation windows. Deterministic, so one run per city and setting.
    python -m baselines.ridge --city mm --longest-gap 6 --completeness 70
"""

import numpy as np

import sys
from pathlib import Path

# Put the O2 root on the import path, so this file runs as a script or as a module.
sys.path.append(str(Path(__file__).resolve().parents[1]))

import config
from Common.schema import HORIZON_H
from Common.splits import TEST, TRAIN, VAL
from data.database import CITIES
from data.load import gbt_features, load_city
from tuning.grid import load_grid
from utils import runlog
from utils.args import run_baseline
from utils.artifacts import ROOT, print_summary, save_scores, settings_tag

BASELINE = "ridge"

def split_inputs(data, split):
    """ Return a split's window ids, shared lookback features [n,f], decoder inputs [n,72,d] and
        targets [n,72].
    """
    ids = data.split_window_ids(split)
    _, x_dec, y = data.gather(ids)
    return ids, gbt_features(data.lookback[ids]), x_dec, y

def fit_hour(x, y, penalties):
    """ Fit ridge on standardized features for every penalty; return a predictor per penalty.
    """
    mean, sd = x.mean(0), x.std(0)
    sd[sd == 0] = 1.0
    z, y_mean = (x - mean) / sd, y.mean()
    gram, cross = z.T @ z, z.T @ (y - y_mean)
    weights = [np.linalg.solve(gram + p * np.eye(len(gram)), cross) for p in penalties]
    return [lambda x_new, w=w: ((x_new - mean) / sd) @ w + y_mean for w in weights]

def run(city, longest_gap=config.DEFAULT_LONGEST_GAP, completeness=config.DEFAULT_COMPLETENESS):
    """ Fit and score ridge for one city under the given filters.
    """
    data = load_city(city, longest_gap, completeness)
    penalties = load_grid()["ridge"]["penalty"]
    _, shared, x_dec, y = split_inputs(data, TRAIN)
    val_ids, val_shared, val_dec, val_y = split_inputs(data, VAL)
    test_ids, test_shared, test_dec, _ = split_inputs(data, TEST)
    features = lambda s, d, h: np.column_stack([s, d[:, h]]).astype(np.float64)

    val_pred = np.empty((len(penalties), len(val_ids), HORIZON_H), np.float32)
    test_pred = np.empty((len(penalties), len(test_ids), HORIZON_H), np.float32)
    for h in range(HORIZON_H):
        predictors = fit_hour(features(shared, x_dec, h), y[:, h].astype(np.float64), penalties)
        for i, predict in enumerate(predictors):
            val_pred[i, :, h] = predict(features(val_shared, val_dec, h))
            test_pred[i, :, h] = predict(features(test_shared, test_dec, h))
    sse = ((val_pred - val_y) ** 2).mean((1, 2))
    best = int(np.argmin(sse))
    runlog.detail("RIDGE PENALTIES", table={"penalty": penalties, "val_rmse": np.sqrt(sse)})

    print(f"[RIDGE] {CITIES[city]} (longest gap: {longest_gap}, station completeness: {completeness})")
    print(f"    penalty = {penalties[best]} (val rmse {np.sqrt(sse[best]):.4f})")
    outdir = ROOT / "baselines" / BASELINE / city / settings_tag(longest_gap, completeness) / "test"
    s = save_scores(outdir, data, test_ids, test_pred[best], val_ids, val_pred[best]
                    if config.SAVE_VAL_PREDICTIONS else None)
    print(f"    {s['n_stations']} stations, {s['n_windows']} test windows")
    print_summary(s)
    return s

if __name__ == "__main__":
    run_baseline(BASELINE, run)
