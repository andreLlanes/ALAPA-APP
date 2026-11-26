""" Seed ensemble per lead: its RMSE against a single seed, and two prediction intervals with their
    coverage of the observations and mean width: the spread of the seeds (central INTERVAL_COVERAGE
    share) and a split-conformal interval (the ensemble plus or minus the INTERVAL_COVERAGE
    quantile of its absolute validation errors at that lead).
"""

import numpy as np
import pandas as pd

import config
from utils.artifacts import station_mean, station_rmse

def spread(pred) -> tuple:
    """ Return the lower and upper percentile of the seeds' predictions [seeds,n,72], in chunks.
    """
    tail = 50 * (1 - config.INTERVAL_COVERAGE)
    bounds = [np.percentile(pred[:, a:a + config.ANALYSIS_CHUNK].astype(np.float32),
                            [tail, 100 - tail], axis=0)
              for a in range(0, pred.shape[1], config.ANALYSIS_CHUNK)]
    return np.concatenate([b[0] for b in bounds]), np.concatenate([b[1] for b in bounds])

def rows(key: dict, group: pd.DataFrame, g: dict) -> list:
    """ Return the per-lead rows of one configuration, or none for fewer than two seeds or without
        validation predictions.
    """
    if len(g["pred"]) < 2 or g["val_pred"] is None:
        return []
    obs, mean = g["obs"], g["pred"].astype(np.float32).mean(0)
    quantile = np.nanquantile(np.abs(g["val_obs"] - g["val_pred"]), config.INTERVAL_COVERAGE, axis=0)
    low, high = spread(g["pred"])
    intervals = {"spread": (low, high), "conformal": (mean - quantile, mean + quantile)}
    curve, overall = station_rmse(g["keys"], mean - obs)
    single = np.mean([pd.read_csv(f"{d}/lead_curve.csv")["rmse"] for d in group["dir"]], axis=0)
    out = [{**key, "lead": lead, "rmse_ensemble": r, "rmse_single": s}
           for lead, (r, s) in enumerate(zip(curve, single), 1)]
    out.append({**key, "lead": "all", "rmse_ensemble": overall, "rmse_single": group["rmse"].mean()})
    for name, (lo, hi) in intervals.items():
        inside = np.where(np.isnan(obs), np.nan, ((obs >= lo) & (obs <= hi)).astype(np.float32))
        for i, (coverage, width) in enumerate(zip(station_mean(g["keys"], inside),
                                                  station_mean(g["keys"], hi - lo))):
            out[i].update({f"{name}_coverage": coverage, f"{name}_width": width})
        out[-1].update({f"{name}_coverage": np.nanmean([r[f"{name}_coverage"] for r in out[:-1]]),
                        f"{name}_width": np.mean([r[f"{name}_width"] for r in out[:-1]])})
    return out
