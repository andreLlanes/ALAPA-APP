""" Persistence baseline: carry the last observed concentration across the horizon (Section 4.8.3),
        y_hat(t + l) = y(t),    l in {1, ..., 72},
    scored on exactly the test windows the models see (same filters, same split), with metrics per
    station averaged with equal weight, as the models are. The last observed concentration is the
    most recent measured hour at or before the origin; its age is recorded as origin_gap_h.
    python -m baselines.persistence --city mm --longest-gap 6 --completeness 70
"""

import numpy as np
import pandas as pd

import sys
from pathlib import Path

# Put the O2 root on the import path, so this file runs as a script or as a module.
sys.path.append(str(Path(__file__).resolve().parents[1]))

import config
from Common.schema import HORIZON_H
from Common.splits import TEST
from data.database import CITIES
from data.load import LOOKBACK, load_city
from utils.args import run_baseline
from utils.artifacts import (ROOT, print_summary, seen_stations, settings_tag, station_scores,
                             write_predictions, write_tables)

BASELINE = "persistence"

def last_observed(lookback_pm25):
    """ Return the most recent finite PM2.5 at or before each window's origin, and its age:
        y_origin (n,), NaN where the whole lookback is missing; gap_h (n,), 0 when the origin
        hour itself was observed, -1 where nothing was observed.
    """
    finite = np.isfinite(lookback_pm25)
    n, lookback = finite.shape
    has_obs = finite.any(axis=1)

    # argmax on the reversed mask gives the distance from the origin back to the
    # last observed hour, in one pass and without a Python-level loop.
    gap_h = np.argmax(finite[:, ::-1], axis=1)
    last_idx = (lookback - 1) - gap_h

    y_origin = np.full(n, np.nan, dtype=float)
    rows = np.arange(n)[has_obs]
    y_origin[has_obs] = lookback_pm25[rows, last_idx[has_obs]]
    gap_h = np.where(has_obs, gap_h, -1)
    return y_origin, gap_h

def origin_gaps(keys, y_origin, gap_h) -> pd.DataFrame:
    """ Per-station counts of windows with and without an observed lookback hour, and the mean
        and longest age of the last observation (hours before the origin), over observed windows.
    """
    seen = np.isfinite(y_origin)
    frame = pd.DataFrame({"location_key": keys, "scored_window": seen,
                          "gap": np.where(seen, gap_h, np.nan)})
    agg = frame.groupby("location_key").agg(windows_scored=("scored_window", "sum"),
                                            windows_no_obs=("scored_window", lambda c: (~c).sum()),
                                            mean_origin_gap_h=("gap", "mean"),
                                            max_origin_gap_h=("gap", "max")).reset_index()
    agg["max_origin_gap_h"] = agg["max_origin_gap_h"].fillna(-1).astype(int)
    return agg.astype({"windows_scored": int, "windows_no_obs": int})

def run(city, longest_gap=config.DEFAULT_LONGEST_GAP, completeness=config.DEFAULT_COMPLETENESS):
    """ Score persistence on one city's test windows under the given filters.
    """
    data = load_city(city, longest_gap, completeness)
    ids = data.split_window_ids(TEST)
    raw_pm = data.stats.inverse_y(data.pm)               # NaN where not measured
    y_origin, gap_h = last_observed(raw_pm[data.start[ids][:, None] + LOOKBACK])
    pred = np.repeat(y_origin[:, None], HORIZON_H, axis=1)  # NaN where nothing was observed
    obs = data.stats.inverse_y(data.gather(ids)[2])

    stations, by_lead, curve, summary = station_scores(data.keys, data.station[ids], pred, obs,
                                                        seen_stations(data))
    stations = stations.merge(origin_gaps(data.keys[data.station[ids]], y_origin, gap_h),
                              on="location_key")
    summary["windows_no_obs"] = int(stations["windows_no_obs"].sum())

    outdir = ROOT / "baselines" / BASELINE / city / settings_tag(longest_gap, completeness) / "test"
    write_tables(outdir, stations, by_lead, curve, summary)
    write_predictions(outdir / "predictions.npz", data, ids, pred, obs)

    print(f"[PERSISTENCE] {CITIES[city]} (longest gap: {longest_gap}, "
          f"station completeness: {completeness})")
    print(f"    {summary['n_stations']} stations, {summary['n_windows']} test windows"
          + (f" ({summary['windows_no_obs']} without an observed lookback hour)"
             if summary["windows_no_obs"] else ""))
    print_summary(summary)
    return summary

if __name__ == "__main__":
    run_baseline("persistence", run)
