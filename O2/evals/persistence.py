""" Persistence baseline: carry the last observed concentration across the horizon (Section 4.8.3),
        y_hat(t + l) = y(t),    l in {1, ..., 72},
    scored on exactly the test windows the models see (same filters, same split), with metrics per
    station averaged with equal weight, as the models are. The last observed concentration is the
    most recent measured hour at or before the origin; its age is recorded as origin_gap_h.
    PYTHONPATH=.. python -m evals.persistence --city mm --longest-gap 6 --completeness 70
"""

import argparse

import numpy as np
import pandas as pd

import config
from Common.metrics import all_metrics
from Common.schema import HORIZON_H
from Common.splits import TEST
from data.database import CITIES
from data.load import LOOKBACK, load_city
from utils.artifacts import ROOT, lead_curve, mark_scored, save_json, seen_stations, settings_tag, summarize

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

def persistence_forecast(lookback_pm25, horizon=HORIZON_H):
    """ Return the (n, horizon) persistence forecast (Eq. 4.19) and its per-window diagnostics.
    """
    y_origin, gap_h = last_observed(lookback_pm25)
    pred = np.repeat(y_origin[:, None], horizon, axis=1)
    return pred, y_origin, gap_h

def evaluate_station(location_key, lookback_pm25, Y, horizon=HORIZON_H):
    """ Score one station's persistence forecasts overall and at each lead time.
        Windows without any observed lookback hour are excluded but counted.
        Returns (overall row, per-lead rows, y_origin).
    """
    pred, y_origin, gap_h = persistence_forecast(lookback_pm25, horizon)
    scored = np.isfinite(y_origin)

    fold = {"location_key": location_key,
            "n_windows": int(len(y_origin)),
            "windows_scored": int(scored.sum()),
            "windows_no_obs": int((~scored).sum())}
    fold.update(all_metrics(pred[scored], Y[scored]))
    observed_gap = gap_h[scored]
    fold["mean_origin_gap_h"] = float(observed_gap.mean()) if observed_gap.size else float("nan")
    fold["max_origin_gap_h"] = int(observed_gap.max()) if observed_gap.size else -1

    by_lead = []
    for lead in range(1, horizon + 1):
        row = {"location_key": location_key, "lead": lead}
        row.update(all_metrics(y_origin[scored], Y[scored, lead - 1]))
        by_lead.append(row)

    return fold, by_lead, y_origin

def _write_window(outdir, folds, by_lead):
    """ Write the per-station, per-lead and station-averaged tables; return the summary.
    """
    folds_df = mark_scored(pd.DataFrame(folds))
    leads_df = pd.DataFrame(by_lead)
    # Station-averaged: each scored station weighted equally regardless of how many
    # forecasts it contributes (Section 4.8.1).
    curve_df = lead_curve(leads_df, folds_df)

    folds_df.to_csv(outdir / "stations.csv", index=False)
    leads_df.to_csv(outdir / "by_lead.csv", index=False)
    curve_df.to_csv(outdir / "lead_curve.csv", index=False)
    summary = {**summarize(folds_df, curve_df),
               "windows_no_obs": int(folds_df["windows_no_obs"].sum())}
    save_json(outdir / "metrics.json", summary)
    return summary

def run(city, longest_gap=config.DEFAULT_LONGEST_GAP, completeness=config.DEFAULT_COMPLETENESS):
    """ Score persistence on one city's test windows under the given filters.
    """
    data = load_city(city, longest_gap, completeness)
    ids = data.split_window_ids(TEST)
    raw_pm = data.stats.inverse_y(data.pm)               # NaN where not measured
    lookback = raw_pm[data.start[ids][:, None] + LOOKBACK]
    Y = data.stats.inverse_y(data.gather(ids)[2])
    seen = seen_stations(data)

    folds, by_lead, y_all = [], [], np.empty(len(ids))
    for s in np.unique(data.station[ids]):
        m = data.station[ids] == s
        fold, leads, y_origin = evaluate_station(str(data.keys[s]), lookback[m], Y[m])
        fold["seen"] = bool(seen[s])
        folds.append(fold)
        by_lead.extend(leads)
        y_all[m] = y_origin

    outdir = ROOT / "baselines" / BASELINE / city / settings_tag(longest_gap, completeness) / "test"
    outdir.mkdir(parents=True, exist_ok=True)
    s = _write_window(outdir, folds, by_lead)
    np.savez_compressed(outdir / "forecasts.npz", location_key=data.keys[data.station[ids]],
                        origin=data.origin[ids], y_origin=y_all.astype(np.float32))

    print(f"[PERSISTENCE] {CITIES[city]} (longest gap: {longest_gap}, "
          f"station completeness: {completeness})")
    print(f"    {s['n_stations']} stations, {s['n_windows']} test windows"
          + (f" ({s['windows_no_obs']} without an observed lookback hour)" if s["windows_no_obs"] else ""))
    print(f"    RMSE {s['rmse']:.3f} +/- {s['rmse_sd']:.3f} | MAE {s['mae']:.3f} | "
          f"MBE {s['mbe']:+.3f} | IOA {s['ioa']:.3f}")
    print(f"    lead 1h RMSE {s['rmse_lead1']:.3f} -> lead {HORIZON_H}h RMSE {s[f'rmse_lead{HORIZON_H}']:.3f}")
    return s

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=list(CITIES))
    ap.add_argument("--longest-gap", type=int, default=config.DEFAULT_LONGEST_GAP,
                    choices=config.LONGEST_GAP_CHOICES)
    ap.add_argument("--completeness", type=int, default=config.DEFAULT_COMPLETENESS,
                    choices=config.COMPLETENESS_CHOICES)
    a = ap.parse_args()
    run(a.city, a.longest_gap, a.completeness)
