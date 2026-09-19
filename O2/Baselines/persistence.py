"""Persistence baseline: carry the last observed concentration across the horizon.

Implements the naive reference of Section 4.8.3,

    y_hat(t + l) = y(t),    l in {1, ..., 72},

evaluated on the same station-hour origins the forecasting models are scored on.
Persistence is strong at short lead times, where concentrations change little
from one hour to the next, and decays as the horizon lengthens; it therefore sets
the bar a model must clear to show it has learned usable dynamics rather than
merely repeating the present.

Leave-one-station-out. Section 4.8.1 refits every component on the retained
stations within each fold. Persistence fits nothing and reads only the withheld
station's own history, so withholding a station changes none of its forecasts:
its per-station score already is its LOSO fold score. Metrics are therefore
reported per station and fold-averaged with each station weighted equally, and
the twenty stratified folds of Section 4.7.3 are a subset of the rows written
here rather than a separate run.

Missing origins (masked source). The clean source guarantees a valid PM2.5 value
at every lookback hour, so y(t) is the origin hour. The masked source keeps
windows whose lookback PM2.5 is NaN, and there the last observed concentration
is the most recent finite hour at or before the origin, which is what an operator
forecasting at time t would actually hold. That hour is used and its age is
recorded as origin_gap_h; a window with no observation anywhere in its lookback
has no persistence forecast at all and is dropped, counted as windows_no_obs.

Outputs, under O2/Outputs/persistence/<citySlug>/<source>/:
    folds.csv       - one row per station (one LOSO fold): overall metrics.
    by_lead.csv     - one row per (station, lead): metrics at each lead time.
    lead_curve.csv  - fold-averaged metrics per lead, for the skill-vs-lead plot.
    forecasts/<station>.npz - origin timestamps and y(t), which regenerate every
                      prediction without storing 72 copies of a constant.
"""

import os

import numpy as np
import pandas as pd

from common_baseline import (  # sets sys.path for the O1 schema and builders
    ALL_CITIES, ALL_SOURCES, result_dir, shard_paths, load_lookback_pm25, safe_name,
)
from schema import HORIZON_H
from metrics import all_metrics, METRIC_NAMES

BASELINE = "persistence"


def last_observed(lookback_pm25):
    """Return the most recent finite PM2.5 at or before each window's origin.

    The origin is the final lookback hour, so the search runs backward from the
    end of the window.

    Args:
        lookback_pm25: (n, lookback) array; NaN marks a missing hour.

    Returns:
        y_origin: (n,) the carried-forward value, NaN where the whole lookback
            is missing.
        gap_h: (n,) hours between that observation and the origin; 0 when the
            origin hour itself was observed, -1 where nothing was observed.
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
    """Return the (n, horizon) persistence forecast and its per-window diagnostics.

    Every lead time takes the same value, so the forecast is the last observed
    concentration broadcast across the horizon (Eq. 4.19).
    """
    y_origin, gap_h = last_observed(lookback_pm25)
    pred = np.repeat(y_origin[:, None], horizon, axis=1)
    return pred, y_origin, gap_h


def evaluate_station(location_key, lookback_pm25, Y, horizon=HORIZON_H):
    """Score one station's persistence forecasts overall and at each lead time.

    Windows without any observed lookback hour are excluded from the metrics but
    still counted, so the reported figures are never quietly computed on a
    shrinking subset.

    Returns:
        fold: dict of overall metrics for the station (one LOSO fold).
        by_lead: list of per-lead metric dicts.
        y_origin: (n,) the carried value per window, for the forecast archive.
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


def run(source, city, horizon=HORIZON_H):
    """Score persistence for every station in a (source, city) and write results.

    Streams the O1 window shards one station at a time and writes the per-fold,
    per-lead, and fold-averaged tables described in the module docstring.
    """
    paths = shard_paths(city, source)
    outdir = result_dir(BASELINE, city, source)
    forecast_dir = os.path.join(outdir, "forecasts")
    os.makedirs(forecast_dir, exist_ok=True)

    folds, by_lead = [], []
    for path in paths:
        location_key, lookback_pm25, Y, origins = load_lookback_pm25(path)
        if lookback_pm25.shape[0] == 0:
            continue
        if Y.shape[1] != horizon:
            raise ValueError(f"{path}: expected a {horizon}h horizon, found {Y.shape[1]}")

        fold, station_leads, y_origin = evaluate_station(
            location_key, lookback_pm25, Y, horizon)
        if fold["windows_scored"] == 0:
            continue

        folds.append(fold)
        by_lead.extend(station_leads)
        np.savez_compressed(
            os.path.join(forecast_dir, f"{safe_name(location_key)}.npz"),
            meta_origin=origins,
            y_origin=y_origin.astype(np.float32),
        )

    if not folds:
        print(f"[{BASELINE}] {source}/{city}: no scorable windows; nothing written")
        return

    folds_df = pd.DataFrame(folds)
    leads_df = pd.DataFrame(by_lead)
    # Fold-averaged: each station weighted equally regardless of how many
    # forecasts it contributes (Section 4.8.1).
    curve_df = leads_df.groupby("lead")[list(METRIC_NAMES)].mean().reset_index()
    curve_df.insert(1, "n_folds",
                    leads_df.groupby("lead")["location_key"].nunique().to_numpy())

    folds_df.to_csv(os.path.join(outdir, "folds.csv"), index=False)
    leads_df.to_csv(os.path.join(outdir, "by_lead.csv"), index=False)
    curve_df.to_csv(os.path.join(outdir, "lead_curve.csv"), index=False)

    windows = int(folds_df["windows_scored"].sum())
    dropped = int(folds_df["windows_no_obs"].sum())
    note = f" ({dropped} dropped, no observed lookback hour)" if dropped else ""
    mean_rmse = folds_df["rmse"].mean()
    spread_rmse = folds_df["rmse"].std()
    mean_mae = folds_df["mae"].mean()
    mean_mbe = folds_df["mbe"].mean()
    mean_ioa = folds_df["ioa"].mean()
    first_lead = curve_df.loc[0, "rmse"]
    last_lead = curve_df.loc[horizon - 1, "rmse"]

    print(f"[{BASELINE}] {source}/{city}: {len(folds_df)} stations, "
          f"{windows} windows scored{note}")
    print(f"  fold-averaged RMSE {mean_rmse:.3f} +/- {spread_rmse:.3f} ug/m3 | "
          f"MAE {mean_mae:.3f} | MBE {mean_mbe:+.3f} | IOA {mean_ioa:.3f}")
    print(f"  lead 1h RMSE {first_lead:.3f} -> lead {horizon}h RMSE {last_lead:.3f} "
          f"-> {outdir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, choices=ALL_SOURCES)
    p.add_argument("--city", required=True, choices=ALL_CITIES)
    run(**vars(p.parse_args()))
