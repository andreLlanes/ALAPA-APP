"""Persistence baseline: carry the last observed concentration across the horizon.

Implements the naive reference of Section 4.8.3,

    y_hat(t + l) = y(t),    l in {1, ..., 72},

evaluated on the same station-hour origins the forecasting models are scored on.
Persistence is strong at short lead times, where concentrations change little
from one hour to the next, and decays as the horizon lengthens; it therefore sets
the bar a model must clear to show it has learned usable dynamics rather than
merely repeating the present.

Evaluation. The baselines are scored on one protocol only: the test partition of
the chronological 70/15/15 split (the "fixed" protocol), with the dates frozen in
O2/common/split_dates.json. Each station is forecast from its own readings, and
the models are scored the same way on the same windows, so the skill score of
Section 4.8.3 is computed here. Rolling origin and leave-one-station-out are
model-only protocols (ranking stability across seasons, and spatial
generalization through kriging); the baselines are not run under them.

Stations. Only the study's station pool is scored (completeness >= 70%, read from
the fold file's training_pool), the same stations every other method uses.

Interpolated hours (clean source). Filled target hours (Y_filled) are not scored,
since they are estimates rather than measurements. Windows whose origin hour was
filled (origin_filled) are not scored at all: that value was interpolated from
the hours just after it, which are the window's first targets, so carrying it
forward would leak the answer. Both are counted in folds.csv.

Missing origins (masked source). The clean source guarantees a valid PM2.5 value
at every lookback hour, so y(t) is the origin hour. The masked source keeps
windows whose lookback PM2.5 is NaN, and there the last observed concentration
is the most recent finite hour at or before the origin, which is what an operator
forecasting at time t would actually hold. That hour is used and its age is
recorded as origin_gap_h; a window with no observation anywhere in its lookback
has no persistence forecast at all and is dropped, counted as windows_no_obs.

Outputs, under O2/Outputs/persistence/<citySlug>/<source>/<protocol>/:
    windows.json    - the protocol's cut timestamps, spans, and counts.
    summary.csv     - one row per evaluation window: fold-averaged metrics.
    <window>/folds.csv      - one row per station.
    <window>/by_lead.csv    - one row per (station, lead).
    <window>/lead_curve.csv - fold-averaged metrics per lead.
    <window>/forecasts/<station>.npz - origins and y(t), which regenerate every
                      prediction without storing 72 copies of a constant.
"""

import json
import os

import numpy as np
import pandas as pd

from common_baseline import (  # sets sys.path for the O1 schema and builders
    ALL_CITIES, ALL_SOURCES, result_dir, shard_paths, load_origins,
    load_lookback_pm25, load_filled_flags, station_pool, safe_name,
)
from schema import HORIZON_H
from metrics import all_metrics, METRIC_NAMES
from splits import PARTITIONS, frozen_fixed_cuts, partition_mask, describe

BASELINE = "persistence"
PROTOCOLS = ("fixed",)
DEFAULT_PROTOCOL = "fixed"
DEFAULT_SPLIT = "test"


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


def evaluate_station(location_key, lookback_pm25, Y, horizon=HORIZON_H,
                     Y_filled=None, origin_filled=None):
    """Score one station's persistence forecasts overall and at each lead time.

    Windows without any observed lookback hour, and windows whose origin hour was
    interpolated, are excluded from the metrics but still counted, so the
    reported figures are never quietly computed on a shrinking subset.
    Interpolated target hours are blanked, so the metrics skip them.

    Returns:
        fold: dict of overall metrics for the station.
        by_lead: list of per-lead metric dicts.
        y_origin: (n,) the carried value per window, for the forecast archive.
    """
    pred, y_origin, gap_h = persistence_forecast(lookback_pm25, horizon)
    origin_ok = (np.ones(len(y_origin), dtype=bool) if origin_filled is None
                 else ~origin_filled)
    scored = np.isfinite(y_origin) & origin_ok
    Y = np.array(Y, dtype=float)
    if Y_filled is not None:
        Y[Y_filled] = np.nan      # metrics ignore non-finite pairs

    fold = {"location_key": location_key,
            "n_windows": int(len(y_origin)),
            "windows_scored": int(scored.sum()),
            "windows_no_obs": int((~np.isfinite(y_origin)).sum()),
            "windows_origin_filled": int((~origin_ok).sum()),
            "filled_targets_skipped": (int(Y_filled[scored].sum())
                                       if Y_filled is not None else 0)}
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


def pooled_origins(paths):
    """Return every station's forecast origins concatenated.

    Reads only the origin arrays, so defining the evaluation windows costs
    kilobytes per station rather than loading the window tensors.
    """
    collected = [o for o in (load_origins(p) for p in paths) if o.size]
    if not collected:
        raise ValueError("no origins found in any shard")
    return np.concatenate(collected)


def resolve_windows(paths, protocol, city, split=DEFAULT_SPLIT):
    """Return the evaluation windows for a protocol, plus a record of them.

    The cut dates are the city's frozen ones (O2/common/split_dates.json), never
    recomputed here, so persistence uses exactly the models' periods. Each window
    pairs a folder name with the mask that selects its origins.

    Returns:
        windows: list of {"name", "mask"} dicts.
        record: JSON-serializable description written to windows.json.
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}; got {protocol!r}")
    origins = pooled_origins(paths)
    cuts = frozen_fixed_cuts(city)
    windows = [{"name": split,
                "mask": lambda o, c=cuts: partition_mask(o, c, split)}]
    record = {"protocol": "fixed", "split": split,
              "train_end": str(cuts[0]), "val_end": str(cuts[1]),
              "partitions": describe(origins, cuts)}
    return windows, record


def _write_window(outdir, folds, by_lead, horizon):
    """Write one window's per-fold, per-lead, and fold-averaged tables.

    Returns the fold-averaged summary row, or None when nothing was scorable.
    """
    if not folds:
        return None
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

    return {
        "n_stations": len(folds_df),
        "windows_scored": int(folds_df["windows_scored"].sum()),
        "windows_no_obs": int(folds_df["windows_no_obs"].sum()),
        "windows_origin_filled": int(folds_df["windows_origin_filled"].sum()),
        "filled_targets_skipped": int(folds_df["filled_targets_skipped"].sum()),
        "rmse": folds_df["rmse"].mean(),
        "rmse_sd": folds_df["rmse"].std(),
        "mae": folds_df["mae"].mean(),
        "mbe": folds_df["mbe"].mean(),
        "ioa": folds_df["ioa"].mean(),
        "r2": folds_df["r2"].mean(),
        "rmse_lead1": curve_df.loc[0, "rmse"],
        f"rmse_lead{horizon}": curve_df.loc[horizon - 1, "rmse"],
    }


def run(source, city, protocol=DEFAULT_PROTOCOL, split=DEFAULT_SPLIT,
        horizon=HORIZON_H):
    """Score persistence for one (source, city) under one protocol.

    Defines the protocol's evaluation windows from the frozen dates, then streams
    the O1 window shards once, scoring every window from each station's single
    load rather than re-reading the shards per window.
    """
    paths = shard_paths(city, source)
    windows, record = resolve_windows(paths, protocol, city, split)
    pool = station_pool(city)
    record["station_pool"] = (f"{len(pool)} stations (fold file training_pool)"
                              if pool is not None else "all stations (no fold file for this city)")

    protocol_dir = result_dir(BASELINE, city, source, protocol)
    with open(os.path.join(protocol_dir, "windows.json"), "w") as f:
        json.dump(record, f, indent=2)

    acc = {w["name"]: {"folds": [], "by_lead": [], "absent": 0} for w in windows}
    for path in paths:
        location_key, lookback_pm25, Y, origins = load_lookback_pm25(path)
        if pool is not None and location_key not in pool:
            continue
        if lookback_pm25.shape[0] == 0:
            continue
        if Y.shape[1] != horizon:
            raise ValueError(f"{path}: expected a {horizon}h horizon, found {Y.shape[1]}")
        Y_filled, origin_filled = load_filled_flags(path)

        for window in windows:
            store = acc[window["name"]]
            keep = window["mask"](origins)
            if not keep.any():
                store["absent"] += 1
                continue

            fold, station_leads, y_origin = evaluate_station(
                location_key, lookback_pm25[keep], Y[keep], horizon,
                None if Y_filled is None else Y_filled[keep],
                None if origin_filled is None else origin_filled[keep])
            if fold["windows_scored"] == 0:
                store["absent"] += 1
                continue

            store["folds"].append(fold)
            store["by_lead"].extend(station_leads)
            forecast_dir = result_dir(BASELINE, city, source, protocol,
                                      window["name"], "forecasts")
            np.savez_compressed(
                os.path.join(forecast_dir, f"{safe_name(location_key)}.npz"),
                meta_origin=origins[keep],
                y_origin=y_origin.astype(np.float32),
            )

    summary_rows = []
    for window in windows:
        name = window["name"]
        store = acc[name]
        outdir = result_dir(BASELINE, city, source, protocol, name)
        row = _write_window(outdir, store["folds"], store["by_lead"], horizon)
        if row is None:
            print(f"[{BASELINE}] {source}/{city} [{protocol}/{name}]: "
                  f"no scorable windows")
            continue
        row = {"window": name, "stations_absent": store["absent"], **row}
        summary_rows.append(row)

        dropped = []
        if row["windows_no_obs"]:
            dropped.append(f"{row['windows_no_obs']} with no observed lookback hour")
        if row["windows_origin_filled"]:
            dropped.append(f"{row['windows_origin_filled']} with an interpolated origin")
        note = f" (not scored: {'; '.join(dropped)})" if dropped else ""
        if row["filled_targets_skipped"]:
            note += f"; {row['filled_targets_skipped']} interpolated target hours skipped"
        absent = (f", {store['absent']} stations absent from this period"
                  if store["absent"] else "")
        print(f"[{BASELINE}] {source}/{city} [{protocol}/{name}]: "
              f"{row['n_stations']} stations, {row['windows_scored']} windows "
              f"scored{note}{absent}")
        print(f"  fold-averaged RMSE {row['rmse']:.3f} +/- {row['rmse_sd']:.3f} "
              f"ug/m3 | MAE {row['mae']:.3f} | MBE {row['mbe']:+.3f} | "
              f"IOA {row['ioa']:.3f}")
        print(f"  lead 1h RMSE {row['rmse_lead1']:.3f} -> "
              f"lead {horizon}h RMSE {row[f'rmse_lead{horizon}']:.3f}")

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(protocol_dir, "summary.csv"), index=False)
    _print_period(record)
    print(f"  -> {protocol_dir}")


def _print_period(record):
    """Print the calendar period the fixed split actually used."""
    part = record["partitions"].get(record["split"])
    if part and part["n"]:
        print(f"  fixed split: {record['split']} period {part['start'][:10]} to "
              f"{part['end'][:10]} ({part['days']:.0f} days, "
              f"{part['frac']:.1%} of origins)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, choices=ALL_SOURCES)
    p.add_argument("--city", required=True, choices=ALL_CITIES)
    p.add_argument("--split", default=DEFAULT_SPLIT,
                   choices=list(PARTITIONS) + ["all"],
                   help="partition of the chronological split to score (default: test)")
    run(**vars(p.parse_args()))
