"""Persistence baseline: carry the last observed concentration across the horizon.

Implements the naive reference of Section 4.8.3,

    y_hat(t + l) = y(t),    l in {1, ..., 72},

evaluated on the same station-hour origins the forecasting models are scored on.
Persistence is strong at short lead times, where concentrations change little
from one hour to the next, and decays as the horizon lengthens; it therefore sets
the bar a model must clear to show it has learned usable dynamics rather than
merely repeating the present.

Protocols. Both temporal designs of Section 4.7.3 are supported and write to
separate folders:

    fixed   - the test partition of the chronological 70/15/15 split, one
              contiguous period. This is the partition leave-one-station-out
              holds fixed while it varies the station.
    rolling - four origins advancing by three months, each scored on the three
              months that follow, together spanning a full wet and dry cycle.
              This varies the test period that the fixed split holds constant,
              so agreement between the two shows a ranking does not depend on
              the particular months tested on.

Because persistence fits nothing, only each window's test period affects its
score; the training spans are still recorded so the models can reuse exactly the
same windows and the skill score of Section 4.8.3 compares like with like.

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

Outputs, under O2/Outputs/persistence/<citySlug>/<source>/<protocol>/:
    windows.json    - the protocol's cut timestamps, spans, and counts.
    summary.csv     - one row per evaluation window: fold-averaged metrics.
    <window>/folds.csv      - one row per station (one LOSO fold).
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
    load_lookback_pm25, safe_name,
)
from schema import HORIZON_H
from metrics import all_metrics, METRIC_NAMES
from splits import (
    PARTITIONS, ROLLING_ORIGINS, ROLLING_TEST_MONTHS,
    chronological_cuts, partition_mask, describe,
    rolling_origin_windows, rolling_mask, describe_rolling,
)

BASELINE = "persistence"
PROTOCOLS = ("fixed", "rolling")
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


def pooled_origins(paths):
    """Return every station's forecast origins concatenated.

    Reads only the origin arrays, so defining the evaluation windows costs
    kilobytes per station rather than loading the window tensors.
    """
    collected = [o for o in (load_origins(p) for p in paths) if o.size]
    if not collected:
        raise ValueError("no origins found in any shard")
    return np.concatenate(collected)


def resolve_windows(paths, protocol, split=DEFAULT_SPLIT,
                    n_origins=ROLLING_ORIGINS, test_months=ROLLING_TEST_MONTHS):
    """Return the evaluation windows for a protocol, plus a record of them.

    Each window pairs a folder name with the mask that selects its origins, so
    the scoring loop treats the fixed split and the rolling origins identically.

    Returns:
        windows: list of {"name", "mask"} dicts.
        record: JSON-serializable description written to windows.json.
    """
    origins = pooled_origins(paths)

    if protocol == "fixed":
        cuts = chronological_cuts(origins)
        windows = [{"name": split,
                    "mask": lambda o, c=cuts: partition_mask(o, c, split)}]
        record = {"protocol": "fixed", "split": split,
                  "train_end": str(cuts[0]), "val_end": str(cuts[1]),
                  "partitions": describe(origins, cuts)}
        return windows, record

    if protocol == "rolling":
        wins = rolling_origin_windows(origins, n_origins, test_months)
        windows = [{"name": f"origin{w['origin']}",
                    "mask": lambda o, w=w: rolling_mask(o, w, "test")}
                   for w in wins]
        record = {"protocol": "rolling", "n_origins": n_origins,
                  "test_months": test_months,
                  "windows": describe_rolling(origins, wins)}
        return windows, record

    raise ValueError(f"protocol must be one of {PROTOCOLS}; got {protocol!r}")


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
        n_origins=ROLLING_ORIGINS, test_months=ROLLING_TEST_MONTHS,
        horizon=HORIZON_H):
    """Score persistence for one (source, city) under one protocol.

    Defines the protocol's evaluation windows from the pooled origins, then
    streams the O1 window shards once, scoring every window from each station's
    single load rather than re-reading the shards per window.
    """
    paths = shard_paths(city, source)
    windows, record = resolve_windows(paths, protocol, split, n_origins, test_months)

    protocol_dir = result_dir(BASELINE, city, source, protocol)
    with open(os.path.join(protocol_dir, "windows.json"), "w") as f:
        json.dump(record, f, indent=2)

    acc = {w["name"]: {"folds": [], "by_lead": [], "absent": 0} for w in windows}
    for path in paths:
        location_key, lookback_pm25, Y, origins = load_lookback_pm25(path)
        if lookback_pm25.shape[0] == 0:
            continue
        if Y.shape[1] != horizon:
            raise ValueError(f"{path}: expected a {horizon}h horizon, found {Y.shape[1]}")

        for window in windows:
            store = acc[window["name"]]
            keep = window["mask"](origins)
            if not keep.any():
                store["absent"] += 1
                continue

            fold, station_leads, y_origin = evaluate_station(
                location_key, lookback_pm25[keep], Y[keep], horizon)
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

        note = (f" ({row['windows_no_obs']} dropped, no observed lookback hour)"
                if row["windows_no_obs"] else "")
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
    _print_periods(record, protocol)
    print(f"  -> {protocol_dir}")


def _print_periods(record, protocol):
    """Print the calendar periods the protocol actually used."""
    if protocol == "fixed":
        part = record["partitions"].get(record["split"])
        if part and part["n"]:
            print(f"  fixed split: {record['split']} period {part['start'][:10]} to "
                  f"{part['end'][:10]} ({part['days']:.0f} days, "
                  f"{part['frac']:.1%} of origins)")
        return
    print(f"  rolling origin: {record['n_origins']} origins advancing by "
          f"{record['test_months']} months (expanding training window)")
    for w in record["windows"]:
        print(f"    origin {w['origin']}: train through {w['train_end'][:10]} "
              f"({w['n_train']} origins) -> test {w['test_start'][:10]} to "
              f"{w['test_end'][:10]} ({w['n_test']} origins)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, choices=ALL_SOURCES)
    p.add_argument("--city", required=True, choices=ALL_CITIES)
    p.add_argument("--protocol", default=DEFAULT_PROTOCOL, choices=PROTOCOLS,
                   help="temporal evaluation design (default: fixed)")
    p.add_argument("--split", default=DEFAULT_SPLIT,
                   choices=list(PARTITIONS) + ["all"],
                   help="partition to score under --protocol fixed (default: test)")
    p.add_argument("--origins", type=int, default=ROLLING_ORIGINS,
                   dest="n_origins",
                   help="number of rolling origins (default: 4)")
    p.add_argument("--test-months", type=int, default=ROLLING_TEST_MONTHS,
                   dest="test_months",
                   help="length of each rolling test period, in months (default: 3)")
    run(**vars(p.parse_args()))
