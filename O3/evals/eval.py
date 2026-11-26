""" Compare every trained run with the persistence and climatology baselines on the same data
    settings: station-averaged metrics, mean and sd over seeds, overall and over seen/unseen stations.
    Significance is tested as planned contrasts, each transfer variant against its matched cold start
    and its unweighted-pooling control: a Diebold-Mariano test over forecast origins (primary,
    Holm-adjusted within each architecture), with the number of stations whose RMSE improved
    reported alongside as a descriptive check.
    Writes artifacts/comparison.csv and artifacts/contrasts.csv.
    python -m evals.eval [--city mm]
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

import sys

# Put the O2 root on the import path, so this file runs as a script or as a module.
sys.path.append(str(Path(__file__).resolve().parents[1]))

import config
from Common.metrics import METRIC_NAMES
from Common.schema import HORIZON_H
from data.database import CITIES
from utils.artifacts import ROOT, load_json, read_predictions

MODELS = ("lstm", "gnn", "gbt", "dlinear")
BASELINES = ("persistence", "climatology", "ridge")
KEYS = ["city", "settings", "method", "decoder", "source", "variant"]

def flatten(summary: dict) -> dict:
    """ Pick the reported numbers out of a metrics.json summary.
    """
    return {**{m: summary.get(m) for m in METRIC_NAMES},
            "rmse_seen": summary["seen"].get("rmse", float("nan")),
            "rmse_unseen": summary["unseen"].get("rmse", float("nan")),
            "n_stations": summary["n_stations"], "n_windows": summary["n_windows"]}

def model_runs() -> pd.DataFrame:
    """ One row per trained seed: artifacts/<model>/<pair>/<mode>_<settings>/<variant>/seed<n>.
    """
    rows = []
    for model in MODELS:
        for path in (ROOT / model).glob("*/*/*/seed*/metrics.json"):
            seed_dir, variant_dir, settings_dir, pair_dir = path.parents[0:4]
            mode, settings = settings_dir.name.split("_", 1)
            city = pair_dir.name.split("-")[-1]
            rows.append({"city": city, "settings": settings, "method": model.upper(),
                         "decoder": mode, "source": pair_dir.name.split("-")[0] if "-" in pair_dir.name else "",
                         "variant": variant_dir.name, "seed": seed_dir.name[4:],
                         "dir": str(seed_dir), **flatten(load_json(path))})
    return pd.DataFrame(rows)

def baseline_runs() -> pd.DataFrame:
    """ One row per baseline result: artifacts/baselines/<name>/<city>/<settings>/test.
    """
    rows = []
    for name in BASELINES:
        for path in (ROOT / "baselines" / name).glob("*/*/test/metrics.json"):
            city, settings = path.parents[2].name, path.parents[1].name
            rows.append({"city": city, "settings": settings, "method": name, "decoder": "",
                         "source": "", "variant": "", "seed": "", "dir": str(path.parent),
                         **flatten(load_json(path))})
    return pd.DataFrame(rows)

def all_runs(city=None) -> pd.DataFrame:
    """ One row per trained seed and per baseline result, with its data filters.
    """
    runs = pd.concat([model_runs(), baseline_runs()], ignore_index=True)
    if city:
        runs = runs[runs["city"] == city]
    # Baselines are stored per gap/completeness; ablation does not change the test set.
    runs["filters"] = runs["settings"].str.extract(r"(gap\d+_comp\d+)", expand=False)
    return runs

def interval(mean, sd, n):
    """ Return the CI_LEVEL confidence interval of a mean over n seeds.
    """
    half = t.ppf((1 + config.CI_LEVEL) / 2, (n - 1).clip(lower=1)) * sd / np.sqrt(n)
    return mean - half, mean + half

def compare(runs: pd.DataFrame) -> pd.DataFrame:
    """ Average each configuration over its seeds and place the baselines beside it.
    """
    if runs.empty:
        return runs
    values = list(METRIC_NAMES) + ["rmse_seen", "rmse_unseen"]
    grouped = runs.groupby(KEYS + ["filters"], dropna=False)
    table = grouped[values].mean()
    table["rmse_sd"], table["seeds"] = grouped["rmse"].std(), grouped["rmse"].count()
    table["rmse_ci_low"], table["rmse_ci_high"] = interval(table["rmse"], table["rmse_sd"], table["seeds"])
    table = table.reset_index()
    return table.sort_values(["city", "filters", "settings", "rmse"]).reset_index(drop=True)

def station_rmse(runs: pd.DataFrame, lead=None) -> dict:
    """ Return {configuration: RMSE per scored station}, averaged over the configuration's seeds,
        overall or at one forecast lead.
    """
    frames = []
    for run in runs.itertuples():
        scored = pd.read_csv(f"{run.dir}/stations.csv").query("scored")
        rmse = scored[["location_key", "rmse"]] if lead is None else (
            pd.read_csv(f"{run.dir}/by_lead.csv").query("lead == @lead")
            .merge(scored[["location_key"]])[["location_key", "rmse"]])
        frames.append(rmse.assign(**{k: getattr(run, k) for k in KEYS}))
    means = pd.concat(frames).groupby(KEYS + ["location_key"], dropna=False)["rmse"].mean()
    return {key: group.droplevel(KEYS) for key, group in means.groupby(level=KEYS, dropna=False)}

def load_group(group: pd.DataFrame) -> dict:
    """ Load the saved predictions of one configuration's seeds, which share their windows:
        keys, origin, obs [n,72], pred [seeds,n,72], and the mean of the seeds' validation
        predictions with their observations (None when not saved).
    """
    first, preds, val_sum, val_obs = None, [], None, None
    has_val = all((Path(path) / "val_predictions.npz").exists() for path in group["dir"])
    for path in group["dir"]:
        keys, origin, pred, obs = read_predictions(f"{path}/predictions.npz")
        if first is None:
            first = (keys, origin, obs)
        elif not (np.array_equal(keys, first[0]) and np.array_equal(origin, first[1])):
            raise ValueError(f"{path}: test windows differ between seeds")
        preds.append(pred)
        if has_val:
            *_, val_pred, val_obs = read_predictions(f"{path}/val_predictions.npz")
            val_sum = val_pred.astype(np.float32) + (0 if val_sum is None else val_sum)
    return {"keys": first[0], "origin": first[1], "obs": first[2].astype(np.float32),
            "pred": np.stack(preds),
            "val_pred": val_sum / len(preds) if has_val else None,
            "val_obs": val_obs.astype(np.float32) if has_val else None}

def holm(p: np.ndarray) -> np.ndarray:
    """ Holm-adjust a set of p-values for multiple comparisons (NaNs stay NaN).
    """
    out = np.full(len(p), np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    order = ok[np.argsort(p[ok])]
    adjusted = np.minimum(1.0, (len(order) - np.arange(len(order))) * p[order])
    out[order] = np.maximum.accumulate(adjusted)
    return out

# Transfer variants whose benefit the paper claims; each is tested against its matched cold start
# and its unweighted-pooling control. zeroshot and pooled are themselves controls, not tested.
TESTED_VARIANTS = ("param_frozen", "param_full", "mmd_frozen", "mmd_full", "weighted")

def diebold_mariano(loss_a, loss_b, horizon=HORIZON_H):
    """ Two-sided Diebold-Mariano test on a per-origin loss differential d = loss_a - loss_b, with a
        Newey-West long-run variance (lag = horizon - 1, the overlap of hourly h-step forecasts) and
        the Harvey-Leybourne-Newbold small-sample correction. Returns (stat, p, n_origins); a
        negative stat means loss_a < loss_b, i.e. configuration a forecasts more accurately.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3 or np.allclose(d, d[0]):
        return float("nan"), float("nan"), n
    dev = d - d.mean()
    lag = min(horizon - 1, n - 1)
    lrv = np.mean(dev * dev) + 2.0 * sum(
        (1 - k / (lag + 1)) * np.mean(dev[k:] * dev[:-k]) for k in range(1, lag + 1))
    if lrv <= 0:
        return float("nan"), float("nan"), n
    stat = d.mean() / np.sqrt(lrv / n)
    factor = (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
    if factor > 0:
        stat *= np.sqrt(factor)
    return float(stat), float(2 * t.cdf(-abs(stat), df=n - 1)), n

def origin_losses(group_a: pd.DataFrame, group_b: pd.DataFrame):
    """ Align two configurations' seed-mean test forecasts on their shared (station, origin) windows
        and return their per-origin squared-error loss series (mean over stations and the 72 leads),
        ordered by origin.
    """
    def frame(group, name):
        g = load_group(group)
        mse = np.nanmean((g["pred"].mean(0) - g["obs"]) ** 2, axis=1)   # per-window MSE over leads
        return pd.DataFrame({"station": g["keys"], "origin": g["origin"], name: mse})
    merged = frame(group_a, "a").merge(frame(group_b, "b"), on=["station", "origin"])
    per_origin = merged.groupby("origin")[["a", "b"]].mean().sort_index()
    return per_origin["a"].to_numpy(), per_origin["b"].to_numpy()

def window_losses(group: pd.DataFrame, by: str) -> pd.DataFrame:
    """ Per-window squared-error loss (mean over the 72 leads) of a configuration's seed-mean
        forecasts, loading each value of `by` (a fold or a rolling origin, whose windows differ)
        separately: columns station, origin, loss.
    """
    frames = []
    for _, rows in group.groupby(by):
        g = load_group(rows)
        loss = np.nanmean((g["pred"].astype(np.float32).mean(0) - g["obs"]) ** 2, axis=1)
        frames.append(pd.DataFrame({"station": g["keys"], "origin": g["origin"], "loss": loss}))
    return pd.concat(frames) if frames else pd.DataFrame(columns=["station", "origin", "loss"])

def paired_dm(group_a: pd.DataFrame, group_b: pd.DataFrame, by: str):
    """ Diebold-Mariano of configuration a against b on their shared windows, with the loss of each
        forecast origin averaged over its stations. Returns (stat, p, n_origins).
    """
    merged = window_losses(group_a, by).merge(window_losses(group_b, by), on=["station", "origin"],
                                              suffixes=("_a", "_b"))
    per_origin = merged.groupby("origin")[["loss_a", "loss_b"]].mean().sort_index()
    return diebold_mariano(per_origin["loss_a"], per_origin["loss_b"])

def _comparator_key(key: tuple, **changes) -> tuple:
    """ A KEYS tuple with some fields replaced (to name a variant's matched control). """
    fields = dict(zip(KEYS, key))
    fields.update(changes)
    return tuple(fields[k] for k in KEYS)

def planned_contrasts(runs: pd.DataFrame) -> pd.DataFrame:
    """ Test each transfer variant against (a) its matched cold start and (b) its matched
        unweighted-pooling control, same model, decoder, source and filters. The primary test is
        Diebold-Mariano over forecast origins, Holm-adjusted within each architecture (city, filters,
        model). A station-paired test is not computed: too few Metro Manila stations clear the
        100-window threshold for it to reach significance, so the count of stations whose RMSE
        improved is reported instead as a descriptive check.
    """
    groups = {key: g for key, g in runs.groupby(KEYS, dropna=False)}
    per_station = station_rmse(runs)
    rows = []
    for key, group in groups.items():
        info = dict(zip(KEYS, key))
        if info["source"] == "" or info["variant"] not in TESTED_VARIANTS:
            continue
        controls = {"cold_start": _comparator_key(key, source="", variant="none"),
                    "pooling": _comparator_key(key, variant="pooled")}
        for kind, ckey in controls.items():
            if ckey not in groups:
                continue
            stat, p, n_origins = diebold_mariano(*origin_losses(group, groups[ckey]))
            a, b = per_station.get(key), per_station.get(ckey)
            pairs = (pd.concat([a, b], axis=1, join="inner")
                     if a is not None and b is not None else pd.DataFrame())
            rows.append({**info, "filters": group["filters"].iloc[0], "vs": kind,
                         "n_origins": n_origins, "dm_stat": stat, "dm_p": p,
                         "n_stations": len(pairs),
                         "n_better_stations": int((pairs.iloc[:, 0] < pairs.iloc[:, 1]).sum()) if len(pairs) else 0})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["dm_p_holm"] = np.nan
    for _, idx in table.groupby(["city", "filters", "method"]).groups.items():
        table.loc[idx, "dm_p_holm"] = holm(table.loc[idx, "dm_p"].to_numpy())
    return table.sort_values(["city", "filters", "method", "source", "variant", "vs"]).reset_index(drop=True)

def print_table(table: pd.DataFrame):
    """ Print one block per city and filter setting, best RMSE first.
    """
    for (city, filters), group in table.groupby(["city", "filters"], sort=False):
        gap, comp = re.findall(r"\d+", filters)
        print(f"[EVAL] {CITIES[city]} (longest gap: {gap}, station completeness: {comp})")
        for r in group.itertuples():
            ablation = r.settings.split("_", 2)[-1] if r.method.lower() in MODELS else ""
            variant = r.variant if r.variant not in ("", "none") else ""
            route = f"{r.source}->{variant}" if r.source else variant
            name = " ".join(x for x in (r.method, r.decoder, route, ablation) if x)
            sd = (f" +/- {r.rmse_sd:.3f} [{r.rmse_ci_low:.3f}, {r.rmse_ci_high:.3f}]"
                  if r.seeds > 1 else "")
            print(f"    {name:<44} RMSE {r.rmse:.3f}{sd} | MAE {r.mae:.3f} | MBE {r.mbe:+.3f} | "
                  f"IOA {r.ioa:.3f} | seen {r.rmse_seen:.3f} | unseen {r.rmse_unseen:.3f}")

def print_contrasts(table: pd.DataFrame):
    """ Print the planned contrasts: Diebold-Mariano over origins (primary) and the improved-station count.
    """
    if table.empty:
        print("[CONTRASTS] no transfer variants with matched controls yet")
        return
    for (city, filters, method), group in table.groupby(["city", "filters", "method"], sort=False):
        print(f"[CONTRASTS] {CITIES[city]} {method} ({filters})")
        for r in group.itertuples():
            side = "lower" if r.dm_stat < 0 else "higher"
            print(f"    {r.source}->{r.variant} vs {r.vs:<10} DM {r.dm_stat:+.2f} "
                  f"p = {r.dm_p:.3f} Holm = {r.dm_p_holm:.3f} ({side} loss, {r.n_origins} origins) | "
                  f"{r.n_better_stations}/{r.n_stations} stations improved")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", choices=list(CITIES))
    runs = all_runs(ap.parse_args().city)
    table = compare(runs)
    if table.empty:
        raise SystemExit("[EVAL] no results yet")
    table.to_csv(ROOT / "comparison.csv", index=False)
    print_table(table)
    contrasts = planned_contrasts(runs)
    contrasts.to_csv(ROOT / "contrasts.csv", index=False)
    print_contrasts(contrasts)
    print(f"    -> {ROOT / 'comparison.csv'}, {ROOT / 'contrasts.csv'}")
