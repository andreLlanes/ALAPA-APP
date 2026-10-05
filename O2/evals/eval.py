""" Compare every trained run with the persistence and climatology baselines on the same data
    settings: station-averaged metrics, mean and sd over seeds, overall and over seen/unseen stations,
    and a paired Wilcoxon test of each row against the best row, station by station
    (Holm-adjusted within each city and filter setting).
    Writes artifacts/comparison.csv.
"""

import argparse
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import config
from Common.metrics import METRIC_NAMES
from data.database import CITIES
from utils.artifacts import ROOT, load_json

MODELS = ("lstm", "gnn", "gbt")
BASELINES = ("persistence", "climatology")
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

def compare(city=None) -> pd.DataFrame:
    """ Average each configuration over its seeds and place the baselines beside it.
    """
    runs = pd.concat([model_runs(), baseline_runs()], ignore_index=True)
    if runs.empty:
        return runs
    if city:
        runs = runs[runs["city"] == city]
    values = list(METRIC_NAMES) + ["rmse_seen", "rmse_unseen"]
    table = runs.groupby(KEYS, dropna=False)[values].mean()
    table["rmse_sd"] = runs.groupby(KEYS, dropna=False)["rmse"].std()
    table["seeds"] = runs.groupby(KEYS, dropna=False)["rmse"].count()
    # Baselines are stored per gap/completeness; ablation does not change the test set.
    table = table.reset_index()
    table["filters"] = table["settings"].str.extract(r"(gap\d+_comp\d+)", expand=False)
    table["p_vs_best"] = paired_tests(table, station_rmse(runs))
    return table.sort_values(["city", "filters", "settings", "rmse"]).reset_index(drop=True)

def station_rmse(runs: pd.DataFrame) -> dict:
    """ Return {configuration: RMSE per scored station}, averaged over the configuration's seeds.
    """
    frames = []
    for run in runs.itertuples():
        s = pd.read_csv(f"{run.dir}/stations.csv")
        s = s[s["scored"]][["location_key", "rmse"]]
        frames.append(s.assign(**{k: getattr(run, k) for k in KEYS}))
    means = pd.concat(frames).groupby(KEYS + ["location_key"], dropna=False)["rmse"].mean()
    return {key: group.droplevel(KEYS) for key, group in means.groupby(level=KEYS, dropna=False)}

def holm(p: np.ndarray) -> np.ndarray:
    """ Holm-adjust a set of p-values for multiple comparisons (NaNs stay NaN).
    """
    out = np.full(len(p), np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    order = ok[np.argsort(p[ok])]
    adjusted = np.minimum(1.0, (len(order) - np.arange(len(order))) * p[order])
    out[order] = np.maximum.accumulate(adjusted)
    return out

def paired_tests(table: pd.DataFrame, per_station: dict) -> np.ndarray:
    """ Wilcoxon signed-rank p-value of each row against the lowest-RMSE row of its city and
        filter setting, paired over the stations both scored, Holm-adjusted within that block
        (NaN for the best row itself).
    """
    p = np.full(len(table), np.nan)
    for _, block in table.groupby(["city", "filters"]):
        empty = pd.Series(dtype=float)
        best = per_station.get(tuple(block.loc[block["rmse"].idxmin(), KEYS]), empty)
        for i, row in block.iterrows():
            if i == block["rmse"].idxmin():
                continue
            pairs = pd.concat([per_station.get(tuple(row[KEYS]), empty), best], axis=1, join="inner")
            if len(pairs) >= config.MIN_PAIRED_STATIONS and (pairs.iloc[:, 0] != pairs.iloc[:, 1]).any():
                p[i] = wilcoxon(pairs.iloc[:, 0], pairs.iloc[:, 1]).pvalue
        p[block.index] = holm(p[block.index])
    return p

def print_table(table: pd.DataFrame):
    """ Print one block per city and filter setting, best RMSE first.
    """
    for (city, filters), group in table.groupby(["city", "filters"], sort=False):
        gap, comp = re.findall(r"\d+", filters)
        print(f"[EVAL] {CITIES[city]} (longest gap: {gap}, station completeness: {comp})")
        for r in group.itertuples():
            ablation = r.settings.split("_", 2)[-1] if r.method in ("LSTM", "GNN", "GBT") else ""
            source = f"{r.source}->" if r.source else ""
            variant = r.variant if r.variant not in ("", "none") else ""
            name = " ".join(x for x in (r.method, r.decoder, source, variant, ablation) if x)
            sd = f" +/- {r.rmse_sd:.3f}" if r.seeds > 1 else ""
            p = "best" if np.isnan(r.p_vs_best) and r.rmse == group["rmse"].min() else f"p = {r.p_vs_best:.3f}"
            print(f"    {name:<44} RMSE {r.rmse:.3f}{sd} | MAE {r.mae:.3f} | MBE {r.mbe:+.3f} | "
                  f"IOA {r.ioa:.3f} | seen {r.rmse_seen:.3f} | unseen {r.rmse_unseen:.3f} | {p}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", choices=list(CITIES))
    table = compare(ap.parse_args().city)
    if table.empty:
        raise SystemExit("[EVAL] no results yet")
    table.to_csv(ROOT / "comparison.csv", index=False)
    print_table(table)
    print(f"    -> {ROOT / 'comparison.csv'}")
