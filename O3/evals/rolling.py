""" Compare the rolling-origin runs (train.py --rolling true) with the baselines on the same data
    settings. For every rolling origin (test period), each configuration's station-averaged
    metrics are averaged over its seeds; the origins are then averaged with equal weight, and their
    spread shows how steady a configuration is across the year. Diebold-Mariano tests over the
    forecast origins of all test periods compare every configuration with persistence, and each
    transfer variant with its cold start and its pooled control, Holm-adjusted within each
    architecture; the number of test periods whose RMSE improved is reported alongside.
    Writes artifacts/rolling_comparison.csv and artifacts/rolling_contrasts.csv.
    python evals/rolling.py [--city mm]
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Put the O2 root on the import path, so this file runs as a script or as a module.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from Common.metrics import METRIC_NAMES
from data.database import CITIES
from evals.eval import BASELINES, TESTED_VARIANTS, flatten, holm, paired_dm
from utils import runlog
from utils.artifacts import ROOT, load_json

ROLLING = ROOT / "rolling"
MODELS = ("lstm", "gnn", "gbt")
KEYS = ["city", "filters", "method", "decoder", "source", "variant"]

def runs(city=None) -> pd.DataFrame:
    """ One row per rolling-origin result: every model seed and every baseline, per test period.
    """
    rows = []
    for model in MODELS:
        for path in (ROLLING / model).glob("*/*/origin*/*/seed*/metrics.json"):
            seed_dir, variant_dir, origin_dir, settings_dir, pair_dir = path.parents[0:5]
            mode, settings = settings_dir.name.split("_", 1)
            rows.append({"city": pair_dir.name.split("-")[-1], "settings": settings,
                         "method": model.upper(), "decoder": mode,
                         "source": pair_dir.name.split("-")[0] if "-" in pair_dir.name else "",
                         "variant": variant_dir.name, "seed": seed_dir.name[4:],
                         "rolling_origin": int(origin_dir.name[6:]), "dir": str(seed_dir),
                         **flatten(load_json(path))})
    for name in BASELINES:
        for path in (ROLLING / "baselines" / name).glob("*/*/origin*/metrics.json"):
            rows.append({"city": path.parents[2].name, "settings": path.parents[1].name,
                         "method": name, "decoder": "", "source": "", "variant": "", "seed": "",
                         "rolling_origin": int(path.parent.name[6:]), "dir": str(path.parent),
                         **flatten(load_json(path))})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    if city:
        frame = frame[frame["city"] == city]
    frame["filters"] = frame["settings"].str.extract(r"(gap\d+_comp\d+)", expand=False)
    return frame

def compare(frame: pd.DataFrame) -> pd.DataFrame:
    """ Average each configuration over its seeds per rolling origin, then over the origins; keep
        each origin's RMSE and the sd of RMSE over origins.
    """
    by_origin = frame.groupby(KEYS + ["rolling_origin"], dropna=False)
    origins = by_origin[list(METRIC_NAMES)].mean().join(by_origin["seed"].count().rename("seeds"))
    grouped = origins.reset_index().groupby(KEYS, dropna=False)
    table = grouped[list(METRIC_NAMES)].mean()
    table["rmse_sd_origins"] = grouped["rmse"].std()
    table["origins"], table["seeds"] = grouped["rmse"].count(), grouped["seeds"].min()
    each = origins["rmse"].unstack("rolling_origin")
    each.columns = [f"rmse_origin{k}" for k in each.columns]
    return table.join(each).reset_index().sort_values(["city", "filters", "rmse"]).reset_index(drop=True)

def contrast(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """ Diebold-Mariano of configuration a against b over every test period's forecast origins,
        and the number of shared periods where a's seed-mean RMSE is lower.
    """
    stat, p, n_origins = paired_dm(a, b, "rolling_origin")
    period_rmse = lambda g: g.groupby("rolling_origin")["rmse"].mean()
    periods = pd.concat([period_rmse(a), period_rmse(b)], axis=1, join="inner")
    return {"n_origins": n_origins, "dm_stat": stat, "dm_p": p, "n_periods": len(periods),
            "n_better_periods": int((periods.iloc[:, 0] < periods.iloc[:, 1]).sum())}

def contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    """ Test every configuration against persistence, and each transfer variant against its cold
        start and its pooled control (same model, decoder, source and filters).
    """
    groups = {key: g for key, g in frame.groupby(KEYS, dropna=False)}
    rows = []
    for key, group in groups.items():
        info = dict(zip(KEYS, key))
        if info["method"] in BASELINES:
            continue
        controls = {"persistence": (info["city"], info["filters"], "persistence", "", "", "")}
        if info["source"] and info["variant"] in TESTED_VARIANTS:
            controls["cold_start"] = (info["city"], info["filters"], info["method"], info["decoder"],
                                      "", "none")
            controls["pooling"] = (info["city"], info["filters"], info["method"], info["decoder"],
                                   info["source"], "pooled")
        for kind, control in controls.items():
            if control in groups:
                rows.append({**info, "vs": kind, **contrast(group, groups[control])})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["dm_p_holm"] = np.nan
    for _, idx in table.groupby(["city", "filters", "method"]).groups.items():
        table.loc[idx, "dm_p_holm"] = holm(table.loc[idx, "dm_p"].to_numpy())
    return table.sort_values(["city", "filters", "method", "source", "variant", "vs"]).reset_index(drop=True)

def name_of(r) -> str:
    """ Display name of a configuration row.
    """
    variant = r.variant if r.variant not in ("", "none") else ""
    route = f"{r.source}->{variant}" if r.source else variant
    return " ".join(x for x in (r.method, r.decoder, route) if x)

def print_table(table: pd.DataFrame, tests: pd.DataFrame):
    """ Print one block per city and filter setting, best mean RMSE first, then its contrasts.
    """
    for (city, filters), group in table.groupby(["city", "filters"], sort=False):
        gap, comp = re.findall(r"\d+", filters)
        print(f"[ROLLING] {CITIES[city]} (longest gap: {gap}, station completeness: {comp})")
        each = [c for c in group.columns if c.startswith("rmse_origin")]
        for r in group.itertuples():
            per = " ".join(f"{getattr(r, c):.3f}" for c in each)
            sd = f" +/- {r.rmse_sd_origins:.3f}" if r.origins > 1 else ""
            print(f"    {name_of(r):<36} RMSE {r.rmse:.3f}{sd} over {int(r.origins)} origins "
                  f"({per}) | MAE {r.mae:.3f} | MBE {r.mbe:+.3f} | seeds {int(r.seeds)}")
        if tests.empty:
            continue
        for r in tests[(tests["city"] == city) & (tests["filters"] == filters)].itertuples():
            side = "lower" if r.dm_stat < 0 else "higher"
            print(f"    {name_of(r):<30} vs {r.vs:<11} DM {r.dm_stat:+.2f} p = {r.dm_p:.3f} "
                  f"Holm = {r.dm_p_holm:.3f} ({side} loss, {r.n_origins} origins) | "
                  f"{r.n_better_periods}/{r.n_periods} periods improved")

def report(city=None, filters=None):
    """ Write both tables and print them (one city and filter setting, or everything found).
    """
    frame = runs(city)
    if frame.empty:
        print("[ROLLING] no results yet")
        return
    table, tests = compare(frame), contrasts(frame)
    table.to_csv(ROOT / "rolling_comparison.csv", index=False)
    tests.to_csv(ROOT / "rolling_contrasts.csv", index=False)
    if filters:
        table = table[table["filters"] == filters]
    print_table(table, tests)
    print(f"    -> {ROOT / 'rolling_comparison.csv'}, {ROOT / 'rolling_contrasts.csv'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", choices=list(CITIES))
    with runlog.logged("rolling_eval"):
        report(ap.parse_args().city)
