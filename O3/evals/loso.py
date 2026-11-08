""" Compare the leave-one-station-out runs (train.py --spatial true) with the kriged persistence
    baseline on the same data settings. A fold is one withheld station and weighs equally: metrics
    are averaged over the scored folds for each seed, then over seeds (mean, sd and 95% CI),
    overall and per density band. Diebold-Mariano tests over forecast origins compare every
    configuration with kriged persistence, and each transfer variant with its matched cold start
    and its pooled control, Holm-adjusted within each architecture; the number of folds whose RMSE
    improved is reported alongside.
    Writes artifacts/loso_comparison.csv and artifacts/loso_contrasts.csv.
    python evals/loso.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Put the O2 root on the import path, so this file runs as a script or as a module.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from Common.metrics import METRIC_NAMES
from data.load import loso_folds
from evals.eval import TESTED_VARIANTS, holm, interval, paired_dm
from utils import runlog
from utils.artifacts import ROOT

LOSO = ROOT / "loso"
MODELS = ("lstm", "gnn", "gbt")
KEYS = ["filters", "method", "decoder", "source", "variant"]

def runs() -> pd.DataFrame:
    """ One row per kriged fold result: every model seed and every kriged persistence fold, with
        the withheld station's scores.
    """
    rows = []
    for model in MODELS:
        for path in (LOSO / model).glob("*/*/fold*/*/seed*/metrics.json"):
            seed_dir, variant_dir, fold_dir, settings_dir, pair_dir = path.parents[0:5]
            mode, settings = settings_dir.name.split("_", 1)
            rows.append({"settings": settings, "method": model.upper(), "decoder": mode,
                         "source": pair_dir.name.split("-")[0] if "-" in pair_dir.name else "",
                         "variant": variant_dir.name, "seed": seed_dir.name[4:],
                         "fold": int(fold_dir.name[4:]), "dir": str(seed_dir)})
    for path in (LOSO / "baselines" / "persistence").glob("*/*/fold*/metrics.json"):
        rows.append({"settings": path.parents[1].name, "method": "persistence", "decoder": "",
                     "source": "", "variant": "", "seed": "", "fold": int(path.parent.name[4:]),
                     "dir": str(path.parent)})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    scores = pd.DataFrame([pd.read_csv(Path(d) / "stations.csv").iloc[0] for d in frame["dir"]])
    frame = pd.concat([frame, scores[["location_key", "n_windows", "scored", *METRIC_NAMES]]
                       .reset_index(drop=True)], axis=1)
    frame["scored"] = frame["scored"].astype(bool)
    frame["band"] = frame["fold"].map({f["fold_id"]: f["density_band"] for f in loso_folds()})
    frame["filters"] = frame["settings"].str.extract(r"(gap\d+_comp\d+)", expand=False)
    return frame

def compare(frame: pd.DataFrame) -> pd.DataFrame:
    """ Average each configuration over its scored folds per seed, then over seeds; overall
        (band "all") and per density band.
    """
    scored = frame[frame["scored"]]
    parts = []
    for band, rows in [("all", scored), *scored.groupby("band")]:
        by_seed = rows.groupby(KEYS + ["seed"], dropna=False)
        seeds = by_seed[list(METRIC_NAMES)].mean().join(by_seed["fold"].nunique().rename("folds"))
        grouped = seeds.reset_index().groupby(KEYS, dropna=False)
        table = grouped[list(METRIC_NAMES)].mean()
        table["rmse_sd"], table["seeds"] = grouped["rmse"].std(), grouped["rmse"].count()
        table["folds"] = grouped["folds"].min()
        table["rmse_ci_low"], table["rmse_ci_high"] = interval(table["rmse"], table["rmse_sd"],
                                                               table["seeds"])
        parts.append(table.reset_index().assign(band=band))
    if not parts:
        return pd.DataFrame()
    unscored = (frame[~frame["scored"]].groupby(KEYS, dropna=False)["fold"].nunique()
                .rename("folds_unscored"))
    table = pd.concat(parts).join(unscored, on=KEYS).fillna({"folds_unscored": 0})
    return table.sort_values(["filters", "band", "rmse"]).reset_index(drop=True)

def contrast(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """ Diebold-Mariano of configuration a against b on their shared withheld windows of the scored
        folds (loss per origin averaged over the folds), and the number of shared folds where a's
        RMSE is lower.
    """
    stat, p, n_origins = paired_dm(a[a["scored"]], b[b["scored"]], "fold")
    fold_rmse = lambda g: g[g["scored"]].groupby("fold")["rmse"].mean()
    folds = pd.concat([fold_rmse(a), fold_rmse(b)], axis=1, join="inner")
    return {"n_origins": n_origins, "dm_stat": stat, "dm_p": p, "n_folds": len(folds),
            "n_better_folds": int((folds.iloc[:, 0] < folds.iloc[:, 1]).sum())}

def contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    """ Test every configuration against kriged persistence, and each transfer variant against its
        cold start and its pooled control (same model, decoder, source and filters).
    """
    groups = {key: g for key, g in frame.groupby(KEYS, dropna=False)}
    rows = []
    for key, group in groups.items():
        info = dict(zip(KEYS, key))
        if info["method"] == "persistence":
            continue
        controls = {"persistence": (info["filters"], "persistence", "", "", "")}
        if info["source"] and info["variant"] in TESTED_VARIANTS:
            controls["cold_start"] = (info["filters"], info["method"], info["decoder"], "", "none")
            controls["pooling"] = (info["filters"], info["method"], info["decoder"], info["source"],
                                   "pooled")
        for kind, control in controls.items():
            if control in groups:
                result = contrast(group, groups[control])
                if result["n_folds"]:
                    rows.append({**info, "vs": kind, **result})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["dm_p_holm"] = np.nan
    for _, idx in table.groupby(["filters", "method"]).groups.items():
        table.loc[idx, "dm_p_holm"] = holm(table.loc[idx, "dm_p"].to_numpy())
    return table.sort_values(["filters", "method", "source", "variant", "vs"]).reset_index(drop=True)

def print_table(table: pd.DataFrame, filters: str):
    """ Print the overall LOSO comparison of one filter setting, best RMSE first.
    """
    gap, comp = re.findall(r"\d+", filters)
    rows = table[(table["filters"] == filters) & (table["band"] == "all")]
    print(f"[LOSO] Metro Manila, regression-kriged (longest gap: {gap}, station completeness: {comp})")
    if rows.empty:
        print("    no scored folds yet")
    for r in rows.itertuples():
        variant = r.variant if r.variant not in ("", "none") else ""
        name = " ".join(x for x in (r.method, r.decoder, f"{r.source}->{variant}" if r.source else variant) if x)
        sd = f" +/- {r.rmse_sd:.3f}" if r.seeds > 1 else ""
        print(f"    {name:<36} RMSE {r.rmse:.3f}{sd} | MAE {r.mae:.3f} | MBE {r.mbe:+.3f} | "
              f"IOA {r.ioa:.3f} | folds {int(r.folds)} (+{int(r.folds_unscored)} unscored)")

def print_contrasts(table: pd.DataFrame, filters: str):
    """ Print the Diebold-Mariano contrasts of one filter setting.
    """
    for r in table[table["filters"] == filters].itertuples():
        side = "lower" if r.dm_stat < 0 else "higher"
        variant = r.variant if r.variant not in ("", "none") else ""
        name = " ".join(x for x in (r.method, r.decoder, f"{r.source}->{variant}" if r.source else variant) if x)
        print(f"    {name:<30} vs {r.vs:<11} DM {r.dm_stat:+.2f} p = {r.dm_p:.3f} "
              f"Holm = {r.dm_p_holm:.3f} ({side} loss, {r.n_origins} origins) | "
              f"{r.n_better_folds}/{r.n_folds} folds improved")

def report(filters: str = None):
    """ Write both tables and print them (one filter setting, or every setting found).
    """
    frame = runs()
    if frame.empty:
        print("[LOSO] no results yet")
        return
    table, tests = compare(frame), contrasts(frame)
    table.to_csv(ROOT / "loso_comparison.csv", index=False)
    tests.to_csv(ROOT / "loso_contrasts.csv", index=False)
    for f in [filters] if filters else sorted(frame["filters"].dropna().unique()):
        print_table(table, f)
        if not tests.empty:
            print_contrasts(tests, f)
    print(f"    -> {ROOT / 'loso_comparison.csv'}, {ROOT / 'loso_contrasts.csv'}")

if __name__ == "__main__":
    with runlog.logged("loso_eval"):
        report()
