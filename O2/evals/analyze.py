""" Analyses of the saved predictions and scores (no retraining), for every finished configuration:
    lead_comparison.csv, ensemble.csv, exceedance.csv, seasons.csv, distance_gain.csv and
    distance_gain_summary.csv in artifacts/. The permutation and degraded-met analyses rerun the
    trained models and have their own commands.
    python -m evals.analyze [--city mm]
"""

import argparse

import pandas as pd

import sys
from pathlib import Path

# Put the O2 root on the import path, so this file runs as a script or as a module.
sys.path.append(str(Path(__file__).resolve().parents[1]))

import config
from data.database import CITIES
from evals import distance_gain, ensemble, exceedance, lead, seasons
from evals.eval import KEYS, all_runs, load_group
from utils import runlog
from utils.artifacts import ROOT

def save(name, frame, title, show=True):
    """ Write a table to artifacts/ and print it (only the leads of interest for per-lead tables).
    """
    frame.to_csv(ROOT / name, index=False)
    shown = frame[frame["lead"].isin([*config.LEAD_TEST_HOURS, "all"])] if "lead" in frame else frame
    print(f"[{title}] {len(frame)} rows -> {ROOT / name}")
    if show and len(shown):
        print("    " + shown.to_string(index=False, float_format="%.3f").replace("\n", "\n    "))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", choices=list(CITIES))
    args = ap.parse_args()
    with runlog.logged(f"analyze_{args.city or 'all'}", args):
        runs = all_runs(args.city)
        if runs.empty:
            raise SystemExit("[ANALYZE] no results yet")
        results = {"ensemble": [], "exceedance": [], "seasons": []}
        for values, group in runs.groupby(KEYS, dropna=False):
            key, loaded = dict(zip(KEYS, values)), load_group(group)
            for name, module in (("ensemble", ensemble), ("exceedance", exceedance),
                                 ("seasons", seasons)):
                results[name] += module.rows(key, group, loaded)
        save("lead_comparison.csv", lead.table(runs), "LEAD")
        for name, rows in results.items():
            save(f"{name}.csv", pd.DataFrame(rows), name.upper())
        stations, summary = distance_gain.table(runs)
        save("distance_gain.csv", stations, "DISTANCE GAIN", show=False)
        save("distance_gain_summary.csv", summary, "DISTANCE GAIN SUMMARY")

if __name__ == "__main__":
    main()
