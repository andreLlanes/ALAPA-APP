""" Rescore finished runs after changing their test inputs, for the permutation and degraded-met
    analyses: for every variant and seed in ANALYSIS_SEEDS, the station-averaged RMSE per lead and
    overall under each edit, and its change from the unedited inputs.
    Writes <configuration folder>/<analysis>.csv.
"""

import argparse

import pandas as pd

import config
import train
from Common.splits import TEST
from data.load import HORIZON
from utils import runlog
from utils.args import add_arguments, blocks, config_folder, names, read_arguments, variants
from utils.artifacts import load_json, station_rmse

def rescore(args, analysis, make_edits):
    """ Rescore every variant of one ablation block; make_edits(data, ids) -> {name: edit}.
    """
    for block in blocks(args):
        cdir, rows = config_folder(args, block), []
        for variant in variants(args):
            seeds = [s for s in config.ANALYSIS_SEEDS
                     if (cdir / variant / f"seed{s}" / "metrics.json").exists()]
            if not seeds:
                print(f"[{analysis.upper()}] no finished seeds in {cdir / variant}")
                continue
            train.LOADERS.clear()
            data = train.variant_data(args, block, variant)
            ids = data.split_window_ids(TEST)
            edits = {"none": None, **make_edits(data, ids)}
            for seed in seeds:
                out = cdir / variant / f"seed{seed}"
                params = load_json(out / "params.json")
                model = train.load_model(args, params, seed, out)
                for name, edit in edits.items():
                    pred, got = train.predict_split(args, model, data, params, TEST, edit)
                    error = (pred - data.pm[data.start[got][:, None] + HORIZON]) * data.stats.pm_std
                    curve, overall = station_rmse(data.station[got], error)
                    rows += [{"variant": variant, "edit": name, "seed": seed, "lead": lead, "rmse": v}
                             for lead, v in [("all", overall), *enumerate(curve, 1)]]
        if rows:
            report(args, analysis, block, cdir, pd.DataFrame(rows))

def report(args, analysis, block, cdir, rows):
    """ Average the seeds, save the table and print the overall change per variant and edit.
    """
    keys = ["variant", "seed", "lead"]
    rows = rows.merge(rows[rows["edit"] == "none"][keys + ["rmse"]], on=keys, suffixes=("", "_none"))
    rows["delta_rmse"] = rows["rmse"] - rows.pop("rmse_none")
    table = rows.groupby(["variant", "edit", "lead"], sort=False).agg(
        rmse=("rmse", "mean"), rmse_sd=("rmse", "std"), delta_rmse=("delta_rmse", "mean"),
        delta_sd=("delta_rmse", "std"), seeds=("seed", "nunique")).reset_index()
    table.to_csv(cdir / f"{analysis}.csv", index=False)
    print(f"[{analysis.upper()}] {args.model.upper()} "
          f"{names(args.sources) + ' -> ' if args.transfer else ''}{names(args.city)} "
          f"(longest gap: {args.longest_gap}, station completeness: {args.completeness}, "
          f"ablation: {args.data_ablation}, block: {block})")
    overall = table[table["lead"] == "all"].drop(columns="lead")
    print("    " + overall.to_string(index=False, float_format="%.3f").replace("\n", "\n    "))
    print(f"    -> {cdir / f'{analysis}.csv'}")

def main(analysis, models, make_edits):
    """ Run an analysis from the command line, under its own log.
    """
    ap = argparse.ArgumentParser()
    add_arguments(ap)
    args = read_arguments(ap)
    if args.model not in models:
        ap.error(f"--model for {analysis}: {', '.join(models)}")
    pair = f"{args.source}-" if args.transfer else ""
    with runlog.logged(f"{analysis}_{pair}{args.city}", args):
        rescore(args, analysis, make_edits)
