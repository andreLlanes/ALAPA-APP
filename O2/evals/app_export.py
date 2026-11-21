""" Export O2's error progression for the web app: the Metro Manila test RMSE at every lead hour
    (1-72), per model, for the Manila-only run and the Bangkok and Los Angeles transfer runs, with
    persistence and climatology for reference. The app shows it in its own panel, apart from O3's
    maps; the two folders use different splits, so their numbers are never compared.

    The transfer variant shown per model and source is picked on validation RMSE with the same
    rule as the M5 slide (figures/m5_figs.py): the mean over seeds of the station-averaged
    validation RMSE, over stations with at least MIN_WINDOWS validation windows.

    Run from the repo root after O2's analysis has written lead_comparison.csv:
        python O2/evals/app_export.py
    Writes frontend/public/data/o2_tests.json (only that file; O3's map files are not touched).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "O2" / "artifacts"
OUT = REPO / "frontend" / "public" / "data" / "o2_tests.json"
SETTINGS = "gap6_comp70_abl100"
TAG = f"direct_{SETTINGS}"
SEEDS = range(1, 31)
LEADS = 72
MIN_WINDOWS = 100  # O2/config.py MIN_TEST_WINDOWS, as in figures/m5_figs.py
# Transfer variants that compete per model (the others are not on M5), as in figures/m5_figs.py.
MODELS = {"lstm": ("param_frozen", "param_full"), "gnn": ("mmd_frozen", "mmd_full"), "gbt": ("weighted",)}
SOURCES = {"bk": "Bangkok", "la": "Los Angeles"}
HOW = {"none": "trained on Metro Manila only",
       "param_frozen": "parameter transfer, encoder frozen", "param_full": "parameter transfer, all layers",
       "mmd_frozen": "parameter transfer with MMD, encoder frozen",
       "mmd_full": "parameter transfer with MMD, all layers", "weighted": "MMD-weighted instances"}


def val_rmse(folder: Path) -> float:
    """ Mean over seeds of the station-averaged validation RMSE (scored stations only). """
    per_seed = []
    for s in SEEDS:
        with np.load(folder / f"seed{s}" / "val_predictions.npz") as f:
            keys, obs, pred = f["location_key"], f["obs"].astype(float), f["pred"].astype(float)
        _, codes = np.unique(keys, return_inverse=True)
        scored = np.bincount(codes) >= MIN_WINDOWS
        sq = pd.DataFrame((pred - obs) ** 2).groupby(codes)
        per_seed.append((np.sqrt(sq.sum().sum(1) / sq.count().sum(1)))[scored].mean())
    return float(np.mean(per_seed))


def curve(table: pd.DataFrame, method: str, source=None, variant=None) -> pd.DataFrame:
    """ The 72 lead rows of one configuration, in lead order. """
    rows = table[(table.method == method)
                 & (table.source.isna() if source is None else table.source == source)
                 & (table.variant.isna() if variant is None else table.variant == variant)]
    rows = rows.sort_values("lead")
    if len(rows) != LEADS or not np.array_equal(rows.lead.to_numpy(), np.arange(1, LEADS + 1)):
        raise SystemExit(f"lead_comparison.csv: expected {LEADS} leads for {method} {source} {variant}, "
                         f"found {len(rows)}")
    return rows


def values(col: pd.Series) -> list:
    return [None if not np.isfinite(v) else round(float(v), 2) for v in col]


def line(rows: pd.DataFrame, id_: str, label: str, variant: str) -> dict:
    seeds = int(rows.seeds.min())
    if seeds != len(SEEDS):
        raise SystemExit(f"{label} {variant}: {seeds} seeds in lead_comparison.csv, expected {len(SEEDS)}")
    return {"id": id_, "label": label, "variant": variant, "how": HOW[variant], "seeds": seeds,
            "rmse": values(rows.rmse), "lo": values(rows.rmse_ci_low), "hi": values(rows.rmse_ci_high)}


def main():
    table = pd.read_csv(ART / "lead_comparison.csv")
    table = table[(table.city == "mm") & (table.settings == SETTINGS)]
    if table.empty:
        raise SystemExit(f"lead_comparison.csv has no Metro Manila rows at {SETTINGS}")

    models = {}
    for m, variants in MODELS.items():
        lines = [line(curve(table, m.upper(), None, "none"), "mm", "Manila only", "none")]
        for src, name in SOURCES.items():
            if len(variants) > 1:
                scores = {v: val_rmse(ART / m / f"{src}-mm" / TAG / v) for v in variants}
                best = min(scores, key=scores.get)
                print(f"{src}-mm {m.upper()}: {best}  (validation RMSE "
                      + ", ".join(f"{v} {s:.3f}" for v, s in scores.items()) + ")")
            else:
                best = variants[0]
                print(f"{src}-mm {m.upper()}: {best}")
            lines.append(line(curve(table, m.upper(), src, best), src, f"{name} transfer", best))
        models[m] = {"lines": lines}

    # Test facts for the panel, from a Manila-only run: the forecast origins it was scored on and
    # the stations scored (at least MIN_WINDOWS test windows).
    run = ART / "lstm" / "mm" / TAG / "none" / "seed1"
    with np.load(run / "predictions.npz") as f:
        origins = f["origin"].astype("datetime64[h]")
        with_windows = len(np.unique(f["location_key"]))
    scored = int(pd.read_csv(run / "lead_curve.csv")["n_stations"].max())
    hour = lambda t: f"{pd.Timestamp(t):%Y-%m-%dT%H}:00Z"

    out = {"version": 1, "source": "O2/artifacts/lead_comparison.csv", "metric": "rmse",
           "settings": {"split": "70/15/15", "completeness": 70, "longest_gap": 6},
           "test": {"first_origin_utc": hour(origins.min()), "last_origin_utc": hour(origins.max()),
                    "stations_scored": scored, "stations_with_windows": with_windows},
           "seeds": len(SEEDS), "leads": LEADS,
           "baselines": {b: values(curve(table, b).rmse) for b in ("persistence", "climatology")},
           "models": models}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.part")
    tmp.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    tmp.replace(OUT)
    print(f"wrote {OUT.relative_to(REPO)} ({OUT.stat().st_size / 1024:.1f} KB): test origins "
          f"{out['test']['first_origin_utc']} to {out['test']['last_origin_utc']}, {scored} stations scored")
    for m, entry in models.items():
        print(f"  {m.upper()}: " + " | ".join(f"{l['label']} ({l['variant']}) lead 1 {l['rmse'][0]}, "
                                              f"24 {l['rmse'][23]}, 72 {l['rmse'][71]}" for l in entry["lines"]))


if __name__ == "__main__":
    main()
