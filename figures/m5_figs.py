""" Figure for the M5 overview slide (Chapter 5), from the Kaggle runs (O2/artifacts):
      m5_metrics.png  Metro Manila test scores on all five metrics (RMSE, MAE, MBE, IOA, R²) for the
                      two baselines, the three models trained on Manila only (M2) and with transfer
                      from Bangkok and from Los Angeles (M3); each model score is the mean of 30 seeds
    Scores come from O2's own evaluation (artifacts/comparison.csv: stations averaged with equal
    weight, then seeds). The transfer shown for each model is the milestone's (LSTM parameter
    transfer, GNN parameter transfer with MMD, GBT MMD-weighted instances); where it was run with the
    encoder frozen and with every layer fine-tuned, the run with the lower validation RMSE is shown.
    python figures/m5_figs.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shared
from shared import AXIS, GRID, INK, INK2, MUTED, PLANE

ART = shared.REPO / "O2" / "artifacts"
TAG = "direct_gap6_comp70_abl100"
SEEDS = range(1, 31)
MIN_WINDOWS = 100  # O2/config.py MIN_TEST_WINDOWS
MODELS = {"LSTM": ("param_frozen", "param_full"), "GNN": ("mmd_frozen", "mmd_full"), "GBT": ("weighted",)}
SOURCES = {"bk": "Bangkok", "la": "Los Angeles"}
METRICS = [("rmse", "RMSE (µg/m³)", "lower is better"), ("mae", "MAE (µg/m³)", "lower is better"),
           ("mbe", "MBE (µg/m³)", "closer to 0 is better"), ("ioa", "IOA", "higher is better (max 1)"),
           ("r2", "R²", "higher is better (max 1)")]


def val_rmse(folder):
    """ Mean over seeds of the station-averaged validation RMSE (scored stations only). """
    per_seed = []
    for s in SEEDS:
        with np.load(folder / f"seed{s}" / "val_predictions.npz") as f:
            keys, obs, pred = f["location_key"], f["obs"].astype(float), f["pred"].astype(float)
        names, codes = np.unique(keys, return_inverse=True)
        scored = np.bincount(codes) >= MIN_WINDOWS
        sq = pd.DataFrame((pred - obs) ** 2).groupby(codes)
        per_seed.append((np.sqrt(sq.sum().sum(1) / sq.count().sum(1)))[scored].mean())
    return float(np.mean(per_seed))


table = pd.read_csv(ART / "comparison.csv")
table = table[(table.city == "mm") & (table.settings == "gap6_comp70_abl100")]
row = lambda method, source=None, variant=None: table[
    (table.method == method) & ((table.source == source) if source else table.source.isna())
    & ((table.variant == variant) if variant else True)].squeeze()

columns = [("Baselines", "Persistence", row("persistence")), ("Baselines", "Climatology", row("climatology"))]
columns += [("Manila only (M2)", m, row(m, variant="none")) for m in MODELS]
chosen = {}
for src, src_name in SOURCES.items():
    for m, variants in MODELS.items():
        scores = {v: val_rmse(ART / m.lower() / f"{src}-mm" / TAG / v) for v in variants} if len(variants) > 1 else {}
        best = min(scores, key=scores.get) if scores else variants[0]
        chosen[(src, m)] = best
        columns.append((f"Transfer from {src_name} (M3)", m, row(m, src, best)))
        print(f"{src}-mm {m}: {best}" + (f"  (validation RMSE {scores})" if scores else ""))
for group, name, r in columns:
    if not isinstance(r, pd.Series):
        raise SystemExit(f"no single comparison.csv row for {group} / {name}")
models_only = [r for g, _, r in columns if g != "Baselines"]
ci = max((r.rmse_ci_high - r.rmse_ci_low) / 2 for r in models_only)
n_stations = pd.read_json(ART / "lstm" / "mm" / TAG / "none" / "seed1" / "metrics.json", typ="series")["n_stations"]

# ---------- The table ----------
shared.use_style(11)
LABEL_W, COL_W, GROUP_GAP = 2.1, 0.78, 0.22
groups = list(dict.fromkeys(g for g, _, _ in columns))
xs, x = [], 0.2 + LABEL_W
for i, (g, _, _) in enumerate(columns):
    if i and g != columns[i - 1][0]:
        x += GROUP_GAP
    xs.append(x + COL_W / 2)
    x += COL_W
W = x + 0.2
ROW_H, TOP = 0.5, 1.25
H = TOP + ROW_H * len(METRICS) + 0.62
fig = plt.figure(figsize=(W, H))
cv = fig.add_axes([0, 0, 1, 1])
cv.set_xlim(0, W)
cv.set_ylim(0, H)
cv.axis("off")
cv.text(0.2, H - 0.18, "Metro Manila test scores", fontsize=12, weight="semibold", va="top")
y_group, y_model = H - 0.66, H - 0.98
for g in groups:
    idx = [i for i, c in enumerate(columns) if c[0] == g]
    left, right = xs[idx[0]] - COL_W / 2 + 0.04, xs[idx[-1]] + COL_W / 2 - 0.04
    cv.text((left + right) / 2, y_group, g, fontsize=9.5, weight="semibold", ha="center", va="center")
    cv.plot([left, right], [y_group - 0.16] * 2, color=AXIS, lw=0.8)
for xi, (_, name, _) in zip(xs, columns):
    cv.text(xi, y_model, name, fontsize=9.5, color=INK2, ha="center", va="center")
y = H - TOP
cv.plot([0.15, W - 0.15], [y] * 2, color=INK, lw=0.8)
for j, (key, label, hint) in enumerate(METRICS):
    yy = y - ROW_H * (j + 0.5)
    values = np.array([r[key] for _, _, r in columns])
    target = np.abs(values) if key == "mbe" else (-values if key in ("ioa", "r2") else values)
    best = int(np.argmin(target))
    cv.text(0.2, yy + 0.08, label, fontsize=10, weight="semibold", va="center")
    cv.text(0.2, yy - 0.12, hint, fontsize=8, color=MUTED, va="center")
    cv.add_patch(plt.Rectangle((xs[best] - COL_W / 2 + 0.05, yy - ROW_H / 2 + 0.06), COL_W - 0.1, ROW_H - 0.12,
                               color=PLANE, lw=0, zorder=0))
    for i, (xi, v) in enumerate(zip(xs, values)):
        text = f"{v:+.2f}" if key == "mbe" else f"{v:.2f}"
        cv.text(xi, yy, text.replace("-", "−"), fontsize=10, ha="center", va="center",
                weight="bold" if i == best else "normal", color=INK)
    cv.plot([0.15, W - 0.15], [yy - ROW_H / 2] * 2, color=INK if j == len(METRICS) - 1 else GRID, lw=0.8)
how = {"param_frozen": "parameter transfer, encoder frozen", "param_full": "parameter transfer, all layers",
       "mmd_frozen": "parameter transfer with MMD, encoder frozen", "mmd_full": "parameter transfer with MMD, all layers",
       "weighted": "MMD-weighted instances"}
picked = "; ".join(f"{m} {how[chosen[('bk', m)]]}" + ("" if chosen[("bk", m)] == chosen[("la", m)]
                   else f" (LA: {how[chosen[('la', m)]].split(', ')[-1]})") for m in MODELS)
fig.text(0.2 / W, 0.1 / H,
         f"Test period 27 Mar to 27 Jun 2026, {n_stations} scored stations, each weighted equally. Model scores are "
         f"the mean of 30 seeds (95% CI of RMSE within ±{ci:.2f}). Shaded: best in the row.\n"
         f"Transfer shown: {picked}.", fontsize=8.5, color=MUTED, va="bottom", linespacing=1.45)
fig.savefig(shared.out_path("m5_metrics.png"), dpi=200)
plt.close(fig)
print(f"max RMSE CI half-width {ci:.3f}; saved to {shared.OUT}")
