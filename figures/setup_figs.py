""" Figures for the experimental setup slide (Chapter 5), read from the code the Kaggle runs used (O2):
      setup_experiments.png  the training setups each model was run in (O2/models/transfer.py variants)
      setup_grid.png         the hyperparameter grid searched on validation RMSE (O2/grid_tuning.json)
    python figures/setup_figs.py
"""

import ast
import json
from itertools import product

import numpy as np
import matplotlib.pyplot as plt

import shared
from shared import AXIS, GRID, INK, INK2, MUTED

O2 = shared.REPO / "O2"
GRID_FILE = json.loads((O2 / "grid_tuning.json").read_text())
SOURCES = ("Bangkok", "Los Angeles", "Bangkok + LA")
SEEDS = 30  # O2/config.py DEFAULT_SEEDS (seeds 1-30); seed 0 tunes


def constants(path, names):
    """ Read module-level tuple constants from a source file without importing it (no torch needed).
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {t.id: ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
            for t in node.targets if isinstance(t, ast.Name) and t.id in names}


variants = constants(O2 / "models" / "transfer.py", {"NN_VARIANTS", "GBT_VARIANTS"})
nn, gbt = variants["NN_VARIANTS"], variants["GBT_VARIANTS"]
shared.use_style(11)


def canvas(width, height):
    fig = plt.figure(figsize=(width, height))
    cv = fig.add_axes([0, 0, 1, 1])
    cv.set_xlim(0, width)
    cv.set_ylim(0, height)
    cv.axis("off")
    return fig, cv


# ---------- Training setups ----------
has = lambda *names: {"LSTM": any(n in nn for n in names), "GNN": any(n in nn for n in names),
                      "GBT": any(n in gbt for n in names)}
rows = [("Single city", "one city's own windows: Manila, Bangkok or Los Angeles", {"LSTM": True, "GNN": True, "GBT": True}),
        ("Zero-shot", "source city only, then applied to Manila as is", has("zeroshot")),
        ("Pooled", "source and Manila windows together, from scratch", has("pooled")),
        ("Parameter transfer", "pretrained on the source, fine-tuned on Manila\n(encoder frozen, or all layers)",
         has("param_frozen", "param_full")),
        ("MMD transfer", "pretrained on the source while aligned to Manila (MMD),\nthen fine-tuned (frozen or all layers)",
         has("mmd_frozen", "mmd_full")),
        ("MMD-weighted instances", "source windows weighted by each station's MMD\ndistance to Manila, plus Manila's windows",
         {"LSTM": False, "GNN": False, "GBT": "weighted" in gbt})]
n_configs = 3 * 3 + len(SOURCES) * (2 * len(nn) + len(gbt))
widths = [2.0, 4.6, 0.75, 0.75, 0.75]
W, row_h = sum(widths) + 0.4, 0.52
H = 0.75 + 0.5 + row_h * len(rows) + 0.36 + 0.6
fig, cv = canvas(W, H)
x = 0.2 + np.concatenate([[0], np.cumsum(widths)[:-1]])
cv.text(0.2, H - 0.18, "Training setups per model", fontsize=12, weight="semibold", va="top")
y = H - 0.95
for xi, w, name in zip(x, widths, ("Setup", "Trained on", "LSTM", "GNN", "GBT")):
    centred = name in ("LSTM", "GNN", "GBT")
    cv.text(xi + (w / 2 if centred else 0), y, name, fontsize=10, weight="semibold", va="center",
            ha="center" if centred else "left")
cv.plot([0.15, W - 0.15], [y - 0.26] * 2, color=INK, lw=0.8)
GROUP_H = 0.36  # the label row above the transfer setups
for i, (name, what, models) in enumerate(rows):
    yy = y - 0.26 - row_h * (i + 0.5) - (GROUP_H if i else 0)
    cv.text(x[0], yy, name, fontsize=10, weight="semibold", va="center")
    cv.text(x[1], yy, what, fontsize=9, color=INK2, va="center", linespacing=1.15)
    for xi, w, m in zip(x[2:], widths[2:], ("LSTM", "GNN", "GBT")):
        cv.text(xi + w / 2, yy, "✓" if models[m] else "–", fontsize=12 if models[m] else 10,
                color=INK if models[m] else AXIS, ha="center", va="center", family="Segoe UI Symbol")
    if i == 0:  # the transfer setups follow, under their own label
        cv.plot([0.15, W - 0.15], [yy - row_h / 2] * 2, color=INK, lw=0.8)
        cv.text(x[0], yy - row_h / 2 - GROUP_H / 2,
                f"Transfer into Manila, from each source: {', '.join(SOURCES)}",
                fontsize=9.5, color=INK2, weight="semibold", va="center")
        cv.plot([0.15, W - 0.15], [yy - row_h / 2 - GROUP_H] * 2, color=AXIS, lw=0.8)
    elif i < len(rows) - 1:
        cv.plot([0.15, W - 0.15], [yy - row_h / 2] * 2, color=GRID, lw=0.8)
cv.plot([0.15, W - 0.15], [yy - row_h / 2] * 2, color=INK, lw=0.8)
cv.text(0.2, 0.12, f"{n_configs} configurations in all. Settings are tuned on validation RMSE with seed 0 (zero-shot "
        f"reuses the source city's); each is then trained and scored with {SEEDS} seeds (1 to {SEEDS}).",
        fontsize=8.5, color=MUTED, va="bottom")
fig.savefig(shared.out_path("setup_experiments.png"), dpi=200)
plt.close(fig)

# ---------- Hyperparameter grid ----------
LABELS = {"hidden": "hidden size", "layers": "layers", "lr": "learning rate", "batch_size": "batch size",
          "k": "neighbours per station (k)", "max_depth": "tree depth", "eta": "learning rate (eta)",
          "num_boost_round": "trees", "subsample": "row sampling", "colsample_bytree": "column sampling",
          "fine_tune_lr": "fine-tuning learning rate", "lambda": "MMD weight (λ)",
          "tau_scales": "weight scale τ (× median station distance)"}
fmt = lambda v: f"{v:g}"
blocks = [("LSTM", GRID_FILE["lstm"]), ("GNN", GRID_FILE["gnn"]), ("GBT", GRID_FILE["gbt"]),
          ("Transfer", {"fine_tune_lr": GRID_FILE["fine_tune"]["fine_tune_lr"], "lambda": GRID_FILE["mmd"]["lambda"],
                        "tau_scales": GRID_FILE["tau_scales"]})]
notes = {"Transfer": "parameter and MMD fine-tuning; MMD adds λ; GBT weighting adds τ to the tree grid"}
n_lines = sum(len(b) for _, b in blocks)
W, line_h = 8.4, 0.3
H = 0.75 + 0.45 + line_h * n_lines + 0.22 * len(blocks) + 0.55
fig, cv = canvas(W, H)
cv.text(0.2, H - 0.18, "Hyperparameter search grid", fontsize=12, weight="semibold", va="top")
xs = (0.2, 1.55, 4.75)
y = H - 0.95
for xi, name in zip(xs, ("Model", "Setting", "Values tried")):
    cv.text(xi, y, name, fontsize=10, weight="semibold", va="center")
cv.plot([0.15, W - 0.15], [y - 0.2] * 2, color=INK, lw=0.8)
y -= 0.2 + 0.11
for b, (model, grid) in enumerate(blocks):
    combos = len(list(product(*grid.values()))) if model != "Transfer" else None
    y0 = y
    for j, (key, values) in enumerate(grid.items()):
        yy = y - line_h * (j + 0.5)
        cv.text(xs[1], yy, LABELS[key], fontsize=9.5, color=INK2, va="center")
        cv.text(xs[2], yy, ", ".join(fmt(v) for v in values), fontsize=9.5, color=INK, va="center")
    cv.text(xs[0], y0 - line_h * 0.5, model, fontsize=10, weight="semibold", va="center")
    cv.text(xs[0], y0 - line_h * 1.5, f"{combos} combinations" if combos else "", fontsize=8.5, color=MUTED, va="center")
    y -= line_h * len(grid) + 0.11
    cv.plot([0.15, W - 0.15], [y] * 2, color=INK if b == len(blocks) - 1 else GRID, lw=0.8)
    y -= 0.11
cv.text(0.2, 0.12, "Every combination is trained once with seed 0 and the lowest validation RMSE is kept. "
        "Values with one option are fixed.", fontsize=8.5, color=MUTED, va="bottom")
fig.savefig(shared.out_path("setup_grid.png"), dpi=200)
plt.close(fig)
print(f"{n_configs} configurations; NN variants {nn}; GBT variants {gbt}; saved to {shared.OUT}")
