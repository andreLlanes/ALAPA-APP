""" Slide figure for M3 (transfer), from the Kaggle runs (O2/artifacts):

      m3_transfer.png    one table per model: Metro Manila test RMSE of the model trained on Manila
                         only (M2) and with transfer from Bangkok and from Los Angeles (M3), overall
                         (comparison.csv) and at 1, 24 and 72 hours ahead (lead_comparison.csv).
                         Each transfer row is the milestone's method; where it was run with the
                         encoder frozen and with every layer fine-tuned, the run with the lower
                         validation RMSE is shown, the same rule as the M5 slide (m5_figs.py).

    python -E figures/m3_transfer_figs.py
"""

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shared
from shared import AXIS, GRID, INK, INK2, MUTED, PLANE

sys.path.insert(0, str(shared.REPO / "O2" / "evals"))
from app_export import HOW, MODELS, SOURCES, TAG, val_rmse  # O2's validation pick, as on the M5 slide

ART = shared.REPO / "O2" / "artifacts"
SETTINGS = "gap6_comp70_abl100"
LEADS = (1, 24, 72)
NAMES = {"lstm": "LSTM", "gnn": "GNN", "gbt": "GBT"}

overall = pd.read_csv(ART / "comparison.csv")
overall = overall[(overall.city == "mm") & (overall.settings == SETTINGS)]
lead = pd.read_csv(ART / "lead_comparison.csv")
lead = lead[(lead.city == "mm") & (lead.settings == SETTINGS) & lead.lead.isin(LEADS)]


def pick(table, method, source, variant):
    rows = table[(table.method == method) & ((table.source == source) if source else table.source.isna())
                 & (table.variant == variant)]
    return rows


tables, ci = {}, 0.0
for m, variants in MODELS.items():
    method = NAMES[m]
    rows = [("Manila only (M2)", None, "none")]
    for src, src_name in SOURCES.items():
        if len(variants) > 1:
            scores = {v: val_rmse(ART / m / f"{src}-mm" / TAG / v) for v in variants}
            best = min(scores, key=scores.get)
            print(f"{src}-mm {method}: {best}  (validation RMSE {', '.join(f'{k} {s:.4f}' for k, s in scores.items())})")
        else:
            best = variants[0]
        rows.append((f"{src_name} transfer (M3)", src, best))
    out = []
    for label, src, variant in rows:
        o = pick(overall, method, src, variant).squeeze()
        by_lead = pick(lead, method, src, variant).set_index("lead")
        if not isinstance(o, pd.Series) or len(by_lead) != len(LEADS):
            raise SystemExit(f"missing results for {method} {src} {variant}")
        ci = max(ci, (o.rmse_ci_high - o.rmse_ci_low) / 2,
                 *((by_lead.rmse_ci_high - by_lead.rmse_ci_low) / 2))
        out.append((label, variant, [o.rmse] + [by_lead.at[h, "rmse"] for h in LEADS]))
    tables[m] = out
    base = out[0][2]
    for label, _, vals in out[1:]:
        print(f"  {method} {label}: " + ", ".join(
            f"{c} {v:.2f} ({v - b:+.2f})" for c, v, b in zip(["overall", "1 h", "24 h", "72 h"], vals, base)))
n_stations = pd.read_json(ART / "lstm" / "mm" / TAG / "none" / "seed1" / "metrics.json", typ="series")["n_stations"]

# ---------- Three tables side by side ----------
shared.use_style(11)
COLS = ["All 72 h", "1 h ahead", "24 h ahead", "72 h ahead"]
LABEL_W, COL_W, GAP, PAD = 1.6, 0.8, 0.4, 0.2
TABLE_W = LABEL_W + COL_W * len(COLS)
W = PAD * 2 + TABLE_W * 3 + GAP * 2
ROW_H, TOP = 0.58, 1.55
H = TOP + 0.62 + ROW_H * 3 + 0.5
fig = plt.figure(figsize=(W, H))
cv = fig.add_axes([0, 0, 1, 1])
cv.set_xlim(0, W)
cv.set_ylim(0, H)
cv.axis("off")
cv.text(PAD, H - 0.2, "Manila only vs. transfer: Metro Manila test RMSE (µg/m³)", fontsize=13, weight="semibold", va="top")
cv.text(PAD, H - 0.55, "The GBT with Bangkok transfer beats Manila only at all 72 hours ahead. LSTM transfer helps only in hours 1–3, "
        "GNN transfer at no hour.",
        fontsize=10.5, color=INK2, va="top")

for t, (m, rows) in enumerate(tables.items()):
    x0 = PAD + t * (TABLE_W + GAP)
    xs = [x0 + LABEL_W + COL_W * (i + 0.5) for i in range(len(COLS))]
    y_title = H - TOP + 0.38
    cv.text(x0, y_title, NAMES[m], fontsize=12, weight="semibold", va="center")
    methods = {HOW[v] for _, v, _ in rows[1:]}
    cv.text(x0, y_title - 0.27, "Transfer: " + " / ".join(sorted(methods)), fontsize=8.5, color=MUTED, va="center")
    y_head = H - TOP - 0.2
    for xi, c in zip(xs, COLS):
        cv.text(xi, y_head, c, fontsize=10, color=INK2, ha="center", va="center")
    y = y_head - 0.22
    cv.plot([x0, x0 + TABLE_W], [y] * 2, color=INK, lw=0.8)
    base = rows[0][2]
    for j, (label, _, vals) in enumerate(rows):
        yy = y - ROW_H * (j + 0.5)
        cv.text(x0, yy, label, fontsize=10, weight="semibold" if j == 0 else "normal", va="center")
        for i, (xi, v, b) in enumerate(zip(xs, vals, base)):
            better = j > 0 and v < b
            if better:
                cv.add_patch(plt.Rectangle((xi - COL_W / 2 + 0.04, yy - ROW_H / 2 + 0.05), COL_W - 0.08,
                                           ROW_H - 0.1, color=PLANE, lw=0, zorder=0))
            if j == 0:
                cv.text(xi, yy, f"{v:.2f}", fontsize=10.5, ha="center", va="center")
            else:
                cv.text(xi, yy + 0.09, f"{v:.2f}", fontsize=10.5, ha="center", va="center",
                        weight="bold" if better else "normal")
                cv.text(xi, yy - 0.14, f"{v - b:+.2f}".replace("-", "−"), fontsize=8, color=MUTED,
                        ha="center", va="center")
        cv.plot([x0, x0 + TABLE_W], [yy - ROW_H / 2] * 2, color=INK if j == len(rows) - 1 else GRID, lw=0.8)

fig.text(PAD / W, 0.12 / H,
         f"Test period 27 Mar to 27 Jun 2026, {n_stations} scored stations, each weighted equally; each value is the "
         f"mean of 30 seeds (95% CI within ±{ci:.2f}). Small numbers: change from Manila only.\n"
         "Bold and shaded: lower error than Manila only. Where frozen and fully fine-tuned versions were both run, "
         "the one with the lower validation RMSE is shown, as on the M5 slide.",
         fontsize=8.5, color=MUTED, va="bottom", linespacing=1.45)
path = shared.out_path("m3_transfer.png")
fig.savefig(path, dpi=200)
print(f"largest 95% CI half-width {ci:.3f}; wrote {path}")
