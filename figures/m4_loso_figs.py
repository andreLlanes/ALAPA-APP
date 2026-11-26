""" Figure for the M4 results slide, from O3's leave-one-station-out table (O3/artifacts/loso_comparison.csv,
    written by O3/evals/loso.py):

      m4_loso.png    RMSE of the regression-kriged map at stations it never saw: each of the 10 withheld
                     stations (O3/loso_stations.json) is left out of training, validation and kriging, and
                     the map's value at its location is scored against its measured PM2.5 over O3's test
                     period. One bar per model (mean of 3 seeds, 95% CI) and the kriged persistence baseline.
                     These are O3 numbers (60/20/20 split), not comparable with O2's M2/M3 scores.

    python -E figures/m4_loso_figs.py
"""

import pandas as pd
import matplotlib.pyplot as plt

import shared
from shared import BLUE, INK, INK2, MUTED, ORANGE

AQUA = "#1baf7a"  # model colours as on the M2/M3 slides (m2m3_figs.py)
ROWS = [("GBT", AQUA), ("GNN", ORANGE), ("LSTM", BLUE), ("persistence", "#c3c2b7")]
LABELS = {"persistence": "Persistence, kriged\n(baseline)"}
FILTERS = "gap6_comp60"  # O3's data settings: longest gap 6 h, completeness 60%
TEST = "20 Mar – 27 Jun 2026"  # O3's test period (forecasts made), as in its training logs

t = pd.read_csv(shared.O3_DIR / "artifacts" / "loso_comparison.csv", keep_default_na=False)
t = t[(t.band == "all") & (t.filters == FILTERS) & t.variant.isin(["none", ""])].set_index("method")
for m, _ in ROWS:
    r = t.loc[m]
    print(f"{m}: RMSE {r.rmse:.2f}" + (f" (95% CI {float(r.rmse_ci_low):.2f}-{float(r.rmse_ci_high):.2f})"
                                       if r.rmse_ci_low != "" else "") + f", {int(r.folds)} folds, {int(r.seeds)} seeds")

shared.use_style(12)
fig, ax = plt.subplots(figsize=(8, 3.9))
fig.subplots_adjust(left=0.2, right=0.95, top=0.86, bottom=0.3)
ax.grid(axis="x", color=shared.GRID, lw=0.8)
ax.grid(axis="y", visible=False)
ys = range(len(ROWS))[::-1]
for y, (m, colour) in zip(ys, ROWS):
    r = t.loc[m]
    ax.barh(y, r.rmse, height=0.56, color=colour, zorder=2)
    end = r.rmse
    if r.rmse_ci_low != "":
        lo, hi = float(r.rmse_ci_low), float(r.rmse_ci_high)
        ax.plot([lo, hi], [y, y], color=INK, lw=1.4, zorder=3, solid_capstyle="butt")
        for x in (lo, hi):
            ax.plot([x, x], [y - 0.1, y + 0.1], color=INK, lw=1.4, zorder=3)
        end = hi
    ax.text(end + 0.25, y, f"{r.rmse:.2f}", va="center", fontsize=12, color=INK,
            weight="semibold" if m == ROWS[0][0] else "normal")
ax.set_yticks(list(ys))
ax.set_yticklabels([LABELS.get(m, m) for m, _ in ROWS], fontsize=12)
ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 14.5)
ax.set_xlabel("RMSE at the hidden station (µg/m³), lower is better")
ax.spines["left"].set_visible(False)

fig.text(0.02, 0.95, "How well the map predicts places it never saw", fontsize=14.5, weight="semibold", va="top")
r0 = t.loc[ROWS[0][0]]
fig.text(0.02, 0.03,
         f"Leave-one-station-out, Metro Manila: each of {int(r0.folds)} stations is hidden in turn from training and kriging, "
         f"and the map's value at its\nlocation is compared with its measured PM2.5 over O3's test period ({TEST}). "
         f"Mean of {int(r0.seeds)} seeds; lines show the 95% interval.\nPersistence: each other station's last reading, "
         "kriged the same way. O3's own split, so not comparable with the M2/M3 scores.",
         fontsize=9, color=MUTED, va="bottom", linespacing=1.45)

path = shared.out_path("m4_loso.png")
fig.savefig(path, dpi=200)
print(f"wrote {path}")
