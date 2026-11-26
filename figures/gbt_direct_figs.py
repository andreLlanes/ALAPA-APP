""" Slide figure for the GBT's structure (M2, slide 27):

      m2_gbt_direct.png    O2's direct GBT: 72 separate tree models, one per forecast hour. Each gets
                           the same 21 features from the past 72 h plus the weather and time of the
                           hour it forecasts; none sees another model's forecast, so no arrows join
                           them (O2/models/gbt.py, direct mode, the default in O2/utils/args.py).

    python -E figures/gbt_direct_figs.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import shared  # puts O3 on the path; its schema is identical to O2's
from shared import AXIS, BLUE, BLUE_RAMP, INK, INK2, MUTED, PLANE
from Common.schema import LAG_OFFSETS, MET_COLS, ROLL_STATS, ROLL_WINDOWS, TIME_COLS

shared.use_style(12)
fig = plt.figure(figsize=(13.33, 6.1))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(2.5, 60)
ax.axis("off")


def box(x0, x1, yc, h, fill, edge, lw=1.2):
    ax.add_patch(FancyBboxPatch((x0, yc - h / 2), x1 - x0, h, boxstyle="round,pad=0,rounding_size=0.6",
                                fc=fill, ec=edge, lw=lw, zorder=2))


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13, color=INK2,
                                 lw=1.3, shrinkA=0, shrinkB=0, zorder=1))


# Columns (x) and rows (y): models 1, 2, 3, a gap for 4-71, then 72.
HIST, CHIP, TREE, OUT = (2, 26), (31, 50), (55, 71), (77, 95)
ROWS = [(1, 44.5), (2, 37), (3, 29.5), (None, 23), (72, 16.5)]
ROW_H = 5.6
UP, DOWN = 1.3, -1.4  # the history arrow above, the weather chip below, inside each row

n_hist = len(LAG_OFFSETS) + len(ROLL_WINDOWS) * len(ROLL_STATS)
lags = ", ".join(str(k) for k in LAG_OFFSETS[1:-1]) + f" and {LAG_OFFSETS[-1]}"
wins = ", ".join(str(w) for w in ROLL_WINDOWS[:-1]) + f" and {ROLL_WINDOWS[-1]}"

ax.text(2, 57, "GBT: 72 separate models, one for each forecast hour", fontsize=17, weight="semibold", va="center")
ax.text(2, 53.3, "t = the hour the forecast is made", fontsize=11.5, color=INK2, va="center")

# Column headers.
for (x0, x1), label in [(HIST, "Same for all 72 models"), (CHIP, "Different for each model"),
                        (TREE, "72 tree models"), (OUT, "Forecast")]:
    ax.text((x0 + x1) / 2, 49.6, label, ha="center", va="center", fontsize=11.5, color=INK2, weight="semibold")

# The shared history: one box spanning every row.
top, bottom = ROWS[0][1] + ROW_H / 2, ROWS[-1][1] - ROW_H / 2
box(*HIST, (top + bottom) / 2, top - bottom, BLUE_RAMP[0], BLUE)
hx = (HIST[0] + HIST[1]) / 2
ax.text(hx, top - 3.2, "Past 72 h of PM2.5", ha="center", va="center", fontsize=13, weight="semibold")
ax.text(hx, top - 6.2, f"{n_hist} features", ha="center", va="center", fontsize=11.5, color=INK2)
ax.text(hx, top - 10.5,
        f"{len(LAG_OFFSETS)} readings:\nnow and {lags}\nhours before\n\n"
        f"{len(ROLL_WINDOWS) * len(ROLL_STATS)} statistics:\n{', '.join(ROLL_STATS)} of the\nlast {wins} hours",
        ha="center", va="top", fontsize=11.5, linespacing=1.4)

for lead, y in ROWS:
    if lead is None:
        for x0, x1 in (CHIP, TREE, OUT):
            for dy in (-1.1, 0, 1.1):  # a vertical ellipsis; Segoe UI has no glyph for it
                ax.plot((x0 + x1) / 2, y + dy, "o", ms=3.2, color=MUTED)
        ax.text((TREE[0] + TREE[1]) / 2 + 4.2, y, "models 4–71", va="center", fontsize=10.5, color=MUTED)
        continue
    # Two inputs into each model: the shared history (top) and its own hour's weather and time.
    arrow(HIST[1], y + UP, TREE[0], y + UP)
    box(*CHIP, y + DOWN, 2.6, PLANE, AXIS, lw=1)
    ax.text((CHIP[0] + CHIP[1]) / 2, y + DOWN, f"weather & time at t+{lead}", ha="center", va="center", fontsize=11)
    arrow(CHIP[1], y + DOWN, TREE[0], y + DOWN)
    box(*TREE, y, ROW_H - 0.6, "white", INK2)
    ax.text((TREE[0] + TREE[1]) / 2, y, f"Tree model {lead}", ha="center", va="center", fontsize=12.5,
            weight="semibold")
    arrow(TREE[1], y, OUT[0], y)
    box(*OUT, y, 3.4, "white", BLUE)
    ax.text((OUT[0] + OUT[1]) / 2, y, f"PM2.5 at t+{lead}", ha="center", va="center", fontsize=12.5)

ax.text(2, 8.6, f"Weather & time = {len(MET_COLS)} weather values and {len(TIME_COLS)} time-of-day/week/year "
        "values for the hour being forecast.", fontsize=11, color=INK2, va="center")
ax.text(2, 5.2, "Each model is trained on its own and never sees another model's forecast, "
        "so there are no arrows between them.", fontsize=11, color=INK2, va="center")

path = shared.out_path("m2_gbt_direct.png")
fig.savefig(path, dpi=200)
print(f"Wrote {path}")
