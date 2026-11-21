""" Figures for the interpolation slide (Chapter 4, Data Preprocessing): a real Metro Manila sample
    with its filled hours, and the four kinds of gap. The filling is done by O3's own interpolate(),
    so the charts show exactly what the models get.
    python figures/interpolation_figs.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import shared  # puts O3 on the path; import before O3 modules
from shared import AXIS, BLUE, INK, INK2, MUTED, ORANGE, PLANE
import config
from Common.schema import HORIZON_H, KEY_COL, LOOKBACK_H, PM25_COL, TIME_COL
from data.load import interpolate

# The real example, picked from `interpolation_stats.py --candidates` (one clear 5-hour gap).
# The zoom box and label positions below are tuned to this sample.
EXAMPLE_STATION = "openaq:4049149"
EXAMPLE_ORIGIN = pd.Timestamp("2025-09-09T10", tz="UTC")  # last history hour = when the forecast is made

shared.use_style(13)
fill = lambda history: interpolate(history[None, :])[0]

# ---------- A real Manila sample ----------
rows = shared.read_city("mm", [KEY_COL, TIME_COL, PM25_COL])
s = rows[rows[KEY_COL] == EXAMPLE_STATION].set_index(TIME_COL)[PM25_COL]
hours = pd.date_range(EXAMPLE_ORIGIN - pd.Timedelta(hours=LOOKBACK_H - 1),
                      EXAMPLE_ORIGIN + pd.Timedelta(hours=HORIZON_H), freq="h")
v = s.reindex(hours).to_numpy(float)
x = np.arange(-(LOOKBACK_H - 1), HORIZON_H + 1)  # 0 = the hour the forecast is made
past = x <= 0
filled = fill(v[past])
gap = np.flatnonzero(np.isnan(v[past]))
if not len(gap) or np.isnan(v[~past]).any():
    raise ValueError("the example must have a history gap and a fully measured forecast")
g0, g1 = x[gap[0]], x[gap[-1]]
a0, a1 = g0 - 1, g1 + 1  # the readings either side of the gap
fx = np.arange(a0, a1 + 1)
fy = filled[fx + LOOKBACK_H - 1]
before, after = v[a0 + LOOKBACK_H - 1], v[a1 + LOOKBACK_H - 1]

fig = plt.figure(figsize=(12.6, 2.95))
gs = fig.add_gridspec(1, 2, width_ratios=[3.3, 1], wspace=0.10, left=0.05, right=0.99, top=0.76, bottom=0.23)
ax, zx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
top = 46
ax.set_ylim(0, top)
ax.axvspan(0.5, HORIZON_H + 0.5, color=PLANE, zorder=0, lw=0)
ax.axvline(0.5, color=AXIS, lw=1.2, zorder=1)
ax.plot(x, v, color=BLUE, lw=2, zorder=3, solid_capstyle="round")
ax.plot(x, v, "o", color=BLUE, ms=3, zorder=3)
ax.plot(fx, fy, "--", color=ORANGE, lw=2, zorder=4)
ax.plot(x[gap], filled[gap], "o", mfc="white", mec=ORANGE, mew=2, ms=7, zorder=5)
ax.add_patch(Rectangle((a0 - 1.5, 4), (a1 - a0) + 3, 26, fill=False, ec=INK2, lw=1, zorder=6))
ax.text((a0 + a1) / 2, 31.5, "zoom →", ha="center", va="bottom", fontsize=11.5, color=INK2)
ax.set_xlim(-LOOKBACK_H - 0.5, HORIZON_H + 0.5)
ax.set_xticks([-72, -48, -24, 0, 24, 48, 72])
ax.set_xticklabels(["−72 h", "−48 h", "−24 h", "now", "+24 h", "+48 h", "+72 h"], fontsize=12)
ax.set_yticks([0, 20, 40])
ax.set_ylabel("PM2.5 (µg/m³)", fontsize=12)
ax.text(-36, top * 1.04, f"Past {LOOKBACK_H} hours (model input)", ha="center", va="bottom", fontsize=14, weight="semibold")
ax.text(36.5, top * 1.04, f"Next {HORIZON_H} hours (forecast target)", ha="center", va="bottom", fontsize=14, weight="semibold")
ax.text(-56, top * 0.93, "short gaps filled", ha="center", va="top", fontsize=12.5, color=ORANGE, weight="semibold")
ax.text(14, top * 0.93, "never filled", ha="center", va="top", fontsize=12.5, color=INK2)

lo, hi = a0 - 5, a1 + 5
m = (x >= lo) & (x <= hi)
zx.plot(x[m], v[m], color=BLUE, lw=2, zorder=3)
zx.plot(x[m], v[m], "o", color=BLUE, ms=5.5, zorder=3)
zx.plot(fx, fy, "--", color=ORANGE, lw=2, zorder=4)
zx.plot(x[gap], filled[gap], "o", mfc="white", mec=ORANGE, mew=2, ms=9, zorder=5)
zx.axvspan(g0 - 0.5, g1 + 0.5, color=ORANGE, alpha=0.08, lw=0, zorder=0)
zx.set_xlim(lo - 0.6, hi + 0.6)
zx.set_ylim(4, 30)
zx.set_xticks([])
zx.set_yticks([10, 20, 30])
zx.tick_params(labelsize=11)
zx.text(a0, before - 2.2, f"{before:.1f}", ha="center", va="top", fontsize=11.5, color=INK2)
zx.text(a1, after + 1.6, f"{after:.1f}", ha="center", va="bottom", fontsize=11.5, color=INK2)
zx.text((g0 + g1) / 2, 5.0, f"{len(gap)} missing hours", ha="center", va="bottom", fontsize=12, color=INK)
zx.set_title("Zoom on the filled gap", fontsize=13, weight="semibold", pad=8)
ax.plot([], [], "-o", color=BLUE, ms=4, lw=2, label="Measured")
ax.plot([], [], "--o", color=ORANGE, mfc="white", mec=ORANGE, mew=2, ms=7, lw=2, label="Filled")
fig.legend(loc="lower left", bbox_to_anchor=(0.045, -0.01), ncol=2, frameon=False, fontsize=12, handlelength=2.4)
made = EXAMPLE_ORIGIN.tz_convert(shared.LOCAL_TZ)
fig.text(0.99, 0.025, f"Real sample: Metro Manila station {EXAMPLE_STATION.split(':')[-1]} (OpenAQ); "
         f"forecast made {made.strftime('%I %p').lstrip('0')}, {made.day} {made:%b %Y}.",
         ha="right", fontsize=10.5, color=MUTED)
fig.savefig(shared.out_path("interpolation_example.png"), dpi=200)
plt.close(fig)

# ---------- The four kinds of gap (illustration) ----------
t = np.arange(24)
base = 20 + 8 * np.sin((t - 4) / 24 * 2 * np.pi * 1.6) + np.array(
    [0.6, -0.8, 0.3, 1.1, -0.4, 0.2, -1.0, 0.7, 0.1, -0.6, 0.9, -0.3, 0.4, -0.9, 0.5, 0.0,
     -0.5, 0.8, -0.2, 0.6, -0.7, 0.3, -0.1, 0.4])
limit = config.DEFAULT_LONGEST_GAP
cases = [
    ("1  Gap in the middle", "Straight line between\nthe readings on either side", np.r_[8:12], True),
    ("2  Gap at the start", "Copy the first reading back", np.r_[0:3], True),
    ("3  Latest hours missing", "Carry the last reading forward", np.r_[21:24], True),
    (f"4  Gap longer than {limit} hours", "Not filled: sample not used", np.r_[7:7 + limit + 2], False),
]
fig, axes = plt.subplots(1, 4, figsize=(12.6, 2.2), sharey=True)
fig.subplots_adjust(left=0.02, right=0.99, top=0.68, bottom=0.29, wspace=0.10)
for a, (title, rule, hole, is_filled) in zip(axes, cases):
    y = base.copy()
    y[hole] = np.nan
    a.grid(False)
    a.set_xticks([])
    a.set_yticks([])
    a.spines["left"].set_visible(False)
    a.set_ylim(5, 36)
    a.set_xlim(-1, 24)
    if is_filled:
        f = fill(y)
        a.plot(t, y, "-", color=BLUE, lw=2)
        a.plot(t, y, "o", color=BLUE, ms=3.5)
        lo_, hi_ = max(hole[0] - 1, 0), min(hole[-1] + 1, 23)
        a.plot(np.arange(lo_, hi_ + 1), f[lo_:hi_ + 1], "--", color=ORANGE, lw=2)
        a.plot(hole, f[hole], "o", mfc="white", mec=ORANGE, mew=2, ms=6.5)
    else:
        a.plot(t, y, "-o", color=AXIS, lw=2, ms=3.5)
        a.axvspan(hole[0] - 0.5, hole[-1] + 0.5, color=PLANE, lw=0)
        a.text((hole[0] + hole[-1]) / 2, 29, f"{len(hole)} h", ha="center", fontsize=12, color=INK2)
    a.set_title(title, fontsize=13.5, weight="semibold", loc="left", pad=6)
    a.text(0, -0.06, rule, transform=a.transAxes, va="top", fontsize=12.5, color=INK if is_filled else INK2)
    a.axvline(-0.6, color=AXIS, lw=1)
    a.axvline(23.6, color=AXIS, lw=1)
axes[1].text(-0.3, 34, f"← start of the {LOOKBACK_H} h", fontsize=10.5, color=MUTED, va="top")
axes[2].text(23.3, 34, "now →", fontsize=10.5, color=MUTED, ha="right", va="top")
fig.text(0.02, 0.95, "Gap handling by gap type", fontsize=14.5, weight="semibold", va="top")
fig.text(0.99, 0.95, "Illustration, not real data", fontsize=10.5, color=MUTED, ha="right", va="top")
fig.savefig(shared.out_path("interpolation_cases.png"), dpi=200)
plt.close(fig)

print("saved to", shared.OUT)
