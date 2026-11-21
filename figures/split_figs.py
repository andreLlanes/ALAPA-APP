""" Figures for the split slide (Chapter 4, Data Preprocessing), from the cleaned data and O3's own
    filters and split (data/load.py prepare_city, Common/splits.py):
      split_timeline.png  every kept Metro Manila window, by the date its forecast is made,
                          coloured train / validation / test
      split_purge.png     a zoom on the train/validation cut: the 72 h purge, the last training
                          window and the first validation window
    python figures/split_figs.py                                    O3 settings (60/20/20, completeness 60)
    python figures/split_figs.py --train 0.70 --val 0.15 --completeness 70   O2's settings
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import shared  # puts O3 on the path; import before O3 modules
from shared import BLUE, INK, INK2, ORANGE
import config
from Common.schema import HORIZON_H, LOOKBACK_H
from matplotlib.patches import Patch, Rectangle
from Common.splits import PURGE_HOURS, PURGED, TEST, TRAIN, VAL, chronological_cuts, partition_labels
from data.load import loso_keys
from data.windows import city_arrays

parser = argparse.ArgumentParser()
parser.add_argument("--train", type=float, default=0.60, help="training share of windows (O3: 0.60, O2: 0.70)")
parser.add_argument("--val", type=float, default=0.20, help="validation share of windows (O3: 0.20, O2: 0.15)")
parser.add_argument("--completeness", type=int, default=config.DEFAULT_COMPLETENESS, help="station cut, %%")
parser.add_argument("--longest-gap", type=int, default=config.DEFAULT_LONGEST_GAP, help="history gap limit, h")
args = parser.parse_args()
AQUA, PURGE_GREY = "#1baf7a", "#c3c2b7"
PARTS = {TRAIN: ("Train", BLUE), VAL: ("Validation", AQUA), TEST: ("Test", ORANGE), PURGED: ("Purged", PURGE_GREY)}
shares = (args.train, args.val, 1 - args.train - args.val)
suffix = "" if (args.train, args.val) == (0.60, 0.20) else f"_{round(args.train * 100)}_{round(args.val * 100)}"

# The same filters and split as O3's prepare_city.
shared.use_style(11)
a = city_arrays(shared.read_city("mm"))
keep = (a["completeness"] * 100 >= args.completeness) | np.isin(a["keys"], loso_keys())
ok = keep[a["station"]] & a["scored"] & (a["gap"] <= args.longest_gap)
origin = a["origin"][ok].astype("datetime64[h]")
cuts = chronological_cuts(origin, args.train, args.val)
label = partition_labels(origin, cuts)
local = pd.DatetimeIndex(origin).tz_localize("UTC").tz_convert(shared.LOCAL_TZ).tz_localize(None)
cut_local = [pd.Timestamp(c).tz_localize("UTC").tz_convert(shared.LOCAL_TZ).tz_localize(None) for c in cuts]
print("cuts (UTC):", [str(c) for c in cuts], " counts:",
      {PARTS[k][0]: int((label == k).sum()) for k in PARTS})

# ---------- Every kept window, by forecast date ----------
days = pd.Series(1, index=local).groupby([local.normalize(), label]).sum().unstack(fill_value=0)
days = days.reindex(pd.date_range(days.index.min(), days.index.max(), freq="D"), fill_value=0)
fig, ax = plt.subplots(figsize=(7.3, 3.6))
fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.14)
bottom = np.zeros(len(days))
for code in (TRAIN, PURGED, VAL, TEST):
    if code in days:
        ax.bar(days.index, days[code], width=1.0, bottom=bottom, color=PARTS[code][1], lw=0, align="edge")
        bottom += days[code].to_numpy()
top = bottom.max() * 1.3
ax.set_ylim(0, top)
for c in cut_local:
    ax.axvline(c, color=INK, lw=0.9)
    first_cut = c == cut_local[0]  # labels sit outside the narrow validation span
    ax.text(c + pd.Timedelta(days=-4 if first_cut else 4), top * 0.97, f"{c.day} {c:%b %Y}", va="top", fontsize=8.5,
            color=INK, ha="right" if first_cut else "left")
for code, share in zip((TRAIN, VAL, TEST), shares):
    mid = local[label == code][np.argsort(local[label == code])][(label == code).sum() // 2]  # half its windows either side
    ax.text(mid, top * 0.72, f"{PARTS[code][0]}\n{share:.0%}", ha="center", va="center", fontsize=9.5,
            color=INK, weight="semibold")
ax.set_ylabel("Windows per day", fontsize=9.5)
ax.yaxis.set_major_locator(plt.MaxNLocator(3))
ax.tick_params(labelsize=9)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.set_xlim(days.index[0], days.index[-1] + pd.Timedelta(days=1))
fig.text(0.08, 0.965, "Kept windows per day by data split, Metro Manila", fontsize=11.5,
         weight="semibold", va="top")
fig.savefig(shared.out_path(f"split_timeline{suffix}.png"), dpi=200)
plt.close(fig)

# ---------- Zoom on the train/validation cut ----------
cut = cut_local[0]
lo, hi = cut - pd.Timedelta(days=7), cut + pd.Timedelta(days=4)
near = (local >= lo) & (local < hi)
hour_part = pd.Series(label[near], index=local[near]).groupby(level=0).first()
h = pd.Timedelta(hours=1)
# The last training window's forecast ends before the cut; the first validation window starts after it.
last_train = local[label == TRAIN].max()
first_val = local[label == VAL].min()
fig, ax = plt.subplots(figsize=(7.3, 3.0))
fig.subplots_adjust(left=0.03, right=0.97, top=0.84, bottom=0.12)
ax.grid(False)
for t0, code in hour_part.items():  # strip: which part the windows made at each hour belong to
    ax.add_patch(Rectangle((mdates.date2num(t0), 0), 1 / 24, 0.45, color=PARTS[int(code)][1], lw=0))
ax.text(lo + 2 * h, 0.55, "Each hour as a forecast time ('now'): which part its windows belong to", fontsize=8.5,
        color=INK2, va="bottom")
for y, now, name, colour in ((2.0, last_train, "Last training window", BLUE),
                             (1.25, first_val, "First validation window", AQUA)):
    ax.plot([now - (LOOKBACK_H - 1) * h, now + h], [y, y], color=colour, lw=6, solid_capstyle="butt")
    ax.plot([now + h, now + (HORIZON_H + 1) * h], [y, y], color=colour, lw=6, alpha=0.4, solid_capstyle="butt")
    ax.plot([now + h, now + h], [y - 0.13, y + 0.13], color=INK, lw=1)
    ax.text(now - (LOOKBACK_H - 1) * h, y + 0.16, f"{name}: past 72 h (dark), then the next 72 h it forecasts (light)",
            fontsize=8.5, color=INK, va="bottom")
ax.axvline(cut, color=INK, lw=1.0)
ax.text(cut, 2.55, f"cut: {cut.day} {cut:%b %Y}, " + f"{cut:%I %p}".lstrip("0"), ha="center", va="bottom",
        fontsize=8.5, color=INK, bbox=dict(fc="white", ec="none", pad=1), zorder=5)
purge_start = cut - pd.Timedelta(hours=PURGE_HOURS)
ax.annotate("", (purge_start, 0.22), (cut, 0.22), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9))
ax.text(purge_start + (cut - purge_start) / 2, 0.24, f"{PURGE_HOURS} h purge", ha="center", va="bottom",
        fontsize=8.5, color=INK, bbox=dict(fc=PURGE_GREY, ec="none", pad=1))
ax.set_ylim(0, 2.85)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlim(lo, hi)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.tick_params(labelsize=9)
handles = [Patch(color=PARTS[c][1], label=PARTS[c][0]) for c in (TRAIN, PURGED, VAL)]
fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.97, 0.97), ncol=3, frameon=False, fontsize=8.5,
           handlelength=1.2)
fig.text(0.03, 0.965, f"{PURGE_HOURS}-hour purge at the train–validation cut", fontsize=11.5, weight="semibold",
         va="top")
fig.savefig(shared.out_path(f"split_purge{suffix}.png"), dpi=200)
plt.close(fig)
print("saved to", shared.OUT)
