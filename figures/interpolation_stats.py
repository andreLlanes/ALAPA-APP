""" Interpolation numbers for Metro Manila under O3's default settings, and the backup chart
    comparing the straight-line fill with copying the last reading.
    Prints: how many samples and history hours are filled, the longest gap per sample, and an
    accuracy check (real readings hidden, refilled both ways, compared with the truth).
    python figures/interpolation_stats.py                 numbers + backup chart
    python figures/interpolation_stats.py --candidates    also list real samples to use as the slide example
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shared  # puts O3 on the path; import before O3 modules
from shared import INK, INK2, MUTED, ORANGE
import config
from Common.schema import LOOKBACK_H, TIME_COL
from Common.splits import chronological_cuts
from data.load import loso_keys
from data.windows import city_arrays

parser = argparse.ArgumentParser()
parser.add_argument("--candidates", action="store_true", help="list example samples (one 3-5 h gap)")
args = parser.parse_args()

rows = shared.read_city("mm")
a = city_arrays(rows)
# O3's filters: station completeness (LOSO stations always kept), all forecast hours observed, and
# the longest history gap.
keep = (a["completeness"] * 100 >= config.DEFAULT_COMPLETENESS) | np.isin(a["keys"], loso_keys())
ok = keep[a["station"]] & a["scored"] & (a["gap"] <= config.DEFAULT_LONGEST_GAP)
start, station, gap = a["start"][ok], a["station"][ok], a["gap"][ok]
pm = a["pm"]
lookback = pm[start[:, None] + np.arange(LOOKBACK_H)]
miss = np.isnan(lookback)
print(f"Settings: completeness >= {config.DEFAULT_COMPLETENESS}%, longest gap <= {config.DEFAULT_LONGEST_GAP} h")
print(f"Samples kept: {len(start):,}")
print(f"Samples with at least one filled hour: {int(miss.any(1).sum()):,} ({miss.any(1).mean():.1%})")
print(f"History hours filled: {int(miss.sum()):,} of {miss.size:,} ({miss.mean():.2%})")
print("Longest gap per sample (hours: samples):", pd.Series(gap).value_counts().sort_index().to_dict())
print(f"Latest hour missing (carried forward): {int(miss[:, -1].sum()):,}")
print(f"First hour missing (copied back): {int(miss[:, 0].sum()):,}")

# Accuracy check on training-period hours: take k+2 consecutive observed hours of one station, hide
# the middle k, refill them with a straight line or by copying the last reading, and compare.
cuts = chronological_cuts(a["origin"][ok].astype("datetime64[h]"))
train_end = np.datetime64(cuts[0])
hours = ((pd.to_datetime(rows[TIME_COL], utc=True) - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(hours=1)).to_numpy()
row_station = a["row_station"]
in_train = hours.astype("datetime64[h]") <= train_end
kept_row = keep[row_station]
results = []
for k in range(1, config.DEFAULT_LONGEST_GAP + 1):
    span = k + 2
    idx = np.arange(len(pm) - span + 1)
    seg = pm[idx[:, None] + np.arange(span)]
    same_run = (row_station[idx] == row_station[idx + span - 1]) & (hours[idx + span - 1] - hours[idx] == span - 1)
    seg = seg[same_run & ~np.isnan(seg).any(1) & kept_row[idx] & in_train[idx + span - 1]]
    before, after, truth = seg[:, :1], seg[:, -1:], seg[:, 1:-1]
    line = before + np.arange(1, k + 1) / (k + 1) * (after - before)
    results.append({"gap_h": k, "cases": len(seg),
                    "line_error": float(np.abs(line - truth).mean()),
                    "copy_error": float(np.abs(before - truth).mean()),
                    "typical_level": float(np.median(truth))})
acc = pd.DataFrame(results)
print(f"\nAccuracy check (average error, µg/m³), training period up to {train_end}:")
print(acc.round(2).to_string(index=False))
print(f"Straight line is {1 - (acc.line_error / acc.copy_error).mean():.0%} lower on average")

shared.use_style(14)
fig, ax = plt.subplots(figsize=(8.5, 5.2))
fig.subplots_adjust(left=0.12, right=0.70, top=0.86, bottom=0.2)
fig.text(0.12, 0.965, "Interpolation error by gap length", fontsize=16, weight="semibold", va="top")
g, line_e, copy_e = acc.gap_h.to_numpy(), acc.line_error.to_numpy(), acc.copy_error.to_numpy()
ax.plot(g, copy_e, "-o", color=MUTED, lw=2, ms=8, mec="white", mew=2)
ax.plot(g, line_e, "-o", color=ORANGE, lw=2.5, ms=9, mec="white", mew=2)
ax.set_xlim(0.7, g[-1] + 0.3)
ax.set_ylim(0, np.ceil(copy_e.max()) + 0.5)
ax.set_xticks(g)
ax.set_xlabel("Length of the gap (hours)")
ax.set_ylabel("Average error (µg/m³)")
ax.text(g[-1] + 0.18, copy_e[-1], f"Copy the last reading\n{copy_e[-1]:.1f} at {g[-1]} h", va="center",
        fontsize=13, color=INK2, clip_on=False)
ax.text(g[-1] + 0.18, line_e[-1], f"Straight line (used)\n{line_e[-1]:.1f} at {g[-1]} h", va="center",
        fontsize=13, color=INK, weight="semibold", clip_on=False)
ax.text(1, copy_e[0] + 0.45, f"{copy_e[0]:.1f}", ha="center", fontsize=12, color=INK2)
ax.text(1, line_e[0] - 0.75, f"{line_e[0]:.1f}", ha="center", fontsize=12, color=INK)
fig.text(0.12, 0.03, "Check: real readings hidden, then refilled and compared with the truth.\n"
         f"Training period (to {pd.Timestamp(train_end):%b %Y}), {int(keep.sum())} stations, "
         f"{round(acc.cases.min() / 1000)}k–{round(acc.cases.max() / 1000)}k cases per gap length.", fontsize=10.5, color=MUTED)
fig.savefig(shared.out_path("backup_interpolation_accuracy.png"), dpi=200)
plt.close(fig)

if args.candidates:
    # One interior gap of 3-5 h, everything else observed, away from the edges of the history.
    origin = a["origin"][ok].astype("datetime64[h]")
    found = []
    for i in np.flatnonzero((gap >= 3) & (gap <= 5)):
        m = miss[i]
        pos = np.flatnonzero(m)
        if m[0] or m[-1] or m.sum() != gap[i] or pos[0] < 30 or pos[-1] > 66:
            continue
        found.append((a["keys"][station[i]], str(origin[i]), int(gap[i]), float(np.nanstd(lookback[i]))))
    c = pd.DataFrame(found, columns=["station", "origin_utc", "gap_h", "spread"])
    c = c.assign(day=c.origin_utc.str[:10]).sort_values("spread", ascending=False).drop_duplicates(["station", "day"])
    c.drop(columns="day").to_csv(shared.out_path("interpolation_candidates.csv"), index=False)
    print(f"\n{len(c)} candidate samples (most varied first); the top 15:")
    print(c.drop(columns="day").head(15).to_string(index=False))
print("saved to", shared.OUT)
