""" Figures for the evaluation setup slide (Chapter 4), from the cleaned data and the projects' own
    settings (no trained model needed):
      evaluation_baselines.png  one Metro Manila test window and the two reference forecasts every
                                model is compared with: persistence (the last reading, held for 72 h)
                                and climatology (the station's training-period mean at each hour of
                                day), with their error at each forecast hour
      evaluation_loso.png       the leave-one-station-out test stations of the regression-kriging
                                evaluation (O3/loso_stations.json), by density band
    The temporal models were trained with O2's settings (70/15/15 split, station completeness 70), so
    the example window is from O2's test period; the leave-one-station-out folds are O3's.
    python figures/evaluation_figs.py
"""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shared  # puts O3 on the path; import before O3 modules
from shared import BLUE, BLUE_RAMP, GRID, INK, INK2, MUTED, PLANE
from Common.schema import HORIZON_H, LOOKBACK_H, TIME_COL
from Common.splits import TEST, TRAIN, chronological_cuts, partition_labels
from data.load import LOSO_PATH, covered_rows
from data.windows import city_arrays

# O2's data settings (the Kaggle runs) and the example window, from its test period.
O2_TRAIN, O2_VAL, O2_COMPLETENESS, LONGEST_GAP = 0.70, 0.15, 70, 6
MIN_OBS_PER_HOUR = 10  # climatology: training hours needed at every hour of day (both configs)
EXAMPLE_STATION = "openaq:5976206"
EXAMPLE_ORIGIN = pd.Timestamp("2026-04-02T10", tz="Asia/Manila")  # when the forecast is made
PERSIST, CLIM = INK, "#8f6bbd"

shared.use_style(11)
df = shared.read_city("mm")
a = city_arrays(df)

# ---------- One test window and the two baselines ----------
# The same filters and split as O2's load_city (no stations protected for LOSO there).
keep = a["completeness"] * 100 >= O2_COMPLETENESS
ids = np.flatnonzero(keep[a["station"]] & a["scored"] & (a["gap"] <= LONGEST_GAP))
label = partition_labels(a["origin"][ids].astype("datetime64[h]"),
                         chronological_cuts(a["origin"][ids].astype("datetime64[h]"), O2_TRAIN, O2_VAL))
origin_h = (EXAMPLE_ORIGIN.tz_convert("UTC") - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(hours=1)
ex = int(np.flatnonzero(a["keys"] == EXAMPLE_STATION)[0])
w = ids[(a["station"][ids] == ex) & (a["origin"][ids] == origin_h)]
if len(w) != 1 or label[np.searchsorted(ids, w[0])] != TEST:
    raise SystemExit(f"{EXAMPLE_STATION} at {EXAMPLE_ORIGIN} is not an O2 test window")
start = a["start"][w[0]]
history = a["pm"][start:start + LOOKBACK_H]
obs = a["pm"][start + LOOKBACK_H:start + LOOKBACK_H + HORIZON_H]

# Persistence: the most recent reading at or before the forecast time, held for all 72 hours.
persistence = np.full(HORIZON_H, history[np.isfinite(history)][-1])
# Climatology: the station's mean PM2.5 at each local hour of day, over the measured hours its
# training windows cover (the city-wide means if any hour of day has too few).
local_hour = pd.to_datetime(df[TIME_COL], utc=True).dt.tz_convert(shared.LOCAL_TZ).dt.hour.to_numpy()
fit = covered_rows(a["start"][ids[label == TRAIN]], len(a["pm"])) & ~np.isnan(a["pm"])


def lookup(rows):
    counts = np.bincount(local_hour[rows], minlength=24)
    return None if (counts < MIN_OBS_PER_HOUR).any() else np.bincount(local_hour[rows], a["pm"][rows], 24) / counts


own = lookup(fit & (a["row_station"] == ex))
climatology = (own if own is not None else lookup(fit))[local_hour[start + LOOKBACK_H:start + LOOKBACK_H + HORIZON_H]]
rmse = {name: float(np.sqrt(np.mean((f - obs) ** 2))) for name, f in (("p", persistence), ("c", climatology))}

past, ahead = np.arange(-(LOOKBACK_H - 1), 1), np.arange(1, HORIZON_H + 1)
fig, (ax, ex_ax) = plt.subplots(2, 1, figsize=(7.3, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
fig.subplots_adjust(left=0.08, right=0.84, top=0.88, bottom=0.17, hspace=0.12)
for a_ in (ax, ex_ax):
    a_.axvspan(0.5, HORIZON_H + 0.5, color=PLANE, lw=0, zorder=0)
    a_.axvline(0.5, color=INK, lw=0.9, zorder=1)
ax.plot(past, history, color=BLUE, lw=1.5, zorder=3)
ax.plot(ahead, obs, color=BLUE, lw=1.5, zorder=3)
ax.plot(ahead, persistence, "--", color=PERSIST, lw=1.6, zorder=4)
ax.plot(ahead, climatology, color=CLIM, lw=1.6, zorder=4)
top = np.nanmax([history.max(), obs.max(), climatology.max()]) * 1.22
ax.set_ylim(0, top)
ax.set_ylabel("PM2.5 (µg/m³)", fontsize=9.5)
ax.yaxis.set_major_locator(plt.MaxNLocator(4))
ax.tick_params(labelsize=9)
ax.text(-70, top * 0.97, "Past 72 hours (model input)", fontsize=8.5, color=INK2, va="top")
ax.text(2, top * 0.97, "Next 72 hours: scored at every hour", fontsize=8.5, color=INK2, va="top")
ends = {"measured": (obs[-1], BLUE), f"persistence\nRMSE {rmse['p']:.1f}": (persistence[-1], PERSIST),
        f"climatology\nRMSE {rmse['c']:.1f}": (climatology[-1], CLIM)}
order = sorted(ends.items(), key=lambda kv: kv[1][0])
placed = []
for name, (y, colour) in order:  # direct labels at the right edge, pushed apart
    y = max(y, placed[-1] + top * 0.16) if placed else y
    placed.append(y)
    ax.text(HORIZON_H + 2.5, y, name, fontsize=8.5, color=INK, va="center", linespacing=1.1,
            bbox=dict(fc="white", ec="none", pad=0.5))
    ax.plot([HORIZON_H + 0.6, HORIZON_H + 2.2], [y, y], color=colour, lw=1.6, clip_on=False,
            ls="--" if colour == PERSIST else "-")
ex_ax.plot(ahead, np.abs(persistence - obs), "--", color=PERSIST, lw=1.4)
ex_ax.plot(ahead, np.abs(climatology - obs), color=CLIM, lw=1.4)
ex_ax.set_ylim(0, None)
ex_ax.yaxis.set_major_locator(plt.MaxNLocator(3))
ex_ax.tick_params(labelsize=9)
ex_ax.set_ylabel("Error\n(µg/m³)", fontsize=9.5)
ex_ax.set_xticks([-71, -48, -24, 0, 24, 48, 72])
ex_ax.set_xticklabels(["−71", "−48", "−24", "now", "+24", "+48", "+72"], fontsize=9)
ex_ax.set_xlim(-LOOKBACK_H + 0.5, HORIZON_H + 0.5)
ex_ax.set_xlabel("hours from when the forecast is made", fontsize=9)
made = f"{EXAMPLE_ORIGIN:%I %p}".lstrip("0") + f", {EXAMPLE_ORIGIN.day} {EXAMPLE_ORIGIN:%b %Y}"
fig.text(0.08, 0.965, "Persistence and climatology baselines, one test window", fontsize=11.5,
         weight="semibold", va="top")
fig.text(0.08, 0.015, f"Station {EXAMPLE_STATION.split(':')[-1]}, Metro Manila; forecast made {made} (test period). "
         "RMSE over the 72 forecast hours.", fontsize=8.5, color=MUTED, va="bottom")
fig.savefig(shared.out_path("evaluation_baselines.png"), dpi=200)
plt.close(fig)
print(f"example {EXAMPLE_STATION} {made}: persistence RMSE {rmse['p']:.2f}, climatology RMSE {rmse['c']:.2f} "
      f"({'station' if own is not None else 'city-wide'} lookup)")

# ---------- The leave-one-station-out test stations ----------
loso = json.loads(LOSO_PATH.read_text())
folds = pd.DataFrame(loso["folds"])
pool = pd.DataFrame(loso["training_pool"])
bands = loso["bands"]
others = pool[~pool.location_key.isin(folds.location_key)]
colours = [BLUE_RAMP[4], BLUE_RAMP[2], BLUE_RAMP[1], BLUE_RAMP[0]]  # core (dense) -> periphery (sparse)

FW, FH, MAP_H = 3.9, 6.8, 4.5
fig = plt.figure(figsize=(FW, FH))
ax = fig.add_axes([0.62 / FW, 1.75 / FH, shared.manila_map_width(MAP_H) / FW, MAP_H / FH])
shared.manila_basemap(ax)
ax.plot(others.longitude, others.latitude, "o", mfc="white", mec=MUTED, mew=0.9, ms=4, zorder=4)
for b, colour in zip(bands, colours):
    f = folds[folds.band == b["band"]]
    ax.plot(f.longitude, f.latitude, "o", color=colour, ms=8, mec=INK, mew=0.9, zorder=5)
fig.text(0.04, 1 - 0.12 / FH, "Leave-one-station-out test\nstations, Metro Manila", fontsize=11.5,
         weight="semibold", va="top")
legend = [(dict(marker="o", color=c, ms=8, mec=INK, mew=0.9, lw=0),
           f"{b['name']}: {b['n_selected']} withheld ({b['knn_km_min']:.1f}–{b['knn_km_max']:.1f} km)")
          for b, c in zip(bands, colours)]
legend.append((dict(marker="o", mfc="white", mec=MUTED, mew=0.9, ms=4, lw=0),
               f"other stations ({len(others)}): their forecasts are kriged"))
fig.text(0.05, 1.5 / FH, "Withheld station, by density band\n(km to its 3rd-nearest eligible station)",
         fontsize=9, color=INK, va="top")
for i, (style, text) in enumerate(legend):
    y = (1.05 - i * 0.22) / FH
    fig.lines.append(plt.Line2D([0.08], [y], transform=fig.transFigure, **style))
    fig.text(0.15, y, text, fontsize=9, color=INK2, va="center")
fig.savefig(shared.out_path("evaluation_loso.png"), dpi=200)
plt.close(fig)
print(f"LOSO: {len(folds)} folds from {loso['n_eligible']} eligible stations; {len(others)} others; saved to {shared.OUT}")
