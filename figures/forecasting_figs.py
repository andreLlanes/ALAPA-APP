""" Figures for the forecasting slide (Chapter 4), from the cleaned data and O3's own code
    (data/batch.py for the station graph, data/load.py for the GBT inputs):
      forecasting_graph.png         the GNN's station graph at one forecast hour: every Metro Manila
                                    station with a window then, linked to its k nearest stations
      forecasting_gbt_inputs.png    what the GBT takes from one real history: PM2.5 lags and rolling
                                    statistics
    Both use the window of windowing_matrix.png (station 4049149, forecast made 6 PM, 9 Sep 2025).
    python figures/forecasting_figs.py [--k 4]
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shared  # puts O3 on the path; import before O3 modules
from shared import BLUE, GRID, INK, INK2, MUTED, ORANGE
import config
from Common.schema import LAG_OFFSETS, LOOKBACK_H, ROLL_STATS, ROLL_WINDOWS
from data.batch import build_station_graph, station_distances_km
from data.load import gbt_features, interpolate, loso_keys
from data.windows import city_arrays

EXAMPLE_STATION = "openaq:4049149"
EXAMPLE_ORIGIN = pd.Timestamp("2025-09-09T10", tz="UTC")  # last history hour = when the forecast is made
EPOCH = pd.Timestamp(0, tz="UTC")
LINK, ABSENT = "#9d9c96", "#b9b8b1"

parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, default=4, help="neighbours per station (tuned among 4, 8, 12; the chart shows one)")
args = parser.parse_args()

shared.use_style(11)
a = city_arrays(shared.read_city("mm"))
ex = int(np.flatnonzero(a["keys"] == EXAMPLE_STATION)[0])

# ---------- The GNN's graph at one forecast hour ----------
# The same filters as O3's prepare_city; the graph is built from the stations with a window then.
keep = (a["completeness"] * 100 >= config.DEFAULT_COMPLETENESS) | np.isin(a["keys"], loso_keys())
ok = keep[a["station"]] & a["scored"] & (a["gap"] <= config.DEFAULT_LONGEST_GAP)
origin_h = (EXAMPLE_ORIGIN - EPOCH) // pd.Timedelta(hours=1)
# Manila's graphs are small (most hours have one to three stations with a window), so the chart shows
# the busiest hour and states the typical size.
per_hour = pd.Series(a["station"][ok]).groupby(a["origin"][ok]).nunique()
graph_h = int(per_hour.idxmax())
graph_when = pd.Timestamp(graph_h, unit="h", tz="UTC").tz_convert(shared.LOCAL_TZ)
here = np.unique(a["station"][ok & (a["origin"] == graph_h)])
lat, lon = a["lat"][here], a["lon"][here]
edge_index, weight = build_station_graph(station_distances_km(lat, lon), args.k)
pairs = {}
for s, d, w in zip(*edge_index, weight):  # each link is stored both ways; draw it once
    pairs[(min(s, d), max(s, d))] = w
absent = np.setdiff1d(np.arange(len(a["keys"])), here)

FW, FH, MAP_H = 3.9, 6.6, 4.5
fig = plt.figure(figsize=(FW, FH))
ax = fig.add_axes([0.62 / FW, 1.45 / FH, shared.manila_map_width(MAP_H) / FW, MAP_H / FH])
shared.manila_basemap(ax)
for (s, d), w in pairs.items():
    ax.plot(lon[[s, d]], lat[[s, d]], color=INK2, lw=0.6 + 2.4 * w, solid_capstyle="round", zorder=2)
ax.plot(a["lon"][absent], a["lat"][absent], "o", mfc="white", mec=ABSENT, mew=1.0, ms=4, zorder=4)
ax.plot(lon, lat, "o", color=BLUE, ms=6, mec="white", mew=0.8, zorder=5)
graph_at = f"{graph_when:%I %p}".lstrip("0") + f", {graph_when.day} {graph_when:%b %Y}"
fig.text(0.04, 1 - 0.12 / FH, "GNN station graph, Metro Manila", fontsize=11.5, weight="semibold", va="top")
legend = [(dict(marker="o", color=BLUE, ms=6, mec="white", lw=0), f"station with a window at this hour ({len(here)})"),
          (dict(marker="o", mfc="white", mec=ABSENT, mew=1.0, ms=4, lw=0), f"no window at this hour: left out ({len(absent)})"),
          (dict(color=INK2, lw=1.8), f"link to one of its {args.k} nearest (thicker = closer)")]
for i, (style, label) in enumerate(legend):
    y = (1.12 - i * 0.24) / FH
    xs = [0.05, 0.11] if "marker" not in style else [0.08]
    fig.lines.append(plt.Line2D(xs, [y] * len(xs), transform=fig.transFigure, **style))
    fig.text(0.15, y, label, fontsize=9.5, color=INK2, va="center")
fig.text(0.05, 0.08 / FH, f"{graph_at}, the busiest hour; most hours have 1 to {int(per_hour.quantile(0.9))} "
         f"stations.\nk = {args.k} shown; k was tuned among 4, 8 and 12.", fontsize=8.5, color=MUTED, va="bottom")
fig.savefig(shared.out_path("forecasting_graph.png"), dpi=200)
plt.close(fig)
print(f"graph at {graph_at}: {len(here)} stations, {len(pairs)} links; stations per hour: median "
      f"{per_hour.median():.0f}, 90th percentile {per_hour.quantile(0.9):.0f}, max {per_hour.max()}")

# ---------- What the GBT takes from the past 72 hours ----------
o = int(np.flatnonzero((a["station"] == ex) & (a["origin"] == origin_h))[0])
raw = a["pm"][a["start"][o]:a["start"][o] + LOOKBACK_H]
history = interpolate(raw[None, :])[0]
filled = np.isnan(raw)
feats = gbt_features(history[None, :])[0]
lag_values = dict(zip(LAG_OFFSETS, feats[:len(LAG_OFFSETS)]))
rolls = feats[len(LAG_OFFSETS):].reshape(len(ROLL_WINDOWS), len(ROLL_STATS))
x = np.arange(-(LOOKBACK_H - 1), 1)  # 0 = now

fig, (ax, rx) = plt.subplots(2, 1, figsize=(7.3, 4.4), sharex=True, gridspec_kw=dict(height_ratios=[2.3, 1.25]))
fig.subplots_adjust(left=0.08, right=0.97, top=0.87, bottom=0.12, hspace=0.12)
ax.plot(x, np.where(filled, np.nan, history), color=BLUE, lw=1.6, zorder=2)
f0, f1 = np.flatnonzero(filled)[[0, -1]] if filled.any() else (None, None)
if filled.any():  # the filled hours, as on the interpolation slide (one gap in this window)
    ax.plot(x[max(f0 - 1, 0):f1 + 2], history[max(f0 - 1, 0):f1 + 2], "--", color=ORANGE, lw=1.6, zorder=3)
lx = -np.array(LAG_OFFSETS)
ax.plot(lx, [lag_values[k] for k in LAG_OFFSETS], "o", color=INK, ms=5.5, mec="white", mew=1, zorder=4)
top = np.nanmax(history) * 1.25
ax.set_ylim(0, top)
ax.set_ylabel("PM2.5 (µg/m³)", fontsize=9.5)
ax.yaxis.set_major_locator(plt.MaxNLocator(4))
ax.tick_params(labelsize=9)
ax.text(-70.5, top * 0.97, f"● 9 lagged readings: now and 1, 2, 3, 6, 12, 24, 48 and 71 hours before",
        fontsize=8.5, color=INK, va="top")
if filled.any():
    ax.text(x[(f0 + f1) // 2], history[f0:f1 + 1].min() - top * 0.04, "filled", color=ORANGE, fontsize=8.5,
            ha="center", va="top")
rx.grid(False)
rx.spines["left"].set_visible(False)
rx.set_yticks([])
for i, w in enumerate(ROLL_WINDOWS):
    y = len(ROLL_WINDOWS) - 1 - i
    rx.add_patch(plt.Rectangle((-w + 0.5, y - 0.32), w, 0.64, color=GRID, lw=0))
    vals = ", ".join(f"{name} {v:.1f}" for name, v in zip(ROLL_STATS, rolls[i]))
    rx.text(-w - 0.5, y, f"last {w} h:  {vals}", ha="right", va="center", fontsize=8.5, color=INK2)
rx.set_ylim(-0.6, len(ROLL_WINDOWS) - 0.4)
rx.text(-70.5, len(ROLL_WINDOWS) - 0.45, "12 rolling statistics (µg/m³)", fontsize=8.5, color=INK, va="top")
rx.set_xticks([-71, -48, -24, -12, -6, -3, 0])  # -2 and -1 are lags too, unlabelled for room
rx.set_xticklabels(["−71", "−48", "−24", "−12", "−6", "−3", "now"], fontsize=8.5)
rx.set_xlabel("hours before the forecast is made", fontsize=9)
rx.set_xlim(-LOOKBACK_H + 0.5, 1.5)
for t in lx:
    ax.axvline(t, color=GRID, lw=0.7, zorder=0)
fig.text(0.08, 0.965, "GBT input features from the past 72 hours", fontsize=11.5, weight="semibold", va="top")
fig.savefig(shared.out_path("forecasting_gbt_inputs.png"), dpi=200)
plt.close(fig)
print("saved to", shared.OUT)
