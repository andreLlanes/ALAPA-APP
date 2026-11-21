""" Figure for the M4 overview slide (Chapter 5), from the web app's map data (frontend/public/data,
    made by O3/spatial/export.py from O3's LSTM run, `train.py --map true`):
      m4_surfaces.png  the regression-kriged 100 m PM2.5 surface for Metro Manila at the start of the
                       forecast and the same morning hour on each of the next three days
    The map is a past forecast from O3's test period (a backtest) that used recorded weather. The
    script also scores the map against the station readings the app ships (readings.json), the way
    the app does: forecast at each station's 100 m cell against its reading, every station-hour with
    both, split into the stations whose forecasts were kriged and the ones that weren't.
    python figures/m4_figs.py
"""

import gzip
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pyproj import Transformer

import shared
from shared import AXIS, INK, INK2, MUTED

LEADS = (1, 24, 48, 72)  # the first forecast hour, then the same hour on each following day
PM_CMAP = LinearSegmentedColormap.from_list("pm", ["#f3edf7", "#c7b2dd", "#8f6bbd", "#5e3c99", "#36195e"])

manifest, layer = shared.web_map()  # the LSTM's map
readings = json.loads((shared.WEB_DATA / "readings.json").read_text())
g = manifest["grid"]
with gzip.open(shared.WEB_DATA / layer["pm25"]["file"]) as f:
    code = np.frombuffer(f.read(), np.uint8).reshape(manifest["horizon"], g["rows"], g["cols"])
pm = np.where(code == layer["pm25"]["nodata"], np.nan, layer["pm25"]["max"] * (code / 254.0) ** 2)
origin = pd.Timestamp(manifest["origin_utc"]).tz_convert(shared.LOCAL_TZ)
to_grid = Transformer.from_crs(4326, g["epsg"], always_xy=True)
to_lonlat = Transformer.from_crs(g["epsg"], 4326, always_xy=True)

# ---------- Check against the readings (same rule as the app's Checks panel) ----------
# As in the app (readings.ts): reading index `lookback` is the first forecast hour (the issue time).
if pd.Timestamp(readings["start_utc"]) + pd.Timedelta(hours=readings["lookback"]) != pd.Timestamp(manifest["issued_utc"]):
    raise SystemExit("readings.json does not line up with the map's forecast hours")
errors = {True: [], False: []}
for s in readings["stations"]:
    x, y = to_grid.transform(s["lon"], s["lat"])
    col, row = int((x - g["x0"]) // g["cell"]), int((g["y0"] - y) // g["cell"])
    if not (0 <= row < g["rows"] and 0 <= col < g["cols"]):
        continue
    obs = np.array([np.nan if v is None else v for v in s["pm25"][readings["lookback"]:]], float)
    d = pm[:, row, col] - obs
    errors[s["in_map"]].append(d[np.isfinite(d)])
check = {}
for used, parts in errors.items():
    d = np.concatenate(parts)
    check[used] = dict(stations=sum(len(p) > 0 for p in parts), hours=len(d), rmse=float(np.sqrt(np.mean(d ** 2))),
                       mae=float(np.mean(np.abs(d))), bias=float(np.mean(d)))
    print(f"{'kriged' if used else 'not kriged'} stations: {check[used]['stations']}, {check[used]['hours']} "
          f"station-hours, RMSE {check[used]['rmse']:.2f}, MAE {check[used]['mae']:.2f}, bias {check[used]['bias']:+.2f}")

# ---------- The surface across the forecast ----------
shared.use_style(11)
x = g["x0"] + np.arange(g["cols"] + 1) * g["cell"]  # cell edges, for pcolormesh
y = g["y0"] - np.arange(g["rows"] + 1) * g["cell"]
lon, lat = to_lonlat.transform(*np.meshgrid(x, y))
stations = pd.DataFrame(manifest["stations"])
shown = [pm[k - 1] for k in LEADS]
lo, hi = np.nanpercentile(np.concatenate([s[np.isfinite(s)] for s in shown]), [1, 99])
at = lambda ts: f"{ts:%I %p}".lstrip("0") + f", {ts.day} {ts:%b}"

MAP_H = 3.5
map_w = shared.manila_map_width(MAP_H)
GAP, LEFT = 0.12, 0.25
FW, FH = LEFT + len(LEADS) * map_w + (len(LEADS) - 1) * GAP + 1.05, MAP_H + 1.5
fig = plt.figure(figsize=(FW, FH))
for i, (lead, values) in enumerate(zip(LEADS, shown)):
    ax = fig.add_axes([(LEFT + i * (map_w + GAP)) / FW, 0.62 / FH, map_w / FW, MAP_H / FH])
    ax.set_facecolor(shared.WATER)
    ax.grid(False)
    mesh = ax.pcolormesh(lon, lat, values, cmap=PM_CMAP, vmin=lo, vmax=hi, shading="flat", rasterized=True)
    ax.plot(stations.lon, stations.lat, "o", color="white", ms=3.4, mec=INK, mew=0.8, zorder=5)
    ax.set_xlim(shared.MM_BBOX[0], shared.MM_BBOX[2])
    ax.set_ylim(shared.MM_BBOX[1], shared.MM_BBOX[3])
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(True)
        side.set_color(AXIS)
    ax.set_title(f"+{lead} h", fontsize=10.5, weight="semibold", loc="left", pad=16)
    ax.text(0, 1.012, at(origin + pd.Timedelta(hours=lead)), transform=ax.transAxes, fontsize=8.5, color=INK2,
            va="bottom")
    if i == 0:
        ax.text(120.935, 14.495, "Manila\nBay", fontsize=8, color=shared.WATER_TEXT, style="italic", va="center")
        x1, y0 = shared.MM_BBOX[2] - 0.012, shared.MM_BBOX[1] + 0.012
        km5 = 5 / (111.32 * shared.MM_COSLAT)
        ax.plot([x1 - km5, x1], [y0, y0], color=INK2, lw=2, solid_capstyle="butt", zorder=6)
        ax.text(x1 - km5 / 2, y0 + 0.006, "5 km", ha="center", va="bottom", fontsize=8, color=INK2, zorder=6)
cax = fig.add_axes([(FW - 0.75) / FW, 0.62 / FH, 0.11 / FW, MAP_H / FH])
cb = fig.colorbar(mesh, cax=cax)
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=8.5, length=2)
cb.set_label("PM2.5 (µg/m³)", fontsize=9, color=INK2)
fig.text(LEFT / FW, 1 - 0.14 / FH, "Forecast PM2.5 surfaces at 100 m, Metro Manila", fontsize=12,
         weight="semibold", va="top")
fig.text(LEFT / FW, 0.1 / FH, f"Past forecast made {at(origin)} 2026 (test period) with recorded weather. The LSTM's "
         f"forecasts at the {len(stations)} stations (dots),\naveraged over {len(layer['seeds'])} seeds, "
         "regression-kriged to every 100 m cell.", fontsize=8.5, color=MUTED, va="bottom", linespacing=1.4)
fig.savefig(shared.out_path("m4_surfaces.png"), dpi=200)
plt.close(fig)
print(f"colour scale {lo:.1f}-{hi:.1f} µg/m³; saved to {shared.OUT}")
