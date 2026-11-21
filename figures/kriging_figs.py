""" Figures for the regression-kriging slide (Chapter 4), from the web app's map data
    (frontend/public/data, made by O3/spatial/export.py from `train.py --map true` runs), the
    LSTM's map (shared.FIGURE_MODEL):
      kriging_map.png        the kriged PM2.5 forecast on the 100 m grid and its kriging uncertainty,
                             for one forecast hour, with the stations whose forecasts were kriged
      kriging_variogram.png  the variogram fitted for that run, with the distances between those
                             stations
    The map is a past forecast from the test period (a backtest) that used recorded weather.
    python figures/kriging_figs.py [--lead 1]
"""

import argparse
import gzip
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pyproj import Transformer

import shared
from shared import INK, INK2, MUTED, GRID
from data.batch import station_distances_km
from spatial.variogram import Variogram

parser = argparse.ArgumentParser()
parser.add_argument("--lead", type=int, default=1, help="forecast hour shown, 1-72")
args = parser.parse_args()

shared.use_style(11)
manifest, layer = shared.web_map()
g = manifest["grid"]
shape = (manifest["horizon"], g["rows"], g["cols"])


def bands(name):
    """ Read one of the app's byte grids: [72, rows, cols] uint8.
    """
    with gzip.open(shared.WEB_DATA / name) as f:
        return np.frombuffer(f.read(), np.uint8).reshape(shape)


h = args.lead - 1
pm_code, sd_code = bands(layer["pm25"]["file"])[h], bands(layer["sd"]["file"])[h]
nodata = layer["pm25"]["nodata"]
pm = np.where(pm_code == nodata, np.nan, layer["pm25"]["max"] * (pm_code / 254.0) ** 2)
sd = np.where(sd_code == layer["sd"].get("nodata", 255), np.nan, sd_code / 254.0 * layer["sd"]["scale"][h])
x = g["x0"] + np.arange(g["cols"] + 1) * g["cell"]  # cell edges, for pcolormesh
y = g["y0"] - np.arange(g["rows"] + 1) * g["cell"]
lon, lat = Transformer.from_crs(g["epsg"], 4326, always_xy=True).transform(*np.meshgrid(x, y))
stations = pd.DataFrame(manifest["stations"])
origin = pd.Timestamp(manifest["origin_utc"]).tz_convert(shared.LOCAL_TZ)
shown = origin + pd.Timedelta(hours=args.lead)
at = lambda ts: f"{ts:%I %p}".lstrip("0") + f", {ts.day} {ts:%b %Y}"

# ---------- The kriged forecast and its uncertainty ----------
PM_CMAP = LinearSegmentedColormap.from_list("pm", ["#f3edf7", "#c7b2dd", "#8f6bbd", "#5e3c99", "#36195e"])
SD_CMAP = LinearSegmentedColormap.from_list("sd", ["#f4f3ef", "#b9b8b1", "#6d6c66", "#2b2a27"])
MAP_H = 4.5
map_w = shared.manila_map_width(MAP_H)
FW, FH = 0.6 + 2 * map_w + 0.95, 6.5
fig = plt.figure(figsize=(FW, FH))
panels = [("Forecast PM2.5 (µg/m³)", pm, PM_CMAP), ("Kriging uncertainty (SD, µg/m³)", sd, SD_CMAP)]
for i, (label, values, cmap) in enumerate(panels):
    left = 0.6 + i * (map_w + 0.35)
    ax = fig.add_axes([left / FW, 1.25 / FH, map_w / FW, MAP_H / FH])
    shared.manila_basemap(ax)
    lo, hi = np.nanpercentile(values, [1, 99])
    mesh = ax.pcolormesh(lon, lat, values, cmap=cmap, vmin=lo, vmax=hi, shading="flat", zorder=2,
                         rasterized=True)
    ax.plot(stations.lon, stations.lat, "o", color="white", ms=4.2, mec=INK, mew=0.9, zorder=5)
    if i:
        ax.set_yticklabels([])
    ax.set_title(label, fontsize=10, loc="left", pad=6, color=INK)
    cax = fig.add_axes([left / FW, 0.72 / FH, map_w / FW, 0.11 / FH])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8.5, length=2)
    cb.locator = plt.MaxNLocator(4)
    cb.update_ticks()
fig.text(0.6 / FW, 1 - 0.14 / FH, "Regression-kriging output, Metro Manila", fontsize=11.5, weight="semibold", va="top")
fig.text(0.6 / FW, 0.12 / FH, f"Past forecast made {at(origin)}, shown for {at(shown)} (+{args.lead} h); "
         f"recorded weather.\nDots: the {len(stations)} stations whose forecasts were kriged.",
         fontsize=8.5, color=MUTED, va="bottom")
fig.savefig(shared.out_path("kriging_map.png"), dpi=200)
plt.close(fig)

# ---------- The fitted variogram ----------
v = layer["variogram"]
model = Variogram(v["model"], v["nugget"], v["psill"], v["range_km"])
d = station_distances_km(stations.lat.to_numpy(), stations.lon.to_numpy())
pairs = d[np.triu_indices(len(d), 1)]
practical = {"exponential": 3.0, "gaussian": np.sqrt(3.0), "spherical": 1.0}[v["model"]] * v["range_km"]
far = max(pairs.max(), practical) * 1.05
lag = np.linspace(0, far, 400)
sill = v["nugget"] + v["psill"]

fig, (ax, rx) = plt.subplots(2, 1, figsize=(6.4, 3.6), sharex=True, gridspec_kw=dict(height_ratios=[3, 0.55]))
fig.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.15, hspace=0.08)
ax.plot(lag[1:], model(lag[1:]), color=INK, lw=2)
ax.plot([0], [0], "o", color=INK, ms=4)
ax.axhline(sill, color=GRID, lw=1, ls="--", zorder=0)
ax.axvline(practical, color=GRID, lw=1, ls="--", zorder=0)
ax.text(far, sill * 1.02, f"sill {sill:.2f}", ha="right", va="bottom", fontsize=8.5, color=INK2)
ax.text(practical + far * 0.01, sill * 0.35, f"practical range\n{practical:.1f} km", fontsize=8.5, color=INK2)
ax.annotate(f"nugget {v['nugget']:.2f}", (0, v["nugget"]), (far * 0.06, sill * 0.12), fontsize=8.5, color=INK2,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.set_ylim(0, sill * 1.18)
ax.set_ylabel("Semivariance\n(standardized residuals)", fontsize=9)
ax.tick_params(labelsize=8.5)
rx.grid(False)
rx.set_yticks([])
rx.spines["left"].set_visible(False)
rx.plot(pairs, np.zeros_like(pairs), "|", color=INK2, ms=9, alpha=0.35)
rx.text(0, 0.5, f"distances between the {len(stations)} stations", fontsize=8.5, color=INK2, va="bottom",
        transform=rx.get_yaxis_transform())
rx.set_ylim(-0.5, 1.2)
rx.set_xlim(0, far)
rx.set_xlabel("Distance between two stations (km)", fontsize=9)
rx.tick_params(labelsize=8.5)
fig.text(0.11, 0.965, f"Fitted residual variogram ({v['model']})", fontsize=11.5, weight="semibold", va="top")
fig.savefig(shared.out_path("kriging_variogram.png"), dpi=200)
plt.close(fig)
print(f"lead +{args.lead} h ({at(shown)}): PM2.5 {np.nanmin(pm):.1f}-{np.nanmax(pm):.1f}, SD {np.nanmin(sd):.2f}-"
      f"{np.nanmax(sd):.2f}; {len(stations)} stations; practical range {practical:.1f} km; saved to {shared.OUT}")
