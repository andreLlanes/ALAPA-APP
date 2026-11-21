""" Shared pieces for the slide-figure scripts: repo paths, access to O3's code, city data and chart
    style.
"""

import gzip
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
O3_DIR = REPO / "O3"
OUT = Path(__file__).resolve().parent / "Outputs"
# The scripts import O3's own code (fill rule, completeness, settings), so every chart shows what
# the models actually do. O3 goes after this folder on the path, so its config/Common/data resolve.
sys.path.insert(1, str(O3_DIR))

# O3's copies of the database come first; Los Angeles may only be copied under O2.
CACHE_DIRS = [O3_DIR / "data" / "cache", REPO / "O2" / "data" / "cache"]
CITY_NAMES = {"mm": "Metro Manila", "bk": "Bangkok", "la": "Los Angeles"}
LOCAL_TZ = "Asia/Manila"

# Chart colours (blue = measured, orange = filled) and ink.
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, PLANE = "#e1e0d9", "#c3c2b7", "#f3f2ee"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BLUE_RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95"]  # few -> many

# Metro Manila base map: the study box (O1/ETL CITY_BBOXES: min lon, min lat, max lon, max lat), the
# land outline from the web app's map grid (frontend/public/data, made by O3/spatial/export.py) and
# place names for orientation, nudged off the stations.
MM_BBOX = (120.93, 14.34, 121.15, 14.78)
MM_COSLAT = float(np.cos(np.radians((MM_BBOX[1] + MM_BBOX[3]) / 2)))  # a degree of longitude is cos(lat) as long
WEB_DATA = REPO / "frontend" / "public" / "data"
LAND, WATER, WATER_TEXT, PLACE_TEXT = "#ecebe4", "#e4edf6", "#6f8bab", "#7d7c77"
MM_PLACES = {"Quezon City": (14.676, 121.044), "Caloocan": (14.651, 120.972), "Marikina": (14.690, 121.125),
             "Manila": (14.607, 120.965), "Makati": (14.555, 121.024), "Pasig": (14.562, 121.103),
             "Parañaque": (14.479, 121.020), "Las Piñas": (14.430, 120.985), "Muntinlupa": (14.408, 121.042)}


def read_city(code: str, columns=None) -> pd.DataFrame:
    """ Read a city's cleaned rows (a local copy of openaq.merged_clean), ordered by station, then time.
    """
    for folder in CACHE_DIRS:
        path = folder / f"{code}.parquet"
        if path.exists():
            return pd.read_parquet(path, columns=columns)
    raise FileNotFoundError(f"No {code}.parquet in {' or '.join(map(str, CACHE_DIRS))}; "
                            "copy the database first (python O3/data/database.py)")


def out_path(name: str) -> Path:
    """ Return the path of an output file, creating figures/Outputs/ if needed.
    """
    OUT.mkdir(exist_ok=True)
    return OUT / name


# The slide maps (kriging, M4) show this model's map; the app holds one per model since export v2.
FIGURE_MODEL = "lstm"


def web_map(model=FIGURE_MODEL):
    """ Read the web app's manifest (frontend/public/data, O3/spatial/export.py); return it and one
        model's map entry (pm25 and sd encodings, seeds, variogram). A version 1 manifest holds a
        single map, returned whatever the model; model None picks a version 2 manifest's default.
    """
    manifest = json.loads((WEB_DATA / "manifest.json").read_text())
    if manifest.get("version", 1) == 1:
        return manifest, {"id": manifest["model"]["id"], "pm25": manifest["pm25"], "sd": manifest["sd"],
                          "seeds": manifest["model"]["seeds"], "variogram": manifest.get("variogram")}
    model = model or manifest["default_model"]
    for entry in manifest["models"]:
        if entry["id"] == model:
            return manifest, entry
    raise KeyError(f"the web app has no {model} map (it has {', '.join(m['id'] for m in manifest['models'])})")


def land_mask():
    """ Return (lon, lat, land) for the web app's 100 m grid (byte 255 = water or no value), or None.
    """
    from pyproj import Transformer
    try:
        manifest, entry = web_map(None)
        g = manifest["grid"]
        with gzip.open(WEB_DATA / entry["pm25"]["file"]) as f:
            first_hour = np.frombuffer(f.read(g["rows"] * g["cols"]), np.uint8)
    except (FileNotFoundError, KeyError):
        print("No web map grid; drawing the map without a land outline")
        return None
    x = g["x0"] + (np.arange(g["cols"]) + 0.5) * g["cell"]
    y = g["y0"] - (np.arange(g["rows"]) + 0.5) * g["cell"]
    lon, lat = Transformer.from_crs(g["epsg"], 4326, always_xy=True).transform(*np.meshgrid(x, y))
    return lon, lat, first_hour.reshape(g["rows"], g["cols"]) != 255


def manila_map_width(height: float) -> float:
    """ Return the width (inches) of a Metro Manila map of this height that keeps its true shape.
    """
    return height * (MM_BBOX[2] - MM_BBOX[0]) * MM_COSLAT / (MM_BBOX[3] - MM_BBOX[1])


def manila_basemap(ax):
    """ Draw Metro Manila's land and water, place names (zorder 3), a 5 km scale bar and lat/lon
        ticks. Draw links below the names (zorder 2) and markers above them (zorder 4+).
    """
    ax.set_facecolor(WATER)
    ax.grid(False)
    mask = land_mask()
    if mask is not None:
        ax.contourf(*mask[:2], mask[2].astype(float), levels=[0.5, 1.5], colors=[LAND])
    else:
        ax.set_facecolor(LAND)
    halo = [pe.withStroke(linewidth=2.5, foreground=LAND)]
    for name, (lat, lon) in MM_PLACES.items():
        ax.text(lon, lat, name, fontsize=8, color=PLACE_TEXT, ha="center", va="center", path_effects=halo, zorder=3)
    ax.text(120.934, 14.585, "Manila\nBay", fontsize=8.5, color=WATER_TEXT, style="italic", va="center")
    ax.text(121.11, 14.40, "Laguna\nde Bay", fontsize=8.5, color=WATER_TEXT, style="italic", ha="center", va="center")
    x1, y0 = MM_BBOX[2] - 0.012, MM_BBOX[1] + 0.012
    km5 = 5 / (111.32 * MM_COSLAT)  # 5 km in degrees of longitude
    ax.plot([x1 - km5, x1], [y0, y0], color=INK2, lw=2, solid_capstyle="butt", zorder=6)
    ax.text(x1 - km5 / 2, y0 + 0.005, "5 km", ha="center", va="bottom", fontsize=9, color=INK2, zorder=6)
    ax.set_xlim(MM_BBOX[0], MM_BBOX[2])
    ax.set_ylim(MM_BBOX[1], MM_BBOX[3])
    ax.set_xticks([121.0, 121.1])
    ax.set_xticklabels(["121.0°E", "121.1°E"], fontsize=9)
    ax.set_yticks([14.4, 14.5, 14.6, 14.7])
    ax.set_yticklabels([f"{v:.1f}°N" for v in (14.4, 14.5, 14.6, 14.7)], fontsize=9)
    for side in ax.spines.values():
        side.set_visible(True)
        side.set_color(AXIS)


def use_style(size: float = 13):
    """ Apply the shared chart style: system sans, hairline grid and axes, no top/right spines.
    """
    plt.rcParams.update({
        "font.family": "Segoe UI", "font.size": size, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 1,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })
