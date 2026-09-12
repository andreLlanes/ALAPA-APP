"""
Road network density for Metro Manila on the study's common 100 m grid
(Section: Road Network Density). Each cell is assigned the total length of road
within it (metres); cells with no roads are zero. Output is a single-band
GeoTIFF aligned cell-for-cell with the building-fraction and GTFS layers.

Roads are fetched from OpenStreetMap via Overpass with the same tiled,
cached, resumable approach as the building fetcher (mm_buildings.py): the MM
bounding box is split into tiles, each queried and cached to its own file, so a
stopped run resumes from cache. All highway types are included except clearly
non-vehicular pedestrian ways (footway, path, steps, cycleway, pedestrian);
set ROAD_INCLUDE_ALL=1 to include literally every highway tag.

Length per cell is computed exactly: road lines are projected to UTM 51N and
intersected with each grid cell's square, and the clipped segment lengths are
summed per cell. Processing is per tile so the whole network never needs to sit
in memory at once.
"""

import json
import os
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from affine import Affine
from rasterio.features import rasterize
from shapely.geometry import LineString, box

CRS = "EPSG:32651"
ORIGIN_E = 276700.0
ORIGIN_N = 1635100.0
CELL = 100.0
NCOLS = 242
NROWS = 490
GRID_TRANSFORM = Affine(CELL, 0, ORIGIN_E, 0, -CELL, ORIGIN_N)

MM_BBOX = (14.34, 120.93, 14.78, 121.15)
TILE_STEP_DEG = float(os.environ.get("ROAD_TILE_STEP_DEG", "0.05"))

CACHE_DIR = os.environ.get("ROAD_CACHE_DIR", "mm_roads_cache")
OUTPUT_TIF = os.environ.get("ROAD_OUTPUT", "road_density.tif")

INCLUDE_ALL = os.environ.get("ROAD_INCLUDE_ALL", "0") == "1"
EXCLUDED_TYPES = {"footway", "path", "steps", "cycleway", "pedestrian",
                  "bridleway", "corridor"}

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
USER_AGENT = os.environ.get("ROAD_USER_AGENT", "MM-AirQuality-Pipeline/1.0")
QUERY_TIMEOUT = int(os.environ.get("ROAD_QUERY_TIMEOUT", "180"))
HTTP_TIMEOUT = int(os.environ.get("ROAD_HTTP_TIMEOUT", "300"))
MAX_RETRIES = int(os.environ.get("ROAD_MAX_RETRIES", "5"))
BACKOFF_BASE = float(os.environ.get("ROAD_BACKOFF_BASE", "4"))
SLEEP_BETWEEN_TILES = float(os.environ.get("ROAD_SLEEP_BETWEEN", "2"))


def frange(start, stop, step):
    v = start
    while v < stop - 1e-9:
        yield v
        v += step


def build_tiles(bbox, step):
    min_lat, min_lon, max_lat, max_lon = bbox
    tiles = []
    for lat0 in frange(min_lat, max_lat, step):
        lat1 = min(lat0 + step, max_lat)
        for lon0 in frange(min_lon, max_lon, step):
            lon1 = min(lon0 + step, max_lon)
            tiles.append((round(lat0, 4), round(lon0, 4),
                          round(lat1, 4), round(lon1, 4)))
    return tiles


def tile_cache_path(tile):
    lat0, lon0, lat1, lon1 = tile
    name = f"tile_{lat0}_{lon0}_{lat1}_{lon1}.json".replace("-", "m")
    return os.path.join(CACHE_DIR, name)


def overpass_query(tile):
    lat0, lon0, lat1, lon1 = tile
    return f"""
[out:json][timeout:{QUERY_TIMEOUT}];
way["highway"]({lat0},{lon0},{lat1},{lon1});
out geom;
""".strip()


def fetch_tile(tile):
    cache_path = tile_cache_path(tile)
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f), "cache"

    query = overpass_query(tile)
    last_err = None
    for attempt in range(MAX_RETRIES):
        endpoint = OVERPASS_ENDPOINTS[attempt % len(OVERPASS_ENDPOINTS)]
        try:
            r = requests.post(endpoint, data={"data": query},
                              headers={"User-Agent": USER_AGENT},
                              timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(data, f)
            return data, "live"
        except Exception as e:
            last_err = e
            wait = BACKOFF_BASE * (2 ** attempt)
            print(f"    attempt {attempt + 1}/{MAX_RETRIES} on "
                  f"{endpoint.split('/')[2]} failed ({e}); retrying in {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError(f"tile {tile} failed after {MAX_RETRIES} attempts: {last_err}")


def lines_from_json(road_json):
    records = []
    for el in road_json.get("elements", []):
        if el.get("type") == "way" and "geometry" in el:
            hwy = el.get("tags", {}).get("highway", "")
            if not INCLUDE_ALL and hwy in EXCLUDED_TYPES:
                continue
            coords = [(pt["lon"], pt["lat"]) for pt in el["geometry"]]
            if len(coords) >= 2:
                records.append({"highway": hwy, "geometry": LineString(coords)})
    return records


def accumulate_lengths(gdf_utm, length_grid):
    """Clip each road line to the cells it crosses and add segment length in m."""
    for geom in gdf_utm.geometry:
        if geom is None or geom.is_empty:
            continue
        minx, miny, maxx, maxy = geom.bounds
        c0 = int((minx - ORIGIN_E) // CELL)
        c1 = int((maxx - ORIGIN_E) // CELL)
        r0 = int((ORIGIN_N - maxy) // CELL)
        r1 = int((ORIGIN_N - miny) // CELL)
        for r in range(max(r0, 0), min(r1, NROWS - 1) + 1):
            for c in range(max(c0, 0), min(c1, NCOLS - 1) + 1):
                cell_minx = ORIGIN_E + c * CELL
                cell_maxy = ORIGIN_N - r * CELL
                cell = box(cell_minx, cell_maxy - CELL, cell_minx + CELL, cell_maxy)
                seg = geom.intersection(cell)
                if not seg.is_empty:
                    length_grid[r, c] += seg.length


def main():
    tiles = build_tiles(MM_BBOX, TILE_STEP_DEG)
    mode = "ALL highway types" if INCLUDE_ALL else "vehicular roads (peds excluded)"
    print(f"MM bbox split into {len(tiles)} tiles; including {mode}")

    length_grid = np.zeros((NROWS, NCOLS), dtype="float64")
    total_lines = 0

    for i, tile in enumerate(tiles, 1):
        try:
            road_json, source = fetch_tile(tile)
        except RuntimeError as e:
            print(f"[{i}/{len(tiles)}] {tile} — SKIPPED: {e}")
            continue

        recs = lines_from_json(road_json)
        if recs:
            gdf = gpd.GeoDataFrame(recs, geometry="geometry", crs="EPSG:4326")
            gdf = gdf.to_crs(CRS)
            accumulate_lengths(gdf, length_grid)
            total_lines += len(gdf)
            del gdf
        print(f"[{i}/{len(tiles)}] {tile} — {len(recs):>6} roads [{source}] "
              f"· total lines {total_lines}")

        del road_json
        if source == "live":
            time.sleep(SLEEP_BETWEEN_TILES)

    grid = length_grid.astype("float32")
    print(f"\nRoad length: {(grid > 0).sum()} nonzero cells, "
          f"total {grid.sum() / 1000:.1f} km, max cell {grid.max():.0f} m")

    with rasterio.open(
        OUTPUT_TIF, "w", driver="GTiff",
        height=NROWS, width=NCOLS, count=1, dtype="float32",
        crs=CRS, transform=GRID_TRANSFORM, compress="deflate",
    ) as dst:
        dst.write(grid, 1)
        dst.set_band_description(1, "road_length_m")
    print(f"Wrote {OUTPUT_TIF}")
    return grid


if __name__ == "__main__":
    main()