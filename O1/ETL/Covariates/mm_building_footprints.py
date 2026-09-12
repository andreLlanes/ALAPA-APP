"""
Building footprints for Metro Manila from OpenStreetMap via the Overpass API.

Bounding box (min_lat, min_lon, max_lat, max_lon) = (14.34, 120.93, 14.78, 121.15).

The box is split into a grid of small tiles; each tile is queried, cached to its
own file, and stitched into a single GeoDataFrame. A stopped run resumes by
loading already-cached tiles instead of re-querying them. Tiles are processed one
at a time and each tile's raw JSON is discarded after parsing, so peak memory is
one tile of JSON plus the accumulating polygon list.

Output is a GeoJSON of building polygons in EPSG:4326 with osm_id, name, and
building type.
"""

import json
import os
import time

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Polygon

MM_BBOX = (14.34, 120.93, 14.78, 121.15)

TILE_STEP_DEG = float(os.environ.get("BLDG_TILE_STEP_DEG", "0.05"))

CACHE_DIR = os.environ.get("BLDG_CACHE_DIR", "mm_buildings_cache")
OUTPUT_GEOJSON = os.environ.get("BLDG_OUTPUT", "mm_buildings.geojson")

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

USER_AGENT = os.environ.get("BLDG_USER_AGENT", "MM-AirQuality-Pipeline/1.0")
QUERY_TIMEOUT = int(os.environ.get("BLDG_QUERY_TIMEOUT", "180"))
HTTP_TIMEOUT = int(os.environ.get("BLDG_HTTP_TIMEOUT", "300"))
MAX_RETRIES = int(os.environ.get("BLDG_MAX_RETRIES", "5"))
BACKOFF_BASE = float(os.environ.get("BLDG_BACKOFF_BASE", "4"))
SLEEP_BETWEEN_TILES = float(os.environ.get("BLDG_SLEEP_BETWEEN", "2"))


def frange(start, stop, step):
    """Inclusive-ish float range; last tile is clamped to `stop` by the caller."""
    v = start
    while v < stop - 1e-9:
        yield v
        v += step


def build_tiles(bbox, step):
    """Split (min_lat, min_lon, max_lat, max_lon) into a list of sub-bboxes."""
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
    """Overpass QL for all buildings in one tile. bbox order is (S, W, N, E)."""
    lat0, lon0, lat1, lon1 = tile
    return f"""
[out:json][timeout:{QUERY_TIMEOUT}];
way["building"]({lat0},{lon0},{lat1},{lon1});
out geom;
""".strip()


def fetch_tile(tile):
    """Return the tile's Overpass JSON, from cache if present, else live with retry.

    Cached tiles are read straight off disk. Live queries rotate through the
    endpoint list and back off exponentially on failure / HTTP 429 / 504.
    """
    cache_path = tile_cache_path(tile)
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f), "cache"

    query = overpass_query(tile)
    last_err = None
    for attempt in range(MAX_RETRIES):
        endpoint = OVERPASS_ENDPOINTS[attempt % len(OVERPASS_ENDPOINTS)]
        try:
            r = requests.post(
                endpoint,
                data={"data": query},
                headers={"User-Agent": USER_AGENT},
                timeout=HTTP_TIMEOUT,
            )
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


def polygons_from_json(bldg_json):
    """Stitch each way's vertices into a Polygon record. Skips degenerate ways."""
    records = []
    for el in bldg_json.get("elements", []):
        if el.get("type") == "way" and "geometry" in el:
            coords = [(pt["lon"], pt["lat"]) for pt in el["geometry"]]
            if len(coords) >= 3:
                tags = el.get("tags", {})
                records.append({
                    "osm_id": el.get("id"),
                    "name": tags.get("name", ""),
                    "building": tags.get("building", "yes"),
                    "geometry": Polygon(coords),
                })
    return records


def main():
    tiles = build_tiles(MM_BBOX, TILE_STEP_DEG)
    print(f"Metro Manila bbox {MM_BBOX} split into {len(tiles)} tiles "
          f"of {TILE_STEP_DEG}° each")

    all_records = []
    seen_ids = set()

    for i, tile in enumerate(tiles, 1):
        try:
            bldg_json, source = fetch_tile(tile)
        except RuntimeError as e:
            print(f"[{i}/{len(tiles)}] {tile} — SKIPPED: {e}")
            continue

        recs = polygons_from_json(bldg_json)
        new = [r for r in recs if r["osm_id"] not in seen_ids]
        seen_ids.update(r["osm_id"] for r in new)
        all_records.extend(new)

        print(f"[{i}/{len(tiles)}] {tile} — {len(recs):>6} ways "
              f"({len(new):>6} new) [{source}] · total {len(all_records)}")

        del bldg_json
        if source == "live":
            time.sleep(SLEEP_BETWEEN_TILES)

    gdf = gpd.GeoDataFrame(
        pd.DataFrame(all_records, columns=["osm_id", "name", "building", "geometry"]),
        geometry="geometry",
        crs="EPSG:4326",
    )
    print(f"\nParsed {len(gdf)} building polygons total")

    gdf.to_file(OUTPUT_GEOJSON, driver="GeoJSON")
    print(f"Wrote {OUTPUT_GEOJSON}")
    return gdf


if __name__ == "__main__":
    main()