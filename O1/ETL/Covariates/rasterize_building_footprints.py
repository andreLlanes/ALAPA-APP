"""
Rasterize OSM building footprints to building-coverage fraction on the study's
common 100 m grid (Section: Building Footprint Rasterization).

Output: a single-band GeoTIFF where each cell holds the fraction of its area
covered by building footprints, in [0, 1]. Cells with no buildings are 0.
All tagged footprints are included regardless of building type.

Common grid (UTM Zone 51N, EPSG:32651):
    upper-left origin : easting 276700 m, northing 1635100 m
    cell size         : 100 m
    dimensions        : 242 cols x 490 rows  (24.2 km x 49.0 km, 118580 cells)

Method — supersampled coverage:
    Each 100 m cell is divided into an N x N subgrid (default 10 x 10 -> 10 m
    subcells) and the footprints are burned as a boolean mask at that fine
    resolution. The mask is then block-averaged back to 100 m: the mean of each
    N x N block is the fraction of that cell covered. Footprints are streamed from
    disk in chunks and burned onto the fine mask one chunk at a time, so the full
    layer never sits in RAM at once.
"""

import os

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from rasterio.features import rasterize

CRS = "EPSG:32651"
ORIGIN_E = 276700.0
ORIGIN_N = 1635100.0
CELL = 100.0
NCOLS = 242
NROWS = 490

SUBSAMPLE = int(os.environ.get("BLDG_SUBSAMPLE", "10"))

CHUNK_SIZE = int(os.environ.get("BLDG_CHUNK_SIZE", "50000"))

INPUT_GEOJSON = os.environ.get("BLDG_INPUT", "mm_buildings.geojson")
OUTPUT_TIF = os.environ.get("BLDG_RASTER_OUT", "building_fraction.tif")

GRID_TRANSFORM = Affine(CELL, 0, ORIGIN_E, 0, -CELL, ORIGIN_N)
FINE_TRANSFORM = Affine(CELL / SUBSAMPLE, 0, ORIGIN_E,
                        0, -CELL / SUBSAMPLE, ORIGIN_N)


def _burn_chunk(gdf, fine):
    """Reproject one chunk to the grid CRS and OR its footprints into `fine`."""
    if gdf.crs is None:
        raise ValueError("input chunk has no CRS")
    gdf = gdf.to_crs(CRS)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    if len(gdf) == 0:
        return 0

    chunk_mask = rasterize(
        ((geom, 1) for geom in gdf.geometry),
        out_shape=fine.shape,
        transform=FINE_TRANSFORM,
        fill=0,
        all_touched=False,
        dtype="uint8",
    )
    np.bitwise_or(fine, chunk_mask, out=fine)
    del chunk_mask
    return len(gdf)


def coverage_fraction_from_file(path, chunk_size=CHUNK_SIZE):
    """Stream footprints from `path` and return the (NROWS, NCOLS) fraction grid.

    The file is read `chunk_size` features at a time; each chunk is reprojected
    and burned onto a single persistent fine mask, so the full layer is never
    materialized in memory at once. Falls back to a whole-file read only if the
    installed geopandas/pyogrio can't slice rows.
    """
    fine_h = NROWS * SUBSAMPLE
    fine_w = NCOLS * SUBSAMPLE
    fine = np.zeros((fine_h, fine_w), dtype="uint8")

    total = 0
    try:
        import pyogrio
        info = pyogrio.read_info(path)
        n = info["features"]
        for start in range(0, n, chunk_size):
            gdf = gpd.read_file(path, rows=slice(start, min(start + chunk_size, n)))
            total += _burn_chunk(gdf, fine)
            del gdf
            print(f"  burned {min(start + chunk_size, n):>8,}/{n:,} features")
    except (ImportError, KeyError, TypeError):
        print("  (row-sliced read unavailable; reading whole file)")
        gdf = gpd.read_file(path)
        total += _burn_chunk(gdf, fine)
        del gdf

    frac = _blocks_to_fraction(fine)
    del fine
    print(f"  burned {total:,} valid polygons total")
    return frac


def _blocks_to_fraction(fine):
    """Block-average the SUBSAMPLE x SUBSAMPLE mask down to coverage in [0,1]."""
    return (fine.reshape(NROWS, SUBSAMPLE, NCOLS, SUBSAMPLE)
                .mean(axis=(1, 3))
                .astype("float32"))


def coverage_fraction(gdf):
    """Coverage fraction for an already-loaded GeoDataFrame."""
    fine_h = NROWS * SUBSAMPLE
    fine_w = NCOLS * SUBSAMPLE
    fine = np.zeros((fine_h, fine_w), dtype="uint8")
    _burn_chunk(gdf, fine)
    frac = _blocks_to_fraction(fine)
    del fine
    return frac


def write_raster(frac, path):
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=NROWS, width=NCOLS,
        count=1, dtype="float32",
        crs=CRS, transform=GRID_TRANSFORM,
        nodata=None, compress="deflate",
    ) as dst:
        dst.write(frac, 1)
        dst.set_band_description(1, "building_area_fraction")


def main():
    print(f"Streaming footprints from {INPUT_GEOJSON} "
          f"(chunk={CHUNK_SIZE:,}, subsample={SUBSAMPLE})")
    frac = coverage_fraction_from_file(INPUT_GEOJSON)
    print(f"Coverage grid {frac.shape}: "
          f"min={frac.min():.3f} max={frac.max():.3f} mean={frac.mean():.4f}, "
          f"{(frac > 0).sum()} non-zero cells")

    write_raster(frac, OUTPUT_TIF)
    print(f"Wrote {OUTPUT_TIF}")
    return frac


if __name__ == "__main__":
    main()