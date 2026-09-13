"""
GTFS transit-intensity emission surfaces for Metro Manila, on the study's common
100 m grid (Section: GTFS Transit Intensity Surface).

Produces four per-band relative-emission rasters (AM peak, midday, PM peak, night)
aligned cell-for-cell with the building-fraction and other covariate layers, so the
whole stack shares one grid for regression kriging.

Method:
  - Keep only road public transport: jeepney (PUJ) and bus (PUB) routes. Rail
    (LRT/MRT/PNR, route_type 2) is excluded.
  - Four service bands: AM 06-09, midday 09-16, PM 16-19, night 19-06.
  - Per band, each route's dispatch frequency f_b = 3600 / headway_secs (veh/hr),
    averaged over the band weighted by temporal overlap with the band window.
  - Fixed relative emission weights: E_jeep = 1.0, E_bus = 3.64.
  - Per cell i and band b: E_hat = E_bus * F_bus + E_jeep * F_jeep, where F is the
    frequency-weighted total of overlapping routes of each type in that cell.
  - Near-road dispersion: each band surface is convolved with a radial exponential
    decay kernel exp(-r / lambda), lambda = 1/0.0026 m ~= 385 m (the black-carbon
    near-road decay rate from a published meta-analysis, used as an exhaust proxy
    because PM2.5 has no sharp published gradient). The kernel is truncated at 3
    lambda and mass-conserving, so emission spreads off the road with exponential
    falloff rather than sitting only on the centreline. Override the rate via
    GTFS_DECAY_RATE. Set GTFS_DECAY_RATE very high to effectively disable.

Route geometry is built from stop_times sequence order as the union of all
distinct trip paths per route; emission is assigned once per covered cell.
Output is a 4-band GeoTIFF in EPSG:32651 on the common grid.
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.features import rasterize
from shapely.geometry import LineString

CRS = "EPSG:32651"
ORIGIN_E = 276700.0
ORIGIN_N = 1635100.0
CELL = 100.0
NCOLS = 242
NROWS = 490
GRID_TRANSFORM = Affine(CELL, 0, ORIGIN_E, 0, -CELL, ORIGIN_N)

GTFS_DIR = os.environ.get("GTFS_DIR", "gtfs-dotc")
OUTPUT_TIF = os.environ.get("GTFS_OUTPUT", "gtfs_emissions.tif")

E_JEEP = 1.0
E_BUS = 3.64

DECAY_RATE = float(os.environ.get("GTFS_DECAY_RATE", "0.0026"))
DECAY_LAMBDA = 1.0 / DECAY_RATE
KERNEL_TRUNC = float(os.environ.get("GTFS_KERNEL_TRUNC", "3.0"))

BANDS = {
    "am_peak": (6 * 3600, 9 * 3600),
    "midday":  (9 * 3600, 16 * 3600),
    "pm_peak": (16 * 3600, 19 * 3600),
    "night":   (19 * 3600, 30 * 3600),
}
NIGHT_EXTRA = (0, 6 * 3600)


def gtfs_secs(t):
    h, m, s = map(int, str(t).split(":"))
    return h * 3600 + m * 60 + s


def overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def classify(route_id):
    """Jeepney (PUJ) or bus (PUB) from the route_id; None for rail/other."""
    rid = str(route_id).upper()
    if "PUJ" in rid:
        return "jeep"
    if "PUB" in rid:
        return "bus"
    return None


def load_routes():
    routes = pd.read_csv(os.path.join(GTFS_DIR, "routes.txt"))
    routes["veh_class"] = routes["route_id"].map(classify)
    kept = routes[routes["veh_class"].notna()].copy()
    print(f"Routes: {len(kept)} kept "
          f"({(kept.veh_class == 'jeep').sum()} jeep, "
          f"{(kept.veh_class == 'bus').sum()} bus); "
          f"{len(routes) - len(kept)} rail/other dropped")
    return kept


def build_route_band(routes_kept):
    """Per-(route, band) frequency-weighted emission."""
    trips = pd.read_csv(os.path.join(GTFS_DIR, "trips.txt"))
    frequencies = pd.read_csv(os.path.join(GTFS_DIR, "frequencies.txt"))

    trips_kept = trips.merge(routes_kept[["route_id", "veh_class"]], on="route_id")
    freq = frequencies.merge(
        trips_kept[["trip_id", "route_id", "veh_class"]], on="trip_id")
    freq["start_s"] = freq["start_time"].map(gtfs_secs)
    freq["end_s"] = freq["end_time"].map(gtfs_secs)

    for band, (b0, b1) in BANDS.items():
        ov = freq.apply(lambda r: overlap(r.start_s, r.end_s, b0, b1), axis=1)
        if band == "night":
            ov = ov + freq.apply(
                lambda r: overlap(r.start_s, r.end_s, *NIGHT_EXTRA), axis=1)
        freq[f"ov_{band}"] = ov

    records = []
    for band in BANDS:
        sub = freq[freq[f"ov_{band}"] > 0].copy()
        if sub.empty:
            continue
        sub["w"] = sub[f"ov_{band}"]
        sub["f"] = 3600.0 / sub["headway_secs"]
        g = (sub.groupby(["route_id", "veh_class"])
                .apply(lambda d: np.average(d["f"], weights=d["w"]),
                       include_groups=False)
                .rename("veh_per_hr").reset_index())
        g["band"] = band
        records.append(g)

    route_band = pd.concat(records, ignore_index=True)
    route_band["E"] = np.where(route_band["veh_class"] == "bus", E_BUS, E_JEEP)
    route_band["emis"] = route_band["E"] * route_band["veh_per_hr"]
    return route_band


def build_route_lines(routes_kept):
    """Route geometry as the union of all distinct trip paths per route.

    Paths are built from stop_times sequence order. Every distinct stop-sequence
    a route runs is turned into a line, so together they cover every street
    segment the route uses. Emission is assigned once per covered cell, so
    unioning widens coverage without inflating magnitude. Returns one row per
    (route_id, distinct-path) line.
    """
    trips = pd.read_csv(os.path.join(GTFS_DIR, "trips.txt"))
    stops = pd.read_csv(os.path.join(GTFS_DIR, "stops.txt"))
    stop_times = pd.read_csv(os.path.join(GTFS_DIR, "stop_times.txt"))

    trips_kept = trips.merge(routes_kept[["route_id", "veh_class"]], on="route_id")
    st = stop_times.merge(trips_kept[["trip_id", "route_id"]], on="trip_id")
    st = st.merge(stops[["stop_id", "stop_lat", "stop_lon"]], on="stop_id")
    st = st.sort_values(["trip_id", "stop_sequence"])

    sig = st.groupby("trip_id")["stop_id"].apply(tuple).rename("sig")
    trip_route = trips_kept.set_index("trip_id")["route_id"]
    trip_meta = pd.DataFrame({"route_id": trip_route, "sig": sig}).dropna()
    rep_trips = (trip_meta.reset_index()
                          .drop_duplicates(["route_id", "sig"])["trip_id"])

    st_rep = st[st["trip_id"].isin(set(rep_trips))]

    rows = []
    for (route_id, trip_id), d in st_rep.groupby(["route_id", "trip_id"]):
        if len(d) > 1:
            rows.append({"route_id": route_id,
                         "geometry": LineString(zip(d["stop_lon"], d["stop_lat"]))})
    route_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    n_routes = route_gdf["route_id"].nunique()
    print(f"Built {len(route_gdf)} path-lines across {n_routes} routes")
    return route_gdf.to_crs(CRS)


def cell_index_for_lines(route_gdf):
    """Map each route to the deduplicated set of grid cells its paths intersect.

    All of a route's path-lines are burned together and the touched cells
    deduplicated, so a cell covered by two paths of the same route is counted
    once. Emission is assigned per route per cell, not per path, which prevents
    double-counting.
    """
    route_to_cells = {}
    for route_id, grp in route_gdf.groupby("route_id"):
        geoms = [(g, 1) for g in grp.geometry if g is not None and not g.is_empty]
        if not geoms:
            continue
        mask = rasterize(
            geoms, out_shape=(NROWS, NCOLS),
            transform=GRID_TRANSFORM, fill=0, all_touched=True, dtype="uint8")
        route_to_cells[route_id] = np.argwhere(mask == 1)
    return route_to_cells


def build_surfaces(route_band, route_to_cells):
    """Accumulate E_hat per cell per band onto the common grid."""
    surfaces = {b: np.zeros((NROWS, NCOLS), dtype="float32") for b in BANDS}
    for band in BANDS:
        rb = route_band[route_band["band"] == band].set_index("route_id")["emis"]
        surf = surfaces[band]
        for route_id, emis in rb.items():
            cells = route_to_cells.get(route_id)
            if cells is None:
                continue
            surf[cells[:, 0], cells[:, 1]] += emis
        print(f"  {band}: {(surf > 0).sum()} nonzero cells, "
              f"max {surf.max():.1f}")
    return surfaces


def decay_kernel():
    radius_cells = int(np.ceil(KERNEL_TRUNC * DECAY_LAMBDA / CELL))
    offs = np.arange(-radius_cells, radius_cells + 1)
    dy, dx = np.meshgrid(offs, offs, indexing="ij")
    dist_m = np.hypot(dy, dx) * CELL
    k = np.exp(-dist_m / DECAY_LAMBDA).astype("float64")
    k[dist_m > KERNEL_TRUNC * DECAY_LAMBDA] = 0.0
    k /= k.sum()
    return k.astype("float32")


def apply_decay(surfaces):
    from scipy.signal import fftconvolve
    k = decay_kernel()
    print(f"  decay kernel: lambda={DECAY_LAMBDA:.0f} m, "
          f"radius={(k.shape[0] - 1) // 2} cells, sum={k.sum():.4f}")
    out = {}
    for band, surf in surfaces.items():
        sm = fftconvolve(surf.astype("float32"), k, mode="same").astype("float32")
        sm[sm < 0] = 0.0
        out[band] = sm
        print(f"  {band}: post-decay max {sm.max():.2f}, "
              f"{(sm > 1e-6).sum()} cells > 0")
    return out


def emission_road_correlation(surfaces, road_raster="road_density.tif"):
    if not os.path.exists(road_raster):
        print(f"  ({road_raster} not found in outputs; skipping correlation check)")
        return None
    with rasterio.open(road_raster) as r:
        road = r.read(1).astype("float64")
    band0 = list(surfaces)[0]
    emis = surfaces[band0].astype("float64")
    if road.shape != emis.shape:
        print(f"  (road raster shape {road.shape} != {emis.shape}; skipping)")
        return None
    mask = (road > 0) | (emis > 0)
    if mask.sum() < 2:
        return None
    corr = np.corrcoef(emis[mask], road[mask])[0, 1]
    print(f"  Pearson r (emission {band0} vs road density, "
          f"non-zero cells) = {corr:.3f}")
    return corr


def write_stack(surfaces, path):
    band_order = list(BANDS)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=NROWS, width=NCOLS, count=len(band_order),
        dtype="float32", crs=CRS, transform=GRID_TRANSFORM,
        compress="deflate",
    ) as dst:
        for i, band in enumerate(band_order, 1):
            dst.write(surfaces[band], i)
            dst.set_band_description(i, f"emis_{band}")
    print(f"Wrote {path} ({len(band_order)} bands: {', '.join(band_order)})")


def main():
    routes_kept = load_routes()
    route_band = build_route_band(routes_kept)
    route_gdf = build_route_lines(routes_kept)
    route_to_cells = cell_index_for_lines(route_gdf)
    surfaces = build_surfaces(route_band, route_to_cells)
    surfaces = apply_decay(surfaces)
    write_stack(surfaces, OUTPUT_TIF)
    emission_road_correlation(surfaces)
    return surfaces


if __name__ == "__main__":
    main()