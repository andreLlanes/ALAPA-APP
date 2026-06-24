import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from pykrige.ok import OrdinaryKriging
from rasterio.crs import CRS
from rasterio.transform import from_origin
import rasterio
from shapely.geometry import box
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


SENSOR_COORDS = {
    "cubao-1HR-clean": (14.6173, 121.0593),
    "lawton-1HR-clean": (14.5946, 120.9794),
    "makati-1HR-clean": (14.5547, 121.0244),
    "pasay-1HR-clean": (14.5375, 121.0014),
    "SDG-1HR-clean": (14.3537, 120.5921),
}

DATA_DIR   = Path("data")
OUTPUT_DIR = Path("output/kriging")

GRID_RESOLUTION_M = 100
VARIOGRAM_MODEL = "spherical"
NLAGS = 4
REQUIRE_MIN_2_SENSORS = False

BBOX_WGS84 = {
    "min_lon": 120.855, "max_lon": 121.205,
    "min_lat":  14.390, "max_lat":  14.810,
}

CRS_UTM   = CRS.from_epsg(32651)
CRS_WGS84 = CRS.from_epsg(4326)

NODATA = -9999.0

BANDS = ["pm1", "pm25", "pm10", "temperature", "humidity"]
BAND_NAMES = {
    1:  "PM1 (ug/m3)",
    2:  "PM2.5 (ug/m3)",
    3:  "PM10 (ug/m3)",
    4:  "Temperature (degC)",
    5:  "Humidity (%)",
    6:  "Variance - PM1",
    7:  "Variance - PM2.5",
    8:  "Variance - PM10",
    9:  "Variance - Temperature",
    10: "Variance - Humidity",
}
def build_grid():
    bbox_gdf = gpd.GeoDataFrame(
        geometry=[box(BBOX_WGS84["min_lon"], BBOX_WGS84["min_lat"],
                      BBOX_WGS84["max_lon"], BBOX_WGS84["max_lat"])],
        crs=CRS_WGS84,
    ).to_crs(CRS_UTM)

    minx, miny, maxx, maxy = bbox_gdf.total_bounds
    minx = np.floor(minx / GRID_RESOLUTION_M) * GRID_RESOLUTION_M
    miny = np.floor(miny / GRID_RESOLUTION_M) * GRID_RESOLUTION_M
    maxx = np.ceil(maxx  / GRID_RESOLUTION_M) * GRID_RESOLUTION_M
    maxy = np.ceil(maxy  / GRID_RESOLUTION_M) * GRID_RESOLUTION_M

    cols = int(round((maxx - minx) / GRID_RESOLUTION_M))
    rows = int(round((maxy - miny) / GRID_RESOLUTION_M))

    transform = from_origin(minx, maxy, GRID_RESOLUTION_M, GRID_RESOLUTION_M)

    grid_xs = minx + (np.arange(cols) + 0.5) * GRID_RESOLUTION_M
    grid_ys = maxy - (np.arange(rows) + 0.5) * GRID_RESOLUTION_M  # N→S

    meta = {
        "driver":    "GTiff",
        "dtype":     "float32",
        "width":     cols,
        "height":    rows,
        "count":     len(BANDS) * 2, 
        "crs":       CRS_UTM,
        "transform": transform,
        "nodata":    NODATA,
        "compress":  "deflate",
        "tiled":     True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    print(f"  Grid: {rows} rows × {cols} cols = {rows * cols:,} cells  "
          f"({GRID_RESOLUTION_M} m pixels, EPSG:32651)")
    return grid_xs, grid_ys, meta, rows, cols

def _normalise_columns(df):
    col_map = {}
    for col in df.columns:
        clean = re.sub(r"\(.*?\)", "", col)
        clean = re.sub(r"[^a-zA-Z0-9]", "_", clean)
        clean = re.sub(r"_+", "_", clean).strip("_").lower()
        col_map[col] = clean
    df = df.rename(columns=col_map)
    rename = {}
    for col in df.columns:
        lc = col.lower()
        if "timestamp" in lc or lc in ("time", "date", "datetime"):
            rename[col] = "timestamp"
        elif re.search(r"pm\s*2[\._]?5", lc) or "pm25" in lc:
            rename[col] = "pm25"
        elif re.search(r"pm\s*10", lc) or "pm10" in lc:
            rename[col] = "pm10"
        elif re.search(r"\bpm\s*1\b", lc) or lc in ("pm1", "pm_1"):
            rename[col] = "pm1"
        elif "temp" in lc:
            rename[col] = "temperature"
        elif "humid" in lc:
            rename[col] = "humidity"
    return df.rename(columns=rename)


def load_sensor_csv(path, lat, lon):
    df = pd.read_csv(path, sep=None, engine="python", encoding_errors="replace")
    df = _normalise_columns(df)
    if "timestamp" not in df.columns:
        raise ValueError(f"{path.name}: no timestamp column found.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    for var in BANDS:
        if var not in df.columns:
            df[var] = np.nan
    df["hour"] = df["timestamp"].dt.floor("h")
    hourly = df.groupby("hour", as_index=False)[BANDS].mean()
    hourly["lat"] = lat
    hourly["lon"] = lon
    return hourly


def load_all_sensors(data_dir):
    frames = []
    for csv_path in sorted(data_dir.glob("*.csv")):
        stem = csv_path.stem
        if stem not in SENSOR_COORDS:
            print(f"  ⚠  {csv_path.name} not in SENSOR_COORDS — skipping")
            continue
        lat, lon = SENSOR_COORDS[stem]
        df = load_sensor_csv(csv_path, lat, lon)
        df["sensor_id"] = stem
        frames.append(df)
        print(f"  {csv_path.name}: {len(df)} hourly records  "
              f"({df['hour'].min()} → {df['hour'].max()})")
    if not frames:
        raise FileNotFoundError(f"No CSVs matched SENSOR_COORDS keys in {data_dir}.")
    return pd.concat(frames, ignore_index=True)


def project_sensors(df):
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=CRS_WGS84
    ).to_crs(CRS_UTM)
    df = df.copy()
    df["utm_x"] = gdf.geometry.x.values
    df["utm_y"] = gdf.geometry.y.values
    return df

def krige_variable(sx, sy, sv, grid_xs, grid_ys, rows, cols):
    """
    Fit variogram and Krige one variable onto the full grid.
    Returns (z_2d, ss_2d) both shape (rows, cols), float32.
    """
    ok = OrdinaryKriging(
        sx, sy, sv,
        variogram_model=VARIOGRAM_MODEL,
        nlags=NLAGS,
        verbose=False,
        enable_plotting=False,
    )
    z, ss = ok.execute("grid", grid_xs, grid_ys)
    return z.data.astype(np.float32), ss.data.astype(np.float32)

def loso_cv(sx, sy, vals):
    """Leave-One-Station-Out Cross Validation."""
    obs = []
    pred = []
    n = len(vals)

    for i in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False
        try:
            ok = OrdinaryKriging(
                sx[train_mask], sy[train_mask], vals[train_mask],
                variogram_model=VARIOGRAM_MODEL,
                nlags=NLAGS,
                verbose=False,
                enable_plotting=False,
            )
            z, _ = ok.execute("points", np.array([sx[i]]), np.array([sy[i]]))
            obs.append(float(vals[i]))
            pred.append(float(z[0]))
        except Exception:
            continue

    if len(obs) < 2:
        return None

    obs = np.array(obs)
    pred = np.array(pred)

    return {
        "MAE": float(mean_absolute_error(obs, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(obs, pred))),
        "Bias": float(np.mean(pred - obs)),
        "R2": float(r2_score(obs, pred)),
    }

def write_geotiff(z_bands, ss_bands, rows, cols, meta, out_path):
    """
    Write 10-band GeoTIFF: bands 1-5 = values, bands 6-10 = variance.
    Embeds band descriptions and GDAL statistics for QGIS compatibility.
    """
    with rasterio.open(out_path, "w", **meta) as dst:
        for i, var in enumerate(BANDS):
            val_band  = i + 1          # 1–5
            var_band  = i + 6          # 6–10

            z_surf  = np.where(np.isnan(z_bands[var]),  NODATA, z_bands[var])
            ss_surf = np.where(np.isnan(ss_bands[var]), NODATA, ss_bands[var])

            dst.write(z_surf.astype(np.float32),  val_band)
            dst.write(ss_surf.astype(np.float32), var_band)

            dst.set_band_description(val_band, BAND_NAMES[val_band])
            dst.set_band_description(var_band, BAND_NAMES[var_band])

            # Embed statistics so QGIS doesn't need to rescan
            for band_idx, surf in [(val_band, z_bands[var]),
                                   (var_band, ss_bands[var])]:
                valid = surf[~np.isnan(surf)]
                if len(valid) > 0:
                    dst.update_tags(band_idx,
                        STATISTICS_MINIMUM=f"{float(valid.min()):.6f}",
                        STATISTICS_MAXIMUM=f"{float(valid.max()):.6f}",
                        STATISTICS_MEAN=f"{float(valid.mean()):.6f}",
                        STATISTICS_STDDEV=f"{float(valid.std()):.6f}",
                    )

def run_pipeline():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n── Building 100 m grid ──")
    grid_xs, grid_ys, meta, rows, cols = build_grid()

    print("\n── Loading sensor CSVs ──")
    sensor_df = load_all_sensors(DATA_DIR)
    sensor_df = project_sensors(sensor_df)

    hours   = sorted(sensor_df["hour"].unique())
    min_sns = 2 if REQUIRE_MIN_2_SENSORS else 1
    print(f"\n── Processing {len(hours)} hours  "
          f"(min sensors: {min_sns}, variogram: {VARIOGRAM_MODEL}, nlags: {NLAGS}) ──")

    written = 0
    skipped = 0
    all_cv_results = []

    for hour in hours:
        hour_df = sensor_df[sensor_df["hour"] == hour].dropna(subset=BANDS, how="all")
        active  = hour_df.dropna(subset=["pm25"])

        if len(active) < min_sns:
            skipped += 1
            continue

        sx = active["utm_x"].values
        sy = active["utm_y"].values

        ts_str = pd.Timestamp(hour).strftime("%Y-%m-%d_%H")

        cv_rows = []
        for var in BANDS:
            vals = active[var].values.astype(np.float32)
            ok_mask = ~np.isnan(vals)

            if ok_mask.sum() < 3:
                continue

            result = loso_cv(sx[ok_mask], sy[ok_mask], vals[ok_mask])

            if result is not None:
                result["hour"] = ts_str
                result["variable"] = var
                result["n_stations"] = int(ok_mask.sum())
                cv_rows.append(result)

        if cv_rows:
            cv_df = pd.DataFrame(cv_rows)
            cv_df.to_csv(OUTPUT_DIR / f"{ts_str}_cv.csv", index=False)
            all_cv_results.append(cv_df)

        z_bands  = {}
        ss_bands = {}

        for var in BANDS:
            vals    = active[var].values.astype(np.float32)
            ok_mask = ~np.isnan(vals)

            if not ok_mask.any():
                z_bands[var]  = np.full((rows, cols), np.nan, dtype=np.float32)
                ss_bands[var] = np.full((rows, cols), np.nan, dtype=np.float32)
                continue

            # Single unique value — flat surface, zero variance
            if ok_mask.sum() < 2 or np.unique(vals[ok_mask]).size < 2:
                z_bands[var]  = np.full((rows, cols), float(vals[ok_mask][0]),
                                        dtype=np.float32)
                ss_bands[var] = np.zeros((rows, cols), dtype=np.float32)
                continue

            z_bands[var], ss_bands[var] = krige_variable(
                sx[ok_mask], sy[ok_mask], vals[ok_mask],
                grid_xs, grid_ys, rows, cols,
            )

        ts_str   = pd.Timestamp(hour).strftime("%Y-%m-%d_%H")
        out_path = OUTPUT_DIR / f"{ts_str}.tif"
        write_geotiff(z_bands, ss_bands, rows, cols, meta, out_path)
        written += 1

    if all_cv_results:
        pd.concat(all_cv_results, ignore_index=True).to_csv(
            OUTPUT_DIR / "LOSO_CV_Summary.csv",
            index=False
        )

    print(f"\n── Done ──")
    print(f"  GeoTIFFs written : {written}")
    print(f"  Hours skipped    : {skipped}")
    print(f"  Output folder    : {OUTPUT_DIR.resolve()}")
    print(f"\n  Band layout:")
    for b, name in BAND_NAMES.items():
        print(f"    Band {b:>2}: {name}")
    print(f"\n  Pixel size : {GRID_RESOLUTION_M} m × {GRID_RESOLUTION_M} m  (EPSG:32651)")
    print(f"  Nodata     : {NODATA}")
    print(f"\n  Tip: bands 6–10 are Kriging variance.")
    print(f"  High variance = far from sensors = less reliable estimate.")


if __name__ == "__main__":
    run_pipeline()