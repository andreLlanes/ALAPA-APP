""" Export forecast maps (spatial/surface.py, train.py --map true) for the app: a manifest and, for
    each model's map, two gzip-compressed byte grids on the common 100 m grid, 72 forecast hours x
    490 x 242 cells, lead 1 first, each hour row by row from the north-west corner:
        pm25_<model>.u8.gz  PM2.5: byte = round(254 sqrt(v / PM_MAX)), so a byte step is about
                    0.7 ug/m3 at 15 ug/m3 and 1.5 at 100; 255 = water or no value
        sd_<model>.u8.gz    kriging standard deviation: byte = round(254 sd / scale), one scale per
                    hour in the manifest; 255 = no value
        readings.json  hourly PM2.5 of every station of the city inside the grid, from the 72h
                    lookback before the origin to the last forecast hour (null where no reading),
                    flagged by whether its forecast made the map. For a past origin these include
                    what was measured during the forecast hours, which the app compares with the
                    map; a live origin has none there yet.
    The maps given must be of the same origin and stations (each model's map of the same forecast),
    so they share the clock, the station list and the readings. The manifest holds those, the
    encodings, and per model its seeds, variogram and leave-one-station-out RMSE (all folds, from
    artifacts/loso_comparison.csv) when that table has the model. The app opens on the default
    model: --default, else the lowest of those RMSEs, else the first map. The app's range is the
    kriging part only: the forecast's own error is not in it yet. The manifest is written last, so
    the app never reads it beside half-written files; byte grids an earlier export left that it no
    longer names are then removed.
    python spatial/export.py --map artifacts/<model>/mm/<settings>/<variant>/map_<origin>.json [...]
                             [--model gbt] [--default gbt] [--copy data/cache/mm.parquet]
                             [--out ../frontend/public/data]
"""

import argparse
import gzip
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as reproject_points
from scipy.stats import norm

# Put the O3 root on the import path, so this file runs as a script or as a module.
sys.path.append(str(Path(__file__).resolve().parents[1]))

import config
from Common.schema import HORIZON_H, KEY_COL, LOOKBACK_H, PM25_COL, TIME_COL
from data.database import copy_path
from spatial.covariates import GRID

MODELS = ("lstm", "gnn", "gbt")
PM_MAX = 500.0      # ug/m3 at byte 254; higher values are stored as PM_MAX
NODATA = 255
OUT = Path(__file__).resolve().parents[2] / "frontend" / "public" / "data"
LOSO_TABLE = Path(__file__).resolve().parents[1] / "artifacts" / "loso_comparison.csv"

def read_bands(path: Path) -> np.ndarray:
    """ Read a map GeoTIFF [72, rows, cols]; stop if it is not 72 hours on the common grid.
    """
    with rasterio.open(path) as src:
        if (src.crs is None or src.crs.to_epsg() != config.GRID_EPSG or src.shape != config.GRID_SHAPE
                or not src.transform.almost_equals(GRID) or src.count != HORIZON_H):
            raise ValueError(f"{path.name} is not {HORIZON_H} hours on the common grid")
        return src.read().astype(np.float64)

def encode_pm(pm: np.ndarray) -> np.ndarray:
    """ PM2.5 (ug/m3) to bytes on the square-root scale; NaN to NODATA.
    """
    code = np.round(254 * np.sqrt(np.clip(pm, 0.0, PM_MAX) / PM_MAX))
    return np.where(np.isfinite(pm), code, NODATA).astype(np.uint8)

def decode_pm(code: np.ndarray) -> np.ndarray:
    """ Bytes back to PM2.5 (ug/m3), as the app decodes them.
    """
    return np.where(code == NODATA, np.nan, PM_MAX * (code / 254.0) ** 2)

def encode_sd(sd: np.ndarray):
    """ Standard deviations [72, rows, cols] to bytes, each hour on its own linear scale (its
        largest value); returns (bytes, scales [72]).
    """
    scale = np.nanmax(sd.reshape(len(sd), -1), axis=1)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    code = np.clip(np.round(254 * sd / scale[:, None, None]), 0, 254)
    return np.where(np.isfinite(sd), code, NODATA).astype(np.uint8), scale

def model_of(path: Path, given):
    """ The model and variant of a map: the model given, or the one in its folder
        (artifacts/<model>/<city>/<settings>/<variant>/), else None.
    """
    found = next((p for p in path.resolve().parts if p in MODELS), None)
    return given or found, (path.parent.name if found else None)

def source_of(path: Path) -> str:
    """ A map record's path below artifacts/, which names its model, city, settings and variant.
    """
    parts = path.resolve().parts
    return "/".join(parts[parts.index("artifacts") + 1:]) if "artifacts" in parts else path.name

def loso_score(path: Path, model: str, variant):
    """ The map's run in the leave-one-station-out table (evals/loso.py, all folds), matched on the
        filters, decoder, source and variant of its folder (artifacts/<model>/<city>/<settings>/
        <variant>/): {rmse, folds, seeds}, or None without the table or a single matching row.
    """
    parts = path.resolve().parts
    if not LOSO_TABLE.exists() or model not in parts or len(parts) < parts.index(model) + 4:
        return None
    city, settings = parts[parts.index(model) + 1], parts[parts.index(model) + 2]
    filters = re.search(r"gap\d+_comp\d+", settings)
    if filters is None:
        return None
    table = pd.read_csv(LOSO_TABLE, keep_default_na=False)
    rows = table[(table["band"] == "all") & (table["method"] == model.upper())
                 & (table["filters"] == filters.group(0)) & (table["decoder"] == settings.split("_")[0])
                 & (table["source"] == (city.split("-")[0] if "-" in city else ""))
                 & (table["variant"] == (variant or "none"))]
    if len(rows) != 1:
        return None
    r = rows.iloc[0]
    return {"rmse": round(float(r["rmse"]), 4), "folds": int(float(r["folds"])), "seeds": int(float(r["seeds"]))}

def is_reference(v) -> bool:
    return str(v).strip().lower() in ("t", "true", "1")

def read_copy(copy: Path) -> pd.DataFrame:
    return pd.read_parquet(copy, columns=[KEY_COL, TIME_COL, PM25_COL, "latitude", "longitude", "is_monitor"])

def stations(keys, rows: pd.DataFrame, origin: pd.Timestamp, name: str):
    """ Coordinates and grade of the map's stations, and the time of their latest reading at or
        before the origin.
    """
    rows = rows[rows[KEY_COL].isin(keys)]
    info = rows.groupby(KEY_COL)[["latitude", "longitude", "is_monitor"]].first()
    missing = sorted(set(keys) - set(info.index))
    if missing:
        raise ValueError(f"{len(missing)} map station(s) not in {name}: {', '.join(missing[:5])}")
    out = [{"id": k, "lat": round(float(info.at[k, "latitude"]), 6),
            "lon": round(float(info.at[k, "longitude"]), 6), "reference": is_reference(info.at[k, "is_monitor"])}
           for k in keys]
    seen = rows.loc[rows[PM25_COL].notna() & (rows[TIME_COL] <= origin), TIME_COL]
    return out, (seen.max() if len(seen) else origin)

def readings(rows: pd.DataFrame, origin: pd.Timestamp, used) -> dict:
    """ Hourly PM2.5 of every station inside the grid with a reading from the lookback before the
        origin to its last forecast hour: hour k is origin - (LOOKBACK_H - 1) + k, so hour
        LOOKBACK_H is the first forecast hour. Stations whose forecast made the map come first.
    """
    start = origin - pd.Timedelta(hours=LOOKBACK_H - 1)
    hours = LOOKBACK_H + HORIZON_H
    rows = rows[(rows[TIME_COL] >= start) & (rows[TIME_COL] < start + pd.Timedelta(hours=hours))]
    seen = rows[rows[PM25_COL].notna()]
    info = rows[rows[KEY_COL].isin(seen[KEY_COL].unique())].groupby(KEY_COL)[
        ["latitude", "longitude", "is_monitor"]].first()
    x, y = reproject_points("EPSG:4326", f"EPSG:{config.GRID_EPSG}", info["longitude"].tolist(),
                            info["latitude"].tolist())
    col = np.floor((np.array(x) - config.GRID_ORIGIN[0]) / config.GRID_CELL)
    row = np.floor((config.GRID_ORIGIN[1] - np.array(y)) / config.GRID_CELL)
    inside = (col >= 0) & (col < config.GRID_SHAPE[1]) & (row >= 0) & (row < config.GRID_SHAPE[0])
    info = info[inside]

    used = set(used)
    hour = ((seen[TIME_COL] - start) / pd.Timedelta(hours=1)).round().astype(int)
    grid = {k: [None] * hours for k in info.index}
    for k, h, v in zip(seen[KEY_COL], hour, seen[PM25_COL]):
        if k in grid:
            grid[k][h] = round(float(v), 1)
    order = sorted(info.index, key=lambda k: (k not in used, k))
    return {"start_utc": utc(start), "hours": hours, "lookback": LOOKBACK_H,
            "stations": [{"id": k, "lat": round(float(info.at[k, "latitude"]), 6),
                          "lon": round(float(info.at[k, "longitude"]), 6),
                          "reference": is_reference(info.at[k, "is_monitor"]), "in_map": k in used,
                          "pm25": grid[k]} for k in order]}

def write_gzip(path: Path, data: np.ndarray) -> int:
    with gzip.open(path, "wb", compresslevel=9) as f:
        f.write(np.ascontiguousarray(data).tobytes())
    return path.stat().st_size

def utc(t: pd.Timestamp) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")

def main():
    ap = argparse.ArgumentParser(description="Export forecast maps for the app.")
    ap.add_argument("--map", type=Path, nargs="+", required=True,
                    help="each model's map record, map_<origin>.json, all of the same origin")
    ap.add_argument("--model", choices=MODELS, help="with one map, its model (default: read from its folder)")
    ap.add_argument("--default", choices=MODELS,
                    help="the model the app opens on (default: the lowest leave-one-station-out RMSE, "
                         "else the first map)")
    ap.add_argument("--copy", type=Path, default=copy_path(config.LOSO_CITY),
                    help="the city's database copy, for station coordinates")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    if args.model and len(args.map) > 1:
        ap.error("--model names a single map's model; with several, each is read from its folder")
    if not args.copy.exists():
        raise SystemExit(f"{args.copy} not found: pass --copy with the city's database copy (.parquet)")

    maps = []
    for path in args.map:
        model, variant = model_of(path, args.model)
        if model is None:
            raise SystemExit(f"{path}: its folder names no model; pass --model")
        maps.append({"path": path, "record": json.loads(path.read_text(encoding="utf-8")), "model": model,
                     "variant": variant, "loso": loso_score(path, model, variant)})
    first = maps[0]["record"]
    ids = [m["model"] for m in maps]
    if len(set(ids)) < len(ids):
        raise SystemExit(f"two maps of one model ({', '.join(ids)}): give one map per model")
    for m in maps[1:]:
        if m["record"]["origin_utc"] != first["origin_utc"] or m["record"]["stations"] != first["stations"]:
            raise SystemExit(f"{m['path']}: not the same origin and stations as {maps[0]['path']}")
    if args.default and args.default not in ids:
        ap.error(f"--default {args.default} is not among the maps given ({', '.join(ids)})")
    if args.default:
        default, default_by = args.default, "choice"
    elif len(maps) > 1 and all(m["loso"] for m in maps):
        default, default_by = min(maps, key=lambda m: m["loso"]["rmse"])["model"], "loso_rmse"
    else:
        default, default_by = ids[0], "order"

    origin = pd.Timestamp(first["origin_utc"], tz="UTC")
    rows = read_copy(args.copy)
    station_list, as_of = stations(first["stations"], rows, origin, args.copy.name)
    observed = readings(rows, origin, first["stations"])

    # One model at a time, so only one pair of 72-hour grids is in memory.
    args.out.mkdir(parents=True, exist_ok=True)
    sizes, entries, water, lines = {}, [], None, []
    for m in maps:
        stem = m["path"].with_suffix("")
        pm = read_bands(Path(f"{stem}_pm25.tif"))
        sd = np.sqrt(np.clip(read_bands(Path(f"{stem}_variance.tif")), 0.0, None))
        if not np.array_equal(np.isnan(pm), np.isnan(sd)):
            raise ValueError(f"{m['path'].name} ({m['model']}): the PM2.5 and variance maps blank different cells")
        if water is None:
            water = np.isnan(pm)
        elif not np.array_equal(np.isnan(pm), water):
            raise SystemExit(f"{m['path']}: blanks different cells from {maps[0]['path']}")
        pm_code = encode_pm(pm)
        sd_code, sd_scale = encode_sd(sd)
        pm_file, sd_file = f"pm25_{m['model']}.u8.gz", f"sd_{m['model']}.u8.gz"
        sizes[pm_file] = write_gzip(args.out / pm_file, pm_code)
        sizes[sd_file] = write_gzip(args.out / sd_file, sd_code)
        entries.append({
            "id": m["model"], "variant": m["variant"], "seeds": m["record"].get("seeds_averaged", []),
            "source": source_of(m["path"]),
            "pm25": {"file": pm_file, "encoding": "sqrt", "max": PM_MAX, "nodata": NODATA},
            "sd": {"file": sd_file, "encoding": "linear", "scale": [round(float(s), 4) for s in sd_scale],
                   "nodata": NODATA, "includes": "kriging"},
            "variogram": m["record"].get("variogram"),
            "loso": m["loso"],
        })
        land = np.isfinite(pm)
        error = np.abs(decode_pm(pm_code)[land] - np.clip(pm[land], 0.0, PM_MAX))
        loso = f"LOSO RMSE {m['loso']['rmse']:.2f}" if m["loso"] else "no LOSO score"
        variant = f" ({m['variant']})" if m["variant"] else ""
        lines.append(f"    {m['model']}{variant}: PM2.5 {np.nanmin(pm):.1f}-{np.nanmax(pm):.1f} ug/m3, "
                     f"byte error max {error.max():.2f}, {loso}")
        del pm, sd, pm_code, sd_code

    (args.out / "readings.json").write_text(json.dumps(observed, separators=(",", ":")), encoding="utf-8")
    sizes["readings.json"] = (args.out / "readings.json").stat().st_size
    after = sum(v is not None for s in observed["stations"] for v in s["pm25"][LOOKBACK_H:])
    manifest = {
        "version": 2,
        "created_utc": utc(pd.Timestamp(datetime.now(timezone.utc))),
        "grid": {"epsg": config.GRID_EPSG, "x0": config.GRID_ORIGIN[0], "y0": config.GRID_ORIGIN[1],
                 "cell": config.GRID_CELL, "rows": config.GRID_SHAPE[0], "cols": config.GRID_SHAPE[1]},
        "origin_utc": utc(origin),
        "issued_utc": utc(origin + pd.Timedelta(hours=1)),
        "data_as_of_utc": utc(as_of),
        "horizon": HORIZON_H,
        "stations": station_list,
        "interval": {"coverage": config.INTERVAL_COVERAGE,
                     "z": round(float(norm.ppf(0.5 + config.INTERVAL_COVERAGE / 2)), 6)},
        "default_model": default,
        "default_by": default_by,
        "models": entries,
        "readings": {"file": "readings.json", "stations": len(observed["stations"]),
                     "forecast_hour_readings": after},
    }
    tmp = args.out / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    tmp.replace(args.out / "manifest.json")
    named = {e["pm25"]["file"] for e in entries} | {e["sd"]["file"] for e in entries}
    stale = sorted(p.name for p in args.out.glob("*.u8.gz") if p.name.startswith(("pm25", "sd")) and p.name not in named)
    for name in stale:
        (args.out / name).unlink()

    why = {"choice": "given", "loso_rmse": "lowest LOSO RMSE", "order": "first map"}[default_by]
    print(f"[EXPORT] {len(maps)} map(s) of {utc(origin)}: {', '.join(ids)}; the app opens on {default} ({why}), "
          f"issued {utc(origin + pd.Timedelta(hours=1))}, {len(station_list)} stations, data as of {utc(as_of)}")
    print(f"    readings: {len(observed['stations'])} stations ({len(station_list)} made the map), "
          f"{after} readings in the forecast hours{' (a past forecast)' if after else ''}")
    print(f"    {int((~water[0]).sum())} land cells x {HORIZON_H} h")
    for line in lines:
        print(line)
    print("    " + ", ".join(f"{name} {size / 1e6:.2f} MB" for name, size in sizes.items()) + f" -> {args.out}"
          + (f"; removed {', '.join(stale)}" if stale else ""))

if __name__ == "__main__":
    main()
