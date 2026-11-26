""" The forecast map (train.py --map true): regression-kriging of every station's forecasts onto
    every land cell of the common 100 m grid, for the 72 forecast hours of the latest origin with
    at least MIN_RK_STATIONS forecasting stations. The forecasts are the mean over the run's seeds;
    the variogram is fitted from their validation forecasts, as in leave-one-station-out scoring.
    Writes two 72-band GeoTIFFs on the common grid, PM2.5 (ug/m3, set to 0 where negative) and the
    kriging variance (a relative uncertainty, lower near stations), and a record of the fit. Water
    cells are blank. When NDVI is in an hour's trend, a cell without valid NDVI in that quarter
    takes the trend fitted without NDVI.
"""

import numpy as np
import pandas as pd
import rasterio

import config
from Common.schema import HORIZON_H
from spatial import covariates
from spatial.covariates import COVARIATES
from spatial.kriging import ordinary_weights, residual_variance, trend
from spatial.loso import Geometry, by_origin, hours_of, origin_hours, pooled_variogram
from utils import runlog
from utils.artifacts import save_json

def latest_origin(data, ids):
    """ Return (origin in hours since the epoch, the window ids at it) for the latest origin among
        ids with at least MIN_RK_STATIONS forecasting stations.
    """
    groups = by_origin(origin_hours(data, ids))
    enough = [o for o, pos in groups.items() if len(pos) >= config.MIN_RK_STATIONS]
    if not enough:
        raise ValueError(f"no test origin has {config.MIN_RK_STATIONS} stations with a forecast")
    origin = max(enough)
    return origin, np.sort(ids[groups[origin]])

def stamp(origin: int) -> str:
    """ File-name time of an origin (UTC), e.g. 20260327T00.
    """
    return pd.Timestamp(origin, unit="h", tz="UTC").strftime("%Y%m%dT%H")

def record_path(folder, origin: int):
    """ Path of a map's record; the map is done when it exists.
    """
    return folder / f"map_{stamp(origin)}.json"

def write(path, bands: np.ndarray, descriptions):
    """ Write [bands, rows, cols] to a GeoTIFF on the common grid, NaN as no-data.
    """
    with rasterio.open(path, "w", driver="GTiff", height=config.GRID_SHAPE[0],
                       width=config.GRID_SHAPE[1], count=len(bands), dtype="float32",
                       crs=f"EPSG:{config.GRID_EPSG}", transform=covariates.GRID, nodata=np.nan,
                       compress="deflate") as dst:
        dst.write(bands.astype(np.float32))
        for i, text in enumerate(descriptions, 1):
            dst.set_band_description(i, text)

def make_map(folder, data, val, latest, seeds, title):
    """ Map one origin. val and latest are (window ids, forecasts [n,72] in ug/m3, the mean over
        seeds) for the validation windows and for the windows at the mapped origin.
    """
    cov = covariates.load()
    geo = Geometry(data, cov)
    variogram, candidates, pooled = pooled_variogram(data, geo, cov, *val)
    if variogram is None:
        raise ValueError("no variogram: too few validation origins with enough stations")
    ids, pred = latest
    origin = int(origin_hours(data, ids[:1])[0])
    s = data.station[ids]
    hours = hours_of(origin)
    x = cov.at(geo.row[s], geo.col[s], hours)
    rows, cols = np.nonzero(~cov.water)
    cx = (config.GRID_ORIGIN[0] + (cols + 0.5) * config.GRID_CELL) / 1000.0
    cy = (config.GRID_ORIGIN[1] - (rows + 0.5) * config.GRID_CELL) / 1000.0
    dist_cells = np.hypot(geo.xy[s, 0][:, None] - cx, geo.xy[s, 1][:, None] - cy)
    weights, ok_var = ordinary_weights(variogram, geo.dist(s, s), dist_cells)

    surface = np.full((2, HORIZON_H) + config.GRID_SHAPE, np.nan, np.float32)
    without_ndvi = np.array([k != "ndvi" for k in COVARIATES])
    dropped = filled_without_ndvi = 0
    for h in range(HORIZON_H):
        cells = cov.at(rows, cols, hours[h:h + 1])[0]
        full = trend(x[h], pred[:, h])
        dropped += int(not full[1].all())
        fits = [(np.ones(len(rows), bool), full)]
        missing = ~np.isfinite(cells[:, 0])
        if full[1][0] and missing.any():
            fits = [(~missing, full), (missing, trend(x[h], pred[:, h], without_ndvi))]
            filled_without_ndvi = max(filled_without_ndvi, int(missing.sum()))
        for where, (beta, used, rank, e) in fits:
            z = beta[0] + cells[where][:, used] @ beta[1:] + e @ weights[:, where]
            surface[0, h, rows[where], cols[where]] = np.maximum(z, 0.0)
            surface[1, h, rows[where], cols[where]] = residual_variance(e, rank) * ok_var[where]

    utc = pd.Timestamp(origin, unit="h", tz="UTC")
    leads = [f"lead {h + 1}h: {utc + pd.Timedelta(hours=h + 1):%Y-%m-%dT%H} UTC" for h in range(HORIZON_H)]
    folder.mkdir(parents=True, exist_ok=True)
    name = f"map_{stamp(origin)}"
    write(folder / f"{name}_pm25.tif", surface[0], leads)
    write(folder / f"{name}_variance.tif", surface[1], leads)
    record = {"origin_utc": f"{utc:%Y-%m-%dT%H}",
              "origin_local": f"{utc.tz_convert(config.LOCAL_TZ):%Y-%m-%dT%H}",
              "stations": [str(k) for k in data.keys[s]], "seeds_averaged": list(seeds),
              "validation_hours_pooled": int(pooled),
              "variogram_units": "standardized residuals; range in km",
              "variogram": {"model": variogram.model, "nugget": variogram.nugget,
                            "psill": variogram.psill, "range_km": variogram.range},
              "variogram_candidates": candidates,
              "land_cells": int(len(rows)), "water_cells": int(cov.water.sum()),
              "hours_covariate_left_out": dropped,
              "cells_without_ndvi_per_hour_max": filled_without_ndvi,
              "blank_land_cells_per_hour_max": int(np.isnan(surface[0][:, rows, cols]).sum(1).max())}
    save_json(record_path(folder, origin), record)
    runlog.detail(f"MAP {title}", record)
    print(f"    MAP {record['origin_local']} Manila ({record['origin_utc']} UTC): {len(s)} stations, "
          f"mean of {len(seeds)} seed(s); {variogram.model} variogram -> "
          f"{folder / (name + '_pm25.tif')}")
