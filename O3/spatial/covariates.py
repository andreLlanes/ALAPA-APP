""" Covariates of the regression-kriging trend, read from the rasters of the common 100 m grid
    (EPSG:32651): building fraction and road density (static), transit emission intensity (one
    band per service band, chosen by the local hour predicted) and NDVI (one raster per quarter,
    chosen by the local date predicted, with a band flagging cells with a valid observation).
    A cell is water when it has no buildings and its median NDVI over the quarters with a valid
    observation is below WATER_NDVI (water has negative NDVI), or it has none. Each covariate is
    standardized with its mean and std over land cells, so stations and cells share one scale.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from rasterio.warp import transform as reproject_points

import config
from utils import runlog

FOLDER = Path(__file__).resolve().parents[1] / config.SPATIAL_DIR
COVARIATES = ("ndvi", "road", "building", "emission")
GRID = Affine(config.GRID_CELL, 0.0, config.GRID_ORIGIN[0], 0.0, -config.GRID_CELL,
              config.GRID_ORIGIN[1])

def read_layer(name: str):
    """ Read every band of one raster; return (bands [b, rows, cols] float64, band names), with NaN
        where a value is not finite or equals the raster's no-data value (a no-data value of 0 is
        kept, since 0 is a real value in these layers). Stops if the raster is not on the common
        grid, since its cells would not match the other layers.
    """
    path = FOLDER / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (set SPATIAL_DIR and the file names in config.py)")
    with rasterio.open(path) as src:
        if (src.crs is None or src.crs.to_epsg() != config.GRID_EPSG or src.shape != config.GRID_SHAPE
                or not src.transform.almost_equals(GRID)):
            raise ValueError(f"{path} is not on the common grid (EPSG:{config.GRID_EPSG}, origin "
                             f"{config.GRID_ORIGIN}, {config.GRID_CELL:g} m, {config.GRID_SHAPE})")
        bands = src.read().astype(np.float64)
        if src.nodata is not None and src.nodata != 0:
            bands[bands == src.nodata] = np.nan
        return bands, list(src.descriptions)

def named_bands(name: str, wanted) -> np.ndarray:
    """ Read the bands of one raster by name, in the order wanted: [len(wanted), rows, cols].
        A raster without band names is read in band order when it has exactly that many bands.
    """
    bands, names = read_layer(name)
    if not any(names) and len(bands) == len(wanted):
        return bands
    missing = [b for b in wanted if b not in names]
    if missing:
        raise ValueError(f"{name}: no band named {', '.join(missing)} (found {names})")
    return np.stack([bands[names.index(b)] for b in wanted])

def ndvi_file(quarter: int, start: pd.Timestamp) -> str:
    """ Find quarter's NDVI raster (1-based) and check that the start date in its name is the
        quarter's start.
    """
    pattern = config.NDVI_FILE.format(quarter=quarter)
    found = sorted(FOLDER.glob(pattern))
    if len(found) != 1:
        raise FileNotFoundError(f"{len(found)} files match {FOLDER / pattern}; expected one")
    head, tail = Path(pattern).name.split("*")
    named = found[0].name[len(head):len(found[0].name) - len(tail)]
    try:
        date = pd.Timestamp(named).date()
    except ValueError:
        raise ValueError(f"{found[0].name}: no start date in the name") from None
    if date != start.date():
        raise ValueError(f"{found[0].name}: starts {date}, but quarter {quarter} starts "
                         f"{start.date()} (NDVI_START in config.py)")
    return str(found[0].relative_to(FOLDER))

def check_land(name: str, values: np.ndarray, needed: np.ndarray):
    """ Stop if a layer has no value (NaN) in a cell where one is needed.
    """
    missing = np.isnan(values) & needed
    if missing.any():
        raise ValueError(f"{name}: no value in {int(missing.sum())} land cell(s)")

def standardize(values: np.ndarray, mask: np.ndarray):
    """ Return the mean and std of values over the masked cells (a zero std is set to 1).
    """
    picked = values[mask]
    sd = float(picked.std())
    return float(picked.mean()), sd if sd > 1e-12 else 1.0

class Covariates:
    """ The four covariate layers, their water mask and standardization, and lookups at stations.
    """

    def __init__(self):
        starts = [pd.Timestamp(config.NDVI_START, tz=config.LOCAL_TZ)
                  + pd.DateOffset(months=config.NDVI_MONTHS * q)
                  for q in range(config.NDVI_QUARTERS + 1)]
        self.quarter_starts = pd.DatetimeIndex(starts)
        self.building = named_bands(config.BUILDINGS_FILE, [config.BUILDINGS_BAND])[0]
        self.road = named_bands(config.ROADS_FILE, [config.ROADS_BAND])[0]
        self.emission = named_bands(config.EMISSION_FILE, list(config.EMISSION_BANDS))
        ndvi = [named_bands(ndvi_file(q + 1, starts[q]), list(config.NDVI_BANDS))
                for q in range(config.NDVI_QUARTERS)]
        self.ndvi = np.stack([q[0] for q in ndvi])
        self.ndvi_valid = np.stack([q[1] for q in ndvi]) > 0
        seen = self.ndvi_valid.any(0)
        median = np.full(seen.shape, np.nan)
        median[seen] = np.nanmedian(np.where(self.ndvi_valid, self.ndvi, np.nan)[:, seen], axis=0)
        self.water = (self.building == 0) & (~seen | (median < config.WATER_NDVI))
        land = ~self.water
        check_land(config.BUILDINGS_FILE, self.building, land)
        check_land(config.ROADS_FILE, self.road, land)
        check_land(config.EMISSION_FILE, self.emission, np.broadcast_to(land, self.emission.shape))
        check_land("NDVI", self.ndvi, self.ndvi_valid & land)
        self.scale = {"ndvi": standardize(self.ndvi, self.ndvi_valid & land),
                      "road": standardize(self.road, land),
                      "building": standardize(self.building, land),
                      "emission": standardize(self.emission, np.broadcast_to(land, self.emission.shape))}
        self.band_of_hour = np.array([band_index(h) for h in range(24)])
        r = {b: float(np.corrcoef(self.road[land], self.emission[i][land])[0, 1])
             for i, b in enumerate(config.EMISSION_BANDS)}
        print(f"[SPATIAL] Covariates: {int(land.sum())} land cells, {int(self.water.sum())} water; "
              f"road-emission r over land cells: "
              + ", ".join(f"{b.removeprefix('emis_')} {v:.3f}" for b, v in r.items()))
        runlog.detail("SPATIAL COVARIATES", {
            "folder": FOLDER, "land cells": int(land.sum()), "water cells": int(self.water.sum()),
            "road-emission r (land cells)": r,
            "standardization (mean, std over land)": self.scale})

    def cells(self, lat: np.ndarray, lon: np.ndarray):
        """ Project stations to the grid; return (x, y) in km and the (row, col) of their cells.
        """
        x, y = (np.asarray(v) for v in reproject_points("EPSG:4326", f"EPSG:{config.GRID_EPSG}",
                                                       np.asarray(lon, float), np.asarray(lat, float)))
        col = np.floor((x - config.GRID_ORIGIN[0]) / config.GRID_CELL).astype(int)
        row = np.floor((config.GRID_ORIGIN[1] - y) / config.GRID_CELL).astype(int)
        outside = (row < 0) | (row >= config.GRID_SHAPE[0]) | (col < 0) | (col >= config.GRID_SHAPE[1])
        if outside.any():
            raise ValueError(f"{int(outside.sum())} station(s) fall outside the common grid")
        return x / 1000.0, y / 1000.0, row, col

    def at(self, row: np.ndarray, col: np.ndarray, hours: np.ndarray) -> np.ndarray:
        """ Standardized covariates [hours, cells, 4] (COVARIATES order) at the given cells, for
            target hours given as hours since the epoch (UTC); NDVI is NaN where the cell has no
            valid NDVI in the hour's quarter. Stops if an hour falls outside the NDVI quarters.
        """
        local = pd.to_datetime(np.asarray(hours, np.int64), unit="h", utc=True).tz_convert(config.LOCAL_TZ)
        quarter = self.quarter_starts.searchsorted(local, side="right") - 1
        if ((quarter < 0) | (quarter >= config.NDVI_QUARTERS)).any():
            raise ValueError(f"hours outside the NDVI quarters ({self.quarter_starts[0].date()} to "
                             f"{self.quarter_starts[-1].date()})")
        band = self.band_of_hour[local.hour.to_numpy()]
        q, r, c = quarter[:, None], row[None, :], col[None, :]
        invalid = ~self.ndvi_valid[q, r, c]
        layers = {"ndvi": np.where(invalid, np.nan, self.ndvi[q, r, c]),
                  "road": np.broadcast_to(self.road[r, c], invalid.shape),
                  "building": np.broadcast_to(self.building[r, c], invalid.shape),
                  "emission": self.emission[band[:, None], r, c]}
        return np.stack([(layers[k] - self.scale[k][0]) / self.scale[k][1] for k in COVARIATES], -1)

def band_index(hour: int) -> int:
    """ Return the service band (EMISSION_BANDS order) of a local clock hour.
    """
    for i, (start, end) in enumerate(config.EMISSION_BANDS.values()):
        if (start <= hour < end) if start < end else (hour >= start or hour < end):
            return i
    raise ValueError(f"hour {hour} is in no service band")

@lru_cache(maxsize=1)
def load() -> Covariates:
    """ Read the covariate layers once per process.
    """
    return Covariates()
