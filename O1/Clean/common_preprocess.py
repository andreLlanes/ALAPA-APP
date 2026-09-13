"""Cleaning and transform steps shared by the Clean stage.

Inputs match the raw database tables:
    gs_measurements : location_key, timestamp_utc, pm25, city
    ifs_met         : city, latitude, longitude, timestamp_utc, temperature_c,
                      humidity_pct, wind_speed_ms, wind_gusts_ms, wind_dir_deg,
                      surface_pressure_hpa, precipitation_mm

clean_openaq (PM2.5) and transform_openmeteo (meteorology) run independently.
ifs_met has no location_key -- grid cells are keyed by city/latitude/longitude --
so joining stations to their nearest grid cell is a separate step, not done here.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
from schema import PM25_COL, KEY_COL, TIME_COL, MET_COLS

# Grid-cell identity for ifs_met (clean-side only; not in schema).
GRID_KEY_COLS = ["city", "latitude", "longitude"]

# QC / gap-fill thresholds.
FLATLINE_MIN_RUN = 6
COMPLETENESS_MIN = 0.90
GAP_FILL_LIMIT = 3


def _reindex_hourly(g: pd.DataFrame) -> pd.DataFrame:
    """Reindex one station onto a gap-free hourly grid spanning its active range."""
    full = pd.date_range(g[TIME_COL].min(), g[TIME_COL].max(), freq="h", tz="UTC")
    g = g.set_index(TIME_COL).reindex(full)
    g.index.name = TIME_COL
    return g.reset_index()


def _remove_flatlines(s: pd.Series, min_run: int = FLATLINE_MIN_RUN) -> pd.Series:
    """Null out runs of ``min_run`` or more identical consecutive values.

    Targets stuck sensors. NaNs break a run, so a constant stretch split by a gap
    is treated as two separate runs.
    """
    s = s.copy()
    notna = s.notna()
    changed = (s != s.shift()) | (~notna) | (~notna.shift(fill_value=False))
    grp = changed.cumsum()
    run_len = s.groupby(grp).transform("size")
    s[notna & (run_len >= min_run)] = np.nan
    return s


def _passes_completeness(s: pd.Series, threshold: float = COMPLETENESS_MIN) -> bool:
    """Return whether a station meets the minimum non-missing fraction."""
    return len(s) > 0 and s.notna().mean() >= threshold


def _interpolate_short_gaps(s: pd.Series, limit: int = GAP_FILL_LIMIT) -> pd.Series:
    """Linearly interpolate interior gaps up to ``limit`` hours; leave longer gaps NaN."""
    s = s.copy()
    isna = s.isna()
    grp = (isna != isna.shift()).cumsum()
    run_len = s.groupby(grp).transform("size")
    long_gap = isna & (run_len > limit)
    filled = s.interpolate(method="linear", limit=limit, limit_area="inside")
    filled[long_gap] = np.nan
    return filled


def clean_openaq(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full PM2.5 QC and gap-fill pipeline over all stations.

    Order: drop pm25 <= 0, mean-collapse duplicate timestamps, reindex to an
    hourly grid, remove flatlines, drop stations below the completeness threshold,
    then interpolate short gaps. Constant per-station columns (city, is_mobile)
    are carried through unchanged.
    """
    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True)
    df = df[df[PM25_COL] > 0]

    passthrough_cols = [c for c in ("city", "is_mobile") if c in df.columns]
    agg = {PM25_COL: "mean", **{c: "first" for c in passthrough_cols}}
    df = df.groupby([KEY_COL, TIME_COL], as_index=False).agg(agg)

    out_stations = []
    for key, g in df.groupby(KEY_COL, sort=False):
        passthrough_vals = {c: g[c].iloc[0] for c in passthrough_cols}
        g = g[[TIME_COL, PM25_COL]].sort_values(TIME_COL)
        g = _reindex_hourly(g)
        g[PM25_COL] = _remove_flatlines(g[PM25_COL])
        if not _passes_completeness(g[PM25_COL]):
            continue
        g[PM25_COL] = _interpolate_short_gaps(g[PM25_COL])
        g[KEY_COL] = key
        for c, v in passthrough_vals.items():
            g[c] = v
        out_stations.append(g)

    if not out_stations:
        cols = [KEY_COL, TIME_COL, PM25_COL] + passthrough_cols
        return pd.DataFrame(columns=cols)
    return pd.concat(out_stations, ignore_index=True)


def transform_openmeteo(df: pd.DataFrame) -> pd.DataFrame:
    """Convert ifs_met wind to u/v components; pass other met variables through.

    Wind speed/direction become orthogonal components (u = -s*sin(theta),
    v = -s*cos(theta)); gusts, temperature, humidity, precipitation, and surface
    pressure are unchanged. Grid cells stay keyed by city/latitude/longitude.
    Raises if the expected columns are absent; warns on any missing met values.
    """
    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True)
    theta = np.deg2rad(df["wind_dir_deg"])
    s = df["wind_speed_ms"]
    df["wind_u"] = -s * np.sin(theta)
    df["wind_v"] = -s * np.cos(theta)
    df = df.drop(columns=["wind_speed_ms", "wind_dir_deg"])

    expected = GRID_KEY_COLS + [TIME_COL] + MET_COLS
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"ifs_met frame missing columns: {missing}")
    n_missing = int(df[MET_COLS].isna().sum().sum())
    if n_missing:
        print(f"[transform_openmeteo] warning: {n_missing} missing met values; "
              f"affected rows will be dropped downstream.")
    return df[expected]


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical sin/cos encodings for hour (T=24), weekday (T=7), month (T=12).

    Month uses ``t = month - 1`` so the period spans 0-11.
    """
    df = df.copy()
    ts = df[TIME_COL].dt

    def enc(t, period, name):
        df[f"{name}_sin"] = np.sin(2 * np.pi * t / period)
        df[f"{name}_cos"] = np.cos(2 * np.pi * t / period)

    enc(ts.hour, 24, "hour")
    enc(ts.dayofweek, 7, "dow")
    enc(ts.month - 1, 12, "month")
    return df"""Cleaning and transform steps shared by the Clean stage.

Inputs match the raw database tables:
    gs_measurements : location_key, timestamp_utc, pm25, city
    ifs_met         : city, latitude, longitude, timestamp_utc, temperature_c,
                      humidity_pct, wind_speed_ms, wind_gusts_ms, wind_dir_deg,
                      surface_pressure_hpa, precipitation_mm

clean_openaq (PM2.5) and transform_openmeteo (meteorology) run independently.
ifs_met has no location_key -- grid cells are keyed by city/latitude/longitude --
so joining stations to their nearest grid cell is a separate step, not done here.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
from schema import PM25_COL, KEY_COL, TIME_COL, MET_COLS

# Grid-cell identity for ifs_met (clean-side only; not in schema).
GRID_KEY_COLS = ["city", "latitude", "longitude"]

# QC / gap-fill thresholds.
FLATLINE_MIN_RUN = 6
COMPLETENESS_MIN = 0.90
GAP_FILL_LIMIT = 3


def _reindex_hourly(g: pd.DataFrame) -> pd.DataFrame:
    """Reindex one station onto a gap-free hourly grid spanning its active range."""
    full = pd.date_range(g[TIME_COL].min(), g[TIME_COL].max(), freq="h", tz="UTC")
    g = g.set_index(TIME_COL).reindex(full)
    g.index.name = TIME_COL
    return g.reset_index()


def _remove_flatlines(s: pd.Series, min_run: int = FLATLINE_MIN_RUN) -> pd.Series:
    """Null out runs of ``min_run`` or more identical consecutive values.

    Targets stuck sensors. NaNs break a run, so a constant stretch split by a gap
    is treated as two separate runs.
    """
    s = s.copy()
    notna = s.notna()
    changed = (s != s.shift()) | (~notna) | (~notna.shift(fill_value=False))
    grp = changed.cumsum()
    run_len = s.groupby(grp).transform("size")
    s[notna & (run_len >= min_run)] = np.nan
    return s


def _passes_completeness(s: pd.Series, threshold: float = COMPLETENESS_MIN) -> bool:
    """Return whether a station meets the minimum non-missing fraction."""
    return len(s) > 0 and s.notna().mean() >= threshold


def _interpolate_short_gaps(s: pd.Series, limit: int = GAP_FILL_LIMIT) -> pd.Series:
    """Linearly interpolate interior gaps up to ``limit`` hours; leave longer gaps NaN."""
    s = s.copy()
    isna = s.isna()
    grp = (isna != isna.shift()).cumsum()
    run_len = s.groupby(grp).transform("size")
    long_gap = isna & (run_len > limit)
    filled = s.interpolate(method="linear", limit=limit, limit_area="inside")
    filled[long_gap] = np.nan
    return filled


def clean_openaq(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full PM2.5 QC and gap-fill pipeline over all stations.

    Order: drop pm25 <= 0, mean-collapse duplicate timestamps, reindex to an
    hourly grid, remove flatlines, drop stations below the completeness threshold,
    then interpolate short gaps. Constant per-station columns (city, is_mobile)
    are carried through unchanged.
    """
    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True)
    df = df[df[PM25_COL] > 0]

    passthrough_cols = [c for c in ("city", "is_mobile") if c in df.columns]
    agg = {PM25_COL: "mean", **{c: "first" for c in passthrough_cols}}
    df = df.groupby([KEY_COL, TIME_COL], as_index=False).agg(agg)

    out_stations = []
    for key, g in df.groupby(KEY_COL, sort=False):
        passthrough_vals = {c: g[c].iloc[0] for c in passthrough_cols}
        g = g[[TIME_COL, PM25_COL]].sort_values(TIME_COL)
        g = _reindex_hourly(g)
        g[PM25_COL] = _remove_flatlines(g[PM25_COL])
        if not _passes_completeness(g[PM25_COL]):
            continue
        g[PM25_COL] = _interpolate_short_gaps(g[PM25_COL])
        g[KEY_COL] = key
        for c, v in passthrough_vals.items():
            g[c] = v
        out_stations.append(g)

    if not out_stations:
        cols = [KEY_COL, TIME_COL, PM25_COL] + passthrough_cols
        return pd.DataFrame(columns=cols)
    return pd.concat(out_stations, ignore_index=True)


def transform_openmeteo(df: pd.DataFrame) -> pd.DataFrame:
    """Convert ifs_met wind to u/v components; pass other met variables through.

    Wind speed/direction become orthogonal components (u = -s*sin(theta),
    v = -s*cos(theta)); gusts, temperature, humidity, precipitation, and surface
    pressure are unchanged. Grid cells stay keyed by city/latitude/longitude.
    Raises if the expected columns are absent; warns on any missing met values.
    """
    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True)
    theta = np.deg2rad(df["wind_dir_deg"])
    s = df["wind_speed_ms"]
    df["wind_u"] = -s * np.sin(theta)
    df["wind_v"] = -s * np.cos(theta)
    df = df.drop(columns=["wind_speed_ms", "wind_dir_deg"])

    expected = GRID_KEY_COLS + [TIME_COL] + MET_COLS
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"ifs_met frame missing columns: {missing}")
    n_missing = int(df[MET_COLS].isna().sum().sum())
    if n_missing:
        print(f"[transform_openmeteo] warning: {n_missing} missing met values; "
              f"affected rows will be dropped downstream.")
    return df[expected]


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical sin/cos encodings for hour (T=24), weekday (T=7), month (T=12).

    Month uses ``t = month - 1`` so the period spans 0-11.
    """
    df = df.copy()
    ts = df[TIME_COL].dt

    def enc(t, period, name):
        df[f"{name}_sin"] = np.sin(2 * np.pi * t / period)
        df[f"{name}_cos"] = np.cos(2 * np.pi * t / period)

    enc(ts.hour, 24, "hour")
    enc(ts.dayofweek, 7, "dow")
    enc(ts.month - 1, 12, "month")
    return df