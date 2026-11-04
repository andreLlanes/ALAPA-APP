""" Cleaning and transform steps shared by the Clean stage.

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

import numpy as np
import pandas as pd

from Common.schema import TIME_COL, MET_COLS

# Grid-cell identity for ifs_met (clean-side only; not in schema).
GRID_KEY_COLS = ["city", "latitude", "longitude"]

# QC thresholds and physical bounds.
FLATLINE_MIN_RUN = 6          # >= this many identical consecutive values is a flatline
GAP_EDGE_KEEP = 6             # keep this many NaN hours each side of a gap; drop the rest
PM25_MAX = 1000.0             # upper physical-plausibility bound (ug/m3); above this = fault
# Temporal features are encoded in each city's own local time (LA handles DST).
CITY_TZ = {
    "Metro Manila": "Asia/Manila",
    "Bangkok": "Asia/Bangkok",
    "Los Angeles": "America/Los_Angeles",
}


def _reindex_hourly(g: pd.DataFrame) -> pd.DataFrame:
    """ Reindex one station onto a gap-free hourly grid spanning its active range.
    """
    full = pd.date_range(g[TIME_COL].min(), g[TIME_COL].max(), freq="h", tz="UTC")
    g = g.set_index(TIME_COL).reindex(full)
    g.index.name = TIME_COL
    return g.reset_index()


def _remove_flatlines(s: pd.Series, min_run: int = FLATLINE_MIN_RUN) -> pd.Series:
    """ Null out runs of ``min_run`` or more identical consecutive values.

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


def _gap_length(s: pd.Series):
    """ Return the NaN-run length each missing hour belongs to; NA where present.

        No interpolation is performed here: filling is deferred to training so it can
        be fit within each train/val/test split, avoiding leakage. A present value
        gets <NA> (not applicable); every hour inside a k-hour gap gets the integer k,
        so any short/long threshold can be applied downstream.
    """
    isna = s.isna()
    grp = (isna != isna.shift()).cumsum()
    run_len = s.groupby(grp).transform("size")
    return run_len.where(isna).astype("Int64")


def _trim_long_gaps(g: pd.DataFrame, pm25_col: str, edge: int = GAP_EDGE_KEEP) -> pd.DataFrame:
    """ Drop the deep interior of over-long NaN gaps, keeping ``edge`` hours each side.

        No interpolation: a gap of length L stays entirely NaN when L <= 2*edge; when
        longer, only the ``edge`` hours nearest each end are kept and the interior rows
        are removed. The series is no longer a contiguous hourly grid across a trimmed
        gap, so windows never span it (follows_previous_hour rejects the seam).
    """
    isna = g[pm25_col].isna().to_numpy()
    if not isna.any():
        return g
    # run id and within-run position for NaN hours
    grp = (isna != np.concatenate(([not isna[0]], isna[:-1]))).cumsum()
    run_len = pd.Series(isna).groupby(grp).transform("size").to_numpy()
    pos = np.arange(len(isna)) - pd.Series(np.arange(len(isna))).groupby(grp).transform("min").to_numpy()
    # drop a NaN hour if its gap is longer than 2*edge AND it is not within edge of either end
    drop = isna & (run_len > 2 * edge) & (pos >= edge) & (pos < run_len - edge)
    return g.loc[~drop].reset_index(drop=True)


def transform_openmeteo(df: pd.DataFrame) -> pd.DataFrame:
    """ Convert ifs_met wind to u/v components; pass other met variables through.

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
    """ Add cyclical sin/cos encodings for hour (T=24), weekday (T=7), month (T=12).

        Timestamps are converted to the frame's city local time (CITY_TZ) before
        encoding, so each city's diurnal and weekly cycles reference its own clock.
        Month uses ``t = month - 1`` (0-11). Assumes a single city per frame.
    """
    df = df.copy()
    tz = CITY_TZ[df["city"].iloc[0]]
    local = df[TIME_COL].dt.tz_convert(tz)

    def enc(t, period, name):
        df[f"{name}_sin"] = np.sin(2 * np.pi * t / period)
        df[f"{name}_cos"] = np.cos(2 * np.pi * t / period)

    enc(local.dt.hour, 24, "hour")
    enc(local.dt.dayofweek, 7, "dow")
    enc(local.dt.month - 1, 12, "month")
    return df
