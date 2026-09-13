"""Shared schema for the PM2.5 forecasting pipeline.

Single source of truth for column names, feature groups, and horizon geometry,
imported by both the cleaning stage (``Clean/``) and the dataset builders
(``Builders/``) so they cannot disagree on the data contract.
"""

import numpy as np

PM25_COL = "pm25"
KEY_COL = "location_key"
TIME_COL = "timestamp_utc"

# Meteorological covariates; wind is stored as u/v components, not speed/direction.
MET_COLS = [
    "wind_u",
    "wind_v",
    "wind_gusts_ms",
    "temperature_c",
    "humidity_pct",
    "precipitation_mm",
    "surface_pressure_hpa",
]

# Cyclical time encodings (sine/cosine pairs).
TIME_COLS = [
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
]

# 72h lookback + 72h forecast = 144h sample.
LOOKBACK_H = 72
HORIZON_H = 72
WINDOW_H = LOOKBACK_H + HORIZON_H

# Encoder sees the full past; decoder sees met + time only (non-autoregressive).
ENCODER_COLS = [PM25_COL] + MET_COLS + TIME_COLS
DECODER_COLS = MET_COLS + TIME_COLS

# GBT backward-looking PM2.5 features: lag offsets and rolling windows (hours).
LAG_OFFSETS = [1, 2, 3, 6, 12, 24, 48, 71]
ROLL_WINDOWS = [3, 6, 12, 24]
ROLL_STATS = ("mean", "max", "std")


def contiguous_step_mask(times: np.ndarray) -> np.ndarray:
    """Flag timestamps exactly one hour after their predecessor.

    Lets window builders reject samples that span a time gap. ``times`` must be
    sorted ascending; the first element is always ``True``.
    """
    one_hour = np.timedelta64(1, "h")
    ok = np.empty(len(times), dtype=bool)
    ok[0] = True
    ok[1:] = (times[1:] - times[:-1]) == one_hour
    return ok