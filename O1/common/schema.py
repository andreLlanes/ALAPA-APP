"""Shared schema for the PM2.5 forecasting pipeline.

Single source of truth for column names, feature groups, and horizon geometry,
imported by both the cleaning stage (``Clean/``) and the dataset builders
(``Builders/``) so they cannot disagree on the data contract.
"""

import numpy as np

PM25_COL = "pm25"
KEY_COL = "location_key"
TIME_COL = "timestamp_utc"

# Merged tables written by Clean/ and read by Builders/ and O2. Versioned so a
# rebuild under a changed schema never writes into, or reads from, an older one;
# bump the suffix here and every reader follows.
MERGED_TABLE = "openaq.merged_clean_v2"   # rows with pm25 present
MASKED_TABLE = "openaq.merged_masked_v2"  # every active-range hour, long gaps as NULL

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


def exclusion_window_starts(g, lookback: int = LOOKBACK_H, horizon: int = HORIZON_H) -> np.ndarray:
    """Return the start rows of every valid exclusion window in one station's frame.

    The single definition of a usable window, shared by the LSTM builder and the
    baselines so they are always scored on the same windows. A window starting at
    row ``s`` is valid when the encoder features are complete over the lookback,
    the decoder features and PM2.5 target are complete over the horizon, and every
    step is exactly one hour after the last. ``g`` must be sorted by time.
    """
    window_len = lookback + horizon
    n = len(g)
    if n < window_len:
        return np.empty(0, dtype=int)

    enc_ok = ~np.isnan(g[ENCODER_COLS].to_numpy(dtype=float)).any(axis=1)
    dec_ok = ~np.isnan(g[DECODER_COLS].to_numpy(dtype=float)).any(axis=1)
    pm_ok = ~np.isnan(g[PM25_COL].to_numpy(dtype=float))
    step_ok = contiguous_step_mask(g[TIME_COL].to_numpy())

    def all_true(mask, lo, hi):
        """For each start s, whether mask[s+lo : s+hi] is entirely True."""
        c = np.concatenate([[0], np.cumsum(mask)])
        s = np.arange(n - window_len + 1)
        return (c[s + hi] - c[s + lo]) == (hi - lo)

    valid = (all_true(enc_ok, 0, lookback)
             & all_true(dec_ok, lookback, window_len)
             & all_true(pm_ok, lookback, window_len)
             & all_true(step_ok, 1, window_len))
    return np.flatnonzero(valid)