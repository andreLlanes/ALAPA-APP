"""Masked-source GBT table builder.

Differs from the exclusion build_gbt_tables in three ways:
    - A window is kept when the target (origin+h pm25) and the target-hour met
      are present; the lookback pm25 may contain NaN (the masked case).
    - Lag features gather pm25 directly, so a lag on a missing hour is NaN and is
      left for the tree's native missing-value handling.
    - Rolling mean/max/std are computed over present hours only (NaN-aware); a
      rolling window that is entirely missing yields NaN.

Rolling std uses ddof=1 over present values, so a window with fewer than two
present values yields NaN, which native tree handling absorbs. Vectorized per
station.
"""

import warnings

import numpy as np
import pandas as pd

from schema import KEY_COL, TIME_COL, PM25_COL, MET_COLS, TIME_COLS, contiguous_step_mask
from schema import LAG_OFFSETS, ROLL_WINDOWS, ROLL_STATS

LOOKBACK_H = 72
HORIZON_H = 72


def build_gbt_tables_masked(df, lookback=LOOKBACK_H, horizon=HORIZON_H, stride=1):
    """Build one tabular frame per lead time, keeping lookback-NaN windows.

    A window is retained when its target and target-hour met are present; the
    lookback pm25 may be missing. Lag and rolling features therefore carry NaN
    (NaN-aware for rollings), which XGBoost/LightGBM handle natively.

    Returns:
        tables: dict mapping lead time (1..horizon) to a DataFrame.
        feature_cols: predictor column names, shared across all lead tables.
    """
    window_len = lookback + horizon
    lag_names = [f"pm25_lag_{l}" for l in LAG_OFFSETS]
    roll_names = [f"pm25_roll{w}_{s}" for w in ROLL_WINDOWS for s in ROLL_STATS]
    feature_cols = lag_names + roll_names + TIME_COLS + MET_COLS
    lag_offsets = np.asarray(LAG_OFFSETS)

    per_lead = {h: {"back": [], "met": [], "tenc": [], "y": [], "key": [], "origin": []}
                for h in range(1, horizon + 1)}

    for key, g in df.groupby(KEY_COL, sort=False):
        g = g.sort_values(TIME_COL).reset_index(drop=True)
        pm25 = g[PM25_COL].to_numpy(dtype=float)
        met = g[MET_COLS].to_numpy(dtype=float)
        tenc = g[TIME_COLS].to_numpy(dtype=float)
        times = g[TIME_COL].to_numpy()
        n = len(g)
        if n < window_len:
            continue

        pm_ok = ~np.isnan(pm25)
        met_ok = ~np.isnan(met).any(axis=1)
        step_ok = contiguous_step_mask(times)
        starts = np.arange(0, n - window_len + 1, stride)
        if len(starts) == 0:
            continue

        # Validity requires the target and horizon met present and a contiguous
        # window; lookback pm25 is deliberately NOT required (that is the mask).
        csum_pm = np.concatenate(([0], np.cumsum(pm_ok)))
        pm_horizon_ok = (csum_pm[starts + window_len]
                         - csum_pm[starts + lookback]) == horizon
        csum_met = np.concatenate(([0], np.cumsum(met_ok)))
        met_horizon_ok = (csum_met[starts + window_len]
                          - csum_met[starts + lookback]) == horizon
        csum_step = np.concatenate(([0], np.cumsum(step_ok)))
        step_win_ok = (csum_step[starts + window_len]
                       - csum_step[starts + 1]) == (window_len - 1)

        valid = pm_horizon_ok & met_horizon_ok & step_win_ok
        s = starts[valid]
        if len(s) == 0:
            continue
        mid = s + lookback
        origin_idx = mid - 1

        lag_cols = pm25[origin_idx[:, None] - lag_offsets[None, :]]

        roll_blocks = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for w in ROLL_WINDOWS:
                idx = origin_idx[:, None] - np.arange(w)[None, :]
                seg = pm25[idx]
                roll_blocks.append(np.nanmean(seg, axis=1))
                roll_blocks.append(np.nanmax(seg, axis=1))
                if w > 1:
                    roll_blocks.append(np.nanstd(seg, axis=1, ddof=1))
                else:
                    roll_blocks.append(np.full(len(s), np.nan))
        roll_cols = np.column_stack(roll_blocks)
        backward = np.column_stack([lag_cols, roll_cols])

        origins = times[origin_idx]
        keys = np.full(len(s), key, dtype=object)
        for h in range(1, horizon + 1):
            ti = mid + h - 1
            per_lead[h]["back"].append(backward)
            per_lead[h]["met"].append(met[ti])
            per_lead[h]["tenc"].append(tenc[ti])
            per_lead[h]["y"].append(pm25[ti])
            per_lead[h]["key"].append(keys)
            per_lead[h]["origin"].append(origins)

    tables = {}
    col_order = [KEY_COL, "origin"] + feature_cols + ["y"]
    for h in range(1, horizon + 1):
        b = per_lead[h]
        if not b["back"]:
            tables[h] = pd.DataFrame(columns=col_order)
            continue
        feat = np.column_stack([np.concatenate(b["back"]),
                                np.concatenate(b["tenc"]),
                                np.concatenate(b["met"])])
        tbl = pd.DataFrame(feat, columns=feature_cols)
        tbl.insert(0, "origin", np.concatenate(b["origin"]))
        tbl.insert(0, KEY_COL, np.concatenate(b["key"]))
        tbl["y"] = np.concatenate(b["y"])
        tables[h] = tbl[col_order]
    return tables, feature_cols