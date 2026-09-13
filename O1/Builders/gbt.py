"""GBT dataset builder: tabular recasting with normalized per-station output.

Two Parquet files per station, which avoids duplicating the backward-looking
feature block across all 72 lead times (that duplication was ~90% of total
dataset size):
    <station>_backward.parquet - location_key, origin, lags, rolling stats
                                 (one row per origin).
    <station>_leads.parquet    - location_key, origin, lead, target-hour time
                                 encodings, target-hour met, y (72 rows per origin).
Rejoin on (location_key, origin) at training time to recover the exact flat table;
this is lossless de-duplication.

    clean source  -> build_gbt_tables (exclusion: full 144h span valid)
    masked source -> build_gbt_tables_masked (keep lookback-NaN; lags carry NaN,
                     rolling stats over present hours; native tree handling)

There is no separate mask array: NaN in the feature columns is the mask, consumed
natively by XGBoost/LightGBM. Feature columns are float32 and unnormalized.
"""

import os

import numpy as np
import pandas as pd

from common_build import station_keys, load_station, shard_dir, safe_name  # sets sys.path
from schema import (
    KEY_COL, TIME_COL, PM25_COL, MET_COLS, TIME_COLS,
    LAG_OFFSETS, ROLL_WINDOWS, ROLL_STATS, LOOKBACK_H, HORIZON_H, contiguous_step_mask,
)
from _masked_gbt import build_gbt_tables_masked


def build_gbt_tables(df, lookback=LOOKBACK_H, horizon=HORIZON_H, stride=1):
    """Build one tabular frame per lead time (exclusion source).

    For each (station, origin) with a fully valid 144h span, every lead time h
    gets a row of: the backward-looking PM2.5 features (lags and rolling stats,
    shared across h), the target-hour time encodings and met (specific to h), and
    target y = pm25 at origin + h.

    Vectorized per station: valid origins are found once via prefix sums, the
    backward features are gathered by fancy indexing, and each lead table is
    assembled by column-stacking rather than per-row dicts.

    Returns:
        tables: dict mapping lead time (1..horizon) to a DataFrame.
        feature_cols: predictor column names, shared across all lead tables.
    """
    window_len = lookback + horizon
    lag_names = [f"pm25_lag_{l}" for l in LAG_OFFSETS]
    roll_names = [f"pm25_roll{w}_{s}" for w in ROLL_WINDOWS for s in ROLL_STATS]
    feature_cols = lag_names + roll_names + TIME_COLS + MET_COLS

    lag_offsets = np.asarray(LAG_OFFSETS)
    per_lead_backward = {h: [] for h in range(1, horizon + 1)}
    per_lead_target_met = {h: [] for h in range(1, horizon + 1)}
    per_lead_target_tenc = {h: [] for h in range(1, horizon + 1)}
    per_lead_y = {h: [] for h in range(1, horizon + 1)}
    per_lead_key = {h: [] for h in range(1, horizon + 1)}
    per_lead_origin = {h: [] for h in range(1, horizon + 1)}

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

        def all_true_over(mask, length):
            csum = np.concatenate(([0], np.cumsum(mask)))
            return (csum[starts + length] - csum[starts]) == length

        pm_full_ok = all_true_over(pm_ok, window_len)
        csum_met = np.concatenate(([0], np.cumsum(met_ok)))
        met_horizon_ok = (csum_met[starts + window_len]
                          - csum_met[starts + lookback]) == horizon
        csum_step = np.concatenate(([0], np.cumsum(step_ok)))
        step_win_ok = (csum_step[starts + window_len]
                       - csum_step[starts + 1]) == (window_len - 1)

        valid = pm_full_ok & met_horizon_ok & step_win_ok
        s = starts[valid]
        if len(s) == 0:
            continue
        mid = s + lookback
        origin_idx = mid - 1

        lag_cols = pm25[origin_idx[:, None] - lag_offsets[None, :]]
        roll_blocks = []
        for w in ROLL_WINDOWS:
            idx = origin_idx[:, None] - np.arange(w)[None, :]
            seg = pm25[idx]
            roll_blocks.append(seg.mean(axis=1))
            roll_blocks.append(seg.max(axis=1))
            roll_blocks.append(seg.std(axis=1, ddof=1) if w > 1
                               else np.zeros(len(s)))
        roll_cols = np.column_stack(roll_blocks)
        backward = np.column_stack([lag_cols, roll_cols])

        origins = times[origin_idx]
        keys = np.full(len(s), key, dtype=object)

        for h in range(1, horizon + 1):
            ti = mid + h - 1
            per_lead_backward[h].append(backward)
            per_lead_target_met[h].append(met[ti])
            per_lead_target_tenc[h].append(tenc[ti])
            per_lead_y[h].append(pm25[ti])
            per_lead_key[h].append(keys)
            per_lead_origin[h].append(origins)

    tables = {}
    col_order = [KEY_COL, "origin"] + feature_cols + ["y"]
    for h in range(1, horizon + 1):
        if not per_lead_backward[h]:
            tables[h] = pd.DataFrame(columns=col_order)
            continue
        backward = np.concatenate(per_lead_backward[h])
        tmet = np.concatenate(per_lead_target_met[h])
        ttenc = np.concatenate(per_lead_target_tenc[h])
        y = np.concatenate(per_lead_y[h])
        keys = np.concatenate(per_lead_key[h])
        origins = np.concatenate(per_lead_origin[h])

        feat_matrix = np.column_stack([backward, ttenc, tmet])
        tbl = pd.DataFrame(feat_matrix, columns=feature_cols)
        tbl.insert(0, "origin", origins)
        tbl.insert(0, KEY_COL, keys)
        tbl["y"] = y
        tables[h] = tbl[col_order]
    return tables, feature_cols


def build(source, city):
    """Write normalized backward/leads Parquet pairs for every station.

    Per station, the 72 lead tables are stacked, downcast to float32, then split:
    the backward feature block (identical across leads) is stored once per origin,
    and only the per-lead columns go to the leads file.
    """
    keys = station_keys(source, city)
    outdir = shard_dir("gbt", city, source)
    builder = build_gbt_tables if source == "clean" else build_gbt_tables_masked

    lag_names = [f"pm25_lag_{l}" for l in LAG_OFFSETS]
    roll_names = [f"pm25_roll{w}_{s}" for w in ROLL_WINDOWS for s in ROLL_STATS]
    backward_cols = lag_names + roll_names
    lead_cols = list(TIME_COLS) + list(MET_COLS)

    total_back = total_lead = 0
    written = 0
    for location_key in keys:
        df = load_station(source, location_key)
        if df.empty:
            continue
        tables, _ = builder(df)
        frames = []
        for h, tbl in tables.items():
            if len(tbl):
                tbl = tbl.copy()
                tbl["lead"] = h
                frames.append(tbl)
        if not frames:
            continue
        station_tbl = pd.concat(frames, ignore_index=True)
        floatcols = station_tbl.select_dtypes("float64").columns
        station_tbl[floatcols] = station_tbl[floatcols].astype("float32")

        backward = (station_tbl[["location_key", "origin"] + backward_cols]
                    .drop_duplicates(["location_key", "origin"])
                    .reset_index(drop=True))
        leads = station_tbl[["location_key", "origin", "lead"] + lead_cols + ["y"]]

        stem = safe_name(location_key)
        backward.to_parquet(os.path.join(outdir, f"{stem}_backward.parquet"), index=False)
        leads.to_parquet(os.path.join(outdir, f"{stem}_leads.parquet"), index=False)
        total_back += len(backward)
        total_lead += len(leads)
        written += 1

    print(f"[gbt] {source}/{city}: {written} stations -> {total_back} backward rows "
          f"(1/origin) + {total_lead} lead rows -> {outdir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, choices=["clean", "masked"])
    p.add_argument("--city", required=True)
    build(**vars(p.parse_args()))