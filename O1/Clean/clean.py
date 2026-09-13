"""Grade-aware PM2.5 cleaning for a grade-merged frame.

Input rows are raw PM2.5 with is_monitor and station coordinates already attached
(location_key, timestamp_utc, pm25, is_monitor, plus constant per-station columns
city/latitude/longitude that pass through untouched).

Cleaning rules:
    negative/zero removal   - uniform (physically impossible for any grade).
    duplicate collapse      - uniform (mean of duplicate station-hours).
    reindex to hourly grid  - uniform (complete hourly timeline per station).
    flatline removal (>=6)  - low-cost sensors only (is_monitor is False);
                              reference monitors may hold a constant value legitimately.
    short-gap interpolation - uniform (<= 3 consecutive hours).
    spikes                  - never removed (may be real pollution episodes).

Completeness is not filtered here: low-cost sensors report intermittently, so
station selection by completeness is deferred to training where it can be tuned.
Unknown grade (is_monitor NULL) is treated as reference, i.e. the flatline rule
is skipped (see IS_MONITOR_DEFAULT).
"""

import pandas as pd

from common_preprocess import (
    PM25_COL, KEY_COL, TIME_COL,
    _reindex_hourly, _remove_flatlines, _interpolate_short_gaps,
)

PASSTHROUGH_COLS = ("city", "latitude", "longitude", "is_monitor")
IS_MONITOR_DEFAULT = True


def clean_pm25(df: pd.DataFrame) -> pd.DataFrame:
    """Clean all stations in a frame and return the concatenated result.

    Applies the grade-aware rules described in the module docstring. Returns an
    empty frame with the expected columns if no station survives.
    """
    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True)
    df = df[df[PM25_COL] > 0]

    passthrough = [c for c in PASSTHROUGH_COLS if c in df.columns]
    agg = {PM25_COL: "mean", **{c: "first" for c in passthrough}}
    df = df.groupby([KEY_COL, TIME_COL], as_index=False).agg(agg)

    out_stations = []
    for key, g in df.groupby(KEY_COL, sort=False):
        vals = {c: g[c].iloc[0] for c in passthrough}
        is_ref = bool(vals.get("is_monitor", IS_MONITOR_DEFAULT))
        if "is_monitor" in vals and pd.isna(vals["is_monitor"]):
            is_ref = IS_MONITOR_DEFAULT

        g = g[[TIME_COL, PM25_COL]].sort_values(TIME_COL)
        g = _reindex_hourly(g)
        if not is_ref:
            g[PM25_COL] = _remove_flatlines(g[PM25_COL])
        g[PM25_COL] = _interpolate_short_gaps(g[PM25_COL])
        g[KEY_COL] = key
        for c, v in vals.items():
            g[c] = v
        out_stations.append(g)

    if not out_stations:
        return pd.DataFrame(columns=[KEY_COL, TIME_COL, PM25_COL] + passthrough)
    return pd.concat(out_stations, ignore_index=True)


def clean_one_station(df: pd.DataFrame):
    """Clean one station's raw PM2.5 and return (cleaned_frame, stats).

    Applies the grade-aware rules described in the module docstring and adds two
    boolean label columns: ``short_gap_filled`` (value came from interpolation)
    and ``long_gap_missing`` (still NaN after interpolation).

    The stats dict records per-station provenance counts (all ints unless noted):
        raw_rows               - rows received.
        nonpositive_dropped    - rows removed for pm25 <= 0.
        duplicate_rows         - duplicate station-hours collapsed by mean.
        active_range_hours     - hours from first to last observation (grid size).
        gap_hours_inserted     - NaN hours added by the hourly reindex.
        is_reference           - bool; whether the flatline rule was skipped.
        flatline_hours_removed - hours nulled by the flatline rule (0 if reference).
        short_gap_filled_count - hours filled by <= 3h interpolation.
        long_gap_missing_count - hours left NaN after interpolation.
        final_rows             - rows out (equals active_range_hours).
        final_valid            - non-null pm25 rows out.
    """
    stats = {"raw_rows": len(df)}
    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True)

    before = len(df)
    df = df[df[PM25_COL] > 0]
    stats["nonpositive_dropped"] = before - len(df)

    passthrough = [c for c in PASSTHROUGH_COLS if c in df.columns]
    vals = {c: df[c].iloc[0] for c in passthrough} if len(df) else {}

    before = len(df)
    df = df.groupby([KEY_COL, TIME_COL], as_index=False).agg(
        {PM25_COL: "mean", **{c: "first" for c in passthrough}})
    stats["duplicate_rows"] = before - len(df)

    if df.empty:
        stats.update(active_range_hours=0, gap_hours_inserted=0, is_reference=True,
                     flatline_hours_removed=0, short_gap_filled_count=0,
                     long_gap_missing_count=0, final_rows=0, final_valid=0)
        cols = [KEY_COL, TIME_COL, PM25_COL, "short_gap_filled", "long_gap_missing"] + passthrough
        return pd.DataFrame(columns=cols), stats

    key = df[KEY_COL].iloc[0]
    is_ref = bool(vals.get("is_monitor", IS_MONITOR_DEFAULT))
    if "is_monitor" in vals and pd.isna(vals["is_monitor"]):
        is_ref = IS_MONITOR_DEFAULT
    stats["is_reference"] = is_ref

    present = len(df)
    g = df[[TIME_COL, PM25_COL]].sort_values(TIME_COL)
    g = _reindex_hourly(g)
    stats["active_range_hours"] = len(g)
    stats["gap_hours_inserted"] = len(g) - present

    if not is_ref:
        before_valid = g[PM25_COL].notna().sum()
        g[PM25_COL] = _remove_flatlines(g[PM25_COL])
        stats["flatline_hours_removed"] = int(before_valid - g[PM25_COL].notna().sum())
    else:
        stats["flatline_hours_removed"] = 0

    gap_before = g[PM25_COL].isna()
    valid_before = g[PM25_COL].notna().sum()
    g[PM25_COL] = _interpolate_short_gaps(g[PM25_COL])
    g["short_gap_filled"] = gap_before & g[PM25_COL].notna()
    g["long_gap_missing"] = g[PM25_COL].isna()
    stats["short_gap_filled_count"] = int(g[PM25_COL].notna().sum() - valid_before)
    stats["long_gap_missing_count"] = int(g[PM25_COL].isna().sum())

    g[KEY_COL] = key
    for c, v in vals.items():
        g[c] = v
    stats["final_rows"] = len(g)
    stats["final_valid"] = int(g[PM25_COL].notna().sum())
    return g, stats