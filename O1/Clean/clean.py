""" Grade-aware PM2.5 cleaning for a grade-merged frame.

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

from Common.schema import PM25_COL, KEY_COL, TIME_COL

# Threshold for the short/long split in the provenance counts only (not used in
# cleaning logic -- gap_length stores the raw run-length). Local to the cleaning
# stage since the shared schema no longer defines it.
SHORT_GAP_MAX_H = 6
from preprocess import (
    PM25_MAX,
    _reindex_hourly, _remove_flatlines, _gap_length, _trim_long_gaps,
)

PASSTHROUGH_COLS = ("city", "latitude", "longitude", "is_monitor")
IS_MONITOR_DEFAULT = True


def clean_one_station(df: pd.DataFrame):
    """ Clean one station's raw PM2.5 and return (cleaned_frame, stats).

        Applies the grade-aware rules described in the module docstring and adds a
        ``gap_length`` column giving the NaN-run length of each missing hour (<NA>
        where present). No interpolation is done here -- gaps stay NaN so filling can
        be fit per split at training time, avoiding leakage.

        The stats dict records per-station provenance counts (all ints unless noted):
            raw_rows               - rows received.
            nonpositive_dropped    - rows removed for pm25 <= 0.
            over_max_dropped       - rows removed for pm25 > PM25_MAX (sensor fault).
            duplicate_rows         - duplicate station-hours collapsed by mean.
            active_range_hours     - hours from first to last observation (grid size).
            gap_hours_inserted     - NaN hours added by the hourly reindex.
            is_reference           - bool; whether the flatline rule was skipped.
            flatline_hours_removed - hours nulled by the flatline rule (0 if reference).
            gap_interior_dropped   - interior hours removed from gaps > 2*GAP_EDGE_KEEP.
            short_gap_count        - missing hours in gaps <= SHORT_GAP_MAX_H (post-trim).
            long_gap_count         - missing hours in longer gaps (post-trim; <= 2*edge).
            final_rows             - rows out (active_range_hours minus gap_interior_dropped).
            final_valid            - non-null pm25 rows out.
    """
    stats = {"raw_rows": len(df)}
    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True)

    before = len(df)
    df = df[df[PM25_COL] > 0]
    stats["nonpositive_dropped"] = before - len(df)

    before = len(df)
    df = df[df[PM25_COL] <= PM25_MAX]
    stats["over_max_dropped"] = before - len(df)

    passthrough = [c for c in PASSTHROUGH_COLS if c in df.columns]
    vals = {c: df[c].iloc[0] for c in passthrough} if len(df) else {}

    before = len(df)
    df = df.groupby([KEY_COL, TIME_COL], as_index=False).agg(
        {PM25_COL: "mean", **{c: "first" for c in passthrough}})
    stats["duplicate_rows"] = before - len(df)

    if df.empty:
        stats.update(active_range_hours=0, gap_hours_inserted=0, is_reference=True,
                     flatline_hours_removed=0, gap_interior_dropped=0,
                     short_gap_count=0, long_gap_count=0, final_rows=0, final_valid=0)
        cols = [KEY_COL, TIME_COL, PM25_COL, "gap_length"] + passthrough
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

    before_trim = len(g)
    g = _trim_long_gaps(g, PM25_COL)
    stats["gap_interior_dropped"] = before_trim - len(g)

    gap_length = _gap_length(g[PM25_COL])
    g["gap_length"] = gap_length
    stats["short_gap_count"] = int((gap_length <= SHORT_GAP_MAX_H).sum())
    stats["long_gap_count"] = int((gap_length > SHORT_GAP_MAX_H).sum())

    g[KEY_COL] = key
    for c, v in vals.items():
        g[c] = v
    stats["final_rows"] = len(g)
    stats["final_valid"] = int(g[PM25_COL].notna().sum())
    return g, stats