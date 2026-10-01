#!/usr/bin/env python3
r"""Climatology baseline for PM2.5 forecasting (Section 4.8.3, Eq. 4.20).

For each station, the prediction for a target hour is the training-period mean
PM2.5 at that hour of day,

    yhat_{t+l} = ybar_{h(t+l)},

so it carries no information about current conditions and does not decay with
the horizon. It is fitted and scored once per evaluation period:

    main      - the 70/15/15 chronological split. Fitted on the training period,
                scored on windows whose forecast origin lies in the test period.
    ro_1..4   - the rolling origin: four three-month test blocks anchored to the
                end of the record, each fitted on everything before its block.

Scoring rules, shared with the models so the skill score of Eq. 4.21 compares
like with like:
    * Windows come from schema.exclusion_window_starts, the same rule the LSTM
      builder uses, so climatology is scored on exactly the windows the models see.
    * Interpolated hours (short_gap_filled) are never used: the hourly means are
      fitted on measured hours only, and filled target hours are not scored.
    * Metrics come from O2/common/metrics.py. They are computed per station (one
      LOSO fold each), then averaged with every station weighted equally
      (Section 4.8.1), both overall and at each lead time.
    * A station is scored in a period only if it has training-period data for
      every hour of day (at least MIN_OBS_PER_HOUR each) and test windows.
    * Eligible stations come from the frozen fold file (O2/folds/loso_folds.json)
      when it exists, so every script uses the same stations; until then they are
      selected here by measured-hour completeness.

Outputs, under O2/Outputs/climatology/<citySlug>/<period>/:
    stations.csv    - one row per scored station: overall metrics.
    by_lead.csv     - one row per (station, lead).
    lead_curve.csv  - station-averaged metrics per lead, for the skill curves.
    summary.json    - the period's dates, counts and station-averaged metrics,
                      plus the 20-fold average once the folds are frozen.
The fitted lookups are stored in Postgres (CLIMATOLOGY_TABLE), one version per
(period, city), so a rerun replaces only its own rows.

    python O2/climatology_baseline.py --city "Metro Manila"
    python O2/climatology_baseline.py --city all --periods main
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import psycopg2.extras

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_REPO_ROOT, "O1", "Builders"),  # common_build also adds O1/common
           os.path.join(_HERE, "common")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common_build import (  # noqa: E402  (path set just above)
    ALL_CITIES, CITY_SLUG, get_conn, load_station, station_keys,
)
from schema import (  # noqa: E402
    HORIZON_H, LOOKBACK_H, MERGED_TABLE, PM25_COL, TIME_COL, exclusion_window_starts,
)
from metrics import METRIC_NAMES, all_metrics  # noqa: E402

CLIMATOLOGY_TABLE = "openaq.climatology_v2"
OUTPUT_ROOT = os.path.join(_HERE, "Outputs", "climatology")
FOLD_FILE = os.path.join(_HERE, "folds", "loso_folds.json")

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
ROLLING_ORIGINS = 4
ROLLING_TEST_MONTHS = 3
MIN_OBS_PER_HOUR = 10       # measured training hours needed at each hour of day
MIN_COMPLETENESS = 0.90     # fallback only, used until the folds are frozen


def record_span(city):
    """Return the first and last timestamp in a city's merged record."""
    with get_conn().cursor() as cur:
        cur.execute(f"SELECT MIN({TIME_COL}), MAX({TIME_COL}) FROM {MERGED_TABLE} "
                    f"WHERE city = %s", (city,))
        first, last = cur.fetchone()
    if first is None:
        raise ValueError(f"No rows for city={city!r} in {MERGED_TABLE}.")
    return pd.Timestamp(first), pd.Timestamp(last)


def evaluation_periods(city, which=("main", "rolling"), train_end=None, test_start=None):
    """Return the periods to fit and score, as dicts with half-open date bounds.

    The main split cuts calendar time 70/15/15 unless explicit dates are given,
    which is how frozen cut dates are passed in. Rolling-origin blocks are laid
    back from the end of the record, ROLLING_TEST_MONTHS each.
    """
    first, last = record_span(city)
    end = last + pd.Timedelta(hours=1)
    span = end - first
    periods = []
    if "main" in which:
        t_end = pd.Timestamp(train_end, tz="UTC") if train_end else (first + span * TRAIN_FRAC).floor("h")
        t_start = (pd.Timestamp(test_start, tz="UTC") if test_start
                   else (first + span * (TRAIN_FRAC + VAL_FRAC)).floor("h"))
        periods.append({"period": "main", "train_end": t_end,
                        "test_start": t_start, "test_end": end})
    if "rolling" in which:
        edges = [end - pd.DateOffset(months=ROLLING_TEST_MONTHS * (ROLLING_ORIGINS - k))
                 for k in range(ROLLING_ORIGINS + 1)]
        for k in range(ROLLING_ORIGINS):
            periods.append({"period": f"ro_{k + 1}", "train_end": edges[k],
                            "test_start": edges[k], "test_end": edges[k + 1]})
    return periods


def load_fold_file(path=FOLD_FILE):
    """Return (eligible keys, fold keys) from the frozen folds, or (None, None)."""
    if not os.path.exists(path):
        return None, None
    with open(path, encoding="utf-8") as f:
        record = json.load(f)
    eligible = {s["location_key"] for s in record["eligible_stations"]}
    folds = {s["location_key"] for s in record["folds"]}
    return eligible, folds


def measured_completeness(g):
    """Measured (not interpolated) hours over the station's active range."""
    times = g[TIME_COL]
    active = (times.max() - times.min()) / pd.Timedelta(hours=1) + 1
    measured = (g[PM25_COL].notna() & ~g["short_gap_filled"].fillna(False).astype(bool)).sum()
    return float(measured / active) if active > 0 else 0.0


def fit_lookup(hours, values, measured, times, train_end):
    """Return the 24-value hour-of-day lookup and per-hour counts, or (None, counts).

    Fitted on measured training hours only. A station missing any hour of day
    (fewer than MIN_OBS_PER_HOUR observations) gets no lookup, so it is never
    scored on a partial set of hours.
    """
    use = measured & (times < train_end)
    counts = np.bincount(hours[use], minlength=24)
    if (counts < MIN_OBS_PER_HOUR).any():
        return None, counts
    sums = np.bincount(hours[use], weights=values[use], minlength=24)
    return sums / counts, counts


def score_station(location_key, g, starts, lookup, period):
    """Score one station's test windows; return (overall row, per-lead rows, n)."""
    times = g[TIME_COL].dt.tz_convert(None).to_numpy()   # naive UTC datetime64
    origins = times[starts + LOOKBACK_H - 1]
    lo = np.datetime64(period["test_start"].tz_convert(None))
    hi = np.datetime64(period["test_end"].tz_convert(None))
    sel = starts[(origins >= lo) & (origins < hi)]
    if sel.size == 0:
        return None, [], 0

    idx = sel[:, None] + LOOKBACK_H + np.arange(HORIZON_H)   # target rows per window
    y = g[PM25_COL].to_numpy(dtype=float)[idx]
    filled = g["short_gap_filled"].fillna(False).to_numpy(dtype=bool)[idx]
    y[filled] = np.nan                                        # never score a filled hour
    pred = lookup[g["hour"].to_numpy()[idx]]

    row = {"location_key": location_key, "n_windows": int(sel.size),
           "filled_targets_skipped": int(filled.sum())}
    row.update(all_metrics(pred, y))
    leads = []
    for lead in range(HORIZON_H):
        lr = {"location_key": location_key, "lead": lead + 1}
        lr.update(all_metrics(pred[:, lead], y[:, lead]))
        leads.append(lr)
    return row, leads, int(sel.size)


def persist_lookups(rows, city, period_id):
    """Replace this (period, city)'s lookups in CLIMATOLOGY_TABLE; others are kept."""
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {CLIMATOLOGY_TABLE} (
                period text NOT NULL,
                city text NOT NULL,
                location_key text NOT NULL,
                hour_of_day integer NOT NULL,
                climatology_pm25 double precision NOT NULL,
                n_obs integer NOT NULL,
                train_end timestamptz NOT NULL,
                PRIMARY KEY (period, city, location_key, hour_of_day)
            )""")
        cur.execute(f"DELETE FROM {CLIMATOLOGY_TABLE} WHERE period = %s AND city = %s",
                    (period_id, city))
        if rows:
            psycopg2.extras.execute_values(
                cur,
                f"INSERT INTO {CLIMATOLOGY_TABLE} (period, city, location_key, hour_of_day, "
                f"climatology_pm25, n_obs, train_end) VALUES %s",
                rows)
    conn.commit()


def station_average(frame):
    """Average metrics across stations, each station weighted equally."""
    return {m: float(frame[m].mean()) for m in METRIC_NAMES} if len(frame) else {}


def run_city(city, which, train_end=None, test_start=None, min_completeness=MIN_COMPLETENESS,
             persist=True, fold_file=FOLD_FILE):
    """Fit and score climatology for every requested period of one city."""
    periods = evaluation_periods(city, which, train_end, test_start)
    eligible, folds = load_fold_file(fold_file)
    source = "fold file" if eligible is not None else f"completeness >= {min_completeness:.0%}"

    results = {p["period"]: {"stations": [], "leads": [], "lookups": [], "skipped": 0}
               for p in periods}
    n_considered = 0
    for key in station_keys("clean", city):
        if eligible is not None and key not in eligible:
            continue
        g = load_station("clean", key)
        if g.empty:
            continue
        g = g.sort_values(TIME_COL).reset_index(drop=True)
        if eligible is None and measured_completeness(g) < min_completeness:
            continue
        n_considered += 1

        g["hour"] = g[TIME_COL].dt.hour
        hours = g["hour"].to_numpy()
        values = g[PM25_COL].to_numpy(dtype=float)
        measured = ~np.isnan(values) & ~g["short_gap_filled"].fillna(False).to_numpy(dtype=bool)
        times = g[TIME_COL]
        starts = exclusion_window_starts(g)

        for p in periods:
            res = results[p["period"]]
            lookup, counts = fit_lookup(hours, values, measured, times, p["train_end"])
            if lookup is None:
                res["skipped"] += 1
                continue
            row, leads, n = score_station(key, g, starts, lookup, p)
            if row is None:
                res["skipped"] += 1
                continue
            res["stations"].append(row)
            res["leads"].extend(leads)
            res["lookups"].extend(
                (p["period"], city, key, h, float(lookup[h]), int(counts[h]), p["train_end"])
                for h in range(24))

    slug = CITY_SLUG.get(city, city.replace(" ", "-"))
    print(f"[climatology] {city}: {n_considered} stations considered ({source})")
    for p in periods:
        pid = p["period"]
        res = results[pid]
        outdir = os.path.join(OUTPUT_ROOT, slug, pid)
        os.makedirs(outdir, exist_ok=True)
        stations = pd.DataFrame(res["stations"])
        leads = pd.DataFrame(res["leads"])
        summary = {
            "city": city, "period": pid,
            "train_end": str(p["train_end"]), "test_start": str(p["test_start"]),
            "test_end": str(p["test_end"]), "station_source": source,
            "stations_scored": int(len(stations)), "stations_skipped": res["skipped"],
            "windows": int(stations["n_windows"].sum()) if len(stations) else 0,
            "station_average": station_average(stations),
        }
        if folds is not None and len(stations):
            in_folds = stations[stations["location_key"].isin(folds)]
            summary["fold_stations_scored"] = int(len(in_folds))
            summary["fold_average"] = station_average(in_folds)
        if len(stations):
            stations.to_csv(os.path.join(outdir, "stations.csv"), index=False)
            leads.to_csv(os.path.join(outdir, "by_lead.csv"), index=False)
            curve = leads.groupby("lead")[list(METRIC_NAMES)].mean().reset_index()
            curve.insert(1, "n_stations", leads.groupby("lead")["location_key"].nunique().to_numpy())
            curve.to_csv(os.path.join(outdir, "lead_curve.csv"), index=False)
        with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        if persist:
            persist_lookups(res["lookups"], city, pid)

        avg = summary["station_average"]
        line = (f"  {pid:<5} test {p['test_start']:%Y-%m-%d}..{p['test_end']:%Y-%m-%d}  "
                f"stations {summary['stations_scored']:3d} (skipped {res['skipped']:2d})  "
                f"windows {summary['windows']:7d}")
        if avg:
            line += (f"  RMSE {avg['rmse']:.2f}  MAE {avg['mae']:.2f}  "
                     f"MBE {avg['mbe']:+.2f}  IOA {avg['ioa']:.3f}")
        print(line)


def parse_args():
    p = argparse.ArgumentParser(description="Fit and score the climatology baseline.")
    p.add_argument("--city", default="Metro Manila", help='city name or "all"')
    p.add_argument("--periods", default="all", choices=["all", "main", "rolling"])
    p.add_argument("--train-end", help="frozen training cutoff for the main split (UTC)")
    p.add_argument("--test-start", help="frozen test start for the main split (UTC)")
    p.add_argument("--min-completeness", type=float, default=MIN_COMPLETENESS,
                   help="used only until the fold file exists")
    p.add_argument("--fold-file", default=FOLD_FILE)
    p.add_argument("--no-persist", action="store_true",
                   help=f"do not write lookups to {CLIMATOLOGY_TABLE}")
    return p.parse_args()


def main():
    args = parse_args()
    which = ("main", "rolling") if args.periods == "all" else (args.periods,)
    cities = ALL_CITIES if args.city == "all" else [args.city]
    for city in cities:
        run_city(city, which, args.train_end, args.test_start, args.min_completeness,
                 persist=not args.no_persist, fold_file=args.fold_file)


if __name__ == "__main__":
    main()
