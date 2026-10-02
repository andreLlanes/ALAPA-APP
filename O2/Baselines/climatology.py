#!/usr/bin/env python3
r"""Climatology baseline for PM2.5 forecasting (Section 4.8.3, Eq. 4.20).

For each station, the prediction for a target hour is the training-period mean
PM2.5 at that hour of day,

    yhat_{t+l} = ybar_{h(t+l)},

so it carries no information about current conditions and does not decay with
the horizon. It is fitted and scored once per evaluation period, using the dates
frozen in O2/common/split_dates.json (see O2/common/splits.py):

    main      - the 70/15/15 chronological split. Fitted on the hours the training
                partition covers, scored on the test partition's windows.
    ro_1..4   - the rolling origin. Each fitted on its block's training hours and
                scored on its three-month test block.

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
    * Eligible stations come from the frozen fold file (O2/Folds/loso_folds.json)
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

    python O2/Baselines/climatology.py --city "Metro Manila"
    python O2/Baselines/climatology.py --city all --periods main
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import psycopg2.extras

_HERE = os.path.dirname(os.path.abspath(__file__))   # O2/Baselines
_O2_ROOT = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_O2_ROOT)
for _p in (os.path.join(_REPO_ROOT, "O1", "Builders"),  # common_build also adds O1/common
           os.path.join(_O2_ROOT, "common")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common_build import (  # noqa: E402  (path set just above)
    ALL_CITIES, CITY_SLUG, get_conn, load_station, station_keys,
)
from schema import (  # noqa: E402
    HORIZON_H, LOOKBACK_H, PM25_COL, TIME_COL, exclusion_window_starts,
)
from metrics import METRIC_NAMES, all_metrics  # noqa: E402
from splits import (  # noqa: E402
    frozen_fixed_cuts, frozen_rolling_windows, partition_mask, rolling_mask,
)

CLIMATOLOGY_TABLE = "openaq.climatology_v2"
OUTPUT_ROOT = os.path.join(_O2_ROOT, "Outputs", "climatology")
FOLD_FILE = os.path.join(_O2_ROOT, "Folds", "loso_folds.json")

MIN_OBS_PER_HOUR = 10       # measured training hours needed at each hour of day
MIN_COMPLETENESS = 0.70     # fallback only, used until the folds are frozen


def evaluation_periods(city, which=("main", "rolling")):
    """Return the periods to fit and score, from the city's frozen split dates.

    Each period carries ``fit_end`` (climatology is fitted on measured hours up to
    and including it: the hours the training windows cover) and ``test_mask``, a
    function selecting the test windows from an array of forecast origins.
    """
    periods = []
    if "main" in which:
        cuts = frozen_fixed_cuts(city)
        periods.append({"period": "main", "fit_end": cuts[0],
                        "test_label": f"after {str(cuts[1])[:16]}",
                        "test_mask": lambda o, c=cuts: partition_mask(o, c, "test")})
    if "rolling" in which:
        for w in frozen_rolling_windows(city):
            periods.append({"period": f"ro_{w['origin']}", "fit_end": w["val_start"],
                            "test_label": f"{str(w['test_start'])[:10]}..{str(w['test_end'])[:10]}",
                            "test_mask": lambda o, w=w: rolling_mask(o, w, "test")})
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


def fit_lookup(hours, values, measured, times, fit_end):
    """Return the 24-value hour-of-day lookup and per-hour counts, or (None, counts).

    Fitted on measured hours up to ``fit_end`` (naive UTC datetime64, like
    ``times``). A station missing any hour of day (fewer than MIN_OBS_PER_HOUR
    observations) gets no lookup, so it is never scored on a partial set of hours.
    """
    use = measured & (times <= fit_end)
    counts = np.bincount(hours[use], minlength=24)
    if (counts < MIN_OBS_PER_HOUR).any():
        return None, counts
    sums = np.bincount(hours[use], weights=values[use], minlength=24)
    return sums / counts, counts


def score_station(location_key, g, starts, lookup, period):
    """Score one station's test windows; return (overall row, per-lead rows, n)."""
    times = g[TIME_COL].dt.tz_convert(None).to_numpy()   # naive UTC datetime64
    origins = times[starts + LOOKBACK_H - 1]
    sel = starts[period["test_mask"](origins)]
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


def run_city(city, which, min_completeness=MIN_COMPLETENESS, persist=True, fold_file=FOLD_FILE):
    """Fit and score climatology for every requested period of one city."""
    periods = evaluation_periods(city, which)
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
        times = g[TIME_COL].dt.tz_convert(None).to_numpy()   # naive UTC datetime64
        starts = exclusion_window_starts(g)

        for p in periods:
            res = results[p["period"]]
            lookup, counts = fit_lookup(hours, values, measured, times, p["fit_end"])
            if lookup is None:
                res["skipped"] += 1
                continue
            row, leads, n = score_station(key, g, starts, lookup, p)
            if row is None:
                res["skipped"] += 1
                continue
            res["stations"].append(row)
            res["leads"].extend(leads)
            fit_end = pd.Timestamp(p["fit_end"], tz="UTC").to_pydatetime()
            res["lookups"].extend(
                (p["period"], city, key, h, float(lookup[h]), int(counts[h]), fit_end)
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
            "fit_end": str(p["fit_end"]), "test": p["test_label"],
            "split_dates": "O2/common/split_dates.json", "station_source": source,
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
        line = (f"  {pid:<5} test {p['test_label']:<22}  "
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
        run_city(city, which, args.min_completeness,
                 persist=not args.no_persist, fold_file=args.fold_file)


if __name__ == "__main__":
    main()
