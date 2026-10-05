""" Climatology baseline (Section 4.8.3, Eq. 4.20): for each station, the prediction for a target
    hour is the training-period mean PM2.5 at that local hour of day,
        yhat_{t+l} = ybar_{h(t+l)},
    so it carries no information about current conditions and does not decay with the horizon.
    Fitted on the measured hours covered by the run's training windows and scored on exactly the
    test windows the models see, with metrics per station averaged with equal weight.
    A station without config.MIN_OBS_PER_HOUR training hours at every hour of day falls back to the
    city-wide lookup, so climatology covers the same stations as the models.
    PYTHONPATH=.. python -m evals.climatology --city mm --longest-gap 6 --completeness 70
"""

import argparse

import numpy as np
import pandas as pd

import config
from Common.metrics import all_metrics
from Common.schema import DECODER_COLS, HORIZON_H
from Common.splits import TEST, TRAIN
from data.database import CITIES
from data.load import HORIZON, covered_rows, load_city
from utils.artifacts import ROOT, lead_curve, mark_scored, save_json, seen_stations, settings_tag, summarize

BASELINE = "climatology"

def local_hour(feat_raw) -> np.ndarray:
    """ Recover each row's local hour of day (0-23) from its hour_sin/hour_cos encoding.
    """
    sin = feat_raw[:, DECODER_COLS.index("hour_sin")]
    cos = feat_raw[:, DECODER_COLS.index("hour_cos")]
    return np.rint(np.arctan2(sin, cos) * 24 / (2 * np.pi)).astype(int) % 24

def fit_lookup(hours, values):
    """ Return the 24-value hour-of-day lookup and per-hour counts, or (None, counts).
        A station missing any hour of day (fewer than config.MIN_OBS_PER_HOUR observations) gets no
        lookup, so it is never scored on a partial set of hours.
    """
    counts = np.bincount(hours, minlength=24)
    if (counts < config.MIN_OBS_PER_HOUR).any():
        return None, counts
    sums = np.bincount(hours, weights=values, minlength=24)
    return sums / counts, counts

def score_station(location_key, pred, y):
    """ Score one station's test windows; return (overall row, per-lead rows).
    """
    row = {"location_key": location_key, "n_windows": int(len(y))}
    row.update(all_metrics(pred, y))
    leads = []
    for lead in range(HORIZON_H):
        lr = {"location_key": location_key, "lead": lead + 1}
        lr.update(all_metrics(pred[:, lead], y[:, lead]))
        leads.append(lr)
    return row, leads

def run(city, longest_gap=config.DEFAULT_LONGEST_GAP, completeness=config.DEFAULT_COMPLETENESS):
    """ Fit and score climatology for one city under the given filters.
    """
    data = load_city(city, longest_gap, completeness)
    raw_pm = data.stats.inverse_y(data.pm)
    hours = local_hour(data.feat * data.stats.feat_std + data.stats.feat_mean)
    fit_rows = covered_rows(data.start[data.split_window_ids(TRAIN)], len(raw_pm)) & ~np.isnan(raw_pm)
    ids = data.split_window_ids(TEST)
    target_rows = data.start[ids][:, None] + HORIZON
    seen = seen_stations(data)
    # City-wide lookup from every station's training hours, for stations that cannot fit their own.
    city_lookup, city_counts = fit_lookup(hours[fit_rows], raw_pm[fit_rows])
    if city_lookup is None:
        raise ValueError(f"{city}: fewer than {config.MIN_OBS_PER_HOUR} training hours at some hour of day")

    stations, leads, lookups, fallback = [], [], [], 0
    for s in np.unique(data.station[ids]):
        rows = fit_rows & (data.row_station == s)
        lookup, counts = fit_lookup(hours[rows], raw_pm[rows])
        if lookup is None:
            lookup, counts, fallback = city_lookup, city_counts, fallback + 1
        m = data.station[ids] == s
        row, station_leads = score_station(str(data.keys[s]), lookup[hours[target_rows[m]]],
                                           raw_pm[target_rows[m]])
        row.update(seen=bool(seen[s]), lookup="city" if lookup is city_lookup else "station")
        stations.append(row)
        leads.extend(station_leads)
        lookups += [{"location_key": str(data.keys[s]), "hour_of_day": h, "lookup": row["lookup"],
                     "climatology_pm25": float(lookup[h]), "n_obs": int(counts[h])} for h in range(24)]

    print(f"[CLIMATOLOGY] {CITIES[city]} (longest gap: {longest_gap}, "
          f"station completeness: {completeness})")
    outdir = ROOT / "baselines" / BASELINE / city / settings_tag(longest_gap, completeness) / "test"
    outdir.mkdir(parents=True, exist_ok=True)
    stations, leads = mark_scored(pd.DataFrame(stations)), pd.DataFrame(leads)
    curve = lead_curve(leads, stations)
    stations.to_csv(outdir / "stations.csv", index=False)
    leads.to_csv(outdir / "by_lead.csv", index=False)
    curve.to_csv(outdir / "lead_curve.csv", index=False)
    pd.DataFrame(lookups).to_csv(outdir / "lookups.csv", index=False)
    s = {**summarize(stations, curve), "stations_city_lookup": fallback}
    save_json(outdir / "metrics.json", s)

    print(f"    {s['n_stations']} stations scored ({fallback} on the city-wide lookup), "
          f"{s['n_windows']} test windows")
    print(f"    RMSE {s['rmse']:.3f} +/- {s['rmse_sd']:.3f} | MAE {s['mae']:.3f} | "
          f"MBE {s['mbe']:+.3f} | IOA {s['ioa']:.3f}")
    return s

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=list(CITIES))
    ap.add_argument("--longest-gap", type=int, default=config.DEFAULT_LONGEST_GAP,
                    choices=config.LONGEST_GAP_CHOICES)
    ap.add_argument("--completeness", type=int, default=config.DEFAULT_COMPLETENESS,
                    choices=config.COMPLETENESS_CHOICES)
    a = ap.parse_args()
    run(a.city, a.longest_gap, a.completeness)
