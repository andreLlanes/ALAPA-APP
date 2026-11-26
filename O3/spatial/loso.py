""" Leave-one-station-out scoring by regression-kriging (train.py --spatial true). A fold withholds
    one Metro Manila station from training, validation and every forecast input; its forecast is
    the regression-kriging of the retained stations' forecasts, scored on its own test windows
    with at least LOSO_MIN_OBSERVED of their forecast hours observed (on the observed hours).
    The forecasts come from every window with a lookback and met data, whether or not its forecast
    hours were observed. The variogram is fitted once per run from its validation forecasts. An
    origin with fewer than MIN_RK_STATIONS forecasting stations is skipped as a whole window; a fold
    counts in the averages when at least MIN_TEST_WINDOWS of its windows are kriged.
"""

import numpy as np
import pandas as pd

import config
from baselines.persistence import last_observed
from Common.metrics import METRIC_NAMES, all_metrics
from Common.schema import HORIZON_H
from Common.splits import TEST, VAL
from data.database import CITIES
from data.load import LOOKBACK, load_city
from spatial import covariates
from spatial.kriging import hour_residuals, krige_origin
from spatial.variogram import Pool, fit
from utils import runlog
from utils.artifacts import (ROOT, lead_curve, mark_scored, save_json,
                             settings_tag, station_scores, summarize, write_tables)

LEADS = np.arange(1, HORIZON_H + 1)

class Geometry:
    """ Grid cells and projected positions (km) of the target city's stations (the last city loaded).
    """

    def __init__(self, data, cov):
        n = len(data.keys)
        city = np.flatnonzero(data.city == data.city[-1])
        x, y, row, col = cov.cells(data.lat[city], data.lon[city])
        self.row, self.col = np.full(n, -1), np.full(n, -1)
        self.xy = np.full((n, 2), np.nan)
        self.row[city], self.col[city], self.xy[city] = row, col, np.column_stack([x, y])

    def dist(self, a, b) -> np.ndarray:
        """ Distances (km) between stations a and stations b.
        """
        return np.linalg.norm(self.xy[a][:, None] - self.xy[b][None], axis=-1)

def hours_of(origin: int) -> np.ndarray:
    """ Target hours (since the epoch, UTC) of the 72 forecast hours of an origin.
    """
    return origin + LEADS

def by_origin(origins: np.ndarray) -> dict:
    """ Group window positions by origin (hours since the epoch).
    """
    order = np.argsort(origins, kind="stable")
    ordered = origins[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    ends = np.r_[starts[1:], len(order)]
    return {int(ordered[s]): order[s:e] for s, e in zip(starts, ends)}

def origin_hours(data, ids) -> np.ndarray:
    """ Origins of windows as hours since the epoch.
    """
    return data.origin[ids].astype("datetime64[h]").astype(np.int64)

def pooled_variogram(data, geo, cov, ids, pred):
    """ Fit the run's variogram from the trend residuals of every validation origin with enough
        stations; return (Variogram or None, candidate fits, hours pooled).
    """
    stations = np.unique(data.station[ids])
    pool = Pool(float(geo.dist(stations, stations).max()) if len(stations) > 1 else 0.0)
    for origin, pos in by_origin(origin_hours(data, ids)).items():
        if len(pos) < config.MIN_RK_STATIONS:
            continue
        s = data.station[ids[pos]]
        e = hour_residuals(cov.at(geo.row[s], geo.col[s], hours_of(origin)), pred[pos].T)
        pool.add(e[np.isfinite(e).all(1)], geo.dist(s, s))
    if pool.hours == 0:
        return None, [], 0
    try:
        variogram, candidates = fit(pool)
    except ValueError:
        return None, [], pool.hours
    return variogram, candidates, pool.hours

def empty_scores(key):
    """ Score tables of a withheld station with no kriged window.
    """
    stations = mark_scored(pd.DataFrame([{"location_key": str(key), "seen": False, "n_windows": 0,
                                          **all_metrics([], [])}]))
    by_lead = pd.DataFrame(columns=["location_key", "lead", "n_pairs", *METRIC_NAMES])
    curve = lead_curve(by_lead, stations)
    return stations, by_lead, curve, summarize(stations, curve)

def score(out, data, val, test, title):
    """ Krige the withheld station's test windows from the retained stations' forecasts and save
        its predictions, scores and kriging record to `out`. val and test are (window ids,
        forecasts [n,72] in ug/m3) of the retained stations.
    """
    cov = covariates.load()
    geo = Geometry(data, cov)
    w = data.withheld
    key = data.keys[w]
    variogram, candidates, pooled = pooled_variogram(data, geo, cov, *val)
    test_ids, test_pred = test
    groups = by_origin(origin_hours(data, test_ids))
    target = data.withheld_window_ids()
    kept, preds, variances, n_stations = [], [], [], []
    dropped = deficient = clipped = no_ndvi = 0
    for t in target:
        origin = int(origin_hours(data, [t])[0])
        pos = groups.get(origin, np.empty(0, int))
        if variogram is None or len(pos) < config.MIN_RK_STATIONS:
            continue
        s = data.station[test_ids[pos]]
        hours = hours_of(origin)
        x = cov.at(geo.row[s], geo.col[s], hours)
        x_target = cov.at(geo.row[[w]], geo.col[[w]], hours)[:, 0]
        no_ndvi += int((~np.isfinite(x[..., 0]).all(1) | ~np.isfinite(x_target[:, 0])).sum())
        pred, var, d, r, c = krige_origin(variogram, x, test_pred[pos].T, geo.dist(s, s),
                                          x_target, geo.dist(s, [w])[:, 0])
        kept.append(t)
        preds.append(pred)
        variances.append(var)
        n_stations.append(len(s))
        dropped, deficient, clipped = dropped + d, deficient + r, clipped + c
    kept = np.asarray(kept, int)
    if len(kept):
        pred = np.stack(preds)
        obs = data.stats.inverse_y(data.gather(kept)[2])
        tables = station_scores(data.keys, np.full(len(kept), w), pred, obs, np.zeros(len(data.keys), bool))
    else:
        pred = obs = np.empty((0, HORIZON_H))
        tables = empty_scores(key)

    out.mkdir(parents=True, exist_ok=True)
    dtype = config.STORED_PREDICTION_DTYPE
    np.savez_compressed(out / "predictions.npz", location_key=np.full(len(kept), key),
                        origin=data.origin[kept], pred=pred.astype(dtype), obs=obs.astype(dtype),
                        variance=np.asarray(variances, np.float32).reshape(len(kept), HORIZON_H))
    record = {"withheld": str(key), "test_windows": int(len(target)), "kriged_windows": int(len(kept)),
              "skipped_windows": int(len(target) - len(kept)),
              "min_rk_stations": config.MIN_RK_STATIONS,
              "stations_per_origin": ({"min": int(min(n_stations)), "median": float(np.median(n_stations)),
                                       "max": int(max(n_stations))} if n_stations else None),
              "validation_hours_pooled": int(pooled),
              "variogram_units": "standardized residuals; range in km",
              "variogram": None if variogram is None else {
                  "model": variogram.model, "nugget": variogram.nugget, "psill": variogram.psill,
                  "range_km": variogram.range},
              "variogram_candidates": candidates,
              "hours_covariate_left_out": dropped, "hours_without_ndvi": no_ndvi,
              "hours_rank_deficient": deficient,
              "hours_set_to_zero": clipped}
    save_json(out / "spatial.json", record)
    write_tables(out, *tables)
    runlog.detail(f"LOSO {title}", record)

    fitted = ("no variogram (too few validation origins with enough stations)" if variogram is None else
              f"{variogram.model} variogram (nugget {variogram.nugget:.3f}, partial sill "
              f"{variogram.psill:.3f}, range {variogram.range:.2f} km)")
    print(f"    LOSO {key}: kriged {len(kept)} of {len(target)} test windows "
          f"(fewer than {config.MIN_RK_STATIONS} stations in {len(target) - len(kept)}); {fitted}")
    if len(kept):
        row, by_lead = tables[0].iloc[0], tables[1].set_index("lead")["rmse"]
        note = "" if row["scored"] else (f" (not counted: fewer than {config.MIN_TEST_WINDOWS} "
                                         f"kriged windows)")
        print(f"    RMSE {row['rmse']:.3f} | MAE {row['mae']:.3f} | MBE {row['mbe']:+.3f} | "
              f"IOA {row['ioa']:.3f} | lead 1h RMSE {by_lead[1]:.3f} -> lead {HORIZON_H}h "
              f"RMSE {by_lead[HORIZON_H]:.3f}{note}")

def fold_folder(fold: dict, longest_gap: int, completeness: int):
    """ Folder of the kriged persistence baseline of one fold.
    """
    return (ROOT / "loso" / "baselines" / "persistence" / config.LOSO_CITY
            / settings_tag(longest_gap, completeness) / f"fold{fold['fold_id']:02d}")

def persistence(fold: dict, longest_gap: int, completeness: int):
    """ Kriged persistence for one fold: each retained station's last observed PM2.5 carried over
        the horizon, then regression-kriged to the withheld station like a model's forecasts.
    """
    out = fold_folder(fold, longest_gap, completeness)
    if (out / "metrics.json").exists():
        return
    title = (f"[PERSISTENCE, KRIGED] {CITIES[config.LOSO_CITY]}, fold {fold['fold_id']} "
             f"({fold['location_key']})")
    print(title)
    data = load_city(config.LOSO_CITY, longest_gap, completeness, withheld=fold["location_key"])
    raw = data.stats.inverse_y(data.pm)
    inputs = []
    for split in (VAL, TEST):
        ids = data.forecast_window_ids(split)
        last, _ = last_observed(raw[data.start[ids][:, None] + LOOKBACK])
        inputs.append((ids, np.repeat(last[:, None], HORIZON_H, axis=1)))
    score(out, data, *inputs, title)
