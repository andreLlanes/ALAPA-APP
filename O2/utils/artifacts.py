""" Save each run's outputs: test predictions and per-station, per-lead and station-averaged scores.
    Layout: artifacts/<model>/<city or source-city>/<ar|direct>_<settings>/<variant>/seed<seed>/
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
from Common.metrics import METRIC_NAMES, all_metrics
from Common.schema import HORIZON_H
from Common.splits import TRAIN

ROOT = Path(__file__).resolve().parents[1] / "artifacts"

def settings_tag(longest_gap, completeness, ablation=100, block=0) -> str:
    """ Name the data settings of a run, e.g. gap6_comp70_abl25_block1.
    """
    tag = f"gap{longest_gap}_comp{completeness}_abl{ablation}"
    return tag + (f"_block{block}" if ablation < 100 else "")

def config_dir(model, city, source, autoregressive, tag) -> Path:
    """ Return the folder of one configuration (all its variants and seeds).
    """
    pair = f"{source}-{city}" if source else city
    return ROOT / model / pair / f"{'ar' if autoregressive else 'direct'}_{tag}"

def save_json(path, obj):
    """ Write JSON, creating the folder first.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)

def load_json(path):
    """ Read JSON, or return None when the file does not exist.
    """
    if not Path(path).exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def seen_stations(data) -> np.ndarray:
    """ Mark the stations that have at least one training window in this run.
    """
    seen = np.zeros(len(data.keys), bool)
    seen[np.unique(data.station[data.split_window_ids(TRAIN)])] = True
    return seen

def station_scores(keys, station, pred, obs, seen):
    """ Score every station overall and per lead, then average stations with equal weight
        (Section 4.8.1), also separately over seen and unseen stations.
        Returns (stations, by_lead, lead_curve, summary).
    """
    rows, leads = [], []
    for s in np.unique(station):
        m = station == s
        rows.append({"location_key": str(keys[s]), "seen": bool(seen[s]),
                     "n_windows": int(m.sum()), **all_metrics(pred[m], obs[m])})
        leads += [{"location_key": str(keys[s]), "lead": h + 1,
                   **all_metrics(pred[m, h], obs[m, h])} for h in range(HORIZON_H)]
    stations = mark_scored(pd.DataFrame(rows))
    by_lead = pd.DataFrame(leads)
    curve = lead_curve(by_lead, stations)
    return stations, by_lead, curve, summarize(stations, curve)

def mark_scored(stations: pd.DataFrame) -> pd.DataFrame:
    """ Flag the stations with at least MIN_TEST_WINDOWS test windows; only they count in averages.
    """
    stations["scored"] = stations["n_windows"] >= config.MIN_TEST_WINDOWS
    return stations

def lead_curve(by_lead: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """ Average every lead over the scored stations, each station weighted equally.
    """
    rows = by_lead[by_lead["location_key"].isin(stations.loc[stations["scored"], "location_key"])]
    curve = rows.groupby("lead")[list(METRIC_NAMES)].mean().reset_index()
    curve.insert(1, "n_stations", rows.groupby("lead")["location_key"].nunique().to_numpy())
    return curve

def summarize(stations, curve) -> dict:
    """ Station-averaged metrics over the scored stations: overall, seen and unseen, lead 1 and 72.
    """
    scored = stations[stations["scored"]]
    average = lambda f: {m: float(f[m].mean()) for m in METRIC_NAMES} if len(f) else {}
    edge = lambda i: float(curve["rmse"].iloc[i]) if len(curve) else float("nan")
    return {"n_stations": int(len(scored)), "n_stations_unscored": int(len(stations) - len(scored)),
            "n_windows": int(scored["n_windows"].sum()),
            **average(scored), "rmse_sd": float(scored["rmse"].std()),
            "seen": average(scored[scored["seen"]]),
            "unseen": average(scored[~scored["seen"]]),
            "rmse_lead1": edge(0), f"rmse_lead{HORIZON_H}": edge(-1)}

def save_scores(directory, data, ids, pred):
    """ Save one run's test predictions (ug/m3) and its scores; return the summary.
        pred is normalized, in the order of ids.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pred = data.stats.inverse_y(pred)
    obs = data.stats.inverse_y(data.gather(ids)[2])
    np.savez_compressed(directory / "predictions.npz", location_key=data.keys[data.station[ids]],
                        origin=data.origin[ids], pred=pred.astype(np.float32),
                        obs=obs.astype(np.float32))
    stations, by_lead, curve, summary = station_scores(data.keys, data.station[ids], pred, obs,
                                                       seen_stations(data))
    stations.to_csv(directory / "stations.csv", index=False)
    by_lead.to_csv(directory / "by_lead.csv", index=False)
    curve.to_csv(directory / "lead_curve.csv", index=False)
    save_json(directory / "metrics.json", summary)
    return summary
