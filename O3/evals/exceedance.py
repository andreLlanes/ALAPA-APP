""" Forecasts of high PM2.5, per 24-hour forecast block (hours 1-24, 25-48, 49-72). Events: the
    block mean above EXCEEDANCE_PRIMARY and EXCEEDANCE_SENSITIVITY, and the worst hours: an hour
    above the TOP_QUANTILE of the city's training PM2.5. Counts are pooled over the scored
    stations' windows; a block counts when BLOCK_MIN_OBSERVED of its hours are observed.
    Reports POD, FAR, CSI and the base rate, averaged over seeds.
"""

import re
from functools import cache

import numpy as np
import pandas as pd

import config
from data.load import load_city
from models.transfer import training_rows
from utils.artifacts import station_codes

@cache
def training_quantile(city: str, filters: str) -> float:
    """ Return the TOP_QUANTILE of the city's PM2.5 over its training rows.
    """
    gap, completeness = map(int, re.findall(r"\d+", filters))
    return float(np.quantile(training_rows(load_city(city, gap, completeness))[0][:, 0],
                             config.TOP_QUANTILE))

def blocks(values) -> tuple:
    """ Split [n,72] values into forecast blocks [n,B,24]; return their block means [n,B] (NaN
        when too few hours are observed) and the hours.
    """
    hours = values.reshape(len(values), -1, config.BLOCK_HOURS)
    seen = ~np.isnan(hours)
    mean = np.where(seen, hours, 0).sum(2) / np.maximum(seen.sum(2), 1)
    return np.where(seen.mean(2) >= config.BLOCK_MIN_OBSERVED, mean, np.nan), hours

def counts(event_pred, event_obs, valid) -> np.ndarray:
    """ Count hits, misses and false alarms per block over the valid cases ([n,B,k] arrays).
    """
    hit, miss, alarm = (event_pred & event_obs, ~event_pred & event_obs, event_pred & ~event_obs)
    return np.stack([(x & valid).sum((0, 2)) for x in (hit, miss, alarm)] + [valid.sum((0, 2))])

def skill(c) -> dict:
    """ Return POD, FAR, CSI and the base rate from counts [hits, misses, alarms, cases].
    """
    hit, miss, alarm, total = c
    ratio = lambda a, b: np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 0)
    return {"pod": ratio(hit, hit + miss), "far": ratio(alarm, hit + alarm),
            "csi": ratio(hit, hit + miss + alarm), "base_rate": ratio(hit + miss, total)}

def rows(key: dict, group: pd.DataFrame, g: dict) -> list:
    """ Return one row per event and block for one configuration.
    """
    codes, scored = station_codes(g["keys"])
    keep = scored[codes]
    obs_mean, obs_hours = blocks(g["obs"][keep])
    top = training_quantile(key["city"], group["filters"].iloc[0])
    events = [(f"above_{t}", t, "block") for t in (config.EXCEEDANCE_PRIMARY,
                                                   config.EXCEEDANCE_SENSITIVITY)]
    events.append(("top", top, "hour"))
    out = []
    for name, threshold, unit in events:
        per_seed = []
        for pred in g["pred"]:
            pred_mean, pred_hours = blocks(pred[keep].astype(np.float32))
            if unit == "block":
                valid = (~np.isnan(obs_mean) & ~np.isnan(pred_mean))[..., None]
                c = counts((pred_mean > threshold)[..., None], (obs_mean > threshold)[..., None], valid)
            else:
                valid = ~np.isnan(obs_hours) & ~np.isnan(pred_hours)
                c = counts(pred_hours > threshold, obs_hours > threshold, valid)
            per_seed.append((skill(c), c[3]))
        mean = {k: np.mean([skills[k] for skills, _ in per_seed], axis=0) for k in per_seed[0][0]}
        out += [{**key, "event": name, "threshold": threshold, "block": b + 1,
                 "cases": per_seed[0][1][b], **{k: v[b] for k, v in mean.items()}}
                for b in range(len(mean["pod"]))]
    return out
