""" Transfer gain against source distance, per target station: the RMSE reduction of each transfer
    variant over the same model trained on the target alone, and the MMD^2 between the station's
    training hours and the pooled source cities' (same kernel as the GBT source weights). Reports
    the Spearman correlation between distance and gain.
"""

import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config
from data.load import load_city
from evals.eval import KEYS, station_rmse
from models.transfer import station_distances
from utils.artifacts import ROOT, load_json, save_json

def distances(source: str, city: str, filters: str) -> pd.Series:
    """ Return each target station's MMD^2 to the pooled source cities (cached).
    """
    path = ROOT / "distances" / f"{source}-{city}_{filters}.json"
    saved = load_json(path)
    if saved is None:
        gap, completeness = map(int, re.findall(r"\d+", filters))
        stations = load_city(city, gap, completeness)
        found = station_distances(stations, load_city(source.split("+"), gap, completeness), {})
        saved = dict(zip(stations.keys[found["station"]].tolist(),
                         found[config.DISTANCE_COLUMN].tolist()))
        save_json(path, saved)
    return pd.Series(saved)

def table(runs: pd.DataFrame) -> tuple:
    """ Return (one row per transfer configuration and target station, one row per configuration
        with its Spearman correlation).
    """
    transfers = runs[runs["source"] != ""][KEYS + ["filters"]].drop_duplicates()
    rmse, stations, summary = station_rmse(runs), [], []
    for row in transfers.itertuples(index=False):
        key = row._asdict()
        plain = rmse.get(tuple({"source": "", "variant": "none"}.get(k, key[k]) for k in KEYS))
        if plain is None:
            continue
        found = pd.concat({"mmd2": distances(key["source"], key["city"], key["filters"]),
                           "rmse_plain": plain,
                           "rmse_transfer": rmse[tuple(key[k] for k in KEYS)]}, axis=1, join="inner")
        found["gain"] = (found["rmse_plain"] - found["rmse_transfer"]) / found["rmse_plain"]
        rho, p = spearmanr(found["mmd2"], found["gain"]) if len(found) > 2 else (np.nan, np.nan)
        label = {k: key[k] for k in KEYS}
        stations.append(found.rename_axis("location_key").reset_index().assign(**label))
        summary.append({**label, "stations": len(found), "spearman": rho, "p": p})
    return (pd.concat(stations, ignore_index=True) if stations else pd.DataFrame(),
            pd.DataFrame(summary))
