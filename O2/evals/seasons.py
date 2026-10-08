""" Error by season and by month of the forecast origin: RMSE and MAE averaged over the scored
    stations (each weighted equally within the period), with the number of windows, averaged over
    seeds. Seasons per city are in config.SEASONS.
"""

import numpy as np
import pandas as pd

import config
from utils.artifacts import station_codes

def rows(key: dict, group: pd.DataFrame, g: dict) -> list:
    """ Return one row per month and per season for one configuration.
    """
    codes, scored = station_codes(g["keys"])
    month = g["origin"].astype("datetime64[M]").astype(int) % 12 + 1
    season_of = {m: name for name, months in config.SEASONS[key["city"]].items() for m in months}
    periods = {"month": month, "season": np.array([season_of[m] for m in range(1, 13)])[month - 1]}
    tables = []
    for pred in g["pred"]:
        error = pred.astype(np.float32) - g["obs"]
        seen = ~np.isnan(error)
        error = np.where(seen, error, 0)
        windows = pd.DataFrame({"station": codes, "squared": np.square(error).sum(1),
                                "absolute": np.abs(error).sum(1), "hours": seen.sum(1)})
        for kind, label in periods.items():
            sums = windows.assign(period=label).groupby(["period", "station"]).agg(
                squared=("squared", "sum"), absolute=("absolute", "sum"), hours=("hours", "sum"),
                windows=("hours", "size"))
            sums = sums[scored[sums.index.get_level_values("station")]]
            sums["rmse"] = np.sqrt(sums["squared"] / sums["hours"])
            sums["mae"] = sums["absolute"] / sums["hours"]
            tables.append(sums.groupby("period").agg(
                rmse=("rmse", "mean"), mae=("mae", "mean"), windows=("windows", "sum"),
                stations=("windows", "size")).reset_index().assign(kind=kind))
    mean = (pd.concat(tables).groupby(["kind", "period"], sort=False).mean()
            .astype({"windows": int, "stations": int}).reset_index())
    return [{**key, **r} for r in mean.to_dict("records")]
