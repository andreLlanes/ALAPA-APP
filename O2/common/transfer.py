#Source-sample weights for the GBT's instance transfer (Eq. 4.6).

import math
import os

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TABLE = os.path.join(os.path.dirname(_HERE), "MMD", "Outputs", "station_mmd.csv")

# Candidate values, spanning "discount distant stations sharply" to "pool almost
# uniformly". The median pooled distance (0.2465 on the full-feature table) sits
# near the top of this range and is the script's own suggestion, not a decision.
TAU_GRID = (0.02, 0.05, 0.10, 0.25, 0.50)

# Unset on purpose: select on validation RMSE over TAU_GRID, then fix it here with
# the value and the date. Until then callers must pass tau, so no run can report
# a number under an unchosen default.
TAU = None

DISTANCE_COLUMN = "mmd2_pair"  # mmd2_fixed is the common-kernel robustness check


def load_distances(path=DEFAULT_TABLE, city=None, column=DISTANCE_COLUMN):
    """Return {location_key: d_s} for stations that have a usable distance.

    Stations whose status is not "ok" are absent: they had no training-partition
    hours, or fewer than one week, so no distance was measured and they take no
    part in transfer.
    """
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"]
    if city is not None:
        df = df[df["city"] == city]
    return dict(zip(df["location_key"], df[column].astype(float)))


def station_weights(tau, path=DEFAULT_TABLE, city=None, column=DISTANCE_COLUMN):
    """Return {location_key: w_s} under Eq. 4.6.

    Distances are clipped at zero first: the unbiased estimator can return a small
    negative value for a station that matches Metro Manila closely, and w must
    never exceed the weight of a Metro Manila row.
    """
    if tau is None or tau <= 0:
        raise ValueError("tau must be a positive number chosen on validation data")
    return {k: math.exp(-max(d, 0.0) / tau)
            for k, d in load_distances(path, city, column).items()}


def weight_column(location_keys, tau, target_keys, path=DEFAULT_TABLE,
                  column=DISTANCE_COLUMN):
    """Weights aligned to a training table's location_key column.

    ``target_keys`` is the set of Metro Manila stations in the pool; they take
    w = 1. Every other key must appear in the distance table, so a source station
    that was never measured raises instead of being silently treated as target
    data.
    """
    w = station_weights(tau, path, None, column)
    target = set(target_keys)
    out = []
    for key in location_keys:
        if key in target:
            out.append(1.0)
        elif key in w:
            out.append(w[key])
        else:
            raise KeyError(
                f"{key} is neither a Metro Manila station nor a measured source "
                "station; drop it before training or re-run station_distance.py")
    return out
