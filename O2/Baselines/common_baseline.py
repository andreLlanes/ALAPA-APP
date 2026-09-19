"""Shared helpers for the O2 baselines.

Baselines are scored on exactly the windows the forecasting models see, so the
evaluation set is taken from the O1 builder output rather than rebuilt from the
database: each LSTM shard under O1/Outputs/lstm/<citySlug>/<source>/ already
carries one station's valid (origin, horizon) pairs, which is the station-hour
unit of analysis. Reading them keeps the baseline and the models on identical
origins, which is what makes the skill score of Section 4.8.3 a fair comparison.

Shards are read one station at a time, and only the columns a baseline needs are
kept, so no whole city is ever held in memory. Column names, the horizon, and the
city/source vocabulary are imported from O1 rather than restated, so O2 cannot
drift from the data contract the datasets were built under.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_O2_ROOT = os.path.dirname(_HERE)
_PROJECT_ROOT = os.path.dirname(_O2_ROOT)
_O1_ROOT = os.path.join(_PROJECT_ROOT, "O1")

for _p in (os.path.join(_O1_ROOT, "common"),
           os.path.join(_O1_ROOT, "Builders"),
           os.path.join(_O2_ROOT, "common")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common_build import (  # noqa: E402  (path bootstrap must run first)
    CITY_SLUG, ALL_CITIES, ALL_SOURCES, safe_name, OUTPUT_ROOT as O1_OUTPUT_ROOT,
)

# Baseline results live beside the datasets they score, under O2/Outputs/.
OUTPUT_ROOT = os.path.join(_O2_ROOT, "Outputs")

# The LSTM shards define the evaluation set: they are the only builder output
# that stores the raw lookback PM2.5 series a naive forecast needs.
WINDOW_MODEL = "lstm"


def city_slug(city):
    """Return the short slug used in output paths for a full city name."""
    return CITY_SLUG.get(city, city.replace(" ", "-"))


def windows_dir(city, source, model=WINDOW_MODEL):
    """Return the O1 folder holding one (city, source)'s window shards."""
    return os.path.join(O1_OUTPUT_ROOT, model, city_slug(city), source)


def result_dir(baseline, city, source):
    """Return (and create) the output folder for one baseline run.

    Layout mirrors O1: Outputs/<baseline>/<citySlug>/<source>/.
    """
    d = os.path.join(OUTPUT_ROOT, baseline, city_slug(city), source)
    os.makedirs(d, exist_ok=True)
    return d


def shard_paths(city, source, model=WINDOW_MODEL):
    """Return the sorted per-station shard paths for a (city, source).

    ``nodes.npz`` (the GNN graph ingredients) is not a station shard and is
    skipped. Raises if the dataset folder is missing, since that means the O1
    builder has not been run for this combination.
    """
    d = windows_dir(city, source, model)
    if not os.path.isdir(d):
        raise FileNotFoundError(
            f"No window shards at {d}. Build them first with "
            f'python O1/Builders/main.py --source {source} --city "{city}" --model {model}'
        )
    names = sorted(f for f in os.listdir(d)
                   if f.endswith(".npz") and f != "nodes.npz")
    return [os.path.join(d, n) for n in names]


def load_lookback_pm25(path):
    """Return one station's (location_key, lookback PM2.5, targets, origins).

    Only the PM2.5 column of the encoder block is retained: the full encoder
    tensor is dropped as soon as that column is copied out, which keeps peak
    memory at a few megabytes per station instead of the tens of megabytes the
    whole block would cost.

    Returns:
        location_key: str station key, read from the shard rather than the
            filename (which is path-sanitized and therefore lossy).
        lookback_pm25: (n, lookback) float array; NaN where the hour was missing
            (masked source only).
        Y: (n, horizon) float array of target PM2.5.
        origins: (n,) datetime64[ns] forecast origins.
    """
    with np.load(path, allow_pickle=False) as z:
        encoder_cols = [str(c) for c in z["encoder_cols"]]
        if "pm25" not in encoder_cols:
            raise ValueError(f"{path} has no pm25 encoder column; got {encoder_cols}")
        pm_idx = encoder_cols.index("pm25")

        X_enc = z["X_enc"]
        lookback_pm25 = np.array(X_enc[:, :, pm_idx], dtype=float)
        del X_enc

        Y = np.asarray(z["Y"], dtype=float)
        origins = z["meta_origin"]
        keys = z["meta_location_key"]
        location_key = str(keys[0]) if len(keys) else os.path.splitext(os.path.basename(path))[0]

    return location_key, lookback_pm25, Y, origins
