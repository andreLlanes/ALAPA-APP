""" Grid search on validation RMSE: try every combination with the tuning seed and keep the best.
"""

import hashlib
import itertools
import json
from pathlib import Path

import config
from utils.artifacts import load_json, save_json

GRID_PATH = Path(__file__).resolve().parents[1] / "grid_tuning.json"

def load_grid() -> dict:
    """ Read grid_tuning.json.
    """
    with open(GRID_PATH, encoding="utf-8") as f:
        return json.load(f)

def combinations(grid: dict) -> list:
    """ Expand {name: [values]} into every combination, as a list of {name: value}.
    """
    names = list(grid)
    return [dict(zip(names, values)) for values in itertools.product(*(grid[n] for n in names))]

def point_name(params: dict) -> str:
    """ Return a short stable name for a set of hyperparameters (used in file names).
    """
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:10]

def search(grid: dict, score, path) -> dict:
    """ Score every combination (lower is better) and return the best point {params, val}.
        score(params, seed) trains with the tuning seed and returns the validation RMSE. Finished
        points are saved to `path`, so a rerun skips them.
    """
    record = load_json(path) or {"points": []}
    points = combinations(grid)
    for i, params in enumerate(points):
        if any(q["params"] == params for q in record["points"]):
            continue
        val = score(params, config.TUNING_SEED)
        record["points"].append({"params": params, "val": val})
        save_json(path, record)
        print(f"    POINT {i + 1}/{len(points)}: {params} -> val = {val:.4f}")
    return min((p for p in record["points"] if p["params"] in points), key=lambda p: p["val"])
