""" Grid search on validation RMSE: try every combination with the first seed and keep the best.
"""

import hashlib
import itertools
import json
import shutil
from pathlib import Path

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

def _remove(model_path):
    """ Delete a saved model, whether a file (.pt) or a folder (GBT).
    """
    path = Path(model_path)
    path.with_suffix(".pt").unlink(missing_ok=True)
    if path.is_dir():
        shutil.rmtree(path)

def search(grid: dict, score, path, seed: int) -> dict:
    """ Score every combination (lower is better) and return the best point:
        {params, val, model, seed}. score(params, model_path, seed) trains with that seed, saves
        its model at model_path and returns the validation RMSE. Every point trains with the seed
        the search started with, and only the best point's model is kept, so the run with that
        seed can reuse it. Finished points are saved to `path`, so a rerun skips them.
    """
    path = Path(path)
    record = load_json(path) or {"points": [], "seed": seed}
    points = combinations(grid)
    missing = [p for p in points if not any(q["params"] == p for q in record["points"])]
    for i, params in enumerate(points):
        if params not in missing:
            continue
        model = path.parent / "models" / point_name(params)
        val = score(params, model, record["seed"])
        record["points"].append({"params": params, "val": val, "model": str(model)})
        best = min(record["points"], key=lambda p: p["val"])
        for p in record["points"]:
            if p is not best:
                _remove(p["model"])
        save_json(path, record)
        print(f"    POINT {i + 1}/{len(points)}: {params} -> val = {val:.4f}")
    tried = [p for p in record["points"] if p["params"] in points]
    record["best"] = min(tried, key=lambda p: p["val"])
    save_json(path, record)
    return {**record["best"], "seed": record["seed"]}
