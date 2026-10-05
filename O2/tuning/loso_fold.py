""" Select and freeze the twenty leave-one-station-out folds (Section 4.7.3).
    One Metro Manila station is withheld at a time; the forecast at a withheld station is the
    kriged surface of the retained stations' forecasts, so no method uses its own data. A withheld
    station is scored on its windows in the test partition of the run's split.

    Selection, in order:
        1. Eligibility: completeness >= 70% (rule 1a) and >= MIN_GRADED_WINDOWS test windows
           under the default filters (rule 1b). No training history is required.
        2. Density: distance to the k-th nearest eligible neighbour.
        3. Bands: eligible stations grouped into density bands by quantile.
        4. Draw: folds spread evenly across bands, drawn under a fixed seed; a band short of its
           share gives all it has and the shortfall moves to the nearest band.
    Two lists are saved: the training pool (rule 1a; every fold trains on it minus the withheld
    station) and the eligible stations (1a and 1b), from which the folds are drawn.
    The file is not overwritten without --overwrite: changing the folds invalidates every result.
    PYTHONPATH=.. python -m tuning.loso_fold
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config
from Common.splits import TEST
from data.batch import station_distances_km
from data.database import CITIES, read_copy
from data.load import LOSO_PATH, load_city
from data.windows import city_arrays
from utils.artifacts import load_json, save_json

CITY = "mm"
N_FOLDS = 20
COMPLETENESS_MIN = 0.70    # completeness over the active range (rule 1a)
MIN_GRADED_WINDOWS = 100   # test windows under the default filters (rule 1b)
K_NEIGHBOR = 3
N_BANDS = 4
BAND_NAMES = {3: ("core", "middle", "periphery"),
              4: ("core", "inner", "outer", "periphery")}
SEED = 2026

def station_stats(city: str = CITY) -> pd.DataFrame:
    """ Return one row per station: coordinates, completeness, and test windows under the defaults.
    """
    arrays = city_arrays(read_copy(city))
    stats = pd.DataFrame({"location_key": arrays["keys"], "latitude": arrays["lat"],
                          "longitude": arrays["lon"], "completeness": arrays["completeness"]})
    data = load_city(city, config.DEFAULT_LONGEST_GAP, int(COMPLETENESS_MIN * 100))
    counts = pd.Series(data.keys[data.station[data.split_window_ids(TEST)]]).value_counts()
    stats["graded_windows"] = stats["location_key"].map(counts).fillna(0).astype(int)
    return stats

def training_pool(stats: pd.DataFrame, threshold: float = COMPLETENESS_MIN) -> pd.DataFrame:
    """ Return stations with coordinates that meet the quality rule (rule 1a).
    """
    has_coords = stats["latitude"].notna() & stats["longitude"].notna()
    keep = has_coords & (stats["completeness"] >= threshold)
    return stats[keep].sort_values("location_key").reset_index(drop=True)

def eligible_stations(stats: pd.DataFrame, threshold: float = COMPLETENESS_MIN,
                      min_windows: int = MIN_GRADED_WINDOWS) -> pd.DataFrame:
    """ Return training-pool stations that can also be scored (rules 1a and 1b).
    """
    pool = training_pool(stats, threshold)
    keep = pool["graded_windows"].fillna(0) >= min_windows
    return pool[keep].reset_index(drop=True)

def add_density_bands(eligible: pd.DataFrame, k: int = K_NEIGHBOR,
                      n_bands: int = N_BANDS) -> pd.DataFrame:
    """ Attach each station's k-th-nearest-neighbour distance and its density band (0 = densest).
        Quantile edges split the eligible stations into bands of near-equal size.
    """
    if len(eligible) <= k:
        raise ValueError(f"Need more than k={k} eligible stations; have {len(eligible)}.")
    dist = station_distances_km(eligible["latitude"].to_numpy(), eligible["longitude"].to_numpy())
    # Column 0 of each sorted row is the station itself (distance 0).
    knn = np.sort(dist, axis=1)[:, k]

    out = eligible.copy()
    out["knn_distance_km"] = knn
    # Rank first so tied distances cannot collapse two quantile edges into one.
    out["band"] = pd.qcut(out["knn_distance_km"].rank(method="first"),
                          n_bands, labels=False).astype(int)
    names = BAND_NAMES.get(n_bands, tuple(f"band_{b}" for b in range(n_bands)))
    out["density_band"] = out["band"].map(dict(enumerate(names)))
    return out

def allocate(band_sizes: list[int], n_folds: int) -> list[int]:
    """ Split n_folds across bands as evenly as their sizes allow; the remainder goes to the
        sparser bands, and a band short of its share passes the shortfall to the nearest band.
    """
    n_bands = len(band_sizes)
    if sum(band_sizes) < n_folds:
        raise ValueError(f"Only {sum(band_sizes)} eligible stations for {n_folds} folds.")
    base, extra = divmod(n_folds, n_bands)
    quota = [base + (1 if b >= n_bands - extra else 0) for b in range(n_bands)]
    take = [min(q, s) for q, s in zip(quota, band_sizes)]

    for b in range(n_bands):
        # A band may already hold more than its quota from an earlier transfer.
        short = max(quota[b] - take[b], 0)
        # Nearest bands first, outward from the short band.
        for other in sorted(range(n_bands), key=lambda o: (abs(o - b), -o)):
            if short == 0:
                break
            spare = band_sizes[other] - take[other]
            give = min(spare, short)
            take[other] += give
            short -= give
    return take

def select_folds(banded: pd.DataFrame, n_folds: int = N_FOLDS, seed: int = SEED) -> pd.DataFrame:
    """ Draw the held-out stations from each band under a fixed seed (sorted by key first, so the
        result depends only on the seed and the station set).
    """
    n_bands = int(banded["band"].max()) + 1
    sizes = [int((banded["band"] == b).sum()) for b in range(n_bands)]
    take = allocate(sizes, n_folds)

    rng = np.random.default_rng(seed)
    picks = []
    for b in range(n_bands):
        pool = banded[banded["band"] == b].sort_values("location_key")
        idx = rng.choice(len(pool), size=take[b], replace=False)
        picks.append(pool.iloc[np.sort(idx)])

    folds = pd.concat(picks).sort_values(["band", "location_key"]).reset_index(drop=True)
    folds.insert(0, "fold_id", np.arange(1, len(folds) + 1))
    return folds

def build_record(stats, pool, banded, folds, settings: dict) -> dict:
    """ Assemble the frozen record: settings, bands, training pool, eligible stations, folds.
    """
    bands = []
    for b in range(settings["n_bands"]):
        in_band = banded[banded["band"] == b]
        bands.append({
            "band": b,
            "name": in_band["density_band"].iloc[0] if len(in_band) else None,
            "knn_km_min": round(float(in_band["knn_distance_km"].min()), 3),
            "knn_km_max": round(float(in_band["knn_distance_km"].max()), 3),
            "n_eligible": int(len(in_band)),
            "n_selected": int((folds["band"] == b).sum()),
        })

    def rows(frame, cols):
        return [{c: (round(float(v), 6) if isinstance(v, (float, np.floating)) else
                     int(v) if isinstance(v, (np.integer,)) else v)
                 for c, v in zip(cols, r)} for r in frame[cols].itertuples(index=False)]

    pool_cols = ["location_key", "latitude", "longitude", "completeness", "graded_windows"]
    station_cols = pool_cols + ["knn_distance_km", "band", "density_band"]
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": settings,
        "n_stations_total": int(len(stats)),
        "n_training_pool": int(len(pool)),
        "n_eligible": int(len(banded)),
        "bands": bands,
        "training_pool": rows(pool, pool_cols),
        "eligible_stations": rows(banded, station_cols),
        "folds": rows(folds, ["fold_id"] + station_cols),
    }

def load_folds(path=LOSO_PATH) -> dict:
    """ Read the frozen fold record. Downstream code uses this and never re-selects.
    """
    return load_json(path)

def iter_loso_folds(record: dict):
    """ Yield (fold_id, held_out_key, train_keys) for each frozen fold; each fold trains on the
        whole training pool except the withheld station.
    """
    pool = [s["location_key"] for s in record.get("training_pool", record["eligible_stations"])]
    for fold in record["folds"]:
        held = fold["location_key"]
        yield fold["fold_id"], held, [k for k in pool if k != held]

def main():
    p = argparse.ArgumentParser(description="Select and freeze the LOSO folds (Section 4.7.3).")
    p.add_argument("--city", default=CITY, choices=list(CITIES))
    p.add_argument("--n-folds", type=int, default=N_FOLDS)
    p.add_argument("--k", type=int, default=K_NEIGHBOR)
    p.add_argument("--n-bands", type=int, default=N_BANDS)
    p.add_argument("--completeness", type=float, default=COMPLETENESS_MIN)
    p.add_argument("--min-windows", type=int, default=MIN_GRADED_WINDOWS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--overwrite", action="store_true", help="replace the frozen fold file")
    args = p.parse_args()
    if LOSO_PATH.exists() and not args.overwrite:
        raise SystemExit(f"{LOSO_PATH} already exists. The folds are frozen; pass --overwrite "
                         f"only if every result scored on them will be rerun.")

    stats = station_stats(args.city)
    pool = training_pool(stats, args.completeness)
    banded = add_density_bands(eligible_stations(stats, args.completeness, args.min_windows),
                               args.k, args.n_bands)
    folds = select_folds(banded, args.n_folds, args.seed)
    settings = {"city": CITIES[args.city], "n_folds": args.n_folds,
                "completeness_min": args.completeness, "min_graded_windows": args.min_windows,
                "longest_gap": config.DEFAULT_LONGEST_GAP, "k_neighbor": args.k, "n_bands": args.n_bands,
                "seed": args.seed,
                "scoring": {"method": "regression-kriging of the retained stations' forecasts",
                            "period": "test partition of the run's split"}}
    record = build_record(stats, pool, banded, folds, settings)
    save_json(LOSO_PATH, record)

    print(f"[LOSO] {CITIES[args.city]}: {len(stats)} stations | training pool "
          f"(completeness >= {args.completeness:.0%}): {len(pool)} | eligible "
          f"(+ >= {args.min_windows} test windows): {len(banded)} | k = {args.k}")
    for b in record["bands"]:
        print(f"    {b['name']:<10} {b['knn_km_min']:6.2f}-{b['knn_km_max']:6.2f} km  "
              f"eligible {b['n_eligible']:3d}  selected {b['n_selected']:2d}")
    print(f"    Froze {len(folds)} folds to {LOSO_PATH}")

if __name__ == "__main__":
    main()
