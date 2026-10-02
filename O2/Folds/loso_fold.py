#!/usr/bin/env python3
"""Select and freeze the twenty leave-one-station-out folds (Section 4.7.3).

Spatial generalization is assessed by withholding one Metro Manila station at a
time. Refitting every configuration per fold is expensive, so the held-out set is
limited to twenty stations, chosen once and frozen so that every configuration
(baselines, LSTM, GNN, GBT, every transfer variant) is scored on identical folds,
which is what makes the paired comparisons of Section 4.8.5 valid.

LOSO is scored through regression-kriging: the forecast at a withheld station is
the kriged surface of the retained stations' forecasts, so no method uses any of
the withheld station's own data. A withheld station is scored only on its windows
in the test period of the frozen split (O2/common/split_dates.json): Section
4.7.3 has leave-one-station-out hold the test period fixed, and kriged forecasts
at earlier times would draw on in-sample predictions at the retained stations.

Selection, in order:
    1. Eligibility. A station is eligible only if
       a. it reported at least 70% of the hours in its own active range. Reported
          hours are counted after QC and before gap filling, so interpolated hours
          do not count. (The 75% EPA criterion leaves too few stations for twenty
          folds.)
       b. it has at least MIN_GRADED_WINDOWS usable windows in the scored period,
          counted with the models' window rule (schema.exclusion_window_starts)
          and the split's purge gap, so no fold is empty or nearly so.
       No training-period history is required: under kriging the withheld
       station's own past is never used.
    2. Density. Each eligible station's density is the geodesic distance to its
       k-th nearest eligible neighbor, the same measure Section 4.8.1 plots the
       per-fold error against. Small distance = densely monitored core.
    3. Bands. Eligible stations are grouped into density bands by quantile of
       that distance, from the dense core to the sparse periphery.
    4. Draw. The twenty folds are spread evenly across the bands, and stations
       within a band are drawn at random under a fixed seed. A band with fewer
       stations than its share gives all of them, and the shortfall is taken
       from the other bands, nearest band first.

Two station lists are saved, because holding a station out and training on it
need different rules:
    training pool - every station meeting the study's quality rule (rule 1a).
                    Each fold trains on all of them except the one withheld.
    eligible      - the training-pool stations that can also be scored (rule
                    1b); the twenty folds are drawn from these.
A station with a good record that stopped reporting before the test period
fails rule 1b, so it is never withheld, but it still trains every fold.

Output is one JSON file (committed, unlike the gitignored CSV outputs) holding
the settings, the eligible stations, and the folds. Downstream code reads it
with ``load_folds`` and iterates it with ``iter_loso_folds``; it never
re-selects. An existing file is not overwritten without --overwrite, since
changing the folds invalidates every result already scored on them.

Temporal partitioning (the 70/15/15 split and the rolling origin) lives in
O2/common/splits.py and is not repeated here.

    python O2/Folds/loso_fold.py                      # select from the database, freeze
    python O2/Folds/loso_fold.py --demo --out demo.json   # synthetic stations, no database

The frozen file is written next to this script, as O2/Folds/loso_folds.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv

_HERE = os.path.dirname(os.path.abspath(__file__))   # O2/Folds
_O2_ROOT = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_O2_ROOT)
load_dotenv(os.path.join(_REPO_ROOT, ".env"))
for _p in (os.path.join(_REPO_ROOT, "O1", "common"), os.path.join(_O2_ROOT, "common")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from schema import MASKED_TABLE  # noqa: E402  (path set just above)
from splits import PURGE_HOURS, SPLIT_DATES_PATH, frozen_fixed_cuts, partition_mask  # noqa: E402

SOURCE_TABLE = MASKED_TABLE
CITY = "Metro Manila"
DEFAULT_OUT = os.path.join(_HERE, "loso_folds.json")

N_FOLDS = 20
COMPLETENESS_MIN = 0.70    # measured hours over the active range (rule 1a)
MIN_GRADED_WINDOWS = 100   # usable windows in the test period (rule 1b)
K_NEIGHBOR = 3
N_BANDS = 4
BAND_NAMES = {3: ("core", "middle", "periphery"),
              4: ("core", "inner", "outer", "periphery")}
SEED = 2026

EARTH_RADIUS_KM = 6371.0088


def resolve_pg_dsn() -> str:
    """Build the Postgres DSN from PG_DSN, or from the PG_* component vars."""
    dsn = os.environ.get("PG_DSN")
    if dsn:
        return dsn
    host = os.environ.get("PG_HOST")
    port = os.environ.get("PG_PORT", "5432")
    dbname = os.environ.get("PG_DB")
    user = os.environ.get("PG_USER")
    password = os.environ.get("PG_PASSWORD")
    if not all([host, dbname, user, password]):
        raise ValueError("Missing PG_DSN or PG_HOST/PG_DB/PG_USER/PG_PASSWORD.")
    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"


def fetch_station_stats(city: str = CITY, table: str = SOURCE_TABLE) -> pd.DataFrame:
    """Return one row per station: coordinates, active-range hours, reported hours.

    The masked table keeps every hour of each station's active range (long-gap NaNs
    included), so its row count is the active range. A reported hour is one with
    PM2.5 that was not produced by short-gap interpolation.
    """
    import psycopg2  # imported here so --demo runs without a database driver

    query = f"""
        SELECT location_key,
               AVG(latitude)  AS latitude,
               AVG(longitude) AS longitude,
               COUNT(*)       AS active_hours,
               COUNT(*) FILTER (
                   WHERE pm25 IS NOT NULL AND NOT COALESCE(short_gap_filled, FALSE)
               )              AS reported_hours
        FROM {table}
        WHERE city = %s
        GROUP BY location_key
        ORDER BY location_key
    """
    with psycopg2.connect(resolve_pg_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (city,))
            cols = [d[0] for d in cur.description or ()]
            rows = cur.fetchall()
    frame = pd.DataFrame(rows, columns=cols)
    if frame.empty:
        raise ValueError(f"No stations found for city={city!r} in {table}.")
    for c in ("latitude", "longitude"):
        frame[c] = frame[c].astype(float)
    return frame


def count_graded_windows(keys, cuts) -> dict:
    """Return {station: usable windows in the scored period} for the given stations.

    The scored period is the test partition of the frozen split, selected with
    splits.partition_mask exactly as it will be at scoring time. Windows follow
    the models' rule (schema.exclusion_window_starts) on the clean merged table.
    """
    builders = os.path.join(_REPO_ROOT, "O1", "Builders")
    if builders not in sys.path:
        sys.path.insert(0, builders)
    from common_build import load_station  # noqa: E402  (database access)
    from schema import LOOKBACK_H, TIME_COL, exclusion_window_starts  # noqa: E402

    counts = {}
    for key in keys:
        g = load_station("clean", key)
        if g.empty:
            counts[key] = 0
            continue
        g = g.sort_values(TIME_COL).reset_index(drop=True)
        starts = exclusion_window_starts(g)
        origins = g[TIME_COL].dt.tz_convert(None).to_numpy()[starts + LOOKBACK_H - 1]
        counts[key] = int(partition_mask(origins, cuts, "test").sum())
    return counts


def demo_station_stats(n: int = 69, seed: int = 7) -> pd.DataFrame:
    """Synthetic stations: a dense central cluster plus a sparse periphery.

    Mimics the shape of the Metro Manila network (concentrated in the central
    districts) so the selection logic can be exercised without the database.
    """
    rng = np.random.default_rng(seed)
    n_core = int(n * 0.65)
    core = rng.normal([14.58, 121.03], [0.025, 0.02], size=(n_core, 2))
    edge = np.column_stack([rng.uniform(14.34, 14.78, n - n_core),
                            rng.uniform(120.93, 121.15, n - n_core)])
    coords = np.vstack([core, edge])
    active = rng.integers(6_000, 24_000, size=n)
    completeness = np.clip(rng.beta(9, 1.2, size=n), 0, 1)
    return pd.DataFrame({
        "location_key": [f"demo_{i:03d}" for i in range(n)],
        "latitude": coords[:, 0],
        "longitude": coords[:, 1],
        "active_hours": active,
        "reported_hours": np.floor(active * completeness).astype(int),
        "graded_windows": rng.integers(0, 1500, size=n),
    })


def haversine_matrix_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Return the pairwise great-circle distance matrix, in kilometres."""
    lat_r = np.radians(lat)[:, None]
    lon_r = np.radians(lon)[:, None]
    dlat = lat_r - lat_r.T
    dlon = lon_r - lon_r.T
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_r) * np.cos(lat_r.T) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def add_completeness(stats: pd.DataFrame) -> pd.DataFrame:
    """Attach measured-hour completeness over each station's active range."""
    stats = stats.copy()
    stats["completeness"] = stats["reported_hours"] / stats["active_hours"].where(
        stats["active_hours"] > 0)
    return stats


def training_pool(stats: pd.DataFrame, threshold: float = COMPLETENESS_MIN) -> pd.DataFrame:
    """Return stations with coordinates that meet the quality rule (rule 1a)."""
    has_coords = stats["latitude"].notna() & stats["longitude"].notna()
    keep = has_coords & (stats["completeness"] >= threshold)
    return stats[keep].sort_values("location_key").reset_index(drop=True)


def eligible_stations(stats: pd.DataFrame, threshold: float = COMPLETENESS_MIN,
                      min_windows: int = MIN_GRADED_WINDOWS) -> pd.DataFrame:
    """Return training-pool stations that can also be scored (rules 1a and 1b)."""
    pool = training_pool(stats, threshold)
    keep = pool["graded_windows"].fillna(0) >= min_windows
    return pool[keep].reset_index(drop=True)


def add_density_bands(eligible: pd.DataFrame, k: int = K_NEIGHBOR,
                      n_bands: int = N_BANDS) -> pd.DataFrame:
    """Attach each station's k-th-nearest-neighbor distance and its density band.

    Band 0 is the densest (shortest distance). Quantile edges split the eligible
    stations into bands of near-equal size.
    """
    if len(eligible) <= k:
        raise ValueError(f"Need more than k={k} eligible stations; have {len(eligible)}.")
    dist = haversine_matrix_km(eligible["latitude"].to_numpy(),
                               eligible["longitude"].to_numpy())
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
    """Split n_folds across bands as evenly as their sizes allow.

    Each band first gets an equal share (the remainder going to the sparser bands,
    which the manuscript wants represented). A band that cannot fill its share
    gives what it has, and the shortfall moves to the nearest band with spare
    stations, so the draw stays as close to even as possible.
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
    """Draw the held-out stations from each band under a fixed seed.

    Stations are sorted by key within each band before drawing, so the result
    depends only on the seed and the station set, not on query order.
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


def build_record(stats: pd.DataFrame, pool: pd.DataFrame, banded: pd.DataFrame,
                 folds: pd.DataFrame, settings: dict) -> dict:
    """Assemble the frozen JSON record: settings, bands, training pool, eligible, folds."""
    n_bands = settings["n_bands"]
    bands = []
    for b in range(n_bands):
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


def load_folds(path: str = DEFAULT_OUT) -> dict:
    """Read the frozen fold record. Downstream code uses this, never re-selects."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def iter_loso_folds(record: dict):
    """Yield (fold_id, held_out_key, train_keys) for each frozen fold.

    Each fold trains on the whole training pool except the one withheld. (Files
    frozen before the training pool was recorded fall back to the eligible list.)
    """
    pool = [s["location_key"]
            for s in record.get("training_pool", record["eligible_stations"])]
    for fold in record["folds"]:
        held = fold["location_key"]
        yield fold["fold_id"], held, [k for k in pool if k != held]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select and freeze the LOSO folds (Section 4.7.3).")
    p.add_argument("--city", default=CITY)
    p.add_argument("--source-table", default=SOURCE_TABLE)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--n-folds", type=int, default=N_FOLDS)
    p.add_argument("--k", type=int, default=K_NEIGHBOR,
                   help="k for the k-th-nearest-neighbor density distance")
    p.add_argument("--n-bands", type=int, default=N_BANDS)
    p.add_argument("--completeness", type=float, default=COMPLETENESS_MIN)
    p.add_argument("--min-windows", type=int, default=MIN_GRADED_WINDOWS,
                   help="usable windows required in the test period")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--demo", action="store_true",
                   help="use synthetic stations instead of the database")
    p.add_argument("--overwrite", action="store_true",
                   help="replace an existing frozen fold file")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if os.path.exists(args.out) and not args.overwrite:
        raise SystemExit(
            f"{args.out} already exists. The folds are frozen; pass --overwrite only if "
            f"every result scored on them will be rerun.")

    cuts = frozen_fixed_cuts(args.city)
    if args.demo:
        stats = add_completeness(demo_station_stats())
    else:
        stats = add_completeness(fetch_station_stats(args.city, args.source_table))
        # Windows are only counted where completeness already passes; that is the
        # slow step (one table read per station).
        passing = stats.loc[stats["completeness"] >= args.completeness, "location_key"]
        counts = count_graded_windows(passing, cuts)
        # Stations failing completeness were not counted; record them as 0.
        stats["graded_windows"] = stats["location_key"].map(counts).fillna(0).astype(int)
    pool = training_pool(stats, args.completeness)
    eligible = eligible_stations(stats, args.completeness, args.min_windows)
    banded = add_density_bands(eligible, args.k, args.n_bands)
    folds = select_folds(banded, args.n_folds, args.seed)

    settings = {
        "city": args.city,
        "source": "demo" if args.demo else args.source_table,
        "n_folds": args.n_folds,
        "completeness_min": args.completeness,
        "min_graded_windows": args.min_windows,
        "k_neighbor": args.k,
        "n_bands": args.n_bands,
        "seed": args.seed,
        "scoring": {
            "method": "regression-kriging of the retained stations' forecasts",
            "period": "test partition of the frozen split",
            "scored_after_utc": str(cuts[1]),
            "purge_hours": PURGE_HOURS,
            "split_dates": os.path.relpath(SPLIT_DATES_PATH, _REPO_ROOT).replace(os.sep, "/"),
        },
    }
    record = build_record(stats, pool, banded, folds, settings)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    print(f"{len(stats)} stations | training pool (completeness >= {args.completeness:.0%}): "
          f"{len(pool)} | eligible (+ >= {args.min_windows} test windows after "
          f"{str(cuts[1])[:16]}): {len(banded)} | k={args.k}")
    for b in record["bands"]:
        print(f"  {b['name']:<10} {b['knn_km_min']:6.2f}-{b['knn_km_max']:6.2f} km  "
              f"eligible {b['n_eligible']:3d}  selected {b['n_selected']:2d}")
    print(f"Froze {len(folds)} folds to {args.out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI error handling
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
