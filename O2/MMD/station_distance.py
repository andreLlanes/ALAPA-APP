import argparse
import json
import os
import sys
import time
import zlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_O2_ROOT = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_O2_ROOT)
for _p in (os.path.join(_REPO_ROOT, "O1", "common"), os.path.join(_O2_ROOT, "common"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from schema import PM25_COL, MET_COLS, TIME_COLS
from cities import (TARGET_CITY, SOURCE_CITIES, ALL_CITIES, CITY_SLUG, LA_WINDOW,
                    split_bounds, local_time_features)
from db import fetch
import mmd

TABLE = "openaq.merged_clean"
OUTPUT_DIR = os.path.join(_HERE, "Outputs")
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
DEFAULT_OUT = os.path.join(OUTPUT_DIR, "station_mmd.csv")

N_CACHE = 4000    # source rows pulled per station
N_SAMPLE = 1000   # rows per side in each MMD draw
N_GLOBAL = 1000   # rows per city behind the fixed bandwidth
REPEATS = 3
SEED = 2026
MIN_ROWS = 168    # one week of complete hours; fewer leaves d_s undefined

DB_COLS = [PM25_COL] + MET_COLS


def list_stations():
    """Return {city: [location_key, ...]} for every station in merged_clean.

    A loose index scan over the primary key visits each station once instead of
    scanning the whole table.
    """
    _, rows = fetch(f"""
        WITH RECURSIVE s AS (
            (SELECT location_key, city FROM {TABLE} ORDER BY location_key LIMIT 1)
            UNION ALL
            SELECT n.location_key, n.city
            FROM s, LATERAL (
                SELECT location_key, city FROM {TABLE}
                WHERE location_key > s.location_key
                ORDER BY location_key LIMIT 1
            ) n
        )
        SELECT location_key, city FROM s
    """)
    out = {c: [] for c in ALL_CITIES}
    for key, city in rows:
        if city in out:
            out[city].append(key)
    return out


def _cache_dir(city, train):
    n = "all" if city == TARGET_CITY else N_CACHE
    name = f"{CITY_SLUG[city]}_{train[0]:%Y%m%d}-{train[1]:%Y%m%d%H}_n{n}_s{SEED}"
    d = os.path.join(CACHE_DIR, name)
    os.makedirs(d, exist_ok=True)
    return d


def _pull_station(key, train, sampled):
    complete = " AND ".join(f"{c} IS NOT NULL" for c in DB_COLS)
    order = ("md5(%s || ':' || location_key || ':' || "
             "extract(epoch FROM timestamp_utc)::bigint::text)"
             if sampled else "timestamp_utc")
    query = f"""
        SELECT extract(epoch FROM timestamp_utc)::bigint, latitude, longitude,
               {", ".join(DB_COLS)}, count(*) OVER ()
        FROM {TABLE}
        WHERE location_key = %s AND timestamp_utc >= %s AND timestamp_utc < %s
          AND {complete}
        ORDER BY {order}
        LIMIT %s
    """
    params = [key, train[0].to_pydatetime(), train[1].to_pydatetime()]
    if sampled:
        params.append(str(SEED))
    params.append(N_CACHE if sampled else None)
    _, rows = fetch(query, params)

    if not rows:
        return {"t": np.empty(0, np.int64),
                "X": np.empty((0, len(DB_COLS)), np.float32),
                "lat": np.nan, "lon": np.nan, "n_avail": 0}
    arr = np.array(rows, dtype=object)
    order_t = np.argsort(arr[:, 0].astype(np.int64), kind="stable")
    arr = arr[order_t]
    return {
        "t": arr[:, 0].astype(np.int64),
        "X": arr[:, 3:3 + len(DB_COLS)].astype(np.float32),
        "lat": np.nan if arr[0, 1] is None else float(arr[0, 1]),
        "lon": np.nan if arr[0, 2] is None else float(arr[0, 2]),
        "n_avail": int(arr[0, -1]),
    }


def _read_cache(path):
    if not os.path.exists(path):
        return None
    try:
        with np.load(path) as z:
            if list(z["cols"]) != DB_COLS:
                return None
            return {"t": z["t"], "X": z["X"], "lat": float(z["lat"]),
                    "lon": float(z["lon"]), "n_avail": int(z["n_avail"])}
    except (OSError, ValueError, KeyError, EOFError):
        return None


def _write_cache(path, rec):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, cols=np.array(DB_COLS), **rec)
    os.replace(tmp, path)


def load_city(city, keys, train):
    sampled = city != TARGET_CITY
    folder = _cache_dir(city, train)
    out, pulled = {}, 0
    for i, key in enumerate(keys, 1):
        path = os.path.join(folder, key.replace(":", "-").replace("/", "-") + ".npz")
        rec = _read_cache(path)
        if rec is None:
            rec = _pull_station(key, train, sampled)
            _write_cache(path, rec)
            pulled += 1
            print(f"  [{i}/{len(keys)}] {city} {key}: "
                  f"{len(rec['t'])} of {rec['n_avail']} complete training hours")
        out[key] = rec
    print(f"{city}: {len(keys)} stations ({pulled} pulled, {len(keys) - pulled} cached)")
    return out


def features(rec, city, include_pm25):
    x = rec["X"].astype(np.float64)
    if not include_pm25:
        x = x[:, 1:]
    return np.hstack([x, local_time_features(rec["t"], city)])


def _rng(*parts):
    return np.random.default_rng(
        [SEED] + [p if isinstance(p, int) else zlib.crc32(p.encode()) for p in parts])


def _draw(a, n, rng):
    if len(a) <= n:
        return a.astype(np.float64)
    return a[np.sort(rng.choice(len(a), n, replace=False))].astype(np.float64)


def compute_distances(sources, target_pool, source_pools):
    pair = {k: [] for c in sources for k in sources[c]}
    fixed = {k: [] for k in pair}
    sigma_fixed = []
    total = len(pair)

    for r in range(REPEATS):
        started = time.time()
        y = _draw(target_pool, N_SAMPLE, _rng(r, "target"))
        d_yy = mmd.sq_dists(y)
        yy_upper = mmd.upper_values(d_yy)

        pooled = np.vstack(
            [_draw(target_pool, N_GLOBAL, _rng(r, "global", TARGET_CITY))]
            + [_draw(source_pools[c], N_GLOBAL, _rng(r, "global", c)) for c in SOURCE_CITIES])
        s_med = mmd.pooled_median(mmd.upper_values(mmd.sq_dists(pooled)))
        sigma_fixed.append(s_med)
        s_fix = mmd.bandwidths(s_med)
        k_yy_fix = mmd.within_mean(d_yy, s_fix)

        done = 0
        for c, stations in sources.items():
            for key, a in stations.items():
                x = _draw(a, N_SAMPLE, _rng(r, key))
                d_xx = mmd.sq_dists(x)
                d_xy = mmd.sq_dists(x, y)
                s_pair = mmd.bandwidths(
                    mmd.pooled_median(mmd.upper_values(d_xx), yy_upper, d_xy))
                pair[key].append(mmd.within_mean(d_xx, s_pair)
                                 + mmd.within_mean(d_yy, s_pair)
                                 - 2.0 * mmd.cross_mean(d_xy, s_pair))
                fixed[key].append(mmd.within_mean(d_xx, s_fix) + k_yy_fix
                                  - 2.0 * mmd.cross_mean(d_xy, s_fix))
                done += 1
                if done % 100 == 0 or done == total:
                    print(f"  repeat {r + 1}/{REPEATS}: {done}/{total} stations "
                          f"({time.time() - started:.0f}s)")
    return pair, fixed, sigma_fixed


def summarize(df):
    """Print the Bangkok-versus-Los Angeles comparison and a tau suggestion."""
    ok = df[df["status"] == "ok"]
    print("\n" + "=" * 72)
    print("STATION DISTANCE TO METRO MANILA (unbiased MMD^2)")
    print("=" * 72)
    for col in ("mmd2_pair", "mmd2_fixed"):
        print(f"\n{col}")
        print(f"  {'city':<13}{'n':>5}{'median':>10}{'q25':>10}{'q75':>10}{'min':>10}{'max':>10}")
        for c in SOURCE_CITIES:
            v = ok.loc[ok["city"] == c, col]
            if v.empty:
                continue
            print(f"  {c:<13}{len(v):>5}{v.median():>10.4f}{v.quantile(.25):>10.4f}"
                  f"{v.quantile(.75):>10.4f}{v.min():>10.4f}{v.max():>10.4f}")

    bk = ok.loc[ok["city"] == "Bangkok", "mmd2_pair"].to_numpy()
    la = ok.loc[ok["city"] == "Los Angeles", "mmd2_pair"].to_numpy()
    if len(bk) and len(la):
        print("\nPremise check: are Bangkok stations closer to Metro Manila?")
        for col in ("mmd2_pair", "mmd2_fixed"):
            b = ok.loc[ok["city"] == "Bangkok", col].to_numpy()
            l = ok.loc[ok["city"] == "Los Angeles", col].to_numpy()
            p = stats.mannwhitneyu(b, l, alternative="less").pvalue
            p_closer = ((b[:, None] < l[None, :]).mean()
                        + 0.5 * (b[:, None] == l[None, :]).mean())
            print(f"  {col:<11} P(random BK station closer than random LA station) = "
                  f"{p_closer:.3f}   Mann-Whitney one-sided p = {p:.2e}")
        print(f"  LA stations closer than the Bangkok median: "
              f"{(la < np.median(bk)).mean():.1%}")

        rho = stats.spearmanr(ok["mmd2_pair"], ok["mmd2_fixed"])[0]
        print(f"\nRobustness: Spearman rank agreement, pair vs fixed kernel = {rho:.3f}")
        iqr = np.subtract(*np.percentile(ok["mmd2_pair"], [75, 25]))
        noise = ok["mmd2_pair_sd"].median()
        print(f"Stability: median repeat sd {noise:.4f} vs between-station IQR {iqr:.4f} "
              f"({noise / iqr:.1%} of the spread)")

        tau = float(np.median(np.clip(ok["mmd2_pair"], 0, None)))
        print(f"\nSuggested shared tau for Eq. 4.6 (median pooled d_s) = {tau:.4f}")
        for c, v in (("Bangkok", bk), ("Los Angeles", la)):
            w = mmd.sample_weights(v, tau)
            print(f"  {c:<13} weights: median {np.median(w):.3f}, "
                  f"range {w.min():.3f} - {w.max():.3f}")

    skipped = df[df["status"] != "ok"]["status"].value_counts()
    if not skipped.empty:
        print("\nStations without a distance: "
              + ", ".join(f"{n} {s}" for s, n in skipped.items()))


def main():
    """Pull, standardize, measure, and write the station distance table."""
    p = argparse.ArgumentParser(description=(
        __doc__ or "Station-level MMD distances to Metro Manila (Section 4.5.2)."
    ).split("\n\n")[0])
    p.add_argument("--exclude-mnl", default="",
                   help="comma-separated Metro Manila location_keys to leave out")
    p.add_argument("--no-pm25", action="store_true",
                   help="measure meteorology and time only (strict covariate shift)")
    p.add_argument("--out", default=DEFAULT_OUT, help="output CSV path")
    p.add_argument("--seed", type=int, default=None,
                   help="seeds the cached row sample and every draw; a different "
                        "value pulls an independent sample per station, which is "
                        "how the reported stability is checked against resampling")
    args = p.parse_args()

    if args.seed is not None:
        globals()["SEED"] = args.seed

    exclude = {k.strip() for k in args.exclude_mnl.split(",") if k.strip()}
    include_pm25 = not args.no_pm25
    feature_names = ([PM25_COL] if include_pm25 else []) + MET_COLS + TIME_COLS

    bounds = {c: split_bounds(c) for c in ALL_CITIES}
    print(f"Training partitions (LA_WINDOW={LA_WINDOW}):")
    for c in ALL_CITIES:
        s, e = bounds[c]["train"]
        print(f"  {c:<13} {s:%Y-%m-%d %H:%M} -> {e:%Y-%m-%d %H:%M} UTC")

    stations = list_stations()
    unknown = exclude - set(stations[TARGET_CITY])
    if unknown:
        raise SystemExit(f"--exclude-mnl keys not found in Metro Manila: {sorted(unknown)}")
    data = {c: load_city(c, stations[c], bounds[c]["train"]) for c in ALL_CITIES}

    # Standardization statistics come from the retained Metro Manila hours only.
    target_keys = [k for k, rec in data[TARGET_CITY].items()
                   if len(rec["t"]) and k not in exclude]
    target_pool = np.vstack([features(data[TARGET_CITY][k], TARGET_CITY, include_pm25)
                             for k in target_keys])
    mu = target_pool.mean(axis=0)
    sd = target_pool.std(axis=0)
    sd[sd == 0] = 1.0
    target_pool = ((target_pool - mu) / sd).astype(np.float32)
    print(f"\nReference: {len(target_keys)} Metro Manila stations, "
          f"{len(target_pool):,} training hours, {len(feature_names)} features"
          + (f" ({len(exclude)} excluded)" if exclude else ""))

    sources, source_pools, rows = {}, {}, []
    for c in SOURCE_CITIES:
        sources[c] = {}
        for key, rec in data[c].items():
            n = len(rec["t"])
            status = "ok" if n >= MIN_ROWS else ("no_training_rows" if n == 0 else "too_few_rows")
            rows.append({"location_key": key, "city": c, "latitude": rec["lat"],
                         "longitude": rec["lon"], "n_train_hours": rec["n_avail"],
                         "n_used": min(n, N_SAMPLE), "status": status})
            if n:
                z = ((features(rec, c, include_pm25) - mu) / sd).astype(np.float32)
                if status == "ok":
                    sources[c][key] = z
                source_pools.setdefault(c, []).append(z)
        source_pools[c] = np.vstack(source_pools[c])

    print(f"Measuring {sum(map(len, sources.values()))} source stations, "
          f"{REPEATS} repeats x {N_SAMPLE} hours per side\n")
    pair, fixed, sigma_fixed = compute_distances(sources, target_pool, source_pools)

    df = pd.DataFrame(rows)
    df["mmd2_pair"] = df["location_key"].map(lambda k: np.mean(pair[k]) if k in pair else np.nan)
    df["mmd2_pair_sd"] = df["location_key"].map(lambda k: np.std(pair[k], ddof=1) if k in pair else np.nan)
    df["mmd2_fixed"] = df["location_key"].map(lambda k: np.mean(fixed[k]) if k in fixed else np.nan)
    df["mmd2_fixed_sd"] = df["location_key"].map(lambda k: np.std(fixed[k], ddof=1) if k in fixed else np.nan)
    df = df[["location_key", "city", "latitude", "longitude", "n_train_hours", "n_used",
             "mmd2_pair", "mmd2_pair_sd", "mmd2_fixed", "mmd2_fixed_sd", "status"]]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "table": TABLE,
        "la_window": LA_WINDOW,
        "train_partitions": {c: [str(bounds[c]["train"][0]), str(bounds[c]["train"][1])]
                             for c in ALL_CITIES},
        "features": feature_names,
        "time_encoding": "local time per city",
        "standardization": "Metro Manila training hours (retained stations)",
        "excluded_mnl": sorted(exclude),
        "reference_stations": len(target_keys),
        "reference_hours": int(len(target_pool)),
        "n_cache": N_CACHE, "n_sample": N_SAMPLE, "n_global": N_GLOBAL,
        "repeats": REPEATS, "seed": SEED, "min_rows": MIN_ROWS,
        "sigma_fixed_per_repeat": sigma_fixed,
        "bandwidth_scales": mmd.BANDWIDTH_SCALES.tolist(),
    }
    meta_path = os.path.splitext(args.out)[0] + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nWrote {args.out}\nWrote {meta_path}")

    summarize(df)


if __name__ == "__main__":
    main()
