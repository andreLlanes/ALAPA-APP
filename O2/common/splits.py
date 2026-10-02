"""Temporal partitioning: the chronological split and the rolling origin.

Section 4.7.3 defines two ways of cutting the record in time, and this module
supplies both. Neither shuffles, so no sample from a later period ever informs a
model evaluated on an earlier one.

**Chronological split.** The record is split into training, validation, and test
partitions in the proportions 70, 15, 15 percent *of the usable forecast windows*,
pooled across every station of a city. Counting windows rather than calendar time
matters for a network that grew late: in Metro Manila most stations started in
mid-2025, so a 70% calendar cut would leave only about a third of the windows for
training. The two cuts are timestamps, applied identically to every station, so
no two stations disagree about which side of a cut an hour is on.

**Rolling origin.** Temporal generalization is assessed by advancing the split
point through the record: four three-month test blocks laid back from the end of
the record, each trained on everything before it (an expanding window, never a
sliding one). Inside each block's pre-test history, the most recent
ROLLING_VAL_FRAC of the windows is held out as that block's validation set, for
early stopping.

**Purge gap.** A window's origin is its last lookback hour and its targets run
PURGE_HOURS (the 72 h horizon) past it. A training window ending just before a cut
would therefore share target hours with the windows just after it, so every
training or validation window must end before the next partition begins: its
origin must lie at least PURGE_HOURS before the cut. The windows in that gap are
dropped and counted as ``purged``.

**Frozen dates.** Cuts computed from the data would move whenever the station
set changes, so they are computed once by this module's command line and frozen
in split_dates.json (committed). ``frozen_fixed_cuts`` and
``frozen_rolling_windows`` return them; every script should use those rather than
recomputing.

    python O2/common/splits.py --city "Metro Manila" --dry-run    # show, do not write
    python O2/common/splits.py --city "Metro Manila"              # compute and freeze

Because every test period is a fixed calendar window shared by all stations, a
station whose record does not reach into one contributes nothing to it and drops
out of that evaluation. Callers should check the excluded count rather than
assume every station is represented in every window.
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
PARTITIONS = ("train", "val", "test")

# Four origins advancing by three months, spanning a full annual cycle.
ROLLING_ORIGINS = 4
ROLLING_TEST_MONTHS = 3
ROLLING_VAL_FRAC = 0.15

# The forecast horizon: a training or validation window must end this far
# before the next partition starts, so no target hour is shared across a cut.
PURGE_HOURS = 72

SPLIT_DATES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "split_dates.json")

_HOUR = np.timedelta64(1, "h")


def _purge(purge_h):
    return np.timedelta64(int(purge_h), "h")


def chronological_cuts(origins, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    """Return the two cut timestamps splitting pooled origins 70/15/15 by count.

    Args:
        origins: array of forecast origins, in any order; pooled across every
            station in one city.
        train_frac: share of origins ending at (and including) the first cut.
        val_frac: share of origins between the two cuts.

    Returns:
        (train_end, val_end): an origin is training when <= train_end (less the
        purge gap), validation when <= val_end (less the purge gap), test
        otherwise. See ``partition_mask``.

    Raises:
        ValueError: if no origins are supplied, or the fractions leave no room
            for a test partition.
    """
    origins = np.asarray(origins)
    if origins.size == 0:
        raise ValueError("cannot split an empty set of origins")
    if not 0 < train_frac < 1 or not 0 < val_frac < 1:
        raise ValueError("train_frac and val_frac must each lie in (0, 1)")
    if train_frac + val_frac >= 1:
        raise ValueError(
            f"train_frac + val_frac = {train_frac + val_frac} leaves no test partition")

    ordered = np.sort(origins)
    n = len(ordered)
    train_end = ordered[max(int(round(n * train_frac)) - 1, 0)]
    val_end = ordered[max(int(round(n * (train_frac + val_frac))) - 1, 0)]
    return train_end, val_end


def partition_mask(origins, cuts, partition, purge_h=PURGE_HOURS):
    """Return the boolean mask selecting one partition's origins.

    Comparisons are made against the cut timestamps, not sample positions, so a
    given origin hour always lands in the same partition for every station.
    Training and validation windows within ``purge_h`` hours of their cut are
    excluded, so none of their target hours fall in the next partition.
    """
    if partition not in PARTITIONS and partition != "all":
        raise ValueError(f"partition must be one of {PARTITIONS + ('all',)}; got {partition!r}")
    origins = np.asarray(origins)
    if partition == "all":
        return np.ones(origins.shape, dtype=bool)
    train_end, val_end = cuts
    gap = _purge(purge_h)
    if partition == "train":
        return origins <= train_end - gap
    if partition == "val":
        return (origins > train_end) & (origins <= val_end - gap)
    return origins > val_end


def describe(origins, cuts, purge_h=PURGE_HOURS):
    """Summarize each partition: count, realized share, and calendar span.

    Also counts the windows dropped by the purge gap, so a run can report the
    test period it actually used rather than assume it.
    """
    origins = np.asarray(origins)
    total = len(origins)
    summary = {}
    for partition in PARTITIONS:
        sel = origins[partition_mask(origins, cuts, partition, purge_h)]
        if sel.size == 0:
            summary[partition] = {"n": 0, "frac": 0.0, "start": None,
                                  "end": None, "days": 0.0}
            continue
        start, end = sel.min(), sel.max()
        summary[partition] = {
            "n": int(sel.size),
            "frac": float(sel.size / total) if total else 0.0,
            "start": str(start),
            "end": str(end),
            "days": float((end - start) / np.timedelta64(1, "D")),
        }
    kept = sum(summary[p]["n"] for p in PARTITIONS)
    summary["purged"] = {"n": int(total - kept)}
    return summary


def rolling_origin_windows(origins, n_origins=ROLLING_ORIGINS,
                           test_months=ROLLING_TEST_MONTHS,
                           val_frac=ROLLING_VAL_FRAC, purge_h=PURGE_HOURS):
    """Return the advancing train/validation/test windows of the rolling origin.

    The test periods are anchored to the end of the record and laid back from
    it, so the ``n_origins * test_months`` months they span are the most recent
    available. Each test period is half-open in time, [test_start, test_end).

    Within each block, the windows that end before the test period begins (origin
    at least ``purge_h`` before ``test_start``) form its history. The most recent
    ``val_frac`` of them is validation, starting after ``val_start``; the rest,
    less a purge gap before ``val_start``, is training.

    Returns:
        A list of dicts, one per origin, with the 1-based ``origin`` number and
        ``train_start``, ``val_start``, ``test_start`` and ``test_end`` as numpy
        datetime64 values. ``train_end`` is kept as an alias of ``val_start`` (the
        cut between training and validation).

    Raises:
        ValueError: if the record is empty, the arguments are invalid, or a block
            has no history before its test period.
    """
    origins = np.asarray(origins)
    if origins.size == 0:
        raise ValueError("cannot build rolling origins from an empty set of origins")
    if n_origins < 1 or test_months < 1:
        raise ValueError("n_origins and test_months must both be >= 1")
    if not 0 < val_frac < 1:
        raise ValueError("val_frac must lie in (0, 1)")

    first = pd.Timestamp(origins.min())
    # Half-open windows, so the last period must reach just past the final origin.
    end = pd.Timestamp(origins.max()) + pd.Timedelta(1, "h")
    edges = [end - pd.DateOffset(months=test_months * (n_origins - k))
             for k in range(n_origins + 1)]
    gap = _purge(purge_h)

    windows = []
    for k in range(n_origins):
        test_start = np.datetime64(edges[k])
        history = np.sort(origins[origins <= test_start - gap])
        if history.size == 0:
            raise ValueError(
                f"rolling origin {k + 1}: no windows end before its test period "
                f"({edges[k]}); the record is too short for {n_origins} origins")
        val_start = history[max(int(round(history.size * (1 - val_frac))) - 1, 0)]
        windows.append({"origin": k + 1,
                        "train_start": np.datetime64(first),
                        "val_start": val_start,
                        "train_end": val_start,
                        "test_start": test_start,
                        "test_end": np.datetime64(edges[k + 1])})
    return windows


def rolling_mask(origins, window, part="test", purge_h=PURGE_HOURS):
    """Return the boolean mask selecting one rolling-origin window's origins.

    ``train`` ends a purge gap before ``val_start``; ``val`` runs from
    ``val_start`` to a purge gap before ``test_start``; ``test`` is the half-open
    period [test_start, test_end).
    """
    if part not in ("train", "val", "test"):
        raise ValueError(f"part must be 'train', 'val' or 'test'; got {part!r}")
    origins = np.asarray(origins)
    gap = _purge(purge_h)
    val_start = window.get("val_start", window["train_end"])
    if part == "train":
        return origins <= val_start - gap
    if part == "val":
        return (origins > val_start) & (origins <= window["test_start"] - gap)
    return (origins >= window["test_start"]) & (origins < window["test_end"])


def describe_rolling(origins, windows, purge_h=PURGE_HOURS):
    """Summarize each rolling-origin window: counts and calendar spans."""
    origins = np.asarray(origins)
    summary = []
    for window in windows:
        test = origins[rolling_mask(origins, window, "test", purge_h)]
        summary.append({
            "origin": window["origin"],
            "val_start": str(window.get("val_start", window["train_end"])),
            "n_train": int(rolling_mask(origins, window, "train", purge_h).sum()),
            "n_val": int(rolling_mask(origins, window, "val", purge_h).sum()),
            "test_start": str(window["test_start"]),
            "test_end": str(window["test_end"]),
            "n_test": int(test.size),
            "test_days": float((window["test_end"] - window["test_start"])
                               / np.timedelta64(1, "D")),
        })
    return summary


# ---------------------------------------------------------------------------
# Frozen dates
# ---------------------------------------------------------------------------

def compute_split_dates(origins):
    """Compute every cut for one city from its pooled origins, ready to freeze."""
    origins = np.asarray(origins)
    cuts = chronological_cuts(origins)
    rolling = rolling_origin_windows(origins)
    return {
        "method": "70/15/15 of pooled usable windows (forecast origins)",
        "purge_hours": PURGE_HOURS,
        "n_windows": int(origins.size),
        "fixed": {"train_end": str(cuts[0]), "val_end": str(cuts[1]),
                  "partitions": describe(origins, cuts)},
        "rolling": {"n_origins": ROLLING_ORIGINS, "test_months": ROLLING_TEST_MONTHS,
                    "val_frac": ROLLING_VAL_FRAC,
                    "windows": [{"origin": w["origin"], "val_start": str(w["val_start"]),
                                 "test_start": str(w["test_start"]),
                                 "test_end": str(w["test_end"])} for w in rolling],
                    "summary": describe_rolling(origins, rolling)},
    }


def load_split_dates(city, path=SPLIT_DATES_PATH):
    """Return the frozen record for one city, or None if it has not been frozen."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f).get(city)


def frozen_fixed_cuts(city, path=SPLIT_DATES_PATH):
    """Return the frozen (train_end, val_end) for a city, for ``partition_mask``."""
    record = load_split_dates(city, path)
    if record is None:
        raise KeyError(f"no frozen split dates for {city!r} in {path}; "
                       f"run python O2/common/splits.py --city \"{city}\"")
    return (np.datetime64(record["fixed"]["train_end"]),
            np.datetime64(record["fixed"]["val_end"]))


def frozen_rolling_windows(city, path=SPLIT_DATES_PATH):
    """Return the frozen rolling-origin windows for a city, for ``rolling_mask``."""
    record = load_split_dates(city, path)
    if record is None:
        raise KeyError(f"no frozen split dates for {city!r} in {path}; "
                       f"run python O2/common/splits.py --city \"{city}\"")
    out = []
    for w in record["rolling"]["windows"]:
        val_start = np.datetime64(w["val_start"])
        out.append({"origin": w["origin"], "val_start": val_start, "train_end": val_start,
                    "test_start": np.datetime64(w["test_start"]),
                    "test_end": np.datetime64(w["test_end"])})
    return out


def _city_origins(city):
    """Pool the forecast origins of every usable window in a city's clean source.

    Uses the same window rule as the LSTM builder (schema.exclusion_window_starts),
    so the frozen dates are cut on exactly the windows the models train on.
    """
    import sys
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    builders = os.path.join(repo, "O1", "Builders")
    if builders not in sys.path:
        sys.path.insert(0, builders)
    from common_build import load_station, station_keys  # noqa: E402
    from schema import LOOKBACK_H, TIME_COL, exclusion_window_starts  # noqa: E402

    collected = []
    for key in station_keys("clean", city):
        g = load_station("clean", key)
        if g.empty:
            continue
        g = g.sort_values(TIME_COL).reset_index(drop=True)
        starts = exclusion_window_starts(g)
        if starts.size:
            times = g[TIME_COL].dt.tz_convert(None).to_numpy()
            collected.append(times[starts + LOOKBACK_H - 1])
    if not collected:
        raise ValueError(f"no usable windows for {city!r}")
    return np.concatenate(collected)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Compute and freeze the split dates for a city.")
    p.add_argument("--city", required=True)
    p.add_argument("--dry-run", action="store_true", help="print the dates without saving")
    p.add_argument("--overwrite", action="store_true",
                   help="replace this city's frozen dates (invalidates results scored on them)")
    p.add_argument("--path", default=SPLIT_DATES_PATH)
    args = p.parse_args()

    record = compute_split_dates(_city_origins(args.city))
    record["frozen_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    fixed = record["fixed"]
    parts = fixed["partitions"]
    print(f"{args.city}: {record['n_windows']} usable windows")
    print(f"  fixed:   train_end {fixed['train_end']}   val_end {fixed['val_end']}")
    for name in PARTITIONS:
        s = parts[name]
        print(f"    {name:5s} {s['n']:7d} windows ({100 * s['frac']:4.1f}%)  {s['start']} .. {s['end']}")
    print(f"    purged {parts['purged']['n']} windows")
    for w in record["rolling"]["summary"]:
        print(f"  ro_{w['origin']}:    val from {w['val_start']}  test {w['test_start']} .. "
              f"{w['test_end']}  (train {w['n_train']}, val {w['n_val']}, test {w['n_test']})")

    if args.dry_run:
        return
    store = {}
    if os.path.exists(args.path):
        with open(args.path, encoding="utf-8") as f:
            store = json.load(f)
    if args.city in store and not args.overwrite:
        raise SystemExit(f"{args.city} is already frozen in {args.path}; "
                         f"pass --overwrite only if every result scored on it will be rerun.")
    store[args.city] = record
    with open(args.path, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)
    print(f"Froze {args.city} to {args.path}")


if __name__ == "__main__":
    main()
