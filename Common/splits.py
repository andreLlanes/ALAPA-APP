""" Temporal partitioning (Section 4.7.3): the chronological split and the rolling origin.
    Neither shuffles, so no later window informs a model evaluated on an earlier one.
    Cuts are computed from each run's windows after the completeness and longest-gap filters,
    so they follow the data instead of being fixed dates.
"""

import numpy as np
import pandas as pd

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
PARTITIONS = ("train", "val", "test")
TRAIN, VAL, TEST, PURGED = 0, 1, 2, -1

# Four origins advancing by three months, spanning a full annual cycle.
ROLLING_ORIGINS = 4
ROLLING_TEST_MONTHS = 3
ROLLING_VAL_FRAC = 0.15

# The forecast horizon: a train/val window must end this far before the next
# partition starts, so no target hour is shared across a cut.
PURGE_HOURS = 72

def _purge(purge_h):
    return np.timedelta64(int(purge_h), "h")

def chronological_cuts(origins, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    """ Find the two cut timestamps that split the pooled origins 70/15/15 by window count.
        Counting windows rather than calendar time keeps a late-growing network from
        leaving little for training. Returns (train_end, val_end).
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
    """ Select one partition's origins by comparing against the cut timestamps, so an
        hour lands in the same partition for every station.
    """
    if partition not in PARTITIONS and partition != "all":
        raise ValueError(f"partition must be one of {PARTITIONS + ('all',)}; got {partition!r}")
    origins = np.asarray(origins)
    if partition == "all":
        return np.ones(origins.shape, dtype=bool)
    train_end, val_end = cuts
    gap = _purge(purge_h)
    # Train/val windows within the purge gap of their cut are dropped.
    if partition == "train":
        return origins <= train_end - gap
    if partition == "val":
        return (origins > train_end) & (origins <= val_end - gap)
    return origins > val_end

def partition_labels(origins, cuts, purge_h=PURGE_HOURS):
    """ Label every origin TRAIN, VAL, TEST or PURGED (inside a purge gap).
    """
    labels = np.full(len(origins), PURGED, np.int8)
    for code, name in ((TRAIN, "train"), (VAL, "val"), (TEST, "test")):
        labels[partition_mask(origins, cuts, name, purge_h)] = code
    return labels

def describe(origins, cuts, purge_h=PURGE_HOURS):
    """ Summarize each partition (count, share, calendar span) and count the purged windows,
        so a run reports the periods it actually used.
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
    """ Build the rolling-origin blocks: test periods of `test_months` laid back from the end
        of the record, each trained on everything before it (expanding window). The most
        recent `val_frac` of each block's history is its validation set.
        Returns one dict per block: origin, train_start, val_start (= train_end), test_start, test_end.
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
    """ Select one rolling-origin block's train, val or test origins.
    """
    if part not in ("train", "val", "test"):
        raise ValueError(f"part must be 'train', 'val' or 'test'; got {part!r}")
    origins = np.asarray(origins)
    gap = _purge(purge_h)
    val_start = window.get("val_start", window["train_end"])
    # Train ends a purge gap before val_start; val ends a purge gap before test_start.
    if part == "train":
        return origins <= val_start - gap
    if part == "val":
        return (origins > val_start) & (origins <= window["test_start"] - gap)
    return (origins >= window["test_start"]) & (origins < window["test_end"])

def describe_rolling(origins, windows, purge_h=PURGE_HOURS):
    """ Summarize each rolling-origin block: counts and calendar spans.
    """
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
