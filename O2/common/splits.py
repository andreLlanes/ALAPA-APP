"""Temporal partitioning: the chronological split and the rolling origin.

Section 4.7.3 defines two ways of cutting the record in time, and this module
supplies both. Neither shuffles, so no sample from a later period ever informs a
model evaluated on an earlier one.

**Chronological split.** The record is split into training, validation, and test
partitions in the proportions 70, 15, 15 percent. The cuts are computed once per
(city, source) over the pooled forecast origins and the same two timestamps are
applied to every station, so all stations are scored on the same calendar period
and fold-averaging compares like with like. Cuts fall on timestamp boundaries
rather than sample indices: all windows sharing an origin hour land in the same
partition, so no two stations disagree about which side of a cut a given hour is
on. The realized proportions are therefore approximate, exact only to the
granularity of the origin timestamps, and ``describe`` reports what was actually
achieved.

**Rolling origin.** Temporal generalization is assessed by advancing the split
point through the record. A model is trained on all data up to a given origin and
evaluated on the three months that follow; the origin then advances by three
months and the procedure repeats, the preceding test period absorbed into
training. Four origins are used, so the evaluated periods together span twelve
months and cover a full wet and dry cycle. Training is therefore an expanding
window, never a sliding one: every fold keeps all the history before its test
period.

The two protocols answer different questions and are both applied. The
chronological split holds one test period fixed, which is what leave-one-station-
out varies stations against; the rolling origin varies the test period instead,
so agreement between them indicates that a configuration's standing does not
depend on the particular months it was tested on.

Because every test period here is a fixed calendar window shared by all stations,
a station whose record does not reach into one contributes nothing to it and
drops out of that evaluation. Callers should check the excluded count rather than
assume every station is represented in every window.
"""

import numpy as np
import pandas as pd

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
PARTITIONS = ("train", "val", "test")

# Four origins advancing by three months, spanning a full annual cycle.
ROLLING_ORIGINS = 4
ROLLING_TEST_MONTHS = 3


def chronological_cuts(origins, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    """Return the two cut timestamps splitting pooled origins 70/15/15 in time.

    Args:
        origins: array of forecast origins, in any order; pooled across every
            station in one (city, source).
        train_frac: share of origins ending at (and including) the first cut.
        val_frac: share of origins between the two cuts.

    Returns:
        (train_end, val_end): the last origin of the training partition and the
        last origin of the validation partition. An origin belongs to train when
        it is <= train_end, to val when it is <= val_end, and to test otherwise.

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


def partition_mask(origins, cuts, partition):
    """Return the boolean mask selecting one partition's origins.

    Comparisons are made against the cut timestamps, not sample positions, so a
    given origin hour always lands in the same partition for every station.
    """
    if partition not in PARTITIONS and partition != "all":
        raise ValueError(f"partition must be one of {PARTITIONS + ('all',)}; got {partition!r}")
    origins = np.asarray(origins)
    if partition == "all":
        return np.ones(origins.shape, dtype=bool)
    train_end, val_end = cuts
    if partition == "train":
        return origins <= train_end
    if partition == "val":
        return (origins > train_end) & (origins <= val_end)
    return origins > val_end


def describe(origins, cuts):
    """Summarize each partition: count, realized share, and calendar span.

    Lets a run print the test period it actually used, so the fixed window can be
    checked against the intended design rather than assumed.
    """
    origins = np.asarray(origins)
    total = len(origins)
    summary = {}
    for partition in PARTITIONS:
        sel = origins[partition_mask(origins, cuts, partition)]
        if sel.size == 0:
            summary[partition] = {"n": 0, "frac": 0.0, "start": None,
                                  "end": None, "days": 0.0}
            continue
        start, end = sel.min(), sel.max()
        span_days = float((end - start) / np.timedelta64(1, "D"))
        summary[partition] = {
            "n": int(sel.size),
            "frac": float(sel.size / total) if total else 0.0,
            "start": str(start),
            "end": str(end),
            "days": span_days,
        }
    return summary


def rolling_origin_windows(origins, n_origins=ROLLING_ORIGINS,
                           test_months=ROLLING_TEST_MONTHS):
    """Return the advancing train/test windows of the rolling-origin protocol.

    The evaluated periods are anchored to the end of the record and laid back
    from it, so the ``n_origins * test_months`` months they span are the most
    recent ones available. Each window is half-open in time, [start, end), so
    consecutive periods neither overlap nor leave an hour unscored, and the final
    period includes the last origin in the record.

    Training is an expanding window: a fold trains on every origin strictly
    before its test period, which is what "the preceding test period absorbed
    into training" means.

    Args:
        origins: array of forecast origins, pooled across every station in one
            (city, source).
        n_origins: how many origins to advance through.
        test_months: length of each test period, in calendar months.

    Returns:
        A list of dicts, one per origin, each carrying the 1-based ``origin``
        number, ``train_start``, ``train_end``, ``test_start``, and ``test_end``
        as numpy datetime64 values. ``train_end`` equals ``test_start``: training
        is everything strictly before the test period.

    Raises:
        ValueError: if the record is empty, the arguments are non-positive, or
            the record does not begin early enough to leave any training data
            before the first test period.
    """
    origins = np.asarray(origins)
    if origins.size == 0:
        raise ValueError("cannot build rolling origins from an empty set of origins")
    if n_origins < 1 or test_months < 1:
        raise ValueError("n_origins and test_months must both be >= 1")

    first = pd.Timestamp(origins.min())
    # Half-open windows, so the last period must reach just past the final origin.
    end = pd.Timestamp(origins.max()) + pd.Timedelta(1, "h")
    edges = [end - pd.DateOffset(months=test_months * (n_origins - k))
             for k in range(n_origins + 1)]

    if edges[0] <= first:
        span = test_months * n_origins
        raise ValueError(
            f"record starts {first} but the first test period opens at {edges[0]}, "
            f"leaving no training data; {span} months of evaluation need a longer "
            f"record or fewer origins"
        )

    return [{"origin": k + 1,
             "train_start": np.datetime64(first),
             "train_end": np.datetime64(edges[k]),
             "test_start": np.datetime64(edges[k]),
             "test_end": np.datetime64(edges[k + 1])}
            for k in range(n_origins)]


def rolling_mask(origins, window, part="test"):
    """Return the boolean mask selecting one rolling-origin window's origins.

    ``train`` is every origin strictly before the test period (the expanding
    window); ``test`` is the half-open period [test_start, test_end).
    """
    if part not in ("train", "test"):
        raise ValueError(f"part must be 'train' or 'test'; got {part!r}")
    origins = np.asarray(origins)
    if part == "train":
        return origins < window["test_start"]
    return (origins >= window["test_start"]) & (origins < window["test_end"])


def describe_rolling(origins, windows):
    """Summarize each rolling-origin window: counts and calendar spans.

    Lets a run print the periods it actually trained and tested on, so the
    advancing design can be checked rather than assumed.
    """
    origins = np.asarray(origins)
    summary = []
    for window in windows:
        n_train = int(rolling_mask(origins, window, "train").sum())
        test = origins[rolling_mask(origins, window, "test")]
        summary.append({
            "origin": window["origin"],
            "train_end": str(window["train_end"]),
            "n_train": n_train,
            "test_start": str(window["test_start"]),
            "test_end": str(window["test_end"]),
            "n_test": int(test.size),
            "test_days": float((window["test_end"] - window["test_start"])
                               / np.timedelta64(1, "D")),
        })
    return summary
