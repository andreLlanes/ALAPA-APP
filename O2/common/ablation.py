"""Training-data ablation: contiguous block draws at reduced data volumes.

Section 4.7.5 reduces the Metro Manila training data to 25, 50, and 75 percent of
the full training partition and compares the best transfer configuration against
the no-transfer configuration at each level, which is how H5 is read: the benefit
of transfer should shrink as target data grows.

Data are removed in contiguous blocks, never as individual hours. Hourly
observations are strongly autocorrelated, so dropping random hours would leave a
model with nearly the same information as the full record and flatten the curve
into a null result. Blocks must therefore span at least one week, which exceeds
the 144 h a single sample covers (72 h lookback plus 72 h horizon).

Only the training partition is reduced; validation and test stay at full size, so
a change in performance reflects the reduction in training data rather than a
change in what is being predicted.

Window erosion. The O1 builders store complete windows, so selecting an origin
also drags in its 72 h of lookback and 72 h of horizon. An origin sitting near
the edge of a retained block therefore reaches into a discarded block, and the
data the ablation meant to remove comes back in through the window. Eroding each
retained run by the window geometry prevents that, at the cost of 143 eligible
origins per run, which is why the default block is four weeks rather than the
one-week floor:

    block length      eligible origins      retained within the block
    1 week (168 h)    26                    15 percent
    2 weeks (336 h)   194                   58 percent
    4 weeks (672 h)   530                   79 percent

At the one-week floor, erosion would discard most of every block and a nominal
25 percent level would deliver about 4 percent. Erosion is applied only at
boundaries that touch a discarded block: the outer edges of the training span are
left alone, so the 100 percent level reproduces the full training partition
exactly and remains comparable to the reduced ones.

Draws are seeded deterministically from the level and draw number, so the three
draws at each level are reproducible and every model configuration trains on
byte-identical data at a given (level, draw).
"""

import json
import os
import sys

import numpy as np

_O1_COMMON = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "O1", "common"))
if _O1_COMMON not in sys.path:
    sys.path.insert(0, _O1_COMMON)

from schema import LOOKBACK_H, HORIZON_H  # noqa: E402  (path bootstrap must run first)

ABLATION_LEVELS = (0.25, 0.50, 0.75, 1.00)
ABLATION_DRAWS = 3

# Section 4.7.5 requires at least one week; four weeks keeps erosion affordable.
MIN_BLOCK_HOURS = 168
DEFAULT_BLOCK_HOURS = 672
SEED_BASE = 20260906

_HOUR = np.timedelta64(1, "h")


def draw_seed(draw, seed_base=SEED_BASE):
    """Return the deterministic seed for one draw.

    The seed depends on the draw alone and not on the level, so every level
    within a draw grows along the same permutation of candidate blocks and the
    retained sets nest: the blocks kept at 25 percent are a subset of those kept
    at 50. The ablation curve then varies only how much data the model saw, not
    which stretches of calendar time it saw, which is what H5 is about.

    Being derived rather than drawn from a global stream also means a single
    level can be re-run later and reproduce exactly the same blocks.
    """
    return int(seed_base + draw)


def training_span(train_origins):
    """Return the half-open [start, end) calendar span of the training origins."""
    origins = np.asarray(train_origins)
    if origins.size == 0:
        raise ValueError("cannot ablate an empty training partition")
    return origins.min(), origins.max() + _HOUR


def _assemble(picked, edges, span_start, span_end, erode, lookback, horizon):
    """Turn a set of candidate block indices into merged, eroded blocks.

    Consecutive indices are merged into single runs first. Merging matters: two
    adjacent blocks form one longer stretch, eroded once at each end rather than
    twice, so it yields more usable origins than two isolated blocks of the same
    total length.
    """
    runs = []
    for idx in sorted(picked):
        if runs and idx == runs[-1][1] + 1:
            runs[-1][1] = idx
        else:
            runs.append([idx, idx])

    blocks = []
    for lo, hi in runs:
        start, end = edges[lo], min(edges[hi + 1], span_end)
        origin_start, origin_end = start, end
        if erode:
            # Only trim where a discarded block actually abuts this run; the
            # outer edges of the training span have nothing to leak from.
            if start > span_start:
                origin_start = start + (lookback - 1) * _HOUR
            if end < span_end:
                origin_end = end - horizon * _HOUR
        if origin_start >= origin_end:
            continue
        blocks.append({"start": start, "end": end,
                       "origin_start": origin_start, "origin_end": origin_end})
    return blocks


def draw_blocks(train_origins, level, seed, block_hours=DEFAULT_BLOCK_HOURS,
                erode=True, calibrate=True, lookback=LOOKBACK_H, horizon=HORIZON_H):
    """Return the retained blocks for one ablation level and draw.

    The training span is chopped into equal candidate blocks and a random subset
    is kept. How many are kept depends on ``calibrate``:

        calibrate=True  - keep however many blocks bring the share of retained
            *origins* closest to ``level``. Section 4.7.5 reduces the data "to 25,
            50, and 75 percent of the full training partition", which is a claim
            about volume, and block rounding plus erosion otherwise leave a
            nominal 25 percent delivering nearer 18. Calibrating makes the label
            on the ablation curve true.
        calibrate=False - keep ``round(level * n_blocks)`` blocks, so ``level`` is
            a share of blocks rather than of data.

    Calibration draws a single permutation from the seed and grows the retained
    set along it, so the result stays reproducible and the levels remain nested:
    the blocks kept at 25 percent are a subset of those kept at 50.

    Args:
        train_origins: origins of the training partition, pooled across stations.
        level: target share, in (0, 1].
        seed: seed for the draw, normally from ``draw_seed``.
        block_hours: candidate block length; must be at least MIN_BLOCK_HOURS.
        erode: trim each run by the window geometry so no retained origin reaches
            into a discarded block. Disable only to reproduce a naive ablation.
        calibrate: target retained origins rather than retained blocks.
        lookback, horizon: window geometry, defaulting to the O1 schema.

    Returns:
        A list of block dicts with ``start``/``end`` (the retained calendar span)
        and ``origin_start``/``origin_end`` (the half-open range of origins
        eligible after erosion). Runs left with no eligible origin are dropped.

    Raises:
        ValueError: on an empty partition, a level outside (0, 1], or a block
            shorter than the one-week floor.
    """
    if not 0 < level <= 1:
        raise ValueError(f"level must lie in (0, 1]; got {level}")
    if block_hours < MIN_BLOCK_HOURS:
        raise ValueError(
            f"block_hours must be at least {MIN_BLOCK_HOURS} (one week); got {block_hours}")

    origins = np.asarray(train_origins)
    span_start, span_end = training_span(origins)
    total_hours = int((span_end - span_start) / _HOUR)
    n_blocks = max(1, -(-total_hours // block_hours))  # ceiling division

    edges = [span_start + i * block_hours * _HOUR for i in range(n_blocks)]
    edges.append(span_end)

    order = np.random.default_rng(seed).permutation(n_blocks)

    if not calibrate:
        keep = min(max(int(round(level * n_blocks)), 1), n_blocks)
        return _assemble(order[:keep], edges, span_start, span_end,
                         erode, lookback, horizon)

    # Grow the retained set along the permutation until the realized share of
    # origins is as close to the target as it can get. Retained origins increase
    # with k, so the search can stop as soon as it reaches the target.
    total = origins.size
    best, best_err = None, None
    for keep in range(1, n_blocks + 1):
        blocks = _assemble(order[:keep], edges, span_start, span_end,
                           erode, lookback, horizon)
        realized = ablation_mask(origins, blocks).sum() / total if total else 0.0
        err = abs(realized - level)
        if best_err is None or err < best_err:
            best, best_err = blocks, err
        if realized >= level:
            break
    return best


def ablation_mask(origins, blocks):
    """Return the boolean mask selecting origins retained by an ablation draw."""
    origins = np.asarray(origins)
    mask = np.zeros(origins.shape, dtype=bool)
    for block in blocks:
        mask |= (origins >= block["origin_start"]) & (origins < block["origin_end"])
    return mask


def describe_draw(train_origins, blocks, level, draw, seed):
    """Summarize one draw: what was asked for against what it actually delivered.

    ``realized`` is the share of training origins the draw keeps, which sits
    below ``nominal`` because erosion trims each run. Reporting both is the point:
    the ablation curve should be plotted against the realized volume.
    """
    origins = np.asarray(train_origins)
    kept = int(ablation_mask(origins, blocks).sum())
    total = int(origins.size)
    retained_hours = sum(float((b["end"] - b["start"]) / _HOUR) for b in blocks)
    return {
        "level": level,
        "draw": draw,
        "seed": seed,
        "n_blocks": len(blocks),
        "nominal": level,
        "realized": (kept / total) if total else 0.0,
        "n_origins": kept,
        "n_origins_full": total,
        "retained_days": retained_hours / 24.0,
        "blocks": [{k: str(v) for k, v in b.items()} for b in blocks],
    }


def build_plan(train_origins, levels=ABLATION_LEVELS, draws=ABLATION_DRAWS,
               block_hours=DEFAULT_BLOCK_HOURS, erode=True, calibrate=True,
               seed_base=SEED_BASE, lookback=LOOKBACK_H, horizon=HORIZON_H):
    """Return every (level, draw) summary for an ablation plan.

    Three draws per level let an unlucky selection be told apart from a real
    effect, which a single draw at each level could not (Section 4.7.5).

    Draws that reproduce an earlier draw's blocks at the same level are dropped
    rather than emitted. At 100 percent there is nothing left to vary, so all
    draws are identical and only the first survives; emitting the others would
    schedule byte-identical training runs whose results could not differ. Draw
    numbers therefore stay tied to their seed and may be non-contiguous within a
    level, which is what keeps a surviving draw reproducible.
    """
    plan = []
    for level in levels:
        seen = set()
        for draw in range(1, draws + 1):
            seed = draw_seed(draw, seed_base)
            blocks = draw_blocks(train_origins, level, seed, block_hours,
                                 erode, calibrate, lookback, horizon)
            fingerprint = tuple((str(b["origin_start"]), str(b["origin_end"]))
                                for b in blocks)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            plan.append(describe_draw(train_origins, blocks, level, draw, seed))
    return plan


# ---------------------------------------------------------------------------
# Reading the manifest back.
#
# The manifest written by Ablation/main.py is the contract between the ablation
# plan and the trainers: every configuration at a given (level, draw) must train
# on the same blocks, or a difference between two configurations could be the
# draw rather than the transfer. These helpers are the only supported way to
# consume it, so no trainer has to re-parse the JSON or reconstruct timestamps
# and none can drift from another.
# ---------------------------------------------------------------------------

MANIFEST_NAME = "manifest.json"


def manifest_path(output_root, city_slug, source):
    """Return the manifest path for one (city, source)."""
    return os.path.join(output_root, "ablation", city_slug, source, MANIFEST_NAME)


def load_manifest(path):
    """Load an ablation manifest, restoring its timestamps to datetime64.

    The manifest stores times as ISO strings so the file stays readable; every
    boundary is converted back here so callers compare datetimes, never strings.

    Raises:
        FileNotFoundError: if the manifest has not been built yet, with the
            command that builds it.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No ablation manifest at {path}. Build it first with "
            f"python O2/Ablation/main.py --source <source> --city \"Metro Manila\""
        )
    with open(path) as f:
        manifest = json.load(f)

    for draw in manifest["draws"]:
        draw["blocks"] = [{k: np.datetime64(v) for k, v in block.items()}
                          for block in draw["blocks"]]
    return manifest


def manifest_cuts(manifest):
    """Return the chronological split cuts the manifest was built against.

    Lets a trainer derive its validation and test masks from the same record
    that defined the ablation, instead of recomputing the split and risking a
    different answer.
    """
    return np.datetime64(manifest["train_end"]), np.datetime64(manifest["val_end"])


def available_draws(manifest):
    """Return the (level, draw) pairs the manifest actually contains.

    Duplicate draws are dropped when the plan is built, so a level may carry
    fewer pairs than were requested; iterate this rather than assuming a grid.
    """
    return [(d["level"], d["draw"]) for d in manifest["draws"]]


def find_draw(manifest, level, draw):
    """Return one draw's record, raising with the available pairs if absent."""
    for record in manifest["draws"]:
        if record["level"] == level and record["draw"] == draw:
            return record
    raise KeyError(
        f"no (level={level}, draw={draw}) in this manifest; "
        f"available: {available_draws(manifest)}"
    )


def apply_ablation(origins, manifest, level, draw):
    """Return the mask selecting the training origins retained by one draw.

    This is what a trainer calls. Pass the origins of the samples it is about to
    train on and use the returned mask to drop the rest.

    Passing every origin, including validation and test, is safe: the manifest's
    blocks lie inside the training span, so an origin from a later partition
    matches no block and is excluded. The mask therefore both applies the
    ablation and enforces that only the training partition is reduced, which is
    what Section 4.7.5 requires. Validation and test masks come from
    ``manifest_cuts`` with ``splits.partition_mask``.
    """
    return ablation_mask(origins, find_draw(manifest, level, draw)["blocks"])
