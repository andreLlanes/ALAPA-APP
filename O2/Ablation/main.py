"""Ablation manifest builder.

Writes the training-data ablation plan of Section 4.7.5: for each level (25, 50,
75, 100 percent) and each of three draws, the contiguous blocks of the training
partition that survive.

The manifest is the contract between this stage and the trainers. Every model
configuration at a given (level, draw) must read the same blocks, otherwise the
Bangkok-versus-no-transfer comparison at that level is not controlled and any
difference could be the draw rather than the transfer. Writing the plan once,
here, is what makes that guarantee checkable.

Ablation is defined on the training partition of the chronological 70/15/15 split
(Section 4.7.3). It is not applied to the rolling origin: that would multiply the
experiment by the number of origins, which Section 4.7.5 does not ask for.
Validation and test partitions are never touched.

Blocks are drawn over the pooled origins of all stations, so the same calendar
stretches are removed everywhere. That is what the hypothesis is about: a shorter
monitoring record for the city, not a different record per station.

Only the target city is ablated. H5 asks how transfer benefit varies with the
volume of *Metro Manila* training data; the source cities keep their full record,
since ablating them would change what is being transferred rather than how much
local data it is being transferred to. Another city is therefore refused unless
--any-city is passed, so a stray --city all cannot quietly produce manifests that
mean nothing.

    python main.py --source masked --city "Metro Manila"
    python main.py --source all --city "Metro Manila" --block-hours 336

Outputs, under O2/Outputs/ablation/<citySlug>/<source>/:
    manifest.json - the full plan: blocks, seeds, and realized volumes.
    summary.csv   - one row per (level, draw), for eyeballing nominal vs realized.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_O2_ROOT = os.path.dirname(_HERE)

# Shard discovery lives with the baselines; both stages read the same O1 output,
# so it is imported rather than duplicated here.
for _p in (os.path.join(_O2_ROOT, "common"), os.path.join(_O2_ROOT, "Baselines")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common_baseline import (  # noqa: E402  (path bootstrap must run first)
    ALL_CITIES, ALL_SOURCES, result_dir, shard_paths, load_origins,
)
from splits import chronological_cuts, partition_mask  # noqa: E402
from ablation import (  # noqa: E402
    ABLATION_LEVELS, ABLATION_DRAWS, DEFAULT_BLOCK_HOURS, MIN_BLOCK_HOURS,
    SEED_BASE, build_plan, training_span,
)
from schema import LOOKBACK_H, HORIZON_H  # noqa: E402

STAGE = "ablation"
TARGET_CITY = "Metro Manila"


def training_origins(paths):
    """Return the pooled origins falling in the chronological training partition.

    Reads only the origin arrays, so the plan costs kilobytes per station rather
    than loading any window tensors.
    """
    collected = [o for o in (load_origins(p) for p in paths) if o.size]
    if not collected:
        raise ValueError("no origins found in any shard")
    origins = np.concatenate(collected)
    cuts = chronological_cuts(origins)
    return origins[partition_mask(origins, cuts, "train")], cuts


def build(source, city, levels=ABLATION_LEVELS, draws=ABLATION_DRAWS,
          block_hours=DEFAULT_BLOCK_HOURS, erode=True, calibrate=True,
          seed_base=SEED_BASE):
    """Build and write the ablation manifest for one (source, city)."""
    paths = shard_paths(city, source)
    train_origins, cuts = training_origins(paths)
    span_start, span_end = training_span(train_origins)

    plan = build_plan(train_origins, levels, draws, block_hours, erode, calibrate,
                      seed_base, LOOKBACK_H, HORIZON_H)

    outdir = result_dir(STAGE, city, source)
    manifest = {
        "city": city,
        "source": source,
        "protocol": "fixed",
        "partition": "train",
        "block_hours": block_hours,
        "min_block_hours": MIN_BLOCK_HOURS,
        "erode": erode,
        "calibrate": calibrate,
        "lookback_h": LOOKBACK_H,
        "horizon_h": HORIZON_H,
        "seed_base": seed_base,
        "train_end": str(cuts[0]),
        "val_end": str(cuts[1]),
        "train_span": {"start": str(span_start), "end": str(span_end),
                       "n_origins": int(train_origins.size)},
        "draws": plan,
    }
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    summary = pd.DataFrame([{k: d[k] for k in
                             ("level", "draw", "seed", "n_blocks", "nominal",
                              "realized", "n_origins", "n_origins_full",
                              "retained_days")}
                            for d in plan])
    summary.to_csv(os.path.join(outdir, "summary.csv"), index=False)

    weeks = block_hours / 168
    print(f"[{STAGE}] {source}/{city}: training partition "
          f"{str(span_start)[:10]} to {str(span_end)[:10]}, "
          f"{train_origins.size} origins")
    print(f"  blocks of {block_hours} h ({weeks:.0f} week"
          f"{'s' if weeks != 1 else ''}), erosion "
          f"{'on' if erode else 'OFF (retained blocks will leak into discarded ones)'}"
          f", up to {draws} draws per level, levels targeting "
          f"{'retained data volume' if calibrate else 'block count'}")
    for level in levels:
        rows = summary[summary["level"] == level]
        n = len(rows)
        dup = draws - n
        note = f" [{dup} duplicate draw{'s' if dup > 1 else ''} dropped]" if dup else ""
        print(f"  level {level:.0%}: {n} draw{'s' if n != 1 else ''}, realized "
              f"{rows['realized'].min():.1%}-{rows['realized'].max():.1%} "
              f"(mean {rows['realized'].mean():.1%}), "
              f"{rows['n_origins'].min()}-{rows['n_origins'].max()} origins, "
              f"{rows['n_blocks'].min()}-{rows['n_blocks'].max()} blocks{note}")
    print(f"  -> {outdir}")


def main():
    """Parse arguments and build the manifest for each selected combination."""
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, choices=["clean", "masked", "all"])
    p.add_argument("--city", default=TARGET_CITY,
                   help=f'target city to ablate (default: "{TARGET_CITY}")')
    p.add_argument("--any-city", action="store_true", dest="any_city",
                   help="allow ablating a city other than the transfer target")
    p.add_argument("--levels", type=float, nargs="+", default=list(ABLATION_LEVELS),
                   help="training-volume levels (default: 0.25 0.5 0.75 1.0)")
    p.add_argument("--draws", type=int, default=ABLATION_DRAWS,
                   help="block draws per level (default: 3)")
    p.add_argument("--block-hours", type=int, default=DEFAULT_BLOCK_HOURS,
                   dest="block_hours",
                   help=f"candidate block length in hours "
                        f"(default: {DEFAULT_BLOCK_HOURS}, floor: {MIN_BLOCK_HOURS})")
    p.add_argument("--no-erode", action="store_false", dest="erode",
                   help="keep origins whose window reaches into a discarded block")
    p.add_argument("--no-calibrate", action="store_false", dest="calibrate",
                   help="treat levels as a share of blocks, not of retained data")
    p.add_argument("--seed", type=int, default=SEED_BASE, dest="seed_base",
                   help="base seed for the draws")
    args = p.parse_args()

    sources = ALL_SOURCES if args.source == "all" else [args.source]
    cities = ALL_CITIES if args.city == "all" else [args.city]

    if not args.any_city:
        rejected = [c for c in cities if c != TARGET_CITY]
        if rejected:
            print(f"[{STAGE}] refusing {', '.join(rejected)}: Section 4.7.5 ablates "
                  f"the transfer target only. Pass --any-city to override.")
        cities = [c for c in cities if c == TARGET_CITY]
        if not cities:
            return

    for city in cities:
        for source in sources:
            try:
                build(source, city, args.levels, args.draws, args.block_hours,
                      args.erode, args.calibrate, args.seed_base)
            except (FileNotFoundError, ValueError) as exc:
                print(f"[{STAGE}] {source}/{city}: skipped, {exc}")


if __name__ == "__main__":
    main()
