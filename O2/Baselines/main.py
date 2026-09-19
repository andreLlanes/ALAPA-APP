"""Baseline entry point.

Scores the naive references of Section 4.8.3 on the window datasets built by O1
and writes per-fold, per-lead, and fold-averaged tables under
Outputs/<baseline>/<citySlug>/<source>/. --source, --city, and --baseline all
accept "all" to run every combination.

    python main.py --source masked --city "Metro Manila" --baseline persistence
    python main.py --source all --city all --baseline all

The O1 builders must have been run for a (city, source) first, since the shards
they write define the origins a baseline is scored on.
"""

import argparse

from common_baseline import ALL_SOURCES, ALL_CITIES
import persistence as persistence_baseline

BASELINES = {
    "persistence": persistence_baseline.run,
}


def main():
    """Parse arguments and run each selected (baseline, city, source) combination."""
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, choices=["clean", "masked", "all"])
    p.add_argument("--city", required=True,
                   help='full city name (e.g. "Metro Manila") or "all"')
    p.add_argument("--baseline", required=True,
                   choices=list(BASELINES) + ["all"])
    args = p.parse_args()

    sources = ALL_SOURCES if args.source == "all" else [args.source]
    cities = ALL_CITIES if args.city == "all" else [args.city]
    baselines = list(BASELINES) if args.baseline == "all" else [args.baseline]

    for baseline in baselines:
        for city in cities:
            for source in sources:
                try:
                    BASELINES[baseline](source, city)
                except FileNotFoundError as exc:
                    print(f"[{baseline}] {source}/{city}: skipped, {exc}")


if __name__ == "__main__":
    main()
