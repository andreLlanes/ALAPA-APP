"""Baseline entry point.

Scores the naive references of Section 4.8.3 on the window datasets built by O1
and writes per-fold, per-lead, and fold-averaged tables under
Outputs/<baseline>/<citySlug>/<source>/fixed/<split>/. --source, --city and
--baseline accept "all" to run every combination.

The baselines are scored on the test partition of the chronological 70/15/15
split only, with the dates frozen in O2/common/split_dates.json; that is where the
skill score is computed. Rolling origin and leave-one-station-out are model-only.

    python main.py --source clean --city "Metro Manila" --baseline persistence
    python main.py --source all --city all --baseline all

The O1 builders must have been run for a (city, source) first, since the shards
they write define the origins a baseline is scored on.
"""

import argparse

from common_baseline import ALL_SOURCES, ALL_CITIES
from splits import PARTITIONS
import persistence as persistence_baseline

BASELINES = {
    "persistence": persistence_baseline.run,
}


def main():
    """Parse arguments and run each selected combination."""
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, choices=["clean", "masked", "all"])
    p.add_argument("--city", required=True,
                   help='full city name (e.g. "Metro Manila") or "all"')
    p.add_argument("--baseline", required=True,
                   choices=list(BASELINES) + ["all"])
    p.add_argument("--split", default="test",
                   choices=list(PARTITIONS) + ["all"],
                   help="partition of the chronological split to score (default: test)")
    args = p.parse_args()

    sources = ALL_SOURCES if args.source == "all" else [args.source]
    cities = ALL_CITIES if args.city == "all" else [args.city]
    baselines = list(BASELINES) if args.baseline == "all" else [args.baseline]

    for baseline in baselines:
        for city in cities:
            for source in sources:
                try:
                    BASELINES[baseline](source, city, split=args.split)
                except (FileNotFoundError, KeyError, ValueError) as exc:
                    print(f"[{baseline}] {source}/{city}: skipped, {exc}")


if __name__ == "__main__":
    main()
