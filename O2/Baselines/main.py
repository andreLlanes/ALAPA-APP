"""Baseline entry point.

Scores the naive references of Section 4.8.3 on the window datasets built by O1
and writes per-fold, per-lead, and fold-averaged tables under
Outputs/<baseline>/<citySlug>/<source>/<protocol>/<window>/. --source, --city,
--baseline, and --protocol all accept "all" to run every combination.

Two temporal protocols of Section 4.7.3 are available, and both are normally
run:

    fixed   - the test partition of the chronological 70/15/15 split.
    rolling - four origins advancing by three months, each scored on the three
              months that follow.

    python main.py --source masked --city "Metro Manila" --baseline persistence
    python main.py --source all --city all --baseline all --protocol all

The O1 builders must have been run for a (city, source) first, since the shards
they write define the origins a baseline is scored on.
"""

import argparse

from common_baseline import ALL_SOURCES, ALL_CITIES
from splits import PARTITIONS, ROLLING_ORIGINS, ROLLING_TEST_MONTHS
import persistence as persistence_baseline

BASELINES = {
    "persistence": persistence_baseline.run,
}
PROTOCOLS = list(persistence_baseline.PROTOCOLS)


def main():
    """Parse arguments and run each selected combination."""
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, choices=["clean", "masked", "all"])
    p.add_argument("--city", required=True,
                   help='full city name (e.g. "Metro Manila") or "all"')
    p.add_argument("--baseline", required=True,
                   choices=list(BASELINES) + ["all"])
    p.add_argument("--protocol", default="fixed",
                   choices=PROTOCOLS + ["all"],
                   help="temporal evaluation design (default: fixed)")
    p.add_argument("--split", default="test",
                   choices=list(PARTITIONS) + ["all"],
                   help="partition to score under the fixed protocol (default: test)")
    p.add_argument("--origins", type=int, default=ROLLING_ORIGINS, dest="n_origins",
                   help="number of rolling origins (default: 4)")
    p.add_argument("--test-months", type=int, default=ROLLING_TEST_MONTHS,
                   dest="test_months",
                   help="length of each rolling test period, in months (default: 3)")
    args = p.parse_args()

    sources = ALL_SOURCES if args.source == "all" else [args.source]
    cities = ALL_CITIES if args.city == "all" else [args.city]
    baselines = list(BASELINES) if args.baseline == "all" else [args.baseline]
    protocols = PROTOCOLS if args.protocol == "all" else [args.protocol]

    for baseline in baselines:
        for city in cities:
            for source in sources:
                for protocol in protocols:
                    try:
                        BASELINES[baseline](source, city, protocol=protocol,
                                            split=args.split,
                                            n_origins=args.n_origins,
                                            test_months=args.test_months)
                    except FileNotFoundError as exc:
                        print(f"[{baseline}] {source}/{city} [{protocol}]: "
                              f"skipped, {exc}")
                    except ValueError as exc:
                        print(f"[{baseline}] {source}/{city} [{protocol}]: "
                              f"skipped, {exc}")


if __name__ == "__main__":
    main()
