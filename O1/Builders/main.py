"""Dataset builder entry point.

Reads a merged table (clean or masked) for a city and writes model-ready
per-station shards under Outputs/<model>/<citySlug>/<source>/. --source and
--city accept "all" to run every combination.

    python main.py --source masked --city "Metro Manila" --model lstm
    python main.py --source all --city all --model all

Datasets are unnormalized; train-only normalization happens at training time.
For GNN, k is deferred to training: the builder writes a nodes.npz with the graph
ingredients so the graph can be built at any k without rebuilding the dataset.
"""

import argparse

from common_build import ALL_SOURCES, ALL_CITIES
import lstm as lstm_builder
import gnn as gnn_builder
import gbt as gbt_builder

MODELS = {
    "lstm": lstm_builder.build,
    "gnn": gnn_builder.build,
    "gbt": gbt_builder.build,
}


def main():
    """Parse arguments and run each selected (model, city, source) combination."""
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, choices=["clean", "masked", "all"])
    p.add_argument("--city", required=True,
                   help='full city name (e.g. "Metro Manila") or "all"')
    p.add_argument("--model", required=True, choices=["lstm", "gnn", "gbt", "all"])
    args = p.parse_args()

    sources = ALL_SOURCES if args.source == "all" else [args.source]
    cities = ALL_CITIES if args.city == "all" else [args.city]
    models = list(MODELS) if args.model == "all" else [args.model]

    for model in models:
        for city in cities:
            for source in sources:
                MODELS[model](source, city)


if __name__ == "__main__":
    main()