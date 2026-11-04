""" Command-line arguments shared by train.py and the analyses: which configuration to run or read,
    and where its runs are saved.
"""

import argparse

import config
from data.database import CITIES
from models import transfer
from utils import runlog
from utils.artifacts import config_dir, settings_tag

def boolean(text: str) -> bool:
    """ Parse true/false arguments.
    """
    if text.lower() not in ("true", "false"):
        raise argparse.ArgumentTypeError("expected true or false")
    return text.lower() == "true"

def add_arguments(ap):
    """ Add the arguments that identify one configuration.
    """
    ap.add_argument("--model", required=True, choices=["lstm", "gnn", "gbt"])
    ap.add_argument("--city", required=True, choices=list(CITIES))
    ap.add_argument("--transfer", type=boolean, default=False)
    ap.add_argument("--source", help="source city, or cities pooled as one source, e.g. bk+la")
    ap.add_argument("--training-variant", default="all",
                    choices=sorted({*transfer.NN_VARIANTS, *transfer.GBT_VARIANTS}) + ["all"])
    ap.add_argument("--autoregressive", type=boolean, default=False)
    ap.add_argument("--longest-gap", type=int, default=config.DEFAULT_LONGEST_GAP,
                    choices=config.LONGEST_GAP_CHOICES)
    ap.add_argument("--completeness", type=int, default=config.DEFAULT_COMPLETENESS,
                    choices=config.COMPLETENESS_CHOICES)
    ap.add_argument("--data-ablation", type=int, default=config.DEFAULT_ABLATION,
                    choices=config.ABLATION_CHOICES)
    ap.add_argument("--block", default=config.DEFAULT_BLOCK, choices=config.BLOCK_CHOICES)
    ap.add_argument("--device", default=config.DEFAULT_DEVICE)

def read_arguments(ap):
    """ Parse and check the command line; args.sources lists the source cities.
    """
    args = ap.parse_args()
    args.sources = sorted(set(args.source.split("+"))) if args.source else []
    args.source = "+".join(args.sources)
    if args.transfer and (not args.sources or args.city in args.sources
                          or not set(args.sources) <= set(CITIES)):
        ap.error("--transfer true needs --source: other cities, joined with +, e.g. bk+la")
    allowed = transfer.GBT_VARIANTS if args.model == "gbt" else transfer.NN_VARIANTS
    if args.training_variant not in allowed + ("all",):
        ap.error(f"--training-variant for {args.model}: {', '.join(allowed)} or all")
    return args

def names(codes) -> str:
    """ Return city names for city codes, joined with +, e.g. Bangkok + Los Angeles.
    """
    return " + ".join(CITIES[c] for c in ([codes] if isinstance(codes, str) else codes))

def blocks(args) -> list:
    """ Return the ablation blocks to run: block 0 alone at full data.
    """
    return [0] if args.data_ablation == 100 else (
        [0, 1, 2] if args.block == "all" else [int(args.block)])

def variants(args) -> list:
    """ Return the variants of the configuration that args selects.
    """
    if not args.transfer:
        return ["none"]
    names_ = transfer.GBT_VARIANTS if args.model == "gbt" else transfer.NN_VARIANTS
    return list(names_) if args.training_variant == "all" else [args.training_variant]

def config_folder(args, block):
    """ Return the folder of the configuration's runs for one ablation block.
    """
    tag = settings_tag(args.longest_gap, args.completeness, args.data_ablation, block)
    return config_dir(args.model, args.city, args.source if args.transfer else None, args.autoregressive, tag)

def run_baseline(name, run):
    """ Run a baseline from the command line (--city, --longest-gap, --completeness) under its
        own log.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=list(CITIES))
    ap.add_argument("--longest-gap", type=int, default=config.DEFAULT_LONGEST_GAP,
                    choices=config.LONGEST_GAP_CHOICES)
    ap.add_argument("--completeness", type=int, default=config.DEFAULT_COMPLETENESS,
                    choices=config.COMPLETENESS_CHOICES)
    args = ap.parse_args()
    with runlog.logged(f"{name}_{args.city}", args):
        run(args.city, args.longest_gap, args.completeness)
