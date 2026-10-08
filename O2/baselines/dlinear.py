""" DLinear baseline: the PM2.5 lookback is split into a moving-average trend and a remainder, and
    each is mapped to the 72-hour forecast by one linear layer. Trained, tuned and scored like the
    LSTM, for every seed, by train.py's plain run.
    python -m baselines.dlinear --city mm --longest-gap 6 --completeness 70
"""

import argparse

import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path

# Put the O2 root on the import path, so this file runs as a script or as a module.
sys.path.append(str(Path(__file__).resolve().parents[1]))

import config
from Common.schema import HORIZON_H, LOOKBACK_H
from data.database import CITIES
from utils import runlog

class DLinear(nn.Module):
    """ Forecast [n,72] from the PM2.5 lookback alone (encoder column 0).
    """

    def __init__(self, kernel: int):
        super().__init__()
        self.autoregressive = False
        self.kernel = kernel
        self.trend = nn.Linear(LOOKBACK_H, HORIZON_H)
        self.remainder = nn.Linear(LOOKBACK_H, HORIZON_H)

    def forward(self, x_enc, **_):
        """ Return (prediction, the lookback as the representation).
        """
        pm = x_enc[..., 0]
        pad = (self.kernel - 1) // 2
        trend = F.avg_pool1d(F.pad(pm[:, None], (pad, pad), mode="replicate"), self.kernel, 1)[:, 0]
        return self.trend(trend) + self.remainder(pm - trend), pm

def main():
    """ Run the plain training run for DLinear (train is imported here because it imports this file).
    """
    import train
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=list(CITIES))
    ap.add_argument("--longest-gap", type=int, default=config.DEFAULT_LONGEST_GAP,
                    choices=config.LONGEST_GAP_CHOICES)
    ap.add_argument("--completeness", type=int, default=config.DEFAULT_COMPLETENESS,
                    choices=config.COMPLETENESS_CHOICES)
    ap.add_argument("--seed", type=lambda t: [int(s) for s in t.split(",")],
                    default=config.DEFAULT_SEEDS)
    ap.add_argument("--device", default=config.DEFAULT_DEVICE)
    args = argparse.Namespace(**vars(ap.parse_args()), model="dlinear", transfer=False, source="",
                              sources=[], autoregressive=False, training_variant="all",
                              data_ablation=100, block="0")
    with runlog.logged(f"dlinear_{args.city}", args):
        train.run(args)

if __name__ == "__main__":
    main()
