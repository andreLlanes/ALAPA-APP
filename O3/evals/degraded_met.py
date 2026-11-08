""" Forecast skill when the decoder's met inputs are degraded, for LSTM, GNN and GBT: all seven met
    variables get noise that grows with the lead, std = f x training std x (lead - 1) / 71 for each
    f in NOISE_LEVELS (physical limits kept), or are held at their origin-hour values for all 72
    hours. Noise is a fixed draw shared by every model.
    python -m evals.degraded_met --model lstm --city mm [--transfer true --source bk+la]
"""

import numpy as np

import sys
from pathlib import Path

# Put the O2 root on the import path, so this file runs as a script or as a module.
sys.path.append(str(Path(__file__).resolve().parents[1]))

import config
from Common.schema import HORIZON_H, LOOKBACK_H, MET_COLS
from evals.rescore import main

N_MET = len(MET_COLS)

def make_edits(data, ids) -> dict:
    """ Return the noise edits and the hold edit.
    """
    mean, std = data.stats.feat_mean[:N_MET], data.stats.feat_std[:N_MET]
    low, high = (np.array([config.MET_BOUNDS.get(c, (None, None))[i] for c in MET_COLS], float)
                 for i in (0, 1))
    low, high = np.nan_to_num(low, nan=-np.inf), np.nan_to_num(high, nan=np.inf)
    draws = np.random.default_rng(config.PERTURBATION_SEED).standard_normal(
        (config.NOISE_TABLE_SIZE, HORIZON_H, N_MET)).astype(np.float32)
    growth = (np.arange(HORIZON_H) / (HORIZON_H - 1))[:, None]

    def noisy(level):
        def edit(x_enc, x_dec, batch_ids):
            raw = (x_dec[..., :N_MET] * std + mean
                   + level * std * growth * draws[batch_ids % config.NOISE_TABLE_SIZE])
            x_dec[..., :N_MET] = (np.clip(raw, low, high) - mean) / std
            return x_enc, x_dec
        return edit

    def hold(x_enc, x_dec, batch_ids):
        x_dec[..., :N_MET] = data.feat[data.start[batch_ids] + LOOKBACK_H - 1, :N_MET][:, None]
        return x_enc, x_dec

    return {**{f"noise_{level}": noisy(level) for level in config.NOISE_LEVELS}, "hold": hold}

if __name__ == "__main__":
    main("degraded_met", ("lstm", "gnn", "gbt"), make_edits)
