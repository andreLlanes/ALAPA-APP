""" Permutation importance of the LSTM and GNN inputs: each input group is shuffled across the test
    windows (every window takes another window's values of that group, in the encoder and the
    decoder) and the RMSE increase is reported: PM2.5 history, each met variable, time encodings.
    python -m evals.permutation --model lstm --city mm [--transfer true --source bk+la]
"""

import numpy as np

import sys
from pathlib import Path

# Put the O2 root on the import path, so this file runs as a script or as a module.
sys.path.append(str(Path(__file__).resolve().parents[1]))

import config
from Common.schema import DECODER_COLS, ENCODER_COLS, MET_COLS, TIME_COLS
from evals.rescore import main

GROUPS = {"pm25": ([ENCODER_COLS[0]], []),
          **{name: ([name], [name]) for name in MET_COLS},
          "time": (TIME_COLS, TIME_COLS)}

def shuffled(data, ids, order, encoder, decoder):
    """ Return the edit that gives window ids[i] the inputs of window ids[order[i]].
    """
    def edit(x_enc, x_dec, batch_ids):
        donor_enc, donor_dec, _ = data.gather(ids[order[np.searchsorted(ids, batch_ids)]])
        x_enc[..., encoder] = donor_enc[..., encoder]
        x_dec[..., decoder] = donor_dec[..., decoder]
        return x_enc, x_dec
    return edit

def make_edits(data, ids) -> dict:
    """ Return one shuffling edit per input group, each with its own fixed permutation.
    """
    return {name: shuffled(data, ids,
                           np.random.default_rng([config.PERTURBATION_SEED, i]).permutation(len(ids)),
                           [ENCODER_COLS.index(c) for c in encoder],
                           [DECODER_COLS.index(c) for c in decoder])
            for i, (name, (encoder, decoder)) in enumerate(GROUPS.items())}

if __name__ == "__main__":
    main("permutation", ("lstm", "gnn"), make_edits)
