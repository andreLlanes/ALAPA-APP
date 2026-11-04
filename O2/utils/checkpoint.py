""" Save and resume training checkpoints, so an interrupted run continues where it stopped.
"""

import os
from pathlib import Path

import torch

def save_checkpoint(path, state: dict):
    """ Write a checkpoint atomically: a crash mid-write never leaves a corrupt file.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)

def load_checkpoint(path):
    """ Load a checkpoint, or return None when there is none yet.
    """
    if path is None or not os.path.exists(path):
        return None
    return torch.load(path, weights_only=False)
