""" Split each station's windows chronologically: 70/15/15, purged by the forecast horizon.
"""

import numpy as np

from Common.schema import HORIZON_H

TRAIN, VAL, TEST, PURGED = 0, 1, 2, -1
CUTS = (0.70, 0.85)


def station_split(station: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """ Label each window TRAIN/VAL/TEST/PURGED: per station, in time order, 70% train,
        15% val, 15% test. Windows must be sorted by station, then origin (in hours).
    """
    if np.any(np.diff(station) < 0):
        raise ValueError("windows must be sorted by station")
    _, first, count = np.unique(station, return_index=True, return_counts=True)
    group = np.repeat(np.arange(len(first)), count)
    pos = np.arange(len(station)) - first[group]  # position within the station
    val_start = (count * CUTS[0]).astype(int)[group]
    test_start = (count * CUTS[1]).astype(int)[group]

    label = np.where(pos < val_start, TRAIN, np.where(pos < test_start, VAL, TEST))
    # Purge: a val/test window whose origin is within 72h of the previous part's last origin
    # would forecast hours that part already covers.
    last_train = origin[first[group] + np.maximum(val_start - 1, 0)]
    last_val = origin[first[group] + np.maximum(test_start - 1, 0)]
    boundary = np.where(label == VAL, last_train, last_val)
    label[(label != TRAIN) & (origin < boundary + HORIZON_H)] = PURGED
    return label
