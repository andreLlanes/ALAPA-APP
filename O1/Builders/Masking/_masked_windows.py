"""Masked-source window builders for the LSTM and GNN.

The exclusion builders (build_lstm_windows / build_gnn_windows) drop any window
with a NaN in a required column, which is correct for the clean source. For the
masked source these builders instead keep windows whose lookback pm25 is NaN
(exactly what the mask marks), while still dropping a window when:
    - the target (horizon pm25) has any NaN (cannot train against a missing target),
    - the encoder met/time features have any NaN (not what masking covers),
    - the decoder met/time features have any NaN, or
    - the window is not time-contiguous.

Both return an extra enc_mask of shape (n, lookback): 1 where the lookback pm25
was NaN, 0 elsewhere. pm25 stays NaN in X_enc; the trainer fills it with a neutral
placeholder after normalization.
"""

import numpy as np
import pandas as pd

from schema import KEY_COL, TIME_COL, PM25_COL, contiguous_step_mask
from schema import ENCODER_COLS, DECODER_COLS

LOOKBACK_H = 72
HORIZON_H = 72

_PM25_ENC_IDX = ENCODER_COLS.index(PM25_COL)
_ENC_NONPM_IDX = [i for i, c in enumerate(ENCODER_COLS) if c != PM25_COL]


def build_lstm_windows_masked(df, lookback=LOOKBACK_H, horizon=HORIZON_H, stride=1):
    """Slice masked LSTM windows, keeping those whose lookback pm25 is NaN.

    Returns X_enc, X_dec, Y, enc_mask, and a meta DataFrame (location_key, start,
    origin, end). See the module docstring for the retention rules and mask.
    """
    window_len = lookback + horizon
    Xe, Xd, Ys, masks, rows = [], [], [], [], []

    for key, g in df.groupby(KEY_COL, sort=False):
        g = g.sort_values(TIME_COL).reset_index(drop=True)
        enc_vals = g[ENCODER_COLS].to_numpy(dtype=float)
        dec_vals = g[DECODER_COLS].to_numpy(dtype=float)
        pm25 = g[PM25_COL].to_numpy(dtype=float)
        times = g[TIME_COL].to_numpy()

        enc_nonpm_ok = ~np.isnan(enc_vals[:, _ENC_NONPM_IDX]).any(axis=1)
        dec_ok = ~np.isnan(dec_vals).any(axis=1)
        pm_ok = ~np.isnan(pm25)
        step_ok = contiguous_step_mask(times)

        for start in range(0, len(g) - window_len + 1, stride):
            mid = start + lookback
            end = start + window_len
            if not enc_nonpm_ok[start:mid].all():
                continue
            if not dec_ok[mid:end].all():
                continue
            if not pm_ok[mid:end].all():
                continue
            if not step_ok[start + 1:end].all():
                continue

            block = enc_vals[start:mid]
            mask = ~np.isfinite(block[:, _PM25_ENC_IDX])
            Xe.append(block)
            Xd.append(dec_vals[mid:end])
            Ys.append(pm25[mid:end])
            masks.append(mask.astype(np.int8))
            rows.append({KEY_COL: key, "start": times[start],
                         "origin": times[mid - 1], "end": times[end - 1]})

    if not Xe:
        return (np.empty((0, lookback, len(ENCODER_COLS))),
                np.empty((0, horizon, len(DECODER_COLS))),
                np.empty((0, horizon)),
                np.empty((0, lookback), dtype=np.int8),
                pd.DataFrame(columns=[KEY_COL, "start", "origin", "end"]))

    return (np.stack(Xe), np.stack(Xd), np.stack(Ys),
            np.stack(masks), pd.DataFrame(rows))


def build_gnn_windows_masked(df, node_order, lookback=LOOKBACK_H,
                             horizon=HORIZON_H, stride=1):
    """Slice masked GNN windows, tagging each with its node index.

    Same retention rules and mask as build_lstm_windows_masked, but the meta
    DataFrame additionally carries a node_index so tensors align to graph nodes.
    """
    node_of = {k: i for i, k in enumerate(node_order)}
    window_len = lookback + horizon
    Xe, Xd, Ys, masks, rows = [], [], [], [], []

    for key, g in df.groupby(KEY_COL, sort=False):
        if key not in node_of:
            continue
        g = g.sort_values(TIME_COL).reset_index(drop=True)
        enc_vals = g[ENCODER_COLS].to_numpy(dtype=float)
        dec_vals = g[DECODER_COLS].to_numpy(dtype=float)
        pm25 = g[PM25_COL].to_numpy(dtype=float)
        times = g[TIME_COL].to_numpy()

        enc_nonpm_ok = ~np.isnan(enc_vals[:, _ENC_NONPM_IDX]).any(axis=1)
        dec_ok = ~np.isnan(dec_vals).any(axis=1)
        pm_ok = ~np.isnan(pm25)
        step_ok = contiguous_step_mask(times)

        for start in range(0, len(g) - window_len + 1, stride):
            mid = start + lookback
            end = start + window_len
            if not enc_nonpm_ok[start:mid].all():
                continue
            if not dec_ok[mid:end].all():
                continue
            if not pm_ok[mid:end].all():
                continue
            if not step_ok[start + 1:end].all():
                continue

            block = enc_vals[start:mid]
            mask = ~np.isfinite(block[:, _PM25_ENC_IDX])
            Xe.append(block)
            Xd.append(dec_vals[mid:end])
            Ys.append(pm25[mid:end])
            masks.append(mask.astype(np.int8))
            rows.append({KEY_COL: key, "node_index": node_of[key],
                         "start": times[start], "origin": times[mid - 1],
                         "end": times[end - 1]})

    if not Xe:
        return (np.empty((0, lookback, len(ENCODER_COLS))),
                np.empty((0, horizon, len(DECODER_COLS))),
                np.empty((0, horizon)),
                np.empty((0, lookback), dtype=np.int8),
                pd.DataFrame(columns=[KEY_COL, "node_index", "start", "origin", "end"]))

    return (np.stack(Xe), np.stack(Xd), np.stack(Ys),
            np.stack(masks), pd.DataFrame(rows))