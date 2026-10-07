"""LSTM dataset builder: windowing and per-station shard output.

The encoder sees the full 72h past (pm25 + met + time); the decoder sees met +
time only over the 72h horizon (non-autoregressive). Writes one float32 .npz per
station to Outputs/lstm/<citySlug>/<source>/<station>.npz.

    clean source  -> build_lstm_windows (exclusion: drop any window with a NaN)
    masked source -> build_lstm_windows_masked (keep lookback-pm25 NaN + emit mask)

Clean shards also carry Y_filled ((n, 72) bool) and origin_filled ((n,) bool),
marking interpolated target and origin hours. Filled targets are estimates, so
scoring and training losses should skip them.

Datasets are unnormalized; train-only normalization happens at training time.
"""

import os

import numpy as np
import pandas as pd

from common_build import station_keys, load_station, shard_dir, safe_name  # sets sys.path
from schema import (
    KEY_COL, TIME_COL, PM25_COL, ENCODER_COLS, DECODER_COLS,
    LOOKBACK_H, HORIZON_H, exclusion_window_starts, filled_flags,
)
from _masked_windows import build_lstm_windows_masked


def build_lstm_windows(df, lookback=LOOKBACK_H, horizon=HORIZON_H, stride=1,
                       return_filled=False):
    """Slice a station's frame into exclusion windows (no NaN anywhere).

    A window is emitted only when the encoder features, decoder features, target,
    and step contiguity are all valid across its span; the rule itself lives in
    schema.exclusion_window_starts so the baselines select the same windows.

    Returns:
        X_enc: (n, lookback, |ENCODER_COLS|) past features.
        X_dec: (n, horizon, |DECODER_COLS|) known-future features.
        Y: (n, horizon) target pm25.
        meta: DataFrame with location_key, start, origin, end per window.
        With return_filled=True, also Y_filled ((n, horizon) bool) and
        origin_filled ((n,) bool): which targets and origins were interpolated.
    """
    window_len = lookback + horizon
    Xe, Xd, Ys, rows, Yf, Of = [], [], [], [], [], []

    for key, g in df.groupby(KEY_COL, sort=False):
        g = g.sort_values(TIME_COL).reset_index(drop=True)
        enc_vals = g[ENCODER_COLS].to_numpy(dtype=float)
        dec_vals = g[DECODER_COLS].to_numpy(dtype=float)
        pm25 = g[PM25_COL].to_numpy(dtype=float)
        times = g[TIME_COL].to_numpy()

        starts = exclusion_window_starts(g, lookback, horizon)
        starts = starts[starts % stride == 0]
        if starts.size:
            target_filled, origin_filled = filled_flags(g, starts, lookback, horizon)
            Yf.append(target_filled)
            Of.append(origin_filled)

        for start in starts:
            mid = start + lookback
            end = start + window_len
            Xe.append(enc_vals[start:mid])
            Xd.append(dec_vals[mid:end])
            Ys.append(pm25[mid:end])
            rows.append({KEY_COL: key, "start": times[start],
                         "origin": times[mid - 1], "end": times[end - 1]})

    if not Xe:
        out = (np.empty((0, lookback, len(ENCODER_COLS))),
               np.empty((0, horizon, len(DECODER_COLS))),
               np.empty((0, horizon)),
               pd.DataFrame(columns=[KEY_COL, "start", "origin", "end"]))
        filled = (np.empty((0, horizon), dtype=bool), np.empty(0, dtype=bool))
    else:
        out = (np.stack(Xe), np.stack(Xd), np.stack(Ys), pd.DataFrame(rows))
        filled = (np.concatenate(Yf), np.concatenate(Of))
    return out + filled if return_filled else out


def build(source, city):
    """Window every station in a (source, city) and write one npz shard each."""
    keys = station_keys(source, city)
    outdir = shard_dir("lstm", city, source)
    total_windows = 0
    written = 0
    for location_key in keys:
        df = load_station(source, location_key)
        if df.empty:
            continue
        Y_filled = origin_filled = None
        if source == "clean":
            X_enc, X_dec, Y, meta, Y_filled, origin_filled = build_lstm_windows(
                df, return_filled=True)
            enc_mask = None
        else:
            X_enc, X_dec, Y, enc_mask, meta = build_lstm_windows_masked(df)
        if X_enc.shape[0] == 0:
            continue

        payload = {
            "X_enc": X_enc.astype(np.float32),
            "X_dec": X_dec.astype(np.float32),
            "Y": Y.astype(np.float32),
            "encoder_cols": np.array(ENCODER_COLS),
            "decoder_cols": np.array(DECODER_COLS),
            "meta_location_key": meta["location_key"].to_numpy().astype(str),
            "meta_origin": meta["origin"].dt.tz_localize(None).to_numpy().astype("datetime64[ns]"),
        }
        if Y_filled is not None:
            # Interpolated hours: skip filled targets when scoring or in the loss.
            payload["Y_filled"] = Y_filled
            payload["origin_filled"] = origin_filled
        if enc_mask is not None:
            payload["enc_mask"] = enc_mask.astype(np.int8)

        np.savez_compressed(os.path.join(outdir, f"{safe_name(location_key)}.npz"), **payload)
        total_windows += X_enc.shape[0]
        written += 1

    print(f"[lstm] {source}/{city}: {written} station shards, {total_windows} windows -> {outdir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, choices=["clean", "masked"])
    p.add_argument("--city", required=True)
    build(**vars(p.parse_args()))