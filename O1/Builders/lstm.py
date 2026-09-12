"""LSTM dataset builder: windowing and per-station shard output.

The encoder sees the full 72h past (pm25 + met + time); the decoder sees met +
time only over the 72h horizon (non-autoregressive). Writes one float32 .npz per
station to Outputs/lstm/<citySlug>/<source>/<station>.npz.

    clean source  -> build_lstm_windows (exclusion: drop any window with a NaN)
    masked source -> build_lstm_windows_masked (keep lookback-pm25 NaN + emit mask)

Datasets are unnormalized; train-only normalization happens at training time.
"""

import os

import numpy as np
import pandas as pd

from common_build import station_keys, load_station, shard_dir, safe_name  # sets sys.path
from schema import (
    KEY_COL, TIME_COL, PM25_COL, ENCODER_COLS, DECODER_COLS,
    LOOKBACK_H, HORIZON_H, contiguous_step_mask,
)
from _masked_windows import build_lstm_windows_masked


def build_lstm_windows(df, lookback=LOOKBACK_H, horizon=HORIZON_H, stride=1):
    """Slice a station's frame into exclusion windows (no NaN anywhere).

    A window is emitted only when the encoder features, decoder features, target,
    and step contiguity are all valid across its span.

    Returns:
        X_enc: (n, lookback, |ENCODER_COLS|) past features.
        X_dec: (n, horizon, |DECODER_COLS|) known-future features.
        Y: (n, horizon) target pm25.
        meta: DataFrame with location_key, start, origin, end per window.
    """
    window_len = lookback + horizon
    Xe, Xd, Ys, rows = [], [], [], []

    for key, g in df.groupby(KEY_COL, sort=False):
        g = g.sort_values(TIME_COL).reset_index(drop=True)
        enc_vals = g[ENCODER_COLS].to_numpy(dtype=float)
        dec_vals = g[DECODER_COLS].to_numpy(dtype=float)
        pm25 = g[PM25_COL].to_numpy(dtype=float)
        times = g[TIME_COL].to_numpy()

        enc_ok = ~np.isnan(enc_vals).any(axis=1)
        dec_ok = ~np.isnan(dec_vals).any(axis=1)
        pm_ok = ~np.isnan(pm25)
        step_ok = contiguous_step_mask(times)

        for start in range(0, len(g) - window_len + 1, stride):
            mid = start + lookback
            end = start + window_len
            if not enc_ok[start:mid].all():
                continue
            if not dec_ok[mid:end].all():
                continue
            if not pm_ok[mid:end].all():
                continue
            if not step_ok[start + 1:end].all():
                continue
            Xe.append(enc_vals[start:mid])
            Xd.append(dec_vals[mid:end])
            Ys.append(pm25[mid:end])
            rows.append({KEY_COL: key, "start": times[start],
                         "origin": times[mid - 1], "end": times[end - 1]})

    if not Xe:
        return (np.empty((0, lookback, len(ENCODER_COLS))),
                np.empty((0, horizon, len(DECODER_COLS))),
                np.empty((0, horizon)),
                pd.DataFrame(columns=[KEY_COL, "start", "origin", "end"]))
    return np.stack(Xe), np.stack(Xd), np.stack(Ys), pd.DataFrame(rows)


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
        if source == "clean":
            X_enc, X_dec, Y, meta = build_lstm_windows(df)
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