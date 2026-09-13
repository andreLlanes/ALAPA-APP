"""GNN dataset builder: windowing and per-station shard output.

Writes per-station float32 shards to Outputs/gnn/<citySlug>/<source>/<station>.npz
and a nodes.npz (node order + coordinates) in the same folder.

The graph is NOT built here. k (neighbors per node) is a hyperparameter tuned at
training time, so the builder emits only the graph ingredients (coordinates); the
graph is constructed at training for each k via build_graph, which is kept in this
module for training to import.

    clean source  -> build_gnn_windows (exclusion)
    masked source -> build_gnn_windows_masked (keep lookback-pm25 NaN + emit mask)

Station coordinates come from the merged table itself. Datasets are unnormalized.
"""

import os

import numpy as np
import pandas as pd

from common_build import load_station, station_coords, shard_dir, safe_name  # sets sys.path
from schema import (
    KEY_COL, TIME_COL, PM25_COL, ENCODER_COLS, DECODER_COLS,
    LOOKBACK_H, HORIZON_H, contiguous_step_mask,
)
from _masked_windows import build_gnn_windows_masked

EARTH_RADIUS_KM = 6371.0088


def haversine_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Return the pairwise great-circle distance matrix (km), zero on the diagonal."""
    lat_r = np.radians(lat)[:, None]
    lon_r = np.radians(lon)[:, None]
    dlat = lat_r - lat_r.T
    dlon = lon_r - lon_r.T
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_r) * np.cos(lat_r.T) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_graph(stations: pd.DataFrame, node_order: list[str], k: int,
                sigma: float | None = None):
    """Build the symmetrized, Gaussian-weighted k-NN station graph.

    Each node connects to its k nearest neighbors by geodesic distance; the
    directed k-NN set is symmetrized by union so message passing is bidirectional,
    and edges are weighted by a Gaussian kernel of their distance. This is called
    at training time, once per candidate k.

    Args:
        stations: DataFrame with location_key, latitude, longitude.
        node_order: Ordered location_keys defining node indices 0..N-1; must match
            the node_index carried by the window shards.
        k: Neighbors per node.
        sigma: Kernel bandwidth. Defaults to the mean of all directed k-NN edge
            distances (the network's own neighbor-distance scale).

    Returns:
        A dict with edge_index ((2, E) int, symmetric), edge_weight ((E,) float),
        node_order, the sigma used, and dist (the (N, N) distance matrix in km).
    """
    coords = (stations.drop_duplicates(KEY_COL)
              .set_index(KEY_COL).loc[node_order])
    lat = coords["latitude"].to_numpy(dtype=float)
    lon = coords["longitude"].to_numpy(dtype=float)
    N = len(node_order)
    if k >= N:
        raise ValueError(f"k={k} must be < number of stations ({N})")

    dist = haversine_matrix(lat, lon)
    np.fill_diagonal(dist, np.inf)

    knn_idx = np.argsort(dist, axis=1)[:, :k]
    src = np.repeat(np.arange(N), k)
    dst = knn_idx.reshape(-1)
    knn_dist = dist[src, dst]

    if sigma is None:
        sigma = float(knn_dist.mean())
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    directed = set(zip(src.tolist(), dst.tolist()))
    undirected = set()
    for i, j in directed:
        undirected.add((i, j))
        undirected.add((j, i))

    edges = np.array(sorted(undirected), dtype=int).T
    d = dist[edges[0], edges[1]]
    w = np.exp(-(d ** 2) / (2 * sigma ** 2))

    return {
        "edge_index": edges,
        "edge_weight": w.astype(float),
        "node_order": list(node_order),
        "sigma": sigma,
        "dist": dist,
    }


def build_gnn_windows(df, node_order, lookback=LOOKBACK_H, horizon=HORIZON_H,
                      stride=1):
    """Slice exclusion windows (as in the LSTM) and tag each with its node index.

    Returns X_enc, X_dec, Y, and a meta DataFrame that additionally carries a
    node_index column so tensors align to graph nodes.
    """
    node_of = {k: i for i, k in enumerate(node_order)}
    window_len = lookback + horizon
    Xe, Xd, Ys, rows = [], [], [], []

    for key, g in df.groupby(KEY_COL, sort=False):
        if key not in node_of:
            continue
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
            rows.append({KEY_COL: key, "node_index": node_of[key],
                         "start": times[start], "origin": times[mid - 1],
                         "end": times[end - 1]})

    if not Xe:
        return (np.empty((0, lookback, len(ENCODER_COLS))),
                np.empty((0, horizon, len(DECODER_COLS))),
                np.empty((0, horizon)),
                pd.DataFrame(columns=[KEY_COL, "node_index", "start",
                                      "origin", "end"]))
    return np.stack(Xe), np.stack(Xd), np.stack(Ys), pd.DataFrame(rows)


def build(source, city):
    """Write a nodes.npz plus one window shard per station for a (source, city).

    nodes.npz holds node_order and coordinates so build_graph can construct the
    graph at any k during training; k is not fixed here.
    """
    coords = station_coords(source, city)
    if coords.empty:
        print(f"[gnn] {source}/{city}: no station coordinates; nothing to build")
        return
    node_order = sorted(coords["location_key"].unique())
    node_of = {key: i for i, key in enumerate(node_order)}
    outdir = shard_dir("gnn", city, source)

    coord_lookup = coords.drop_duplicates("location_key").set_index("location_key")
    lat = coord_lookup.loc[node_order, "latitude"].to_numpy(dtype=np.float64)
    lon = coord_lookup.loc[node_order, "longitude"].to_numpy(dtype=np.float64)
    np.savez_compressed(
        os.path.join(outdir, "nodes.npz"),
        node_order=np.array(node_order).astype(str),
        latitude=lat,
        longitude=lon,
        encoder_cols=np.array(ENCODER_COLS),
        decoder_cols=np.array(DECODER_COLS),
    )

    total_windows = 0
    written = 0
    for location_key in node_order:
        df = load_station(source, location_key)
        if df.empty:
            continue
        if source == "clean":
            X_enc, X_dec, Y, meta = build_gnn_windows(df, node_order)
            enc_mask = None
        else:
            X_enc, X_dec, Y, enc_mask, meta = build_gnn_windows_masked(df, node_order)
        if X_enc.shape[0] == 0:
            continue

        payload = {
            "X_enc": X_enc.astype(np.float32),
            "X_dec": X_dec.astype(np.float32),
            "Y": Y.astype(np.float32),
            "node_index": np.full(X_enc.shape[0], node_of[location_key], dtype=np.int64),
            "meta_origin": meta["origin"].dt.tz_localize(None).to_numpy().astype("datetime64[ns]"),
        }
        if enc_mask is not None:
            payload["enc_mask"] = enc_mask.astype(np.int8)

        np.savez_compressed(os.path.join(outdir, f"{safe_name(location_key)}.npz"), **payload)
        total_windows += X_enc.shape[0]
        written += 1

    print(f"[gnn] {source}/{city}: {len(node_order)} nodes, {written} station shards, "
          f"{total_windows} windows -> {outdir}  (k deferred to training)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, choices=["clean", "masked"])
    p.add_argument("--city", required=True)
    build(**vars(p.parse_args()))