""" Batch windows for the models: independent windows for the LSTM, per-origin station graphs
    for the GNN. Pass the run's np.random.Generator to shuffle (train), or None for a fixed
    order (val/test).
"""

import numpy as np

def lstm_batches(data, ids: np.ndarray, batch_size: int, rng=None, device=None):
    """ Yield LSTM batches (x_enc, x_dec, y, ids) of batch_size windows from the given window ids.
        With `device`, the inputs are gathered directly on that device instead of as NumPy.
    """
    # Each window is an independent sample.
    order = rng.permutation(ids) if rng is not None else ids
    for i in range(0, len(order), batch_size):
        batch_ids = order[i:i + batch_size]
        gathered = data.gather(batch_ids) if device is None else data.gather_on(batch_ids, device)
        yield (*gathered, batch_ids)

def edited(batches, edit):
    """ Wrap batch factories so that edit(x_enc, x_dec, ids) -> (x_enc, x_dec) changes the inputs
        of every batch, whatever the model.
    """
    def generate():
        for x_enc, x_dec, y, *rest in batches():
            yield (*edit(x_enc, x_dec, rest[-1]), y, *rest)
    return generate

def station_distances_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """ Compute the distance in km between every pair of stations
        (haversine: great-circle distance from latitude/longitude).
    """
    p, l = np.radians(lat), np.radians(lon)
    a = (np.sin((p[:, None] - p) / 2) ** 2
         + np.cos(p[:, None]) * np.cos(p) * np.sin((l[:, None] - l) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))

def build_station_graph(dist: np.ndarray, k: int):
    """ Create the station graph: link each station to its k nearest stations, weighting
        closer ones more. Return (edge_index [2,E], edge_weight [E]).
    """
    n = len(dist)
    if n < 2:
        return np.empty((2, 0), np.int64), np.empty(0, np.float32)
    # Nearest k neighbours per station, never itself.
    d = dist + np.diag(np.full(n, np.inf))
    nbr = np.argsort(d, 1)[:, :min(k, n - 1)]
    # Link both ways if either station picked the other.
    adj = np.zeros((n, n), bool)
    adj[np.arange(n)[:, None], nbr] = True
    adj |= adj.T
    src, dst = np.nonzero(adj)
    # Gaussian weight; bandwidth = median distance to the chosen neighbours.
    near = np.take_along_axis(d, nbr, 1)
    near = near[near > 0]
    sigma = np.median(near) if near.size else 1.0
    weight = np.exp(-dist[src, dst] ** 2 / (2 * sigma ** 2))
    return np.stack([src, dst]), weight.astype(np.float32)

def normalized_adjacency(edge_index, edge_weight, n):
    """ NumPy form of D^-1/2 (A + I) D^-1/2 with self-loops: return (indices [2,E'], values [E']),
        the same normalized adjacency the model used to rebuild on every forward pass, computed
        once per unique station graph instead.
    """
    loops = np.arange(n)
    src = np.concatenate([edge_index[0], loops])
    dst = np.concatenate([edge_index[1], loops])
    w = np.concatenate([edge_weight.astype(np.float32), np.ones(n, np.float32)])
    deg = np.zeros(n, np.float32)
    np.add.at(deg, dst, w)
    inv = (1.0 / np.sqrt(deg)).astype(np.float32)
    values = (w * inv[src] * inv[dst]).astype(np.float32)
    return np.stack([dst, src]), values

class GraphBatches:
    """ Build GNN batches from window ids: each origin hour of each city becomes a graph of the
        stations that have a window then, so cities never share a graph. Each unique station graph's
        normalized adjacency is computed once; a batch's block-diagonal adjacency is assembled from
        those pieces. Call the object once per epoch to iterate.
    """

    def __init__(self, data, ids: np.ndarray, k: int, batch_size: int, device=None):
        self.data, self.batch_size, self.device = data, batch_size, device
        # Cuts are shared timestamps, so every window at an hour is in the same split.
        city = np.unique(data.city, return_inverse=True)[1][data.station[ids]]
        graph_key = data.origin[ids].astype("int64") * (city.max() + 1) + city
        pool = ids[np.lexsort((data.station[ids], graph_key))]
        graph_key = np.sort(graph_key)
        bounds = np.append(np.flatnonzero(np.diff(graph_key)) + 1, len(pool))
        dist = station_distances_km(data.lat, data.lon)
        graphs, self.groups = {}, []
        for a, b in zip(np.append(0, bounds[:-1]), bounds):
            stations = data.station[pool[a:b]]
            key = stations.tobytes()
            if key not in graphs:
                edge_index, edge_weight = build_station_graph(dist[np.ix_(stations, stations)], k)
                idx, val = normalized_adjacency(edge_index, edge_weight, len(stations))
                graphs[key] = (idx, val, len(stations))
            self.groups.append((pool[a:b], graphs[key]))

    def __call__(self, rng=None):
        """ Yield GNN batches (x_enc, x_dec, y, adj, ids), stacking several origins into one
            disconnected graph whose block-diagonal normalized adjacency is `adj` (a sparse tensor).
        """
        import torch
        device = self.device or "cpu"
        n = len(self.groups)
        order = rng.permutation(n) if rng is not None else np.arange(n)
        for i in range(0, n, self.batch_size):
            chunk = [self.groups[j] for j in order[i:i + self.batch_size]]
            ids = np.concatenate([window_ids for window_ids, _ in chunk])
            sizes = [graph[2] for _, graph in chunk]
            offset = np.cumsum([0] + sizes[:-1])
            idx = np.concatenate([graph[0] + o for (_, graph), o in zip(chunk, offset)], axis=1)
            val = np.concatenate([graph[1] for _, graph in chunk])
            total = int(np.sum(sizes))
            adj = torch.sparse_coo_tensor(torch.from_numpy(idx).to(device),
                                          torch.from_numpy(val).to(device), (total, total),
                                          check_invariants=False).coalesce()
            gathered = (self.data.gather(ids) if self.device is None
                        else self.data.gather_on(ids, self.device))
            yield (*gathered, adj, ids)
