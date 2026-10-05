""" Batch windows for the models: independent windows for the LSTM, per-origin station graphs
    for the GNN. Pass the run's np.random.Generator to shuffle (train), or None for a fixed
    order (val/test).
"""

import numpy as np

def lstm_batches(data, ids: np.ndarray, batch_size: int, rng=None):
    """ Yield LSTM batches (x_enc, x_dec, y, ids) of batch_size windows from the given window ids.
    """
    # Each window is an independent sample.
    order = rng.permutation(ids) if rng is not None else ids
    for i in range(0, len(order), batch_size):
        batch_ids = order[i:i + batch_size]
        yield (*data.gather(batch_ids), batch_ids)

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

class GraphBatches:
    """ Build GNN batches for one split: each origin hour becomes a graph of the stations that
        have a window at that hour. Call the object once per epoch to iterate.
    """

    def __init__(self, data, split: int, k: int, batch_size: int):
        self.data, self.batch_size = data, batch_size
        # Cuts are shared timestamps, so every window at an hour is in the same split:
        # a graph only ever holds that split's windows.
        pool = data.split_window_ids(split)
        pool = pool[np.lexsort((data.station[pool], data.origin[pool]))]
        bounds = np.append(np.unique(data.origin[pool], return_index=True)[1], len(pool))
        dist = station_distances_km(data.lat, data.lon)
        graphs, self.groups = {}, []
        for a, b in zip(bounds[:-1], bounds[1:]):
            stations = data.station[pool[a:b]]
            key = stations.tobytes()
            if key not in graphs:
                graphs[key] = build_station_graph(dist[np.ix_(stations, stations)], k)
            self.groups.append((pool[a:b], graphs[key]))

    def __call__(self, rng=None):
        """ Yield GNN batches (x_enc, x_dec, y, edge_index, edge_weight, ids), stacking
            several origins into one disconnected graph.
        """
        n = len(self.groups)
        order = rng.permutation(n) if rng is not None else np.arange(n)
        for i in range(0, n, self.batch_size):
            chunk = [self.groups[j] for j in order[i:i + self.batch_size]]
            ids = np.concatenate([g[0] for g in chunk])
            offset = np.cumsum([0] + [len(g[0]) for g in chunk[:-1]])
            edge_index = np.concatenate([g[1][0] + o for g, o in zip(chunk, offset)], 1)
            edge_weight = np.concatenate([g[1][1] for g in chunk])
            yield (*self.data.gather(ids), edge_index, edge_weight, ids)
