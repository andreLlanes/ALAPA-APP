"""Maximum Mean Discrepancy (Section 4.5.2, Eqs. 4.4-4.6), in NumPy.

The kernel averages five Gaussians whose bandwidths are 1/4, 1/2, 1, 2 and 4
times sigma_med, the median positive squared distance over the pooled samples:
k(x, x') = mean_u exp(-||x - x'||^2 / sigma_u). MMD^2 is the unbiased estimator
of Gretton et al. (2012): self-pairs are excluded from the within-set means, so a
value may be slightly negative when both samples share a distribution.

The station-level distances and the GBT sample weights are built on these
functions. The GNN alignment term needs a differentiable copy of the same kernel
at training time.
"""

from functools import lru_cache

import numpy as np

BANDWIDTH_SCALES = 2.0 ** (np.arange(1, 6) - 3)  # u = 1..5  ->  2^(u-3)


def sq_dists(a, b=None):
    a = np.asarray(a, dtype=np.float64)
    same = b is None
    b = a if same else np.asarray(b, dtype=np.float64)
    d = (np.einsum("ij,ij->i", a, a)[:, None]
         + np.einsum("ij,ij->i", b, b)[None, :]
         - 2.0 * (a @ b.T))
    np.maximum(d, 0.0, out=d)
    if same:
        np.fill_diagonal(d, 0.0)
    return d


@lru_cache(maxsize=8)
def _triu(n):
    return np.triu_indices(n, k=1)


def upper_values(d):
    return d[_triu(d.shape[0])]


def pooled_median(*blocks):
    v = np.concatenate([np.ravel(b) for b in blocks])
    return float(np.median(v[v > 0]))


def median_heuristic(d_xx, d_yy, d_xy):
    return pooled_median(upper_values(d_xx), upper_values(d_yy), d_xy)


def bandwidths(sigma_med):
    return sigma_med * BANDWIDTH_SCALES


def _kernel_sum(d, sigmas):
    return sum(float(np.exp(-d / s).sum()) for s in sigmas) / len(sigmas)


def within_mean(d, sigmas):
    n = d.shape[0]
    return (_kernel_sum(d, sigmas) - n) / (n * (n - 1))


def cross_mean(d, sigmas):
    return _kernel_sum(d, sigmas) / d.size


def mmd2(x, y, sigma_med=None):
    d_xx, d_yy, d_xy = sq_dists(x), sq_dists(y), sq_dists(x, y)
    if sigma_med is None:
        sigma_med = median_heuristic(d_xx, d_yy, d_xy)
    s = bandwidths(sigma_med)
    return within_mean(d_xx, s) + within_mean(d_yy, s) - 2.0 * cross_mean(d_xy, s)


def sample_weights(d, tau):
    return np.exp(-np.clip(np.asarray(d, dtype=float), 0.0, None) / tau)
