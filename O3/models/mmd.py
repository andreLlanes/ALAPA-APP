""" Maximum Mean Discrepancy (Section 4.5.2, Eqs. 4.4-4.6): NumPy for station distances,
    torch for the differentiable alignment term.
    The kernel averages five Gaussians with bandwidths 1/4, 1/2, 1, 2 and 4 times sigma_med,
    the median positive squared distance over the pooled samples. MMD^2 is the unbiased
    estimator of Gretton et al. (2012), so it can be slightly negative for identical distributions.
"""

from functools import lru_cache

import numpy as np
import torch

BANDWIDTH_SCALES = 2.0 ** (np.arange(1, 6) - 3)  # u = 1..5  ->  2^(u-3)

# NumPy: station distances and GBT weights

def sq_dists(a, b=None):
    """ Compute squared Euclidean distances between the rows of a and b (or within a).
    """
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
    """ Return the values above the diagonal of a square distance matrix.
    """
    return d[_triu(d.shape[0])]

def pooled_median(*blocks):
    """ Return the median positive value over all the given distance blocks.
    """
    v = np.concatenate([np.ravel(b) for b in blocks])
    return float(np.median(v[v > 0]))

def bandwidths(sigma_med):
    """ Return the five kernel bandwidths for a sigma_med.
    """
    return sigma_med * BANDWIDTH_SCALES

def _kernel_sum(d, sigmas):
    return sum(float(np.exp(-d / s).sum()) for s in sigmas) / len(sigmas)

def within_mean(d, sigmas):
    """ Average the kernel within one sample, excluding self-pairs.
    """
    n = d.shape[0]
    return (_kernel_sum(d, sigmas) - n) / (n * (n - 1))

def cross_mean(d, sigmas):
    """ Average the kernel across two samples.
    """
    return _kernel_sum(d, sigmas) / d.size

def sample_weights(d, tau):
    """ Turn station distances into GBT sample weights: exp(-max(d, 0) / tau) (Eq. 4.6).
    """
    return np.exp(-np.clip(np.asarray(d, dtype=float), 0.0, None) / tau)

# Torch: alignment term for LSTM/GNN pretraining

def torch_sq_dists(a, b=None):
    """ Compute squared Euclidean distances between the rows of a and b (or within a).
    """
    same = b is None
    b = a if same else b
    d = (torch.cdist(a, b, p=2.0) ** 2).clamp_min(0.0)
    if same:
        d = d - torch.diag_embed(torch.diagonal(d))
    return d

def torch_median_heuristic(d_xx, d_yy, d_xy):
    """ Return sigma_med from the three distance blocks (detached: the bandwidth is not trained).
    """
    iu = torch.triu_indices(d_xx.shape[0], d_xx.shape[0], offset=1, device=d_xx.device)
    iv = torch.triu_indices(d_yy.shape[0], d_yy.shape[0], offset=1, device=d_yy.device)
    pooled = torch.cat([d_xx[iu[0], iu[1]], d_yy[iv[0], iv[1]], d_xy.reshape(-1)]).detach()
    positive = pooled[pooled > 0]
    if positive.numel() == 0:
        return pooled.mean().clamp_min(torch.finfo(pooled.dtype).eps)
    return positive.median()

def _torch_kernel_mean(d, sigmas, exclude_diagonal):
    total = sum(torch.exp(-d / s).sum() for s in sigmas) / len(sigmas)
    if not exclude_diagonal:
        return total / d.numel()
    # k(x, x) = 1 at every bandwidth, so the diagonal contributes exactly n.
    n = d.shape[0]
    return (total - n) / (n * (n - 1))

def torch_mmd2(x, y, sigma_med=None, unbiased=True):
    """ Compute a differentiable MMD^2 between representation batches x and y ([rows, features]).
    """
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError("x and y must be 2-D (batch, features)")
    if unbiased and (x.shape[0] < 2 or y.shape[0] < 2):
        raise ValueError("the unbiased estimator needs at least 2 rows per side")
    d_xx, d_yy, d_xy = torch_sq_dists(x), torch_sq_dists(y), torch_sq_dists(x, y)
    if sigma_med is None:
        sigma_med = torch_median_heuristic(d_xx, d_yy, d_xy)
    sigmas = [sigma_med * float(s) for s in BANDWIDTH_SCALES]
    return (_torch_kernel_mean(d_xx, sigmas, unbiased) + _torch_kernel_mean(d_yy, sigmas, unbiased)
            - 2.0 * _torch_kernel_mean(d_xy, sigmas, False))

class MMDAlignment(torch.nn.Module):
    """ Compute the alignment term lambda * MMD^2 (Section 4.5.4), with lambda ramped in linearly
        over warmup_steps so an untrained encoder's noise does not dominate the first steps.
    """

    def __init__(self, lambda_=1.0, unbiased=True, warmup_steps=0):
        super().__init__()
        self.lambda_ = float(lambda_)
        self.unbiased = bool(unbiased)
        self.warmup_steps = int(warmup_steps)
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))

    def coefficient(self):
        """ Return the current lambda after warmup.
        """
        if self.warmup_steps <= 0:
            return self.lambda_
        return self.lambda_ * min(1.0, float(self.steps) / self.warmup_steps)

    def forward(self, source_repr, target_repr):
        """ Return (weighted term, raw MMD^2) and advance the warmup counter.
        """
        raw = torch_mmd2(source_repr, target_repr, unbiased=self.unbiased)
        term = self.coefficient() * raw
        if self.training:
            self.steps += 1
        return term, raw.detach()
