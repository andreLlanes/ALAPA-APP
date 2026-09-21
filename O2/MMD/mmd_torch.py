import torch

BANDWIDTH_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)


def sq_dists(a, b=None):
    same = b is None
    b = a if same else b
    d = torch.cdist(a, b, p=2.0) ** 2
    d = d.clamp_min(0.0)
    if same:
        d = d - torch.diag_embed(torch.diagonal(d))
    return d


def median_heuristic(d_xx, d_yy, d_xy):
    n = d_xx.shape[0]
    m = d_yy.shape[0]
    iu = torch.triu_indices(n, n, offset=1, device=d_xx.device)
    iv = torch.triu_indices(m, m, offset=1, device=d_yy.device)
    pooled = torch.cat([
        d_xx[iu[0], iu[1]].reshape(-1),
        d_yy[iv[0], iv[1]].reshape(-1),
        d_xy.reshape(-1),
    ]).detach()
    positive = pooled[pooled > 0]
    if positive.numel() == 0:
        return pooled.mean().clamp_min(torch.finfo(pooled.dtype).eps)
    return positive.median()


def _kernel_mean(d, sigmas, exclude_diagonal):
    total = sum(torch.exp(-d / s).sum() for s in sigmas) / len(sigmas)
    if not exclude_diagonal:
        return total / d.numel()
    # k(x, x) = 1 at every bandwidth, so the diagonal contributes exactly n.
    n = d.shape[0]
    return (total - n) / (n * (n - 1))


def mmd2(x, y, sigma_med=None, unbiased=True):
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError("x and y must be 2-D (batch, features)")
    if unbiased and (x.shape[0] < 2 or y.shape[0] < 2):
        raise ValueError("the unbiased estimator needs at least 2 rows per side")

    d_xx, d_yy, d_xy = sq_dists(x), sq_dists(y), sq_dists(x, y)
    if sigma_med is None:
        sigma_med = median_heuristic(d_xx, d_yy, d_xy)
    sigmas = [sigma_med * s for s in BANDWIDTH_SCALES]

    k_xx = _kernel_mean(d_xx, sigmas, unbiased)
    k_yy = _kernel_mean(d_yy, sigmas, unbiased)
    k_xy = _kernel_mean(d_xy, sigmas, False)
    return k_xx + k_yy - 2.0 * k_xy


class MMDAlignment(torch.nn.Module):
    """The alignment term as a module: ``loss = task_loss + lambda * mmd2``.

    ``lambda_`` is the coefficient of Section 4.5.4. ``warmup_steps`` ramps it in
    linearly from zero, which keeps the untrained encoder's noise from dominating
    the objective in the first steps; set it to 0 for a constant coefficient.
    """

    def __init__(self, lambda_=1.0, unbiased=True, warmup_steps=0):
        super().__init__()
        self.lambda_ = float(lambda_)
        self.unbiased = bool(unbiased)
        self.warmup_steps = int(warmup_steps)
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))

    def coefficient(self):
        if self.warmup_steps <= 0:
            return self.lambda_
        return self.lambda_ * min(1.0, float(self.steps) / self.warmup_steps)

    def forward(self, source_repr, target_repr):
        """Return (weighted term, raw MMD^2) and advance the warmup counter."""
        raw = mmd2(source_repr, target_repr, unbiased=self.unbiased)
        term = self.coefficient() * raw
        if self.training:
            self.steps += 1
        return term, raw.detach()
