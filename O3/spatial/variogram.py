""" Residual variogram of the regression-kriging: the empirical semivariogram of the trend
    residuals, pooled over every validation hour of a run, and the theoretical model fitted to it.
    One hour holds too few stations for a stable variogram, so a run fits one and keeps it for every
    test hour (as in Graler, Pebesma and Heuvelink, 2016, The R Journal 8(1)). Each hour's residuals
    are divided by that hour's residual std before pooling, so polluted hours do not dominate the
    shape; an hour rescales the model by its own residual variance, which changes the kriging
    variance but not the kriging weights.
    Spherical, exponential and Gaussian models are fitted by weighted least squares (weights
    N(h) / h^2, the gstat default) and the lowest weighted residual sum of squares is kept.
    `range` is each model's parameter a: the spherical model levels off at a, the exponential
    reaches 95% of its sill at 3a and the Gaussian at sqrt(3) a.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

import config

def spherical(h, nugget, psill, a):
    """ Spherical model: rises to nugget + psill at lag a, flat beyond.
    """
    r = np.minimum(h / a, 1.0)
    return nugget + psill * (1.5 * r - 0.5 * r ** 3)

def exponential(h, nugget, psill, a):
    """ Exponential model: approaches nugget + psill, 95% at 3a.
    """
    return nugget + psill * (1.0 - np.exp(-h / a))

def gaussian(h, nugget, psill, a):
    """ Gaussian model: approaches nugget + psill, 95% at sqrt(3) a.
    """
    return nugget + psill * (1.0 - np.exp(-(h / a) ** 2))

MODELS = {"spherical": spherical, "exponential": exponential, "gaussian": gaussian}

@dataclass
class Variogram:
    """ A fitted model: semivariance at lag h (km), 0 at h = 0.
    """
    model: str
    nugget: float
    psill: float
    range: float

    def __call__(self, h):
        h = np.asarray(h, float)
        return np.where(h > 0, MODELS[self.model](h, self.nugget, self.psill, self.range), 0.0)

class Pool:
    """ Sum squared residual differences, pair counts and lags per lag bin over many hours.
    """

    def __init__(self, max_separation_km: float):
        self.max_separation = max_separation_km
        self.edges = np.linspace(0.0, config.VARIOGRAM_MAX_LAG * max_separation_km,
                                 config.VARIOGRAM_LAGS + 1)
        self.sq = np.zeros(config.VARIOGRAM_LAGS)
        self.n = np.zeros(config.VARIOGRAM_LAGS)
        self.lag = np.zeros(config.VARIOGRAM_LAGS)
        self.hours = 0

    def add(self, residuals: np.ndarray, dist: np.ndarray):
        """ Add the station pairs of every hour in residuals [hours, stations] (standardized),
            with the stations' distance matrix (km).
        """
        i, j = np.triu_indices(dist.shape[0], 1)
        d = dist[i, j]
        b = np.searchsorted(self.edges, d, side="right") - 1
        keep = (d > 0) & (b < config.VARIOGRAM_LAGS)
        k = len(residuals)
        diff2 = ((residuals[:, i[keep]] - residuals[:, j[keep]]) ** 2).sum(0)
        size = config.VARIOGRAM_LAGS
        self.sq += np.bincount(b[keep], diff2, size)
        self.n += k * np.bincount(b[keep], minlength=size)
        self.lag += k * np.bincount(b[keep], d[keep], size)
        self.hours += k

    def empirical(self):
        """ Return the non-empty bins: (mean lag, semivariance, pair count).
        """
        ok = self.n > 0
        return self.lag[ok] / self.n[ok], self.sq[ok] / (2 * self.n[ok]), self.n[ok]

def fit(pool: Pool):
    """ Fit every candidate model to the pooled semivariogram; return (best Variogram, one row per
        candidate with its parameters and weighted residual sum of squares).
    """
    h, gamma, n = pool.empirical()
    if len(h) < 3:
        raise ValueError(f"only {len(h)} non-empty lag bins over {pool.hours} hours; a variogram "
                         f"needs at least 3")
    weight = np.sqrt(n / h ** 2)
    start = [gamma[0], max(gamma.max() - gamma[0], 1e-6), h.max() / 2]
    bounds = ([0.0, 0.0, 1e-6], [np.inf, np.inf, pool.max_separation])
    rows = []
    for name in config.VARIOGRAM_MODELS:
        model = MODELS[name]
        res = least_squares(lambda p: weight * (model(h, *p) - gamma),
                            np.clip(start, bounds[0], [10 * gamma.max(), 10 * gamma.max(),
                                                       pool.max_separation]), bounds=bounds)
        nugget, psill, a = (float(v) for v in res.x)
        rows.append({"model": name, "nugget": nugget, "psill": psill, "range_km": a,
                     "wrss": float(np.sum(res.fun ** 2))})
    best = min(rows, key=lambda r: r["wrss"])
    return Variogram(best["model"], best["nugget"], best["psill"], best["range_km"]), rows
