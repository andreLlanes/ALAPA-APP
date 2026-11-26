""" Regression-kriging at one forecast origin: for each of the 72 forecast hours, an OLS trend of
    the stations' forecasts on the standardized covariates, then ordinary kriging of its residuals
    with a fixed variogram. The prediction at a target is the trend there plus the kriged residual,
    set to 0 where negative; the kriging variance, rescaled by the hour's residual variance, is a
    relative uncertainty (lower near stations), not an error bound.
    A covariate with the same value at every station in an hour cannot be estimated and is left out
    of that hour's trend, and so is NDVI when a station or the target has no valid NDVI that hour.
"""

import numpy as np

def trend(x: np.ndarray, z: np.ndarray, usable=None):
    """ Fit z [n] on an intercept and the covariates x [n, k] that have a value (not NaN) at every
        station, vary across them, and are usable ([k] bool, default all).
        Return (coefficients, covariates used [k] bool, rank, residuals [n]).
    """
    finite = np.isfinite(x).all(0)
    used = finite & (np.ptp(np.where(finite, x, 0.0), axis=0) > 1e-12)
    if usable is not None:
        used &= usable
    design = np.column_stack([np.ones(len(z)), x[:, used]])
    beta, _, rank, _ = np.linalg.lstsq(design, z, rcond=None)
    return beta, used, int(rank), z - design @ beta

def residual_variance(residuals: np.ndarray, rank: int) -> float:
    """ Residual variance of a trend fit, on n - rank degrees of freedom.
    """
    return float(residuals @ residuals / (len(residuals) - rank))

def ordinary_weights(variogram, dist: np.ndarray, dist_target: np.ndarray):
    """ Solve the ordinary-kriging system for one target (dist_target [n]) or many ([n, m]):
        weights [n] or [n, m], each target's summing to 1, and the kriging variance (a number or
        [m]) in the variogram's units.
    """
    n = len(dist)
    lhs = np.ones((n + 1, n + 1))
    lhs[:n, :n] = variogram(dist)
    lhs[n, n] = 0.0
    gamma = variogram(dist_target).reshape(n, -1)
    rhs = np.vstack([gamma, np.ones((1, gamma.shape[1]))])
    try:
        sol = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    weights, variance = sol[:n], (sol[:n] * gamma).sum(0) + sol[n]
    if np.ndim(dist_target) == 1:
        return weights[:, 0], float(variance[0])
    return weights, variance

def hour_residuals(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """ Standardized trend residuals [hours, n] for covariates x [hours, n, k] and forecasts
        z [hours, n]: each hour's residuals divided by its residual std, the input of the pooled
        variogram. An hour the trend fits exactly (residual std below 1e-6 ug/m3, rounding error
        only) is NaN, so its rounding error is never scaled up into the variogram.
    """
    out = np.full(z.shape, np.nan)
    for h in range(len(z)):
        _, _, rank, e = trend(x[h], z[h])
        var = residual_variance(e, rank)
        if var > 1e-12:
            out[h] = e / np.sqrt(var)
    return out

def krige_origin(variogram, x: np.ndarray, z: np.ndarray, dist: np.ndarray,
                 x_target: np.ndarray, dist_target: np.ndarray):
    """ Regression-kriging of one origin's forecasts z [hours, n] (covariates x [hours, n, k],
        station distances dist [n, n]) at one target (covariates x_target [hours, k], distances
        dist_target [n]). The kriging weights depend only on the stations and the variogram, so
        they are solved once for every hour. Return (prediction [hours], variance [hours],
        hours with a covariate left out, hours with a rank-deficient trend, hours set to 0).
    """
    weights, ok_var = ordinary_weights(variogram, dist, dist_target)
    pred, var = np.empty(len(z)), np.empty(len(z))
    dropped = deficient = 0
    for h in range(len(z)):
        beta, used, rank, e = trend(x[h], z[h], np.isfinite(x_target[h]))
        dropped += int(not used.all())
        deficient += int(rank < len(beta))
        pred[h] = beta[0] + x_target[h, used] @ beta[1:] + weights @ e
        var[h] = residual_variance(e, rank) * ok_var
    clipped = int((pred < 0).sum())
    return np.maximum(pred, 0.0), var, dropped, deficient, clipped
