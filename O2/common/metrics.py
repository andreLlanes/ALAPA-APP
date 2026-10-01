"""Forecast accuracy metrics and the skill score.

Single source of truth for how a forecast is scored, imported by the baselines
and (later) by the trained models so a persistence RMSE and an LSTM RMSE are
always computed the same way.

Metrics follow Section 4.8.2 of the manuscript: RMSE is the headline figure, MAE
reports the typical error, MBE exposes directional bias, IOA is a scale-free
agreement score, and R2 is kept for comparability with other air-quality studies.
The skill score of Section 4.8.3 expresses a model's RMSE as a fractional
reduction against a reference forecast.

All functions take matched forecast/observation pairs in ug/m3 and ignore pairs
where either side is non-finite, so a caller may pass arrays that still carry
NaN. Every metric returns NaN rather than raising when no valid pair survives.
"""

import numpy as np

METRIC_NAMES = ("rmse", "mae", "mbe", "ioa", "r2")


def _matched(pred, obs):
    """Return the flattened, finite (pred, obs) pairs shared by both arrays."""
    pred = np.asarray(pred, dtype=float).ravel()
    obs = np.asarray(obs, dtype=float).ravel()
    if pred.shape != obs.shape:
        raise ValueError(f"pred and obs must align; got {pred.shape} vs {obs.shape}")
    ok = np.isfinite(pred) & np.isfinite(obs)
    return pred[ok], obs[ok]


def rmse(pred, obs):
    """Root-mean-square error (Eq. 4.14); penalizes large misses disproportionately."""
    p, o = _matched(pred, obs)
    if p.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((p - o) ** 2)))


def mae(pred, obs):
    """Mean absolute error (Eq. 4.15); the typical error, insensitive to rare misses."""
    p, o = _matched(pred, obs)
    if p.size == 0:
        return float("nan")
    return float(np.mean(np.abs(p - o)))


def mbe(pred, obs):
    """Mean bias error (Eq. 4.16); positive when the forecast over-predicts."""
    p, o = _matched(pred, obs)
    if p.size == 0:
        return float("nan")
    return float(np.mean(p - o))


def ioa(pred, obs):
    """Index of agreement (Eq. 4.17), 0 to 1, referenced to the observation mean.

    Returns NaN only in the degenerate case where every forecast and every
    observation equals the observation mean, which collapses the denominator and
    leaves the score undefined.
    """
    p, o = _matched(pred, obs)
    if p.size == 0:
        return float("nan")
    obar = o.mean()
    denom = np.sum((np.abs(p - obar) + np.abs(o - obar)) ** 2)
    if denom == 0:
        return float("nan")
    return float(1.0 - np.sum((p - o) ** 2) / denom)


def r2(pred, obs):
    """Coefficient of determination (Eq. 4.18), undefined for constant observations."""
    p, o = _matched(pred, obs)
    if p.size == 0:
        return float("nan")
    denom = np.sum((o - o.mean()) ** 2)
    if denom == 0:
        return float("nan")
    return float(1.0 - np.sum((o - p) ** 2) / denom)


def all_metrics(pred, obs):
    """Return every Section 4.8.2 metric plus the count of pairs scored."""
    p, o = _matched(pred, obs)
    return {
        "n_pairs": int(p.size),
        "rmse": rmse(p, o),
        "mae": mae(p, o),
        "mbe": mbe(p, o),
        "ioa": ioa(p, o),
        "r2": r2(p, o),
    }


def skill_score(model_rmse, reference_rmse):
    """Fractional RMSE reduction against a reference forecast (Eq. 4.21).

    Zero when the model matches the reference, positive when it improves on it,
    and negative when it is worse. The same form serves the transfer benefit of
    Eq. 4.22 with the cold-start model as the reference.
    """
    model_rmse = float(model_rmse)
    reference_rmse = float(reference_rmse)
    if not np.isfinite(model_rmse) or not np.isfinite(reference_rmse) or reference_rmse == 0:
        return float("nan")
    return 1.0 - model_rmse / reference_rmse
