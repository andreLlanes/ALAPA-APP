""" Gradient-boosted trees (XGBoost) on lag/rolling PM2.5 features plus the target hour's met/time.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from Common.schema import HORIZON_H
from data.load import GBT_FEATURES, gbt_features
from utils import runlog

class GBTForecaster:
    """ Autoregressive: one next-hour model rolled forward 72 times, each prediction becoming the
        newest lag. Direct: 72 models, model h predicts hour h from the lookback.
    """

    def __init__(self, params: dict, num_boost_round: int, autoregressive: bool,
                 device="cpu", seed=0):
        self.params = {"tree_method": "hist", "device": device, "seed": seed, **params}
        self.rounds = num_boost_round
        self.autoregressive = autoregressive
        self.boosters = []

    @staticmethod
    def matrix(lookback, x_dec_hour, y=None, weight=None):
        """ Build one row per window: lookback lags/rolling stats + met/time of the hour predicted.
        """
        return xgb.DMatrix(np.column_stack([gbt_features(lookback), x_dec_hour]), y, weight=weight,
                           feature_names=GBT_FEATURES)

    def fit(self, lookback, x_dec, y, weight=None, val=None, ckpt_dir=None, verbose=True,
            label="TREES"):
        """ Train every model for exactly num_boost_round rounds (no early stopping). val is an
            optional (lookback, x_dec, y) for logging. With ckpt_dir, finished models are saved
            and reloaded, so an interrupted fit resumes at the next hour. verbose=False hides the
            per-model lines (used while tuning). Per-model scores go to the run log under `label`.
        """
        hours = 1 if self.autoregressive else HORIZON_H
        if ckpt_dir:
            Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        self.boosters, scores = [], []
        for h in range(hours):
            path = Path(ckpt_dir) / f"hour_{h:02d}.ubj" if ckpt_dir else None
            if path and path.exists():
                self.boosters.append(xgb.Booster(model_file=str(path)))
                scores.append({"model": h + 1, "note": "loaded from checkpoint"})
                continue
            dtrain = self.matrix(lookback, x_dec[:, h], y[:, h], weight)
            evals = [(dtrain, "train")]
            if val is not None:
                evals.append((self.matrix(val[0], val[1][:, h], val[2][:, h]), "val"))
            log = {}
            booster = xgb.train({**self.params, "eval_metric": "rmse"}, dtrain, self.rounds,
                                evals=evals, evals_result=log, verbose_eval=False)
            self.boosters.append(booster)
            if path:
                booster.save_model(str(path))
            scores.append({"model": h + 1, **{k: v["rmse"][-1] for k, v in log.items()}})
            if verbose:
                line = ", ".join(f"{k} = {v['rmse'][-1]:.4f}" for k, v in log.items())
                print(f"    TREE {h + 1}/{hours}: {line}")
        runlog.detail(label, {"settings": self.params, "rounds": self.rounds,
                              "weighted rows": weight is not None}, scores)
        return self

    def predict(self, lookback, x_dec) -> np.ndarray:
        """ Forecast [n,72] (normalized).
        """
        if not self.autoregressive:
            return np.column_stack([b.predict(self.matrix(lookback, x_dec[:, h]))
                                    for h, b in enumerate(self.boosters)])
        history, preds = lookback.copy(), []
        for h in range(HORIZON_H):
            pred = self.boosters[0].predict(self.matrix(history, x_dec[:, h]))
            preds.append(pred)
            history = np.column_stack([history[:, 1:], pred])  # prediction becomes the newest lag
        return np.column_stack(preds)

    def importance(self) -> pd.DataFrame:
        """ Return the gain of every feature in each model: one row per forecast hour (a single
            row for the autoregressive model), zero where a model never split on a feature.
        """
        gains = pd.DataFrame([b.get_score(importance_type="gain") for b in self.boosters],
                             columns=GBT_FEATURES).fillna(0.0)
        return gains.rename_axis("lead").reset_index().assign(lead=lambda f: f["lead"] + 1)

    def load(self, directory):
        """ Load the models fit() saved to a folder.
        """
        paths = sorted(Path(directory).glob("hour_*.ubj"))
        self.boosters = [xgb.Booster(model_file=str(p)) for p in paths]
        return self
