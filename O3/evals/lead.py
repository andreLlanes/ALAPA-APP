""" Forecast error at every lead: RMSE mean, sd and confidence interval over seeds, and skill against
    persistence and climatology. Significance is tested only as planned contrasts in evals.eval, not
    per lead.
"""

import pandas as pd

from evals.eval import KEYS, interval

def table(runs: pd.DataFrame) -> pd.DataFrame:
    """ One row per configuration and lead; baselines are single runs.
    """
    curves = pd.concat([pd.read_csv(f"{run.dir}/lead_curve.csv", usecols=["lead", "rmse"])
                        .assign(**{k: getattr(run, k) for k in KEYS + ["filters"]})
                        for run in runs.itertuples()])
    grouped = curves.groupby(KEYS + ["filters", "lead"], dropna=False)["rmse"]
    out = pd.concat([grouped.mean(), grouped.std().rename("rmse_sd"),
                     grouped.count().rename("seeds")], axis=1).reset_index()
    out["rmse_ci_low"], out["rmse_ci_high"] = interval(out["rmse"], out["rmse_sd"], out["seeds"])
    for baseline in ("persistence", "climatology"):
        reference = out[out["method"] == baseline].set_index(["city", "filters", "lead"])["rmse"]
        index = pd.MultiIndex.from_frame(out[["city", "filters", "lead"]])
        out[f"skill_vs_{baseline}"] = 1 - out["rmse"] / reference.reindex(index).to_numpy()
    return out.sort_values(["city", "filters", "settings", "lead", "rmse"]).reset_index(drop=True)
