""" Forecast error at every lead: RMSE mean, sd and confidence interval over seeds, skill against
    persistence and climatology, and a paired Wilcoxon test against the best configuration at the
    leads in LEAD_TEST_HOURS (Holm-adjusted within each city and filter setting).
"""

import numpy as np
import pandas as pd

import config
from evals.eval import KEYS, interval, paired_tests, station_rmse

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
    out["p_vs_best"] = np.nan
    for lead in config.LEAD_TEST_HOURS:
        at_lead = out[out["lead"] == lead]
        out.loc[at_lead.index, "p_vs_best"] = paired_tests(at_lead.reset_index(drop=True),
                                                           station_rmse(runs, lead))
    return out.sort_values(["city", "filters", "settings", "lead", "rmse"]).reset_index(drop=True)
