# O2 — Climatology baseline

This folder contains the climatology baseline used for PM2.5 forecasting.

The baseline follows the standard formulation:

$$
\hat{y}_{t+\ell} = \bar{y}_{h(t+\ell)}
$$

where $$h(t + \ell)$$ is the target hour-of-day and $$\bar{y}_h$$ is the historical
mean PM2.5 concentration for that same station and hour-of-day from the training
period.

## Usage

From the repository root:

```bash
python O2/climatology_baseline.py --city "Metro Manila"            # main split + rolling origin
python O2/climatology_baseline.py --city all --periods main
python O2/climatology_baseline.py --train-end 2025-08-25T20:00 --test-start 2026-01-26T22:00
python O2/climatology_baseline.py --no-persist                      # do not write to Postgres
```

The script reads the O1 table `MERGED_TABLE` (`openaq.merged_clean_v2`, set in
`O1/common/schema.py`) and fits and scores climatology once per evaluation period:

- `main`: the 70/15/15 chronological split by calendar time. Pass `--train-end`
  and `--test-start` to use frozen cut dates instead.
- `ro_1` to `ro_4`: the rolling origin, four three-month test blocks ending at the
  last record, each fitted on everything before its block.

Scoring follows the same rules as the models, so skill scores compare like with like:

- Windows come from `exclusion_window_starts` in `O1/common/schema.py`, the rule
  the LSTM builder uses, so climatology is scored on exactly the models' windows.
- Interpolated hours (`short_gap_filled`) are never used: the hourly means are
  fitted on measured hours only, and filled target hours are not scored.
- Metrics (RMSE, MAE, MBE, IOA, R²) come from `O2/common/metrics.py`, computed per
  station and then averaged with every station weighted equally, overall and at
  each of the 72 lead times.
- A station is scored only if it has at least 10 measured training hours at every
  hour of day, plus test windows.
- Eligible stations come from the frozen fold file `O2/folds/loso_folds.json` when
  it exists. Until then they are chosen by measured-hour completeness
  (`--min-completeness`, default 0.90).

Outputs, under `O2/Outputs/climatology/<citySlug>/<period>/` (gitignored):
`stations.csv` (per-station metrics), `by_lead.csv` (per station and lead),
`lead_curve.csv` (station-averaged metrics per lead) and `summary.json`.

The fitted lookups are stored in `openaq.climatology_v2`, keyed by
(`period`, `city`, `location_key`, `hour_of_day`), with the observation count and
training cutoff. A rerun replaces only its own period and city. The older
`openaq.climatology_baseline` table was fitted on the full record, test period
included, and should not be used.
