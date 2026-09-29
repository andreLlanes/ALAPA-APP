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
python O2/climatology_baseline.py --city "Metro Manila"
python O2/climatology_baseline.py --city all --output-table openaq.climatology_baseline
python O2/climatology_baseline.py --city all --start-date 2023-01-01 --end-date 2026-01-01
```

The script reads from the O1 pipeline table `openaq.merged_clean` by default.
It applies a chronological 70/15/15 train/validation/test split, keeps stations
with at least 90% hourly completeness across their observed lifespan, and fits
the lookup table on the training partition only. Use `--start-date`, `--end-date`,
`--min-completeness`, `--train-fraction`, and `--validation-fraction` to override
these defaults. Pass `--no-evaluate` when only the persisted lookup is needed.

Validation and test scores use the same 72-hour lookback plus 72-hour forecast
window as O1 model datasets. The script reports RMSE, MAE, index of agreement
(IOA), and $R^2$ for complete windows only.

The fitted climatology is stored as a compact lookup table keyed by:

- `city`
- `location_key`
- `hour_of_day`
- `climatology_pm25`

This is lightweight and robust for benchmarking against more recent models such
as persistence, SARIMA, or deep-learning approaches.
