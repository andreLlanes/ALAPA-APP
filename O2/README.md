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
```

The script reads from the O1 pipeline table `openaq.merged_clean` by default and
stores the fitted climatology as a compact lookup table keyed by:

- `city`
- `location_key`
- `hour_of_day`
- `climatology_pm25`

This is lightweight and robust for benchmarking against more recent models such
as persistence, SARIMA, or deep-learning approaches.
