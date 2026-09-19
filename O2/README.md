# O2 — Forecast Baselines and Evaluation

Naive reference forecasts and the metrics every model is scored against, for
72-hour PM2.5 forecasting across Metro Manila, Bangkok, and Los Angeles.

O2 consumes the model-ready datasets O1 writes; it does not touch Postgres. The
baselines are scored on exactly the station-hour origins the forecasting models
see, which is what makes the skill score a fair comparison.

## Stages

1. **Common** (`common/`) — `metrics.py`, the single definition of RMSE, MAE,
   MBE, IOA, R2, and the skill score, imported by the baselines and (later) by
   the trained models so their numbers are always computed the same way.
2. **Baselines** (`Baselines/`) — naive references that require no training.
   - `persistence.py` — carry the last observed concentration across the horizon.
   - `common_baseline.py` — shard discovery and per-station loading, plus the
     path bootstrap that pulls the column names and horizon from `O1/common`.

## Prerequisite

O2 reads the LSTM window shards under `O1/Outputs/lstm/<citySlug>/<source>/`,
which are the only builder output that stores the raw lookback PM2.5 a naive
forecast needs. Build them first:

```bash
python O1/Builders/main.py --source all --city all --model lstm
```

## Running

```bash
cd O2/Baselines

# one combination
python main.py --source masked --city "Metro Manila" --baseline persistence

# every baseline, city, and source (missing datasets are skipped with a note)
python main.py --source all --city all --baseline all
```

## Outputs

Written to `O2/Outputs/<baseline>/<citySlug>/<source>/`:

| File | Contents |
| --- | --- |
| `folds.csv` | One row per station: RMSE, MAE, MBE, IOA, R2, window counts, origin-gap stats. |
| `by_lead.csv` | One row per (station, lead time): the same metrics at each of the 72 leads. |
| `lead_curve.csv` | Fold-averaged metrics per lead, for the skill-vs-lead-time plot. |
| `forecasts/<station>.npz` | Forecast origins and y(t), which regenerate every prediction. |

## Notes

- **Leave-one-station-out.** Persistence fits nothing and reads only the withheld
  station's own history, so withholding a station changes none of its forecasts:
  each per-station row already is that station's LOSO fold score. The twenty
  stratified folds are a subset of the rows in `folds.csv`, not a separate run.
- **Fold averaging** weights each station equally regardless of how many
  forecasts it contributes, so a few high-volume stations cannot dominate.
- **Masked source.** Where the origin hour itself is missing, persistence carries
  the most recent observed hour forward and records its age in `origin_gap_h`,
  which is what an operator forecasting at that time would actually hold. A
  window with no observation anywhere in its lookback has no persistence forecast
  and is dropped, counted in `windows_no_obs`.
- **Metrics ignore non-finite pairs** and return NaN rather than raising when no
  valid pair survives, so a partially missing input degrades instead of crashing.
- Forecasts are scored in ug/m3; the datasets O1 writes are unnormalized, so no
  inverse transform is needed here.
