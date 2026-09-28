# O2 — Forecast Baselines and Evaluation

Naive reference forecasts and the metrics every model is scored against, for
72-hour PM2.5 forecasting across Metro Manila, Bangkok, and Los Angeles.

O2 consumes the model-ready datasets O1 writes; it does not touch Postgres. The
baselines are scored on exactly the station-hour origins and the exact test
period the forecasting models see, which is what makes the skill score a fair
comparison.

## Stages

1. **Common** (`common/`)
   - `metrics.py` — the single definition of RMSE, MAE, MBE, IOA, R2, and the
     skill score, imported by the baselines and (later) by the trained models so
     their numbers are always computed the same way.
   - `splits.py` — the chronological 70/15/15 partition of Section 4.7.3.
2. **Baselines** (`Baselines/`) — naive references that require no training.
   - `persistence.py` — carry the last observed concentration across the horizon.
   - `common_baseline.py` — shard discovery and per-station loading, plus the
     path bootstrap that pulls the column names and horizon from `O1/common`.

## Data partitioning

Section 4.7.3 defines two temporal designs. Both are implemented, and both are
normally run: they answer different questions.

### Chronological split (`--protocol fixed`)

The record is split chronologically into training, validation, and test
partitions in the proportions **70 / 15 / 15**. No shuffling is applied, so no
sample from a later period informs a model evaluated on an earlier one. This is
the partition leave-one-station-out holds fixed while it varies the station.

Two details make the split safe to share across stations and models:

- The cuts are computed once per (city, source) over **every station's origins
  pooled together**, and the same two timestamps are applied to every station, so
  all stations are scored on the same calendar period and fold-averaging compares
  like with like.
- The cuts fall on **timestamp boundaries, not sample indices**, so every window
  sharing an origin hour lands in the same partition and no two stations disagree
  about which side of a cut an hour is on. The realized proportions are therefore
  approximate to the granularity of the origin timestamps; `windows.json` records
  what was actually achieved.

### Rolling origin (`--protocol rolling`)

Temporal generalization is assessed by advancing the split point through the
record. A model trains on all data up to a given origin and is evaluated on the
**three months that follow**; the origin then advances by three months and the
procedure repeats, the preceding test period absorbed into training. **Four
origins** are used, so the evaluated periods together span twelve months and
cover a full wet and dry cycle.

Training is an **expanding window**, never a sliding one: each fold keeps all the
history before its test period. The four test periods are consecutive and
half-open, so they neither overlap nor leave an hour unscored, and no origin is
ever in both the training and test side of the same fold.

Counts default to the manuscript's design but are adjustable with `--origins`
and `--test-months`.

### What both share

Every test period is a fixed calendar window shared by all stations, so a station
whose record does not reach into one contributes nothing to it and drops out of
that evaluation. Each run reports that count rather than leaving it implicit.

Because the rolling origin varies the test period that the fixed split holds
constant, agreement between the two indicates that a configuration's standing
does not depend on the particular months it was tested on.

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

# one combination, scored on the fixed test period
python main.py --source masked --city "Metro Manila" --baseline persistence

# the rolling-origin protocol: four origins, three months each
python main.py --source masked --city "Metro Manila" --baseline persistence --protocol rolling

# everything (missing or too-short datasets are skipped with a note)
python main.py --source all --city all --baseline all --protocol all

# diagnostic only: score a different partition of the fixed split
python main.py --source clean --city "Metro Manila" --baseline persistence --split train
```

`--split` defaults to `test` and applies only to `--protocol fixed`. The
baselines fit nothing, so their train and validation scores leak nothing, but the
figures the models are compared against are the test ones. Every protocol and
window writes to its own folder, so no run can overwrite another.

## Outputs

Written to `O2/Outputs/<baseline>/<citySlug>/<source>/<protocol>/`:

| File | Contents |
| --- | --- |
| `windows.json` | The protocol's cut timestamps, calendar spans, and origin counts. |
| `summary.csv` | One row per evaluation window: fold-averaged metrics. |
| `<window>/folds.csv` | One row per station: RMSE, MAE, MBE, IOA, R2, window counts, origin-gap stats. |
| `<window>/by_lead.csv` | One row per (station, lead time): the same metrics at each of the 72 leads. |
| `<window>/lead_curve.csv` | Fold-averaged metrics per lead, for the skill-vs-lead-time plot. |
| `<window>/forecasts/<station>.npz` | Forecast origins and y(t), which regenerate every prediction. |

`<window>` is the partition name under `fixed` (e.g. `test`) and `origin1` …
`origin4` under `rolling`.

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
