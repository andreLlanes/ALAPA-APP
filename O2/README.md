# O2 — Forecast Baselines and Evaluation

Naive reference forecasts and the metrics every model is scored against, for
72-hour PM2.5 forecasting across Metro Manila, Bangkok, and Los Angeles.

The persistence baseline and ablation consume the model-ready datasets O1 writes and do not touch Postgres; the climatology baseline and the fold selection read the merged tables directly. The
baselines are scored on exactly the station-hour origins and the exact test
period the forecasting models see, which is what makes the skill score a fair
comparison.

## Stages

1. **Common** (`common/`)
   - `metrics.py` — the single definition of RMSE, MAE, MBE, IOA, R2, and the
     skill score, imported by the baselines and (later) by the trained models so
     their numbers are always computed the same way.
   - `splits.py` — the chronological 70/15/15 partition of Section 4.7.3.
   - `ablation.py` — the training-data ablation block sampler of Section 4.7.5.
2. **Baselines** (`Baselines/`) — naive references that require no training.
   - `persistence.py` — carry the last observed concentration across the horizon.
   - `common_baseline.py` — shard discovery and per-station loading, plus the
     path bootstrap that pulls the column names and horizon from `O1/common`.
3. **Ablation** (`Ablation/`) — `main.py` writes the ablation manifest the
   trainers consume.
4. **Climatology** (`climatology_baseline.py`) — the hour-of-day climatology
   baseline, fitted per evaluation period; see *Climatology baseline* below.
5. **Folds** (`loso_fold.py`) — selects and freezes the twenty leave-one-station-out
   folds to `folds/loso_folds.json`, which every other script reads.

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

## Training-data ablation

Section 4.7.5 reduces the Metro Manila training data to 25, 50, and 75 percent of
the full training partition and compares the best transfer configuration against
the no-transfer configuration at each level, which is how H5 is read.

`Ablation/main.py` writes the plan; it does not train anything. **The manifest is
the contract**: every model configuration at a given (level, draw) must read the
same blocks, or the Bangkok-versus-no-transfer comparison at that level is
uncontrolled and a difference could be the draw rather than the transfer.

```bash
cd O2/Ablation
python main.py --source masked --city "Metro Manila"
```

Written to `O2/Outputs/ablation/<citySlug>/<source>/`: `manifest.json` (blocks,
seeds, realized volumes) and `summary.csv` (one row per level and draw).

Only Metro Manila is ablated. H5 asks how transfer benefit varies with the volume
of *target* training data, so the source cities keep their full record; another
city is refused unless `--any-city` is passed.

### Consuming the manifest

Trainers read it through `ablation.py` rather than parsing the JSON themselves,
so no two of them can drift:

```python
from ablation import load_manifest, manifest_path, manifest_cuts, apply_ablation
from splits import partition_mask

manifest = load_manifest(manifest_path(OUTPUT_ROOT, "MM", source))
keep = apply_ablation(origins, manifest, level=0.25, draw=1)   # training mask
val  = partition_mask(origins, manifest_cuts(manifest), "val") # untouched
```

Passing *every* origin to `apply_ablation` is safe: the blocks lie inside the
training span, so validation and test origins match nothing and are excluded. The
mask both applies the ablation and enforces that only the training partition is
reduced. `available_draws(manifest)` lists the pairs actually present — iterate
it rather than assuming a full grid, because duplicate draws are dropped.

Four properties are worth knowing before reading the curve:

- **Contiguous blocks, never random hours.** Hourly data is strongly
  autocorrelated, so removing random hours would leave a model with nearly the
  same information and flatten the curve into a null result. Blocks default to
  four weeks; the manuscript's floor is one week.
- **Window erosion.** The O1 shards store complete windows, so keeping an origin
  also keeps its 72 h lookback and 72 h horizon. An origin near a block edge
  would reach into a discarded block and quietly undo the ablation, so each
  retained run is trimmed by the window geometry. That costs 143 origins per run,
  which is why the default block is four weeks rather than one — at the one-week
  floor, erosion discards about 85 percent of every block. Erosion is applied
  only where a discarded block actually abuts, so the 100 percent level
  reproduces the full training partition exactly.
- **Levels target data volume, not block count.** §4.7.5 reduces the data *to* 25
  percent, which is a claim about volume. Block rounding plus erosion would
  otherwise leave a nominal 25 percent delivering about 18, so the block count is
  calibrated until the realized share hits the target. Both figures are recorded;
  plot against `realized`. `--no-calibrate` restores block-share semantics.
- **Levels nest.** All levels within a draw grow along one permutation, so the
  blocks kept at 25 percent are a subset of those kept at 50. The curve then
  varies only how much data the model saw, not which months it saw.

Draws are seeded from the draw number alone, so a level can be re-run later and
reproduce the same blocks. Three draws per level let an unlucky selection be told
apart from a real effect — except where a draw would duplicate an earlier one, in
which case it is dropped. At 100 percent there is nothing left to vary, so that
level carries a single draw instead of three byte-identical training runs.

Ablation is defined on the **fixed split's training partition** only. It is not
applied to the rolling origin (that would multiply the experiment by four, which
§4.7.5 does not ask for), and validation and test are never touched.

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

## Climatology baseline

The climatology baseline follows the standard formulation:

$$
\hat{y}_{t+\ell} = \bar{y}_{h(t+\ell)}
$$

where $$h(t + \ell)$$ is the target hour-of-day and $$\bar{y}_h$$ is the historical
mean PM2.5 concentration for that same station and hour-of-day from the training
period.

### Usage

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
