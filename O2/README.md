# O2 — Forecast Baselines and Evaluation

Naive reference forecasts and the metrics every model is scored against, for
72-hour PM2.5 forecasting across Metro Manila, Bangkok, and Los Angeles.

The persistence baseline and ablation consume the model-ready datasets O1 writes and do not touch Postgres; the climatology baseline and the fold selection read the merged tables directly. The
baselines are scored on exactly the station-hour origins and the exact test
period the forecasting models see, which is what makes the skill score a fair
comparison.

## Layout

```
O2/
├── common/       shared by every stage: metrics.py, splits.py, ablation.py
├── Baselines/    persistence.py, climatology.py, common_baseline.py, main.py
├── Folds/        loso_fold.py and the frozen loso_folds.json
├── Ablation/     main.py (writes the ablation manifest)
└── Outputs/      results, one folder per baseline or stage (gitignored)
```

## Stages

1. **Common** (`common/`)
   - `metrics.py` — the single definition of RMSE, MAE, MBE, IOA, R2, and the
     skill score, imported by the baselines and (later) by the trained models so
     their numbers are always computed the same way.
   - `splits.py` — the chronological 70/15/15 partition of Section 4.7.3.
   - `ablation.py` — the training-data ablation block sampler of Section 4.7.5.
2. **Baselines** (`Baselines/`) — naive references for the models.
   - `persistence.py` — carry the last observed concentration across the horizon.
   - `climatology.py` — the station's training-period mean at each hour of day,
     fitted per evaluation period; see *Climatology baseline* below.
   - `common_baseline.py` — shard discovery and per-station loading, plus the
     path bootstrap that pulls the column names and horizon from `O1/common`.
3. **Folds** (`Folds/`) — `loso_fold.py` selects and freezes the twenty
   leave-one-station-out folds to `Folds/loso_folds.json`, which every other
   script reads. A station is eligible with at least 70% measured-hour
   completeness and at least 100 usable windows in the test period; the twenty
   are drawn five per density band (distance to the 3rd-nearest eligible
   station) with seed 2026. LOSO is scored through regression-kriging, on the
   withheld station's windows in the test period only (Section 4.7.3 holds the
   test period fixed under LOSO), and the file records that period. The file
   keeps two lists: `training_pool` (every station at ≥70% completeness; each
   fold trains on all of them except the withheld one) and `eligible_stations`
   (those that can also be scored, from which the folds are drawn).
4. **Ablation** (`Ablation/`) — `main.py` writes the ablation manifest the
   trainers consume.

## Data partitioning

Section 4.7.3 defines two temporal designs, both implemented in `common/splits.py`
with frozen dates. Who uses which:

| Protocol | Models | Baselines (persistence, climatology) |
| --- | --- | --- |
| Chronological split, test partition | yes (own-station; also LOSO's fixed test period) | **yes: the skill score is computed here** |
| Rolling origin | yes (ranking stability across seasons) | no |
| Leave-one-station-out (kriged) | yes (spatial generalization, hypothesis tests) | no |

### Chronological split

The record is split chronologically into training, validation, and test
partitions in the proportions **70 / 15 / 15 of the usable forecast windows**
(not of calendar time). No shuffling is applied, so no sample from a later period
informs a model evaluated on an earlier one. Counting windows matters because the
Metro Manila network grew late: a 70% calendar cut would leave only about a third
of the windows for training.

Three details make the split safe to share across stations and models:

- The cuts are computed once per city over **every station's origins pooled
  together**, and the same two timestamps are applied to every station, so all
  stations are scored on the same calendar period and fold-averaging compares
  like with like.
- The cuts are **frozen** in `common/split_dates.json` (committed), so they cannot
  move when stations are added or removed. Compute and freeze them with
  `python O2/common/splits.py --city "Metro Manila"` (add `--dry-run` to only
  print them); read them with `frozen_fixed_cuts(city)` and
  `frozen_rolling_windows(city)`. Metro Manila: training up to 2025-12-28 17:00,
  validation to 2026-02-22 19:00, test after (UTC).
- A **72-hour purge gap** precedes each cut: a window's targets run 72 hours past
  its origin, so training and validation windows must end before the next
  partition begins. The windows in the gap are dropped and counted as `purged`.
- The cuts fall on **timestamp boundaries, not sample indices**, so every window
  sharing an origin hour lands in the same partition and no two stations disagree
  about which side of a cut an hour is on. The realized proportions are therefore
  approximate to the granularity of the origin timestamps; `windows.json` records
  what was actually achieved.

### Rolling origin (models only)

Temporal generalization is assessed by advancing the split point through the
record. A model trains on all data up to a given origin and is evaluated on the
**three months that follow**; the origin then advances by three months and the
procedure repeats, the preceding test period absorbed into training. **Four
origins** are used, so the evaluated periods together span twelve months and
cover a full wet and dry cycle.

Training is an **expanding window**, never a sliding one: each fold keeps all the
history before its test period. Within that history, the most recent **15%** of
the windows is the fold's **validation** set, for early stopping, and the same
72-hour purge gap separates training from validation and validation from test.
The four test periods are consecutive and half-open, so they neither overlap nor
leave an hour unscored, and no origin is ever in two parts of the same fold. The
four blocks are frozen in `common/split_dates.json`.

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

# persistence on the test partition of the chronological split
python main.py --source clean --city "Metro Manila" --baseline persistence

# everything (missing datasets are skipped with a note)
python main.py --source all --city all --baseline all

# diagnostic only: score a different partition of the split
python main.py --source clean --city "Metro Manila" --baseline persistence --split train
```

`--split` defaults to `test`. The baselines fit nothing, so their train and
validation scores leak nothing, but the figures the models are compared against
are the test ones. The baselines are not run under rolling origin or
leave-one-station-out: those are model-only protocols, and the skill score is
computed on the test partition, where models and baselines forecast each station
from its own history on the same windows.

## Outputs

Written to `O2/Outputs/<baseline>/<citySlug>/<source>/fixed/`:

| File | Contents |
| --- | --- |
| `windows.json` | The cut timestamps, calendar spans, and origin counts. |
| `summary.csv` | Station-averaged metrics for the scored partition. |
| `<split>/folds.csv` | One row per station: RMSE, MAE, MBE, IOA, R2, window counts, origin-gap stats. |
| `<split>/by_lead.csv` | One row per (station, lead time): the same metrics at each of the 72 leads. |
| `<split>/lead_curve.csv` | Station-averaged metrics per lead, for the skill-vs-lead-time plot. |
| `<split>/forecasts/<station>.npz` | Forecast origins and y(t), which regenerate every prediction. |

## Notes

- **Station averaging** weights each station equally regardless of how many
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
python O2/Baselines/climatology.py --city "Metro Manila"
python O2/Baselines/climatology.py --city all
python O2/Baselines/climatology.py --no-persist          # do not write to Postgres
```

The script reads the O1 table `MERGED_TABLE` (`openaq.merged_clean_v2`, set in
`O1/common/schema.py`). It fits climatology on the hours the training partition
covers and scores it on the test partition's windows, using the frozen dates in
`common/split_dates.json` (a city must be frozen first). Like persistence, it is
not run under rolling origin or leave-one-station-out.

Scoring follows the same rules as the models, so skill scores compare like with like:

- Windows come from `exclusion_window_starts` in `O1/common/schema.py`, the rule
  the LSTM builder uses, so climatology is scored on exactly the models' windows.
- Interpolated hours (`short_gap_filled`) are never used: the hourly means are
  fitted on measured hours only, filled target hours are not scored, and windows
  whose origin hour was filled are not scored at all (as in persistence).
- Metrics (RMSE, MAE, MBE, IOA, R²) come from `O2/common/metrics.py`, computed per
  station and then averaged with every station weighted equally, overall and at
  each of the 72 lead times.
- A station is scored only if it has at least 10 measured training hours at every
  hour of day, plus test windows.
- Stations come from the frozen fold file's `training_pool` (completeness ≥70%),
  the pool every method uses. For a city the fold file does not cover, they are
  chosen by measured-hour completeness (`--min-completeness`, default 0.70).

## Interpolated hours (all baselines, and the models)

Clean-source shards carry `Y_filled` (interpolated target hours) and
`origin_filled` (the origin hour was interpolated). Filled targets are estimates,
so they are never scored. A filled origin was interpolated from the hours just
after it, which are the window's own first targets, so its inputs leak part of
the answer: such windows are not scored either (about 10% of Manila windows).
Model training and evaluation should apply the same two rules.

Outputs, under `O2/Outputs/climatology/<citySlug>/main/` (gitignored):
`stations.csv` (per-station metrics), `by_lead.csv` (per station and lead),
`lead_curve.csv` (station-averaged metrics per lead) and `summary.json`.

The fitted lookups are stored in `openaq.climatology_v2`, keyed by
(`period`, `city`, `location_key`, `hour_of_day`), with the observation count and
training cutoff. A rerun replaces only its own period and city. The older
`openaq.climatology_baseline` table was fitted on the full record, test period
included, and should not be used.
