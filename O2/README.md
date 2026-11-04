# O2: PM2.5 forecasting (LSTM, GNN, GBT) with transfer learning

Forecasts 72 hours of PM2.5 from a 72-hour lookback, for Metro Manila (`mm`), Bangkok (`bk`) and Los Angeles (`la`).

## Setup

Put `.env` at the repo root with `PG_DSN` (or `PG_HOST`, `PG_PORT`, `PG_DB`, `PG_USER`, `PG_PASSWORD`).

Install the packages once (from inside `O2/`):

```
pip install -r requirements.txt
```

Run everything from inside `O2/`, after setting `PYTHONPATH` once per terminal so `Common/` is importable:

```
export PYTHONPATH=..          # macOS / Linux
$env:PYTHONPATH = ".."        # Windows PowerShell
```

## Run

```
python data/database.py                                             # copy the database (once)
python tuning/loso_fold.py                                          # select the 20 LOSO stations (once)

python train.py --model lstm --city mm                              # tune, train 30 seeds, score, compare
python train.py --model gnn --city mm --transfer true --source bk+la
python baselines/dlinear.py --city mm                               # DLinear (also --longest-gap, --completeness, --seed, --device)

python evals/eval.py                                                # comparison.csv
python evals/analyze.py                                             # analyses of saved predictions
python evals/permutation.py --model lstm --city mm                  # needs trained LSTM / GNN
python evals/degraded_met.py --model lstm --city mm                 # needs trained models
```

Each script runs either as a file (above) or as a module (`python -m data.database`, `python -m evals.analyze`, ...); both forms need `PYTHONPATH=..` set so `Common/` imports.

After the database changes: `python data/database.py --refresh`, then delete `artifacts/`.

## Arguments (`train.py`, `evals.permutation`, `evals.degraded_met`)

| Argument | Values | Default |
|---|---|---|
| `--model` | lstm, gnn, gbt | required |
| `--city` | mm, bk, la | required |
| `--transfer` | true, false | false |
| `--source` | one city or several joined with `+` (`bk`, `bk+la`), not the target | required with transfer |
| `--training-variant` | LSTM/GNN: zeroshot, pooled, param_frozen, param_full, mmd_frozen, mmd_full · GBT: zeroshot, pooled, weighted · all | all |
| `--autoregressive` | true, false | false |
| `--longest-gap` | 0–12 | 6 |
| `--completeness` | 0, 25, 50, 70, 90 | 70 |
| `--data-ablation` | 25, 50, 75, 100 | 100 |
| `--block` | 0, 1, 2, all | all |
| `--seed` (`train.py`) | e.g. 1,2,3 | 1–30 (`DEFAULT_SEEDS`) |
| `--device` | cpu, cuda | cpu |

`evals.analyze` and `evals.eval` take `--city` only. `baselines.persistence`, `.climatology` and `.ridge` take `--city`, `--longest-gap`, `--completeness`.

## Method

- **Data:** the database is copied to `data/cache/` once; every run reads that copy. Windows are filtered (completeness, then longest lookback gap), then split 70/15/15 by window count with a 72h purge. No fixed dates.
- **Gaps:** lookback gaps are interpolated within the window; windows with any missing forecast hour are dropped.
- **LOSO stations** (`loso_stations.json`) are never removed by the completeness filter.
- **Seeds:** seed 0 (`TUNING_SEED`) tunes the hyperparameters and is never a result. Seeds 1–30 each train a fresh model with the tuned values. Tuned values are shared across seeds, never across settings: every model, city, variant, gap, completeness and ablation block tunes its own.
- **Final models:** `model.pt` (LSTM/GNN) or `trees/` (GBT) per seed; validation and test predictions are stored as float16.
- **Joint source:** `--source bk+la` pools the cities into one source, trained and scored like a single city; its validation score is the mean of the per-city validation RMSEs. Folders read `bk+la-mm`. Cities never share a GNN graph.
- **Variants (LSTM/GNN):** `zeroshot` applies the source model to the target unchanged; `pooled` trains from scratch on source and target together (normalization fitted on both, early stopping on the target); `param_*` and `mmd_*` pretrain on the source (MMD-aligned for `mmd`) and fine-tune frozen (encoder fixed) or full.
- **Variants (GBT):** `zeroshot` trains on the source only; `pooled` adds every source window at weight 1; `weighted` weighs each source station by exp(−d/τ) from its MMD distance to the target, with τ tuned per source-target pair.
- **Baselines:** persistence, climatology and Ridge (per forecast hour, on the GBT features, penalty chosen on validation) run after `train.py` on the same test windows. DLinear is a trained model (`baselines.dlinear`).
- **GBT importance:** gain per feature and lead hour, in each run's `feature_importance.csv`.
- **Scoring:** metrics per station, then averaged with equal weight over stations with at least 100 test windows; reported overall and for seen/unseen stations. `comparison.csv` adds mean, sd and 95% CI over seeds, and a Holm-adjusted paired Wilcoxon test against the best row.
- **Resuming:** rerunning a command skips finished seeds and grid points and resumes interrupted training (except MMD-aligned pretraining, which restarts).
- **Settings:** all fixed settings live in `config.py`; tuned ones in `grid_tuning.json`.

## Analyses

`evals.analyze` reads saved predictions and scores (no retraining) and writes to `artifacts/`:

| File | Content |
|---|---|
| `lead_comparison.csv` | RMSE per lead: mean, sd and CI over seeds; skill against persistence and climatology; Wilcoxon p-value against the best row at leads 1, 24, 48, 72 |
| `ensemble.csv` | per lead: seed-ensemble RMSE against a single seed; coverage and width of the 5th–95th percentile seed spread and of a split-conformal interval (ensemble ± the 90% quantile of its absolute validation errors per lead) |
| `exceedance.csv` | per 24-hour block (hours 1–24, 25–48, 49–72): POD, FAR, CSI and base rate for block means above 50 and 35 µg/m³, and for the worst hours (above the 90th percentile of the city's training PM2.5); baselines included |
| `seasons.csv` | RMSE and MAE with window counts by forecast-origin month and by season (`SEASONS` in `config.py`) |
| `distance_gain.csv`, `distance_gain_summary.csv` | per target station: RMSE gain of each transfer variant over the same model trained on the target alone, against the station's MMD² to the pooled source cities; Spearman correlation per variant |

`evals.permutation` (LSTM, GNN) shuffles one input group across the test windows (PM2.5 history, each met variable, time encodings) and reports the RMSE increase. `evals.degraded_met` (all models) degrades the decoder's seven met variables: noise with std = f × training std × (lead − 1) / 71 for f in `NOISE_LEVELS`, clipped to physical limits, or held at the origin-hour values for all 72 hours. Both rerun seeds 1–10 (`ANALYSIS_SEEDS`) and write `permutation.csv` / `degraded_met.csv` in the configuration's folder.

## Outputs

```
artifacts/<model>/<city or source-city>/<ar|direct>_<settings>/
    tuning.json
    <variant>/seed<n>/   model.pt | trees/  predictions.npz  val_predictions.npz  stations.csv
                         by_lead.csv  lead_curve.csv  metrics.json  params.json  [feature_importance.csv]
    permutation.csv  degraded_met.csv
artifacts/baselines/<persistence|climatology|ridge>/<city>/<settings>/test/
artifacts/comparison.csv  lead_comparison.csv  ensemble.csv  exceedance.csv  seasons.csv  distance_gain*.csv
artifacts/logs/<time>_<script or model>_<city or source-city>.txt
```

Every runnable script writes one log with everything it printed, headed by the command, arguments, library versions, config and database copy dates. The log also holds station completeness, filters, split and normalization for every city loaded; every tuning point's history; pretraining and fine-tuning histories; GBT per-model scores; each source station's distance and weight; per-station test scores; and any error with its traceback.

## Not built

LOSO training and scoring (kriging), and rolling-origin evaluation.
