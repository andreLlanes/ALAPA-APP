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
python -m data.database                        # copy the database (once)
python train.py --model lstm --city mm         # tune, train, score, compare
python train.py --model gnn --city mm --transfer true --source bk
python -m evals.eval                           # compare everything trained
python -m tuning.loso_fold                     # select the 20 LOSO stations (once)
```

After the database changes: `python -m data.database --refresh`, then delete `artifacts/`.

## Arguments (`train.py`)

| Argument | Values | Default |
|---|---|---|
| `--model` | lstm, gnn, gbt | required |
| `--city` | mm, bk, la | required |
| `--transfer` | true, false | false |
| `--source` | mm, bk, la (not the city) | required with transfer |
| `--training-variant` | LSTM/GNN: frozen, full, all · GBT: pooled, weighted, all | all |
| `--autoregressive` | true, false | false |
| `--longest-gap` | 0–12 | 6 |
| `--completeness` | 0, 25, 50, 70, 90 | 70 |
| `--data-ablation` | 25, 50, 75, 100 | 100 |
| `--block` | 0, 1, 2, all | all |
| `--seed` | e.g. 42,1234,2026 | 42,1234,2026 |
| `--device` | cpu, cuda | cpu |

## Key details

- **Data:** the database is copied to `data/cache/` once; every run reads that copy. Windows are filtered (completeness, then longest lookback gap), then split 70/15/15 by window count with a 72h purge. No fixed dates.
- **Gaps:** lookback gaps are interpolated within the window; windows with any missing forecast hour are dropped.
- **LOSO stations** (`loso_stations.json`) are never removed by the completeness filter.
- **Tuning:** grid search on validation RMSE (`grid_tuning.json`). Hyperparameters are shared across seeds (tuned with the first seed, whose winning model is reused) but never across settings: every model, city, transfer variant, gap, completeness and ablation block tunes its own.
- **Transfer (LSTM/GNN):** parameter transfer and MMD-aligned pretraining on the source, each fine-tuned frozen (encoder fixed) and/or full. Fine-tuning starts at teacher forcing 0.
- **Transfer (GBT):** target plus source windows, either *pooled* (every source window weighs 1) or *weighted* (each source station by exp(−d/τ), from its MMD distance to the target). τ is tuned with the tree settings, separately for every source-target pair.
- **Directions:** any city can be the source or the target (e.g. `--city bk --source mm`). Runs can be done in any order.
- **Baselines:** persistence and climatology run automatically after training, on the same test windows.
- **Scoring:** metrics per station, then averaged with equal weight over stations with at least 100 test windows; reported overall and for seen/unseen stations, with a Holm-adjusted paired Wilcoxon test against the best row.
- **Resuming:** rerunning a command skips finished seeds and grid points and resumes interrupted training (except MMD-aligned pretraining, which restarts).
- **Settings:** all fixed settings live in `config.py`; tuned ones in `grid_tuning.json`.

## Outputs

```
artifacts/<model>/<city or source-city>/<ar|direct>_<settings>/<variant>/seed<n>/
    predictions.npz  stations.csv  by_lead.csv  lead_curve.csv  metrics.json  params.json
artifacts/baselines/<persistence|climatology>/<city>/<settings>/test/
artifacts/comparison.csv
artifacts/logs/<time>_<model>_<city>_<settings>.txt   one log per train.py run
```

Each log holds everything printed, plus: the command, arguments, config and library versions;
station completeness, filters, split and normalization for every city loaded; every tuning point's
training history; pretraining and fine-tuning histories (with MMD per epoch); GBT per-model scores;
each source station's distance and weight; and per-station test scores.
