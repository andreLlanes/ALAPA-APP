# O1 — PM2.5 Forecasting Data Pipeline

Data acquisition, cleaning, and dataset construction for 72-hour PM2.5
forecasting across Metro Manila, Bangkok, and Los Angeles.

## Pipeline stages

1. **ETL** (`ETL/`) — pull raw data into Postgres.
   - `OpenAQ.py` — ground-station PM2.5 → `gs_measurements`, `gs_stations`.
   - `OpenMeteo.py` — ECMWF IFS meteorology → `ifs_met`.
   - `Covariates/` — spatial covariate rasters (buildings, roads, NDVI, GTFS)
     on the common 100 m grid → `Covariates/Outputs/`.
2. **Clean** (`Clean/`) — grade-aware QC, station↔grid-cell merge, feature
   construction → `merged_clean` and `merged_masked` tables.
3. **Builders** (`Builders/`) — model-ready datasets (LSTM/GNN sequences, GBT
   tables) as per-station shards → `Outputs/`.
4. **Common** (`Common/`) — shared schema (column names, feature layout, horizon)
   imported by Clean and Builders.

## Setup

```bash
cd O1
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
# 1. Backfill raw data
python ETL/OpenAQ.py
python ETL/OpenMeteo.py

# 2. Clean and merge
python Clean/main.py

# 3. Build model-ready datasets
python Builders/main.py --source all --city all --model all
```

Each stage writes a progress ledger (`*_progress.txt`) and resumes from it, so an
interrupted run can be re-run safely.

## Notes

- Datasets in `Outputs/` are unnormalized; train-only normalization happens at
  training time to avoid leakage.
- GBT datasets are stored as normalized backward/leads Parquet pairs (rejoin on
  `location_key, origin`); GNN datasets defer the `k` hyperparameter (a `nodes.npz`
  holds the graph coordinates).
