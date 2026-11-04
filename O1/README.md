# O1: PM2.5 data pipeline (ETL + cleaning)

Pulls raw ground-station PM2.5 and meteorology into Postgres and produces the
cleaned `merged_clean` table that O2 trains on, for Metro Manila, Bangkok and Los
Angeles. Also builds the Metro Manila spatial covariate rasters.

## Setup

Put `.env` at the repo root with `PG_DSN` (or `PG_HOST`, `PG_PORT`, `PG_DB`,
`PG_USER`, `PG_PASSWORD`), plus `OPENAQ_API_KEY` for the ground-station ETL and
`SH_CLIENT_ID` / `SH_CLIENT_SECRET` for NDVI.

```
cd O1
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run everything from inside `O1/`, with `Common/` importable:

```
export PYTHONPATH=..        # Windows: set PYTHONPATH=..
```

The covariate scripts need extra packages (`rasterio`, `geopandas`, `shapely`,
`scipy`, `sentinelhub`, `requests`); install those only if building rasters.

## Run

```
python ETL/openaq_etl.py        # ground-station PM2.5  -> gs_measurements, gs_stations
python ETL/openmeteo_etl.py     # ECMWF IFS meteorology -> ifs_met
python Clean/main.py            # clean + merge + features -> merged_clean
```

ETL and cleaning track progress in `*_progress.txt` and resume from it, so an
interrupted run can be re-run safely. Covariate scripts under `ETL/Covariates/`
are standalone and write GeoTIFFs on the common 100 m grid (EPSG:32651).

## What gets cleaned

Per station, onto a complete hourly grid:

- **Station keys** are `openaq-<id>`.
- **Physical bounds:** PM2.5 `<= 0` or `> 1000` is dropped.
- **Duplicates:** repeated station-hours are collapsed by mean.
- **Flatlines:** runs of 6+ identical values are removed for low-cost sensors
  only; reference monitors may legitimately hold a constant value and are kept.
- **No interpolation.** Gaps stay NaN and carry `gap_length` (the length of the
  gap each missing hour belongs to; NULL when present), so filling can be fit per
  split at training time. Gaps longer than 12 hours are trimmed to 6 hours at each
  edge with the interior removed, so `merged_clean` is not a perfectly contiguous
  grid.
- **Meteorology** is wind u/v (not speed/direction), joined from the nearest IFS
  grid cell. **Temporal features** are cyclical hour/day-of-week/month, encoded in
  each city's own local time.

## Output

Postgres tables (schema `openaq`): `gs_stations`, `gs_measurements`, `ifs_met`,
and `merged_clean` (the modelling input — cleaned PM2.5 joined to meteorology with
time features, one row per station-hour). Cleaning also writes a per-station
`provenance_log.csv`.
