# figures: slide and paper figures

Scripts that draw the charts for the defense slides from the project's own data and code. Each one reads the cleaned data (local copies of `openaq.merged_clean`) and imports O3's code for the fill rule, completeness and settings, so the charts show what the models actually do.

Output goes to `figures/Outputs/`, which git ignores. Re-run a script to update its charts after the data or the O3 settings change.

## Setup

Uses the O3 packages plus matplotlib:

```
pip install -r O3/requirements.txt matplotlib
```

The data must be copied first (`python O3/data/database.py`). The scripts read `O3/data/cache/<city>.parquet`, falling back to `O2/data/cache/` (Los Angeles may only be copied there).

## Run (from the repo root)

```
python figures/openaq_stats.py                 # numbers for the OpenAQ PM2.5 slide, per city
python figures/openaq_figs.py                  # OpenAQ PM2.5 slide: station coverage, readings per city
python figures/openaq_figs.py --trim-la        #   same, with LA from Bangkok's first record (what the models use)

python figures/interpolation_stats.py          # interpolation numbers + backup accuracy chart
python figures/interpolation_stats.py --candidates   # also list real samples for the slide example
python figures/interpolation_figs.py           # interpolation slide: real sample, the four gap cases
python figures/alignment_figs.py               # alignment slide: station-to-weather map, hourly timeline, table excerpt
python figures/windowing_figs.py               # windowing slide: one station's record cut into windows, one window as a matrix
python figures/forecasting_figs.py             # forecasting slide: the GNN's station graph, the GBT's inputs (--k sets the neighbours drawn)
python figures/kriging_figs.py                 # regression-kriging slide: kriged map + uncertainty, fitted variogram (--lead sets the hour)
python figures/evaluation_figs.py              # evaluation setup slide: the two baselines on one test window, the leave-one-station-out stations
python figures/setup_figs.py                   # experimental setup slide: training setups per model, hyperparameter grid (from O2's code)
python figures/m1_figs.py                      # M1 slides: datasets per city, Manila windows per station, Bangkok/LA graphs, GBT rows (uses O2's loader)
python figures/m2m3_figs.py                    # M2 and M3 overview: one Manila test window, Manila-only vs Bangkok transfer forecasts (reads O2/artifacts)
python figures/m2m3_window_figs.py             # M2 and M3, one figure each: the 72 h in / 72 h out window, and how the milestone trains before forecasting it
python figures/m4_figs.py                      # M4 overview: the app's 100 m surface at +1/+24/+48/+72 h; prints the map's check against station readings
python figures/m5_figs.py                      # M5 overview: Manila test scores on all five metrics (reads O2/artifacts/comparison.csv)
python figures/split_figs.py                   # split slide (O3, 60/20/20): windows by forecast date, zoom on the purge
python figures/split_figs.py --train 0.70 --val 0.15 --completeness 70   # same with O2's settings (files end _70_15)
```

| Output | Slide |
|---|---|
| `openaq_coverage.png`, `openaq_cities.png` | OpenAQ PM2.5 (Chapter 5) |
| `interpolation_example.png`, `interpolation_cases.png` | Data Preprocessing: interpolation (Chapter 4) |
| `backup_interpolation_accuracy.png` | Appendix backup: straight line vs copying the last reading |
| `alignment_map.png`, `alignment_timeline.png`, `alignment_table.png` | Data Preprocessing: alignment (Chapter 4) |
| `windowing_record.png`, `windowing_matrix.png` | Data Preprocessing: windowing (Chapter 4) |
| `windowing_record_19jun.png` | Windowing, alternative record: the window at 5 PM, 19 Jun, the last kept before a missing reading |
| `split_timeline.png`, `split_purge.png` | Data Preprocessing: split (Chapter 4) |
| `windowing_matrix.png`, `forecasting_graph.png`, `forecasting_gbt_inputs.png` | Forecasting (Chapter 4) |
| `kriging_map.png`, `kriging_variogram.png` | Regression-Kriging (Chapter 4) |
| `evaluation_baselines.png`, `evaluation_loso.png` | Evaluation Setup (Chapter 4) |
| `setup_experiments.png`, `setup_grid.png` | Experimental Setup (Chapter 5) |
| `m1_datasets.png` | M1: Overview (Chapter 5) |
| `m1_manila.png` | M1: Metro Manila (Chapter 5) |
| `m1_lstm_windows.png`, `m1_graphs.png` | M1: LSTM and GNN, Bangkok and Los Angeles (Chapter 5) |
| `m1_gbt_rows.png` | M1: GBT, Bangkok and Los Angeles (Chapter 5) |
| `m2m3_forecasts.png` | M2 and M3: Overview (Chapter 5) |
| `m2_windowing.png` | M2: the forecast window and how M2 trains (Chapter 5) |
| `m3_windowing.png` | M3: the forecast window and how M3 trains, per model (Chapter 5) |
| `m4_surfaces.png` | M4: Overview (Chapter 5) |
| `m5_metrics.png` | M5: Overview (Chapter 5) |

## Notes

- `shared.py` holds the paths, the chart style and the Metro Manila base map. New figure scripts should use it.
- The kriging figures read the web app's map data (`frontend/public/data`): a past forecast from the test period that used recorded weather, not a live forecast.
- The evaluation example window uses O2's split and filters (70/15/15, completeness 70), because the temporal models were trained with O2 on Kaggle; the leave-one-station-out stations are O3's (`O3/loso_stations.json`).
- `m1_figs.py` builds the datasets with O2's own loader (`O2/data/load.py`, read from `O2/data/cache/`), not O3's, because the trained models are O2's; it stops if O3's split or settings load instead. Loading Los Angeles takes a few minutes.
- `m2m3_figs.py` reads the Kaggle results in `O2/artifacts` (the Manila runs' `predictions.npz` and `val_predictions.npz`). It draws the evaluation slide's example window, whose error is near the median for these models. Each line is the average of 30 seeds. Where a transfer method was run with the encoder frozen and with every layer fine-tuned, it draws the version with the lower validation RMSE.
- The app holds O3's three maps of the same past forecast (LSTM, GNN, GBT; one manifest, `pm25_<model>.u8.gz` and `sd_<model>.u8.gz` per model). `m4_surfaces.png` and the kriging figures draw the LSTM's (`FIGURE_MODEL` in `shared.py`), so they match the slides; `shared.web_map()` reads either manifest format. `m4_figs.py` scores the map against `readings.json` the way the app's Checks panel does: every station-hour with both a reading and a forecast, at the station's 100 m cell.
- `m5_figs.py` takes its scores from O2's own evaluation (`O2/artifacts/comparison.csv`). It picks the transfer variants with the same validation rule as `m2m3_figs.py`.
- The GNN graph is drawn at Manila's busiest forecast hour (11 stations); most hours have 1 to 3, which the chart states.
- The interpolation example (station and forecast time) is set at the top of `interpolation_figs.py`; pick another from `interpolation_stats.py --candidates`.
- The alignment map's land outline comes from the web app's map grid (`frontend/public/data`, made by `O3/spatial/export.py`); without it the map is drawn without an outline.
- The accuracy check in `interpolation_stats.py` is not part of the training pipeline; it only backs the choice of a straight-line fill.
