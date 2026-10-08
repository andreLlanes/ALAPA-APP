""" Training and data settings shared by every run. Tuned hyperparameters live in grid_tuning.json.
"""

# Trim Los Angeles to start at Bangkok's first record, so the two sources differ in
# climate rather than in record length.
MATCH_LA_BKK = True

# Default filters
DEFAULT_LONGEST_GAP = 6
DEFAULT_COMPLETENESS = 70
LONGEST_GAP_CHOICES = list(range(13))
COMPLETENESS_CHOICES = [0, 25, 50, 70, 90]
ABLATION_CHOICES = [25, 50, 75, 100]
DEFAULT_ABLATION = 100   # percent of the training data used

# Run defaults
TUNING_SEED = 0                     # trains every grid point; never used for a reported result
DEFAULT_SEEDS = list(range(1, 31))
ANALYSIS_SEEDS = list(range(1, 11))  # seeds that permutation importance and degraded met rerun
SAVE_VAL_PREDICTIONS = True          # conformal intervals calibrate on them
STORED_PREDICTION_DTYPE = "float16"
DEFAULT_DEVICE = "cpu"
BLOCK_CHOICES = ["0", "1", "2", "all"]
DEFAULT_BLOCK = "all"

# Training loop
MAX_EPOCHS = 100
PATIENCE = 10            # epochs without val improvement, counted once teacher forcing is 0
TF_DECAY_RATE = 0.05     # teacher forcing drops by this much per epoch (1.0 -> 0.0)
DROPOUT = 0.0
LR_DECAY_FACTOR = 0.5    # learning rate multiplier when val stalls (1.0 = off)
LR_DECAY_PATIENCE = 5    # stalled epochs before the learning rate decays
DETERMINISTIC = False    # identical GPU runs, at a speed cost

# Scoring
EVAL_BATCH = 512          # windows per batch when predicting val/test
MIN_TEST_WINDOWS = 100   # a station needs this many test windows to count in the averages
MIN_PAIRED_STATIONS = 5  # common stations needed for the paired Wilcoxon test
MIN_OBS_PER_HOUR = 10    # climatology: training hours needed at each hour of day

# Analyses
INTERVAL_COVERAGE = 0.9
CI_LEVEL = 0.95
LEAD_TEST_HOURS = [1, 24, 48, 72]
NOISE_LEVELS = [0.25, 0.5]   # noise std at the last lead, as a fraction of each variable's training std
NOISE_TABLE_SIZE = 10000     # windows share this many noise draws, matched by window id
PERTURBATION_SEED = 2026     # noise draws and permutations
MET_BOUNDS = {"wind_gusts_ms": (0, None), "precipitation_mm": (0, None),
              "humidity_pct": (0, 100)}   # physical limits kept after adding noise
EXCEEDANCE_PRIMARY = 50      # ug/m3, 24-hour guideline
EXCEEDANCE_SENSITIVITY = 35
BLOCK_HOURS = 24
BLOCK_MIN_OBSERVED = 0.75    # share of a block's hours that must be observed to score it
ANALYSIS_CHUNK = 20000       # windows per step when taking percentiles over seeds
TOP_QUANTILE = 0.9           # "worst hours": observed above this quantile of the city's training PM2.5
SEASONS = {"mm": {"wet": [5, 6, 7, 8, 9, 10], "dry": [11, 12, 1, 2, 3, 4]},
           "bk": {"wet": [5, 6, 7, 8, 9, 10], "dry": [11, 12, 1, 2, 3, 4]},
           "la": {"wet": [11, 12, 1, 2, 3], "dry": [4, 5, 6, 7, 8, 9, 10]}}

# MMD alignment
MMD_WARMUP_STEPS = 0     # steps to ramp lambda in from 0 (0 = constant)

# Station distances (GBT transfer weights)
DISTANCE_COLUMN = "mmd2_pair"
DISTANCE_SAMPLE = 1000   # rows per side in each MMD draw
DISTANCE_GLOBAL = 1000   # rows per city behind the fixed bandwidth
DISTANCE_REPEATS = 3
DISTANCE_MIN_ROWS = 168  # one week of complete hours; fewer leaves the distance undefined
DISTANCE_SEED = 2026

# Leave-one-station-out folds (tuning/loso_fold.py)
LOSO_CITY = "mm"
LOSO_FOLDS = 20
LOSO_COMPLETENESS = 0.70  # completeness over the active range (training pool and eligibility)
LOSO_K_NEIGHBOR = 3       # density = distance to the k-th nearest eligible station
LOSO_BANDS = 4
LOSO_BAND_NAMES = {3: ("core", "middle", "periphery"),
                   4: ("core", "inner", "outer", "periphery")}
LOSO_SEED = 2026
