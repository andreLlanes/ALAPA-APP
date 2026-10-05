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

# MMD alignment
MMD_WARMUP_STEPS = 0     # steps to ramp lambda in from 0 (0 = constant)

# Station distances (GBT transfer weights)
DISTANCE_COLUMN = "mmd2_pair"
DISTANCE_SAMPLE = 1000   # rows per side in each MMD draw
DISTANCE_GLOBAL = 1000   # rows per city behind the fixed bandwidth
DISTANCE_REPEATS = 3
DISTANCE_MIN_ROWS = 168  # one week of complete hours; fewer leaves the distance undefined
DISTANCE_SEED = 2026
