"""IMPORTANT TO READ FOR ALL: Metro Manila stations with no training-period data.

These 18 stations first report after the Metro Manila training cutoff
(2025-08-25 14:00 UTC, see cities.split_bounds), so no model is ever fitted on
them. They are frozen here so every experiment treats them the same way.
Selected from openaq.merged_clean on 2026-09-17.

NEW_STATION_EVAL
    At least 720 test-period hours (30 days). Scored as never-seen locations,
    without refitting, alongside the LOSO folds.
NEW_STATION_INPUT_ONLY
    Fewer than 720 test-period hours. Kept as inputs while they report, never
    scored.

Each comment gives the OpenAQ name, owner, first and last day of data, and
test-period hours.
"""

NEW_STATION_EVAL = [
    "openaq:5975180",  # Valenzuela                   DLSU         2025-10-01 .. 2026-06-30  3494 h
    "openaq:5976206",  # Las Pinas                    DLSU         2025-10-02 .. 2026-06-30  3653 h
    "openaq:6230501",  # North Caloocan Nova Romania  AirGradient  2026-02-06 .. 2026-06-30  3433 h
    "openaq:6303934",  # Sta. Ana Hospital            Clarity      2026-04-11 .. 2026-06-30  1505 h
    "openaq:6303935",  # Near Puregold Tayuman        Clarity      2026-04-11 .. 2026-06-30  1510 h
    "openaq:6303936",  # Anda Circle                  Clarity      2026-04-11 .. 2026-06-30  1508 h
    "openaq:6303937",  # San Sebastian Residences     Clarity      2026-04-11 .. 2026-06-30  1511 h
    "openaq:6338564",  # Rajah Sulayman Park          Clarity      2026-05-07 .. 2026-06-30  1041 h
    "openaq:6340868",  # Tivoli                       AirGradient  2026-05-09 .. 2026-06-30  1249 h
]

NEW_STATION_INPUT_ONLY = [
    "openaq:5963700",  # 313_Cubao                    AirGradient  2025-09-30 .. 2025-10-01     0 h
    "openaq:5974688",  # Ayala MRT                    AirGradient  2025-10-02 .. 2025-10-09     0 h
    "openaq:5995084",  # Pasay Rotonda                AirGradient  2025-10-03 .. 2026-01-19     0 h
    "openaq:6209259",  # Ramon Avancena High School   AirGradient  2026-01-21 .. 2026-06-17    93 h
    "openaq:6265529",  # Marikina                     AirGradient  2026-03-09 .. 2026-03-09     1 h
    "openaq:6275863",  # pasay city                   AirGradient  2026-03-18 .. 2026-05-18   123 h
    "openaq:6384709",  # ADB Courtyard (Demo)         Clarity      2026-06-02 .. 2026-06-03    16 h
    "openaq:6391974",  # Valle1                       AirGradient  2026-06-07 .. 2026-06-30   487 h
    "openaq:6393360",  # Ayala Ave.                   Clarity      2026-06-08 .. 2026-06-30   370 h
]

NO_TRAINING_STATIONS = NEW_STATION_EVAL + NEW_STATION_INPUT_ONLY
