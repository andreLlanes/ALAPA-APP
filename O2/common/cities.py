#Per-city metadata shared by the O2 experiments.


from datetime import datetime, timezone

import numpy as np
import pandas as pd

TARGET_CITY = "Metro Manila"
SOURCE_CITIES = ["Bangkok", "Los Angeles"]
ALL_CITIES = [TARGET_CITY] + SOURCE_CITIES

# Short slugs, matching O1/Builders.
CITY_SLUG = {"Metro Manila": "MM", "Bangkok": "BK", "Los Angeles": "LA"}

CITY_TZ = {
    "Metro Manila": "Asia/Manila",
    "Bangkok": "Asia/Bangkok",
    "Los Angeles": "America/Los_Angeles",
}

# OpenAQ hours are stored under their end time (period.datetimeTo) and the ETL
# stops at this instant, so it is also the last stamp in the PM2.5 tables.
RECORD_END = datetime(2026, 6, 30, tzinfo=timezone.utc)

# Los Angeles is cut to Bangkok's delivered start (no Bangkok station reports
# before 2021-04-29) so the two sources differ in climate rather than in record
# length, which H2 needs. Set to "full" to use LA's 2017 start instead.
LA_WINDOW = "matched"
_LA_START = {
    "matched": datetime(2021, 4, 29, tzinfo=timezone.utc),
    "full": datetime(2017, 1, 1, tzinfo=timezone.utc),
}

CITY_PERIODS = {
    "Metro Manila": (datetime(2023, 9, 6, tzinfo=timezone.utc), RECORD_END),
    "Bangkok": (datetime(2021, 4, 29, tzinfo=timezone.utc), RECORD_END),
    "Los Angeles": (_LA_START[LA_WINDOW], RECORD_END),
}

SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def split_bounds(city):
    """Return {"train" | "val" | "test": (start, end)} half-open UTC intervals."""
    start, end = (pd.Timestamp(t) for t in CITY_PERIODS[city])
    span = end - start
    train_end = (start + span * SPLIT_FRACTIONS["train"]).floor("h")
    val_end = (start + span * (SPLIT_FRACTIONS["train"]
                               + SPLIT_FRACTIONS["val"])).floor("h")
    return {
        "train": (start, train_end),
        "val": (train_end, val_end),
        "test": (val_end, end + pd.Timedelta(hours=1)),
    }


def local_time_features(epoch_s, city):
    """Return the six cyclical encodings, computed in the city's local time."""
    local = pd.to_datetime(np.asarray(epoch_s, dtype=np.int64), unit="s",
                           utc=True).tz_convert(CITY_TZ[city])
    parts = []
    for t, period in ((local.hour, 24), (local.dayofweek, 7), (local.month - 1, 12)):
        angle = 2.0 * np.pi * np.asarray(t, dtype=float) / period
        parts += [np.sin(angle), np.cos(angle)]
    return np.column_stack(parts)
