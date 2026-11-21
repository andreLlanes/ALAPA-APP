""" Print the numbers for the 'OpenAQ PM2.5' slide (Chapter 5), per city, from the cleaned data:
    stations, period, readings, hours missing, station completeness and PM2.5 levels. Los Angeles
    is shown twice: as downloaded, and from Bangkok's first record (what the models use when
    O3's MATCH_LA_BKK is on).
    python figures/openaq_stats.py
"""

import shared  # puts O3 on the path; import before O3 modules
import config
from Common.schema import KEY_COL, PM25_COL, TIME_COL


def describe(name, rows):
    """ Print one city's summary. Hours missing and completeness count the hours between each
        station's first and last reading, as O3's station completeness does.
    """
    obs = rows[rows[PM25_COL].notna()]
    span = obs.groupby(KEY_COL)[TIME_COL]
    hours = (span.max() - span.min()).dt.total_seconds() / 3600 + 1
    comp = obs.groupby(KEY_COL).size() / hours * 100
    q = obs[PM25_COL].quantile([.1, .25, .5, .75, .9, .99]).round(1).tolist()
    print(f"== {name}")
    print(f"  stations {comp.size}   readings {len(obs):,}   "
          f"from {obs[TIME_COL].min():%d %b %Y} to {obs[TIME_COL].max():%d %b %Y} (UTC)")
    print(f"  hours missing {1 - len(obs) / hours.sum():.1%}   completeness median {comp.median():.1f}%   "
          f">= {config.DEFAULT_COMPLETENESS}%: {int((comp >= config.DEFAULT_COMPLETENESS).sum())}   "
          f">= 70%: {int((comp >= 70).sum())}")
    print(f"  PM2.5 mean {obs[PM25_COL].mean():.1f}   p10/p25/median/p75/p90/p99 {q}")


cols = [KEY_COL, TIME_COL, PM25_COL]
for code in ("mm", "bk", "la"):
    describe(shared.CITY_NAMES[code], shared.read_city(code, cols))
bk_start = shared.read_city("bk", [TIME_COL])[TIME_COL].min()
la = shared.read_city("la", cols)
describe(f"Los Angeles from Bangkok's first record ({bk_start:%d %b %Y})", la[la[TIME_COL] >= bk_start])
