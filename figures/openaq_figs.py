""" Figures for the 'OpenAQ PM2.5' slide (Chapter 5): when each Metro Manila station reported (with
    the completeness filter), and hourly readings per city.
    python figures/openaq_figs.py              Los Angeles as downloaded
    python figures/openaq_figs.py --trim-la    Los Angeles from Bangkok's first record (what the models use)
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

import shared  # puts O3 on the path; import before O3 modules
from shared import AXIS, BLUE, INK, INK2, MUTED
import config
from Common.schema import KEY_COL, PM25_COL, TIME_COL
from data.load import loso_keys
from data.windows import station_completeness

parser = argparse.ArgumentParser()
parser.add_argument("--trim-la", action="store_true",
                    help="start Los Angeles at Bangkok's first record, as the models do (MATCH_LA_BKK)")
args = parser.parse_args()
shared.use_style(13)
GAP_GRAY = "#e6e4dc"
RAMP = LinearSegmentedColormap.from_list("blue", shared.BLUE_RAMP)
COLS = [KEY_COL, TIME_COL, PM25_COL]

# ---------- When each Metro Manila station reported ----------
rows = shared.read_city("mm", COLS)
keys, station = np.unique(rows[KEY_COL].to_numpy(str), return_inverse=True)
hours = ((rows[TIME_COL] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(hours=1)).to_numpy(np.int64)
pm = rows[PM25_COL].to_numpy(np.float32)
comp = station_completeness(station, hours, pm, len(keys)) * 100
cut = config.DEFAULT_COMPLETENESS
# O3's station filter: complete enough, or one of the LOSO stations (always kept).
used = (comp >= cut) | np.isin(keys, loso_keys())

obs = rows[rows[PM25_COL].notna()].assign(t=lambda d: d[TIME_COL].dt.tz_convert(shared.LOCAL_TZ).dt.tz_localize(None))
info = obs.groupby(KEY_COL)["t"].agg(["min", "max"]).reindex(keys)
info["comp"], info["used"] = comp, used
# Used stations first, then the rest; each group in order of first reading.
info = pd.concat([info[info.used].sort_values("min"), info[~info.used].sort_values("min")])
n_used = int(info.used.sum())

week_of = lambda t: t.dt.to_period("W-SUN").dt.start_time  # Monday-start weeks
weeks = pd.date_range(week_of(obs.t).min(), week_of(obs.t).max(), freq="W-MON")
counts = (obs.assign(w=week_of(obs.t)).groupby([KEY_COL, "w"]).size()
          .unstack(fill_value=0).reindex(index=info.index, columns=weeks, fill_value=0))
share = counts.to_numpy() / 168
w = weeks.to_numpy()
active = (w[None, :] >= week_of(info["min"]).to_numpy()[:, None]) & (w[None, :] <= week_of(info["max"]).to_numpy()[:, None])

# Facts for the annotations, from the data.
joined = info["min"].dt.to_period("M").value_counts()
peak = joined.idxmax()
monthly = obs.assign(m=obs.t.dt.to_period("M")).groupby("m")[KEY_COL].nunique()
before = monthly[monthly.index < pd.Period(f"{peak.year}-01", "M")]

n = len(info)
ypos = np.arange(n) + np.where(info.used.to_numpy(), 0, 1.6)  # a small break before the unused group
fig = plt.figure(figsize=(7.3, 4.6))
gs = fig.add_gridspec(1, 2, width_ratios=[4.0, 1], wspace=0.05, left=0.03, right=0.965, top=0.80, bottom=0.12)
ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
for i in range(n):
    for j in np.flatnonzero(active[i]):
        c = GAP_GRAY if share[i, j] == 0 else RAMP(min(share[i, j], 1.0))
        ax.add_patch(Rectangle((j, ypos[i] - 0.42), 1.0, 0.84, color=c, lw=0))
ax.grid(False)
ax.set_xlim(0, len(weeks))
ax.set_ylim(ypos[-1] + 1, -1)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
years = [pd.Timestamp(f"{y}-01-01") for y in range(weeks[0].year + 1, weeks[-1].year + 1)]
ax.set_xticks([weeks.searchsorted(t) for t in years])
ax.set_xticklabels([str(t.year) for t in years], fontsize=12)
peak_x = weeks.searchsorted(peak.start_time)
first_peak_row = int(np.argmax(info["min"].dt.to_period("M").to_numpy() == peak))
ax.annotate(f"{joined.max()} stations joined\nin {peak.start_time:%B %Y}", (peak_x - 0.5, first_peak_row + 6),
            xytext=(peak_x - 50, first_peak_row + 10), fontsize=13, color=INK2, va="center",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
if len(before):
    early = n_used if n_used < n else 0  # point at the first unused station, else the first one
    ax.annotate(f"Before {peak.year}: {before.min()} to {before.max()}\nstations at a time",
                (weeks.searchsorted(info["min"].iloc[early]) + 2, ypos[early] - 0.2), xytext=(4, ypos[n_used] - 14),
                fontsize=13, color=INK2, va="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))

bx.grid(False)
bx.barh(ypos, info.comp, height=0.8, color=np.where(info.used, BLUE, AXIS), lw=0)
bx.axvline(cut, color=INK2, lw=1)
bx.set_xlim(0, 100)
bx.set_ylim(ax.get_ylim())
bx.set_yticks([])
bx.spines["left"].set_visible(False)
bx.set_xticks([0, cut, 100])
bx.set_xticklabels(["0", f"{cut}%", "100%"], fontsize=12)
bx.text(50, -2.0, "Hours reported\nfirst → last reading", fontsize=12, color=INK2, ha="center", va="bottom")
bx.text(4, ypos[n_used // 2 + 8], f"{n_used} used", fontsize=13, color="white", weight="semibold", va="center")
bx.text(cut + 4, ypos[min(n_used + 4, n - 1)], f"{n - n_used} not\nused", fontsize=13, color=INK2, va="center")

fig.text(0.03, 0.945, "Weekly reporting coverage by station, Metro Manila", fontsize=16, weight="semibold")
lx, ly, sw = 0.03, 0.855, 0.026
for i, s in enumerate(np.linspace(0.05, 1, 5)):
    fig.patches.append(Rectangle((lx + i * sw, ly), sw, 0.032, color=RAMP(s), transform=fig.transFigure, figure=fig))
fig.text(lx + 5 * sw + 0.01, ly + 0.016, "hours with a reading: few → all", fontsize=12, color=INK2, va="center")
fig.patches.append(Rectangle((0.52, ly), sw, 0.032, color=GAP_GRAY, transform=fig.transFigure, figure=fig))
fig.text(0.52 + sw + 0.01, ly + 0.016, "none that week", fontsize=12, color=INK2, va="center")
fig.text(0.03, 0.02, f"One row per station, one cell per week. OpenAQ after cleaning, "
         f"{obs.t.min():%b %Y} – {obs.t.max():%b %Y}.", fontsize=11, color=MUTED)
fig.savefig(shared.out_path("openaq_coverage.png"), dpi=200)
plt.close(fig)

# ---------- Hourly readings per city ----------
bk_start = shared.read_city("bk", [TIME_COL])[TIME_COL].min()
readings = {}
for code in ("mm", "bk", "la"):
    city = shared.read_city(code, [TIME_COL, PM25_COL])
    if code == "la" and args.trim_la:
        city = city[city[TIME_COL] >= bk_start]
    readings[code] = int(city[PM25_COL].notna().sum())
readings = pd.Series(readings)

fig, ax = plt.subplots(figsize=(5.6, 2.5))
fig.subplots_adjust(left=0.25, right=0.80, top=0.80, bottom=0.08)
vals = readings / 1e6
ax.grid(False)
ax.barh(range(3), vals, height=0.62, color=[BLUE, AXIS, AXIS], lw=0)
ax.set_ylim(2.6, -0.6)
ax.set_xlim(0, np.ceil(vals.max()) + 0.5)
ax.set_yticks(range(3))
ax.set_yticklabels([shared.CITY_NAMES[c] for c in readings.index], fontsize=13, color=INK)
ax.tick_params(axis="y", length=0)
ax.set_xticks([])
ax.spines["bottom"].set_visible(False)
for y, (v, r) in enumerate(zip(vals, readings)):
    ax.text(v + 0.12, y, f"{r:,}" if v < 1 else f"{v:.2f} M", va="center", fontsize=13,
            color=INK if y == 0 else INK2, weight="semibold" if y == 0 else "normal")
fig.text(0.25, 0.92, "Hourly PM2.5 readings per city, after cleaning", fontsize=13, weight="semibold", va="center")
fig.savefig(shared.out_path("openaq_cities.png"), dpi=200)
plt.close(fig)

print("saved to", shared.OUT)
