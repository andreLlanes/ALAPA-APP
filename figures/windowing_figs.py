""" Figures for the windowing slide (Chapter 4, Data Preprocessing), from the cleaned data and O3's
    own window rules (data/windows.py, data/load.py):
      windowing_record.png  one station's record, and for every hour whether a window whose forecast
                            is made at that hour is kept, or why not
      windowing_record_19jun.png  the same, with the window at 5 PM, 19 Jun: the last one kept before
                            a missing reading
      windowing_matrix.png  one kept window as the model receives it: past 72 h x 14 columns,
                            next 72 h x 13 columns (weather + clock), and the 72 PM2.5 targets
    python figures/windowing_figs.py
"""

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

import shared  # puts O3 on the path; import before O3 modules
from shared import AXIS, BLUE, GRID, INK, INK2, MUTED, ORANGE
import config
from Common.schema import KEY_COL, LOOKBACK_H, HORIZON_H, MET_COLS, PM25_COL, TIME_COL, TIME_COLS, WINDOW_H
from data.load import interpolate
from data.windows import city_arrays, longest_lookback_gaps, running_count

EXAMPLE_STATION = "openaq:4049149"
# A stretch of that station's record with every outcome: kept, forecast hours missing, a history
# gap over the limit, and an outage (a long gap whose middle was dropped in cleaning).
RECORD_FROM, RECORD_TO = pd.Timestamp("2025-06-02", tz=shared.LOCAL_TZ), pd.Timestamp("2025-07-09", tz=shared.LOCAL_TZ)
# The window bracketed in windowing_record_19jun.png: the last one kept before the missing reading at
# 6 PM, 22 Jun. An hour later that reading would be in the forecast, so the strip turns grey.
BRACKET_NOW = pd.Timestamp("2025-06-19T17", tz=shared.LOCAL_TZ)
# The window shown as a matrix: the interpolation slide's example (forecast made 6 PM, 9 Sep 2025).
MATRIX_ORIGIN = pd.Timestamp("2025-09-09T10", tz="UTC")
EPOCH = pd.Timestamp(0, tz="UTC")

shared.use_style(11)
rows = shared.read_city("mm")
a = city_arrays(rows)
hours = ((rows[TIME_COL] - EPOCH) // pd.Timedelta(hours=1)).to_numpy(np.int64)
sid = int(np.flatnonzero(a["keys"] == EXAMPLE_STATION)[0])
mine = np.flatnonzero(a["row_station"] == sid)
ex_id = EXAMPLE_STATION.split(":")[-1]

# ---------- The record: is a window with its forecast made at this hour kept? ----------
# The same tests as find_windows and prepare_city, but keeping the reason a window fails.
pm = a["pm"]
breaks = np.ones(len(hours), bool)
breaks[:-1] = (a["row_station"][1:] != a["row_station"][:-1]) | (np.diff(hours) != 1)
breaks_before, missing_before = running_count(breaks), running_count(np.isnan(pm))
bad_feat_before = running_count(np.isnan(a["feat"]).any(1))
first = mine[: len(mine) - WINDOW_H + 1]
outage = (breaks_before[first + WINDOW_H - 1] != breaks_before[first]) | (bad_feat_before[first + WINDOW_H] != bad_feat_before[first])
forecast_missing = missing_before[first + WINDOW_H] != missing_before[first + LOOKBACK_H]
long_gap = longest_lookback_gaps(pm, first) > config.DEFAULT_LONGEST_GAP
REASONS = [("kept", BLUE), ("not kept: a forecast hour has no reading", "#a9a8a1"),
           (f"not kept: history gap over {config.DEFAULT_LONGEST_GAP} h", INK2), ("not kept: outage in the record", "#dcdbd4")]
code = np.select([outage, forecast_missing, long_gap], [3, 1, 2], 0)
origin_hour = hours[first + LOOKBACK_H - 1]

span = pd.date_range(RECORD_FROM, RECORD_TO, freq="h", inclusive="left")
span_h = ((span.tz_convert("UTC") - EPOCH) // pd.Timedelta(hours=1)).to_numpy(np.int64)
status = pd.Series(code, index=origin_hour).reindex(span_h, fill_value=3).to_numpy()  # no row = outage
series = pd.Series(pm[mine], index=hours[mine]).reindex(span_h).to_numpy()
t = span.tz_localize(None)
when = lambda ts: f"{ts.day} {ts:%b}, " + f"{ts:%I %p}".lstrip("0")


def draw_record(now, name, missing=None):
    """ Draw the record and its status strip, with the kept window whose forecast is made at hour
        `now` (an index into the span) drawn as a bracket. `missing`, if given, is the index of an
        hour with no reading, marked on the chart.
    """
    fig, (ax, sx) = plt.subplots(2, 1, figsize=(7.3, 4.2), sharex=True, gridspec_kw=dict(height_ratios=[3.2, 0.45]))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.87, bottom=0.2, hspace=0.25)
    ax.plot(t, series, color=BLUE, lw=1.0)
    ax.set_ylim(0, np.nanmax(series) * 1.35)
    ax.set_ylabel("PM2.5 (µg/m³)", fontsize=9.5)
    ax.tick_params(labelsize=9)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    # The window, drawn as a bracket over its 144 hours.
    y_br = np.nanmax(series) * 1.2
    h0, hn, h1 = t[now - LOOKBACK_H + 1], t[now], t[min(now + HORIZON_H, len(t) - 1)]
    ax.plot([h0, hn], [y_br, y_br], color=INK, lw=2.2, solid_capstyle="butt")
    ax.plot([hn, h1], [y_br, y_br], color=MUTED, lw=2.2, solid_capstyle="butt")
    for x_ in (h0, hn, h1):
        ax.plot([x_, x_], [y_br * 0.96, y_br * 1.04], color=INK, lw=1)
    pad = pd.Timedelta(hours=8)
    ax.text(h0 - pad, y_br, "one window:  past 72 h", ha="right", va="center", fontsize=8.5, color=INK)
    ax.text(h1 + pad, y_br, "next 72 h", ha="left", va="center", fontsize=8.5, color=INK2)
    ax.text(hn, y_br * 1.07, "now", ha="center", va="bottom", fontsize=8.5, color=INK,
            bbox=dict(fc="white", ec="none", pad=1), zorder=4)
    ax.axvline(hn, color=INK, lw=0.8, ls="-", zorder=0)
    sx.axvline(hn, color=INK, lw=0.8, zorder=3)
    if missing is not None:
        m = t[missing]
        ax.axvline(m, ymax=y_br / ax.get_ylim()[1], color=INK2, lw=0.9, ls=(0, (2, 2)), zorder=0)
        ax.text(m + pad, y_br * 0.86, f"{when(m)}: no reading", ha="left", va="center", fontsize=8.5,
                color=INK, bbox=dict(fc="white", ec="none", pad=1), zorder=4)
    colours = [c for _, c in REASONS]
    for i, c in enumerate(status):
        sx.add_patch(Rectangle((mdates.date2num(t[i]), 0), 1 / 24, 1, color=colours[c], lw=0))
    sx.set_ylim(0, 1)
    sx.set_yticks([])
    sx.grid(False)
    for side in ("left", "top", "right"):
        sx.spines[side].set_visible(False)
    sx.set_xlim(t[0], t[-1])
    sx.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    sx.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    sx.tick_params(labelsize=9)
    sx.text(0, 1.15, "Each hour as a possible forecast time ('now'): is that window kept?", transform=sx.transAxes,
            fontsize=8.5, color=INK2)
    for i, (label, c) in enumerate(REASONS):
        x0 = 0.08 + (i % 2) * 0.45
        y0 = 0.075 - (i // 2) * 0.045
        fig.patches.append(Rectangle((x0, y0 - 0.012), 0.018, 0.025, color=c, transform=fig.transFigure, figure=fig))
        fig.text(x0 + 0.025, y0, label, fontsize=8.5, color=INK2, va="center")
    fig.text(0.08, 0.965, f"Kept and dropped windows by forecast hour, station {ex_id}", fontsize=11.5,
             weight="semibold", va="top")
    fig.savefig(shared.out_path(name), dpi=200)
    plt.close(fig)


kept_at = np.flatnonzero(status == 0)
draw_record(kept_at[np.argmin(np.abs(kept_at - int(len(t) * 0.33)))], "windowing_record.png")
# The last window kept before a missing reading: the next window's forecast would include it.
now = span.get_loc(BRACKET_NOW)
missing = now + 1 + np.flatnonzero(np.isnan(series[now + 1:]))[0]
if status[now] != 0 or missing != now + HORIZON_H + 1:
    raise ValueError("BRACKET_NOW must be a kept window whose forecast ends just before a missing reading")
draw_record(now, "windowing_record_19jun.png", missing)

# ---------- One window as the model receives it ----------
o = int(np.flatnonzero((a["row_station"] == sid) & (hours == (MATRIX_ORIGIN - EPOCH) // pd.Timedelta(hours=1)))[0])
w0 = o - LOOKBACK_H + 1
history = interpolate(pm[w0:o + 1][None, :])[0]
filled = np.isnan(pm[w0:o + 1])
target = pm[o + 1:o + 1 + HORIZON_H]
feat = a["feat"][w0:w0 + WINDOW_H]  # columns: MET_COLS + TIME_COLS
names = {"wind_u": "wind east–west", "wind_v": "wind north–south", "wind_gusts_ms": "wind gusts",
         "temperature_c": "temperature", "humidity_pct": "humidity", "precipitation_mm": "rain",
         "surface_pressure_hpa": "pressure", "hour_sin": "hour (sin)", "hour_cos": "hour (cos)",
         "dow_sin": "weekday (sin)", "dow_cos": "weekday (cos)", "month_sin": "month (sin)", "month_cos": "month (cos)"}
labels = ["PM2.5"] + [names[c] for c in MET_COLS + TIME_COLS]
pm_row = np.concatenate([history, target])
values = np.vstack([pm_row, feat.T])


def scaled(row):
    """ Scale a row to 0-1 across the window for colour; a constant row is drawn mid-tone.
    """
    lo, hi = np.nanmin(row), np.nanmax(row)
    return np.full_like(row, 0.5) if hi - lo < 1e-9 else (row - lo) / (hi - lo)


blues = LinearSegmentedColormap.from_list("b", shared.BLUE_RAMP)
greys = LinearSegmentedColormap.from_list("g", ["#ecebe6", "#4a4945"])
n_rows = len(labels)
fig, ax = plt.subplots(figsize=(7.3, 4.6))
fig.subplots_adjust(left=0.2, right=0.84, top=0.8, bottom=0.12)
ax.grid(False)
x = np.arange(-LOOKBACK_H + 1, HORIZON_H + 1)
for i, row in enumerate(values):
    s = scaled(row)
    for j, v in enumerate(s):
        is_target = i == 0 and x[j] > 0
        ax.add_patch(Rectangle((x[j] - 0.5, i), 1, 0.92, color=(greys if is_target else blues)(v), lw=0))
for j in np.flatnonzero(filled):  # interpolated history hours (slide 9)
    ax.add_patch(Rectangle((x[j] - 0.5, 0), 1, 0.92, fill=False, ec=ORANGE, lw=1.2))
ax.axvline(0.5, color=INK, lw=1.2)
ax.axhline(1 - 0.04, xmin=0, xmax=1, color="white", lw=3)
for g in (1 + len(MET_COLS),):  # gap between the weather rows and the clock rows
    ax.axhline(g - 0.04, color="white", lw=3)
ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
ax.set_ylim(n_rows, -0.1)
ax.set_yticks(np.arange(n_rows) + 0.46)
ax.set_yticklabels(labels, fontsize=8.5)
ax.tick_params(axis="y", length=0)
ax.set_xticks([-71, -47, -23, 0, 24, 48, 72])
ax.set_xticklabels(["−72 h", "−48 h", "−24 h", "now", "+24 h", "+48 h", "+72 h"], fontsize=8.5)
for side in ax.spines.values():
    side.set_visible(False)
# What goes where.
ax.annotate("", (-71.5, -0.9), (0.5, -0.9), arrowprops=dict(arrowstyle="-", color=INK, lw=1.2), annotation_clip=False)
ax.text(-35.5, -1.1, "Input 1: past 72 h × 14 columns", ha="center", va="bottom", fontsize=9.5, weight="semibold")
ax.annotate("", (0.5, -0.9), (72.5, -0.9), arrowprops=dict(arrowstyle="-", color=INK2, lw=1.2), annotation_clip=False)
ax.text(36.5, -1.1, "Input 2: next 72 h × 13 columns", ha="center", va="bottom", fontsize=9.5, weight="semibold")
ax.text(74, 0.46, "Target: the next\n72 PM2.5 values\n(not given to the\nmodel; it is\ngraded on them)",
        fontsize=8.5, color=INK, va="center", ha="left")
ax.text(74, 1 + len(MET_COLS) / 2, "weather from the\nnearest grid point", fontsize=8.5, color=INK2, va="center")
ax.text(74, 1 + len(MET_COLS) + len(TIME_COLS) / 2, "clock, in\nManila time", fontsize=8.5, color=INK2, va="center")
local = MATRIX_ORIGIN.tz_convert(shared.LOCAL_TZ)
fig.text(0.02, 0.965, "Model inputs and targets for one window", fontsize=11.5, weight="semibold", va="top")
fig.patches.append(Rectangle((0.2, 0.025), 0.016, 0.024, fill=False, ec=ORANGE, lw=1.2, transform=fig.transFigure,
                             figure=fig))  # legend: the outline marks interpolated hours
fig.text(0.225, 0.037, "filled by interpolation", fontsize=8.5, color=INK2, va="center")
fig.savefig(shared.out_path("windowing_matrix.png"), dpi=200)
plt.close(fig)
print("saved to", shared.OUT)
