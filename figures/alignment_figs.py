""" Figures for the alignment slide (Chapter 4, Data Preprocessing), all from the cleaned data, where
    O1/Clean stored each station's nearest weather grid point and the joined weather:
      alignment_map.png       Metro Manila stations and the weather grid points they are joined to
      alignment_timeline.png  one station's PM2.5 and its grid point's weather on the same hours
      alignment_table.png     consecutive rows of the joined table, as stored
    python figures/alignment_figs.py

    The map's land outline comes from the web app's map grid (see shared.manila_basemap); without it
    the map is drawn without an outline.
"""

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import shared  # puts O3 on the path; import before O3 modules
from shared import BLUE, GRID, INK, INK2, MUTED, PLANE
from Common.schema import KEY_COL, PM25_COL, TIME_COL

# The example: the same station and hour as the interpolation slide's example.
EXAMPLE_STATION = "openaq:4049149"
EXAMPLE_HOUR = pd.Timestamp("2025-09-09T10", tz="UTC")
LINK = "#9d9c96"

shared.use_style(11)
rows = shared.read_city("mm")
stations = rows.groupby(KEY_COL)[["latitude", "longitude", "grid_latitude", "grid_longitude"]].first()
cells = stations[["grid_latitude", "grid_longitude"]].drop_duplicates().to_numpy()
ex = stations.loc[EXAMPLE_STATION]
ex_km = 111.32 * np.hypot((ex.longitude - ex.grid_longitude) * shared.MM_COSLAT, ex.latitude - ex.grid_latitude)
ex_id = EXAMPLE_STATION.split(":")[-1]

# ---------- Map ----------
# The map keeps its true shape (about half as wide as tall); the figure is sized around it.
FW, FH, MAP_H = 3.9, 6.4, 4.5
fig = plt.figure(figsize=(FW, FH))
ax = fig.add_axes([0.62 / FW, 1.25 / FH, shared.manila_map_width(MAP_H) / FW, MAP_H / FH])
shared.manila_basemap(ax)
for _, s in stations.iterrows():
    ax.plot([s.longitude, s.grid_longitude], [s.latitude, s.grid_latitude], color=LINK, lw=0.8, zorder=2)
ax.plot(stations.longitude, stations.latitude, "o", color=BLUE, ms=4.5, mec="white", mew=0.7, zorder=4)
ax.plot(cells[:, 1], cells[:, 0], "s", color=INK, ms=8, mec="white", mew=1.1, zorder=5)
ax.plot([ex.longitude, ex.grid_longitude], [ex.latitude, ex.grid_latitude], color=INK, lw=1.8, zorder=6)
ax.plot(ex.longitude, ex.latitude, "o", color=BLUE, ms=7, mec=INK, mew=1.4, zorder=7)
fig.text(0.04, 1 - 0.12 / FH, "PM2.5 stations and nearest weather\ngrid points, Metro Manila",
         fontsize=11.5, weight="semibold", va="top")
legend = [(dict(marker="o", color=BLUE, ms=4.5, mec="white", lw=0), f"OpenAQ PM2.5 station ({len(stations)})"),
          (dict(marker="s", color=INK, ms=7, mec="white", lw=0), f"ECMWF IFS 9 km grid point ({len(cells)})"),
          (dict(color=LINK, lw=1.0), "station → nearest grid point"),
          (dict(color=INK, lw=1.8), f"example: station {ex_id}, {ex_km:.1f} km")]
for i, (style, label) in enumerate(legend):
    y = (0.92 - i * 0.24) / FH
    xs = [0.05, 0.11] if "marker" not in style else [0.08]
    fig.lines.append(plt.Line2D(xs, [y] * len(xs), transform=fig.transFigure, **style))
    fig.text(0.15, y, label, fontsize=9.5, color=INK2, va="center")
fig.savefig(shared.out_path("alignment_map.png"), dpi=200)
plt.close(fig)

# ---------- Timeline: one station's PM2.5 and its grid point's weather, hour by hour ----------
local = EXAMPLE_HOUR.tz_convert(shared.LOCAL_TZ)
start = (local - pd.Timedelta(days=2)).normalize().tz_convert("UTC")  # three local days up to the example
st = rows[rows[KEY_COL] == EXAMPLE_STATION].set_index(TIME_COL)
st = st.reindex(pd.date_range(start, start + pd.Timedelta(days=3), freq="h", inclusive="left"))
t = st.index.tz_convert(shared.LOCAL_TZ).tz_localize(None)
mark = local.tz_localize(None) - pd.Timedelta(minutes=30)  # the line sits on the example hour's point
panels = [("PM2.5 (µg/m³), from the station (OpenAQ)", st[PM25_COL], "line", BLUE),
          ("Temperature (°C), from its grid point (IFS)", st.temperature_c, "line", INK2),
          ("Humidity (%), from its grid point (IFS)", st.humidity_pct, "line", INK2),
          ("Rain (mm per hour), from its grid point (IFS)", st.precipitation_mm, "bar", INK2),
          ("Wind speed (m/s), from its grid point (IFS)", np.hypot(st.wind_u, st.wind_v), "line", INK2)]
fig, axes = plt.subplots(len(panels), 1, figsize=(7.3, 5.6), sharex=True,
                         gridspec_kw=dict(height_ratios=[1.4, 1, 1, 0.8, 1]))
fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.07, hspace=0.32)
for a, (label, values, kind, colour) in zip(axes, panels):
    for day in pd.date_range(t[0].normalize(), t[-1], freq="D"):  # night: 6 PM - 6 AM
        a.axvspan(day - pd.Timedelta(hours=6), day + pd.Timedelta(hours=6), color=PLANE, lw=0, zorder=0)
    if kind == "bar":
        a.bar(t, values, width=1 / 24 * 0.8, color=colour, align="edge", zorder=2)
    else:
        a.plot(t, values, "-o" if colour == BLUE else "-", color=colour, lw=1.5, ms=2.2, zorder=2)
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    lo = 0 if kind == "bar" or label.startswith(("PM2.5", "Wind")) else lo - 0.1 * (hi - lo)
    a.set_ylim(lo, hi + 0.45 * (hi - lo))  # headroom for the label
    a.text(0.005, 0.97, label, transform=a.transAxes, fontsize=8.5, color=INK2, va="top")
    a.axvline(mark, color=INK, lw=1.0, zorder=3)
    a.yaxis.set_major_locator(plt.MaxNLocator(3))
    a.tick_params(labelsize=8.5)
    a.grid(axis="y", color=GRID, lw=0.7)
miss = np.flatnonzero(st[PM25_COL].isna().to_numpy())
if len(miss):
    axes[0].text(t[miss].mean(), 1, f"no reading for {len(miss)} h;\nweather still joined", fontsize=8,
                 color=INK2, ha="center", va="bottom")
axes[-1].set_xlim(t[0], t[-1] + pd.Timedelta(hours=1))
axes[-1].xaxis.set_major_locator(mdates.DayLocator())
axes[-1].xaxis.set_major_formatter(plt.NullFormatter())
axes[-1].xaxis.set_minor_locator(mdates.HourLocator(byhour=12))  # day names centred on noon
axes[-1].xaxis.set_minor_formatter(mdates.DateFormatter("%a %d %b"))
axes[-1].tick_params(axis="x", which="minor", length=0, labelsize=9)
axes[0].text(mark, axes[0].get_ylim()[1], f"{local:%I %p}".lstrip("0") + f", {local:%a} {local.day} {local:%b}",
             fontsize=8.5, color=INK, va="bottom", ha="center")
fig.text(0.07, 0.975, f"Hourly PM2.5 and weather, station {ex_id}", fontsize=11.5, weight="semibold", va="top")
fig.savefig(shared.out_path("alignment_timeline.png"), dpi=200)
plt.close(fig)

# ---------- Table: consecutive joined rows, as stored ----------
num = lambda v, d: f"{v:.{d}f}".replace("-0.00", "0.00")
cols = [("timestamp_utc", lambda ts, r: f"{ts:%Y-%m-%d %H:%M}"),
        ("Manila time", lambda ts, r: f"{ts.tz_convert(shared.LOCAL_TZ):%I %p}".lstrip("0")),
        ("pm25", lambda ts, r: num(r.pm25, 1)), ("temperature_c", lambda ts, r: num(r.temperature_c, 1)),
        ("humidity_pct", lambda ts, r: num(r.humidity_pct, 0)), ("precipitation_mm", lambda ts, r: num(r.precipitation_mm, 1)),
        ("wind_u", lambda ts, r: num(r.wind_u, 2)), ("wind_v", lambda ts, r: num(r.wind_v, 2)),
        ("hour_sin", lambda ts, r: num(r.hour_sin, 2)), ("hour_cos", lambda ts, r: num(r.hour_cos, 2))]
excerpt = st.loc[EXAMPLE_HOUR - pd.Timedelta(hours=3): EXAMPLE_HOUR + pd.Timedelta(hours=2)]
widths = np.array([1.45, 0.95, 0.6, 1.05, 0.95, 1.2, 0.7, 0.7, 0.75, 0.75])
W, row_h = widths.sum() + 0.3, 0.3
H = row_h * (len(excerpt) + 1) + 0.5  # caption + header + rows, with a small bottom margin
fig = plt.figure(figsize=(W, H))
cv = fig.add_axes([0, 0, 1, 1])
cv.set_xlim(0, W)
cv.set_ylim(0, H)
cv.axis("off")
x = 0.15 + np.concatenate([[0], np.cumsum(widths)[:-1]])
y_head = H - 0.5
cv.text(0.15, H - 0.12, f"openaq.merged_clean, station {ex_id} (excerpt). 'Manila time' is added for reading; "
        "gusts, pressure and the day/month columns are not shown.", fontsize=9, color=MUTED, va="top")
for xi, (name, _) in zip(x, cols):
    cv.text(xi, y_head, name, fontsize=8.5, family="Consolas", color=INK, va="center", weight="bold")
for yy in (y_head + row_h / 2, y_head - row_h / 2):
    cv.plot([0.1, W - 0.1], [yy, yy], color=INK, lw=0.8)
for i, (ts, r) in enumerate(excerpt.iterrows()):
    y = y_head - row_h * (i + 1)
    if ts == EXAMPLE_HOUR:
        cv.add_patch(plt.Rectangle((0.1, y - row_h / 2), W - 0.2, row_h, color="#e8f0fb", lw=0))
    for xi, (_, fmt) in zip(x, cols):
        cv.text(xi, y, fmt(ts, r), fontsize=8.5, family="Consolas", color=INK, va="center")
cv.plot([0.1, W - 0.1], [y - row_h / 2] * 2, color=INK, lw=0.8)
fig.savefig(shared.out_path("alignment_table.png"), dpi=200)
plt.close(fig)

old = shared.OUT / "alignment_row.png"  # replaced by the timeline and the table
if old.exists():
    old.unlink()
print("saved to", shared.OUT)
