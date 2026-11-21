""" Figures for the M1 slides (Chapter 5, milestone 1: model-ready datasets). The temporal models were
    trained on Kaggle with O2's code, so the datasets are built with O2's own loader
    (O2/data/load.py prepare_city: completeness 70, longest gap 6, 70/15/15 split):
      m1_datasets.png  stations, hourly rows and windows per split, for the three cities
      m1_manila.png    Metro Manila's windows per station, by split
      m1_lstm_windows.png  one window per city as the LSTM receives it, from the graph's hour
      m1_graphs.png    the GNN's station graph for Bangkok and Los Angeles at a typical training hour
      m1_gbt_rows.png  consecutive rows of the GBT's training table (forecast hour +1), Bangkok and
                       Los Angeles
    python figures/m1_figs.py [--k 4]
"""

import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

import shared
from shared import AXIS, BLUE, GRID, INK, INK2, MUTED, ORANGE

# O2's loader and settings and the repo-root Common (O2's 70/15/15 split), ahead of O3 on the path.
sys.path[:0] = [str(shared.REPO / "O2"), str(shared.REPO)]
import config
import Common.splits as splits
from Common.schema import DECODER_COLS, HORIZON_H, LOOKBACK_H, MET_COLS, TIME_COLS
from Common.splits import TEST, TRAIN, VAL
from data.batch import build_station_graph, station_distances_km
from data.load import GBT_FEATURES, gbt_features, interpolate, loso_keys, prepare_city

if (splits.TRAIN_FRAC, config.DEFAULT_COMPLETENESS) != (0.70, 70):
    raise SystemExit("O2's code did not load (expected the 70/15/15 split and completeness 70)")

parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, default=4, help="neighbours per station (tuned among 4, 8, 12; the chart shows one)")
args = parser.parse_args()

AQUA = "#1baf7a"
PARTS = ((TRAIN, "Train", BLUE), (VAL, "Validation", AQUA), (TEST, "Test", ORANGE))
CITY_TZ = {"mm": "Asia/Manila", "bk": "Asia/Bangkok", "la": "America/Los_Angeles"}
PLACES = {"bk": {"Bangkok": (13.756, 100.502), "Nonthaburi": (13.862, 100.514), "Pathum Thani": (14.020, 100.525),
                 "Samut Prakan": (13.599, 100.597)},
          "la": {"Los Angeles": (34.052, -118.244), "Long Beach": (33.770, -118.194), "Pasadena": (34.148, -118.144),
                 "Santa Monica": (34.019, -118.491), "Anaheim": (33.836, -117.914), "Riverside": (33.953, -117.396),
                 "San Bernardino": (34.108, -117.290), "Santa Clarita": (34.392, -118.543), "Irvine": (33.684, -117.826)}}
shared.use_style(11)
local = lambda h, city: pd.Timestamp(h).tz_localize("UTC").tz_convert(CITY_TZ[city])
clock = lambda ts: f"{ts:%I %p}".lstrip("0") + f", {ts.day} {ts:%b %Y}"


def typical_graph(d, city):
    """ The GNN's graph at the latest training hour of median size: its stations, links and weights.
    """
    m = d["split"] == TRAIN
    per_origin = pd.Series(d["station"][m]).groupby(d["origin"][m]).nunique()
    median = int(per_origin.median())
    at = per_origin.index[(per_origin - median).abs() == (per_origin - median).abs().min()].max()
    here = np.unique(d["station"][m & (d["origin"] == at)])
    lat, lon = d["lat"][here], d["lon"][here]
    edge_index, weight = build_station_graph(station_distances_km(lat, lon), args.k)
    links = {}
    for s, t, w in zip(*edge_index, weight):  # each link is stored both ways
        links[(min(s, t), max(s, t))] = w
    used = np.unique(d["station"])
    # One window of that hour as the LSTM receives it: the graph station with the most training windows.
    windows = np.flatnonzero(m & (d["origin"] == at))
    w = windows[np.argmax(np.bincount(d["station"][m], minlength=len(d["keys"]))[d["station"][windows]])]
    start = d["start"][w]
    raw = d["pm"][start:start + LOOKBACK_H]
    pm_row = np.concatenate([interpolate(raw[None, :])[0], d["pm"][start + LOOKBACK_H:start + LOOKBACK_H + HORIZON_H]])
    window = dict(station=str(d["keys"][d["station"][w]]).split(":")[-1], filled=np.isnan(raw),
                  values=np.vstack([pm_row, d["feat"][start:start + LOOKBACK_H + HORIZON_H].T]))
    return dict(at=local(at, city), lat=lat, lon=lon, links=links, median=median, max=int(per_origin.max()),
                all_lat=d["lat"][used], all_lon=d["lon"][used], absent=np.setdiff1d(used, here), window=window)


def gbt_rows(d, city, n=4):
    """ n consecutive training rows (8 AM onward, local) of the busiest station's +1 h GBT table, from
        the latest morning whose histories have no missing hour.
    """
    m = d["split"] == TRAIN
    s = np.bincount(d["station"][m]).argmax()
    ids = np.flatnonzero(m & (d["station"] == s))
    origin = d["origin"][ids].astype("datetime64[h]").astype(np.int64)
    pos = {o: i for i, o in zip(ids, origin)}
    miss = np.concatenate([[0], np.cumsum(np.isnan(d["pm"]))])
    for o in sorted(origin, reverse=True):
        run = [pos.get(o + j) for j in range(n)]
        if (local(np.datetime64(int(o), "h"), city).hour == 8 and None not in run
                and all(miss[d["start"][w] + LOOKBACK_H] == miss[d["start"][w]] for w in run)):
            break
    else:
        raise SystemExit(f"{city}: no {n} consecutive complete training windows at 8 AM")
    rows = []
    for w in run:
        start = d["start"][w]
        history = interpolate(d["pm"][start:start + LOOKBACK_H][None, :])[0]
        values = dict(zip(GBT_FEATURES, np.concatenate([gbt_features(history[None, :])[0], d["feat"][start + LOOKBACK_H]])))
        rows.append({"made": local(d["origin"][w], city), **values, "target": d["pm"][start + LOOKBACK_H]})
    return str(d["keys"][s]).split(":")[-1], pd.DataFrame(rows)


summary, graphs, tables = [], {}, {}
for city in ("mm", "bk", "la"):
    d = prepare_city(city, config.DEFAULT_LONGEST_GAP, config.DEFAULT_COMPLETENESS)
    keep = (d["completeness"] * 100 >= config.DEFAULT_COMPLETENESS) | np.isin(d["keys"], loso_keys())
    rows = keep[d["row_station"]]
    summary.append(dict(city=shared.CITY_NAMES[city], first=local(d["origin"].min(), city),
                        last=local(d["origin"].max(), city), stations=len(d["keys"]), kept=int(keep.sum()),
                        used=len(np.unique(d["station"])), rows=int(rows.sum()),
                        measured=float(np.isfinite(d["pm"][rows]).mean()),
                        **{name: int((d["split"] == code).sum()) for code, name, _ in PARTS}))
    if city == "mm":
        manila = pd.DataFrame({"station": d["keys"][d["station"]], "part": d["split"]}).value_counts().unstack(fill_value=0)
    else:
        graphs[city], tables[city] = typical_graph(d, city), gbt_rows(d, city)
    del d
summary = pd.DataFrame(summary)
print(summary.to_string(index=False))

# ---------- Model-ready datasets by city ----------
cols = [("City", 1.35, lambda r: r.city, "left"),
        ("Forecasts made", 1.75, lambda r: f"{r['first']:%b %Y} – {r['last']:%b %Y}", "left"),
        ("Stations", 0.85, lambda r: f"{r.stations}", "right"),
        ("Kept", 0.6, lambda r: f"{r.kept}", "right"),
        ("With windows", 1.05, lambda r: f"{r.used}", "right"),
        ("Hourly rows", 1.05, lambda r: f"{r.rows:,}", "right"),
        ("Measured", 0.9, lambda r: f"{r.measured:.0%}", "right"),
        ("Train", 1.0, lambda r: f"{r.Train:,}", "right"),
        ("Validation", 0.95, lambda r: f"{r.Validation:,}", "right"),
        ("Test", 0.9, lambda r: f"{r.Test:,}", "right")]
widths = np.array([c[1] for c in cols])
W, row_h = widths.sum() + 0.4, 0.42
H = 0.75 + 0.45 + row_h * (len(summary) + 1) + 0.55
fig = plt.figure(figsize=(W, H))
cv = fig.add_axes([0, 0, 1, 1])
cv.set_xlim(0, W)
cv.set_ylim(0, H)
cv.axis("off")
left = 0.2 + np.concatenate([[0], np.cumsum(widths)[:-1]])
cv.text(0.2, H - 0.18, "Model-ready datasets by city", fontsize=12, weight="semibold", va="top")
y_group = H - 0.85
for (a, b, label) in ((2, 4, "Stations"), (5, 6, "Hourly rows, kept stations"), (7, 9, "Windows")):
    x0, x1 = left[a] + 0.05, left[b] + widths[b] - 0.05
    cv.plot([x0, x1], [y_group - 0.17] * 2, color=AXIS, lw=0.9)
    cv.text((x0 + x1) / 2, y_group, label, ha="center", va="center", fontsize=9, color=INK2)
y_head = y_group - 0.45
names = {"Stations": "All", "Hourly rows": "Rows"}
for x, w, (name, _, _, align) in zip(left, widths, cols):
    cv.text(x + (w - 0.12 if align == "right" else 0), y_head, names.get(name, name), fontsize=9.5,
            weight="semibold", ha=align, va="center", color=INK)
cv.plot([0.15, W - 0.15], [y_head - row_h / 2] * 2, color=INK, lw=0.8)
for i, r in summary.iterrows():
    y = y_head - row_h * (i + 1)
    for x, w, (_, _, fmt, align) in zip(left, widths, cols):
        cv.text(x + (w - 0.12 if align == "right" else 0), y, fmt(r), fontsize=10, ha=align, va="center",
                color=INK, weight="semibold" if align == "left" and fmt is cols[0][2] else "normal")
    cv.plot([0.15, W - 0.15], [y - row_h / 2] * 2, color=GRID if i < len(summary) - 1 else INK, lw=0.8)
cv.text(0.2, 0.12, "Kept: at least 70% of hours measured. A window is 72 hours of history and the 72 hours it "
        "forecasts; windows are split 70/15/15 in time order.", fontsize=8.5, color=MUTED, va="bottom")
fig.savefig(shared.out_path("m1_datasets.png"), dpi=200)
plt.close(fig)

# ---------- Metro Manila: windows per station ----------
manila["total"] = manila.sum(axis=1)
manila = manila.sort_values("total")
fig, ax = plt.subplots(figsize=(7.3, 4.9))
fig.subplots_adjust(left=0.15, right=0.97, top=0.88, bottom=0.15)
ax.grid(axis="x", color=GRID, lw=0.8)
ax.grid(axis="y", visible=False)
y = np.arange(len(manila))
left_edge = np.zeros(len(manila))
for code, name, colour in PARTS:
    n = manila.get(code, pd.Series(0, index=manila.index)).to_numpy()
    ax.barh(y, n, left=left_edge, height=0.7, color=colour, lw=0, label=name)
    left_edge += n
for yi, (_, r) in zip(y, manila.iterrows()):
    test = int(r.get(TEST, 0))
    ax.text(r.total + 40, yi, f"{int(r.total):,}" + (f"  ({test:,} test)" if test else ""), va="center",
            fontsize=8.5, color=INK2)
ax.set_yticks(y)
ax.set_yticklabels([k.split(":")[-1] for k in manila.index], fontsize=8.5)
ax.set_xlim(0, manila.total.max() * 1.22)
ax.set_xlabel("Windows", fontsize=9.5)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.tick_params(axis="x", labelsize=9)
ax.set_ylabel("OpenAQ station", fontsize=9.5)
ax.legend(loc="lower right", frameon=False, fontsize=9, handlelength=1.2)
mm = summary.iloc[0]
fig.text(0.15, 0.965, "Windows per station by data split, Metro Manila", fontsize=11.5, weight="semibold", va="top")
fig.text(0.15, 0.015, f"{mm.kept} of {mm.stations} stations pass the 70% completeness cut; these {mm.used} have at "
         "least one window.", fontsize=8.5, color=MUTED, va="bottom")
fig.savefig(shared.out_path("m1_manila.png"), dpi=200)
plt.close(fig)

# ---------- The GNN's graphs, Bangkok and Los Angeles ----------
MAP_H = 4.4
extent, widths_in = {}, []
for city, g in graphs.items():
    pad_lat = (g["all_lat"].max() - g["all_lat"].min()) * 0.06
    pad_lon = (g["all_lon"].max() - g["all_lon"].min()) * 0.06
    lat0, lat1 = g["all_lat"].min() - pad_lat, g["all_lat"].max() + pad_lat
    lon0, lon1 = g["all_lon"].min() - pad_lon, g["all_lon"].max() + pad_lon
    coslat = np.cos(np.radians((lat0 + lat1) / 2))
    extent[city] = (lon0, lon1, lat0, lat1, coslat)
    widths_in.append(MAP_H * (lon1 - lon0) * coslat / (lat1 - lat0))
FW, FH = 0.75 + sum(widths_in) + 0.7 + 0.3, 6.6
fig = plt.figure(figsize=(FW, FH))
x = 0.75
halo = [pe.withStroke(linewidth=2.5, foreground="white")]
for (city, g), w in zip(graphs.items(), widths_in):
    lon0, lon1, lat0, lat1, coslat = extent[city]
    ax = fig.add_axes([x / FW, 1.35 / FH, w / FW, MAP_H / FH])
    ax.set_facecolor("#f6f5f1")
    ax.grid(False)
    for side in ax.spines.values():
        side.set_visible(True)
        side.set_color(AXIS)
    for (s, t), wt in g["links"].items():
        ax.plot(g["lon"][[s, t]], g["lat"][[s, t]], color=INK2, lw=0.4 + 1.8 * wt, solid_capstyle="round", zorder=2)
    ax.plot(g["all_lon"], g["all_lat"], "o", mfc="white", mec="#b9b8b1", mew=0.9, ms=3.2, zorder=3)
    ax.plot(g["lon"], g["lat"], "o", color=BLUE, ms=4.5, mec="white", mew=0.6, zorder=4)
    for name, (plat, plon) in PLACES[city].items():
        if lon0 < plon < lon1 and lat0 < plat < lat1:
            ax.text(plon, plat, name, fontsize=8, color="#7d7c77", ha="center", va="center", path_effects=halo, zorder=5)
    km = 10 if city == "bk" else 20
    deg = km / (111.32 * coslat)
    bx, by = lon1 - (lon1 - lon0) * 0.04, lat0 + (lat1 - lat0) * 0.04
    ax.plot([bx - deg, bx], [by, by], color=INK2, lw=2, solid_capstyle="butt", zorder=6)
    ax.text(bx - deg / 2, by + (lat1 - lat0) * 0.012, f"{km} km", ha="center", va="bottom", fontsize=8.5, color=INK2, zorder=6)
    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{abs(v):.1f}°{'E' if v >= 0 else 'W'}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}°N"))
    ax.tick_params(labelsize=8.5)
    ax.set_title(shared.CITY_NAMES[city], fontsize=10.5, loc="left", pad=5, color=INK, weight="semibold")
    fig.text(x / FW, 1.0 / FH, f"{clock(g['at'])}: {len(g['lat'])} stations\n(training median {g['median']}, "
             f"max {g['max']})", fontsize=8.5, color=INK2, va="top")
    x += w + 0.7
fig.text(0.75 / FW, 1 - 0.14 / FH, "GNN station graphs, Bangkok and Los Angeles", fontsize=11.5, weight="semibold", va="top")
legend = [(dict(marker="o", color=BLUE, ms=4.5, mec="white", lw=0), "station with a window at this hour"),
          (dict(marker="o", mfc="white", mec="#b9b8b1", mew=0.9, ms=3.2, lw=0), "other station (no window at this hour)"),
          (dict(color=INK2, lw=1.4), f"link to one of its {args.k} nearest (thicker = closer)")]
for i, (style, label) in enumerate(legend):
    yy = (0.42 - i * 0.16) / FH
    xs = [0.75 / FW + 0.004, 0.75 / FW + 0.03] if "marker" not in style else [0.75 / FW + 0.017]
    fig.lines.append(plt.Line2D(xs, [yy] * len(xs), transform=fig.transFigure, **style))
    fig.text(0.75 / FW + 0.045, yy, label, fontsize=8.5, color=INK2, va="center")
fig.text(1 - 0.3 / FW, 0.1 / FH, f"k = {args.k} shown; k was tuned among 4, 8 and 12.", fontsize=8.5, color=MUTED,
         ha="right", va="bottom")
fig.savefig(shared.out_path("m1_graphs.png"), dpi=200)
plt.close(fig)

# ---------- The LSTM's input: one window per city, from the hour of the graph ----------
NAMES = {"wind_u": "wind east–west", "wind_v": "wind north–south", "wind_gusts_ms": "wind gusts",
         "temperature_c": "temperature", "humidity_pct": "humidity", "precipitation_mm": "rain",
         "surface_pressure_hpa": "pressure", "hour_sin": "hour (sin)", "hour_cos": "hour (cos)",
         "dow_sin": "weekday (sin)", "dow_cos": "weekday (cos)", "month_sin": "month (sin)", "month_cos": "month (cos)"}
labels = ["PM2.5"] + [NAMES[c] for c in MET_COLS + TIME_COLS]
blues = LinearSegmentedColormap.from_list("b", shared.BLUE_RAMP)
greys = LinearSegmentedColormap.from_list("g", ["#ecebe6", "#4a4945"])
hours = np.arange(-LOOKBACK_H + 1, HORIZON_H + 1)
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), sharey=True)
fig.subplots_adjust(left=0.13, right=0.97, top=0.76, bottom=0.17, wspace=0.14)
for ax, (city, g) in zip(axes, graphs.items()):
    ax.grid(False)
    for i, row in enumerate(g["window"]["values"]):  # each row coloured by its own range, as on the windowing slide
        lo, hi = np.nanmin(row), np.nanmax(row)
        dry = labels[i] == "rain" and hi < 1e-9  # a constant row is drawn mid-tone, except no rain at all (light)
        scaled = np.full_like(row, 0.0 if dry else 0.5) if hi - lo < 1e-9 else (row - lo) / (hi - lo)
        if dry:
            ax.text(-70, i + 0.45, "no rain", fontsize=8, color=INK2, va="center", zorder=3)
        for j, v in enumerate(scaled):
            ax.add_patch(Rectangle((hours[j] - 0.5, i), 1, 0.9, lw=0,
                                   color=(greys if i == 0 and hours[j] > 0 else blues)(v)))
    for j in np.flatnonzero(g["window"]["filled"]):
        ax.add_patch(Rectangle((hours[j] - 0.5, 0), 1, 0.9, fill=False, ec=ORANGE, lw=1.1))
    ax.axvline(0.5, color=INK, lw=1.2)
    ax.set_xlim(hours[0] - 0.5, hours[-1] + 0.5)
    ax.set_ylim(len(labels), -0.1)
    ax.set_yticks(np.arange(len(labels)) + 0.45)
    ax.set_yticklabels(labels, fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([-71, -23, 0, 24, 72])
    ax.set_xticklabels(["−72 h", "−24 h", "now", "+24 h", "+72 h"], fontsize=8)
    for side in ax.spines.values():
        side.set_visible(False)
    for (x0, x1, text, colour) in ((-71.5, 0.5, "past 72 h × 14", INK), (0.5, 72.5, "next 72 h × 13", INK2)):
        ax.annotate("", (x0 + 1, -0.75), (x1 - 1, -0.75), arrowprops=dict(arrowstyle="-", color=colour, lw=1.1),
                    annotation_clip=False)
        ax.text((x0 + x1) / 2, -0.95, text, ha="center", va="bottom", fontsize=8.5, weight="semibold")
    ax.text(-71.5, -2.35, f"{shared.CITY_NAMES[city]}, station {g['window']['station']}", fontsize=10,
            weight="semibold", va="bottom")
    ax.text(-71.5, -1.75, f"forecast made {clock(g['at'])}", fontsize=8.5, color=INK2, va="bottom")
fig.text(0.13, 0.965, "LSTM input windows, Bangkok and Los Angeles", fontsize=11.5, weight="semibold", va="top")
keys = [(dict(color="#8e8d88", lw=0), "target: the next 72 PM2.5 values (graded, not given)")]
if any(g["window"]["filled"].any() for g in graphs.values()):
    keys.insert(0, (dict(fill=False, ec=ORANGE, lw=1.1), "filled by interpolation"))
for i, (patch, text) in enumerate(keys):
    x0 = 0.13 + i * 0.2
    fig.patches.append(Rectangle((x0, 0.035), 0.012, 0.03, transform=fig.transFigure, figure=fig, **patch))
    fig.text(x0 + 0.018, 0.05, text, fontsize=8.5, color=INK2, va="center")
fig.savefig(shared.out_path("m1_lstm_windows.png"), dpi=200)
plt.close(fig)

# ---------- Rows of the GBT's training table ----------
num = lambda d: (lambda v: f"{v:.{d}f}")
gcols = [("Forecast made", None, 1.35, lambda v: f"{v:%d %b %Y}, " + f"{v:%I %p}".lstrip("0")),
         ("now", "pm25_lag0", 0.62, num(1)), ("1 h ago", "pm25_lag1", 0.72, num(1)),
         ("24 h ago", "pm25_lag24", 0.78, num(1)), ("71 h ago", "pm25_lag71", 0.78, num(1)),
         ("mean", "pm25_mean24", 0.66, num(1)), ("max", "pm25_max24", 0.62, num(1)), ("std", "pm25_std24", 0.62, num(1)),
         ("temp. (°C)", "temperature_c_target", 0.92, num(1)), ("humidity (%)", "humidity_pct_target", 1.05, num(0)),
         ("rain (mm)", "precipitation_mm_target", 0.85, num(1)), ("…", None, 0.35, lambda v: "…"),
         ("PM2.5 +1 h", "target", 1.0, num(1))]
gw = np.array([c[2] for c in gcols])
W, row_h = gw.sum() + 0.4, 0.3
block = 0.45 + 0.42 + row_h * 5
H = 0.6 + block * len(tables) + 0.55
fig = plt.figure(figsize=(W, H))
cv = fig.add_axes([0, 0, 1, 1])
cv.set_xlim(0, W)
cv.set_ylim(0, H)
cv.axis("off")
gx = 0.2 + np.concatenate([[0], np.cumsum(gw)[:-1]])
cv.text(0.2, H - 0.16, "GBT training rows for forecast hour +1", fontsize=12, weight="semibold", va="top")
top = H - 0.75
for city, (station, rows) in tables.items():
    cv.text(0.2, top, f"{shared.CITY_NAMES[city]}, station {station}", fontsize=10, weight="semibold", va="center")
    y_group = top - 0.38
    for a, b, label in ((1, 4, "PM2.5 lags (µg/m³)"), (5, 7, "last 24 h (µg/m³)"), (8, 11, "weather at the forecast hour"),
                        (12, 12, "target")):
        x0, x1 = gx[a] + 0.04, gx[b] + gw[b] - 0.04
        cv.plot([x0, x1], [y_group - 0.12] * 2, color=AXIS, lw=0.9)
        cv.text((x0 + x1) / 2, y_group, label, ha="center", va="center", fontsize=8.5, color=INK2)
    y_head = y_group - 0.34
    for xx, w, (name, *_rest) in zip(gx, gw, gcols):
        cv.text(xx + (0 if name == "Forecast made" else w - 0.1), y_head, name, fontsize=8.5, weight="semibold",
                ha="left" if name == "Forecast made" else "right", va="center")
    cv.plot([0.15, W - 0.15], [y_head - row_h / 2] * 2, color=INK, lw=0.8)
    for i, r in rows.iterrows():
        yy = y_head - row_h * (i + 1)
        for xx, w, (name, key, _, fmt) in zip(gx, gw, gcols):
            v = r["made"] if name == "Forecast made" else (r[key] if key else None)
            cv.text(xx + (0 if name == "Forecast made" else w - 0.1), yy, fmt(v), fontsize=8.5,
                    ha="left" if name == "Forecast made" else "right", va="center", color=INK)
    cv.plot([0.15, W - 0.15], [yy - row_h / 2] * 2, color=INK, lw=0.8)
    top -= block
n_feat = len(GBT_FEATURES)
cv.text(0.2, 0.12, f"Each forecast hour has its own model and table (72 in all). A row has {n_feat} inputs: 9 PM2.5 lags, "
        f"12 rolling statistics (mean, max, std over 3, 6, 12, 24 h)\nand the {len(DECODER_COLS)} weather and time values "
        "at the forecast hour; 10 are shown. Local time.", fontsize=8.5, color=MUTED, va="bottom")
fig.savefig(shared.out_path("m1_gbt_rows.png"), dpi=200)
plt.close(fig)
for city, g in graphs.items():
    print(f"{city} graph at {clock(g['at'])}: {len(g['lat'])} stations, {len(g['links'])} links "
          f"(training median {g['median']}, max {g['max']})")
print("saved to", shared.OUT)
