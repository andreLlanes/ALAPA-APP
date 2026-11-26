""" Figures for the M2 and M3 slides (Chapter 5): the 144-hour window each milestone forecasts, and
    how it trains before forecasting it, from the Kaggle runs (O2/artifacts). Both draw the same
    Metro Manila test window (the evaluation slide's example): the past 72 h the model is given and
    the next 72 h it forecasts.
      m2_windowing.png  M2: every model trains on Metro Manila windows alone, then forecasts
      m3_windowing.png  M3: the LSTM and GNN first train a source model on Bangkok or Los Angeles
                        windows and fine-tune it on Manila windows; the GBT trains once on Manila
                        windows plus Bangkok or LA windows weighted by similarity
    Every city is cut into windows by the same rule, so the window glyphs are the same everywhere.
    The fine-tuning named on M3 is the variant M5 shows, picked on validation RMSE.
    python figures/m2m3_window_figs.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

import shared
from shared import AXIS, BLUE, GRID, INK, INK2, MUTED, ORANGE, PLANE

ART = shared.REPO / "O2" / "artifacts"
TAG = "direct_gap6_comp70_abl100"  # O2's data settings: longest gap 6 h, completeness 70%
SEEDS = range(1, 31)
MIN_WINDOWS = 100                  # O2/config.py MIN_TEST_WINDOWS
HORIZON = 72
EXAMPLE_STATION = "openaq:5976206"  # the evaluation slide's example window (as in m2m3_figs.py)
EXAMPLE_ORIGIN = pd.Timestamp("2026-04-02T10", tz="Asia/Manila")  # last hour the model sees
# The fine-tuning variants that compete per model on M5 (figures/m5_figs.py).
FINE_TUNE = {"LSTM": ("lstm", ("param_frozen", "param_full")), "GNN": ("gnn", ("mmd_frozen", "mmd_full"))}
METHOD = {"param": "parameter transfer", "mmd": "parameter transfer with MMD"}
TUNING = {"frozen": "encoder frozen", "full": "every layer fine-tuned"}
SOURCES = ("bk", "la")
MANILA, SOURCE = BLUE, ORANGE
W = 11.0
STRIP_H, ROW_GAP = 1.9, 0.8  # inches


def val_rmse(folder):
    """ Mean over seeds of the station-averaged validation RMSE (scored stations only), as on M5. """
    per_seed = []
    for s in SEEDS:
        with np.load(folder / f"seed{s}" / "val_predictions.npz") as f:
            keys, obs, pred = f["location_key"], f["obs"].astype(float), f["pred"].astype(float)
        _, codes = np.unique(keys, return_inverse=True)
        scored = np.bincount(codes) >= MIN_WINDOWS
        sq = pd.DataFrame((pred - obs) ** 2).groupby(codes)
        per_seed.append((np.sqrt(sq.sum().sum(1) / sq.count().sum(1)))[scored].mean())
    return float(np.mean(per_seed))


def example_window():
    """ The example window's next 72 measured hours and the 72 before them (the observations of the
        window that ends where this one starts), from a Manila-only run's saved test windows.
    """
    with np.load(ART / "lstm" / "mm" / TAG / "none" / "seed1" / "predictions.npz") as f:
        keys, origin, obs = f["location_key"], f["origin"], f["obs"].astype(float)
    o = np.datetime64(EXAMPLE_ORIGIN.tz_convert("UTC").tz_localize(None), "h")
    i = np.flatnonzero((keys == EXAMPLE_STATION) & (origin == o))
    b = np.flatnonzero((keys == EXAMPLE_STATION) & (origin == o - np.timedelta64(HORIZON, "h")))
    if len(i) != 1 or len(b) != 1:
        raise SystemExit(f"{EXAMPLE_STATION} at {EXAMPLE_ORIGIN}: not a test window, or no window 72 h before")
    return obs[b[0]], obs[i[0]]


def strip(fig, H, top_in, history, ahead_obs):
    """ The window from real readings, with what each half carries bracketed under it. """
    b0 = top_in - STRIP_H
    ax = fig.add_axes([0.75 / W, b0 / H, (W - 1.05) / W, STRIP_H / H])
    past, ahead = np.arange(-(HORIZON - 1), 1), np.arange(1, HORIZON + 1)
    top = np.nanmax([history.max(), ahead_obs.max()]) * 1.2
    ax.axvspan(0.5, HORIZON + 0.5, color=PLANE, lw=0, zorder=0)
    ax.axvline(0.5, color=INK, lw=0.9, zorder=1)
    ax.plot(past, history, color=INK, lw=1.6, zorder=3)
    ax.plot(ahead, ahead_obs, color=MUTED, lw=1.4, zorder=3)
    ax.text(-69, top * 0.95, "Past 72 hours, given to the model", fontsize=9, color=INK2, va="top")
    ax.text(3, top * 0.95, "Next 72 hours, forecast by the model", fontsize=9, color=INK2, va="top")
    peak = int(np.nanargmax(ahead_obs))
    ax.annotate("measured later, used only to grade", xy=(ahead[peak], ahead_obs[peak]),
                xytext=(ahead[peak] - 31, top * 0.78), fontsize=8.5, color=MUTED, va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkA=2, shrinkB=3))
    ax.set_xlim(-HORIZON + 0.5, HORIZON + 0.5)
    ax.set_ylim(0, top)
    ax.set_xticks([-71, -48, -24, 0.5, 24, 48, 72])
    ax.set_xticklabels(["−71 h", "−48 h", "−24 h", "now", "+24 h", "+48 h", "+72 h"], fontsize=8.5)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.tick_params(labelsize=8.5)
    ax.set_ylabel("PM2.5 (µg/m³)", fontsize=9)
    xt = ax.get_xaxis_transform()
    for x0, x1, text in ((-HORIZON + 1.5, -0.5, "Given PM2.5, weather and clock for each hour (72 h × 14)"),
                         (1.5, HORIZON - 0.5, "Given weather and clock (72 h × 13), forecasts the 72 PM2.5 values")):
        ax.plot([x0, x0, x1, x1], [-0.2, -0.24, -0.24, -0.2], transform=xt, color=INK2, lw=0.9, clip_on=False)
        ax.text((x0 + x1) / 2, -0.3, text, transform=xt, fontsize=8.8, color=INK, ha="center", va="top")
    return b0


def glyph(cv, x, y, colour, alpha=1.0, w=0.42, h=0.14):
    """ A small window: the past half filled, the forecast half outlined. """
    cv.add_patch(Rectangle((x, y), w / 2, h, facecolor=colour, edgecolor=colour, lw=0.8, alpha=alpha))
    cv.add_patch(Rectangle((x + w / 2, y), w / 2, h, facecolor="white", edgecolor=colour, lw=0.8, alpha=alpha))


def box(cv, x0, x1, y, title, sub="", kind=None):
    """ One step: a rounded box with its windows drawn on the left, a title and a line under it.
        kind: "manila", "source" or "this" (one window), "both" (a Manila window and fading
        source windows: the GBT's weighting), or None for an empty step.
    """
    h = 0.6
    empty = kind is None
    cv.add_patch(FancyBboxPatch((x0, y - h / 2), x1 - x0, h, boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor="white" if empty else PLANE, edgecolor=AXIS if empty else GRID,
                                lw=0.9, ls=(0, (3, 2)) if empty else "-"))
    if empty:
        cv.text((x0 + x1) / 2, y, title, fontsize=8.8, color=MUTED, ha="center", va="center")
        return
    gx = x0 + 0.15
    if kind == "both":
        glyph(cv, gx, y + 0.03, MANILA, h=0.11)
        for k, a in enumerate((0.95, 0.55, 0.25)):
            glyph(cv, gx + k * 0.15, y - 0.16, SOURCE, a, w=0.12, h=0.08)
    else:
        glyph(cv, gx, y - 0.07, {"manila": MANILA, "source": SOURCE, "this": INK}[kind])
    tx = x0 + 0.74
    cv.text(tx, y + 0.09, title, fontsize=9.2, weight="semibold", va="center")
    cv.text(tx, y - 0.13, sub, fontsize=8.4, color=INK2, va="center")


def arrow(cv, x0, x1, y):
    cv.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=9, color=INK2, lw=0.9))


def draw(name, title, subtitle, columns, rows, footnote, data):
    """ One milestone's figure: the window strip, then one row per training path. columns are
        (header, (x0, x1)) in order; each row is (y index, label, sub, [one step per column]),
        a step being (title, sub, kind) or None for an empty one. The last column is the window.
    """
    H = 5.5 + ROW_GAP * (len(rows) - 1)
    fig = plt.figure(figsize=(W, H))
    cv = fig.add_axes([0, 0, 1, 1])
    cv.set_xlim(0, W)
    cv.set_ylim(0, H)
    cv.axis("off")
    cv.text(0.3, H - 0.22, title, fontsize=12.5, weight="semibold", va="top")
    cv.text(0.3, H - 0.55, subtitle, fontsize=9.5, color=INK2, va="top")
    head_y = strip(fig, H, H - 0.95, *data) - 1.1
    for label, (x0, _) in columns:
        cv.text(x0, head_y, label, fontsize=9, weight="semibold", color=INK2, va="center")
    cv.plot([0.3, W - 0.3], [head_y - 0.17] * 2, color=AXIS, lw=0.8)
    for k, (label, sub, steps) in enumerate(rows):
        y = head_y - 0.6 - ROW_GAP * k
        cv.text(0.3, y + 0.1, label, fontsize=13, weight="semibold", va="center")
        cv.text(0.3, y - 0.15, sub, fontsize=8.6, color=INK2, va="center")
        for c, ((_, (x0, x1)), step) in enumerate(zip(columns, steps)):
            box(cv, x0, x1, y, *(step or ("nothing first",)))
            if c + 1 < len(columns):
                arrow(cv, x1 + 0.04, columns[c + 1][1][0] - 0.04, y)
    cv.text(0.3, 0.12, footnote, fontsize=8.2, color=MUTED, va="bottom", linespacing=1.45)
    fig.savefig(shared.out_path(name), dpi=200)
    plt.close(fig)
    print(f"saved to {shared.OUT / name}")


# The fine-tuning M3 uses per model, picked on validation for each source; M3's footnote names it.
chosen = {}
for model_name, (model, variants) in FINE_TUNE.items():
    chosen[model_name] = {src: min(variants, key=lambda v: val_rmse(ART / model / f"{src}-mm" / TAG / v))
                          for src in SOURCES}
    print(f"{model_name}: " + ", ".join(f"{s}-mm {v}" for s, v in chosen[model_name].items()))
data = example_window()

shared.use_style(10)
made = f"{EXAMPLE_ORIGIN:%I %p}".lstrip("0") + f", {EXAMPLE_ORIGIN.day} {EXAMPLE_ORIGIN:%b %Y}"
where = f"Station {EXAMPLE_STATION.split(':')[-1]}, forecast made {made} (test period)."
weather = "the next 72 hours' weather is the recorded weather"
WINDOW = ("The window above", "72 h in, 72 h out", "this")

draw("m2_windowing.png", "The 72-hour window in M2",
     "Each model learns from Metro Manila's windows only, then forecasts the next 72 hours.",
     [("Trains on", (2.25, 6.3)), ("Then forecasts", (6.65, 10.7))],
     [("LSTM, GNN, GBT", "Manila only", [("Metro Manila windows", "trains the model from scratch", "manila"), WINDOW])],
     f"{where} The {weather[4:]}.", data)

how = lambda v: f"{METHOD[v.split('_')[0]]} ({TUNING[v.split('_')[1]]})"
ft = "; ".join(f"{n} {how(p['bk'])}" + ("" if p["bk"] == p["la"] else f", from LA {how(p['la'])}")
               for n, p in chosen.items())
draw("m3_windowing.png", "The 72-hour window in M3",
     "Each model also learns from Bangkok or Los Angeles windows, cut the same way, before forecasting Manila.",
     [("First learns from", (2.25, 4.6)), ("Then trains on", (4.95, 8.15)), ("Then forecasts", (8.5, 10.7))],
     [("LSTM, GNN", "pretrain, then fine-tune",
       [("Bangkok or LA windows", "trains a source model", "source"),
        ("Metro Manila windows", "fine-tunes that source model", "manila"), WINDOW]),
      ("GBT", "one weighted training",
       [None, ("Manila + Bangkok or LA windows", "trains once, far-off windows count less", "both"), WINDOW])],
     f"{where} Every city is cut into windows by the same rule, and {weather}.\n"
     f"Transfer shown as on M5, {ft}; GBT MMD-weighted instances.", data)
