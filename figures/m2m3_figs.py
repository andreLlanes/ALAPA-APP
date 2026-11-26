""" Figure for the M2 and M3 overview slide (Chapter 5), from the Kaggle runs (O2/artifacts):
      m2m3_forecasts.png  one Metro Manila test window (the evaluation slide's example) with the
                          72-hour forecasts of the three models trained on Manila alone (M2) and
                          with transfer from Bangkok (M3); each line is the average of 30 seeds
    The transfer drawn for each model is the milestone's: LSTM parameter transfer, GNN parameter
    transfer with MMD alignment, GBT MMD-weighted instances. Where that transfer was run twice
    (encoder frozen, or every layer fine-tuned), the run with the lower validation RMSE is drawn.
    python figures/m2m3_figs.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shared
from shared import BLUE, INK, INK2, MUTED, ORANGE, PLANE

ART = shared.REPO / "O2" / "artifacts"
TAG = "direct_gap6_comp70_abl100"  # O2's data settings: longest gap 6 h, completeness 70%
SEEDS = range(1, 31)
MIN_WINDOWS = 100                  # O2/config.py MIN_TEST_WINDOWS: fewer and a station isn't scored
HORIZON = 72
EXAMPLE_STATION = "openaq:5976206"  # the evaluation slide's example window
EXAMPLE_ORIGIN = pd.Timestamp("2026-04-02T10", tz="Asia/Manila")  # last hour the model sees
AQUA = "#1baf7a"
SOURCE, SOURCE_NAME = "bk", "Bangkok"
MODELS = {"LSTM": ("lstm", BLUE, ("param_frozen", "param_full")),
          "GNN": ("gnn", ORANGE, ("mmd_frozen", "mmd_full")),
          "GBT": ("gbt", AQUA, ("weighted",))}
METHOD = {"param": "parameter transfer", "mmd": "parameter transfer with MMD", "weighted": "MMD-weighted instances"}
TUNING = {"frozen": "encoder frozen", "full": "all layers fine-tuned"}


def load(folder, file="predictions.npz"):
    """ Return (station keys, origins, observations, [seeds, n, 72] predictions) of a run's seeds,
        every seed in the first seed's window order.
    """
    preds, ref = [], None
    for s in SEEDS:
        with np.load(folder / f"seed{s}" / file) as f:
            key = pd.MultiIndex.from_arrays([f["location_key"], f["origin"]])
            if ref is None:
                ref, obs, keys, origin = key, f["obs"].astype(float), f["location_key"], f["origin"]
            order = key.get_indexer(ref)
            if (order < 0).any():
                raise SystemExit(f"{folder} seed{s}: different windows from seed 1")
            preds.append(f["pred"].astype(float)[order])
    return keys, origin, obs, np.stack(preds)


def val_rmse(folder):
    """ Mean over seeds of the station-averaged validation RMSE (scored stations only). """
    keys, _, obs, preds = load(folder, "val_predictions.npz")
    names, codes = np.unique(keys, return_inverse=True)
    scored = np.bincount(codes) >= MIN_WINDOWS
    per_seed = []
    for p in preds:
        sq = pd.DataFrame((p - obs) ** 2).groupby(codes)
        total, count = sq.sum().sum(1), sq.count().sum(1)
        per_seed.append(np.sqrt(total / count)[scored].mean())
    return float(np.mean(per_seed))


def window(keys, origin, obs, preds):
    """ The example window: its observations, the mean forecast over seeds, and the 72 measured
        hours before it (the observations of the window that ends where this one starts).
    """
    o = np.datetime64(EXAMPLE_ORIGIN.tz_convert("UTC").tz_localize(None), "h")
    i = np.flatnonzero((keys == EXAMPLE_STATION) & (origin == o))
    b = np.flatnonzero((keys == EXAMPLE_STATION) & (origin == o - np.timedelta64(HORIZON, "h")))
    if len(i) != 1 or len(b) != 1:
        raise SystemExit(f"{EXAMPLE_STATION} at {EXAMPLE_ORIGIN}: not a test window, or no window 72 h before")
    return obs[i[0]], preds[:, i[0]].mean(0), obs[b[0]]


forecasts = {"M2": {}, "M3": {}}
chosen = {}
for name, (model, colour, variants) in MODELS.items():
    run = load(ART / model / "mm" / TAG / "none")
    obs, forecasts["M2"][name], history = window(*run)
    scores = {v: val_rmse(ART / model / f"{SOURCE}-mm" / TAG / v) for v in variants} if len(variants) > 1 else {}
    best = min(scores, key=scores.get) if scores else variants[0]
    chosen[name] = best
    t_obs, forecasts["M3"][name], _ = window(*load(ART / model / f"{SOURCE}-mm" / TAG / best))
    if not np.allclose(t_obs, obs, equal_nan=True):
        raise SystemExit(f"{name}: the transfer run's observations differ")
    print(f"{name}: transfer {best}" + (f" (validation RMSE {', '.join(f'{v} {s:.2f}' for v, s in scores.items())})"
                                        if scores else ""))

shared.use_style(11)
past, ahead = np.arange(-(HORIZON - 1), 1), np.arange(1, HORIZON + 1)
top = np.nanmax([history.max(), obs.max(), *(f.max() for g in forecasts.values() for f in g.values())]) * 1.18
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
fig.subplots_adjust(left=0.065, right=0.985, top=0.78, bottom=0.2, wspace=0.1)
titles = {"M2": "Trained on Manila only (M2)", "M3": f"With transfer from {SOURCE_NAME} (M3)"}
for ax, (key, lines) in zip(axes, forecasts.items()):
    ax.axvspan(0.5, HORIZON + 0.5, color=PLANE, lw=0, zorder=0)
    ax.axvline(0.5, color=INK, lw=0.9, zorder=1)
    ax.plot(past, history, color=INK, lw=1.5, zorder=3)
    ax.plot(ahead, obs, color=INK, lw=1.5, zorder=3)
    for name, f in lines.items():
        ax.plot(ahead, f, color=MODELS[name][1], lw=1.9, zorder=4)
    ax.set_title(titles[key], fontsize=10.5, weight="semibold", loc="left", pad=6)
    ax.text(-69, top * 0.97, "Past 72 hours", fontsize=8.5, color=INK2, va="top")
    ax.text(3, top * 0.97, "Next 72 hours (forecast)", fontsize=8.5, color=INK2, va="top")
    ax.set_xticks([-71, -48, -24, 0, 24, 48, 72])
    ax.set_xticklabels(["−71", "−48", "−24", "now", "+24", "+48", "+72"], fontsize=9)
    ax.set_xlim(-HORIZON + 0.5, HORIZON + 0.5)
    ax.set_xlabel("hours from when the forecast is made", fontsize=9)
    ax.tick_params(labelsize=9)
axes[0].set_ylim(0, top)
axes[0].yaxis.set_major_locator(plt.MaxNLocator(4))
axes[0].set_ylabel("PM2.5 (µg/m³)", fontsize=9.5)

fig.text(0.065, 0.965, "72-hour forecasts on one Manila test window", fontsize=12, weight="semibold", va="top")
fig.legend([plt.Line2D([], [], color=c, lw=2) for c in [INK] + [m[1] for m in MODELS.values()]],
           ["Measured", *MODELS], loc="center left", bbox_to_anchor=(0.058, 0.885), ncol=4, frameon=False,
           fontsize=9.5, handlelength=1.8, columnspacing=2.2)
made = f"{EXAMPLE_ORIGIN:%I %p}".lstrip("0") + f", {EXAMPLE_ORIGIN.day} {EXAMPLE_ORIGIN:%b %Y}"
how = lambda v: (METHOD[v.split("_")[0]] + (f", {TUNING[v.split('_')[1]]}" if "_" in v else ""))
fig.text(0.065, 0.015,
         f"Station {EXAMPLE_STATION.split(':')[-1]}, forecast made {made} (test period). Each line is the average "
         f"of a model's 30 trained copies (seeds 1 to 30).\nTransfer shown: "
         + "; ".join(f"{n} {how(v)}" for n, v in chosen.items()) + ".",
         fontsize=8.5, color=MUTED, va="bottom", linespacing=1.4)
fig.savefig(shared.out_path("m2m3_forecasts.png"), dpi=200)
plt.close(fig)
print(f"saved to {shared.OUT}")
