""" Slide figure for the hindcast on the advisor's data (O2/data/advisor/05_ncr_meteorology_pm25_hourly_selected.csv:
    85 NCR stations, hourly met and gap-filled PM2.5, 1 Apr 2024 - 31 Dec 2025), from the groupmate's run:
    O2/artifacts/hindcast/hindcast_results.csv (one row per setup) and hindcast_by_lead.csv (per hour ahead).

      hindcast.png   left: RMSE at each hour ahead, 1-72, for the LSTM, GNN and GBT trained on Metro Manila
                     alone; right: overall RMSE of each model trained on Manila alone and with Bangkok or Los
                     Angeles, using the transfer variant O2 picked by validation RMSE (figures/m5_figs.py).
                     One seed per setup; every number is the equal-weight mean over the 85 stations.

    python -E figures/hindcast_figs.py
"""

import pandas as pd
import matplotlib.pyplot as plt

import shared
from shared import BLUE, INK, INK2, MUTED, ORANGE

ART = shared.REPO / "O2" / "artifacts" / "hindcast"
AQUA = "#1baf7a"  # the GBT's colour on the other M2/M3 slides
MODELS = {"lstm": ("LSTM", BLUE), "gnn": ("GNN", ORANGE), "gbt": ("GBT", AQUA)}
# The transfer variant per model, as chosen on O2's validation data for the M3/M5 slides.
VARIANT = {"lstm": "param_full", "gnn": "mmd_full", "gbt": "weighted"}
SOURCES = [("", "Manila only", "o"), ("bk", "+ Bangkok", "s"), ("la", "+ Los Angeles", "^")]

res = pd.read_csv(ART / "hindcast_results.csv", keep_default_na=False)
lead = pd.read_csv(ART / "hindcast_by_lead.csv", keep_default_na=False)


def pick(frame, model, source):
    variant = "none" if not source else VARIANT[model]
    return frame[(frame.method == model) & (frame.source == source) & (frame.variant == variant)]


stations, windows, seeds = res.n_stations.max(), res.n_windows.max(), res.seed.nunique()
curves = {m: pick(lead, m, "").set_index("lead").rmse for m in MODELS}
best = pd.concat(curves, axis=1).idxmin(axis=1)
print(f"{stations} stations, {windows} windows, {seeds} seed(s); lowest Manila-only model per hour: "
      f"{best.value_counts().to_dict()}")
overall = {}
for m in MODELS:
    for s, label, _ in SOURCES:
        r = pick(res, m, s).iloc[0]
        overall[m, s] = r.rmse
        print(f"{MODELS[m][0]:5s} {label:14s} ({r.variant}): RMSE {r.rmse:.2f}, MBE {r.mbe:+.2f}, R2 {r.r2:+.3f}, "
              f"seen {r.rmse_seen:.2f} / unseen {r.rmse_unseen:.2f}")
    print(f"      1 h {curves[m][1]:.2f}, 24 h {curves[m][24]:.2f}, 72 h {curves[m][72]:.2f}")

shared.use_style(12)
fig = plt.figure(figsize=(12, 5.8))
ax = fig.add_axes([0.07, 0.2, 0.5, 0.56])
dx = fig.add_axes([0.69, 0.2, 0.27, 0.56])

# ---------- Left: error by hour ahead, Manila-only models ----------
NUDGE = {"gnn": 0.4, "lstm": -0.4}  # LSTM and GNN end within 0.1 of each other
for m, (name, colour) in MODELS.items():
    ax.plot(curves[m].index, curves[m].values, color=colour, lw=2, zorder=3)
    ax.text(73, curves[m][72] + NUDGE.get(m, 0), name, va="center", fontsize=11, color=INK2)
ax.set_xlim(1, 72)
ax.set_ylim(0, 14)
ax.set_xticks([1, 12, 24, 36, 48, 60, 72])
ax.set_xlabel("hours ahead")
ax.set_ylabel("RMSE (µg/m³), lower is better")
ax.set_title("Trained on Metro Manila only", fontsize=12, loc="left", color=INK, pad=8)

# ---------- Right: overall RMSE, Manila only vs with each source ----------
rows = list(MODELS)[::-1]
for y, m in enumerate(rows):
    name, colour = MODELS[m]
    vals = [overall[m, s] for s, _, _ in SOURCES]
    dx.plot([min(vals), max(vals)], [y, y], color=shared.GRID, lw=2, zorder=1)
    for (s, _, marker), v in zip(SOURCES, vals):
        hollow = s == ""  # drawn last, larger and unfilled, so it stays visible over a filled marker
        dx.plot(v, y, marker=marker, ms=13 if hollow else 9, color=colour, mfc="none" if hollow else colour,
                mew=1.8, zorder=4 if hollow else 3)
    top = f"{vals[0]:.2f}"
    if abs(vals[1] - vals[0]) < 0.05:  # the two markers sit on top of each other
        top = f"Manila only {vals[0]:.2f}, + Bangkok {vals[1]:.2f}"
    dx.text(vals[0], y + 0.3, top, ha="center", fontsize=9.5, color=MUTED)
    dx.text(min(vals[1:]), y - 0.36, f"{min(vals[1:]):.2f}", ha="center", fontsize=9.5, color=INK2)
dx.set_yticks(range(len(rows)))
dx.set_yticklabels([MODELS[m][0] for m in rows])
dx.tick_params(axis="y", length=0)
dx.set_ylim(-1.5, len(rows) - 0.4)
dx.set_xlim(8.8, 11.8)
dx.set_xlabel("RMSE (µg/m³), all 72 hours")
dx.spines["left"].set_visible(False)
dx.grid(axis="y", visible=False)
dx.set_title("Without and with transfer", fontsize=12, loc="left", color=INK, pad=8)
handles = [plt.Line2D([], [], marker=mk, ls="", ms=8, color=INK2, mfc="none" if not s else INK2, mew=1.6)
           for s, _, mk in SOURCES]
dx.legend(handles, [label for _, label, _ in SOURCES], loc="lower center", ncol=3,
          frameon=False, fontsize=10, handletextpad=0.3, columnspacing=1.0)

fig.text(0.07, 0.93, "Hindcast on NCR dataset", fontsize=16, weight="semibold")
fig.text(0.07, 0.875, "GBT has the lowest error at every hour ahead. Adding Los Angeles data lowers the error of "
         "all three models.", fontsize=12, color=INK2)
fig.text(0.07, 0.035,
         f"{stations} NCR stations, 1 Apr 2024 – 31 Dec 2025, {windows:,} forecast windows; PM2.5 includes the "
         f"advisor's gap-filled values. One training run (seed 1)\nper setup; each number is the mean over stations, "
         "weighted equally. Transfer variants as chosen on O2 validation: LSTM all-layer parameter transfer, GNN MMD,\n"
         "GBT weighted instances. Not comparable with the M2/M3 test scores (different period and stations).",
         fontsize=9, color=MUTED, linespacing=1.45)

path = shared.out_path("hindcast.png")
fig.savefig(path, dpi=200)
print(f"Wrote {path}")
