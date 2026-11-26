""" Slide figure for M2 (Manila-only models), from the Kaggle runs (O2/artifacts/lead_comparison.csv):

      m2_lead.png    forecast error (RMSE) at each hour ahead, 1-72, for the LSTM, GNN and GBT trained
                     on Metro Manila alone, with the persistence and climatology baselines. Each model
                     line is the mean of 30 seeds; every line is the equal-weight mean over the scored
                     test stations (O2/utils/artifacts.py lead_curve).

    python -E figures/m2_lead_figs.py
"""

import pandas as pd
import matplotlib.pyplot as plt

import shared
from shared import BLUE, INK, INK2, MUTED, ORANGE

ART = shared.REPO / "O2" / "artifacts"
SETTINGS = "gap6_comp70_abl100"  # O2's data settings: longest gap 6 h, completeness 70%
AQUA = "#1baf7a"                 # the GBT's colour on the other M2/M3 slides (m2m3_figs.py)
MODELS = {"LSTM": BLUE, "GNN": ORANGE, "GBT": AQUA}
BASELINES = {"persistence": (INK2, (0, (5, 3))), "climatology": (MUTED, (0, (1.5, 2.5)))}

d = pd.read_csv(ART / "lead_comparison.csv")
d = d[(d.city == "mm") & (d.settings == SETTINGS) & d.source.isna()]  # Manila only, no transfer
rmse = d.pivot_table(index="lead", columns="method", values="rmse")
seeds = d[d.method.isin(MODELS)].groupby("method").seeds.first()
stations = pd.read_csv(ART / "gbt" / "mm" / f"direct_{SETTINGS}" / "none" / "seed1" / "lead_curve.csv").n_stations.iloc[0]

for m in MODELS:
    lose_p = [h for h in rmse.index if rmse.at[h, m] >= rmse.at[h, "persistence"]]
    lose_c = [h for h in rmse.index if rmse.at[h, m] >= rmse.at[h, "climatology"]]
    print(f"{m}: 1 h {rmse.at[1, m]:.2f}, 24 h {rmse.at[24, m]:.2f}, 72 h {rmse.at[72, m]:.2f}; "
          f"not better than persistence at {lose_p or 'no hour'}, than climatology at {lose_c or 'no hour'}")
print(f"persistence 1 h {rmse.at[1, 'persistence']:.2f}; seeds {dict(seeds)}; {stations} stations")

shared.use_style(12)
fig, ax = plt.subplots(figsize=(11, 5.6))
fig.subplots_adjust(left=0.08, right=0.86, top=0.80, bottom=0.21)

for name, (colour, dash) in BASELINES.items():
    ax.plot(rmse.index, rmse[name], color=colour, lw=1.6, ls=dash, zorder=2)
for name, colour in MODELS.items():
    ax.plot(rmse.index, rmse[name], color=colour, lw=2, zorder=3)

# Direct labels for the baselines only: the three models end within 0.4 of each other, too close to
# label apart, so the legend names them.
for n in BASELINES:
    ax.text(73.2, rmse.at[72, n], n.capitalize(), va="center", fontsize=11, color=INK2)

# Hour 1, the one hour persistence beats two of the models.
p1 = rmse.at[1, "persistence"]
ax.annotate(f"Hour 1: persistence {p1:.1f}\nLSTM {rmse.at[1, 'LSTM']:.1f}, GNN {rmse.at[1, 'GNN']:.1f}, "
            f"GBT {rmse.at[1, 'GBT']:.1f}",
            xy=(1, p1), xytext=(5, 4.2), fontsize=10.5, color=INK2, va="center",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9, shrinkA=2, shrinkB=3))

ax.set_xlim(1, 72)
ax.set_ylim(0, 24)
ax.set_xticks([1, 12, 24, 36, 48, 60, 72])
ax.set_xlabel("hours ahead")
ax.set_ylabel("RMSE (µg/m³), lower is better")

fig.text(0.08, 0.93, "M2: forecast error by hour ahead, models trained on Metro Manila only",
         fontsize=16, weight="semibold")
fig.text(0.08, 0.875, "All three beat climatology at every hour. GBT beats persistence at every hour; "
         "LSTM and GNN from hour 2.", fontsize=12, color=INK2)
handles = [plt.Line2D([], [], color=c, lw=2) for c in MODELS.values()] + \
          [plt.Line2D([], [], color=c, lw=1.6, ls=s) for c, s in BASELINES.values()]
fig.legend(handles, list(MODELS) + ["Persistence (last reading repeated)", "Climatology (usual level for the hour)"],
           loc="upper left", bbox_to_anchor=(0.075, 0.845), ncol=5, frameon=False, fontsize=10.5,
           handlelength=2.2, columnspacing=1.4)
fig.text(0.08, 0.06, f"Metro Manila test period, forecasts made 27 Mar – 27 Jun 2026, {stations} scored stations "
         f"(each weighted equally). Model lines are the mean of {seeds.max()} seeds.",
         fontsize=10, color=MUTED)

path = shared.out_path("m2_lead.png")
fig.savefig(path, dpi=200)
print(f"Wrote {path}")
