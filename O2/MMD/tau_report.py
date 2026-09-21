import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_O2_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_O2_ROOT, "common"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cities import SOURCE_CITIES  # noqa: E402
from transfer import TAU_GRID, DISTANCE_COLUMN  # noqa: E402
import mmd  # noqa: E402

DEFAULT_TABLE = os.path.join(_HERE, "Outputs", "station_mmd.csv")


def report(table, target_rows, column, grid):
    df = pd.read_csv(table)
    ok = df[df["status"] == "ok"].copy()
    print(f"Table: {table}")
    print(f"Distance column: {column}")
    print(f"Metro Manila training rows in the pool: {target_rows:,}\n")

    print(f"{'tau':>6} │ {'city':<13}{'median w':>10}{'min w':>8}{'max w':>8}"
          f"{'eff. rows':>12}{'x Manila':>10}")
    print("─" * 70)
    for tau in grid:
        ok["w"] = mmd.sample_weights(ok[column].to_numpy(), tau)
        ok["eff"] = ok["w"] * ok["n_train_hours"]
        first = True
        for city in SOURCE_CITIES:
            c = ok[ok["city"] == city]
            if c.empty:
                continue
            eff = c["eff"].sum()
            print(f"{tau if first else '':>6} │ {city:<13}{c['w'].median():>10.3f}"
                  f"{c['w'].min():>8.3f}{c['w'].max():>8.3f}{eff:>12,.0f}"
                  f"{eff / target_rows:>10.2f}")
            first = False
        # How lopsided the source pool is: if Los Angeles contributes as much
        # weighted data as Bangkok, the "dissimilar source" control is weak.
        effs = {c: ok.loc[ok["city"] == c, "eff"].sum() for c in SOURCE_CITIES}
        if all(effs.values()):
            ratio = effs[SOURCE_CITIES[1]] / effs[SOURCE_CITIES[0]]
            print(f"{'':>6} │ {'':13}{'':26}LA / BK weighted data = {ratio:.2f}")
        print("─" * 70)

    print("\nReading this table")
    print("  A tau is too large when Los Angeles still contributes a substantial")
    print("  share: the dissimilar-source control then tests weak downweighting")
    print("  rather than transfer from a dissimilar city.")
    print("  A tau is too small when Bangkok's own effective rows fall near or")
    print("  below Metro Manila's: there is then little source data left to")
    print("  transfer from, and the configuration approaches the cold start.")
    print("\nSelect on validation RMSE across this grid, then fix TAU in")
    print("O2/common/transfer.py with the value, the date, and the run that chose it.")


def main():
    p = argparse.ArgumentParser(description=(
        __doc__ or "What each candidate tau does to the GBT training pool (Eq. 4.6)."
    ).split("\n\n")[0])
    p.add_argument("--table", default=DEFAULT_TABLE)
    p.add_argument("--column", default=DISTANCE_COLUMN,
                   choices=("mmd2_pair", "mmd2_fixed"))
    p.add_argument("--target-rows", type=int, default=110842,
                   help="Metro Manila training hours (station_mmd.meta.json: reference_hours)")
    p.add_argument("--grid", default=",".join(str(t) for t in TAU_GRID))
    args = p.parse_args()
    grid = [float(t) for t in args.grid.split(",") if t.strip()]
    report(args.table, args.target_rows, args.column, np.array(grid))


if __name__ == "__main__":
    main()
