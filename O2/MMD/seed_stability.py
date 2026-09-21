import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats

COLUMN = "mmd2_pair"


def load(path, column):
    df = pd.read_csv(path)
    return df[df["status"] == "ok"][["location_key", "city", column, column + "_sd"]]


def compare(path_a, path_b, column):
    a = load(path_a, column).rename(columns={column: "a", column + "_sd": "a_sd"})
    b = load(path_b, column).rename(columns={column: "b", column + "_sd": "b_sd"})
    m = a.merge(b.drop(columns="city"), on="location_key")
    if m.empty:
        raise SystemExit("the two tables share no stations with status ok")
    m["diff"] = m["b"] - m["a"]
    m["abs_diff"] = m["diff"].abs()

    print(f"A: {path_a}\nB: {path_b}\ncolumn: {column}")
    print(f"stations compared: {len(m)} "
          f"(A had {len(a)}, B had {len(b)})\n")

    print(f"{'city':<14}{'n':>5}{'median |B-A|':>14}{'p95 |B-A|':>11}"
          f"{'IQR(A)':>10}{'|B-A|/IQR':>11}{'within-cache sd':>17}")
    print("-" * 82)
    for city in list(dict.fromkeys(m["city"])) + ["ALL"]:
        c = m if city == "ALL" else m[m["city"] == city]
        iqr = np.subtract(*np.percentile(c["a"], [75, 25]))
        med = c["abs_diff"].median()
        print(f"{city:<14}{len(c):>5}{med:>14.4f}{c['abs_diff'].quantile(.95):>11.4f}"
              f"{iqr:>10.4f}{med / iqr:>11.1%}{c['a_sd'].median():>17.4f}")

    rho = stats.spearmanr(m["a"], m["b"]).statistic
    print(f"\nRank agreement between samples: Spearman rho = {rho:.4f}")

    # The claim that has to survive resampling.
    cities = [c for c in dict.fromkeys(m["city"])]
    if len(cities) == 2:
        first, second = cities
        for label, col in (("sample A", "a"), ("sample B", "b")):
            x = m.loc[m["city"] == first, col].to_numpy()
            y = m.loc[m["city"] == second, col].to_numpy()
            closer = ((x[:, None] < y[None, :]).mean()
                      + 0.5 * (x[:, None] == y[None, :]).mean())
            p = stats.mannwhitneyu(x, y, alternative="less").pvalue
            print(f"  {label}: median {first} {np.median(x):.4f} vs {second} "
                  f"{np.median(y):.4f}; P({first} closer) = {closer:.3f}, p = {p:.1e}")

    # Weight impact: what the difference is worth once Eq. 4.6 is applied.
    print("\nEffect on the Eq. 4.6 weights")
    for tau in (0.05, 0.10, 0.25):
        wa = np.exp(-m["a"].clip(lower=0) / tau)
        wb = np.exp(-m["b"].clip(lower=0) / tau)
        d = (wb - wa).abs()
        print(f"  tau = {tau:<5} median |dw| = {d.median():.4f}, "
              f"p95 = {d.quantile(.95):.4f}, max = {d.max():.4f}")

    print("\nRead this as: the ranking and the city separation are what the study "
          "reports,\nso stability of those matters more than the size of |B-A|.")
    return m


def main():
    p = argparse.ArgumentParser(description=(
        __doc__ or "Compare two station distance tables built with different seeds."
    ).split("\n\n")[0])
    p.add_argument("table_a")
    p.add_argument("table_b")
    p.add_argument("--column", default=COLUMN, choices=("mmd2_pair", "mmd2_fixed"))
    p.add_argument("--out", default="", help="optional CSV of the per-station comparison")
    args = p.parse_args()
    m = compare(args.table_a, args.table_b, args.column)
    if args.out:
        m.to_csv(args.out, index=False)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
