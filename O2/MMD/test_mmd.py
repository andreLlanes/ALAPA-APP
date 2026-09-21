import sys

import numpy as np

import mmd

try:
    import torch

    import mmd_torch
except ImportError:  # torch is only needed by the GNN
    torch = None

RNG = np.random.default_rng(2026)
FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def null_case():
    print("\nNull case (same distribution)")
    vals = [mmd.mmd2(RNG.normal(size=(300, 8)), RNG.normal(size=(300, 8)))
            for _ in range(20)]
    v = np.array(vals)
    check("|MMD^2| stays near zero", np.abs(v).max() < 0.01,
          f"max |value| = {np.abs(v).max():.5f}")
    check("unbiased estimator goes negative sometimes", (v < 0).any(),
          f"{(v < 0).sum()}/20 draws negative")


def shift_cases():
    print("\nShift detection")
    base = RNG.normal(size=(300, 8))
    null = mmd.mmd2(base, RNG.normal(size=(300, 8)))
    values = []
    for shift in (0.25, 0.5, 1.0, 2.0):
        y = RNG.normal(size=(300, 8)) + shift
        values.append(mmd.mmd2(base, y))
    check("mean shift is detected", values[0] > abs(null) * 5,
          f"null = {null:+.5f}, shift 0.25 = {values[0]:.5f}")
    check("distance grows with the shift", all(np.diff(values) > 0),
          " < ".join(f"{v:.4f}" for v in values))

    spread = mmd.mmd2(base, RNG.normal(size=(300, 8)) * 2.0)
    check("variance-only shift is detected", spread > abs(null) * 5,
          f"scale x2 = {spread:.5f}")


def ranking_stability():
    print("\nRanking stability across seeds")
    target = RNG.normal(size=(600, 8))
    stations = {f"s{i}": RNG.normal(size=(600, 8)) + 0.15 * i for i in range(6)}
    orders = []
    for seed in (1, 2, 3):
        r = np.random.default_rng(seed)
        d = {k: mmd.mmd2(v[r.choice(len(v), 300, replace=False)],
                         target[r.choice(len(target), 300, replace=False)])
             for k, v in stations.items()}
        orders.append([k for k, _ in sorted(d.items(), key=lambda kv: kv[1])])
    check("same ordering under three seeds", all(o == orders[0] for o in orders),
          " | ".join(",".join(o) for o in orders))


def weight_behaviour():
    print("\nSample weights (Eq. 4.6)")
    d = np.array([-0.01, 0.0, 0.03, 0.25, 1.0])
    w = mmd.sample_weights(d, tau=0.25)
    check("weights are ordered by distance", np.all(np.diff(w) <= 0),
          " > ".join(f"{x:.3f}" for x in w))
    check("a negative distance clips to weight 1", w[0] == 1.0)
    check("weights never exceed a Metro Manila sample", w.max() <= 1.0)


def torch_matches_numpy():
    print("\nTorch kernel vs NumPy kernel")
    if torch is None:
        check("torch available", False, "torch is not installed; GNN alignment untested")
        return
    x = RNG.normal(size=(120, 6))
    y = RNG.normal(size=(150, 6)) + 0.4
    want = mmd.mmd2(x, y)
    got = mmd_torch.mmd2(torch.tensor(x, dtype=torch.float64),
                         torch.tensor(y, dtype=torch.float64)).item()
    check("unbiased values agree", abs(want - got) < 1e-9,
          f"numpy {want:.10f} vs torch {got:.10f}")

    fixed = 3.5
    want_f = mmd.mmd2(x, y, sigma_med=fixed)
    got_f = mmd_torch.mmd2(torch.tensor(x, dtype=torch.float64),
                           torch.tensor(y, dtype=torch.float64),
                           sigma_med=torch.tensor(fixed, dtype=torch.float64)).item()
    check("fixed-bandwidth values agree", abs(want_f - got_f) < 1e-9,
          f"numpy {want_f:.10f} vs torch {got_f:.10f}")

    biased = mmd_torch.mmd2(torch.tensor(x), torch.tensor(y), unbiased=False).item()
    check("biased form stays non-negative", biased >= 0, f"{biased:.6f}")


def torch_gradients():
    print("\nTorch gradients")
    if torch is None:
        check("torch available", False, "torch is not installed")
        return
    torch.manual_seed(2026)
    target = torch.randn(128, 6)
    source = (torch.randn(128, 6) + 1.5).requires_grad_(True)
    # Adam, not SGD: the term's gradient norm is ~1e-2 on a batch this size, so a
    # plain SGD step of a sensible size barely moves it.
    opt = torch.optim.Adam([source], lr=0.05)

    start = mmd_torch.mmd2(source, target).item()
    for _ in range(200):
        opt.zero_grad()
        loss = mmd_torch.mmd2(source, target)
        loss.backward()
        opt.step()
    end = mmd_torch.mmd2(source, target).item()

    check("gradient flows to the source batch", source.grad is not None
          and torch.isfinite(source.grad).all().item())
    check("optimizing the term reduces the distance", end < start * 0.1,
          f"{start:.5f} -> {end:.5f}")

    align = mmd_torch.MMDAlignment(lambda_=0.5, warmup_steps=10).train()
    first, _ = align(torch.randn(64, 6), torch.randn(64, 6))
    for _ in range(20):
        align(torch.randn(64, 6), torch.randn(64, 6))
    last_coefficient = align.coefficient()
    check("warmup ramps the coefficient to lambda", abs(last_coefficient - 0.5) < 1e-9,
          f"start {float(first):.6f}, coefficient now {last_coefficient}")


def main():
    print("MMD checks (Section 4.5.2, Eqs. 4.4-4.6)")
    null_case()
    shift_cases()
    ranking_stability()
    weight_behaviour()
    torch_matches_numpy()
    torch_gradients()
    print("\n" + ("All checks passed." if not FAILURES
                  else f"FAILED: {', '.join(FAILURES)}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
