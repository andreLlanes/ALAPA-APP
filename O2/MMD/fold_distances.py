#Per-fold station distances for leave-one-station-out evaluation.

import argparse
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_O2_ROOT = os.path.dirname(_HERE)
if os.path.join(_O2_ROOT, "common") not in sys.path:
    sys.path.insert(0, os.path.join(_O2_ROOT, "common"))

OUT_DIR = os.path.join(_HERE, "Outputs", "folds")
SCRIPT = os.path.join(_HERE, "station_distance.py")


def frozen_folds():
    """The frozen LOSO stations, or a clear failure if they are not frozen yet."""
    try:
        from folds import FOLD_STATIONS
    except ImportError:
        raise SystemExit(
            "O2/common/folds.py does not exist yet, so there is no frozen fold "
            "list to run.\nFreeze the folds first, or pass --stations "
            "openaq:...,openaq:... to run specific ones.")
    return list(FOLD_STATIONS)


def out_path(key):
    return os.path.join(OUT_DIR, key.replace(":", "-").replace("/", "-") + ".csv")


def main():
    p = argparse.ArgumentParser(description=(
        __doc__ or "Per-fold station distances for leave-one-station-out evaluation."
    ).split("\n\n")[0])
    p.add_argument("--stations", default="",
                   help="comma-separated Metro Manila keys; default is the frozen folds")
    p.add_argument("--no-pm25", action="store_true",
                   help="meteorology and time only, matching the robustness table")
    p.add_argument("--force", action="store_true", help="rebuild folds already written")
    p.add_argument("--dry-run", action="store_true", help="list the runs and stop")
    args = p.parse_args()

    keys = [k.strip() for k in args.stations.split(",") if k.strip()] or frozen_folds()
    os.makedirs(OUT_DIR, exist_ok=True)

    todo = [k for k in keys if args.force or not os.path.exists(out_path(k))]
    print(f"{len(keys)} folds, {len(todo)} to run, "
          f"{len(keys) - len(todo)} already written in {OUT_DIR}")
    if args.dry_run:
        for k in todo:
            print(f"  would write {out_path(k)}")
        return 0

    started = time.time()
    for i, key in enumerate(todo, 1):
        cmd = [sys.executable, SCRIPT, "--exclude-mnl", key, "--out", out_path(key)]
        if args.no_pm25:
            cmd.append("--no-pm25")
        print(f"\n[{i}/{len(todo)}] excluding {key}")
        result = subprocess.run(cmd, cwd=_HERE)
        if result.returncode != 0:
            raise SystemExit(f"fold {key} failed with exit code {result.returncode}")

    if todo:
        print(f"\nWrote {len(todo)} fold tables in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
