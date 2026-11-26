#!/usr/bin/env bash
# run_gbt.sh — run the GBT training batch. Run from inside O2/.
set -u

cmds=(
  "--model gbt --city mm"
  "--model gbt --city bk"
  "--model gbt --city la"
  "--model gbt --city mm --transfer true --source bk --training-variant all"
  "--model gbt --city mm --transfer true --source la --training-variant all"
  "--model gbt --city mm --transfer true --source bk+la --training-variant all"
)

for c in "${cmds[@]}"; do
  echo "=== uv run train.py $c ==="
  uv run train.py $c || echo "!!! FAILED: $c"
done
echo "=== batch finished ==="
