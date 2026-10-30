#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# Shared data/reference streams across model seeds. Override budgets with CLI flags.
for dataset in gaussian banana_mixture gaussian_mixture; do
  for dimension in 2 4 8 16 32; do
    for seed in 0 1 2 3; do
      .venv/bin/python -m scripts.nd.run --datasets "$dataset" --dimension "$dimension" \
        --seed "$seed" --data-seed 20261020 --gaussian-kind ar1 --methods lf fm \
        --output runs/scaling "$@"
    done
  done
done
