#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
for method in lf lf_unpaired fm; do
    .venv/bin/python -m scripts.mnist.run --method "$method" "$@"
done
