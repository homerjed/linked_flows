#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
.venv/bin/python -m scripts.1d.run --methods lf fm lsd psd esd --loss l1 --clip-norm 1 --learning-rate .001 --schedule cosine --final-lr-ratio .1 --output runs/flow_maps "$@"
