#!/usr/bin/env bash
set -euo pipefail
case ${1:-cpu} in
    cpu|cuda12) ;;
    *) echo "Usage: bash setup.sh [cpu|cuda12]" >&2; exit 2 ;;
esac
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements-dev.txt
if [[ ${1:-cpu} == cuda12 ]]; then
    uv pip install --python .venv/bin/python 'jax[cuda12]==0.11.2'
fi
