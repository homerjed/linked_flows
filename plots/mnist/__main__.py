"""Regenerate figures: python -m plots.mnist RUN_DIRECTORY."""

import json
from pathlib import Path
import sys

import numpy as np
from . import save

directory = Path(sys.argv[1])
with np.load(directory / "evaluation.npz") as arrays:
    save(
        directory,
        json.loads((directory / "history.json").read_text()),
        arrays["samples"],
        arrays["reference"],
        arrays["labels"] if "labels" in arrays else None,
        arrays["validation_targets"] if "validation_targets" in arrays else None,
        arrays["validation_predictions"]
        if "validation_predictions" in arrays
        else None,
    )
