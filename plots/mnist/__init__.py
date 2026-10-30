"""Saved MNIST grids and loss curves (also rerunnable from a run directory)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def grid(images, path, title, labels=None):
    columns = min(8, len(images))
    if labels is None:
        rows = min(8, (len(images) + columns - 1) // columns)
        indices = [
            np.arange(len(images))[i * columns : (i + 1) * columns] for i in range(rows)
        ]
    else:
        rows = 10
        columns = min(columns, max(np.count_nonzero(labels == c) for c in range(10)))
        indices = [np.flatnonzero(labels == c)[:columns] for c in range(10)]
    fig, axes = plt.subplots(
        rows, columns, figsize=(max(3, columns), rows), squeeze=False
    )
    for row, members in enumerate(indices):
        for col, ax in enumerate(axes[row]):
            ax.set_xticks([])
            ax.set_yticks([])
            if col < len(members):
                ax.imshow(images[members[col], 0], cmap="gray", vmin=-1, vmax=1)
            if labels is not None and col == 0:
                ax.set_ylabel(str(row))
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save(
    run_dir, history, samples, reference, labels=None, targets=None, predictions=None
):
    run_dir = Path(run_dir)
    grid(samples, run_dir / "samples.png", "Fresh generated samples", labels)
    grid(reference, run_dir / "reference.png", "Official MNIST test reference", labels)
    if targets is not None:
        grid(targets, run_dir / "validation_targets.png", "Held-out assigned targets")
        grid(
            predictions,
            run_dir / "validation_predictions.png",
            "Held-out endpoint predictions",
        )
    fig, ax = plt.subplots(figsize=(6, 4))
    for key in ("train_loss", "validation_loss"):
        ax.plot(
            [row["epoch"] for row in history], [row[key] for row in history], label=key
        )
    ax.set(xlabel="Epoch", ylabel="MSE", title="Training and validation objectives")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "loss.png", dpi=120)
    plt.close(fig)
