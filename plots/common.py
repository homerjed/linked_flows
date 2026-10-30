"""Small plots of saved runs; native losses are method-specific objectives."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "lf_unpaired": "LF unpaired",
    "lf": "LF",
    "fm": "FM Heun",
    "fm_paired": "FM paired Heun",
    "lsd": "LSD",
    "psd": "PSD",
    "esd": "ESD",
    "fm_euler": "FM Euler",
    "fm_paired_euler": "FM paired Euler",
}


def plot_run(directory):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    with np.load(directory / "samples.npz") as archive:
        samples = dict(archive)
    with np.load(directory / "history.npz") as archive:
        history = dict(archive)
    methods = [name for name in LABELS if name in samples]
    dimension = config["dimension"]
    colors = dict(zip(methods, plt.get_cmap("tab10").colors))
    if dimension == 1:
        fig, axes = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True)
        all_values = np.concatenate(
            [samples[name][:, 0] for name in ["reference", *methods]]
        )
        low, high = float(all_values.min()), float(all_values.max())
        bins = np.linspace(low - 0.2 * (high - low), high + 0.2 * (high - low), 80)
        axes[0].hist(
            samples["reference"][:, 0],
            bins=bins,
            density=True,
            color=".8",
            label="Target",
        )
        order = np.argsort(samples["source"][:, 0])
        order = order[np.linspace(0, len(order) - 1, min(1500, len(order))).astype(int)]
        for method in methods:
            axes[0].hist(
                samples[method][:, 0],
                bins=bins,
                density=True,
                histtype="step",
                color=colors[method],
                label=LABELS[method],
            )
            axes[1].plot(
                samples["source"][order, 0],
                samples[method][order, 0],
                color=colors[method],
                label=LABELS[method],
                lw=1,
            )
        if "analytic" in samples:
            axes[1].plot(
                samples["source"][order, 0],
                samples["analytic"][order, 0],
                "k--",
                label="Analytic",
                lw=1,
            )
        axes[0].set(title="Generated density", xlabel="x", ylabel="Density")
        axes[1].set(title="Learned transport", xlabel="Source z", ylabel="Generated x")
        axes[1].legend(fontsize=7)
        loss_axes = axes[2:]
    else:
        columns = min(5, len(methods) + 3)
        rows = (len(methods) + 2 + columns) // columns
        fig, grid = plt.subplots(
            rows,
            columns,
            figsize=(3.4 * columns, 3.1 * rows),
            squeeze=False,
            constrained_layout=True,
        )
        for ax in grid.ravel():
            ax.set_visible(False)
        point_axes = grid.ravel()[: len(methods) + 1]
        all_values = np.concatenate(
            [samples[name][:, :2] for name in ["reference", *methods]]
        )
        low, high = all_values.min(axis=0), all_values.max(axis=0)
        margin = 0.12 * (high - low)
        for ax, name in zip(point_axes, ["reference", *methods]):
            ax.set_visible(True)
            values = samples[name][:3000]
            ax.scatter(
                values[:, 0],
                values[:, 1],
                s=2,
                alpha=0.35,
                color=".4" if name == "reference" else colors[name],
                rasterized=True,
            )
            ax.set(
                title="Target" if name == "reference" else LABELS[name],
                xlabel="Coordinate 1",
                ylabel="Coordinate 2",
                xlim=(low[0] - margin[0], high[0] + margin[0]),
                ylim=(low[1] - margin[1], high[1] + margin[1]),
            )
        loss_axes = grid.ravel()[len(methods) + 1 : len(methods) + 3]
        for ax in loss_axes:
            ax.set_visible(True)
    for method in config["methods"]:
        for ax, field in zip(loss_axes, ["native_loss", "endpoint_mse"]):
            ax.plot(
                history[f"{method}_step"],
                history[f"{method}_{field}"],
                label=LABELS[method],
                color=colors[method],
            )
            ax.set_yscale("log")
            ax.set_xlabel("Training updates")
    loss_axes[0].set(
        title="Native held-out objectives", ylabel="Loss (objectives differ)"
    )
    loss_axes[1].set(title="Common endpoint diagnostic", ylabel="Endpoint MSE")
    loss_axes[0].legend(fontsize=7)
    fig.suptitle(f"{config['dataset']} · {dimension}D · seed {config['seed']}")
    fig.savefig(directory / "comparison.png", dpi=160)
    fig.savefig(directory / "comparison.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Replot one or more saved experiment directories"
    )
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    for directory in args.runs:
        plot_run(directory)


if __name__ == "__main__":
    main()
