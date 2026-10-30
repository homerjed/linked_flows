"""Run 1D LF/FM or LSD/PSD/ESD comparisons."""

from importlib import import_module

_data = import_module("data.1d")
Distribution, NAMES = _data.Distribution, _data.NAMES
from scripts.train import arguments, run


def main():
    args = arguments("1d", NAMES)
    for name in args.datasets:
        target = Distribution(
            name,
            *args.uniform_bounds,
            args.mixture_means,
            args.mixture_stds,
            args.mixture_weights,
        )
        run(args, target)


if __name__ == "__main__":
    main()
