"""Run the source's ten 2D datasets with training-only normalization."""

from importlib import import_module

_data = import_module("data.2d")
Distribution, NAMES = _data.Distribution, _data.NAMES
from scripts.train import arguments, run


def main():
    args = arguments("2d", NAMES)
    for name in args.datasets:
        run(args, Distribution(name))


if __name__ == "__main__":
    main()
