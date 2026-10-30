"""Run ND targets; choose --gaussian-kind ar1 for dimension scaling."""

from data.nd import Distribution, NAMES
from scripts.train import arguments, run


def main():
    args = arguments("nd", NAMES)
    for name in args.datasets:
        target = Distribution(
            name,
            args.dimension,
            args.target_seed,
            args.beta,
            args.gaussian_kind,
            args.rho,
            args.mean,
        )
        run(args, target)


if __name__ == "__main__":
    main()
