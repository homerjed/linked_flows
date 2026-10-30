"""Small distribution diagnostics shared by every experiment."""

from .distances import (
    distribution_metrics,
    energy_statistic,
    energy_test,
    sliced_wasserstein,
    wasserstein,
)

__all__ = [
    "distribution_metrics",
    "energy_statistic",
    "energy_test",
    "sliced_wasserstein",
    "wasserstein",
]
