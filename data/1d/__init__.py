"""One-dimensional targets from comparison_1d.py (normal source)."""

from dataclasses import dataclass
import numpy as np
from scipy.special import ndtr, ndtri

NAMES = ("uniform", "beta", "gaussian_mixture")


@dataclass
class Distribution:
    name: str
    low: float = 1.0
    high: float = 3.0
    means: tuple = (-2.0, 0.0, 2.0)
    stds: tuple = (0.35, 0.35, 0.35)
    weights: tuple = (1 / 3, 1 / 3, 1 / 3)
    dimension: int = 1

    def __post_init__(self):
        if self.name not in NAMES or not self.low < self.high:
            raise ValueError("Unknown target or invalid uniform bounds")
        if not (len(self.means) == len(self.stds) == len(self.weights)):
            raise ValueError("Mixture parameters must have equal lengths")
        if min(self.stds) <= 0 or min(self.weights) <= 0:
            raise ValueError("Mixture scales and weights must be positive")
        self.weights = np.asarray(self.weights) / np.sum(self.weights)

    def sample(self, rng, n):
        if self.name == "uniform":
            values = rng.uniform(self.low, self.high, n)
        elif self.name == "beta":
            values = rng.beta(0.5, 0.5, n)
        else:
            k = rng.choice(len(self.weights), n, p=self.weights)
            values = np.asarray(self.means)[k] + np.asarray(self.stds)[k] * rng.normal(
                size=n
            )
        return np.asarray(values, np.float32)[:, None]

    def cdf(self, x):
        if self.name == "uniform":
            return np.clip((np.asarray(x) - self.low) / (self.high - self.low), 0, 1)
        if self.name == "beta":
            return 2 / np.pi * np.arcsin(np.sqrt(np.clip(x, 0, 1)))
        return np.sum(
            self.weights * ndtr((np.asarray(x)[..., None] - self.means) / self.stds),
            axis=-1,
        )

    def transport(self, z):
        """Known monotone transport, used only for evaluation."""
        p = ndtr(np.asarray(z, dtype=float))
        if self.name == "uniform":
            return self.low + (self.high - self.low) * p
        if self.name == "beta":
            return np.sin(np.pi * p / 2) ** 2
        q = ndtri(np.clip(p, 1e-15, 1 - 1e-15))
        component_q = np.asarray(self.means) + q[..., None] * self.stds
        low, high = component_q.min(axis=-1), component_q.max(axis=-1)
        for _ in range(64):
            middle = (low + high) / 2
            below = self.cdf(middle) < p
            low, high = np.where(below, middle, low), np.where(below, high, middle)
        return (low + high) / 2
