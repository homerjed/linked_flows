"""Correlated Gaussian, coupled banana, and asymmetric five-mode targets."""

import numpy as np

NAMES = ("gaussian", "banana_mixture", "gaussian_mixture")


def symmetric_sqrt(matrix):
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.T


def vine_correlation(rng, dimension, beta):
    raw = 2 * rng.beta(beta, beta, dimension * (dimension - 1) // 2) - 1
    partial, correlation = np.zeros((dimension, dimension)), np.eye(dimension)
    index = 0
    for k in range(dimension - 1):
        for i in range(k + 1, dimension):
            value = partial[k, i] = raw[index]
            index += 1
            for level in range(k - 1, -1, -1):
                a, b = partial[level, i], partial[level, k]
                value = value * np.sqrt(max(0.0, (1 - a * a) * (1 - b * b))) + a * b
            correlation[k, i] = correlation[i, k] = value
    permutation = rng.permutation(dimension)
    return correlation[np.ix_(permutation, permutation)]


class Distribution:
    def __init__(
        self,
        name,
        dimension=16,
        seed=16,
        beta=0.9,
        gaussian_kind="vine",
        rho=0.5,
        mean=0.5,
    ):
        if (
            name not in NAMES
            or dimension < 2
            or (name == "banana_mixture" and dimension % 2)
        ):
            raise ValueError("Unknown target, dimension < 2, or odd banana dimension")
        self.name, self.dimension = name, dimension
        if name == "gaussian":
            rng = lambda stream: np.random.default_rng(
                np.random.SeedSequence([seed, stream])
            )
            if gaussian_kind == "vine":
                self.mean = rng(0).uniform(-1, 1, dimension)
                scales = rng(1).uniform(-1, 1, dimension)
                self.covariance = (
                    scales[:, None]
                    * vine_correlation(rng(2), dimension, beta)
                    * scales[None, :]
                )
            else:
                self.mean = np.full(dimension, mean)
                self.covariance = rho ** np.abs(
                    np.arange(dimension)[:, None] - np.arange(dimension)
                )
            self.root = symmetric_sqrt(self.covariance)
        elif name == "gaussian_mixture":
            self.weights = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
            offsets, alternating = (
                np.array([-2.5, -1.0, 0.5, 1.5, 3.0]),
                np.array([-0.5, 1.5, -1.5, 0.75, -0.25]),
            )
            scales, correlations = (
                np.array([0.35, 0.50, 0.65, 0.45, 0.60]),
                np.array([-0.4, 0.15, 0.45, -0.25, 0.65]),
            )
            self.component_means = offsets[:, None] + alternating[:, None] * np.where(
                np.arange(dimension) % 2, -1.0, 1.0
            )
            lags = np.abs(np.arange(dimension)[:, None] - np.arange(dimension))
            self.component_covariances = (
                scales[:, None, None] ** 2 * correlations[:, None, None] ** lags
            )
            self.roots = np.linalg.cholesky(self.component_covariances)
            self.mean = self.weights @ self.component_means
            centered = self.component_means - self.mean
            self.covariance = np.einsum(
                "k,kij->ij", self.weights, self.component_covariances
            ) + np.einsum("k,ki,kj->ij", self.weights, centered, centered)
        else:
            pairs = dimension // 2
            latent_covariance = 0.65 * np.eye(pairs) + 0.35
            self.scale = np.tile(
                [np.sqrt(5.0), np.sqrt(0.75**2 + 2 * 0.8**2 + 0.3**2)], pairs
            )
            self.mean, self.covariance = (
                np.zeros(dimension),
                np.empty((dimension, dimension)),
            )
            self.covariance[::2, ::2] = 4 + latent_covariance
            self.covariance[1::2, 1::2] = (
                0.75**2 + 2 * 0.8**2 * latent_covariance**2 + 0.3**2 * np.eye(pairs)
            )
            self.covariance[::2, 1::2] = self.covariance[1::2, ::2] = 1.5
            self.covariance /= self.scale[:, None] * self.scale[None, :]

    def sample(self, rng, n):
        d = self.dimension
        if self.name == "gaussian":
            values = self.mean + rng.normal(size=(n, d)) @ self.root
        elif self.name == "gaussian_mixture":
            labels = rng.choice(5, n, p=self.weights)
            values = self.component_means[labels] + np.einsum(
                "nij,nj->ni", self.roots[labels], rng.normal(size=(n, d))
            )
        else:
            sign = 2 * rng.integers(0, 2, (n, 1)) - 1
            latent = np.sqrt(0.35) * rng.normal(size=(n, 1)) + np.sqrt(
                0.65
            ) * rng.normal(size=(n, d // 2))
            values = np.empty((n, d))
            values[:, ::2] = 2 * sign + latent
            values[:, 1::2] = (
                0.75 * sign + 0.8 * (latent**2 - 1) + 0.3 * rng.normal(size=(n, d // 2))
            )
            values /= self.scale
        return values.astype(np.float32)

    def transport(self, z):
        if self.name != "gaussian":
            raise ValueError("Analytic Gaussian transport only")
        return self.mean + np.asarray(z) @ self.root
