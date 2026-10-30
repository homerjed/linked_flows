"""Empirical Wasserstein distances and the unbiased squared-energy statistic."""

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import ks_2samp


def _matrix(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0 or len(x) < 2:
        raise ValueError("Need at least two samples.")
    x = x.reshape(len(x), -1)
    if not x.shape[1] or not np.isfinite(x).all():
        raise ValueError("Samples must be finite and nonempty.")
    return x


def wasserstein(x, y, p=2):
    """Exact empirical 1D W1 or W2, including unequal sample counts."""
    if p not in (1, 2):
        raise ValueError("p must be 1 or 2")
    x, y = np.sort(np.asarray(x).ravel()), np.sort(np.asarray(y).ravel())
    if not len(x) or not len(y):
        raise ValueError("Empty samples")
    if len(x) == len(y):
        return float(np.mean(np.abs(x - y) ** p) ** (1 / p))
    edges = np.unique(
        np.concatenate((np.arange(len(x) + 1) / len(x), np.arange(len(y) + 1) / len(y)))
    )
    middle = (edges[1:] + edges[:-1]) / 2
    delta = x[(middle * len(x)).astype(int)] - y[(middle * len(y)).astype(int)]
    return float(np.sum(np.diff(edges) * np.abs(delta) ** p) ** (1 / p))


def sliced_wasserstein(x, y, projections=128, seed=0):
    """Root mean squared 1D W2 over shared random unit directions."""
    x, y = _matrix(x), _matrix(y)
    if x.shape[1] != y.shape[1] or projections < 1:
        raise ValueError("Need matching dimensions and positive projections.")
    directions = np.random.default_rng(seed).normal(size=(projections, x.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    values = []
    for block in np.array_split(directions, max(1, (projections + 15) // 16)):
        xp, yp = x @ block.T, y @ block.T
        values.extend(wasserstein(a, b) ** 2 for a, b in zip(xp.T, yp.T))
    return float(np.sqrt(np.mean(values)))


def energy_statistic(x, y):
    """Unbiased squared energy U-statistic; finite-sample values may be negative."""
    x, y = _matrix(x), _matrix(y)
    return float(2 * cdist(x, y).mean() - pdist(x).mean() - pdist(y).mean())


def energy_test(x, y, permutations=199, seed=0):
    """Whole-row label permutation test with the plus-one correction.

    Quadratic memory: pass an explicitly selected small held-out sample.
    """
    x, y = _matrix(x), _matrix(y)
    if permutations < 1:
        raise ValueError("permutations must be positive")
    n, m = len(x), len(y)
    pooled = np.concatenate((x, y))
    distances = cdist(pooled, pooled)

    def statistic(order):
        a, b = order[:n], order[n:]
        return (
            2 * distances[np.ix_(a, b)].mean()
            - distances[np.ix_(a, a)].sum() / (n * (n - 1))
            - distances[np.ix_(b, b)].sum() / (m * (m - 1))
        )

    observed = statistic(np.arange(n + m))
    rng = np.random.default_rng(seed)
    tolerance = 128 * np.finfo(float).eps * max(abs(observed), distances.mean())
    exceed = sum(
        statistic(rng.permutation(n + m)) >= observed - tolerance
        for _ in range(permutations)
    )
    return {
        "energy_u": float(observed),
        "energy_pvalue": (1 + exceed) / (1 + permutations),
        "permutations": permutations,
    }


def distribution_metrics(samples, reference, seed=0, projections=128, max_samples=2048):
    """W2/moments on all samples; energy on a recorded seeded subset only."""
    x, y = _matrix(samples), _matrix(reference)
    if x.shape[1] != y.shape[1] or max_samples < 2:
        raise ValueError("Need matching dimensions and max_samples >= 2.")
    rng = np.random.default_rng(seed)
    a = x[rng.choice(len(x), min(len(x), max_samples), replace=False)]
    b = y[rng.choice(len(y), min(len(y), max_samples), replace=False)]
    covariance_x, covariance_y = np.atleast_2d(np.cov(x.T)), np.atleast_2d(np.cov(y.T))
    result = {
        "n": len(x),
        "n_reference": len(y),
        "dimension": x.shape[1],
        "sliced_w2": sliced_wasserstein(x, y, projections, seed),
        "projections": projections,
        "metric_seed": seed,
        "energy_u": energy_statistic(a, b),
        "energy_n": len(a),
        "energy_n_reference": len(b),
        "mean_rmse": float(np.sqrt(np.mean((x.mean(0) - y.mean(0)) ** 2))),
        "covariance_relative_error": float(
            np.linalg.norm(covariance_x - covariance_y)
            / max(np.linalg.norm(covariance_y), 1e-15)
        ),
    }
    if x.shape[1] == 1:
        ks = ks_2samp(x[:, 0], y[:, 0])
        result.update(
            w1=wasserstein(x, y, 1),
            w2=wasserstein(x, y, 2),
            ks_statistic=float(ks.statistic),
            ks_pvalue=float(ks.pvalue),
        )
    return result
