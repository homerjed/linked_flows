"""Exact one-to-one assignment in source order, including classwise pairing."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def pair_indices(source, target, metric="euclidean", labels=None, max_cost_gib=4):
    """Return p with source[i] paired to target[p[i]].

    Images are flattened. In 1D, rank pairing is exact for both metrics.
    Otherwise SciPy solves the dense assignment. The memory limit covers
    each float64 cost matrix; inputs and solver workspace require extra RAM.
    With labels, source and target share the supplied label ordering and each
    class is solved separately, so labels[p] equals labels.
    """
    source, target = np.asarray(source), np.asarray(target)
    if source.shape != target.shape or source.ndim < 1 or not len(source):
        raise ValueError("source and target must have equal nonempty shapes")
    source, target = source.reshape(len(source), -1), target.reshape(len(target), -1)
    if source.shape[1] == 0 or not (
        np.isfinite(source).all() and np.isfinite(target).all()
    ):
        raise ValueError("pairing requires finite features")
    if metric not in ("euclidean", "squared_euclidean", "sqeuclidean"):
        raise ValueError("metric must be euclidean or squared_euclidean")
    if not np.isfinite(max_cost_gib) or max_cost_gib <= 0:
        raise ValueError("max_cost_gib must be positive")
    metric = "euclidean" if metric == "euclidean" else "sqeuclidean"
    groups = [np.arange(len(source))]
    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape != (len(source),) or labels.dtype.kind not in "iu":
            raise ValueError("labels must be one integer per example")
        groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    permutation = np.empty(len(source), dtype=np.int64)
    for indices in groups:
        z, x = source[indices], target[indices]
        if source.shape[1] == 1:
            rows = np.argsort(z[:, 0], kind="stable")
            columns = np.argsort(x[:, 0], kind="stable")
        else:
            required = 8 * len(indices) ** 2 / 1024**3
            if required > max_cost_gib:
                raise MemoryError(
                    f"Assignment cost matrix needs {required:.2f} GiB; "
                    f"limit is {max_cost_gib:.2f} GiB. Reduce training pool size."
                )
            rows, columns = linear_sum_assignment(cdist(z, x, metric=metric))
        permutation[indices[rows]] = indices[columns]
    return permutation
