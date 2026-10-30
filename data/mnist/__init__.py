"""MNIST gzip cache, normalization, and reproducible 80/20 splits."""

import gzip
import hashlib
from pathlib import Path
import struct
from urllib.request import urlopen

import numpy as np

FILES = (
    ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873", (60000, 28, 28)),
    ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432", (60000,)),
    ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3", (10000, 28, 28)),
    ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c", (10000,)),
)


def load(directory="data/mnist/cache", download=True):
    """Return uint8 train images/labels and official test images/labels.

    Pass a directory containing the four gz files, or an original data cache
    containing mnist_raw/. Only missing archives are downloaded.
    """
    directory = Path(directory)
    if (directory / "mnist_raw").is_dir():
        directory /= "mnist_raw"
    arrays = []
    for filename, checksum, shape in FILES:
        path = directory / filename
        if not path.exists():
            if not download:
                raise FileNotFoundError(path)
            directory.mkdir(parents=True, exist_ok=True)
            with urlopen(
                "https://storage.googleapis.com/cvdf-datasets/mnist/" + filename,
                timeout=60,
            ) as response:
                payload = response.read()
            if hashlib.md5(payload).hexdigest() != checksum:
                raise ValueError(f"MNIST checksum mismatch: {filename}")
            path.write_bytes(payload)
        payload = path.read_bytes()
        if hashlib.md5(payload).hexdigest() != checksum:
            raise ValueError(f"MNIST checksum mismatch: {path}")
        raw = gzip.decompress(payload)
        header_size = 4 * (len(shape) + 1)
        header = struct.unpack(">" + "I" * (len(shape) + 1), raw[:header_size])
        if header != (0x800 + len(shape), *shape) or len(raw) != header_size + np.prod(
            shape
        ):
            raise ValueError(f"Invalid MNIST IDX data: {path}")
        array = np.frombuffer(raw, dtype=np.uint8, offset=header_size).reshape(shape)
        arrays.append(array[:, None] if len(shape) == 3 else array)
    return tuple(arrays)


def normalize(images):
    return np.asarray(images, dtype=np.float32) / np.float32(127.5) - np.float32(1)


def quotas(total, counts, minimum, maximum):
    """Proportional integer counts, with exact total and class bounds."""
    if total < np.sum(minimum) or total > np.sum(maximum):
        raise ValueError("Requested size cannot retain every class in both splits")
    ideal = total * counts / counts.sum()
    result = np.clip(np.floor(ideal).astype(int), minimum, maximum)
    while result.sum() < total:
        result[np.argmax(np.where(result < maximum, ideal - result, -np.inf))] += 1
    while result.sum() > total:
        result[np.argmax(np.where(result > minimum, result - ideal, -np.inf))] -= 1
    return result


def split_indices(labels, n_total, seed=0, conditional=False):
    """Select from official training data, then split; conditional data are class-sorted."""
    if not 2 <= n_total <= len(labels):
        raise ValueError("n_total must be between 2 and the training dataset size")
    selection_rng, split_rng = (
        np.random.default_rng(seed),
        np.random.default_rng(seed + 6000),
    )
    n_train = int(0.8 * n_total)
    if not conditional:
        selected = selection_rng.choice(len(labels), n_total, replace=False)
        order = split_rng.permutation(n_total)
        return selected[order[:n_train]], selected[order[n_train:]]
    counts = np.bincount(labels, minlength=10)
    sizes = quotas(n_total, counts, np.full(10, 2), counts)
    selected = [
        np.sort(selection_rng.choice(np.flatnonzero(labels == c), n, replace=False))
        for c, n in enumerate(sizes)
    ]
    train_sizes = quotas(n_train, sizes, np.ones(10, dtype=int), sizes - 1)
    train, validation = [], []
    for rows, n in zip(selected, train_sizes):
        mask = np.zeros(len(rows), dtype=bool)
        mask[split_rng.choice(len(rows), n, replace=False)] = True
        train.append(rows[mask])
        validation.append(rows[~mask])
    return np.concatenate(train), np.concatenate(validation)


def evaluation_indices(labels, n_eval, seed=0, conditional=False):
    """Two disjoint official-test reference sets, balanced when conditional."""
    rng = np.random.default_rng(seed + 5000)
    if n_eval < 1 or 2 * n_eval > len(labels):
        raise ValueError("Two disjoint test reference sets need 1 <= n_eval <= 5000")
    if not conditional:
        return tuple(
            rng.choice(len(labels), 2 * n_eval, replace=False).reshape(2, n_eval)
        )
    if n_eval < 10:
        raise ValueError("Conditional evaluation needs at least ten samples")
    counts = np.full(10, n_eval // 10)
    counts[: n_eval % 10] += 1
    if np.any(2 * counts > np.bincount(labels, minlength=10)):
        raise ValueError(
            "Not enough test images per class for two balanced reference sets"
        )
    pairs = [
        rng.choice(np.flatnonzero(labels == c), 2 * n, replace=False).reshape(2, n)
        for c, n in enumerate(counts)
    ]
    return tuple(np.concatenate([np.sort(pair[i]) for pair in pairs]) for i in (0, 1))
