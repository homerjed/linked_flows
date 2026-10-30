"""The ten source 2D laws, including their finite-batch sampling designs.

`sample(..., iid=True)` replaces grids and mirrored pairs with independent
continuous draws. Normalize using training mean/std only, in the runner.
"""

import numpy as np

NAMES = (
    "checkerboard",
    "circles",
    "circles_repeat",
    "eight_gaussians",
    "two_moons",
    "pinwheel",
    "rings",
    "s_curve",
    "swiss_roll",
    "two_spirals",
)


def polar(theta, radius):
    return (
        np.column_stack((np.cos(theta), np.sin(theta))) * np.asarray(radius)[..., None]
    )


class Distribution:
    dimension = 2

    def __init__(self, name):
        if name not in NAMES:
            raise ValueError(f"Unknown 2D target: {name}")
        self.name = name

    def sample(self, rng, n, iid=False, dtype=np.float32):
        name = self.name
        if name == "checkerboard":
            tiles = np.array(
                [(i, j) for i in range(4) for j in range(4) if (i + j) % 2 == 0]
            )
            values = (
                -1
                + 0.5 * tiles[rng.integers(0, 8, n)]
                + np.column_stack((rng.uniform(0, 0.5, n), rng.uniform(0, 0.5, n)))
            )
        elif name in ("circles", "circles_repeat", "rings"):
            circles = name != "rings"
            radii = [1.0, 0.5] if circles else [3.0, 2.25, 1.5, 0.75]
            if iid:
                radius = np.asarray(radii)[rng.integers(0, len(radii), n)]
                values = polar(rng.uniform(0, 2 * np.pi, n), radius)
            else:
                counts = (
                    [n // 2, n - n // 2]
                    if circles
                    else [n // 4] * 3 + [n - 3 * (n // 4)]
                )
                values = rng.permutation(
                    np.concatenate(
                        [
                            polar(np.linspace(0, 2 * np.pi, count, endpoint=False), r)
                            for count, r in zip(counts, radii)
                        ]
                    )
                )
            values += rng.normal(0, 0.08, (n, 2))
            if circles:
                values *= 3
        elif name == "eight_gaussians":
            h = 1 / np.sqrt(2)
            centers = 4 * np.array(
                [(1, 0), (-1, 0), (0, 1), (0, -1), (h, h), (h, -h), (-h, h), (-h, -h)]
            )
            values = np.empty((n, 2))
            for i in range(n):
                noise = rng.normal(0, 0.5, 2)
                values[i] = centers[rng.integers(0, 8)] + noise
            values /= 1.414
        elif name == "two_moons":
            if iid:
                angle, lower = (
                    rng.uniform(0, np.pi, n),
                    rng.integers(0, 2, n).astype(bool),
                )
                values = np.column_stack(
                    (
                        np.where(lower, 1 - np.cos(angle), np.cos(angle)),
                        np.where(lower, 0.5 - np.sin(angle), np.sin(angle)),
                    )
                )
            else:
                a, b = np.linspace(0, np.pi, n // 2), np.linspace(0, np.pi, n - n // 2)
                values = rng.permutation(
                    np.concatenate(
                        (
                            np.column_stack((np.cos(a), np.sin(a))),
                            np.column_stack((1 - np.cos(b), 0.5 - np.sin(b))),
                        )
                    )
                )
            values += rng.normal(0, 0.1, (n, 2))
        elif name == "pinwheel":
            features = rng.normal(size=(n, 2)) * [0.3, 0.1] + [1, 0]
            counts = np.full(5, n // 5)
            counts[: n % 5] += 1
            labels = rng.integers(0, 5, n) if iid else np.repeat(np.arange(5), counts)
            angles = labels * (2 * np.pi / 5) + 0.25 * np.exp(features[:, 0])
            c, s = np.cos(angles), np.sin(angles)
            values = 2 * np.column_stack(
                (
                    features[:, 0] * c + features[:, 1] * s,
                    -features[:, 0] * s + features[:, 1] * c,
                )
            )
            if not iid:
                values = rng.permutation(values)
        elif name in ("s_curve", "swiss_roll"):
            u = rng.uniform(size=n)
            rng.uniform(size=n)  # Preserve draws for the omitted middle coordinate.
            noise = 0.1 * rng.normal(size=(3, n))[[0, 2]].T
            if name == "s_curve":
                t = 3 * np.pi * (u - 0.5)
                values = np.column_stack((np.sin(t), np.sign(t) * (np.cos(t) - 1)))
            else:
                t = 1.5 * np.pi * (1 + 2 * u)
                values = polar(t, t)
            values += noise
        else:
            size = n if iid else (n + 1) // 2
            theta = 3 * np.pi * np.sqrt(rng.random(size))
            base = np.column_stack(
                (
                    -theta * np.cos(theta) + 0.5 * rng.random(size),
                    theta * np.sin(theta) + 0.5 * rng.random(size),
                )
            )
            values = (
                base * (2 * rng.integers(0, 2, n) - 1)[:, None]
                if iid
                else np.concatenate((base, -base[: n // 2]))
            )
            values = values / 3 + rng.normal(0, 0.1, (n, 2))
        return np.asarray(values, dtype=dtype)
