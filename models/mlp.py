"""Small residual generator, velocity network, or flow-map slope network."""

import equinox as eqx
import jax
import jax.numpy as jnp


class MLP(eqx.Module):
    net: eqx.nn.MLP
    kind: str = eqx.field(static=True)
    residual: bool = eqx.field(static=True)

    def __init__(
        self,
        dimension,
        width=128,
        depth=3,
        kind="lf",
        *,
        key,
        activation="silu",
        residual=True,
    ):
        if kind not in ("lf", "fm", "fmap"):
            raise ValueError("kind must be lf, fm, or fmap")
        if activation not in ("silu", "elu"):
            raise ValueError("activation must be silu or elu")
        self.kind = kind
        self.residual = residual
        self.net = eqx.nn.MLP(
            dimension + {"lf": 0, "fm": 1, "fmap": 2}[kind],
            dimension,
            width,
            depth,
            activation=getattr(jax.nn, activation),
            key=key,
        )

    def __call__(self, x, t=0.0, s=0.0, label=None):
        """Evaluate one vector; labels are accepted for a common model API."""
        if self.kind == "lf":
            return x + self.net(x) if self.residual else self.net(x)
        times = (t,) if self.kind == "fm" else (s, t)
        return self.net(jnp.concatenate((x, jnp.asarray(times).reshape(-1))))
