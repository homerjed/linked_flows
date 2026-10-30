"""Linked flow: fit a direct map to fixed or periodically repaired pairs."""

import equinox as eqx
import jax
import jax.numpy as jnp

from .pairing import pair_indices


@eqx.filter_jit
def generate(model, z, labels=None):
    if labels is None:
        return jax.vmap(model)(z)
    return jax.vmap(lambda x, label: model(x, label=label))(z, labels)


def endpoint_loss(model, z, x, labels=None, power=2):
    """Mean absolute (power=1) or squared (power=2) endpoint error."""
    if power not in (1, 2):
        raise ValueError("power must be 1 or 2")
    return jnp.mean(jnp.abs(generate(model, z, labels) - x) ** power)
