"""Straight-path conditional flow matching and fixed-step Heun sampling."""

import equinox as eqx
import jax
import jax.numpy as jnp


def velocity(model, x, t, labels=None):
    t = jnp.broadcast_to(jnp.asarray(t), (len(x),))
    if labels is None:
        return jax.vmap(lambda row, time: model(row, t=time))(x, t)
    return jax.vmap(lambda row, time, label: model(row, t=time, label=label))(
        x, t, labels
    )


def loss(model, z, x, t, labels=None):
    """The caller supplies independent or paired endpoints and uniform times."""
    t = jnp.asarray(t).reshape((-1,) + (1,) * (z.ndim - 1))
    at_t = (1 - t) * z + t * x
    return jnp.mean((velocity(model, at_t, t.reshape(-1), labels) - (x - z)) ** 2)


@eqx.filter_jit
def generate(model, z, labels=None):
    """One Euler step, useful for the original one-step FM diagnostic."""
    return z + velocity(model, z, 0.0, labels)


@eqx.filter_jit
def sample(model, z, steps=64, labels=None):
    """Integrate from 0 to 1 using exactly 2 * steps velocity evaluations."""
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer")
    h = 1.0 / steps

    def advance(i, x):
        t = i * h
        first = velocity(model, x, t, labels)
        second = velocity(model, x + h * first, t + h, labels)
        return x + 0.5 * h * (first + second)

    return jax.lax.fori_loop(0, steps, advance, z)
