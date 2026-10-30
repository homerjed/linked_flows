"""LSD, PSD, and ESD self-distillation from scripts/flow_maps_1d.py.

The map is X(s,t,x) = x + (t-s) * slope(x,s,t). Teachers use the current
network with stopped gradients; there is no pretrained or EMA teacher.
"""

import equinox as eqx
import jax
import jax.numpy as jnp


def flow_map(model, x, s, t, label=None):
    return x + (t - s) * model(x, t=t, s=s, label=label)


def residual(model, x, s, t, method, label=None):
    """Stopped-teacher residuals, with x fixed in time partial derivatives."""
    if method == "LSD":
        end, dt = jax.jvp(
            lambda end_time: flow_map(model, x, s, end_time, label),
            (t,),
            (jnp.ones_like(t),),
        )
        return dt - jax.lax.stop_gradient(model(end, t=t, s=t, label=label))
    if method == "ESD":
        _, ds = jax.jvp(
            lambda start_time: flow_map(model, x, start_time, t, label),
            (s,),
            (jnp.ones_like(s),),
        )
        velocity = model(x, t=s, s=s, label=label)
        _, advective = jax.jvp(
            lambda position: flow_map(model, position, s, t, label), (x,), (velocity,)
        )
        return ds + jax.lax.stop_gradient(advective)
    if method == "PSD":
        middle = (s + t) / 2
        first = model(x, t=middle, s=s, label=label)
        midpoint = x + (middle - s) * first
        teacher = 0.5 * (first + model(midpoint, t=t, s=middle, label=label))
        return model(x, t=t, s=s, label=label) - jax.lax.stop_gradient(teacher)
    raise ValueError("method must be LSD, PSD, or ESD")


def loss(model, z, x, t, s, method, diagonal_weight=0.75, *, r=None, labels=None):
    """Combine diagonal FM MSE and self-distillation MSE.

    Pass independent uniform arrays s,t; they are sorted for the SD interval.
    r supplies independent diagonal times, as in the original experiment.
    If omitted, the unsorted t is reused (same expectation, correlated draws).
    When supplying already sorted s,t, provide independent uniform r.
    """
    if not 0 < diagonal_weight < 1:
        raise ValueError("diagonal_weight must lie strictly between 0 and 1")
    r = jnp.asarray(t if r is None else r)
    s, t = jnp.minimum(s, t), jnp.maximum(s, t)
    shape = (-1,) + (1,) * (z.ndim - 1)
    at_r = (1 - r.reshape(shape)) * z + r.reshape(shape) * x
    at_s = (1 - s.reshape(shape)) * z + s.reshape(shape) * x
    if labels is None:
        predicted = jax.vmap(lambda row, time: model(row, t=time, s=time))(at_r, r)
        errors = jax.vmap(
            lambda row, start, end: residual(model, row, start, end, method)
        )(at_s, s, t)
    else:
        predicted = jax.vmap(
            lambda row, time, label: model(row, t=time, s=time, label=label)
        )(at_r, r, labels)
        errors = jax.vmap(
            lambda row, start, end, label: residual(
                model, row, start, end, method, label
            )
        )(at_s, s, t, labels)
    diagonal = jnp.mean((predicted - (x - z)) ** 2)
    return diagonal_weight * diagonal + (1 - diagonal_weight) * jnp.mean(errors**2)


@eqx.filter_jit
def generate(model, z, labels=None):
    if labels is None:
        return jax.vmap(lambda x: flow_map(model, x, 0.0, 1.0))(z)
    return jax.vmap(lambda x, label: flow_map(model, x, 0.0, 1.0, label))(z, labels)
