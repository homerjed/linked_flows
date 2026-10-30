"""Small U-Net shared by endpoint and velocity regression."""

import equinox as eqx
import jax
import jax.numpy as jnp


class DoubleConv(eqx.Module):
    conv1: eqx.nn.Conv2d
    norm1: eqx.nn.GroupNorm
    conv2: eqx.nn.Conv2d
    norm2: eqx.nn.GroupNorm

    def __init__(self, channels_in, channels_out, key):
        k1, k2 = jax.random.split(key)
        self.conv1 = eqx.nn.Conv2d(
            channels_in, channels_out, 3, padding=1, use_bias=False, key=k1
        )
        self.norm1 = eqx.nn.GroupNorm(groups=8, channels=channels_out)
        self.conv2 = eqx.nn.Conv2d(
            channels_out, channels_out, 3, padding=1, use_bias=False, key=k2
        )
        self.norm2 = eqx.nn.GroupNorm(groups=8, channels=channels_out)

    def __call__(self, x):
        return jax.nn.silu(
            self.norm2(self.conv2(jax.nn.silu(self.norm1(self.conv1(x)))))
        )


class UNet(eqx.Module):
    encoder28: DoubleConv
    down14: eqx.nn.Conv2d
    encoder14: DoubleConv
    down7: eqx.nn.Conv2d
    bottleneck7: DoubleConv
    mix7: DoubleConv
    up14: eqx.nn.Conv2d
    decoder14: DoubleConv
    up28: eqx.nn.Conv2d
    decoder28: DoubleConv
    head: eqx.nn.Conv2d
    conditional: bool = eqx.field(static=True)
    flow: bool = eqx.field(static=True)

    def __init__(self, key, base_channels=32, conditional=False, flow=False):
        if base_channels < 8 or base_channels % 8:
            raise ValueError("base_channels must be a positive multiple of 8")
        self.conditional, self.flow = conditional, flow
        keys = iter(jax.random.split(key, 11))
        b = base_channels
        self.encoder28 = DoubleConv(
            1 + int(flow) + 10 * int(conditional), b, next(keys)
        )
        self.down14 = eqx.nn.Conv2d(b, 2 * b, 3, stride=2, padding=1, key=next(keys))
        self.encoder14 = DoubleConv(2 * b, 2 * b, next(keys))
        self.down7 = eqx.nn.Conv2d(2 * b, 4 * b, 3, stride=2, padding=1, key=next(keys))
        self.bottleneck7 = DoubleConv(4 * b, 4 * b, next(keys))
        self.mix7 = DoubleConv(4 * b, 4 * b, next(keys))
        self.up14 = eqx.nn.Conv2d(4 * b, 2 * b, 3, padding=1, key=next(keys))
        self.decoder14 = DoubleConv(4 * b, 2 * b, next(keys))
        self.up28 = eqx.nn.Conv2d(2 * b, b, 3, padding=1, key=next(keys))
        self.decoder28 = DoubleConv(2 * b, b, next(keys))
        self.head = eqx.nn.Conv2d(b, 1, 1, key=next(keys))

    def __call__(self, x, t=0.0, s=0.0, label=None):
        """One CHW image; time is used only for flow matching."""
        if x.shape != (1, 28, 28):
            raise ValueError("UNet expects (1, 28, 28); use jax.vmap for batches")
        inputs = [x]
        if self.flow:
            inputs.append(jnp.broadcast_to(jnp.asarray(t, x.dtype), x.shape))
        if self.conditional:
            if label is None:
                raise ValueError("A conditional UNet requires a digit label")
            inputs.append(
                jnp.broadcast_to(jax.nn.one_hot(label, 10)[:, None, None], (10, 28, 28))
            )
        skip28 = self.encoder28(jnp.concatenate(inputs))
        skip14 = self.encoder14(jax.nn.silu(self.down14(skip28)))
        x = self.mix7(self.bottleneck7(jax.nn.silu(self.down7(skip14))))
        x = jax.nn.silu(self.up14(jax.image.resize(x, (x.shape[0], 14, 14), "nearest")))
        x = self.decoder14(jnp.concatenate((x, skip14)))
        x = jax.nn.silu(self.up28(jax.image.resize(x, (x.shape[0], 28, 28), "nearest")))
        x = self.head(self.decoder28(jnp.concatenate((x, skip28))))
        return x if self.flow else jnp.tanh(x)
