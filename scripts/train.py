"""Shared fixed-budget toy runner; run through scripts.1d/2d/nd.run."""

import argparse
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.stats import kstest

import fm
import fmap
import lf
from models import MLP
from metrics import distribution_metrics

METHODS = ("lf_unpaired", "lf", "fm", "fm_paired", "lsd", "psd", "esd")
MAPS = ("lsd", "psd", "esd")


def arguments(family, names):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=names, default=list(names))
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS[:4]))
    p.add_argument("--profile", choices=("smoke", "quick", "full"), default="quick")
    p.add_argument("--output", type=Path, default=Path("runs") / family)
    p.add_argument("--seed", type=int, default=16 if family == "nd" else 0)
    p.add_argument(
        "--data-seed", type=int, help="Fix training/reference data across model seeds"
    )
    for name in (
        "n_train",
        "n_test",
        "n_eval",
        "probe_size",
        "steps",
        "batch_size",
        "width",
        "depth",
        "eval_every",
        "ode_steps",
    ):
        p.add_argument("--" + name.replace("_", "-"), type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--weight-decay", type=float)
    p.add_argument("--final-lr-ratio", type=float, default=0.05)
    p.add_argument("--schedule", choices=("constant", "cosine"))
    p.add_argument("--activation", choices=("silu", "elu"))
    p.add_argument("--residual", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument(
        "--loss",
        choices=("l1", "l2"),
        default="l2",
        help="LF endpoint loss; use l1 for flow-map comparison",
    )
    p.add_argument(
        "--coupling",
        choices=("independent", "paired"),
        default="independent",
        help="LSD/PSD/ESD endpoint coupling",
    )
    p.add_argument("--diagonal-weight", type=float, default=0.75)
    p.add_argument(
        "--clip-norm",
        type=float,
        default=0.0,
        help="0 disables gradient clipping; flow-map recipe uses 1",
    )
    p.add_argument(
        "--max-cost-gib",
        type=float,
        default=4.0,
        help="Dense exact-pairing memory guard; full 2D needs --max-cost-gib 128 and sufficient RAM",
    )
    p.add_argument("--metric-samples", type=int, default=2048)
    p.add_argument("--projections", type=int, default=128)
    if family == "1d":
        p.add_argument("--uniform-bounds", nargs=2, type=float, default=[1.0, 3.0])
        p.add_argument("--mixture-means", nargs=3, type=float, default=[-2.0, 0.0, 2.0])
        p.add_argument(
            "--mixture-stds", nargs=3, type=float, default=[0.35, 0.35, 0.35]
        )
        p.add_argument(
            "--mixture-weights", nargs=3, type=float, default=[1.0, 1.0, 1.0]
        )
    if family == "nd":
        p.add_argument("--dimension", type=int, default=16)
        p.add_argument("--target-seed", type=int, default=16)
        p.add_argument("--beta", type=float, default=0.9)
        p.add_argument("--gaussian-kind", choices=("vine", "ar1"), default="vine")
        p.add_argument("--rho", type=float, default=0.5)
        p.add_argument("--mean", type=float, default=0.5)
    a = p.parse_args()
    defaults = dict(
        n_train=4096 if family == "1d" else 1024,
        n_test=512,
        n_eval=2048,
        probe_size=256,
        steps=1000,
        batch_size=128,
        width=64,
        depth=3,
        eval_every=100,
        ode_steps=32,
        learning_rate=0.001 if family == "nd" else 0.002,
        weight_decay=0.0001 if family == "nd" else 0.0,
        schedule="constant" if family == "nd" else "cosine",
        activation="elu" if family == "nd" else "silu",
        residual=family != "nd",
    )
    if a.profile == "smoke":
        defaults.update(
            n_train=128,
            n_test=96,
            n_eval=256,
            probe_size=32,
            steps=12,
            batch_size=16,
            width=16,
            depth=2,
            eval_every=6,
            ode_steps=8,
        )
    if a.profile == "full":
        defaults.update(
            n_train=5000 if family == "nd" else 100000,
            n_test=2000 if family == "nd" else 4096,
            n_eval={"1d": 20000, "2d": 100000, "nd": 10000}[family],
            probe_size=512 if family == "nd" else 2048,
            steps=200000 if family == "nd" else 10000,
            batch_size=256,
            width={"1d": 128, "2d": 200, "nd": 64}[family],
            depth=4 if family == "2d" else 3,
            eval_every=2000 if family == "nd" else 100,
            ode_steps=64,
        )
    for name, value in defaults.items():
        if getattr(a, name) is None:
            setattr(a, name, value)
    a.family = family
    a.dimension = getattr(a, "dimension", 1 if family == "1d" else 2)
    a.data_seed = a.seed if a.data_seed is None else a.data_seed
    positive = (
        "n_train",
        "n_test",
        "n_eval",
        "probe_size",
        "steps",
        "batch_size",
        "width",
        "depth",
        "eval_every",
        "ode_steps",
        "metric_samples",
        "projections",
    )
    if (
        any(getattr(a, name) < 1 for name in positive)
        or min(a.n_test, a.n_eval, a.metric_samples) < 2
    ):
        p.error(
            "Sizes/steps must be positive; test/evaluation/metric sizes must be >= 2"
        )
    numeric = [
        a.learning_rate,
        a.weight_decay,
        a.clip_norm,
        a.diagonal_weight,
        a.max_cost_gib,
        a.final_lr_ratio,
    ]
    if (
        not np.isfinite(numeric).all()
        or a.learning_rate <= 0
        or min(a.weight_decay, a.clip_norm) < 0
        or not 0 < a.diagonal_weight < 1
        or a.max_cost_gib <= 0
        or not 0 <= a.final_lr_ratio <= 1
    ):
        p.error("Invalid optimizer, clipping, diagonal-weight or memory settings")
    if family == "nd" and (
        not np.isfinite([a.beta, a.rho, a.mean]).all() or a.beta <= 0 or abs(a.rho) >= 1
    ):
        p.error("beta must be positive and abs(rho) < 1")
    if len(set(a.methods)) != len(a.methods) or len(set(a.datasets)) != len(a.datasets):
        p.error("Datasets and methods must be unique")
    return a


def make_data(a, target):
    rng = lambda stream: np.random.default_rng(
        np.random.SeedSequence([a.data_seed, stream])
    )
    normal = lambda stream, n: (
        rng(stream).normal(size=(n, a.dimension)).astype(np.float32)
    )
    sample = lambda stream, n: (
        target.sample(rng(stream), n, dtype=np.float64)
        if a.family == "2d"
        else target.sample(rng(stream), n)
    )
    z, x = normal(3, a.n_train), sample(4, a.n_train)
    z_test, x_test = normal(5, a.n_test), sample(11, a.n_test)
    reference = (
        target.sample(rng(7), a.n_eval, iid=True, dtype=np.float64)
        if a.family == "2d"
        else target.sample(rng(7), a.n_eval)
    )
    reference2 = (
        target.sample(rng(10), a.n_eval, iid=True, dtype=np.float64)
        if a.family == "2d"
        else target.sample(rng(10), a.n_eval)
    )
    mean, scale = (
        (x.astype(float).mean(0), x.astype(float).std(0))
        if a.family == "2d"
        else (np.zeros(a.dimension), np.ones(a.dimension))
    )
    if np.any(scale == 0):
        raise ValueError("Training target has zero variance")
    x, x_test, reference, reference2 = [
        ((v - mean) / scale).astype(np.float32)
        for v in (x, x_test, reference, reference2)
    ]
    needs_pairs = any(m in ("lf", "fm_paired") for m in a.methods) or (
        a.coupling == "paired" and any(m in MAPS for m in a.methods)
    )
    pairing = (
        lf.pair_indices(z, x, metric="squared_euclidean", max_cost_gib=a.max_cost_gib)
        if needs_pairs
        else np.arange(len(z))
    )
    test_pairing = lf.pair_indices(
        z_test, x_test, metric="squared_euclidean", max_cost_gib=a.max_cost_gib
    )
    endpoint = x_test[test_pairing]
    return dict(
        train_source=z,
        train_target=x,
        train_pairing=pairing,
        test_source=z_test,
        test_target=x_test,
        test_pairing=test_pairing,
        endpoint_reference=endpoint,
        source=normal(6, a.n_eval),
        reference=reference,
        reference2=reference2,
        target_mean=mean,
        target_scale=scale,
    )


def objective(model, z, x, r, s, t, method, a):
    if method.startswith("lf"):
        return lf.endpoint_loss(model, z, x, power=1 if a.loss == "l1" else 2)
    if method.startswith("fm"):
        return fm.loss(model, z, x, r)
    return fmap.loss(
        model, z, x, t, s, method.upper(), diagonal_weight=a.diagonal_weight, r=r
    )


@eqx.filter_jit
def predict(model, z, method, steps, euler=False):
    if method.startswith("lf"):
        return lf.generate(model, z)
    if method.startswith("fm"):
        return fm.generate(model, z) if euler else fm.sample(model, z, steps=steps)
    return fmap.generate(model, z)


def generate(model, z, method, a, euler=False):
    return np.concatenate(
        [
            np.asarray(
                predict(model, jnp.asarray(z[i : i + 1024]), method, a.ode_steps, euler)
            )
            for i in range(0, len(z), 1024)
        ]
    )


def fit(a, data, method):
    kind = (
        "lf" if method.startswith("lf") else "fm" if method.startswith("fm") else "fmap"
    )
    model = MLP(
        a.dimension,
        a.width,
        a.depth,
        kind,
        key=jax.random.key(a.seed),
        activation=a.activation,
        residual=a.residual,
    )
    rate = (
        optax.cosine_decay_schedule(a.learning_rate, a.steps, alpha=a.final_lr_ratio)
        if a.schedule == "cosine"
        else a.learning_rate
    )
    optimizer = (
        optax.adamw(rate, weight_decay=a.weight_decay)
        if a.weight_decay
        else optax.adam(rate)
    )
    if a.clip_norm:
        optimizer = optax.chain(optax.clip_by_global_norm(a.clip_norm), optimizer)
    parameters, static = eqx.partition(model, eqx.is_inexact_array)
    state = optimizer.init(parameters)
    paired = method in ("lf", "fm_paired") or (
        method in MAPS and a.coupling == "paired"
    )
    z = jnp.asarray(data["train_source"])
    x = jnp.asarray(
        data["train_target"][data["train_pairing"]] if paired else data["train_target"]
    )
    independent = method == "fm" or (method in MAPS and a.coupling == "independent")

    @eqx.filter_jit
    def train_block(parameters, state, key, count):
        def step(carry, unused):
            parameters, state, key = carry
            key, ki, kj, kr, kst = jax.random.split(key, 5)
            i = jax.random.randint(ki, (a.batch_size,), 0, len(z))
            j = jax.random.randint(kj, (a.batch_size,), 0, len(x))
            r = jax.random.uniform(kr, (a.batch_size,))
            st = jnp.sort(jax.random.uniform(kst, (a.batch_size, 2)), axis=1)
            loss_fn = lambda params: objective(
                eqx.combine(params, static),
                z[i],
                x[j if independent else i],
                r,
                st[:, 0],
                st[:, 1],
                method,
                a,
            )
            value, grads = jax.value_and_grad(loss_fn)(parameters)
            updates, state = optimizer.update(grads, state, parameters)
            return (eqx.apply_updates(parameters, updates), state, key), value

        return jax.lax.scan(step, (parameters, state, key), None, length=count)

    probe_count = min(a.probe_size, a.n_test)
    probe_z = data["test_source"][:probe_count]
    probe_x = (
        data["test_target"][data["test_pairing"]][:probe_count]
        if paired
        else data["test_target"][:probe_count]
    )
    random = np.random.default_rng(np.random.SeedSequence([a.data_seed, 100]))
    r, st = (
        jnp.asarray(random.random(probe_count), dtype=jnp.float32),
        jnp.asarray(
            np.sort(random.random((probe_count, 2)), axis=1), dtype=jnp.float32
        ),
    )
    evaluate_native = eqx.filter_jit(
        lambda m: objective(
            m,
            jnp.asarray(probe_z),
            jnp.asarray(probe_x),
            r,
            st[:, 0],
            st[:, 1],
            method,
            a,
        )
    )
    history = dict(step=[], native_loss=[], endpoint_mse=[], train_loss=[])
    key, completed = jax.random.key(a.seed + 101), 0
    start = time.perf_counter()
    while True:
        output = generate(model, probe_z, method, a)
        native = float(evaluate_native(model))
        mse = float(np.mean((output - data["endpoint_reference"][:probe_count]) ** 2))
        if not np.isfinite([native, mse]).all():
            raise FloatingPointError(
                f"Nonfinite {method} objective or samples at step {completed}"
            )
        history["step"].append(completed)
        history["native_loss"].append(native)
        history["endpoint_mse"].append(mse)
        print(
            f"  {method:11s} {completed:7d}/{a.steps}: native={native:.5g}, endpoint MSE={mse:.5g}",
            flush=True,
        )
        if completed == a.steps:
            break
        count = min(a.eval_every, a.steps - completed)
        (parameters, state, key), losses = train_block(parameters, state, key, count)
        model = eqx.combine(parameters, static)
        history["train_loss"].extend(np.asarray(losses).tolist())
        completed += count
    return model, history, time.perf_counter() - start


def run(a, target):
    directory = a.output / target.name / f"d{a.dimension}" / f"seed_{a.seed}"
    directory.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config["dataset"] = target.name
    config["endpoint_diagnostic"] = "independent held-out exact pairing"
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"{target.name}: {directory}", flush=True)
    data = make_data(a, target)
    saved = dict(data)
    if hasattr(target, "covariance"):
        saved.update(
            population_mean=target.mean, population_covariance=target.covariance
        )
    if a.family == "1d" or (a.family == "nd" and target.name == "gaussian"):
        saved["analytic"] = target.transport(data["source"])
    metric = lambda samples: distribution_metrics(
        samples,
        data["reference"],
        seed=a.data_seed,
        projections=a.projections,
        max_samples=a.metric_samples,
    )
    metrics, history = {"reference_baseline": metric(data["reference2"])}, {}
    for method in a.methods:
        model, record, seconds = fit(a, data, method)
        history.update({f"{method}_{name}": values for name, values in record.items()})
        eqx.tree_serialise_leaves(directory / f"{method}.eqx", model)
        for euler in [False, True] if method.startswith("fm") else [False]:
            name = method + "_euler" if euler else method
            started = time.perf_counter()
            saved[name] = generate(model, data["source"], method, a, euler)
            generation_seconds = time.perf_counter() - started
            metrics[name] = metric(saved[name])
            metrics[name].update(
                training_seconds=seconds,
                generation_seconds=generation_seconds,
                nfe=1 if euler or not method.startswith("fm") else 2 * a.ode_steps,
            )
            if "analytic" in saved:
                metrics[name]["analytic_map_mse"] = float(
                    np.mean((saved[name] - saved["analytic"]) ** 2)
                )
            if a.family == "1d":
                metrics[name]["known_cdf_ks"] = float(
                    kstest(saved[name][:, 0], target.cdf).statistic
                )
        np.savez_compressed(directory / "history.npz", **history)
        np.savez_compressed(directory / "samples.npz", **saved)
        (directory / "metrics.json").write_text(
            json.dumps(metrics, indent=2, allow_nan=False) + "\n"
        )
    from plots.common import plot_run

    plot_run(directory)
    return directory
