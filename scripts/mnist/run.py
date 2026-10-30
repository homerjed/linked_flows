"""MNIST endpoint regression and flow matching: python -m scripts.mnist.run --help."""

import argparse
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from data.mnist import load, normalize, split_indices, evaluation_indices
import fm
import lf
from metrics import distribution_metrics
from models import UNet
from plots.mnist import save as save_plots


PROFILES = {
    "smoke": dict(n_total=128, n_eval=128, epochs=3, base_channels=8, batch_size=16),
    "quick": dict(
        n_total=2048, n_eval=512, epochs=100, base_channels=32, batch_size=64
    ),
    "full": dict(
        n_total=60000, n_eval=1024, epochs=200, base_channels=32, batch_size=64
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("lf", "lf_unpaired", "fm"), default="lf")
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument(
        "--repair-every",
        type=int,
        default=0,
        help="Fresh LF noise and matching every N epochs; 0 keeps fixed pairs",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--steps", type=int, default=64, help="FM Heun integration steps"
    )
    parser.add_argument(
        "--metric", choices=("euclidean", "sqeuclidean"), default="euclidean"
    )
    parser.add_argument(
        "--max-cost-gib",
        type=float,
        default=32,
        help="Exact assignment matrix memory cap; solver uses additional memory",
    )
    parser.add_argument("--data-dir", default="data/mnist/cache")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--out", type=Path)
    for name in PROFILES["smoke"]:
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    args = parser.parse_args()
    for name, value in PROFILES[args.profile].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if (
        min(args.n_total, args.n_eval, args.epochs, args.batch_size, args.steps) < 1
        or args.seed < 0
        or args.repair_every < 0
    ):
        parser.error(
            "Sizes/epochs/steps must be positive; seed and repair interval must be nonnegative"
        )
    if args.method == "fm" and args.repair_every:
        parser.error(
            "FM draws fresh independent noise each minibatch and does not use --repair-every"
        )
    if args.n_eval < (20 if args.conditional else 2):
        parser.error(
            "Evaluation needs at least two samples per requested class (20 with --conditional)"
        )
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    run_dir = (
        args.out
        or Path("runs/mnist")
        / f"{args.method}_{'conditional_' if args.conditional else ''}{args.profile}_repair{args.repair_every}_seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "config.json").exists():
        parser.error(f"Run already exists: {run_dir}; choose another --out")
    config = dict(vars(args), out=str(run_dir))
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    images, labels, test_images, test_labels = load(
        args.data_dir, download=not args.no_download
    )
    train_ids, validation_ids = split_indices(
        labels, args.n_total, args.seed, args.conditional
    )
    reference_ids, repeat_ids = evaluation_indices(
        test_labels, args.n_eval, args.seed, args.conditional
    )
    train, validation = images[train_ids], images[validation_ids]
    train_labels, validation_labels = labels[train_ids], labels[validation_ids]
    np.savez(
        run_dir / "split.npz",
        train=train_ids,
        validation=validation_ids,
        reference=reference_ids,
        repeat=repeat_ids,
        train_labels=train_labels,
        validation_labels=validation_labels,
    )
    flow = args.method == "fm"
    model = UNet(
        jax.random.key(args.seed + 2000), args.base_channels, args.conditional, flow
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(args.learning_rate)
    )
    state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    def objective(model, z, x, t, y):
        return (
            fm.loss(model, z, x, t, labels=y)
            if flow
            else lf.endpoint_loss(model, z, x, labels=y)
        )

    @eqx.filter_jit
    def update(model, state, z, x, t, y):
        loss, grads = eqx.filter_value_and_grad(objective)(model, z, x, t, y)
        updates, state = optimizer.update(
            grads, state, eqx.filter(model, eqx.is_inexact_array)
        )
        return eqx.apply_updates(model, updates), state, loss

    evaluate_batch = eqx.filter_jit(objective)

    def evaluate(model, z, images, p, times, labels):
        total = 0.0
        for start in range(0, len(z), args.batch_size):
            rows = slice(start, start + args.batch_size)
            y = jnp.asarray(labels[rows]) if args.conditional else None
            total += len(z[rows]) * float(
                evaluate_batch(
                    model,
                    jnp.asarray(z[rows]),
                    jnp.asarray(normalize(images[p[rows]])),
                    jnp.asarray(times[rows]),
                    y,
                )
            )
        return total / len(z)

    def prepare(images, labels, seed):
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, 0x464D11]) if flow else seed
        )
        z = rng.standard_normal(images.shape, dtype=np.float32)
        p = np.arange(len(images))
        if args.method == "lf":
            p = lf.pair_indices(
                z.reshape(len(z), -1),
                normalize(images).reshape(len(z), -1),
                metric=args.metric,
                labels=labels if args.conditional else None,
                max_cost_gib=args.max_cost_gib,
            )
        elif args.method == "lf_unpaired":
            groups = (
                [np.flatnonzero(labels == c) for c in range(10)]
                if args.conditional
                else [p.copy()]
            )
            for rows in groups:
                p[rows] = rng.permutation(rows)
        time_rng = (
            np.random.default_rng(np.random.SeedSequence([seed, 0x464D12]))
            if flow
            else rng
        )
        return z, p, time_rng.random(len(z), dtype=np.float32)

    z_train, p_train, t_train = prepare(train, train_labels, args.seed + 1000)
    z_validation, p_validation, t_validation = prepare(
        validation, validation_labels, args.seed + 1500
    )
    events = [{"epoch": 1, "noise_seed": args.seed + 1000}]
    history = [
        {
            "epoch": 0,
            "train_loss": evaluate(
                model, z_train, train, p_train, t_train, train_labels
            ),
            "validation_loss": evaluate(
                model,
                z_validation,
                validation,
                p_validation,
                t_validation,
                validation_labels,
            ),
        }
    ]
    for epoch in range(args.epochs):
        started = time.perf_counter()
        if args.repair_every and epoch and epoch % args.repair_every == 0:
            seed = int(
                np.random.SeedSequence(
                    [args.seed, 1000, epoch // args.repair_every]
                ).generate_state(1)[0]
            )
            z_train, p_train, t_train = prepare(train, train_labels, seed)
            events.append(
                {
                    "epoch": epoch + 1,
                    "noise_seed": seed,
                    "before_update_loss": evaluate(
                        model, z_train, train, p_train, t_train, train_labels
                    ),
                }
            )
        order = np.random.default_rng(
            np.random.SeedSequence([args.seed + 3000, epoch])
        ).permutation(len(train))
        noise_rng = np.random.default_rng(
            np.random.SeedSequence([args.seed + 3000, epoch, 0x464D01])
        )
        time_rng = np.random.default_rng(
            np.random.SeedSequence([args.seed + 3000, epoch, 0x464D02])
        )
        online_loss = 0.0
        for start in range(0, len(train), args.batch_size):
            rows = order[start : start + args.batch_size]
            z = (
                noise_rng.standard_normal((len(rows), 1, 28, 28), dtype=np.float32)
                if flow
                else z_train[rows]
            )
            t = time_rng.random(len(rows), dtype=np.float32) if flow else t_train[rows]
            target_rows = rows if flow else p_train[rows]
            y = jnp.asarray(train_labels[rows]) if args.conditional else None
            model, state, loss = update(
                model,
                state,
                jnp.asarray(z),
                jnp.asarray(normalize(train[target_rows])),
                jnp.asarray(t),
                y,
            )
            loss = float(loss)
            if not np.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss in epoch {epoch + 1}")
            online_loss += len(rows) * loss
        record = {
            "epoch": epoch + 1,
            "online_loss": online_loss / len(train),
            "train_loss": evaluate(
                model, z_train, train, p_train, t_train, train_labels
            ),
            "validation_loss": evaluate(
                model,
                z_validation,
                validation,
                p_validation,
                t_validation,
                validation_labels,
            ),
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    eqx.tree_serialise_leaves(run_dir / "model.eqx", model)
    (run_dir / "pairing_events.json").write_text(json.dumps(events, indent=2) + "\n")
    np.savez(
        run_dir / "pairs.npz",
        train_noise=z_train,
        train_assignment=p_train,
        train_times=t_train,
        validation_noise=z_validation,
        validation_assignment=p_validation,
        validation_times=t_validation,
    )

    @eqx.filter_jit
    def predict(model, z, y):
        if flow:
            return fm.sample(model, z, steps=args.steps, labels=y)
        return (
            jax.vmap(model)(z)
            if y is None
            else jax.vmap(lambda x, label: model(x, label=label))(z, y)
        )

    def predict_batches(z, labels):
        return np.concatenate(
            [
                np.asarray(
                    predict(
                        model,
                        jnp.asarray(z[i : i + args.batch_size]),
                        jnp.asarray(labels[i : i + args.batch_size])
                        if args.conditional
                        else None,
                    )
                )
                for i in range(0, len(z), args.batch_size)
            ]
        )

    fresh_noise = np.random.default_rng(args.seed + 4000).standard_normal(
        (args.n_eval, 1, 28, 28), dtype=np.float32
    )
    evaluation_labels = test_labels[reference_ids]
    samples = predict_batches(fresh_noise, evaluation_labels)
    reference, repeat = (
        normalize(test_images[reference_ids]),
        normalize(test_images[repeat_ids]),
    )
    metrics = {
        "generated_vs_test": distribution_metrics(samples, reference, seed=args.seed),
        "test_vs_test": distribution_metrics(reference, repeat, seed=args.seed),
        "sample_min": float(samples.min()),
        "sample_max": float(samples.max()),
        "fraction_outside_image_range": float(np.mean(np.abs(samples) > 1)),
    }
    if args.conditional:
        metrics["per_class"] = {
            str(c): distribution_metrics(
                samples[evaluation_labels == c],
                reference[evaluation_labels == c],
                seed=args.seed,
            )
            for c in range(10)
        }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    )
    arrays = dict(
        samples=samples, reference=reference, repeat=repeat, noise=fresh_noise
    )
    if args.conditional:
        arrays["labels"] = evaluation_labels
    if not flow:
        arrays["validation_targets"] = normalize(validation[p_validation[:64]])
        arrays["validation_predictions"] = predict_batches(
            z_validation[:64], validation_labels[:64]
        )
    np.savez_compressed(run_dir / "evaluation.npz", **arrays)
    save_plots(
        run_dir,
        history,
        samples,
        reference,
        evaluation_labels if args.conditional else None,
        arrays.get("validation_targets"),
        arrays.get("validation_predictions"),
    )
    print(f"Saved {run_dir}", flush=True)


if __name__ == "__main__":
    main()
