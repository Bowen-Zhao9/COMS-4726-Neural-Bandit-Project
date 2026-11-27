"""Run from repository root with python -m improvement.run_adamw --help."""
import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name, "1")

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
from time import perf_counter

import numpy as np

from reproduce import bandits, data
from improvement.adamw_bandit import ADAMW_LRS, AdamWNeuralBandit

OPT, DRIFT, EVERY = 0.93, 500_000, 10_000


def run(task):
    args, name, drift, seed = task
    idx, labels = data.load()
    rng = np.random.default_rng(seed)
    if args.optimizer == "sgd":
        algo = bandits.NeuralBandit(data.K, data.D, rng, per_action=(name == "nb3"))
    else:
        algo = AdamWNeuralBandit(data.K, data.D, rng, per_action=(name == "nb3"),
                                lrs=args.lrs, weight_decay=args.weight_decay)
    # Identical architecture and random draws preserve the baseline start position.
    start = int(rng.integers(len(labels)))
    curve = np.zeros(args.T // EVERY + 1)
    total = 0
    before = perf_counter()
    for t in range(args.T):
        i = (start + t) % len(labels)
        label = (int(labels[i]) + (t // DRIFT if drift else 0)) % data.K
        action = algo.select(idx[i])
        reward = int(action == label)
        algo.update(idx[i], action, reward)
        total += reward
        if (t + 1) % EVERY == 0:
            curve[(t + 1) // EVERY] = OPT * (t + 1) - total
    seconds = perf_counter() - before
    stem = f"{name}_{'nonstat' if drift else 'stat'}_seed{seed}"
    np.save(Path(args.out) / f"{stem}.npy", curve)
    summary = dict(algorithm=name, drift=drift, seed=seed, start=start,
                   steps=args.T, reward=total, final_regret=OPT * args.T - total,
                   loop_seconds=seconds)
    (Path(args.out) / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default="adamw")
    parser.add_argument("--lrs", nargs=3, type=float, default=ADAMW_LRS)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--T", type=int, default=2_500_000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--algos", nargs="+", choices=("nb2", "nb3"), default=["nb2", "nb3"])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.T <= 0 or args.T % EVERY:
        parser.error("T must be a positive multiple of 10000")
    if args.seeds <= 0 or args.jobs <= 0 or args.seed_offset < 0:
        parser.error("seeds/jobs must be positive and seed-offset nonnegative")
    if not all(np.isfinite(lr) and lr > 0 for lr in args.lrs):
        parser.error("learning rates must be finite and positive")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight-decay must be finite and nonnegative")
    # Never overwrite an earlier experiment. Use a fresh output directory.
    Path(args.out).mkdir(parents=True, exist_ok=False)
    settings = dict(vars(args), baseline_commit="5a4c5638722d3d225d2844a2db00730cb06000ca",
                    effective_lrs=list(bandits.LRS if args.optimizer == "sgd" else args.lrs),
                    effective_weight_decay=0 if args.optimizer == "sgd" else args.weight_decay,
                    drift_interval=DRIFT, record_every=EVERY, reference_accuracy=OPT)
    (Path(args.out) / "config.json").write_text(json.dumps(settings, indent=2))
    data.load()  # Build the data cache once before starting workers.
    tasks = [(args, name, drift, seed) for name in args.algos for drift in (False, True)
             for seed in range(args.seed_offset, args.seed_offset + args.seeds)]
    with Pool(args.jobs) as pool:
        for summary in pool.imap_unordered(run, tasks):
            print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
