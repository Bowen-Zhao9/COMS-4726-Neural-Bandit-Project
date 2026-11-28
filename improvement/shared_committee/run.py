"""Play NB2 and NB3 with a shared hidden layer (shared_hidden.py) on the same Forest Cover Type streams as
reproduce/run.py and save their cumulative regret every EVERY steps.

    python run.py                                  # full run: 2 algorithms x 2 streams x 10 seeds, T = 2.5M
    python run.py --T 50000 --seeds 2 --out DIR    # smoke test
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one BLAS thread per process: runs are parallelised across processes

import argparse
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(HERE)), "reproduce"))
import data  # noqa: E402  (reproduce/data.py)
from shared_hidden import SharedNeuralBandit  # noqa: E402

# Same stream and regret as reproduce/run.py
OPT = 0.93  # accuracy of the offline policy the regret is measured against (Sec. 5)
DRIFT = 500_000  # non-stationary stream: every DRIFT steps each class k becomes k + 1 (mod 7)
EVERY = 10_000  # record the regret curve every EVERY steps
ALGOS = ("nb2_shared", "nb3_shared")


def make(name, rng):
    return SharedNeuralBandit(data.K, data.D, rng, per_action=(name == "nb3_shared"))


def run(name, drift, seed, T, out):
    """One run: start at a random position of the shuffled data and loop over it for T steps."""
    idx, y = data.load()
    rng = np.random.default_rng(seed)
    algo, start = make(name, rng), rng.integers(len(y))
    curve, total, t0 = np.zeros(T // EVERY + 1), 0, time.time()
    for t in range(T):
        i = (start + t) % len(y)
        label = (y[i] + t // DRIFT) % data.K if drift else y[i]
        k = algo.select(idx[i])
        r = int(k == label)
        algo.update(idx[i], k, r)
        total += r
        if (t + 1) % EVERY == 0:
            curve[(t + 1) // EVERY] = OPT * (t + 1) - total
    np.save(os.path.join(out, f"{name}_{'nonstat' if drift else 'stat'}_seed{seed}.npy"), curve)
    return name, drift, seed, time.time() - t0


def job(args):
    return run(*args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=2_500_000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--algos", nargs="+", choices=ALGOS, default=list(ALGOS))
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    data.load()  # download and preprocess once, before the workers start
    os.makedirs(a.out, exist_ok=True)
    jobs = [(n, d, s, a.T, a.out) for n in a.algos for d in (False, True) for s in range(a.seeds)]
    with Pool(a.jobs) as pool:
        for n, d, s, sec in pool.imap_unordered(job, jobs):
            print(f"{n:10s} {'nonstat' if d else 'stat':7s} seed {s}: {sec / 60:.1f} min", flush=True)
