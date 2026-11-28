"""Original NB2 and NB3 (reproduce/results) against NB2 and NB3 with a shared hidden layer
(improvement/shared_committee/results): mean cumulative regret over the seeds, stationary stream (left) and
non-stationary stream (right).

    python plot.py [results_dir] [output.png]    # results_dir holds the shared runs
"""
import glob
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run import EVERY, HERE

original = os.path.join(os.path.dirname(os.path.dirname(HERE)), "reproduce", "results")
results = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "shared_hidden.png")

LINES = [  # (results dir, algorithm, label, style): solid = original, dashed = shared hidden layer
    (original, "nb2", "NB2", dict(color="black", ls="-")),
    (original, "nb3", "NB3", dict(color="gray", ls="-")),
    (results, "nb2_shared", "NB2 shared", dict(color="black", ls="--")),
    (results, "nb3_shared", "NB3 shared", dict(color="gray", ls="--")),
]

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, stream in zip(axes, ("stat", "nonstat")):
    for folder, name, label, style in LINES:
        files = glob.glob(os.path.join(folder, f"{name}_{stream}_seed*.npy"))
        if files:
            curve = np.mean([np.load(f) for f in files], axis=0)
            ax.plot(np.arange(len(curve)) * EVERY, curve, label=label, **style)
    ax.set_title("Stationary" if stream == "stat" else "Non-stationary")
    ax.set_xlabel("Time step")
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("Cumulated regret")
axes[1].legend()
fig.tight_layout()
fig.savefig(out, dpi=150)
print("saved", out)
