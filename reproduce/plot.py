"""Figure 1: mean cumulative regret over the seeds, stationary stream (left) and non-stationary stream (right).

    python plot.py [results_dir] [output.png]
"""
import glob
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run import EVERY, HERE

STYLE = {  # line styles follow the paper's Figure 1
    "random": ("Random", dict(color="gray", ls=":")),
    "linucb": ("LinUCB", dict(color="dimgray", ls="--")),
    "nb2": ("NB2", dict(color="black", ls="-")),
    "nb3": ("NB3", dict(color="gray", ls="-")),
}

results = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "figure1.png")

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, stream in zip(axes, ("stat", "nonstat")):
    for name, (label, style) in STYLE.items():
        files = glob.glob(os.path.join(results, f"{name}_{stream}_seed*.npy"))
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
