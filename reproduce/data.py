"""Forest Cover Type as in Sec. 5 of the paper: 94 binary features, 7 classes (= actions).

Each of the 10 continuous variables is cut into 5 equal-frequency bins and one-hot encoded (50 features),
followed by the 4 wilderness and 40 soil indicators. Every row has exactly 12 active features, so a context
is stored as the indices of its 12 ones (x @ W == W[idx].sum(0)).
"""
import os
import urllib.request

import numpy as np
import pandas as pd

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
D, K = 94, 7


def _build():
    raw = os.path.join(DIR, "covtype.data.gz")
    if not os.path.exists(raw):
        os.makedirs(DIR, exist_ok=True)
        urllib.request.urlretrieve(URL, raw)
    df = pd.read_csv(raw, header=None).values
    cont = [5 * j + pd.qcut(df[:, j], 5, labels=False) for j in range(10)]
    wild, soil = df[:, 10:14], df[:, 14:54]
    assert (wild.sum(1) == 1).all() and (soil.sum(1) == 1).all()
    idx = np.column_stack(cont + [50 + wild.argmax(1), 54 + soil.argmax(1)]).astype(np.int16)
    return idx, (df[:, 54] - 1).astype(np.int8)


def load(shuffle_seed=0):
    """Return (idx, y): idx is (581012, 12) active-feature indices, y is the class in 0..6.
    Rows are shuffled once with a fixed seed, as the paper uses one shuffled copy of the data."""
    cache = os.path.join(DIR, "covtype94.npz")
    if not os.path.exists(cache):
        idx, y = _build()
        np.savez(cache, idx=idx, y=y)
    f = np.load(cache)
    perm = np.random.default_rng(shuffle_seed).permutation(len(f["y"]))
    return f["idx"][perm], f["y"][perm]
