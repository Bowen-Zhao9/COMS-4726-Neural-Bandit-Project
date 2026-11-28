"""NeuralBandit2 and NeuralBandit3 whose NeuralBandit1 models share one hidden layer across the actions (issue #2).

In the paper (reproduce/bandits.py) a model with C hidden units has one MLP per action: D -> C sigmoid units ->
1 sigmoid output, i.e. K (D C + C) weights. Here the same model is a single MLP: D -> K C sigmoid units shared by
all actions -> K sigmoid outputs, one per action, i.e. D K C + K C K weights. With D = 94 and K = 7 that is
707 C against 665 C, so both versions have about the same number of weights and the same number of hidden units.

Everything else follows reproduce/bandits.py: no biases, weights from U(-0.5, 0.5), quadratic loss on the
played action's output only, SGD with the model's lambda, no 1/P weight, and the same EXP3 committees.
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(REPO, "reproduce"))
import bandits  # noqa: E402  (reproduce/bandits.py)
from bandits import EXP3, MODELS, sigmoid  # noqa: E402


class SharedCommittee:
    """For each model (C, lambda) in MODELS, one MLP shared by all actions: one sigmoid hidden layer of K C units,
    one sigmoid output per action, no biases, weights drawn from U(-0.5, 0.5).
    The hidden units of all models sit side by side: W1 is (D, S) and v is (K, S), S = K * sum of the C's."""

    def __init__(self, K, D, rng):
        sizes = K * np.array([c for c, _ in MODELS])
        self.start = np.concatenate([[0], np.cumsum(sizes)[:-1]])  # first hidden unit of each model
        self.unit_model = np.repeat(np.arange(len(MODELS)), sizes)  # model owning each hidden unit
        self.unit_lr = np.repeat([lr for _, lr in MODELS], sizes)  # its learning step lambda
        self.W1 = rng.uniform(-0.5, 0.5, (D, sizes.sum()))
        self.v = rng.uniform(-0.5, 0.5, (K, sizes.sum()))

    def forward(self, x):
        """Outputs of all networks, shape (K, M)."""
        self.h = sigmoid(self.W1[x].sum(0))  # (S,), the same hidden layer for every action
        self.out = sigmoid(np.add.reduceat(self.h * self.v, self.start, axis=1))
        return self.out

    def update(self, x, k, r, weight):
        """One backprop step on (r - out)^2 / 2 for action k's output in every model, scaled by weight.
        Only action k's output weights change, but the shared hidden layer also moves the other actions' outputs."""
        o, h = self.out[k], self.h
        d_out = ((r - o) * o * (1 - o))[self.unit_model]  # output delta, repeated for each hidden unit
        d_hid = d_out * self.v[k] * h * (1 - h)
        step = self.unit_lr * weight
        self.v[k] += step * d_out * h
        self.W1[x] += step * d_hid


class SharedNeuralBandit(bandits.NeuralBandit):
    """bandits.NeuralBandit (NB2 if per_action=False, NB3 if per_action=True) built on SharedCommittee."""

    def __init__(self, K, D, rng, per_action, gamma=0.005, gamma_model=0.1):
        self.K, self.rng, self.gamma = K, rng, gamma
        self.net = SharedCommittee(K, D, rng)
        self.exp3 = EXP3(K if per_action else 1, len(MODELS), gamma_model, rng)
