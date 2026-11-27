"""Issue #4 in isolation: original architecture/loss/exploration, AdamW updates."""
import numpy as np

from reproduce import bandits as baseline
from improvement.adamw import AdamW

ADAMW_LRS = (1e-3, 3e-3, 1e-2)


class AdamWCommittee(baseline.Committee):
    def __init__(self, K, D, rng, lrs=ADAMW_LRS, weight_decay=1e-4):
        lrs = np.asarray(lrs, dtype=float)
        if lrs.shape != (len(baseline.LRS),) or not np.all(np.isfinite(lrs) & (lrs > 0)):
            raise ValueError("Supply three finite, positive AdamW learning rates")
        super().__init__(K, D, rng)
        self.unit_lr = np.repeat(np.tile(lrs, len(baseline.SIZES)),
                                 [c for c, _ in baseline.MODELS])
        self.hidden_optimizers = [AdamW(self.W1[:, k], weight_decay) for k in range(K)]
        self.head_optimizers = [AdamW(self.v[k], weight_decay) for k in range(K)]
        self.hidden_gradient = np.zeros_like(self.W1[:, 0])

    def update(self, x, k, r, weight):
        o, h = self.out[k], self.h[k]
        # Keep the baseline half-squared loss. These are descent gradients.
        d_out = ((o - r) * o * (1 - o))[self.unit_model] * weight
        d_hidden = d_out * self.v[k] * h * (1 - h)
        grad_head = d_out * h
        self.hidden_gradient.fill(0)
        self.hidden_gradient[x] = d_hidden
        # Within the selected network, inactive input rows have zero gradients,
        # but still receive Adam's historical momentum and decoupled decay.
        self.hidden_optimizers[k].step(self.W1[:, k], self.hidden_gradient, self.unit_lr)
        self.head_optimizers[k].step(self.v[k], grad_head, self.unit_lr)


class AdamWNeuralBandit(baseline.NeuralBandit):
    def __init__(self, K, D, rng, per_action, gamma=0.005, gamma_model=0.1,
                 lrs=ADAMW_LRS, weight_decay=1e-4):
        self.K, self.rng, self.gamma = K, rng, gamma
        self.net = AdamWCommittee(K, D, rng, lrs, weight_decay)
        self.exp3 = baseline.EXP3(K if per_action else 1, len(baseline.MODELS),
                                  gamma_model, rng)
