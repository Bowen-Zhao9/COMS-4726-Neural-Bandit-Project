"""NumPy AdamW. Use a separate instance for each independently updated tensor."""
import numpy as np


class AdamW:
    def __init__(self, parameter, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8):
        if not np.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if len(betas) != 2 or not all(0 <= b < 1 for b in betas):
            raise ValueError("betas must lie in [0, 1)")
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        self.m = np.zeros_like(parameter)
        self.v = np.zeros_like(parameter)
        self.t = 0
        self.beta1, self.beta2 = betas
        self.weight_decay, self.eps = weight_decay, eps

    def step(self, parameter, gradient, lr):
        """Apply a loss gradient in place; lr may broadcast across model columns."""
        if parameter.shape != self.m.shape or gradient.shape != parameter.shape:
            raise ValueError("parameter and gradient shapes must match optimizer state")
        self.t += 1
        self.m *= self.beta1
        self.m += (1 - self.beta1) * gradient
        self.v *= self.beta2
        self.v += (1 - self.beta2) * gradient * gradient
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        parameter *= 1 - lr * self.weight_decay
        parameter -= lr * m_hat / (np.sqrt(v_hat) + self.eps)
