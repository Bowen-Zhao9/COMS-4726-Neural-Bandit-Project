"""Random, LinUCB, NeuralBandit2 and NeuralBandit3 (Allesiardo et al. 2014).

Every algorithm has select(x) -> action and update(x, action, reward), where the context x is the
array of indices of its active binary features (see data.py).
"""
import numpy as np

SIZES, LRS = (1, 5, 25, 50, 100), (0.01, 0.1, 1.0)
MODELS = [(c, lr) for c in SIZES for lr in LRS]  # the 15 NeuralBandit1 models of Sec. 5


def sigmoid(z):
    return 0.5 * (1 + np.tanh(0.5 * z))  # = 1 / (1 + e^-z), without overflow


def draw(p, rng):
    """Sample an index from the probability vector p."""
    return min(int(np.searchsorted(np.cumsum(p), rng.random())), len(p) - 1)


def greedy_probs(scores, gamma):
    """P(k) = (1 - gamma) 1[k = argmax scores] + gamma / K."""
    p = np.full(len(scores), gamma / len(scores))
    p[np.argmax(scores)] += 1 - gamma
    return p


class Random:
    def __init__(self, K, rng):
        self.K, self.rng = K, rng

    def select(self, x):
        return int(self.rng.integers(self.K))

    def update(self, x, k, r):
        pass


class LinUCB:
    """Disjoint LinUCB (Li et al. 2010) with ridge 1; A^-1 is kept up to date with Sherman-Morrison."""

    def __init__(self, K, D, alpha=1.0):
        self.alpha = alpha
        self.A_inv = np.tile(np.eye(D), (K, 1, 1))
        self.b = np.zeros((K, D))
        self.theta = np.zeros((K, D))

    def select(self, x):
        mean = self.theta[:, x].sum(1)
        var = self.A_inv[:, x[:, None], x].sum((1, 2))  # x^T A^-1 x for each arm
        return int(np.argmax(mean + self.alpha * np.sqrt(np.maximum(var, 0))))

    def update(self, x, k, r):
        u = self.A_inv[k, x].sum(0)  # A^-1 x (A^-1 is symmetric)
        self.A_inv[k] -= np.outer(u, u) / (1 + u[x].sum())
        self.b[k, x] += r
        self.theta[k] = self.A_inv[k] @ self.b[k]


class Committee:
    """For each action, one MLP per model in MODELS: one sigmoid hidden layer of C units, sigmoid output,
    no biases (N = dim(X) C + C weights), weights drawn from U(-0.5, 0.5) (Algorithm 1).
    The hidden units of all models sit side by side: W1 is (D, K, S) and v is (K, S), S = sum of the C's."""

    def __init__(self, K, D, rng):
        sizes = np.array([c for c, _ in MODELS])
        self.start = np.concatenate([[0], np.cumsum(sizes)[:-1]])  # first hidden unit of each model
        self.unit_model = np.repeat(np.arange(len(MODELS)), sizes)  # model owning each hidden unit
        self.unit_lr = np.repeat([lr for _, lr in MODELS], sizes)  # its learning step lambda
        self.W1 = rng.uniform(-0.5, 0.5, (D, K, sizes.sum()))
        self.v = rng.uniform(-0.5, 0.5, (K, sizes.sum()))

    def forward(self, x):
        """Outputs of all networks, shape (K, M)."""
        self.h = sigmoid(self.W1[x].sum(0))  # (K, S)
        self.out = sigmoid(np.add.reduceat(self.h * self.v, self.start, axis=1))
        return self.out

    def update(self, x, k, r, weight):
        """One backprop step on (r - out)^2 / 2 for action k's network in every model, scaled by weight (Eq. 1)."""
        o, h = self.out[k], self.h[k]
        d_out = ((r - o) * o * (1 - o))[self.unit_model]  # output delta, repeated for each hidden unit
        d_hid = d_out * self.v[k] * h * (1 - h)
        step = self.unit_lr * weight
        self.v[k] += step * d_out * h
        self.W1[x, k] += step * d_hid


class EXP3:
    """n independent EXP3 instances over M arms (Eq. 2 and 3); weights are kept as logs to avoid overflow."""

    def __init__(self, n, M, gamma, rng):
        self.logw = np.zeros((n, M))
        self.gamma, self.rng = gamma, rng

    def draw(self):
        w = np.exp(self.logw - self.logw.max(1, keepdims=True))
        M = w.shape[1]
        self.p = (1 - self.gamma) * w / w.sum(1, keepdims=True) + self.gamma / M
        u = self.rng.random((len(w), 1))
        self.m = np.minimum((self.p.cumsum(1) < u).sum(1), M - 1)
        return self.m

    def update(self, r):
        rows = np.arange(len(self.m))
        self.logw[rows, self.m] += self.gamma * r / (self.p[rows, self.m] * self.logw.shape[1])


class NeuralBandit:
    """NeuralBandit2 (per_action=False): one EXP3 picks the model that chooses the action (Algorithm 2).
    NeuralBandit3 (per_action=True): one EXP3 per action picks the model that scores it (Algorithm 3).
    In both, every model's network for the played action is updated with Eq. (1) but without its 1/P
    importance weight, and every EXP3 instance receives the reward. With 1/P (about 1400 on exploration
    steps when gamma = 0.005) most sigmoid networks saturate and stop learning; the author's own
    neuralbandit package also leaves 1/P out."""

    def __init__(self, K, D, rng, per_action, gamma=0.005, gamma_model=0.1):
        self.K, self.rng, self.gamma = K, rng, gamma
        self.net = Committee(K, D, rng)
        self.exp3 = EXP3(K if per_action else 1, len(MODELS), gamma_model, rng)

    def select(self, x):
        scores = self.net.forward(x)  # (K, M)
        s = scores[np.arange(self.K), self.exp3.draw()]  # each action scored by its chosen model
        self.p = greedy_probs(s, self.gamma)
        return draw(self.p, self.rng)

    def update(self, x, k, r):
        self.net.update(x, k, r, 1.0)  # no 1/P weight, see the class docstring
        self.exp3.update(r)
