"""Small deterministic checks; no Covertype download required."""
import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from improvement.adamw import AdamW
from improvement.adamw_bandit import AdamWCommittee, AdamWNeuralBandit
from improvement import run_adamw as runner
from reproduce import bandits


class AdamWTests(unittest.TestCase):
    def test_scalar_reference_and_zero_decay(self):
        for decay in (0.0, 0.02):
            parameter = np.array([0.7, -0.2])
            expected = parameter.copy()
            lr = np.array([0.001, 0.01])
            optimizer = AdamW(parameter, decay)
            m = [0.0, 0.0]
            v = [0.0, 0.0]
            for step, gradient in enumerate(([0.2, -0.3], [0, 0.4], [-0.1, 0]), 1):
                for j in range(2):
                    m[j] = 0.9 * m[j] + 0.1 * gradient[j]
                    v[j] = 0.999 * v[j] + 0.001 * gradient[j] ** 2
                    expected[j] *= 1 - lr[j] * decay
                    expected[j] -= lr[j] * (m[j] / (1 - 0.9 ** step)) / (
                        np.sqrt(v[j] / (1 - 0.999 ** step)) + 1e-8)
                optimizer.step(parameter, np.array(gradient), lr)
                np.testing.assert_allclose(parameter, expected, rtol=1e-12, atol=1e-14)

    def test_selected_action_and_dense_momentum(self):
        net = AdamWCommittee(3, 6, np.random.default_rng(2), weight_decay=0.01)
        old_hidden, old_heads = net.W1.copy(), net.v.copy()
        net.forward(np.array([0, 1]))
        net.update(np.array([0, 1]), 1, 1, 1.0)
        first_hidden = net.W1.copy()
        net.forward(np.array([2, 3]))
        net.update(np.array([2, 3]), 1, 0, 1.0)
        for k in (0, 2):
            np.testing.assert_array_equal(net.W1[:, k], old_hidden[:, k])
            np.testing.assert_array_equal(net.v[k], old_heads[k])
            self.assertEqual(net.hidden_optimizers[k].t, 0)
            self.assertEqual(net.head_optimizers[k].t, 0)
            self.assertFalse(net.hidden_optimizers[k].m.any())
        self.assertEqual(net.hidden_optimizers[1].t, 2)
        self.assertEqual(net.head_optimizers[1].t, 2)
        # Row 0 has no current input but historical momentum still advances.
        self.assertTrue(net.hidden_optimizers[1].m[0].any())
        self.assertFalse(np.array_equal(net.W1[0, 1], first_hidden[0, 1]))

    def test_gradient_sign_and_hidden_backprop(self):
        net = AdamWCommittee(3, 6, np.random.default_rng(8), weight_decay=0)
        x = np.array([0, 2])
        k, model, reward = 1, 0, 1
        unit = net.start[model]
        coordinates = [(net.v, (k, unit)), (net.W1, (0, k, unit))]
        numeric = []
        delta = 1e-5
        for parameter, index in coordinates:
            original = parameter[index]
            parameter[index] = original + delta
            plus = 0.5 * (net.forward(x)[k, model] - reward) ** 2
            parameter[index] = original - delta
            minus = 0.5 * (net.forward(x)[k, model] - reward) ** 2
            parameter[index] = original
            numeric.append((plus - minus) / (2 * delta))
        before = [parameter[index] for parameter, index in coordinates]
        net.forward(x)
        net.update(x, k, reward, 1.0)
        for (parameter, index), previous, grad in zip(coordinates, before, numeric):
            expected = previous - net.unit_lr[unit] * grad / (abs(grad) + 1e-8)
            self.assertAlmostEqual(parameter[index], expected, places=10)

    def test_pairing_and_nb2_nb3_online_updates(self):
        for per_action in (False, True):
            rng1, rng2 = np.random.default_rng(3), np.random.default_rng(3)
            original = bandits.NeuralBandit(3, 6, rng1, per_action)
            improved = AdamWNeuralBandit(3, 6, rng2, per_action)
            np.testing.assert_array_equal(original.net.W1, improved.net.W1)
            np.testing.assert_array_equal(original.net.v, improved.net.v)
            self.assertEqual(rng1.integers(10000), rng2.integers(10000))
            counts = np.zeros(3, dtype=int)
            for t in range(100):
                x = np.array([t % 3, 3 + t % 3])
                action = improved.select(x)
                improved.update(x, action, int(action == (t // 25) % 3))
                counts[action] += 1
            self.assertEqual([o.t for o in improved.net.head_optimizers], counts.tolist())
            self.assertTrue(np.isfinite(improved.net.W1).all())
            self.assertTrue(np.isfinite(improved.exp3.logw).all())

    def test_runner_outputs_on_synthetic_stream(self):
        idx = np.tile(np.arange(12), (21, 1))
        labels = np.arange(21) % 7
        with tempfile.TemporaryDirectory() as out:
            args = argparse.Namespace(optimizer="adamw", lrs=(0.001, 0.003, 0.01),
                                      weight_decay=1e-4, T=100, out=out)
            with patch.object(runner.data, "load", return_value=(idx, labels)), \
                 patch.object(runner, "DRIFT", 25):
                for name in ("nb2", "nb3"):
                    for drift in (False, True):
                        summary = runner.run((args, name, drift, 0))
                        self.assertEqual(summary["steps"], 100)
                        self.assertTrue(np.isfinite(summary["final_regret"]))
            self.assertEqual(len(list(Path(out).glob("*.npy"))), 4)
            self.assertEqual(len(list(Path(out).glob("*.json"))), 4)


if __name__ == "__main__":
    unittest.main()
