import unittest

try:
    import torch
except ImportError:  # The lightweight test environment need not install Torch.
    torch = None

from run_confirmation_text_policies import optimize_to_kl_grid, pressure_metrics


@unittest.skipIf(torch is None, "Torch is exercised in the GPU execution environment")
class ConfirmationTextPolicyTests(unittest.TestCase):
    def test_exact_optimizer_increases_proxy_and_reaches_small_kl(self):
        embeddings = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]
        )
        reward = torch.tensor([[1.0, -1.0], [1.0, -1.0]])
        checkpoints, records = optimize_to_kl_grid(
            embeddings, reward, [0.0, 0.01], 0.5, 100
        )
        self.assertTrue(records["0.01"]["reached"])
        initial = pressure_metrics(embeddings, checkpoints[0.0], reward)[2]
        final = pressure_metrics(embeddings, checkpoints[0.01], reward)[2]
        self.assertGreater(final, initial)


if __name__ == "__main__":
    unittest.main()
