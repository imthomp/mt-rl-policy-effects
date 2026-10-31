import math
import unittest

from synthetic_validation import (
    batch_gradients,
    exact_gradient,
    exact_per_response_token_mean_gradient,
    exact_reinforce_gradient as simple_exact_reinforce_gradient,
    monte_carlo_gradient,
    paths,
    sigmoid,
)
from joint_segmentation_policy import (
    enumerate_trajectories,
    exact_gradient as joint_exact_gradient,
    exact_reinforce_gradient,
    exact_token_mean_gradient,
    monte_carlo_gradient as joint_monte_carlo_gradient,
    report as joint_report,
    sigmoid as joint_sigmoid,
)
from exact_same_string_harness import (
    STRINGS,
    exact_batch_gradient as exact_vector_batch_gradient,
    exact_position_weighted_gradient,
    exact_response_mean_gradient as exact_vector_response_gradient,
    exact_string_gradient as exact_vector_string_gradient,
    enumerate_outcomes,
    monte_carlo_batch_gradients,
    score_sample,
    sigmoid as vector_sigmoid,
)


class SyntheticValidationTests(unittest.TestCase):
    def test_exact_joint_path_gradient_matches_string_gradient(self):
        for theta in (-2.0, 0.0, 0.4, 2.0):
            self.assertAlmostEqual(simple_exact_reinforce_gradient(theta), exact_gradient(theta), places=12)

    def test_alternative_paths_preserve_string_probability_and_reward(self):
        for theta in (-1.0, 0.4, 1.5):
            outcomes = paths(theta)
            p = sigmoid(theta)
            self.assertAlmostEqual(sum(x.probability for x in outcomes), 1.0, places=12)
            self.assertAlmostEqual(sum(x.probability for x in outcomes if x.string == "A"), p, places=12)
            self.assertAlmostEqual(sum(x.probability * x.reward for x in outcomes), p, places=12)

    def test_sampled_gradient_converges_to_exact(self):
        theta = 0.4
        estimate = monte_carlo_gradient(theta, samples=300_000, seed=17)
        self.assertLess(abs(estimate - exact_gradient(theta)), 0.002)

    def test_per_response_token_mean_changes_length_weighting(self):
        theta = 0.4
        self.assertNotAlmostEqual(
            exact_per_response_token_mean_gradient(theta), exact_gradient(theta), places=4
        )

    def test_batch_reductions_are_scalar_related_on_identical_paths(self):
        theta = 0.4
        batch = (paths(theta)[0], paths(theta)[1], paths(theta)[2])
        fixed, token = batch_gradients(theta, batch)
        valid_tokens = sum(item.token_count for item in batch)
        expected_factor = len(batch) / valid_tokens
        self.assertTrue(math.isclose(token, fixed * expected_factor, rel_tol=1e-12, abs_tol=1e-12))

    def test_joint_path_probabilities_marginalize_to_string_policy(self):
        for theta, phi in ((-1.2, -2.0), (0.4, -0.2), (1.3, 1.1)):
            outcomes = enumerate_trajectories(theta, phi)
            self.assertAlmostEqual(sum(math.exp(x.behavior_log_probability) for x in outcomes), 1.0, places=12)
            mass_a = sum(math.exp(x.behavior_log_probability) for x in outcomes if x.string == "A")
            self.assertAlmostEqual(mass_a, joint_sigmoid(theta), places=12)

    def test_joint_policy_gradient_is_invariant_to_segmentation_parameter(self):
        for theta, phi in ((-1.2, -2.0), (0.4, -0.2), (1.3, 1.1)):
            grad_theta, grad_phi = exact_reinforce_gradient(theta, phi)
            self.assertAlmostEqual(grad_theta, joint_exact_gradient(theta), places=12)
            self.assertAlmostEqual(grad_phi, 0.0, places=12)

    def test_joint_sample_score_matches_behavior_probability(self):
        # report() runs independent recomputation checks for 128 sampled paths.
        result = joint_report(theta=0.4, phi=-0.2, samples=1000)
        self.assertAlmostEqual(result["joint_path_probability_mass_for_A"], result["string_A_probability"], places=12)

    def test_joint_monte_carlo_converges_and_length_weighting_changes_gradient(self):
        theta, phi = 0.4, -0.2
        mc_theta, mc_phi = joint_monte_carlo_gradient(theta, phi, samples=300_000, seed=19)
        exact_theta, exact_phi = exact_reinforce_gradient(theta, phi)
        self.assertLess(abs(mc_theta - exact_theta), 0.002)
        self.assertLess(abs(mc_phi - exact_phi), 0.002)
        token_theta, token_phi = exact_token_mean_gradient(theta, phi)
        self.assertNotAlmostEqual(token_theta, exact_theta, places=4)
        self.assertNotAlmostEqual(token_phi, exact_phi, places=4)

    def test_vector_joint_probabilities_normalize_and_marginalize(self):
        theta = (0.3, -0.2, 0.1)
        for phi in (-2.0, 0.0, 1.1):
            outcomes = enumerate_outcomes(theta, phi)
            self.assertAlmostEqual(sum(item.probability for item in outcomes), 1.0, places=12)
            for index, string in enumerate(STRINGS):
                mass = sum(item.probability for item in outcomes if item.string == string)
                self.assertAlmostEqual(mass, math.exp(theta[index]) / sum(map(math.exp, theta)), places=12)

    def test_vector_response_gradient_is_invariant_to_path_sampler(self):
        theta = (0.3, -0.2, 0.1)
        expected = exact_vector_string_gradient(theta)
        for phi in (-2.0, 0.0, 1.1):
            grad_theta, grad_phi = exact_vector_response_gradient(theta, phi)
            for actual, target in zip(grad_theta, expected, strict=True):
                self.assertAlmostEqual(actual, target, places=12)
            self.assertAlmostEqual(grad_phi, 0.0, places=12)

    def test_vector_sample_log_probability_and_scores_are_joint(self):
        theta, phi = (0.3, -0.2, 0.1), -1.1
        for outcome in enumerate_outcomes(theta, phi):
            logp, score_theta, score_phi = score_sample(outcome.string, outcome.path, theta, phi)
            self.assertAlmostEqual(math.exp(logp), outcome.probability, places=12)
            self.assertEqual(score_theta, outcome.score_theta)
            self.assertAlmostEqual(score_phi, outcome.score_phi, places=12)

    def test_global_token_mean_changes_expected_weighting_but_fixed_batch_is_scalar(self):
        theta = (0.3, -0.2, 0.1)
        response, response_phi = exact_vector_batch_gradient(theta, -1.1, 2, token_mean=False)
        token, token_phi = exact_vector_batch_gradient(theta, -1.1, 2, token_mean=True)
        self.assertAlmostEqual(response_phi, 0.0, places=12)
        self.assertNotAlmostEqual(token_phi, 0.0, places=5)
        self.assertNotAlmostEqual(token[0] / response[0], token[2] / response[2], places=4)
        token_other_q, _ = exact_vector_batch_gradient(theta, 1.1, 2, token_mean=True)
        self.assertTrue(any(not math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5) for a, b in zip(token, token_other_q, strict=True)))

        # For any one realized batch, dividing by total tokens is exactly a
        # common scalar multiple of dividing by response count.
        batch = (enumerate_outcomes(theta, -1.1)[0], enumerate_outcomes(theta, -1.1)[-1])
        response_sum = tuple(sum(item.reward * item.score_theta[i] for item in batch) / len(batch) for i in range(3))
        token_sum = tuple(sum(item.reward * item.score_theta[i] for item in batch) / sum(item.token_count for item in batch) for i in range(3))
        factor = len(batch) / sum(item.token_count for item in batch)
        for fixed, normalized in zip(response_sum, token_sum, strict=True):
            self.assertAlmostEqual(normalized, factor * fixed, places=12)

    def test_position_weighting_is_same_at_gamma_one_and_changes_with_gamma(self):
        theta = (0.3, -0.2, 0.1)
        ordinary, _ = exact_vector_response_gradient(theta, -1.1)
        gamma_one = exact_position_weighted_gradient(theta, -1.1, 1.0)
        gamma_half = exact_position_weighted_gradient(theta, -1.1, 0.5)
        for actual, target in zip(gamma_one, ordinary, strict=True):
            self.assertAlmostEqual(actual, target, places=12)
        self.assertTrue(any(not math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5) for a, b in zip(gamma_half, ordinary, strict=True)))

    def test_vector_monte_carlo_batch_estimators_match_exact_enumeration(self):
        theta, phi = (0.3, -0.2, 0.1), -1.1
        exact_response = exact_vector_batch_gradient(theta, phi, 2, token_mean=False)
        exact_token = exact_vector_batch_gradient(theta, phi, 2, token_mean=True)
        estimate = monte_carlo_batch_gradients(theta, phi, batches=30_000, batch_size=2, seed=73)
        for key, target in (("response_mean", exact_response), ("global_token_mean", exact_token)):
            actual_theta, actual_phi = estimate[key]
            for actual, expected in zip(actual_theta, target[0], strict=True):
                self.assertLess(abs(actual - expected), 0.008)
            self.assertLess(abs(actual_phi - target[1]), 0.008)


if __name__ == "__main__":
    unittest.main()
