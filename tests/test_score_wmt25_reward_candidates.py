import unittest

from score_wmt25_reward_candidates import mqm_score


class Wmt25RewardScoringTests(unittest.TestCase):
    def test_official_mqm_weights(self):
        errors = [
            {"severity": "major", "error_type": "accuracy/mistranslation"},
            {"severity": "minor", "error_type": "fluency/punctuation"},
            {"severity": "minor", "error_type": "fluency/grammar"},
        ]
        self.assertAlmostEqual(mqm_score(errors), -6.1)

    def test_nontranslation_is_fixed_score(self):
        errors = [
            {"severity": "major", "error_type": "non-translation!"},
            {"severity": "major", "error_type": "accuracy/mistranslation"},
        ]
        self.assertEqual(mqm_score(errors), -25.0)

    def test_unknown_severity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown MQM severity"):
            mqm_score([{"severity": "unknown", "error_type": "other"}])


if __name__ == "__main__":
    unittest.main()
