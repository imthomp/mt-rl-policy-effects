import unittest

from encode_mqm_confirmation_candidates import arrange, text_digest


class EncodeMQMConfirmationCandidatesTests(unittest.TestCase):
    def test_arrange_builds_rectangular_text_environment(self):
        rows = [
            {"segment_key": context, "system_id": action, "source": f"src-{context}", "hypothesis": f"{context}-{action}"}
            for context in ("c2", "c1")
            for action in ("b", "a")
        ]
        contexts, actions, sources, candidates = arrange(rows)
        self.assertEqual(contexts, ["c1", "c2"])
        self.assertEqual(actions, ["a", "b"])
        self.assertEqual(sources, ["src-c1", "src-c2"])
        self.assertEqual(candidates[0], ["c1-a", "c1-b"])
        self.assertEqual(text_digest(sources, candidates), text_digest(sources, candidates))

    def test_arrange_rejects_missing_candidate(self):
        with self.assertRaises(ValueError):
            arrange(
                [
                    {"segment_key": "c1", "system_id": "a", "source": "s", "hypothesis": "x"},
                    {"segment_key": "c1", "system_id": "b", "source": "s", "hypothesis": "y"},
                    {"segment_key": "c2", "system_id": "a", "source": "s2", "hypothesis": "z"},
                ]
            )


if __name__ == "__main__":
    unittest.main()
