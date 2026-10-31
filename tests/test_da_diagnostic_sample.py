import unittest

import pandas as pd

from prepare_da_diagnostic_sample import select_indices


class DaDiagnosticSampleTests(unittest.TestCase):
    def test_selection_is_deterministic_capped_and_slice_local(self):
        frame = pd.DataFrame(
            {
                "lp": ["en-de"] * 5 + ["zh-en"] * 3,
                "year": [2020] * 5 + [2021] * 3,
            }
        )
        first = select_indices(frame, cap=2, seed=17)
        second = select_indices(frame, cap=2, seed=17)
        self.assertEqual(first, second)
        self.assertEqual({key: len(value) for key, value in first.items()}, {
            ("en-de", 2020): 2,
            ("zh-en", 2021): 2,
        })
        self.assertTrue(all(index < 5 for index in first[("en-de", 2020)]))
        self.assertTrue(all(index >= 5 for index in first[("zh-en", 2021)]))


if __name__ == "__main__":
    unittest.main()
