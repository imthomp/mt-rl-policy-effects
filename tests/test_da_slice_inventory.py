import unittest

import pandas as pd

from inventory_da_slices import build_inventory, direction_class


class DaSliceInventoryTests(unittest.TestCase):
    def test_direction_class(self):
        self.assertEqual(direction_class("en-de"), "en_to_x")
        self.assertEqual(direction_class("zh-en"), "x_to_en")
        self.assertEqual(direction_class("de-fr"), "x_to_y")

    def test_build_inventory_traces_every_input_row(self):
        frame = pd.DataFrame(
            [
                {"lp": "en-de", "year": 2020, "domain": "news", "annotators": 1},
                {"lp": "en-de", "year": 2020, "domain": "news", "annotators": 2},
                {"lp": "zh-en", "year": 2021, "domain": "other", "annotators": 1},
            ]
        )
        rows = build_inventory(frame, {("zh-en", 2021)})
        self.assertEqual(sum(row["label_rows"] for row in rows), len(frame))
        self.assertFalse(rows[0]["used_in_diagnostic_discovery"])
        self.assertTrue(rows[1]["used_in_diagnostic_discovery"])
        self.assertEqual(rows[0]["domain_counts"], {"news": 2})


if __name__ == "__main__":
    unittest.main()
