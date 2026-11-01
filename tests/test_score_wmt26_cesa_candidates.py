import unittest

from score_wmt26_cesa_candidates import split_html_segments


class Wmt26CesaScoringTests(unittest.TestCase):
    def test_html_segmentation_is_lossless(self):
        text = "<p>one</p>\n<p>two</p>\n"
        segments = split_html_segments(text)
        self.assertEqual(segments, ["<p>one</p>", "\n<p>two</p>", "\n"])
        self.assertEqual("".join(segments), text)


if __name__ == "__main__":
    unittest.main()
