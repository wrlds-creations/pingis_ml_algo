from __future__ import annotations

import unittest

from hf_pcen_cnn.sweep_histgb_stack import HISTGB_CONFIGS


class HistgbSweepTests(unittest.TestCase):
    def test_candidates_are_four_distinct_regularization_profiles(self) -> None:
        self.assertEqual(len(HISTGB_CONFIGS), 4)
        signatures = {
            tuple(sorted(config.items()))
            for config in HISTGB_CONFIGS.values()
        }
        self.assertEqual(len(signatures), len(HISTGB_CONFIGS))
        for name, config in HISTGB_CONFIGS.items():
            self.assertTrue(name.startswith("histgb_stack_"))
            self.assertGreater(int(config["max_iter"]), 0)
            self.assertGreater(int(config["min_samples_leaf"]), 0)


if __name__ == "__main__":
    unittest.main()
