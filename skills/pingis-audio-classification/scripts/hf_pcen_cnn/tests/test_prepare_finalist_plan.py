from __future__ import annotations

import unittest

from hf_pcen_cnn.prepare_finalist_plan import (
    CNN_FINALISTS,
    HISTGB_FINALISTS,
    _median_best_epoch,
    _median_threshold,
)


class PrepareFinalistPlanTests(unittest.TestCase):
    def test_all_finalists_are_four_class_candidates(self) -> None:
        self.assertEqual(
            CNN_FINALISTS,
            ("dual_residual", "pcen_compact"),
        )
        self.assertEqual(
            HISTGB_FINALISTS,
            ("histgb_stack_reference", "histgb_stack_flexible"),
        )

    def test_uses_fold_median_without_holdout_tuning(self) -> None:
        report = {
            "folds": [
                {"threshold": 0.95, "training": {"best_epoch": 7}},
                {"threshold": 0.20, "training": {"best_epoch": 13}},
                {"threshold": 0.45, "training": {"best_epoch": 15}},
            ]
        }
        self.assertEqual(_median_threshold(report), 0.45)
        self.assertEqual(_median_best_epoch(report), 13)


if __name__ == "__main__":
    unittest.main()
