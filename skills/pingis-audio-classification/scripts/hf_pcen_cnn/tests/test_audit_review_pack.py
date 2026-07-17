from __future__ import annotations

import unittest

from hf_pcen_cnn.audit_review_pack import (
    classify_rejected_positive,
    greedy_time_match,
    loud_threshold_grid,
)


class AuditReviewPackTests(unittest.TestCase):
    def test_greedy_time_match_is_one_to_one(self) -> None:
        matches, truth = greedy_time_match(
            [1_010.0, 1_030.0, 2_100.0],
            [1_000.0, 2_000.0],
            140.0,
        )

        self.assertEqual(matches, {0: 0, 2: 1})
        self.assertEqual(truth, {0, 1})

    def test_threshold_grid_includes_deployed_value(self) -> None:
        self.assertIn(0.85, loud_threshold_grid())

    def test_rejection_reason_distinguishes_label_and_confidence(self) -> None:
        self.assertEqual(
            classify_rejected_positive(
                {"counted": False, "predicted_label": "voice_music_noise", "timing_reason": "not_racket"}
            ),
            "positive_top_label_reject",
        )
        self.assertEqual(
            classify_rejected_positive(
                {
                    "counted": False,
                    "predicted_label": "racket_bounce",
                    "timing_reason": "low_confidence_loud_bg",
                }
            ),
            "positive_low_confidence",
        )


if __name__ == "__main__":
    unittest.main()
