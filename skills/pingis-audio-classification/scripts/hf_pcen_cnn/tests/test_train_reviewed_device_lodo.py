from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from hf_pcen_cnn.train_reviewed_device_lodo import (
    _binary_racket_metrics,
    _oracle_threshold_diagnostic,
    review_fold_indexes,
)


class ReviewedDeviceLodoTests(unittest.TestCase):
    def test_held_device_never_enters_review_training(self) -> None:
        metadata = pd.DataFrame(
            {
                "cache_complete": [True, True, True, True, True],
                "disposition": [
                    "positive",
                    "hard_negative",
                    "ambiguous",
                    "positive",
                    "hard_negative",
                ],
                "label": [
                    "racket_bounce",
                    "voice_music_noise",
                    None,
                    "racket_bounce",
                    "voice_music_noise",
                ],
                "device_id": ["phone-a", "phone-a", "phone-a", "phone-b", "phone-b"],
            }
        )

        train, held = review_fold_indexes(metadata, "phone-a")

        self.assertEqual(train.tolist(), [3, 4])
        self.assertEqual(held.tolist(), [0, 1, 2])

    def test_binary_metrics_reduce_four_classes_to_racket_or_not(self) -> None:
        labels = np.asarray([0, 1, 2, 3], dtype=np.int64)
        predictions = np.asarray([0, 0, 2, 3], dtype=np.int64)

        report = _binary_racket_metrics(labels, predictions)

        self.assertEqual(report["rows"], 4)
        self.assertEqual(report["precision"], 0.5)
        self.assertEqual(report["recall"], 1.0)

    def test_oracle_threshold_is_marked_as_non_deployable(self) -> None:
        metadata = pd.DataFrame(
            {
                "session_id": ["session-a", "session-a"],
                "device_id": ["phone-a", "phone-a"],
                "scenario_id": ["normal", "normal"],
                "disposition": ["positive", "hard_negative"],
                "matched_event_index": [0, None],
                "reviewed_event_ms": [100.0, None],
            }
        )
        candidates = [
            {
                "onset_ms": 100.0,
                "frame_rms": 1.0,
                "background_db": -80.0,
                "predicted_label": "racket_bounce",
                "confidence": 0.9,
            },
            {
                "onset_ms": 500.0,
                "frame_rms": 1.0,
                "background_db": -80.0,
                "predicted_label": "racket_bounce",
                "confidence": 0.4,
            },
        ]

        report = _oracle_threshold_diagnostic(
            metadata, candidates, tolerance_ms=140.0
        )

        self.assertIn("must not be deployed", report["warning"])
        self.assertEqual(report["counting"]["aggregate"]["f1"], 1.0)
        self.assertEqual(report["counting"]["aggregate"]["fp"], 0)


if __name__ == "__main__":
    unittest.main()
