from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from hf_pcen_cnn.train_reviewed_device_lodo import (
    _binary_racket_metrics,
    _oracle_threshold_diagnostic,
    _truth_times,
    review_fold_indexes,
)


class ReviewedDeviceLodoTests(unittest.TestCase):
    def test_truth_times_include_reviewed_event_without_gate_candidate(self) -> None:
        with TemporaryDirectory() as temporary:
            labels_path = Path(temporary) / "review_labels.json"
            labels_path.write_text(
                """
                {
                  "manual_markers": [
                    {"time_s": 1.0, "label": "racket"},
                    {"time_s": 2.0, "label": "racket_bounce"},
                    {"time_s": 3.0, "label": "other"}
                  ]
                }
                """,
                encoding="utf-8",
            )
            rows = pd.DataFrame(
                {
                    "labels_path": [str(labels_path)],
                    "disposition": ["positive"],
                    "matched_event_index": [0],
                    "reviewed_event_ms": [1000.0],
                }
            )

            self.assertEqual(_truth_times(rows), [1000.0, 2000.0])

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
