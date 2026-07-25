from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from hf_pcen_cnn.evaluate_final_holdout import _score_model


class FinalHoldoutTests(unittest.TestCase):
    def test_scores_only_trainable_candidates_for_class_metrics(self) -> None:
        metadata = pd.DataFrame(
            [
                {
                    "session_id": "s1",
                    "device_id": "device-1",
                    "scenario_id": "scenario-1",
                    "onset_ms": 100.0,
                    "hf_rms": 0.1,
                    "background_db": -40.0,
                    "disposition": "positive",
                    "label": "racket_bounce",
                    "matched_event_index": 0,
                    "reviewed_event_ms": 100.0,
                },
                {
                    "session_id": "s1",
                    "device_id": "device-1",
                    "scenario_id": "scenario-1",
                    "onset_ms": 120.0,
                    "hf_rms": 0.1,
                    "background_db": -40.0,
                    "disposition": "ambiguous",
                    "label": "ambiguous",
                    "matched_event_index": np.nan,
                    "reviewed_event_ms": np.nan,
                },
            ]
        )
        labels = np.array([0, -1], dtype=np.int64)
        probabilities = np.array(
            [
                [0.9, 0.05, 0.03, 0.02],
                [0.8, 0.1, 0.05, 0.05],
            ]
        )

        report = _score_model(
            metadata,
            labels,
            np.array([0, 1]),
            np.array([0]),
            probabilities,
            0.5,
            tolerance_ms=140.0,
        )

        self.assertEqual(report["candidate_four_class"]["rows"], 1)
        self.assertEqual(
            report["candidate_racket_binary"]["confusion_matrix"],
            [[0, 0], [0, 1]],
        )


if __name__ == "__main__":
    unittest.main()
