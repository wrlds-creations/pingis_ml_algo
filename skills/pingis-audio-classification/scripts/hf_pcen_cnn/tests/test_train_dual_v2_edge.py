from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from hf_pcen_cnn.train_dual_v2_edge import (
    ConcatenatedFeature,
    _edge_truth_times,
)


class ConcatenatedFeatureTests(unittest.TestCase):
    def test_indexes_across_array_boundary(self) -> None:
        first = np.arange(12).reshape(3, 2, 2)
        second = np.arange(100, 108).reshape(2, 2, 2)
        feature = ConcatenatedFeature((first, second))

        self.assertEqual(feature.shape, (5, 2, 2))
        np.testing.assert_array_equal(feature[2], first[2])
        np.testing.assert_array_equal(feature[3], second[0])
        np.testing.assert_array_equal(feature[-1], second[-1])

    def test_rejects_mismatched_feature_shapes(self) -> None:
        with self.assertRaises(ValueError):
            ConcatenatedFeature(
                (
                    np.zeros((2, 3, 4)),
                    np.zeros((2, 4, 4)),
                )
            )


class EdgeTruthTests(unittest.TestCase):
    def test_only_unique_racket_events_are_counting_truth(self) -> None:
        metadata = pd.DataFrame(
            {
                "disposition": ["positive", "positive", "positive", "unlabeled"],
                "label": [
                    "racket_bounce",
                    "racket_bounce",
                    "table_bounce",
                    None,
                ],
                "matched_event_index": [3, 3, 4, np.nan],
                "reviewed_event_ms": [1200.0, 1200.0, 1600.0, np.nan],
            }
        )

        self.assertEqual(_edge_truth_times(metadata), [1200.0])


if __name__ == "__main__":
    unittest.main()
