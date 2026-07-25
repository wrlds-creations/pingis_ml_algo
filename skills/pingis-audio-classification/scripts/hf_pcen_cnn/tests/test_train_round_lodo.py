from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from hf_pcen_cnn.train_round_lodo import (
    FABLE_PROBABILITY_COLUMNS,
    NOISE_INDEX,
    RACKET_INDEX,
    _choose_round_grouped_split,
    _feature_matrix,
    _model_probabilities,
)


class RoundGroupedSplitTests(unittest.TestCase):
    def test_keeps_single_group_rare_class_in_training(self) -> None:
        labels = np.asarray(
            [1] * 12
            + [0] * 12
            + [2] * 12
            + [3] * 12,
            dtype=np.int64,
        )
        groups = np.asarray(
            ["rare"] * 12
            + [f"racket-{index // 3}" for index in range(12)]
            + [f"floor-{index // 3}" for index in range(12)]
            + [f"noise-{index // 3}" for index in range(12)],
        )
        indexes = np.arange(len(labels), dtype=np.int64)

        train_indexes, validation_indexes = _choose_round_grouped_split(
            labels,
            groups,
            indexes,
            seed=42,
        )

        self.assertEqual(set(labels[train_indexes]), {0, 1, 2, 3})
        self.assertNotIn("rare", set(groups[validation_indexes]))
        self.assertTrue(np.any(labels[validation_indexes] == RACKET_INDEX))
        self.assertTrue(np.any(labels[validation_indexes] != RACKET_INDEX))
        self.assertFalse(
            set(groups[train_indexes]) & set(groups[validation_indexes])
        )

    def test_feature_matrix_stacks_fable_probabilities_and_gate_evidence(self) -> None:
        row = {
            "fable_feature_rms_mean": 0.25,
            "fable_confidence": 0.82,
            "hf_rms": 0.04,
            "background_rms": 0.01,
            "full_band_peak": 0.5,
            "threshold": 0.02,
        }
        row.update(
            dict(
                zip(
                    FABLE_PROBABILITY_COLUMNS,
                    (0.7, 0.1, 0.05, 0.15),
                    strict=True,
                )
            )
        )

        matrix, names = _feature_matrix(pd.DataFrame([row]))

        self.assertEqual(matrix.shape, (1, len(names)))
        self.assertIn("fable_prob_racket_bounce", names)
        self.assertIn("fable_confidence", names)
        self.assertIn("gate_threshold_ratio", names)
        self.assertAlmostEqual(
            matrix[0, names.index("fable_prob_racket_bounce")],
            0.7,
        )

    def test_binary_probabilities_map_to_racket_and_noise_channels(self) -> None:
        features = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        model = ExtraTreesClassifier(n_estimators=8, random_state=42).fit(
            features,
            labels,
        )

        probabilities = _model_probabilities(
            model,
            features,
            np.arange(len(features)),
            binary_target=True,
        )

        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        self.assertTrue(np.all(probabilities[:, RACKET_INDEX] >= 0.0))
        self.assertTrue(np.all(probabilities[:, NOISE_INDEX] >= 0.0))
        other_indexes = {
            index
            for index in range(probabilities.shape[1])
            if index not in {RACKET_INDEX, NOISE_INDEX}
        }
        self.assertTrue(np.all(probabilities[:, list(other_indexes)] == 0.0))


if __name__ == "__main__":
    unittest.main()
