from __future__ import annotations

import unittest

import numpy as np

from hf_pcen_cnn.train_round_lodo import (
    RACKET_INDEX,
    _choose_round_grouped_split,
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


if __name__ == "__main__":
    unittest.main()
