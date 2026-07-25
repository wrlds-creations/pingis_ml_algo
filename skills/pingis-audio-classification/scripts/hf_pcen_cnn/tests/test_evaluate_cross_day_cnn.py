from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from hf_pcen_cnn.evaluate_cross_day_cnn import (
    MODEL_FACTORIES,
    _checkpoint_fold_by_device,
    _load_frontends,
)


class CrossDayCnnTests(unittest.TestCase):
    def test_factories_match_supported_frontend_contracts(self) -> None:
        self.assertEqual(
            set(MODEL_FACTORIES),
            {"logmel_compact", "pcen_compact", "dual_residual"},
        )
        self.assertEqual(
            MODEL_FACTORIES["logmel_compact"]().features[0].in_channels,
            1,
        )
        self.assertEqual(
            MODEL_FACTORIES["dual_residual"]().features[0].in_channels,
            2,
        )

    def test_fold_numbers_follow_report_order(self) -> None:
        report = {
            "folds": [
                {"held_device": "iphone"},
                {"held_device": "moto"},
                {"held_device": "huawei"},
            ]
        }
        self.assertEqual(
            _checkpoint_fold_by_device(report),
            {"iphone": 1, "moto": 2, "huawei": 3},
        )

    def test_loads_frontends_in_checkpoint_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "logmel.npy", np.ones((2, 3), dtype=np.float16))
            np.save(root / "pcen.npy", np.full((2, 3), 2.0, dtype=np.float16))

            frontends = _load_frontends(root, ("pcen", "logmel"))

            self.assertEqual(float(frontends[0][0, 0]), 2.0)
            self.assertEqual(float(frontends[1][0, 0]), 1.0)
            del frontends

    def test_rejects_unknown_frontend(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported CNN frontends"):
            _load_frontends(Path("."), ("mfcc",))


if __name__ == "__main__":
    unittest.main()
