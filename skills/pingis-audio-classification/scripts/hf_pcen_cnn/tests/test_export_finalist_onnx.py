from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from hf_pcen_cnn.export_finalist_onnx import (
    select_parity_rows,
    stack_model_input,
)


class ExportFinalistOnnxTests(unittest.TestCase):
    def test_stacks_frontends_in_declared_order(self) -> None:
        logmel = np.full((2, 64, 65), 1.0, dtype=np.float32)
        pcen = np.full((2, 64, 65), 2.0, dtype=np.float32)
        result = stack_model_input(
            ("pcen", "logmel"),
            logmel=logmel,
            pcen=pcen,
        )
        self.assertEqual(result.shape, (2, 2, 64, 65))
        self.assertTrue(np.all(result[:, 0] == 2.0))
        self.assertTrue(np.all(result[:, 1] == 1.0))

    def test_parity_rows_prefer_device_and_label_diversity(self) -> None:
        metadata = pd.DataFrame(
            [
                {
                    "cache_complete": True,
                    "device_alias": "a",
                    "label": "racket_bounce",
                    "session_id": "s1",
                    "candidate_index": 0,
                },
                {
                    "cache_complete": True,
                    "device_alias": "a",
                    "label": "voice_music_noise",
                    "session_id": "s1",
                    "candidate_index": 1,
                },
                {
                    "cache_complete": True,
                    "device_alias": "b",
                    "label": "racket_bounce",
                    "session_id": "s2",
                    "candidate_index": 0,
                },
                {
                    "cache_complete": False,
                    "device_alias": "c",
                    "label": "table_bounce",
                    "session_id": "s3",
                    "candidate_index": 0,
                },
            ]
        )
        selected = select_parity_rows(metadata, limit=3)
        self.assertEqual(selected, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
