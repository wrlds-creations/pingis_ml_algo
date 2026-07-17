from __future__ import annotations

import unittest

import numpy as np

from hf_pcen_cnn.frontends import DEFAULT_FRONTEND_CONFIG, frontend_pair


class FrontendTests(unittest.TestCase):
    def test_frontends_share_shape_and_are_finite(self) -> None:
        config = DEFAULT_FRONTEND_CONFIG
        time = np.arange(config.clip_samples, dtype=np.float32) / config.sample_rate
        clip = 0.1 * np.sin(2.0 * np.pi * 1_200.0 * time)
        logmel, pcen = frontend_pair(clip)

        self.assertEqual(logmel.shape, pcen.shape)
        self.assertEqual(logmel.shape[0], 1)
        self.assertEqual(logmel.shape[1], config.n_mels)
        self.assertTrue(np.isfinite(logmel).all())
        self.assertTrue(np.isfinite(pcen).all())

    def test_batch_frontends_are_instance_standardized(self) -> None:
        config = DEFAULT_FRONTEND_CONFIG
        rng = np.random.default_rng(42)
        clips = rng.normal(0.0, 0.02, size=(3, config.clip_samples)).astype(np.float32)
        logmel, pcen = frontend_pair(clips)

        np.testing.assert_allclose(logmel.mean(axis=(1, 2)), 0.0, atol=2e-5)
        np.testing.assert_allclose(pcen.mean(axis=(1, 2)), 0.0, atol=2e-5)
        np.testing.assert_allclose(logmel.std(axis=(1, 2)), 1.0, atol=2e-5)
        np.testing.assert_allclose(pcen.std(axis=(1, 2)), 1.0, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
