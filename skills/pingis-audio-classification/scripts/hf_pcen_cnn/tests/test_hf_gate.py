from __future__ import annotations

import unittest

import numpy as np

from hf_pcen_cnn.hf_gate import HFGateConfig, detect_hf_candidates, extract_full_band_clip


class HFGateTests(unittest.TestCase):
    def test_high_frequency_burst_triggers_near_onset(self) -> None:
        sr = 22_050
        rng = np.random.default_rng(7)
        signal = rng.normal(0.0, 1e-5, sr).astype(np.float32)
        start = int(0.5 * sr)
        length = int(0.008 * sr)
        t = np.arange(length) / sr
        signal[start : start + length] += (0.08 * np.sin(2 * np.pi * 9_000 * t)).astype(np.float32)

        candidates = detect_hf_candidates(
            signal,
            sr,
            HFGateConfig(cutoff_hz=7_000, onset_ratio=3.0, cooldown_ms=80),
        )

        self.assertTrue(candidates)
        self.assertLess(abs(float(candidates[0]["onset_ms"]) - 500.0), 5.0)

    def test_sustained_low_frequency_tone_does_not_pass_hf_gate(self) -> None:
        sr = 22_050
        t = np.arange(sr) / sr
        signal = (0.1 * np.sin(2 * np.pi * 800 * t)).astype(np.float32)

        candidates = detect_hf_candidates(signal, sr, HFGateConfig(cutoff_hz=7_000))

        self.assertEqual(candidates, [])

    def test_clip_is_full_band_and_onset_anchored(self) -> None:
        sr = 22_050
        signal = np.arange(sr, dtype=np.float32)
        onset = int(0.2 * sr)
        clip = extract_full_band_clip(signal, onset, sr, pre_ms=100, total_ms=400)

        self.assertEqual(len(clip), int(0.4 * sr))
        self.assertEqual(clip[int(0.1 * sr)], signal[onset])


if __name__ == "__main__":
    unittest.main()
