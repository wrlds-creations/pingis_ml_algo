from __future__ import annotations

import unittest

from hf_pcen_cnn.timing import replay_fable_timing


def candidate(onset_ms: float, rms: float, probability: float = 0.95, label: str = "racket_bounce") -> dict:
    return {
        "onset_ms": onset_ms,
        "hf_rms": rms,
        "predicted_label": label,
        "prob_racket_bounce": probability,
    }


class TimingTest(unittest.TestCase):
    def test_stronger_second_onset_moves_anchor_without_counting(self) -> None:
        decisions = replay_fable_timing(
            [candidate(1000, 1.0), candidate(1120, 2.0), candidate(1300, 1.5)],
            probability_threshold=0.65,
        )
        self.assertEqual([item.reason for item in decisions], ["counted", "same_bounce", "counted"])

    def test_quiet_echo_is_rejected(self) -> None:
        decisions = replay_fable_timing(
            [candidate(1000, 1.0), candidate(1280, 0.5)],
            probability_threshold=0.65,
        )
        self.assertEqual(decisions[1].reason, "echo_window")

    def test_fast_rebound_requires_high_probability(self) -> None:
        decisions = replay_fable_timing(
            [candidate(1000, 1.0), candidate(1200, 0.8, probability=0.85)],
            probability_threshold=0.65,
        )
        self.assertEqual(decisions[1].reason, "low_probability")
        self.assertTrue(decisions[1].fast_rebound)

    def test_non_racket_does_not_advance_anchor(self) -> None:
        decisions = replay_fable_timing(
            [
                candidate(1000, 1.0, label="voice_music_noise"),
                candidate(1050, 1.0),
            ],
            probability_threshold=0.65,
        )
        self.assertEqual([item.counted for item in decisions], [False, True])


if __name__ == "__main__":
    unittest.main()
