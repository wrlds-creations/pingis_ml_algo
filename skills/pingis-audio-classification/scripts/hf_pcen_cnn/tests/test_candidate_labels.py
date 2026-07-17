from __future__ import annotations

import unittest

from hf_pcen_cnn.candidate_labels import (
    FOUR_CLASSES,
    ReviewedEvent,
    assign_candidates,
    canonical_four_class,
)


class CandidateLabelTests(unittest.TestCase):
    def test_canonical_class_names_are_idempotent(self) -> None:
        for label in FOUR_CLASSES:
            self.assertEqual(canonical_four_class(label), label)

    def test_one_candidate_owns_event_and_tail_is_ambiguous(self) -> None:
        assignments = assign_candidates(
            [980.0, 1_010.0, 1_500.0],
            [ReviewedEvent(1_000.0, "racket_bounce")],
            review_complete=True,
            explicit_negative_session=False,
        )

        self.assertEqual(assignments[0].disposition, "ambiguous")
        self.assertEqual(assignments[1].disposition, "positive")
        self.assertEqual(assignments[1].label, "racket_bounce")
        self.assertEqual(assignments[2].disposition, "hard_negative")

    def test_expected_count_without_timestamps_is_not_training_truth(self) -> None:
        assignments = assign_candidates(
            [500.0],
            [],
            review_complete=False,
            explicit_negative_session=False,
        )
        self.assertEqual(assignments[0].disposition, "unlabeled")

    def test_explicit_negative_session_produces_hard_negatives(self) -> None:
        assignments = assign_candidates(
            [500.0],
            [],
            review_complete=False,
            explicit_negative_session=True,
        )
        self.assertEqual(assignments[0].disposition, "hard_negative")
        self.assertEqual(assignments[0].label, "voice_music_noise")

    def test_four_class_mapping(self) -> None:
        self.assertEqual(canonical_four_class("forehand"), "racket_bounce")
        self.assertEqual(canonical_four_class("table_bounce"), "table_bounce")
        self.assertEqual(canonical_four_class("other_impact"), "floor_other_impact")
        self.assertEqual(canonical_four_class("voice_music_noise"), "voice_music_noise")


if __name__ == "__main__":
    unittest.main()
