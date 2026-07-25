"""Tests for reviewed-round schema and broad hard-negative labels."""

from __future__ import annotations

from hf_pcen_cnn.build_reviewed_round_manifest import (
    negative_class_for_record,
)


def test_negative_class_preserves_supported_subclasses() -> None:
    assert (
        negative_class_for_record({"corrected_scenario_id": "surface_impacts"})
        == "table_bounce"
    )
    assert (
        negative_class_for_record({"corrected_scenario_id": "racket_drop"})
        == "floor_other_impact"
    )
    assert (
        negative_class_for_record({"corrected_scenario_id": "music_1_only"})
        == "voice_music_noise"
    )


def test_unknown_negative_scenario_uses_noise_catch_all() -> None:
    assert (
        negative_class_for_record({"corrected_scenario_id": "future_scenario"})
        == "voice_music_noise"
    )
