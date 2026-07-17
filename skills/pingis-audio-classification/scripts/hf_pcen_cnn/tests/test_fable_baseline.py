from __future__ import annotations

import numpy as np

from hf_pcen_cnn.fable_baseline import (
    CLIP_PRE_SAMPLES,
    CLIP_SAMPLES,
    FableHgbRuntime,
    detect_deployed_fable_candidates,
    extract_fable_clip,
    replay_deployed_fable_timing,
)


def test_fable_gate_finds_transient_after_background_history() -> None:
    rng = np.random.default_rng(7)
    pcm = rng.normal(0.0, 0.0001, 22_050).astype(np.float32)
    pcm[11_025] += 0.8

    candidates = detect_deployed_fable_candidates(pcm)

    assert candidates
    assert abs(int(candidates[0]["onset_sample"]) - 11_025) <= 440


def test_fable_clip_keeps_onset_at_100_ms_with_padding() -> None:
    pcm = np.arange(100, dtype=np.float32)
    clip = extract_fable_clip(pcm, 5)

    assert len(clip) == CLIP_SAMPLES
    assert clip[CLIP_PRE_SAMPLES] == 5
    assert np.count_nonzero(clip[: CLIP_PRE_SAMPLES - 5]) == 0


def test_hgb_runtime_walks_flat_trees_and_softmaxes() -> None:
    runtime = FableHgbRuntime(
        {
            "labels": ["noise", "racket_bounce"],
            "feature_names": ["x"],
            "scaler_mean": [0.0],
            "scaler_std": [1.0],
            "baseline": [0.0, 0.0],
            "trees": [
                [[0, 0.0, 1, 2], [-1.0], [1.0]],
                [[0.0]],
            ],
        }
    )

    prediction = runtime.predict({"x": 1.0})

    assert prediction["label"] == "noise"
    assert prediction["confidence"] > 0.7


def test_timing_uses_quiet_and_loud_thresholds() -> None:
    rows = [
        {
            "onset_ms": 1_000,
            "frame_rms": 1.0,
            "predicted_label": "racket_bounce",
            "confidence": 0.8,
            "background_db": -60.0,
        },
        {
            "onset_ms": 2_000,
            "frame_rms": 1.0,
            "predicted_label": "racket_bounce",
            "confidence": 0.8,
            "background_db": -30.0,
        },
        {
            "onset_ms": 3_000,
            "frame_rms": 1.0,
            "predicted_label": "racket_bounce",
            "confidence": 0.86,
            "background_db": -30.0,
        },
    ]

    decisions = replay_deployed_fable_timing(rows)

    assert [decision.counted for decision in decisions] == [True, False, True]
    assert decisions[1].reason == "low_confidence_loud_bg"
