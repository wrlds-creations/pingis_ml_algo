"""Exact offline replica of the deployed Collector Fable audio path."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from scipy.signal import sosfilt


TARGET_SAMPLE_RATE = 22_050
FRAME_SAMPLES = 220
BACKGROUND_FRAMES = 30
CLIP_PRE_SAMPLES = 2_205
CLIP_POST_SAMPLES = 4_410
CLIP_SAMPLES = CLIP_PRE_SAMPLES + CLIP_POST_SAMPLES

# AudioStreamModule.kt stores each SOS row as b0, b1, b2, a1, a2.
FABLE_BANDPASS_SOS = np.asarray(
    [
        [0.09331299315653971, 0.18662598631307942, 0.09331299315653971, 1.0, 0.19676635870899223, 0.09988498828008321],
        [1.0, -2.0, 1.0, 1.0, -1.1925622397360882, 0.39614858444118883],
        [1.0, 2.0, 1.0, 1.0, 0.6137123805223851, 0.5737584318188238],
        [1.0, -2.0, 1.0, 1.0, -1.6102285405185548, 0.7781414737694868],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class FableGateConfig:
    onset_ratio: float = 1.5
    absolute_min_rms: float = 0.0015
    retrigger_ms: float = 120.0


@dataclass(frozen=True)
class FableTimingConfig:
    quiet_confidence: float = 0.65
    loud_confidence: float = 0.85
    loud_background_db: float = -36.0
    same_bounce_ms: float = 250.0
    group_ms: float = 80.0
    fast_rebound_min_ms: float = 150.0
    echo_ms: float = 300.0
    echo_ratio: float = 0.6
    stronger_anchor_ratio: float = 1.1
    fast_rebound_confidence: float = 0.9


@dataclass(frozen=True)
class FableDecision:
    candidate_index: int
    onset_ms: float
    counted: bool
    reason: str
    fast_rebound: bool = False


def detect_deployed_fable_candidates(
    pcm: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    config: FableGateConfig = FableGateConfig(),
) -> list[dict[str, float | int]]:
    """Run the native causal 1.5-7 kHz gate sample-for-sample offline.

    The native implementation filters every frame, including cooldown frames,
    but updates its rolling background only when cooldown has elapsed. Filtering
    the full signal once with ``sosfilt`` is equivalent to its persistent
    direct-form-II biquad state.
    """
    if sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError(f"Fable gate requires {TARGET_SAMPLE_RATE} Hz audio")
    signal = np.asarray(pcm, dtype=np.float64).reshape(-1)
    filtered = sosfilt(FABLE_BANDPASS_SOS, signal)
    background = np.zeros(BACKGROUND_FRAMES, dtype=np.float64)
    background_index = 0
    background_filled = False
    retrigger_samples = int(round(config.retrigger_ms * sample_rate / 1000.0))
    last_onset_sample: int | None = None
    output: list[dict[str, float | int]] = []

    for onset_sample in range(0, len(filtered) - FRAME_SAMPLES + 1, FRAME_SAMPLES):
        frame = filtered[onset_sample : onset_sample + FRAME_SAMPLES]
        frame_rms = float(np.sqrt(np.mean(frame * frame)))
        in_cooldown = (
            last_onset_sample is not None
            and onset_sample - last_onset_sample < retrigger_samples
        )
        if in_cooldown:
            continue

        if background_filled:
            background_rms = float(background.mean())
        elif background_index > 0:
            background_rms = float(background[:background_index].mean())
        else:
            background_rms = frame_rms
        threshold = max(
            background_rms * config.onset_ratio,
            config.absolute_min_rms,
        )

        if frame_rms >= threshold:
            output.append(
                {
                    "onset_sample": onset_sample,
                    "onset_ms": onset_sample * 1000.0 / sample_rate,
                    "frame_rms": frame_rms,
                    "background_rms": background_rms,
                    "threshold": threshold,
                }
            )
            last_onset_sample = onset_sample
            background.fill(background_rms * 2.0)
            background_index = 0
            background_filled = True
        else:
            background[background_index % BACKGROUND_FRAMES] = frame_rms
            background_index += 1
            if background_index >= BACKGROUND_FRAMES:
                background_filled = True

    return output


def extract_fable_clip(
    pcm: np.ndarray,
    onset_sample: int,
) -> np.ndarray:
    """Extract the deployed 100 ms pre-onset + 200 ms post-onset clip."""
    signal = np.asarray(pcm, dtype=np.float32).reshape(-1)
    start = int(onset_sample) - CLIP_PRE_SAMPLES
    end = int(onset_sample) + CLIP_POST_SAMPLES
    clip = np.zeros(CLIP_SAMPLES, dtype=np.float32)
    source_start = max(0, start)
    source_end = min(len(signal), end)
    if source_end > source_start:
        destination_start = source_start - start
        clip[destination_start : destination_start + source_end - source_start] = signal[
            source_start:source_end
        ]
    return clip


class FableHgbRuntime:
    """Portable evaluator for the exported four-class HGB JSON model."""

    def __init__(self, payload: Mapping[str, object]):
        self.metadata = dict(payload.get("metadata", {}))
        self.labels = tuple(str(value) for value in payload["labels"])
        self.feature_names = tuple(str(value) for value in payload["feature_names"])
        self.mean = np.asarray(payload["scaler_mean"], dtype=np.float64)
        self.std = np.asarray(payload["scaler_std"], dtype=np.float64)
        self.baseline = np.asarray(payload["baseline"], dtype=np.float64)
        self.trees = payload["trees"]
        if not (
            len(self.feature_names) == len(self.mean) == len(self.std)
            and len(self.baseline) == len(self.labels)
        ):
            raise ValueError("Invalid Fable model dimensions")

    @classmethod
    def from_path(cls, path: Path) -> "FableHgbRuntime":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    @staticmethod
    def _tree_value(tree: list, features: np.ndarray) -> float:
        node_index = 0
        while True:
            node = tree[node_index]
            if len(node) == 1:
                return float(node[0])
            feature_index, threshold, left, right = node
            node_index = int(left if features[int(feature_index)] <= float(threshold) else right)

    def predict(self, features: Mapping[str, float]) -> dict[str, object]:
        vector = np.asarray(
            [float(features.get(name, 0.0)) for name in self.feature_names],
            dtype=np.float64,
        )
        vector = np.where(np.isfinite(vector), vector, 0.0)
        safe_std = np.where(np.abs(self.std) > 1e-12, self.std, 1.0)
        scaled = (vector - self.mean) / safe_std
        logits = self.baseline.copy()
        class_count = len(self.labels)
        for tree_index, tree in enumerate(self.trees):
            logits[tree_index % class_count] += self._tree_value(tree, scaled)
        shifted = logits - float(logits.max())
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum()
        best_index = int(np.argmax(probabilities))
        return {
            "label": self.labels[best_index],
            "confidence": float(probabilities[best_index]),
            "probabilities": {
                label: float(probabilities[index])
                for index, label in enumerate(self.labels)
            },
        }


def extract_and_predict_fable(
    runtime: FableHgbRuntime,
    clip: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> tuple[dict[str, float], dict[str, object]]:
    """Extract the deployed 83 features and run the exported HGB model."""
    noise_robust_dir = Path(__file__).resolve().parents[1] / "noise_robust"
    noise_robust_path = str(noise_robust_dir)
    if noise_robust_path not in sys.path:
        sys.path.insert(0, noise_robust_path)
    from noise_robust.nr_features import extract_all_features

    features = extract_all_features(np.asarray(clip, dtype=np.float32), sample_rate)
    return features, runtime.predict(features)


def replay_deployed_fable_timing(
    candidates: Iterable[Mapping[str, object]],
    config: FableTimingConfig = FableTimingConfig(),
) -> list[FableDecision]:
    """Apply the deployed Fable model threshold and timing decisions."""
    ordered = sorted(candidates, key=lambda row: float(row["onset_ms"]))
    decisions: list[FableDecision] = []
    last_counted: tuple[float, float] | None = None
    group_start_ms: float | None = None

    for candidate_index, row in enumerate(ordered):
        onset_ms = float(row["onset_ms"])
        frame_rms = max(float(row["frame_rms"]), 0.0)
        fast_rebound = False
        if last_counted is not None:
            since_counted = onset_ms - last_counted[0]
            rms_ratio = frame_rms / max(last_counted[1], 1e-12)
            if since_counted <= config.same_bounce_ms:
                if rms_ratio >= config.stronger_anchor_ratio:
                    last_counted = (onset_ms, frame_rms)
                    group_start_ms = onset_ms
                    decisions.append(FableDecision(candidate_index, onset_ms, False, "same_bounce"))
                    continue
                if rms_ratio <= config.echo_ratio:
                    decisions.append(FableDecision(candidate_index, onset_ms, False, "echo_window"))
                    continue
                if since_counted < config.fast_rebound_min_ms:
                    decisions.append(FableDecision(candidate_index, onset_ms, False, "same_bounce"))
                    continue
                fast_rebound = True
            else:
                if group_start_ms is not None and onset_ms - group_start_ms <= config.group_ms:
                    decisions.append(FableDecision(candidate_index, onset_ms, False, "group_window"))
                    continue
                if since_counted <= config.echo_ms and rms_ratio <= config.echo_ratio:
                    decisions.append(FableDecision(candidate_index, onset_ms, False, "echo_window"))
                    continue

        predicted_label = str(row["predicted_label"])
        confidence = float(row["confidence"])
        background_db = float(row.get("background_db", -100.0))
        loud = background_db >= config.loud_background_db
        threshold = config.loud_confidence if loud else config.quiet_confidence
        if fast_rebound:
            threshold = max(threshold, config.fast_rebound_confidence)
        if predicted_label != "racket_bounce":
            decisions.append(FableDecision(candidate_index, onset_ms, False, "not_racket", fast_rebound))
            continue
        if confidence < threshold:
            reason = "low_confidence_loud_bg" if loud else "low_confidence"
            decisions.append(FableDecision(candidate_index, onset_ms, False, reason, fast_rebound))
            continue

        last_counted = (onset_ms, frame_rms)
        group_start_ms = onset_ms
        decisions.append(FableDecision(candidate_index, onset_ms, True, "counted", fast_rebound))

    return decisions
