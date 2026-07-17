"""Replay the existing Fable same-bounce and echo timing policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TimingConfig:
    same_bounce_ms: float = 250.0
    fast_rebound_min_ms: float = 150.0
    echo_ms: float = 300.0
    echo_ratio: float = 0.6
    stronger_anchor_ratio: float = 1.1
    fast_rebound_probability: float = 0.9


@dataclass(frozen=True)
class CandidateDecision:
    candidate_index: int
    onset_ms: float
    counted: bool
    reason: str
    fast_rebound: bool = False


def replay_fable_timing(
    candidates: Iterable[dict],
    *,
    probability_threshold: float,
    config: TimingConfig = TimingConfig(),
) -> list[CandidateDecision]:
    """Apply the timing/window behavior from ``FableCounter``.

    Each candidate must provide ``onset_ms``, ``hf_rms``,
    ``predicted_label``, and ``prob_racket_bounce``. The HF onset RMS is used
    as the strength signal because that is the candidate generator's native
    frame measurement in this experiment.
    """
    ordered = sorted(candidates, key=lambda row: float(row["onset_ms"]))
    decisions: list[CandidateDecision] = []
    last_counted: tuple[float, float] | None = None

    for candidate_index, row in enumerate(ordered):
        onset_ms = float(row["onset_ms"])
        frame_rms = max(float(row["hf_rms"]), 0.0)
        fast_rebound = False

        if last_counted is not None:
            since_counted = onset_ms - last_counted[0]
            rms_ratio = frame_rms / max(last_counted[1], 1e-12)
            if since_counted <= config.same_bounce_ms:
                if rms_ratio >= config.stronger_anchor_ratio:
                    last_counted = (onset_ms, frame_rms)
                    decisions.append(
                        CandidateDecision(candidate_index, onset_ms, False, "same_bounce")
                    )
                    continue
                if rms_ratio <= config.echo_ratio:
                    decisions.append(
                        CandidateDecision(candidate_index, onset_ms, False, "echo_window")
                    )
                    continue
                if since_counted < config.fast_rebound_min_ms:
                    decisions.append(
                        CandidateDecision(candidate_index, onset_ms, False, "same_bounce")
                    )
                    continue
                fast_rebound = True
            elif since_counted <= config.echo_ms and rms_ratio <= config.echo_ratio:
                decisions.append(
                    CandidateDecision(candidate_index, onset_ms, False, "echo_window")
                )
                continue

        threshold = probability_threshold
        if fast_rebound:
            threshold = max(threshold, config.fast_rebound_probability)
        is_racket = str(row["predicted_label"]) == "racket_bounce"
        probability = float(row["prob_racket_bounce"])
        if not is_racket:
            decisions.append(
                CandidateDecision(
                    candidate_index, onset_ms, False, "not_racket", fast_rebound
                )
            )
            continue
        if probability < threshold:
            decisions.append(
                CandidateDecision(
                    candidate_index, onset_ms, False, "low_probability", fast_rebound
                )
            )
            continue

        last_counted = (onset_ms, frame_rms)
        decisions.append(
            CandidateDecision(candidate_index, onset_ms, True, "counted", fast_rebound)
        )

    return decisions
