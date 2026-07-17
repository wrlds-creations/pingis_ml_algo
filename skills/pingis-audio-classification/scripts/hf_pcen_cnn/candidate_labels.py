"""Ground-truth mapping and leakage-safe candidate assignment."""

from __future__ import annotations

from dataclasses import dataclass


FOUR_CLASSES = (
    "racket_bounce",
    "table_bounce",
    "floor_other_impact",
    "voice_music_noise",
)


@dataclass(frozen=True)
class ReviewedEvent:
    timestamp_ms: float
    label: str
    trainable: bool = True


@dataclass(frozen=True)
class CandidateAssignment:
    candidate_index: int
    onset_ms: float
    disposition: str
    label: str | None
    matched_event_index: int | None
    distance_ms: float | None
    reason: str


def canonical_four_class(raw_label: str) -> str | None:
    label = str(raw_label or "").strip().lower()
    if label in {
        "racket_bounce",
        "racket_contact",
        "forehand",
        "backhand",
        "forehand_hit",
        "backhand_hit",
        "racket_hit",
    }:
        return "racket_bounce"
    if label == "table_bounce":
        return "table_bounce"
    if label in {
        "floor_other_impact",
        "floor_bounce",
        "other_impact",
        "catch_after_sound",
    }:
        return "floor_other_impact"
    if label in {
        "noise",
        "voice_music_noise",
        "speech_music_noise",
        "talking",
        "music",
        "hard_negative",
        "not_racket_contact",
    }:
        return "voice_music_noise"
    return None


def assign_candidates(
    candidate_onsets_ms: list[float],
    reviewed_events: list[ReviewedEvent],
    *,
    match_ms: float = 140.0,
    ambiguity_ms: float = 300.0,
    review_complete: bool,
    explicit_negative_session: bool,
) -> list[CandidateAssignment]:
    """Assign candidates without turning bounce tails into false negatives.

    Trainable candidate/event pairs are greedily matched by smallest absolute
    time distance. Only one candidate can own a reviewed event. Additional
    candidates around that event are ambiguous, not negative.
    """
    onsets = [float(value) for value in candidate_onsets_ms]
    trainable_indices = [
        index
        for index, event in enumerate(reviewed_events)
        if event.trainable and canonical_four_class(event.label) is not None
    ]
    pairs: list[tuple[float, int, int]] = []
    for candidate_index, onset in enumerate(onsets):
        for event_index in trainable_indices:
            distance = abs(onset - reviewed_events[event_index].timestamp_ms)
            if distance <= match_ms:
                pairs.append((distance, candidate_index, event_index))
    pairs.sort(key=lambda item: (item[0], item[1], item[2]))

    matched_candidates: dict[int, tuple[int, float]] = {}
    matched_events: set[int] = set()
    for distance, candidate_index, event_index in pairs:
        if candidate_index in matched_candidates or event_index in matched_events:
            continue
        matched_candidates[candidate_index] = (event_index, distance)
        matched_events.add(event_index)

    assignments: list[CandidateAssignment] = []
    for candidate_index, onset in enumerate(onsets):
        if candidate_index in matched_candidates:
            event_index, distance = matched_candidates[candidate_index]
            label = canonical_four_class(reviewed_events[event_index].label)
            assignments.append(
                CandidateAssignment(
                    candidate_index,
                    onset,
                    "positive",
                    label,
                    event_index,
                    distance,
                    "nearest_reviewed_event",
                )
            )
            continue

        nearest_index: int | None = None
        nearest_distance: float | None = None
        for event_index, event in enumerate(reviewed_events):
            distance = abs(onset - event.timestamp_ms)
            if nearest_distance is None or distance < nearest_distance:
                nearest_index = event_index
                nearest_distance = distance

        if nearest_distance is not None and nearest_distance <= ambiguity_ms:
            reason = (
                "secondary_candidate_near_event"
                if nearest_index in matched_events
                else "near_reviewed_or_nontrainable_event"
            )
            assignments.append(
                CandidateAssignment(
                    candidate_index,
                    onset,
                    "ambiguous",
                    None,
                    nearest_index,
                    nearest_distance,
                    reason,
                )
            )
            continue

        if explicit_negative_session or review_complete:
            assignments.append(
                CandidateAssignment(
                    candidate_index,
                    onset,
                    "hard_negative",
                    "voice_music_noise",
                    None,
                    nearest_distance,
                    "explicit_negative_session" if explicit_negative_session else "outside_reviewed_events",
                )
            )
        else:
            assignments.append(
                CandidateAssignment(
                    candidate_index,
                    onset,
                    "unlabeled",
                    None,
                    None,
                    nearest_distance,
                    "session_has_counts_but_no_timestamps",
                )
            )
    return assignments
