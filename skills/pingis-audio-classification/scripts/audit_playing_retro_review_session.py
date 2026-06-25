#!/usr/bin/env python3
"""Audit one reviewed playing-retro audio session against its saved candidates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def kind_of(item: dict[str, Any]) -> str:
    final_label = item.get("final_label")
    if final_label == "racket_contact":
        return "racket"
    if final_label == "not_racket_contact":
        return "table"
    if (
        item.get("event_type") == "racket_hit"
        or item.get("contact_kind") == "racket_bounce"
        or item.get("class_label") == "racket_bounce"
        or item.get("suggested_label") == "racket_contact"
    ):
        return "racket"
    if (
        item.get("event_type") == "bounce"
        or item.get("not_racket_kind") == "table_bounce"
        or item.get("class_label") == "table_bounce"
        or item.get("surface_label") == "table_bounce"
        or item.get("suggested_label") == "not_racket_contact"
    ):
        return "table"
    return "other"


def confidence_of(item: dict[str, Any]) -> float:
    prediction = item.get("playing_retro_prediction") or {}
    for key in ("confidence", "contact_confidence", "surface_confidence"):
        value = prediction.get(key) if key == "confidence" else item.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def count_by(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(item.get(key, "<null>")) for item in items).items()))


def count_by_kind(items: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(kind_of(item) for item in items).items()))


def sorted_by_time(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: float(item.get("timestamp_ms", 0)))


def dedupe_same_label_candidates(
    candidates: list[dict[str, Any]],
    gap_ms: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for candidate in sorted_by_time(candidates):
        candidate_kind = kind_of(candidate)
        duplicate_index = -1
        for index in range(len(kept) - 1, -1, -1):
            existing = kept[index]
            if abs(float(existing["timestamp_ms"]) - float(candidate["timestamp_ms"])) > gap_ms:
                break
            if kind_of(existing) == candidate_kind:
                duplicate_index = index
                break
        if duplicate_index >= 0:
            removed.append(kept[duplicate_index])
            kept[duplicate_index] = candidate
        else:
            kept.append(candidate)
    return sorted_by_time(kept), sorted_by_time(removed)


def prediction_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": candidate.get("id"),
            "timestamp_ms": float(candidate.get("timestamp_ms", 0)),
            "kind": kind_of(candidate),
            "confidence": confidence_of(candidate),
            "source": candidate.get("playing_retro_candidate_source"),
        }
        for candidate in sorted_by_time(candidates)
    ]


def truth_rows(markers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "id": marker.get("id"),
            "timestamp_ms": float(marker.get("timestamp_ms", 0)),
            "kind": kind_of(marker),
            "source": marker.get("source"),
        }
        for index, marker in enumerate(sorted_by_time(markers))
    ]


def evaluate_predictions(
    predictions: list[dict[str, Any]],
    truths: list[dict[str, Any]],
    threshold_ms: int,
) -> dict[str, Any]:
    used_truths: set[int] = set()
    true_positive: list[dict[str, Any]] = []
    wrong_class: list[dict[str, Any]] = []
    false_positive: list[dict[str, Any]] = []

    for prediction in predictions:
        nearby = sorted(
            (
                {**truth, "dt_ms": abs(float(truth["timestamp_ms"]) - float(prediction["timestamp_ms"]))}
                for truth in truths
                if abs(float(truth["timestamp_ms"]) - float(prediction["timestamp_ms"])) <= threshold_ms
            ),
            key=lambda item: item["dt_ms"],
        )
        same_kind = next(
            (
                truth for truth in nearby
                if truth["kind"] == prediction["kind"] and int(truth["index"]) not in used_truths
            ),
            None,
        )
        if same_kind is not None:
            used_truths.add(int(same_kind["index"]))
            true_positive.append({"prediction": prediction, "truth": same_kind})
            continue

        any_kind = next((truth for truth in nearby if int(truth["index"]) not in used_truths), None)
        if any_kind is not None:
            used_truths.add(int(any_kind["index"]))
            wrong_class.append({"prediction": prediction, "truth": any_kind})
        else:
            false_positive.append(prediction)

    missed = [truth for truth in truths if int(truth["index"]) not in used_truths]
    return {
        "prediction_count": len(predictions),
        "true_positive": len(true_positive),
        "wrong_class": len(wrong_class),
        "false_positive": len(false_positive),
        "missed": len(missed),
        "true_positive_by_kind": dict(sorted(Counter(item["truth"]["kind"] for item in true_positive).items())),
        "false_positive_by_kind": dict(sorted(Counter(item["kind"] for item in false_positive).items())),
        "missed_by_kind": dict(sorted(Counter(item["kind"] for item in missed).items())),
        "wrong_class_rows": [
            {
                "prediction_ms": round(float(item["prediction"]["timestamp_ms"])),
                "prediction_kind": item["prediction"]["kind"],
                "truth_ms": round(float(item["truth"]["timestamp_ms"])),
                "truth_kind": item["truth"]["kind"],
                "dt_ms": round(float(item["truth"]["dt_ms"])),
            }
            for item in wrong_class
        ],
        "false_positive_rows": [
            {
                "timestamp_ms": round(float(item["timestamp_ms"])),
                "kind": item["kind"],
                "confidence": round(float(item["confidence"]), 3),
                "id": item["id"],
            }
            for item in false_positive
        ],
        "missed_rows": [
            {
                "timestamp_ms": round(float(item["timestamp_ms"])),
                "kind": item["kind"],
                "source": item["source"],
                "id": item["id"],
            }
            for item in missed
        ],
    }


def close_adjacent_pairs(markers: list[dict[str, Any]], gap_ms: int) -> list[dict[str, Any]]:
    rows = sorted_by_time(markers)
    pairs: list[dict[str, Any]] = []
    for previous, current in zip(rows, rows[1:]):
        gap = float(current.get("timestamp_ms", 0)) - float(previous.get("timestamp_ms", 0))
        if gap <= gap_ms:
            pairs.append(
                {
                    "gap_ms": round(gap),
                    "previous_ms": round(float(previous.get("timestamp_ms", 0))),
                    "current_ms": round(float(current.get("timestamp_ms", 0))),
                    "previous_kind": kind_of(previous),
                    "current_kind": kind_of(current),
                    "previous_source": previous.get("source"),
                    "current_source": current.get("source"),
                    "previous_id": previous.get("id"),
                    "current_id": current.get("id"),
                }
            )
    return pairs


def build_audit(session_path: Path, event_index: int, dedupe_gap_ms: int, match_threshold_ms: int) -> dict[str, Any]:
    session = json.loads(session_path.read_text(encoding="utf-8"))
    events = session.get("events") or []
    if event_index >= len(events):
        raise IndexError(f"event_index {event_index} out of range for {len(events)} events")

    event = events[event_index]
    markers = event.get("review", {}).get("markers") or []
    candidates = event.get("model_candidates") or []
    review_candidates = [candidate for candidate in candidates if candidate.get("review_relevant")]
    linked_candidate_ids = {
        marker.get("linked_candidate_id")
        for marker in markers
        if marker.get("linked_candidate_id")
    }
    unlinked_review_candidates = [
        candidate for candidate in review_candidates
        if candidate.get("id") not in linked_candidate_ids
    ]

    manual_markers = [marker for marker in markers if marker.get("source") == "manual"]
    auto_markers = [marker for marker in markers if marker.get("source") == "auto"]
    candidates_by_id = {candidate.get("id"): candidate for candidate in candidates}
    label_changes: list[dict[str, Any]] = []
    nudges: list[dict[str, Any]] = []
    for marker in auto_markers:
        candidate = candidates_by_id.get(marker.get("linked_candidate_id"))
        if not candidate:
            continue
        delta = float(marker.get("timestamp_ms", 0)) - float(candidate.get("timestamp_ms", 0))
        marker_kind = kind_of(marker)
        candidate_kind = kind_of(candidate)
        if marker_kind != candidate_kind:
            label_changes.append(
                {
                    "timestamp_ms": round(float(marker.get("timestamp_ms", 0))),
                    "marker_kind": marker_kind,
                    "candidate_kind": candidate_kind,
                    "confidence": round(confidence_of(candidate), 3),
                    "candidate_id": candidate.get("id"),
                }
            )
        if abs(delta) > 0:
            nudges.append(
                {
                    "marker_ms": round(float(marker.get("timestamp_ms", 0))),
                    "candidate_ms": round(float(candidate.get("timestamp_ms", 0))),
                    "delta_ms": round(delta),
                    "kind": marker_kind,
                    "candidate_id": candidate.get("id"),
                }
            )

    deduped_candidates, removed_candidates = dedupe_same_label_candidates(
        review_candidates,
        dedupe_gap_ms,
    )
    truths = truth_rows(markers)
    baseline_predictions = prediction_rows(review_candidates)
    deduped_predictions = prediction_rows(deduped_candidates)

    return {
        "session_path": str(session_path),
        "session_id": session_path.stem,
        "event_index": event_index,
        "duration_ms": event.get("duration_ms"),
        "wav_filename": event.get("wav_filename"),
        "video_filename": event.get("video_recording", {}).get("filename") or event.get("video_filename"),
        "markers": {
            "total": len(markers),
            "by_kind": count_by_kind(markers),
            "by_source": count_by(markers, "source"),
            "by_status": count_by(markers, "review_status"),
            "manual_total": len(manual_markers),
            "manual_by_kind": count_by_kind(manual_markers),
            "manual_rows": [
                {
                    "timestamp_ms": round(float(marker.get("timestamp_ms", 0))),
                    "kind": kind_of(marker),
                    "id": marker.get("id"),
                }
                for marker in sorted_by_time(manual_markers)
            ],
            "auto_label_changes": label_changes,
            "auto_nudges": nudges,
            "close_adjacent_le_180_ms": close_adjacent_pairs(markers, 180),
        },
        "candidates": {
            "total": len(candidates),
            "review_relevant": len(review_candidates),
            "review_relevant_by_kind": count_by_kind(review_candidates),
            "unlinked_review_relevant": len(unlinked_review_candidates),
            "unlinked_review_relevant_by_kind": count_by_kind(unlinked_review_candidates),
        },
        "dedupe": {
            "gap_ms": dedupe_gap_ms,
            "baseline_review_candidate_markers": len(review_candidates),
            "deduped_review_markers": len(deduped_candidates),
            "removed_total": len(removed_candidates),
            "removed_by_kind": count_by_kind(removed_candidates),
            "removed_rows": [
                {
                    "timestamp_ms": round(float(candidate.get("timestamp_ms", 0))),
                    "kind": kind_of(candidate),
                    "confidence": round(confidence_of(candidate), 3),
                    "id": candidate.get("id"),
                    "source": candidate.get("playing_retro_candidate_source"),
                }
                for candidate in removed_candidates
            ],
        },
        "evaluation": {
            "match_threshold_ms": match_threshold_ms,
            "baseline_t0019": evaluate_predictions(baseline_predictions, truths, match_threshold_ms),
            "candidate_t0020": evaluate_predictions(deduped_predictions, truths, match_threshold_ms),
        },
    }


def print_summary(audit: dict[str, Any]) -> None:
    baseline = audit["evaluation"]["baseline_t0019"]
    candidate = audit["evaluation"]["candidate_t0020"]
    print(f"# Playing Retro Review Audit: {audit['session_id']}")
    print()
    print(f"- Duration: {audit['duration_ms']} ms")
    print(f"- Markers: {audit['markers']['total']} {audit['markers']['by_kind']}")
    print(f"- Manual additions: {audit['markers']['manual_total']} {audit['markers']['manual_by_kind']}")
    print(f"- Auto label changes: {len(audit['markers']['auto_label_changes'])}")
    print(f"- Auto nudges: {len(audit['markers']['auto_nudges'])}")
    print(f"- Review candidates: {audit['candidates']['review_relevant']} {audit['candidates']['review_relevant_by_kind']}")
    print(f"- Dedupe removed: {audit['dedupe']['removed_total']} {audit['dedupe']['removed_by_kind']}")
    print()
    print("| Replay | Predictions | TP | Wrong | FP | Missed |")
    print("|---|---:|---:|---:|---:|---:|")
    print(
        f"| T0019 baseline | {baseline['prediction_count']} | {baseline['true_positive']} | "
        f"{baseline['wrong_class']} | {baseline['false_positive']} | {baseline['missed']} |"
    )
    print(
        f"| T0020 dedupe | {candidate['prediction_count']} | {candidate['true_positive']} | "
        f"{candidate['wrong_class']} | {candidate['false_positive']} | {candidate['missed']} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_json", type=Path)
    parser.add_argument("--event-index", type=int, default=0)
    parser.add_argument("--dedupe-gap-ms", type=int, default=80)
    parser.add_argument("--match-threshold-ms", type=int, default=80)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    audit = build_audit(
        args.session_json,
        args.event_index,
        args.dedupe_gap_ms,
        args.match_threshold_ms,
    )
    print_summary(audit)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
        print()
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
