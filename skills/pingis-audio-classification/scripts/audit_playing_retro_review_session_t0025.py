#!/usr/bin/env python3
"""Audit the T0024-reviewed 2026-06-04 playing-retro session.

This script is analysis-only. It does not train, export app JSON, build an APK,
or change studs_live.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_SESSION = ROOT_DIR / "data" / "audio" / "raw" / "audio_session_2026-06-04_001.json"
EVAL_DIR = ROOT_DIR / "data" / "audio" / "models" / "evaluations"
DEFAULT_JSON = EVAL_DIR / "playing_retro_t0025_2026_06_04_audit.json"
DEFAULT_MD = EVAL_DIR / "playing_retro_t0025_2026_06_04_audit.md"
DEFAULT_MANUAL_CSV = EVAL_DIR / "playing_retro_t0025_2026_06_04_manual_additions.csv"
DEFAULT_GAPS_CSV = EVAL_DIR / "playing_retro_t0025_2026_06_04_close_gaps.csv"

T0024_RACKET_THRESHOLD = 0.0
T0024_TABLE_THRESHOLD = 0.5
SAME_LABEL_DEDUPE_MS = 80
MATCH_MS = 80
NEAR_CANDIDATE_MS = 180
GAP_CANDIDATE_MS = 250
TARGET_LABELS = {"racket_contact", "table_bounce"}


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def timestamp_ms(item: dict[str, Any]) -> float:
    return float(item.get("timestamp_ms") or 0.0)


def sorted_by_time(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=timestamp_ms)


def marker_kind(marker: dict[str, Any] | None) -> str:
    if not marker:
        return "other"
    final_label = marker.get("final_label")
    class_label = marker.get("class_label")
    if final_label == "racket_contact" or class_label == "racket_bounce":
        return "racket"
    if (
        final_label == "not_racket_contact"
        or class_label == "table_bounce"
        or marker.get("not_racket_kind") == "table_bounce"
        or marker.get("surface_label") == "table_bounce"
    ):
        return "table"
    return "other"


def prediction_label(candidate: dict[str, Any] | None) -> str:
    if not candidate:
        return ""
    prediction = candidate.get("playing_retro_prediction") or {}
    label = prediction.get("label")
    if label in {"non_target", "racket_contact", "table_bounce"}:
        return str(label)
    class_label = candidate.get("class_label")
    if class_label == "racket_bounce":
        return "racket_contact"
    if class_label == "table_bounce":
        return "table_bounce"
    return "non_target"


def label_kind(label: str) -> str:
    if label == "racket_contact":
        return "racket"
    if label == "table_bounce":
        return "table"
    return "non_target"


def wanted_label(kind: str) -> str:
    if kind == "racket":
        return "racket_contact"
    if kind == "table":
        return "table_bounce"
    return ""


def confidence(candidate: dict[str, Any] | None) -> float:
    if not candidate:
        return 0.0
    prediction = candidate.get("playing_retro_prediction") or {}
    value = prediction.get("confidence")
    if isinstance(value, (int, float)):
        return float(value)
    for key in ("contact_confidence", "surface_confidence"):
        value = candidate.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def probabilities(candidate: dict[str, Any] | None) -> dict[str, float]:
    if not candidate:
        return {}
    raw = (candidate.get("playing_retro_prediction") or {}).get("probabilities") or {}
    return {
        label: round(float(raw.get(label, 0.0)), 4)
        for label in ("non_target", "racket_contact", "table_bounce")
    }


def candidate_source(candidate: dict[str, Any] | None) -> str:
    if not candidate:
        return ""
    return str(candidate.get("playing_retro_candidate_source") or "")


def candidate_snapshot(candidate: dict[str, Any] | None, reference_ms: float | None = None) -> dict[str, Any]:
    if not candidate:
        return {
            "id": "",
            "timestamp_ms": None,
            "dt_ms": None,
            "source": "",
            "review_relevant": None,
            "prediction": "",
            "confidence": None,
            "probabilities": {},
        }
    candidate_ms = timestamp_ms(candidate)
    return {
        "id": candidate.get("id", ""),
        "timestamp_ms": int(round(candidate_ms)),
        "dt_ms": None if reference_ms is None else int(round(candidate_ms - reference_ms)),
        "source": candidate_source(candidate),
        "review_relevant": bool(candidate.get("review_relevant")),
        "prediction": prediction_label(candidate),
        "confidence": round(confidence(candidate), 4),
        "probabilities": probabilities(candidate),
    }


def nearest(timestamp: float, items: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float | None]:
    if not items:
        return None, None
    candidate = min(items, key=lambda item: abs(timestamp_ms(item) - timestamp))
    return candidate, timestamp_ms(candidate) - timestamp


def count_by(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(item.get(key, "<null>")) for item in items).items()))


def count_by_kind(items: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(marker_kind(item) for item in items).items()))


def count_candidates_by_prediction(items: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(prediction_label(item) for item in items).items()))


def clear_review_threshold(candidate: dict[str, Any]) -> bool:
    label = prediction_label(candidate)
    value = confidence(candidate)
    if label == "racket_contact":
        return value >= T0024_RACKET_THRESHOLD
    if label == "table_bounce":
        return value >= T0024_TABLE_THRESHOLD
    return False


def classify_manual_reason(
    marker: dict[str, Any],
    candidates: list[dict[str, Any]],
    linked_candidate_ids: set[str],
) -> tuple[str, dict[str, Any] | None]:
    marker_ms = timestamp_ms(marker)
    kind = marker_kind(marker)
    expected_label = wanted_label(kind)
    near = [
        candidate
        for candidate in candidates
        if abs(timestamp_ms(candidate) - marker_ms) <= NEAR_CANDIDATE_MS
    ]
    near.sort(key=lambda candidate: abs(timestamp_ms(candidate) - marker_ms))

    if near:
        correct = [candidate for candidate in near if prediction_label(candidate) == expected_label]
        if correct:
            candidate = correct[0]
            if candidate.get("review_relevant"):
                if str(candidate.get("id")) not in linked_candidate_ids:
                    return "visible_candidate_unlinked_or_deleted", candidate
                return "same_label_dedupe_or_neighbor_claim", candidate
            if expected_label == "table_bounce" and confidence(candidate) < T0024_TABLE_THRESHOLD:
                return "table_prediction_below_threshold", candidate
            if candidate_source(candidate) == "recovery_candidate":
                return "recovery_gate_filtered", candidate
            return "hidden_target_prediction", candidate

        candidate = near[0]
        predicted = prediction_label(candidate)
        if predicted == "non_target":
            return "model_predicted_non_target", candidate
        if predicted in TARGET_LABELS:
            return "wrong_racket_table_class", candidate
        return "near_candidate_other", candidate

    wider = [
        candidate
        for candidate in candidates
        if abs(timestamp_ms(candidate) - marker_ms) <= GAP_CANDIDATE_MS
    ]
    wider.sort(key=lambda candidate: abs(timestamp_ms(candidate) - marker_ms))
    if wider:
        return "timing_offset_or_dense_sequence", wider[0]
    return "true_candidate_generation_gap", None


def close_gap_rows(markers: list[dict[str, Any]], max_gap_ms: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordered = sorted_by_time([marker for marker in markers if marker_kind(marker) in {"racket", "table"}])
    for previous, current in zip(ordered, ordered[1:]):
        gap = timestamp_ms(current) - timestamp_ms(previous)
        if gap <= max_gap_ms:
            rows.append({
                "gap_ms": int(round(gap)),
                "previous_ms": int(round(timestamp_ms(previous))),
                "current_ms": int(round(timestamp_ms(current))),
                "previous_kind": marker_kind(previous),
                "current_kind": marker_kind(current),
                "previous_source": previous.get("source", ""),
                "current_source": current.get("source", ""),
                "previous_status": previous.get("review_status", ""),
                "current_status": current.get("review_status", ""),
            })
    return rows


def evaluate_predictions(
    predictions: list[dict[str, Any]],
    truths: list[dict[str, Any]],
    match_ms: int,
) -> dict[str, Any]:
    used_truths: set[int] = set()
    true_positive = 0
    wrong_class = 0
    false_positive = 0

    indexed_truths = [
        {**truth, "truth_index": index}
        for index, truth in enumerate(sorted_by_time(truths))
    ]
    for prediction in sorted_by_time(predictions):
        pred_kind = label_kind(prediction_label(prediction))
        if pred_kind not in {"racket", "table"}:
            continue
        nearby = sorted(
            [
                {**truth, "dt_ms": abs(timestamp_ms(truth) - timestamp_ms(prediction))}
                for truth in indexed_truths
                if abs(timestamp_ms(truth) - timestamp_ms(prediction)) <= match_ms
            ],
            key=lambda truth: truth["dt_ms"],
        )
        same = next(
            (
                truth for truth in nearby
                if marker_kind(truth) == pred_kind and int(truth["truth_index"]) not in used_truths
            ),
            None,
        )
        if same:
            used_truths.add(int(same["truth_index"]))
            true_positive += 1
            continue
        any_kind = next((truth for truth in nearby if int(truth["truth_index"]) not in used_truths), None)
        if any_kind:
            used_truths.add(int(any_kind["truth_index"]))
            wrong_class += 1
        else:
            false_positive += 1

    missed = [
        truth for truth in indexed_truths
        if int(truth["truth_index"]) not in used_truths
    ]
    return {
        "prediction_count": len(predictions),
        "true_positive": true_positive,
        "wrong_class": wrong_class,
        "false_positive": false_positive,
        "missed": len(missed),
        "missed_by_kind": count_by_kind(missed),
    }


def build_manual_rows(markers: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    linked_candidate_ids = {
        str(marker.get("linked_candidate_id"))
        for marker in markers
        if marker.get("linked_candidate_id")
    }
    rows: list[dict[str, Any]] = []
    for marker in sorted_by_time([item for item in markers if item.get("source") == "manual"]):
        marker_ms = timestamp_ms(marker)
        nearest_any, _any_dt = nearest(marker_ms, candidates)
        nearest_saved, _saved_dt = nearest(
            marker_ms,
            [candidate for candidate in candidates if candidate_source(candidate) == "saved_candidate"],
        )
        nearest_recovery, _recovery_dt = nearest(
            marker_ms,
            [candidate for candidate in candidates if candidate_source(candidate) == "recovery_candidate"],
        )
        reason, reason_candidate = classify_manual_reason(marker, candidates, linked_candidate_ids)
        rows.append({
            "marker_id": marker.get("id", ""),
            "timestamp_ms": int(round(marker_ms)),
            "kind": marker_kind(marker),
            "reason_bucket": reason,
            "nearest_any": candidate_snapshot(nearest_any, marker_ms),
            "nearest_saved": candidate_snapshot(nearest_saved, marker_ms),
            "nearest_recovery": candidate_snapshot(nearest_recovery, marker_ms),
            "reason_candidate": candidate_snapshot(reason_candidate, marker_ms),
        })
    return rows


def build_auto_edit_rows(
    markers: list[dict[str, Any]],
    candidates_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for marker in sorted_by_time([item for item in markers if item.get("source") == "auto"]):
        candidate = candidates_by_id.get(str(marker.get("linked_candidate_id") or ""))
        if not candidate:
            continue
        candidate_kind = label_kind(prediction_label(candidate))
        delta_ms = timestamp_ms(marker) - timestamp_ms(candidate)
        label_changed = marker_kind(marker) != candidate_kind
        timestamp_changed = abs(delta_ms) >= 1
        status = str(marker.get("review_status") or "")
        if status == "edited" or label_changed or timestamp_changed:
            rows.append({
                "marker_id": marker.get("id", ""),
                "candidate_id": candidate.get("id", ""),
                "marker_ms": int(round(timestamp_ms(marker))),
                "candidate_ms": int(round(timestamp_ms(candidate))),
                "delta_ms": int(round(delta_ms)),
                "marker_kind": marker_kind(marker),
                "candidate_prediction": prediction_label(candidate),
                "candidate_confidence": round(confidence(candidate), 4),
                "review_status": status,
                "label_changed": label_changed,
                "timestamp_changed": timestamp_changed,
                "probabilities": probabilities(candidate),
            })
    return rows


def build_report(session_path: Path) -> dict[str, Any]:
    session = json.loads(session_path.read_text(encoding="utf-8"))
    events = as_list(session.get("events"))
    if not events:
        raise ValueError(f"No events found in {session_path}")
    event = events[0]
    markers = as_list((event.get("review") or {}).get("markers"))
    candidates = as_list(event.get("model_candidates"))
    pose_candidates = as_list(event.get("video_pose_candidates"))
    candidates_by_id = {str(candidate.get("id")): candidate for candidate in candidates}

    target_markers = [marker for marker in markers if marker_kind(marker) in {"racket", "table"}]
    auto_markers = [marker for marker in markers if marker.get("source") == "auto"]
    manual_rows = build_manual_rows(markers, candidates)
    auto_edit_rows = build_auto_edit_rows(markers, candidates_by_id)
    review_relevant_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("review_relevant") and prediction_label(candidate) in TARGET_LABELS
    ]
    hidden_target_candidates = [
        candidate
        for candidate in candidates
        if not candidate.get("review_relevant") and prediction_label(candidate) in TARGET_LABELS
    ]
    linked_candidate_ids = {
        str(marker.get("linked_candidate_id"))
        for marker in markers
        if marker.get("linked_candidate_id")
    }
    unlinked_review_relevant = [
        candidate
        for candidate in review_relevant_candidates
        if str(candidate.get("id")) not in linked_candidate_ids
    ]

    manual_reason_counts = Counter(row["reason_bucket"] for row in manual_rows)
    manual_reason_by_kind: dict[str, dict[str, int]] = {}
    for row in manual_rows:
        manual_reason_by_kind.setdefault(row["reason_bucket"], {})
        manual_reason_by_kind[row["reason_bucket"]][row["kind"]] = (
            manual_reason_by_kind[row["reason_bucket"]].get(row["kind"], 0) + 1
        )

    nearest_counts = {}
    for limit in (40, 80, 120, 180, 250):
        nearest_counts[f"within_{limit}_ms"] = sum(
            1
            for row in manual_rows
            if row["nearest_any"]["dt_ms"] is not None and abs(int(row["nearest_any"]["dt_ms"])) <= limit
        )

    return {
        "ticket": "T0025",
        "status": "audit_only_no_training_no_export_no_apk",
        "session_path": str(session_path),
        "session_id": session_path.stem,
        "duration_ms": event.get("duration_ms"),
        "review_stage": (event.get("review") or {}).get("review_stage", ""),
        "review_completed_at": (event.get("review") or {}).get("completed_at", ""),
        "wav_filename": event.get("wav_filename", ""),
        "video_filename": event.get("video_filename", "") or (event.get("video_recording") or {}).get("filename", ""),
        "t0024_settings": {
            "model": "playing_retro_audio_rf_v2026_06_03_t0022_multi_window_context",
            "racket_threshold": T0024_RACKET_THRESHOLD,
            "table_threshold": T0024_TABLE_THRESHOLD,
            "same_label_dedupe_ms": SAME_LABEL_DEDUPE_MS,
        },
        "summary": {
            "markers_total": len(markers),
            "markers_by_kind": count_by_kind(target_markers),
            "markers_by_source": count_by(markers, "source"),
            "markers_by_status": count_by(markers, "review_status"),
            "auto_markers": len(auto_markers),
            "manual_markers": len(manual_rows),
            "manual_by_kind": dict(sorted(Counter(row["kind"] for row in manual_rows).items())),
            "manual_nearest_candidate_counts": nearest_counts,
            "manual_reason_counts": dict(sorted(manual_reason_counts.items())),
            "manual_reason_by_kind": dict(sorted((key, dict(sorted(value.items()))) for key, value in manual_reason_by_kind.items())),
            "auto_edit_rows": len(auto_edit_rows),
            "model_candidates_total": len(candidates),
            "model_candidates_by_prediction": count_candidates_by_prediction(candidates),
            "review_relevant_target_candidates": len(review_relevant_candidates),
            "review_relevant_by_prediction": count_candidates_by_prediction(review_relevant_candidates),
            "hidden_target_predictions": len(hidden_target_candidates),
            "hidden_target_by_prediction": count_candidates_by_prediction(hidden_target_candidates),
            "unlinked_review_relevant_targets": len(unlinked_review_relevant),
            "video_pose_candidates": len(pose_candidates),
        },
        "evaluation": {
            "review_relevant_candidates_vs_final_markers": evaluate_predictions(
                review_relevant_candidates,
                target_markers,
                MATCH_MS,
            ),
        },
        "manual_additions": manual_rows,
        "auto_edits": auto_edit_rows,
        "close_gaps_under_120_ms": close_gap_rows(target_markers, 120),
        "close_gaps_under_180_ms": close_gap_rows(target_markers, 180),
        "hidden_target_candidates": [
            candidate_snapshot(candidate)
            for candidate in sorted_by_time(hidden_target_candidates)
        ],
        "unlinked_review_relevant_targets": [
            candidate_snapshot(candidate)
            for candidate in sorted_by_time(unlinked_review_relevant)
        ],
        "changed_app_artifacts": False,
        "changed_studs_live": False,
        "changed_video_model": False,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def compact_candidate(snapshot: dict[str, Any]) -> str:
    if not snapshot or snapshot.get("timestamp_ms") is None:
        return ""
    return (
        f"{snapshot['timestamp_ms']} ({snapshot['dt_ms']:+} ms, "
        f"{snapshot['prediction']}, conf={snapshot['confidence']}, "
        f"{snapshot['source']}, visible={snapshot['review_relevant']})"
    )


def write_manual_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "timestamp_ms",
        "kind",
        "reason_bucket",
        "nearest_any",
        "nearest_saved",
        "nearest_recovery",
        "reason_candidate",
        "marker_id",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "timestamp_ms": row["timestamp_ms"],
                "kind": row["kind"],
                "reason_bucket": row["reason_bucket"],
                "nearest_any": compact_candidate(row["nearest_any"]),
                "nearest_saved": compact_candidate(row["nearest_saved"]),
                "nearest_recovery": compact_candidate(row["nearest_recovery"]),
                "reason_candidate": compact_candidate(row["reason_candidate"]),
                "marker_id": row["marker_id"],
            })


def md_table(rows: list[list[Any]], headers: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    evaluation = report["evaluation"]["review_relevant_candidates_vs_final_markers"]
    lines = [
        "# Playing Retro T0025 Audit: audio_session_2026-06-04_001",
        "",
        "This is an audit-only report. It does not train, export app JSON, build an APK, or change `studs_live`.",
        "",
        "## Summary",
        "",
        f"- Final audio markers: `{summary['markers_total']}` (`{summary['markers_by_kind']}`)",
        f"- Sources: `{summary['markers_by_source']}`",
        f"- Review status: `{summary['markers_by_status']}`",
        f"- Manual additions: `{summary['manual_markers']}` (`{summary['manual_by_kind']}`)",
        f"- Auto markers: `{summary['auto_markers']}`",
        f"- Model candidates: `{summary['model_candidates_total']}` (`{summary['model_candidates_by_prediction']}`)",
        f"- Review-relevant target candidates: `{summary['review_relevant_target_candidates']}` (`{summary['review_relevant_by_prediction']}`)",
        f"- Hidden target predictions: `{summary['hidden_target_predictions']}` (`{summary['hidden_target_by_prediction']}`)",
        f"- Manual additions near any candidate: `{summary['manual_nearest_candidate_counts']}`",
        "",
        "## Review-Relevant Candidate Replay",
        "",
        "| Predictions | TP | Wrong | FP | Missed | Missed by kind |",
        "|---:|---:|---:|---:|---:|---|",
        (
            f"| {evaluation['prediction_count']} | {evaluation['true_positive']} | "
            f"{evaluation['wrong_class']} | {evaluation['false_positive']} | "
            f"{evaluation['missed']} | `{evaluation['missed_by_kind']}` |"
        ),
        "",
        "## Manual Addition Failure Buckets",
        "",
    ]
    rows = [
        [bucket, count, summary["manual_reason_by_kind"].get(bucket, {})]
        for bucket, count in summary["manual_reason_counts"].items()
    ]
    lines.extend(md_table(rows, ["Reason bucket", "Count", "Kinds"]))
    lines.extend([
        "",
        "## Manual Additions",
        "",
    ])
    lines.extend(md_table(
        [
            [
                row["timestamp_ms"],
                row["kind"],
                row["reason_bucket"],
                compact_candidate(row["nearest_any"]),
                compact_candidate(row["reason_candidate"]),
            ]
            for row in report["manual_additions"]
        ],
        ["ms", "kind", "reason", "nearest candidate", "reason candidate"],
    ))
    lines.extend([
        "",
        "## Close Final Marker Gaps Under 120 ms",
        "",
    ])
    lines.extend(md_table(
        [
            [
                row["gap_ms"],
                f"{row['previous_kind']}@{row['previous_ms']}",
                f"{row['current_kind']}@{row['current_ms']}",
                f"{row['previous_source']}->{row['current_source']}",
            ]
            for row in report["close_gaps_under_120_ms"]
        ],
        ["gap ms", "previous", "current", "source"],
    ))
    lines.extend([
        "",
        "## Recommended Next Tickets",
        "",
        "- `T0026`: retrain `spel_retro_audio` with this 06-04 reviewed session plus historical playing data, because most manual additions are near saved candidates rather than true no-candidate gaps.",
        "- `T0027`: replay/tune T0026 against T0024 on 05-28, 05-29, 06-03, and 06-04, with special attention to `non_target` target misses and table threshold.",
        "- `T0028`: export/build/install only if T0027 safely beats T0024.",
        "- `T0029`: revisit candidate/peak recovery only if T0027 still shows material true candidate-generation gaps.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(report: dict[str, Any], output_json: Path, output_md: Path, manual_csv: Path, gaps_csv: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(output_md, report)
    write_manual_csv(manual_csv, report["manual_additions"])
    write_csv(
        gaps_csv,
        report["close_gaps_under_120_ms"],
        [
            "gap_ms",
            "previous_ms",
            "current_ms",
            "previous_kind",
            "current_kind",
            "previous_source",
            "current_source",
            "previous_status",
            "current_status",
        ],
    )


def print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    evaluation = report["evaluation"]["review_relevant_candidates_vs_final_markers"]
    print(f"# T0025 audit {report['session_id']}")
    print(f"markers={summary['markers_total']} by_kind={summary['markers_by_kind']}")
    print(f"auto={summary['auto_markers']} manual={summary['manual_markers']} manual_by_kind={summary['manual_by_kind']}")
    print(f"candidates={summary['model_candidates_total']} by_prediction={summary['model_candidates_by_prediction']}")
    print(f"hidden_targets={summary['hidden_target_predictions']} {summary['hidden_target_by_prediction']}")
    print(f"manual_nearest={summary['manual_nearest_candidate_counts']}")
    print(f"manual_reasons={summary['manual_reason_counts']}")
    print(
        "replay_visible_candidates pred={prediction_count} tp={true_positive} wrong={wrong_class} "
        "fp={false_positive} missed={missed}".format(**evaluation)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit T0025 playing-retro review session.")
    parser.add_argument("session_json", type=Path, nargs="?", default=DEFAULT_SESSION)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--manual-csv", type=Path, default=DEFAULT_MANUAL_CSV)
    parser.add_argument("--gaps-csv", type=Path, default=DEFAULT_GAPS_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.session_json)
    write_outputs(report, args.output_json, args.output_md, args.manual_csv, args.gaps_csv)
    print_summary(report)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    print(f"wrote {args.manual_csv}")
    print(f"wrote {args.gaps_csv}")


if __name__ == "__main__":
    main()
