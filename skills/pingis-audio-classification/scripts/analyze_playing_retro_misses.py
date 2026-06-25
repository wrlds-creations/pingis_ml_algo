#!/usr/bin/env python3
"""Explain playing-retro misses before retraining.

This T0021 diagnostic reads one reviewed session and does not train, export,
build an APK, or change Collector app artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import wave
from bisect import bisect_left
from collections import Counter
from pathlib import Path
from typing import Any

from audit_playing_retro_review_session import (
    confidence_of,
    dedupe_same_label_candidates,
    evaluate_predictions,
    kind_of,
    prediction_rows,
    sorted_by_time,
    truth_rows,
)

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_SESSION = ROOT_DIR / "data" / "audio" / "raw" / "audio_session_2026-06-03_005.json"
EVAL_DIR = ROOT_DIR / "data" / "audio" / "models" / "evaluations"
DEFAULT_JSON = EVAL_DIR / "playing_retro_t0021_miss_analysis.json"
DEFAULT_CSV = EVAL_DIR / "playing_retro_t0021_manual_additions.csv"
DEFAULT_MD = EVAL_DIR / "playing_retro_t0021_miss_analysis.md"

TARGET_KINDS = {"racket", "table"}


def candidate_label(candidate: dict[str, Any] | None) -> str:
    if not candidate:
        return ""
    prediction = candidate.get("playing_retro_prediction") or {}
    label = str(prediction.get("label") or "")
    if label in {"racket_contact", "table_bounce", "non_target"}:
        return label
    kind = kind_of(candidate)
    if kind == "racket":
        return "racket_contact"
    if kind == "table":
        return "table_bounce"
    return "non_target"


def label_kind(label: str) -> str:
    if label == "racket_contact":
        return "racket"
    if label == "table_bounce":
        return "table"
    if label == "non_target":
        return "non_target"
    return "other"


def target_label(kind: str) -> str:
    if kind == "racket":
        return "racket_contact"
    if kind == "table":
        return "table_bounce"
    return ""


def probabilities(candidate: dict[str, Any] | None) -> dict[str, float]:
    if not candidate:
        return {}
    raw = (candidate.get("playing_retro_prediction") or {}).get("probabilities") or {}
    return {
        key: round(float(raw.get(key, 0.0)), 3)
        for key in ("non_target", "racket_contact", "table_bounce")
    }


def probability_summary(candidate: dict[str, Any] | None) -> str:
    probs = dict(candidate.get("probabilities") or {}) if candidate else {}
    if not probs:
        probs = probabilities(candidate)
    if not probs:
        return ""
    return "N={non_target:.3f} R={racket_contact:.3f} T={table_bounce:.3f}".format(**probs)


def nearest(timestamp_ms: float, items: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, int | None]:
    if not items:
        return None, None
    item = min(items, key=lambda row: abs(float(row.get("timestamp_ms", 0)) - timestamp_ms))
    return item, int(round(float(item.get("timestamp_ms", 0)) - timestamp_ms))


def candidate_snapshot(candidate: dict[str, Any] | None, delta_ms: int | None = None) -> dict[str, Any]:
    if not candidate:
        return {
            "id": "",
            "timestamp_ms": "",
            "dt_ms": "",
            "source": "",
            "review_relevant": "",
            "label": "",
            "confidence": "",
            "probabilities": {},
        }
    return {
        "id": candidate.get("id", ""),
        "timestamp_ms": int(round(float(candidate.get("timestamp_ms", 0)))),
        "dt_ms": delta_ms,
        "source": candidate.get("playing_retro_candidate_source", ""),
        "review_relevant": bool(candidate.get("review_relevant")),
        "label": candidate_label(candidate),
        "confidence": round(confidence_of(candidate), 3),
        "probabilities": probabilities(candidate),
    }


def read_wav_samples(wav_path: Path) -> tuple[list[int], int]:
    with wave.open(str(wav_path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            return [], int(wav.getframerate())
        frames = wav.readframes(wav.getnframes())
        sample_count = len(frames) // 2
        samples = [
            int.from_bytes(frames[index * 2:index * 2 + 2], byteorder="little", signed=True)
            for index in range(sample_count)
        ]
        return samples, int(wav.getframerate())


def rms(values: list[int]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(float(value) * float(value) for value in values) / len(values)) / 32768.0


def energy_percentile_fn(wav_path: Path):
    if not wav_path.exists():
        return lambda _timestamp_ms: ""
    samples, sample_rate = read_wav_samples(wav_path)
    if not samples:
        return lambda _timestamp_ms: ""
    window_samples = max(1, int(round(sample_rate * 0.06)))
    step_samples = max(1, int(round(sample_rate * 0.01)))
    frame_values = [
        rms(samples[start:start + window_samples])
        for start in range(0, max(0, len(samples) - window_samples), step_samples)
    ]
    sorted_values = sorted(frame_values)

    def percentile_for_timestamp(timestamp_ms: float) -> float:
        radius_samples = max(1, int(round(sample_rate * 0.03)))
        center = int(round(timestamp_ms / 1000.0 * sample_rate))
        value = rms(samples[max(0, center - radius_samples): min(len(samples), center + radius_samples)])
        if not sorted_values:
            return 0.0
        return round(100.0 * bisect_left(sorted_values, value) / len(sorted_values), 1)

    return percentile_for_timestamp


def truth_context(markers: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    truths = [marker for marker in sorted_by_time(markers) if kind_of(marker) in TARGET_KINDS]
    result: dict[str, dict[str, Any]] = {}
    for index, marker in enumerate(truths):
        previous = truths[index - 1] if index > 0 else None
        next_marker = truths[index + 1] if index + 1 < len(truths) else None
        prev_gap = (
            int(round(float(marker.get("timestamp_ms", 0)) - float(previous.get("timestamp_ms", 0))))
            if previous else None
        )
        next_gap = (
            int(round(float(next_marker.get("timestamp_ms", 0)) - float(marker.get("timestamp_ms", 0))))
            if next_marker else None
        )
        gaps = [gap for gap in (prev_gap, next_gap) if gap is not None]
        nearest_gap = min(gaps) if gaps else None
        if nearest_gap is None:
            bucket = "isolated"
        elif nearest_gap < 80:
            bucket = "under_80ms"
        elif nearest_gap < 120:
            bucket = "80_119ms"
        elif nearest_gap < 180:
            bucket = "120_179ms"
        elif nearest_gap <= 300:
            bucket = "180_300ms"
        else:
            bucket = "isolated"
        result[str(marker.get("id", ""))] = {
            "nearest_truth_gap_ms": nearest_gap,
            "close_event_bucket": bucket,
            "previous_kind": kind_of(previous) if previous else "",
            "previous_gap_ms": prev_gap,
            "next_kind": kind_of(next_marker) if next_marker else "",
            "next_gap_ms": next_gap,
        }
    return result


def reason_for_manual_row(
    marker: dict[str, Any],
    candidates: list[dict[str, Any]],
    linked_candidate_ids: set[str],
    candidate_near_ms: int,
    timing_near_ms: int,
) -> tuple[str, dict[str, Any] | None, int | None]:
    timestamp_ms = float(marker.get("timestamp_ms", 0))
    marker_kind = kind_of(marker)
    wanted_label = target_label(marker_kind)
    near_candidates = [
        candidate for candidate in candidates
        if abs(float(candidate.get("timestamp_ms", 0)) - timestamp_ms) <= candidate_near_ms
    ]
    near_candidates.sort(key=lambda row: abs(float(row.get("timestamp_ms", 0)) - timestamp_ms))
    correct_near = [candidate for candidate in near_candidates if candidate_label(candidate) == wanted_label]
    if correct_near:
        candidate = correct_near[0]
        delta = int(round(float(candidate.get("timestamp_ms", 0)) - timestamp_ms))
        if candidate.get("review_relevant") and str(candidate.get("id", "")) not in linked_candidate_ids:
            return "visible_candidate_unlinked_or_deleted", candidate, delta
        if not candidate.get("review_relevant"):
            source = str(candidate.get("playing_retro_candidate_source") or "")
            if source == "recovery_candidate":
                return "hidden_by_recovery_gate", candidate, delta
            return "hidden_by_non_review_gate", candidate, delta
        return "dense_sequence_candidate_claimed_by_neighbor", candidate, delta
    if near_candidates:
        candidate = near_candidates[0]
        delta = int(round(float(candidate.get("timestamp_ms", 0)) - timestamp_ms))
        if candidate_label(candidate) == "non_target":
            return "model_predicted_non_target", candidate, delta
        return "model_wrong_class", candidate, delta

    timing_candidates = [
        candidate for candidate in candidates
        if abs(float(candidate.get("timestamp_ms", 0)) - timestamp_ms) <= timing_near_ms
    ]
    timing_candidates.sort(key=lambda row: abs(float(row.get("timestamp_ms", 0)) - timestamp_ms))
    correct_timing = [candidate for candidate in timing_candidates if candidate_label(candidate) == wanted_label]
    if correct_timing:
        candidate = correct_timing[0]
        return (
            "timing_offset_or_dense_sequence",
            candidate,
            int(round(float(candidate.get("timestamp_ms", 0)) - timestamp_ms)),
        )
    if timing_candidates:
        candidate = timing_candidates[0]
        return (
            "candidate_generation_gap_nearest_wrong_or_non_target",
            candidate,
            int(round(float(candidate.get("timestamp_ms", 0)) - timestamp_ms)),
        )
    return "candidate_generation_gap", None, None


def build_manual_rows(
    markers: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    candidate_near_ms: int,
    timing_near_ms: int,
    energy_percentile,
) -> list[dict[str, Any]]:
    manual_markers = [marker for marker in sorted_by_time(markers) if marker.get("source") == "manual"]
    saved_candidates = [
        candidate for candidate in candidates
        if candidate.get("playing_retro_candidate_source") == "saved_candidate"
    ]
    recovery_candidates = [
        candidate for candidate in candidates
        if candidate.get("playing_retro_candidate_source") == "recovery_candidate"
    ]
    linked_candidate_ids = {
        str(marker.get("linked_candidate_id"))
        for marker in markers
        if marker.get("linked_candidate_id")
    }
    contexts = truth_context(markers)
    rows: list[dict[str, Any]] = []
    for marker in manual_markers:
        timestamp_ms = float(marker.get("timestamp_ms", 0))
        saved, saved_dt = nearest(timestamp_ms, saved_candidates)
        recovery, recovery_dt = nearest(timestamp_ms, recovery_candidates)
        any_candidate, any_dt = nearest(timestamp_ms, candidates)
        reason, reason_candidate, reason_dt = reason_for_manual_row(
            marker,
            candidates,
            linked_candidate_ids,
            candidate_near_ms,
            timing_near_ms,
        )
        context = contexts.get(str(marker.get("id", "")), {})
        rows.append({
            "marker_id": marker.get("id", ""),
            "timestamp_ms": int(round(timestamp_ms)),
            "kind": kind_of(marker),
            "reason_bucket": reason,
            "energy_percentile": energy_percentile(timestamp_ms),
            "close_event_bucket": context.get("close_event_bucket", ""),
            "nearest_truth_gap_ms": context.get("nearest_truth_gap_ms", ""),
            "neighbor_sequence": "{prev}>{kind}>{next}".format(
                prev=context.get("previous_kind", "") or "-",
                kind=kind_of(marker),
                next=context.get("next_kind", "") or "-",
            ),
            "nearest_saved": candidate_snapshot(saved, saved_dt),
            "nearest_recovery": candidate_snapshot(recovery, recovery_dt),
            "nearest_any": candidate_snapshot(any_candidate, any_dt),
            "reason_candidate": candidate_snapshot(reason_candidate, reason_dt),
        })
    return rows


def build_label_change_rows(markers: list[dict[str, Any]], candidates_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for marker in sorted_by_time([marker for marker in markers if marker.get("source") == "auto"]):
        candidate = candidates_by_id.get(str(marker.get("linked_candidate_id") or ""))
        if not candidate:
            continue
        marker_kind = kind_of(marker)
        candidate_kind = label_kind(candidate_label(candidate))
        if marker_kind == candidate_kind:
            continue
        probs = probabilities(candidate)
        wanted = target_label(marker_kind)
        predicted = candidate_label(candidate)
        margin = round(float(probs.get(predicted, 0.0)) - float(probs.get(wanted, 0.0)), 3)
        rows.append({
            "marker_id": marker.get("id", ""),
            "candidate_id": candidate.get("id", ""),
            "timestamp_ms": int(round(float(marker.get("timestamp_ms", 0)))),
            "marker_kind": marker_kind,
            "candidate_label": predicted,
            "confidence": round(confidence_of(candidate), 3),
            "probabilities": probs,
            "winner_margin_over_truth": margin,
        })
    return rows


def build_nudge_rows(markers: list[dict[str, Any]], candidates_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for marker in sorted_by_time([marker for marker in markers if marker.get("source") == "auto"]):
        candidate = candidates_by_id.get(str(marker.get("linked_candidate_id") or ""))
        if not candidate:
            continue
        delta = int(round(float(marker.get("timestamp_ms", 0)) - float(candidate.get("timestamp_ms", 0))))
        if delta == 0:
            continue
        rows.append({
            "marker_id": marker.get("id", ""),
            "candidate_id": candidate.get("id", ""),
            "marker_ms": int(round(float(marker.get("timestamp_ms", 0)))),
            "candidate_ms": int(round(float(candidate.get("timestamp_ms", 0)))),
            "delta_ms": delta,
            "kind": kind_of(marker),
            "candidate_label": candidate_label(candidate),
            "confidence": round(confidence_of(candidate), 3),
        })
    return rows


def detailed_evaluation(
    predictions: list[dict[str, Any]],
    truths: list[dict[str, Any]],
    threshold_ms: int,
) -> dict[str, Any]:
    used_truths: set[int] = set()
    false_positive: list[dict[str, Any]] = []
    wrong_class: list[dict[str, Any]] = []
    true_positive = 0
    for prediction in predictions:
        nearby = sorted(
            (
                {
                    **truth,
                    "dt_ms": abs(float(truth["timestamp_ms"]) - float(prediction["timestamp_ms"])),
                }
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
            true_positive += 1
            continue
        any_kind = next((truth for truth in nearby if int(truth["index"]) not in used_truths), None)
        if any_kind is not None:
            used_truths.add(int(any_kind["index"]))
            wrong_class.append({"prediction": prediction, "truth": any_kind})
        else:
            false_positive.append(prediction)
    missed = [truth for truth in truths if int(truth["index"]) not in used_truths]
    return {
        "true_positive": true_positive,
        "wrong_class": wrong_class,
        "false_positive": false_positive,
        "missed": missed,
    }


def build_false_positive_rows(
    predictions: list[dict[str, Any]],
    truths: list[dict[str, Any]],
    candidates_by_id: dict[str, dict[str, Any]],
    threshold_ms: int,
) -> list[dict[str, Any]]:
    evaluation = detailed_evaluation(predictions, truths, threshold_ms)
    rows: list[dict[str, Any]] = []
    truth_like = [
        {"timestamp_ms": truth["timestamp_ms"], "kind": truth["kind"], "id": truth["id"]}
        for truth in truths
    ]
    for prediction in evaluation["false_positive"]:
        candidate = candidates_by_id.get(str(prediction.get("id") or ""))
        nearest_truth, nearest_truth_dt = nearest(float(prediction["timestamp_ms"]), truth_like)
        fp_reason = "no_truth_within_match_window"
        if nearest_truth_dt is not None and abs(nearest_truth_dt) <= threshold_ms:
            if nearest_truth and nearest_truth.get("kind") == prediction["kind"]:
                fp_reason = "truth_already_claimed_by_earlier_prediction"
            else:
                fp_reason = "near_wrong_kind_truth"
        rows.append({
            "candidate_id": prediction.get("id", ""),
            "timestamp_ms": int(round(float(prediction["timestamp_ms"]))),
            "kind": prediction["kind"],
            "confidence": round(float(prediction["confidence"]), 3),
            "source": prediction.get("source", ""),
            "fp_reason": fp_reason,
            "nearest_truth_ms": candidate_snapshot(nearest_truth, nearest_truth_dt)["timestamp_ms"],
            "nearest_truth_dt_ms": nearest_truth_dt,
            "nearest_truth_kind": nearest_truth.get("kind", "") if nearest_truth else "",
            "probabilities": probabilities(candidate),
        })
    return rows


def compact_candidate(row: dict[str, Any], field: str) -> str:
    candidate = row[field]
    if not candidate or candidate.get("timestamp_ms") == "":
        return ""
    return "{timestamp_ms} ({dt_ms:+} ms, {label}, {confidence}, {source}, visible={review_relevant})".format(**candidate)


def write_manual_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "timestamp_ms",
        "kind",
        "reason_bucket",
        "energy_percentile",
        "close_event_bucket",
        "nearest_truth_gap_ms",
        "neighbor_sequence",
        "nearest_saved",
        "nearest_saved_probs",
        "nearest_recovery",
        "nearest_recovery_probs",
        "reason_candidate",
        "reason_candidate_probs",
        "marker_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "timestamp_ms": row["timestamp_ms"],
                "kind": row["kind"],
                "reason_bucket": row["reason_bucket"],
                "energy_percentile": row["energy_percentile"],
                "close_event_bucket": row["close_event_bucket"],
                "nearest_truth_gap_ms": row["nearest_truth_gap_ms"],
                "neighbor_sequence": row["neighbor_sequence"],
                "nearest_saved": compact_candidate(row, "nearest_saved"),
                "nearest_saved_probs": probability_summary(row["nearest_saved"]),
                "nearest_recovery": compact_candidate(row, "nearest_recovery"),
                "nearest_recovery_probs": probability_summary(row["nearest_recovery"]),
                "reason_candidate": compact_candidate(row, "reason_candidate"),
                "reason_candidate_probs": probability_summary(row["reason_candidate"]),
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
    manual_rows = report["manual_additions"]
    lines = [
        f"# Playing Retro T0021 Miss Analysis: {report['session_id']}",
        "",
        "This is an analysis-only report. It does not train, export, build, install, or change `studs_live`.",
        "",
        "## Summary",
        "",
        f"- Final reviewed markers: `{report['summary']['markers_total']}` (`{report['summary']['markers_by_kind']}`)",
        f"- Manual additions: `{report['summary']['manual_total']}` (`{report['summary']['manual_by_kind']}`)",
        f"- Auto relabels: `{report['summary']['label_change_total']}`",
        f"- Auto timing nudges: `{report['summary']['nudge_total']}`",
        f"- T0020 false positives after dedupe: `{report['summary']['false_positive_after_t0020']}`",
        f"- Manual additions with local energy below median: `{report['summary']['manual_low_energy_count']}`",
        "",
        "## Failure Buckets",
        "",
    ]
    bucket_rows = [
        [bucket, count, report["summary"]["manual_reason_by_kind"].get(bucket, {})]
        for bucket, count in sorted(report["summary"]["manual_reason_counts"].items())
    ]
    lines.extend(md_table(bucket_rows, ["Reason bucket", "Count", "Kinds"]))
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
                row["energy_percentile"],
                row["close_event_bucket"],
                compact_candidate(row, "nearest_saved"),
                compact_candidate(row, "nearest_recovery"),
            ]
            for row in manual_rows
        ],
        ["ms", "kind", "reason", "energy pct", "close bucket", "nearest saved", "nearest recovery"],
    ))
    lines.extend(["", "## Auto Relabels", ""])
    lines.extend(md_table(
        [
            [
                row["timestamp_ms"],
                row["candidate_label"],
                row["marker_kind"],
                row["confidence"],
                row["winner_margin_over_truth"],
                probability_summary({"playing_retro_prediction": {"probabilities": row["probabilities"]}}),
            ]
            for row in report["auto_label_changes"]
        ],
        ["ms", "model label", "reviewed kind", "conf", "margin", "probs"],
    ))
    lines.extend(["", "## Timing Nudges", ""])
    lines.extend(md_table(
        [
            [row["candidate_ms"], row["marker_ms"], row["delta_ms"], row["kind"], row["confidence"]]
            for row in report["auto_nudges"]
        ],
        ["candidate ms", "review ms", "delta", "kind", "conf"],
    ))
    lines.extend(["", "## False Positives After T0020 Dedupe", ""])
    lines.extend(md_table(
        [
            [
                row["timestamp_ms"],
                row["kind"],
                row["confidence"],
                row["source"],
                row["fp_reason"],
                row["nearest_truth_dt_ms"],
                row["nearest_truth_kind"],
            ]
            for row in report["false_positives_after_t0020"]
        ],
        ["ms", "kind", "conf", "source", "reason", "nearest truth dt", "nearest truth kind"],
    ))
    lines.extend([
        "",
        "## T0022 Training Input Plan",
        "",
        "- Include `audio_session_2026-06-03_005` as `spel_retro_audio` / dense playing data only.",
        "- Add the 20 manual additions as positive target rows at the reviewed timestamps.",
        "- Keep saved app candidates and unmatched non-target rows as the candidate surface; replay-generated peaks stay diagnostic unless T0022 explicitly changes the dataset builder.",
        "- Keep historical playing sessions `audio_session_2026-05-28_002`, `audio_session_2026-05-29_001`, and `audio_session_2026-05-29_002` in the validation gate.",
        "- Do not promote anything to ordinary up/down `studs_live`; ordinary bounce remains a separate regression slice only.",
        "- T0023 should compare model-only retrain against recovery/candidate-gate tuning because most misses are non-target classification near real peaks, not raw no-peak failures.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(session_path: Path, event_index: int, candidate_near_ms: int, timing_near_ms: int, dedupe_gap_ms: int, match_threshold_ms: int) -> dict[str, Any]:
    session = json.loads(session_path.read_text(encoding="utf-8"))
    event = session["events"][event_index]
    markers = event.get("review", {}).get("markers") or []
    candidates = event.get("model_candidates") or []
    candidates_by_id = {str(candidate.get("id")): candidate for candidate in candidates}
    wav_path = session_path.with_suffix("") / str(event.get("wav_filename"))
    energy_percentile = energy_percentile_fn(wav_path)

    review_candidates = [candidate for candidate in candidates if candidate.get("review_relevant")]
    deduped_candidates, _removed = dedupe_same_label_candidates(review_candidates, dedupe_gap_ms)
    truths = truth_rows([marker for marker in markers if kind_of(marker) in TARGET_KINDS])
    manual_rows = build_manual_rows(markers, candidates, candidate_near_ms, timing_near_ms, energy_percentile)
    label_change_rows = build_label_change_rows(markers, candidates_by_id)
    nudge_rows = build_nudge_rows(markers, candidates_by_id)
    false_positive_rows = build_false_positive_rows(
        prediction_rows(deduped_candidates),
        truths,
        candidates_by_id,
        match_threshold_ms,
    )
    evaluation = {
        "baseline_t0019": evaluate_predictions(prediction_rows(review_candidates), truths, match_threshold_ms),
        "candidate_t0020": evaluate_predictions(prediction_rows(deduped_candidates), truths, match_threshold_ms),
    }
    reason_counts = Counter(row["reason_bucket"] for row in manual_rows)
    reason_by_kind: dict[str, dict[str, int]] = {}
    for row in manual_rows:
        reason_by_kind.setdefault(row["reason_bucket"], {})
        reason_by_kind[row["reason_bucket"]][row["kind"]] = reason_by_kind[row["reason_bucket"]].get(row["kind"], 0) + 1
    low_energy_count = sum(
        1 for row in manual_rows
        if isinstance(row["energy_percentile"], (int, float)) and float(row["energy_percentile"]) < 50.0
    )

    return {
        "session_path": str(session_path),
        "session_id": session_path.stem,
        "event_index": event_index,
        "wav_path": str(wav_path),
        "thresholds": {
            "candidate_near_ms": candidate_near_ms,
            "timing_near_ms": timing_near_ms,
            "dedupe_gap_ms": dedupe_gap_ms,
            "match_threshold_ms": match_threshold_ms,
        },
        "summary": {
            "markers_total": len(markers),
            "markers_by_kind": dict(sorted(Counter(kind_of(marker) for marker in markers).items())),
            "manual_total": len(manual_rows),
            "manual_by_kind": dict(sorted(Counter(row["kind"] for row in manual_rows).items())),
            "manual_reason_counts": dict(sorted(reason_counts.items())),
            "manual_reason_by_kind": dict(sorted((key, dict(sorted(value.items()))) for key, value in reason_by_kind.items())),
            "manual_low_energy_count": low_energy_count,
            "label_change_total": len(label_change_rows),
            "nudge_total": len(nudge_rows),
            "false_positive_after_t0020": len(false_positive_rows),
            "t0020_eval": evaluation["candidate_t0020"],
        },
        "manual_additions": manual_rows,
        "auto_label_changes": label_change_rows,
        "auto_nudges": nudge_rows,
        "false_positives_after_t0020": false_positive_rows,
        "evaluation": evaluation,
    }


def print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print(f"# Playing Retro T0021 Miss Analysis: {report['session_id']}")
    print()
    print(f"- Markers: {summary['markers_total']} {summary['markers_by_kind']}")
    print(f"- Manual additions: {summary['manual_total']} {summary['manual_by_kind']}")
    print(f"- Auto relabels: {summary['label_change_total']}")
    print(f"- Auto nudges: {summary['nudge_total']}")
    print(f"- T0020 false positives: {summary['false_positive_after_t0020']}")
    print(f"- Manual additions below median local energy: {summary['manual_low_energy_count']}")
    print()
    print("| Reason bucket | Count | Kinds |")
    print("|---|---:|---|")
    for bucket, count in summary["manual_reason_counts"].items():
        print(f"| `{bucket}` | {count} | {summary['manual_reason_by_kind'].get(bucket, {})} |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_json", type=Path, nargs="?", default=DEFAULT_SESSION)
    parser.add_argument("--event-index", type=int, default=0)
    parser.add_argument("--candidate-near-ms", type=int, default=140)
    parser.add_argument("--timing-near-ms", type=int, default=220)
    parser.add_argument("--dedupe-gap-ms", type=int, default=80)
    parser.add_argument("--match-threshold-ms", type=int, default=80)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    report = build_report(
        args.session_json,
        args.event_index,
        args.candidate_near_ms,
        args.timing_near_ms,
        args.dedupe_gap_ms,
        args.match_threshold_ms,
    )
    print_summary(report)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_manual_csv(args.output_csv, report["manual_additions"])
    write_markdown(args.output_md, report)
    print()
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
