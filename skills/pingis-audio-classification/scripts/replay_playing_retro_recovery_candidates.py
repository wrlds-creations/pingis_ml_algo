"""
Replay T0014 playing-retro recovery candidates on reviewed Tomas/Stiga sessions.

This script does not train, export, build an APK, or touch Collector live audio.
It mirrors the app-side conservative recovery pass:
- dense review envelope peaks
- ignore peaks already covered by saved app candidates
- classify recovery anchors with the exported playing_retro_audio_model.json
- report recovered missed markers and added false positives

Run:
  python skills/pingis-audio-classification/scripts/replay_playing_retro_recovery_candidates.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_playing_retro_candidate_report import (
    MATCH_TOLERANCE_MS,
    TruthMarker,
    build_truth_markers,
    candidate_peaks_from_app,
    default_session_paths,
    match_detectable_candidates,
)
from evaluate_playing_retro_audio_multi_window import (
    WINDOWS,
    context_features,
    extract_window,
    prefixed_features,
)
from preprocess_audio import TARGET_SR, extract_features, load_audio
from replay_live_bounce import resolve_wav_path
from replay_playing_retro_audio_app_export import DEFAULT_APP_MODEL, predict_app_model
from train_playing_retro_audio import EVAL_DIR

FRAME_MS = 12
MIN_ENVELOPE_THRESHOLD = 0.01
LOCAL_FRAME_MS = 4
LOCAL_HOP_MS = 1
DENSE_CANDIDATE_GAP_MS = 28
DENSE_AUTO_REFINE_SEARCH_PRE_MS = 70
DENSE_AUTO_REFINE_SEARCH_POST_MS = 35
RECOVERY_MIN_GAP_FROM_KNOWN_MS = 32
RECOVERY_MIN_GAP_FROM_RECOVERY_MS = 48
RECOVERY_MAX_GAP_FROM_SAVED_MS = 520
RECOVERY_MAX_CANDIDATES = 220
RECOVERY_RACKET_MIN_CONFIDENCE = 0.8
RECOVERY_TABLE_MIN_CONFIDENCE = 0.54
RECOVERY_RACKET_MIN_SAVED_GAP_MS = 120
RECOVERY_TABLE_MIN_SAVED_GAP_MS = 60

PREDICTIONS_CSV = EVAL_DIR / "playing_retro_audio_t0014_recovery_predictions.csv"
SUMMARY_CSV = EVAL_DIR / "playing_retro_audio_t0014_recovery_summary.csv"
REPORT_MD = EVAL_DIR / "playing_retro_audio_t0014_recovery_report.md"


def percentile(values: np.ndarray, q: float) -> float:
    if len(values) == 0:
        return 0.0
    index = max(0, min(len(values) - 1, int(np.floor(q * (len(values) - 1)))))
    return float(np.sort(values)[index])


def local_envelope(y: np.ndarray, start_ms: float, end_ms: float) -> tuple[np.ndarray, float]:
    frame_size = max(24, int(round(TARGET_SR * LOCAL_FRAME_MS / 1000.0)))
    hop_size = max(12, int(round(TARGET_SR * LOCAL_HOP_MS / 1000.0)))
    start_sample = max(0, int(np.floor(start_ms / 1000.0 * TARGET_SR)))
    end_sample = min(len(y), int(np.ceil(end_ms / 1000.0 * TARGET_SR)))
    values: list[float] = []
    for start in range(start_sample, max(start_sample, end_sample - frame_size + 1), hop_size):
        clip = y[start:start + frame_size]
        if len(clip) == frame_size:
            values.append(float(np.sqrt(np.mean(np.square(clip)))))
    return np.asarray(values, dtype=np.float64), (hop_size / TARGET_SR) * 1000.0


def local_peak_indices(values: np.ndarray) -> list[int]:
    if len(values) < 3:
        return []
    mean = float(np.mean(values))
    p85 = percentile(values, 0.85)
    p97 = percentile(values, 0.97)
    threshold = max(MIN_ENVELOPE_THRESHOLD * 0.5, mean * 1.25, p85 * 0.75, p97 * 0.4)
    peaks: list[int] = []
    for index in range(1, len(values) - 1):
        current = float(values[index])
        if current < threshold:
            continue
        if current >= values[index - 1] and current >= values[index + 1]:
            peaks.append(index)
    return peaks


def refine_attack_timestamp(y: np.ndarray, approx_ms: float) -> int:
    total_duration_ms = len(y) / TARGET_SR * 1000.0
    start_ms = max(0.0, approx_ms - DENSE_AUTO_REFINE_SEARCH_PRE_MS)
    end_ms = min(total_duration_ms, approx_ms + DENSE_AUTO_REFINE_SEARCH_POST_MS)
    values, hop_ms = local_envelope(y, start_ms, end_ms)
    if len(values) < 3:
        return int(round(max(0.0, min(total_duration_ms, approx_ms))))
    peaks = local_peak_indices(values)
    if not peaks:
        return int(round(max(0.0, min(total_duration_ms, approx_ms))))
    peak_index = max(peaks, key=lambda index: values[index])
    noise_floor = percentile(values, 0.2)
    peak_value = float(values[peak_index])
    attack_threshold = max(MIN_ENVELOPE_THRESHOLD * 0.4, noise_floor * 1.7, peak_value * 0.18)
    onset_index = peak_index
    while onset_index > 0 and values[onset_index - 1] >= attack_threshold:
        onset_index -= 1
    return int(round(max(0.0, min(total_duration_ms, start_ms + onset_index * hop_ms))))


def dense_review_peaks(y: np.ndarray) -> list[dict[str, Any]]:
    frame_size = max(32, int(round(TARGET_SR * FRAME_MS / 1000.0)))
    hop_size = max(16, int(round(frame_size / 2)))
    envelope: list[float] = []
    for start in range(0, max(0, len(y) - frame_size + 1), hop_size):
        clip = y[start:start + frame_size]
        if len(clip) == frame_size:
            envelope.append(float(np.sqrt(np.mean(np.square(clip)))))
    values = np.asarray(envelope, dtype=np.float64)
    if len(values) < 3:
        return []
    mean = float(np.mean(values))
    p90 = percentile(values, 0.9)
    p98 = percentile(values, 0.98)
    threshold = max(MIN_ENVELOPE_THRESHOLD * 0.65, mean * 1.45, p90 * 0.38, p98 * 0.18)
    local_peaks = [
        {"frame": index, "score": float(values[index])}
        for index in range(1, len(values) - 1)
        if values[index] >= threshold and values[index] >= values[index - 1] and values[index] >= values[index + 1]
    ]
    local_peaks.sort(key=lambda row: row["score"], reverse=True)
    min_gap_frames = max(1, int(round(DENSE_CANDIDATE_GAP_MS / ((hop_size / TARGET_SR) * 1000.0))))
    accepted: list[dict[str, Any]] = []
    for peak in local_peaks:
        if any(abs(int(existing["frame"]) - int(peak["frame"])) <= min_gap_frames for existing in accepted):
            continue
        accepted.append(peak)
    accepted.sort(key=lambda row: int(row["frame"]))
    return [
        {
            "timestamp_ms": int(round((int(peak["frame"]) * hop_size / TARGET_SR) * 1000.0)),
            "refined_timestamp_ms": refine_attack_timestamp(
                y,
                (int(peak["frame"]) * hop_size / TARGET_SR) * 1000.0,
            ),
            "score": float(peak["score"]),
        }
        for peak in accepted
    ]


def nearest_gap(anchor_ms: int, timestamps: list[int]) -> int | None:
    if not timestamps:
        return None
    return min(abs(anchor_ms - timestamp) for timestamp in timestamps)


def recovery_anchors(y: np.ndarray, saved_timestamps: list[int]) -> list[dict[str, Any]]:
    peaks = []
    for peak in dense_review_peaks(y):
        anchor_ms = int(peak["refined_timestamp_ms"])
        known_gap = nearest_gap(anchor_ms, saved_timestamps)
        if known_gap is not None and known_gap < RECOVERY_MIN_GAP_FROM_KNOWN_MS:
            continue
        if known_gap is not None and known_gap > RECOVERY_MAX_GAP_FROM_SAVED_MS:
            continue
        peaks.append(peak)
    peaks.sort(key=lambda row: float(row["score"]), reverse=True)
    selected: list[dict[str, Any]] = []
    for peak in peaks[:RECOVERY_MAX_CANDIDATES]:
        anchor_ms = int(peak["refined_timestamp_ms"])
        selected_gap = nearest_gap(anchor_ms, [int(row["refined_timestamp_ms"]) for row in selected])
        if selected_gap is not None and selected_gap < RECOVERY_MIN_GAP_FROM_RECOVERY_MS:
            continue
        selected.append(peak)
    return sorted(selected, key=lambda row: int(row["refined_timestamp_ms"]))


def feature_row_for_anchor(
    y: np.ndarray,
    anchor_ms: int,
    candidate_timestamps: list[int],
) -> dict[str, float]:
    row: dict[str, float] = {}
    for window_name, before_ms, after_ms in WINDOWS:
        clip = extract_window(y, anchor_ms, before_ms, after_ms)
        row.update(prefixed_features(extract_features(clip, TARGET_SR), window_name))
    row.update(context_features(anchor_ms, candidate_timestamps, is_saved_candidate=False))
    return row


def visible_recovery_prediction(label: str, confidence: float, nearest_saved_gap_ms: int | None) -> bool:
    if label == "racket_contact":
        return (
            confidence >= RECOVERY_RACKET_MIN_CONFIDENCE
            and (nearest_saved_gap_ms is None or nearest_saved_gap_ms >= RECOVERY_RACKET_MIN_SAVED_GAP_MS)
        )
    if label == "table_bounce":
        return (
            confidence >= RECOVERY_TABLE_MIN_CONFIDENCE
            and (nearest_saved_gap_ms is None or nearest_saved_gap_ms >= RECOVERY_TABLE_MIN_SAVED_GAP_MS)
        )
    return False


def nearest_truth(anchor_ms: int, truths: list[TruthMarker]) -> tuple[TruthMarker | None, int | None]:
    best: TruthMarker | None = None
    best_delta: int | None = None
    for truth in truths:
        delta = anchor_ms - truth.timestamp_ms
        if best_delta is None or abs(delta) < abs(best_delta):
            best = truth
            best_delta = delta
    return best, best_delta


def main() -> None:
    model = json.loads(DEFAULT_APP_MODEL.read_text(encoding="utf-8"))
    prediction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for session_path in default_session_paths():
        data = json.loads(session_path.read_text(encoding="utf-8"))
        for event_index, event in enumerate(data.get("events") or []):
            truths = build_truth_markers((event.get("review") or {}).get("markers") or [])
            if not truths:
                continue
            app_candidates = candidate_peaks_from_app(event)
            saved_timestamps = sorted({candidate.timestamp_ms for candidate in app_candidates})
            baseline_matches = match_detectable_candidates(app_candidates, truths, MATCH_TOLERANCE_MS)
            baseline_matched_truth_ids = set(baseline_matches.values())
            missed_truths = [truth for truth in truths if truth.marker_id not in baseline_matched_truth_ids]

            wav_path = resolve_wav_path(session_path, event)
            if wav_path is None:
                continue
            y, _sr = load_audio(str(wav_path))
            anchors = recovery_anchors(y, saved_timestamps)
            all_timestamps = sorted(set(saved_timestamps + [int(anchor["refined_timestamp_ms"]) for anchor in anchors]))

            feature_rows = [
                feature_row_for_anchor(y, int(anchor["refined_timestamp_ms"]), all_timestamps)
                for anchor in anchors
            ]
            if not feature_rows:
                continue
            features_df = pd.DataFrame(feature_rows)
            predictions, confidences, probabilities = predict_app_model(model, features_df)

            visible_rows: list[dict[str, Any]] = []
            for index, anchor in enumerate(anchors):
                label = predictions[index]
                confidence = float(confidences[index])
                anchor_ms = int(anchor["refined_timestamp_ms"])
                truth, delta = nearest_truth(anchor_ms, truths)
                row = {
                    "session_id": session_path.stem,
                    "event_index": event_index,
                    "candidate_id": f"t0014_recovery_{index}_{anchor_ms}",
                    "anchor_ms": anchor_ms,
                    "score": round(float(anchor["score"]), 6),
                    "nearest_saved_gap_ms": nearest_gap(anchor_ms, saved_timestamps),
                    "prediction": label,
                    "confidence": confidence,
                    "probability_racket_contact": probabilities.get("racket_contact", [None] * len(anchors))[index],
                    "probability_table_bounce": probabilities.get("table_bounce", [None] * len(anchors))[index],
                    "probability_non_target": probabilities.get("non_target", [None] * len(anchors))[index],
                    "visible": visible_recovery_prediction(label, confidence, nearest_gap(anchor_ms, saved_timestamps)),
                    "nearest_truth_id": truth.marker_id if truth else "",
                    "nearest_truth_kind": truth.truth_kind if truth else "",
                    "nearest_truth_delta_ms": delta if delta is not None else "",
                    "nearest_truth_was_baseline_matched": bool(truth and truth.marker_id in baseline_matched_truth_ids),
                }
                if row["visible"]:
                    visible_rows.append(row)
                prediction_rows.append(row)

            matched_recovery_truths: set[str] = set()
            recovered = 0
            wrong_class = 0
            duplicate_near_baseline = 0
            false_positive = 0
            for row in sorted(visible_rows, key=lambda item: abs(int(item["nearest_truth_delta_ms"]) if item["nearest_truth_delta_ms"] != "" else 999999)):
                truth_id = str(row["nearest_truth_id"])
                truth_kind = str(row["nearest_truth_kind"])
                delta = row["nearest_truth_delta_ms"]
                near_truth = delta != "" and abs(int(delta)) <= MATCH_TOLERANCE_MS
                if not near_truth:
                    false_positive += 1
                    continue
                if bool(row["nearest_truth_was_baseline_matched"]):
                    duplicate_near_baseline += 1
                    continue
                if truth_id in matched_recovery_truths:
                    false_positive += 1
                    continue
                matched_recovery_truths.add(truth_id)
                if str(row["prediction"]) == truth_kind:
                    recovered += 1
                else:
                    wrong_class += 1

            missed_racket = sum(1 for truth in missed_truths if truth.truth_kind == "racket_contact")
            missed_table = sum(1 for truth in missed_truths if truth.truth_kind == "table_bounce")
            summary_rows.append({
                "session_id": session_path.stem,
                "event_index": event_index,
                "saved_candidates": len(app_candidates),
                "baseline_matched_truths": len(baseline_matched_truth_ids),
                "baseline_missed_truths": len(missed_truths),
                "baseline_missed_racket": missed_racket,
                "baseline_missed_table": missed_table,
                "recovery_candidates": len(anchors),
                "visible_recovery_candidates": len(visible_rows),
                "recovered_correct": recovered,
                "wrong_class_near_missed": wrong_class,
                "duplicate_near_baseline_truth": duplicate_near_baseline,
                "false_positive_visible": false_positive,
            })

    predictions_df = pd.DataFrame(prediction_rows)
    summary_df = pd.DataFrame(summary_rows)
    PREDICTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(PREDICTIONS_CSV, index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    totals = summary_df.sum(numeric_only=True).to_dict()
    lines = [
        "# Playing Retro Audio T0014 Recovery Replay",
        "",
        "This replay mirrors the app-side T0014 recovery pass. It does not train, export, build an APK, or change `studs_live`.",
        "",
        "## Summary",
        "",
        f"- Baseline missed truths: `{int(totals.get('baseline_missed_truths', 0))}`",
        f"- Baseline missed racket/table: `{int(totals.get('baseline_missed_racket', 0))}` / `{int(totals.get('baseline_missed_table', 0))}`",
        f"- Recovery anchors scored: `{int(totals.get('recovery_candidates', 0))}`",
        f"- Visible recovery candidates: `{int(totals.get('visible_recovery_candidates', 0))}`",
        f"- Correct recovered missed truths: `{int(totals.get('recovered_correct', 0))}`",
        f"- Wrong-class near missed truths: `{int(totals.get('wrong_class_near_missed', 0))}`",
        f"- Duplicate near already matched truths: `{int(totals.get('duplicate_near_baseline_truth', 0))}`",
        f"- Visible false positives: `{int(totals.get('false_positive_visible', 0))}`",
        "",
        "## Per Session",
        "",
        "| Session | Missed | Recovery Visible | Recovered | Wrong Class | Duplicate | FP |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {session_id} | {baseline_missed_truths} | {visible_recovery_candidates} | {recovered_correct} | {wrong_class_near_missed} | {duplicate_near_baseline_truth} | {false_positive_visible} |".format(**row)
        )
    lines.extend([
        "",
        "## Outputs",
        "",
        f"- Predictions CSV: `{PREDICTIONS_CSV.as_posix()}`",
        f"- Summary CSV: `{SUMMARY_CSV.as_posix()}`",
    ])
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {PREDICTIONS_CSV}")
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {REPORT_MD}")
    print(
        "baseline_missed={missed} visible={visible} recovered={recovered} wrong={wrong} duplicate={dup} fp={fp}".format(
            missed=int(totals.get("baseline_missed_truths", 0)),
            visible=int(totals.get("visible_recovery_candidates", 0)),
            recovered=int(totals.get("recovered_correct", 0)),
            wrong=int(totals.get("wrong_class_near_missed", 0)),
            dup=int(totals.get("duplicate_near_baseline_truth", 0)),
            fp=int(totals.get("false_positive_visible", 0)),
        )
    )


if __name__ == "__main__":
    main()
