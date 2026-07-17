"""Audit reviewed HF-gate candidates through the deployed Fable classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .audio_io import load_mono_audio
from .candidate_labels import ReviewedEvent, assign_candidates
from .fable_baseline import (
    FableHgbRuntime,
    FableTimingConfig,
    extract_and_predict_fable,
    extract_fable_clip,
    replay_deployed_fable_timing,
)
from .hf_gate import HFGateConfig, TARGET_SAMPLE_RATE, detect_hf_candidates


RACKET_LABEL = "racket_bounce"
DEFAULT_MODEL = Path("apps/collector/src/models/fable_audio_model.json")


def greedy_time_match(
    predicted_ms: Iterable[float],
    truth_ms: Iterable[float],
    tolerance_ms: float,
) -> tuple[dict[int, int], set[int]]:
    """Match predicted and reviewed times one-to-one by smallest distance."""
    predicted = [float(value) for value in predicted_ms]
    truth = [float(value) for value in truth_ms]
    pairs: list[tuple[float, int, int]] = []
    for predicted_index, predicted_time in enumerate(predicted):
        for truth_index, truth_time in enumerate(truth):
            distance = abs(predicted_time - truth_time)
            if distance <= tolerance_ms:
                pairs.append((distance, predicted_index, truth_index))
    pairs.sort(key=lambda item: (item[0], item[1], item[2]))

    matched_predictions: dict[int, int] = {}
    matched_truth: set[int] = set()
    for _, predicted_index, truth_index in pairs:
        if predicted_index in matched_predictions or truth_index in matched_truth:
            continue
        matched_predictions[predicted_index] = truth_index
        matched_truth.add(truth_index)
    return matched_predictions, matched_truth


def loud_threshold_grid() -> list[float]:
    """Return a stable sweep that always includes the deployed 0.85 value."""
    return [round(value / 100.0, 2) for value in range(5, 96, 5)]


def classify_rejected_positive(row: Mapping[str, object]) -> str:
    if bool(row.get("counted")):
        return "positive_counted"
    if str(row.get("predicted_label")) != RACKET_LABEL:
        return "positive_top_label_reject"
    reason = str(row.get("timing_reason", ""))
    if reason.startswith("low_confidence"):
        return "positive_low_confidence"
    return f"positive_timing_{reason or 'unknown'}"


def _manifest_path(pack: Path, stored: str, folder: str) -> Path:
    path = Path(stored)
    if path.exists():
        return path
    fallback = pack / folder / path.name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Review-pack file not found: {stored}")


def _reviewed_times(labels_path: Path) -> list[float]:
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    return sorted(
        float(marker["time_s"]) * 1000.0
        for marker in payload.get("manual_markers", [])
        if str(marker.get("label", "")).strip().lower() in {"racket", "racket_bounce"}
    )


def _evaluated_candidates(
    pcm: np.ndarray,
    reviewed_ms: list[float],
    explicit_negative: bool,
    runtime: FableHgbRuntime,
    gate_config: HFGateConfig,
    match_ms: float,
) -> list[dict[str, object]]:
    candidates = detect_hf_candidates(pcm, TARGET_SAMPLE_RATE, gate_config)
    assignments = assign_candidates(
        [float(candidate["onset_ms"]) for candidate in candidates],
        [ReviewedEvent(timestamp_ms, RACKET_LABEL) for timestamp_ms in reviewed_ms],
        match_ms=match_ms,
        ambiguity_ms=300.0,
        review_complete=True,
        explicit_negative_session=explicit_negative,
    )
    evaluated: list[dict[str, object]] = []
    for candidate, assignment in zip(candidates, assignments, strict=True):
        clip = extract_fable_clip(pcm, int(candidate["onset_sample"]))
        features, prediction = extract_and_predict_fable(runtime, clip)
        probabilities = dict(prediction["probabilities"])
        row: dict[str, object] = {
            "candidate_index": assignment.candidate_index,
            "onset_sample": int(candidate["onset_sample"]),
            "onset_ms": float(candidate["onset_ms"]),
            "frame_rms": float(candidate["hf_rms"]),
            "background_rms": float(candidate["background_rms"]),
            "gate_threshold": float(candidate["threshold"]),
            "full_band_peak": float(candidate["full_band_peak"]),
            "disposition": assignment.disposition,
            "label": assignment.label,
            "matched_event_index": assignment.matched_event_index,
            "distance_ms": assignment.distance_ms,
            "assignment_reason": assignment.reason,
            "predicted_label": str(prediction["label"]),
            "confidence": float(prediction["confidence"]),
            "prob_racket_bounce": float(probabilities.get(RACKET_LABEL, 0.0)),
            "background_db": float(features.get("nr_bg_rms_db", -100.0)),
        }
        for label, probability in probabilities.items():
            row[f"prob_{label}"] = float(probability)
        evaluated.append(row)
    return evaluated


def _apply_timing(
    candidates: list[dict[str, object]],
    *,
    loud_threshold: float,
) -> list[dict[str, object]]:
    config = FableTimingConfig(loud_confidence=loud_threshold)
    decisions = replay_deployed_fable_timing(candidates, config)
    output: list[dict[str, object]] = []
    for row, decision in zip(candidates, decisions, strict=True):
        updated = dict(row)
        updated["counted"] = decision.counted
        updated["timing_reason"] = decision.reason
        updated["fast_rebound"] = decision.fast_rebound
        output.append(updated)
    return output


def _score_session(
    candidates: list[dict[str, object]],
    reviewed_ms: list[float],
    match_ms: float,
) -> tuple[dict[str, int], dict[int, int]]:
    counted = [row for row in candidates if bool(row["counted"])]
    matches, matched_truth = greedy_time_match(
        [float(row["onset_ms"]) for row in counted],
        reviewed_ms,
        match_ms,
    )
    tp = len(matches)
    return {
        "tp": tp,
        "fp": len(counted) - tp,
        "fn": len(reviewed_ms) - len(matched_truth),
        "counted": len(counted),
    }, matches


def _probability_summary(rows: pd.DataFrame) -> dict[str, object]:
    if rows.empty:
        return {"count": 0}
    values = rows["prob_racket_bounce"].astype(float)
    return {
        "count": int(len(rows)),
        "top_label_racket": int(rows["predicted_label"].eq(RACKET_LABEL).sum()),
        "prob_racket_min": float(values.min()),
        "prob_racket_p25": float(values.quantile(0.25)),
        "prob_racket_median": float(values.median()),
        "prob_racket_p75": float(values.quantile(0.75)),
        "prob_racket_max": float(values.max()),
    }


def audit_review_pack(
    review_pack: Path,
    model_path: Path,
    output_dir: Path,
    *,
    cutoff_hz: float = 7_000.0,
    onset_ratio: float = 3.0,
    match_ms: float = 140.0,
) -> dict[str, object]:
    manifest = json.loads((review_pack / "manifest.json").read_text(encoding="utf-8"))
    records = list(manifest["records"])
    runtime = FableHgbRuntime.from_path(model_path)
    gate_config = HFGateConfig(cutoff_hz=cutoff_hz, onset_ratio=onset_ratio)
    candidate_rows: list[dict[str, object]] = []
    session_inputs: list[tuple[dict[str, object], list[dict[str, object]], list[float]]] = []

    for index, record in enumerate(records, start=1):
        wav_path = _manifest_path(review_pack, str(record["raw_wav"]), "raw")
        labels_path = _manifest_path(review_pack, str(record["labels_json"]), "review_labels")
        pcm, _ = load_mono_audio(wav_path, TARGET_SAMPLE_RATE)
        reviewed_ms = _reviewed_times(labels_path)
        evaluated = _evaluated_candidates(
            pcm,
            reviewed_ms,
            int(record["expected_count"]) == 0,
            runtime,
            gate_config,
            match_ms,
        )
        print(f"[{index}/{len(records)}] {record['session_id']}: {len(evaluated)} candidates")
        session_inputs.append((record, evaluated, reviewed_ms))

    deployed_session_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    missing_event_rows: list[dict[str, object]] = []
    for record, evaluated, reviewed_ms in session_inputs:
        timed = _apply_timing(evaluated, loud_threshold=0.85)
        score, counted_matches = _score_session(timed, reviewed_ms, match_ms)
        counted_index = 0
        for row in timed:
            output = {
                "session_id": record["session_id"],
                "device_alias": record["device_alias"],
                "device_folder": record["device_folder"],
                "scenario_id": record["scenario_id"],
                "expected_count": record["expected_count"],
                **row,
            }
            output["error_category"] = (
                classify_rejected_positive(output)
                if output["disposition"] == "positive"
                else "negative_false_count"
                if output["disposition"] == "hard_negative" and output["counted"]
                else ""
            )
            if output["counted"]:
                output["final_matched_event_index"] = counted_matches.get(counted_index)
                output["final_false_count"] = counted_index not in counted_matches
                counted_index += 1
            else:
                output["final_matched_event_index"] = None
                output["final_false_count"] = False
            candidate_rows.append(output)
            if (
                output["disposition"] == "positive" and not output["counted"]
            ) or bool(output["final_false_count"]):
                error_rows.append(output)

        matched_truth = set(counted_matches.values())
        for event_index, reviewed_timestamp_ms in enumerate(reviewed_ms):
            if event_index in matched_truth:
                continue
            event_candidates = sorted(
                (
                    row
                    for row in timed
                    if row["disposition"] == "positive"
                    and row["matched_event_index"] == event_index
                ),
                key=lambda row: float(row["distance_ms"]),
            )
            best_candidate = event_candidates[0] if event_candidates else None
            if best_candidate is None:
                miss_reason = "gate_miss"
            elif any(bool(row["counted"]) for row in event_candidates):
                miss_reason = "final_match_conflict"
            else:
                miss_reason = classify_rejected_positive(best_candidate)
            missing_event_rows.append(
                {
                    "session_id": record["session_id"],
                    "device_alias": record["device_alias"],
                    "device_folder": record["device_folder"],
                    "scenario_id": record["scenario_id"],
                    "reviewed_event_index": event_index,
                    "reviewed_timestamp_ms": reviewed_timestamp_ms,
                    "miss_reason": miss_reason,
                    "candidate_count": len(event_candidates),
                    "candidate_index": (
                        best_candidate["candidate_index"] if best_candidate else None
                    ),
                    "candidate_onset_ms": (
                        best_candidate["onset_ms"] if best_candidate else None
                    ),
                    "candidate_distance_ms": (
                        best_candidate["distance_ms"] if best_candidate else None
                    ),
                    "predicted_label": (
                        best_candidate["predicted_label"] if best_candidate else None
                    ),
                    "confidence": (
                        best_candidate["confidence"] if best_candidate else None
                    ),
                    "prob_racket_bounce": (
                        best_candidate["prob_racket_bounce"] if best_candidate else None
                    ),
                    "background_db": (
                        best_candidate["background_db"] if best_candidate else None
                    ),
                    "timing_reason": (
                        best_candidate["timing_reason"] if best_candidate else None
                    ),
                }
            )

        matched_gate_events = {
            int(row["matched_event_index"])
            for row in evaluated
            if row["disposition"] == "positive" and row["matched_event_index"] is not None
        }
        deployed_session_rows.append(
            {
                "session_id": record["session_id"],
                "device_alias": record["device_alias"],
                "scenario_id": record["scenario_id"],
                "expected_count": record["expected_count"],
                "reviewed_events": len(reviewed_ms),
                "gate_matched_events": len(matched_gate_events),
                "candidate_count": len(timed),
                **score,
            }
        )

    sweep_rows: list[dict[str, object]] = []
    for threshold in loud_threshold_grid():
        total = {"tp": 0, "fp": 0, "fn": 0, "counted": 0}
        exact_sessions = 0
        absolute_errors: list[int] = []
        for record, evaluated, reviewed_ms in session_inputs:
            timed = _apply_timing(evaluated, loud_threshold=threshold)
            score, _ = _score_session(timed, reviewed_ms, match_ms)
            for key in total:
                total[key] += score[key]
            error = score["counted"] - len(reviewed_ms)
            absolute_errors.append(abs(error))
            exact_sessions += int(error == 0)
        precision = total["tp"] / (total["tp"] + total["fp"]) if total["tp"] + total["fp"] else 0.0
        recall = total["tp"] / (total["tp"] + total["fn"]) if total["tp"] + total["fn"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        sweep_rows.append(
            {
                "loud_threshold": threshold,
                **total,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "session_count_mae": float(np.mean(absolute_errors)),
                "exact_sessions": exact_sessions,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_frame = pd.DataFrame(candidate_rows)
    sessions_frame = pd.DataFrame(deployed_session_rows)
    errors_frame = pd.DataFrame(error_rows)
    missing_events_frame = pd.DataFrame(missing_event_rows)
    sweep_frame = pd.DataFrame(sweep_rows)
    candidates_frame.to_csv(output_dir / "candidate_audit.csv", index=False)
    sessions_frame.to_csv(output_dir / "session_audit.csv", index=False)
    errors_frame.to_csv(output_dir / "error_cases.csv", index=False)
    missing_events_frame.to_csv(output_dir / "missing_reviewed_events.csv", index=False)
    sweep_frame.to_csv(output_dir / "loud_threshold_sweep.csv", index=False)

    positive = candidates_frame[candidates_frame["disposition"] == "positive"]
    negatives = candidates_frame[candidates_frame["disposition"] == "hard_negative"]
    false_counts = candidates_frame[candidates_frame["final_false_count"]]
    deployed = sweep_frame[sweep_frame["loud_threshold"] == 0.85].iloc[0].to_dict()
    best = sweep_frame.sort_values(
        ["f1", "session_count_mae", "fp"], ascending=[False, True, True]
    ).iloc[0].to_dict()
    report: dict[str, object] = {
        "review_pack": str(review_pack.resolve()),
        "model_path": str(model_path.resolve()),
        "gate": {"cutoff_hz": cutoff_hz, "onset_ratio": onset_ratio},
        "match_ms": match_ms,
        "sessions": len(records),
        "reviewed_events": int(sum(len(item[2]) for item in session_inputs)),
        "gate_matched_events": int(sessions_frame["gate_matched_events"].sum()),
        "deployed_threshold": deployed,
        "best_swept_threshold": best,
        "positive_candidates": _probability_summary(positive),
        "hard_negative_candidates": _probability_summary(negatives),
        "final_false_counts": _probability_summary(false_counts),
        "positive_rejection_reasons": {
            str(key): int(value)
            for key, value in positive["error_category"].value_counts().items()
        },
        "final_miss_reasons": {
            str(key): int(value)
            for key, value in missing_events_frame["miss_reason"].value_counts().items()
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-pack", type=Path, required=True)
    parser.add_argument("--model-json", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cutoff-hz", type=float, default=7_000.0)
    parser.add_argument("--onset-ratio", type=float, default=3.0)
    parser.add_argument("--match-ms", type=float, default=140.0)
    args = parser.parse_args()
    output_dir = args.output_dir or args.review_pack / "analysis" / "fable_error_audit"
    report = audit_review_pack(
        args.review_pack,
        args.model_json,
        output_dir,
        cutoff_hz=args.cutoff_hz,
        onset_ratio=args.onset_ratio,
        match_ms=args.match_ms,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
