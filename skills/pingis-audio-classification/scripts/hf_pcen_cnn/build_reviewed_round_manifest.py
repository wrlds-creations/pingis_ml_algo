"""Build a classifier-ready candidate manifest from a reviewed STIGA round."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .audio_io import load_mono_audio
from .candidate_labels import CandidateAssignment, ReviewedEvent, assign_candidates
from .fable_baseline import (
    FableHgbRuntime,
    extract_and_predict_fable,
    extract_fable_clip,
)
from .hf_gate import HFGateConfig, TARGET_SAMPLE_RATE, detect_hf_candidates


DEFAULT_MODEL = Path("apps/collector/src/models/fable_audio_model.json")
RACKET_LABELS = frozenset({"racket", "racket_bounce"})

TABLE_SCENARIOS = frozenset(
    {
        "surface_impacts",
        "surface_impacts_music_1",
    }
)
IMPACT_SCENARIOS = frozenset(
    {
        "mixed_household_impacts",
        "racket_handling",
        "racket_drop",
    }
)


def reviewed_racket_times(labels_path: Path) -> list[float]:
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    return sorted(
        float(marker["time_s"]) * 1000.0
        for marker in payload.get("manual_markers", [])
        if str(marker.get("label", "")).strip().lower() in RACKET_LABELS
    )


def negative_class_for_record(record: dict[str, Any]) -> str:
    """Return the most defensible broad non-racket class for a reviewed take."""
    scenario = str(record.get("corrected_scenario_id", "")).strip().lower()
    if scenario in TABLE_SCENARIOS:
        return "table_bounce"
    if scenario in IMPACT_SCENARIOS:
        return "floor_other_impact"
    return "voice_music_noise"


def _candidate_label(
    assignment: CandidateAssignment,
    record: dict[str, Any],
) -> tuple[str | None, str]:
    if assignment.disposition == "positive":
        return assignment.label, "reviewed_racket_timestamp"
    if assignment.disposition != "hard_negative":
        return None, "not_trainable"
    return negative_class_for_record(record), "reviewed_session_scenario"


def _resolve_path(review_pack: Path, record: dict[str, Any], key: str, folder: str) -> Path:
    stored = Path(str(record[key]))
    if stored.exists():
        return stored.resolve()
    fallback = review_pack / folder / stored.name
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(f"Missing {key} for {record.get('session_id')}: {stored}")


def _route_class(input_route: Any) -> str:
    route = str(input_route or "").lower()
    if "builtin" in route or "type_15" in route:
        return "builtin"
    if any(token in route for token in ("bluetooth", "type_7", "type_8", "type_26")):
        return "bluetooth"
    if any(token in route for token in ("wired", "usb", "type_11", "type_22")):
        return "external"
    return "unknown"


def build_reviewed_round_manifest(
    review_pack: Path,
    model_path: Path,
    output_path: Path,
    *,
    cutoff_hz: float = 8_000.0,
    onset_ratio: float = 1.85,
    mad_multiplier: float = 2.0,
    match_ms: float = 140.0,
    ambiguity_ms: float = 300.0,
) -> dict[str, Any]:
    """Generate one row per HF candidate while preserving the frozen split."""
    pack = review_pack.resolve()
    payload = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not records:
        raise ValueError("Review-pack manifest contains no records")

    runtime = FableHgbRuntime.from_path(model_path)
    gate_config = HFGateConfig(
        cutoff_hz=cutoff_hz,
        onset_ratio=onset_ratio,
        mad_multiplier=mad_multiplier,
    )
    rows: list[dict[str, Any]] = []
    session_reports: list[dict[str, Any]] = []

    for record_number, record in enumerate(records, start=1):
        session_id = str(record["session_id"])
        audio_path = _resolve_path(pack, record, "linked_wav", "raw")
        labels_path = _resolve_path(pack, record, "review_labels", "review_labels")
        session_path = _resolve_path(pack, record, "linked_json", "raw")
        session = json.loads(session_path.read_text(encoding="utf-8"))
        capture = session.get("capture", {})
        device = session.get("device", {})
        device_id = str(record.get("install_id") or device.get("installId"))
        if not device_id:
            raise ValueError(f"Session {session_id} has no stable device identity")

        pcm, _ = load_mono_audio(audio_path, TARGET_SAMPLE_RATE)
        reviewed_ms = reviewed_racket_times(labels_path)
        candidates = detect_hf_candidates(pcm, TARGET_SAMPLE_RATE, gate_config)
        explicit_negative = (
            str(record.get("binary_label", "")).strip().lower() == "not_racket"
            and not reviewed_ms
        )
        assignments = assign_candidates(
            [float(candidate["onset_ms"]) for candidate in candidates],
            [ReviewedEvent(timestamp_ms, "racket_bounce") for timestamp_ms in reviewed_ms],
            match_ms=match_ms,
            ambiguity_ms=ambiguity_ms,
            review_complete=True,
            explicit_negative_session=explicit_negative,
        )
        matched_events = {
            assignment.matched_event_index
            for assignment in assignments
            if assignment.matched_event_index is not None
            and assignment.disposition == "positive"
        }
        session_reports.append(
            {
                "session_id": session_id,
                "take_id": record["take_id"],
                "device_id": device_id,
                "device_alias": record.get("canonical_device_alias"),
                "dataset_split": record["dataset_split"],
                "truth_events": len(reviewed_ms),
                "matched_events": len(matched_events),
                "missed_events": len(reviewed_ms) - len(matched_events),
                "candidate_count": len(candidates),
                "duration_s": len(pcm) / TARGET_SAMPLE_RATE,
            }
        )

        for candidate, assignment in zip(candidates, assignments, strict=True):
            label, label_source = _candidate_label(assignment, record)
            clip = extract_fable_clip(pcm, int(candidate["onset_sample"]))
            fable_features, fable_prediction = extract_and_predict_fable(runtime, clip)
            probabilities = dict(fable_prediction["probabilities"])
            row: dict[str, Any] = {
                "config_id": (
                    f"hf{int(cutoff_hz)}_r{onset_ratio:g}_m{mad_multiplier:g}"
                ),
                "cutoff_hz": cutoff_hz,
                "onset_ratio": onset_ratio,
                "mad_multiplier": mad_multiplier,
                "round_id": record.get("canonical_round_id"),
                "take_id": record["take_id"],
                "take_number": record.get("take_number"),
                "physical_event_group_id": record["physical_event_group_id"],
                "dataset_split": record["dataset_split"],
                "session_id": session_id,
                "source": "reviewed:round_CJ-20260723-01",
                "device_id": device_id,
                "device_alias": record.get("canonical_device_alias"),
                "device_manufacturer": record.get("manufacturer")
                or device.get("manufacturer"),
                "device_model": record.get("device_model") or device.get("model"),
                "device_platform": record.get("platform") or device.get("platform"),
                "device_os_version": record.get("os_version") or device.get("osVersion"),
                "scenario_id": record["corrected_scenario_id"],
                "binary_label": record.get("binary_label"),
                "audio_path": str(audio_path),
                "labels_path": str(labels_path),
                "requested_sample_rate": capture.get("requestedSampleRate"),
                "input_sample_rate": capture.get("inputSampleRate"),
                "sample_rate": capture.get("outputSampleRate", TARGET_SAMPLE_RATE),
                "audio_source": capture.get("audioSource") or capture.get("source"),
                "input_route": capture.get("inputRoute"),
                "route_class": _route_class(capture.get("inputRoute")),
                "review_complete": True,
                "explicit_negative": explicit_negative,
                "expected_bounce_count": len(reviewed_ms),
                "candidate_index": assignment.candidate_index,
                "onset_sample": int(candidate["onset_sample"]),
                "onset_ms": float(candidate["onset_ms"]),
                "hf_rms": float(candidate["hf_rms"]),
                "background_rms": float(candidate["background_rms"]),
                "threshold": float(candidate["threshold"]),
                "full_band_peak": float(candidate["full_band_peak"]),
                "disposition": assignment.disposition,
                "label": label,
                "label_source": label_source,
                "matched_event_index": assignment.matched_event_index,
                "reviewed_event_ms": (
                    reviewed_ms[assignment.matched_event_index]
                    if assignment.matched_event_index is not None
                    else None
                ),
                "distance_ms": assignment.distance_ms,
                "reason": assignment.reason,
                "background_db": float(fable_features.get("nr_bg_rms_db", -100.0)),
                "fable_predicted_label": str(fable_prediction["label"]),
                "fable_confidence": float(fable_prediction["confidence"]),
            }
            for feature_name, feature_value in fable_features.items():
                row[f"fable_feature_{feature_name}"] = float(feature_value)
            for class_name, probability in probabilities.items():
                row[f"fable_prob_{class_name}"] = float(probability)
            rows.append(row)

        if record_number % 12 == 0:
            print(f"Processed {record_number} of {len(records)} reviewed sessions")

    frame = pd.DataFrame(rows).sort_values(
        ["dataset_split", "take_number", "device_alias", "onset_ms"],
        ignore_index=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    sessions = pd.DataFrame(session_reports)
    sessions.to_csv(output_path.with_suffix(".sessions.csv"), index=False)
    train_sessions = sessions[sessions["dataset_split"].eq("train")]
    holdout_sessions = sessions[sessions["dataset_split"].eq("holdout")]
    report: dict[str, Any] = {
        "review_pack": str(pack),
        "output": str(output_path.resolve()),
        "model": str(model_path.resolve()),
        "gate": {
            "cutoff_hz": cutoff_hz,
            "onset_ratio": onset_ratio,
            "mad_multiplier": mad_multiplier,
            "match_ms": match_ms,
            "ambiguity_ms": ambiguity_ms,
        },
        "rows": int(len(frame)),
        "sessions": int(len(sessions)),
        "takes": int(sessions["take_id"].nunique()),
        "devices": sorted(sessions["device_alias"].unique().tolist()),
        "splits": {
            str(key): int(value)
            for key, value in frame["dataset_split"].value_counts().items()
        },
        "dispositions": {
            str(key): int(value)
            for key, value in frame["disposition"].value_counts().items()
        },
        "labels": {
            str(key): int(value)
            for key, value in frame["label"].dropna().value_counts().items()
        },
        "route_classes": dict(Counter(frame["route_class"].astype(str))),
        "train_gate_recall": {
            "matched": int(train_sessions["matched_events"].sum()),
            "truth": int(train_sessions["truth_events"].sum()),
            "missed": int(train_sessions["missed_events"].sum()),
            "recall": (
                float(train_sessions["matched_events"].sum())
                / float(train_sessions["truth_events"].sum())
            ),
            "candidates_per_minute": (
                float(train_sessions["candidate_count"].sum())
                / (float(train_sessions["duration_s"].sum()) / 60.0)
            ),
        },
        "sealed_holdout": {
            "take_ids": sorted(holdout_sessions["take_id"].unique().tolist()),
            "sessions": int(len(holdout_sessions)),
            "rows_written_but_forbidden_during_model_selection": int(
                frame["dataset_split"].eq("holdout").sum()
            ),
            "policy": (
                "Holdout rows remain in the manifest for one-shot final evaluation. "
                "Training and model-selection code must select dataset_split=train."
            ),
        },
    }
    output_path.with_suffix(".contract.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-pack", type=Path, required=True)
    parser.add_argument("--model-json", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoff-hz", type=float, default=8_000.0)
    parser.add_argument("--onset-ratio", type=float, default=1.85)
    parser.add_argument("--mad-multiplier", type=float, default=2.0)
    parser.add_argument("--match-ms", type=float, default=140.0)
    parser.add_argument("--ambiguity-ms", type=float, default=300.0)
    args = parser.parse_args()
    report = build_reviewed_round_manifest(
        args.review_pack,
        args.model_json,
        args.output,
        cutoff_hz=args.cutoff_hz,
        onset_ratio=args.onset_ratio,
        mad_multiplier=args.mad_multiplier,
        match_ms=args.match_ms,
        ambiguity_ms=args.ambiguity_ms,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
