"""Build a classifier-ready HF-candidate manifest from a completed review pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .audio_io import load_mono_audio
from .candidate_labels import ReviewedEvent, assign_candidates
from .fable_baseline import (
    FableHgbRuntime,
    extract_and_predict_fable,
    extract_fable_clip,
)
from .hf_gate import HFGateConfig, TARGET_SAMPLE_RATE, detect_hf_candidates


DEFAULT_MODEL = Path("apps/collector/src/models/fable_audio_model.json")


def _resolve_pack_file(pack: Path, stored: str, folder: str) -> Path:
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
        if str(marker.get("label", "")).strip().lower()
        in {"racket", "racket_bounce"}
    )


def build_review_pack_manifest(
    review_pack: Path,
    model_path: Path,
    output_path: Path,
    *,
    cutoff_hz: float = 7_000.0,
    onset_ratio: float = 3.0,
    match_ms: float = 140.0,
) -> dict[str, object]:
    """Generate one row per HF candidate with reviewed and frozen-Fable data."""
    pack = review_pack.resolve()
    payload = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not records:
        raise ValueError("Review-pack manifest contains no records")

    runtime = FableHgbRuntime.from_path(model_path)
    gate_config = HFGateConfig(cutoff_hz=cutoff_hz, onset_ratio=onset_ratio)
    rows: list[dict[str, object]] = []
    device_contract: dict[str, dict[str, object]] = {}

    for record in records:
        session_id = str(record["session_id"])
        audio_path = _resolve_pack_file(pack, str(record["raw_wav"]), "raw")
        labels_path = _resolve_pack_file(pack, str(record["labels_json"]), "review_labels")
        session_path = pack / "raw" / f"{session_id}.json"
        if not session_path.exists():
            source_session = pack.parent.parent / str(record["device_folder"]) / session_id / "session.json"
            session_path = source_session
        session = json.loads(session_path.read_text(encoding="utf-8"))
        capture = session.get("capture", {})
        device = session.get("device", {})
        device_id = str(device.get("installId") or record["device_folder"])
        device_contract[device_id] = {
            "alias": device.get("alias") or record.get("device_alias"),
            "manufacturer": device.get("manufacturer"),
            "model": device.get("model"),
            "platform": device.get("platform"),
            "os_version": device.get("osVersion"),
        }

        pcm, _ = load_mono_audio(audio_path, TARGET_SAMPLE_RATE)
        reviewed_ms = _reviewed_times(labels_path)
        candidates = detect_hf_candidates(pcm, TARGET_SAMPLE_RATE, gate_config)
        explicit_negative = len(reviewed_ms) == 0 and int(record.get("expected_count", 0)) == 0
        assignments = assign_candidates(
            [float(candidate["onset_ms"]) for candidate in candidates],
            [ReviewedEvent(timestamp_ms, "racket_bounce") for timestamp_ms in reviewed_ms],
            match_ms=match_ms,
            ambiguity_ms=300.0,
            review_complete=True,
            explicit_negative_session=explicit_negative,
        )

        for candidate, assignment in zip(candidates, assignments, strict=True):
            clip = extract_fable_clip(pcm, int(candidate["onset_sample"]))
            fable_features, fable_prediction = extract_and_predict_fable(runtime, clip)
            probabilities = dict(fable_prediction["probabilities"])
            row: dict[str, object] = {
                "config_id": f"hf{int(cutoff_hz)}_r{onset_ratio:g}",
                "cutoff_hz": cutoff_hz,
                "onset_ratio": onset_ratio,
                "session_id": session_id,
                "source": "reviewed:20260717_synchronized_phone_pack",
                "device_id": device_id,
                "device_alias": device.get("alias") or record.get("device_alias"),
                "device_manufacturer": device.get("manufacturer"),
                "device_model": device.get("model"),
                "device_platform": device.get("platform"),
                "scenario_id": record["scenario_id"],
                "audio_path": str(audio_path.resolve()),
                "input_sample_rate": capture.get("inputSampleRate"),
                "sample_rate": capture.get("outputSampleRate", TARGET_SAMPLE_RATE),
                "input_route": capture.get("inputRoute"),
                "route_class": "builtin" if "builtin" in str(capture.get("inputRoute", "")).lower() or "type_15" in str(capture.get("inputRoute", "")).lower() else "unknown",
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
                "label": assignment.label,
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
            for label, probability in probabilities.items():
                row[f"fable_prob_{label}"] = float(probability)
            rows.append(row)

    frame = pd.DataFrame(rows).sort_values(
        ["device_id", "session_id", "onset_ms"], ignore_index=True
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    report: dict[str, object] = {
        "review_pack": str(pack),
        "output": str(output_path.resolve()),
        "model": str(model_path.resolve()),
        "gate": {"cutoff_hz": cutoff_hz, "onset_ratio": onset_ratio},
        "rows": int(len(frame)),
        "devices": device_contract,
        "dispositions": {
            str(key): int(value) for key, value in frame["disposition"].value_counts().items()
        },
        "labels": {
            str(key): int(value) for key, value in frame["label"].value_counts().items()
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
    parser.add_argument("--cutoff-hz", type=float, default=7_000.0)
    parser.add_argument("--onset-ratio", type=float, default=3.0)
    parser.add_argument("--match-ms", type=float, default=140.0)
    args = parser.parse_args()
    report = build_review_pack_manifest(
        args.review_pack,
        args.model_json,
        args.output,
        cutoff_hz=args.cutoff_hz,
        onset_ratio=args.onset_ratio,
        match_ms=args.match_ms,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
