"""Build HF candidates from the verified edge-racket physical-event manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .audio_io import load_mono_audio
from .build_reviewed_round_manifest import _route_class
from .candidate_labels import ReviewedEvent, assign_candidates
from .hf_gate import HFGateConfig, TARGET_SAMPLE_RATE, detect_hf_candidates


EVENT_MANIFESTS = (
    "T01_three_device_events.json",
    "T02_table_negative_events.json",
    "T03_diagnostic_edge_events.json",
    "T04_diagnostic_table_events.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_source_wav(raw_root: Path, stored_path: str) -> Path:
    """Resolve a corrected manifest's portable ``data/...`` WAV reference."""
    stored = Path(stored_path)
    if stored.is_absolute() and stored.is_file():
        return stored.resolve()
    candidates = [raw_root / stored]
    if stored.parts and stored.parts[0].lower() == "data":
        candidates.append(raw_root.joinpath(*stored.parts[1:]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Cannot resolve source WAV {stored_path!r} below {raw_root}"
    )


def _load_events(
    event_manifest_path: Path,
    raw_root: Path,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    payload = json.loads(event_manifest_path.read_text(encoding="utf-8"))
    events_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in payload["physical_event_groups"]:
        label = str(group["internal_class_label"])
        split = str(group["dataset_bucket"])
        for view in group["device_views"]:
            audio_path = resolve_source_wav(raw_root, str(view["source_wav"]))
            events_by_session[str(view["session_id"])].append(
                {
                    "timestamp_ms": float(view["peak_ms"]),
                    "label": label,
                    "physical_event_group_id": str(
                        group["physical_event_group_id"]
                    ),
                    "subblock_id": str(
                        group.get("subblock_id")
                        or group.get("negative_block_id")
                        or group.get("diagnostic_block_id")
                        or ""
                    ),
                    "racket_id": group.get("racket_id"),
                    "rubber_face": group.get("rubber_face"),
                    "impact_zone": group.get("impact_zone"),
                    "dataset_split": split,
                    "device_alias": str(view["canonical_device_alias"]),
                    "audio_path": audio_path,
                    "source_wav_sha256": str(view["source_wav_sha256"]),
                }
            )
    for session_events in events_by_session.values():
        session_events.sort(key=lambda event: float(event["timestamp_ms"]))
    return payload, events_by_session


def _background_db(background_rms: float) -> float:
    return 20.0 * math.log10(max(float(background_rms), 1e-12))


def build_edge_round_manifest(
    round_dir: Path,
    raw_root: Path,
    output_path: Path,
    *,
    cutoff_hz: float = 8_000.0,
    onset_ratio: float = 1.85,
    mad_multiplier: float = 2.0,
    match_ms: float = 140.0,
    ambiguity_ms: float = 300.0,
) -> dict[str, Any]:
    """Create classifier rows while keeping T03/T04 diagnostic-only."""
    review_dir = round_dir.resolve() / "subblock_review"
    raw_root = raw_root.resolve()
    gate_config = HFGateConfig(
        cutoff_hz=cutoff_hz,
        onset_ratio=onset_ratio,
        mad_multiplier=mad_multiplier,
    )
    rows: list[dict[str, Any]] = []
    session_reports: list[dict[str, Any]] = []
    source_contracts: list[dict[str, Any]] = []

    for manifest_name in EVENT_MANIFESTS:
        event_path = review_dir / manifest_name
        if not event_path.is_file():
            raise FileNotFoundError(f"Missing reviewed event manifest: {event_path}")
        payload, events_by_session = _load_events(event_path, raw_root)
        source_contracts.append(
            {
                "path": str(event_path),
                "sha256": _sha256(event_path),
                "take_id": str(payload["take_id"]),
                "training_eligibility": str(payload["training_eligibility"]),
                "manual_review_required": bool(
                    payload["manual_review_required"]
                ),
            }
        )
        if payload["manual_review_required"]:
            raise ValueError(f"{event_path} still requires manual review")

        for session_id, session_events in sorted(events_by_session.items()):
            first_event = session_events[0]
            audio_path = Path(first_event["audio_path"])
            if _sha256(audio_path) != first_event["source_wav_sha256"]:
                raise ValueError(f"WAV checksum mismatch: {audio_path}")
            session_path = audio_path.with_name("session.json")
            session = json.loads(session_path.read_text(encoding="utf-8"))
            capture = session.get("capture", {})
            device = session.get("device", {})
            device_id = str(device.get("installId") or "")
            if not device_id:
                raise ValueError(f"Session {session_id} has no installId")

            pcm, _ = load_mono_audio(audio_path, TARGET_SAMPLE_RATE)
            candidates = detect_hf_candidates(
                pcm,
                TARGET_SAMPLE_RATE,
                gate_config,
            )
            reviewed_events = [
                ReviewedEvent(
                    timestamp_ms=float(event["timestamp_ms"]),
                    label=str(event["label"]),
                )
                for event in session_events
            ]
            assignments = assign_candidates(
                [float(candidate["onset_ms"]) for candidate in candidates],
                reviewed_events,
                match_ms=match_ms,
                ambiguity_ms=ambiguity_ms,
                review_complete=False,
                explicit_negative_session=False,
            )
            matched_events = {
                assignment.matched_event_index
                for assignment in assignments
                if assignment.disposition == "positive"
                and assignment.matched_event_index is not None
            }
            labels = Counter(str(event["label"]) for event in session_events)
            dataset_split = str(first_event["dataset_split"])
            take_id = str(payload["take_id"])
            session_reports.append(
                {
                    "session_id": session_id,
                    "take_id": take_id,
                    "device_id": device_id,
                    "device_alias": first_event["device_alias"],
                    "dataset_split": dataset_split,
                    "truth_events": len(session_events),
                    "truth_racket_events": labels["racket_bounce"],
                    "truth_table_events": labels["table_bounce"],
                    "matched_events": len(matched_events),
                    "missed_events": len(session_events) - len(matched_events),
                    "candidate_count": len(candidates),
                    "duration_s": len(pcm) / TARGET_SAMPLE_RATE,
                }
            )

            for candidate, assignment in zip(
                candidates,
                assignments,
                strict=True,
            ):
                matched_event = (
                    session_events[assignment.matched_event_index]
                    if assignment.matched_event_index is not None
                    else None
                )
                label = (
                    assignment.label
                    if assignment.disposition == "positive"
                    else None
                )
                row: dict[str, Any] = {
                    "config_id": (
                        f"hf{int(cutoff_hz)}_r{onset_ratio:g}"
                        f"_m{mad_multiplier:g}"
                    ),
                    "cutoff_hz": cutoff_hz,
                    "onset_ratio": onset_ratio,
                    "mad_multiplier": mad_multiplier,
                    "round_id": str(payload["canonical_round_id"]),
                    "take_id": take_id,
                    "take_number": int(take_id.removeprefix("T")),
                    "physical_event_group_id": (
                        matched_event["physical_event_group_id"]
                        if matched_event
                        else None
                    ),
                    "subblock_id": (
                        matched_event["subblock_id"] if matched_event else None
                    ),
                    "racket_id": (
                        matched_event["racket_id"] if matched_event else None
                    ),
                    "rubber_face": (
                        matched_event["rubber_face"] if matched_event else None
                    ),
                    "impact_zone": (
                        matched_event["impact_zone"] if matched_event else None
                    ),
                    "dataset_split": dataset_split,
                    "session_id": session_id,
                    "source": "reviewed:round_CJ-20260727-01_edge",
                    "device_id": device_id,
                    "device_alias": first_event["device_alias"],
                    "device_manufacturer": device.get("manufacturer"),
                    "device_model": device.get("model"),
                    "device_platform": (
                        device.get("platform")
                        or device.get("operatingSystem")
                    ),
                    "device_os_version": device.get("osVersion"),
                    "scenario_id": (
                        "edge_racket_continuous"
                        if take_id == "T01"
                        else "table_bounce_continuous"
                        if take_id == "T02"
                        else "edge_racket_same_day_diagnostic"
                        if take_id == "T03"
                        else "table_bounce_same_day_diagnostic"
                    ),
                    "binary_label": (
                        "racket_bounce"
                        if labels["racket_bounce"]
                        else "not_racket"
                    ),
                    "audio_path": str(audio_path),
                    "labels_path": "",
                    "event_manifest_path": str(event_path.resolve()),
                    "requested_sample_rate": capture.get(
                        "requestedSampleRate"
                    ),
                    "input_sample_rate": capture.get("inputSampleRate"),
                    "sample_rate": capture.get(
                        "outputSampleRate",
                        TARGET_SAMPLE_RATE,
                    ),
                    "audio_source": (
                        capture.get("audioSource") or capture.get("source")
                    ),
                    "input_route": capture.get("inputRoute"),
                    "route_class": _route_class(capture.get("inputRoute")),
                    "review_complete": True,
                    "explicit_negative": False,
                    "expected_bounce_count": labels["racket_bounce"],
                    "candidate_index": assignment.candidate_index,
                    "onset_sample": int(candidate["onset_sample"]),
                    "onset_ms": float(candidate["onset_ms"]),
                    "hf_rms": float(candidate["hf_rms"]),
                    "background_rms": float(candidate["background_rms"]),
                    "threshold": float(candidate["threshold"]),
                    "full_band_peak": float(candidate["full_band_peak"]),
                    "disposition": assignment.disposition,
                    "label": label,
                    "label_source": (
                        "verified_physical_event"
                        if label
                        else "not_trainable"
                    ),
                    "matched_event_index": assignment.matched_event_index,
                    "reviewed_event_ms": (
                        matched_event["timestamp_ms"]
                        if matched_event
                        else None
                    ),
                    "distance_ms": assignment.distance_ms,
                    "reason": assignment.reason,
                    "background_db": _background_db(
                        float(candidate["background_rms"])
                    ),
                }
                rows.append(row)

    frame = pd.DataFrame(rows).sort_values(
        ["dataset_split", "take_number", "device_alias", "onset_ms"],
        ignore_index=True,
    )
    sessions = pd.DataFrame(session_reports).sort_values(
        ["take_id", "device_alias"],
        ignore_index=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    sessions.to_csv(output_path.with_suffix(".sessions.csv"), index=False)

    split_reports: dict[str, Any] = {}
    for split, split_sessions in sessions.groupby("dataset_split"):
        truth = int(split_sessions["truth_events"].sum())
        matched = int(split_sessions["matched_events"].sum())
        split_reports[str(split)] = {
            "truth_events": truth,
            "matched_events": matched,
            "missed_events": truth - matched,
            "gate_recall": matched / truth if truth else None,
            "candidate_count": int(split_sessions["candidate_count"].sum()),
        }
    report: dict[str, Any] = {
        "schema_version": "edge_round_candidate_manifest_v1",
        "round_dir": str(round_dir.resolve()),
        "raw_root": str(raw_root),
        "output": str(output_path.resolve()),
        "source_contracts": source_contracts,
        "gate": {
            "cutoff_hz": cutoff_hz,
            "onset_ratio": onset_ratio,
            "mad_multiplier": mad_multiplier,
            "match_ms": match_ms,
            "ambiguity_ms": ambiguity_ms,
        },
        "rows": int(len(frame)),
        "sessions": int(len(sessions)),
        "splits": dict(Counter(frame["dataset_split"].astype(str))),
        "dispositions": dict(Counter(frame["disposition"].astype(str))),
        "labels": dict(Counter(frame["label"].dropna().astype(str))),
        "gate_recall_by_split": split_reports,
        "policy": {
            "train_takes": ["T01", "T02"],
            "diagnostic_only_takes": ["T03", "T04"],
            "unmatched_candidates": "unlabeled",
            "raw_files_modified": False,
        },
    }
    output_path.with_suffix(".contract.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-dir", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoff-hz", type=float, default=8_000.0)
    parser.add_argument("--onset-ratio", type=float, default=1.85)
    parser.add_argument("--mad-multiplier", type=float, default=2.0)
    parser.add_argument("--match-ms", type=float, default=140.0)
    parser.add_argument("--ambiguity-ms", type=float, default=300.0)
    args = parser.parse_args()
    report = build_edge_round_manifest(
        args.round_dir,
        args.raw_root,
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
