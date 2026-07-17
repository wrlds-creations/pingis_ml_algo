"""Build HF-gate candidate manifests from reviewed legacy and STIGA audio."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from .audio_io import load_mono_audio
from .candidate_labels import assign_candidates
from .data_sources import (
    AudioSession,
    deduplicate_sessions,
    discover_legacy_sessions,
    discover_stiga_sessions,
)
from .hf_gate import HFGateConfig, TARGET_SAMPLE_RATE, detect_hf_candidates, prepare_hf_gate


MANIFEST_COLUMNS = (
    "config_id",
    "cutoff_hz",
    "onset_ratio",
    "session_id",
    "source",
    "device_id",
    "scenario_id",
    "audio_path",
    "input_sample_rate",
    "sample_rate",
    "input_route",
    "route_class",
    "review_complete",
    "explicit_negative",
    "expected_bounce_count",
    "candidate_index",
    "onset_sample",
    "onset_ms",
    "hf_rms",
    "background_rms",
    "threshold",
    "full_band_peak",
    "disposition",
    "label",
    "matched_event_index",
    "distance_ms",
    "reason",
)


def _parse_legacy_root(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("legacy root must use DEVICE_HINT=PATH")
    device, path = value.split("=", 1)
    if not device.strip() or not path.strip():
        raise argparse.ArgumentTypeError("legacy root must use DEVICE_HINT=PATH")
    return device.strip(), Path(path.strip())


def _config_id(cutoff_hz: float, onset_ratio: float) -> str:
    return f"hf{int(round(cutoff_hz))}_r{onset_ratio:g}"


def _metrics_template() -> dict[str, float | int]:
    return {
        "sessions": 0,
        "duration_s": 0.0,
        "reviewed_events": 0,
        "matched_events": 0,
        "candidates": 0,
        "positive_candidates": 0,
        "ambiguous_candidates": 0,
        "hard_negative_candidates": 0,
        "unlabeled_candidates": 0,
        "explicit_negative_candidates": 0,
        "expected_bounces": 0,
    }


def build_manifest(
    sessions: list[AudioSession],
    cutoffs_hz: list[float],
    onset_ratios: list[float],
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    aggregate: dict[str, dict] = defaultdict(_metrics_template)
    per_session: list[dict] = []

    for session_number, session in enumerate(sessions, start=1):
        try:
            pcm, input_sample_rate = load_mono_audio(session.audio_path, TARGET_SAMPLE_RATE)
        except Exception as exc:  # keep a large corpus auditable instead of aborting silently
            per_session.append(
                {
                    "session_id": session.session_id,
                    "audio_path": str(session.audio_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        duration_s = len(pcm) / TARGET_SAMPLE_RATE
        print(f"[{session_number}/{len(sessions)}] {session.session_id} ({duration_s:.1f}s)")
        for cutoff_hz in cutoffs_hz:
            cutoff_config = HFGateConfig(cutoff_hz=cutoff_hz)
            prepared = prepare_hf_gate(pcm, TARGET_SAMPLE_RATE, cutoff_config)
            for onset_ratio in onset_ratios:
                config = HFGateConfig(cutoff_hz=cutoff_hz, onset_ratio=onset_ratio)
                config_id = _config_id(cutoff_hz, onset_ratio)
                candidates = detect_hf_candidates(
                    pcm,
                    TARGET_SAMPLE_RATE,
                    config,
                    prepared=prepared,
                )
                assignments = assign_candidates(
                    [float(candidate["onset_ms"]) for candidate in candidates],
                    list(session.reviewed_events),
                    review_complete=session.review_complete,
                    explicit_negative_session=session.explicit_negative,
                )
                matched_events = len(
                    {
                        assignment.matched_event_index
                        for assignment in assignments
                        if assignment.disposition == "positive" and assignment.matched_event_index is not None
                    }
                )
                metric = aggregate[config_id]
                metric["sessions"] += 1
                metric["duration_s"] += duration_s
                metric["reviewed_events"] += sum(event.trainable for event in session.reviewed_events)
                metric["matched_events"] += matched_events
                metric["candidates"] += len(candidates)
                metric["positive_candidates"] += sum(a.disposition == "positive" for a in assignments)
                metric["ambiguous_candidates"] += sum(a.disposition == "ambiguous" for a in assignments)
                metric["hard_negative_candidates"] += sum(a.disposition == "hard_negative" for a in assignments)
                metric["unlabeled_candidates"] += sum(a.disposition == "unlabeled" for a in assignments)
                metric["explicit_negative_candidates"] += len(candidates) if session.explicit_negative else 0
                metric["expected_bounces"] += session.expected_bounce_count or 0

                per_session.append(
                    {
                        "config_id": config_id,
                        "session_id": session.session_id,
                        "source": session.source,
                        "device_id": session.device_id,
                        "scenario_id": session.scenario_id,
                        "duration_s": duration_s,
                        "reviewed_events": len(session.reviewed_events),
                        "matched_events": matched_events,
                        "expected_bounce_count": session.expected_bounce_count,
                        "candidate_count": len(candidates),
                        "positive_candidates": sum(a.disposition == "positive" for a in assignments),
                        "ambiguous_candidates": sum(a.disposition == "ambiguous" for a in assignments),
                        "hard_negative_candidates": sum(a.disposition == "hard_negative" for a in assignments),
                        "unlabeled_candidates": sum(a.disposition == "unlabeled" for a in assignments),
                    }
                )

                for candidate, assignment in zip(candidates, assignments, strict=True):
                    rows.append(
                        {
                            "config_id": config_id,
                            "cutoff_hz": cutoff_hz,
                            "onset_ratio": onset_ratio,
                            "session_id": session.session_id,
                            "source": session.source,
                            "device_id": session.device_id,
                            "scenario_id": session.scenario_id,
                            "audio_path": str(session.audio_path),
                            "input_sample_rate": input_sample_rate,
                            "sample_rate": TARGET_SAMPLE_RATE,
                            "input_route": session.input_route,
                            "route_class": session.route_class,
                            "review_complete": session.review_complete,
                            "explicit_negative": session.explicit_negative,
                            "expected_bounce_count": session.expected_bounce_count,
                            "candidate_index": assignment.candidate_index,
                            **candidate,
                            **{
                                key: value
                                for key, value in asdict(assignment).items()
                                if key not in {"candidate_index", "onset_ms"}
                            },
                        }
                    )

    for metric in aggregate.values():
        reviewed = int(metric["reviewed_events"])
        duration_minutes = float(metric["duration_s"]) / 60.0
        metric["gate_recall"] = float(metric["matched_events"]) / reviewed if reviewed else None
        metric["candidates_per_minute"] = float(metric["candidates"]) / duration_minutes if duration_minutes else 0.0
    report = {"aggregate": dict(aggregate), "sessions": per_session}
    return rows, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stiga-root", action="append", default=[], type=Path)
    parser.add_argument("--legacy-root", action="append", default=[], type=_parse_legacy_root)
    parser.add_argument("--cutoffs-hz", default="6000,7000,8000")
    parser.add_argument("--onset-ratios", default="3.0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-sessions", type=int, default=0)
    args = parser.parse_args()

    sessions: list[AudioSession] = []
    for root in args.stiga_root:
        sessions.extend(discover_stiga_sessions(root))
    for device_hint, root in args.legacy_root:
        sessions.extend(
            discover_legacy_sessions(root, device_hint=device_hint, source_name=f"legacy:{device_hint}")
        )
    sessions = deduplicate_sessions(sessions)
    if args.limit_sessions:
        sessions = sessions[: args.limit_sessions]
    if not sessions:
        raise SystemExit("No audio sessions discovered")

    cutoffs = [float(value) for value in args.cutoffs_hz.split(",") if value.strip()]
    onset_ratios = [float(value) for value in args.onset_ratios.split(",") if value.strip()]
    rows, report = build_manifest(sessions, cutoffs, onset_ratios)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "hf_candidate_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    report_path = args.output_dir / "hf_gate_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {manifest_path} ({len(rows)} candidates)")
    print(f"Wrote {report_path}")
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
