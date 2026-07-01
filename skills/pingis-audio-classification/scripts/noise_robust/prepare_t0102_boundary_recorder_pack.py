#!/usr/bin/env python3
"""Prepare/pull/verify the T0102 boundary Fable recorder pack.

This is a data-handoff helper only. It does not train a model, export JSON,
change app behavior, or delete device/local data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import wave
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_t0067_peak_gate_replay import (  # noqa: E402
    PeakGateConfig,
    detect_peak_gate,
    read_wav,
)

DEFAULT_LOCAL_ROOT = ROOT / "data/audio/raw/t0102_boundary_recorder_pack/fable_training_audio"
DEFAULT_OUT_DIR = ROOT / "data/audio/models/evaluations/t0102_boundary_recorder_pack_pull_review_prep"
DEFAULT_DEVICE_ROOT = "/sdcard/Download/pingis_sessions/fable_training_audio"
GATE_ID = "peak_fast_balanced"
PEAK_FAST_BALANCED = PeakGateConfig("raw_abs", 3.0, 220.0, 500.0, 60.0, 0.08, 2.0, 0.0)

TARGET_SCENARIOS: dict[str, dict[str, Any]] = {
    "far_soft_racket_bounce_background": {
        "target_count": 3,
        "polarity": "positive",
        "review": "exact_timestamp_review",
    },
    "soft_high_racket_bounce_background": {
        "target_count": 3,
        "polarity": "positive",
        "review": "exact_timestamp_review",
    },
    "background_sound_only_no_bounce": {
        "target_count": 2,
        "polarity": "negative",
        "review": "interval_hard_negative",
    },
    "talking_counting_background_no_bounce": {
        "target_count": 2,
        "polarity": "negative",
        "review": "interval_hard_negative",
    },
    "racket_handling_background_no_bounce": {
        "target_count": 3,
        "polarity": "negative",
        "review": "interval_hard_negative",
    },
    "catch_after_sound_no_racket": {
        "target_count": 2,
        "polarity": "negative",
        "review": "interval_hard_negative",
    },
    "ambiguous_ball_like_impact_near_phone_no_racket": {
        "target_count": 3,
        "polarity": "negative",
        "review": "interval_hard_negative",
    },
}

TARGET_COLLECTION_GOALS = {
    "t0102_boundary_positive_recovery",
    "t0102_boundary_hard_negative",
}

TARGET_BOUNDARY_BUCKETS = {
    "far_soft_background_racket_positive",
    "soft_high_background_racket_positive",
    "background_only_negative",
    "speech_background_negative",
    "racket_handling_background_negative",
    "catch_after_sound_negative",
    "ambiguous_ball_like_near_phone_negative",
}

FIELDNAMES = [
    "row_role",
    "session_id",
    "json_path",
    "wav_path",
    "wav_exists",
    "wav_duration_s",
    "wav_sample_rate_hz",
    "wav_channels",
    "wav_sample_width_bytes",
    "started_at",
    "created_at",
    "duration_ms",
    "scenario_id",
    "scenario_title",
    "polarity",
    "boundary_bucket",
    "collection_goal",
    "training_role_hint",
    "expected_racket_contacts",
    "expected_count_unclear",
    "recommended_review",
    "issues",
]

TRIGGER_FIELDNAMES = [
    "session_id",
    "trigger_index",
    "event_index",
    "candidate_index",
    "gate_id",
    "onset_ms",
    "estimated_wav_ms",
    "time_s",
    "model_label",
    "model_confidence",
    "reject_reason",
    "native_rms",
    "frame_rms",
    "bg_rms",
    "peak_value",
    "peak_ratio",
    "peak_z",
    "prob_racket_bounce",
    "prob_noise",
    "selection_score",
]

REVIEW_MANIFEST_FIELDNAMES = [
    "session_id",
    "scenario_id",
    "scenario_title",
    "expected_racket_contacts",
    "wav_duration_s",
    "peak_candidate_count",
    "draft_label_count",
    "trigger_csv",
    "review_labels_json",
    "review_command",
    "review_url",
    "issues",
]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        output = float(value)
    except (TypeError, ValueError):
        return default
    return output if math.isfinite(output) else default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def rounded(value: Any, digits: int = 6) -> float:
    return round(safe_float(value), digits)


def peak_score(row: dict[str, Any]) -> float:
    peak_value = safe_float(row.get("peak_value"), 0.0)
    ratio = safe_float(row.get("ratio"), 0.0)
    z = safe_float(row.get("z"), 0.0)
    return peak_value * (1.0 + 0.10 * math.log1p(max(0.0, ratio))) + 0.002 * max(0.0, z)


def normalize_peak_rows(events: list[dict[str, Any]], session_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, event in enumerate(sorted(events, key=lambda item: safe_float(item.get("time_ms"))), start=1):
        time_ms = safe_float(event.get("time_ms"))
        rows.append(
            {
                "session_id": session_id,
                "trigger_index": index,
                "event_index": index,
                "candidate_index": index,
                "gate_id": GATE_ID,
                "onset_ms": round(time_ms, 3),
                "estimated_wav_ms": round(time_ms, 3),
                "time_s": round(time_ms / 1000.0, 6),
                "model_label": "peak_candidate",
                "model_confidence": "",
                "reject_reason": "",
                "native_rms": rounded(event.get("peak_value"), 8),
                "frame_rms": rounded(event.get("peak_value"), 8),
                "bg_rms": rounded(event.get("local_bg"), 8),
                "peak_value": rounded(event.get("peak_value"), 8),
                "peak_ratio": rounded(event.get("ratio"), 3),
                "peak_z": rounded(event.get("z"), 3),
                "prob_racket_bounce": "",
                "prob_noise": "",
                "selection_score": round(peak_score(event), 8),
            }
        )
    return rows


def select_draft_markers(rows: list[dict[str, Any]], expected_count: int) -> list[dict[str, Any]]:
    if expected_count <= 0:
        return []
    ranked = sorted(rows, key=lambda item: safe_float(item.get("selection_score")), reverse=True)
    return sorted(ranked[:expected_count], key=lambda item: safe_float(item.get("time_s")))


def make_review_payload(
    *,
    row: dict[str, Any],
    local_root: Path,
    out_dir: Path,
    selected: list[dict[str, Any]],
    force_labels: bool,
) -> Path:
    session_id = str(row["session_id"])
    labels_path = out_dir / "review_pages" / f"{session_id}_review_labels.json"
    expected_count = safe_int(row.get("expected_racket_contacts"))
    if labels_path.exists() and not force_labels:
        payload = json.loads(labels_path.read_text(encoding="utf-8"))
        payload["expected_count"] = expected_count
        payload.setdefault("review_note", "")
        write_json(labels_path, payload)
        return labels_path

    now = datetime.now(timezone.utc).isoformat()
    markers: list[dict[str, Any]] = []
    for marker_index, candidate in enumerate(selected, start=1):
        markers.append(
            {
                "id": f"auto_peak_{session_id}_{marker_index:03d}",
                "time_s": rounded(candidate.get("time_s"), 6),
                "label": "racket",
                "note": "auto_peak_draft_review_required",
                "created_at": now,
                "source": "auto_waveform_peak_prefill",
                "gate_id": GATE_ID,
                "peak_rank": marker_index,
                "source_candidate_index": candidate.get("candidate_index", ""),
                "peak_value": rounded(candidate.get("peak_value"), 8),
                "selection_score": rounded(candidate.get("selection_score"), 8),
            }
        )

    payload = {
        "session_id": session_id,
        "source_wav": str(local_root / f"{session_id}.wav"),
        "source_json": str(local_root / f"{session_id}.json"),
        "expected_count": expected_count,
        "reported_app_count": -1,
        "manual_only": True,
        "trigger_labels": {},
        "manual_markers": markers,
        "saved_at": now,
        "review_note": (
            "T0102 auto-prefill from peak_fast_balanced. Verify these green markers, "
            "delete false ones, drag mistimed ones, and add missing racket contacts."
        ),
    }
    write_json(labels_path, payload)
    return labels_path


def review_command(
    *,
    session_id: str,
    local_root: Path,
    out_dir: Path,
    trigger_csv: Path,
    expected_count: int,
    port: int,
) -> str:
    script = SCRIPT_DIR / "serve_t0053_trigger_review_ui.py"
    gate_note = (
        "Peak candidates are gray read-only lines. Green labels are draft racket contacts; "
        "review/delete/drag/add, then Save labels."
    )
    parts = [
        "python",
        f'"{script}"',
        f"--port {port}",
        f"--session-id {session_id}",
        f'--raw-dir "{local_root}"',
        f'--eval-dir "{out_dir}"',
        f'--out-dir "{out_dir / "review_pages"}"',
        f'--trigger-csv "{trigger_csv}"',
        "--manual-only",
        f"--expected-count {expected_count}",
        "--reported-app-count -1",
        f'--gate-note "{gate_note}"',
    ]
    return " ".join(parts)


def read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload, None
        return None, "json_root_not_object"
    except Exception as exc:  # noqa: BLE001 - report data issue, do not crash scan
        return None, f"json_read_error:{type(exc).__name__}:{exc}"


def wav_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "duration_s": "",
            "sample_rate_hz": "",
            "channels": "",
            "sample_width_bytes": "",
            "issues": ["missing_wav"],
        }
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            sample_width = wav.getsampwidth()
            frame_count = wav.getnframes()
        issues: list[str] = []
        if channels != 1:
            issues.append(f"unexpected_channels:{channels}")
        if sample_rate != 22050:
            issues.append(f"unexpected_sample_rate:{sample_rate}")
        if sample_width != 2:
            issues.append(f"unexpected_sample_width:{sample_width}")
        return {
            "exists": True,
            "duration_s": round(frame_count / sample_rate, 3) if sample_rate else "",
            "sample_rate_hz": sample_rate,
            "channels": channels,
            "sample_width_bytes": sample_width,
            "issues": issues,
        }
    except Exception as exc:  # noqa: BLE001 - report data issue, do not crash scan
        return {
            "exists": path.exists(),
            "duration_s": "",
            "sample_rate_hz": "",
            "channels": "",
            "sample_width_bytes": "",
            "issues": [f"wav_read_error:{type(exc).__name__}:{exc}"],
        }


def local_wav_path(json_path: Path, payload: dict[str, Any]) -> Path:
    audio = payload.get("audio") if isinstance(payload.get("audio"), dict) else {}
    wav_filename = audio.get("wav_filename") or f"{payload.get('session_id', json_path.stem)}.wav"
    return json_path.with_name(str(wav_filename))


def scenario_fields(payload: dict[str, Any]) -> dict[str, Any]:
    scenario = payload.get("scenario") if isinstance(payload.get("scenario"), dict) else {}
    return {
        "scenario_id": scenario.get("id", ""),
        "scenario_title": scenario.get("title", ""),
        "polarity": scenario.get("polarity", ""),
        "boundary_bucket": scenario.get("boundary_bucket", ""),
        "collection_goal": scenario.get("collection_goal", ""),
        "scenario_training_role_hint": scenario.get("training_role_hint", ""),
    }


def has_t0102_metadata(fields: dict[str, Any]) -> bool:
    return (
        fields["collection_goal"] in TARGET_COLLECTION_GOALS
        or fields["boundary_bucket"] in TARGET_BOUNDARY_BUCKETS
    )


def row_role(fields: dict[str, Any]) -> str:
    if has_t0102_metadata(fields):
        return "target_t0102"
    if fields["scenario_id"] in TARGET_SCENARIOS:
        return "boundary_named_legacy"
    return "legacy_or_other"


def review_recommendation(fields: dict[str, Any], target: bool) -> str:
    if target and fields["scenario_id"] in TARGET_SCENARIOS:
        return str(TARGET_SCENARIOS[fields["scenario_id"]]["review"])
    if fields["polarity"] == "positive":
        return "exact_timestamp_review"
    if fields["polarity"] == "negative":
        return "interval_hard_negative"
    return "diagnostic_review"


def scan_recordings(local_root: Path, since_iso: str | None = None) -> list[dict[str, Any]]:
    since_dt = parse_iso(since_iso)
    rows: list[dict[str, Any]] = []
    for json_path in sorted(local_root.rglob("fable_training_audio_*.json")):
        payload, json_issue = read_json(json_path)
        if payload is None:
            rows.append({
                "row_role": "invalid_json",
                "session_id": json_path.stem,
                "json_path": rel(json_path),
                "issues": json_issue or "invalid_json",
            })
            continue

        started_dt = parse_iso(payload.get("started_at"))
        created_dt = parse_iso(payload.get("created_at"))
        event_dt = started_dt or created_dt
        if since_dt and event_dt and event_dt < since_dt:
            continue

        fields = scenario_fields(payload)
        role = row_role(fields)
        target = role == "target_t0102"
        wav_path = local_wav_path(json_path, payload)
        wav = wav_info(wav_path)
        expected = payload.get("expected_racket_contacts")
        unclear = bool(payload.get("expected_count_unclear", False))
        training_role_hint = payload.get("training_role_hint") or fields["scenario_training_role_hint"]

        issues = list(wav["issues"])
        if target:
            if not fields["boundary_bucket"]:
                issues.append("target_missing_boundary_bucket")
            if not fields["collection_goal"]:
                issues.append("target_missing_collection_goal")
            if not training_role_hint:
                issues.append("target_missing_training_role_hint")
        if fields["polarity"] == "positive" and not unclear:
            if not isinstance(expected, int) or expected <= 0:
                issues.append("positive_expected_count_not_positive")
        if fields["polarity"] == "negative" and not unclear:
            if expected != 0:
                issues.append("negative_expected_count_not_zero")
        if fields["scenario_id"] in TARGET_SCENARIOS:
            expected_polarity = TARGET_SCENARIOS[fields["scenario_id"]]["polarity"]
            if fields["polarity"] != expected_polarity:
                issues.append(f"target_polarity_mismatch:{expected_polarity}")

        if role == "boundary_named_legacy":
            issues.append("legacy_boundary_name_missing_new_t0102_metadata")

        row = {
            "row_role": role,
            "session_id": payload.get("session_id", json_path.stem),
            "json_path": rel(json_path),
            "wav_path": rel(wav_path),
            "wav_exists": wav["exists"],
            "wav_duration_s": wav["duration_s"],
            "wav_sample_rate_hz": wav["sample_rate_hz"],
            "wav_channels": wav["channels"],
            "wav_sample_width_bytes": wav["sample_width_bytes"],
            "started_at": payload.get("started_at", ""),
            "created_at": payload.get("created_at", ""),
            "duration_ms": payload.get("duration_ms", ""),
            "scenario_id": fields["scenario_id"],
            "scenario_title": fields["scenario_title"],
            "polarity": fields["polarity"],
            "boundary_bucket": fields["boundary_bucket"],
            "collection_goal": fields["collection_goal"],
            "training_role_hint": training_role_hint or "",
            "expected_racket_contacts": "" if expected is None else expected,
            "expected_count_unclear": unclear,
            "recommended_review": review_recommendation(fields, target),
            "issues": ";".join(issues),
        }
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]], local_root: Path, since_iso: str | None) -> dict[str, Any]:
    target_rows = [row for row in rows if row.get("row_role") == "target_t0102"]
    issue_rows = [row for row in rows if row.get("issues")]
    coverage: dict[str, dict[str, Any]] = {}
    by_scenario = Counter(str(row.get("scenario_id", "")) for row in target_rows)
    for scenario_id, spec in TARGET_SCENARIOS.items():
        actual = by_scenario.get(scenario_id, 0)
        target = int(spec["target_count"])
        coverage[scenario_id] = {
            "target_count": target,
            "actual_count": actual,
            "remaining": max(0, target - actual),
            "polarity": spec["polarity"],
            "recommended_review": spec["review"],
        }

    by_role = Counter(str(row.get("row_role", "")) for row in rows)
    by_polarity = Counter(str(row.get("polarity", "")) for row in target_rows)
    by_collection_goal = Counter(str(row.get("collection_goal", "")) for row in target_rows)
    by_boundary_bucket = Counter(str(row.get("boundary_bucket", "")) for row in target_rows)
    positive_needing_review = [
        row["session_id"]
        for row in target_rows
        if row.get("polarity") == "positive" and row.get("expected_count_unclear") in (False, "False", "false")
    ]
    negative_candidates = [
        row["session_id"]
        for row in target_rows
        if row.get("polarity") == "negative" and not row.get("issues")
    ]

    return {
        "local_root": rel(local_root),
        "since_iso": since_iso,
        "total_json_rows": len(rows),
        "target_t0102_rows": len(target_rows),
        "boundary_named_legacy_rows": by_role.get("boundary_named_legacy", 0),
        "legacy_or_other_rows": by_role.get("legacy_or_other", 0),
        "invalid_json_rows": by_role.get("invalid_json", 0),
        "rows_with_issues": len(issue_rows),
        "target_by_polarity": dict(sorted(by_polarity.items())),
        "target_by_collection_goal": dict(sorted(by_collection_goal.items())),
        "target_by_boundary_bucket": dict(sorted(by_boundary_bucket.items())),
        "coverage": coverage,
        "positive_sessions_needing_exact_review": positive_needing_review,
        "clean_negative_interval_candidates": negative_candidates,
        "issue_sessions": [
            {
                "session_id": row.get("session_id", ""),
                "row_role": row.get("row_role", ""),
                "scenario_id": row.get("scenario_id", ""),
                "issues": row.get("issues", ""),
            }
            for row in issue_rows[:50]
        ],
    }


def resolve_path(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def prepare_review_pages(
    *,
    target_rows: list[dict[str, Any]],
    out_dir: Path,
    port_start: int,
    force_labels: bool,
) -> list[dict[str, Any]]:
    review_rows: list[dict[str, Any]] = []
    trigger_dir = out_dir / "review_pages" / "trigger_csv"
    positives = [
        row
        for row in target_rows
        if row.get("polarity") == "positive"
        and row.get("expected_count_unclear") in (False, "False", "false")
    ]
    for review_index, row in enumerate(positives):
        session_id = str(row.get("session_id", ""))
        expected_count = safe_int(row.get("expected_racket_contacts"))
        wav_path = resolve_path(row.get("wav_path"))
        raw_dir = wav_path.parent
        trigger_csv = trigger_dir / f"{session_id}_{GATE_ID}_triggers.csv"
        labels_path = out_dir / "review_pages" / f"{session_id}_review_labels.json"
        issues: list[str] = []
        peaks: list[dict[str, Any]] = []
        selected: list[dict[str, Any]] = []

        if expected_count <= 0:
            issues.append("missing_positive_expected_count")
        if not wav_path.exists():
            issues.append("missing_wav_for_review_page")

        if not issues:
            try:
                samples, sample_rate = read_wav(wav_path)
                peaks = normalize_peak_rows(detect_peak_gate(samples, sample_rate, PEAK_FAST_BALANCED), session_id)
                selected = select_draft_markers(peaks, expected_count)
                write_csv(trigger_csv, peaks, TRIGGER_FIELDNAMES)
                labels_path = make_review_payload(
                    row=row,
                    local_root=raw_dir,
                    out_dir=out_dir,
                    selected=selected,
                    force_labels=force_labels,
                )
            except Exception as exc:  # noqa: BLE001 - report prep issue without hiding the scan
                issues.append(f"review_page_prep_error:{type(exc).__name__}:{exc}")

        port = port_start + review_index
        command = ""
        url = ""
        if not issues:
            command = review_command(
                session_id=session_id,
                local_root=raw_dir,
                out_dir=out_dir,
                trigger_csv=trigger_csv,
                expected_count=expected_count,
                port=port,
            )
            url = f"http://127.0.0.1:{port}/"

        review_rows.append(
            {
                "session_id": session_id,
                "scenario_id": row.get("scenario_id", ""),
                "scenario_title": row.get("scenario_title", ""),
                "expected_racket_contacts": expected_count,
                "wav_duration_s": row.get("wav_duration_s", ""),
                "peak_candidate_count": len(peaks),
                "draft_label_count": len(selected),
                "trigger_csv": rel(trigger_csv) if trigger_csv.exists() else "",
                "review_labels_json": rel(labels_path) if labels_path.exists() else "",
                "review_command": command,
                "review_url": url,
                "issues": ";".join(issues),
            }
        )

    write_csv(out_dir / "t0102_review_page_manifest.csv", review_rows, REVIEW_MANIFEST_FIELDNAMES)
    return review_rows


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# T0102 Boundary Recorder Pack Pull/Review Prep",
        "",
        "This report verifies recorder WAV/JSON pairs and separates new T0102 boundary recordings from older recorder files.",
        "",
        "## Scan Summary",
        "",
        f"- Local root: `{summary['local_root']}`",
        f"- Since filter: `{summary['since_iso'] or 'none'}`",
        f"- JSON rows scanned: `{summary['total_json_rows']}`",
        f"- T0102 target rows: `{summary['target_t0102_rows']}`",
        f"- Boundary-named legacy rows: `{summary['boundary_named_legacy_rows']}`",
        f"- Legacy/other rows: `{summary['legacy_or_other_rows']}`",
        f"- Invalid JSON rows: `{summary['invalid_json_rows']}`",
        f"- Rows with issues: `{summary['rows_with_issues']}`",
        "",
    ]
    if summary["target_t0102_rows"] == 0:
        lines += [
            "No T0102 target rows were found. That is expected before Love records the boundary pack with the updated app.",
            "",
        ]

    lines += [
        "## Requested Coverage",
        "",
        "| Scenario | Have | Target | Remaining | Role |",
        "|---|---:|---:|---:|---|",
    ]
    for scenario_id, row in summary["coverage"].items():
        lines.append(
            f"| `{scenario_id}` | {row['actual_count']} | {row['target_count']} | "
            f"{row['remaining']} | {row['recommended_review']} |"
        )

    lines += [
        "",
        "## Next Review Work",
        "",
        f"- Positive clips needing exact timestamp review: `{len(summary['positive_sessions_needing_exact_review'])}`",
        f"- Clean negative interval candidates: `{len(summary['clean_negative_interval_candidates'])}`",
        "",
    ]
    if summary["positive_sessions_needing_exact_review"]:
        lines.append("Positive session IDs:")
        for session_id in summary["positive_sessions_needing_exact_review"]:
            lines.append(f"- `{session_id}`")
        lines.append("")

    if "review_pages_prepared" in summary:
        lines += [
            "## Exact Review Page Prep",
            "",
            f"- Review-page prep enabled: `{summary.get('review_pages_enabled', False)}`",
            f"- Review pages attempted: `{summary.get('review_pages_attempted', 0)}`",
            f"- Review pages prepared: `{summary.get('review_pages_prepared', 0)}`",
            f"- Review manifest: `{summary.get('review_page_manifest', '')}`",
            "",
        ]
        commands = summary.get("review_page_commands") or []
        if commands:
            lines.append("Launch commands:")
            lines.append("")
            for command_row in commands:
                session_id = command_row.get("session_id", "")
                issues = command_row.get("issues", "")
                if issues:
                    lines.append(f"- `{session_id}` skipped: `{issues}`")
                    continue
                lines.extend(
                    [
                        f"`{session_id}` -> {command_row.get('review_url', '')}",
                        "",
                        "```powershell",
                        command_row.get("review_command", ""),
                        "```",
                        "",
                    ]
                )

    if summary["issue_sessions"]:
        lines += [
            "## Issues",
            "",
            "| Session | Role | Scenario | Issues |",
            "|---|---|---|---|",
        ]
        for row in summary["issue_sessions"]:
            lines.append(
                f"| `{row['session_id']}` | `{row['row_role']}` | "
                f"`{row['scenario_id']}` | `{row['issues']}` |"
            )
        lines.append("")

    lines += [
        "## Gate",
        "",
        "Do not train/export from this report alone. Positive clips need exact timestamps; negative clips are hard-negative intervals only when metadata and the user note confirm no real racket contacts.",
        "",
    ]
    return "\n".join(lines)


def maybe_pull(args: argparse.Namespace, local_root: Path) -> None:
    if not args.pull:
        return
    local_root.mkdir(parents=True, exist_ok=True)
    cmd = ["adb", "pull", f"{args.device_root}/.", str(local_root)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--pull", action="store_true", help="Pull from the connected Android device before scanning.")
    parser.add_argument("--device-root", default=DEFAULT_DEVICE_ROOT)
    parser.add_argument("--since-iso", default=None, help="Optional ISO timestamp filter for started_at/created_at.")
    parser.add_argument(
        "--prepare-review-pages",
        action="store_true",
        help="Prepare full-WAV timestamp review inputs for target positive T0102 clips.",
    )
    parser.add_argument("--review-port-start", type=int, default=8767)
    parser.add_argument(
        "--force-review-labels",
        action="store_true",
        help="Overwrite existing auto-prefilled review label JSON files.",
    )
    args = parser.parse_args()

    local_root = args.local_root if args.local_root.is_absolute() else ROOT / args.local_root
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir

    maybe_pull(args, local_root)

    rows = scan_recordings(local_root, since_iso=args.since_iso)
    target_rows = [row for row in rows if row.get("row_role") == "target_t0102"]
    summary = summarize(rows, local_root, args.since_iso)
    review_rows: list[dict[str, Any]] = []
    if args.prepare_review_pages:
        review_rows = prepare_review_pages(
            target_rows=target_rows,
            out_dir=out_dir,
            port_start=args.review_port_start,
            force_labels=args.force_review_labels,
        )
    else:
        write_csv(out_dir / "t0102_review_page_manifest.csv", [], REVIEW_MANIFEST_FIELDNAMES)
    summary.update(
        {
            "review_pages_enabled": bool(args.prepare_review_pages),
            "review_pages_attempted": len(review_rows),
            "review_pages_prepared": len([row for row in review_rows if not row.get("issues")]),
            "review_page_manifest": rel(out_dir / "t0102_review_page_manifest.csv"),
            "review_page_commands": [
                {
                    "session_id": row.get("session_id", ""),
                    "review_url": row.get("review_url", ""),
                    "review_command": row.get("review_command", ""),
                    "issues": row.get("issues", ""),
                }
                for row in review_rows
            ],
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "t0102_recorder_manifest_all.csv", rows, FIELDNAMES)
    write_csv(out_dir / "t0102_recorder_manifest_target.csv", target_rows, FIELDNAMES)
    write_json(out_dir / "t0102_summary.json", summary)
    (out_dir / "t0102_report.md").write_text(markdown_report(summary), encoding="utf-8")

    print(f"Scanned {len(rows)} recorder JSON rows")
    print(f"T0102 target rows: {len(target_rows)}")
    print(f"Rows with issues: {summary['rows_with_issues']}")
    print(f"Review pages prepared: {summary['review_pages_prepared']}")
    print(f"Wrote {rel(out_dir / 't0102_report.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
