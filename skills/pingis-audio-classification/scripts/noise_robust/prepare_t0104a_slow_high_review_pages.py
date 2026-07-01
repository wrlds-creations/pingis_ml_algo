#!/usr/bin/env python3
"""Prepare exact-label review pages for the ambiguous T0104 slow/high runs.

This is a local review helper only. It does not train or export a model, change
app runtime behavior, install an APK, or delete phone/local data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_t0067_peak_gate_replay import PeakGateConfig, detect_peak_gate, read_wav  # noqa: E402


DEFAULT_RAW_DIR = ROOT / "data/audio/raw/t0104_bounce_audio_test_live_validation/bounce_audio_test_debug"
DEFAULT_OUT_DIR = ROOT / "data/audio/models/evaluations/t0104a_slow_high_expected_count_review"
GATE_ID = "peak_fast_balanced"
PEAK_FAST_BALANCED = PeakGateConfig("raw_abs", 3.0, 220.0, 500.0, 60.0, 0.08, 2.0, 0.0)
PREFILL_TARGET_COUNT = 30

SLOW_HIGH_SESSIONS = [
    "bounce_audio_test_session_2026-07-01T13-37-11-083Z",
    "bounce_audio_test_session_2026-07-01T13-38-19-066Z",
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

MANIFEST_FIELDNAMES = [
    "session_id",
    "started_at",
    "stopped_at",
    "saved_expected_count",
    "saved_app_count",
    "review_expected_display_count",
    "wav_duration_s",
    "peak_candidate_count",
    "draft_label_count",
    "trigger_csv",
    "review_labels_json",
    "review_url",
    "review_command",
    "issues",
]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


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


def select_draft_markers(rows: list[dict[str, Any]], target_count: int) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda item: safe_float(item.get("selection_score")), reverse=True)
    return sorted(ranked[:target_count], key=lambda item: safe_float(item.get("time_s")))


def review_command(
    *,
    session_id: str,
    raw_dir: Path,
    out_dir: Path,
    trigger_csv: Path,
    port: int,
) -> str:
    script = SCRIPT_DIR / "serve_t0053_trigger_review_ui.py"
    gate_note = (
        "Slow/high count is uncertain: phone metadata says 20, but review target is 20 vs 30. "
        "Gray lines are peak candidates. Green labels are editable draft racket contacts; "
        "delete extras, drag mistimed labels, add missing contacts, then Save labels."
    )
    return " ".join(
        [
            "python",
            f'"{script}"',
            f"--port {port}",
            f"--session-id {session_id}",
            f'--raw-dir "{raw_dir}"',
            f'--eval-dir "{out_dir}"',
            f'--out-dir "{out_dir / "review_pages"}"',
            f'--trigger-csv "{trigger_csv}"',
            "--manual-only",
            f"--expected-count {PREFILL_TARGET_COUNT}",
            "--reported-app-count -1",
            f'--gate-note "{gate_note}"',
        ]
    )


def make_review_payload(
    *,
    session_id: str,
    raw_dir: Path,
    out_dir: Path,
    selected: list[dict[str, Any]],
    saved_expected_count: int | None,
    saved_app_count: int | None,
    force: bool,
) -> Path:
    labels_path = out_dir / "review_pages" / f"{session_id}_review_labels.json"
    if labels_path.exists() and not force:
        payload = json.loads(labels_path.read_text(encoding="utf-8"))
        payload["expected_count"] = PREFILL_TARGET_COUNT
        payload["reported_app_count"] = saved_app_count if saved_app_count is not None else -1
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
        "source_wav": str(raw_dir / f"{session_id}.wav"),
        "source_json": str(raw_dir / f"{session_id}.json"),
        "expected_count": PREFILL_TARGET_COUNT,
        "reported_app_count": saved_app_count if saved_app_count is not None else -1,
        "saved_expected_count": saved_expected_count,
        "manual_only": True,
        "trigger_labels": {},
        "manual_markers": markers,
        "saved_at": now,
        "review_note": (
            "T0104A auto-prefill from peak_fast_balanced. This clip's saved app metadata says "
            f"{saved_expected_count} contacts, but the real count is uncertain. Verify the green "
            "markers, delete false ones, drag mistimed ones, add missing racket contacts, and save."
        ),
    }
    write_json(labels_path, payload)
    return labels_path


def review_saved_values(payload: dict[str, Any]) -> tuple[int | None, int | None]:
    review = payload.get("review") if isinstance(payload.get("review"), dict) else {}
    expected = review.get("expected_racket_contacts")
    app_count = review.get("app_count_at_stop")
    return (
        int(expected) if isinstance(expected, int) else None,
        int(app_count) if isinstance(app_count, int) else None,
    )


def prepare(args: argparse.Namespace) -> list[dict[str, Any]]:
    raw_dir = args.raw_dir.resolve()
    out_dir = args.out_dir.resolve()
    trigger_dir = out_dir / "review_pages" / "trigger_csv"
    rows: list[dict[str, Any]] = []

    for index, session_id in enumerate(args.session_id):
        port = args.port_start + index
        wav_path = raw_dir / f"{session_id}.wav"
        json_path = raw_dir / f"{session_id}.json"
        trigger_csv = trigger_dir / f"{session_id}_{GATE_ID}_triggers.csv"
        labels_path = out_dir / "review_pages" / f"{session_id}_review_labels.json"
        issues: list[str] = []
        peaks: list[dict[str, Any]] = []
        selected: list[dict[str, Any]] = []
        payload: dict[str, Any] = {}
        saved_expected: int | None = None
        saved_app_count: int | None = None
        duration_s = ""

        if not wav_path.exists():
            issues.append("missing_wav")
        if not json_path.exists():
            issues.append("missing_json")

        if not issues:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            saved_expected, saved_app_count = review_saved_values(payload)
            samples, sample_rate = read_wav(wav_path)
            duration_s = f"{len(samples) / float(sample_rate):.3f}"
            events = detect_peak_gate(samples, sample_rate, PEAK_FAST_BALANCED)
            peaks = normalize_peak_rows(events, session_id)
            selected = select_draft_markers(peaks, args.prefill_count)
            write_csv(trigger_csv, peaks, TRIGGER_FIELDNAMES)
            labels_path = make_review_payload(
                session_id=session_id,
                raw_dir=raw_dir,
                out_dir=out_dir,
                selected=selected,
                saved_expected_count=saved_expected,
                saved_app_count=saved_app_count,
                force=args.force,
            )

        command = ""
        url = ""
        if not issues:
            command = review_command(
                session_id=session_id,
                raw_dir=raw_dir,
                out_dir=out_dir,
                trigger_csv=trigger_csv,
                port=port,
            )
            url = f"http://127.0.0.1:{port}/"

        rows.append(
            {
                "session_id": session_id,
                "started_at": payload.get("started_at", ""),
                "stopped_at": payload.get("stopped_at", ""),
                "saved_expected_count": saved_expected if saved_expected is not None else "",
                "saved_app_count": saved_app_count if saved_app_count is not None else "",
                "review_expected_display_count": PREFILL_TARGET_COUNT,
                "wav_duration_s": duration_s,
                "peak_candidate_count": len(peaks),
                "draft_label_count": len(selected),
                "trigger_csv": rel(trigger_csv) if trigger_csv.exists() else "",
                "review_labels_json": rel(labels_path) if labels_path.exists() else "",
                "review_url": url,
                "review_command": command,
                "issues": ";".join(issues),
            }
        )

    write_csv(out_dir / "t0104a_review_page_manifest.csv", rows, MANIFEST_FIELDNAMES)
    report = [
        "# T0104A Slow/High Expected Count Review Pages",
        "",
        "Local review pages for the two ambiguous T0104 slow/high Bounce audio test runs.",
        "",
        "The app metadata saved these as expected `20`, but Love is unsure whether they were `20` or `30`.",
        "Each page pre-fills up to `30` green draft labels from waveform peak candidates so the true count can be corrected by ear.",
        "",
        "| URL | Session | Saved expected | App count | Peak candidates | Draft labels |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            "| {review_url} | `{session_id}` | {saved_expected_count} | {saved_app_count} | "
            "{peak_candidate_count} | {draft_label_count} |".format(**row)
        )
    report.extend(
        [
            "",
            "Instructions: verify the green labels, delete false labels, drag labels onto the exact bounce peak, add missing contacts, and click Save labels.",
        ]
    )
    (out_dir / "t0104a_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--port-start", type=int, default=8781)
    parser.add_argument("--prefill-count", type=int, default=PREFILL_TARGET_COUNT)
    parser.add_argument("--force", action="store_true", help="Overwrite existing prefilled review label JSON files.")
    parser.add_argument("--session-id", action="append", default=[], help="Session ID to prepare. Defaults to both T0104 slow/high sessions.")
    args = parser.parse_args()
    if not args.session_id:
        args.session_id = SLOW_HIGH_SESSIONS
    rows = prepare(args)
    for row in rows:
        print(
            f"{row['review_url']} {row['session_id']} "
            f"peaks={row['peak_candidate_count']} draft={row['draft_label_count']} issues={row['issues']}"
        )


if __name__ == "__main__":
    main()
