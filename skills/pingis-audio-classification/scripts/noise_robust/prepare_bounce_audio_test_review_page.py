#!/usr/bin/env python3
"""Prepare a full-WAV Bounce audio test review page.

This helper turns one saved `bounce_audio_test_debug` JSON/WAV pair into the
review UI pattern Love has been using:

- grey lines: actual app/native candidate rows from the saved debug JSON;
- green labels: editable waveform peak-prefill draft labels for human review.

It does not train, export, install, or change app/runtime behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_t0067_peak_gate_replay import PeakGateConfig, detect_peak_gate, read_wav  # noqa: E402


DEFAULT_RAW_DIR = ROOT / "data/audio/raw"
DEFAULT_OUT_DIR = ROOT / "data/audio/models/evaluations/bounce_audio_test_review_page"
PREFILL_GATE_ID = "raw_abs_soft_abs003_prefill_only"
PREFILL_GATE = PeakGateConfig("raw_abs", 3.0, 220.0, 500.0, 60.0, 0.03, 2.0, 0.0)
DEFAULT_PREFILL_COUNT = 30

APP_TRIGGER_FIELDNAMES = [
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
    "counted",
    "decision",
    "reject_reason",
    "debug_explanation",
    "native_rms",
    "frame_rms",
    "background_rms",
    "bg_rms",
    "peak_value",
    "peak_ratio",
    "peak_z",
    "prob_racket_bounce",
    "prob_not_racket_bounce",
    "prob_noise",
    "fable_label",
    "fable_confidence",
    "fable_prob_racket_bounce",
    "fable_prob_table_bounce",
    "fable_prob_floor_bounce",
    "fable_prob_noise",
    "prev_gap_ms",
    "next_gap_ms",
    "neighbor_count_500ms",
    "age_ms",
    "native_onset_pos",
    "native_onset_time_ms",
    "received_at_ms",
    "peak_calibrated_floor",
    "peak_calibration_median",
    "peak_calibration_mad",
    "adaptive_threshold",
    "peak_envelope",
    "peak_ratio_min",
    "peak_z_min",
    "selection_score",
]

PREFILL_FIELDNAMES = [
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


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
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


def safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value in (None, ""):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def rounded(value: Any, digits: int = 6) -> float:
    return round(safe_float(value), digits)


def latest_session_id(raw_dir: Path) -> str:
    json_files = sorted(raw_dir.glob("bounce_audio_test_session_*.json"), key=lambda path: path.stat().st_mtime)
    if not json_files:
        raise FileNotFoundError(f"No bounce_audio_test_session_*.json files found in {raw_dir}")
    return json_files[-1].stem


def load_payload(raw_dir: Path, session_id: str) -> dict[str, Any]:
    json_path = raw_dir / f"{session_id}.json"
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    return json.loads(json_path.read_text(encoding="utf-8"))


def review_saved_values(payload: dict[str, Any]) -> tuple[int | None, int | None, str]:
    review = payload.get("review") if isinstance(payload.get("review"), dict) else {}
    scenario = review.get("scenario") if isinstance(review.get("scenario"), dict) else {}
    expected = safe_int(review.get("expected_racket_contacts"))
    app_count = safe_int(review.get("app_count_at_stop"))
    title = str(scenario.get("title") or scenario.get("id") or "")
    return expected, app_count, title


def probability(mapping: Any, key: str) -> float | str:
    if not isinstance(mapping, dict) or key not in mapping:
        return ""
    return rounded(mapping.get(key), 8)


def candidate_time_ms(candidate: dict[str, Any], sample_rate: int) -> float | None:
    pos = safe_int(candidate.get("native_onset_pos"))
    if pos is None and isinstance(candidate.get("native_debug"), dict):
        pos = safe_int(candidate["native_debug"].get("onset_pos"))
    if pos is not None and sample_rate > 0:
        return float(pos) / float(sample_rate) * 1000.0

    estimated = candidate.get("estimated_wav_ms")
    if estimated not in (None, ""):
        return safe_float(estimated)
    if "time_s" in candidate:
        return safe_float(candidate.get("time_s")) * 1000.0
    return None


def app_candidate_rows(payload: dict[str, Any], session_id: str, sample_rate: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return rows
    gate_config = payload.get("peak_gate_config") if isinstance(payload.get("peak_gate_config"), dict) else {}
    default_gate_id = gate_config.get("gateId") or "bounce_audio_test_app_candidate"

    for ordinal, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            continue
        native_debug = candidate.get("native_debug") if isinstance(candidate.get("native_debug"), dict) else {}
        classifier_probs = candidate.get("classifier_probabilities")
        fable_probs = candidate.get("fable_probabilities")
        time_ms = candidate_time_ms(candidate, sample_rate)
        if time_ms is None:
            continue
        index = safe_int(candidate.get("index"), ordinal) or ordinal
        counted = bool(candidate.get("counted"))
        reject_reason = candidate.get("reject_reason") or candidate.get("decision") or ""
        if counted and not reject_reason:
            reject_reason = "counted"
        prob_racket = probability(classifier_probs, "racket_bounce")
        if prob_racket == "":
            prob_racket = rounded(candidate.get("classifier_probability"), 8)
        prob_not_racket = probability(classifier_probs, "not_racket_bounce")
        if prob_not_racket == "" and prob_racket != "":
            prob_not_racket = rounded(1.0 - safe_float(prob_racket), 8)
        rows.append(
            {
                "session_id": session_id,
                "trigger_index": index,
                "event_index": index,
                "candidate_index": index,
                "gate_id": native_debug.get("gate_id") or default_gate_id,
                "onset_ms": round(time_ms, 3),
                "estimated_wav_ms": round(time_ms, 3),
                "time_s": round(time_ms / 1000.0, 6),
                "model_label": candidate.get("classifier_label") or candidate.get("decision") or "app_candidate",
                "model_confidence": prob_racket,
                "counted": "true" if counted else "false",
                "decision": candidate.get("decision") or "",
                "reject_reason": reject_reason,
                "debug_explanation": candidate.get("debug_explanation") or "",
                "native_rms": rounded(candidate.get("frame_rms") or native_debug.get("rms"), 8),
                "frame_rms": rounded(candidate.get("frame_rms") or native_debug.get("rms"), 8),
                "background_rms": rounded(native_debug.get("background_rms") or candidate.get("bg_rms"), 8),
                "bg_rms": rounded(candidate.get("bg_rms") or native_debug.get("background_rms"), 8),
                "peak_value": rounded(candidate.get("peak_value") or native_debug.get("peak_value"), 8),
                "peak_ratio": rounded(candidate.get("peak_ratio") or native_debug.get("peak_ratio"), 3),
                "peak_z": rounded(candidate.get("peak_z") or native_debug.get("peak_z"), 3),
                "prob_racket_bounce": prob_racket,
                "prob_not_racket_bounce": prob_not_racket,
                "prob_noise": probability(fable_probs, "noise"),
                "fable_label": candidate.get("fable_label") or "",
                "fable_confidence": rounded(candidate.get("fable_confidence"), 8),
                "fable_prob_racket_bounce": probability(fable_probs, "racket_bounce"),
                "fable_prob_table_bounce": probability(fable_probs, "table_bounce"),
                "fable_prob_floor_bounce": probability(fable_probs, "floor_bounce"),
                "fable_prob_noise": probability(fable_probs, "noise"),
                "prev_gap_ms": rounded(candidate.get("prev_gap_ms"), 3),
                "next_gap_ms": rounded(candidate.get("next_gap_ms"), 3),
                "neighbor_count_500ms": candidate.get("neighbor_count_500ms") or 0,
                "age_ms": rounded(candidate.get("age_ms"), 3),
                "native_onset_pos": candidate.get("native_onset_pos") or native_debug.get("onset_pos") or "",
                "native_onset_time_ms": candidate.get("native_onset_time_ms") or native_debug.get("onset_time_ms") or "",
                "received_at_ms": candidate.get("received_at_ms") or "",
                "peak_calibrated_floor": rounded(native_debug.get("peak_calibrated_floor"), 8),
                "peak_calibration_median": rounded(native_debug.get("peak_calibration_median"), 8),
                "peak_calibration_mad": rounded(native_debug.get("peak_calibration_mad"), 8),
                "adaptive_threshold": rounded(native_debug.get("adaptive_threshold"), 8),
                "peak_envelope": native_debug.get("peak_envelope") or "",
                "peak_ratio_min": rounded(native_debug.get("peak_ratio_min"), 3),
                "peak_z_min": rounded(native_debug.get("peak_z_min"), 3),
                "selection_score": "",
            }
        )
    return sorted(rows, key=lambda item: safe_float(item.get("time_s")))


def peak_score(event: dict[str, Any]) -> float:
    peak_value = safe_float(event.get("peak_value"))
    ratio = safe_float(event.get("ratio"))
    z = safe_float(event.get("z"))
    return peak_value * (1.0 + 0.10 * math.log1p(max(0.0, ratio))) + 0.002 * max(0.0, z)


def waveform_prefill_rows(events: list[dict[str, Any]], session_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, event in enumerate(sorted(events, key=lambda item: safe_float(item.get("time_ms"))), start=1):
        time_ms = safe_float(event.get("time_ms"))
        rows.append(
            {
                "session_id": session_id,
                "trigger_index": index,
                "event_index": index,
                "candidate_index": index,
                "gate_id": PREFILL_GATE_ID,
                "onset_ms": round(time_ms, 3),
                "estimated_wav_ms": round(time_ms, 3),
                "time_s": round(time_ms / 1000.0, 6),
                "model_label": "waveform_peak_prefill",
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


def make_review_payload(
    *,
    labels_path: Path,
    session_id: str,
    raw_dir: Path,
    selected: list[dict[str, Any]],
    expected_count: int,
    reported_app_count: int,
    scenario_title: str,
    force: bool,
) -> None:
    if labels_path.exists() and not force:
        payload = json.loads(labels_path.read_text(encoding="utf-8"))
        payload["expected_count"] = expected_count
        payload["reported_app_count"] = reported_app_count
        payload["manual_only"] = True
        write_json(labels_path, payload)
        return

    now = datetime.now(timezone.utc).isoformat()
    markers: list[dict[str, Any]] = []
    for marker_index, candidate in enumerate(selected, start=1):
        markers.append(
            {
                "id": f"auto_waveform_{session_id}_{marker_index:03d}",
                "time_s": rounded(candidate.get("time_s"), 6),
                "label": "racket",
                "note": "auto_waveform_peak_draft_review_required",
                "created_at": now,
                "source": "auto_waveform_peak_prefill",
                "gate_id": PREFILL_GATE_ID,
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
        "expected_count": expected_count,
        "reported_app_count": reported_app_count,
        "manual_only": True,
        "trigger_labels": {},
        "manual_markers": markers,
        "saved_at": now,
        "review_note": (
            "Auto-prefill from a soft waveform peak picker. Grey lines are actual saved "
            "Bounce audio test app/native candidates. Green labels are draft racket contacts "
            "for human correction only; drag, delete, add, then Save labels."
        ),
        "scenario_title": scenario_title,
        "prefill_gate": {
            "gate_id": PREFILL_GATE_ID,
            "config": {
                "envelope_mode": PREFILL_GATE.envelope_mode,
                "smooth_ms": PREFILL_GATE.smooth_ms,
                "min_gap_ms": PREFILL_GATE.min_gap_ms,
                "bg_ms": PREFILL_GATE.bg_ms,
                "bg_exclude_ms": PREFILL_GATE.bg_exclude_ms,
                "abs_min": PREFILL_GATE.abs_min,
                "ratio_min": PREFILL_GATE.ratio_min,
                "z_min": PREFILL_GATE.z_min,
            },
        },
    }
    write_json(labels_path, payload)


def server_command(
    *,
    session_id: str,
    raw_dir: Path,
    out_dir: Path,
    labels_dir: Path,
    trigger_csv: Path,
    host: str,
    port: int,
    expected_count: int,
    reported_app_count: int,
    gate_note: str,
) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT_DIR / "serve_t0053_trigger_review_ui.py"),
        "--host",
        host,
        "--port",
        str(port),
        "--session-id",
        session_id,
        "--raw-dir",
        str(raw_dir),
        "--eval-dir",
        str(out_dir),
        "--out-dir",
        str(labels_dir),
        "--trigger-csv",
        str(trigger_csv),
        "--manual-only",
        "--expected-count",
        str(expected_count),
        "--reported-app-count",
        str(reported_app_count),
        "--gate-note",
        gate_note,
    ]


def command_for_display(parts: list[str]) -> str:
    rendered: list[str] = []
    for part in parts:
        if " " in part or "\t" in part:
            rendered.append('"' + part.replace('"', '\\"') + '"')
        else:
            rendered.append(part)
    return " ".join(rendered)


def wait_for_port(host: str, port: int, timeout_s: float = 4.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.25)
            try:
                sock.connect((host, port))
                return True
            except OSError:
                time.sleep(0.1)
    return False


def maybe_start_server(command: list[str], out_dir: Path, host: str, port: int) -> int | None:
    logs_dir = out_dir / "server_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / f"server_{port}.out.log"
    stderr_path = logs_dir / f"server_{port}.err.log"
    stdout = stdout_path.open("a", encoding="utf-8")
    stderr = stderr_path.open("a", encoding="utf-8")
    process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
    if wait_for_port(host, port):
        return process.pid
    return process.pid


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    raw_dir = project_path(args.raw_dir).resolve()
    out_dir = project_path(args.out_dir).resolve()
    if not raw_dir.exists():
        raise FileNotFoundError(raw_dir)
    session_id = args.session_id or latest_session_id(raw_dir)
    payload = load_payload(raw_dir, session_id)
    wav_path = raw_dir / f"{session_id}.wav"
    if not wav_path.exists():
        raise FileNotFoundError(wav_path)

    samples, sample_rate = read_wav(wav_path)
    duration_s = len(samples) / float(sample_rate)
    saved_expected, saved_app_count, scenario_title = review_saved_values(payload)
    expected_count = args.expected_count if args.expected_count is not None else saved_expected
    if expected_count is None:
        expected_count = args.prefill_count
    reported_app_count = args.reported_app_count if args.reported_app_count is not None else saved_app_count
    if reported_app_count is None:
        reported_app_count = -1
    prefill_count = args.prefill_count if args.prefill_count is not None else expected_count

    app_rows = app_candidate_rows(payload, session_id, sample_rate)
    trigger_dir = out_dir / "trigger_csv"
    review_dir = out_dir / "review_pages"
    prefill_trigger_dir = review_dir / "trigger_csv"
    app_trigger_csv = trigger_dir / f"{session_id}_app_candidates.csv"
    write_csv(app_trigger_csv, app_rows, APP_TRIGGER_FIELDNAMES)

    peak_events = detect_peak_gate(samples, sample_rate, PREFILL_GATE)
    prefill_rows = waveform_prefill_rows(peak_events, session_id)
    selected = select_draft_markers(prefill_rows, prefill_count)
    prefill_csv = prefill_trigger_dir / f"{session_id}_{PREFILL_GATE_ID}.csv"
    write_csv(prefill_csv, prefill_rows, PREFILL_FIELDNAMES)

    labels_path = review_dir / f"{session_id}_review_labels.json"
    make_review_payload(
        labels_path=labels_path,
        session_id=session_id,
        raw_dir=raw_dir,
        selected=selected,
        expected_count=expected_count,
        reported_app_count=reported_app_count,
        scenario_title=scenario_title,
        force=args.force,
    )

    gate_note = args.gate_note or (
        "Grey lines are actual saved Bounce audio test app/native candidates. "
        "Green labels are waveform-prefilled draft racket contacts; verify by ear, "
        "drag/delete/add, then Save labels."
    )
    command = server_command(
        session_id=session_id,
        raw_dir=raw_dir,
        out_dir=out_dir,
        labels_dir=review_dir,
        trigger_csv=app_trigger_csv,
        host=args.host,
        port=args.port,
        expected_count=expected_count,
        reported_app_count=reported_app_count,
        gate_note=gate_note,
    )

    server_pid = maybe_start_server(command, out_dir, args.host, args.port) if args.start_server else None
    summary = {
        "session_id": session_id,
        "scenario_title": scenario_title,
        "raw_dir": rel(raw_dir),
        "out_dir": rel(out_dir),
        "wav_duration_s": round(duration_s, 3),
        "sample_rate_hz": sample_rate,
        "expected_count": expected_count,
        "reported_app_count": reported_app_count,
        "app_candidate_count": len(app_rows),
        "waveform_prefill_candidate_count": len(prefill_rows),
        "green_draft_label_count": len(selected),
        "app_trigger_csv": rel(app_trigger_csv),
        "waveform_prefill_csv": rel(prefill_csv),
        "review_labels_json": rel(labels_path),
        "review_url": f"http://{args.host}:{args.port}/",
        "review_command": command_for_display(command),
        "server_pid": server_pid,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out_dir / f"{session_id}_review_page_summary.json", summary)
    (out_dir / f"{session_id}_review_command.txt").write_text(summary["review_command"] + "\n", encoding="utf-8")
    report = [
        f"# Bounce Audio Test Review Page: {session_id}",
        "",
        f"- Scenario: `{scenario_title or 'unknown'}`",
        f"- Expected/app: `{expected_count}/{reported_app_count}`",
        f"- WAV: `{duration_s:.3f}s`, `{sample_rate} Hz`",
        f"- Grey app candidates: `{len(app_rows)}`",
        f"- Green waveform draft labels: `{len(selected)}` from `{len(prefill_rows)}` soft peak candidates",
        f"- Review URL: `{summary['review_url']}`",
        "",
        "Grey lines are the saved app/native candidates. Green labels are draft truth labels and must be corrected/saved by Love before ingestion or training.",
        "",
        "Run command:",
        "",
        "```powershell",
        summary["review_command"],
        "```",
    ]
    write_json(out_dir / "latest_review_page_summary.json", summary)
    (out_dir / f"{session_id}_review_page_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", default="", help="Saved bounce_audio_test_debug session id. Defaults to newest JSON in raw-dir.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8799)
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--reported-app-count", type=int, default=None)
    parser.add_argument("--prefill-count", type=int, default=None, help="Number of green draft labels. Defaults to expected count.")
    parser.add_argument("--gate-note", default="")
    parser.add_argument("--force", action="store_true", help="Overwrite existing review label JSON.")
    parser.add_argument("--start-server", action="store_true", help="Start the existing local review server in the background.")
    return parser.parse_args()


def main() -> None:
    summary = prepare(parse_args())
    print(
        f"{summary['review_url']} {summary['session_id']} "
        f"app_candidates={summary['app_candidate_count']} "
        f"green_labels={summary['green_draft_label_count']} "
        f"expected/app={summary['expected_count']}/{summary['reported_app_count']}"
    )
    if summary.get("server_pid"):
        print(f"server_pid={summary['server_pid']}")
    else:
        print(summary["review_command"])


if __name__ == "__main__":
    main()
