#!/usr/bin/env python3
"""Prepare exact-label review pages for selected T0104 positive runs.

This is a local review helper only. It does not train/export a model, change
app runtime behavior, install an APK, or delete phone/local data.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_t0067_peak_gate_replay import detect_peak_gate, read_wav  # noqa: E402
from prepare_t0104a_slow_high_review_pages import (  # noqa: E402
    DEFAULT_RAW_DIR,
    GATE_ID,
    PEAK_FAST_BALANCED,
    TRIGGER_FIELDNAMES,
    normalize_peak_rows,
    rel,
    rounded,
    select_draft_markers,
    write_csv,
    write_json,
)


DEFAULT_OUT_DIR = ROOT / "data/audio/models/evaluations/t0104b_positive_review_pages"
EXPECTED_DEFAULT = 30
TARGET_SCENARIOS = [
    "normal_racket_bounce",
    "fast_racket_bounce",
    "racket_bounce_speaking_counting",
    "racket_bounce_background_sound",
    "far_soft_racket_bounce_background",
]

MANIFEST_FIELDNAMES = [
    "review_url",
    "port",
    "session_id",
    "started_at",
    "scenario_id",
    "scenario_title",
    "saved_expected_count",
    "saved_app_count",
    "review_expected_count",
    "wav_duration_s",
    "peak_candidate_count",
    "draft_label_count",
    "trigger_csv",
    "review_labels_json",
    "review_command",
    "issues",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _review_info(payload: dict[str, Any]) -> tuple[str, str, str, int | None, int | None]:
    review = payload.get("review") if isinstance(payload.get("review"), dict) else {}
    scenario = review.get("scenario") if isinstance(review.get("scenario"), dict) else {}
    scenario_id = str(scenario.get("id") or "unknown")
    scenario_title = str(scenario.get("title") or scenario_id)
    polarity = str(scenario.get("polarity") or "unknown")
    expected = review.get("expected_racket_contacts")
    app_count = review.get("app_count_at_stop")
    return (
        scenario_id,
        scenario_title,
        polarity,
        expected if isinstance(expected, int) else None,
        app_count if isinstance(app_count, int) else None,
    )


def _find_target_sessions(raw_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for json_path in sorted(raw_dir.glob("bounce_audio_test_session_2026-07-01T*.json")):
        payload = _read_json(json_path)
        if payload.get("type") != "bounce_audio_test_debug_session":
            continue
        scenario_id, scenario_title, polarity, expected, app_count = _review_info(payload)
        if scenario_id not in TARGET_SCENARIOS:
            continue
        rows.append(
            {
                "session_id": json_path.stem,
                "json_path": json_path,
                "wav_path": raw_dir / f"{json_path.stem}.wav",
                "started_at": str(payload.get("started_at") or ""),
                "scenario_id": scenario_id,
                "scenario_title": scenario_title,
                "polarity": polarity,
                "saved_expected_count": expected,
                "saved_app_count": app_count,
                "sort_key": (TARGET_SCENARIOS.index(scenario_id), str(payload.get("started_at") or "")),
            }
        )
    rows.sort(key=lambda item: item["sort_key"])
    return rows


def _make_review_payload(
    *,
    row: dict[str, Any],
    raw_dir: Path,
    out_dir: Path,
    selected: list[dict[str, Any]],
    expected_count: int,
    force: bool,
) -> Path:
    session_id = str(row["session_id"])
    labels_path = out_dir / "review_pages" / f"{session_id}_review_labels.json"
    if labels_path.exists() and not force:
        payload = _read_json(labels_path)
        payload["expected_count"] = expected_count
        payload["reported_app_count"] = row.get("saved_app_count") if row.get("saved_app_count") is not None else -1
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
        "expected_count": expected_count,
        "reported_app_count": row.get("saved_app_count") if row.get("saved_app_count") is not None else -1,
        "manual_only": True,
        "trigger_labels": {},
        "manual_markers": markers,
        "saved_at": now,
        "scenario_id": row.get("scenario_id", ""),
        "scenario_title": row.get("scenario_title", ""),
        "review_note": (
            "T0104B auto-prefill from peak_fast_balanced. Verify green labels, "
            "delete false labels, drag mistimed labels, add missing racket contacts, and save."
        ),
    }
    write_json(labels_path, payload)
    return labels_path


def _review_command(*, session_id: str, raw_dir: Path, out_dir: Path, trigger_csv: Path, expected_count: int, port: int) -> str:
    script = SCRIPT_DIR / "serve_t0053_trigger_review_ui.py"
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
            f"--expected-count {expected_count}",
            "--reported-app-count -1",
            '--gate-note "T0104BPositiveReview"',
        ]
    )


def prepare(args: argparse.Namespace) -> list[dict[str, Any]]:
    raw_dir = args.raw_dir.resolve()
    out_dir = args.out_dir.resolve()
    trigger_dir = out_dir / "review_pages" / "trigger_csv"
    source_rows = _find_target_sessions(raw_dir)
    manifest: list[dict[str, Any]] = []

    for index, row in enumerate(source_rows):
        port = args.port_start + index
        session_id = str(row["session_id"])
        expected_count = int(row.get("saved_expected_count") or EXPECTED_DEFAULT)
        trigger_csv = trigger_dir / f"{session_id}_{GATE_ID}_triggers.csv"
        labels_path = out_dir / "review_pages" / f"{session_id}_review_labels.json"
        issues: list[str] = []
        peak_rows: list[dict[str, Any]] = []
        selected: list[dict[str, Any]] = []
        duration_s = ""

        wav_path = Path(row["wav_path"])
        if not wav_path.exists():
            issues.append("missing_wav")
        if not Path(row["json_path"]).exists():
            issues.append("missing_json")
        if row.get("polarity") != "positive":
            issues.append("not_positive")

        if not issues:
            samples, sample_rate = read_wav(wav_path)
            duration_s = f"{len(samples) / float(sample_rate):.3f}"
            peak_rows = normalize_peak_rows(detect_peak_gate(samples, sample_rate, PEAK_FAST_BALANCED), session_id)
            selected = select_draft_markers(peak_rows, expected_count)
            write_csv(trigger_csv, peak_rows, TRIGGER_FIELDNAMES)
            labels_path = _make_review_payload(
                row=row,
                raw_dir=raw_dir,
                out_dir=out_dir,
                selected=selected,
                expected_count=expected_count,
                force=args.force,
            )

        review_url = f"http://127.0.0.1:{port}/" if not issues else ""
        command = (
            _review_command(
                session_id=session_id,
                raw_dir=raw_dir,
                out_dir=out_dir,
                trigger_csv=trigger_csv,
                expected_count=expected_count,
                port=port,
            )
            if not issues
            else ""
        )
        manifest.append(
            {
                "review_url": review_url,
                "port": port,
                "session_id": session_id,
                "started_at": row.get("started_at", ""),
                "scenario_id": row.get("scenario_id", ""),
                "scenario_title": row.get("scenario_title", ""),
                "saved_expected_count": row.get("saved_expected_count") if row.get("saved_expected_count") is not None else "",
                "saved_app_count": row.get("saved_app_count") if row.get("saved_app_count") is not None else "",
                "review_expected_count": expected_count,
                "wav_duration_s": duration_s,
                "peak_candidate_count": len(peak_rows),
                "draft_label_count": len(selected),
                "trigger_csv": rel(trigger_csv) if trigger_csv.exists() else "",
                "review_labels_json": rel(labels_path) if labels_path.exists() else "",
                "review_command": command,
                "issues": ";".join(issues),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "t0104b_review_page_manifest.csv", manifest, MANIFEST_FIELDNAMES)
    _write_report(out_dir / "t0104b_report.md", manifest)
    return manifest


def _write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# T0104B Positive Review Pages",
        "",
        "Exact-label review pages for selected corrected T0104 positive Bounce audio test runs.",
        "",
        "Gray lines are waveform peak candidates. Green labels are editable draft racket contacts.",
        "",
        "| URL | Scenario | Session | Expected | App | Peaks | Draft labels |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {review_url} | {scenario_title} | `{session_id}` | {review_expected_count} | "
            "{saved_app_count} | {peak_candidate_count} | {draft_label_count} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Instructions: verify the green labels, delete false labels, drag labels onto the exact bounce peak, add missing contacts, and click Save labels.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--port-start", type=int, default=8783)
    parser.add_argument("--force", action="store_true", help="Overwrite existing prefilled review label JSON files.")
    args = parser.parse_args()
    rows = prepare(args)
    for row in rows:
        print(
            f"{row['review_url']} {row['scenario_id']} {row['session_id']} "
            f"expected={row['review_expected_count']} peaks={row['peak_candidate_count']} "
            f"draft={row['draft_label_count']} issues={row['issues']}"
        )


if __name__ == "__main__":
    main()
