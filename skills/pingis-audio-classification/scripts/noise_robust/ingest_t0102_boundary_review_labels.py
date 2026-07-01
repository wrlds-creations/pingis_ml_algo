#!/usr/bin/env python3
"""Ingest T0102 boundary review labels into timestamp CSV artifacts.

This is evaluation plumbing only. It reads the local full-WAV review UI label
JSON files created by ``prepare_t0102_boundary_recorder_pack.py`` and writes
reviewed racket-contact timestamps plus nearest peak-candidate diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BASE_DIR = (
    ROOT
    / "data/audio/models/evaluations/t0102_boundary_recorder_pack_pull_review/fresh_pack_2026_07_01"
)
DEFAULT_OUT_DIR = ROOT / "data/audio/models/evaluations/t0102f_boundary_review_label_ingest"
EXACT_MATCH_WINDOW_MS = 140.0
NEAR_MATCH_WINDOW_MS = 250.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["empty"]
        rows = [{"empty": ""}]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out else default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_manifest_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        if not path.exists():
            continue
        rows.extend(read_csv(path))
    return rows


def parse_expected_overrides(values: list[str]) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected override must use SESSION_ID=COUNT, got {value!r}")
        session_id, count_text = value.split("=", 1)
        session_id = session_id.strip()
        count = safe_int(count_text, -1)
        if not session_id or count < 0:
            raise ValueError(f"Expected override must use SESSION_ID=COUNT, got {value!r}")
        overrides[session_id] = count
    return overrides


def load_racket_markers(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    markers: list[dict[str, Any]] = []
    for marker in payload.get("manual_markers") or []:
        label = str(marker.get("label") or "").strip().lower()
        if label not in {"racket", "racket_bounce", "racket_contact"}:
            continue
        time_s = safe_float(marker.get("time_s"))
        if not time_s == time_s:
            continue
        markers.append({**marker, "time_s": time_s, "label": "racket_bounce"})
    markers.sort(key=lambda row: float(row["time_s"]))
    return payload, markers


def load_trigger_rows(path: Path) -> list[dict[str, Any]]:
    triggers: list[dict[str, Any]] = []
    if not path.exists():
        return triggers
    for fallback_index, row in enumerate(read_csv(path), start=1):
        time_s = safe_float(row.get("time_s"))
        if not time_s == time_s:
            onset_ms = safe_float(row.get("onset_ms"))
            time_s = onset_ms / 1000.0 if onset_ms == onset_ms else float("nan")
        if not time_s == time_s:
            continue
        triggers.append(
            {
                **row,
                "trigger_index": safe_int(row.get("trigger_index"), fallback_index),
                "candidate_index": safe_int(row.get("candidate_index"), fallback_index),
                "time_s": time_s,
                "time_ms": time_s * 1000.0,
            }
        )
    triggers.sort(key=lambda row: float(row["time_s"]))
    return triggers


def nearest_trigger(label_time_s: float, triggers: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float]:
    if not triggers:
        return None, float("nan")
    best = min(triggers, key=lambda row: abs(float(row["time_s"]) - label_time_s))
    delta_ms = (float(best["time_s"]) - label_time_s) * 1000.0
    return best, delta_ms


def greedy_matches(labels: list[dict[str, Any]], triggers: list[dict[str, Any]], window_ms: float) -> dict[int, int]:
    pairs: list[tuple[float, int, int]] = []
    for label_index, label in enumerate(labels):
        label_ms = float(label["time_s"]) * 1000.0
        for trigger_index, trigger in enumerate(triggers):
            delta = abs(float(trigger["time_ms"]) - label_ms)
            if delta <= window_ms:
                pairs.append((delta, label_index, trigger_index))
    pairs.sort(key=lambda item: item[0])
    used_labels: set[int] = set()
    used_triggers: set[int] = set()
    matches: dict[int, int] = {}
    for _, label_index, trigger_index in pairs:
        if label_index in used_labels or trigger_index in used_triggers:
            continue
        used_labels.add(label_index)
        used_triggers.add(trigger_index)
        matches[label_index] = trigger_index
    return matches


def marker_origin(marker: dict[str, Any]) -> str:
    source = str(marker.get("source") or "")
    marker_id = str(marker.get("id") or "")
    if source == "auto_waveform_peak_prefill" or marker_id.startswith("auto_peak_"):
        return "kept_auto_prefill"
    return "manual_added"


def fmt_ms(value: float) -> str:
    return "" if not value == value else f"{value:.3f}"


def md_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return lines


def render_report(summary: dict[str, Any], session_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# T0102/T0103 Boundary Review Label Ingest",
        "",
        "## Summary",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Sessions requested: `{summary['sessions_requested']}`",
        f"- Sessions ingested: `{summary['sessions_ingested']}`",
        f"- Expected contacts: `{summary['expected_total']}`",
        f"- Reviewed racket labels: `{summary['reviewed_total']}`",
        f"- Labels within 140 ms of a peak candidate: `{summary['labels_within_140ms']}`",
        f"- Labels within 250 ms of a peak candidate: `{summary['labels_within_250ms']}`",
        f"- Kept auto-prefill labels: `{summary['kept_auto_prefill_total']}`",
        f"- Manual labels added: `{summary['manual_added_total']}`",
        f"- Deleted draft labels: `{summary['deleted_draft_total']}`",
        f"- Expected-count overrides: `{len(summary['expected_overrides'])}`",
        "",
        "## Sessions",
        "",
    ]
    lines.extend(
        md_table(
            session_rows,
            [
                "session_id",
                "scenario_id",
                "expected_count",
                "expected_override_applied",
                "reviewed_labels",
                "draft_label_count",
                "kept_auto_prefill",
                "manual_added",
                "deleted_draft",
                "within_140ms",
                "within_250ms",
                "max_abs_nearest_delta_ms",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            "These labels are positive timestamp ground truth only. Training/export decisions must still be made against boundary negatives and older safety gates.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="Review manifest CSV. Defaults to strict T0102 plus normal-noisy manifests under base-dir.",
    )
    parser.add_argument(
        "--sessions",
        nargs="*",
        default=[],
        help="Optional session IDs to ingest. If omitted, all manifest rows with label files are ingested.",
    )
    parser.add_argument(
        "--expected-override",
        action="append",
        default=[],
        metavar="SESSION_ID=COUNT",
        help="Override expected racket-contact count for a reviewed session.",
    )
    args = parser.parse_args()

    base_dir = project_path(args.base_dir)
    out_dir = project_path(args.out_dir)
    expected_overrides = parse_expected_overrides(args.expected_override)
    manifest_paths = [project_path(path) for path in args.manifest]
    if not manifest_paths:
        manifest_paths = [
            base_dir / "t0102_review_page_manifest.csv",
            base_dir / "normal_noisy_review" / "t0102_review_page_manifest.csv",
        ]
    requested_sessions = set(args.sessions)
    generated_at = datetime.now(timezone.utc).isoformat()

    manifest_rows = load_manifest_rows(manifest_paths)
    selected_rows = [
        row
        for row in manifest_rows
        if not requested_sessions or row.get("session_id", "") in requested_sessions
    ]

    label_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    for row in selected_rows:
        session_id = row.get("session_id", "")
        labels_path = project_path(row.get("review_labels_json", ""))
        trigger_path = project_path(row.get("trigger_csv", ""))
        manifest_expected = safe_int(row.get("expected_racket_contacts"))
        expected_override_applied = session_id in expected_overrides
        expected = expected_overrides.get(session_id, manifest_expected)
        draft_count = safe_int(row.get("draft_label_count"))
        if not labels_path.exists():
            issues.append({"session_id": session_id, "issue": "missing_review_labels_json", "path": str(labels_path)})
            continue
        payload, labels = load_racket_markers(labels_path)
        triggers = load_trigger_rows(trigger_path)
        exact_matches = greedy_matches(labels, triggers, EXACT_MATCH_WINDOW_MS)
        near_matches = greedy_matches(labels, triggers, NEAR_MATCH_WINDOW_MS)

        nearest_abs_values: list[float] = []
        kept_auto_prefill = 0
        manual_added = 0
        within_140 = 0
        within_250 = 0

        for label_index, marker in enumerate(labels, start=1):
            origin = marker_origin(marker)
            kept_auto_prefill += 1 if origin == "kept_auto_prefill" else 0
            manual_added += 1 if origin == "manual_added" else 0
            nearest, delta_ms = nearest_trigger(float(marker["time_s"]), triggers)
            abs_delta_ms = abs(delta_ms) if delta_ms == delta_ms else float("nan")
            if abs_delta_ms == abs_delta_ms:
                nearest_abs_values.append(abs_delta_ms)
            within_140 += 1 if label_index - 1 in exact_matches else 0
            within_250 += 1 if label_index - 1 in near_matches else 0
            label_rows.append(
                {
                    "session_id": session_id,
                    "scenario_id": row.get("scenario_id", ""),
                    "scenario_title": row.get("scenario_title", ""),
                    "label_index": label_index,
                    "label": "racket_bounce",
                    "reviewed_time_s": f"{float(marker['time_s']):.6f}",
                    "reviewed_time_ms": f"{float(marker['time_s']) * 1000.0:.3f}",
                    "marker_id": marker.get("id", ""),
                    "marker_origin": origin,
                    "marker_note": marker.get("note", ""),
                    "source_candidate_index": marker.get("source_candidate_index", ""),
                    "nearest_candidate_index": "" if nearest is None else nearest.get("candidate_index", ""),
                    "nearest_trigger_index": "" if nearest is None else nearest.get("trigger_index", ""),
                    "nearest_time_s": "" if nearest is None else f"{float(nearest['time_s']):.6f}",
                    "nearest_delta_ms": fmt_ms(delta_ms),
                    "nearest_abs_delta_ms": fmt_ms(abs_delta_ms),
                    "within_140ms": abs_delta_ms == abs_delta_ms and abs_delta_ms <= EXACT_MATCH_WINDOW_MS,
                    "within_250ms": abs_delta_ms == abs_delta_ms and abs_delta_ms <= NEAR_MATCH_WINDOW_MS,
                    "expected_count": expected,
                    "manifest_expected_count": manifest_expected,
                    "expected_override_applied": expected_override_applied,
                }
            )

        assigned_exact_trigger_indexes = {trigger_index for trigger_index in exact_matches.values()}
        for trigger_index, trigger in enumerate(triggers):
            nearest_label_index = ""
            nearest_label_delta_ms = ""
            nearest_label_abs_delta_ms = ""
            if labels:
                label_pos, nearest_label = min(
                    enumerate(labels),
                    key=lambda item: abs(float(trigger["time_s"]) - float(item[1]["time_s"])),
                )
                delta_ms = (float(trigger["time_s"]) - float(nearest_label["time_s"])) * 1000.0
                nearest_label_index = label_pos + 1
                nearest_label_delta_ms = fmt_ms(delta_ms)
                nearest_label_abs_delta_ms = fmt_ms(abs(delta_ms))
            if trigger_index in assigned_exact_trigger_indexes:
                status = "matched_racket_within_140ms"
            elif nearest_label_abs_delta_ms != "" and float(nearest_label_abs_delta_ms) <= NEAR_MATCH_WINDOW_MS:
                status = "near_racket_or_duplicate_within_250ms"
            else:
                status = "unmatched_peak_candidate"
            match_rows.append(
                {
                    "session_id": session_id,
                    "scenario_id": row.get("scenario_id", ""),
                    "candidate_index": trigger.get("candidate_index", ""),
                    "trigger_index": trigger.get("trigger_index", ""),
                    "time_s": f"{float(trigger['time_s']):.6f}",
                    "peak_value": trigger.get("peak_value", ""),
                    "peak_ratio": trigger.get("peak_ratio", ""),
                    "peak_z": trigger.get("peak_z", ""),
                    "selection_score": trigger.get("selection_score", ""),
                    "nearest_label_index": nearest_label_index,
                    "nearest_label_delta_ms": nearest_label_delta_ms,
                    "nearest_label_abs_delta_ms": nearest_label_abs_delta_ms,
                    "match_status": status,
                }
            )

        reviewed = len(labels)
        deleted_draft = max(0, draft_count - kept_auto_prefill)
        if reviewed != expected:
            issues.append(
                {
                    "session_id": session_id,
                    "issue": "reviewed_count_differs_from_expected",
                    "expected": expected,
                    "reviewed": reviewed,
                }
            )
        session_rows.append(
            {
                "session_id": session_id,
                "scenario_id": row.get("scenario_id", ""),
                "scenario_title": row.get("scenario_title", ""),
                "expected_count": expected,
                "manifest_expected_count": manifest_expected,
                "expected_override_applied": expected_override_applied,
                "payload_expected_count": payload.get("expected_count", ""),
                "reviewed_labels": reviewed,
                "draft_label_count": draft_count,
                "peak_candidate_count": safe_int(row.get("peak_candidate_count")),
                "kept_auto_prefill": kept_auto_prefill,
                "manual_added": manual_added,
                "deleted_draft": deleted_draft,
                "within_140ms": within_140,
                "within_250ms": within_250,
                "median_abs_nearest_delta_ms": "" if not nearest_abs_values else f"{statistics.median(nearest_abs_values):.3f}",
                "max_abs_nearest_delta_ms": "" if not nearest_abs_values else f"{max(nearest_abs_values):.3f}",
                "saved_at": payload.get("saved_at", ""),
                "labels_json": str(labels_path),
                "trigger_csv": str(trigger_path),
            }
        )

    summary = {
        "generated_at": generated_at,
        "manifest_paths": [str(path) for path in manifest_paths],
        "out_dir": str(out_dir),
        "expected_overrides": expected_overrides,
        "sessions_requested": len(requested_sessions) if requested_sessions else len(selected_rows),
        "sessions_ingested": len(session_rows),
        "expected_total": sum(safe_int(row.get("expected_count")) for row in session_rows),
        "reviewed_total": sum(safe_int(row.get("reviewed_labels")) for row in session_rows),
        "labels_within_140ms": sum(safe_int(row.get("within_140ms")) for row in session_rows),
        "labels_within_250ms": sum(safe_int(row.get("within_250ms")) for row in session_rows),
        "kept_auto_prefill_total": sum(safe_int(row.get("kept_auto_prefill")) for row in session_rows),
        "manual_added_total": sum(safe_int(row.get("manual_added")) for row in session_rows),
        "deleted_draft_total": sum(safe_int(row.get("deleted_draft")) for row in session_rows),
        "issues": issues,
    }

    write_csv(out_dir / "t0102f_reviewed_positive_labels.csv", label_rows)
    write_csv(out_dir / "t0102f_nearest_peak_matches.csv", match_rows)
    write_csv(out_dir / "t0102f_session_summary.csv", session_rows)
    write_json(out_dir / "t0102f_summary.json", summary)
    (out_dir / "t0102f_report.md").write_text(render_report(summary, session_rows), encoding="utf-8")

    print(f"Wrote {out_dir}")
    print(
        "Ingested "
        f"{summary['sessions_ingested']} sessions, {summary['reviewed_total']}/{summary['expected_total']} labels"
    )
    if issues:
        print(f"Issues: {len(issues)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
