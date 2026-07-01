#!/usr/bin/env python3
"""Ingest T0104B positive review labels.

This is evaluation plumbing only. It reads the local T0104B review UI label
JSON files, writes exact racket-contact timestamp artifacts, and reports peak
candidate coverage. It does not train, export, install, or change app runtime.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from ingest_t0102_boundary_review_labels import (  # noqa: E402
    EXACT_MATCH_WINDOW_MS,
    NEAR_MATCH_WINDOW_MS,
    fmt_ms,
    greedy_matches,
    load_racket_markers,
    load_trigger_rows,
    marker_origin,
    md_table,
    nearest_trigger,
    project_path,
    safe_float,
    safe_int,
    write_csv,
    write_json,
)


DEFAULT_BASE_DIR = ROOT / "data/audio/models/evaluations/t0104b_positive_review_pages"
DEFAULT_MANIFEST = DEFAULT_BASE_DIR / "t0104b_review_page_manifest.csv"
DEFAULT_OUT_DIR = ROOT / "data/audio/models/evaluations/t0104d_t0104b_positive_label_ingest"
EXPECTED_CONFIRMED_COUNT = 30


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def expected_count(row: dict[str, str], payload: dict[str, Any]) -> int:
    for key in ("review_expected_count", "expected_racket_contacts", "expected_count", "saved_expected_count"):
        value = row.get(key)
        if value not in (None, ""):
            return safe_int(value)
    return safe_int(payload.get("expected_count"), EXPECTED_CONFIRMED_COUNT)


def scenario_title(row: dict[str, Any]) -> str:
    return str(row.get("scenario_title") or row.get("scenario_id") or "Unknown")


def aggregate_scenarios(session_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in session_rows:
        key = str(row["scenario_id"])
        bucket = grouped.setdefault(
            key,
            {
                "scenario_id": key,
                "scenario_title": row["scenario_title"],
                "runs": 0,
                "expected_count": 0,
                "reviewed_labels": 0,
                "app_count": 0,
                "peak_candidate_count": 0,
                "kept_auto_prefill": 0,
                "manual_added": 0,
                "deleted_draft": 0,
                "within_140ms": 0,
                "within_250ms": 0,
            },
        )
        bucket["runs"] += 1
        for field in (
            "expected_count",
            "reviewed_labels",
            "app_count",
            "peak_candidate_count",
            "kept_auto_prefill",
            "manual_added",
            "deleted_draft",
            "within_140ms",
            "within_250ms",
        ):
            bucket[field] += safe_int(row.get(field))
    return sorted(grouped.values(), key=lambda item: item["scenario_title"])


def render_report(summary: dict[str, Any], scenario_rows: list[dict[str, Any]], session_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# T0104D T0104B Positive Label Ingest",
        "",
        "## Summary",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Sessions ingested: `{summary['sessions_ingested']}`",
        f"- Expected contacts: `{summary['expected_total']}`",
        f"- Reviewed racket labels: `{summary['reviewed_total']}`",
        f"- Labels within 140 ms of a peak candidate: `{summary['labels_within_140ms']}`",
        f"- Labels within 250 ms of a peak candidate: `{summary['labels_within_250ms']}`",
        f"- Kept auto-prefill labels: `{summary['kept_auto_prefill_total']}`",
        f"- Manual labels added: `{summary['manual_added_total']}`",
        f"- Deleted draft labels: `{summary['deleted_draft_total']}`",
        f"- Issues: `{len(summary['issues'])}`",
        "",
        "## By Scenario",
        "",
    ]
    lines.extend(
        md_table(
            scenario_rows,
            [
                "scenario_title",
                "runs",
                "expected_count",
                "reviewed_labels",
                "app_count",
                "peak_candidate_count",
                "within_140ms",
                "within_250ms",
                "manual_added",
                "deleted_draft",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## By Session",
            "",
        ]
    )
    lines.extend(
        md_table(
            session_rows,
            [
                "session_id",
                "scenario_id",
                "expected_count",
                "reviewed_labels",
                "app_count",
                "draft_label_count",
                "peak_candidate_count",
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
            "## Interpretation",
            "",
            "- These are exact positive labels for the fresh T0104 live validation WAVs. They are suitable for the next candidate-training/evaluation loop.",
            "- The labels alone are not a promotion gate; the next model still has to pass the existing boundary negatives, T0073 rejected unsafe cases, and Round A hard-negative safety.",
            "- `within_140ms` is the strict app-style candidate coverage view; `within_250ms` is the wider near-candidate diagnostic view.",
            "",
        ]
    )
    return "\n".join(lines)


def ingest(manifest_path: Path, out_dir: Path) -> dict[str, Any]:
    rows = read_csv(manifest_path)
    label_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    generated_at = datetime.now(timezone.utc).isoformat()

    for row in rows:
        session_id = str(row.get("session_id") or "")
        labels_path = project_path(row.get("review_labels_json", ""))
        trigger_path = project_path(row.get("trigger_csv", ""))
        if not labels_path.exists():
            issues.append({"session_id": session_id, "issue": "missing_review_labels_json", "path": str(labels_path)})
            continue
        payload, labels = load_racket_markers(labels_path)
        triggers = load_trigger_rows(trigger_path)
        expected = expected_count(row, payload)
        draft_count = safe_int(row.get("draft_label_count"))
        app_count = safe_int(row.get("saved_app_count"))

        exact_matches = greedy_matches(labels, triggers, EXACT_MATCH_WINDOW_MS)
        near_matches = greedy_matches(labels, triggers, NEAR_MATCH_WINDOW_MS)
        assigned_exact_trigger_indexes = {trigger_index for trigger_index in exact_matches.values()}

        nearest_abs_values: list[float] = []
        kept_auto_prefill = 0
        manual_added = 0
        within_140 = 0
        within_250 = 0
        scenario_id = str(row.get("scenario_id") or payload.get("scenario_id") or "")
        title = scenario_title(row)

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
                    "scenario_id": scenario_id,
                    "scenario_title": title,
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
                    "within_140ms": label_index - 1 in exact_matches,
                    "within_250ms": label_index - 1 in near_matches,
                    "expected_count": expected,
                }
            )

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
                    "scenario_id": scenario_id,
                    "candidate_index": trigger.get("candidate_index", ""),
                    "trigger_index": trigger.get("trigger_index", ""),
                    "time_s": f"{safe_float(trigger.get('time_s')):.6f}",
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
        if expected != EXPECTED_CONFIRMED_COUNT:
            issues.append(
                {
                    "session_id": session_id,
                    "issue": "expected_count_not_confirmed_30",
                    "expected": expected,
                }
            )
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
                "scenario_id": scenario_id,
                "scenario_title": title,
                "expected_count": expected,
                "reviewed_labels": reviewed,
                "app_count": app_count,
                "draft_label_count": draft_count,
                "peak_candidate_count": safe_int(row.get("peak_candidate_count")),
                "kept_auto_prefill": kept_auto_prefill,
                "manual_added": manual_added,
                "deleted_draft": deleted_draft,
                "within_140ms": within_140,
                "within_250ms": within_250,
                "median_abs_nearest_delta_ms": ""
                if not nearest_abs_values
                else f"{statistics.median(nearest_abs_values):.3f}",
                "max_abs_nearest_delta_ms": "" if not nearest_abs_values else f"{max(nearest_abs_values):.3f}",
                "saved_at": payload.get("saved_at", ""),
                "labels_json": str(labels_path),
                "trigger_csv": str(trigger_path),
            }
        )

    scenario_rows = aggregate_scenarios(session_rows)
    summary = {
        "generated_at": generated_at,
        "manifest_path": str(manifest_path),
        "out_dir": str(out_dir),
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

    write_csv(out_dir / "t0104d_reviewed_positive_labels.csv", label_rows)
    write_csv(out_dir / "t0104d_nearest_peak_matches.csv", match_rows)
    write_csv(out_dir / "t0104d_session_summary.csv", session_rows)
    write_csv(out_dir / "t0104d_scenario_summary.csv", scenario_rows)
    write_json(out_dir / "t0104d_summary.json", summary)
    (out_dir / "t0104d_report.md").write_text(render_report(summary, scenario_rows, session_rows), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    manifest = project_path(args.manifest)
    out_dir = project_path(args.out_dir)
    summary = ingest(manifest, out_dir)
    print(f"Wrote {out_dir}")
    print(f"Ingested {summary['sessions_ingested']} sessions, {summary['reviewed_total']}/{summary['expected_total']} labels")
    if summary["issues"]:
        print(f"Issues: {len(summary['issues'])}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
