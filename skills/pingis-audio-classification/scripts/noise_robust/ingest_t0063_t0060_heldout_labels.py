#!/usr/bin/env python3
"""Ingest T0060 manual held-out labels from the local review UI.

This is evaluation plumbing only. It converts the manual review JSON saved by
the T0062 browser UI into a simple labels CSV for the T0057 loop and produces
nearest-trigger diagnostics for the 60 replay candidates.
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
SESSION_ID = "fable_live_session_2026-06-29T13-29-50-713Z"
DEFAULT_LABELS_JSON = (
    ROOT
    / "data/audio/models/evaluations/t0061_t0060_trigger_review_ui"
    / f"{SESSION_ID}_review_labels.json"
)
DEFAULT_TRIGGER_CSV = (
    ROOT
    / "data/audio/models/evaluations/t0060_fresh_heldout_c2_loop"
    / "t0057_heldout_count_replay_events.csv"
)
DEFAULT_OUT_DIR = ROOT / "data/audio/models/evaluations/t0063_t0060_heldout_label_ingest"
EXACT_MATCH_WINDOW_MS = 140.0
NEAR_MATCH_WINDOW_MS = 250.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if out == out else default


def load_manual_labels(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    markers = []
    for index, marker in enumerate(payload.get("manual_markers") or [], start=1):
        label = str(marker.get("label") or "").strip().lower()
        if label not in {"racket", "racket_bounce", "racket_contact"}:
            continue
        time_s = finite_float(marker.get("time_s"))
        if not time_s == time_s:
            continue
        markers.append(
            {
                "label_index": index,
                "marker_id": marker.get("id") or f"manual_{index}",
                "label": "racket",
                "time_s": round(time_s, 6),
                "reviewed_time_s": round(time_s, 6),
                "note": marker.get("note", ""),
                "created_at": marker.get("created_at", ""),
            }
        )
    markers.sort(key=lambda row: float(row["time_s"]))
    for index, marker in enumerate(markers, start=1):
        marker["label_index"] = index
    return payload, markers


def load_triggers(path: Path) -> list[dict[str, Any]]:
    triggers = []
    for index, row in enumerate(read_csv(path), start=1):
        onset_ms = finite_float(row.get("onset_ms"))
        if not onset_ms == onset_ms:
            continue
        triggers.append(
            {
                "trigger_index": int(row.get("trigger_index") or index),
                "onset_ms": onset_ms,
                "onset_s": round(onset_ms / 1000.0, 6),
                "counted": str(row.get("counted", "")).strip(),
                "reject_reason": row.get("reject_reason", ""),
                "prob_racket_bounce": row.get("prob_racket_bounce", ""),
                "prob_noise": row.get("prob_noise", ""),
                "frame_rms": row.get("frame_rms", ""),
                "background_rms": row.get("background_rms", ""),
            }
        )
    triggers.sort(key=lambda row: float(row["onset_ms"]))
    return triggers


def nearest_trigger_rows(labels: list[dict[str, Any]], triggers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for label in labels:
        time_ms = float(label["time_s"]) * 1000.0
        nearest = min(triggers, key=lambda trigger: abs(float(trigger["onset_ms"]) - time_ms))
        delta_ms = float(nearest["onset_ms"]) - time_ms
        abs_delta_ms = abs(delta_ms)
        rows.append(
            {
                **label,
                "label_time_ms": round(time_ms, 3),
                "nearest_trigger_index": nearest["trigger_index"],
                "nearest_trigger_onset_ms": round(float(nearest["onset_ms"]), 3),
                "nearest_trigger_delta_ms": round(delta_ms, 3),
                "nearest_trigger_abs_delta_ms": round(abs_delta_ms, 3),
                "within_60ms": abs_delta_ms <= 60.0,
                "within_120ms": abs_delta_ms <= 120.0,
                "within_140ms": abs_delta_ms <= EXACT_MATCH_WINDOW_MS,
                "within_250ms": abs_delta_ms <= NEAR_MATCH_WINDOW_MS,
                "nearest_trigger_counted": nearest.get("counted", ""),
                "nearest_trigger_reject_reason": nearest.get("reject_reason", ""),
                "nearest_trigger_prob_racket_bounce": nearest.get("prob_racket_bounce", ""),
            }
        )
    return rows


def greedy_exact_matches(labels: list[dict[str, Any]], triggers: list[dict[str, Any]]) -> dict[int, int]:
    pairs: list[tuple[float, int, int]] = []
    for label_idx, label in enumerate(labels):
        time_ms = float(label["time_s"]) * 1000.0
        for trigger_idx, trigger in enumerate(triggers):
            delta = abs(float(trigger["onset_ms"]) - time_ms)
            if delta <= EXACT_MATCH_WINDOW_MS:
                pairs.append((delta, label_idx, trigger_idx))
    pairs.sort(key=lambda item: item[0])
    used_labels: set[int] = set()
    used_triggers: set[int] = set()
    matched: dict[int, int] = {}
    for _, label_idx, trigger_idx in pairs:
        if label_idx in used_labels or trigger_idx in used_triggers:
            continue
        used_labels.add(label_idx)
        used_triggers.add(trigger_idx)
        matched[trigger_idx] = label_idx
    return matched


def trigger_match_rows(labels: list[dict[str, Any]], triggers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exact_matches = greedy_exact_matches(labels, triggers)
    rows = []
    for trigger_idx, trigger in enumerate(triggers):
        onset_ms = float(trigger["onset_ms"])
        nearest_label_idx, nearest_label = min(
            enumerate(labels),
            key=lambda item: abs(onset_ms - float(item[1]["time_s"]) * 1000.0),
        )
        label_time_ms = float(nearest_label["time_s"]) * 1000.0
        delta_ms = onset_ms - label_time_ms
        abs_delta_ms = abs(delta_ms)
        assigned_label_idx = exact_matches.get(trigger_idx)
        if assigned_label_idx is not None:
            status = "exact_positive"
        elif abs_delta_ms <= NEAR_MATCH_WINDOW_MS:
            status = "weak_near_racket_or_duplicate"
        else:
            status = "unmatched_negative_or_duplicate"
        rows.append(
            {
                **trigger,
                "nearest_label_index": nearest_label_idx + 1,
                "nearest_label_time_s": nearest_label["time_s"],
                "nearest_label_delta_ms": round(delta_ms, 3),
                "nearest_label_abs_delta_ms": round(abs_delta_ms, 3),
                "assigned_exact_label_index": "" if assigned_label_idx is None else assigned_label_idx + 1,
                "match_status": status,
            }
        )
    return rows


def render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# T0063 T0060 Held-Out Label Ingest",
            "",
            "## Input",
            "",
            f"- Session: `{summary['session_id']}`",
            f"- Labels JSON: `{summary['labels_json']}`",
            f"- Trigger CSV: `{summary['trigger_csv']}`",
            "",
            "## Manual Labels",
            "",
            f"- Expected count: `{summary['expected_count']}`",
            f"- Saved manual racket labels: `{summary['manual_racket_labels']}`",
            f"- First/last label: `{summary['first_label_s']}` s / `{summary['last_label_s']}` s",
            "",
            "## Nearest Trigger Coverage",
            "",
            f"- Replay triggers: `{summary['trigger_count']}`",
            f"- Labels within 60 ms of a trigger: `{summary['labels_within_60ms']}`",
            f"- Labels within 120 ms of a trigger: `{summary['labels_within_120ms']}`",
            f"- Labels within 140 ms of a trigger: `{summary['labels_within_140ms']}`",
            f"- Labels within 250 ms of a trigger: `{summary['labels_within_250ms']}`",
            f"- Median nearest-trigger delta: `{summary['median_abs_nearest_trigger_delta_ms']}` ms",
            f"- Max nearest-trigger delta: `{summary['max_abs_nearest_trigger_delta_ms']}` ms",
            "",
            "## Trigger Classification",
            "",
            f"- Greedy one-to-one exact positives within 140 ms: `{summary['exact_positive_triggers']}`",
            f"- Weak near-racket or duplicate triggers within 250 ms: `{summary['weak_near_racket_or_duplicate_triggers']}`",
            f"- Unmatched negative/duplicate triggers: `{summary['unmatched_negative_or_duplicate_triggers']}`",
            "",
            "## Outputs",
            "",
            f"- Exact labels CSV: `{summary['exact_labels_csv']}`",
            f"- Manual nearest-trigger CSV: `{summary['manual_nearest_trigger_csv']}`",
            f"- Trigger match CSV: `{summary['trigger_match_csv']}`",
            "",
            "## Interpretation",
            "",
            "This file is held-out evaluation truth, not training data by itself. The 140 ms exact window mirrors the existing T0056/T0057 replay tolerance.",
            "",
        ]
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    labels_json = Path(args.labels_json)
    trigger_csv = Path(args.trigger_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload, labels = load_manual_labels(labels_json)
    triggers = load_triggers(trigger_csv)
    if not labels:
        raise ValueError(f"No manual racket labels found in {labels_json}")
    if not triggers:
        raise ValueError(f"No triggers found in {trigger_csv}")

    exact_labels_csv = out_dir / "t0063_exact_heldout_labels.csv"
    manual_nearest_csv = out_dir / "t0063_manual_nearest_trigger.csv"
    trigger_match_csv = out_dir / "t0063_trigger_label_matches.csv"
    summary_json = out_dir / "t0063_summary.json"
    report_md = out_dir / "t0063_report.md"

    exact_rows = [
        {
            "session_id": payload.get("session_id", SESSION_ID),
            "label_index": row["label_index"],
            "marker_id": row["marker_id"],
            "label": "racket",
            "review_label": "racket",
            "time_s": row["time_s"],
            "reviewed_time_s": row["reviewed_time_s"],
            "source": "t0062_manual_review",
        }
        for row in labels
    ]
    nearest_rows = nearest_trigger_rows(labels, triggers)
    trigger_rows = trigger_match_rows(labels, triggers)

    abs_deltas = [float(row["nearest_trigger_abs_delta_ms"]) for row in nearest_rows]
    status_counts: dict[str, int] = {}
    for row in trigger_rows:
        status = str(row["match_status"])
        status_counts[status] = status_counts.get(status, 0) + 1

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "session_id": payload.get("session_id", SESSION_ID),
        "manual_only": bool(payload.get("manual_only")),
        "expected_count": payload.get("expected_count"),
        "reported_app_count": payload.get("reported_app_count"),
        "labels_json": str(labels_json),
        "trigger_csv": str(trigger_csv),
        "manual_racket_labels": len(labels),
        "trigger_count": len(triggers),
        "first_label_s": labels[0]["time_s"],
        "last_label_s": labels[-1]["time_s"],
        "labels_within_60ms": sum(float(row["nearest_trigger_abs_delta_ms"]) <= 60.0 for row in nearest_rows),
        "labels_within_120ms": sum(float(row["nearest_trigger_abs_delta_ms"]) <= 120.0 for row in nearest_rows),
        "labels_within_140ms": sum(float(row["nearest_trigger_abs_delta_ms"]) <= EXACT_MATCH_WINDOW_MS for row in nearest_rows),
        "labels_within_250ms": sum(float(row["nearest_trigger_abs_delta_ms"]) <= NEAR_MATCH_WINDOW_MS for row in nearest_rows),
        "median_abs_nearest_trigger_delta_ms": round(statistics.median(abs_deltas), 3),
        "max_abs_nearest_trigger_delta_ms": round(max(abs_deltas), 3),
        "exact_positive_triggers": status_counts.get("exact_positive", 0),
        "weak_near_racket_or_duplicate_triggers": status_counts.get("weak_near_racket_or_duplicate", 0),
        "unmatched_negative_or_duplicate_triggers": status_counts.get("unmatched_negative_or_duplicate", 0),
        "exact_match_window_ms": EXACT_MATCH_WINDOW_MS,
        "near_match_window_ms": NEAR_MATCH_WINDOW_MS,
        "exact_labels_csv": str(exact_labels_csv),
        "manual_nearest_trigger_csv": str(manual_nearest_csv),
        "trigger_match_csv": str(trigger_match_csv),
        "report_md": str(report_md),
    }

    write_csv(exact_labels_csv, exact_rows)
    write_csv(manual_nearest_csv, nearest_rows)
    write_csv(trigger_match_csv, trigger_rows)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report_md.write_text(render_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-json", default=str(DEFAULT_LABELS_JSON))
    parser.add_argument("--trigger-csv", default=str(DEFAULT_TRIGGER_CSV))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
