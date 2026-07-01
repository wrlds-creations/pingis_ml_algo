#!/usr/bin/env python3
"""Ingest T0071 Round A reviewed labels into replay-ready CSV artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[3]

DEFAULT_T0065_DIR = ROOT / "data/audio/models/evaluations/t0065_fable_training_audio_round_a"
DEFAULT_T0066_DIR = ROOT / "data/audio/models/evaluations/t0066_round_a_exact_label_review"
DEFAULT_T0071_DIR = ROOT / "data/audio/models/evaluations/t0071_round_a_scenario_label_expansion"

BACKGROUND_SCENARIO = "racket_bounce_background_sound"
POSITIVE_SCENARIOS = {
    "normal_racket_bounce",
    "slow_high_racket_bounce",
    "fast_racket_bounce",
    "messy_kid_style_racket_bounce",
    "racket_bounce_speaking_counting",
    BACKGROUND_SCENARIO,
}
NEGATIVE_SCENARIOS = {
    "talking_only_no_bounce",
    "racket_handling_no_bounce",
    "floor_table_other_impact_no_racket",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
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


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def scenario_order(scenario_id: str) -> int:
    order = [
        "normal_racket_bounce",
        "slow_high_racket_bounce",
        "fast_racket_bounce",
        "messy_kid_style_racket_bounce",
        "racket_bounce_speaking_counting",
        "racket_bounce_background_sound",
        "talking_only_no_bounce",
        "racket_handling_no_bounce",
        "floor_table_other_impact_no_racket",
    ]
    try:
        return order.index(scenario_id) + 1
    except ValueError:
        return 999


def manifest_meta(t0065_dir: Path) -> dict[str, dict[str, str]]:
    manifest_path = t0065_dir / "t0065_fable_training_audio_manifest.csv"
    return {row.get("session_id", ""): row for row in read_csv_rows(manifest_path)}


def load_review_markers(path: Path) -> tuple[list[dict[str, Any]], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    markers: list[dict[str, Any]] = []
    for marker in payload.get("manual_markers") or []:
        label = str(marker.get("label") or "")
        if label not in {"racket", "racket_bounce"}:
            continue
        markers.append(marker)
    markers.sort(key=lambda item: safe_float(item.get("time_s")))
    return markers, str(payload.get("saved_at") or "")


def load_t0066_background_labels(t0066_dir: Path) -> dict[str, list[dict[str, str]]]:
    labels_path = t0066_dir / "t0066_reviewed_background_positive_labels.csv"
    labels: dict[str, list[dict[str, str]]] = defaultdict(list)
    if not labels_path.exists():
        return labels
    for row in read_csv_rows(labels_path):
        if row.get("label") in {"racket", "racket_bounce"}:
            labels[row.get("session_id", "")].append(row)
    for rows in labels.values():
        rows.sort(key=lambda item: safe_float(item.get("reviewed_time_s")))
    return labels


def nearest_peak_delta_ms(label_s: float, peak_rows: list[dict[str, str]]) -> tuple[str, str, str]:
    if not peak_rows:
        return "", "", ""
    best = min(
        peak_rows,
        key=lambda row: abs(safe_float(row.get("time_s")) - label_s),
    )
    peak_s = safe_float(best.get("time_s"))
    return (
        str(best.get("candidate_index") or best.get("trigger_index") or ""),
        f"{peak_s:.6f}",
        f"{(peak_s - label_s) * 1000.0:.3f}",
    )


def md_table(rows: list[dict[str, Any]], fields: list[str], labels: list[str] | None = None) -> list[str]:
    labels = labels or fields
    lines = [
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join("---" for _ in labels) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t0065-dir", default=str(DEFAULT_T0065_DIR))
    parser.add_argument("--t0066-dir", default=str(DEFAULT_T0066_DIR))
    parser.add_argument("--t0071-dir", default=str(DEFAULT_T0071_DIR))
    args = parser.parse_args()

    t0065_dir = Path(args.t0065_dir)
    t0066_dir = Path(args.t0066_dir)
    t0071_dir = Path(args.t0071_dir)
    generated_at = datetime.now(timezone.utc).isoformat()

    meta_by_session = manifest_meta(t0065_dir)
    review_manifest = read_csv_rows(t0071_dir / "t0071_review_manifest.csv")
    peak_rows = read_csv_rows(t0071_dir / "t0071_peak_candidate_rows.csv")
    peaks_by_session: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in peak_rows:
        peaks_by_session[row.get("session_id", "")].append(row)
    for rows in peaks_by_session.values():
        rows.sort(key=lambda item: safe_float(item.get("time_s")))

    background_labels = load_t0066_background_labels(t0066_dir)

    positive_label_rows: list[dict[str, Any]] = []
    clip_summary_rows: list[dict[str, Any]] = []
    scenario_totals: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "scenario_title": "",
            "polarity": "",
            "clips": 0,
            "expected": 0,
            "reviewed_labels": 0,
            "peak_candidates": 0,
            "mismatched_clips": 0,
        }
    )
    mismatches: list[dict[str, Any]] = []

    for row in sorted(
        review_manifest,
        key=lambda item: (scenario_order(item.get("scenario_id", "")), item.get("started_at", "")),
    ):
        session_id = row.get("session_id", "")
        scenario_id = row.get("scenario_id", "")
        polarity = row.get("polarity", "")
        scenario_title = row.get("scenario_title", "")
        expected = safe_int(row.get("expected_racket_contacts"))
        original_expected = safe_int(row.get("original_expected_racket_contacts"), expected)
        peak_count = safe_int(row.get("peak_candidate_count"))
        reviewed_count = 0
        saved_at = ""

        if polarity == "positive" and scenario_id in POSITIVE_SCENARIOS:
            if scenario_id == BACKGROUND_SCENARIO:
                labels = background_labels.get(session_id, [])
                reviewed_count = len(labels)
                for index, label in enumerate(labels, start=1):
                    label_s = safe_float(label.get("reviewed_time_s"))
                    nearest_index, nearest_s, delta_ms = nearest_peak_delta_ms(label_s, peaks_by_session[session_id])
                    positive_label_rows.append(
                        {
                            "session_id": session_id,
                            "scenario_id": scenario_id,
                            "scenario_title": scenario_title,
                            "label_index": index,
                            "label": "racket_bounce",
                            "reviewed_time_s": f"{label_s:.6f}",
                            "reviewed_time_ms": f"{label_s * 1000.0:.3f}",
                            "source": "t0066_reviewed_background_positive_labels",
                            "source_label_id": label.get("source_label_id", ""),
                            "note": label.get("note", ""),
                            "nearest_peak_candidate_index": nearest_index,
                            "nearest_peak_time_s": nearest_s,
                            "nearest_peak_delta_ms": delta_ms,
                            "expected_racket_contacts": expected,
                            "original_expected_racket_contacts": original_expected,
                        }
                    )
            else:
                labels_path = Path(row.get("review_labels_json", ""))
                if labels_path.exists():
                    markers, saved_at = load_review_markers(labels_path)
                else:
                    markers = []
                    mismatches.append(
                        {
                            "session_id": session_id,
                            "scenario_id": scenario_id,
                            "issue": "missing_review_json",
                            "path": str(labels_path),
                        }
                    )
                reviewed_count = len(markers)
                for index, marker in enumerate(markers, start=1):
                    label_s = safe_float(marker.get("time_s"))
                    nearest_index, nearest_s, delta_ms = nearest_peak_delta_ms(label_s, peaks_by_session[session_id])
                    positive_label_rows.append(
                        {
                            "session_id": session_id,
                            "scenario_id": scenario_id,
                            "scenario_title": scenario_title,
                            "label_index": index,
                            "label": "racket_bounce",
                            "reviewed_time_s": f"{label_s:.6f}",
                            "reviewed_time_ms": f"{label_s * 1000.0:.3f}",
                            "source": marker.get("source", "manual_review"),
                            "source_label_id": marker.get("id", ""),
                            "note": marker.get("note", ""),
                            "nearest_peak_candidate_index": nearest_index,
                            "nearest_peak_time_s": nearest_s,
                            "nearest_peak_delta_ms": delta_ms,
                            "expected_racket_contacts": expected,
                            "original_expected_racket_contacts": original_expected,
                        }
                    )

        if reviewed_count != expected and polarity == "positive":
            mismatches.append(
                {
                    "session_id": session_id,
                    "scenario_id": scenario_id,
                    "issue": "reviewed_count_differs_from_expected",
                    "expected": expected,
                    "reviewed_count": reviewed_count,
                }
            )

        if polarity == "positive":
            clip_summary_rows.append(
                {
                    "session_id": session_id,
                    "scenario_id": scenario_id,
                    "scenario_title": scenario_title,
                    "expected_racket_contacts": expected,
                    "original_expected_racket_contacts": original_expected,
                    "reviewed_racket_labels": reviewed_count,
                    "peak_candidate_count": peak_count,
                    "current_app_count": safe_int(row.get("current_app_count")),
                    "reviewed_minus_expected": reviewed_count - expected,
                    "peak_minus_reviewed": peak_count - reviewed_count,
                    "saved_at": saved_at,
                    "wav_file": meta_by_session.get(session_id, {}).get("wav_file", ""),
                }
            )

        totals = scenario_totals[scenario_id]
        totals["scenario_title"] = scenario_title
        totals["polarity"] = polarity
        totals["clips"] += 1
        totals["expected"] += expected
        totals["reviewed_labels"] += reviewed_count
        totals["peak_candidates"] += peak_count
        if polarity == "positive" and reviewed_count != expected:
            totals["mismatched_clips"] += 1

    hard_negative_rows: list[dict[str, Any]] = []
    for row in peak_rows:
        session_id = row.get("session_id", "")
        meta = meta_by_session.get(session_id, {})
        scenario_id = meta.get("scenario_id", "")
        if scenario_id not in NEGATIVE_SCENARIOS:
            continue
        hard_negative_rows.append(
            {
                **row,
                "scenario_id": scenario_id,
                "scenario_title": meta.get("scenario_title", ""),
                "label": "non_racket_negative",
                "label_source": "round_a_negative_clip_peak_candidate",
            }
        )

    scenario_rows: list[dict[str, Any]] = []
    for scenario_id, totals in sorted(scenario_totals.items(), key=lambda item: scenario_order(item[0])):
        scenario_rows.append(
            {
                "scenario_id": scenario_id,
                "scenario_title": totals["scenario_title"],
                "polarity": totals["polarity"],
                "clips": totals["clips"],
                "expected": totals["expected"],
                "reviewed_labels": totals["reviewed_labels"],
                "peak_candidates": totals["peak_candidates"],
                "reviewed_minus_expected": totals["reviewed_labels"] - totals["expected"],
                "peak_minus_reviewed": totals["peak_candidates"] - totals["reviewed_labels"],
                "mismatched_clips": totals["mismatched_clips"],
            }
        )

    summary = {
        "generated_at": generated_at,
        "ticket": "T0071-round-a-scenario-label-expansion",
        "positive_reviewed_labels": len(positive_label_rows),
        "positive_clips": len(clip_summary_rows),
        "hard_negative_peak_candidates": len(hard_negative_rows),
        "mismatches": len(mismatches),
    }
    report_lines = [
        "# T0071 Reviewed Label Ingest",
        "",
        f"Generated: {generated_at}",
        "",
        "## Scenario Summary",
        "",
        *md_table(
            scenario_rows,
            [
                "scenario_id",
                "clips",
                "expected",
                "reviewed_labels",
                "peak_candidates",
                "reviewed_minus_expected",
                "peak_minus_reviewed",
                "mismatched_clips",
            ],
            ["scenario", "clips", "expected", "labels", "peaks", "labels-expected", "peaks-labels", "mismatch"],
        ),
        "",
        "## Outputs",
        "",
        "- `t0071_reviewed_positive_labels.csv`: exact reviewed racket contacts for all positive Round A clips.",
        "- `t0071_reviewed_positive_clip_summary.csv`: per-positive-clip checks.",
        "- `t0071_hard_negative_peak_candidates.csv`: all peak candidates from expected-zero Round A clips.",
        "- `t0071_reviewed_label_ingest_summary.json`: machine-readable totals.",
        "",
    ]
    if mismatches:
        report_lines.extend(["## Mismatches", ""])
        report_lines.extend(
            md_table(
                mismatches,
                ["session_id", "scenario_id", "issue", "expected", "reviewed_count", "path"],
            )
        )
    else:
        report_lines.extend(["## Mismatches", "", "None."])

    write_csv(t0071_dir / "t0071_reviewed_positive_labels.csv", positive_label_rows)
    write_csv(t0071_dir / "t0071_reviewed_positive_clip_summary.csv", clip_summary_rows)
    write_csv(t0071_dir / "t0071_hard_negative_peak_candidates.csv", hard_negative_rows)
    write_csv(t0071_dir / "t0071_reviewed_label_ingest_by_scenario.csv", scenario_rows)
    write_csv(t0071_dir / "t0071_reviewed_label_ingest_mismatches.csv", mismatches)
    write_json(t0071_dir / "t0071_reviewed_label_ingest_summary.json", summary)
    (t0071_dir / "t0071_reviewed_label_ingest_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
