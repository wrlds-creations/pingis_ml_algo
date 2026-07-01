#!/usr/bin/env python3
"""Prepare Round A peak-prefilled label review batches for T0071.

This is an evaluation/data-prep utility only. It creates review UI inputs from
the local Round A WAV manifest and preserves already reviewed T0066 background
labels. It does not train a model or change app behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_t0067_peak_gate_replay import (  # noqa: E402
    PeakGateConfig,
    detect_peak_gate,
    read_wav,
    write_csv,
)


DEFAULT_RAW_DIR = ROOT / "data/audio/raw/t0065_fable_training_audio_round_a/fable_training_audio"
DEFAULT_T0065_DIR = ROOT / "data/audio/models/evaluations/t0065_fable_training_audio_round_a"
DEFAULT_T0066_DIR = ROOT / "data/audio/models/evaluations/t0066_round_a_exact_label_review"
DEFAULT_OUT_DIR = ROOT / "data/audio/models/evaluations/t0071_round_a_scenario_label_expansion"

GATE_ID = "peak_fast_balanced"
PEAK_FAST_BALANCED = PeakGateConfig("raw_abs", 3.0, 220.0, 500.0, 60.0, 0.08, 2.0, 0.0)

TARGET_POSITIVE_SCENARIOS = {
    "normal_racket_bounce",
    "slow_high_racket_bounce",
    "fast_racket_bounce",
    "messy_kid_style_racket_bounce",
    "racket_bounce_speaking_counting",
}
BACKGROUND_SCENARIO = "racket_bounce_background_sound"
NEGATIVE_SCENARIOS = {
    "talking_only_no_bounce",
    "racket_handling_no_bounce",
    "floor_table_other_impact_no_racket",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def rounded(value: Any, digits: int = 6) -> float:
    number = safe_float(value, 0.0)
    if not math.isfinite(number):
        return 0.0
    return round(number, digits)


def app_counts_by_session(t0065_dir: Path) -> dict[str, dict[str, str]]:
    path = t0065_dir / "t0065_current_fable_block_replay.csv"
    if not path.exists():
        return {}
    return {row.get("session_id", ""): row for row in read_csv_rows(path)}


def existing_t0066_label_counts(t0066_dir: Path) -> dict[str, int]:
    labels_path = t0066_dir / "t0066_reviewed_background_positive_labels.csv"
    counts: dict[str, int] = defaultdict(int)
    if not labels_path.exists():
        return counts
    for row in read_csv_rows(labels_path):
        if row.get("label") in {"racket", "racket_bounce"}:
            counts[row.get("session_id", "")] += 1
    return counts


def expected_count_overrides(out_dir: Path) -> dict[str, dict[str, str]]:
    path = out_dir / "t0071_expected_count_overrides.csv"
    if not path.exists():
        return {}
    return {row.get("session_id", ""): row for row in read_csv_rows(path)}


def peak_score(row: dict[str, Any]) -> float:
    peak_value = safe_float(row.get("peak_value"), 0.0)
    ratio = safe_float(row.get("ratio"), 0.0)
    z = safe_float(row.get("z"), 0.0)
    return peak_value * (1.0 + 0.10 * math.log1p(max(0.0, ratio))) + 0.002 * max(0.0, z)


def normalize_peak_rows(events: list[dict[str, Any]], session_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, event in enumerate(sorted(events, key=lambda item: safe_float(item.get("time_ms"))), start=1):
        time_ms = safe_float(event.get("time_ms"))
        row = {
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
        rows.append(row)
    return rows


def select_draft_markers(rows: list[dict[str, Any]], expected_count: int) -> list[dict[str, Any]]:
    if expected_count <= 0:
        return []
    ranked = sorted(rows, key=lambda item: safe_float(item.get("selection_score")), reverse=True)
    selected = sorted(ranked[:expected_count], key=lambda item: safe_float(item.get("time_s")))
    return selected


def review_status(scenario_id: str, polarity: str, expected_count: int, peak_count: int) -> str:
    if scenario_id == BACKGROUND_SCENARIO:
        return "already_reviewed_in_t0066"
    if polarity == "negative":
        return "negative_peak_candidates_for_safety_summary"
    if expected_count <= 0:
        return "no_expected_positive_count"
    delta = peak_count - expected_count
    if delta == 0:
        return "auto_prefill_count_match_review_required"
    if abs(delta) <= 1:
        return "auto_prefill_near_match_review_required"
    if abs(delta) <= max(3, round(expected_count * 0.12)):
        return "auto_prefill_count_mismatch_review_required"
    return "manual_review_needed_count_mismatch"


def scenario_priority(scenario_id: str) -> int:
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


def make_review_payload(
    *,
    row: dict[str, str],
    raw_dir: Path,
    out_dir: Path,
    selected: list[dict[str, Any]],
    expected_count: int,
    reported_app_count: int,
    force_labels: bool,
) -> Path:
    session_id = row["session_id"]
    labels_path = out_dir / f"{session_id}_review_labels.json"
    if labels_path.exists() and not force_labels:
        payload = json.loads(labels_path.read_text(encoding="utf-8"))
        payload["expected_count"] = expected_count
        payload["reported_app_count"] = reported_app_count
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
        "source_wav": str(raw_dir / row["wav_file"]),
        "source_json": str(raw_dir / row["json_file"]),
        "expected_count": expected_count,
        "reported_app_count": reported_app_count,
        "manual_only": True,
        "trigger_labels": {},
        "manual_markers": markers,
        "saved_at": now,
        "review_note": (
            "T0071 auto-prefill from peak_fast_balanced. Verify these green markers, "
            "delete false ones, drag mistimed ones, and add missing racket contacts."
        ),
    }
    write_json(labels_path, payload)
    return labels_path


def powershell_command(
    *,
    session_id: str,
    raw_dir: Path,
    out_dir: Path,
    trigger_csv: Path,
    expected_count: int,
    reported_app_count: int,
    port: int,
) -> str:
    script = SCRIPT_DIR / "serve_t0053_trigger_review_ui.py"
    gate_note = (
        "Peak candidates are gray read-only lines. Green labels are draft racket contacts; "
        "review them, delete extras, drag mistimed labels, add missing contacts, then Save labels."
    )
    parts = [
        "python",
        f'"{script}"',
        f"--port {port}",
        f"--session-id {session_id}",
        f'--raw-dir "{raw_dir}"',
        f'--eval-dir "{out_dir}"',
        f'--out-dir "{out_dir}"',
        f'--trigger-csv "{trigger_csv}"',
        "--manual-only",
        f"--expected-count {expected_count}",
        f"--reported-app-count {reported_app_count}",
        f'--gate-note "{gate_note}"',
    ]
    return " ".join(parts)


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
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--t0065-dir", default=str(DEFAULT_T0065_DIR))
    parser.add_argument("--t0066-dir", default=str(DEFAULT_T0066_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--force-labels", action="store_true")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    t0065_dir = Path(args.t0065_dir)
    t0066_dir = Path(args.t0066_dir)
    out_dir = Path(args.out_dir)
    trigger_dir = out_dir / "trigger_csv"

    manifest_path = t0065_dir / "t0065_fable_training_audio_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    generated_at = datetime.now(timezone.utc).isoformat()
    manifest_rows = [row for row in read_csv_rows(manifest_path) if row.get("round") == "round_a"]
    app_rows = app_counts_by_session(t0065_dir)
    t0066_counts = existing_t0066_label_counts(t0066_dir)
    expected_overrides = expected_count_overrides(out_dir)

    all_peak_rows: list[dict[str, Any]] = []
    selected_label_rows: list[dict[str, Any]] = []
    review_manifest: list[dict[str, Any]] = []
    scenario_totals: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "scenario_title": "",
            "polarity": "",
            "clips": 0,
            "expected": 0,
            "peak_candidates": 0,
            "draft_or_reviewed_labels": 0,
            "app_count": 0,
            "statuses": defaultdict(int),
        }
    )

    for row in sorted(
        manifest_rows,
        key=lambda item: (scenario_priority(item.get("scenario_id", "")), item.get("started_at", "")),
    ):
        session_id = row["session_id"]
        scenario_id = row.get("scenario_id", "")
        polarity = row.get("polarity", "")
        original_expected_count = safe_int(row.get("expected_racket_contacts"))
        override = expected_overrides.get(session_id, {})
        expected_count = safe_int(override.get("expected_racket_contacts"), original_expected_count)
        override_note = override.get("note", "")
        wav_path = raw_dir / row["wav_file"]
        y, sr = read_wav(wav_path)
        peaks = normalize_peak_rows(detect_peak_gate(y, sr, PEAK_FAST_BALANCED), session_id)

        trigger_csv = trigger_dir / f"{session_id}_{GATE_ID}_triggers.csv"
        write_csv(trigger_csv, peaks)
        all_peak_rows.extend(peaks)

        app_count = safe_int(app_rows.get(session_id, {}).get("counted"))
        status = review_status(scenario_id, polarity, expected_count, len(peaks))
        labels_path = ""
        existing_labels_path = ""
        draft_count = 0

        if scenario_id == BACKGROUND_SCENARIO:
            existing_json = t0066_dir / f"{session_id}_review_labels.json"
            existing_labels_path = str(existing_json) if existing_json.exists() else ""
            draft_count = t0066_counts.get(session_id, 0)
            for label_index in range(1, draft_count + 1):
                selected_label_rows.append(
                    {
                        "session_id": session_id,
                        "scenario_id": scenario_id,
                        "scenario_title": row.get("scenario_title", ""),
                        "label_index": label_index,
                        "label": "racket_bounce",
                        "label_status": "reviewed_exact_t0066",
                        "source": "t0066_reviewed_background_positive_labels",
                    }
                )
        elif scenario_id in TARGET_POSITIVE_SCENARIOS:
            selected = select_draft_markers(peaks, expected_count)
            draft_count = len(selected)
            labels_path = str(
                make_review_payload(
                    row=row,
                    raw_dir=raw_dir,
                    out_dir=out_dir,
                    selected=selected,
                    expected_count=expected_count,
                    reported_app_count=app_count,
                    force_labels=args.force_labels,
                )
            )
            for label_index, candidate in enumerate(selected, start=1):
                selected_label_rows.append(
                    {
                        "session_id": session_id,
                        "scenario_id": scenario_id,
                        "scenario_title": row.get("scenario_title", ""),
                        "label_index": label_index,
                        "label": "racket_bounce",
                        "time_s": rounded(candidate.get("time_s"), 6),
                        "time_ms": rounded(safe_float(candidate.get("time_s")) * 1000.0, 3),
                        "label_status": "draft_review_required",
                        "source": "auto_waveform_peak_prefill",
                        "gate_id": GATE_ID,
                        "source_candidate_index": candidate.get("candidate_index", ""),
                        "peak_value": candidate.get("peak_value", ""),
                        "selection_score": candidate.get("selection_score", ""),
                    }
                )

        launch_command = ""
        if scenario_id in TARGET_POSITIVE_SCENARIOS:
            launch_command = powershell_command(
                session_id=session_id,
                raw_dir=raw_dir,
                out_dir=out_dir,
                trigger_csv=trigger_csv,
                expected_count=expected_count,
                reported_app_count=app_count,
                port=args.port,
            )

        count_delta = len(peaks) - expected_count if polarity == "positive" else len(peaks)
        manifest_out = {
            "review_priority": scenario_priority(scenario_id),
            "session_id": session_id,
            "started_at": row.get("started_at", ""),
            "scenario_id": scenario_id,
            "scenario_title": row.get("scenario_title", ""),
            "polarity": polarity,
            "original_expected_racket_contacts": original_expected_count,
            "expected_racket_contacts": expected_count,
            "expected_count_override_note": override_note,
            "current_app_count": app_count,
            "peak_candidate_count": len(peaks),
            "draft_or_reviewed_label_count": draft_count,
            "peak_minus_expected": count_delta,
            "status": status,
            "wav_duration_s": row.get("wav_duration_s", ""),
            "trigger_csv": str(trigger_csv),
            "review_labels_json": labels_path,
            "existing_t0066_labels_json": existing_labels_path,
            "review_command": launch_command,
        }
        review_manifest.append(manifest_out)

        totals = scenario_totals[scenario_id]
        totals["scenario_title"] = row.get("scenario_title", "")
        totals["polarity"] = polarity
        totals["clips"] += 1
        totals["expected"] += expected_count
        totals["peak_candidates"] += len(peaks)
        totals["draft_or_reviewed_labels"] += draft_count
        totals["app_count"] += app_count
        totals["statuses"][status] += 1

    scenario_rows: list[dict[str, Any]] = []
    for scenario_id, totals in sorted(scenario_totals.items(), key=lambda item: scenario_priority(item[0])):
        scenario_rows.append(
            {
                "scenario_id": scenario_id,
                "scenario_title": totals["scenario_title"],
                "polarity": totals["polarity"],
                "clips": totals["clips"],
                "expected": totals["expected"],
                "current_app_count": totals["app_count"],
                "peak_candidates": totals["peak_candidates"],
                "draft_or_reviewed_labels": totals["draft_or_reviewed_labels"],
                "peak_minus_expected": totals["peak_candidates"] - totals["expected"],
                "statuses": json.dumps(dict(totals["statuses"]), sort_keys=True),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "t0071_review_manifest.csv", review_manifest)
    write_csv(out_dir / "t0071_peak_candidate_rows.csv", all_peak_rows)
    write_csv(out_dir / "t0071_candidate_label_inputs.csv", selected_label_rows)
    write_csv(out_dir / "t0071_by_scenario.csv", scenario_rows)

    recommended = next(
        (
            row
            for row in review_manifest
            if row["scenario_id"] in TARGET_POSITIVE_SCENARIOS
            and row["status"] == "auto_prefill_count_match_review_required"
        ),
        next((row for row in review_manifest if row["scenario_id"] in TARGET_POSITIVE_SCENARIOS), None),
    )

    summary = {
        "generated_at": generated_at,
        "ticket": "T0071-round-a-scenario-label-expansion",
        "gate_id": GATE_ID,
        "gate_config": {
            "envelope_mode": PEAK_FAST_BALANCED.envelope_mode,
            "smooth_ms": PEAK_FAST_BALANCED.smooth_ms,
            "min_gap_ms": PEAK_FAST_BALANCED.min_gap_ms,
            "bg_ms": PEAK_FAST_BALANCED.bg_ms,
            "bg_exclude_ms": PEAK_FAST_BALANCED.bg_exclude_ms,
            "abs_min": PEAK_FAST_BALANCED.abs_min,
            "ratio_min": PEAK_FAST_BALANCED.ratio_min,
            "z_min": PEAK_FAST_BALANCED.z_min,
        },
        "clips": len(review_manifest),
        "peak_candidates": len(all_peak_rows),
        "draft_or_reviewed_labels": len(selected_label_rows),
        "recommended_session_id": recommended.get("session_id") if recommended else "",
        "recommended_review_command": recommended.get("review_command") if recommended else "",
    }
    write_json(out_dir / "t0071_summary.json", summary)

    report_lines = [
        "# T0071 Round A Scenario Label Expansion",
        "",
        f"Generated: {generated_at}",
        "",
        "This prep pass uses `peak_fast_balanced` to create gray trigger candidates and green draft racket-contact labels for review.",
        "It preserves the already reviewed T0066 background-sound labels and does not train or ship a model.",
        "",
        "## Scenario Summary",
        "",
        *md_table(
            scenario_rows,
            [
                "scenario_id",
                "clips",
                "expected",
                "current_app_count",
                "peak_candidates",
                "draft_or_reviewed_labels",
                "peak_minus_expected",
            ],
            ["scenario", "clips", "expected", "app", "peaks", "labels", "peak-expected"],
        ),
        "",
        "## Recommended First Review",
        "",
    ]
    if recommended:
        report_lines.extend(
            [
                f"- Session: `{recommended['session_id']}`",
                f"- Scenario: {recommended['scenario_title']}",
                f"- Expected contacts: {recommended['expected_racket_contacts']}",
                f"- Peak candidates: {recommended['peak_candidate_count']}",
                f"- Draft labels: {recommended['draft_or_reviewed_label_count']}",
                "",
                "Run:",
                "",
                "```powershell",
                recommended["review_command"],
                "```",
                "",
                f"Then open: `http://127.0.0.1:{args.port}`",
            ]
        )
    else:
        report_lines.append("No positive review session found.")

    report_lines.extend(
        [
            "",
            "## Files",
            "",
            "- `t0071_review_manifest.csv`: one row per Round A clip, including review commands.",
            "- `t0071_peak_candidate_rows.csv`: every peak candidate, used as gray read-only UI markers.",
            "- `t0071_candidate_label_inputs.csv`: draft/reviewed racket-contact label rows.",
            "- `t0071_by_scenario.csv`: scenario-level counts.",
            "- `*_review_labels.json`: editable review UI label state for unreviewed positive clips.",
            "",
            "## Review Rule",
            "",
            "Use the green labels as a draft only. Delete false labels, drag labels onto the exact bounce peak, add missing contacts, and save.",
        ]
    )
    (out_dir / "t0071_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
