"""
Tune T0015 playing-retro recovery visibility thresholds.

This script reads the T0014 recovery prediction rows and sweeps visibility gates
for racket/table recovery candidates. It does not train, export, build an APK,
or change Collector live audio.

Run:
  python skills/pingis-audio-classification/scripts/tune_playing_retro_recovery_thresholds.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from build_playing_retro_candidate_report import MATCH_TOLERANCE_MS
from replay_playing_retro_recovery_candidates import (
    PREDICTIONS_CSV as T0014_PREDICTIONS_CSV,
    SUMMARY_CSV as T0014_SUMMARY_CSV,
    main as run_t0014_replay,
)
from train_playing_retro_audio import EVAL_DIR

OUT_CSV = EVAL_DIR / "playing_retro_audio_t0015_threshold_sweep.csv"
REPORT_MD = EVAL_DIR / "playing_retro_audio_t0015_threshold_gate_report.md"

CURRENT_RACKET_CONFIDENCE = 0.80
CURRENT_RACKET_GAP_MS = 120
CURRENT_TABLE_CONFIDENCE = 0.54
CURRENT_TABLE_GAP_MS = 60

RACKET_CONFIDENCE_GRID = [0.54, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
RACKET_GAP_GRID = [32, 40, 60, 80, 100, 120, 140]
TABLE_CONFIDENCE_GRID = [0.50, 0.54, 0.58, 0.62, 0.66, 0.70]
TABLE_GAP_GRID = [40, 50, 60, 70, 80, 100]

MAX_WRONG_CLASS = 0
MAX_DUPLICATE_NEAR_BASELINE = 0
MAX_VISIBLE_FALSE_POSITIVE = 0
MIN_RECOVERED_FOR_APK_GATE = 6


@dataclass(frozen=True)
class Gate:
    racket_confidence: float
    racket_gap_ms: int
    table_confidence: float
    table_gap_ms: int

    @property
    def name(self) -> str:
        return (
            f"r{self.racket_confidence:.2f}_rgap{self.racket_gap_ms}_"
            f"t{self.table_confidence:.2f}_tgap{self.table_gap_ms}"
        )

    @property
    def is_current(self) -> bool:
        return (
            abs(self.racket_confidence - CURRENT_RACKET_CONFIDENCE) < 1e-9
            and self.racket_gap_ms == CURRENT_RACKET_GAP_MS
            and abs(self.table_confidence - CURRENT_TABLE_CONFIDENCE) < 1e-9
            and self.table_gap_ms == CURRENT_TABLE_GAP_MS
        )


def ensure_t0014_predictions() -> None:
    if T0014_PREDICTIONS_CSV.exists() and T0014_SUMMARY_CSV.exists():
        return
    run_t0014_replay()


def visible_for_gate(row: dict[str, Any], gate: Gate) -> bool:
    prediction = str(row.get("prediction") or "")
    confidence = float(row.get("confidence") or 0)
    nearest_saved_gap = row.get("nearest_saved_gap_ms")
    if nearest_saved_gap == "" or pd.isna(nearest_saved_gap):
        gap_ms = 999999
    else:
        gap_ms = int(float(nearest_saved_gap))
    if prediction == "racket_contact":
        return confidence >= gate.racket_confidence and gap_ms >= gate.racket_gap_ms
    if prediction == "table_bounce":
        return confidence >= gate.table_confidence and gap_ms >= gate.table_gap_ms
    return False


def evaluate_gate(predictions_df: pd.DataFrame, gate: Gate) -> dict[str, Any]:
    visible_rows = [
        row
        for row in predictions_df.to_dict("records")
        if visible_for_gate(row, gate)
    ]
    matched_recovery_truths: set[str] = set()
    recovered_correct = 0
    recovered_racket = 0
    recovered_table = 0
    wrong_class_near_missed = 0
    duplicate_near_baseline_truth = 0
    false_positive_visible = 0

    def sort_key(row: dict[str, Any]) -> int:
        delta = row.get("nearest_truth_delta_ms")
        if delta == "" or pd.isna(delta):
            return 999999
        return abs(int(float(delta)))

    for row in sorted(visible_rows, key=sort_key):
        truth_id = str(row.get("nearest_truth_id") or "")
        truth_kind = str(row.get("nearest_truth_kind") or "")
        prediction = str(row.get("prediction") or "")
        delta_value = row.get("nearest_truth_delta_ms")
        near_truth = (
            delta_value != ""
            and not pd.isna(delta_value)
            and abs(int(float(delta_value))) <= MATCH_TOLERANCE_MS
        )
        if not near_truth:
            false_positive_visible += 1
            continue
        if str(row.get("nearest_truth_was_baseline_matched") or "").lower() == "true":
            duplicate_near_baseline_truth += 1
            continue
        if truth_id in matched_recovery_truths:
            false_positive_visible += 1
            continue
        matched_recovery_truths.add(truth_id)
        if prediction == truth_kind:
            recovered_correct += 1
            if truth_kind == "racket_contact":
                recovered_racket += 1
            elif truth_kind == "table_bounce":
                recovered_table += 1
        else:
            wrong_class_near_missed += 1

    passes_gate = (
        wrong_class_near_missed <= MAX_WRONG_CLASS
        and duplicate_near_baseline_truth <= MAX_DUPLICATE_NEAR_BASELINE
        and false_positive_visible <= MAX_VISIBLE_FALSE_POSITIVE
        and recovered_correct >= MIN_RECOVERED_FOR_APK_GATE
    )

    return {
        "gate": gate.name,
        "racket_confidence": gate.racket_confidence,
        "racket_gap_ms": gate.racket_gap_ms,
        "table_confidence": gate.table_confidence,
        "table_gap_ms": gate.table_gap_ms,
        "is_current": gate.is_current,
        "visible_recovery_candidates": len(visible_rows),
        "recovered_correct": recovered_correct,
        "recovered_racket": recovered_racket,
        "recovered_table": recovered_table,
        "wrong_class_near_missed": wrong_class_near_missed,
        "duplicate_near_baseline_truth": duplicate_near_baseline_truth,
        "false_positive_visible": false_positive_visible,
        "passes_t0016_gate": passes_gate,
    }


def gate_sort_key(row: dict[str, Any]) -> tuple[int, int, int, float, float, int, int]:
    return (
        int(row["passes_t0016_gate"]),
        int(row["recovered_correct"]),
        -int(row["wrong_class_near_missed"]),
        -int(row["duplicate_near_baseline_truth"]),
        -int(row["false_positive_visible"]),
        -int(row["visible_recovery_candidates"]),
        int(row["is_current"]),
    )


def write_report(
    report_path: Path,
    sweep_df: pd.DataFrame,
    selected: dict[str, Any],
    current: dict[str, Any],
    baseline_missed: int,
    baseline_missed_racket: int,
    baseline_missed_table: int,
) -> None:
    top_rows = sweep_df.sort_values(
        by=[
            "passes_t0016_gate",
            "recovered_correct",
            "wrong_class_near_missed",
            "duplicate_near_baseline_truth",
            "false_positive_visible",
            "visible_recovery_candidates",
            "is_current",
        ],
        ascending=[False, False, True, True, True, True, False],
    ).head(12)

    lines = [
        "# Playing Retro Audio T0015 Threshold Gate Report",
        "",
        "This report tunes visibility gates for T0014 recovery candidates. It does not train, export, build an APK, or change `studs_live`.",
        "",
        "## Pass Gate",
        "",
        f"- Wrong-class near missed truth must be <= `{MAX_WRONG_CLASS}`",
        f"- Duplicate near already matched truth must be <= `{MAX_DUPLICATE_NEAR_BASELINE}`",
        f"- Visible false positives must be <= `{MAX_VISIBLE_FALSE_POSITIVE}`",
        f"- Correct recovered missed truths must be >= `{MIN_RECOVERED_FOR_APK_GATE}`",
        "",
        "## Dataset",
        "",
        f"- Baseline missed truths: `{baseline_missed}`",
        f"- Baseline missed racket/table: `{baseline_missed_racket}` / `{baseline_missed_table}`",
        f"- Threshold rows tested: `{len(sweep_df)}`",
        "",
        "## Selected Gate",
        "",
        f"- Gate: `{selected['gate']}`",
        f"- Racket: confidence >= `{selected['racket_confidence']:.2f}`, nearest saved gap >= `{selected['racket_gap_ms']} ms`",
        f"- Table: confidence >= `{selected['table_confidence']:.2f}`, nearest saved gap >= `{selected['table_gap_ms']} ms`",
        f"- Recovered correct: `{selected['recovered_correct']}` (`{selected['recovered_racket']}` racket / `{selected['recovered_table']}` table)",
        f"- Wrong-class / duplicate / visible FP: `{selected['wrong_class_near_missed']}` / `{selected['duplicate_near_baseline_truth']}` / `{selected['false_positive_visible']}`",
        f"- Passes T0016 replay gate: `{bool(selected['passes_t0016_gate'])}`",
        "",
        "## Current T0014 Gate",
        "",
        f"- Gate: `{current['gate']}`",
        f"- Recovered correct: `{current['recovered_correct']}` (`{current['recovered_racket']}` racket / `{current['recovered_table']}` table)",
        f"- Wrong-class / duplicate / visible FP: `{current['wrong_class_near_missed']}` / `{current['duplicate_near_baseline_truth']}` / `{current['false_positive_visible']}`",
        f"- Passes T0016 replay gate: `{bool(current['passes_t0016_gate'])}`",
        "",
        "## Top Candidates",
        "",
        "| Gate | Pass | Visible | Recovered | Racket | Table | Wrong | Duplicate | FP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top_rows.to_dict("records"):
        lines.append(
            "| {gate} | {passes_t0016_gate} | {visible_recovery_candidates} | {recovered_correct} | {recovered_racket} | {recovered_table} | {wrong_class_near_missed} | {duplicate_near_baseline_truth} | {false_positive_visible} |".format(**row)
        )
    lines.extend([
        "",
        "## Outputs",
        "",
        f"- Sweep CSV: `{OUT_CSV.as_posix()}`",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_t0014_predictions()
    predictions_df = pd.read_csv(T0014_PREDICTIONS_CSV)
    summary_df = pd.read_csv(T0014_SUMMARY_CSV)
    baseline_missed = int(summary_df["baseline_missed_truths"].sum())
    baseline_missed_racket = int(summary_df["baseline_missed_racket"].sum())
    baseline_missed_table = int(summary_df["baseline_missed_table"].sum())

    rows: list[dict[str, Any]] = []
    for racket_confidence in RACKET_CONFIDENCE_GRID:
        for racket_gap_ms in RACKET_GAP_GRID:
            for table_confidence in TABLE_CONFIDENCE_GRID:
                for table_gap_ms in TABLE_GAP_GRID:
                    rows.append(evaluate_gate(
                        predictions_df,
                        Gate(
                            racket_confidence=racket_confidence,
                            racket_gap_ms=racket_gap_ms,
                            table_confidence=table_confidence,
                            table_gap_ms=table_gap_ms,
                        ),
                    ))

    sweep_df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(OUT_CSV, index=False)

    selected = max(rows, key=gate_sort_key)
    current_candidates = [row for row in rows if row["is_current"]]
    if not current_candidates:
        raise RuntimeError("Current T0014 gate was not included in the threshold grid.")
    current = current_candidates[0]
    write_report(
        REPORT_MD,
        sweep_df,
        selected,
        current,
        baseline_missed,
        baseline_missed_racket,
        baseline_missed_table,
    )
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {REPORT_MD}")
    print(
        "selected={gate} pass={passed} recovered={recovered} racket={racket} table={table} wrong={wrong} duplicate={dup} fp={fp}".format(
            gate=selected["gate"],
            passed=selected["passes_t0016_gate"],
            recovered=selected["recovered_correct"],
            racket=selected["recovered_racket"],
            table=selected["recovered_table"],
            wrong=selected["wrong_class_near_missed"],
            dup=selected["duplicate_near_baseline_truth"],
            fp=selected["false_positive_visible"],
        )
    )
    print(
        "current={gate} pass={passed} recovered={recovered} racket={racket} table={table} wrong={wrong} duplicate={dup} fp={fp}".format(
            gate=current["gate"],
            passed=current["passes_t0016_gate"],
            recovered=current["recovered_correct"],
            racket=current["recovered_racket"],
            table=current["recovered_table"],
            wrong=current["wrong_class_near_missed"],
            dup=current["duplicate_near_baseline_truth"],
            fp=current["false_positive_visible"],
        )
    )


if __name__ == "__main__":
    main()
