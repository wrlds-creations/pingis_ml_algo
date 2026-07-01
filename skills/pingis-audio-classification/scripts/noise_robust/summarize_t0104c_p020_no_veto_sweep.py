#!/usr/bin/env python3
"""Summarize the T0104C p>=0.20/no-veto safety sweep.

This report is evaluation-only. It reads already generated/pulled T0103 and
T0104 artifacts and writes ignored local report files. It does not train,
export, change app thresholds, or install an APK.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_T0104_CSV = (
    ROOT
    / "data/audio/models/evaluations/t0104_bounce_audio_test_live_validation/t0104_session_summary.csv"
)
DEFAULT_T0103_SWEEP = (
    ROOT
    / "data/audio/models/evaluations/t0103_boundary_label_candidate_phone_gate/candidate_loop_2026_07_01/t0103_policy_sweep.csv"
)
DEFAULT_OUT_DIR = ROOT / "data/audio/models/evaluations/t0104c_p020_no_veto_safety_sweep"
THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.575]
T0103_POLICY_FILTER = {
    "candidate_model_id": "extra_leaf4_t0103",
    "feature_set_id": "base_t0075",
    "weight_strategy_id": "boundary_recall_safety",
    "dedupe_ms": "180",
    "fable_noise_veto_threshold": "",
}


@dataclass(frozen=True)
class ThresholdTotals:
    threshold: float
    positive_count: int
    positive_expected: int
    negative_count: int
    negative_expected: int


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def as_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    return int(float(value))


def as_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def threshold_key(threshold: float) -> str:
    return f"p>={threshold:g}"


def summarize_t0104_live(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[ThresholdTotals]]:
    included = [row for row in rows if str(row.get("include_in_metrics", "True")).lower() != "false"]
    scenario_rows: dict[str, dict[str, Any]] = {}
    for row in included:
        scenario = row.get("scenario_title") or row.get("scenario_id") or "Unknown"
        bucket = scenario_rows.setdefault(
            scenario,
            {
                "scenario": scenario,
                "polarity": row.get("polarity", "unknown"),
                "runs": 0,
                "expected": 0,
                "app_count": 0,
                "candidate_count": 0,
            },
        )
        bucket["runs"] += 1
        bucket["expected"] += as_int(row.get("expected"))
        bucket["app_count"] += as_int(row.get("app_count"))
        bucket["candidate_count"] += as_int(row.get("candidate_count"))
        for threshold in THRESHOLDS:
            key = threshold_key(threshold)
            bucket[key] = bucket.get(key, 0) + as_int(row.get(key))

    totals: list[ThresholdTotals] = []
    for threshold in THRESHOLDS:
        key = threshold_key(threshold)
        positives = [row for row in included if row.get("polarity") == "positive"]
        negatives = [row for row in included if row.get("polarity") == "negative"]
        totals.append(
            ThresholdTotals(
                threshold=threshold,
                positive_count=sum(as_int(row.get(key)) for row in positives),
                positive_expected=sum(as_int(row.get("expected")) for row in positives),
                negative_count=sum(as_int(row.get(key)) for row in negatives),
                negative_expected=sum(as_int(row.get("expected")) for row in negatives),
            )
        )

    return sorted(scenario_rows.values(), key=lambda item: item["scenario"]), totals


def filter_t0103_policy_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for row in rows:
        if all(str(row.get(key, "")) == value for key, value in T0103_POLICY_FILTER.items()):
            output.append(row)
    return sorted(output, key=lambda item: as_float(item["threshold"]))


def summarize_t0103_policy(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in filter_t0103_policy_rows(rows):
        output.append(
            {
                "threshold": as_float(row["threshold"]),
                "boundary_positive_tp_140": as_int(row["boundary_positive_tp_140"]),
                "boundary_positive_truth": as_int(row["boundary_positive_tp_140"])
                + as_int(row["boundary_positive_missed_140"]),
                "boundary_negative_false_counts": as_int(row["boundary_negative_false_counts"]),
                "round_positive_tp_140": as_int(row["round_positive_tp_140"]),
                "round_positive_truth": as_int(row["round_positive_tp_140"])
                + as_int(row["round_positive_missed_140"]),
                "round_negative_false_counts": as_int(row["round_negative_false_counts"]),
                "boundary_far_soft_tp": as_int(row["boundary_positive_far_soft_tp"]),
                "boundary_far_soft_truth": as_int(row["boundary_positive_far_soft_truth"]),
                "boundary_soft_high_tp": as_int(row["boundary_positive_soft_high_tp"]),
                "boundary_soft_high_truth": as_int(row["boundary_positive_soft_high_truth"]),
                "boundary_normal_noisy_tp": as_int(row["boundary_positive_normal_noisy_tp"]),
                "boundary_normal_noisy_truth": as_int(row["boundary_positive_normal_noisy_truth"]),
            }
        )
    return output


def get_threshold_row(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    for row in rows:
        if abs(as_float(row["threshold"]) - threshold) < 1e-9:
            return row
    raise ValueError(f"Missing threshold row {threshold}")


def format_count(count: int, expected: int) -> str:
    return f"{count}/{expected}"


def write_report(
    path: Path,
    live_scenarios: list[dict[str, Any]],
    live_totals: list[ThresholdTotals],
    t0103_rows: list[dict[str, Any]],
) -> None:
    live_020 = next(item for item in live_totals if item.threshold == 0.2)
    live_030 = next(item for item in live_totals if item.threshold == 0.3)
    live_0575 = next(item for item in live_totals if item.threshold == 0.575)
    sweep_020 = get_threshold_row(t0103_rows, 0.2)
    sweep_030 = get_threshold_row(t0103_rows, 0.3)
    sweep_0575 = get_threshold_row(t0103_rows, 0.575)
    lines = [
        "# T0104C p>=0.20 No-Veto Safety Sweep",
        "",
        "## Scope",
        "",
        "- Evaluates Love's live setting: `threshold=0.20`, Fable-noise veto disabled/effectively `1.00`, smart dedupe from the existing app/debug rows.",
        "- Uses already pulled/evaluable artifacts: fresh corrected T0104 `Bounce audio test` debug summaries and the T0103 boundary/Round A policy sweep.",
        "- Does not ingest unfinished T0104B labels, retrain, export, install, or change app behavior.",
        "",
        "## Fresh T0104 Motorola Debug Replay",
        "",
        "| Threshold | Positives | Negative false counts |",
        "|---:|---:|---:|",
    ]
    for total in live_totals:
        lines.append(
            f"| `{total.threshold:g}` | `{format_count(total.positive_count, total.positive_expected)}` | `{total.negative_count}` |"
        )
    lines.extend(
        [
            "",
            "## Fresh T0104 By Scenario",
            "",
            "| Scenario | Expected | App/default 0.575 | p>=0.20 | p>=0.30 | Candidates |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in live_scenarios:
        lines.append(
            f"| {row['scenario']} | `{row['expected']}` | `{row.get('p>=0.575', 0)}` | `{row.get('p>=0.2', 0)}` | `{row.get('p>=0.3', 0)}` | `{row['candidate_count']}` |"
        )
    lines.extend(
        [
            "",
            "## Older T0103 Boundary/Round A Safety Sweep",
            "",
            "Policy slice: `extra_leaf4_t0103`, `base_t0075`, `boundary_recall_safety`, `dedupe=180`, no Fable-noise veto.",
            "",
            "| Threshold | Boundary positives | Boundary false counts | Round A positives | Round A hard-negative false counts |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in t0103_rows:
        lines.append(
            f"| `{row['threshold']:g}` | `{format_count(row['boundary_positive_tp_140'], row['boundary_positive_truth'])}` | `{row['boundary_negative_false_counts']}` | `{format_count(row['round_positive_tp_140'], row['round_positive_truth'])}` | `{row['round_negative_false_counts']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- On the fresh T0104 phone pull, `p>=0.20/no-veto` is much closer to usable: `{format_count(live_020.positive_count, live_020.positive_expected)}` positives. It is not perfectly safe even there: `{live_020.negative_count}` false counts across talking-only/racket-handling negatives.",
            f"- `p>=0.30/no-veto` is a safer-looking live compromise on this exact pull: `{format_count(live_030.positive_count, live_030.positive_expected)}` positives with `{live_030.negative_count}` fresh negative false counts.",
            f"- The installed/default guarded threshold `0.575` is too conservative live: `{format_count(live_0575.positive_count, live_0575.positive_expected)}` positives with `{live_0575.negative_count}` fresh negative false counts.",
            f"- The older T0103 safety sweep says `p>=0.20/no-veto` is not good enough to promote: `{sweep_020['boundary_negative_false_counts']}` fresh-boundary negative false counts and `{sweep_020['round_negative_false_counts']}` Round A hard-negative false counts.",
            f"- Even `p>=0.30/no-veto` remains unsafe on older safety rows: `{sweep_030['boundary_negative_false_counts']}` boundary false counts and `{sweep_030['round_negative_false_counts']}` Round A hard-negative false counts.",
            f"- The guarded `0.575/no-veto` policy is much safer in the older sweep (`{sweep_0575['boundary_negative_false_counts']}` boundary false counts, `{sweep_0575['round_negative_false_counts']}` Round A false counts) but misses too many live positives.",
            "",
            "## Recommendation",
            "",
            "`p>=0.20/no-veto` should remain a diagnostic setting, not a promoted app default. It proves the peak candidates and current score contain many true bounces, so more labeling can help. The next useful step is to ingest the saved T0104B exact labels and train/evaluate a candidate that raises weak real-bounce probabilities without also raising handling/background/impact negatives.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t0104-csv", type=Path, default=DEFAULT_T0104_CSV)
    parser.add_argument("--t0103-sweep", type=Path, default=DEFAULT_T0103_SWEEP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    t0104_rows = read_csv(args.t0104_csv)
    t0103_rows = read_csv(args.t0103_sweep)
    live_scenarios, live_totals = summarize_t0104_live(t0104_rows)
    t0103_summary_rows = summarize_t0103_policy(t0103_rows)

    if not t0103_summary_rows:
        raise SystemExit("No T0103 policy rows matched the expected app-exportable policy filter.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.out_dir / "t0104c_live_threshold_summary.csv",
        live_scenarios,
        [
            "scenario",
            "polarity",
            "runs",
            "expected",
            "app_count",
            "candidate_count",
            *[threshold_key(threshold) for threshold in THRESHOLDS],
        ],
    )
    write_csv(
        args.out_dir / "t0104c_t0103_policy_comparison.csv",
        t0103_summary_rows,
        [
            "threshold",
            "boundary_positive_tp_140",
            "boundary_positive_truth",
            "boundary_negative_false_counts",
            "round_positive_tp_140",
            "round_positive_truth",
            "round_negative_false_counts",
            "boundary_far_soft_tp",
            "boundary_far_soft_truth",
            "boundary_soft_high_tp",
            "boundary_soft_high_truth",
            "boundary_normal_noisy_tp",
            "boundary_normal_noisy_truth",
        ],
    )
    write_json(
        args.out_dir / "t0104c_summary.json",
        {
            "setting_under_test": {
                "threshold": 0.2,
                "fable_noise_veto": "disabled/no-veto",
                "dedupe_ms": 180,
            },
            "fresh_t0104_live_threshold_totals": [item.__dict__ for item in live_totals],
            "fresh_t0104_live_scenarios": live_scenarios,
            "t0103_policy_rows": t0103_summary_rows,
            "recommendation": "keep_p020_no_veto_diagnostic_not_promoted",
        },
    )
    write_report(args.out_dir / "t0104c_report.md", live_scenarios, live_totals, t0103_summary_rows)
    print(f"Wrote {args.out_dir / 't0104c_report.md'}")


if __name__ == "__main__":
    main()
