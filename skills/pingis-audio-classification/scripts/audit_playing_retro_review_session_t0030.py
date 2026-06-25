#!/usr/bin/env python3
"""Audit the T0028-reviewed 2026-06-04_006 playing-retro session.

This script is analysis-only. It does not train, export app JSON, build an APK,
or change studs_live.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import audit_playing_retro_review_session_t0025 as base


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_SESSION = ROOT_DIR / "data" / "audio" / "raw" / "audio_session_2026-06-04_006.json"
EVAL_DIR = ROOT_DIR / "data" / "audio" / "models" / "evaluations"
DEFAULT_JSON = EVAL_DIR / "playing_retro_t0030_2026_06_04_006_audit.json"
DEFAULT_MD = EVAL_DIR / "playing_retro_t0030_2026_06_04_006_audit.md"
DEFAULT_MANUAL_CSV = EVAL_DIR / "playing_retro_t0030_2026_06_04_006_manual_additions.csv"
DEFAULT_GAPS_CSV = EVAL_DIR / "playing_retro_t0030_2026_06_04_006_close_gaps.csv"

T0028_MODEL_ID = "playing_retro_audio_rf_v2026_06_04_t0026_multi_window_context"
T0028_RACKET_THRESHOLD = 0.0
T0028_TABLE_THRESHOLD = 0.45


def md_table(rows: list[list[Any]], headers: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    evaluation = report["evaluation"]["review_relevant_candidates_vs_final_markers"]
    lines = [
        "# Playing Retro T0030 Audit: audio_session_2026-06-04_006",
        "",
        "This is an audit-only report. It does not train, export app JSON, build an APK, or change `studs_live`.",
        "",
        "## Summary",
        "",
        f"- Final audio markers: `{summary['markers_total']}` (`{summary['markers_by_kind']}`)",
        f"- Sources: `{summary['markers_by_source']}`",
        f"- Review status: `{summary['markers_by_status']}`",
        f"- Manual additions: `{summary['manual_markers']}` (`{summary['manual_by_kind']}`)",
        f"- Auto markers: `{summary['auto_markers']}`",
        f"- Model candidates: `{summary['model_candidates_total']}` (`{summary['model_candidates_by_prediction']}`)",
        f"- Review-relevant target candidates: `{summary['review_relevant_target_candidates']}` (`{summary['review_relevant_by_prediction']}`)",
        f"- Hidden target predictions: `{summary['hidden_target_predictions']}` (`{summary['hidden_target_by_prediction']}`)",
        f"- Manual additions near any candidate: `{summary['manual_nearest_candidate_counts']}`",
        "",
        "## Review-Relevant Candidate Replay",
        "",
        "| Predictions | TP | Wrong | FP | Missed | Missed by kind |",
        "|---:|---:|---:|---:|---:|---|",
        (
            f"| {evaluation['prediction_count']} | {evaluation['true_positive']} | "
            f"{evaluation['wrong_class']} | {evaluation['false_positive']} | "
            f"{evaluation['missed']} | `{evaluation['missed_by_kind']}` |"
        ),
        "",
        "## Manual Addition Failure Buckets",
        "",
    ]
    lines.extend(md_table(
        [
            [bucket, count, summary["manual_reason_by_kind"].get(bucket, {})]
            for bucket, count in summary["manual_reason_counts"].items()
        ],
        ["Reason bucket", "Count", "Kinds"],
    ))
    lines.extend([
        "",
        "## Manual Additions",
        "",
    ])
    lines.extend(md_table(
        [
            [
                row["timestamp_ms"],
                row["kind"],
                row["reason_bucket"],
                base.compact_candidate(row["nearest_any"]),
                base.compact_candidate(row["reason_candidate"]),
            ]
            for row in report["manual_additions"]
        ],
        ["ms", "kind", "reason", "nearest candidate", "reason candidate"],
    ))
    lines.extend([
        "",
        "## Close Final Marker Gaps Under 120 ms",
        "",
    ])
    lines.extend(md_table(
        [
            [
                row["gap_ms"],
                f"{row['previous_kind']}@{row['previous_ms']}",
                f"{row['current_kind']}@{row['current_ms']}",
                f"{row['previous_source']}->{row['current_source']}",
            ]
            for row in report["close_gaps_under_120_ms"]
        ],
        ["gap ms", "previous", "current", "source"],
    ))
    lines.extend([
        "",
        "## Recommended Next Tickets",
        "",
        "- `T0030`: train and replay a local `spel_retro_audio` candidate with this session plus historical playing data.",
        "- `T0031`: export/build/install only if T0030 replay safely beats the installed T0028 baseline.",
        "- Keep `Studsdetektor`, ordinary bounce, app JSON export, APK build, and video-stroke work out of this audit step.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(report: dict[str, Any], output_json: Path, output_md: Path, manual_csv: Path, gaps_csv: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(output_md, report)
    base.write_manual_csv(manual_csv, report["manual_additions"])
    base.write_csv(
        gaps_csv,
        report["close_gaps_under_120_ms"],
        [
            "gap_ms",
            "previous_ms",
            "current_ms",
            "previous_kind",
            "current_kind",
            "previous_source",
            "current_source",
            "previous_status",
            "current_status",
        ],
    )


def print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    evaluation = report["evaluation"]["review_relevant_candidates_vs_final_markers"]
    print(f"# T0030 audit {report['session_id']}")
    print(f"markers={summary['markers_total']} by_kind={summary['markers_by_kind']}")
    print(f"auto={summary['auto_markers']} manual={summary['manual_markers']} manual_by_kind={summary['manual_by_kind']}")
    print(f"candidates={summary['model_candidates_total']} by_prediction={summary['model_candidates_by_prediction']}")
    print(f"hidden_targets={summary['hidden_target_predictions']} {summary['hidden_target_by_prediction']}")
    print(f"manual_nearest={summary['manual_nearest_candidate_counts']}")
    print(f"manual_reasons={summary['manual_reason_counts']}")
    print(
        "replay_visible_candidates pred={prediction_count} tp={true_positive} wrong={wrong_class} "
        "fp={false_positive} missed={missed}".format(**evaluation)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit T0030 playing-retro review session.")
    parser.add_argument("session_json", type=Path, nargs="?", default=DEFAULT_SESSION)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--manual-csv", type=Path, default=DEFAULT_MANUAL_CSV)
    parser.add_argument("--gaps-csv", type=Path, default=DEFAULT_GAPS_CSV)
    return parser.parse_args()


def main() -> None:
    base.T0024_RACKET_THRESHOLD = T0028_RACKET_THRESHOLD
    base.T0024_TABLE_THRESHOLD = T0028_TABLE_THRESHOLD

    args = parse_args()
    report = base.build_report(args.session_json)
    report["ticket"] = "T0030"
    report["t0028_settings"] = {
        "model": T0028_MODEL_ID,
        "racket_threshold": T0028_RACKET_THRESHOLD,
        "table_threshold": T0028_TABLE_THRESHOLD,
        "same_label_dedupe_ms": base.SAME_LABEL_DEDUPE_MS,
    }
    report.pop("t0024_settings", None)
    write_outputs(report, args.output_json, args.output_md, args.manual_csv, args.gaps_csv)
    print_summary(report)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    print(f"wrote {args.manual_csv}")
    print(f"wrote {args.gaps_csv}")


if __name__ == "__main__":
    main()
