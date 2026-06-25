#!/usr/bin/env python3
"""Replay the T0030 joblib playing-retro model against the installed T0028 baseline.

This is a local replay/tuning step. It does not retrain, export app JSON, build
an APK, or change studs_live.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import replay_playing_retro_audio_t0027 as base
from train_playing_retro_audio import EVAL_DIR, MODEL_ROOT


MODEL_ID = "playing_retro_audio_rf_v2026_06_04_t0030_multi_window_context"
MODEL_DIR = MODEL_ROOT / MODEL_ID
BASELINE_MODEL_ID = "playing_retro_audio_rf_v2026_06_04_t0026_multi_window_context"
BASELINE_MODEL_DIR = MODEL_ROOT / BASELINE_MODEL_ID
DATASET_CSV = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "audio"
    / "processed"
    / "playing_retro_audio_multi_window_dataset_t0030_2026_06_04_006.csv"
)

DEFAULT_SESSIONS = [
    "audio_session_2026-05-28_002",
    "audio_session_2026-05-29_001",
    "audio_session_2026-05-29_002",
    "audio_session_2026-06-03_005",
    "audio_session_2026-06-04_001",
    "audio_session_2026-06-04_006",
]

PREDICTIONS_CSV = EVAL_DIR / "playing_retro_audio_t0030_replay_predictions.csv"
EVAL_CSV = EVAL_DIR / "playing_retro_audio_t0030_replay_eval.csv"
SWEEP_CSV = EVAL_DIR / "playing_retro_audio_t0030_threshold_sweep.csv"
REPORT_JSON = EVAL_DIR / "playing_retro_audio_t0030_replay_report.json"
REPORT_MD = EVAL_DIR / "playing_retro_audio_t0030_replay_report.md"

T0028_RACKET_THRESHOLD = 0.0
T0028_TABLE_THRESHOLD = 0.45


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def add_model_predictions(
    candidate_df: pd.DataFrame,
    model_dir: Path,
    prediction_column: str,
    confidence_column: str,
    probability_prefix: str,
) -> None:
    classifier, scaler, encoder, features = base.load_t0026_model(model_dir)
    predictions, confidences, probabilities = base.predict_joblib_model(
        classifier,
        scaler,
        encoder,
        features,
        candidate_df,
    )
    candidate_df[prediction_column] = predictions
    candidate_df[confidence_column] = confidences
    for label, values in probabilities.items():
        candidate_df[f"{probability_prefix}_{label}"] = values


def metric_delta(selected: dict[str, Any], baseline: dict[str, Any]) -> dict[str, int]:
    return {
        "true_positive": int(selected["true_positive"]) - int(baseline["true_positive"]),
        "wrong_class": int(selected["wrong_class"]) - int(baseline["wrong_class"]),
        "false_positive": int(selected["false_positive"]) - int(baseline["false_positive"]),
        "missed": int(selected["missed"]) - int(baseline["missed"]),
    }


def write_markdown(report: dict[str, Any]) -> None:
    baseline = report["marker_replay"]["t0028_baseline"]
    selected = report["marker_replay"]["t0030_selected"]
    delta = report["marker_replay"]["delta"]
    lines = [
        "# Playing Retro Audio T0030 Replay Report",
        "",
        "This is a local replay/tuning report. No app JSON, APK, `studs_live`, or video model artifact was changed.",
        "",
        "## Decision",
        "",
        f"- T0028 baseline model: `{BASELINE_MODEL_ID}`",
        f"- T0030 candidate model: `{MODEL_ID}`",
        f"- Selected racket threshold: `{selected['racket_threshold']}`",
        f"- Selected table threshold: `{selected['table_threshold']}`",
        f"- Recommendation: `{report['recommendation']}`",
        "",
        "## Marker Replay",
        "",
        "| Model | Predictions | TP | Wrong | FP | Missed | Racket TP | Table TP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| T0028 baseline | {baseline['predictions']} | {baseline['true_positive']} | "
            f"{baseline['wrong_class']} | {baseline['false_positive']} | {baseline['missed']} | "
            f"{baseline['tp_racket']} | {baseline['tp_table']} |"
        ),
        (
            f"| T0030 selected | {selected['predictions']} | {selected['true_positive']} | "
            f"{selected['wrong_class']} | {selected['false_positive']} | {selected['missed']} | "
            f"{selected['tp_racket']} | {selected['tp_table']} |"
        ),
        (
            f"| Delta |  | {delta['true_positive']:+} | {delta['wrong_class']:+} | "
            f"{delta['false_positive']:+} | {delta['missed']:+} |  |  |"
        ),
        "",
        "## Per Session",
        "",
        "| Session | T0028 TP/Wrong/FP/Missed | T0030 TP/Wrong/FP/Missed |",
        "|---|---|---|",
    ]
    baseline_sessions = report["marker_replay"]["per_session"]["t0028_baseline"]
    selected_sessions = report["marker_replay"]["per_session"]["t0030_selected"]
    for session_id in report["sessions"]:
        left = baseline_sessions.get(session_id, {})
        right = selected_sessions.get(session_id, {})
        lines.append(
            f"| `{session_id}` | "
            f"{left.get('true_positive', 0)}/{left.get('wrong_class', 0)}/{left.get('false_positive', 0)}/{left.get('missed', 0)} | "
            f"{right.get('true_positive', 0)}/{right.get('wrong_class', 0)}/{right.get('false_positive', 0)}/{right.get('missed', 0)} |"
        )
    lines.extend([
        "",
        "## Outputs",
        "",
        f"- Predictions CSV: `{PREDICTIONS_CSV.as_posix()}`",
        f"- Evaluation CSV: `{EVAL_CSV.as_posix()}`",
        f"- Threshold sweep CSV: `{SWEEP_CSV.as_posix()}`",
        f"- JSON report: `{REPORT_JSON.as_posix()}`",
        "",
    ])
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay T0030 playing-retro model and tune thresholds.")
    parser.add_argument("--dataset-csv", default=str(DATASET_CSV))
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--baseline-model-dir", default=str(BASELINE_MODEL_DIR))
    parser.add_argument("--session", action="append", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sessions = args.session or DEFAULT_SESSIONS
    dataset_path = Path(args.dataset_csv)
    if not dataset_path.exists():
        raise SystemExit(f"Missing {dataset_path}; run T0030 training first.")

    dataset = pd.read_csv(dataset_path)
    target = dataset[dataset["session_id"].astype(str).isin(set(sessions))].copy()
    candidate_df = target[target["row_type"].fillna("").astype(str) == "candidate"].copy()
    if candidate_df.empty:
        raise SystemExit("No candidate rows found for requested sessions.")

    add_model_predictions(
        candidate_df,
        Path(args.baseline_model_dir),
        "t0028_prediction",
        "t0028_confidence",
        "t0028_probability",
    )
    add_model_predictions(
        candidate_df,
        Path(args.model_dir),
        "t0026_prediction",
        "t0026_confidence",
        "t0030_probability",
    )
    candidate_df["baseline_prediction"] = candidate_df["t0028_prediction"]
    candidate_df["is_recovery_candidate"] = base.recovery_mask(candidate_df)
    candidate_df["nearest_saved_gap_ms"] = base.nearest_saved_gap_by_row(candidate_df)

    eval_rows: list[dict[str, Any]] = []
    for row in base.grouped_metric_rows("t0028_candidate_prediction", candidate_df, "t0028_prediction"):
        eval_rows.append({"kind": "candidate", **row})
    for row in base.grouped_metric_rows("t0030_candidate_prediction", candidate_df, "t0026_prediction"):
        eval_rows.append({"kind": "candidate", **row})

    truths = base.truth_rows_for_sessions(sessions)
    baseline_events = base.prediction_events(
        candidate_df,
        prediction_column="t0028_prediction",
        confidence_column="t0028_confidence",
        racket_threshold=T0028_RACKET_THRESHOLD,
        table_threshold=T0028_TABLE_THRESHOLD,
        baseline_visible_only=False,
    )
    baseline_summary, baseline_session_rows = base.evaluate_marker_predictions(
        baseline_events,
        truths,
        scope="t0028_baseline",
        match_ms=base.REPLAY_MATCH_MS,
        dedupe_ms=base.SAME_LABEL_DEDUPE_MS,
    )
    sweep_df, best_by_score = base.sweep_thresholds(candidate_df, truths)
    selected = base.select_safe_threshold(sweep_df, baseline_summary)
    selected_events = base.prediction_events(
        candidate_df,
        prediction_column="t0026_prediction",
        confidence_column="t0026_confidence",
        racket_threshold=float(selected["racket_threshold"]),
        table_threshold=float(selected["table_threshold"]),
        baseline_visible_only=False,
    )
    selected_summary, selected_session_rows = base.evaluate_marker_predictions(
        selected_events,
        truths,
        scope="t0030_selected",
        match_ms=base.REPLAY_MATCH_MS,
        dedupe_ms=base.SAME_LABEL_DEDUPE_MS,
    )
    selected_summary["racket_threshold"] = float(selected["racket_threshold"])
    selected_summary["table_threshold"] = float(selected["table_threshold"])
    selected_summary["score"] = float(selected["score"])

    prediction_columns = [
        "session_id",
        "event_index",
        "wav_filename",
        "candidate_id",
        "anchor_ms",
        "label",
        "candidate_status",
        "candidate_predicted_kind",
        "t0028_prediction",
        "t0028_confidence",
        "t0026_prediction",
        "t0026_confidence",
        "t0030_probability_racket_contact",
        "t0030_probability_table_bounce",
        "t0030_probability_non_target",
        "source_rule",
        "match_outcome",
        "close_event_bucket",
        "neighbor_sequence",
        "is_recovery_candidate",
        "nearest_saved_gap_ms",
    ]
    PREDICTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    candidate_df[[column for column in prediction_columns if column in candidate_df.columns]].to_csv(PREDICTIONS_CSV, index=False)
    eval_df = pd.DataFrame(eval_rows)
    EVAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(EVAL_CSV, index=False)
    sweep_df.to_csv(SWEEP_CSV, index=False)

    report = {
        "ticket": "T0030",
        "status": "local_replay_only_not_exported",
        "model_id": MODEL_ID,
        "baseline_model_id": BASELINE_MODEL_ID,
        "sessions": sessions,
        "candidate_rows": int(len(candidate_df)),
        "truth_rows": int(len(truths)),
        "recommendation": base.recommendation(selected_summary, baseline_summary),
        "marker_replay": {
            "t0028_baseline": baseline_summary,
            "t0030_selected": selected_summary,
            "best_by_score": best_by_score,
            "delta": metric_delta(selected_summary, baseline_summary),
            "per_session": {
                "t0028_baseline": base.per_session_totals(baseline_session_rows),
                "t0030_selected": base.per_session_totals(selected_session_rows),
            },
        },
        "outputs": {
            "predictions_csv": str(PREDICTIONS_CSV),
            "eval_csv": str(EVAL_CSV),
            "sweep_csv": str(SWEEP_CSV),
            "report_json": str(REPORT_JSON),
            "report_md": str(REPORT_MD),
        },
        "changed_app_artifacts": False,
        "changed_studs_live": False,
        "changed_video_model": False,
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2, default=json_default), encoding="utf-8")
    write_markdown(report)

    print(f"Wrote {PREDICTIONS_CSV}")
    print(f"Wrote {EVAL_CSV}")
    print(f"Wrote {SWEEP_CSV}")
    print(f"Wrote {REPORT_JSON}")
    print(f"Wrote {REPORT_MD}")
    print(
        "t0028_baseline: pred={pred} tp={tp} wrong={wrong} fp={fp} missed={missed}".format(
            pred=baseline_summary["predictions"],
            tp=baseline_summary["true_positive"],
            wrong=baseline_summary["wrong_class"],
            fp=baseline_summary["false_positive"],
            missed=baseline_summary["missed"],
        )
    )
    print(
        "t0030_selected: racket_thr={racket} table_thr={table} pred={pred} tp={tp} wrong={wrong} fp={fp} missed={missed}".format(
            racket=selected_summary["racket_threshold"],
            table=selected_summary["table_threshold"],
            pred=selected_summary["predictions"],
            tp=selected_summary["true_positive"],
            wrong=selected_summary["wrong_class"],
            fp=selected_summary["false_positive"],
            missed=selected_summary["missed"],
        )
    )
    print(f"recommendation={report['recommendation']}")


if __name__ == "__main__":
    main()
