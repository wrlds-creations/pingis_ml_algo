"""
Retrain the local T0026 playing-retro audio candidate.

This script is intentionally separate from the T0022 script so the 2026-06-04
review can be added without overwriting older cached datasets or app exports.
It trains local joblib artifacts only; it does not export Collector JSON, build
an APK, or change studs_live.

Run:
  python skills/pingis-audio-classification/scripts/train_playing_retro_audio_t0026.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from build_playing_retro_candidate_report import MATCH_TOLERANCE_MS
from evaluate_playing_retro_audio_multi_window import (
    VARIANTS,
    WINDOWS,
    Variant,
    build_multi_window_dataset,
    feature_columns_for_mode,
    metric_summary,
    ordinary_fallback_dataset,
    predict_labels,
    train_variant,
)
from train_playing_retro_audio import (
    EVAL_DIR,
    MODEL_ROOT,
    OUT_DIR,
    TARGET_LABELS,
    build_candidate_rows,
    grouped_metrics,
    mapped_baseline_prediction,
    selected_playing_events,
)


FOCUS_SESSION = "audio_session_2026-06-04_001"
MODEL_ID = "playing_retro_audio_rf_v2026_06_04_t0026_multi_window_context"

CANDIDATE_ROWS_CSV = OUT_DIR / "playing_retro_audio_candidate_rows_t0026_2026_06_04.csv"
DATASET_CSV = OUT_DIR / "playing_retro_audio_multi_window_dataset_t0026_2026_06_04.csv"
EVAL_CSV = EVAL_DIR / "playing_retro_audio_t0026_retrain_eval.csv"
PREDICTIONS_CSV = EVAL_DIR / "playing_retro_audio_t0026_retrain_predictions.csv"
REPORT_MD = EVAL_DIR / "playing_retro_audio_t0026_retrain_report.md"
MODEL_DIR = MODEL_ROOT / MODEL_ID

DEFAULT_HOLDOUT_SESSIONS = [
    "audio_session_2026-05-28_002",
    "audio_session_2026-05-29_001",
    "audio_session_2026-05-29_002",
    "audio_session_2026-06-03_005",
    FOCUS_SESSION,
]

REFERENCE_T0022 = {
    "audio_session_2026-05-28_002": {
        "accuracy": 0.913,
        "racket_contact_recall": 0.910,
        "table_bounce_recall": 0.946,
        "non_target_recall": 0.879,
    },
    "audio_session_2026-05-29_001": {
        "accuracy": 0.921,
        "racket_contact_recall": 0.939,
        "table_bounce_recall": 0.958,
        "non_target_recall": 0.859,
    },
    "audio_session_2026-05-29_002": {
        "accuracy": 0.908,
        "racket_contact_recall": 0.896,
        "table_bounce_recall": 0.924,
        "non_target_recall": 0.875,
    },
    "audio_session_2026-06-03_005": {
        "accuracy": 0.848,
        "racket_contact_recall": 0.905,
        "table_bounce_recall": 0.935,
        "non_target_recall": 0.667,
    },
}


def as_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def safe_recall(summary: dict[str, Any], label: str) -> float:
    value = summary.get(f"{label}_recall")
    return float(value) if value is not None and not pd.isna(value) else 0.0


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def build_fresh_dataset(min_racket: int, min_table: int, match_tolerance_ms: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = selected_playing_events(min_racket=min_racket, min_table=min_table)
    if not events:
        raise SystemExit("No playing-retro training events selected.")

    candidate_rows = pd.DataFrame(build_candidate_rows(events, tolerance_ms=match_tolerance_ms))
    if candidate_rows.empty:
        raise SystemExit("No candidate rows produced.")

    dataset = build_multi_window_dataset(candidate_rows)
    if dataset.empty:
        raise SystemExit("No multi-window dataset rows produced.")
    return candidate_rows, dataset


def prediction_metadata(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "session_id",
        "event_index",
        "wav_filename",
        "candidate_id",
        "anchor_ms",
        "label",
        "source_rule",
        "row_type",
        "candidate_status",
        "candidate_predicted_kind",
        "match_outcome",
        "close_event_bucket",
        "neighbor_sequence",
        "source_config",
        "ctx_prev_gap_1000",
        "ctx_next_gap_1000",
        "ctx_density_300ms",
    ]
    existing = [column for column in columns if column in df.columns]
    return df[existing].copy()


def evaluate_variant(
    dataset: pd.DataFrame,
    variant: Variant,
    holdout_sessions: list[str],
) -> tuple[list[str], list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    features = feature_columns_for_mode(dataset, variant.feature_mode)
    eval_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    session_summaries: list[dict[str, Any]] = []

    for holdout_session in holdout_sessions:
        train_df = dataset[dataset["session_id"].astype(str) != holdout_session].copy()
        holdout_df = dataset[dataset["session_id"].astype(str) == holdout_session].copy()
        if train_df.empty or holdout_df.empty:
            continue

        classifier, scaler, label_encoder = train_variant(train_df, features, variant)
        predictions = predict_labels(classifier, scaler, label_encoder, holdout_df, features)

        eval_rows.extend(
            {"variant": variant.name, "holdout_session": holdout_session, **row}
            for row in grouped_metrics(
                "cross_session_holdout",
                holdout_df,
                predictions,
                ["session_id", "evaluation_bucket", "close_event_bucket", "source_rule"],
            )
        )

        pred_df = prediction_metadata(holdout_df)
        pred_df.insert(0, "variant", variant.name)
        pred_df.insert(1, "holdout_session", holdout_session)
        pred_df["prediction"] = predictions
        pred_df["correct"] = pred_df["label"].astype(str) == pred_df["prediction"].astype(str)
        prediction_frames.append(pred_df)

        summary = metric_summary(holdout_df, predictions)
        summary["session_id"] = holdout_session
        reference = REFERENCE_T0022.get(holdout_session)
        if reference:
            for metric in ["accuracy", "racket_contact_recall", "table_bounce_recall", "non_target_recall"]:
                value = as_float(summary.get(metric))
                summary[f"{metric}_vs_t0022"] = None if value is None else value - reference[metric]
        session_summaries.append(summary)

    predictions_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    if predictions_df.empty:
        combined_summary = {"rows": 0, "accuracy": 0.0}
    else:
        combined_summary = metric_summary(predictions_df, predictions_df["prediction"].astype(str).to_numpy())

    focus = next((item for item in session_summaries if item["session_id"] == FOCUS_SESSION), None)
    old_sessions = [item for item in session_summaries if item["session_id"] in REFERENCE_T0022]
    target_recalls = [
        mean([safe_recall(item, "racket_contact"), safe_recall(item, "table_bounce")])
        for item in session_summaries
    ]
    focus_target_recall = (
        mean([safe_recall(focus, "racket_contact"), safe_recall(focus, "table_bounce")])
        if focus
        else 0.0
    )
    non_target_mean = mean([safe_recall(item, "non_target") for item in session_summaries])
    old_reference_safe = all(
        (
            as_float(item.get("racket_contact_recall_vs_t0022")) is not None
            and as_float(item.get("table_bounce_recall_vs_t0022")) is not None
            and as_float(item.get("non_target_recall_vs_t0022")) is not None
            and float(item["racket_contact_recall_vs_t0022"]) >= -0.05
            and float(item["table_bounce_recall_vs_t0022"]) >= -0.03
            and float(item["non_target_recall_vs_t0022"]) >= -0.08
        )
        for item in old_sessions
    )
    selection_score = 0.55 * focus_target_recall + 0.30 * mean(target_recalls) + 0.15 * non_target_mean

    summary = {
        "name": variant.name,
        "description": variant.description,
        "config": variant.__dict__,
        "feature_count": len(features),
        "holdout": combined_summary,
        "holdout_by_session": session_summaries,
        "old_reference_safe": bool(old_reference_safe),
        "focus_target_recall": focus_target_recall,
        "selection_score": float(selection_score),
    }
    return features, eval_rows, predictions_df, summary


def choose_selected_variant(summaries: list[dict[str, Any]]) -> str:
    safe = [item for item in summaries if item.get("old_reference_safe")]
    pool = safe or summaries
    pool.sort(
        key=lambda item: (
            item["selection_score"],
            safe_recall(item["holdout"], "racket_contact"),
            item["holdout"]["accuracy"],
        ),
        reverse=True,
    )
    return str(pool[0]["name"])


def final_focus_summary(dataset: pd.DataFrame, predictions: np.ndarray) -> dict[str, Any]:
    focus_df = dataset[dataset["session_id"].astype(str) == FOCUS_SESSION].copy()
    if focus_df.empty:
        return {"rows": 0}

    pred = pd.Series(predictions, index=dataset.index).loc[focus_df.index]
    candidate_rows = focus_df[focus_df["source_rule"].astype(str) != "manual_missed_marker"].copy()
    baseline_pred = mapped_baseline_prediction(candidate_rows["candidate_predicted_kind"])
    baseline_metrics = metric_summary(candidate_rows, baseline_pred.to_numpy()) if not candidate_rows.empty else {"rows": 0}

    target_labels = {"racket_contact", "table_bounce"}
    candidate_target_rows = candidate_rows[candidate_rows["label"].astype(str).isin(target_labels)].copy()
    candidate_target_baseline = mapped_baseline_prediction(candidate_target_rows["candidate_predicted_kind"])
    baseline_non_target_mask = candidate_target_baseline == "non_target"
    baseline_wrong_class_mask = (
        (candidate_target_baseline.isin(target_labels))
        & (candidate_target_baseline.astype(str) != candidate_target_rows["label"].astype(str))
    )

    final_for_candidate_targets = pred.loc[candidate_target_rows.index] if not candidate_target_rows.empty else pd.Series(dtype=str)
    manual_rows = focus_df[focus_df["source_rule"].astype(str) == "manual_missed_marker"].copy()
    final_for_manual = pred.loc[manual_rows.index] if not manual_rows.empty else pd.Series(dtype=str)

    return {
        "rows": int(len(focus_df)),
        "baseline_candidate_rows": int(len(candidate_rows)),
        "baseline_candidate_metrics": baseline_metrics,
        "baseline_target_candidate_errors": int(
            (candidate_target_baseline.to_numpy() != candidate_target_rows["label"].astype(str).to_numpy()).sum()
        ),
        "baseline_non_target_target_rows": int(baseline_non_target_mask.sum()),
        "final_correct_on_baseline_non_target_target_rows": int(
            (
                final_for_candidate_targets.loc[candidate_target_rows.index[baseline_non_target_mask.to_numpy()]]
                == candidate_target_rows.loc[baseline_non_target_mask.to_numpy(), "label"].astype(str)
            ).sum()
        ) if int(baseline_non_target_mask.sum()) else 0,
        "baseline_wrong_class_target_rows": int(baseline_wrong_class_mask.sum()),
        "final_correct_on_baseline_wrong_class_target_rows": int(
            (
                final_for_candidate_targets.loc[candidate_target_rows.index[baseline_wrong_class_mask.to_numpy()]]
                == candidate_target_rows.loc[baseline_wrong_class_mask.to_numpy(), "label"].astype(str)
            ).sum()
        ) if int(baseline_wrong_class_mask.sum()) else 0,
        "manual_missed_marker_rows": int(len(manual_rows)),
        "final_correct_manual_missed_marker_rows": int(
            (final_for_manual.to_numpy() == manual_rows["label"].astype(str).to_numpy()).sum()
        ) if not manual_rows.empty else 0,
        "final_focus_metrics": metric_summary(focus_df, pred.to_numpy()),
    }


def session_class_rows(dataset: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for session_id, group in dataset.groupby("session_id"):
        labels = group["label"].value_counts().to_dict()
        source_rules = group["source_rule"].value_counts().to_dict()
        rows.append({
            "session_id": str(session_id),
            "rows": int(len(group)),
            "racket_contact": int(labels.get("racket_contact", 0)),
            "table_bounce": int(labels.get("table_bounce", 0)),
            "non_target": int(labels.get("non_target", 0)),
            "manual_missed_marker": int(source_rules.get("manual_missed_marker", 0)),
            "candidate_rows": int(len(group) - int(source_rules.get("manual_missed_marker", 0))),
        })
    rows.sort(key=lambda item: item["session_id"])
    return rows


def metric_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.3f}"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected_variant"]
    selected_summary = next(item for item in report["variants"] if item["name"] == selected)
    focus_holdout = next(
        (item for item in selected_summary["holdout_by_session"] if item["session_id"] == FOCUS_SESSION),
        None,
    )
    focus = report["final_selected_model"]["focus_session"]
    lines = [
        "# Playing Retro Audio T0026 Retrain Report",
        "",
        "This is a local retrain only. No Collector app model JSON, APK, `studs_live`, or video model artifact was changed.",
        "",
        "## Decision",
        "",
        f"- Selected local candidate: `{MODEL_ID}`",
        f"- Selected variant: `{selected}`",
        f"- Model dir: `{MODEL_DIR.as_posix()}`",
        f"- Recommendation: `{report['recommendation']}`",
        "- Why: 06-04 was added to training and the selected variant stays safe versus T0022 reference sessions, but app promotion needs marker-level T0027 replay against the installed T0024 baseline.",
        "",
        "## Data",
        "",
        f"- Rows: `{report['dataset_rows']}`",
        f"- Sessions: `{report['dataset_sessions']}`",
        f"- Labels: `{report['label_counts']}`",
        f"- Holdout sessions: `{report['holdout_sessions']}`",
        f"- Windows: `{report['windows']}`",
        "- 06-04 manual additions enter training in two ways: nearby saved candidates become corrected target rows, while unmatched reviewed markers become `manual_missed_marker` rows.",
        "",
        "| Session | Rows | Racket | Table | Non-target | Manual missed | Candidate rows |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["session_class_rows"]:
        lines.append(
            f"| `{row['session_id']}` | {row['rows']} | {row['racket_contact']} | "
            f"{row['table_bounce']} | {row['non_target']} | {row['manual_missed_marker']} | "
            f"{row['candidate_rows']} |"
        )

    lines.extend([
        "",
        "## Variant Holdout Summary",
        "",
        "| Variant | Safe vs T0022 refs | Score | Acc | Racket | Table | Non-target |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for item in sorted(report["variants"], key=lambda value: value["selection_score"], reverse=True):
        holdout = item["holdout"]
        lines.append(
            f"| `{item['name']}` | `{item['old_reference_safe']}` | "
            f"{metric_text(item['selection_score'])} | {metric_text(holdout.get('accuracy'))} | "
            f"{metric_text(holdout.get('racket_contact_recall'))} | "
            f"{metric_text(holdout.get('table_bounce_recall'))} | "
            f"{metric_text(holdout.get('non_target_recall'))} |"
        )

    lines.extend([
        "",
        "## Selected Variant Leave-One-Session-Out",
        "",
        "| Holdout | Acc | Racket | Table | Non-target | Racket vs T0022 | Table vs T0022 | Non-target vs T0022 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for item in selected_summary["holdout_by_session"]:
        lines.append(
            f"| `{item['session_id']}` | {metric_text(item.get('accuracy'))} | "
            f"{metric_text(item.get('racket_contact_recall'))} | "
            f"{metric_text(item.get('table_bounce_recall'))} | "
            f"{metric_text(item.get('non_target_recall'))} | "
            f"{metric_text(item.get('racket_contact_recall_vs_t0022'))} | "
            f"{metric_text(item.get('table_bounce_recall_vs_t0022'))} | "
            f"{metric_text(item.get('non_target_recall_vs_t0022'))} |"
        )

    lines.extend([
        "",
        "## 2026-06-04 Held-Out Check",
        "",
        "This is leave-one-session-out: the selected variant is trained without the 06-04 session, then tested on 06-04 candidate rows.",
        "",
    ])
    if focus_holdout:
        lines.extend([
            f"- Rows: `{int(focus_holdout['rows'])}`",
            f"- Accuracy: `{metric_text(focus_holdout.get('accuracy'))}`",
            f"- Racket recall: `{metric_text(focus_holdout.get('racket_contact_recall'))}`",
            f"- Table recall: `{metric_text(focus_holdout.get('table_bounce_recall'))}`",
            f"- Non-target recall: `{metric_text(focus_holdout.get('non_target_recall'))}`",
            "",
        ])

    lines.extend([
        "",
        "## 2026-06-04 Focus Check",
        "",
        "This section is in-sample for the final saved model because T0026 intentionally trains the final candidate on the reviewed 2026-06-04 clip. Use T0027 replay for promotion evidence.",
        "",
        f"- Final 06-04 rows: `{focus['rows']}`",
        f"- Baseline candidate target errors: `{focus['baseline_target_candidate_errors']}`",
        f"- Baseline target rows called non-target: `{focus['baseline_non_target_target_rows']}`",
        f"- Final correct on those non-target target rows: `{focus['final_correct_on_baseline_non_target_target_rows']}`",
        f"- Baseline wrong-class target rows: `{focus['baseline_wrong_class_target_rows']}`",
        f"- Final correct on wrong-class target rows: `{focus['final_correct_on_baseline_wrong_class_target_rows']}`",
        f"- Manual missed marker rows: `{focus['manual_missed_marker_rows']}`",
        f"- Final correct manual missed marker rows: `{focus['final_correct_manual_missed_marker_rows']}`",
        "",
        "## Outputs",
        "",
        f"- Candidate rows CSV: `{CANDIDATE_ROWS_CSV.as_posix()}`",
        f"- Multi-window dataset CSV: `{DATASET_CSV.as_posix()}`",
        f"- Evaluation CSV: `{EVAL_CSV.as_posix()}`",
        f"- Predictions CSV: `{PREDICTIONS_CSV.as_posix()}`",
        f"- JSON report: `{(MODEL_DIR / 'report.json').as_posix()}`",
        f"- Model dir: `{MODEL_DIR.as_posix()}`",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train local T0026 playing-retro audio candidate.")
    parser.add_argument("--min-racket", type=int, default=5)
    parser.add_argument("--min-table", type=int, default=10)
    parser.add_argument("--match-tolerance-ms", type=int, default=MATCH_TOLERANCE_MS)
    parser.add_argument("--holdout-session", action="append", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    holdout_sessions = args.holdout_session or DEFAULT_HOLDOUT_SESSIONS
    candidate_rows, dataset = build_fresh_dataset(args.min_racket, args.min_table, args.match_tolerance_ms)

    holdout_sessions = [session for session in holdout_sessions if session in set(dataset["session_id"].astype(str))]
    if not holdout_sessions:
        raise SystemExit("No requested holdout sessions exist in the T0026 dataset.")

    CANDIDATE_ROWS_CSV.parent.mkdir(parents=True, exist_ok=True)
    candidate_rows.to_csv(CANDIDATE_ROWS_CSV, index=False)
    DATASET_CSV.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(DATASET_CSV, index=False)

    eval_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    variant_summaries: list[dict[str, Any]] = []
    features_by_variant: dict[str, list[str]] = {}

    for variant in VARIANTS:
        features, variant_eval_rows, prediction_df, summary = evaluate_variant(dataset, variant, holdout_sessions)
        features_by_variant[variant.name] = features
        eval_rows.extend(variant_eval_rows)
        if not prediction_df.empty:
            prediction_frames.append(prediction_df)
        variant_summaries.append(summary)

    selected_name = choose_selected_variant(variant_summaries)
    selected_variant = next(variant for variant in VARIANTS if variant.name == selected_name)
    selected_features = features_by_variant[selected_name]

    final_classifier, final_scaler, final_encoder = train_variant(dataset, selected_features, selected_variant)
    final_predictions = predict_labels(final_classifier, final_scaler, final_encoder, dataset, selected_features)

    base_feature_names = sorted({column.split("_", 1)[1] for column in dataset.columns if column.startswith("normal_")})
    ordinary_df = ordinary_fallback_dataset(selected_features, base_feature_names)
    ordinary_predictions = predict_labels(final_classifier, final_scaler, final_encoder, ordinary_df, selected_features)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_classifier, MODEL_DIR / "playing_retro_audio_rf_classifier.pkl")
    joblib.dump(final_scaler, MODEL_DIR / "playing_retro_audio_feature_scaler.pkl")
    joblib.dump(final_encoder, MODEL_DIR / "playing_retro_audio_label_encoder.pkl")
    joblib.dump(selected_features, MODEL_DIR / "playing_retro_audio_feature_cols.pkl")

    eval_df = pd.DataFrame(eval_rows)
    EVAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(EVAL_CSV, index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(PREDICTIONS_CSV, index=False)

    report: dict[str, Any] = {
        "model_id": MODEL_ID,
        "ticket": "T0026",
        "status": "local_candidate_only_not_exported",
        "recommendation": "proceed_to_t0027_replay_tune_before_export",
        "training_choice": "fresh_multi_window_context_dataset_from_historical_playing_plus_2026_06_04",
        "candidate_rows": int(len(candidate_rows)),
        "dataset_rows": int(len(dataset)),
        "dataset_sessions": int(dataset["session_id"].nunique()),
        "label_counts": {str(k): int(v) for k, v in dataset["label"].value_counts().to_dict().items()},
        "source_rules": {str(k): int(v) for k, v in dataset["source_rule"].value_counts().to_dict().items()},
        "session_class_rows": session_class_rows(dataset),
        "holdout_sessions": holdout_sessions,
        "windows": [{"name": name, "before_ms": before, "after_ms": after} for name, before, after in WINDOWS],
        "variants": variant_summaries,
        "selected_variant": selected_name,
        "final_selected_model": {
            "name": selected_name,
            "training_rows": int(len(dataset)),
            "feature_count": len(selected_features),
            "ordinary_regression": metric_summary(ordinary_df, ordinary_predictions),
            "focus_session": final_focus_summary(dataset, final_predictions),
        },
        "outputs": {
            "candidate_rows_csv": str(CANDIDATE_ROWS_CSV),
            "dataset_csv": str(DATASET_CSV),
            "eval_csv": str(EVAL_CSV),
            "predictions_csv": str(PREDICTIONS_CSV),
            "report_md": str(REPORT_MD),
            "model_dir": str(MODEL_DIR),
        },
        "changed_app_artifacts": False,
        "changed_studs_live": False,
        "changed_video_model": False,
    }
    (MODEL_DIR / "report.json").write_text(json.dumps(report, indent=2, default=json_default), encoding="utf-8")
    write_markdown(REPORT_MD, report)

    focus = report["final_selected_model"]["focus_session"]
    print(f"Wrote {CANDIDATE_ROWS_CSV}")
    print(f"Wrote {DATASET_CSV}")
    print(f"Wrote {EVAL_CSV}")
    print(f"Wrote {PREDICTIONS_CSV}")
    print(f"Wrote {REPORT_MD}")
    print(f"Saved local model to {MODEL_DIR}")
    print(f"rows={len(dataset)} sessions={dataset['session_id'].nunique()} labels={report['label_counts']}")
    print(f"selected_variant={selected_name} feature_count={len(selected_features)}")
    print(
        "focus_baseline_non_target_target_rows={misses} final_correct={correct} "
        "manual_missed_rows={manual} manual_correct={manual_correct}".format(
            misses=focus["baseline_non_target_target_rows"],
            correct=focus["final_correct_on_baseline_non_target_target_rows"],
            manual=focus["manual_missed_marker_rows"],
            manual_correct=focus["final_correct_manual_missed_marker_rows"],
        )
    )


if __name__ == "__main__":
    main()
