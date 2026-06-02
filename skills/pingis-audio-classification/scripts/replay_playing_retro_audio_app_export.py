"""
Replay the exported playing-retro app model on saved Review candidates.

This is a T0010 controlled replay path. It reads the separate
`playing_retro_audio_model.json` app export, not the joblib classifier and not
the normal Collector `audio_model.json`.

Run:
  python skills/pingis-audio-classification/scripts/replay_playing_retro_audio_app_export.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluate_playing_retro_audio_multi_window import (
    MULTI_WINDOW_DATASET_CSV,
    build_multi_window_dataset,
    read_or_build_candidate_rows,
)
from train_playing_retro_audio import EVAL_DIR, TARGET_LABELS

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_APP_MODEL = ROOT_DIR / "apps" / "collector" / "src" / "models" / "playing_retro_audio_model.json"
DEFAULT_SESSIONS = [
    "audio_session_2026-05-28_002",
    "audio_session_2026-05-29_001",
    "audio_session_2026-05-29_002",
]
PREDICTIONS_CSV = EVAL_DIR / "playing_retro_audio_t0010_app_export_replay_predictions.csv"
EVAL_CSV = EVAL_DIR / "playing_retro_audio_t0010_app_export_replay_eval.csv"
REPORT_JSON = EVAL_DIR / "playing_retro_audio_t0010_app_export_replay_report.json"
REPORT_MD = EVAL_DIR / "playing_retro_audio_t0010_app_export_replay_report.md"


def is_leaf(node: list[float], n_classes: int) -> bool:
    if len(node) != n_classes:
        return len(node) != 4
    total = 0.0
    for value in node:
        if value < 0 or value > 1:
            return False
        total += value
    return abs(total - 1.0) < 0.01


def traverse_tree(tree: list[list[float]], scaled_features: np.ndarray, n_classes: int) -> np.ndarray:
    index = 0
    while not is_leaf(tree[index], n_classes):
        feature_index, threshold, left_child, right_child = tree[index]
        index = int(left_child) if scaled_features[int(feature_index)] <= threshold else int(right_child)
    return np.asarray(tree[index], dtype=np.float64)


def predict_app_model(model: dict[str, Any], dataset: pd.DataFrame) -> tuple[list[str], list[float], dict[str, list[float]]]:
    feature_names = model["feature_names"]
    missing_features = [feature for feature in feature_names if feature not in dataset.columns]
    if missing_features:
        raise ValueError(f"Dataset is missing app model features: {missing_features[:10]}")

    raw = dataset[feature_names].fillna(0).to_numpy(dtype=np.float64)
    scaler_mean = np.asarray(model["scaler_mean"], dtype=np.float64)
    scaler_std = np.asarray(model["scaler_std"], dtype=np.float64)
    scaler_std = np.where(scaler_std == 0, 1.0, scaler_std)
    scaled = (raw - scaler_mean) / scaler_std

    labels = model["labels"]
    n_classes = len(labels)
    probability_rows: list[np.ndarray] = []
    for row in scaled:
        prob_sum = np.zeros(n_classes, dtype=np.float64)
        for tree in model["trees"]:
            prob_sum += traverse_tree(tree, row, n_classes)
        probability_rows.append(prob_sum / max(1, len(model["trees"])))

    predictions: list[str] = []
    confidences: list[float] = []
    probabilities: dict[str, list[float]] = {label: [] for label in labels}
    for row in probability_rows:
        max_index = int(np.argmax(row))
        predictions.append(str(labels[max_index]))
        confidences.append(round(float(row[max_index]), 6))
        for index, label in enumerate(labels):
            probabilities[label].append(round(float(row[index]), 6))
    return predictions, confidences, probabilities


def map_saved_candidate_prediction(value: Any) -> str:
    label = str(value or "")
    if label == "racket_contact":
        return "racket_contact"
    if label == "table_bounce":
        return "table_bounce"
    return "non_target"


def metric_row(scope: str, group: str, df: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    truth = df["label"].astype(str).to_numpy()
    predictions = df[prediction_column].astype(str).to_numpy()
    row = {
        "scope": scope,
        "group": group,
        "rows": int(len(df)),
        "accuracy": float(np.mean(truth == predictions)) if len(df) else 0.0,
    }
    for label in TARGET_LABELS:
        truth_mask = truth == label
        pred_mask = predictions == label
        row[f"{label}_rows"] = int(truth_mask.sum())
        row[f"{label}_recall"] = float(np.mean(predictions[truth_mask] == label)) if truth_mask.any() else None
        row[f"pred_{label}"] = int(pred_mask.sum())
    non_target_mask = truth == "non_target"
    row["non_target_predicted_target"] = int(((predictions != "non_target") & non_target_mask).sum())
    row["target_predicted_non_target"] = int(((truth != "non_target") & (predictions == "non_target")).sum())
    return row


def grouped_metric_rows(scope: str, df: pd.DataFrame, prediction_column: str, group_columns: list[str]) -> list[dict[str, Any]]:
    rows = [metric_row(scope, "all", df, prediction_column)]
    for column in group_columns:
        if column not in df.columns:
            continue
        for value in sorted(df[column].fillna("").astype(str).unique()):
            group_df = df[df[column].fillna("").astype(str) == value]
            if group_df.empty:
                continue
            rows.append(metric_row(scope, f"{column}={value or 'unspecified'}", group_df, prediction_column))
    return rows


def load_or_build_dataset(rebuild_dataset: bool) -> pd.DataFrame:
    if rebuild_dataset or not MULTI_WINDOW_DATASET_CSV.exists():
        rows = read_or_build_candidate_rows()
        dataset = build_multi_window_dataset(rows)
        MULTI_WINDOW_DATASET_CSV.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(MULTI_WINDOW_DATASET_CSV, index=False)
        return dataset
    return pd.read_csv(MULTI_WINDOW_DATASET_CSV)


def write_report(
    report_path: Path,
    report: dict[str, Any],
    eval_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> None:
    lines = [
        "# Playing Retro Audio T0010 App Export Replay",
        "",
        "This replay uses the separate `playing_retro_audio_model.json` app export. It does not use or change Collector `audio_model.json`, `audio_contact_model.json`, `studs_live`, APKs, or visible Review behavior.",
        "",
        "## Scope",
        "",
        f"- App model: `{report['app_model']}`",
        f"- Model version: `{report['model_version']}`",
        f"- Sessions: `{', '.join(report['sessions'])}`",
        f"- Saved candidate rows replayed: `{report['candidate_rows']}`",
        f"- Manual missed marker rows not classifiable by saved-candidate replay: `{report['missed_marker_rows']}`",
        "",
        "## Summary",
        "",
        "| Scope | Accuracy | Racket Recall | Table Recall | Non-target Recall | Target->Non-target | Non-target->Target |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scope in ["playing_retro_app_export", "saved_candidate_baseline"]:
        row = eval_df[(eval_df["scope"] == scope) & (eval_df["group"] == "all")].iloc[0]
        lines.append(
            "| `{scope}` | {accuracy:.3f} | {racket:.3f} | {table:.3f} | {non_target:.3f} | {target_miss} | {target_fp} |".format(
                scope=scope,
                accuracy=float(row["accuracy"]),
                racket=float(row["racket_contact_recall"]),
                table=float(row["table_bounce_recall"]),
                non_target=float(row["non_target_recall"]),
                target_miss=int(row["target_predicted_non_target"]),
                target_fp=int(row["non_target_predicted_target"]),
            )
        )

    lines.extend([
        "",
        "## By Session",
        "",
        "| Session | Rows | Accuracy | Racket Recall | Table Recall | Non-target Recall | Predictions R/T/N |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    session_rows = eval_df[
        (eval_df["scope"] == "playing_retro_app_export")
        & eval_df["group"].str.startswith("session_id=")
    ]
    for _, row in session_rows.iterrows():
        session_id = str(row["group"]).split("=", 1)[1]
        lines.append(
            "| `{session}` | {rows} | {accuracy:.3f} | {racket:.3f} | {table:.3f} | {non_target:.3f} | {pred_r}/{pred_t}/{pred_n} |".format(
                session=session_id,
                rows=int(row["rows"]),
                accuracy=float(row["accuracy"]),
                racket=float(row["racket_contact_recall"]),
                table=float(row["table_bounce_recall"]),
                non_target=float(row["non_target_recall"]),
                pred_r=int(row["pred_racket_contact"]),
                pred_t=int(row["pred_table_bounce"]),
                pred_n=int(row["pred_non_target"]),
            )
        )

    lines.extend([
        "",
        "## Close-Gap Behavior",
        "",
        "| Gap Bucket | Rows | Accuracy | Racket Recall | Table Recall | Non-target Recall |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    gap_rows = eval_df[
        (eval_df["scope"] == "playing_retro_app_export")
        & eval_df["group"].str.startswith("close_event_bucket=")
    ]
    for _, row in gap_rows.iterrows():
        bucket = str(row["group"]).split("=", 1)[1]
        lines.append(
            "| `{bucket}` | {rows} | {accuracy:.3f} | {racket} | {table} | {non_target} |".format(
                bucket=bucket,
                rows=int(row["rows"]),
                accuracy=float(row["accuracy"]),
                racket="" if pd.isna(row["racket_contact_recall"]) else f"{float(row['racket_contact_recall']):.3f}",
                table="" if pd.isna(row["table_bounce_recall"]) else f"{float(row['table_bounce_recall']):.3f}",
                non_target="" if pd.isna(row["non_target_recall"]) else f"{float(row['non_target_recall']):.3f}",
            )
        )

    lines.extend([
        "",
        "## Outputs",
        "",
        f"- Predictions CSV: `{PREDICTIONS_CSV.as_posix()}`",
        f"- Evaluation CSV: `{EVAL_CSV.as_posix()}`",
        f"- JSON report: `{REPORT_JSON.as_posix()}`",
        "",
        "## Notes",
        "",
        "- Metrics are candidate-level and use reviewed marker matches as labels.",
        "- This is an app-export sanity replay, not a generalization claim: the exported T0009 model was fit on all selected T0007/T0008 rows, including these sessions.",
        "- The 15 missed marker rows in the target sessions are not replayed because no saved app candidate exists at those timestamps.",
        "- Fixing missed markers requires a later candidate-generation/UI step, not only candidate reclassification.",
        "- `close_event_bucket` and `neighbor_sequence` are reporting metadata only and are not app model features.",
        "",
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay playing-retro app JSON on saved Review candidates.")
    parser.add_argument("--app-model", default=str(DEFAULT_APP_MODEL))
    parser.add_argument("--sessions", nargs="+", default=DEFAULT_SESSIONS)
    parser.add_argument("--rebuild-dataset", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_model_path = Path(args.app_model)
    model = json.loads(app_model_path.read_text(encoding="utf-8"))
    dataset = load_or_build_dataset(args.rebuild_dataset)
    target_sessions = set(args.sessions)
    target_dataset = dataset[dataset["session_id"].astype(str).isin(target_sessions)].copy()
    candidate_df = target_dataset[target_dataset["row_type"].fillna("").astype(str) == "candidate"].copy()
    missed_df = target_dataset[target_dataset["row_type"].fillna("").astype(str) == "missed_marker"].copy()

    predictions, confidences, probabilities = predict_app_model(model, candidate_df)
    candidate_df["playing_retro_prediction"] = predictions
    candidate_df["playing_retro_confidence"] = confidences
    for label, values in probabilities.items():
        candidate_df[f"playing_retro_probability_{label}"] = values
    candidate_df["saved_candidate_prediction"] = candidate_df["candidate_predicted_kind"].map(map_saved_candidate_prediction)

    prediction_columns = [
        "session_id",
        "event_index",
        "wav_filename",
        "candidate_id",
        "anchor_ms",
        "label",
        "playing_retro_prediction",
        "playing_retro_confidence",
        "playing_retro_probability_racket_contact",
        "playing_retro_probability_table_bounce",
        "playing_retro_probability_non_target",
        "saved_candidate_prediction",
        "candidate_predicted_kind",
        "candidate_confidence",
        "source_rule",
        "match_outcome",
        "matched_truth_kind",
        "candidate_to_truth_offset_ms",
        "close_event_bucket",
        "neighbor_sequence",
    ]
    prediction_columns = [column for column in prediction_columns if column in candidate_df.columns]
    PREDICTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    candidate_df[prediction_columns].to_csv(PREDICTIONS_CSV, index=False)

    eval_rows: list[dict[str, Any]] = []
    group_columns = ["session_id", "close_event_bucket", "source_rule", "neighbor_sequence"]
    eval_rows.extend(grouped_metric_rows("playing_retro_app_export", candidate_df, "playing_retro_prediction", group_columns))
    eval_rows.extend(grouped_metric_rows("saved_candidate_baseline", candidate_df, "saved_candidate_prediction", group_columns))
    eval_df = pd.DataFrame(eval_rows)
    EVAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(EVAL_CSV, index=False)

    overall = eval_df[(eval_df["scope"] == "playing_retro_app_export") & (eval_df["group"] == "all")].iloc[0].to_dict()
    baseline = eval_df[(eval_df["scope"] == "saved_candidate_baseline") & (eval_df["group"] == "all")].iloc[0].to_dict()
    report = {
        "app_model": str(app_model_path),
        "model_version": model.get("metadata", {}).get("model_version"),
        "feature_version": model.get("metadata", {}).get("feature_version"),
        "sessions": args.sessions,
        "candidate_rows": int(len(candidate_df)),
        "missed_marker_rows": int(len(missed_df)),
        "missed_marker_labels": missed_df["label"].value_counts().to_dict() if not missed_df.empty else {},
        "overall": overall,
        "saved_candidate_baseline": baseline,
        "outputs": {
            "predictions_csv": str(PREDICTIONS_CSV),
            "eval_csv": str(EVAL_CSV),
            "report_md": str(REPORT_MD),
        },
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_report(REPORT_MD, report, eval_df, candidate_df)

    print(f"Wrote {PREDICTIONS_CSV}")
    print(f"Wrote {EVAL_CSV}")
    print(f"Wrote {REPORT_JSON}")
    print(f"Wrote {REPORT_MD}")
    print(
        "playing_retro_app_export: rows={rows} acc={acc:.3f} racket={racket:.3f} table={table:.3f} non_target={non_target:.3f}".format(
            rows=int(overall["rows"]),
            acc=float(overall["accuracy"]),
            racket=float(overall["racket_contact_recall"]),
            table=float(overall["table_bounce_recall"]),
            non_target=float(overall["non_target_recall"]),
        )
    )
    print(
        "saved_candidate_baseline: rows={rows} acc={acc:.3f} racket={racket:.3f} table={table:.3f} non_target={non_target:.3f}".format(
            rows=int(baseline["rows"]),
            acc=float(baseline["accuracy"]),
            racket=float(baseline["racket_contact_recall"]),
            table=float(baseline["table_bounce_recall"]),
            non_target=float(baseline["non_target_recall"]),
        )
    )
    print(f"missed_marker_rows_not_replayed={len(missed_df)} labels={report['missed_marker_labels']}")


if __name__ == "__main__":
    main()
