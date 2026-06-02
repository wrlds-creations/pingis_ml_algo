"""
Train and evaluate a local playing-retro audio candidate.

This is a separate retro-analysis model path. It does not export to the
Collector app and does not change the live `studs_live` audio model.

Default training data:
- reviewed sessions with both racket and table truth
- saved app `model_candidates` as the primary candidate-centered rows
- manually added reviewed markers when the app missed a truth event

Default regression data:
- raw rows from `audio_dataset.csv` outside dense playing sessions, mapped to
  racket_contact / table_bounce / non_target

Run:
  python skills/pingis-audio-classification/scripts/train_playing_retro_audio.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

from build_playing_retro_candidate_report import (
    MATCH_TOLERANCE_MS,
    base_context,
    build_rows_for_group,
    build_truth_markers,
    candidate_peaks_from_app,
)
from preprocess_audio import TARGET_SR, extract_features, load_audio
from replay_live_bounce import RAW_DIR, extract_live_clip, resolve_wav_path


ROOT_DIR = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT_DIR / "data" / "audio" / "processed"
MODEL_ROOT = ROOT_DIR / "data" / "audio" / "models" / "playing_retro_candidates"
EVAL_DIR = ROOT_DIR / "data" / "audio" / "models" / "evaluations"
AUDIO_DATASET = OUT_DIR / "audio_dataset.csv"

DEFAULT_DATASET_CSV = OUT_DIR / "playing_retro_audio_candidate_dataset.csv"
DEFAULT_ROWS_CSV = OUT_DIR / "playing_retro_audio_candidate_rows_t0005.csv"
DEFAULT_EVAL_CSV = EVAL_DIR / "playing_retro_audio_candidate_eval.csv"
DEFAULT_REPORT_MD = EVAL_DIR / "playing_retro_audio_candidate_report.md"
DEFAULT_MODEL_DIR = MODEL_ROOT / "playing_retro_audio_rf_v2026_06_02_app_candidates_100_200"

EXCLUDED_TRAINING_SESSIONS = {
    "audio_session_2026-05-26_002",
    "audio_session_2026-05-26_003",
    "audio_session_2026-05-26_004",
}

DEFAULT_HOLDOUT_SESSIONS = ["audio_session_2026-05-29_002"]
TARGET_LABELS = ["racket_contact", "table_bounce", "non_target"]
META_COLS = {
    "label",
    "source_label",
    "session_id",
    "event_index",
    "wav_filename",
    "candidate_id",
    "anchor_ms",
    "source_rule",
    "row_type",
    "candidate_status",
    "candidate_predicted_kind",
    "candidate_source",
    "source_config",
    "match_outcome",
    "matched_truth_kind",
    "evaluation_bucket",
    "scenario_id",
    "background_condition",
    "bounce_context",
    "close_event_bucket",
    "neighbor_sequence",
    "sample_weight",
}


def truth_counts(event: dict[str, Any]) -> tuple[int, int]:
    markers = (event.get("review") or {}).get("markers") or []
    truths = build_truth_markers(markers)
    racket = sum(1 for marker in truths if marker.truth_kind == "racket_contact")
    table = sum(1 for marker in truths if marker.truth_kind == "table_bounce")
    return racket, table


def selected_playing_events(min_racket: int, min_table: int) -> list[tuple[Path, int, dict[str, Any]]]:
    selected: list[tuple[Path, int, dict[str, Any]]] = []
    for session_path in sorted(RAW_DIR.glob("audio_session_*.json")):
        session_id = session_path.stem
        if session_id in EXCLUDED_TRAINING_SESSIONS:
            continue
        data = json.loads(session_path.read_text(encoding="utf-8"))
        for event_index, event in enumerate(data.get("events") or []):
            racket, table = truth_counts(event)
            if racket < min_racket or table < min_table:
                continue
            if not event.get("model_candidates"):
                continue
            if resolve_wav_path(session_path, event) is None:
                continue
            selected.append((session_path, event_index, event))
    return selected


def build_candidate_rows(
    events: list[tuple[Path, int, dict[str, Any]]],
    tolerance_ms: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for session_path, event_index, event in events:
        markers = (event.get("review") or {}).get("markers") or []
        truths = build_truth_markers(markers)
        candidates = candidate_peaks_from_app(event)
        context = base_context(session_path, event_index, event)
        config_names = sorted({candidate.source_config for candidate in candidates}) or ["saved_model_candidates"]
        for config_name in config_names:
            config_candidates = [candidate for candidate in candidates if candidate.source_config == config_name]
            rows.extend(build_rows_for_group(
                context=context,
                source="app_saved",
                config=config_name,
                candidates=config_candidates,
                truths=truths,
                tolerance_ms=tolerance_ms,
            ))
    return rows


def label_for_report_row(row: dict[str, Any]) -> str | None:
    truth = str(row.get("matched_truth_kind") or "")
    if truth in {"racket_contact", "table_bounce"}:
        return truth
    if str(row.get("match_outcome") or "") == "false_positive":
        return "non_target"
    return None


def sample_weight_for_row(row: dict[str, Any], label: str) -> float:
    if label == "non_target":
        return 0.60
    if str(row.get("row_type") or "") == "missed_marker":
        return 1.10
    if str(row.get("candidate_status") or "") == "analysis_only":
        return 0.85
    return 1.00


def build_feature_dataset(report_rows: list[dict[str, Any]]) -> pd.DataFrame:
    by_event: dict[tuple[str, int], tuple[Path, dict[str, Any], np.ndarray]] = {}
    dataset_rows: list[dict[str, Any]] = []

    for row in report_rows:
        label = label_for_report_row(row)
        if label is None:
            continue
        session_id = str(row["session_id"])
        event_index = int(row.get("event_index") or 0)
        session_path = RAW_DIR / f"{session_id}.json"
        key = (session_id, event_index)
        if key not in by_event:
            data = json.loads(session_path.read_text(encoding="utf-8"))
            event = data["events"][event_index]
            wav_path = resolve_wav_path(session_path, event)
            if wav_path is None:
                continue
            y, _sr = load_audio(str(wav_path))
            by_event[key] = (wav_path, event, y)

        wav_path, _event, y = by_event[key]
        timestamp = row.get("candidate_timestamp_ms")
        source_rule = "candidate_peak"
        if timestamp == "" or pd.isna(timestamp):
            timestamp = row.get("matched_truth_timestamp_ms")
            source_rule = "manual_missed_marker"
        if timestamp == "" or pd.isna(timestamp):
            continue
        anchor_ms = int(round(float(timestamp)))
        onset_sample = int(round(anchor_ms / 1000.0 * TARGET_SR))
        clip = extract_live_clip(y, onset_sample)
        try:
            features = extract_features(clip, TARGET_SR)
        except Exception:
            continue

        features.update({
            "label": label,
            "source_label": label,
            "session_id": session_id,
            "event_index": event_index,
            "wav_filename": wav_path.name,
            "candidate_id": row.get("candidate_id", ""),
            "anchor_ms": anchor_ms,
            "source_rule": source_rule if source_rule == "manual_missed_marker" else str(row.get("match_outcome") or ""),
            "row_type": row.get("row_type", ""),
            "candidate_status": row.get("candidate_status", ""),
            "candidate_predicted_kind": row.get("candidate_predicted_kind", ""),
            "candidate_source": row.get("candidate_source", ""),
            "source_config": row.get("source_config", ""),
            "match_outcome": row.get("match_outcome", ""),
            "matched_truth_kind": row.get("matched_truth_kind", ""),
            "evaluation_bucket": row.get("evaluation_bucket", ""),
            "scenario_id": row.get("scenario_id", ""),
            "background_condition": row.get("background_condition", ""),
            "bounce_context": row.get("bounce_context", ""),
            "close_event_bucket": row.get("close_event_bucket", ""),
            "neighbor_sequence": row.get("neighbor_sequence", ""),
            "sample_weight": sample_weight_for_row(row, label),
        })
        dataset_rows.append(features)

    return pd.DataFrame(dataset_rows)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in META_COLS]


def train_model(train_df: pd.DataFrame, features: list[str]) -> tuple[RandomForestClassifier, StandardScaler, LabelEncoder]:
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["label"].astype(str).to_numpy())
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[features].fillna(0).to_numpy(dtype=np.float32))
    classifier = RandomForestClassifier(
        n_estimators=350,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    classifier.fit(x_train, y_train, sample_weight=train_df["sample_weight"].astype(float).to_numpy())
    return classifier, scaler, label_encoder


def predict_labels(
    classifier: RandomForestClassifier,
    scaler: StandardScaler,
    label_encoder: LabelEncoder,
    df: pd.DataFrame,
    features: list[str],
) -> np.ndarray:
    x = scaler.transform(df[features].fillna(0).to_numpy(dtype=np.float32))
    return label_encoder.inverse_transform(classifier.predict(x))


def mapped_baseline_prediction(values: pd.Series) -> pd.Series:
    text = values.fillna("").astype(str)
    return text.map(lambda value: value if value in {"racket_contact", "table_bounce"} else "non_target")


def metric_row(scope: str, df: pd.DataFrame, predictions: np.ndarray, group: str) -> dict[str, Any]:
    truth = df["label"].astype(str).to_numpy()
    result = {
        "scope": scope,
        "group": group,
        "rows": int(len(df)),
        "accuracy": float(np.mean(truth == predictions)) if len(df) else 0.0,
    }
    for label in TARGET_LABELS:
        mask = truth == label
        result[f"{label}_rows"] = int(mask.sum())
        result[f"{label}_recall"] = float(np.mean(predictions[mask] == label)) if mask.any() else None
        result[f"pred_{label}"] = int((predictions == label).sum())
    return result


def grouped_metrics(scope: str, df: pd.DataFrame, predictions: np.ndarray, columns: list[str]) -> list[dict[str, Any]]:
    rows = [metric_row(scope, df, predictions, "all")]
    pred_series = pd.Series(predictions, index=df.index)
    for column in columns:
        if column not in df.columns:
            continue
        for value in sorted(df[column].fillna("").astype(str).unique()):
            mask = df[column].fillna("").astype(str) == value
            if int(mask.sum()) == 0:
                continue
            rows.append(metric_row(scope, df[mask], pred_series[mask].to_numpy(), f"{column}={value or 'unspecified'}"))
    return rows


def build_ordinary_regression_dataset(features: list[str]) -> pd.DataFrame:
    if not AUDIO_DATASET.exists():
        return pd.DataFrame()
    df = pd.read_csv(AUDIO_DATASET, low_memory=False)
    if "augmentation" in df.columns:
        df = df[df["augmentation"].fillna("").astype(str) == "none"].copy()
    if "session_id" in df.columns:
        df = df[~df["session_id"].astype(str).isin(EXCLUDED_TRAINING_SESSIONS)].copy()
    if "scenario_id" in df.columns:
        df = df[df["scenario_id"].fillna("").astype(str) != "playing_dense_audio"].copy()
    label_map = {
        "racket_bounce": "racket_contact",
        "table_bounce": "table_bounce",
        "floor_bounce": "non_target",
        "noise": "non_target",
    }
    df["label"] = df["label"].map(label_map)
    df = df[df["label"].isin(TARGET_LABELS)].copy()
    for column in features:
        if column not in df.columns:
            df[column] = 0.0
    return df


def write_report(
    path: Path,
    dataset: pd.DataFrame,
    holdout_sessions: list[str],
    eval_rows: list[dict[str, Any]],
    report: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Playing Retro Audio Candidate Report",
        "",
        "This is a local T0005 candidate. No Collector app model JSON, APK, or live detector behavior was changed.",
        "",
        "## Training Choice",
        "",
        "- Trained from all matchable saved app candidate peaks, not only review-relevant peaks.",
        "- Added manually reviewed missed racket/table markers as candidate-centered rows at the marker timestamp.",
        "- Labeled unmatched app candidates as `non_target` with lower sample weight.",
        "- Replay candidates were used for T0004 diagnostics, not as T0005 training rows, to avoid multiplying the same WAV by timing config.",
        "- Ordinary up/down bounce is reported as a regression slice only; this candidate is not promoted into `studs_live`.",
        "",
        "## Dataset",
        "",
        f"- Rows: `{len(dataset)}`",
        f"- Sessions: `{dataset['session_id'].nunique()}`",
        f"- Labels: `{dataset['label'].value_counts().to_dict()}`",
        f"- Source rules: `{dataset['source_rule'].value_counts().to_dict()}`",
        f"- Close-event buckets: `{dataset['close_event_bucket'].value_counts().to_dict()}`",
        f"- Holdout sessions: `{holdout_sessions}`",
        "",
        "## Key Metrics",
        "",
        f"- Holdout accuracy: `{report['holdout_accuracy']:.3f}`",
        f"- Old app prediction holdout accuracy: `{report['old_app_holdout_accuracy']:.3f}`",
        f"- Ordinary regression accuracy: `{report.get('ordinary_regression_accuracy', 0.0):.3f}`",
        f"- Ordinary regression rows: `{report.get('ordinary_regression_rows', 0)}`",
        "",
        "## Output Files",
        "",
        f"- Dataset CSV: `{DEFAULT_DATASET_CSV.as_posix()}`",
        f"- Evaluation CSV: `{DEFAULT_EVAL_CSV.as_posix()}`",
        f"- Model dir: `{DEFAULT_MODEL_DIR.as_posix()}`",
        "",
    ]
    for row in eval_rows[:30]:
        if row["scope"] not in {"holdout", "ordinary_regression"} or row["group"] != "all":
            continue
        lines.extend([
            f"### {row['scope']}",
            "",
            f"- rows: `{row['rows']}`",
            f"- accuracy: `{row['accuracy']:.3f}`",
            f"- racket recall: `{row.get('racket_contact_recall')}`",
            f"- table recall: `{row.get('table_bounce_recall')}`",
            f"- non-target recall: `{row.get('non_target_recall')}`",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train local playing-retro audio RF candidate.")
    parser.add_argument("--min-racket", type=int, default=5)
    parser.add_argument("--min-table", type=int, default=10)
    parser.add_argument("--match-tolerance-ms", type=int, default=MATCH_TOLERANCE_MS)
    parser.add_argument("--holdout-session", action="append", default=DEFAULT_HOLDOUT_SESSIONS)
    parser.add_argument("--dataset-csv", default=str(DEFAULT_DATASET_CSV))
    parser.add_argument("--candidate-rows-csv", default=str(DEFAULT_ROWS_CSV))
    parser.add_argument("--eval-csv", default=str(DEFAULT_EVAL_CSV))
    parser.add_argument("--report-md", default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = selected_playing_events(args.min_racket, args.min_table)
    if not events:
        raise SystemExit("No playing-retro training events selected.")

    report_rows = build_candidate_rows(events, args.match_tolerance_ms)
    candidate_rows_csv = Path(args.candidate_rows_csv)
    candidate_rows_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report_rows).to_csv(candidate_rows_csv, index=False)

    dataset = build_feature_dataset(report_rows)
    if dataset.empty:
        raise SystemExit("No training dataset rows produced.")
    dataset_csv = Path(args.dataset_csv)
    dataset_csv.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(dataset_csv, index=False)

    features = feature_columns(dataset)
    holdout_sessions = [session for session in args.holdout_session if session in set(dataset["session_id"])]
    if not holdout_sessions:
        holdout_sessions = [str(sorted(dataset["session_id"].unique())[-1])]
    train_df = dataset[~dataset["session_id"].isin(holdout_sessions)].copy()
    holdout_df = dataset[dataset["session_id"].isin(holdout_sessions)].copy()
    if train_df.empty or holdout_df.empty:
        raise SystemExit("Need both train and holdout rows.")

    holdout_model, holdout_scaler, holdout_encoder = train_model(train_df, features)
    holdout_pred = predict_labels(holdout_model, holdout_scaler, holdout_encoder, holdout_df, features)

    baseline_candidate_df = holdout_df[holdout_df["source_rule"] != "manual_missed_marker"].copy()
    baseline_pred = mapped_baseline_prediction(baseline_candidate_df["candidate_predicted_kind"]).to_numpy()

    final_model, final_scaler, final_encoder = train_model(dataset, features)
    all_pred = predict_labels(final_model, final_scaler, final_encoder, dataset, features)
    ordinary_df = build_ordinary_regression_dataset(features)
    ordinary_pred = np.array([])
    if not ordinary_df.empty:
        ordinary_pred = predict_labels(final_model, final_scaler, final_encoder, ordinary_df, features)

    eval_rows: list[dict[str, Any]] = []
    eval_rows.extend(grouped_metrics(
        "holdout",
        holdout_df,
        holdout_pred,
        ["session_id", "evaluation_bucket", "close_event_bucket", "source_rule"],
    ))
    if not baseline_candidate_df.empty:
        eval_rows.extend(grouped_metrics(
            "holdout_old_app_prediction",
            baseline_candidate_df,
            baseline_pred,
            ["session_id", "evaluation_bucket", "close_event_bucket"],
        ))
    eval_rows.extend(grouped_metrics(
        "final_model_all_training_rows",
        dataset,
        all_pred,
        ["session_id", "evaluation_bucket", "close_event_bucket"],
    ))
    if not ordinary_df.empty:
        eval_rows.extend(grouped_metrics(
            "ordinary_regression",
            ordinary_df,
            ordinary_pred,
            ["scenario_id", "background_condition"],
        ))

    eval_csv = Path(args.eval_csv)
    eval_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(eval_rows).to_csv(eval_csv, index=False)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_dir / "playing_retro_audio_rf_classifier.pkl")
    joblib.dump(final_scaler, model_dir / "playing_retro_audio_feature_scaler.pkl")
    joblib.dump(final_encoder, model_dir / "playing_retro_audio_label_encoder.pkl")
    joblib.dump(features, model_dir / "playing_retro_audio_feature_cols.pkl")

    holdout_report = classification_report(
        holdout_df["label"],
        holdout_pred,
        labels=TARGET_LABELS,
        zero_division=0,
        output_dict=True,
    )
    ordinary_accuracy = float(np.mean(ordinary_df["label"].to_numpy() == ordinary_pred)) if not ordinary_df.empty else 0.0
    report = {
        "model_id": model_dir.name,
        "training_choice": "all_matchable_saved_app_candidate_peaks_plus_manual_missed_markers",
        "rows": int(len(dataset)),
        "sessions": sorted(dataset["session_id"].unique().tolist()),
        "labels": {str(k): int(v) for k, v in dataset["label"].value_counts().to_dict().items()},
        "source_rules": {str(k): int(v) for k, v in dataset["source_rule"].value_counts().to_dict().items()},
        "close_event_buckets": {str(k): int(v) for k, v in dataset["close_event_bucket"].value_counts().to_dict().items()},
        "holdout_sessions": holdout_sessions,
        "holdout_accuracy": float(np.mean(holdout_df["label"].to_numpy() == holdout_pred)),
        "holdout_report": holdout_report,
        "holdout_confusion_matrix": confusion_matrix(
            holdout_df["label"],
            holdout_pred,
            labels=TARGET_LABELS,
        ).tolist(),
        "old_app_holdout_accuracy": (
            float(np.mean(baseline_candidate_df["label"].to_numpy() == baseline_pred))
            if not baseline_candidate_df.empty
            else None
        ),
        "ordinary_regression_rows": int(len(ordinary_df)),
        "ordinary_regression_accuracy": ordinary_accuracy,
        "outputs": {
            "dataset_csv": str(dataset_csv),
            "candidate_rows_csv": str(candidate_rows_csv),
            "eval_csv": str(eval_csv),
            "report_md": str(args.report_md),
            "model_dir": str(model_dir),
        },
    }
    (model_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_report(Path(args.report_md), dataset, holdout_sessions, eval_rows, report)

    print(f"Wrote {candidate_rows_csv}")
    print(f"Wrote {dataset_csv}")
    print(f"Wrote {eval_csv}")
    print(f"Wrote {args.report_md}")
    print(f"Saved local model to {model_dir}")
    print(f"rows={len(dataset)} sessions={dataset['session_id'].nunique()} labels={dataset['label'].value_counts().to_dict()}")
    print(f"holdout_sessions={holdout_sessions} holdout_accuracy={report['holdout_accuracy']:.3f}")
    print(f"old_app_holdout_accuracy={report['old_app_holdout_accuracy']:.3f}")
    print(f"ordinary_regression_rows={len(ordinary_df)} ordinary_regression_accuracy={ordinary_accuracy:.3f}")


if __name__ == "__main__":
    main()
