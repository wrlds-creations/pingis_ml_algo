"""
Evaluate real multi-window and candidate-context features for spel_retro_audio.

This is a local T0007 experiment. It does not export Collector app model JSON,
build an APK, or change `studs_live`.

Run:
  python skills/pingis-audio-classification/scripts/evaluate_playing_retro_audio_multi_window.py
  python skills/pingis-audio-classification/scripts/evaluate_playing_retro_audio_multi_window.py --rebuild-dataset
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from train_playing_retro_audio import (
    DEFAULT_HOLDOUT_SESSIONS,
    DEFAULT_ROWS_CSV,
    EVAL_DIR,
    MODEL_ROOT,
    OUT_DIR,
    TARGET_LABELS,
    build_candidate_rows,
    build_ordinary_regression_dataset,
    feature_columns,
    grouped_metrics,
    label_for_report_row,
    sample_weight_for_row,
    selected_playing_events,
)
from preprocess_audio import TARGET_SR, extract_features, load_audio
from replay_live_bounce import RAW_DIR, resolve_wav_path


MULTI_WINDOW_DATASET_CSV = OUT_DIR / "playing_retro_audio_multi_window_dataset_t0007.csv"
EVAL_CSV = EVAL_DIR / "playing_retro_audio_t0007_multi_window_eval.csv"
HOLDOUT_PREDICTIONS_CSV = EVAL_DIR / "playing_retro_audio_t0007_holdout_predictions.csv"
REPORT_MD = EVAL_DIR / "playing_retro_audio_t0007_multi_window_report.md"
MODEL_DIR = MODEL_ROOT / "playing_retro_audio_rf_v2026_06_02_multi_window_context"

WINDOWS = [
    ("tight", 60, 140),
    ("normal", 100, 200),
    ("wide", 160, 320),
]

TRUTH_DERIVED_META_COLS = {
    "close_event_bucket",
    "neighbor_sequence",
    "matched_truth_kind",
    "matched_truth_timestamp_ms",
    "candidate_to_truth_offset_ms",
    "nearest_truth_marker_id",
    "nearest_truth_kind",
    "nearest_truth_timestamp_ms",
    "nearest_truth_offset_ms",
    "truth_nearest_neighbor_ms",
}

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
    "evaluation_bucket",
    "scenario_id",
    "background_condition",
    "bounce_context",
    "sample_weight",
    *TRUTH_DERIVED_META_COLS,
}


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    feature_mode: str
    n_estimators: int
    min_samples_leaf: int
    class_weight: str | None
    racket_weight: float = 1.0
    wrong_racket_as_table_weight: float = 1.0
    missed_marker_weight: float = 1.0
    tight_racket_weight: float = 1.0
    table_weight: float = 1.0
    non_target_weight: float = 1.0
    false_positive_weight: float = 1.0


VARIANTS = [
    Variant(
        name="multi_window_balanced",
        description="tight+normal+wide windows without candidate context; balanced RF.",
        feature_mode="windows",
        n_estimators=450,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
    ),
    Variant(
        name="multi_window_unbalanced",
        description="tight+normal+wide windows without candidate context; RF without class_weight.",
        feature_mode="windows",
        n_estimators=450,
        min_samples_leaf=2,
        class_weight=None,
    ),
    Variant(
        name="multi_window_context",
        description="multi-window features plus non-leaky candidate timing/context features.",
        feature_mode="context",
        n_estimators=450,
        min_samples_leaf=2,
        class_weight=None,
    ),
    Variant(
        name="multi_window_context_safe_weighted",
        description="multi-window/context plus mild T0006-style hard-racket weighting.",
        feature_mode="context",
        n_estimators=450,
        min_samples_leaf=2,
        class_weight=None,
        racket_weight=1.10,
        wrong_racket_as_table_weight=1.20,
        missed_marker_weight=1.15,
        tight_racket_weight=1.10,
    ),
    Variant(
        name="multi_window_context_racket_weighted",
        description="multi-window/context plus stronger racket weighting; included as a tradeoff probe.",
        feature_mode="context",
        n_estimators=450,
        min_samples_leaf=3,
        class_weight=None,
        racket_weight=1.25,
        wrong_racket_as_table_weight=1.45,
        missed_marker_weight=1.30,
        tight_racket_weight=1.20,
        table_weight=1.03,
        non_target_weight=1.10,
        false_positive_weight=1.20,
    ),
]

REFERENCE_METRICS = {
    "t0005_baseline": {
        "accuracy": 0.7590361445783133,
        "racket_contact_recall": 0.6037735849056604,
        "table_bounce_recall": 0.9243697478991597,
        "non_target_recall": 0.625,
    },
    "t0006_safe_racket_weighted": {
        "accuracy": 0.7710843373493976,
        "racket_contact_recall": 0.6226415094339622,
        "table_bounce_recall": 0.9327731092436975,
        "non_target_recall": 0.625,
    },
}


def read_or_build_candidate_rows() -> pd.DataFrame:
    if DEFAULT_ROWS_CSV.exists():
        return pd.read_csv(DEFAULT_ROWS_CSV)
    events = selected_playing_events(min_racket=5, min_table=10)
    return pd.DataFrame(build_candidate_rows(events, tolerance_ms=80))


def timestamp_value(row: dict[str, Any] | pd.Series) -> float | None:
    timestamp = row.get("candidate_timestamp_ms")
    if timestamp == "" or pd.isna(timestamp):
        timestamp = row.get("matched_truth_timestamp_ms")
    if timestamp == "" or pd.isna(timestamp):
        return None
    return float(timestamp)


def extract_window(y: np.ndarray, anchor_ms: int, before_ms: int, after_ms: int) -> np.ndarray:
    length = int(round((before_ms + after_ms) / 1000.0 * TARGET_SR))
    clip = np.zeros(length, dtype=np.float32)
    anchor_sample = int(round(anchor_ms / 1000.0 * TARGET_SR))
    start = anchor_sample - int(round(before_ms / 1000.0 * TARGET_SR))
    end = anchor_sample + int(round(after_ms / 1000.0 * TARGET_SR))
    src_start = max(0, start)
    src_end = min(len(y), end)
    dst_start = src_start - start
    if src_end > src_start:
        clip[dst_start:dst_start + (src_end - src_start)] = y[src_start:src_end]
    return clip


def prefixed_features(features: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in features.items()}


def candidate_timestamps_by_event(rows: pd.DataFrame) -> dict[tuple[str, int, str], list[int]]:
    result: dict[tuple[str, int, str], list[int]] = {}
    candidate_rows = rows[rows["candidate_timestamp_ms"].notna()].copy()
    candidate_rows = candidate_rows[candidate_rows["candidate_timestamp_ms"].astype(str) != ""]
    for key, group in candidate_rows.groupby(["session_id", "event_index", "source_config"]):
        timestamps = sorted({int(round(float(value))) for value in group["candidate_timestamp_ms"].tolist()})
        result[(str(key[0]), int(key[1]), str(key[2]))] = timestamps
    return result


def context_features(anchor_ms: int, candidate_timestamps: list[int], is_saved_candidate: bool) -> dict[str, float]:
    prev_values = [anchor_ms - ts for ts in candidate_timestamps if ts < anchor_ms]
    next_values = [ts - anchor_ms for ts in candidate_timestamps if ts > anchor_ms]
    prev_gap = min(prev_values) if prev_values else None
    next_gap = min(next_values) if next_values else None
    nearest_gap = min([gap for gap in [prev_gap, next_gap] if gap is not None], default=None)
    nearest_index = 0
    if candidate_timestamps:
        nearest_index = int(np.argmin([abs(anchor_ms - ts) for ts in candidate_timestamps]))
    count = len(candidate_timestamps)

    def clipped_gap(value: int | None) -> float:
        return 1.0 if value is None else float(min(value, 1000) / 1000.0)

    return {
        "ctx_is_saved_candidate": 1.0 if is_saved_candidate else 0.0,
        "ctx_candidate_count_log": float(np.log1p(count)),
        "ctx_candidate_index_norm": float(nearest_index / max(1, count - 1)) if count else 0.0,
        "ctx_has_prev_candidate": 1.0 if prev_gap is not None else 0.0,
        "ctx_has_next_candidate": 1.0 if next_gap is not None else 0.0,
        "ctx_prev_gap_1000": clipped_gap(prev_gap),
        "ctx_next_gap_1000": clipped_gap(next_gap),
        "ctx_nearest_gap_1000": clipped_gap(nearest_gap),
        "ctx_density_150ms": float(sum(1 for ts in candidate_timestamps if abs(anchor_ms - ts) <= 150)),
        "ctx_density_300ms": float(sum(1 for ts in candidate_timestamps if abs(anchor_ms - ts) <= 300)),
        "ctx_density_600ms": float(sum(1 for ts in candidate_timestamps if abs(anchor_ms - ts) <= 600)),
    }


def build_multi_window_dataset(candidate_rows: pd.DataFrame) -> pd.DataFrame:
    timestamps_by_event = candidate_timestamps_by_event(candidate_rows)
    by_event: dict[tuple[str, int], tuple[Path, np.ndarray]] = {}
    dataset_rows: list[dict[str, Any]] = []

    for row in candidate_rows.to_dict("records"):
        label = label_for_report_row(row)
        if label is None:
            continue
        timestamp = timestamp_value(row)
        if timestamp is None:
            continue
        anchor_ms = int(round(timestamp))
        session_id = str(row["session_id"])
        event_index = int(row.get("event_index") or 0)
        event_key = (session_id, event_index)
        if event_key not in by_event:
            session_path = RAW_DIR / f"{session_id}.json"
            data = json.loads(session_path.read_text(encoding="utf-8"))
            event = data["events"][event_index]
            wav_path = resolve_wav_path(session_path, event)
            if wav_path is None:
                continue
            y, _sr = load_audio(str(wav_path))
            by_event[event_key] = (wav_path, y)

        wav_path, y = by_event[event_key]
        feature_row: dict[str, Any] = {}
        for window_name, before_ms, after_ms in WINDOWS:
            clip = extract_window(y, anchor_ms, before_ms, after_ms)
            try:
                feature_row.update(prefixed_features(extract_features(clip, TARGET_SR), window_name))
            except Exception:
                continue

        candidate_timestamps = timestamps_by_event.get(
            (session_id, event_index, str(row.get("source_config") or "")),
            [],
        )
        is_saved_candidate = str(row.get("row_type") or "") == "candidate"
        feature_row.update(context_features(anchor_ms, candidate_timestamps, is_saved_candidate))

        # Keep truth-derived fields only as metadata for reporting, never as features.
        feature_row.update({
            "label": label,
            "source_label": label,
            "session_id": session_id,
            "event_index": event_index,
            "wav_filename": wav_path.name,
            "candidate_id": row.get("candidate_id", ""),
            "anchor_ms": anchor_ms,
            "source_rule": (
                "manual_missed_marker"
                if str(row.get("row_type") or "") == "missed_marker"
                else str(row.get("match_outcome") or "")
            ),
            "row_type": row.get("row_type", ""),
            "candidate_status": row.get("candidate_status", ""),
            "candidate_predicted_kind": row.get("candidate_predicted_kind", ""),
            "candidate_source": row.get("candidate_source", ""),
            "source_config": row.get("source_config", ""),
            "match_outcome": row.get("match_outcome", ""),
            "matched_truth_kind": row.get("matched_truth_kind", ""),
            "matched_truth_timestamp_ms": row.get("matched_truth_timestamp_ms", ""),
            "candidate_to_truth_offset_ms": row.get("candidate_to_truth_offset_ms", ""),
            "nearest_truth_marker_id": row.get("nearest_truth_marker_id", ""),
            "nearest_truth_kind": row.get("nearest_truth_kind", ""),
            "nearest_truth_timestamp_ms": row.get("nearest_truth_timestamp_ms", ""),
            "nearest_truth_offset_ms": row.get("nearest_truth_offset_ms", ""),
            "truth_nearest_neighbor_ms": row.get("truth_nearest_neighbor_ms", ""),
            "evaluation_bucket": row.get("evaluation_bucket", ""),
            "scenario_id": row.get("scenario_id", ""),
            "background_condition": row.get("background_condition", ""),
            "bounce_context": row.get("bounce_context", ""),
            "close_event_bucket": row.get("close_event_bucket", ""),
            "neighbor_sequence": row.get("neighbor_sequence", ""),
            "sample_weight": sample_weight_for_row(row, label),
        })
        dataset_rows.append(feature_row)

    return pd.DataFrame(dataset_rows)


def feature_columns_for_mode(dataset: pd.DataFrame, mode: str) -> list[str]:
    all_features = [column for column in dataset.columns if column not in META_COLS]
    if mode == "windows":
        return [column for column in all_features if not column.startswith("ctx_")]
    if mode == "context":
        return all_features
    raise ValueError(f"Unknown feature mode: {mode}")


def adjusted_weights(df: pd.DataFrame, variant: Variant) -> pd.Series:
    weights = df["sample_weight"].astype(float).copy()
    close_bucket = df["close_event_bucket"].fillna("").astype(str)
    weights[df["label"] == "racket_contact"] *= variant.racket_weight
    weights[df["source_rule"] == "wrong_class_racket_as_table"] *= variant.wrong_racket_as_table_weight
    weights[df["source_rule"] == "manual_missed_marker"] *= variant.missed_marker_weight
    weights[
        (df["label"] == "racket_contact")
        & close_bucket.isin(["under_80ms", "80_119ms", "120_179ms"])
    ] *= variant.tight_racket_weight
    weights[df["label"] == "table_bounce"] *= variant.table_weight
    weights[df["label"] == "non_target"] *= variant.non_target_weight
    weights[df["source_rule"] == "false_positive"] *= variant.false_positive_weight
    return weights


def train_variant(
    train_df: pd.DataFrame,
    features: list[str],
    variant: Variant,
) -> tuple[RandomForestClassifier, StandardScaler, LabelEncoder]:
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["label"].astype(str).to_numpy())
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[features].fillna(0).to_numpy(dtype=np.float32))
    classifier = RandomForestClassifier(
        n_estimators=variant.n_estimators,
        min_samples_leaf=variant.min_samples_leaf,
        class_weight=variant.class_weight,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    classifier.fit(x_train, y_train, sample_weight=adjusted_weights(train_df, variant).to_numpy())
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


def metric_summary(df: pd.DataFrame, predictions: np.ndarray) -> dict[str, Any]:
    truth = df["label"].astype(str).to_numpy()
    result: dict[str, Any] = {
        "rows": int(len(df)),
        "accuracy": float(np.mean(truth == predictions)),
    }
    for label in TARGET_LABELS:
        mask = truth == label
        result[f"{label}_rows"] = int(mask.sum())
        result[f"{label}_recall"] = float(np.mean(predictions[mask] == label)) if mask.any() else None
        result[f"pred_{label}"] = int((predictions == label).sum())
    return result


def ordinary_fallback_dataset(features: list[str], base_features: list[str]) -> pd.DataFrame:
    ordinary_df = build_ordinary_regression_dataset(base_features)
    projected: dict[str, Any] = {}
    for feature in features:
        if feature.startswith("ctx_"):
            projected[feature] = 0.0
            continue
        if "_" not in feature:
            projected[feature] = 0.0
            continue
        _window, base_name = feature.split("_", 1)
        projected[feature] = ordinary_df[base_name] if base_name in ordinary_df.columns else 0.0
    return pd.concat([ordinary_df, pd.DataFrame(projected, index=ordinary_df.index)], axis=1).copy()


def add_predictions(
    rows: list[pd.DataFrame],
    variant: Variant,
    holdout_df: pd.DataFrame,
    predictions: np.ndarray,
) -> None:
    pred_df = holdout_df[[
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
        "ctx_prev_gap_1000",
        "ctx_next_gap_1000",
        "ctx_density_300ms",
    ]].copy()
    pred_df.insert(0, "variant", variant.name)
    pred_df["prediction"] = predictions
    pred_df["correct"] = pred_df["label"].astype(str) == pred_df["prediction"].astype(str)
    rows.append(pred_df)


def variant_passes_gate(summary: dict[str, Any]) -> bool:
    holdout = summary["holdout"]
    return (
        holdout["racket_contact_recall"] > REFERENCE_METRICS["t0006_safe_racket_weighted"]["racket_contact_recall"]
        and holdout["table_bounce_recall"] >= REFERENCE_METRICS["t0006_safe_racket_weighted"]["table_bounce_recall"]
        and holdout["non_target_recall"] >= REFERENCE_METRICS["t0006_safe_racket_weighted"]["non_target_recall"]
    )


def choose_selected_variant(report_variants: list[dict[str, Any]]) -> str | None:
    passing = [item for item in report_variants if variant_passes_gate(item)]
    if not passing:
        return None
    passing.sort(
        key=lambda item: (
            item["holdout"]["racket_contact_recall"],
            item["holdout"]["accuracy"],
            item["ordinary_regression"]["accuracy"],
        ),
        reverse=True,
    )
    return str(passing[0]["name"])


def write_report(path: Path, report: dict[str, Any], eval_df: pd.DataFrame) -> None:
    lines = [
        "# Playing Retro Audio T0007 Multi-Window Report",
        "",
        "This is a local experiment. No Collector app model JSON, APK, or `studs_live` behavior was changed.",
        "",
        "## Feature Set",
        "",
        "- Real WAV features are extracted per candidate at `tight -60/+140 ms`, `normal -100/+200 ms`, and `wide -160/+320 ms`.",
        "- Candidate-context features are computed only from app candidate timestamps: previous/next gap, nearest gap, candidate density, and relative candidate index.",
        "- Truth-derived fields such as `close_event_bucket` and `neighbor_sequence` are kept only for reporting and are excluded from model features.",
        "- Ordinary regression uses a fallback projection of existing ordinary features into each window because older ordinary rows do not all carry raw event timestamps.",
        "- Ordinary fallback metrics are advisory for regression risk only; this model family remains playing-retro-only and must not affect `studs_live`.",
        "",
        "## Decision",
        "",
    ]
    selected = report.get("selected_variant")
    if selected:
        lines.extend([
            f"- Selected local candidate: `{selected}`",
            f"- Model dir: `{MODEL_DIR.as_posix()}`",
            "- Status: local candidate only; not exported to the app.",
        ])
    else:
        lines.extend([
            "- Selected local candidate: `none`",
            "- Status: no multi-window/context variant passed the T0007 playing-retro gate.",
        ])
    lines.extend([
        "",
        "## Holdout Comparison",
        "",
        "| Variant | Accuracy | Racket Recall | Table Recall | Non-target Recall | Ordinary Accuracy |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| `t0005_baseline` | {REFERENCE_METRICS['t0005_baseline']['accuracy']:.3f} | "
            f"{REFERENCE_METRICS['t0005_baseline']['racket_contact_recall']:.3f} | "
            f"{REFERENCE_METRICS['t0005_baseline']['table_bounce_recall']:.3f} | "
            f"{REFERENCE_METRICS['t0005_baseline']['non_target_recall']:.3f} | `see T0005` |"
        ),
        (
            f"| `t0006_safe_racket_weighted` | {REFERENCE_METRICS['t0006_safe_racket_weighted']['accuracy']:.3f} | "
            f"{REFERENCE_METRICS['t0006_safe_racket_weighted']['racket_contact_recall']:.3f} | "
            f"{REFERENCE_METRICS['t0006_safe_racket_weighted']['table_bounce_recall']:.3f} | "
            f"{REFERENCE_METRICS['t0006_safe_racket_weighted']['non_target_recall']:.3f} | `see T0006` |"
        ),
    ])
    for item in report["variants"]:
        holdout = item["holdout"]
        ordinary = item["ordinary_regression"]
        lines.append(
            f"| `{item['name']}` | {holdout['accuracy']:.3f} | "
            f"{holdout['racket_contact_recall']:.3f} | {holdout['table_bounce_recall']:.3f} | "
            f"{holdout['non_target_recall']:.3f} | {ordinary['accuracy']:.3f} |"
        )
    lines.extend([
        "",
        "## Source-Rule Findings",
        "",
    ])
    source_rows = eval_df[(eval_df["scope"] == "holdout") & eval_df["group"].str.startswith("source_rule=")]
    for item in report["variants"]:
        wrong = source_rows[
            (source_rows["variant"] == item["name"])
            & (source_rows["group"] == "source_rule=wrong_class_racket_as_table")
        ]
        matched_table = source_rows[
            (source_rows["variant"] == item["name"])
            & (source_rows["group"] == "source_rule=matched_table")
        ]
        if wrong.empty or matched_table.empty:
            continue
        lines.append(
            f"- `{item['name']}`: wrong-class racket-as-table recall "
            f"`{float(wrong.iloc[0]['racket_contact_recall']):.3f}`, matched-table recall "
            f"`{float(matched_table.iloc[0]['table_bounce_recall']):.3f}`"
        )
    lines.extend([
        "",
        "## Outputs",
        "",
        f"- Multi-window dataset CSV: `{MULTI_WINDOW_DATASET_CSV.as_posix()}`",
        f"- Evaluation CSV: `{EVAL_CSV.as_posix()}`",
        f"- Holdout predictions CSV: `{HOLDOUT_PREDICTIONS_CSV.as_posix()}`",
        f"- JSON report: `{(MODEL_DIR / 'report.json').as_posix()}`",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate T0007 multi-window playing-retro audio variants.")
    parser.add_argument("--rebuild-dataset", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rebuild_dataset or not MULTI_WINDOW_DATASET_CSV.exists():
        candidate_rows = read_or_build_candidate_rows()
        dataset = build_multi_window_dataset(candidate_rows)
        MULTI_WINDOW_DATASET_CSV.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(MULTI_WINDOW_DATASET_CSV, index=False)
    else:
        dataset = pd.read_csv(MULTI_WINDOW_DATASET_CSV)

    base_feature_names = sorted({column.split("_", 1)[1] for column in dataset.columns if column.startswith("normal_")})
    holdout_sessions = [session for session in DEFAULT_HOLDOUT_SESSIONS if session in set(dataset["session_id"])]
    train_df = dataset[~dataset["session_id"].isin(holdout_sessions)].copy()
    holdout_df = dataset[dataset["session_id"].isin(holdout_sessions)].copy()

    eval_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    report: dict[str, Any] = {
        "dataset_rows": int(len(dataset)),
        "holdout_sessions": holdout_sessions,
        "windows": [{"name": name, "before_ms": before, "after_ms": after} for name, before, after in WINDOWS],
        "variants": [],
    }

    for variant in VARIANTS:
        features = feature_columns_for_mode(dataset, variant.feature_mode)
        ordinary_df = ordinary_fallback_dataset(features, base_feature_names)
        classifier, scaler, label_encoder = train_variant(train_df, features, variant)
        holdout_pred = predict_labels(classifier, scaler, label_encoder, holdout_df, features)
        ordinary_pred = predict_labels(classifier, scaler, label_encoder, ordinary_df, features)

        add_predictions(prediction_rows, variant, holdout_df, holdout_pred)
        eval_rows.extend(
            {"variant": variant.name, **row}
            for row in grouped_metrics(
                "holdout",
                holdout_df,
                holdout_pred,
                ["session_id", "evaluation_bucket", "close_event_bucket", "source_rule"],
            )
        )
        eval_rows.extend(
            {"variant": variant.name, **row}
            for row in grouped_metrics(
                "ordinary_regression",
                ordinary_df,
                ordinary_pred,
                ["scenario_id", "background_condition"],
            )
        )

        summary = {
            "name": variant.name,
            "description": variant.description,
            "config": variant.__dict__,
            "feature_count": len(features),
            "holdout": metric_summary(holdout_df, holdout_pred),
            "ordinary_regression": metric_summary(ordinary_df, ordinary_pred),
        }
        report["variants"].append(summary)

    report["selected_variant"] = choose_selected_variant(report["variants"])
    eval_df = pd.DataFrame(eval_rows)
    EVAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(EVAL_CSV, index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(HOLDOUT_PREDICTIONS_CSV, index=False)

    selected_name = report["selected_variant"]
    if selected_name:
        selected_variant = next(variant for variant in VARIANTS if variant.name == selected_name)
        selected_features = feature_columns_for_mode(dataset, selected_variant.feature_mode)
        final_classifier, final_scaler, final_encoder = train_variant(dataset, selected_features, selected_variant)
        final_ordinary_df = ordinary_fallback_dataset(selected_features, base_feature_names)
        final_ordinary_pred = predict_labels(
            final_classifier,
            final_scaler,
            final_encoder,
            final_ordinary_df,
            selected_features,
        )
        report["final_selected_model"] = {
            "name": selected_name,
            "training_rows": int(len(dataset)),
            "feature_count": len(selected_features),
            "ordinary_regression": metric_summary(final_ordinary_df, final_ordinary_pred),
        }
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_classifier, MODEL_DIR / "playing_retro_audio_rf_classifier.pkl")
        joblib.dump(final_scaler, MODEL_DIR / "playing_retro_audio_feature_scaler.pkl")
        joblib.dump(final_encoder, MODEL_DIR / "playing_retro_audio_label_encoder.pkl")
        joblib.dump(selected_features, MODEL_DIR / "playing_retro_audio_feature_cols.pkl")
    else:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        report["final_selected_model"] = None

    (MODEL_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_report(REPORT_MD, report, eval_df)

    print(f"Wrote {MULTI_WINDOW_DATASET_CSV}")
    print(f"Wrote {EVAL_CSV}")
    print(f"Wrote {HOLDOUT_PREDICTIONS_CSV}")
    print(f"Wrote {REPORT_MD}")
    print(f"selected_variant={report['selected_variant']}")
    for item in report["variants"]:
        holdout = item["holdout"]
        ordinary = item["ordinary_regression"]
        print(
            "{name}: holdout_acc={acc:.3f} racket={racket:.3f} table={table:.3f} "
            "non_target={non_target:.3f} ordinary_acc={ordinary_acc:.3f}".format(
                name=item["name"],
                acc=holdout["accuracy"],
                racket=holdout["racket_contact_recall"],
                table=holdout["table_bounce_recall"],
                non_target=holdout["non_target_recall"],
                ordinary_acc=ordinary["accuracy"],
            )
        )


if __name__ == "__main__":
    main()
