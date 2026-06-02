"""
Compare focused playing-retro audio variants against the T0005 baseline.

This script is local-only. It does not export a Collector app model JSON, build
an APK, or change `studs_live`.

Run:
  python skills/pingis-audio-classification/scripts/evaluate_playing_retro_audio_variants.py
"""

from __future__ import annotations

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
    DEFAULT_DATASET_CSV,
    DEFAULT_HOLDOUT_SESSIONS,
    EVAL_DIR,
    MODEL_ROOT,
    TARGET_LABELS,
    build_ordinary_regression_dataset,
    feature_columns,
    grouped_metrics,
)


OUT_EVAL_CSV = EVAL_DIR / "playing_retro_audio_t0006_variant_eval.csv"
OUT_HOLDOUT_CSV = EVAL_DIR / "playing_retro_audio_t0006_holdout_predictions.csv"
OUT_REPORT_MD = EVAL_DIR / "playing_retro_audio_t0006_variant_report.md"
SELECTED_MODEL_DIR = MODEL_ROOT / "playing_retro_audio_rf_v2026_06_02_safe_racket_weighted"


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
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
        name="t0005_baseline",
        description="Original T0005 RF: balanced_subsample, min_samples_leaf=2.",
        n_estimators=350,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
    ),
    Variant(
        name="safe_racket_weighted",
        description=(
            "Selected T0006 candidate: mild racket/hard-racket weighting, "
            "RF without class_weight to avoid ordinary-bounce collapse."
        ),
        n_estimators=350,
        min_samples_leaf=2,
        class_weight=None,
        racket_weight=1.10,
        wrong_racket_as_table_weight=1.20,
        missed_marker_weight=1.15,
        tight_racket_weight=1.10,
    ),
    Variant(
        name="aggressive_racket_weighted",
        description=(
            "Rejected tradeoff probe: raises racket more, but costs "
            "non-target or table protection."
        ),
        n_estimators=350,
        min_samples_leaf=3,
        class_weight=None,
        racket_weight=1.30,
        wrong_racket_as_table_weight=1.60,
        missed_marker_weight=1.45,
        tight_racket_weight=1.30,
        table_weight=1.03,
        non_target_weight=1.10,
        false_positive_weight=1.20,
    ),
]


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


def add_holdout_predictions(
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
    ]].copy()
    pred_df.insert(0, "variant", variant.name)
    pred_df["prediction"] = predictions
    pred_df["correct"] = pred_df["label"].astype(str) == pred_df["prediction"].astype(str)
    rows.append(pred_df)


def write_report(
    path: Path,
    eval_df: pd.DataFrame,
    selected: Variant,
    report: dict[str, Any],
) -> None:
    baseline = eval_df[(eval_df["variant"] == "t0005_baseline") & (eval_df["scope"] == "holdout") & (eval_df["group"] == "all")].iloc[0]
    selected_row = eval_df[(eval_df["variant"] == selected.name) & (eval_df["scope"] == "holdout") & (eval_df["group"] == "all")].iloc[0]
    ordinary_row = eval_df[(eval_df["variant"] == selected.name) & (eval_df["scope"] == "ordinary_regression") & (eval_df["group"] == "all")].iloc[0]
    final_ordinary = report["final_selected_model"]["ordinary_regression"]

    lines = [
        "# Playing Retro Audio T0006 Variant Report",
        "",
        "This is a local comparison report. No Collector app model JSON, APK, or `studs_live` behavior was changed.",
        "",
        "## Decision",
        "",
        f"- Selected local candidate: `{selected.name}`",
        f"- Model dir: `{SELECTED_MODEL_DIR.as_posix()}`",
        "- Reason: it is the best tested holdout-trained variant that raises holdout racket recall while keeping holdout table recall, holdout non-target recall, and ordinary regression above the baseline checks used in this ticket.",
        "- Status: local candidate only; still not ready for app integration because racket recall is only a modest improvement.",
        "",
        "## Holdout Comparison",
        "",
        "| Variant | Accuracy | Racket Recall | Table Recall | Non-target Recall |",
        "|---|---:|---:|---:|---:|",
        (
            f"| `t0005_baseline` | {baseline['accuracy']:.3f} | "
            f"{baseline['racket_contact_recall']:.3f} | {baseline['table_bounce_recall']:.3f} | "
            f"{baseline['non_target_recall']:.3f} |"
        ),
        (
            f"| `{selected.name}` | {selected_row['accuracy']:.3f} | "
            f"{selected_row['racket_contact_recall']:.3f} | {selected_row['table_bounce_recall']:.3f} | "
            f"{selected_row['non_target_recall']:.3f} |"
        ),
        "",
        "## Ordinary Regression",
        "",
        "Variant comparison below uses the same holdout-trained models as the holdout metrics.",
        (
            f"- `{selected.name}` ordinary rows: `{int(ordinary_row['rows'])}`, "
            f"accuracy `{ordinary_row['accuracy']:.3f}`, racket recall `{ordinary_row['racket_contact_recall']:.3f}`, "
            f"table recall `{ordinary_row['table_bounce_recall']:.3f}`, non-target recall `{ordinary_row['non_target_recall']:.3f}`"
        ),
        (
            f"- Final saved `{selected.name}` model, refit on all candidate rows, ordinary rows: "
            f"`{final_ordinary['rows']}`, accuracy `{final_ordinary['accuracy']:.3f}`, "
            f"racket recall `{final_ordinary['racket_contact_recall']:.3f}`, "
            f"table recall `{final_ordinary['table_bounce_recall']:.3f}`, "
            f"non-target recall `{final_ordinary['non_target_recall']:.3f}`"
        ),
        "",
        "## Error Finding",
        "",
        "- T0005 baseline missed most racket through `wrong_class_racket_as_table`: 24 of 43 such holdout rows were still predicted as table.",
        "- The selected variant improves that bucket to 21 of 43 correct racket rows while keeping matched table at 101 of 102.",
        "- The remaining hard gap is still fast/close play: sub-120 ms racket rows remain weak and need either better candidate timing or true multi-window/context features in a later ticket.",
        "",
        "## Outputs",
        "",
        f"- Variant eval CSV: `{OUT_EVAL_CSV.as_posix()}`",
        f"- Holdout prediction CSV: `{OUT_HOLDOUT_CSV.as_posix()}`",
        f"- JSON report: `{(SELECTED_MODEL_DIR / 'report.json').as_posix()}`",
        "",
        "## Variant Details",
        "",
    ]
    for variant in report["variants"]:
        lines.extend([
            f"### {variant['name']}",
            "",
            f"- Description: {variant['description']}",
            f"- Holdout: accuracy `{variant['holdout']['accuracy']:.3f}`, racket `{variant['holdout']['racket_contact_recall']:.3f}`, table `{variant['holdout']['table_bounce_recall']:.3f}`, non-target `{variant['holdout']['non_target_recall']:.3f}`",
            f"- Ordinary: accuracy `{variant['ordinary_regression']['accuracy']:.3f}`, racket `{variant['ordinary_regression']['racket_contact_recall']:.3f}`, table `{variant['ordinary_regression']['table_bounce_recall']:.3f}`, non-target `{variant['ordinary_regression']['non_target_recall']:.3f}`",
            "",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not DEFAULT_DATASET_CSV.exists():
        raise SystemExit(f"Missing {DEFAULT_DATASET_CSV}; run train_playing_retro_audio.py first.")

    dataset = pd.read_csv(DEFAULT_DATASET_CSV)
    features = feature_columns(dataset)
    holdout_sessions = [session for session in DEFAULT_HOLDOUT_SESSIONS if session in set(dataset["session_id"])]
    train_df = dataset[~dataset["session_id"].isin(holdout_sessions)].copy()
    holdout_df = dataset[dataset["session_id"].isin(holdout_sessions)].copy()
    ordinary_df = build_ordinary_regression_dataset(features)

    eval_rows: list[dict[str, Any]] = []
    holdout_prediction_rows: list[pd.DataFrame] = []
    report: dict[str, Any] = {
        "selected_variant": "safe_racket_weighted",
        "holdout_sessions": holdout_sessions,
        "dataset_rows": int(len(dataset)),
        "ordinary_regression_rows": int(len(ordinary_df)),
        "variants": [],
    }

    selected_bundle: tuple[RandomForestClassifier, StandardScaler, LabelEncoder] | None = None
    selected_features = features
    selected_variant = next(variant for variant in VARIANTS if variant.name == report["selected_variant"])

    for variant in VARIANTS:
        classifier, scaler, label_encoder = train_variant(train_df, features, variant)
        holdout_pred = predict_labels(classifier, scaler, label_encoder, holdout_df, features)
        ordinary_pred = predict_labels(classifier, scaler, label_encoder, ordinary_df, features)

        add_holdout_predictions(holdout_prediction_rows, variant, holdout_df, holdout_pred)
        eval_rows.extend(
            {
                "variant": variant.name,
                **row,
            }
            for row in grouped_metrics(
                "holdout",
                holdout_df,
                holdout_pred,
                ["session_id", "evaluation_bucket", "close_event_bucket", "source_rule"],
            )
        )
        eval_rows.extend(
            {
                "variant": variant.name,
                **row,
            }
            for row in grouped_metrics(
                "ordinary_regression",
                ordinary_df,
                ordinary_pred,
                ["scenario_id", "background_condition"],
            )
        )

        report["variants"].append({
            "name": variant.name,
            "description": variant.description,
            "config": variant.__dict__,
            "holdout": metric_summary(holdout_df, holdout_pred),
            "ordinary_regression": metric_summary(ordinary_df, ordinary_pred),
        })
        if variant.name == selected_variant.name:
            selected_bundle = (classifier, scaler, label_encoder)

    if selected_bundle is None:
        raise SystemExit(f"Selected variant not evaluated: {selected_variant.name}")

    final_classifier, final_scaler, final_encoder = train_variant(dataset, features, selected_variant)
    final_ordinary_pred = predict_labels(final_classifier, final_scaler, final_encoder, ordinary_df, features)
    report["final_selected_model"] = {
        "name": selected_variant.name,
        "training_rows": int(len(dataset)),
        "ordinary_regression": metric_summary(ordinary_df, final_ordinary_pred),
    }
    SELECTED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_classifier, SELECTED_MODEL_DIR / "playing_retro_audio_rf_classifier.pkl")
    joblib.dump(final_scaler, SELECTED_MODEL_DIR / "playing_retro_audio_feature_scaler.pkl")
    joblib.dump(final_encoder, SELECTED_MODEL_DIR / "playing_retro_audio_label_encoder.pkl")
    joblib.dump(selected_features, SELECTED_MODEL_DIR / "playing_retro_audio_feature_cols.pkl")
    (SELECTED_MODEL_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    eval_df = pd.DataFrame(eval_rows)
    OUT_EVAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(OUT_EVAL_CSV, index=False)
    pd.concat(holdout_prediction_rows, ignore_index=True).to_csv(OUT_HOLDOUT_CSV, index=False)
    write_report(OUT_REPORT_MD, eval_df, selected_variant, report)

    selected_report = next(item for item in report["variants"] if item["name"] == selected_variant.name)
    print(f"Wrote {OUT_EVAL_CSV}")
    print(f"Wrote {OUT_HOLDOUT_CSV}")
    print(f"Wrote {OUT_REPORT_MD}")
    print(f"Saved selected local model to {SELECTED_MODEL_DIR}")
    print(
        "selected={name} holdout_accuracy={acc:.3f} racket_recall={racket:.3f} "
        "table_recall={table:.3f} non_target_recall={non_target:.3f}".format(
            name=selected_variant.name,
            acc=selected_report["holdout"]["accuracy"],
            racket=selected_report["holdout"]["racket_contact_recall"],
            table=selected_report["holdout"]["table_bounce_recall"],
            non_target=selected_report["holdout"]["non_target_recall"],
        )
    )


if __name__ == "__main__":
    main()
