"""
Cross-session validation for the T0007 spel_retro_audio feature family.

This script is local-only. It does not export Collector app model JSON, build
an APK, or change `studs_live`.

Run:
  python skills/pingis-audio-classification/scripts/evaluate_playing_retro_audio_cross_session.py
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
    VARIANTS as ALL_MULTI_WINDOW_VARIANTS,
    build_multi_window_dataset,
    feature_columns_for_mode,
    metric_summary as multi_metric_summary,
    ordinary_fallback_dataset,
    predict_labels as predict_multi_window_labels,
    read_or_build_candidate_rows,
    train_variant as train_multi_window_variant,
)
from evaluate_playing_retro_audio_variants import (
    VARIANTS as ALL_ONE_WINDOW_VARIANTS,
    metric_summary as one_window_metric_summary,
    predict_labels as predict_one_window_labels,
    train_variant as train_one_window_variant,
)
from train_playing_retro_audio import (
    DEFAULT_DATASET_CSV,
    EVAL_DIR,
    TARGET_LABELS,
    build_ordinary_regression_dataset,
    feature_columns as one_window_feature_columns,
    grouped_metrics,
)


HOLDOUT_SESSIONS = [
    "audio_session_2026-05-28_002",
    "audio_session_2026-05-29_001",
    "audio_session_2026-05-29_002",
]

ONE_WINDOW_VARIANTS = [
    variant
    for variant in ALL_ONE_WINDOW_VARIANTS
    if variant.name in {"t0005_baseline", "safe_racket_weighted"}
]
MULTI_WINDOW_VARIANTS = [
    variant
    for variant in ALL_MULTI_WINDOW_VARIANTS
    if variant.name in {
        "multi_window_context",
        "multi_window_context_safe_weighted",
        "multi_window_context_racket_weighted",
    }
]
SELECTED_T0007_VARIANT = "multi_window_context_racket_weighted"
SELECTED_T0006_VARIANT = "safe_racket_weighted"

EVAL_CSV = EVAL_DIR / "playing_retro_audio_t0008_cross_session_eval.csv"
PREDICTIONS_CSV = EVAL_DIR / "playing_retro_audio_t0008_cross_session_predictions.csv"
REPORT_MD = EVAL_DIR / "playing_retro_audio_t0008_cross_session_report.md"
REPORT_JSON = EVAL_DIR / "playing_retro_audio_t0008_cross_session_report.json"


def load_multi_window_dataset(rebuild: bool) -> pd.DataFrame:
    if rebuild or not MULTI_WINDOW_DATASET_CSV.exists():
        rows = read_or_build_candidate_rows()
        dataset = build_multi_window_dataset(rows)
        MULTI_WINDOW_DATASET_CSV.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(MULTI_WINDOW_DATASET_CSV, index=False)
        return dataset
    return pd.read_csv(MULTI_WINDOW_DATASET_CSV)


def label_counts_by_session(df: pd.DataFrame, sessions: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for session in sessions:
        session_df = df[df["session_id"].astype(str) == session]
        result[session] = {
            "rows": int(len(session_df)),
            "labels": {str(k): int(v) for k, v in session_df["label"].value_counts().to_dict().items()},
            "source_rules": {
                str(k): int(v)
                for k, v in session_df["source_rule"].value_counts().to_dict().items()
            },
        }
    return result


def add_predictions(
    rows: list[pd.DataFrame],
    *,
    family: str,
    variant_name: str,
    holdout_session: str,
    holdout_df: pd.DataFrame,
    predictions: np.ndarray,
) -> None:
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
    ]
    for optional in ["ctx_prev_gap_1000", "ctx_next_gap_1000", "ctx_density_300ms"]:
        if optional in holdout_df.columns:
            columns.append(optional)
    pred_df = holdout_df[columns].copy()
    pred_df.insert(0, "family", family)
    pred_df.insert(1, "variant", variant_name)
    pred_df.insert(2, "holdout_session", holdout_session)
    pred_df["prediction"] = predictions
    pred_df["correct"] = pred_df["label"].astype(str) == pred_df["prediction"].astype(str)
    rows.append(pred_df)


def add_grouped_eval_rows(
    eval_rows: list[dict[str, Any]],
    *,
    family: str,
    variant_name: str,
    holdout_session: str,
    scope: str,
    df: pd.DataFrame,
    predictions: np.ndarray,
    group_columns: list[str],
) -> None:
    eval_rows.extend(
        {
            "family": family,
            "variant": variant_name,
            "holdout_session": holdout_session,
            **row,
        }
        for row in grouped_metrics(scope, df, predictions, group_columns)
    )


def evaluate_one_window(
    dataset: pd.DataFrame,
    holdout_sessions: list[str],
    eval_rows: list[dict[str, Any]],
    prediction_rows: list[pd.DataFrame],
) -> list[dict[str, Any]]:
    features = one_window_feature_columns(dataset)
    ordinary_df = build_ordinary_regression_dataset(features)
    summaries: list[dict[str, Any]] = []

    for holdout_session in holdout_sessions:
        train_df = dataset[dataset["session_id"].astype(str) != holdout_session].copy()
        holdout_df = dataset[dataset["session_id"].astype(str) == holdout_session].copy()
        if train_df.empty or holdout_df.empty:
            continue
        for variant in ONE_WINDOW_VARIANTS:
            classifier, scaler, label_encoder = train_one_window_variant(train_df, features, variant)
            holdout_pred = predict_one_window_labels(classifier, scaler, label_encoder, holdout_df, features)
            ordinary_pred = predict_one_window_labels(classifier, scaler, label_encoder, ordinary_df, features)

            add_predictions(
                prediction_rows,
                family="one_window",
                variant_name=variant.name,
                holdout_session=holdout_session,
                holdout_df=holdout_df,
                predictions=holdout_pred,
            )
            add_grouped_eval_rows(
                eval_rows,
                family="one_window",
                variant_name=variant.name,
                holdout_session=holdout_session,
                scope="cross_session_holdout",
                df=holdout_df,
                predictions=holdout_pred,
                group_columns=["session_id", "evaluation_bucket", "close_event_bucket", "source_rule"],
            )
            add_grouped_eval_rows(
                eval_rows,
                family="one_window",
                variant_name=variant.name,
                holdout_session=holdout_session,
                scope="ordinary_regression",
                df=ordinary_df,
                predictions=ordinary_pred,
                group_columns=["scenario_id", "background_condition"],
            )
            summaries.append({
                "family": "one_window",
                "variant": variant.name,
                "holdout_session": holdout_session,
                "holdout": one_window_metric_summary(holdout_df, holdout_pred),
                "ordinary_regression": one_window_metric_summary(ordinary_df, ordinary_pred),
            })
    return summaries


def evaluate_multi_window(
    dataset: pd.DataFrame,
    holdout_sessions: list[str],
    eval_rows: list[dict[str, Any]],
    prediction_rows: list[pd.DataFrame],
) -> list[dict[str, Any]]:
    base_feature_names = sorted({
        column.split("_", 1)[1]
        for column in dataset.columns
        if column.startswith("normal_")
    })
    summaries: list[dict[str, Any]] = []

    for holdout_session in holdout_sessions:
        train_df = dataset[dataset["session_id"].astype(str) != holdout_session].copy()
        holdout_df = dataset[dataset["session_id"].astype(str) == holdout_session].copy()
        if train_df.empty or holdout_df.empty:
            continue
        for variant in MULTI_WINDOW_VARIANTS:
            features = feature_columns_for_mode(dataset, variant.feature_mode)
            ordinary_df = ordinary_fallback_dataset(features, base_feature_names)
            classifier, scaler, label_encoder = train_multi_window_variant(train_df, features, variant)
            holdout_pred = predict_multi_window_labels(classifier, scaler, label_encoder, holdout_df, features)
            ordinary_pred = predict_multi_window_labels(classifier, scaler, label_encoder, ordinary_df, features)

            add_predictions(
                prediction_rows,
                family="multi_window",
                variant_name=variant.name,
                holdout_session=holdout_session,
                holdout_df=holdout_df,
                predictions=holdout_pred,
            )
            add_grouped_eval_rows(
                eval_rows,
                family="multi_window",
                variant_name=variant.name,
                holdout_session=holdout_session,
                scope="cross_session_holdout",
                df=holdout_df,
                predictions=holdout_pred,
                group_columns=["session_id", "evaluation_bucket", "close_event_bucket", "source_rule"],
            )
            add_grouped_eval_rows(
                eval_rows,
                family="multi_window",
                variant_name=variant.name,
                holdout_session=holdout_session,
                scope="ordinary_regression_advisory",
                df=ordinary_df,
                predictions=ordinary_pred,
                group_columns=["scenario_id", "background_condition"],
            )
            summaries.append({
                "family": "multi_window",
                "variant": variant.name,
                "holdout_session": holdout_session,
                "holdout": multi_metric_summary(holdout_df, holdout_pred),
                "ordinary_regression_advisory": multi_metric_summary(ordinary_df, ordinary_pred),
            })
    return summaries


def all_rows(eval_df: pd.DataFrame) -> pd.DataFrame:
    return eval_df[
        (eval_df["scope"] == "cross_session_holdout")
        & (eval_df["group"] == "all")
    ].copy()


def find_metric(
    eval_df: pd.DataFrame,
    *,
    family: str,
    variant: str,
    holdout_session: str,
    group: str,
    metric: str,
) -> float | None:
    row = eval_df[
        (eval_df["family"] == family)
        & (eval_df["variant"] == variant)
        & (eval_df["holdout_session"] == holdout_session)
        & (eval_df["scope"] == "cross_session_holdout")
        & (eval_df["group"] == group)
    ]
    if row.empty:
        return None
    value = row.iloc[0].get(metric)
    if pd.isna(value):
        return None
    return float(value)


def verdict(eval_df: pd.DataFrame, holdout_sessions: list[str]) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    for session in holdout_sessions:
        selected = {
            metric: find_metric(
                eval_df,
                family="multi_window",
                variant=SELECTED_T0007_VARIANT,
                holdout_session=session,
                group="all",
                metric=metric,
            )
            for metric in [
                "accuracy",
                "racket_contact_recall",
                "table_bounce_recall",
                "non_target_recall",
            ]
        }
        t0006 = {
            metric: find_metric(
                eval_df,
                family="one_window",
                variant=SELECTED_T0006_VARIANT,
                holdout_session=session,
                group="all",
                metric=metric,
            )
            for metric in [
                "accuracy",
                "racket_contact_recall",
                "table_bounce_recall",
                "non_target_recall",
            ]
        }
        comparisons.append({
            "holdout_session": session,
            "selected": selected,
            "t0006": t0006,
            "racket_delta": (
                selected["racket_contact_recall"] - t0006["racket_contact_recall"]
                if selected["racket_contact_recall"] is not None and t0006["racket_contact_recall"] is not None
                else None
            ),
            "table_delta": (
                selected["table_bounce_recall"] - t0006["table_bounce_recall"]
                if selected["table_bounce_recall"] is not None and t0006["table_bounce_recall"] is not None
                else None
            ),
            "non_target_delta": (
                selected["non_target_recall"] - t0006["non_target_recall"]
                if selected["non_target_recall"] is not None and t0006["non_target_recall"] is not None
                else None
            ),
        })

    complete = [
        item for item in comparisons
        if item["racket_delta"] is not None
        and item["table_delta"] is not None
        and item["non_target_delta"] is not None
    ]
    all_racket_not_worse = all(item["racket_delta"] >= -0.001 for item in complete)
    any_racket_better = any(item["racket_delta"] > 0.001 for item in complete)
    all_table_protected = all(item["table_delta"] >= -0.02 for item in complete)
    all_non_target_protected = all(item["non_target_delta"] >= -0.02 for item in complete)
    if complete and all_racket_not_worse and any_racket_better and all_table_protected and all_non_target_protected:
        label = "passes_cross_session_gate"
        explanation = "T0007 selected variant matches or improves racket recall on every requested holdout, improves at least one holdout, and avoids meaningful table/non-target regression."
    elif complete and all_racket_not_worse:
        label = "mixed_tradeoff"
        explanation = "T0007 selected variant does not lose racket recall, but at least one table or non-target slice regresses."
    else:
        label = "not_general_enough"
        explanation = "T0007 selected variant does not improve racket recall consistently across the requested holdouts."
    return {
        "label": label,
        "explanation": explanation,
        "comparisons": comparisons,
    }


def fmt(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value)


def write_report(
    path: Path,
    *,
    eval_df: pd.DataFrame,
    report: dict[str, Any],
) -> None:
    summary = all_rows(eval_df)
    lines = [
        "# Playing Retro Audio T0008 Cross-Session Report",
        "",
        "This is a local validation report. No Collector app model JSON, APK, or `studs_live` behavior was changed.",
        "",
        "## Decision",
        "",
        f"- Verdict: `{report['verdict']['label']}`",
        f"- Meaning: {report['verdict']['explanation']}",
        "- `multi_window` ordinary metrics are advisory only because older ordinary rows do not preserve exact multi-window timestamps.",
        "",
        "## Holdout Comparison",
        "",
        "| Holdout | Family | Variant | Rows | Accuracy | Racket Recall | Table Recall | Non-target Recall |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values(["holdout_session", "family", "variant"]).to_dict("records"):
        lines.append(
            f"| `{row['holdout_session']}` | `{row['family']}` | `{row['variant']}` | "
            f"{int(row['rows'])} | {fmt(row['accuracy'])} | {fmt(row['racket_contact_recall'])} | "
            f"{fmt(row['table_bounce_recall'])} | {fmt(row['non_target_recall'])} |"
        )

    lines.extend([
        "",
        "## T0007 Selected vs T0006",
        "",
        "| Holdout | Racket Delta | Table Delta | Non-target Delta |",
        "|---|---:|---:|---:|",
    ])
    for item in report["verdict"]["comparisons"]:
        lines.append(
            f"| `{item['holdout_session']}` | {fmt(item['racket_delta'])} | "
            f"{fmt(item['table_delta'])} | {fmt(item['non_target_delta'])} |"
        )

    lines.extend([
        "",
        "## Error Slices",
        "",
        "| Holdout | Variant | wrong_class_racket_as_table Racket Recall | matched_table Table Recall | under_120ms Accuracy |",
        "|---|---|---:|---:|---:|",
    ])
    for session in report["holdout_sessions"]:
        for family, variant_name in [
            ("one_window", "t0005_baseline"),
            ("one_window", "safe_racket_weighted"),
            ("multi_window", SELECTED_T0007_VARIANT),
        ]:
            wrong_racket = find_metric(
                eval_df,
                family=family,
                variant=variant_name,
                holdout_session=session,
                group="source_rule=wrong_class_racket_as_table",
                metric="racket_contact_recall",
            )
            matched_table = find_metric(
                eval_df,
                family=family,
                variant=variant_name,
                holdout_session=session,
                group="source_rule=matched_table",
                metric="table_bounce_recall",
            )
            under_120_values = []
            for bucket in ["under_80ms", "80_119ms"]:
                group = f"close_event_bucket={bucket}"
                under_120_values.append((
                    find_metric(
                        eval_df,
                        family=family,
                        variant=variant_name,
                        holdout_session=session,
                        group=group,
                        metric="accuracy",
                    ),
                    find_metric(
                        eval_df,
                        family=family,
                        variant=variant_name,
                        holdout_session=session,
                        group=group,
                        metric="rows",
                    ),
                ))
            weighted = [
                (accuracy, rows)
                for accuracy, rows in under_120_values
                if accuracy is not None and rows is not None and rows > 0
            ]
            under_120 = (
                float(sum(accuracy * rows for accuracy, rows in weighted) / sum(rows for _accuracy, rows in weighted))
                if weighted
                else None
            )
            lines.append(
                f"| `{session}` | `{variant_name}` | {fmt(wrong_racket)} | "
                f"{fmt(matched_table)} | {fmt(under_120)} |"
            )

    lines.extend([
        "",
        "## Outputs",
        "",
        f"- Evaluation CSV: `{EVAL_CSV.as_posix()}`",
        f"- Prediction CSV: `{PREDICTIONS_CSV.as_posix()}`",
        f"- JSON report: `{REPORT_JSON.as_posix()}`",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-session validate T0007 playing-retro audio variants.")
    parser.add_argument("--rebuild-multi-window-dataset", action="store_true")
    parser.add_argument("--holdout-session", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not DEFAULT_DATASET_CSV.exists():
        raise SystemExit(f"Missing {DEFAULT_DATASET_CSV}; run train_playing_retro_audio.py first.")

    holdout_sessions = args.holdout_session or HOLDOUT_SESSIONS
    one_window_dataset = pd.read_csv(DEFAULT_DATASET_CSV)
    multi_window_dataset = load_multi_window_dataset(args.rebuild_multi_window_dataset)
    holdout_sessions = [
        session
        for session in holdout_sessions
        if session in set(one_window_dataset["session_id"]) and session in set(multi_window_dataset["session_id"])
    ]
    if not holdout_sessions:
        raise SystemExit("No requested holdout sessions exist in both datasets.")

    eval_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    summaries = []
    summaries.extend(evaluate_one_window(one_window_dataset, holdout_sessions, eval_rows, prediction_rows))
    summaries.extend(evaluate_multi_window(multi_window_dataset, holdout_sessions, eval_rows, prediction_rows))

    eval_df = pd.DataFrame(eval_rows)
    EVAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(EVAL_CSV, index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(PREDICTIONS_CSV, index=False)

    report = {
        "holdout_sessions": holdout_sessions,
        "one_window_dataset": {
            "rows": int(len(one_window_dataset)),
            "sessions": int(one_window_dataset["session_id"].nunique()),
            "holdout_counts": label_counts_by_session(one_window_dataset, holdout_sessions),
        },
        "multi_window_dataset": {
            "rows": int(len(multi_window_dataset)),
            "sessions": int(multi_window_dataset["session_id"].nunique()),
            "holdout_counts": label_counts_by_session(multi_window_dataset, holdout_sessions),
        },
        "variants": summaries,
        "verdict": verdict(eval_df, holdout_sessions),
        "outputs": {
            "eval_csv": str(EVAL_CSV),
            "predictions_csv": str(PREDICTIONS_CSV),
            "report_md": str(REPORT_MD),
            "report_json": str(REPORT_JSON),
        },
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_report(REPORT_MD, eval_df=eval_df, report=report)

    print(f"Wrote {EVAL_CSV}")
    print(f"Wrote {PREDICTIONS_CSV}")
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {REPORT_JSON}")
    print(f"verdict={report['verdict']['label']}")
    selected_rows = all_rows(eval_df)[
        (all_rows(eval_df)["family"] == "multi_window")
        & (all_rows(eval_df)["variant"] == SELECTED_T0007_VARIANT)
    ]
    for row in selected_rows.sort_values("holdout_session").to_dict("records"):
        print(
            "{session}: selected_t0007 acc={acc:.3f} racket={racket:.3f} "
            "table={table:.3f} non_target={non_target:.3f}".format(
                session=row["holdout_session"],
                acc=row["accuracy"],
                racket=row["racket_contact_recall"],
                table=row["table_bounce_recall"],
                non_target=row["non_target_recall"],
            )
        )


if __name__ == "__main__":
    main()
