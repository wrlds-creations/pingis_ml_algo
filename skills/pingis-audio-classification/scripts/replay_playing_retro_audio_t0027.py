"""
Replay the T0026 joblib playing-retro model against reviewed dense sessions.

This is a local T0027 replay/tuning step. It does not retrain, export app JSON,
build an APK, or change studs_live.

Run:
  python skills/pingis-audio-classification/scripts/replay_playing_retro_audio_t0027.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from build_playing_retro_candidate_report import MATCH_TOLERANCE_MS, build_truth_markers
from replay_live_bounce import RAW_DIR
from train_playing_retro_audio import EVAL_DIR, TARGET_LABELS, mapped_baseline_prediction
from train_playing_retro_audio_t0026 import DATASET_CSV, MODEL_DIR, MODEL_ID


DEFAULT_SESSIONS = [
    "audio_session_2026-05-28_002",
    "audio_session_2026-05-29_001",
    "audio_session_2026-05-29_002",
    "audio_session_2026-06-03_005",
    "audio_session_2026-06-04_001",
]

PREDICTIONS_CSV = EVAL_DIR / "playing_retro_audio_t0027_replay_predictions.csv"
EVAL_CSV = EVAL_DIR / "playing_retro_audio_t0027_replay_eval.csv"
SWEEP_CSV = EVAL_DIR / "playing_retro_audio_t0027_threshold_sweep.csv"
REPORT_JSON = EVAL_DIR / "playing_retro_audio_t0027_replay_report.json"
REPORT_MD = EVAL_DIR / "playing_retro_audio_t0027_replay_report.md"

SAME_LABEL_DEDUPE_MS = 80
REPLAY_MATCH_MS = MATCH_TOLERANCE_MS
THRESHOLDS = [0.0, 0.45, 0.50, 0.54, 0.58, 0.62, 0.66, 0.70, 0.74, 0.78, 0.82]


def load_t0026_model(model_dir: Path) -> tuple[Any, Any, Any, list[str]]:
    classifier = joblib.load(model_dir / "playing_retro_audio_rf_classifier.pkl")
    scaler = joblib.load(model_dir / "playing_retro_audio_feature_scaler.pkl")
    encoder = joblib.load(model_dir / "playing_retro_audio_label_encoder.pkl")
    features = joblib.load(model_dir / "playing_retro_audio_feature_cols.pkl")
    return classifier, scaler, encoder, list(features)


def predict_joblib_model(
    classifier: Any,
    scaler: Any,
    encoder: Any,
    features: list[str],
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        raise ValueError(f"T0026 replay dataset is missing features: {missing[:10]}")
    x = scaler.transform(df[features].fillna(0).to_numpy(dtype=np.float32))
    encoded = classifier.predict(x)
    predictions = encoder.inverse_transform(encoded)
    probabilities_raw = classifier.predict_proba(x)
    classes = [str(label) for label in encoder.classes_]
    confidences = np.max(probabilities_raw, axis=1)
    probabilities = {
        label: probabilities_raw[:, index]
        for index, label in enumerate(classes)
    }
    return predictions, confidences, probabilities


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
    row["target_predicted_non_target"] = int(((truth != "non_target") & (predictions == "non_target")).sum())
    row["non_target_predicted_target"] = int(((truth == "non_target") & (predictions != "non_target")).sum())
    row["wrong_target_class"] = int(
        (
            ((truth == "racket_contact") & (predictions == "table_bounce"))
            | ((truth == "table_bounce") & (predictions == "racket_contact"))
        ).sum()
    )
    return row


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def grouped_metric_rows(scope: str, df: pd.DataFrame, prediction_column: str) -> list[dict[str, Any]]:
    rows = [metric_row(scope, "all", df, prediction_column)]
    for column in ["session_id", "close_event_bucket", "source_rule", "candidate_status"]:
        if column not in df.columns:
            continue
        for value in sorted(df[column].fillna("").astype(str).unique()):
            group_df = df[df[column].fillna("").astype(str) == value]
            if not group_df.empty:
                rows.append(metric_row(scope, f"{column}={value or 'unspecified'}", group_df, prediction_column))
    return rows


def marker_kind(label: str) -> str | None:
    if label == "racket_contact":
        return "racket"
    if label == "table_bounce":
        return "table"
    return None


def truth_rows_for_sessions(sessions: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for session_id in sessions:
        session_path = RAW_DIR / f"{session_id}.json"
        data = json.loads(session_path.read_text(encoding="utf-8"))
        for event_index, event in enumerate(data.get("events") or []):
            for truth in build_truth_markers((event.get("review") or {}).get("markers") or []):
                kind = marker_kind(truth.truth_kind)
                if not kind:
                    continue
                rows.append({
                    "session_id": session_id,
                    "event_index": event_index,
                    "truth_id": truth.marker_id,
                    "timestamp_ms": truth.timestamp_ms,
                    "kind": kind,
                })
    return pd.DataFrame(rows)


def recovery_mask(df: pd.DataFrame) -> pd.Series:
    return df["candidate_id"].fillna("").astype(str).str.contains("recovery", case=False, regex=False)


def nearest_saved_gap_by_row(df: pd.DataFrame) -> pd.Series:
    gaps = pd.Series(np.nan, index=df.index, dtype="float64")
    is_recovery = recovery_mask(df)
    for (_session_id, _event_index), group in df.groupby(["session_id", "event_index"], dropna=False):
        saved = group[~is_recovery.loc[group.index]]["anchor_ms"].astype(float).to_numpy()
        if len(saved) == 0:
            continue
        for index, row in group[is_recovery.loc[group.index]].iterrows():
            gaps.loc[index] = float(np.min(np.abs(saved - float(row["anchor_ms"]))))
    return gaps


def prediction_events(
    df: pd.DataFrame,
    prediction_column: str,
    confidence_column: str | None,
    racket_threshold: float,
    table_threshold: float,
    baseline_visible_only: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, row in df.iterrows():
        if baseline_visible_only and str(row.get("candidate_status") or "") != "review_relevant":
            continue
        label = str(row[prediction_column])
        kind = marker_kind(label)
        if not kind:
            continue
        confidence = float(row[confidence_column]) if confidence_column else 1.0
        threshold = racket_threshold if kind == "racket" else table_threshold
        if confidence < threshold:
            continue
        rows.append({
            "source_index": int(index),
            "session_id": str(row["session_id"]),
            "event_index": int(row["event_index"]),
            "timestamp_ms": float(row["anchor_ms"]),
            "kind": kind,
            "confidence": confidence,
            "candidate_id": str(row.get("candidate_id") or ""),
            "is_recovery": bool(row.get("is_recovery_candidate", False)),
        })
    return pd.DataFrame(rows)


def dedupe_same_label(predictions: pd.DataFrame, gap_ms: int) -> tuple[pd.DataFrame, int]:
    if predictions.empty:
        return predictions.copy(), 0
    kept: list[dict[str, Any]] = []
    removed = 0
    for row in predictions.sort_values(["session_id", "event_index", "timestamp_ms"]).to_dict("records"):
        duplicate_index = -1
        for index in range(len(kept) - 1, -1, -1):
            existing = kept[index]
            if existing["session_id"] != row["session_id"] or int(existing["event_index"]) != int(row["event_index"]):
                break
            if float(row["timestamp_ms"]) - float(existing["timestamp_ms"]) > gap_ms:
                break
            if str(existing["kind"]) == str(row["kind"]):
                duplicate_index = index
                break
        if duplicate_index >= 0:
            kept[duplicate_index] = row
            removed += 1
        else:
            kept.append(row)
    return pd.DataFrame(kept), removed


def evaluate_marker_predictions(
    predictions: pd.DataFrame,
    truths: pd.DataFrame,
    scope: str,
    match_ms: int,
    dedupe_ms: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    deduped, removed = dedupe_same_label(predictions, dedupe_ms)
    total_tp = 0
    total_wrong = 0
    total_fp = 0
    total_missed = 0
    tp_by_kind = {"racket": 0, "table": 0}
    missed_by_kind = {"racket": 0, "table": 0}
    session_rows: list[dict[str, Any]] = []

    keys = sorted(set(zip(truths["session_id"], truths["event_index"])) | set(zip(deduped.get("session_id", []), deduped.get("event_index", []))))
    for session_id, event_index in keys:
        truth_group = truths[(truths["session_id"] == session_id) & (truths["event_index"] == event_index)].copy()
        pred_group = deduped[(deduped["session_id"] == session_id) & (deduped["event_index"] == event_index)].copy() if not deduped.empty else pd.DataFrame()
        used_truths: set[int] = set()
        tp = 0
        wrong = 0
        fp = 0

        for _, prediction in pred_group.sort_values("timestamp_ms").iterrows():
            nearby = truth_group.assign(
                dt_ms=(truth_group["timestamp_ms"].astype(float) - float(prediction["timestamp_ms"])).abs()
            )
            nearby = nearby[nearby["dt_ms"] <= match_ms].sort_values("dt_ms")
            same_kind = nearby[
                (nearby["kind"].astype(str) == str(prediction["kind"]))
                & (~nearby.index.isin(used_truths))
            ]
            if not same_kind.empty:
                truth_index = int(same_kind.index[0])
                used_truths.add(truth_index)
                tp += 1
                tp_by_kind[str(prediction["kind"])] += 1
                continue
            any_kind = nearby[~nearby.index.isin(used_truths)]
            if not any_kind.empty:
                truth_index = int(any_kind.index[0])
                used_truths.add(truth_index)
                wrong += 1
            else:
                fp += 1

        missed_rows = truth_group[~truth_group.index.isin(used_truths)]
        missed = int(len(missed_rows))
        for kind, count in missed_rows["kind"].value_counts().to_dict().items():
            missed_by_kind[str(kind)] += int(count)

        total_tp += tp
        total_wrong += wrong
        total_fp += fp
        total_missed += missed
        session_rows.append({
            "scope": scope,
            "session_id": session_id,
            "event_index": int(event_index),
            "predictions": int(len(pred_group)),
            "true_positive": tp,
            "wrong_class": wrong,
            "false_positive": fp,
            "missed": missed,
        })

    summary = {
        "scope": scope,
        "predictions": int(len(deduped)),
        "same_label_duplicates_removed": int(removed),
        "true_positive": int(total_tp),
        "wrong_class": int(total_wrong),
        "false_positive": int(total_fp),
        "missed": int(total_missed),
        "tp_racket": int(tp_by_kind["racket"]),
        "tp_table": int(tp_by_kind["table"]),
        "missed_racket": int(missed_by_kind["racket"]),
        "missed_table": int(missed_by_kind["table"]),
    }
    return summary, session_rows


def sweep_thresholds(candidate_df: pd.DataFrame, truths: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for racket_threshold in THRESHOLDS:
        for table_threshold in THRESHOLDS:
            predictions = prediction_events(
                candidate_df,
                prediction_column="t0026_prediction",
                confidence_column="t0026_confidence",
                racket_threshold=racket_threshold,
                table_threshold=table_threshold,
                baseline_visible_only=False,
            )
            summary, _session_rows = evaluate_marker_predictions(
                predictions,
                truths,
                scope="t0026_threshold_sweep",
                match_ms=REPLAY_MATCH_MS,
                dedupe_ms=SAME_LABEL_DEDUPE_MS,
            )
            score = (
                summary["true_positive"]
                - 2.0 * summary["false_positive"]
                - 1.5 * summary["wrong_class"]
                - 0.35 * summary["missed"]
            )
            rows.append({
                "racket_threshold": racket_threshold,
                "table_threshold": table_threshold,
                "score": float(score),
                **summary,
            })
    sweep_df = pd.DataFrame(rows)
    return sweep_df, sweep_df.sort_values(
        ["score", "true_positive", "false_positive", "wrong_class", "predictions"],
        ascending=[False, False, True, True, True],
    ).iloc[0].to_dict()


def select_safe_threshold(sweep_df: pd.DataFrame, baseline_summary: dict[str, Any]) -> dict[str, Any]:
    safe = sweep_df[
        (sweep_df["true_positive"] >= int(baseline_summary["true_positive"]))
        & (sweep_df["false_positive"] <= int(baseline_summary["false_positive"]))
        & (sweep_df["wrong_class"] <= int(baseline_summary["wrong_class"]))
    ].copy()
    if safe.empty:
        return sweep_df.sort_values(
            ["score", "true_positive", "false_positive", "wrong_class", "predictions"],
            ascending=[False, False, True, True, True],
        ).iloc[0].to_dict()
    return safe.sort_values(
        ["score", "true_positive", "false_positive", "wrong_class", "predictions"],
        ascending=[False, False, True, True, True],
    ).iloc[0].to_dict()


def recommendation(selected: dict[str, Any], baseline: dict[str, Any]) -> str:
    tp_gain = int(selected["true_positive"]) - int(baseline["true_positive"])
    fp_gain = int(selected["false_positive"]) - int(baseline["false_positive"])
    wrong_gain = int(selected["wrong_class"]) - int(baseline["wrong_class"])
    if tp_gain > 0 and fp_gain <= 0 and wrong_gain <= 0:
        return "proceed_t0028_export_with_selected_thresholds"
    if tp_gain >= 0 and fp_gain <= 2 and wrong_gain <= 1:
        return "proceed_t0028_export_cautiously_with_selected_thresholds"
    return "do_not_export_yet_tune_or_collect_more"


def metric_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.3f}"


def write_markdown(report: dict[str, Any], eval_df: pd.DataFrame) -> None:
    baseline = report["marker_replay"]["baseline_visible"]
    selected = report["marker_replay"]["t0026_selected"]
    candidate_rows = eval_df[(eval_df["kind"] == "candidate") & (eval_df["group"] == "all")]
    lines = [
        "# Playing Retro Audio T0027 Replay Report",
        "",
        "This is a local replay/tuning report. No Collector app model JSON, APK, `studs_live`, or video model artifact was changed.",
        "",
        "## Recommendation",
        "",
        f"- Result: `{report['recommendation']}`",
        f"- T0026 model: `{MODEL_ID}`",
        f"- Racket threshold: `{selected['racket_threshold']}`",
        f"- Table threshold: `{selected['table_threshold']}`",
        f"- Same-label dedupe: `{SAME_LABEL_DEDUPE_MS} ms`",
        "",
        "## Marker Replay",
        "",
        "| Scope | Predictions | TP | Wrong | FP | Missed | TP racket/table | Missed racket/table | Dedupe removed |",
        "|---|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    for item in [baseline, selected]:
        lines.append(
            f"| `{item['scope']}` | {int(item['predictions'])} | {int(item['true_positive'])} | "
            f"{int(item['wrong_class'])} | {int(item['false_positive'])} | {int(item['missed'])} | "
            f"{int(item['tp_racket'])}/{int(item['tp_table'])} | "
            f"{int(item['missed_racket'])}/{int(item['missed_table'])} | "
            f"{int(item['same_label_duplicates_removed'])} |"
        )

    lines.extend([
        "",
        "## Candidate-Level Replay",
        "",
        "| Scope | Accuracy | Racket Recall | Table Recall | Non-target Recall | Target->Non-target | Non-target->Target |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in candidate_rows.iterrows():
        lines.append(
            f"| `{row['scope']}` | {metric_text(row['accuracy'])} | "
            f"{metric_text(row['racket_contact_recall'])} | {metric_text(row['table_bounce_recall'])} | "
            f"{metric_text(row['non_target_recall'])} | {int(row['target_predicted_non_target'])} | "
            f"{int(row['non_target_predicted_target'])} |"
        )

    lines.extend([
        "",
        "## Selected Per Session",
        "",
        "| Session | Baseline TP/Wrong/FP/Missed | T0026 TP/Wrong/FP/Missed |",
        "|---|---|---|",
    ])
    per_session = report["marker_replay"]["per_session"]
    for session_id in report["sessions"]:
        base = per_session["baseline_visible"].get(session_id, {})
        t0026 = per_session["t0026_selected"].get(session_id, {})
        lines.append(
            f"| `{session_id}` | {base.get('true_positive', 0)}/{base.get('wrong_class', 0)}/"
            f"{base.get('false_positive', 0)}/{base.get('missed', 0)} | "
            f"{t0026.get('true_positive', 0)}/{t0026.get('wrong_class', 0)}/"
            f"{t0026.get('false_positive', 0)}/{t0026.get('missed', 0)} |"
        )

    lines.extend(["", "## Focus Sessions", ""])
    for focus_session_id, focus in report["focus_sessions"].items():
        lines.extend([
            f"### {focus_session_id}",
            "",
            f"- Baseline target candidate errors fixed by T0026 candidate replay: `{focus['t0026_correct_on_baseline_target_errors']}` / `{focus['baseline_target_candidate_errors']}`",
            f"- Baseline target rows called non-target fixed by T0026: `{focus['t0026_correct_on_baseline_non_target_target_rows']}` / `{focus['baseline_non_target_target_rows']}`",
            f"- Recovery/analysis rows selected by T0026 threshold: `{focus['selected_recovery_or_analysis_predictions']}`",
            "",
        ])

    lines.extend([
        "## Outputs",
        "",
        f"- Predictions CSV: `{PREDICTIONS_CSV.as_posix()}`",
        f"- Evaluation CSV: `{EVAL_CSV.as_posix()}`",
        f"- Threshold sweep CSV: `{SWEEP_CSV.as_posix()}`",
        f"- JSON report: `{REPORT_JSON.as_posix()}`",
        "",
    ])
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def per_session_totals(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    df = pd.DataFrame(rows)
    if df.empty:
        return result
    for session_id, group in df.groupby("session_id"):
        result[str(session_id)] = {
            "predictions": int(group["predictions"].sum()),
            "true_positive": int(group["true_positive"].sum()),
            "wrong_class": int(group["wrong_class"].sum()),
            "false_positive": int(group["false_positive"].sum()),
            "missed": int(group["missed"].sum()),
        }
    return result


def focus_summary(candidate_df: pd.DataFrame, selected_predictions: pd.DataFrame, session_id: str) -> dict[str, Any]:
    focus = candidate_df[candidate_df["session_id"].astype(str) == session_id].copy()
    target = focus[focus["label"].isin(["racket_contact", "table_bounce"])].copy()
    baseline = mapped_baseline_prediction(target["candidate_predicted_kind"])
    baseline_errors = baseline.astype(str) != target["label"].astype(str)
    baseline_non_target = baseline.astype(str) == "non_target"
    t0026_correct = focus["t0026_prediction"].astype(str) == focus["label"].astype(str)
    selected_focus = selected_predictions[selected_predictions["session_id"].astype(str) == session_id]
    selected_recovery_or_analysis = selected_focus[
        selected_focus["is_recovery"].astype(bool)
        | selected_focus["source_index"].isin(
            focus[focus["candidate_status"].astype(str) != "review_relevant"].index.astype(int)
        )
    ]
    return {
        "baseline_target_candidate_errors": int(baseline_errors.sum()),
        "t0026_correct_on_baseline_target_errors": int(
            t0026_correct.loc[target.index[baseline_errors.to_numpy()]].sum()
        ) if int(baseline_errors.sum()) else 0,
        "baseline_non_target_target_rows": int(baseline_non_target.sum()),
        "t0026_correct_on_baseline_non_target_target_rows": int(
            t0026_correct.loc[target.index[baseline_non_target.to_numpy()]].sum()
        ) if int(baseline_non_target.sum()) else 0,
        "selected_recovery_or_analysis_predictions": int(len(selected_recovery_or_analysis)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay T0026 playing-retro model and tune thresholds.")
    parser.add_argument("--dataset-csv", default=str(DATASET_CSV))
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--session", action="append", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sessions = args.session or DEFAULT_SESSIONS
    dataset_path = Path(args.dataset_csv)
    if not dataset_path.exists():
        raise SystemExit(f"Missing {dataset_path}; run T0026 training first.")
    model_dir = Path(args.model_dir)
    classifier, scaler, encoder, features = load_t0026_model(model_dir)

    dataset = pd.read_csv(dataset_path)
    target = dataset[dataset["session_id"].astype(str).isin(set(sessions))].copy()
    candidate_df = target[target["row_type"].fillna("").astype(str) == "candidate"].copy()
    if candidate_df.empty:
        raise SystemExit("No candidate rows found for requested sessions.")

    predictions, confidences, probabilities = predict_joblib_model(classifier, scaler, encoder, features, candidate_df)
    candidate_df["baseline_prediction"] = mapped_baseline_prediction(candidate_df["candidate_predicted_kind"])
    candidate_df["t0026_prediction"] = predictions
    candidate_df["t0026_confidence"] = confidences
    candidate_df["is_recovery_candidate"] = recovery_mask(candidate_df)
    candidate_df["nearest_saved_gap_ms"] = nearest_saved_gap_by_row(candidate_df)
    for label, values in probabilities.items():
        candidate_df[f"t0026_probability_{label}"] = values

    eval_rows: list[dict[str, Any]] = []
    for row in grouped_metric_rows("baseline_candidate_prediction", candidate_df, "baseline_prediction"):
        eval_rows.append({"kind": "candidate", **row})
    for row in grouped_metric_rows("t0026_candidate_prediction", candidate_df, "t0026_prediction"):
        eval_rows.append({"kind": "candidate", **row})

    truths = truth_rows_for_sessions(sessions)
    baseline_events = prediction_events(
        candidate_df,
        prediction_column="baseline_prediction",
        confidence_column=None,
        racket_threshold=0.0,
        table_threshold=0.0,
        baseline_visible_only=True,
    )
    baseline_summary, baseline_session_rows = evaluate_marker_predictions(
        baseline_events,
        truths,
        scope="baseline_visible",
        match_ms=REPLAY_MATCH_MS,
        dedupe_ms=SAME_LABEL_DEDUPE_MS,
    )
    sweep_df, best_by_score = sweep_thresholds(candidate_df, truths)
    selected = select_safe_threshold(sweep_df, baseline_summary)
    selected_events = prediction_events(
        candidate_df,
        prediction_column="t0026_prediction",
        confidence_column="t0026_confidence",
        racket_threshold=float(selected["racket_threshold"]),
        table_threshold=float(selected["table_threshold"]),
        baseline_visible_only=False,
    )
    selected_summary, selected_session_rows = evaluate_marker_predictions(
        selected_events,
        truths,
        scope="t0026_selected",
        match_ms=REPLAY_MATCH_MS,
        dedupe_ms=SAME_LABEL_DEDUPE_MS,
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
        "baseline_prediction",
        "t0026_prediction",
        "t0026_confidence",
        "t0026_probability_racket_contact",
        "t0026_probability_table_bounce",
        "t0026_probability_non_target",
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
        "ticket": "T0027",
        "status": "local_replay_only_not_exported",
        "model_id": MODEL_ID,
        "model_dir": str(model_dir),
        "sessions": sessions,
        "candidate_rows": int(len(candidate_df)),
        "truth_rows": int(len(truths)),
        "recommendation": recommendation(selected_summary, baseline_summary),
        "marker_replay": {
            "baseline_visible": baseline_summary,
            "t0026_selected": selected_summary,
            "best_by_score": best_by_score,
            "per_session": {
                "baseline_visible": per_session_totals(baseline_session_rows),
                "t0026_selected": per_session_totals(selected_session_rows),
            },
        },
        "focus_sessions": {
            "audio_session_2026-06-03_005": focus_summary(candidate_df, selected_events, "audio_session_2026-06-03_005"),
            "audio_session_2026-06-04_001": focus_summary(candidate_df, selected_events, "audio_session_2026-06-04_001"),
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
    write_markdown(report, eval_df)

    print(f"Wrote {PREDICTIONS_CSV}")
    print(f"Wrote {EVAL_CSV}")
    print(f"Wrote {SWEEP_CSV}")
    print(f"Wrote {REPORT_JSON}")
    print(f"Wrote {REPORT_MD}")
    print(
        "baseline: pred={pred} tp={tp} wrong={wrong} fp={fp} missed={missed}".format(
            pred=baseline_summary["predictions"],
            tp=baseline_summary["true_positive"],
            wrong=baseline_summary["wrong_class"],
            fp=baseline_summary["false_positive"],
            missed=baseline_summary["missed"],
        )
    )
    print(
        "t0026_selected: racket_thr={racket} table_thr={table} pred={pred} tp={tp} wrong={wrong} fp={fp} missed={missed}".format(
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
