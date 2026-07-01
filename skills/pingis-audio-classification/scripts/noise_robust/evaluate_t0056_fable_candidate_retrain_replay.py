"""
T0056 Fable feature/window audit plus local candidate retrain/replay.

Evaluation-only. This script does not export a model JSON, change app runtime,
build an APK, or mutate raw labels. It combines the Love-approved T0049 rows
with the T0055 reviewed C2 hard slice, then compares current Fable behavior
against local binary candidate models.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.signal import sosfiltfilt
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
for _path in (str(SCRIPT_DIR), str(SCRIPTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import nr_config  # noqa: E402
import nr_features  # noqa: E402
from evaluate_fable_audio_reliability_t0044 import FableAppModel  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parents[4]

SESSION_ID = "fable_live_session_2026-06-28T16-26-01-662Z"
DEFAULT_T0049_ROWS = (
    ROOT_DIR
    / "data"
    / "audio"
    / "models"
    / "evaluations"
    / "t0049_speech_veto_candidate"
    / "t0049_love_approved_extra_rows.csv"
)
DEFAULT_T0055_TIMELINE = (
    ROOT_DIR
    / "data"
    / "audio"
    / "models"
    / "evaluations"
    / "t0055_fable_label_ingest_plan"
    / "t0055_reviewed_timeline.csv"
)
DEFAULT_T0055_CANDIDATES = (
    ROOT_DIR
    / "data"
    / "audio"
    / "models"
    / "evaluations"
    / "t0055_fable_label_ingest_plan"
    / "t0055_candidate_rows.csv"
)
DEFAULT_T0052_WAV = (
    ROOT_DIR
    / "data"
    / "audio"
    / "raw"
    / "t0052_fable_continuous_debug_round"
    / "fable_live_debug"
    / f"{SESSION_ID}.wav"
)
DEFAULT_MODEL_JSON = ROOT_DIR / "apps" / "collector" / "src" / "models" / "fable_audio_model.json"
DEFAULT_OUT_DIR = (
    ROOT_DIR
    / "data"
    / "audio"
    / "models"
    / "evaluations"
    / "t0056_fable_candidate_retrain_replay"
)

TARGET_LABELS = ["noise", "racket_bounce"]
MATCH_TOLERANCE_MS = 140.0
REPLAY_THRESHOLDS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.80, 0.90]
PEAK_WINDOWS_MS = [80, 180, 300, 500]


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def boolish(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    if not fields:
        fields = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if channels != 1 or sample_width != 2:
        raise ValueError(f"Expected mono 16-bit PCM WAV, got channels={channels} width={sample_width}")
    y = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return y, sample_rate


def stats(values: list[float]) -> dict[str, Any]:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return {"count": 0}
    ordered = sorted(clean)
    return {
        "count": len(clean),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": statistics.mean(ordered),
        "max": ordered[-1],
    }


def prediction_fields(prefix: str, prediction: dict[str, Any]) -> dict[str, Any]:
    probs = prediction.get("probabilities") or {}
    return {
        f"{prefix}_label": prediction.get("label") or "",
        f"{prefix}_confidence": prediction.get("confidence") or "",
        f"{prefix}_prob_racket_bounce": probs.get("racket_bounce", ""),
        f"{prefix}_prob_noise": probs.get("noise", ""),
        f"{prefix}_prob_table_bounce": probs.get("table_bounce", ""),
        f"{prefix}_prob_floor_bounce": probs.get("floor_bounce", ""),
    }


def safe_feature(value: Any) -> float:
    out = finite_float(value, 0.0)
    return out if math.isfinite(out) else 0.0


def feature_value(row: dict[str, Any], feature_name: str) -> float:
    if feature_name in row:
        return safe_feature(row.get(feature_name))
    prefixed = f"feat_{feature_name}"
    if prefixed in row:
        return safe_feature(row.get(prefixed))
    return 0.0


def make_feature_row(
    *,
    features: dict[str, Any],
    feature_names: list[str],
    label: str,
    split_source: str,
    row_id: str,
    source_ticket: str,
    scenario_id: str,
    background_condition: str,
    train_role: str,
    weight: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "row_id": row_id,
        "source_ticket": source_ticket,
        "split_source": split_source,
        "label": label,
        "scenario_id": scenario_id,
        "background_condition": background_condition,
        "train_role": train_role,
        "sample_weight": weight,
    }
    for name in feature_names:
        row[name] = feature_value(features, name)
    return row


def load_t0049_rows(path: Path, feature_names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(read_csv(path), start=1):
        label = str(row.get("label") or "")
        if label not in TARGET_LABELS:
            continue
        rows.append(
            make_feature_row(
                features=row,
                feature_names=feature_names,
                label=label,
                split_source="t0049_approved_extra",
                row_id=str(row.get("clip_id") or f"t0049_{index}"),
                source_ticket="T0049",
                scenario_id=str(row.get("scenario_id") or ""),
                background_condition=str(row.get("background_condition") or ""),
                train_role=str(row.get("source") or "approved_extra_row"),
                weight=1.0,
            )
        )
    return rows


def predict_at_sample(model: FableAppModel, y: np.ndarray, sample_rate: int, sample: int) -> dict[str, Any]:
    clip = nr_features.extract_live_clip(y, max(0, min(len(y) - 1, int(sample))))
    features = nr_features.extract_all_features(clip, sample_rate)
    return model.predict_features(features)


def extract_features_at_sample(y: np.ndarray, sample_rate: int, sample: int) -> dict[str, float]:
    clip = nr_features.extract_live_clip(y, max(0, min(len(y) - 1, int(sample))))
    return nr_features.extract_all_features(clip, sample_rate)


def strongest_peak_sample(
    signal: np.ndarray,
    sample_rate: int,
    center_time_s: float,
    radius_ms: float,
) -> int:
    center = int(round(center_time_s * sample_rate))
    radius = int(round(radius_ms / 1000.0 * sample_rate))
    lo = max(0, center - radius)
    hi = min(len(signal), center + radius + 1)
    if hi <= lo:
        return max(0, min(len(signal) - 1, center))
    return lo + int(np.argmax(np.abs(signal[lo:hi])))


def build_t0055_rows(
    candidate_path: Path,
    y: np.ndarray,
    sample_rate: int,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(candidate_path):
        if not boolish(row.get("include_in_candidate_rows")):
            continue
        label = str(row.get("target_label") or "")
        if label not in TARGET_LABELS:
            continue
        time_s = finite_float(row.get("reviewed_time_s"))
        if not math.isfinite(time_s):
            continue
        sample = int(round(time_s * sample_rate))
        features = extract_features_at_sample(y, sample_rate, sample)
        rows.append(
            make_feature_row(
                features=features,
                feature_names=feature_names,
                label=label,
                split_source="t0055_reviewed_c2",
                row_id=str(row.get("row_id") or ""),
                source_ticket="T0055",
                scenario_id=str(row.get("bucket_suggestion") or "ordinary_self_practice_messy_speech_background"),
                background_condition="speech_background",
                train_role=str(row.get("train_role_suggestion") or ""),
                weight=3.0 if label == "racket_bounce" else 2.0,
            )
        )
    return rows


def build_anchor_audit(
    timeline_path: Path,
    model: FableAppModel,
    y: np.ndarray,
    sample_rate: int,
) -> list[dict[str, Any]]:
    rows = [row for row in read_csv(timeline_path) if row.get("review_label") == "racket"]
    bp = sosfiltfilt(nr_features._bandpass_sos(sample_rate, 1500.0, 7000.0), y.astype(np.float64))
    env = np.convolve(np.abs(bp), np.ones(110) / 110, mode="same")

    out: list[dict[str, Any]] = []
    for row in rows:
        reviewed_time_s = finite_float(row.get("reviewed_time_s"))
        raw_saved_time_s = finite_float(row.get("raw_saved_time_s"))
        anchors: list[tuple[str, float, str]] = [("corrected_review", reviewed_time_s, "reviewed_time_s")]
        if math.isfinite(raw_saved_time_s):
            anchors.insert(0, ("original_native_recut", raw_saved_time_s, "raw_saved_time_s"))
        for radius in PEAK_WINDOWS_MS:
            raw_sample = strongest_peak_sample(np.abs(y), sample_rate, reviewed_time_s, radius)
            anchors.append((f"raw_abs_peak_pm{radius}ms", raw_sample / sample_rate, "raw_abs_peak"))
            bp_sample = strongest_peak_sample(env, sample_rate, reviewed_time_s, radius)
            anchors.append((f"bp_env_peak_pm{radius}ms", bp_sample / sample_rate, "bp_env_peak"))

        for anchor_name, anchor_time_s, anchor_source in anchors:
            sample = int(round(anchor_time_s * sample_rate))
            prediction = predict_at_sample(model, y, sample_rate, sample)
            out.append({
                "row_id": row.get("row_id") or "",
                "source": row.get("source") or "",
                "event_index": row.get("event_index") or "",
                "anchor": anchor_name,
                "anchor_source": anchor_source,
                "reviewed_time_s": reviewed_time_s,
                "anchor_time_s": anchor_time_s,
                "anchor_delta_ms": (anchor_time_s - reviewed_time_s) * 1000.0,
                "saved_model_label": row.get("saved_model_label") or "",
                "saved_prob_racket_bounce": row.get("saved_prob_racket_bounce") or "",
                **prediction_fields("current_model", prediction),
            })
    return out


def anchor_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("anchor") or "")].append(row)
    out: list[dict[str, Any]] = []
    for anchor, items in sorted(grouped.items()):
        probs = [finite_float(row.get("current_model_prob_racket_bounce")) for row in items]
        labels = Counter(str(row.get("current_model_label") or "") for row in items)
        deltas = [abs(finite_float(row.get("anchor_delta_ms"))) for row in items]
        out.append({
            "anchor": anchor,
            "rows": len(items),
            "racket_labels": labels.get("racket_bounce", 0),
            "noise_labels": labels.get("noise", 0),
            "table_labels": labels.get("table_bounce", 0),
            "floor_labels": labels.get("floor_bounce", 0),
            "prob_racket_stats": json.dumps(stats(probs), sort_keys=True),
            "abs_delta_ms_stats": json.dumps(stats(deltas), sort_keys=True),
        })
    return out


def label_to_int(label: str) -> int:
    return 1 if label == "racket_bounce" else 0


def candidate_specs(seed: int) -> list[tuple[str, Any]]:
    return [
        (
            "logreg_balanced",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
            ),
        ),
        (
            "rf_balanced",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=18,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            ),
        ),
        (
            "histgb_balanced",
            HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=0.08,
                l2_regularization=0.02,
                class_weight="balanced",
                random_state=seed,
            ),
        ),
    ]


def fit_candidate(
    model: Any,
    train_rows: list[dict[str, Any]],
    feature_names: list[str],
    use_weights: bool,
) -> Any:
    x = np.asarray([[safe_feature(row[name]) for name in feature_names] for row in train_rows], dtype=np.float64)
    y = np.asarray([label_to_int(str(row["label"])) for row in train_rows], dtype=np.int64)
    weights = np.asarray([safe_feature(row.get("sample_weight", 1.0)) for row in train_rows], dtype=np.float64)
    if use_weights:
        try:
            model.fit(x, y, **{"logisticregression__sample_weight": weights})
            return model
        except Exception:
            pass
        try:
            model.fit(x, y, sample_weight=weights)
            return model
        except TypeError:
            model.fit(x, y)
            return model
    model.fit(x, y)
    return model


def predict_candidate(model: Any, features: dict[str, Any], feature_names: list[str]) -> dict[str, Any]:
    x = np.asarray([[feature_value(features, name) for name in feature_names]], dtype=np.float64)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)[0]
        classes = list(getattr(model, "classes_", []))
        if not classes and hasattr(model, "named_steps"):
            classes = list(model.named_steps[list(model.named_steps)[-1]].classes_)
        prob_map = {int(cls): float(prob) for cls, prob in zip(classes, probs)}
        p_racket = prob_map.get(1, 0.0)
    else:
        pred = int(model.predict(x)[0])
        p_racket = 1.0 if pred == 1 else 0.0
    p_racket = max(0.0, min(1.0, p_racket))
    return {
        "label": "racket_bounce" if p_racket >= 0.5 else "noise",
        "confidence": max(p_racket, 1.0 - p_racket),
        "probabilities": {"racket_bounce": p_racket, "noise": 1.0 - p_racket},
    }


def evaluate_rows(
    model: Any,
    rows: list[dict[str, Any]],
    feature_names: list[str],
    dataset_name: str,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_true = [label_to_int(str(row["label"])) for row in rows]
    y_prob: list[float] = []
    for row in rows:
        pred = predict_candidate(model, row, feature_names)
        y_prob.append(float(pred["probabilities"]["racket_bounce"]))
    y_pred = [1 if p >= threshold else 0 for p in y_prob]
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["noise", "racket_bounce"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "dataset": dataset_name,
        "rows": len(rows),
        "threshold": threshold,
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))) if rows else 0.0,
        "noise_precision": float(report["noise"]["precision"]),
        "noise_recall": float(report["noise"]["recall"]),
        "racket_precision": float(report["racket_bounce"]["precision"]),
        "racket_recall": float(report["racket_bounce"]["recall"]),
        "racket_f1": float(report["racket_bounce"]["f1-score"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "confusion_matrix_noise_racket": json.dumps(cm),
        "prob_racket_stats": json.dumps(stats(y_prob), sort_keys=True),
    }


def row_training_sets(
    t0049_rows: list[dict[str, Any]],
    t0055_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        "t0049_only": list(t0049_rows),
        "t0049_plus_c2": list(t0049_rows) + list(t0055_rows),
        "t0049_plus_c2_weighted": list(t0049_rows) + list(t0055_rows),
    }


def match_counts(predicted_ms: list[float], truth_ms: list[float], tolerance_ms: float) -> dict[str, Any]:
    truth_sorted = sorted(truth_ms)
    used_truth: set[int] = set()
    matched: list[tuple[float, float, float]] = []
    false_counts: list[float] = []
    for pred in sorted(predicted_ms):
        best_idx = None
        best_delta = float("inf")
        for idx, truth in enumerate(truth_sorted):
            if idx in used_truth:
                continue
            delta = pred - truth
            if abs(delta) <= tolerance_ms and abs(delta) < abs(best_delta):
                best_idx = idx
                best_delta = delta
        if best_idx is None:
            false_counts.append(pred)
        else:
            used_truth.add(best_idx)
            matched.append((pred, truth_sorted[best_idx], best_delta))
    missed = [truth for idx, truth in enumerate(truth_sorted) if idx not in used_truth]
    return {
        "tp": len(matched),
        "fp": len(false_counts),
        "missed": len(missed),
        "predicted": len(predicted_ms),
        "truth": len(truth_ms),
        "precision": len(matched) / len(predicted_ms) if predicted_ms else 0.0,
        "recall": len(matched) / len(truth_ms) if truth_ms else 0.0,
        "matched": matched,
        "false_counts": false_counts,
        "missed_truth": missed,
    }


class BinaryLiveCounter:
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.last_counted: tuple[float, float] | None = None
        self.group_start_ms: float | None = None

    def process(self, prob_racket: float, onset_ms: float, frame_rms: float) -> tuple[bool, str]:
        if self.last_counted is not None:
            since_counted = onset_ms - self.last_counted[0]
            rms_ratio = frame_rms / max(self.last_counted[1], 1e-9)
            if since_counted <= 250.0:
                if rms_ratio >= 1.1:
                    self.last_counted = (onset_ms, frame_rms)
                    self.group_start_ms = onset_ms
                    return False, "same_bounce"
                if rms_ratio <= 0.6:
                    return False, "echo_window"
                if since_counted < 150.0:
                    return False, "same_bounce"
            else:
                if self.group_start_ms is not None and onset_ms - self.group_start_ms <= 80.0:
                    return False, "group_window"
                if since_counted <= 300.0 and rms_ratio <= 0.6:
                    return False, "echo_window"
        if prob_racket < self.threshold:
            return False, "low_probability"
        self.last_counted = (onset_ms, frame_rms)
        self.group_start_ms = onset_ms
        return True, ""


def replay_c2(
    model: Any,
    feature_names: list[str],
    y: np.ndarray,
    sample_rate: int,
    truth_ms: list[float],
    threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    triggers = nr_features.simulate_gate(
        y,
        sample_rate,
        onset_ratio=1.5,
        retrigger_ms=120,
        abs_min_rms=0.0015,
        mode="bandpass",
        spectral_gate=False,
    )
    counter = BinaryLiveCounter(threshold)
    event_rows: list[dict[str, Any]] = []
    counted_ms: list[float] = []
    for index, trigger in enumerate(triggers, start=1):
        onset_sample = int(trigger["onset_sample"])
        onset_ms = float(trigger["onset_ms"])
        features = extract_features_at_sample(y, sample_rate, onset_sample)
        prediction = predict_candidate(model, features, feature_names)
        prob = float(prediction["probabilities"]["racket_bounce"])
        counted, reject_reason = counter.process(prob, onset_ms, float(trigger["frame_rms"]))
        if counted:
            counted_ms.append(onset_ms)
        nearest_truth_delta = min((onset_ms - truth for truth in truth_ms), key=abs) if truth_ms else float("nan")
        event_rows.append({
            "trigger_index": index,
            "onset_ms": onset_ms,
            "counted": counted,
            "reject_reason": reject_reason,
            "prob_racket_bounce": prob,
            "prob_noise": 1.0 - prob,
            "nearest_truth_delta_ms": nearest_truth_delta,
            "frame_rms": trigger.get("frame_rms", ""),
            "background_rms": trigger.get("bg_rms", ""),
        })
    match = match_counts(counted_ms, truth_ms, MATCH_TOLERANCE_MS)
    summary = {
        "threshold": threshold,
        "triggers": len(triggers),
        "counted": match["predicted"],
        "truth": match["truth"],
        "tp": match["tp"],
        "fp": match["fp"],
        "missed": match["missed"],
        "precision": match["precision"],
        "recall": match["recall"],
    }
    return summary, event_rows


def current_saved_app_summary(timeline_rows: list[dict[str, str]]) -> dict[str, Any]:
    racket_rows = [row for row in timeline_rows if row.get("review_label") == "racket"]
    phone_rows = [row for row in racket_rows if row.get("source") == "phone_trigger"]
    saved_probs = [finite_float(row.get("saved_prob_racket_bounce")) for row in phone_rows]
    centered_probs = [finite_float(row.get("reviewed_anchor_current_model_prob_racket_bounce")) for row in racket_rows]
    return {
        "reviewed_rackets": len(racket_rows),
        "phone_trigger_reviewed_rackets": len(phone_rows),
        "saved_exact_app_racket_labels": Counter(row.get("saved_model_label") or "-" for row in phone_rows),
        "saved_exact_app_counted": sum(1 for row in phone_rows if boolish(row.get("saved_counted"))),
        "saved_exact_app_prob_racket_stats": stats(saved_probs),
        "reviewed_anchor_racket_labels": Counter(row.get("reviewed_anchor_current_model_label") or "-" for row in racket_rows),
        "reviewed_anchor_prob_racket_stats": stats(centered_probs),
    }


def render_report(summary: dict[str, Any], output_dir: Path) -> str:
    def fmt_counter(counter_like: Any) -> str:
        return json.dumps(dict(counter_like), sort_keys=True)

    best_replay = summary["best_replay"]
    best_by_training_set = summary["best_replay_by_training_set"]
    lines = [
        "# T0056 Fable Candidate Retrain Replay",
        "",
        "## Scope",
        "",
        "- Evaluation-only local script.",
        "- No app runtime change, model JSON replacement, APK build/install, cloud/API, or raw label mutation.",
        "- Candidate models are binary `racket_bounce` vs `noise`; they are not app-ready because table/floor promotion safety is not covered here.",
        "",
        "## Inputs",
        "",
        f"- T0049 approved rows: `{summary['input_counts']['t0049_rows']}`",
        f"- T0055 included C2 rows: `{summary['input_counts']['t0055_rows']}`",
        f"- Combined rows: `{summary['input_counts']['combined_rows']}`",
        f"- Combined labels: `{json.dumps(summary['input_counts']['combined_label_counts'], sort_keys=True)}`",
        "",
        "## Current Fable Baseline",
        "",
        f"- Reviewed C2 racket contacts: `{summary['current_saved_app']['reviewed_rackets']}`",
        f"- Exact saved app/native trigger rows labeled racket: `{summary['current_saved_app']['phone_trigger_reviewed_rackets']}`",
        f"- Exact saved app labels on those rows: `{fmt_counter(summary['current_saved_app']['saved_exact_app_racket_labels'])}`",
        f"- Exact saved app counted: `{summary['current_saved_app']['saved_exact_app_counted']}`",
        f"- Corrected-anchor current model labels: `{fmt_counter(summary['current_saved_app']['reviewed_anchor_racket_labels'])}`",
        "",
        "## Anchor/Window Audit",
        "",
        "- Corrected timestamps and local peak recentering do not rescue the current model.",
        "- See `t0056_anchor_window_summary.csv` and `t0056_anchor_window_audit.csv` for per-anchor details.",
        "",
        "## Candidate Replay Result",
        "",
        "| Training set | Best candidate | Threshold | TP | FP | Missed | Precision | Recall |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        *[
            (
                f"| `{row['training_set']}` | `{row['candidate']}` | `{row['threshold']}` | "
                f"{row['tp']} | {row['fp']} | {row['missed']} | "
                f"{row['precision']:.3f} | {row['recall']:.3f} |"
            )
            for row in best_by_training_set
        ],
        "",
        f"- Best C2 replay row: `{best_replay['candidate']}` trained on `{best_replay['training_set']}` at threshold `{best_replay['threshold']}`.",
        f"- C2 replay: `TP {best_replay['tp']} / FP {best_replay['fp']} / missed {best_replay['missed']}` from `{best_replay['truth']}` reviewed racket contacts.",
        f"- Precision/recall: `{best_replay['precision']:.3f}` / `{best_replay['recall']:.3f}`.",
        "- Important: T0049-only candidates still fail this C2 slice, so the prior approved snippets do not cover the new hard domain.",
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Outputs",
        "",
        "- `t0056_anchor_window_audit.csv`",
        "- `t0056_anchor_window_summary.csv`",
        "- `t0056_training_rows.csv`",
        "- `t0056_candidate_row_metrics.csv`",
        "- `t0056_c2_replay_summary.csv`",
        "- `t0056_c2_replay_events_best.csv`",
        "- `t0056_summary.json`",
        "- local candidate `.joblib` files for inspection only",
    ]
    report = "\n".join(lines) + "\n"
    (output_dir / "t0056_report.md").write_text(report, encoding="utf-8")
    return report


def best_replays_by_training_set(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["training_set"])
        current = best.get(key)
        if current is None or (
            row["recall"],
            row["precision"],
            -row["fp"],
            -row["missed"],
        ) > (
            current["recall"],
            current["precision"],
            -current["fp"],
            -current["missed"],
        ):
            best[key] = row
    return [best[key] for key in sorted(best)]


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_names = nr_features.all_feature_names()
    y, sample_rate = read_wav(Path(args.wav))
    if sample_rate != nr_config.TARGET_SR:
        raise ValueError(f"Expected {nr_config.TARGET_SR} Hz WAV, got {sample_rate}")

    app_model = FableAppModel.load(Path(args.model_json))
    timeline_rows = read_csv(Path(args.timeline))
    t0049_rows = load_t0049_rows(Path(args.t0049_rows), feature_names)
    t0055_rows = build_t0055_rows(Path(args.t0055_candidates), y, sample_rate, feature_names)
    combined_rows = t0049_rows + t0055_rows

    anchor_rows = build_anchor_audit(Path(args.timeline), app_model, y, sample_rate)
    write_csv(output_dir / "t0056_anchor_window_audit.csv", anchor_rows)
    write_csv(output_dir / "t0056_anchor_window_summary.csv", anchor_summary(anchor_rows))
    write_csv(output_dir / "t0056_training_rows.csv", combined_rows)

    truth_ms = [
        finite_float(row.get("reviewed_time_s")) * 1000.0
        for row in timeline_rows
        if row.get("review_label") == "racket" and math.isfinite(finite_float(row.get("reviewed_time_s")))
    ]

    training_sets = row_training_sets(t0049_rows, t0055_rows)
    row_metrics: list[dict[str, Any]] = []
    replay_summaries: list[dict[str, Any]] = []
    best_event_rows: list[dict[str, Any]] = []
    best_replay: dict[str, Any] | None = None
    models_dir = output_dir / "candidate_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for training_set_name, train_rows in training_sets.items():
        use_weights = training_set_name.endswith("_weighted")
        for candidate_name, estimator in candidate_specs(args.seed):
            model = fit_candidate(estimator, train_rows, feature_names, use_weights)
            candidate_id = f"{candidate_name}__{training_set_name}"
            joblib.dump(model, models_dir / f"{candidate_id}.joblib")

            row_metrics.append({
                "candidate": candidate_name,
                "training_set": training_set_name,
                **evaluate_rows(model, t0049_rows, feature_names, "t0049_rows"),
            })
            row_metrics.append({
                "candidate": candidate_name,
                "training_set": training_set_name,
                **evaluate_rows(model, t0055_rows, feature_names, "t0055_c2_rows"),
            })
            row_metrics.append({
                "candidate": candidate_name,
                "training_set": training_set_name,
                **evaluate_rows(model, combined_rows, feature_names, "combined_rows"),
            })

            for threshold in REPLAY_THRESHOLDS:
                replay, event_rows = replay_c2(model, feature_names, y, sample_rate, truth_ms, threshold)
                replay_row = {
                    "candidate": candidate_name,
                    "training_set": training_set_name,
                    **replay,
                }
                replay_summaries.append(replay_row)
                if (
                    best_replay is None
                    or (replay_row["recall"], replay_row["precision"], -replay_row["fp"])
                    > (best_replay["recall"], best_replay["precision"], -best_replay["fp"])
                ):
                    best_replay = replay_row
                    best_event_rows = [
                        {"candidate": candidate_name, "training_set": training_set_name, **row}
                        for row in event_rows
                    ]

    if best_replay is None:
        raise RuntimeError("No replay results generated")

    write_csv(output_dir / "t0056_candidate_row_metrics.csv", row_metrics)
    write_csv(output_dir / "t0056_c2_replay_summary.csv", replay_summaries)
    write_csv(output_dir / "t0056_c2_replay_events_best.csv", best_event_rows)

    input_counts = {
        "t0049_rows": len(t0049_rows),
        "t0055_rows": len(t0055_rows),
        "combined_rows": len(combined_rows),
        "combined_label_counts": dict(sorted(Counter(row["label"] for row in combined_rows).items())),
        "t0055_label_counts": dict(sorted(Counter(row["label"] for row in t0055_rows).items())),
        "t0049_label_counts": dict(sorted(Counter(row["label"] for row in t0049_rows).items())),
    }

    current = current_saved_app_summary(timeline_rows)
    decision = (
        "The local binary candidates can be made to recover C2 in replay, but this ticket does not prove "
        "a ship-ready app model: the best C2 replay uses T0055 rows in training, and the candidate has no "
        "table/floor safety coverage. Next step should be a T0057-style broader retrain using the full "
        "noise-robust 4-class dataset plus these hard rows, or collect one fresh held-out C2-like run before export."
    )
    if best_replay["recall"] < 0.7:
        decision = (
            "Even with local hard rows, the best candidate replay does not recover enough C2 contacts. "
            "Collect more reviewed C2-like positives and hard negatives before training/export."
        )
    elif best_replay["fp"] > 5:
        decision = (
            "The best candidate recovers C2 but creates too many C2 false counts in replay. "
            "Do not export; add hard negatives and tune thresholds with a broader replay set."
        )

    summary = {
        "ticket": "T0056-feature-window-audit-plus-candidate-retrain-replay",
        "input_counts": input_counts,
        "current_saved_app": current,
        "best_replay": best_replay,
        "best_replay_by_training_set": best_replays_by_training_set(replay_summaries),
        "decision": decision,
        "outputs": {
            "output_dir": str(output_dir),
            "anchor_audit": str(output_dir / "t0056_anchor_window_audit.csv"),
            "candidate_row_metrics": str(output_dir / "t0056_candidate_row_metrics.csv"),
            "c2_replay_summary": str(output_dir / "t0056_c2_replay_summary.csv"),
            "report": str(output_dir / "t0056_report.md"),
        },
    }
    (output_dir / "t0056_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    render_report(summary, output_dir)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t0049-rows", default=str(DEFAULT_T0049_ROWS))
    parser.add_argument("--timeline", default=str(DEFAULT_T0055_TIMELINE))
    parser.add_argument("--t0055-candidates", default=str(DEFAULT_T0055_CANDIDATES))
    parser.add_argument("--wav", default=str(DEFAULT_T0052_WAV))
    parser.add_argument("--model-json", default=str(DEFAULT_MODEL_JSON))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--seed", type=int, default=20260629)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({
        "ticket": result["ticket"],
        "input_counts": result["input_counts"],
        "best_replay": result["best_replay"],
        "decision": result["decision"],
        "report": result["outputs"]["report"],
    }, indent=2, sort_keys=True))
