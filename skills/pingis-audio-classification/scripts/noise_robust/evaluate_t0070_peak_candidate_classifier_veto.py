#!/usr/bin/env python3
"""T0070 local peak-candidate classifier/veto evaluation.

This is evaluation-only. It trains local diagnostic classifiers from existing
peak-centered rows, writes ignored artifacts under /data, and does not export a
Collector model JSON, build an APK, or change app/runtime behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
for _path in (str(SCRIPT_DIR), str(SCRIPTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import nr_features  # noqa: E402
from evaluate_fable_audio_reliability_t0044 import FableAppModel  # noqa: E402
from evaluate_t0067_peak_gate_replay import (  # noqa: E402
    load_exact_positive_truth,
    match_predictions,
    read_csv,
    read_wav,
    selected_review_rows,
    write_csv,
)
from evaluate_t0069_peak_fable_hybrid_replay import (  # noqa: E402
    DEFAULT_HELDOUT_LABELS,
    DEFAULT_HELDOUT_WAV,
    DEFAULT_MODEL_JSON,
    DEFAULT_RAW_DIR,
    DEFAULT_T0065_DIR,
    DEFAULT_T0066_DIR,
    HELDOUT_SESSION_ID,
    finite_float,
    intish,
    normalized_gate_events,
    truth_times_from_labels,
)

DEFAULT_T0069_DIR = ROOT / "data/audio/models/evaluations/t0069_peak_fable_hybrid_replay"
DEFAULT_OUT_DIR = ROOT / "data/audio/models/evaluations/t0070_peak_candidate_classifier_veto"
MATCH_TOLERANCES_MS = (140.0, 250.0)
LABEL_MATCH_TOLERANCE_MS = 140.0


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    label: str
    estimator: Any


@dataclass(frozen=True)
class PolicySpec:
    model_id: str
    model_label: str
    threshold: float
    dedupe_ms: float

    @property
    def pipeline_id(self) -> str:
        threshold_id = str(self.threshold).replace(".", "p")
        dedupe_id = int(self.dedupe_ms)
        return f"{self.model_id}_thr{threshold_id}_smart{dedupe_id}"

    @property
    def pipeline_label(self) -> str:
        return f"{self.model_label} p>={self.threshold:g} smart{int(self.dedupe_ms)}"


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            "rf_depth6_leaf4",
            "RF depth6 leaf4",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=4,
                class_weight="balanced_subsample",
                random_state=7001,
                n_jobs=-1,
            ),
        ),
        ModelSpec(
            "rf_depth10_leaf3",
            "RF depth10 leaf3",
            RandomForestClassifier(
                n_estimators=400,
                max_depth=10,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=7002,
                n_jobs=-1,
            ),
        ),
        ModelSpec(
            "extra_leaf4",
            "ExtraTrees leaf4",
            ExtraTreesClassifier(
                n_estimators=500,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=7003,
                n_jobs=-1,
            ),
        ),
        ModelSpec(
            "logreg_balanced",
            "LogReg balanced",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=7004,
                ),
            ),
        ),
    ]


def threshold_grid() -> list[float]:
    return [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90]


def feature_name_list(model: FableAppModel) -> list[str]:
    base = [
        "frame_rms",
        "bg_rms",
        "peak_value",
        "peak_ratio",
        "peak_z",
        "prev_gap_ms",
        "next_gap_ms",
        "neighbor_count_500ms",
        "prob_racket_bounce",
        "prob_noise",
        "prob_floor_bounce",
        "prob_table_bounce",
        "model_confidence",
        "model_is_racket",
        "model_is_noise",
        "model_is_floor",
        "model_is_table",
    ]
    return base + [f"feat_{name}" for name in model.feature_names]


def prediction_probabilities(model: FableAppModel, features: dict[str, Any]) -> dict[str, Any]:
    prediction = model.predict_features({key: finite_float(value, 0.0) for key, value in features.items()})
    probs = prediction.get("probabilities") or {}
    return {
        "model_label": str(prediction.get("label") or ""),
        "model_confidence": finite_float(prediction.get("confidence"), 0.0),
        "prob_racket_bounce": finite_float(probs.get("racket_bounce"), 0.0),
        "prob_noise": finite_float(probs.get("noise"), 0.0),
        "prob_floor_bounce": finite_float(probs.get("floor_bounce"), 0.0),
        "prob_table_bounce": finite_float(probs.get("table_bounce"), 0.0),
    }


def build_candidate_rows_for_session(
    *,
    model: FableAppModel,
    session_id: str,
    y: np.ndarray,
    sr: int,
    meta: dict[str, Any],
    truth_ms: list[float] | None = None,
    dataset_role: str,
    label_candidates: bool,
) -> list[dict[str, Any]]:
    gate_rows = sorted(normalized_gate_events(y, sr, "peak_fast_balanced"), key=lambda row: finite_float(row["time_ms"]))
    times = [finite_float(row["time_ms"]) for row in gate_rows]
    matches = match_predictions(times, truth_ms or [], LABEL_MATCH_TOLERANCE_MS) if truth_ms else {"matches": []}
    positive_by_candidate: dict[int, tuple[int, float]] = {
        int(pred_idx): (int(truth_idx), float(delta)) for pred_idx, truth_idx, delta in matches["matches"]
    }
    rows: list[dict[str, Any]] = []
    for idx, gate in enumerate(gate_rows):
        time_ms = finite_float(gate["time_ms"])
        onset_sample = int(gate["onset_sample"])
        clip = nr_features.extract_live_clip(y, onset_sample)
        features = nr_features.extract_all_features(clip, sr)
        fable = prediction_probabilities(model, features)
        prev_gap = time_ms - times[idx - 1] if idx > 0 else 99999.0
        next_gap = times[idx + 1] - time_ms if idx + 1 < len(times) else 99999.0
        neighbors_500 = sum(1 for other in times if 0.0 < abs(other - time_ms) <= 500.0)
        label = ""
        label_source = "unlabeled"
        nearest_truth_delta = ""
        if label_candidates:
            if idx in positive_by_candidate:
                label = 1
                label_source = "reviewed_racket_match"
                nearest_truth_delta = positive_by_candidate[idx][1]
            elif truth_ms:
                label = 0
                label_source = "reviewed_positive_clip_extra_peak"
                nearest_truth_delta = min((time_ms - truth for truth in truth_ms), key=abs) if truth_ms else ""
            else:
                label = 0
                label_source = "scenario_expected_zero_hard_negative"
        row = {
            "session_id": session_id,
            "scenario_id": meta.get("scenario_id", ""),
            "scenario_title": meta.get("scenario_title", ""),
            "polarity": meta.get("polarity", ""),
            "dataset_role": dataset_role,
            "candidate_index": idx + 1,
            "time_ms": time_ms,
            "onset_sample": onset_sample,
            "label": label,
            "label_source": label_source,
            "nearest_truth_delta_ms": nearest_truth_delta,
            "expected_contacts": meta.get("expected_racket_contacts", ""),
            "frame_rms": finite_float(gate.get("frame_rms"), 0.0),
            "bg_rms": finite_float(gate.get("bg_rms"), 0.0),
            "peak_value": finite_float(gate.get("peak_value"), 0.0),
            "peak_ratio": finite_float(gate.get("peak_ratio"), 0.0),
            "peak_z": finite_float(gate.get("peak_z"), 0.0),
            "prev_gap_ms": prev_gap,
            "next_gap_ms": next_gap,
            "neighbor_count_500ms": neighbors_500,
            **fable,
            "model_is_racket": 1 if fable["model_label"] == "racket_bounce" else 0,
            "model_is_noise": 1 if fable["model_label"] == "noise" else 0,
            "model_is_floor": 1 if fable["model_label"] == "floor_bounce" else 0,
            "model_is_table": 1 if fable["model_label"] == "table_bounce" else 0,
        }
        for name in model.feature_names:
            row[f"feat_{name}"] = finite_float(features.get(name), 0.0)
        rows.append(row)
    return rows


def build_all_candidate_rows(args: argparse.Namespace, model: FableAppModel) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    raw_dir = Path(args.raw_dir)
    t0065_dir = Path(args.t0065_dir)
    t0066_dir = Path(args.t0066_dir)
    manifest_rows = [row for row in read_csv(t0065_dir / "t0065_fable_training_audio_manifest.csv") if row.get("round") == "round_a"]
    manifest_by_session = {row["session_id"]: row for row in manifest_rows}
    selected_rows = selected_review_rows(t0066_dir)
    truth_by_session = load_exact_positive_truth(t0066_dir)
    heldout_truth = truth_times_from_labels(Path(args.heldout_labels))

    selected_session_ids = {row["session_id"] for row in selected_rows}
    selected_candidate_rows: list[dict[str, Any]] = []
    for item in selected_rows:
        sid = item["session_id"]
        y, sr = read_wav(raw_dir / f"{sid}.wav")
        selected_candidate_rows.extend(
            build_candidate_rows_for_session(
                model=model,
                session_id=sid,
                y=y,
                sr=sr,
                meta=item,
                truth_ms=truth_by_session.get(sid),
                dataset_role="selected_train_oof",
                label_candidates=True,
            )
        )

    round_a_candidate_rows: list[dict[str, Any]] = []
    for item in manifest_rows:
        sid = item["session_id"]
        y, sr = read_wav(raw_dir / f"{sid}.wav")
        is_selected = sid in selected_session_ids
        round_a_candidate_rows.extend(
            build_candidate_rows_for_session(
                model=model,
                session_id=sid,
                y=y,
                sr=sr,
                meta=item,
                truth_ms=truth_by_session.get(sid) if is_selected else None,
                dataset_role="round_a_replay",
                label_candidates=False,
            )
        )

    heldout_candidate_rows: list[dict[str, Any]] = []
    heldout_wav = Path(args.heldout_wav)
    if heldout_wav.exists() and heldout_truth:
        y, sr = read_wav(heldout_wav)
        heldout_candidate_rows.extend(
            build_candidate_rows_for_session(
                model=model,
                session_id=HELDOUT_SESSION_ID,
                y=y,
                sr=sr,
                meta={
                    "scenario_id": "heldout_c2",
                    "scenario_title": "Held-out C2 speech/background",
                    "polarity": "positive",
                    "expected_racket_contacts": str(len(heldout_truth)),
                },
                truth_ms=heldout_truth,
                dataset_role="heldout_c2",
                label_candidates=True,
            )
        )

    return selected_candidate_rows, round_a_candidate_rows, heldout_candidate_rows


def numeric_matrix(rows: list[dict[str, Any]], features: list[str]) -> np.ndarray:
    matrix = np.zeros((len(rows), len(features)), dtype=np.float64)
    for row_idx, row in enumerate(rows):
        for col_idx, name in enumerate(features):
            matrix[row_idx, col_idx] = finite_float(row.get(name), 0.0)
    return matrix


def labels_and_weights(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray([intish(row.get("label")) for row in rows], dtype=np.int32)
    weights = []
    for row in rows:
        source = str(row.get("label_source") or "")
        if source == "reviewed_racket_match":
            weights.append(2.0)
        elif source == "scenario_expected_zero_hard_negative":
            weights.append(1.2)
        else:
            weights.append(0.75)
    return y, np.asarray(weights, dtype=np.float64)


def fit_estimator(spec: ModelSpec, rows: list[dict[str, Any]], features: list[str]) -> Any:
    x = numeric_matrix(rows, features)
    y, weights = labels_and_weights(rows)
    estimator = clone(spec.estimator)
    try:
        if hasattr(estimator, "steps"):
            final_step_name = estimator.steps[-1][0]
            estimator.fit(x, y, **{f"{final_step_name}__sample_weight": weights})
        else:
            estimator.fit(x, y, sample_weight=weights)
    except (TypeError, ValueError):
        estimator.fit(x, y)
    return estimator


def predict_positive_probability(estimator: Any, rows: list[dict[str, Any]], features: list[str]) -> np.ndarray:
    if not rows:
        return np.zeros(0, dtype=np.float64)
    x = numeric_matrix(rows, features)
    probabilities = estimator.predict_proba(x)
    classes = list(getattr(estimator, "classes_", []))
    if not classes and hasattr(estimator, "named_steps"):
        final = list(estimator.named_steps.values())[-1]
        classes = list(getattr(final, "classes_", []))
    if 1 not in classes:
        return np.zeros(len(rows), dtype=np.float64)
    return probabilities[:, classes.index(1)]


def add_probabilities(rows: list[dict[str, Any]], probs: np.ndarray, model_id: str, model_label: str, probability_key: str = "clf_prob") -> list[dict[str, Any]]:
    out = []
    for row, prob in zip(rows, probs):
        out.append({**row, probability_key: float(prob), "classifier_id": model_id, "classifier_label": model_label})
    return out


def make_oof_predictions(selected_rows: list[dict[str, Any]], spec: ModelSpec, features: list[str]) -> list[dict[str, Any]]:
    session_ids = sorted({row["session_id"] for row in selected_rows})
    predictions: list[dict[str, Any]] = []
    for holdout_sid in session_ids:
        train_rows = [row for row in selected_rows if row["session_id"] != holdout_sid]
        holdout_rows = [row for row in selected_rows if row["session_id"] == holdout_sid]
        if not train_rows or len({intish(row.get("label")) for row in train_rows}) < 2:
            continue
        estimator = fit_estimator(spec, train_rows, features)
        probs = predict_positive_probability(estimator, holdout_rows, features)
        predictions.extend(add_probabilities(holdout_rows, probs, spec.model_id, spec.label, "oof_prob"))
    return sorted(predictions, key=lambda row: (row["session_id"], finite_float(row["time_ms"])))


def accepted_after_dedupe(rows: list[dict[str, Any]], prob_key: str, threshold: float, dedupe_ms: float) -> list[dict[str, Any]]:
    accepted = []
    for row in sorted(rows, key=lambda item: finite_float(item["time_ms"])):
        if finite_float(row.get(prob_key), 0.0) >= threshold:
            accepted.append(dict(row))
    clusters: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in accepted:
        if current and finite_float(row["time_ms"]) - finite_float(current[-1]["time_ms"]) > dedupe_ms:
            clusters.append(current)
            current = []
        current.append(row)
    if current:
        clusters.append(current)
    counted: list[dict[str, Any]] = []
    for cluster in clusters:
        counted.append(max(cluster, key=lambda item: finite_float(item.get(prob_key), 0.0)))
    return counted


def score_exact_sessions(
    *,
    rows: list[dict[str, Any]],
    policy: PolicySpec,
    prob_key: str,
    truth_by_session: dict[str, list[float]],
    selected_rows_meta: list[dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected_by_sid = {row["session_id"]: row for row in selected_rows_meta}
    detail: list[dict[str, Any]] = []
    matches_by_tol = {tol: {"tp": 0, "fp": 0, "missed": 0} for tol in MATCH_TOLERANCES_MS}
    truth_total = 0
    positive_counted = 0
    negative_counted = 0
    negative_by_scenario: Counter[str] = Counter()
    candidates_positive = 0
    candidates_negative = 0
    rows_by_session: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_session.setdefault(row["session_id"], []).append(row)
    for sid, session_rows in rows_by_session.items():
        meta = selected_by_sid.get(sid, {})
        counted = accepted_after_dedupe(session_rows, prob_key, policy.threshold, policy.dedupe_ms)
        pred_ms = [finite_float(row["time_ms"]) for row in counted]
        if sid in truth_by_session:
            truth = truth_by_session[sid]
            truth_total += len(truth)
            positive_counted += len(pred_ms)
            candidates_positive += len(session_rows)
            detail_row: dict[str, Any] = {
                "pipeline_id": policy.pipeline_id,
                "pipeline_label": policy.pipeline_label,
                "session_id": sid,
                "scenario_title": meta.get("scenario_title", ""),
                "polarity": "positive_exact",
                "truth": len(truth),
                "candidates": len(session_rows),
                "counted": len(pred_ms),
            }
            for tol in MATCH_TOLERANCES_MS:
                matched = match_predictions(pred_ms, truth, tol)
                matches_by_tol[tol]["tp"] += matched["tp"]
                matches_by_tol[tol]["fp"] += matched["fp"]
                matches_by_tol[tol]["missed"] += matched["missed"]
                detail_row[f"tp_{int(tol)}ms"] = matched["tp"]
                detail_row[f"fp_{int(tol)}ms"] = matched["fp"]
                detail_row[f"missed_{int(tol)}ms"] = matched["missed"]
            detail.append(detail_row)
        else:
            negative_counted += len(pred_ms)
            candidates_negative += len(session_rows)
            negative_by_scenario[meta.get("scenario_title", "")] += len(pred_ms)
            detail.append(
                {
                    "pipeline_id": policy.pipeline_id,
                    "pipeline_label": policy.pipeline_label,
                    "session_id": sid,
                    "scenario_title": meta.get("scenario_title", ""),
                    "polarity": "expected_zero",
                    "truth": 0,
                    "candidates": len(session_rows),
                    "counted": len(pred_ms),
                }
            )
    summary: dict[str, Any] = {
        "pipeline_id": policy.pipeline_id,
        "pipeline_label": policy.pipeline_label,
        "classifier_id": policy.model_id,
        "classifier_label": policy.model_label,
        "threshold": policy.threshold,
        "dedupe_ms": policy.dedupe_ms,
        "truth": truth_total,
        "positive_candidates": candidates_positive,
        "positive_counted": positive_counted,
        "selected_expected_zero_candidates": candidates_negative,
        "selected_expected_zero_false_counts": negative_counted,
        "selected_expected_zero_by_scenario": json.dumps(dict(sorted(negative_by_scenario.items())), sort_keys=True),
    }
    for tol in MATCH_TOLERANCES_MS:
        tp = matches_by_tol[tol]["tp"]
        fp_pos = matches_by_tol[tol]["fp"]
        missed = matches_by_tol[tol]["missed"]
        all_fp = fp_pos + negative_counted
        precision = tp / (tp + all_fp) if (tp + all_fp) else 0.0
        recall = tp / truth_total if truth_total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        summary.update(
            {
                f"tp_{int(tol)}ms": tp,
                f"positive_fp_{int(tol)}ms": fp_pos,
                f"missed_{int(tol)}ms": missed,
                f"precision_including_negatives_{int(tol)}ms": precision,
                f"recall_{int(tol)}ms": recall,
                f"f1_including_negatives_{int(tol)}ms": f1,
            }
        )
    return summary, detail


def score_single_exact(
    *,
    rows: list[dict[str, Any]],
    policy: PolicySpec,
    prob_key: str,
    truth_ms: list[float],
) -> dict[str, Any]:
    counted = accepted_after_dedupe(rows, prob_key, policy.threshold, policy.dedupe_ms)
    pred_ms = [finite_float(row["time_ms"]) for row in counted]
    out: dict[str, Any] = {
        "pipeline_id": policy.pipeline_id,
        "pipeline_label": policy.pipeline_label,
        "classifier_id": policy.model_id,
        "classifier_label": policy.model_label,
        "threshold": policy.threshold,
        "dedupe_ms": policy.dedupe_ms,
        "candidates": len(rows),
        "counted": len(pred_ms),
        "truth": len(truth_ms),
    }
    for tol in MATCH_TOLERANCES_MS:
        matched = match_predictions(pred_ms, truth_ms, tol)
        precision = matched["tp"] / (matched["tp"] + matched["fp"]) if (matched["tp"] + matched["fp"]) else 0.0
        recall = matched["tp"] / len(truth_ms) if truth_ms else 0.0
        out.update(
            {
                f"tp_{int(tol)}ms": matched["tp"],
                f"fp_{int(tol)}ms": matched["fp"],
                f"missed_{int(tol)}ms": matched["missed"],
                f"precision_{int(tol)}ms": precision,
                f"recall_{int(tol)}ms": recall,
            }
        )
    return out


def round_a_replay_rows(
    *,
    rows: list[dict[str, Any]],
    policy: PolicySpec,
    prob_key: str,
    manifest_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    by_session: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_session.setdefault(row["session_id"], []).append(row)
    out: list[dict[str, Any]] = []
    for meta in manifest_rows:
        sid = meta["session_id"]
        session_rows = by_session.get(sid, [])
        counted = accepted_after_dedupe(session_rows, prob_key, policy.threshold, policy.dedupe_ms)
        expected = intish(meta.get("expected_racket_contacts"))
        out.append(
            {
                "pipeline_id": policy.pipeline_id,
                "pipeline_label": policy.pipeline_label,
                "classifier_id": policy.model_id,
                "classifier_label": policy.model_label,
                "threshold": policy.threshold,
                "dedupe_ms": policy.dedupe_ms,
                "session_id": sid,
                "scenario_id": meta.get("scenario_id", ""),
                "scenario_title": meta.get("scenario_title", ""),
                "polarity": meta.get("polarity", ""),
                "expected_contacts": expected,
                "candidate_count": len(session_rows),
                "counted": len(counted),
                "count_error": len(counted) - expected,
                "abs_count_error": abs(len(counted) - expected),
                "duration_s": finite_float(meta.get("wav_duration_s"), 0.0),
            }
        )
    return out


def summarize_round_a(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_scenario: dict[tuple[str, str], dict[str, Any]] = {}
    by_pipeline: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = (row["pipeline_id"], row["scenario_id"])
        scenario = by_scenario.setdefault(
            key,
            {
                "pipeline_id": row["pipeline_id"],
                "pipeline_label": row["pipeline_label"],
                "classifier_id": row["classifier_id"],
                "scenario_id": row["scenario_id"],
                "scenario_title": row["scenario_title"],
                "polarity": row["polarity"],
                "clips": 0,
                "expected_contacts": 0,
                "candidate_count": 0,
                "counted": 0,
                "abs_count_error": 0,
                "duration_s": 0.0,
            },
        )
        total = by_pipeline.setdefault(
            row["pipeline_id"],
            {
                "pipeline_id": row["pipeline_id"],
                "pipeline_label": row["pipeline_label"],
                "classifier_id": row["classifier_id"],
                "classifier_label": row["classifier_label"],
                "threshold": row["threshold"],
                "dedupe_ms": row["dedupe_ms"],
                "clips": 0,
                "positive_expected": 0,
                "positive_counted": 0,
                "positive_abs_count_error": 0,
                "negative_false_counts": 0,
                "candidate_count": 0,
            },
        )
        for target in (scenario, total):
            target["clips"] += 1
            target["candidate_count"] += int(row["candidate_count"])
        scenario["expected_contacts"] += int(row["expected_contacts"])
        scenario["counted"] += int(row["counted"])
        scenario["abs_count_error"] += int(row["abs_count_error"])
        scenario["duration_s"] += finite_float(row.get("duration_s"), 0.0)
        if row["polarity"] == "positive":
            total["positive_expected"] += int(row["expected_contacts"])
            total["positive_counted"] += int(row["counted"])
            total["positive_abs_count_error"] += int(row["abs_count_error"])
        else:
            total["negative_false_counts"] += int(row["counted"])
    for scenario in by_scenario.values():
        scenario["count_error"] = int(scenario["counted"]) - int(scenario["expected_contacts"])
    scenario_rows = sorted(by_scenario.values(), key=lambda row: (row["pipeline_id"], row["scenario_id"]))
    scenario_lookup = {(row["pipeline_id"], row["scenario_id"]): row for row in scenario_rows}
    for total in by_pipeline.values():
        for scenario_id, field in [
            ("fast_racket_bounce", "fast_counted"),
            ("racket_bounce_background_sound", "background_counted"),
            ("talking_only_no_bounce", "talking_only_false_counts"),
            ("racket_handling_no_bounce", "racket_handling_false_counts"),
            ("floor_table_other_impact_no_racket", "floor_table_other_false_counts"),
        ]:
            scenario = scenario_lookup.get((total["pipeline_id"], scenario_id))
            total[field] = int(scenario["counted"]) if scenario else 0
            total[f"{field}_expected"] = int(scenario["expected_contacts"]) if scenario else 0
            total[f"{field}_error"] = int(scenario["count_error"]) if scenario else 0
    total_rows = sorted(
        by_pipeline.values(),
        key=lambda row: (
            int(row.get("negative_false_counts", 999999)),
            int(row.get("positive_abs_count_error", 999999)),
            -int(row.get("background_counted", 0)),
        ),
    )
    return scenario_rows, total_rows


def load_t0069_baselines(t0069_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    exact_path = t0069_dir / "t0069_exact_selected_comparison.csv"
    heldout_path = t0069_dir / "t0069_heldout_c2_exact_comparison.csv"
    round_path = t0069_dir / "t0069_round_a_pipeline_summary.csv"
    baseline_ids = {
        "rms_current_app_counter",
        "peak_fb_prob0p02_smart300",
        "peak_fb_prob0p5_smart240",
    }
    exact = [row for row in read_csv_dicts(exact_path) if row.get("pipeline_id") in baseline_ids] if exact_path.exists() else []
    heldout = [row for row in read_csv_dicts(heldout_path) if row.get("pipeline_id") in baseline_ids] if heldout_path.exists() else []
    round_rows = [row for row in read_csv_dicts(round_path) if row.get("pipeline_id") in baseline_ids] if round_path.exists() else []
    for rows, source in [(exact, "t0069_exact"), (heldout, "t0069_heldout"), (round_rows, "t0069_round_a")]:
        for row in rows:
            row["source"] = source
    return exact, heldout, round_rows


def policy_sweep(
    *,
    selected_rows: list[dict[str, Any]],
    round_a_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
    feature_names: list[str],
    selected_meta: list[dict[str, str]],
    truth_by_session: dict[str, list[float]],
    heldout_truth: list[float],
    manifest_rows: list[dict[str, str]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    exact_summaries: list[dict[str, Any]] = []
    exact_details: list[dict[str, Any]] = []
    heldout_summaries: list[dict[str, Any]] = []
    round_block_rows: list[dict[str, Any]] = []
    oof_prediction_rows: list[dict[str, Any]] = []
    final_prediction_rows: list[dict[str, Any]] = []

    for spec in model_specs():
        oof_rows = make_oof_predictions(selected_rows, spec, feature_names)
        oof_prediction_rows.extend(oof_rows)
        estimator = fit_estimator(spec, selected_rows, feature_names)
        round_probs = predict_positive_probability(estimator, round_a_rows, feature_names)
        heldout_probs = predict_positive_probability(estimator, heldout_rows, feature_names)
        round_scored = add_probabilities(round_a_rows, round_probs, spec.model_id, spec.label)
        heldout_scored = add_probabilities(heldout_rows, heldout_probs, spec.model_id, spec.label)
        final_prediction_rows.extend(round_scored)
        final_prediction_rows.extend(heldout_scored)
        for threshold in threshold_grid():
            for dedupe_ms in (240.0, 300.0):
                policy = PolicySpec(spec.model_id, spec.label, threshold, dedupe_ms)
                exact_summary, exact_detail = score_exact_sessions(
                    rows=oof_rows,
                    policy=policy,
                    prob_key="oof_prob",
                    truth_by_session=truth_by_session,
                    selected_rows_meta=selected_meta,
                )
                exact_summaries.append(exact_summary)
                exact_details.extend(exact_detail)
                if heldout_scored:
                    heldout_summaries.append(
                        score_single_exact(
                            rows=heldout_scored,
                            policy=policy,
                            prob_key="clf_prob",
                            truth_ms=heldout_truth,
                        )
                    )
                round_block_rows.extend(
                    round_a_replay_rows(
                        rows=round_scored,
                        policy=policy,
                        prob_key="clf_prob",
                        manifest_rows=manifest_rows,
                    )
                )
    return exact_summaries, exact_details, heldout_summaries, round_block_rows, oof_prediction_rows, final_prediction_rows


def choose_candidates(
    exact_summaries: list[dict[str, Any]],
    heldout_summaries: list[dict[str, Any]],
    round_totals: list[dict[str, Any]],
    t0069_exact: list[dict[str, Any]],
    t0069_round: list[dict[str, Any]],
) -> dict[str, Any]:
    exact_by_id = {row["pipeline_id"]: row for row in exact_summaries}
    heldout_by_id = {row["pipeline_id"]: row for row in heldout_summaries}
    round_by_id = {row["pipeline_id"]: row for row in round_totals}
    baseline_exact = next((row for row in t0069_exact if row.get("pipeline_id") == "rms_current_app_counter"), {})
    baseline_round = next((row for row in t0069_round if row.get("pipeline_id") == "rms_current_app_counter"), {})

    def as_float(row: dict[str, Any], key: str) -> float:
        return finite_float(row.get(key), 0.0)

    def as_int(row: dict[str, Any], key: str) -> int:
        return intish(row.get(key))

    broad_passes = []
    runtime_passes = []
    for pipeline_id, exact in exact_by_id.items():
        round_row = round_by_id.get(pipeline_id, {})
        if not round_row:
            continue
        broad = (
            as_float(exact, "recall_140ms") >= as_float(baseline_exact, "recall_140ms")
            and as_int(exact, "selected_expected_zero_false_counts")
            <= as_int(baseline_exact, "selected_expected_zero_false_counts")
            and as_int(round_row, "negative_false_counts") <= as_int(baseline_round, "negative_false_counts")
            and as_int(round_row, "positive_abs_count_error") <= as_int(baseline_round, "positive_abs_count_error")
        )
        if not broad:
            continue
        broad_passes.append(pipeline_id)
        if as_int(round_row, "talking_only_false_counts") <= as_int(baseline_round, "talking_only_false_counts"):
            runtime_passes.append(pipeline_id)

    def rank_key(pipeline_id: str) -> tuple[int, int, float, int]:
        round_row = round_by_id[pipeline_id]
        exact = exact_by_id[pipeline_id]
        return (
            -as_int(round_row, "negative_false_counts"),
            -as_int(round_row, "positive_abs_count_error"),
            as_float(exact, "recall_140ms"),
            as_int(round_row, "background_counted"),
        )

    best_broad = max(broad_passes, key=rank_key) if broad_passes else ""
    best_runtime = max(runtime_passes, key=rank_key) if runtime_passes else ""
    best_exact = max(
        exact_summaries,
        key=lambda row: (
            as_float(row, "f1_including_negatives_140ms"),
            as_float(row, "recall_140ms"),
            -as_int(row, "selected_expected_zero_false_counts"),
        ),
    )
    best_heldout = max(
        heldout_summaries,
        key=lambda row: (
            as_float(row, "recall_140ms"),
            -as_int(row, "fp_140ms"),
            -abs(as_int(row, "counted") - as_int(row, "truth")),
        ),
    ) if heldout_summaries else {}
    recommendation = "do_not_export"
    if best_runtime:
        recommendation = "runtime_candidate_worth_manual_review_before_export"
    elif best_broad:
        recommendation = "aggregate_improvement_but_bucket_regression_blocks_export"
    return {
        "recommendation": recommendation,
        "best_exact_f1_pipeline": best_exact["pipeline_id"],
        "best_exact_f1_label": best_exact["pipeline_label"],
        "best_heldout_pipeline": best_heldout.get("pipeline_id", ""),
        "best_heldout_label": best_heldout.get("pipeline_label", ""),
        "best_broad_pipeline": best_broad,
        "best_broad_label": exact_by_id.get(best_broad, {}).get("pipeline_label", "") if best_broad else "",
        "best_runtime_pipeline": best_runtime,
        "best_runtime_label": exact_by_id.get(best_runtime, {}).get("pipeline_label", "") if best_runtime else "",
        "broad_pass_count": len(broad_passes),
        "runtime_pass_count": len(runtime_passes),
    }


def md_table(rows: list[dict[str, Any]], fields: list[str], labels: list[str] | None = None, limit: int | None = None) -> list[str]:
    if labels is None:
        labels = fields
    shown = rows[:limit] if limit is not None else rows
    lines = [
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join("---" for _ in labels) + " |",
    ]
    for row in shown:
        values = []
        for field in fields:
            value = row.get(field, "")
            number = finite_float(value, float("nan"))
            if math.isfinite(number) and str(value).strip() != "":
                if "recall" in field or "precision" in field or "f1" in field:
                    values.append(f"{number:.3f}")
                elif abs(number - round(number)) < 1e-9:
                    values.append(str(int(round(number))))
                else:
                    values.append(f"{number:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def render_report(
    *,
    summary: dict[str, Any],
    exact_summaries: list[dict[str, Any]],
    heldout_summaries: list[dict[str, Any]],
    round_totals: list[dict[str, Any]],
    scenario_rows: list[dict[str, Any]],
    t0069_exact: list[dict[str, Any]],
    t0069_heldout: list[dict[str, Any]],
    t0069_round: list[dict[str, Any]],
) -> str:
    recommendation = summary["recommendation"]
    exact_display = sorted(
        exact_summaries,
        key=lambda row: (
            -finite_float(row.get("f1_including_negatives_140ms"), 0.0),
            -finite_float(row.get("recall_140ms"), 0.0),
        ),
    )
    heldout_display = sorted(
        heldout_summaries,
        key=lambda row: (-finite_float(row.get("recall_140ms"), 0.0), intish(row.get("fp_140ms"))),
    )
    round_display = sorted(
        round_totals,
        key=lambda row: (intish(row.get("negative_false_counts")), intish(row.get("positive_abs_count_error"))),
    )
    best_round_tradeoff = min(
        round_totals,
        key=lambda row: intish(row.get("positive_abs_count_error")) + intish(row.get("negative_false_counts")),
    ) if round_totals else {}
    selected_pipeline_ids = {
        recommendation.get("best_exact_f1_pipeline"),
        recommendation.get("best_broad_pipeline"),
        recommendation.get("best_runtime_pipeline"),
        recommendation.get("best_heldout_pipeline"),
        best_round_tradeoff.get("pipeline_id"),
    }
    selected_pipeline_ids.discard(None)
    selected_pipeline_ids.discard("")
    comparison_round = [
        *t0069_round,
        *[row for row in round_totals if row["pipeline_id"] in selected_pipeline_ids],
    ]
    seen = set()
    comparison_round_unique = []
    for row in comparison_round:
        key = row.get("pipeline_id")
        if key in seen:
            continue
        seen.add(key)
        comparison_round_unique.append(row)
    scenario_display = [
        row
        for row in scenario_rows
        if row["pipeline_id"] in selected_pipeline_ids
    ]
    lines = [
        "# T0070 Peak-Candidate Classifier/Veto Evaluation",
        "",
        f"Generated at: `{summary['generated_at']}`",
        "",
        "## Scope",
        "",
        "- Evaluation only: no app model JSON, runtime logic, APK, camera, cloud/API, or AWS change.",
        "- Candidate source is T0068 `Peak fast balanced`.",
        "- Training rows come from T0066 reviewed background positives plus scenario-derived expected-zero hard negatives.",
        "- T0060 C2 exact labels are kept out of training and used as a sanity holdout.",
        "- Extra peaks in reviewed positive clips are treated as negative candidates; this is a practical label assumption and should be revisited if manual review finds missed racket contacts there.",
        "",
        "## Dataset",
        "",
        f"- Selected train/out-of-fold candidates: `{summary['selected_candidate_rows']}`",
        f"- Selected positives / negatives: `{summary['selected_positive_rows']}` / `{summary['selected_negative_rows']}`",
        "- Exact positive labels are currently narrow: the candidate classifier is trained on reviewed `Racket bounce + background sound` positives plus hard negatives, then stress-tested against the full Round A scenario mix.",
        f"- Round A replay candidates: `{summary['round_a_candidate_rows']}`",
        f"- Held-out C2 candidates: `{summary['heldout_candidate_rows']}`",
        "",
        "## Out-Of-Fold T0066 Exact Metrics",
        "",
        *md_table(
            exact_display,
            [
                "pipeline_label",
                "tp_140ms",
                "missed_140ms",
                "positive_fp_140ms",
                "selected_expected_zero_false_counts",
                "precision_including_negatives_140ms",
                "recall_140ms",
                "f1_including_negatives_140ms",
            ],
            ["Pipeline", "TP", "Miss", "Pos FP", "Neg FP", "Precision", "Recall", "F1"],
            limit=12,
        ),
        "",
        "## Held-Out T0060 C2",
        "",
        *md_table(
            heldout_display,
            [
                "pipeline_label",
                "candidates",
                "counted",
                "tp_140ms",
                "fp_140ms",
                "missed_140ms",
                "precision_140ms",
                "recall_140ms",
            ],
            ["Pipeline", "Cand.", "Count", "TP", "FP", "Miss", "Precision", "Recall"],
            limit=12,
        ),
        "",
        "## Round A Replay",
        "",
        *md_table(
            round_display,
            [
                "pipeline_label",
                "positive_counted",
                "positive_expected",
                "positive_abs_count_error",
                "negative_false_counts",
                "fast_counted",
                "background_counted",
                "talking_only_false_counts",
                "racket_handling_false_counts",
                "floor_table_other_false_counts",
            ],
            ["Pipeline", "Pos Count", "Pos Exp", "Pos Abs Err", "Neg FP", "Fast", "Background", "Talking FP", "Handling FP", "Floor/Table FP"],
            limit=12,
        ),
        "",
        "## Comparison To T0069 Baselines",
        "",
        *md_table(
            comparison_round_unique,
            [
                "pipeline_label",
                "positive_counted",
                "positive_expected",
                "positive_abs_count_error",
                "negative_false_counts",
                "background_counted",
                "talking_only_false_counts",
                "racket_handling_false_counts",
                "floor_table_other_false_counts",
            ],
            ["Pipeline", "Pos Count", "Pos Exp", "Pos Abs Err", "Neg FP", "Background", "Talking FP", "Handling FP", "Floor/Table FP"],
        ),
        "",
        "## Scenario Detail For Selected Classifier Pipelines",
        "",
        *md_table(
            scenario_display,
            ["pipeline_label", "scenario_title", "expected_contacts", "counted", "count_error", "candidate_count"],
            ["Pipeline", "Scenario", "Expected", "Count", "Error", "Cand."],
        ),
        "",
        "## Recommendation",
        "",
    ]
    if recommendation["recommendation"] == "runtime_candidate_worth_manual_review_before_export":
        lines.append(
            "- A classifier/veto candidate beats the current aggregate and bucket gates in this offline replay. It is still not exported; next step is manual bad-case review and parity planning before any app candidate."
        )
    elif recommendation["recommendation"] == "aggregate_improvement_but_bucket_regression_blocks_export":
        lines.append(
            "- A classifier/veto candidate improves aggregate metrics but still has a bucket regression, so do not export or install yet."
        )
    else:
        lines.append(
            "- Do not export or install from this T0070 run. The trained classifiers are useful diagnostics but do not yet beat the current path safely by scenario."
        )
    lines += [
        f"- Best OOF exact-F1 pipeline: `{recommendation.get('best_exact_f1_label')}`.",
        f"- Best held-out C2 pipeline: `{recommendation.get('best_heldout_label')}`.",
        f"- Best broad aggregate pipeline: `{recommendation.get('best_broad_label') or 'none'}`.",
        f"- Runtime-pass candidates: `{recommendation.get('runtime_pass_count')}`.",
        "- The next useful work is to inspect false positives/false negatives from the best classifier, then decide whether to collect/label more talking/handling/floor/table negatives or proceed to a guarded app candidate.",
        "",
        "## Outputs",
        "",
        "- `t0070_candidate_rows_selected.csv`",
        "- `t0070_candidate_rows_round_a.csv`",
        "- `t0070_candidate_rows_heldout_c2.csv`",
        "- `t0070_oof_exact_comparison.csv`",
        "- `t0070_heldout_c2_comparison.csv`",
        "- `t0070_round_a_block_replay.csv`",
        "- `t0070_round_a_by_scenario.csv`",
        "- `t0070_round_a_pipeline_summary.csv`",
        "- `t0070_summary.json`",
    ]
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = FableAppModel.load(Path(args.model_json))
    feature_names = feature_name_list(model)
    selected_candidate_rows, round_a_candidate_rows, heldout_candidate_rows = build_all_candidate_rows(args, model)
    selected_meta = selected_review_rows(Path(args.t0066_dir))
    truth_by_session = load_exact_positive_truth(Path(args.t0066_dir))
    heldout_truth = truth_times_from_labels(Path(args.heldout_labels))
    manifest_rows = [row for row in read_csv(Path(args.t0065_dir) / "t0065_fable_training_audio_manifest.csv") if row.get("round") == "round_a"]

    exact_summaries, exact_details, heldout_summaries, round_block_rows, oof_predictions, final_predictions = policy_sweep(
        selected_rows=selected_candidate_rows,
        round_a_rows=round_a_candidate_rows,
        heldout_rows=heldout_candidate_rows,
        feature_names=feature_names,
        selected_meta=selected_meta,
        truth_by_session=truth_by_session,
        heldout_truth=heldout_truth,
        manifest_rows=manifest_rows,
    )
    scenario_rows, round_total_rows = summarize_round_a(round_block_rows)
    t0069_exact, t0069_heldout, t0069_round = load_t0069_baselines(Path(args.t0069_dir))
    recommendation = choose_candidates(exact_summaries, heldout_summaries, round_total_rows, t0069_exact, t0069_round)

    exact_summaries.sort(
        key=lambda row: (
            -finite_float(row.get("f1_including_negatives_140ms"), 0.0),
            -finite_float(row.get("recall_140ms"), 0.0),
        )
    )
    heldout_summaries.sort(
        key=lambda row: (-finite_float(row.get("recall_140ms"), 0.0), intish(row.get("fp_140ms")))
    )
    round_total_rows.sort(
        key=lambda row: (intish(row.get("negative_false_counts")), intish(row.get("positive_abs_count_error")))
    )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_count": len(feature_names),
        "models_evaluated": len(model_specs()),
        "thresholds_evaluated": len(threshold_grid()),
        "policies_evaluated": len(exact_summaries),
        "selected_candidate_rows": len(selected_candidate_rows),
        "selected_positive_rows": sum(1 for row in selected_candidate_rows if intish(row.get("label")) == 1),
        "selected_negative_rows": sum(1 for row in selected_candidate_rows if intish(row.get("label")) == 0),
        "round_a_candidate_rows": len(round_a_candidate_rows),
        "heldout_candidate_rows": len(heldout_candidate_rows),
        "label_match_tolerance_ms": LABEL_MATCH_TOLERANCE_MS,
        "match_tolerances_ms": list(MATCH_TOLERANCES_MS),
        "recommendation": recommendation,
    }

    write_csv(out_dir / "t0070_candidate_rows_selected.csv", selected_candidate_rows)
    write_csv(out_dir / "t0070_candidate_rows_round_a.csv", round_a_candidate_rows)
    write_csv(out_dir / "t0070_candidate_rows_heldout_c2.csv", heldout_candidate_rows)
    write_csv(out_dir / "t0070_oof_predictions.csv", oof_predictions)
    write_csv(out_dir / "t0070_final_model_predictions.csv", final_predictions)
    write_csv(out_dir / "t0070_oof_exact_comparison.csv", exact_summaries)
    write_csv(out_dir / "t0070_oof_exact_clip_rows.csv", exact_details)
    write_csv(out_dir / "t0070_heldout_c2_comparison.csv", heldout_summaries)
    write_csv(out_dir / "t0070_round_a_block_replay.csv", round_block_rows)
    write_csv(out_dir / "t0070_round_a_by_scenario.csv", scenario_rows)
    write_csv(out_dir / "t0070_round_a_pipeline_summary.csv", round_total_rows)
    (out_dir / "t0070_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "t0070_peak_candidate_classifier_report.md").write_text(
        render_report(
            summary=summary,
            exact_summaries=exact_summaries,
            heldout_summaries=heldout_summaries,
            round_totals=round_total_rows,
            scenario_rows=scenario_rows,
            t0069_exact=t0069_exact,
            t0069_heldout=t0069_heldout,
            t0069_round=t0069_round,
        ),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--t0065-dir", default=str(DEFAULT_T0065_DIR))
    parser.add_argument("--t0066-dir", default=str(DEFAULT_T0066_DIR))
    parser.add_argument("--t0069-dir", default=str(DEFAULT_T0069_DIR))
    parser.add_argument("--model-json", default=str(DEFAULT_MODEL_JSON))
    parser.add_argument("--heldout-wav", default=str(DEFAULT_HELDOUT_WAV))
    parser.add_argument("--heldout-labels", default=str(DEFAULT_HELDOUT_LABELS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
