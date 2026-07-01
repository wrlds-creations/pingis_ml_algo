#!/usr/bin/env python3
"""T0103 boundary-label candidate loop.

This is an evaluation/training gate for the separate Bounce audio test path.
It joins the fresh T0102/T0103 boundary labels with the previous noisy target
and Round A safety rows, then compares the current app candidate with a small
set of trainable candidate policies. It does not export an app model or change
runtime behavior.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = ROOT / "data/audio/models/evaluations/t0103_boundary_label_candidate_phone_gate/candidate_loop_2026_07_01"
T0103_INGEST_DIR = ROOT / "data/audio/models/evaluations/t0103_boundary_label_candidate_phone_gate/full_label_ingest_2026_07_01"
T0102_REVIEW_DIR = ROOT / "data/audio/models/evaluations/t0102_boundary_recorder_pack_pull_review/fresh_pack_2026_07_01"
T0102_RAW_DIR = ROOT / "data/audio/raw/t0102_boundary_recorder_pack/fable_training_audio"
T0097_SCRIPT = ROOT / "data/audio/models/evaluations/t0097_target_domain_bounce_audio_candidate/run_t0097_target_domain_candidate.py"
T0099_SCRIPT = ROOT / "data/audio/models/evaluations/t0099_negative_aware_target_domain_candidate/run_t0099_negative_aware_candidate.py"
T0098_SCRIPT = ROOT / "data/audio/models/evaluations/t0098_t0097_false_count_broader_safety/run_t0098_false_count_safety.py"
T0075_APP_MODEL = ROOT / "apps/collector/src/models/fable_extra_trees_candidate_t0075.json"
DEFAULT_APP_MODEL_OUT = ROOT / "apps/collector/src/models/fable_extra_trees_candidate_t0103.json"

MATCH_TOLERANCE_MS = 140.0
THRESHOLDS = [0.20, 0.30, 0.40, 0.50, 0.575, 0.65, 0.75, 0.85]
DEDUPES_MS = [180.0, 220.0]
NOISE_VETO_THRESHOLDS: list[float | None] = [None, 0.95]
SELECTED_MODEL_ID = "extra_leaf4_t0103"
SELECTED_FEATURE_SET_ID = "base_t0075"
SELECTED_WEIGHT_STRATEGY_ID = "boundary_recall_safety"
SELECTED_THRESHOLD = 0.575
SELECTED_DEDUPE_MS = 180.0
POSITIVE_CLASS = 1
POSITIVE_LABEL = "racket_bounce"
NEGATIVE_LABEL = "not_racket_bounce"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    label: str
    estimator: Any


@dataclass(frozen=True)
class FeatureSet:
    feature_set_id: str
    label: str
    features: list[str]
    app_exportable: bool


@dataclass(frozen=True)
class WeightStrategy:
    strategy_id: str
    label: str
    weight_fn: Callable[[dict[str, Any]], float]


def load_module(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ff(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def intish(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fields.append(key)
                    seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def md_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return lines


def policy_id(parts: list[Any]) -> str:
    out = "_".join(str(part) for part in parts)
    return (
        out.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("+", "plus")
        .replace(".", "p")
        .replace("none", "noveto")
    )


def load_truth_labels() -> tuple[dict[str, list[float]], dict[str, dict[str, Any]]]:
    labels = read_csv(T0103_INGEST_DIR / "t0102f_reviewed_positive_labels.csv")
    truth: dict[str, list[float]] = defaultdict(list)
    meta: dict[str, dict[str, Any]] = {}
    for row in labels:
        sid = row["session_id"]
        truth[sid].append(ff(row.get("reviewed_time_ms")))
        meta.setdefault(
            sid,
            {
                "session_id": sid,
                "scenario_id": row.get("scenario_id", ""),
                "scenario_title": row.get("scenario_title", ""),
                "polarity": "positive",
                "expected_racket_contacts": row.get("expected_count", ""),
            },
        )
    return {sid: sorted(times) for sid, times in truth.items()}, meta


def load_t0102_manifest() -> dict[str, dict[str, Any]]:
    rows = read_csv(T0102_REVIEW_DIR / "t0102_recorder_manifest_target.csv")
    normal_rows = read_csv(T0102_REVIEW_DIR / "normal_noisy_review" / "t0102_review_page_manifest.csv")
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        sid = row["session_id"]
        out[sid] = {
            **row,
            "session_id": sid,
            "expected_racket_contacts": row.get("expected_racket_contacts", ""),
        }
    for row in normal_rows:
        sid = row["session_id"]
        out[sid] = {
            **row,
            "session_id": sid,
            "polarity": "positive",
            "expected_racket_contacts": row.get("expected_racket_contacts", ""),
        }
    return out


def boundary_group(row: dict[str, Any]) -> str:
    scenario = str(row.get("scenario_id") or "")
    mapping = {
        "far_soft_racket_bounce_background": "boundary_positive_far_soft",
        "soft_high_racket_bounce_background": "boundary_positive_soft_high",
        "racket_bounce_background_sound": "boundary_positive_normal_noisy",
        "background_sound_only_no_bounce": "boundary_negative_background_only",
        "talking_counting_background_no_bounce": "boundary_negative_talking_counting",
        "racket_handling_background_no_bounce": "boundary_negative_handling",
        "catch_after_sound_no_racket": "boundary_negative_catch_after",
        "ambiguous_ball_like_impact_near_phone_no_racket": "boundary_negative_ambiguous_impact",
    }
    return mapping.get(scenario, f"boundary_{scenario or 'unknown'}")


def make_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            "extra_leaf4_t0103",
            "ExtraTrees leaf4 T0103",
            ExtraTreesClassifier(
                n_estimators=240,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=10304,
                n_jobs=-1,
            ),
        ),
        ModelSpec(
            "extra_leaf8_t0103",
            "ExtraTrees leaf8 T0103",
            ExtraTreesClassifier(
                n_estimators=240,
                min_samples_leaf=8,
                class_weight="balanced",
                random_state=10308,
                n_jobs=-1,
            ),
        ),
        ModelSpec(
            "logreg_balanced_t0103",
            "LogReg balanced T0103",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(class_weight="balanced", max_iter=2000, random_state=10311),
            ),
        ),
    ]


def make_weight_strategies(t0099: Any) -> list[WeightStrategy]:
    def default(row: dict[str, Any]) -> float:
        return float(t0099.base_weight(row))

    def boundary_recall_safety(row: dict[str, Any]) -> float:
        weight = float(t0099.base_weight(row))
        domain = str(row.get("domain") or "")
        label = intish(row.get("label"))
        if domain == "boundary_t0103" and label == 1:
            weight *= 2.2
        if domain == "boundary_t0103" and label == 0:
            weight *= 2.0
        if domain == "round_a" and label == 0:
            weight *= 3.5
        if intish(row.get("is_rejected_unsafe")):
            weight *= 8.0
        return weight

    def strict_safety(row: dict[str, Any]) -> float:
        weight = float(t0099.base_weight(row))
        domain = str(row.get("domain") or "")
        label = intish(row.get("label"))
        if domain in {"boundary_t0103", "round_a"} and label == 0:
            weight *= 5.0
        if intish(row.get("is_rejected_unsafe")):
            weight *= 10.0
        return weight

    return [
        WeightStrategy("combined_default", "Default combined weights", default),
        WeightStrategy("boundary_recall_safety", "Boundary positives plus safety negatives", boundary_recall_safety),
        WeightStrategy("strict_safety", "Strict boundary/Round A negative weighting", strict_safety),
    ]


def numeric_matrix(rows: list[dict[str, Any]], features: list[str]) -> np.ndarray:
    matrix = np.zeros((len(rows), len(features)), dtype=np.float64)
    for row_idx, row in enumerate(rows):
        for col_idx, name in enumerate(features):
            matrix[row_idx, col_idx] = ff(row.get(name), 0.0)
    return matrix


def fit_estimator(estimator: Any, rows: list[dict[str, Any]], features: list[str], strategy: WeightStrategy) -> Any:
    x = numeric_matrix(rows, features)
    y = np.asarray([intish(row.get("label")) for row in rows], dtype=np.int32)
    weights = np.asarray([strategy.weight_fn(row) for row in rows], dtype=np.float64)
    fitted = clone(estimator)
    try:
        if hasattr(fitted, "steps"):
            final_step_name = fitted.steps[-1][0]
            fitted.fit(x, y, **{f"{final_step_name}__sample_weight": weights})
        else:
            fitted.fit(x, y, sample_weight=weights)
    except (TypeError, ValueError):
        fitted.fit(x, y)
    return fitted


def predict_probability(estimator: Any, rows: list[dict[str, Any]], features: list[str]) -> np.ndarray:
    if not rows:
        return np.zeros(0, dtype=np.float64)
    probs = estimator.predict_proba(numeric_matrix(rows, features))
    classes = list(getattr(estimator, "classes_", []))
    if not classes and hasattr(estimator, "named_steps"):
        classes = list(getattr(list(estimator.named_steps.values())[-1], "classes_", []))
    return probs[:, classes.index(1)] if 1 in classes else np.zeros(len(rows), dtype=np.float64)


def build_boundary_rows(t0097: Any, model: Any) -> tuple[list[dict[str, Any]], dict[str, list[float]], list[str]]:
    truth, label_meta = load_truth_labels()
    manifest = load_t0102_manifest()
    session_ids = sorted(set(truth) | {sid for sid, row in manifest.items() if row.get("polarity") == "negative"})
    rows: list[dict[str, Any]] = []
    acoustic_names: list[str] = []
    for sid in session_ids:
        meta = {**manifest.get(sid, {}), **label_meta.get(sid, {})}
        meta.setdefault("session_id", sid)
        meta.setdefault("scenario_id", "")
        meta.setdefault("scenario_title", "")
        meta.setdefault("polarity", "positive" if sid in truth else "negative")
        meta["session_group"] = boundary_group(meta)
        wav_path = T0102_RAW_DIR / f"{sid}.wav"
        if not wav_path.exists():
            raise FileNotFoundError(wav_path)
        y, sr = t0097.read_wav(wav_path)
        candidate_rows = t0097.build_candidate_rows_for_session(
            model=model,
            session_id=sid,
            y=y,
            sr=sr,
            meta=meta,
            truth_ms=truth.get(sid),
            dataset_role="t0103_boundary_loso",
            label_candidates=True,
        )
        for row in candidate_rows:
            acoustic = t0097.transient_features(y, sr, intish(row.get("onset_sample")))
            if not acoustic_names:
                acoustic_names = list(acoustic.keys())
            row.update(acoustic)
            row["domain"] = "boundary_t0103"
            row["domain_session_id"] = f"boundary::{sid}"
            row["eval_group"] = meta["session_group"]
            row["session_group"] = meta["session_group"]
            row["manual_review"] = ""
            row["is_rejected_unsafe"] = 0
            row["label"] = intish(row.get("label"))
            rows.append(row)
    return rows, truth, acoustic_names


def build_combined_rows() -> tuple[list[dict[str, Any]], dict[str, list[float]], dict[str, list[float]], list[str], dict[str, Any]]:
    t0097 = load_module("t0097_helpers_t0103", T0097_SCRIPT)
    t0098 = load_module("t0098_helpers_t0103", T0098_SCRIPT)
    t0099 = load_module("t0099_helpers_t0103", T0099_SCRIPT)
    model = t0097.FableAppModel.load(t0097.DEFAULT_MODEL_JSON)
    noisy_rows, noisy_truth, round_rows, round_truth, acoustic_names = t0099.enrich_rows(t0097, t0098)
    boundary_rows, boundary_truth, boundary_acoustic = build_boundary_rows(t0097, model)
    for name in boundary_acoustic:
        if name not in acoustic_names:
            acoustic_names.append(name)
    rows = noisy_rows + round_rows + boundary_rows
    modules = {"t0097": t0097, "t0099": t0099, "model": model}
    return rows, boundary_truth, round_truth, acoustic_names, modules


def app_model_probability(rows: list[dict[str, Any]]) -> np.ndarray:
    export = load_module("t0075_app_parity_t0103", ROOT / "skills/pingis-audio-classification/scripts/noise_robust/export_t0075_fable_extra_trees_app_parity.py")
    model = json.loads(T0075_APP_MODEL.read_text(encoding="utf-8"))
    return export.app_style_probabilities(model, rows)


def class_label(value: int) -> str:
    return POSITIVE_LABEL if int(value) == POSITIVE_CLASS else NEGATIVE_LABEL


def export_tree_full_precision(estimator: Any) -> list[list[float]]:
    tree = estimator.tree_
    nodes: list[list[float]] = []
    for index in range(tree.node_count):
        if tree.children_left[index] == -1:
            counts = tree.value[index][0].astype(float)
            total = float(counts.sum())
            probabilities = (counts / total).tolist() if total > 0 else counts.tolist()
            nodes.append([float(value) for value in probabilities])
        else:
            nodes.append(
                [
                    int(tree.feature[index]),
                    float(tree.threshold[index]),
                    int(tree.children_left[index]),
                    int(tree.children_right[index]),
                ]
            )
    return nodes


def export_candidate_model(
    estimator: Any,
    features: list[str],
    training_rows: list[dict[str, Any]],
    selected_policy: dict[str, Any] | None,
) -> dict[str, Any]:
    classes = [int(value) for value in estimator.classes_]
    if POSITIVE_CLASS not in classes:
        raise RuntimeError("Estimator does not expose positive class 1")
    positive_rows = sum(1 for row in training_rows if intish(row.get("label")) == 1)
    total_nodes = sum(int(tree.tree_.node_count) for tree in estimator.estimators_)
    labels = [class_label(value) for value in classes]
    return {
        "metadata": {
            "model_version": "fable_extra_trees_candidate_t0103",
            "source_ticket": "T0103-boundary-label-candidate-phone-gate",
            "selection_source": selected_policy.get("policy_id") if selected_policy else "",
            "model_type": "extra_trees_binary_peak_candidate",
            "candidate_gate": "peak_fast_balanced",
            "feature_version": "t0103_peak_candidate_features_plus_fable83",
            "selected_threshold": SELECTED_THRESHOLD,
            "smart_dedupe_ms": SELECTED_DEDUPE_MS,
            "fable_noise_veto_threshold": None,
            "positive_class": POSITIVE_CLASS,
            "positive_label": POSITIVE_LABEL,
            "classes": classes,
            "tree_count": len(estimator.estimators_),
            "total_nodes": total_nodes,
            "training_rows": len(training_rows),
            "training_positive_candidates": positive_rows,
            "training_negative_candidates": len(training_rows) - positive_rows,
            "normal_fable_model_unchanged": True,
            "runtime_status": "guarded_bounce_audio_test_only_not_production",
        },
        "labels": labels,
        "feature_names": features,
        "scaler_mean": [0.0 for _ in features],
        "scaler_std": [1.0 for _ in features],
        "trees": [export_tree_full_precision(tree) for tree in estimator.estimators_],
    }


def make_feature_sets(t0097: Any, model: Any, acoustic_names: list[str]) -> list[FeatureSet]:
    built = t0097.make_feature_sets(model, acoustic_names)
    out: list[FeatureSet] = []
    for item in built:
        if item.feature_set_id == "base_t0075":
            out.append(FeatureSet(item.feature_set_id, item.label, item.features, True))
        elif item.feature_set_id == "base_plus_transient":
            out.append(FeatureSet(item.feature_set_id, item.label, item.features, False))
    return out


def make_oof_predictions(
    rows: list[dict[str, Any]],
    specs: list[ModelSpec],
    feature_sets: list[FeatureSet],
    strategies: list[WeightStrategy],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    session_keys = sorted({str(row["domain_session_id"]) for row in rows})
    for spec in specs:
        for feature_set in feature_sets:
            if spec.model_id.startswith("logreg") and feature_set.feature_set_id == "base_t0075":
                continue
            for strategy in strategies:
                for heldout_key in session_keys:
                    train_rows = [row for row in rows if str(row["domain_session_id"]) != heldout_key]
                    heldout_rows = [row for row in rows if str(row["domain_session_id"]) == heldout_key]
                    if len({intish(row.get("label")) for row in train_rows}) < 2:
                        continue
                    fitted = fit_estimator(spec.estimator, train_rows, feature_set.features, strategy)
                    probs = predict_probability(fitted, heldout_rows, feature_set.features)
                    for row, prob in zip(heldout_rows, probs):
                        out.append(
                            {
                                **row,
                                "candidate_model_id": spec.model_id,
                                "candidate_model_label": spec.label,
                                "feature_set_id": feature_set.feature_set_id,
                                "feature_set_label": feature_set.label,
                                "app_exportable": feature_set.app_exportable and spec.model_id.startswith("extra_"),
                                "weight_strategy_id": strategy.strategy_id,
                                "weight_strategy_label": strategy.label,
                                "oof_prob": float(prob),
                            }
                        )
    return sorted(
        out,
        key=lambda row: (
            str(row.get("candidate_model_id")),
            str(row.get("feature_set_id")),
            str(row.get("weight_strategy_id")),
            str(row.get("domain")),
            str(row.get("session_id")),
            ff(row.get("time_ms")),
        ),
    )


def accepted_rows(
    rows: list[dict[str, Any]],
    threshold: float,
    dedupe_ms: float,
    *,
    prob_key: str,
    fable_noise_veto_threshold: float | None,
) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: ff(item.get("time_ms"))):
        if ff(row.get(prob_key), 0.0) < threshold:
            continue
        if fable_noise_veto_threshold is not None and ff(row.get("prob_noise"), 0.0) >= fable_noise_veto_threshold:
            continue
        eligible.append(row)
    clusters: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in eligible:
        if current and ff(row.get("time_ms")) - ff(current[-1].get("time_ms")) > dedupe_ms:
            clusters.append(current)
            current = []
        current.append(row)
    if current:
        clusters.append(current)
    return [max(cluster, key=lambda item: ff(item.get(prob_key))) for cluster in clusters]


def match_predictions(pred_ms: list[float], truth_ms: list[float]) -> dict[str, Any]:
    pairs: list[tuple[float, int, int]] = []
    for pred_idx, pred in enumerate(pred_ms):
        for truth_idx, truth in enumerate(truth_ms):
            delta = abs(pred - truth)
            if delta <= MATCH_TOLERANCE_MS:
                pairs.append((delta, pred_idx, truth_idx))
    pairs.sort(key=lambda item: item[0])
    used_pred: set[int] = set()
    used_truth: set[int] = set()
    for _delta, pred_idx, truth_idx in pairs:
        if pred_idx in used_pred or truth_idx in used_truth:
            continue
        used_pred.add(pred_idx)
        used_truth.add(truth_idx)
    return {
        "tp": len(used_truth),
        "fp": len(pred_ms) - len(used_pred),
        "missed": len(truth_ms) - len(used_truth),
    }


def score_policy(
    rows: list[dict[str, Any]],
    truth: dict[str, list[float]],
    threshold: float,
    dedupe_ms: float,
    *,
    prob_key: str,
    fable_noise_veto_threshold: float | None,
    domain: str | None = None,
) -> dict[str, Any]:
    selected = [row for row in rows if domain is None or row.get("domain") == domain]
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_session[str(row["session_id"])].append(row)
    group_counts = Counter()
    positive_tp = positive_fp = positive_missed = positive_pred = positive_truth = 0
    negative_false = 0
    false_rows: list[dict[str, Any]] = []
    for sid, session_rows in by_session.items():
        counted = accepted_rows(
            session_rows,
            threshold,
            dedupe_ms,
            prob_key=prob_key,
            fable_noise_veto_threshold=fable_noise_veto_threshold,
        )
        pred_ms = [ff(row.get("time_ms")) for row in counted]
        group = str(session_rows[0].get("eval_group") or session_rows[0].get("session_group") or "")
        if sid in truth:
            match = match_predictions(pred_ms, truth[sid])
            positive_tp += match["tp"]
            positive_fp += match["fp"]
            positive_missed += match["missed"]
            positive_pred += len(pred_ms)
            positive_truth += len(truth[sid])
            group_counts[(group, "truth")] += len(truth[sid])
            group_counts[(group, "tp")] += match["tp"]
            group_counts[(group, "pred")] += len(pred_ms)
            group_counts[(group, "fp")] += match["fp"]
            group_counts[(group, "missed")] += match["missed"]
        else:
            negative_false += len(pred_ms)
            group_counts[(group, "false")] += len(pred_ms)
            false_rows.extend(counted)
    precision = positive_tp / max(1, positive_tp + positive_fp)
    recall = positive_tp / max(1, positive_truth)
    return {
        "threshold": threshold,
        "dedupe_ms": int(dedupe_ms),
        "fable_noise_veto_threshold": "" if fable_noise_veto_threshold is None else fable_noise_veto_threshold,
        "positive_truth": positive_truth,
        "positive_predicted": positive_pred,
        "positive_tp_140": positive_tp,
        "positive_fp_140": positive_fp,
        "positive_missed_140": positive_missed,
        "positive_precision_140": round(precision, 4),
        "positive_recall_140": round(recall, 4),
        "positive_count_error": positive_pred - positive_truth,
        "negative_false_counts": negative_false,
        "false_row_count": len(false_rows),
        "group_counts": dict(group_counts),
    }


def summarize_boundary_groups(score: dict[str, Any]) -> dict[str, Any]:
    counts = score.get("group_counts", {})
    groups = [
        "boundary_positive_far_soft",
        "boundary_positive_soft_high",
        "boundary_positive_normal_noisy",
        "boundary_negative_background_only",
        "boundary_negative_talking_counting",
        "boundary_negative_handling",
        "boundary_negative_catch_after",
        "boundary_negative_ambiguous_impact",
    ]
    out: dict[str, Any] = {}
    for group in groups:
        truth = int(counts.get((group, "truth"), 0))
        tp = int(counts.get((group, "tp"), 0))
        false = int(counts.get((group, "false"), 0))
        out[f"{group}_tp"] = tp
        out[f"{group}_truth"] = truth
        out[f"{group}_recall"] = round(tp / max(1, truth), 4) if truth else ""
        out[f"{group}_false"] = false
    return out


def candidate_generation_summary(rows: list[dict[str, Any]], truth: dict[str, list[float]]) -> list[dict[str, Any]]:
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("domain") == "boundary_t0103":
            by_session[str(row["session_id"])].append(row)
    out: list[dict[str, Any]] = []
    for sid, session_rows in sorted(by_session.items()):
        if sid not in truth:
            out.append(
                {
                    "session_id": sid,
                    "scenario_id": session_rows[0].get("scenario_id", ""),
                    "session_group": session_rows[0].get("session_group", ""),
                    "truth": 0,
                    "candidate_count": len(session_rows),
                    "candidate_covered_140": "",
                    "coverage_140": "",
                }
            )
            continue
        candidate_ms = [ff(row.get("time_ms")) for row in session_rows]
        match = match_predictions(candidate_ms, truth[sid])
        out.append(
            {
                "session_id": sid,
                "scenario_id": session_rows[0].get("scenario_id", ""),
                "session_group": session_rows[0].get("session_group", ""),
                "truth": len(truth[sid]),
                "candidate_count": len(session_rows),
                "candidate_covered_140": match["tp"],
                "coverage_140": round(match["tp"] / max(1, len(truth[sid])), 4),
            }
        )
    return out


def evaluate_current_app_baseline(boundary_rows: list[dict[str, Any]], boundary_truth: dict[str, list[float]]) -> dict[str, Any]:
    probs = app_model_probability(boundary_rows)
    app_rows = []
    for row, prob in zip(boundary_rows, probs):
        item = dict(row)
        item["app_prob"] = float(prob)
        app_rows.append(item)
    score = score_policy(
        app_rows,
        boundary_truth,
        0.30,
        220.0,
        prob_key="app_prob",
        fable_noise_veto_threshold=0.95,
        domain="boundary_t0103",
    )
    out = {**score, **summarize_boundary_groups(score), "policy_id": "current_app_t0075_p0p3_noise0p95_smart220"}
    out.pop("group_counts", None)
    return out


def truth_from_candidate_rows(rows: list[dict[str, Any]], *, domain: str) -> dict[str, list[float]]:
    truth: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("domain") == domain and intish(row.get("label")) == 1:
            truth[str(row.get("session_id"))].append(ff(row.get("time_ms")))
    return {sid: sorted(times) for sid, times in truth.items()}


def sweep_predictions(predictions: list[dict[str, Any]], boundary_truth: dict[str, list[float]], round_truth: dict[str, list[float]]) -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    combos = sorted(
        {
            (
                row.get("candidate_model_id", ""),
                row.get("candidate_model_label", ""),
                row.get("feature_set_id", ""),
                row.get("feature_set_label", ""),
                row.get("weight_strategy_id", ""),
                row.get("weight_strategy_label", ""),
                bool(row.get("app_exportable")),
            )
            for row in predictions
        }
    )
    for model_id, model_label, feature_set_id, feature_set_label, strategy_id, strategy_label, app_exportable in combos:
        rows = [
            row
            for row in predictions
            if row.get("candidate_model_id") == model_id
            and row.get("feature_set_id") == feature_set_id
            and row.get("weight_strategy_id") == strategy_id
        ]
        for threshold in THRESHOLDS:
            for dedupe_ms in DEDUPES_MS:
                for noise_veto in NOISE_VETO_THRESHOLDS:
                    boundary_score = score_policy(
                        rows,
                        boundary_truth,
                        threshold,
                        dedupe_ms,
                        prob_key="oof_prob",
                        fable_noise_veto_threshold=noise_veto,
                        domain="boundary_t0103",
                    )
                    round_score = score_policy(
                        rows,
                        round_truth,
                        threshold,
                        dedupe_ms,
                        prob_key="oof_prob",
                        fable_noise_veto_threshold=noise_veto,
                        domain="round_a",
                    )
                    policy = {
                        "policy_id": policy_id([model_id, feature_set_id, strategy_id, f"thr{threshold}", f"dedupe{int(dedupe_ms)}", f"veto{noise_veto}"]),
                        "candidate_model_id": model_id,
                        "candidate_model_label": model_label,
                        "feature_set_id": feature_set_id,
                        "feature_set_label": feature_set_label,
                        "weight_strategy_id": strategy_id,
                        "weight_strategy_label": strategy_label,
                        "app_exportable": int(app_exportable),
                        "threshold": threshold,
                        "dedupe_ms": int(dedupe_ms),
                        "fable_noise_veto_threshold": "" if noise_veto is None else noise_veto,
                        "boundary_positive_tp_140": boundary_score["positive_tp_140"],
                        "boundary_positive_missed_140": boundary_score["positive_missed_140"],
                        "boundary_positive_recall_140": boundary_score["positive_recall_140"],
                        "boundary_positive_precision_140": boundary_score["positive_precision_140"],
                        "boundary_positive_count_error": boundary_score["positive_count_error"],
                        "boundary_negative_false_counts": boundary_score["negative_false_counts"],
                        "round_positive_tp_140": round_score["positive_tp_140"],
                        "round_positive_missed_140": round_score["positive_missed_140"],
                        "round_positive_recall_140": round_score["positive_recall_140"],
                        "round_negative_false_counts": round_score["negative_false_counts"],
                    }
                    policy.update(summarize_boundary_groups(boundary_score))
                    policies.append(policy)
    return policies


def select_candidate(policy_rows: list[dict[str, Any]], baseline: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    exportable = [row for row in policy_rows if intish(row.get("app_exportable")) == 1]
    practical = [
        row
        for row in exportable
        if ff(row.get("boundary_positive_recall_140")) >= 0.78
        and intish(row.get("boundary_negative_false_counts")) <= 6
        and intish(row.get("round_negative_false_counts")) <= 8
        and ff(row.get("round_positive_recall_140")) >= 0.95
    ]
    if practical:
        ranked = sorted(
            practical,
            key=lambda row: (
                -ff(row.get("boundary_positive_recall_140")),
                intish(row.get("boundary_negative_false_counts")),
                intish(row.get("round_negative_false_counts")),
                abs(ff(row.get("boundary_positive_count_error"))),
            ),
        )
        return "exportable_candidate_worth_phone_test", ranked[0]
    near = [
        row
        for row in policy_rows
        if ff(row.get("boundary_positive_recall_140")) > ff(baseline.get("positive_recall_140"))
        and intish(row.get("boundary_negative_false_counts")) <= max(6, intish(baseline.get("negative_false_counts")))
    ]
    if near:
        ranked = sorted(
            near,
            key=lambda row: (
                -ff(row.get("boundary_positive_recall_140")),
                intish(row.get("boundary_negative_false_counts")),
                intish(row.get("round_negative_false_counts")),
            ),
        )
        return "offline_improved_but_not_exportable_or_not_safe_enough", ranked[0]
    return "no_candidate_worth_phone_test_yet", None


def train_and_export_selected_app_model(
    rows: list[dict[str, Any]],
    feature_sets: list[FeatureSet],
    strategies: list[WeightStrategy],
    selected_policy: dict[str, Any] | None,
    app_model_out: Path,
    out_dir: Path,
) -> dict[str, Any]:
    features = next(item.features for item in feature_sets if item.feature_set_id == SELECTED_FEATURE_SET_ID)
    strategy = next(item for item in strategies if item.strategy_id == SELECTED_WEIGHT_STRATEGY_ID)
    spec = next(item for item in make_model_specs() if item.model_id == SELECTED_MODEL_ID)
    estimator = fit_estimator(spec.estimator, rows, features, strategy)
    model_json = export_candidate_model(estimator, features, rows, selected_policy)
    app_model_out.parent.mkdir(parents=True, exist_ok=True)
    app_model_out.write_text(json.dumps(model_json, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    archive_path = out_dir / "fable_extra_trees_candidate_t0103.json"
    archive_path.write_text(json.dumps(model_json, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "app_model_out": str(app_model_out),
        "archive_path": str(archive_path),
        "training_rows": len(rows),
        "feature_count": len(features),
        "tree_count": len(model_json["trees"]),
        "total_nodes": model_json["metadata"]["total_nodes"],
    }


def render_report(
    summary: dict[str, Any],
    candidate_summary: list[dict[str, Any]],
    baseline: dict[str, Any],
    top_rows: list[dict[str, Any]],
    selected: dict[str, Any] | None,
) -> str:
    lines = [
        "# T0103 Boundary Candidate Phone Gate",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        "## Decision",
        "",
        f"- Recommendation: `{summary['recommendation']}`",
        f"- App install prepared: `{summary['app_install_prepared']}`",
        "",
        "## Candidate Generation",
        "",
    ]
    lines.extend(md_table(candidate_summary, ["session_id", "session_group", "truth", "candidate_count", "candidate_covered_140", "coverage_140"]))
    lines.extend(
        [
            "",
            "## Current App Baseline",
            "",
            "Current `Bounce audio test` baseline here means T0075 app model, `p>=0.30`, Fable noise veto `>=0.95`, smart dedupe `220 ms`.",
            "",
        ]
    )
    baseline_fields = [
        "policy_id",
        "positive_tp_140",
        "positive_missed_140",
        "positive_recall_140",
        "negative_false_counts",
        "boundary_positive_far_soft_recall",
        "boundary_positive_soft_high_recall",
        "boundary_positive_normal_noisy_recall",
        "boundary_negative_background_only_false",
        "boundary_negative_talking_counting_false",
        "boundary_negative_handling_false",
        "boundary_negative_catch_after_false",
        "boundary_negative_ambiguous_impact_false",
    ]
    lines.extend(md_table([baseline], baseline_fields))
    lines.extend(["", "## Top Candidate Policies", ""])
    candidate_fields = [
        "policy_id",
        "app_exportable",
        "boundary_positive_tp_140",
        "boundary_positive_recall_140",
        "boundary_negative_false_counts",
        "round_positive_recall_140",
        "round_negative_false_counts",
        "threshold",
        "fable_noise_veto_threshold",
    ]
    lines.extend(md_table(top_rows, candidate_fields))
    lines.extend(["", "## Selected", ""])
    if selected:
        lines.extend(md_table([selected], candidate_fields))
    else:
        lines.append("_None._")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This pass is evaluation/training only. It does not export an app model, install an APK, or replace current behavior.",
            "- A candidate is considered phone-test-worthy only when it improves boundary recall and keeps new negatives plus older Round A hard negatives under control.",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--app-model-out", type=Path, default=DEFAULT_APP_MODEL_OUT)
    parser.add_argument("--export-app-model", action="store_true")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing candidate/prediction/policy CSVs in out-dir and only regenerate summary/report.",
    )
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_existing and (out_dir / "t0103_candidate_rows.csv").exists() and (out_dir / "t0103_policy_sweep.csv").exists():
        rows = read_csv(out_dir / "t0103_candidate_rows.csv")
        modules = None
        acoustic_names: list[str] = []
        _truth, _meta = load_truth_labels()
        boundary_truth = _truth
        round_truth = truth_from_candidate_rows(rows, domain="round_a")
        boundary_rows = [row for row in rows if row.get("domain") == "boundary_t0103"]
        candidate_summary = read_csv(out_dir / "t0103_candidate_generation_summary.csv")
        baseline = evaluate_current_app_baseline(boundary_rows, boundary_truth)
        predictions = read_csv(out_dir / "t0103_oof_predictions.csv") if (out_dir / "t0103_oof_predictions.csv").exists() else []
        policy_rows = read_csv(out_dir / "t0103_policy_sweep.csv")
    else:
        rows, boundary_truth, round_truth, acoustic_names, modules = build_combined_rows()
        boundary_rows = [row for row in rows if row.get("domain") == "boundary_t0103"]
        candidate_summary = candidate_generation_summary(rows, boundary_truth)
        baseline = evaluate_current_app_baseline(boundary_rows, boundary_truth)

        feature_sets = make_feature_sets(modules["t0097"], modules["model"], acoustic_names)
        specs = make_model_specs()
        strategies = make_weight_strategies(modules["t0099"])
        predictions = make_oof_predictions(rows, specs, feature_sets, strategies)
        policy_rows = sweep_predictions(predictions, boundary_truth, round_truth)
    recommendation, selected = select_candidate(policy_rows, baseline)
    export_summary: dict[str, Any] | None = None

    if args.export_app_model:
        if recommendation != "exportable_candidate_worth_phone_test":
            raise RuntimeError(f"Refusing app export because recommendation is {recommendation!r}")
        if modules is None:
            rows, _boundary_truth2, _round_truth2, acoustic_names, modules = build_combined_rows()
        feature_sets_for_export = make_feature_sets(modules["t0097"], modules["model"], acoustic_names)
        strategies_for_export = make_weight_strategies(modules["t0099"])
        app_model_out = args.app_model_out if args.app_model_out.is_absolute() else ROOT / args.app_model_out
        export_summary = train_and_export_selected_app_model(
            rows,
            feature_sets_for_export,
            strategies_for_export,
            selected,
            app_model_out,
            out_dir,
        )

    top_rows = sorted(
        policy_rows,
        key=lambda row: (
            intish(row.get("boundary_negative_false_counts")),
            intish(row.get("round_negative_false_counts")),
            -ff(row.get("boundary_positive_recall_140")),
        ),
    )[:24]
    top_recall_rows = sorted(
        policy_rows,
        key=lambda row: (
            -ff(row.get("boundary_positive_recall_140")),
            intish(row.get("boundary_negative_false_counts")),
            intish(row.get("round_negative_false_counts")),
        ),
    )[:24]
    top_combined = top_rows + [row for row in top_recall_rows if row not in top_rows]

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows),
        "boundary_rows": len(boundary_rows),
        "boundary_truth": sum(len(values) for values in boundary_truth.values()),
        "round_truth": sum(len(values) for values in round_truth.values()),
        "candidate_generation_boundary_positive_covered_140": sum(
            intish(row.get("candidate_covered_140")) for row in candidate_summary if intish(row.get("truth")) > 0
        ),
        "current_app_baseline": baseline,
        "recommendation": recommendation,
        "selected_policy_id": selected.get("policy_id") if selected else "",
        "app_install_prepared": bool(export_summary),
        "export_summary": export_summary,
        "outputs": {
            "candidate_rows": "t0103_candidate_rows.csv",
            "oof_predictions": "t0103_oof_predictions.csv",
            "policy_sweep": "t0103_policy_sweep.csv",
            "report": "t0103_report.md",
        },
    }

    if not args.reuse_existing:
        write_csv(out_dir / "t0103_candidate_rows.csv", rows)
        write_csv(out_dir / "t0103_oof_predictions.csv", predictions)
        write_csv(out_dir / "t0103_policy_sweep.csv", policy_rows)
        write_csv(out_dir / "t0103_candidate_generation_summary.csv", candidate_summary)
    write_json(out_dir / "t0103_summary.json", summary)
    (out_dir / "t0103_report.md").write_text(
        render_report(summary, candidate_summary, baseline, top_combined, selected),
        encoding="utf-8",
    )

    print(f"Wrote {out_dir}")
    print(f"Boundary baseline recall={baseline['positive_recall_140']} negatives={baseline['negative_false_counts']}")
    print(f"Recommendation: {recommendation}")
    if selected:
        print(f"Selected: {selected['policy_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
