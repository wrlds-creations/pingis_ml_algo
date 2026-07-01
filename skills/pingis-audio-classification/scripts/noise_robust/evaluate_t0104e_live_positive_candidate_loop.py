#!/usr/bin/env python3
"""T0104E live-positive candidate loop.

Evaluation/training only. This joins the T0103 candidate rows with the latest
T0104 live labels and sweeps app-exportable ExtraTrees candidates. It does not
export an app model, install an APK, or change runtime behavior.
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
from sklearn.ensemble import ExtraTreesClassifier


ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = ROOT / "data/audio/models/evaluations/t0104e_live_positive_candidate_loop"
T0103_SCRIPT = ROOT / "skills/pingis-audio-classification/scripts/noise_robust/evaluate_t0103_boundary_candidate_loop.py"
T0103_DIR = ROOT / "data/audio/models/evaluations/t0103_boundary_label_candidate_phone_gate/candidate_loop_2026_07_01"
T0104_SUMMARY_CSV = ROOT / "data/audio/models/evaluations/t0104_bounce_audio_test_live_validation/t0104_session_summary.csv"
T0104D_LABELS_CSV = ROOT / "data/audio/models/evaluations/t0104d_t0104b_positive_label_ingest/t0104d_reviewed_positive_labels.csv"
T0104A_REVIEW_DIR = ROOT / "data/audio/models/evaluations/t0104a_slow_high_expected_count_review/review_pages"
T0104B_REVIEW_DIR = ROOT / "data/audio/models/evaluations/t0104b_positive_review_pages/review_pages"
T0104_RAW_DEBUG_DIR = ROOT / "data/audio/raw/t0104_bounce_audio_test_live_validation/bounce_audio_test_debug"
T0103_APP_MODEL = ROOT / "apps/collector/src/models/fable_extra_trees_candidate_t0103.json"
T0103_ARCHIVE_MODEL = T0103_DIR / "fable_extra_trees_candidate_t0103.json"
DEFAULT_APP_MODEL_OUT = ROOT / "apps/collector/src/models/fable_extra_trees_candidate_t0104e.json"
EXPORT_PARITY_SCRIPT = ROOT / "skills/pingis-audio-classification/scripts/noise_robust/export_t0075_fable_extra_trees_app_parity.py"

MATCH_TOLERANCE_MS = 140.0
THRESHOLDS = [0.20, 0.25, 0.30, 0.40, 0.50, 0.575, 0.65, 0.75]
DEDUPES_MS = [180.0, 220.0]
NOISE_VETO_THRESHOLDS: list[float | None] = [None, 0.95, 1.0]
SELECTED_MODEL_ID = "extra_leaf2_t0104e"
SELECTED_FEATURE_SET_ID = "base_t0075"
SELECTED_WEIGHT_STRATEGY_ID = "live_recall_safety"
SELECTED_THRESHOLD = 0.575
SELECTED_DEDUPE_MS = 180.0
SELECTED_FABLE_NOISE_VETO_THRESHOLD: float | None = None
POSITIVE_CLASS = 1
POSITIVE_LABEL = "racket_bounce"
NEGATIVE_LABEL = "not_racket_bounce"


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
            writer.writerow(row)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def md_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return lines


def policy_id(parts: list[Any]) -> str:
    return "_".join(str(part).replace(".", "p").replace(" ", "-").replace("/", "-") for part in parts)


def load_feature_names() -> list[str]:
    model_path = T0103_APP_MODEL if T0103_APP_MODEL.exists() else T0103_ARCHIVE_MODEL
    model = json.loads(model_path.read_text(encoding="utf-8"))
    return list(model["feature_names"])


def load_t0104_summary() -> dict[str, dict[str, Any]]:
    return {row["session_id"]: row for row in read_csv(T0104_SUMMARY_CSV)}


def load_t0104d_truth() -> tuple[dict[str, list[float]], dict[str, dict[str, Any]]]:
    truth: dict[str, list[float]] = defaultdict(list)
    meta: dict[str, dict[str, Any]] = {}
    for row in read_csv(T0104D_LABELS_CSV):
        if row.get("label") != POSITIVE_LABEL:
            continue
        sid = row["session_id"]
        truth[sid].append(ff(row["reviewed_time_ms"]))
        meta.setdefault(
            sid,
            {
                "session_id": sid,
                "scenario_id": row.get("scenario_id", ""),
                "scenario_title": row.get("scenario_title", ""),
                "polarity": "positive",
                "expected_contacts": intish(row.get("expected_count"), 30),
                "truth_source": "t0104d_reviewed_positive_labels",
            },
        )
    return {sid: sorted(times) for sid, times in truth.items()}, meta


def load_t0104a_slow_high_truth() -> tuple[str, list[float], dict[str, Any]]:
    sid = "bounce_audio_test_session_2026-07-01T13-37-11-083Z"
    review_path = T0104A_REVIEW_DIR / f"{sid}_review_labels.json"
    payload = json.loads(review_path.read_text(encoding="utf-8"))
    times = sorted(ff(marker.get("time_s")) * 1000.0 for marker in payload.get("manual_markers", []))
    if len(times) != 20:
        raise RuntimeError(f"Expected 20 confirmed slow/high labels for {sid}, got {len(times)}")
    meta = {
        "session_id": sid,
        "scenario_id": "slow_high_racket_bounce",
        "scenario_title": "Slow/high racket bounce",
        "polarity": "positive",
        "expected_contacts": 20,
        "truth_source": "t0104a_confirmed_slow_high_first_run",
    }
    return sid, times, meta


def load_live_truth_and_meta() -> tuple[dict[str, list[float]], dict[str, dict[str, Any]], set[str]]:
    truth, meta = load_t0104d_truth()
    slow_sid, slow_times, slow_meta = load_t0104a_slow_high_truth()
    truth[slow_sid] = slow_times
    meta[slow_sid] = slow_meta

    summary = load_t0104_summary()
    included_sessions = {
        sid
        for sid, row in summary.items()
        if str(row.get("include_in_metrics", "")).lower() == "true"
    }
    for sid, row in summary.items():
        if sid in meta or sid not in included_sessions:
            continue
        if row.get("polarity") == "negative":
            meta[sid] = {
                "session_id": sid,
                "scenario_id": row.get("scenario_id", ""),
                "scenario_title": row.get("scenario_title", ""),
                "polarity": "negative",
                "expected_contacts": 0,
                "truth_source": "t0104_negative_live_session",
            }
    return truth, meta, included_sessions


def trigger_csv_for_session(sid: str) -> Path | None:
    candidates = [
        T0104B_REVIEW_DIR / "trigger_csv" / f"{sid}_peak_fast_balanced_triggers.csv",
        T0104A_REVIEW_DIR / "trigger_csv" / f"{sid}_peak_fast_balanced_triggers.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def trigger_times_by_index(sid: str) -> dict[int, float]:
    path = trigger_csv_for_session(sid)
    if path is None:
        return {}
    out: dict[int, float] = {}
    for row in read_csv(path):
        out[intish(row.get("candidate_index"))] = ff(row.get("estimated_wav_ms") or row.get("onset_ms"))
    return out


def nearest_delta_ms(time_ms: float, truth_ms: list[float]) -> float:
    if not truth_ms:
        return 999999.0
    return min((time_ms - truth for truth in truth_ms), key=lambda delta: abs(delta))


def live_eval_group(row: dict[str, Any]) -> str:
    scenario = str(row.get("scenario_id") or "")
    mapping = {
        "normal_racket_bounce": "live_positive_normal",
        "slow_high_racket_bounce": "live_positive_slow_high",
        "fast_racket_bounce": "live_positive_fast",
        "racket_bounce_speaking_counting": "live_positive_speaking_counting",
        "racket_bounce_background_sound": "live_positive_background",
        "far_soft_racket_bounce_background": "live_positive_far_soft_background",
        "talking_only_no_bounce": "live_negative_talking",
        "racket_handling_no_bounce": "live_negative_handling",
    }
    return mapping.get(scenario, f"live_{scenario or 'unknown'}")


def row_from_live_candidate(
    *,
    sid: str,
    candidate: dict[str, Any],
    meta: dict[str, Any],
    feature_names: list[str],
    truth_ms: list[float],
    trigger_times: dict[int, float],
) -> dict[str, Any]:
    index = intish(candidate.get("index") or candidate.get("id"))
    continuous = candidate.get("_continuous_audio", {})
    sample_rate = ff(continuous.get("sample_rate_hz"), 22050.0)
    fallback_ms = ff(candidate.get("native_onset_pos")) / max(1.0, sample_rate) * 1000.0
    time_ms = trigger_times.get(index, fallback_ms)
    delta = nearest_delta_ms(time_ms, truth_ms)
    label = 1 if truth_ms and abs(delta) <= MATCH_TOLERANCE_MS else 0

    feature_vector = candidate.get("feature_vector") or {}
    fable_probs = candidate.get("fable_probabilities") or {}
    native_debug = candidate.get("native_debug") or {}
    row: dict[str, Any] = {
        "session_id": sid,
        "scenario_id": meta.get("scenario_id", ""),
        "scenario_title": meta.get("scenario_title", ""),
        "polarity": meta.get("polarity", ""),
        "dataset_role": "t0104_live_loso",
        "candidate_index": index,
        "time_ms": time_ms,
        "onset_sample": candidate.get("native_onset_pos", ""),
        "label": label,
        "label_source": "t0104_live_review_match" if label else ("t0104_live_extra_peak" if truth_ms else "t0104_live_negative"),
        "nearest_truth_delta_ms": delta if truth_ms else "",
        "expected_contacts": meta.get("expected_contacts", ""),
        "model_label": candidate.get("fable_label", ""),
        "model_confidence": ff(candidate.get("fable_confidence")),
        "prob_racket_bounce": ff(fable_probs.get("racket_bounce")),
        "prob_noise": ff(fable_probs.get("noise")),
        "prob_floor_bounce": ff(fable_probs.get("floor_bounce")),
        "prob_table_bounce": ff(fable_probs.get("table_bounce")),
        "model_is_racket": 1 if candidate.get("fable_label") == "racket_bounce" else 0,
        "model_is_noise": 1 if candidate.get("fable_label") == "noise" else 0,
        "model_is_floor": 1 if candidate.get("fable_label") == "floor_bounce" else 0,
        "model_is_table": 1 if candidate.get("fable_label") == "table_bounce" else 0,
        "domain": "live_t0104",
        "domain_session_id": f"live::{sid}",
        "eval_group": live_eval_group(meta),
        "session_group": live_eval_group(meta),
        "manual_review": meta.get("truth_source", ""),
        "is_rejected_unsafe": 0,
        "source_model_probability": ff(candidate.get("classifier_probability")),
        "source_decision": candidate.get("decision", ""),
        "source_reject_reason": candidate.get("reject_reason", ""),
        "native_gate_id": native_debug.get("gate_id", ""),
    }
    for name in feature_names:
        row[name] = feature_vector.get(name, row.get(name, 0.0))
    return row


def build_live_rows(feature_names: list[str]) -> tuple[list[dict[str, Any]], dict[str, list[float]], list[dict[str, Any]]]:
    truth, meta_by_session, included_sessions = load_live_truth_and_meta()
    rows: list[dict[str, Any]] = []
    session_summary: list[dict[str, Any]] = []
    for sid in sorted(included_sessions):
        if sid not in meta_by_session:
            continue
        json_path = T0104_RAW_DEBUG_DIR / f"{sid}.json"
        if not json_path.exists():
            raise FileNotFoundError(json_path)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        continuous = payload.get("continuous_audio") or {}
        trigger_times = trigger_times_by_index(sid)
        truth_ms = truth.get(sid, [])
        session_rows: list[dict[str, Any]] = []
        for candidate in payload.get("candidates", []):
            candidate["_continuous_audio"] = continuous
            session_rows.append(
                row_from_live_candidate(
                    sid=sid,
                    candidate=candidate,
                    meta=meta_by_session[sid],
                    feature_names=feature_names,
                    truth_ms=truth_ms,
                    trigger_times=trigger_times,
                )
            )
        rows.extend(session_rows)
        candidate_ms = [ff(row.get("time_ms")) for row in session_rows]
        match = match_predictions(candidate_ms, truth_ms) if truth_ms else {"tp": "", "fp": "", "missed": ""}
        session_summary.append(
            {
                "session_id": sid,
                "scenario_id": meta_by_session[sid].get("scenario_id", ""),
                "eval_group": live_eval_group(meta_by_session[sid]),
                "truth": len(truth_ms),
                "candidate_count": len(session_rows),
                "candidate_positive_rows": sum(intish(row.get("label")) for row in session_rows),
                "candidate_covered_140": match["tp"],
                "candidate_missed_140": match["missed"],
                "app_count_at_stop": (payload.get("review") or {}).get("app_count_at_stop", ""),
                "truth_source": meta_by_session[sid].get("truth_source", ""),
            }
        )
    return rows, truth, session_summary


def load_base_rows() -> list[dict[str, Any]]:
    return read_csv(T0103_DIR / "t0103_candidate_rows.csv")


def make_model_specs(t0103: Any) -> list[Any]:
    return [
        t0103.ModelSpec(
            "extra_leaf2_t0104e",
            "ExtraTrees leaf2 T0104E",
            ExtraTreesClassifier(
                n_estimators=160,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=10402,
                n_jobs=-1,
            ),
        ),
        t0103.ModelSpec(
            "extra_leaf4_t0104e",
            "ExtraTrees leaf4 T0104E",
            ExtraTreesClassifier(
                n_estimators=160,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=10404,
                n_jobs=-1,
            ),
        ),
    ]


def make_weight_strategies(t0103: Any, t0099: Any) -> list[Any]:
    def base(row: dict[str, Any]) -> float:
        return float(t0099.base_weight(row))

    def live_recall_safety(row: dict[str, Any]) -> float:
        weight = base(row)
        domain = str(row.get("domain") or "")
        group = str(row.get("eval_group") or "")
        label = intish(row.get("label"))
        if domain == "live_t0104" and label == 1:
            weight *= 3.0
        if group == "live_positive_far_soft_background" and label == 1:
            weight *= 2.0
        if domain == "live_t0104" and label == 0:
            weight *= 1.8
        if domain == "boundary_t0103" and label == 1:
            weight *= 2.0
        if domain == "boundary_t0103" and label == 0:
            weight *= 2.2
        if domain == "round_a" and label == 0:
            weight *= 3.8
        if intish(row.get("is_rejected_unsafe")):
            weight *= 8.0
        return weight

    def strict_negative_guard(row: dict[str, Any]) -> float:
        weight = base(row)
        domain = str(row.get("domain") or "")
        label = intish(row.get("label"))
        if domain == "live_t0104" and label == 1:
            weight *= 2.4
        if domain in {"live_t0104", "boundary_t0103", "round_a"} and label == 0:
            weight *= 5.0
        if domain == "boundary_t0103" and label == 1:
            weight *= 1.8
        if intish(row.get("is_rejected_unsafe")):
            weight *= 10.0
        return weight

    def live_far_soft_push(row: dict[str, Any]) -> float:
        weight = live_recall_safety(row)
        if str(row.get("eval_group") or "") == "live_positive_far_soft_background" and intish(row.get("label")) == 1:
            weight *= 1.8
        return weight

    return [
        t0103.WeightStrategy("live_recall_safety", "Live positives plus safety negatives", live_recall_safety),
        t0103.WeightStrategy("strict_negative_guard", "Strict live/boundary/Round A negatives", strict_negative_guard),
        t0103.WeightStrategy("live_far_soft_push", "Live far-soft positives pushed with safety guards", live_far_soft_push),
    ]


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
    return {"tp": len(used_truth), "fp": len(pred_ms) - len(used_pred), "missed": len(truth_ms) - len(used_truth)}


def truth_from_candidate_rows(rows: list[dict[str, Any]], *, domain: str) -> dict[str, list[float]]:
    truth: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("domain") == domain and intish(row.get("label")) == 1:
            truth[str(row.get("session_id"))].append(ff(row.get("time_ms")))
    return {sid: sorted(times) for sid, times in truth.items()}


def accepted_rows(
    rows: list[dict[str, Any]],
    threshold: float,
    dedupe_ms: float,
    *,
    prob_key: str,
    fable_noise_veto_threshold: float | None,
) -> list[dict[str, Any]]:
    eligible = []
    for row in sorted(rows, key=lambda item: ff(item.get("time_ms"))):
        if ff(row.get(prob_key)) < threshold:
            continue
        if fable_noise_veto_threshold is not None and ff(row.get("prob_noise")) >= fable_noise_veto_threshold:
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
        "group_counts": dict(group_counts),
    }


def group_metric(score: dict[str, Any], group: str, field: str) -> Any:
    counts = score.get("group_counts", {})
    if field == "recall":
        truth = int(counts.get((group, "truth"), 0))
        tp = int(counts.get((group, "tp"), 0))
        return round(tp / max(1, truth), 4) if truth else ""
    if field == "tp":
        return int(counts.get((group, "tp"), 0))
    if field == "truth":
        return int(counts.get((group, "truth"), 0))
    if field == "false":
        return int(counts.get((group, "false"), 0))
    if field == "missed":
        return int(counts.get((group, "missed"), 0))
    return ""


def add_live_group_summaries(row: dict[str, Any], live_score: dict[str, Any]) -> None:
    for group in [
        "live_positive_normal",
        "live_positive_slow_high",
        "live_positive_fast",
        "live_positive_speaking_counting",
        "live_positive_background",
        "live_positive_far_soft_background",
        "live_negative_talking",
        "live_negative_handling",
    ]:
        if group.startswith("live_positive"):
            row[f"{group}_recall"] = group_metric(live_score, group, "recall")
            row[f"{group}_tp"] = group_metric(live_score, group, "tp")
            row[f"{group}_truth"] = group_metric(live_score, group, "truth")
        else:
            row[f"{group}_false"] = group_metric(live_score, group, "false")


def sweep_predictions(
    predictions: list[dict[str, Any]],
    live_truth: dict[str, list[float]],
    boundary_truth: dict[str, list[float]],
    round_truth: dict[str, list[float]],
    noisy_truth: dict[str, list[float]],
) -> list[dict[str, Any]]:
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
                    live_score = score_policy(
                        rows,
                        live_truth,
                        threshold,
                        dedupe_ms,
                        prob_key="oof_prob",
                        fable_noise_veto_threshold=noise_veto,
                        domain="live_t0104",
                    )
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
                    noisy_score = score_policy(
                        rows,
                        noisy_truth,
                        threshold,
                        dedupe_ms,
                        prob_key="oof_prob",
                        fable_noise_veto_threshold=noise_veto,
                        domain="noisy_target",
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
                        "live_positive_tp_140": live_score["positive_tp_140"],
                        "live_positive_missed_140": live_score["positive_missed_140"],
                        "live_positive_recall_140": live_score["positive_recall_140"],
                        "live_positive_precision_140": live_score["positive_precision_140"],
                        "live_positive_count_error": live_score["positive_count_error"],
                        "live_negative_false_counts": live_score["negative_false_counts"],
                        "boundary_positive_recall_140": boundary_score["positive_recall_140"],
                        "boundary_negative_false_counts": boundary_score["negative_false_counts"],
                        "round_positive_recall_140": round_score["positive_recall_140"],
                        "round_negative_false_counts": round_score["negative_false_counts"],
                        "noisy_positive_recall_140": noisy_score["positive_recall_140"],
                        "noisy_negative_false_counts": noisy_score["negative_false_counts"],
                    }
                    add_live_group_summaries(policy, live_score)
                    policies.append(policy)
    return policies


def score_named_policy(
    name: str,
    rows: list[dict[str, Any]],
    live_truth: dict[str, list[float]],
    boundary_truth: dict[str, list[float]],
    round_truth: dict[str, list[float]],
    *,
    threshold: float,
    dedupe_ms: float,
    prob_key: str,
    fable_noise_veto_threshold: float | None,
) -> dict[str, Any]:
    live_score = score_policy(rows, live_truth, threshold, dedupe_ms, prob_key=prob_key, fable_noise_veto_threshold=fable_noise_veto_threshold, domain="live_t0104")
    boundary_score = score_policy(rows, boundary_truth, threshold, dedupe_ms, prob_key=prob_key, fable_noise_veto_threshold=fable_noise_veto_threshold, domain="boundary_t0103")
    round_score = score_policy(rows, round_truth, threshold, dedupe_ms, prob_key=prob_key, fable_noise_veto_threshold=fable_noise_veto_threshold, domain="round_a")
    out = {
        "policy_id": name,
        "threshold": threshold,
        "dedupe_ms": int(dedupe_ms),
        "fable_noise_veto_threshold": "" if fable_noise_veto_threshold is None else fable_noise_veto_threshold,
        "live_positive_tp_140": live_score["positive_tp_140"],
        "live_positive_missed_140": live_score["positive_missed_140"],
        "live_positive_recall_140": live_score["positive_recall_140"],
        "live_negative_false_counts": live_score["negative_false_counts"],
        "boundary_positive_recall_140": boundary_score["positive_recall_140"],
        "boundary_negative_false_counts": boundary_score["negative_false_counts"],
        "round_positive_recall_140": round_score["positive_recall_140"],
        "round_negative_false_counts": round_score["negative_false_counts"],
    }
    add_live_group_summaries(out, live_score)
    return out


def baseline_rows_with_t0103_app_probability(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    export = load_module("t0104e_app_parity", EXPORT_PARITY_SCRIPT)
    model_path = T0103_APP_MODEL if T0103_APP_MODEL.exists() else T0103_ARCHIVE_MODEL
    model = json.loads(model_path.read_text(encoding="utf-8"))
    probs = export.app_style_probabilities(model, rows)
    out = []
    for row, prob in zip(rows, probs):
        item = dict(row)
        item["t0103_app_prob"] = float(prob)
        out.append(item)
    return out


def select_candidate(policy_rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any] | None]:
    exportable = [row for row in policy_rows if intish(row.get("app_exportable")) == 1]
    practical = [
        row
        for row in exportable
        if ff(row.get("live_positive_recall_140")) >= 0.88
        and intish(row.get("live_negative_false_counts")) <= 2
        and ff(row.get("boundary_positive_recall_140")) >= 0.78
        and intish(row.get("boundary_negative_false_counts")) <= 8
        and ff(row.get("round_positive_recall_140")) >= 0.95
        and intish(row.get("round_negative_false_counts")) <= 10
    ]
    if practical:
        ranked = sorted(
            practical,
            key=lambda row: (
                -ff(row.get("live_positive_recall_140")),
                intish(row.get("live_negative_false_counts")),
                intish(row.get("boundary_negative_false_counts")),
                intish(row.get("round_negative_false_counts")),
                -ff(row.get("live_positive_far_soft_background_recall"), -1.0),
            ),
        )
        return "candidate_worth_guarded_phone_test", ranked[0]
    near = [
        row
        for row in exportable
        if ff(row.get("live_positive_recall_140")) >= 0.84
        and intish(row.get("live_negative_false_counts")) <= 4
        and intish(row.get("boundary_negative_false_counts")) <= 12
        and intish(row.get("round_negative_false_counts")) <= 16
    ]
    if near:
        ranked = sorted(
            near,
            key=lambda row: (
                -ff(row.get("live_positive_recall_140")),
                intish(row.get("live_negative_false_counts")),
                intish(row.get("boundary_negative_false_counts")),
                intish(row.get("round_negative_false_counts")),
            ),
        )
        return "near_candidate_needs_caution_before_phone_test", ranked[0]
    return "no_candidate_worth_phone_test_yet", None


def class_label(value: int) -> str:
    return POSITIVE_LABEL if int(value) == POSITIVE_CLASS else NEGATIVE_LABEL


def export_candidate_model(
    t0103: Any,
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
            "model_version": "fable_extra_trees_candidate_t0104e",
            "source_ticket": "T0106-bounce-audio-test-model-switcher",
            "training_source_ticket": "T0104E-live-positive-candidate-loop",
            "selection_source": selected_policy.get("policy_id") if selected_policy else "",
            "model_type": "extra_trees_binary_peak_candidate",
            "candidate_gate": "peak_fast_balanced",
            "feature_version": "t0104e_peak_candidate_features_plus_fable83",
            "selected_threshold": SELECTED_THRESHOLD,
            "smart_dedupe_ms": SELECTED_DEDUPE_MS,
            "fable_noise_veto_threshold": SELECTED_FABLE_NOISE_VETO_THRESHOLD,
            "positive_class": POSITIVE_CLASS,
            "positive_label": POSITIVE_LABEL,
            "classes": classes,
            "tree_count": len(estimator.estimators_),
            "total_nodes": total_nodes,
            "training_rows": len(training_rows),
            "training_positive_candidates": positive_rows,
            "training_negative_candidates": len(training_rows) - positive_rows,
            "normal_fable_model_unchanged": True,
            "runtime_status": "diagnostic_bounce_audio_test_switch_only_not_production",
            "offline_caution": (
                "T0104E was a near-miss candidate, not a production promotion. "
                "Use only for guarded phone comparison against T0103."
            ),
        },
        "labels": labels,
        "feature_names": features,
        "scaler_mean": [0.0 for _ in features],
        "scaler_std": [1.0 for _ in features],
        "trees": [t0103.export_tree_full_precision(tree) for tree in estimator.estimators_],
    }


def train_and_export_selected_app_model(
    t0103: Any,
    rows: list[dict[str, Any]],
    feature_sets: list[Any],
    specs: list[Any],
    strategies: list[Any],
    selected_policy: dict[str, Any] | None,
    app_model_out: Path,
    out_dir: Path,
) -> dict[str, Any]:
    features = next(item.features for item in feature_sets if item.feature_set_id == SELECTED_FEATURE_SET_ID)
    strategy = next(item for item in strategies if item.strategy_id == SELECTED_WEIGHT_STRATEGY_ID)
    spec = next(item for item in specs if item.model_id == SELECTED_MODEL_ID)
    estimator = t0103.fit_estimator(spec.estimator, rows, features, strategy)
    model_json = export_candidate_model(t0103, estimator, features, rows, selected_policy)
    app_model_out.parent.mkdir(parents=True, exist_ok=True)
    app_model_out.write_text(json.dumps(model_json, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    archive_path = out_dir / "fable_extra_trees_candidate_t0104e.json"
    archive_path.write_text(json.dumps(model_json, indent=2, ensure_ascii=False), encoding="utf-8")

    export = load_module("t0104e_export_parity", EXPORT_PARITY_SCRIPT)
    app_probs = export.app_style_probabilities(model_json, rows)
    fit_probs = t0103.predict_probability(estimator, rows, features)
    max_probability_diff = float(np.max(np.abs(app_probs - fit_probs))) if len(rows) else 0.0

    return {
        "app_model_out": str(app_model_out),
        "archive_path": str(archive_path),
        "training_rows": len(rows),
        "feature_count": len(features),
        "tree_count": len(model_json["trees"]),
        "total_nodes": model_json["metadata"]["total_nodes"],
        "max_probability_diff": max_probability_diff,
    }


def render_report(
    summary: dict[str, Any],
    live_session_summary: list[dict[str, Any]],
    baselines: list[dict[str, Any]],
    top_rows: list[dict[str, Any]],
    selected: dict[str, Any] | None,
) -> str:
    lines = [
        "# T0104E Live Positive Candidate Loop",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        "## Decision",
        "",
        f"- Recommendation: `{summary['recommendation']}`",
        f"- App/model export prepared: `{summary['app_model_export_prepared']}`",
        "",
        "## Live Candidate Coverage",
        "",
    ]
    lines.extend(
        md_table(
            live_session_summary,
            [
                "session_id",
                "eval_group",
                "truth",
                "candidate_count",
                "candidate_covered_140",
                "candidate_missed_140",
                "app_count_at_stop",
                "truth_source",
            ],
        )
    )
    lines.extend(["", "## Baselines", ""])
    lines.extend(
        [
            "The current-app baseline rows are useful for fresh T0104 held-out live sessions. "
            "Their boundary/Round A columns are final-fit/in-sample context; use T0104C/T0103 OOF policy sweeps for promotion safety.",
            "",
        ]
    )
    common_fields = [
        "policy_id",
        "threshold",
        "fable_noise_veto_threshold",
        "live_positive_tp_140",
        "live_positive_recall_140",
        "live_negative_false_counts",
        "boundary_positive_recall_140",
        "boundary_negative_false_counts",
        "round_positive_recall_140",
        "round_negative_false_counts",
        "live_positive_far_soft_background_recall",
    ]
    lines.extend(md_table(baselines, common_fields))
    lines.extend(["", "## Top Candidate Policies", ""])
    top_fields = [
        "policy_id",
        "app_exportable",
        "threshold",
        "fable_noise_veto_threshold",
        "live_positive_tp_140",
        "live_positive_recall_140",
        "live_negative_false_counts",
        "boundary_positive_recall_140",
        "boundary_negative_false_counts",
        "round_positive_recall_140",
        "round_negative_false_counts",
        "live_positive_far_soft_background_recall",
    ]
    lines.extend(md_table(top_rows, top_fields))
    lines.extend(["", "## Selected", ""])
    if selected:
        lines.extend(md_table([selected], top_fields))
    else:
        lines.append("_None._")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This is primarily offline evaluation. Any exported app model is diagnostic-only for the separate `Bounce audio test` selector.",
            "- The excluded slow/high run `bounce_audio_test_session_2026-07-01T13-38-19-066Z` is not used because Love marked the true count unclear.",
            "- Phone-test worthiness requires live-positive recall improvement while preserving fresh negative, T0103 boundary-negative, and Round A hard-negative safety.",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--app-model-out", type=Path, default=DEFAULT_APP_MODEL_OUT)
    parser.add_argument(
        "--export-app-model",
        action="store_true",
        help="Export the selected near-miss T0104E model for guarded Bounce audio test comparison only.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    t0103 = load_module("t0103_candidate_loop_for_t0104e", T0103_SCRIPT)
    t0099 = t0103.load_module("t0099_helpers_t0104e", t0103.T0099_SCRIPT)
    feature_names = load_feature_names()

    base_rows = load_base_rows()
    live_rows, live_truth, live_session_summary = build_live_rows(feature_names)
    rows = base_rows + live_rows

    boundary_truth = truth_from_candidate_rows(base_rows, domain="boundary_t0103")
    round_truth = truth_from_candidate_rows(base_rows, domain="round_a")
    noisy_truth = truth_from_candidate_rows(base_rows, domain="noisy_target")

    feature_sets = [t0103.FeatureSet("base_t0075", "App 100-feature vector", feature_names, True)]
    specs = make_model_specs(t0103)
    strategies = make_weight_strategies(t0103, t0099)
    predictions = t0103.make_oof_predictions(rows, specs, feature_sets, strategies)
    policy_rows = sweep_predictions(predictions, live_truth, boundary_truth, round_truth, noisy_truth)
    recommendation, selected = select_candidate(policy_rows)
    export_summary: dict[str, Any] | None = None

    if args.export_app_model:
        if selected is None:
            raise RuntimeError("Refusing app export because no T0104E candidate was selected")
        app_model_out = args.app_model_out if args.app_model_out.is_absolute() else ROOT / args.app_model_out
        export_summary = train_and_export_selected_app_model(
            t0103,
            rows,
            feature_sets,
            specs,
            strategies,
            selected,
            app_model_out,
            out_dir,
        )

    app_rows = baseline_rows_with_t0103_app_probability(rows)
    baselines = [
        score_named_policy(
            "current_t0103_app_thr0p575_no_veto_dedupe180",
            app_rows,
            live_truth,
            boundary_truth,
            round_truth,
            threshold=0.575,
            dedupe_ms=180.0,
            prob_key="t0103_app_prob",
            fable_noise_veto_threshold=None,
        ),
        score_named_policy(
            "diagnostic_t0103_app_thr0p30_no_veto_dedupe180",
            app_rows,
            live_truth,
            boundary_truth,
            round_truth,
            threshold=0.30,
            dedupe_ms=180.0,
            prob_key="t0103_app_prob",
            fable_noise_veto_threshold=None,
        ),
        score_named_policy(
            "diagnostic_t0103_app_thr0p20_no_veto_dedupe180",
            app_rows,
            live_truth,
            boundary_truth,
            round_truth,
            threshold=0.20,
            dedupe_ms=180.0,
            prob_key="t0103_app_prob",
            fable_noise_veto_threshold=None,
        ),
    ]

    top_safe = sorted(
        policy_rows,
        key=lambda row: (
            intish(row.get("live_negative_false_counts")),
            intish(row.get("boundary_negative_false_counts")),
            intish(row.get("round_negative_false_counts")),
            -ff(row.get("live_positive_recall_140")),
            -ff(row.get("live_positive_far_soft_background_recall"), -1.0),
        ),
    )[:20]
    top_recall = sorted(
        policy_rows,
        key=lambda row: (
            -ff(row.get("live_positive_recall_140")),
            intish(row.get("live_negative_false_counts")),
            intish(row.get("boundary_negative_false_counts")),
            intish(row.get("round_negative_false_counts")),
        ),
    )[:20]
    top_rows = top_safe + [row for row in top_recall if row not in top_safe]

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows),
        "base_rows": len(base_rows),
        "live_rows": len(live_rows),
        "live_truth": sum(len(values) for values in live_truth.values()),
        "boundary_truth": sum(len(values) for values in boundary_truth.values()),
        "round_truth": sum(len(values) for values in round_truth.values()),
        "feature_count": len(feature_names),
        "models": [spec.model_id for spec in specs],
        "weight_strategies": [strategy.strategy_id for strategy in strategies],
        "thresholds": THRESHOLDS,
        "dedupes_ms": DEDUPES_MS,
        "noise_veto_thresholds": ["" if item is None else item for item in NOISE_VETO_THRESHOLDS],
        "baselines": baselines,
        "recommendation": recommendation,
        "selected_policy_id": selected.get("policy_id") if selected else "",
        "selected_policy": selected,
        "app_model_export_prepared": bool(export_summary),
        "export_summary": export_summary,
        "outputs": {
            "candidate_rows": "t0104e_candidate_rows.csv",
            "oof_predictions": "t0104e_oof_predictions.csv",
            "policy_sweep": "t0104e_policy_sweep.csv",
            "report": "t0104e_report.md",
        },
    }

    write_csv(out_dir / "t0104e_candidate_rows.csv", rows)
    write_csv(out_dir / "t0104e_oof_predictions.csv", predictions)
    write_csv(out_dir / "t0104e_policy_sweep.csv", policy_rows)
    write_csv(out_dir / "t0104e_live_session_summary.csv", live_session_summary)
    write_csv(out_dir / "t0104e_baselines.csv", baselines)
    write_json(out_dir / "t0104e_summary.json", summary)
    (out_dir / "t0104e_report.md").write_text(
        render_report(summary, live_session_summary, baselines, top_rows, selected),
        encoding="utf-8",
    )

    print(f"Wrote {out_dir}")
    print(f"Live truth: {summary['live_truth']} labels; live rows: {len(live_rows)}")
    print(f"Recommendation: {recommendation}")
    if selected:
        print(f"Selected: {selected['policy_id']}")
        print(
            "Selected live recall="
            f"{selected['live_positive_recall_140']} negatives={selected['live_negative_false_counts']} "
            f"boundary_false={selected['boundary_negative_false_counts']} round_false={selected['round_negative_false_counts']}"
        )
    if export_summary:
        print(f"Exported app model: {export_summary['app_model_out']}")
        print(f"Export parity max diff: {export_summary['max_probability_diff']:.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
