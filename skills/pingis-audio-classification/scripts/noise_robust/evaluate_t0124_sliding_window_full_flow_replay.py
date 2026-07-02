#!/usr/bin/env python3
"""T0124 full-flow replay for recovered PCM candidates.

Evaluation only. This answers whether replacing the current peak-candidate
timestamps with softer peak or sliding-window PCM timestamps helps after the
existing app-style T0104E stack is applied:

WAV -> candidate timestamps -> live clip extraction -> Fable features/model
-> T0104E ExtraTrees -> threshold/noise veto/smart dedupe -> scored count.

It does not train, export, install, or change app/runtime behavior.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import nr_config  # noqa: E402
import nr_features  # noqa: E402
from evaluate_fable_audio_reliability_t0044 import FableAppModel  # noqa: E402
from evaluate_t0123_sliding_window_pcm_candidate_audit import (  # noqa: E402
    EvalSession,
    detect_for_config,
    get_configs,
    load_sessions,
    read_wav,
)

OUT_DIR = ROOT / "data/audio/models/evaluations/t0124_sliding_window_full_flow_replay"
FABLE_MODEL_JSON = ROOT / "apps/collector/src/models/fable_audio_model.json"
T0104E_MODEL_JSON = ROOT / "apps/collector/src/models/fable_extra_trees_candidate_t0104e.json"

MATCH_TOLERANCE_MS = 140.0
FAR_GAP_MS = 99999.0
FRAME_SIZE = 220

CONFIG_IDS = [
    "current_peak_abs008",
    "soft_peak_abs003",
    "soft_peak_hp_abs0015",
    "sw_raw_abs030_r2_z4",
    "sw_raw_abs020_r3_z6",
    "sw_hp_abs006_r4_z10",
]

POLICIES = [
    {
        "policy_id": "strict_t0104e_p0575_no_veto",
        "threshold": 0.575,
        "fable_noise_veto_threshold": 1.0,
        "smart_dedupe_ms": 180.0,
    },
    {
        "policy_id": "love_t0104e_p025_veto098",
        "threshold": 0.25,
        "fable_noise_veto_threshold": 0.98,
        "smart_dedupe_ms": 180.0,
    },
    {
        "policy_id": "diagnostic_t0104e_p020_no_veto",
        "threshold": 0.20,
        "fable_noise_veto_threshold": 1.0,
        "smart_dedupe_ms": 180.0,
    },
    {
        "policy_id": "diagnostic_t0104e_p030_no_veto",
        "threshold": 0.30,
        "fable_noise_veto_threshold": 1.0,
        "smart_dedupe_ms": 180.0,
    },
]


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


def pct(part: int | float, total: int | float) -> str:
    return "0.0%" if not total else f"{float(part) / float(total) * 100.0:.1f}%"


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
    if not fields:
        fields = ["empty"]
        rows = [{"empty": ""}]
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


class RfJsonModel:
    """Python mirror of apps/collector/src/rfRuntime.ts for ExtraTrees JSON."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.metadata = dict(payload.get("metadata") or {})
        self.labels = [str(v) for v in payload["labels"]]
        self.feature_names = [str(v) for v in payload["feature_names"]]
        self.scaler_mean = np.asarray(payload["scaler_mean"], dtype=np.float64)
        self.scaler_std = np.asarray(payload["scaler_std"], dtype=np.float64)
        self.trees = payload["trees"]
        if len(self.feature_names) != len(self.scaler_mean):
            raise ValueError("RF model feature/scaler length mismatch")
        if len(self.feature_names) != len(self.scaler_std):
            raise ValueError("RF model feature/std length mismatch")

    @classmethod
    def load(cls, path: Path) -> "RfJsonModel":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def is_leaf(self, node: list[float]) -> bool:
        n_classes = len(self.labels)
        if len(node) != n_classes:
            return len(node) != 4
        total = 0.0
        for value in node:
            if value < 0.0 or value > 1.0:
                return False
            total += value
        return abs(total - 1.0) < 0.01

    def traverse_tree(self, tree: list[list[float]], scaled: np.ndarray) -> list[float]:
        idx = 0
        while not self.is_leaf(tree[idx]):
            node = tree[idx]
            feature_idx = int(node[0])
            threshold = float(node[1])
            idx = int(node[2]) if scaled[feature_idx] <= threshold else int(node[3])
        return tree[idx]

    def predict_features(self, features: dict[str, float]) -> dict[str, Any]:
        raw = np.zeros(len(self.feature_names), dtype=np.float64)
        for idx, name in enumerate(self.feature_names):
            value = ff(features.get(name), 0.0)
            raw[idx] = value
        std = self.scaler_std.copy()
        std[std == 0.0] = 1.0
        scaled = (raw - self.scaler_mean) / std

        n_classes = len(self.labels)
        prob_sum = np.zeros(n_classes, dtype=np.float64)
        for tree in self.trees:
            proba = self.traverse_tree(tree, scaled)
            for class_idx in range(n_classes):
                prob_sum[class_idx] += float(proba[class_idx])
        probs = prob_sum / max(1, len(self.trees))
        best_idx = int(np.argmax(probs))
        return {
            "label": self.labels[best_idx],
            "confidence": float(probs[best_idx]),
            "probabilities": {label: float(probs[idx]) for idx, label in enumerate(self.labels)},
        }


def frame_rms_at(y: np.ndarray, sample: int) -> float:
    if len(y) == 0:
        return 0.0
    start = max(0, sample)
    end = min(len(y), start + FRAME_SIZE)
    if end <= start:
        return 0.0
    frame = y[start:end].astype(np.float64)
    return float(np.sqrt(np.mean(frame * frame)))


def fable_model_flags(prediction: dict[str, Any]) -> dict[str, int]:
    label = str(prediction.get("label", ""))
    return {
        "model_is_racket": 1 if label == "racket_bounce" else 0,
        "model_is_noise": 1 if label == "noise" else 0,
        "model_is_floor": 1 if label == "floor_bounce" else 0,
        "model_is_table": 1 if label == "table_bounce" else 0,
    }


def build_feature_vector(
    row: dict[str, Any],
    fable_features: dict[str, float],
    fable_prediction: dict[str, Any],
    model: RfJsonModel,
) -> dict[str, float]:
    fable_probs = fable_prediction.get("probabilities") or {}
    vector: dict[str, float] = {
        "frame_rms": ff(row.get("frame_rms")),
        "bg_rms": ff(row.get("bg_rms")),
        "peak_value": ff(row.get("peak_value")),
        "peak_ratio": ff(row.get("peak_ratio")),
        "peak_z": ff(row.get("peak_z")),
        "prev_gap_ms": ff(row.get("prev_gap_ms"), FAR_GAP_MS),
        "next_gap_ms": ff(row.get("next_gap_ms"), FAR_GAP_MS),
        "neighbor_count_500ms": ff(row.get("neighbor_count_500ms")),
        "prob_racket_bounce": ff(fable_probs.get("racket_bounce")),
        "prob_noise": ff(fable_probs.get("noise")),
        "prob_floor_bounce": ff(fable_probs.get("floor_bounce")),
        "prob_table_bounce": ff(fable_probs.get("table_bounce")),
        "model_confidence": ff(fable_prediction.get("confidence")),
        **fable_model_flags(fable_prediction),
    }
    for name, value in fable_features.items():
        vector[f"feat_{name}"] = ff(value)
    return {name: ff(vector.get(name)) for name in model.feature_names}


def normalize_candidate(
    event: dict[str, Any],
    y: np.ndarray,
    sr: int,
    candidate_index: int,
) -> dict[str, Any]:
    time_ms = ff(event.get("time_ms"), ff(event.get("time_s")) * 1000.0)
    sample = intish(event.get("sample"), int(round(time_ms / 1000.0 * sr)))
    sample = max(0, min(len(y) - 1, sample)) if len(y) else 0
    peak_value = ff(event.get("peak_value"))
    bg = ff(event.get("local_bg"), ff(event.get("bg_rms"), 1e-8))
    ratio = ff(event.get("ratio"), peak_value / max(bg, 1e-8))
    z = ff(event.get("z"), (peak_value - bg) / max(ff(event.get("local_mad"), 1e-8), 1e-8))
    return {
        "candidate_index": candidate_index,
        "time_ms": time_ms,
        "time_s": time_ms / 1000.0,
        "sample": sample,
        "frame_rms": frame_rms_at(y, sample),
        "bg_rms": bg,
        "peak_value": peak_value,
        "peak_ratio": ratio,
        "peak_z": z,
        "peak_local_mad": ff(event.get("local_mad")),
        "score": ff(event.get("score"), z + math.log(max(ratio, 1.0))),
    }


def add_timing_features(candidates: list[dict[str, Any]]) -> None:
    candidates.sort(key=lambda row: ff(row.get("time_ms")))
    for idx, row in enumerate(candidates):
        time_ms = ff(row.get("time_ms"))
        row["prev_gap_ms"] = time_ms - ff(candidates[idx - 1].get("time_ms")) if idx > 0 else FAR_GAP_MS
        row["next_gap_ms"] = (
            ff(candidates[idx + 1].get("time_ms")) - time_ms if idx + 1 < len(candidates) else FAR_GAP_MS
        )
        row["neighbor_count_500ms"] = sum(
            1
            for other in candidates
            if 0.0 < abs(ff(other.get("time_ms")) - time_ms) <= 500.0
        )


def classify_candidates(
    session: EvalSession,
    config_id: str,
    family: str,
    y: np.ndarray,
    sr: int,
    events: list[dict[str, Any]],
    fable_model: FableAppModel,
    t0104e_model: RfJsonModel,
    feature_cache: dict[tuple[Path, int], tuple[dict[str, float], dict[str, Any]]],
) -> list[dict[str, Any]]:
    candidates = [normalize_candidate(event, y, sr, idx) for idx, event in enumerate(events, start=1)]
    add_timing_features(candidates)

    rows: list[dict[str, Any]] = []
    for row in candidates:
        sample = intish(row.get("sample"))
        cache_key = (session.wav_path, sample)
        if cache_key not in feature_cache:
            clip = nr_features.extract_live_clip(y, sample)
            features = nr_features.extract_all_features(clip, nr_config.TARGET_SR)
            prediction = fable_model.predict_features(features)
            feature_cache[cache_key] = (features, prediction)
        fable_features, fable_prediction = feature_cache[cache_key]
        feature_vector = build_feature_vector(row, fable_features, fable_prediction, t0104e_model)
        classifier_prediction = t0104e_model.predict_features(feature_vector)
        classifier_probs = classifier_prediction.get("probabilities") or {}
        fable_probs = fable_prediction.get("probabilities") or {}
        probability = ff(classifier_probs.get("racket_bounce"))

        rows.append(
            {
                **row,
                "config_id": config_id,
                "family": family,
                "session_id": session.session_id,
                "scenario_id": session.scenario_id,
                "scenario_title": session.scenario_title,
                "polarity": session.polarity,
                "expected_count": session.expected_count,
                "truth_count": len(session.label_times_ms),
                "fable_label": fable_prediction.get("label", ""),
                "fable_confidence": ff(fable_prediction.get("confidence")),
                "fable_prob_racket_bounce": ff(fable_probs.get("racket_bounce")),
                "fable_prob_noise": ff(fable_probs.get("noise")),
                "fable_prob_floor_bounce": ff(fable_probs.get("floor_bounce")),
                "fable_prob_table_bounce": ff(fable_probs.get("table_bounce")),
                "classifier_label": classifier_prediction.get("label", ""),
                "classifier_confidence": ff(classifier_prediction.get("confidence")),
                "classifier_probability": probability,
            }
        )
    return rows


def apply_policy(
    candidate_rows: list[dict[str, Any]],
    policy: dict[str, Any],
) -> list[dict[str, Any]]:
    threshold = ff(policy.get("threshold"))
    veto_threshold = ff(policy.get("fable_noise_veto_threshold"), 1.0)
    smart_dedupe_ms = ff(policy.get("smart_dedupe_ms"), 180.0)

    rows: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    for source in sorted(candidate_rows, key=lambda row: ff(row.get("time_ms"))):
        row = dict(source)
        row["policy_id"] = policy["policy_id"]
        row["threshold"] = threshold
        row["fable_noise_veto_threshold"] = veto_threshold
        row["smart_dedupe_ms"] = smart_dedupe_ms
        row["counted"] = False

        fable_noise_veto = (
            veto_threshold < 1.0
            and row.get("fable_label") == "noise"
            and ff(row.get("fable_confidence")) >= veto_threshold
        )
        if fable_noise_veto:
            row["decision"] = "rejected_fable_noise_veto"
            row["reject_reason"] = "fable_noise_veto"
        elif ff(row.get("classifier_probability")) >= threshold:
            row["decision"] = "accepted_pending_dedupe"
            row["reject_reason"] = ""
            accepted.append(row)
        else:
            row["decision"] = "classified_low_probability"
            row["reject_reason"] = "below_threshold"
        rows.append(row)

    accepted.sort(key=lambda row: ff(row.get("time_ms")))
    clusters: list[list[dict[str, Any]]] = []
    cluster: list[dict[str, Any]] = []
    for row in accepted:
        if cluster and ff(row.get("time_ms")) - ff(cluster[-1].get("time_ms")) > smart_dedupe_ms:
            clusters.append(cluster)
            cluster = []
        cluster.append(row)
    if cluster:
        clusters.append(cluster)

    accepted_ids_to_final: dict[int, tuple[bool, str, str]] = {}
    for items in clusters:
        winner = max(
            items,
            key=lambda row: (ff(row.get("classifier_probability")), ff(row.get("peak_value"))),
        )
        for row in items:
            if row is winner:
                accepted_ids_to_final[id(row)] = (True, "counted", "")
            else:
                accepted_ids_to_final[id(row)] = (
                    False,
                    "deduped_lower_probability",
                    "deduped_lower_probability",
                )

    for row in rows:
        if row["decision"] != "accepted_pending_dedupe":
            continue
        counted, decision, reject_reason = accepted_ids_to_final.get(id(row), (False, row["decision"], ""))
        row["counted"] = counted
        row["decision"] = decision
        row["reject_reason"] = reject_reason
    return rows


def match_counted(counted_ms: list[float], truth_ms: tuple[float, ...]) -> dict[str, Any]:
    pairs: list[tuple[float, int, int]] = []
    for pred_idx, pred in enumerate(counted_ms):
        for truth_idx, truth in enumerate(truth_ms):
            delta = abs(pred - truth)
            if delta <= MATCH_TOLERANCE_MS:
                pairs.append((delta, pred_idx, truth_idx))
    pairs.sort(key=lambda item: item[0])
    used_pred: set[int] = set()
    used_truth: set[int] = set()
    deltas: list[float] = []
    for delta, pred_idx, truth_idx in pairs:
        if pred_idx in used_pred or truth_idx in used_truth:
            continue
        used_pred.add(pred_idx)
        used_truth.add(truth_idx)
        deltas.append(delta)
    return {
        "true_positive": len(used_truth),
        "false_positive": len(counted_ms) - len(used_pred),
        "missed": len(truth_ms) - len(used_truth),
        "median_abs_delta_ms": float(np.median(deltas)) if deltas else "",
    }


def summarize_session_decisions(
    session: EvalSession,
    config_id: str,
    family: str,
    policy: dict[str, Any],
    decision_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    counted_ms = [ff(row.get("time_ms")) for row in decision_rows if row.get("counted")]
    score = match_counted(counted_ms, session.label_times_ms)
    decisions = Counter(str(row.get("decision", "")) for row in decision_rows)
    reject_reasons = Counter(str(row.get("reject_reason", "")) for row in decision_rows if row.get("reject_reason"))
    return {
        "config_id": config_id,
        "family": family,
        "policy_id": policy["policy_id"],
        "threshold": policy["threshold"],
        "fable_noise_veto_threshold": policy["fable_noise_veto_threshold"],
        "smart_dedupe_ms": policy["smart_dedupe_ms"],
        "session_id": session.session_id,
        "scenario_id": session.scenario_id,
        "scenario_title": session.scenario_title,
        "polarity": session.polarity,
        "source": session.source,
        "expected_count": session.expected_count,
        "truth_count": len(session.label_times_ms),
        "candidate_count": len(decision_rows),
        "classified_count": sum(1 for row in decision_rows if row.get("classifier_probability") != ""),
        "counted": len(counted_ms),
        "true_positive": score["true_positive"],
        "false_positive": score["false_positive"],
        "missed": score["missed"],
        "recall": pct(score["true_positive"], len(session.label_times_ms)),
        "median_abs_delta_ms": score["median_abs_delta_ms"],
        "low_probability": decisions["classified_low_probability"],
        "fable_noise_vetoed": decisions["rejected_fable_noise_veto"],
        "deduped": decisions["deduped_lower_probability"],
        "js_error": decisions["js_error"],
        "reject_reasons_json": json.dumps(reject_reasons, sort_keys=True),
    }


def summarize_policy(session_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in session_rows:
        grouped[(row["config_id"], row["family"], row["policy_id"])].append(row)
    summaries: list[dict[str, Any]] = []
    for (config_id, family, policy_id), rows in grouped.items():
        positive = [row for row in rows if row["polarity"] == "positive"]
        negative = [row for row in rows if row["polarity"] == "negative"]
        truth_total = sum(intish(row["truth_count"]) for row in positive)
        tp = sum(intish(row["true_positive"]) for row in positive)
        fp_pos = sum(intish(row["false_positive"]) for row in positive)
        missed = sum(intish(row["missed"]) for row in positive)
        false_negative = sum(intish(row["counted"]) for row in negative)
        summaries.append(
            {
                "config_id": config_id,
                "family": family,
                "policy_id": policy_id,
                "positive_sessions": len(positive),
                "negative_sessions": len(negative),
                "truth_total": truth_total,
                "true_positive": tp,
                "recall": pct(tp, truth_total),
                "missed": missed,
                "positive_unmatched_counted": fp_pos,
                "negative_false_counts": false_negative,
                "total_false_counts": fp_pos + false_negative,
                "positive_candidates": sum(intish(row["candidate_count"]) for row in positive),
                "negative_candidates": sum(intish(row["candidate_count"]) for row in negative),
                "low_probability": sum(intish(row["low_probability"]) for row in rows),
                "fable_noise_vetoed": sum(intish(row["fable_noise_vetoed"]) for row in rows),
                "deduped": sum(intish(row["deduped"]) for row in rows),
            }
        )
    summaries.sort(
        key=lambda row: (
            row["policy_id"] != "love_t0104e_p025_veto098",
            intish(row["negative_false_counts"]),
            -intish(row["true_positive"]),
            intish(row["total_false_counts"]),
        )
    )
    return summaries


def summarize_scenario(session_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in session_rows:
        grouped[
            (
                row["config_id"],
                row["family"],
                row["policy_id"],
                row["polarity"],
                row["scenario_id"],
            )
        ].append(row)
    summaries: list[dict[str, Any]] = []
    for (config_id, family, policy_id, polarity, scenario_id), rows in grouped.items():
        truth_total = sum(intish(row["truth_count"]) for row in rows)
        tp = sum(intish(row["true_positive"]) for row in rows)
        counted = sum(intish(row["counted"]) for row in rows)
        summaries.append(
            {
                "config_id": config_id,
                "family": family,
                "policy_id": policy_id,
                "polarity": polarity,
                "scenario_id": scenario_id,
                "scenario_title": rows[0].get("scenario_title", ""),
                "sessions": len(rows),
                "truth_total": truth_total,
                "counted": counted,
                "true_positive": tp,
                "missed": sum(intish(row["missed"]) for row in rows),
                "false_counts": sum(intish(row["false_positive"]) for row in rows),
                "recall": pct(tp, truth_total),
            }
        )
    summaries.sort(key=lambda row: (row["policy_id"], row["config_id"], row["polarity"], row["scenario_id"]))
    return summaries


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_sessions()
    if not sessions:
        raise SystemExit("No evaluation sessions found")

    fable_model = FableAppModel.load(FABLE_MODEL_JSON)
    t0104e_model = RfJsonModel.load(T0104E_MODEL_JSON)

    all_configs = {config_id: (family, cfg) for config_id, family, cfg in get_configs()}
    missing = [config_id for config_id in CONFIG_IDS if config_id not in all_configs]
    if missing:
        raise SystemExit(f"Missing T0123 candidate configs: {missing}")

    wav_cache: dict[Path, tuple[np.ndarray, int]] = {}
    feature_cache: dict[tuple[Path, int], tuple[dict[str, float], dict[str, Any]]] = {}
    scored_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}

    candidate_rows: list[dict[str, Any]] = []
    decision_rows_out: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    config_payload: dict[str, Any] = {}

    for config_id in CONFIG_IDS:
        family, cfg = all_configs[config_id]
        config_payload[config_id] = {"family": family, "config": asdict(cfg)}
        for session in sessions:
            if not session.wav_path.exists():
                continue
            if session.wav_path not in wav_cache:
                wav_cache[session.wav_path] = read_wav(session.wav_path)
            y, sr = wav_cache[session.wav_path]
            if sr != nr_config.TARGET_SR:
                raise ValueError(f"Expected {nr_config.TARGET_SR} Hz WAV, got {sr}: {session.wav_path}")

            events = detect_for_config(y, sr, family, cfg)
            scored = classify_candidates(
                session=session,
                config_id=config_id,
                family=family,
                y=y,
                sr=sr,
                events=events,
                fable_model=fable_model,
                t0104e_model=t0104e_model,
                feature_cache=feature_cache,
            )
            scored_cache[(config_id, session.session_id)] = scored
            for row in scored:
                candidate_rows.append(
                    {
                        key: row.get(key, "")
                        for key in [
                            "config_id",
                            "family",
                            "session_id",
                            "scenario_id",
                            "polarity",
                            "candidate_index",
                            "time_ms",
                            "frame_rms",
                            "bg_rms",
                            "peak_value",
                            "peak_ratio",
                            "peak_z",
                            "prev_gap_ms",
                            "next_gap_ms",
                            "neighbor_count_500ms",
                            "fable_label",
                            "fable_confidence",
                            "fable_prob_racket_bounce",
                            "fable_prob_noise",
                            "classifier_label",
                            "classifier_probability",
                        ]
                    }
                )

            for policy in POLICIES:
                decisions = apply_policy(scored, policy)
                session_rows.append(summarize_session_decisions(session, config_id, family, policy, decisions))
                for row in decisions:
                    decision_rows_out.append(
                        {
                            "config_id": config_id,
                            "family": family,
                            "policy_id": policy["policy_id"],
                            "session_id": session.session_id,
                            "scenario_id": session.scenario_id,
                            "polarity": session.polarity,
                            "candidate_index": row.get("candidate_index", ""),
                            "time_ms": row.get("time_ms", ""),
                            "classifier_probability": row.get("classifier_probability", ""),
                            "fable_label": row.get("fable_label", ""),
                            "fable_confidence": row.get("fable_confidence", ""),
                            "fable_prob_noise": row.get("fable_prob_noise", ""),
                            "decision": row.get("decision", ""),
                            "reject_reason": row.get("reject_reason", ""),
                            "counted": row.get("counted", False),
                        }
                    )

    policy_rows = summarize_policy(session_rows)
    scenario_rows = summarize_scenario(session_rows)

    write_csv(OUT_DIR / "t0124_candidate_scores.csv", candidate_rows)
    write_csv(OUT_DIR / "t0124_candidate_decisions.csv", decision_rows_out)
    write_csv(OUT_DIR / "t0124_session_summary.csv", session_rows)
    write_csv(OUT_DIR / "t0124_policy_summary.csv", policy_rows)
    write_csv(OUT_DIR / "t0124_scenario_summary.csv", scenario_rows)
    write_json(
        OUT_DIR / "t0124_summary.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sessions": len(sessions),
            "configs": config_payload,
            "policies": POLICIES,
            "model": {
                "fable_model_json": str(FABLE_MODEL_JSON.relative_to(ROOT)),
                "candidate_model_json": str(T0104E_MODEL_JSON.relative_to(ROOT)),
                "candidate_model_metadata": t0104e_model.metadata,
            },
            "top_policy_rows": policy_rows[:12],
            "notes": [
                "Evaluation only: no model training/export and no app/runtime change.",
                "The classifier receives T0104E feature names and raw ExtraTrees probabilities.",
                "Sliding-window candidate stats are mapped into the same peak feature fields as app peak candidates.",
            ],
        },
    )

    love_rows = [row for row in policy_rows if row["policy_id"] == "love_t0104e_p025_veto098"]
    strict_rows = [row for row in policy_rows if row["policy_id"] == "strict_t0104e_p0575_no_veto"]
    t0119_rows = [
        row
        for row in session_rows
        if row["session_id"] == "bounce_audio_test_session_2026-07-02T17-48-43-807Z"
        and row["policy_id"] == "love_t0104e_p025_veto098"
    ]

    love_rows_by_tp = sorted(
        love_rows,
        key=lambda row: (-intish(row["true_positive"]), intish(row["total_false_counts"])),
    )
    best_love = love_rows_by_tp[0] if love_rows_by_tp else {}

    report_lines = [
        "# T0124 Sliding-Window Full-Flow Replay",
        "",
        "Evaluation-only replay. No model training, model export, app runtime change, APK install, push, or production promotion.",
        "",
        "## Short Conclusion",
        "",
        f"- Best `p=0.25` / Fable-noise-veto `0.98` row by true positives: "
        f"`{best_love.get('config_id', '')}` -> "
        f"{best_love.get('true_positive', '')}/{best_love.get('truth_total', '')} "
        f"({best_love.get('recall', '')}), "
        f"negative false counts `{best_love.get('negative_false_counts', '')}`, "
        f"positive unmatched counted `{best_love.get('positive_unmatched_counted', '')}`.",
        "- This measures the full stack, not just timestamp coverage: candidate generation, Fable feature/model, T0104E ExtraTrees, threshold, noise veto, and smart dedupe.",
        "- Treat any sliding-window win as an offline signal only. The app still needs a separate implementation and phone test before promotion.",
        "",
        "## Compared Methods",
        "",
        "- `current_peak_abs008`: current repo `Bounce audio test` peak gate reference.",
        "- `soft_peak_abs003` / `soft_peak_hp_abs0015`: lower-floor peak recovery checks.",
        "- `sw_*`: sliding-window PCM local-maximum candidate generators from T0123.",
        "",
        "## Policy Summary",
        "",
        *md_table(
            policy_rows,
            [
                "config_id",
                "family",
                "policy_id",
                "truth_total",
                "true_positive",
                "recall",
                "missed",
                "positive_unmatched_counted",
                "negative_false_counts",
                "total_false_counts",
                "positive_candidates",
                "negative_candidates",
                "low_probability",
                "fable_noise_vetoed",
                "deduped",
            ],
        ),
        "",
        "## Love Test Policy Only",
        "",
        *md_table(
            love_rows,
            [
                "config_id",
                "family",
                "truth_total",
                "true_positive",
                "recall",
                "missed",
                "positive_unmatched_counted",
                "negative_false_counts",
                "total_false_counts",
                "low_probability",
                "fable_noise_vetoed",
                "deduped",
            ],
        ),
        "",
        "## Strict Reference Only",
        "",
        *md_table(
            strict_rows,
            [
                "config_id",
                "family",
                "truth_total",
                "true_positive",
                "recall",
                "missed",
                "positive_unmatched_counted",
                "negative_false_counts",
                "total_false_counts",
            ],
        ),
        "",
        "## Critical T0119 Speaking/Counting Clip",
        "",
        *md_table(
            t0119_rows,
            [
                "config_id",
                "family",
                "candidate_count",
                "counted",
                "true_positive",
                "false_positive",
                "missed",
                "recall",
                "low_probability",
                "fable_noise_vetoed",
                "deduped",
            ],
        ),
        "",
        "## Outputs",
        "",
        "- `t0124_policy_summary.csv`",
        "- `t0124_session_summary.csv`",
        "- `t0124_scenario_summary.csv`",
        "- `t0124_candidate_scores.csv`",
        "- `t0124_candidate_decisions.csv`",
        "- `t0124_summary.json`",
        "",
    ]
    (OUT_DIR / "t0124_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Wrote T0124 outputs to {OUT_DIR}")
    print("Best Love-policy rows:")
    for row in love_rows[:8]:
        print(
            f"  {row['config_id']}: TP {row['true_positive']}/{row['truth_total']} "
            f"({row['recall']}), neg false {row['negative_false_counts']}, "
            f"pos unmatched {row['positive_unmatched_counted']}"
        )


if __name__ == "__main__":
    main()
