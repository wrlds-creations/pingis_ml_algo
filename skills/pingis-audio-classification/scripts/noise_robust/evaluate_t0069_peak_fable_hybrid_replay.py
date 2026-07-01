#!/usr/bin/env python3
"""T0069 peak-candidate -> Fable/veto -> smart-dedupe replay.

Evaluation only. This script does not train, export, build, install, or change
any Collector runtime behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
for _path in (str(SCRIPT_DIR), str(SCRIPTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import nr_features  # noqa: E402
from evaluate_fable_audio_reliability_t0044 import (  # noqa: E402
    FableAppModel,
    FableRuntimeConfig,
)
from evaluate_t0067_peak_gate_replay import (  # noqa: E402
    PeakGateConfig,
    detect_peak_gate,
    load_exact_positive_truth,
    match_predictions,
    read_csv,
    read_wav,
    selected_review_rows,
    write_csv,
)
from evaluate_t0068_rms_vs_peak_gate_audit import (  # noqa: E402
    RmsGateConfig,
    detect_rms_gate,
)

DEFAULT_RAW_DIR = ROOT / "data/audio/raw/t0065_fable_training_audio_round_a/fable_training_audio"
DEFAULT_T0065_DIR = ROOT / "data/audio/models/evaluations/t0065_fable_training_audio_round_a"
DEFAULT_T0066_DIR = ROOT / "data/audio/models/evaluations/t0066_round_a_exact_label_review"
DEFAULT_MODEL_JSON = ROOT / "apps/collector/src/models/fable_audio_model.json"
DEFAULT_HELDOUT_WAV = (
    ROOT
    / "data/audio/raw/t0060_fresh_heldout_c2/fable_live_debug"
    / "fable_live_session_2026-06-29T13-29-50-713Z.wav"
)
DEFAULT_HELDOUT_LABELS = (
    ROOT
    / "data/audio/models/evaluations/t0063_t0060_heldout_label_ingest"
    / "t0063_exact_heldout_labels.csv"
)
DEFAULT_OUT_DIR = ROOT / "data/audio/models/evaluations/t0069_peak_fable_hybrid_replay"

TOLERANCES_MS = (140.0, 250.0)
HELDOUT_SESSION_ID = "fable_live_session_2026-06-29T13-29-50-713Z"

# Mirrors apps/collector/src/fableEngine.ts for the current installed Fable path.
CURRENT_APP_CONFIG = FableRuntimeConfig(
    quiet_confidence=0.65,
    loud_confidence=0.9,
    loud_bg_db=-42.0,
    merge_ms=120,
    same_bounce_ms=250,
    group_ms=80,
    echo_ms=300,
    echo_ratio=0.6,
)

RMS_CURRENT = RmsGateConfig("bandpass", 1.5, 120, 0.0015, False)
PEAK_FAST_BALANCED = PeakGateConfig("raw_abs", 3.0, 220.0, 500.0, 60.0, 0.08, 2.0, 0.0)


@dataclass(frozen=True)
class PipelineSpec:
    pipeline_id: str
    label: str
    gate_id: str
    mode: str
    min_racket_prob: float = 0.0
    require_argmax: bool = False
    max_surface_prob: float | None = None
    max_noise_prob: float | None = None
    dedupe_ms: float = 0.0
    notes: str = ""


def finite_float(value: Any, default: float = float("nan")) -> float:
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


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def truth_times_from_labels(path: Path) -> list[float]:
    if not path.exists():
        return []
    out: list[float] = []
    for row in read_csv_dicts(path):
        label = str(row.get("label") or row.get("review_label") or "").strip().lower()
        if label not in {"racket", "racket_bounce", "racket_contact"}:
            continue
        time_s = finite_float(row.get("reviewed_time_s"), finite_float(row.get("time_s")))
        if math.isfinite(time_s):
            out.append(time_s * 1000.0)
    return sorted(out)


def frame_rms_at(y: np.ndarray, sr: int, sample: int, window_ms: float = 10.0) -> float:
    size = max(1, int(round(sr * window_ms / 1000.0)))
    start = max(0, int(sample))
    end = min(len(y), start + size)
    if end <= start:
        return 0.0
    frame = np.asarray(y[start:end], dtype=np.float64)
    return float(np.sqrt(np.mean(frame * frame)))


def normalized_gate_events(y: np.ndarray, sr: int, gate_id: str) -> list[dict[str, Any]]:
    if gate_id == "rms_current":
        raw = detect_rms_gate(y, sr, RMS_CURRENT)
    elif gate_id == "peak_fast_balanced":
        raw = detect_peak_gate(y, sr, PEAK_FAST_BALANCED)
    else:
        raise ValueError(f"Unknown gate_id: {gate_id}")

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        time_ms = finite_float(item.get("time_ms"), finite_float(item.get("onset_ms")))
        if not math.isfinite(time_ms):
            continue
        onset_sample = int(item.get("onset_sample") or round(time_ms / 1000.0 * sr))
        rows.append(
            {
                "candidate_index": index,
                "gate_id": gate_id,
                "time_ms": time_ms,
                "time_s": time_ms / 1000.0,
                "onset_sample": onset_sample,
                "frame_rms": finite_float(item.get("frame_rms"), frame_rms_at(y, sr, onset_sample)),
                "bg_rms": finite_float(item.get("bg_rms"), finite_float(item.get("local_bg"))),
                "peak_value": finite_float(item.get("peak_value"), finite_float(item.get("frame_rms"))),
                "peak_ratio": finite_float(item.get("ratio")),
                "peak_z": finite_float(item.get("z")),
            }
        )
    return rows


def classify_candidates(
    *,
    model: FableAppModel,
    y: np.ndarray,
    sr: int,
    session_id: str,
    gate_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in normalized_gate_events(y, sr, gate_id):
        onset_sample = int(candidate["onset_sample"])
        clip = nr_features.extract_live_clip(y, onset_sample)
        features = nr_features.extract_all_features(clip, sr)
        prediction = model.predict_features(features)
        probs = prediction.get("probabilities") or {}
        p_racket = finite_float(probs.get("racket_bounce"), 0.0)
        p_noise = finite_float(probs.get("noise"), 0.0)
        p_floor = finite_float(probs.get("floor_bounce"), 0.0)
        p_table = finite_float(probs.get("table_bounce"), 0.0)
        ratio = finite_float(candidate.get("peak_ratio"), 0.0)
        peak_value = finite_float(candidate.get("peak_value"), 0.0)
        frame_rms = finite_float(candidate.get("frame_rms"), 0.0)
        score = p_racket + 0.01 * math.log1p(max(0.0, ratio)) + 0.03 * min(1.0, max(peak_value, frame_rms))
        rows.append(
            {
                **candidate,
                "session_id": session_id,
                "model_label": str(prediction.get("label") or ""),
                "model_confidence": finite_float(prediction.get("confidence"), 0.0),
                "prob_racket_bounce": p_racket,
                "prob_noise": p_noise,
                "prob_floor_bounce": p_floor,
                "prob_table_bounce": p_table,
                "nr_bg_rms_db": finite_float(features.get("nr_bg_rms_db"), -100.0),
                "nr_snr_db_est": finite_float(features.get("nr_snr_db_est"), 0.0),
                "nr_bp_peak_ratio": finite_float(features.get("nr_bp_peak_ratio"), 0.0),
                "nr_bp_peak_db": finite_float(features.get("nr_bp_peak_db"), -100.0),
                "score": score,
            }
        )
    return rows


def threshold_for_candidate(row: dict[str, Any], fast_rebound: bool) -> tuple[float, str]:
    bg_rms_db = finite_float(row.get("nr_bg_rms_db"), -100.0)
    loud = bg_rms_db >= CURRENT_APP_CONFIG.loud_bg_db
    threshold = CURRENT_APP_CONFIG.loud_confidence if loud else CURRENT_APP_CONFIG.quiet_confidence
    if fast_rebound:
        threshold = max(threshold, 0.9)
    return threshold, "loud" if loud else "quiet"


def apply_current_fable_counter(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    last_counted: tuple[float, float] | None = None
    group_start_ms: float | None = None

    for candidate in sorted(candidates, key=lambda item: float(item["time_ms"])):
        row = dict(candidate)
        onset_ms = finite_float(row.get("time_ms"))
        frame_rms = max(finite_float(row.get("frame_rms"), 0.0), 0.0)
        fast_rebound = False
        reject_reason = ""
        counted = False

        if last_counted is not None:
            since_counted = onset_ms - last_counted[0]
            rms_ratio = frame_rms / max(last_counted[1], 1e-9)
            row["since_last_counted_ms"] = since_counted
            row["rms_ratio_to_last_counted"] = rms_ratio
            if since_counted <= CURRENT_APP_CONFIG.same_bounce_ms:
                if rms_ratio >= 1.1:
                    last_counted = (onset_ms, frame_rms)
                    group_start_ms = onset_ms
                    reject_reason = "same_bounce"
                elif rms_ratio <= CURRENT_APP_CONFIG.echo_ratio:
                    reject_reason = "echo_window"
                elif since_counted < 150:
                    reject_reason = "same_bounce"
                else:
                    fast_rebound = True
            elif group_start_ms is not None and onset_ms - group_start_ms <= CURRENT_APP_CONFIG.group_ms:
                reject_reason = "group_window"
            elif since_counted <= CURRENT_APP_CONFIG.echo_ms and rms_ratio <= CURRENT_APP_CONFIG.echo_ratio:
                reject_reason = "echo_window"

        threshold, bg_mode = threshold_for_candidate(row, fast_rebound)
        if not reject_reason:
            if row.get("model_label") != "racket_bounce":
                reject_reason = "not_racket"
            elif finite_float(row.get("model_confidence"), 0.0) < threshold:
                reject_reason = "low_confidence_loud_bg" if bg_mode == "loud" else "low_confidence"
            else:
                counted = True
                last_counted = (onset_ms, frame_rms)
                group_start_ms = onset_ms

        row.update(
            {
                "counted": counted,
                "reject_reason": reject_reason,
                "confidence_threshold": threshold,
                "bg_mode": bg_mode,
                "policy_score": finite_float(row.get("score"), 0.0),
            }
        )
        rows.append(row)
    return rows


def initial_policy_decision(candidate: dict[str, Any], policy: PipelineSpec) -> tuple[bool, str]:
    p_racket = finite_float(candidate.get("prob_racket_bounce"), 0.0)
    p_noise = finite_float(candidate.get("prob_noise"), 0.0)
    p_floor = finite_float(candidate.get("prob_floor_bounce"), 0.0)
    p_table = finite_float(candidate.get("prob_table_bounce"), 0.0)
    p_surface = max(p_floor, p_table)

    if policy.require_argmax and candidate.get("model_label") != "racket_bounce":
        return False, "classifier_not_racket"
    if p_racket < policy.min_racket_prob:
        return False, "low_racket_prob"
    if policy.max_surface_prob is not None and p_surface >= policy.max_surface_prob and p_surface > p_racket:
        return False, "surface_veto"
    if policy.max_noise_prob is not None and p_noise >= policy.max_noise_prob and p_noise > p_racket:
        return False, "noise_veto"
    return True, "accepted_before_dedupe"


def apply_probability_policy(candidates: list[dict[str, Any]], policy: PipelineSpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    accepted_indices: list[int] = []
    for candidate in sorted(candidates, key=lambda item: float(item["time_ms"])):
        accepted, reason = initial_policy_decision(candidate, policy)
        row = {
            **candidate,
            "counted": False,
            "accepted_before_dedupe": accepted,
            "reject_reason": "" if accepted else reason,
            "confidence_threshold": policy.min_racket_prob,
            "bg_mode": "policy",
            "policy_score": finite_float(candidate.get("score"), 0.0),
        }
        rows.append(row)
        if accepted:
            accepted_indices.append(len(rows) - 1)

    if policy.dedupe_ms <= 0:
        for idx in accepted_indices:
            rows[idx]["counted"] = True
            rows[idx]["reject_reason"] = ""
        return rows

    clusters: list[list[int]] = []
    current: list[int] = []
    for idx in accepted_indices:
        if current and finite_float(rows[idx]["time_ms"]) - finite_float(rows[current[-1]]["time_ms"]) > policy.dedupe_ms:
            clusters.append(current)
            current = []
        current.append(idx)
    if current:
        clusters.append(current)

    for cluster in clusters:
        keep = max(cluster, key=lambda idx: finite_float(rows[idx].get("policy_score"), 0.0))
        for idx in cluster:
            if idx == keep:
                rows[idx]["counted"] = True
                rows[idx]["reject_reason"] = ""
            else:
                rows[idx]["counted"] = False
                rows[idx]["reject_reason"] = "smart_dedupe"
    return rows


def pipeline_specs() -> list[PipelineSpec]:
    specs = [
        PipelineSpec(
            "rms_current_app_counter",
            "RMS current + Fable app counter",
            "rms_current",
            "app_counter",
            notes="Current RMS/native-style candidates and current app Fable counter.",
        ),
        PipelineSpec(
            "peak_fb_app_counter",
            "Peak fast balanced + Fable app counter",
            "peak_fast_balanced",
            "app_counter",
            notes="Peak candidates passed through the current app Fable counter.",
        ),
        PipelineSpec(
            "peak_fb_argmax_smart240",
            "Peak + Fable argmax + smart240",
            "peak_fast_balanced",
            "probability",
            require_argmax=True,
            dedupe_ms=240.0,
            notes="Count only candidates whose Fable argmax is racket_bounce.",
        ),
    ]
    for prob in (0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50):
        suffix = str(prob).replace(".", "p")
        specs.append(
            PipelineSpec(
                f"peak_fb_prob{suffix}_smart240",
                f"Peak + p_racket>={prob:g} + smart240",
                "peak_fast_balanced",
                "probability",
                min_racket_prob=prob,
                dedupe_ms=240.0,
                notes="Fable racket probability threshold, no explicit noise/surface veto.",
            )
        )
    for prob in (0.02, 0.05, 0.10):
        suffix = str(prob).replace(".", "p")
        specs.append(
            PipelineSpec(
                f"peak_fb_prob{suffix}_surface080_smart240",
                f"Peak + p_racket>={prob:g} + surface veto + smart240",
                "peak_fast_balanced",
                "probability",
                min_racket_prob=prob,
                max_surface_prob=0.80,
                dedupe_ms=240.0,
                notes="Reject strong table/floor predictions when they beat racket.",
            )
        )
    for prob in (0.02, 0.05):
        suffix = str(prob).replace(".", "p")
        specs.append(
            PipelineSpec(
                f"peak_fb_prob{suffix}_noise099_surface080_smart240",
                f"Peak + p_racket>={prob:g} + noise/surface veto + smart240",
                "peak_fast_balanced",
                "probability",
                min_racket_prob=prob,
                max_noise_prob=0.99,
                max_surface_prob=0.80,
                dedupe_ms=240.0,
                notes="Reject very strong noise plus strong table/floor predictions.",
            )
        )
    for prob in (0.01, 0.02, 0.05):
        suffix = str(prob).replace(".", "p")
        specs.append(
            PipelineSpec(
                f"peak_fb_prob{suffix}_smart300",
                f"Peak + p_racket>={prob:g} + smart300",
                "peak_fast_balanced",
                "probability",
                min_racket_prob=prob,
                dedupe_ms=300.0,
                notes="More aggressive dedupe window.",
            )
        )
    return specs


def apply_pipeline(candidates: list[dict[str, Any]], policy: PipelineSpec) -> list[dict[str, Any]]:
    if policy.mode == "app_counter":
        rows = apply_current_fable_counter(candidates)
    elif policy.mode == "probability":
        rows = apply_probability_policy(candidates, policy)
    else:
        raise ValueError(f"Unknown policy mode: {policy.mode}")
    return [
        {
            **row,
            "pipeline_id": policy.pipeline_id,
            "pipeline_label": policy.label,
            "pipeline_mode": policy.mode,
        }
        for row in rows
    ]


def counted_times(rows: list[dict[str, Any]]) -> list[float]:
    return [finite_float(row.get("time_ms")) for row in rows if row.get("counted") is True]


def summarize_match(pred_ms: list[float], truth_ms: list[float], tolerance_ms: float) -> dict[str, Any]:
    result = match_predictions(pred_ms, truth_ms, tolerance_ms)
    deltas = [float(delta) for _, _, delta in result["matches"]]
    abs_deltas = [abs(delta) for delta in deltas]
    return {
        "tp": result["tp"],
        "fp": result["fp"],
        "missed": result["missed"],
        "median_abs_delta_ms": float(np.median(abs_deltas)) if abs_deltas else "",
        "p95_abs_delta_ms": float(np.percentile(abs_deltas, 95)) if abs_deltas else "",
    }


def score_selected_exact(
    *,
    policies: list[PipelineSpec],
    selected_rows: list[dict[str, str]],
    truth_by_session: dict[str, list[float]],
    get_results: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for policy in policies:
        matches_by_tol = {tol: {"tp": 0, "fp": 0, "missed": 0, "median_values": [], "p95_values": []} for tol in TOLERANCES_MS}
        truth_total = 0
        positive_counted = 0
        positive_candidates = 0
        negative_counted = 0
        negative_candidates = 0
        negative_by_scenario: Counter[str] = Counter()
        reject_reasons: Counter[str] = Counter()

        for selected in selected_rows:
            sid = selected["session_id"]
            results = get_results(policy, sid)
            pred_ms = counted_times(results)
            candidate_count = len(results)
            for result in results:
                reject_reasons[str(result.get("reject_reason") or "counted")] += 1

            if sid in truth_by_session:
                truth = truth_by_session[sid]
                truth_total += len(truth)
                positive_counted += len(pred_ms)
                positive_candidates += candidate_count
                detail = {
                    "pipeline_id": policy.pipeline_id,
                    "pipeline_label": policy.label,
                    "session_id": sid,
                    "scenario_title": selected.get("scenario_title", ""),
                    "polarity": "positive_exact",
                    "truth": len(truth),
                    "candidates": candidate_count,
                    "counted": len(pred_ms),
                }
                for tol in TOLERANCES_MS:
                    matched = summarize_match(pred_ms, truth, tol)
                    aggregate = matches_by_tol[tol]
                    aggregate["tp"] += matched["tp"]
                    aggregate["fp"] += matched["fp"]
                    aggregate["missed"] += matched["missed"]
                    if matched["median_abs_delta_ms"] != "":
                        aggregate["median_values"].append(float(matched["median_abs_delta_ms"]))
                    if matched["p95_abs_delta_ms"] != "":
                        aggregate["p95_values"].append(float(matched["p95_abs_delta_ms"]))
                    detail[f"tp_{int(tol)}ms"] = matched["tp"]
                    detail[f"fp_{int(tol)}ms"] = matched["fp"]
                    detail[f"missed_{int(tol)}ms"] = matched["missed"]
                detail_rows.append(detail)
            elif selected.get("polarity") == "negative":
                negative_counted += len(pred_ms)
                negative_candidates += candidate_count
                negative_by_scenario[selected.get("scenario_title", "")] += len(pred_ms)
                detail_rows.append(
                    {
                        "pipeline_id": policy.pipeline_id,
                        "pipeline_label": policy.label,
                        "session_id": sid,
                        "scenario_title": selected.get("scenario_title", ""),
                        "polarity": "expected_zero",
                        "truth": 0,
                        "candidates": candidate_count,
                        "counted": len(pred_ms),
                        "tp_140ms": 0,
                        "fp_140ms": len(pred_ms),
                        "missed_140ms": 0,
                    }
                )

        summary = {
            "pipeline_id": policy.pipeline_id,
            "pipeline_label": policy.label,
            "gate_id": policy.gate_id,
            "mode": policy.mode,
            "min_racket_prob": policy.min_racket_prob,
            "require_argmax": policy.require_argmax,
            "max_surface_prob": policy.max_surface_prob if policy.max_surface_prob is not None else "",
            "max_noise_prob": policy.max_noise_prob if policy.max_noise_prob is not None else "",
            "dedupe_ms": policy.dedupe_ms,
            "exact_truth": truth_total,
            "positive_candidates": positive_candidates,
            "positive_counted": positive_counted,
            "selected_expected_zero_candidates": negative_candidates,
            "selected_expected_zero_false_counts": negative_counted,
            "selected_expected_zero_by_scenario": json.dumps(dict(sorted(negative_by_scenario.items())), sort_keys=True),
            "reject_reason_counts": json.dumps(dict(sorted(reject_reasons.items())), sort_keys=True),
            "notes": policy.notes,
        }
        for tol in TOLERANCES_MS:
            aggregate = matches_by_tol[tol]
            tp = int(aggregate["tp"])
            fp_pos = int(aggregate["fp"])
            missed = int(aggregate["missed"])
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
                    f"median_abs_delta_{int(tol)}ms": (
                        float(np.median(aggregate["median_values"])) if aggregate["median_values"] else ""
                    ),
                    f"p95_abs_delta_{int(tol)}ms": (
                        float(np.median(aggregate["p95_values"])) if aggregate["p95_values"] else ""
                    ),
                }
            )
        summary_rows.append(summary)
    summary_rows.sort(
        key=lambda row: (
            -float(row.get("f1_including_negatives_140ms", 0.0)),
            -float(row.get("recall_140ms", 0.0)),
            int(row.get("selected_expected_zero_false_counts", 10**9)),
        )
    )
    return summary_rows, detail_rows


def score_heldout_exact(
    *,
    policies: list[PipelineSpec],
    truth_ms: list[float],
    get_results: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not truth_ms:
        return rows
    for policy in policies:
        results = get_results(policy, HELDOUT_SESSION_ID)
        pred_ms = counted_times(results)
        row: dict[str, Any] = {
            "pipeline_id": policy.pipeline_id,
            "pipeline_label": policy.label,
            "gate_id": policy.gate_id,
            "candidates": len(results),
            "counted": len(pred_ms),
            "truth": len(truth_ms),
        }
        for tol in TOLERANCES_MS:
            matched = summarize_match(pred_ms, truth_ms, tol)
            precision = matched["tp"] / (matched["tp"] + matched["fp"]) if (matched["tp"] + matched["fp"]) else 0.0
            recall = matched["tp"] / len(truth_ms) if truth_ms else 0.0
            row.update(
                {
                    f"tp_{int(tol)}ms": matched["tp"],
                    f"fp_{int(tol)}ms": matched["fp"],
                    f"missed_{int(tol)}ms": matched["missed"],
                    f"precision_{int(tol)}ms": precision,
                    f"recall_{int(tol)}ms": recall,
                }
            )
        rows.append(row)
    rows.sort(
        key=lambda row: (
            -float(row.get("recall_140ms", 0.0)),
            int(row.get("fp_140ms", 10**9)),
        )
    )
    return rows


def round_a_block_rows(
    *,
    policies: list[PipelineSpec],
    manifest_rows: list[dict[str, str]],
    get_results: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy in policies:
        for item in manifest_rows:
            sid = item["session_id"]
            results = get_results(policy, sid)
            count = len(counted_times(results))
            candidates = len(results)
            accepted_pre_dedupe = sum(1 for result in results if result.get("accepted_before_dedupe") is True)
            expected = intish(item.get("expected_racket_contacts"))
            rows.append(
                {
                    "pipeline_id": policy.pipeline_id,
                    "pipeline_label": policy.label,
                    "gate_id": policy.gate_id,
                    "session_id": sid,
                    "scenario_id": item.get("scenario_id", ""),
                    "scenario_title": item.get("scenario_title", ""),
                    "polarity": item.get("polarity", ""),
                    "expected_contacts": expected,
                    "candidate_count": candidates,
                    "accepted_before_dedupe": accepted_pre_dedupe,
                    "counted": count,
                    "count_error": count - expected,
                    "abs_count_error": abs(count - expected),
                    "duration_s": finite_float(item.get("wav_duration_s"), 0.0),
                }
            )
    return rows


def summarize_round_a(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_scenario: dict[tuple[str, str], dict[str, Any]] = {}
    by_pipeline: dict[str, dict[str, Any]] = {}
    for row in rows:
        pipeline_id = str(row["pipeline_id"])
        scenario_key = (pipeline_id, str(row["scenario_id"]))
        scenario = by_scenario.setdefault(
            scenario_key,
            {
                "pipeline_id": pipeline_id,
                "pipeline_label": row["pipeline_label"],
                "gate_id": row["gate_id"],
                "scenario_id": row["scenario_id"],
                "scenario_title": row["scenario_title"],
                "polarity": row["polarity"],
                "clips": 0,
                "expected_contacts": 0,
                "candidate_count": 0,
                "accepted_before_dedupe": 0,
                "counted": 0,
                "abs_count_error": 0,
                "duration_s": 0.0,
            },
        )
        total = by_pipeline.setdefault(
            pipeline_id,
            {
                "pipeline_id": pipeline_id,
                "pipeline_label": row["pipeline_label"],
                "gate_id": row["gate_id"],
                "clips": 0,
                "positive_expected": 0,
                "positive_counted": 0,
                "positive_abs_count_error": 0,
                "negative_false_counts": 0,
                "candidate_count": 0,
                "accepted_before_dedupe": 0,
                "total_expected": 0,
                "total_counted": 0,
                "total_abs_count_error": 0,
            },
        )
        for target in (scenario, total):
            target["clips"] += 1
            target["candidate_count"] += int(row["candidate_count"])
            target["accepted_before_dedupe"] += int(row["accepted_before_dedupe"])
        scenario["expected_contacts"] += int(row["expected_contacts"])
        scenario["counted"] += int(row["counted"])
        scenario["abs_count_error"] += int(row["abs_count_error"])
        scenario["duration_s"] += finite_float(row.get("duration_s"), 0.0)

        total["total_expected"] += int(row["expected_contacts"])
        total["total_counted"] += int(row["counted"])
        total["total_abs_count_error"] += int(row["abs_count_error"])
        if row.get("polarity") == "positive":
            total["positive_expected"] += int(row["expected_contacts"])
            total["positive_counted"] += int(row["counted"])
            total["positive_abs_count_error"] += int(row["abs_count_error"])
        else:
            total["negative_false_counts"] += int(row["counted"])

    for item in by_scenario.values():
        item["count_error"] = int(item["counted"]) - int(item["expected_contacts"])
        item["counts_per_min"] = item["counted"] / (item["duration_s"] / 60.0) if item["duration_s"] else 0.0

    scenario_rows = sorted(by_scenario.values(), key=lambda row: (row["pipeline_id"], row["scenario_id"]))

    scenario_lookup = {(row["pipeline_id"], row["scenario_id"]): row for row in scenario_rows}
    total_rows: list[dict[str, Any]] = []
    for item in by_pipeline.values():
        item["total_error"] = int(item["total_counted"]) - int(item["total_expected"])
        for scenario_id, output_field in [
            ("fast_racket_bounce", "fast_counted"),
            ("racket_bounce_background_sound", "background_counted"),
            ("talking_only_no_bounce", "talking_only_false_counts"),
            ("racket_handling_no_bounce", "racket_handling_false_counts"),
            ("floor_table_other_impact_no_racket", "floor_table_other_false_counts"),
        ]:
            scenario = scenario_lookup.get((item["pipeline_id"], scenario_id))
            item[output_field] = int(scenario["counted"]) if scenario else 0
            if scenario:
                item[f"{output_field}_expected"] = int(scenario["expected_contacts"])
                item[f"{output_field}_error"] = int(scenario["count_error"])
        total_rows.append(item)
    total_rows.sort(
        key=lambda row: (
            int(row.get("negative_false_counts", 10**9)),
            int(row.get("positive_abs_count_error", 10**9)),
            abs(int(row.get("fast_counted_error", 10**9))),
        )
    )
    return scenario_rows, total_rows


def compact_event_rows(rows: list[dict[str, Any]], policy: PipelineSpec, meta: dict[str, str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("counted") is not True:
            continue
        out.append(
            {
                "pipeline_id": policy.pipeline_id,
                "pipeline_label": policy.label,
                "session_id": row.get("session_id", ""),
                "scenario_id": meta.get("scenario_id", ""),
                "scenario_title": meta.get("scenario_title", ""),
                "time_ms": round(finite_float(row.get("time_ms")), 3),
                "model_label": row.get("model_label", ""),
                "model_confidence": row.get("model_confidence", ""),
                "prob_racket_bounce": row.get("prob_racket_bounce", ""),
                "prob_noise": row.get("prob_noise", ""),
                "prob_floor_bounce": row.get("prob_floor_bounce", ""),
                "prob_table_bounce": row.get("prob_table_bounce", ""),
                "policy_score": row.get("policy_score", ""),
                "frame_rms": row.get("frame_rms", ""),
                "peak_value": row.get("peak_value", ""),
                "peak_ratio": row.get("peak_ratio", ""),
            }
        )
    return out


def md_table(rows: list[dict[str, Any]], fields: list[str], labels: list[str] | None = None, limit: int | None = None) -> list[str]:
    if labels is None:
        labels = fields
    shown = rows[:limit] if limit is not None else rows
    lines = [
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join("---" for _ in labels) + " |",
    ]
    for row in shown:
        values: list[str] = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                if "recall" in field or "precision" in field or "f1" in field:
                    values.append(f"{value:.3f}")
                elif "delta" in field or "per_" in field:
                    values.append(f"{value:.1f}")
                else:
                    values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def choose_recommendation(exact_rows: list[dict[str, Any]], round_totals: list[dict[str, Any]]) -> dict[str, Any]:
    exact_by_id = {row["pipeline_id"]: row for row in exact_rows}
    round_by_id = {row["pipeline_id"]: row for row in round_totals}
    baseline = exact_by_id["rms_current_app_counter"]
    baseline_round = round_by_id["rms_current_app_counter"]
    peak_rows = [row for row in exact_rows if str(row["pipeline_id"]).startswith("peak_")]
    best_exact = max(
        peak_rows,
        key=lambda row: (
            float(row.get("f1_including_negatives_140ms", 0.0)),
            float(row.get("recall_140ms", 0.0)),
            -int(row.get("selected_expected_zero_false_counts", 0)),
        ),
    )
    broad_candidates: list[dict[str, Any]] = []
    runtime_candidates: list[dict[str, Any]] = []
    for exact in peak_rows:
        round_row = round_by_id[exact["pipeline_id"]]
        broad_beats_baseline = (
            float(exact.get("recall_140ms", 0.0)) >= float(baseline.get("recall_140ms", 0.0))
            and int(exact.get("selected_expected_zero_false_counts", 10**9))
            <= int(baseline.get("selected_expected_zero_false_counts", 0))
            and int(round_row.get("negative_false_counts", 10**9))
            <= int(baseline_round.get("negative_false_counts", 0))
            and int(round_row.get("positive_abs_count_error", 10**9))
            <= int(baseline_round.get("positive_abs_count_error", 0))
        )
        if not broad_beats_baseline:
            continue
        candidate = {
            **exact,
            "round_negative_false_counts": int(round_row.get("negative_false_counts", 0)),
            "round_positive_abs_count_error": int(round_row.get("positive_abs_count_error", 0)),
            "talking_only_false_counts": int(round_row.get("talking_only_false_counts", 0)),
            "background_counted": int(round_row.get("background_counted", 0)),
        }
        broad_candidates.append(candidate)
        if int(round_row.get("talking_only_false_counts", 10**9)) <= int(baseline_round.get("talking_only_false_counts", 0)):
            runtime_candidates.append(candidate)

    def conservative_key(row: dict[str, Any]) -> tuple[float, int, int, int]:
        return (
            -int(row.get("round_negative_false_counts", 10**9)),
            -int(row.get("talking_only_false_counts", 10**9)),
            -int(row.get("round_positive_abs_count_error", 10**9)),
            float(row.get("recall_140ms", 0.0)),
        )

    best_broad = max(broad_candidates, key=conservative_key) if broad_candidates else None
    best_runtime = max(runtime_candidates, key=conservative_key) if runtime_candidates else None
    best_broad_round = round_by_id[best_broad["pipeline_id"]] if best_broad else None
    talking_regression = (
        best_broad_round is not None
        and int(best_broad_round.get("talking_only_false_counts", 0))
        > int(baseline_round.get("talking_only_false_counts", 0))
    )
    return {
        "baseline_pipeline": baseline["pipeline_id"],
        "best_exact_f1_peak_pipeline": best_exact["pipeline_id"],
        "best_exact_f1_peak_label": best_exact["pipeline_label"],
        "best_conservative_peak_pipeline": best_broad["pipeline_id"] if best_broad else "",
        "best_conservative_peak_label": best_broad["pipeline_label"] if best_broad else "",
        "talking_only_baseline_false_counts": int(baseline_round.get("talking_only_false_counts", 0)),
        "talking_only_best_conservative_false_counts": (
            int(best_broad_round.get("talking_only_false_counts", 0)) if best_broad_round else ""
        ),
        "broad_peak_candidate_beats_aggregate_baseline": best_broad is not None,
        "talking_only_regression_blocks_runtime": talking_regression,
        "beats_current_rms_fable_baseline": best_runtime is not None,
        "recommendation": (
            "peak_hybrid_candidate_ready_for_followup_runtime_candidate"
            if best_runtime is not None
            else "do_not_replace_current_runtime_with_tested_peak_fable_hybrids"
        ),
    }


def render_report(
    summary: dict[str, Any],
    exact_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
    round_totals: list[dict[str, Any]],
    scenario_rows: list[dict[str, Any]],
) -> str:
    recommendation = summary["recommendation"]
    exact_display = sorted(
        exact_rows,
        key=lambda row: (
            0 if row["pipeline_id"] == "rms_current_app_counter" else 1,
            -float(row.get("f1_including_negatives_140ms", 0.0)),
        ),
    )
    round_display = sorted(
        round_totals,
        key=lambda row: (
            0 if row["pipeline_id"] == "rms_current_app_counter" else 1,
            int(row.get("negative_false_counts", 10**9)),
            int(row.get("positive_abs_count_error", 10**9)),
        ),
    )
    scenario_display = [
        row
        for row in scenario_rows
        if row["pipeline_id"]
        in {
            "rms_current_app_counter",
            recommendation["best_exact_f1_peak_pipeline"],
            recommendation["best_conservative_peak_pipeline"],
            "peak_fb_app_counter",
        }
    ]

    lines = [
        "# T0069 Peak/Fable/Veto/Dedupe Replay",
        "",
        f"Generated at: `{summary['generated_at']}`",
        "",
        "## Scope",
        "",
        "- Evaluation only: no model JSON, app runtime, APK, training export, camera logic, cloud/API, or AWS change.",
        "- Current baseline is recomputed RMS/native-style candidates plus the current Fable app counter.",
        "- Peak candidate source is T0068 `Peak fast balanced`.",
        "- Selected expected-zero negatives are scenario-derived hard-negative clips, not individually timestamp-reviewed permanent truth.",
        "",
        "## Exact T0066 Background Labels",
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
                "positive_counted",
            ],
            [
                "Pipeline",
                "TP",
                "Miss",
                "Pos FP",
                "Neg FP",
                "Precision",
                "Recall",
                "F1",
                "Pos Count",
            ],
            limit=12,
        ),
        "",
        "## Held-Out C2 Exact Labels",
        "",
        *md_table(
            heldout_rows,
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
        "## Round A Block Replay",
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
            [
                "Pipeline",
                "Pos Count",
                "Pos Exp",
                "Pos Abs Err",
                "Neg FP",
                "Fast",
                "Background",
                "Talking FP",
                "Handling FP",
                "Floor/Table FP",
            ],
            limit=12,
        ),
        "",
        "## Scenario Detail For Baseline And Best Peak",
        "",
        *md_table(
            scenario_display,
            [
                "pipeline_label",
                "scenario_title",
                "expected_contacts",
                "counted",
                "count_error",
                "candidate_count",
                "accepted_before_dedupe",
            ],
            ["Pipeline", "Scenario", "Expected", "Count", "Error", "Cand.", "Pre-Dedupe"],
        ),
        "",
        "## Recommendation",
        "",
    ]
    if recommendation["beats_current_rms_fable_baseline"]:
        lines.append(
            "- A tested peak hybrid beats the recomputed current RMS + Fable baseline on this offline replay. "
            "The next ticket can be a carefully scoped app-runtime candidate with RMS kept as fallback and a fresh Motorola validation round."
        )
    elif recommendation["broad_peak_candidate_beats_aggregate_baseline"]:
        lines.append(
            "- A conservative peak hybrid beats the aggregate baseline on exact recall, selected hard negatives, Round A positive count error, and total Round A negative false counts, "
            "but it still is not runtime-ready because talking-only false counts regress."
        )
    else:
        lines.append(
            "- Do not replace the current runtime with the tested peak + current-Fable/veto/dedupe chains. "
            "Peak candidates are better than RMS as raw timing candidates, but the current Fable probabilities/vetoes still do not turn them into a safe final counter across all scenarios."
        )
    lines += [
        f"- Best tested peak chain by exact selected F1: `{recommendation['best_exact_f1_peak_label']}`.",
        f"- Best conservative aggregate peak chain: `{recommendation['best_conservative_peak_label'] or 'none'}`.",
        f"- Talking-only false counts baseline vs best conservative peak: `{recommendation['talking_only_baseline_false_counts']}` vs `{recommendation['talking_only_best_conservative_false_counts']}`.",
        "- If the best peak chain has high recall but unsafe false counts, the next lever is a trained classifier/veto using the newly labeled data, not another raw gate tweak.",
        "",
        "## Outputs",
        "",
        "- `t0069_exact_selected_comparison.csv`",
        "- `t0069_exact_selected_clip_rows.csv`",
        "- `t0069_heldout_c2_exact_comparison.csv`",
        "- `t0069_round_a_block_replay.csv`",
        "- `t0069_round_a_by_scenario.csv`",
        "- `t0069_round_a_pipeline_summary.csv`",
        "- `t0069_counted_events.csv`",
        "- `t0069_peak_candidate_predictions_selected.csv`",
        "- `t0069_summary.json`",
    ]
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw_dir = Path(args.raw_dir)
    t0065_dir = Path(args.t0065_dir)
    t0066_dir = Path(args.t0066_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = FableAppModel.load(Path(args.model_json))
    policies = pipeline_specs()
    manifest_rows = [row for row in read_csv(t0065_dir / "t0065_fable_training_audio_manifest.csv") if row.get("round") == "round_a"]
    selected_rows = selected_review_rows(t0066_dir)
    truth_by_session = load_exact_positive_truth(t0066_dir)
    heldout_truth = truth_times_from_labels(Path(args.heldout_labels))

    session_wavs: dict[str, Path] = {row["session_id"]: raw_dir / f"{row['session_id']}.wav" for row in manifest_rows}
    session_meta: dict[str, dict[str, str]] = {row["session_id"]: row for row in manifest_rows}
    for row in selected_rows:
        session_wavs.setdefault(row["session_id"], raw_dir / f"{row['session_id']}.wav")
        session_meta.setdefault(row["session_id"], row)
    if Path(args.heldout_wav).exists() and heldout_truth:
        session_wavs[HELDOUT_SESSION_ID] = Path(args.heldout_wav)
        session_meta[HELDOUT_SESSION_ID] = {
            "session_id": HELDOUT_SESSION_ID,
            "scenario_id": "heldout_c2",
            "scenario_title": "Held-out C2 speech/background",
            "polarity": "positive",
            "expected_racket_contacts": str(len(heldout_truth)),
        }

    audio_cache: dict[str, tuple[np.ndarray, int]] = {}
    candidate_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
    result_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}

    def get_audio(session_id: str) -> tuple[np.ndarray, int]:
        if session_id not in audio_cache:
            audio_cache[session_id] = read_wav(session_wavs[session_id])
        return audio_cache[session_id]

    def get_candidates(gate_id: str, session_id: str) -> list[dict[str, Any]]:
        key = (gate_id, session_id)
        if key not in candidate_cache:
            y, sr = get_audio(session_id)
            candidate_cache[key] = classify_candidates(
                model=model,
                y=y,
                sr=sr,
                session_id=session_id,
                gate_id=gate_id,
            )
        return candidate_cache[key]

    def get_results(policy: PipelineSpec, session_id: str) -> list[dict[str, Any]]:
        key = (policy.pipeline_id, session_id)
        if key not in result_cache:
            result_cache[key] = apply_pipeline(get_candidates(policy.gate_id, session_id), policy)
        return result_cache[key]

    exact_rows, exact_detail_rows = score_selected_exact(
        policies=policies,
        selected_rows=selected_rows,
        truth_by_session=truth_by_session,
        get_results=get_results,
    )
    heldout_rows = score_heldout_exact(policies=policies, truth_ms=heldout_truth, get_results=get_results)
    block_rows = round_a_block_rows(policies=policies, manifest_rows=manifest_rows, get_results=get_results)
    scenario_rows, round_total_rows = summarize_round_a(block_rows)
    recommendation = choose_recommendation(exact_rows, round_total_rows)

    counted_event_rows: list[dict[str, Any]] = []
    for policy in policies:
        for session_id in session_wavs:
            counted_event_rows.extend(compact_event_rows(get_results(policy, session_id), policy, session_meta.get(session_id, {})))

    selected_peak_prediction_rows: list[dict[str, Any]] = []
    for selected in selected_rows:
        sid = selected["session_id"]
        for row in get_candidates("peak_fast_balanced", sid):
            selected_peak_prediction_rows.append(
                {
                    "session_id": sid,
                    "scenario_title": selected.get("scenario_title", ""),
                    "polarity": selected.get("polarity", ""),
                    "time_ms": round(finite_float(row.get("time_ms")), 3),
                    "peak_value": row.get("peak_value", ""),
                    "peak_ratio": row.get("peak_ratio", ""),
                    "model_label": row.get("model_label", ""),
                    "model_confidence": row.get("model_confidence", ""),
                    "prob_racket_bounce": row.get("prob_racket_bounce", ""),
                    "prob_noise": row.get("prob_noise", ""),
                    "prob_floor_bounce": row.get("prob_floor_bounce", ""),
                    "prob_table_bounce": row.get("prob_table_bounce", ""),
                    "nr_bg_rms_db": row.get("nr_bg_rms_db", ""),
                    "nr_bp_peak_ratio": row.get("nr_bp_peak_ratio", ""),
                    "score": row.get("score", ""),
                }
            )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_json": str(Path(args.model_json)),
        "tolerances_ms": list(TOLERANCES_MS),
        "pipelines_evaluated": len(policies),
        "round_a_clips": len(manifest_rows),
        "selected_review_clips": len(selected_rows),
        "t0066_exact_truth": sum(len(values) for values in truth_by_session.values()),
        "heldout_c2_truth": len(heldout_truth),
        "gates": {
            "rms_current": asdict(RMS_CURRENT),
            "peak_fast_balanced": asdict(PEAK_FAST_BALANCED),
        },
        "recommendation": recommendation,
    }

    write_csv(out_dir / "t0069_exact_selected_comparison.csv", exact_rows)
    write_csv(out_dir / "t0069_exact_selected_clip_rows.csv", exact_detail_rows)
    write_csv(out_dir / "t0069_heldout_c2_exact_comparison.csv", heldout_rows)
    write_csv(out_dir / "t0069_round_a_block_replay.csv", block_rows)
    write_csv(out_dir / "t0069_round_a_by_scenario.csv", scenario_rows)
    write_csv(out_dir / "t0069_round_a_pipeline_summary.csv", round_total_rows)
    write_csv(out_dir / "t0069_counted_events.csv", counted_event_rows)
    write_csv(out_dir / "t0069_peak_candidate_predictions_selected.csv", selected_peak_prediction_rows)
    (out_dir / "t0069_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "t0069_peak_fable_hybrid_report.md").write_text(
        render_report(summary, exact_rows, heldout_rows, round_total_rows, scenario_rows),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--t0065-dir", default=str(DEFAULT_T0065_DIR))
    parser.add_argument("--t0066-dir", default=str(DEFAULT_T0066_DIR))
    parser.add_argument("--model-json", default=str(DEFAULT_MODEL_JSON))
    parser.add_argument("--heldout-wav", default=str(DEFAULT_HELDOUT_WAV))
    parser.add_argument("--heldout-labels", default=str(DEFAULT_HELDOUT_LABELS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
