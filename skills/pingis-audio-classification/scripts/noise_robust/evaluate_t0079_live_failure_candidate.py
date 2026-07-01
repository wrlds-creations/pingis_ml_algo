#!/usr/bin/env python3
"""T0079 live-failure candidate replay.

Evaluation-only. This script adds the T0076 live clean/noisy failures to the
reviewed Round A candidate rows, runs session-held classifier replays, and
reports whether a next candidate is worth export. It does not export an app
model, change Collector behavior, build an APK, or install anything.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
for _path in (str(SCRIPT_DIR), str(SCRIPTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from evaluate_t0067_peak_gate_replay import write_csv  # noqa: E402
from evaluate_t0069_peak_fable_hybrid_replay import finite_float, intish  # noqa: E402
from evaluate_t0070_peak_candidate_classifier_veto import (  # noqa: E402
    ModelSpec,
    PolicySpec,
    accepted_after_dedupe,
    add_probabilities,
    fit_estimator,
    model_specs,
    predict_positive_probability,
    read_csv_dicts,
    round_a_replay_rows,
    score_single_exact,
    summarize_round_a,
)
from evaluate_t0072_round_a_reviewed_classifier_replay import heldout_truth_ms, load_manifest  # noqa: E402

DEFAULT_T0072_DIR = ROOT / "data/audio/models/evaluations/t0072_round_a_reviewed_classifier_replay"
DEFAULT_T0078_DIR = ROOT / "data/audio/models/evaluations/t0078_noisy_live_label_analysis"
DEFAULT_T0071_DIR = ROOT / "data/audio/models/evaluations/t0071_round_a_scenario_label_expansion"
DEFAULT_T0063_LABELS = ROOT / "data/audio/models/evaluations/t0063_t0060_heldout_label_ingest/t0063_exact_heldout_labels.csv"
DEFAULT_RAW_DIR = ROOT / "data/audio/raw/t0076_bounce_audio_test_debug"
DEFAULT_CANDIDATE_JSON = ROOT / "apps/collector/src/models/fable_extra_trees_candidate_t0075.json"
DEFAULT_OUT_DIR = ROOT / "data/audio/models/evaluations/t0079_live_failure_candidate"

CLEAN_SESSION_ID = "bounce_audio_test_session_2026-06-30T10-08-41-546Z"
NOISY_SESSION_ID = "bounce_audio_test_session_2026-06-30T09-37-08-304Z"
SAMPLE_RATE_HZ = 22050.0


@dataclass(frozen=True)
class LiveSessionSpec:
    session_id: str
    scenario_id: str
    scenario_title: str
    expected_contacts: int
    review_role: str


LIVE_SESSIONS = {
    CLEAN_SESSION_ID: LiveSessionSpec(
        session_id=CLEAN_SESSION_ID,
        scenario_id="live_clean_normal_bounce",
        scenario_title="T0076 live clean normal bounce",
        expected_contacts=10,
        review_role="live_clean_expected_all_candidates_positive",
    ),
    NOISY_SESSION_ID: LiveSessionSpec(
        session_id=NOISY_SESSION_ID,
        scenario_id="live_noisy_background_bounce",
        scenario_title="T0076 live noisy/background bounce",
        expected_contacts=30,
        review_role="live_noisy_manual_labels",
    ),
}


def load_feature_names(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = payload.get("feature_names")
    if not isinstance(names, list) or not names:
        raise ValueError(f"No feature_names in {path}")
    return [str(name) for name in names]


def extra_model_specs() -> list[ModelSpec]:
    specs = list(model_specs())
    specs.append(
        ModelSpec(
            "extra_leaf2_live",
            "ExtraTrees leaf2 live",
            ExtraTreesClassifier(
                n_estimators=700,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=7901,
                n_jobs=-1,
            ),
        )
    )
    specs.append(
        ModelSpec(
            "rf_depth8_leaf2_live",
            "RF depth8 leaf2 live",
            RandomForestClassifier(
                n_estimators=500,
                max_depth=8,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=7902,
                n_jobs=-1,
            ),
        )
    )
    return specs


def threshold_grid() -> list[float]:
    return [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.525, 0.55, 0.575, 0.60, 0.65, 0.70]


def live_time_ms(candidate: dict[str, Any]) -> float:
    onset_pos = candidate.get("native_onset_pos")
    if isinstance(onset_pos, (int, float)) and np.isfinite(onset_pos):
        return float(onset_pos) / SAMPLE_RATE_HZ * 1000.0
    return finite_float(candidate.get("native_onset_time_ms"), 0.0)


def base_live_row(
    *,
    spec: LiveSessionSpec,
    candidate: dict[str, Any],
    index: int,
    feature_names: list[str],
) -> dict[str, Any]:
    vector = candidate.get("feature_vector") or {}
    if not isinstance(vector, dict):
        vector = {}
    time_ms = live_time_ms(candidate)
    row: dict[str, Any] = {
        "session_id": spec.session_id,
        "scenario_id": spec.scenario_id,
        "scenario_title": spec.scenario_title,
        "polarity": "positive",
        "dataset_role": spec.review_role,
        "candidate_index": index,
        "time_ms": time_ms,
        "onset_sample": intish(candidate.get("native_onset_pos")),
        "expected_contacts": spec.expected_contacts,
        "t0075_probability": finite_float(candidate.get("classifier_probability"), 0.0),
        "t0075_counted": bool(candidate.get("counted")),
        "label": 0,
        "label_source": "live_unlabeled_candidate",
        "nearest_truth_delta_ms": "",
    }
    for name in feature_names:
        row[name] = finite_float(vector.get(name), finite_float(candidate.get(name), 0.0))
    return row


def load_live_clean_rows(raw_dir: Path, feature_names: list[str]) -> list[dict[str, Any]]:
    spec = LIVE_SESSIONS[CLEAN_SESSION_ID]
    payload = json.loads((raw_dir / f"{spec.session_id}.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(payload.get("candidates") or [], start=1):
        row = base_live_row(spec=spec, candidate=candidate, index=index, feature_names=feature_names)
        row["label"] = 1
        row["label_source"] = "reviewed_racket_match"
        row["nearest_truth_delta_ms"] = 0.0
        rows.append(row)
    return rows


def load_noisy_truth_map(t0078_dir: Path) -> dict[int, dict[str, Any]]:
    rows = read_csv_dicts(t0078_dir / "t0078_noisy_candidates_with_truth.csv")
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        out[intish(row.get("trigger_index"))] = row
    return out


def load_live_noisy_rows(raw_dir: Path, t0078_dir: Path, feature_names: list[str]) -> tuple[list[dict[str, Any]], int]:
    spec = LIVE_SESSIONS[NOISY_SESSION_ID]
    payload = json.loads((raw_dir / f"{spec.session_id}.json").read_text(encoding="utf-8"))
    truth_map = load_noisy_truth_map(t0078_dir)
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(payload.get("candidates") or [], start=1):
        row = base_live_row(spec=spec, candidate=candidate, index=index, feature_names=feature_names)
        truth = truth_map.get(index, {})
        is_racket = truth.get("truth_label") == "racket"
        row["label"] = 1 if is_racket else 0
        row["label_source"] = "reviewed_racket_match" if is_racket else "live_background_unmatched_negative"
        row["nearest_truth_delta_ms"] = truth.get("truth_match_delta_ms", "")
        row["truth_index"] = truth.get("truth_index", "")
        rows.append(row)
    matched_positive_count = sum(1 for row in rows if intish(row.get("label")) == 1)
    return rows, matched_positive_count


def load_all_rows(
    *,
    t0072_dir: Path,
    raw_dir: Path,
    t0078_dir: Path,
    feature_names: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    round_rows = read_csv_dicts(t0072_dir / "t0072_candidate_rows_round_a.csv")
    for row in round_rows:
        row["source_group"] = "round_a"
    clean_rows = load_live_clean_rows(raw_dir, feature_names)
    noisy_rows, noisy_matched = load_live_noisy_rows(raw_dir, t0078_dir, feature_names)
    for row in clean_rows:
        row["source_group"] = "live_clean"
    for row in noisy_rows:
        row["source_group"] = "live_noisy"
    live_rows = clean_rows + noisy_rows
    meta = {
        "round_a_rows": len(round_rows),
        "live_clean_rows": len(clean_rows),
        "live_noisy_rows": len(noisy_rows),
        "live_noisy_matched_positive_candidates": noisy_matched,
        "live_noisy_expected_contacts": LIVE_SESSIONS[NOISY_SESSION_ID].expected_contacts,
    }
    return round_rows, live_rows, round_rows + live_rows, meta


def make_loso_predictions(rows: list[dict[str, Any]], spec: ModelSpec, features: list[str]) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    session_ids = sorted({str(row["session_id"]) for row in rows})
    for holdout_sid in session_ids:
        train_rows = [row for row in rows if row["session_id"] != holdout_sid]
        holdout_rows = [row for row in rows if row["session_id"] == holdout_sid]
        if not train_rows or len({intish(row.get("label")) for row in train_rows}) < 2:
            continue
        estimator = fit_estimator(spec, train_rows, features)
        probs = predict_positive_probability(estimator, holdout_rows, features)
        predictions.extend(add_probabilities(holdout_rows, probs, spec.model_id, spec.label, "oof_prob"))
    return sorted(predictions, key=lambda row: (row["session_id"], finite_float(row["time_ms"])))


def score_live_session(
    rows: list[dict[str, Any]],
    *,
    session_id: str,
    prob_key: str,
    policy: PolicySpec,
    expected_contacts: int,
) -> dict[str, Any]:
    session_rows = [row for row in rows if row["session_id"] == session_id]
    counted = accepted_after_dedupe(session_rows, prob_key, policy.threshold, policy.dedupe_ms)
    tp = sum(1 for row in counted if intish(row.get("label")) == 1)
    fp = sum(1 for row in counted if intish(row.get("label")) == 0)
    positives = sum(1 for row in session_rows if intish(row.get("label")) == 1)
    return {
        "pipeline_id": policy.pipeline_id,
        "pipeline_label": policy.pipeline_label,
        "session_id": session_id,
        "expected_contacts": expected_contacts,
        "candidate_positive_labels": positives,
        "candidate_count": len(session_rows),
        "counted": len(counted),
        "tp_candidates": tp,
        "fp_candidates": fp,
        "missed_candidate_positives": max(0, positives - tp),
        "missed_vs_expected_contacts": max(0, expected_contacts - tp),
    }


def scenario_count(block_rows: list[dict[str, Any]], scenario_id: str) -> dict[str, int]:
    row = next((item for item in block_rows if item.get("scenario_id") == scenario_id), {})
    return {
        "counted": intish(row.get("counted")),
        "expected": intish(row.get("expected_contacts")),
    }


def score_policy(
    *,
    policy: PolicySpec,
    predictions: list[dict[str, Any]],
    manifest_rows: list[dict[str, str]],
    heldout_rows: list[dict[str, Any]],
    heldout_truth: list[float],
    all_train_rows: list[dict[str, Any]],
    spec: ModelSpec,
    features: list[str],
) -> dict[str, Any]:
    round_predictions = [row for row in predictions if row.get("source_group") == "round_a"]
    live_predictions = [row for row in predictions if str(row.get("source_group", "")).startswith("live_")]
    round_blocks = round_a_replay_rows(
        rows=round_predictions,
        policy=policy,
        prob_key="oof_prob",
        manifest_rows=manifest_rows,
    )
    _, round_totals = summarize_round_a(round_blocks)
    round_total = next(row for row in round_totals if row["pipeline_id"] == policy.pipeline_id)
    normal = scenario_count(round_blocks, "normal_racket_bounce")
    slow_high = scenario_count(round_blocks, "slow_high_racket_bounce")
    messy = scenario_count(round_blocks, "messy_kid_style_racket_bounce")
    speaking = scenario_count(round_blocks, "racket_bounce_speaking_counting")
    clean = score_live_session(
        live_predictions,
        session_id=CLEAN_SESSION_ID,
        prob_key="oof_prob",
        policy=policy,
        expected_contacts=LIVE_SESSIONS[CLEAN_SESSION_ID].expected_contacts,
    )
    noisy = score_live_session(
        live_predictions,
        session_id=NOISY_SESSION_ID,
        prob_key="oof_prob",
        policy=policy,
        expected_contacts=LIVE_SESSIONS[NOISY_SESSION_ID].expected_contacts,
    )

    final_estimator = fit_estimator(spec, all_train_rows, features)
    heldout_scored = add_probabilities(
        heldout_rows,
        predict_positive_probability(final_estimator, heldout_rows, features),
        spec.model_id,
        spec.label,
        "final_prob",
    )
    heldout = score_single_exact(rows=heldout_scored, policy=policy, prob_key="final_prob", truth_ms=heldout_truth)

    return {
        "pipeline_id": policy.pipeline_id,
        "pipeline_label": policy.pipeline_label,
        "classifier_id": policy.model_id,
        "classifier_label": policy.model_label,
        "threshold": policy.threshold,
        "dedupe_ms": policy.dedupe_ms,
        "round_positive_counted": intish(round_total.get("positive_counted")),
        "round_positive_expected": intish(round_total.get("positive_expected")),
        "round_positive_abs_error": intish(round_total.get("positive_abs_count_error")),
        "round_negative_false_counts": intish(round_total.get("negative_false_counts")),
        "round_talking_false_counts": intish(round_total.get("talking_only_false_counts")),
        "round_handling_false_counts": intish(round_total.get("racket_handling_false_counts")),
        "round_floor_table_false_counts": intish(round_total.get("floor_table_other_false_counts")),
        "round_background_counted": intish(round_total.get("background_counted")),
        "round_background_expected": intish(round_total.get("background_counted_expected")),
        "round_fast_counted": intish(round_total.get("fast_counted")),
        "round_fast_expected": intish(round_total.get("fast_counted_expected")),
        "round_normal_counted": normal["counted"],
        "round_normal_expected": normal["expected"],
        "round_slow_high_counted": slow_high["counted"],
        "round_slow_high_expected": slow_high["expected"],
        "round_messy_counted": messy["counted"],
        "round_messy_expected": messy["expected"],
        "round_speaking_counted": speaking["counted"],
        "round_speaking_expected": speaking["expected"],
        "live_clean_counted": clean["counted"],
        "live_clean_tp": clean["tp_candidates"],
        "live_clean_fp": clean["fp_candidates"],
        "live_clean_expected": clean["expected_contacts"],
        "live_noisy_counted": noisy["counted"],
        "live_noisy_tp": noisy["tp_candidates"],
        "live_noisy_fp": noisy["fp_candidates"],
        "live_noisy_candidate_fn": noisy["missed_candidate_positives"],
        "live_noisy_expected_miss": noisy["missed_vs_expected_contacts"],
        "live_noisy_expected": noisy["expected_contacts"],
        "heldout_c2_counted": heldout["counted"],
        "heldout_c2_truth": heldout["truth"],
        "heldout_c2_tp_140ms": heldout["tp_140ms"],
        "heldout_c2_fp_140ms": heldout["fp_140ms"],
        "heldout_c2_missed_140ms": heldout["missed_140ms"],
        "heldout_c2_precision_140ms": heldout["precision_140ms"],
        "heldout_c2_recall_140ms": heldout["recall_140ms"],
    }


def current_t0075_baseline(t0078_dir: Path, raw_dir: Path) -> dict[str, Any]:
    noisy_summary = json.loads((t0078_dir / "t0078_summary.json").read_text(encoding="utf-8"))
    clean_payload = json.loads((raw_dir / f"{CLEAN_SESSION_ID}.json").read_text(encoding="utf-8"))
    return {
        "pipeline_id": "t0075_live_current_p0575",
        "pipeline_label": "T0075 app p>=0.575 smart220",
        "round_positive_counted": 932,
        "round_positive_expected": 960,
        "round_positive_abs_error": 28,
        "round_negative_false_counts": 0,
        "round_talking_false_counts": 0,
        "round_handling_false_counts": 0,
        "round_floor_table_false_counts": 0,
        "round_background_counted": 171,
        "round_background_expected": 196,
        "live_clean_counted": intish(clean_payload.get("review", {}).get("app_count_at_stop")),
        "live_clean_expected": 10,
        "live_noisy_counted": noisy_summary["current_app_counted_metrics"]["count"],
        "live_noisy_tp": noisy_summary["current_app_counted_metrics"]["tp"],
        "live_noisy_fp": noisy_summary["current_app_counted_metrics"]["fp"],
        "live_noisy_expected": 30,
        "heldout_c2_counted": 23,
        "heldout_c2_truth": 31,
    }


def score_t0075_live_threshold(live_rows: list[dict[str, Any]], policy: PolicySpec) -> tuple[dict[str, Any], dict[str, Any]]:
    clean = score_live_session(
        live_rows,
        session_id=CLEAN_SESSION_ID,
        prob_key="t0075_probability",
        policy=policy,
        expected_contacts=LIVE_SESSIONS[CLEAN_SESSION_ID].expected_contacts,
    )
    noisy = score_live_session(
        live_rows,
        session_id=NOISY_SESSION_ID,
        prob_key="t0075_probability",
        policy=policy,
        expected_contacts=LIVE_SESSIONS[NOISY_SESSION_ID].expected_contacts,
    )
    return clean, noisy


def t0075_threshold_only_baselines(
    *,
    t0072_dir: Path,
    t0071_dir: Path,
    raw_dir: Path,
    t0078_dir: Path,
    heldout_labels: Path,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    """Recompute old T0075 threshold-only policies on the same slices.

    T0074 only swept the safe threshold range. T0076/T0078 showed that lower
    thresholds help live recall, so this table makes those unsafe alternatives
    visible in the T0079 report without retraining.
    """

    manifest_rows = load_manifest(t0071_dir)
    oof_rows = [
        row for row in read_csv_dicts(t0072_dir / "t0072_oof_predictions.csv")
        if row.get("classifier_id") == "extra_leaf4"
    ]
    final_rows = [
        row for row in read_csv_dicts(t0072_dir / "t0072_final_predictions.csv")
        if row.get("classifier_id") == "extra_leaf4"
        and row.get("scenario_id") == "heldout_c2_speaking_background"
    ]
    live_rows = load_live_clean_rows(raw_dir, feature_names)
    noisy_rows, _ = load_live_noisy_rows(raw_dir, t0078_dir, feature_names)
    live_rows.extend(noisy_rows)
    heldout_truth = heldout_truth_ms(heldout_labels)

    rows: list[dict[str, Any]] = []
    for threshold in [0.20, 0.30, 0.50, 0.575]:
        policy = PolicySpec("t0075_threshold_only", "T0075 threshold-only", threshold, 220.0)
        round_blocks = round_a_replay_rows(
            rows=oof_rows,
            policy=policy,
            prob_key="oof_prob",
            manifest_rows=manifest_rows,
        )
        _, round_totals = summarize_round_a(round_blocks)
        round_total = next(row for row in round_totals if row["pipeline_id"] == policy.pipeline_id)
        normal = scenario_count(round_blocks, "normal_racket_bounce")
        slow_high = scenario_count(round_blocks, "slow_high_racket_bounce")
        messy = scenario_count(round_blocks, "messy_kid_style_racket_bounce")
        speaking = scenario_count(round_blocks, "racket_bounce_speaking_counting")
        clean, noisy = score_t0075_live_threshold(live_rows, policy)
        heldout = score_single_exact(rows=final_rows, policy=policy, prob_key="clf_prob", truth_ms=heldout_truth)
        rows.append({
            "pipeline_id": policy.pipeline_id,
            "pipeline_label": f"T0075 threshold-only p>={threshold:g} smart220",
            "threshold": threshold,
            "dedupe_ms": policy.dedupe_ms,
            "round_positive_counted": intish(round_total.get("positive_counted")),
            "round_positive_expected": intish(round_total.get("positive_expected")),
            "round_positive_abs_error": intish(round_total.get("positive_abs_count_error")),
            "round_negative_false_counts": intish(round_total.get("negative_false_counts")),
            "round_talking_false_counts": intish(round_total.get("talking_only_false_counts")),
            "round_handling_false_counts": intish(round_total.get("racket_handling_false_counts")),
            "round_floor_table_false_counts": intish(round_total.get("floor_table_other_false_counts")),
            "round_background_counted": intish(round_total.get("background_counted")),
            "round_background_expected": intish(round_total.get("background_counted_expected")),
            "round_fast_counted": intish(round_total.get("fast_counted")),
            "round_fast_expected": intish(round_total.get("fast_counted_expected")),
            "round_normal_counted": normal["counted"],
            "round_normal_expected": normal["expected"],
            "round_slow_high_counted": slow_high["counted"],
            "round_slow_high_expected": slow_high["expected"],
            "round_messy_counted": messy["counted"],
            "round_messy_expected": messy["expected"],
            "round_speaking_counted": speaking["counted"],
            "round_speaking_expected": speaking["expected"],
            "live_clean_counted": clean["counted"],
            "live_clean_tp": clean["tp_candidates"],
            "live_clean_fp": clean["fp_candidates"],
            "live_clean_expected": clean["expected_contacts"],
            "live_noisy_counted": noisy["counted"],
            "live_noisy_tp": noisy["tp_candidates"],
            "live_noisy_fp": noisy["fp_candidates"],
            "live_noisy_candidate_fn": noisy["missed_candidate_positives"],
            "live_noisy_expected_miss": noisy["missed_vs_expected_contacts"],
            "live_noisy_expected": noisy["expected_contacts"],
            "heldout_c2_counted": heldout["counted"],
            "heldout_c2_truth": heldout["truth"],
            "heldout_c2_tp_140ms": heldout["tp_140ms"],
            "heldout_c2_fp_140ms": heldout["fp_140ms"],
            "heldout_c2_missed_140ms": heldout["missed_140ms"],
            "heldout_c2_precision_140ms": heldout["precision_140ms"],
            "heldout_c2_recall_140ms": heldout["recall_140ms"],
        })
    return rows


def select_recommendation(policy_rows: list[dict[str, Any]]) -> dict[str, Any]:
    exportable = [
        row for row in policy_rows
        if intish(row["round_negative_false_counts"]) <= 5
        and intish(row["round_talking_false_counts"]) == 0
        and intish(row["round_floor_table_false_counts"]) <= 2
        and intish(row["live_clean_tp"]) >= 8
        and intish(row["live_noisy_tp"]) >= 24
        and intish(row["live_noisy_fp"]) <= 3
    ]
    if not exportable:
        best = sorted(
            policy_rows,
            key=lambda row: (
                intish(row["round_negative_false_counts"]),
                -intish(row["live_clean_tp"]) - intish(row["live_noisy_tp"]),
                intish(row["round_positive_abs_error"]),
            ),
        )[0]
        return {
            "recommendation": "do_not_export_yet",
            "reason": "No tested policy recovered the live failures while preserving hard-negative safety.",
            "best_near_miss_pipeline": best["pipeline_id"],
            "best_near_miss_label": best["pipeline_label"],
        }
    best = sorted(
        exportable,
        key=lambda row: (
            -intish(row["live_clean_tp"]) - intish(row["live_noisy_tp"]),
            intish(row["round_negative_false_counts"]),
            intish(row["round_positive_abs_error"]),
        ),
    )[0]
    return {
        "recommendation": "candidate_worth_export_parity_before_apk",
        "pipeline_id": best["pipeline_id"],
        "pipeline_label": best["pipeline_label"],
    }


def md_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], limit: int | None = None) -> str:
    shown = rows[:limit] if limit else rows
    lines = [
        "| " + " | ".join(title for title, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in shown:
        lines.append("| " + " | ".join(str(row.get(key, "")) for _, key in columns) + " |")
    return "\n".join(lines)


def write_report(
    out_dir: Path,
    summary: dict[str, Any],
    policy_rows: list[dict[str, Any]],
    baseline: dict[str, Any],
    threshold_baselines: list[dict[str, Any]],
) -> None:
    top = sorted(
        policy_rows,
        key=lambda row: (
            intish(row["round_negative_false_counts"]),
            -intish(row["live_clean_tp"]) - intish(row["live_noisy_tp"]),
            intish(row["round_positive_abs_error"]),
        ),
    )[:15]
    live_best = sorted(
        policy_rows,
        key=lambda row: (
            -intish(row["live_clean_tp"]) - intish(row["live_noisy_tp"]),
            intish(row["round_negative_false_counts"]),
        ),
    )[:15]
    columns = [
        ("Policy", "pipeline_label"),
        ("Round pos", "round_positive_counted"),
        ("Round err", "round_positive_abs_error"),
        ("Neg FP", "round_negative_false_counts"),
        ("Talk", "round_talking_false_counts"),
        ("Handling", "round_handling_false_counts"),
        ("Floor/table", "round_floor_table_false_counts"),
        ("Clean TP", "live_clean_tp"),
        ("Noisy TP", "live_noisy_tp"),
        ("Noisy FP", "live_noisy_fp"),
        ("C2", "heldout_c2_counted"),
    ]
    report = [
        "# T0079 Live Failure Candidate",
        "",
        f"Generated at: `{summary['generated_at']}`",
        "",
        "## Recommendation",
        "",
        f"- `{summary['recommendation']['recommendation']}`",
        f"- Reason: {summary['recommendation'].get('reason', summary['recommendation'].get('pipeline_label', ''))}",
        "",
        "## Baseline",
        "",
        md_table([baseline], [
            ("Policy", "pipeline_label"),
            ("Round pos", "round_positive_counted"),
            ("Round err", "round_positive_abs_error"),
            ("Neg FP", "round_negative_false_counts"),
            ("Clean", "live_clean_counted"),
            ("Noisy", "live_noisy_counted"),
            ("C2", "heldout_c2_counted"),
        ]),
        "",
        "## T0075 Threshold-Only Comparison",
        "",
        md_table(threshold_baselines, columns),
        "",
        "## Safest Tested Policies",
        "",
        md_table(top, columns),
        "",
        "## Best Live Recall Policies",
        "",
        md_table(live_best, columns),
        "",
        "## Caveat",
        "",
        "The two live sessions are very small. Leave-one-session-out predictions are useful diagnostics, but training on these same failures is not production proof. A candidate still needs app-export parity and fresh Motorola validation before APK promotion.",
    ]
    (out_dir / "t0079_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t0072-dir", type=Path, default=DEFAULT_T0072_DIR)
    parser.add_argument("--t0078-dir", type=Path, default=DEFAULT_T0078_DIR)
    parser.add_argument("--t0071-dir", type=Path, default=DEFAULT_T0071_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--candidate-json", type=Path, default=DEFAULT_CANDIDATE_JSON)
    parser.add_argument("--heldout-labels", type=Path, default=DEFAULT_T0063_LABELS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    feature_names = load_feature_names(args.candidate_json)
    round_rows, live_rows, all_rows, data_meta = load_all_rows(
        t0072_dir=args.t0072_dir,
        raw_dir=args.raw_dir,
        t0078_dir=args.t0078_dir,
        feature_names=feature_names,
    )
    manifest_rows = load_manifest(args.t0071_dir)
    heldout_rows = read_csv_dicts(args.t0072_dir / "t0072_candidate_rows_heldout_c2.csv")
    heldout_truth = heldout_truth_ms(args.heldout_labels)

    all_predictions: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    for spec in extra_model_specs():
        predictions = make_loso_predictions(all_rows, spec, feature_names)
        all_predictions.extend(predictions)
        for threshold in threshold_grid():
            policy = PolicySpec(spec.model_id, spec.label, threshold, 220.0)
            policy_rows.append(
                score_policy(
                    policy=policy,
                    predictions=predictions,
                    manifest_rows=manifest_rows,
                    heldout_rows=heldout_rows,
                    heldout_truth=heldout_truth,
                    all_train_rows=all_rows,
                    spec=spec,
                    features=feature_names,
                )
            )

    baseline = current_t0075_baseline(args.t0078_dir, args.raw_dir)
    threshold_baselines = t0075_threshold_only_baselines(
        t0072_dir=args.t0072_dir,
        t0071_dir=args.t0071_dir,
        raw_dir=args.raw_dir,
        t0078_dir=args.t0078_dir,
        heldout_labels=args.heldout_labels,
        feature_names=feature_names,
    )
    recommendation = select_recommendation(policy_rows)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ticket": "T0079-train-fused-bounce-audio-candidate-from-live-failures",
        **data_meta,
        "models_evaluated": len(extra_model_specs()),
        "policies_evaluated": len(policy_rows),
        "baseline": baseline,
        "t0075_threshold_only_baselines": threshold_baselines,
        "recommendation": recommendation,
    }

    write_csv(args.out_dir / "t0079_loso_predictions.csv", all_predictions)
    write_csv(args.out_dir / "t0079_policy_sweep.csv", policy_rows)
    write_csv(args.out_dir / "t0079_t0075_threshold_only_baselines.csv", threshold_baselines)
    write_csv(args.out_dir / "t0079_live_rows.csv", live_rows)
    (args.out_dir / "t0079_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(args.out_dir, summary, policy_rows, baseline, threshold_baselines)
    print(json.dumps(summary["recommendation"], indent=2))
    print(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()
