#!/usr/bin/env python3
"""T0072 Round A reviewed peak-candidate classifier/veto replay.

Evaluation-only. This script trains local diagnostic classifiers from T0071
reviewed Round A labels, scores them with leave-one-session-out probabilities,
and writes ignored artifacts. It does not export a model, change app runtime
behavior, build an APK, or install anything.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
for _path in (str(SCRIPT_DIR), str(SCRIPTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from evaluate_fable_audio_reliability_t0044 import FableAppModel  # noqa: E402
from evaluate_t0067_peak_gate_replay import match_predictions, read_wav, write_csv  # noqa: E402
from evaluate_t0069_peak_fable_hybrid_replay import (  # noqa: E402
    DEFAULT_HELDOUT_WAV,
    DEFAULT_MODEL_JSON,
    DEFAULT_RAW_DIR,
    HELDOUT_SESSION_ID,
    finite_float,
    intish,
)
from evaluate_t0070_peak_candidate_classifier_veto import (  # noqa: E402
    MATCH_TOLERANCES_MS,
    PolicySpec,
    accepted_after_dedupe,
    add_probabilities,
    build_candidate_rows_for_session,
    feature_name_list,
    fit_estimator,
    make_oof_predictions,
    md_table,
    model_specs,
    predict_positive_probability,
    read_csv_dicts,
    round_a_replay_rows,
    score_exact_sessions,
    score_single_exact,
    summarize_round_a,
    threshold_grid,
)

DEFAULT_T0071_DIR = ROOT / "data/audio/models/evaluations/t0071_round_a_scenario_label_expansion"
DEFAULT_T0063_LABELS = ROOT / "data/audio/models/evaluations/t0063_t0060_heldout_label_ingest/t0063_exact_heldout_labels.csv"
DEFAULT_T0065_DIR = ROOT / "data/audio/models/evaluations/t0065_fable_training_audio_round_a"
DEFAULT_OUT_DIR = ROOT / "data/audio/models/evaluations/t0072_round_a_reviewed_classifier_replay"
LABEL_MATCH_TOLERANCE_MS = 140.0


def truth_by_session(t0071_dir: Path) -> dict[str, list[float]]:
    rows = read_csv_dicts(t0071_dir / "t0071_reviewed_positive_labels.csv")
    out: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        out[row["session_id"]].append(finite_float(row.get("reviewed_time_ms"), 0.0))
    return {sid: sorted(values) for sid, values in out.items()}


def heldout_truth_ms(path: Path) -> list[float]:
    if not path.exists():
        return []
    rows = read_csv_dicts(path)
    values: list[float] = []
    for row in rows:
        if row.get("label") in {"racket", "racket_bounce"} or row.get("review_label") in {"racket", "racket_bounce"}:
            values.append(finite_float(row.get("reviewed_time_s"), finite_float(row.get("time_s"))) * 1000.0)
    return sorted(values)


def load_manifest(t0071_dir: Path) -> list[dict[str, str]]:
    rows = read_csv_dicts(t0071_dir / "t0071_review_manifest.csv")
    return sorted(rows, key=lambda row: (intish(row.get("review_priority")), row.get("started_at", "")))


def build_round_a_rows(
    *,
    model: FableAppModel,
    raw_dir: Path,
    manifest_rows: list[dict[str, str]],
    truth: dict[str, list[float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for meta in manifest_rows:
        sid = meta["session_id"]
        wav_path = raw_dir / f"{sid}.wav"
        y, sr = read_wav(wav_path)
        exact = truth.get(sid)
        role = "round_a_reviewed_positive" if exact else "round_a_expected_zero_hard_negative"
        rows.extend(
            build_candidate_rows_for_session(
                model=model,
                session_id=sid,
                y=y,
                sr=sr,
                meta=meta,
                truth_ms=exact,
                dataset_role=role,
                label_candidates=True,
            )
        )
    return rows


def build_heldout_rows(*, model: FableAppModel, wav_path: Path, truth_ms: list[float]) -> list[dict[str, Any]]:
    if not wav_path.exists() or not truth_ms:
        return []
    y, sr = read_wav(wav_path)
    meta = {
        "session_id": HELDOUT_SESSION_ID,
        "scenario_id": "heldout_c2_speaking_background",
        "scenario_title": "Held-out C2 speaking/background",
        "polarity": "positive",
        "expected_racket_contacts": len(truth_ms),
    }
    return build_candidate_rows_for_session(
        model=model,
        session_id=HELDOUT_SESSION_ID,
        y=y,
        sr=sr,
        meta=meta,
        truth_ms=truth_ms,
        dataset_role="heldout_c2_exact",
        label_candidates=True,
    )


def row_metrics(rows: list[dict[str, Any]], prob_key: str, model_id: str, model_label: str) -> dict[str, Any]:
    labels = np.asarray([intish(row.get("label")) for row in rows], dtype=np.int32)
    probs = np.asarray([finite_float(row.get(prob_key), 0.0) for row in rows], dtype=np.float64)
    out: dict[str, Any] = {
        "classifier_id": model_id,
        "classifier_label": model_label,
        "rows": len(rows),
        "positives": int(labels.sum()),
        "negatives": int(len(labels) - labels.sum()),
    }
    if len(set(labels.tolist())) == 2:
        out["roc_auc"] = float(roc_auc_score(labels, probs))
        out["average_precision"] = float(average_precision_score(labels, probs))
    else:
        out["roc_auc"] = ""
        out["average_precision"] = ""
    for threshold in (0.25, 0.35, 0.50, 0.70, 0.90):
        pred = probs >= threshold
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        suffix = str(threshold).replace(".", "p")
        out[f"row_precision_{suffix}"] = precision
        out[f"row_recall_{suffix}"] = recall
        out[f"row_fp_{suffix}"] = fp
    return out


def baseline_replay_rows(manifest_rows: list[dict[str, str]], mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for meta in manifest_rows:
        expected = intish(meta.get("expected_racket_contacts"))
        if mode == "current_app":
            counted = intish(meta.get("current_app_count"))
            pipeline_id = "current_app_fable_t0065_corrected"
            label = "Current app/Fable block replay"
        elif mode == "peak_only":
            counted = intish(meta.get("peak_candidate_count"))
            pipeline_id = "peak_only_t0071"
            label = "Peak-only candidate count"
        else:
            raise ValueError(mode)
        rows.append(
            {
                "pipeline_id": pipeline_id,
                "pipeline_label": label,
                "classifier_id": mode,
                "classifier_label": label,
                "threshold": "",
                "dedupe_ms": "",
                "session_id": meta["session_id"],
                "scenario_id": meta.get("scenario_id", ""),
                "scenario_title": meta.get("scenario_title", ""),
                "polarity": meta.get("polarity", ""),
                "expected_contacts": expected,
                "candidate_count": intish(meta.get("peak_candidate_count")),
                "counted": counted,
                "count_error": counted - expected,
                "abs_count_error": abs(counted - expected),
                "duration_s": finite_float(meta.get("wav_duration_s"), 0.0),
            }
        )
    return rows


def baseline_totals(manifest_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_blocks: list[dict[str, Any]] = []
    baseline_blocks.extend(baseline_replay_rows(manifest_rows, "current_app"))
    baseline_blocks.extend(baseline_replay_rows(manifest_rows, "peak_only"))
    scenario_rows, total_rows = summarize_round_a(baseline_blocks)
    enrich_total_scenario_fields(scenario_rows, total_rows)
    return baseline_blocks, scenario_rows, total_rows


def enrich_total_scenario_fields(scenario_rows: list[dict[str, Any]], total_rows: list[dict[str, Any]]) -> None:
    scenario_fields = [
        ("normal_racket_bounce", "normal_counted"),
        ("slow_high_racket_bounce", "slow_high_counted"),
        ("fast_racket_bounce", "fast_counted"),
        ("messy_kid_style_racket_bounce", "messy_counted"),
        ("racket_bounce_speaking_counting", "speaking_counted"),
        ("racket_bounce_background_sound", "background_counted"),
        ("talking_only_no_bounce", "talking_only_false_counts"),
        ("racket_handling_no_bounce", "racket_handling_false_counts"),
        ("floor_table_other_impact_no_racket", "floor_table_other_false_counts"),
    ]
    lookup = {(row["pipeline_id"], row["scenario_id"]): row for row in scenario_rows}
    for total in total_rows:
        for scenario_id, field in scenario_fields:
            scenario = lookup.get((total["pipeline_id"], scenario_id))
            total[field] = intish(scenario.get("counted")) if scenario else 0
            total[f"{field}_expected"] = intish(scenario.get("expected_contacts")) if scenario else 0
            total[f"{field}_error"] = intish(scenario.get("count_error")) if scenario else 0


def policies() -> list[PolicySpec]:
    out: list[PolicySpec] = []
    for spec in model_specs():
        for threshold in threshold_grid():
            for dedupe_ms in (220.0, 240.0, 300.0):
                out.append(PolicySpec(spec.model_id, spec.label, threshold, dedupe_ms))
    return out


def score_policy_candidates(
    *,
    candidate_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
    features: list[str],
    manifest_rows: list[dict[str, str]],
    truth: dict[str, list[float]],
    heldout_truth: list[float],
) -> dict[str, list[dict[str, Any]]]:
    all_oof_predictions: list[dict[str, Any]] = []
    all_final_predictions: list[dict[str, Any]] = []
    row_metric_rows: list[dict[str, Any]] = []
    exact_summaries: list[dict[str, Any]] = []
    exact_details: list[dict[str, Any]] = []
    heldout_summaries: list[dict[str, Any]] = []
    oof_round_blocks: list[dict[str, Any]] = []

    manifest_meta = [{**row, "scenario_title": row.get("scenario_title", "")} for row in manifest_rows]
    policy_by_model = defaultdict(list)
    for policy in policies():
        policy_by_model[policy.model_id].append(policy)

    for spec in model_specs():
        oof_rows = make_oof_predictions(candidate_rows, spec, features)
        all_oof_predictions.extend(oof_rows)
        row_metric_rows.append(row_metrics(oof_rows, "oof_prob", spec.model_id, spec.label))

        estimator = fit_estimator(spec, candidate_rows, features)
        final_round_probs = predict_positive_probability(estimator, candidate_rows, features)
        final_round_rows = add_probabilities(candidate_rows, final_round_probs, spec.model_id, spec.label, "clf_prob")
        all_final_predictions.extend(final_round_rows)
        if heldout_rows:
            heldout_probs = predict_positive_probability(estimator, heldout_rows, features)
            heldout_scored = add_probabilities(heldout_rows, heldout_probs, spec.model_id, spec.label, "clf_prob")
            all_final_predictions.extend(heldout_scored)
        else:
            heldout_scored = []

        for policy in policy_by_model[spec.model_id]:
            exact_summary, exact_detail = score_exact_sessions(
                rows=oof_rows,
                policy=policy,
                prob_key="oof_prob",
                truth_by_session=truth,
                selected_rows_meta=manifest_meta,
            )
            exact_summaries.append(exact_summary)
            exact_details.extend(exact_detail)
            oof_round_blocks.extend(
                round_a_replay_rows(
                    rows=oof_rows,
                    policy=policy,
                    prob_key="oof_prob",
                    manifest_rows=manifest_rows,
                )
            )
            if heldout_scored:
                heldout_summaries.append(
                    score_single_exact(
                        rows=heldout_scored,
                        policy=policy,
                        prob_key="clf_prob",
                        truth_ms=heldout_truth,
                    )
                )
    scenario_rows, total_rows = summarize_round_a(oof_round_blocks)
    enrich_total_scenario_fields(scenario_rows, total_rows)
    return {
        "oof_predictions": all_oof_predictions,
        "final_predictions": all_final_predictions,
        "row_metrics": row_metric_rows,
        "exact_summaries": exact_summaries,
        "exact_details": exact_details,
        "heldout_summaries": heldout_summaries,
        "oof_round_blocks": oof_round_blocks,
        "oof_scenario_rows": scenario_rows,
        "oof_total_rows": total_rows,
    }


def get_total(total_rows: list[dict[str, Any]], pipeline_id: str) -> dict[str, Any]:
    return next((row for row in total_rows if row.get("pipeline_id") == pipeline_id), {})


def scenario_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["pipeline_id"], row["scenario_id"]): row for row in rows}


def choose_recommendation(
    *,
    exact_summaries: list[dict[str, Any]],
    heldout_summaries: list[dict[str, Any]],
    oof_total_rows: list[dict[str, Any]],
    oof_scenario_rows: list[dict[str, Any]],
    baseline_total_rows: list[dict[str, Any]],
    baseline_scenario_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    current = get_total(baseline_total_rows, "current_app_fable_t0065_corrected")
    peak = get_total(baseline_total_rows, "peak_only_t0071")
    exact_by_id = {row["pipeline_id"]: row for row in exact_summaries}
    heldout_by_id = {row["pipeline_id"]: row for row in heldout_summaries}
    oof_by_id = {row["pipeline_id"]: row for row in oof_total_rows}
    scen = scenario_lookup(oof_scenario_rows)
    base_scen = scenario_lookup(baseline_scenario_rows)

    def as_int(row: dict[str, Any], key: str) -> int:
        return intish(row.get(key))

    def as_float(row: dict[str, Any], key: str) -> float:
        return finite_float(row.get(key), 0.0)

    def scenario_count(pipeline_id: str, scenario_id: str) -> int:
        return as_int(scen.get((pipeline_id, scenario_id), {}), "counted")

    def base_count(pipeline_id: str, scenario_id: str) -> int:
        return as_int(base_scen.get((pipeline_id, scenario_id), {}), "counted")

    broad_passes: list[str] = []
    strict_passes: list[str] = []
    for row in oof_total_rows:
        pid = row["pipeline_id"]
        if as_int(row, "positive_abs_count_error") <= as_int(current, "positive_abs_count_error") and as_int(row, "negative_false_counts") <= as_int(current, "negative_false_counts"):
            broad_passes.append(pid)
        strict = (
            pid in broad_passes
            and scenario_count(pid, "talking_only_no_bounce") <= base_count("current_app_fable_t0065_corrected", "talking_only_no_bounce")
            and scenario_count(pid, "racket_handling_no_bounce") <= base_count("current_app_fable_t0065_corrected", "racket_handling_no_bounce")
            and scenario_count(pid, "floor_table_other_impact_no_racket") <= base_count("current_app_fable_t0065_corrected", "floor_table_other_impact_no_racket")
            and scenario_count(pid, "racket_bounce_background_sound") >= base_count("current_app_fable_t0065_corrected", "racket_bounce_background_sound")
            and scenario_count(pid, "racket_bounce_speaking_counting") >= base_count("current_app_fable_t0065_corrected", "racket_bounce_speaking_counting")
        )
        if strict:
            strict_passes.append(pid)

    def rank(pid: str) -> tuple[int, int, int, float]:
        row = oof_by_id[pid]
        exact = exact_by_id.get(pid, {})
        return (
            -as_int(row, "negative_false_counts"),
            -as_int(row, "positive_abs_count_error"),
            scenario_count(pid, "racket_bounce_background_sound"),
            as_float(exact, "recall_140ms"),
        )

    best_strict = max(strict_passes, key=rank) if strict_passes else ""
    best_broad = max(broad_passes, key=rank) if broad_passes else ""
    screened_passes = [
        pid
        for pid in strict_passes
        if as_float(heldout_by_id.get(pid, {}), "recall_140ms") >= 0.80
        and as_int(heldout_by_id.get(pid, {}), "fp_140ms") <= 3
    ]

    def screened_rank(pid: str) -> tuple[int, int, float, int]:
        row = oof_by_id[pid]
        heldout = heldout_by_id.get(pid, {})
        return (
            -(as_int(row, "positive_abs_count_error") + as_int(row, "negative_false_counts")),
            -as_int(row, "negative_false_counts"),
            as_float(heldout, "recall_140ms"),
            scenario_count(pid, "racket_bounce_background_sound"),
        )

    best_screened = max(screened_passes, key=screened_rank) if screened_passes else ""
    best_exact = max(
        exact_summaries,
        key=lambda row: (
            as_float(row, "f1_including_negatives_140ms"),
            as_float(row, "recall_140ms"),
            -as_int(row, "selected_expected_zero_false_counts"),
        ),
    )
    best_count = min(
        oof_total_rows,
        key=lambda row: (
            as_int(row, "positive_abs_count_error") + as_int(row, "negative_false_counts"),
            as_int(row, "negative_false_counts"),
            -as_int(row, "background_counted"),
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
    if best_screened:
        recommendation = "offline_candidate_worth_bad_case_review_before_export"
    elif best_strict:
        recommendation = "round_a_pass_but_heldout_blocks_export"
    elif best_broad:
        recommendation = "aggregate_improvement_but_bucket_regression_blocks_export"

    return {
        "recommendation": recommendation,
        "best_strict_pipeline": best_strict,
        "best_strict_label": exact_by_id.get(best_strict, {}).get("pipeline_label", ""),
        "best_screened_pipeline": best_screened,
        "best_screened_label": exact_by_id.get(best_screened, {}).get("pipeline_label", ""),
        "best_broad_pipeline": best_broad,
        "best_broad_label": exact_by_id.get(best_broad, {}).get("pipeline_label", ""),
        "best_exact_f1_pipeline": best_exact["pipeline_id"],
        "best_exact_f1_label": best_exact["pipeline_label"],
        "best_count_pipeline": best_count["pipeline_id"],
        "best_count_label": best_count["pipeline_label"],
        "best_heldout_pipeline": best_heldout.get("pipeline_id", ""),
        "best_heldout_label": best_heldout.get("pipeline_label", ""),
        "strict_pass_count": len(strict_passes),
        "screened_pass_count": len(screened_passes),
        "broad_pass_count": len(broad_passes),
        "current_positive_abs_count_error": as_int(current, "positive_abs_count_error"),
        "current_negative_false_counts": as_int(current, "negative_false_counts"),
        "peak_positive_abs_count_error": as_int(peak, "positive_abs_count_error"),
        "peak_negative_false_counts": as_int(peak, "negative_false_counts"),
    }


def render_report(
    *,
    summary: dict[str, Any],
    recommendation: dict[str, Any],
    baseline_total_rows: list[dict[str, Any]],
    baseline_scenario_rows: list[dict[str, Any]],
    row_metrics_rows: list[dict[str, Any]],
    exact_summaries: list[dict[str, Any]],
    heldout_summaries: list[dict[str, Any]],
    oof_total_rows: list[dict[str, Any]],
    oof_scenario_rows: list[dict[str, Any]],
) -> str:
    top_count = sorted(
        oof_total_rows,
        key=lambda row: (
            intish(row.get("positive_abs_count_error")) + intish(row.get("negative_false_counts")),
            intish(row.get("negative_false_counts")),
            -intish(row.get("background_counted")),
        ),
    )
    top_exact = sorted(
        exact_summaries,
        key=lambda row: (
            -finite_float(row.get("f1_including_negatives_140ms"), 0.0),
            -finite_float(row.get("recall_140ms"), 0.0),
            intish(row.get("selected_expected_zero_false_counts")),
        ),
    )
    top_heldout = sorted(
        heldout_summaries,
        key=lambda row: (
            -finite_float(row.get("recall_140ms"), 0.0),
            intish(row.get("fp_140ms")),
            abs(intish(row.get("counted")) - intish(row.get("truth"))),
        ),
    )
    best_pid = recommendation.get("best_count_pipeline", "")
    best_scenario_rows = [row for row in oof_scenario_rows if row.get("pipeline_id") == best_pid]

    lines = [
        "# T0072 Round A Reviewed Classifier Replay",
        "",
        f"Generated at: `{summary['generated_at']}`",
        "",
        "## Recommendation",
        "",
        f"- Recommendation: `{recommendation['recommendation']}`",
        f"- Best count-balance pipeline: `{recommendation.get('best_count_label')}`",
        f"- Best strict-pass pipeline: `{recommendation.get('best_strict_label') or 'none'}`",
        f"- Best strict + held-out screened pipeline: `{recommendation.get('best_screened_label') or 'none'}`",
        f"- Best held-out C2 pipeline: `{recommendation.get('best_heldout_label') or 'none'}`",
        "",
        "This is still evaluation-only. No model JSON, app runtime, APK, cloud/API, or camera behavior changed.",
        "",
        "## Dataset",
        "",
        f"- Round A candidates: `{summary['round_a_candidate_rows']}`",
        f"- Training/evaluation positives: `{summary['round_a_positive_candidate_rows']}`",
        f"- Training/evaluation negatives: `{summary['round_a_negative_candidate_rows']}`",
        f"- T0071 reviewed positive labels: `{summary['reviewed_positive_labels']}`",
        f"- Held-out C2 candidates: `{summary['heldout_candidate_rows']}`",
        "",
        "## Baselines",
        "",
        *md_table(
            baseline_total_rows,
            [
                "pipeline_label",
                "positive_expected",
                "positive_counted",
                "positive_abs_count_error",
                "negative_false_counts",
                "background_counted",
                "talking_only_false_counts",
                "racket_handling_false_counts",
                "floor_table_other_false_counts",
            ],
            ["Baseline", "Pos Exp", "Pos Count", "Pos Abs Err", "Neg FP", "BG Count", "Talk FP", "Handling FP", "Impact FP"],
        ),
        "",
        "## Row-Level OOF Metrics",
        "",
        *md_table(
            sorted(row_metrics_rows, key=lambda row: -finite_float(row.get("average_precision"), 0.0)),
            ["classifier_label", "rows", "positives", "negatives", "roc_auc", "average_precision", "row_precision_0p5", "row_recall_0p5", "row_fp_0p5"],
            ["Classifier", "Rows", "Pos", "Neg", "ROC AUC", "Avg Prec", "Prec@0.5", "Rec@0.5", "FP@0.5"],
        ),
        "",
        "## Best OOF Exact Event Metrics",
        "",
        *md_table(
            top_exact,
            [
                "pipeline_label",
                "truth",
                "positive_counted",
                "selected_expected_zero_false_counts",
                "tp_140ms",
                "positive_fp_140ms",
                "missed_140ms",
                "precision_including_negatives_140ms",
                "recall_140ms",
                "f1_including_negatives_140ms",
            ],
            ["Pipeline", "Truth", "Pos Count", "Neg FP", "TP", "Pos FP", "Miss", "Prec", "Recall", "F1"],
            limit=12,
        ),
        "",
        "## Best OOF Count Replay",
        "",
        *md_table(
            top_count,
            [
                "pipeline_label",
                "positive_expected",
                "positive_counted",
                "positive_abs_count_error",
                "negative_false_counts",
                "background_counted",
                "speaking_counted",
                "talking_only_false_counts",
                "racket_handling_false_counts",
                "floor_table_other_false_counts",
            ],
            ["Pipeline", "Pos Exp", "Pos Count", "Pos Abs Err", "Neg FP", "BG", "Speaking", "Talk FP", "Handling FP", "Impact FP"],
            limit=12,
        ),
        "",
        "## Best Count Pipeline By Scenario",
        "",
        *md_table(
            best_scenario_rows,
            ["scenario_title", "expected_contacts", "counted", "count_error", "candidate_count"],
            ["Scenario", "Expected", "Counted", "Error", "Candidates"],
        ),
        "",
        "## Held-Out C2 Exact Check",
        "",
        *md_table(
            top_heldout,
            ["pipeline_label", "truth", "candidates", "counted", "tp_140ms", "fp_140ms", "missed_140ms", "precision_140ms", "recall_140ms"],
            ["Pipeline", "Truth", "Cand", "Count", "TP", "FP", "Miss", "Prec", "Recall"],
            limit=12,
        ),
        "",
        "## Baseline Scenario Counts",
        "",
        *md_table(
            baseline_scenario_rows,
            ["pipeline_label", "scenario_title", "expected_contacts", "counted", "count_error", "candidate_count"],
            ["Baseline", "Scenario", "Expected", "Counted", "Error", "Candidates"],
        ),
        "",
        "## Outputs",
        "",
        "- `t0072_candidate_rows_round_a.csv`",
        "- `t0072_candidate_rows_heldout_c2.csv`",
        "- `t0072_oof_predictions.csv`",
        "- `t0072_final_predictions.csv`",
        "- `t0072_row_metrics.csv`",
        "- `t0072_oof_exact_comparison.csv`",
        "- `t0072_oof_round_a_by_scenario.csv`",
        "- `t0072_oof_round_a_pipeline_summary.csv`",
        "- `t0072_heldout_c2_comparison.csv`",
        "- `t0072_baseline_by_scenario.csv`",
        "- `t0072_summary.json`",
    ]
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0071_dir = Path(args.t0071_dir)
    raw_dir = Path(args.raw_dir)
    model = FableAppModel.load(Path(args.model_json))
    manifest_rows = load_manifest(t0071_dir)
    truth = truth_by_session(t0071_dir)
    reviewed_positive_labels = sum(len(values) for values in truth.values())
    candidate_rows = build_round_a_rows(model=model, raw_dir=raw_dir, manifest_rows=manifest_rows, truth=truth)
    features = feature_name_list(model)
    heldout_truth = heldout_truth_ms(Path(args.heldout_labels))
    heldout_rows = build_heldout_rows(model=model, wav_path=Path(args.heldout_wav), truth_ms=heldout_truth)

    baseline_blocks, baseline_scenario_rows, baseline_total_rows = baseline_totals(manifest_rows)
    scored = score_policy_candidates(
        candidate_rows=candidate_rows,
        heldout_rows=heldout_rows,
        features=features,
        manifest_rows=manifest_rows,
        truth=truth,
        heldout_truth=heldout_truth,
    )
    recommendation = choose_recommendation(
        exact_summaries=scored["exact_summaries"],
        heldout_summaries=scored["heldout_summaries"],
        oof_total_rows=scored["oof_total_rows"],
        oof_scenario_rows=scored["oof_scenario_rows"],
        baseline_total_rows=baseline_total_rows,
        baseline_scenario_rows=baseline_scenario_rows,
    )

    exact_sorted = sorted(
        scored["exact_summaries"],
        key=lambda row: (
            -finite_float(row.get("f1_including_negatives_140ms"), 0.0),
            -finite_float(row.get("recall_140ms"), 0.0),
        ),
    )
    total_sorted = sorted(
        scored["oof_total_rows"],
        key=lambda row: (
            intish(row.get("positive_abs_count_error")) + intish(row.get("negative_false_counts")),
            intish(row.get("negative_false_counts")),
            -intish(row.get("background_counted")),
        ),
    )
    heldout_sorted = sorted(
        scored["heldout_summaries"],
        key=lambda row: (
            -finite_float(row.get("recall_140ms"), 0.0),
            intish(row.get("fp_140ms")),
        ),
    )
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ticket": "T0072-round-a-reviewed-classifier-replay",
        "round_a_candidate_rows": len(candidate_rows),
        "round_a_positive_candidate_rows": sum(1 for row in candidate_rows if intish(row.get("label")) == 1),
        "round_a_negative_candidate_rows": sum(1 for row in candidate_rows if intish(row.get("label")) == 0),
        "reviewed_positive_labels": reviewed_positive_labels,
        "heldout_candidate_rows": len(heldout_rows),
        "heldout_truth_labels": len(heldout_truth),
        "policies_evaluated": len(scored["oof_total_rows"]),
        "recommendation": recommendation,
        "best_exact_f1": exact_sorted[0] if exact_sorted else {},
        "best_count_replay": total_sorted[0] if total_sorted else {},
        "best_heldout": heldout_sorted[0] if heldout_sorted else {},
    }

    write_csv(out_dir / "t0072_candidate_rows_round_a.csv", candidate_rows)
    write_csv(out_dir / "t0072_candidate_rows_heldout_c2.csv", heldout_rows)
    write_csv(out_dir / "t0072_oof_predictions.csv", scored["oof_predictions"])
    write_csv(out_dir / "t0072_final_predictions.csv", scored["final_predictions"])
    write_csv(out_dir / "t0072_row_metrics.csv", scored["row_metrics"])
    write_csv(out_dir / "t0072_oof_exact_comparison.csv", scored["exact_summaries"])
    write_csv(out_dir / "t0072_oof_exact_clip_rows.csv", scored["exact_details"])
    write_csv(out_dir / "t0072_oof_round_a_block_replay.csv", scored["oof_round_blocks"])
    write_csv(out_dir / "t0072_oof_round_a_by_scenario.csv", scored["oof_scenario_rows"])
    write_csv(out_dir / "t0072_oof_round_a_pipeline_summary.csv", scored["oof_total_rows"])
    write_csv(out_dir / "t0072_heldout_c2_comparison.csv", scored["heldout_summaries"])
    write_csv(out_dir / "t0072_baseline_block_replay.csv", baseline_blocks)
    write_csv(out_dir / "t0072_baseline_by_scenario.csv", baseline_scenario_rows)
    write_csv(out_dir / "t0072_baseline_summary.csv", baseline_total_rows)
    (out_dir / "t0072_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "t0072_round_a_reviewed_classifier_report.md").write_text(
        render_report(
            summary=summary,
            recommendation=recommendation,
            baseline_total_rows=baseline_total_rows,
            baseline_scenario_rows=baseline_scenario_rows,
            row_metrics_rows=scored["row_metrics"],
            exact_summaries=scored["exact_summaries"],
            heldout_summaries=scored["heldout_summaries"],
            oof_total_rows=scored["oof_total_rows"],
            oof_scenario_rows=scored["oof_scenario_rows"],
        ),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--model-json", default=str(DEFAULT_MODEL_JSON))
    parser.add_argument("--t0071-dir", default=str(DEFAULT_T0071_DIR))
    parser.add_argument("--heldout-wav", default=str(DEFAULT_HELDOUT_WAV))
    parser.add_argument("--heldout-labels", default=str(DEFAULT_T0063_LABELS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    summary = run(parser.parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
