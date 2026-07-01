#!/usr/bin/env python3
"""T0075 export/parity check for the Fable ExtraTrees candidate.

This is export/parity-only. It creates a separate candidate app JSON for the
T0074-selected ExtraTrees policy and validates that app-style tree traversal
matches the Python estimator. It does not replace `fable_audio_model.json`, wire
runtime code, build an APK, or install anything.
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

ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
for _path in (str(SCRIPT_DIR), str(SCRIPTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from evaluate_fable_audio_reliability_t0044 import FableAppModel  # noqa: E402
from evaluate_t0067_peak_gate_replay import write_csv  # noqa: E402
from evaluate_t0069_peak_fable_hybrid_replay import HELDOUT_SESSION_ID, finite_float, intish  # noqa: E402
from evaluate_t0070_peak_candidate_classifier_veto import (  # noqa: E402
    PolicySpec,
    accepted_after_dedupe,
    add_probabilities,
    feature_name_list,
    fit_estimator,
    md_table,
    model_specs,
    predict_positive_probability,
    read_csv_dicts,
    round_a_replay_rows,
    score_exact_sessions,
    score_single_exact,
    summarize_round_a,
)
from evaluate_t0072_round_a_reviewed_classifier_replay import (  # noqa: E402
    DEFAULT_MODEL_JSON,
    DEFAULT_T0063_LABELS,
    DEFAULT_T0071_DIR,
    enrich_total_scenario_fields,
    heldout_truth_ms,
    load_manifest,
    truth_by_session,
)

DEFAULT_T0072_DIR = ROOT / "data/audio/models/evaluations/t0072_round_a_reviewed_classifier_replay"
DEFAULT_T0073_DIR = ROOT / "data/audio/models/evaluations/t0073_fable_candidate_bad_case_export_prep"
DEFAULT_T0074_DIR = ROOT / "data/audio/models/evaluations/t0074_fable_app_style_parity_safety_gate"
DEFAULT_OUT_DIR = ROOT / "data/audio/models/evaluations/t0075_fable_extra_trees_app_export_parity"
DEFAULT_APP_MODEL_OUT = ROOT / "apps/collector/src/models/fable_extra_trees_candidate_t0075.json"

SELECTED_CLASSIFIER_ID = "extra_leaf4"
SELECTED_CLASSIFIER_LABEL = "ExtraTrees leaf4"
SELECTED_THRESHOLD = 0.575
SELECTED_DEDUPE_MS = 220.0
POSITIVE_CLASS = 1
POSITIVE_LABEL = "racket_bounce"
NEGATIVE_LABEL = "not_racket_bounce"
MANUAL_CASE_NEARBY_MS = 140.0


def selected_spec() -> Any:
    for spec in model_specs():
        if spec.model_id == SELECTED_CLASSIFIER_ID:
            return spec
    raise RuntimeError(f"Missing model spec: {SELECTED_CLASSIFIER_ID}")


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


def export_candidate_model(estimator: Any, features: list[str], training_rows: list[dict[str, Any]]) -> dict[str, Any]:
    classes = [int(value) for value in estimator.classes_]
    if POSITIVE_CLASS not in classes:
        raise RuntimeError("Estimator does not expose positive class 1")
    positive_rows = sum(1 for row in training_rows if intish(row.get("label")) == 1)
    total_nodes = sum(int(tree.tree_.node_count) for tree in estimator.estimators_)
    labels = [class_label(value) for value in classes]
    return {
        "metadata": {
            "model_version": "fable_extra_trees_candidate_t0075",
            "source_ticket": "T0075-fable-extra-trees-app-export-parity",
            "selection_source": "T0074 recommended policy",
            "model_type": "extra_trees_binary_peak_candidate",
            "candidate_gate": "peak_fast_balanced",
            "feature_version": "t0072_peak_candidate_features_plus_fable83",
            "selected_threshold": SELECTED_THRESHOLD,
            "smart_dedupe_ms": SELECTED_DEDUPE_MS,
            "positive_class": POSITIVE_CLASS,
            "positive_label": POSITIVE_LABEL,
            "classes": classes,
            "tree_count": len(estimator.estimators_),
            "total_nodes": total_nodes,
            "training_rows": len(training_rows),
            "training_positive_candidates": positive_rows,
            "training_negative_candidates": len(training_rows) - positive_rows,
            "normal_fable_model_unchanged": True,
            "runtime_status": "candidate_export_only_not_wired",
        },
        "labels": labels,
        "feature_names": features,
        "scaler_mean": [0.0 for _ in features],
        "scaler_std": [1.0 for _ in features],
        "trees": [export_tree_full_precision(tree) for tree in estimator.estimators_],
    }


def is_leaf(node: list[float], n_classes: int) -> bool:
    if len(node) != n_classes:
        return len(node) != 4
    total = 0.0
    for value in node:
        if value < 0 or value > 1:
            return False
        total += value
    return abs(total - 1.0) < 0.01


def traverse_tree(tree: list[list[float]], scaled_features: np.ndarray, n_classes: int) -> np.ndarray:
    index = 0
    while not is_leaf(tree[index], n_classes):
        feature_index, threshold, left_child, right_child = tree[index]
        index = int(left_child) if scaled_features[int(feature_index)] <= float(threshold) else int(right_child)
    return np.asarray(tree[index], dtype=np.float64)


def app_style_probabilities(model: dict[str, Any], rows: list[dict[str, Any]]) -> np.ndarray:
    features = model["feature_names"]
    labels = model["labels"]
    n_classes = len(labels)
    positive_index = labels.index(POSITIVE_LABEL)
    scaler_mean = np.asarray(model["scaler_mean"], dtype=np.float64)
    scaler_std = np.asarray(model["scaler_std"], dtype=np.float64)
    scaler_std = np.where(scaler_std == 0, 1.0, scaler_std)

    probabilities: list[float] = []
    for row in rows:
        raw = np.asarray([finite_float(row.get(name), 0.0) for name in features], dtype=np.float64)
        scaled = (raw - scaler_mean) / scaler_std
        prob_sum = np.zeros(n_classes, dtype=np.float64)
        for tree in model["trees"]:
            prob_sum += traverse_tree(tree, scaled, n_classes)
        probabilities.append(float((prob_sum / max(1, len(model["trees"])))[positive_index]))
    return np.asarray(probabilities, dtype=np.float64)


def row_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("session_id", "")), intish(row.get("candidate_index"))


def load_reference_final_predictions(t0072_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows = read_csv_dicts(t0072_dir / "t0072_final_predictions.csv")
    return {
        row_key(row): row
        for row in rows
        if row.get("classifier_id") == SELECTED_CLASSIFIER_ID
    }


def probability_parity_rows(
    *,
    rows: list[dict[str, Any]],
    split: str,
    sklearn_probs: np.ndarray,
    app_probs: np.ndarray,
    reference: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row, sklearn_prob, app_prob in zip(rows, sklearn_probs, app_probs):
        ref_row = reference.get(row_key(row), {})
        ref_prob = finite_float(ref_row.get("clf_prob"), float("nan"))
        sklearn_app_diff = abs(float(sklearn_prob) - float(app_prob))
        ref_app_diff = abs(float(ref_prob) - float(app_prob)) if math.isfinite(ref_prob) else ""
        ref_sklearn_diff = abs(float(ref_prob) - float(sklearn_prob)) if math.isfinite(ref_prob) else ""
        out.append(
            {
                "split": split,
                "session_id": row.get("session_id", ""),
                "scenario_id": row.get("scenario_id", ""),
                "candidate_index": row.get("candidate_index", ""),
                "time_ms": round(finite_float(row.get("time_ms"), 0.0), 3),
                "label": row.get("label", ""),
                "sklearn_prob": float(sklearn_prob),
                "app_prob": float(app_prob),
                "t0072_final_prob": "" if not math.isfinite(ref_prob) else float(ref_prob),
                "abs_diff_sklearn_app": sklearn_app_diff,
                "abs_diff_t0072_app": ref_app_diff,
                "abs_diff_t0072_sklearn": ref_sklearn_diff,
            }
        )
    return out


def max_numeric(rows: list[dict[str, Any]], field: str) -> float:
    values = [finite_float(row.get(field), float("nan")) for row in rows]
    finite = [value for value in values if math.isfinite(value)]
    return max(finite) if finite else float("nan")


def scored_rows(
    rows: list[dict[str, Any]],
    app_probs: np.ndarray,
) -> list[dict[str, Any]]:
    out = add_probabilities(rows, app_probs, SELECTED_CLASSIFIER_ID, SELECTED_CLASSIFIER_LABEL, "app_prob")
    for row in out:
        row["clf_prob"] = row["app_prob"]
        row["oof_prob"] = row["app_prob"]
    return out


def accepted_by_session(rows: list[dict[str, Any]], prob_key: str, threshold: float, dedupe_ms: float) -> dict[str, list[dict[str, Any]]]:
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_session[str(row.get("session_id", ""))].append(row)
    return {
        session_id: accepted_after_dedupe(session_rows, prob_key, threshold, dedupe_ms)
        for session_id, session_rows in by_session.items()
    }


def manual_fp_cases(t0073_dir: Path) -> list[dict[str, str]]:
    path = t0073_dir / "t0073_bad_cases.csv"
    if not path.exists():
        return []
    return [row for row in read_csv_dicts(path) if row.get("case_id", "").startswith("round_fp_")]


def manual_fp_status_rows(
    *,
    cases: list[dict[str, str]],
    round_scored: list[dict[str, Any]],
    policy: PolicySpec,
) -> list[dict[str, Any]]:
    by_id = {row_key(row): row for row in round_scored}
    accepted = accepted_by_session(round_scored, "app_prob", policy.threshold, policy.dedupe_ms)
    accepted_ids = {row_key(row) for values in accepted.values() for row in values}
    out: list[dict[str, Any]] = []
    for case in cases:
        sid = case.get("session_id", "")
        candidate_index = intish(case.get("candidate_index"))
        candidate_time_ms = finite_float(case.get("candidate_time_ms"), 0.0)
        candidate = by_id.get((sid, candidate_index), {})
        session_accepted = accepted.get(sid, [])
        exact_counted = (sid, candidate_index) in accepted_ids
        nearby_rows = [
            row
            for row in session_accepted
            if abs(finite_float(row.get("time_ms"), 0.0) - candidate_time_ms) <= MANUAL_CASE_NEARBY_MS
        ]
        nearest = min(
            nearby_rows,
            key=lambda row: abs(finite_float(row.get("time_ms"), 0.0) - candidate_time_ms),
            default=None,
        )
        still_counted = exact_counted or bool(nearby_rows)
        out.append(
            {
                "pipeline_id": policy.pipeline_id,
                "pipeline_label": policy.pipeline_label,
                "threshold": policy.threshold,
                "dedupe_ms": policy.dedupe_ms,
                "case_id": case.get("case_id", ""),
                "manual_review": case.get("manual_review", ""),
                "manual_note": case.get("manual_note", ""),
                "session_id": sid,
                "scenario_id": case.get("scenario_id", ""),
                "candidate_index": candidate_index,
                "candidate_time_ms": round(candidate_time_ms, 3),
                "old_t0073_candidate_prob": finite_float(case.get("candidate_prob"), 0.0),
                "candidate_app_prob": finite_float(candidate.get("app_prob"), 0.0),
                "exact_candidate_counted": int(exact_counted),
                "nearby_counted_140ms": int(bool(nearby_rows)),
                "case_still_counted": int(still_counted),
                "nearest_counted_candidate_index": intish(nearest.get("candidate_index")) if nearest else "",
                "nearest_counted_time_ms": round(finite_float(nearest.get("time_ms"), 0.0), 3) if nearest else "",
                "nearest_counted_app_prob": round(finite_float(nearest.get("app_prob"), 0.0), 6) if nearest else "",
            }
        )
    return out


def manual_fp_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {
        "manual_fp_cases": len(rows),
        "manual_fp_counted": sum(intish(row.get("case_still_counted")) for row in rows),
        "acceptable_bounce_like_counted": sum(
            intish(row.get("case_still_counted"))
            for row in rows
            if row.get("manual_review") == "acceptable_bounce_like"
        ),
        "rejected_unsafe_counted": sum(
            intish(row.get("case_still_counted"))
            for row in rows
            if row.get("manual_review") == "reject_unsafe_false_positive"
        ),
    }
    for row in rows:
        out[f"{row.get('case_id')}_counted"] = intish(row.get("case_still_counted"))
    return out


def selected_t0074_policy_row(t0074_dir: Path) -> dict[str, Any]:
    path = t0074_dir / "t0074_policy_sweep.csv"
    if not path.exists():
        return {}
    for row in read_csv_dicts(path):
        if row.get("pipeline_id") == "extra_leaf4_thr0p575_smart220":
            return row
    return {}


def render_report(
    *,
    summary: dict[str, Any],
    app_model_out: Path,
    parity_rows: list[dict[str, Any]],
    replay_total: dict[str, Any],
    replay_scenarios: list[dict[str, Any]],
    exact_summary: dict[str, Any],
    heldout_summary: dict[str, Any],
    manual_summary: dict[str, Any],
    manual_rows: list[dict[str, Any]],
    t0074_row: dict[str, Any],
) -> str:
    scenario_focus = [
        row
        for row in replay_scenarios
        if row.get("pipeline_id") == replay_total.get("pipeline_id")
    ]
    worst_parity = sorted(
        parity_rows,
        key=lambda row: finite_float(row.get("abs_diff_sklearn_app"), 0.0),
        reverse=True,
    )[:5]
    lines = [
        "# T0075 Fable ExtraTrees App Export Parity",
        "",
        f"Generated at: `{summary['generated_at']}`",
        "",
        "## Result",
        "",
        f"- Candidate app JSON: `{app_model_out.as_posix()}`",
        f"- Policy: `ExtraTrees leaf4 p>=0.575 smart220`",
        f"- Export status: `{summary['export_status']}`",
        f"- Max Python-vs-app probability diff: `{summary['max_abs_diff_sklearn_app']:.12g}`",
        f"- Max T0072-final-vs-app probability diff: `{summary['max_abs_diff_t0072_app']:.12g}`",
        f"- Active `fable_audio_model.json` changed: `no`",
        "",
        "This is an export/parity ticket only. The candidate is not wired into live counting and no APK was built or installed.",
        "",
        "## Final-Fit Replay",
        "",
        "These rows use one final model trained on all T0071 Round A candidate rows. They prove export behavior, not new generalization.",
        "",
        *md_table(
            [replay_total],
            [
                "positive_expected",
                "positive_counted",
                "positive_abs_count_error",
                "negative_false_counts",
                "normal_counted",
                "slow_high_counted",
                "fast_counted",
                "messy_counted",
                "speaking_counted",
                "background_counted",
            ],
            [
                "Pos Exp",
                "Pos Count",
                "Pos Abs Err",
                "Neg FP",
                "Normal",
                "Slow/high",
                "Fast",
                "Messy",
                "Speaking",
                "BG",
            ],
        ),
        "",
        "## Safety Baseline",
        "",
        "T0074 remains the real safety baseline because it used out-of-fold Round A probabilities plus held-out C2.",
        "",
        *md_table(
            [t0074_row] if t0074_row else [],
            [
                "positive_expected",
                "positive_counted",
                "positive_abs_count_error",
                "negative_false_counts",
                "rejected_unsafe_counted",
                "heldout_counted",
                "heldout_missed_140ms",
            ],
            [
                "Pos Exp",
                "Pos Count",
                "Pos Abs Err",
                "Neg FP",
                "Rejected FP Counted",
                "C2 Count",
                "C2 Miss",
            ],
        ),
        "",
        "## Exact And Manual Checks",
        "",
        *md_table(
            [
                {
                    **exact_summary,
                    **{
                        "heldout_counted": heldout_summary.get("counted", ""),
                        "heldout_truth": heldout_summary.get("truth", ""),
                        "heldout_missed_140ms": heldout_summary.get("missed_140ms", ""),
                        "manual_fp_counted": manual_summary.get("manual_fp_counted", ""),
                        "rejected_unsafe_counted": manual_summary.get("rejected_unsafe_counted", ""),
                    },
                }
            ],
            [
                "truth",
                "positive_counted",
                "tp_140ms",
                "positive_fp_140ms",
                "missed_140ms",
                "recall_140ms",
                "heldout_truth",
                "heldout_counted",
                "heldout_missed_140ms",
                "manual_fp_counted",
                "rejected_unsafe_counted",
            ],
            [
                "Truth",
                "Counted",
                "TP",
                "Pos FP",
                "Missed",
                "Recall",
                "C2 Truth",
                "C2 Count",
                "C2 Miss",
                "Manual FP",
                "Rejected FP",
            ],
        ),
        "",
        "## Scenario Replay",
        "",
        *md_table(
            scenario_focus,
            ["scenario_id", "expected_contacts", "counted", "count_error", "candidate_count"],
            ["Scenario", "Expected", "Counted", "Error", "Candidates"],
        ),
        "",
        "## Manual False-Positive Cases",
        "",
        *md_table(
            manual_rows,
            ["case_id", "manual_review", "old_t0073_candidate_prob", "candidate_app_prob", "case_still_counted"],
            ["Case", "Review", "Old Prob", "App Prob", "Still Counted"],
        ),
        "",
        "## Worst Parity Rows",
        "",
        *md_table(
            worst_parity,
            ["split", "session_id", "candidate_index", "sklearn_prob", "app_prob", "abs_diff_sklearn_app"],
            ["Split", "Session", "Idx", "Python", "App", "Diff"],
        ),
        "",
        "## Outputs",
        "",
        f"- Summary JSON: `{(DEFAULT_OUT_DIR / 't0075_summary.json').as_posix()}`",
        f"- Parity rows: `{(DEFAULT_OUT_DIR / 't0075_probability_parity_rows.csv').as_posix()}`",
        f"- Final-fit block replay: `{(DEFAULT_OUT_DIR / 't0075_final_fit_round_a_block_replay.csv').as_posix()}`",
        f"- Scenario replay: `{(DEFAULT_OUT_DIR / 't0075_final_fit_round_a_by_scenario.csv').as_posix()}`",
        f"- Manual FP status: `{(DEFAULT_OUT_DIR / 't0075_manual_fp_status.csv').as_posix()}`",
        "",
        "## Next",
        "",
        "If this parity result is accepted, the next ticket should be a tiny guarded app integration that can switch the live Fable counter to this candidate on-device, followed immediately by live testing on the Motorola.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export and parity-check the T0075 Fable ExtraTrees candidate.")
    parser.add_argument("--t0072-dir", default=str(DEFAULT_T0072_DIR))
    parser.add_argument("--t0073-dir", default=str(DEFAULT_T0073_DIR))
    parser.add_argument("--t0074-dir", default=str(DEFAULT_T0074_DIR))
    parser.add_argument("--t0071-dir", default=str(DEFAULT_T0071_DIR))
    parser.add_argument("--heldout-labels", default=str(DEFAULT_T0063_LABELS))
    parser.add_argument("--model-json", default=str(DEFAULT_MODEL_JSON))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--app-model-out", default=str(DEFAULT_APP_MODEL_OUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0072_dir = Path(args.t0072_dir)
    t0073_dir = Path(args.t0073_dir)
    t0074_dir = Path(args.t0074_dir)
    t0071_dir = Path(args.t0071_dir)
    out_dir = Path(args.out_dir)
    app_model_out = Path(args.app_model_out)

    round_rows = read_csv_dicts(t0072_dir / "t0072_candidate_rows_round_a.csv")
    heldout_rows = read_csv_dicts(t0072_dir / "t0072_candidate_rows_heldout_c2.csv")
    manifest_rows = load_manifest(t0071_dir)
    truth = truth_by_session(t0071_dir)
    heldout_truth = heldout_truth_ms(Path(args.heldout_labels))

    fable_model = FableAppModel.load(Path(args.model_json))
    features = feature_name_list(fable_model)
    spec = selected_spec()
    estimator = fit_estimator(spec, round_rows, features)
    exported = export_candidate_model(estimator, features, round_rows)

    app_model_out.parent.mkdir(parents=True, exist_ok=True)
    app_model_out.write_text(json.dumps(exported, separators=(",", ":")), encoding="utf-8")

    round_sklearn_probs = predict_positive_probability(estimator, round_rows, features)
    heldout_sklearn_probs = predict_positive_probability(estimator, heldout_rows, features)
    round_app_probs = app_style_probabilities(exported, round_rows)
    heldout_app_probs = app_style_probabilities(exported, heldout_rows)

    reference = load_reference_final_predictions(t0072_dir)
    parity_rows = probability_parity_rows(
        rows=round_rows,
        split="round_a_final_fit",
        sklearn_probs=round_sklearn_probs,
        app_probs=round_app_probs,
        reference=reference,
    )
    parity_rows.extend(
        probability_parity_rows(
            rows=heldout_rows,
            split="heldout_c2_final_fit",
            sklearn_probs=heldout_sklearn_probs,
            app_probs=heldout_app_probs,
            reference=reference,
        )
    )

    round_scored = scored_rows(round_rows, round_app_probs)
    heldout_scored = scored_rows(heldout_rows, heldout_app_probs)
    policy = PolicySpec(SELECTED_CLASSIFIER_ID, SELECTED_CLASSIFIER_LABEL, SELECTED_THRESHOLD, SELECTED_DEDUPE_MS)

    round_blocks = round_a_replay_rows(
        rows=round_scored,
        policy=policy,
        prob_key="app_prob",
        manifest_rows=manifest_rows,
    )
    scenario_rows, total_rows = summarize_round_a(round_blocks)
    enrich_total_scenario_fields(scenario_rows, total_rows)
    replay_total = next(row for row in total_rows if row["pipeline_id"] == policy.pipeline_id)

    exact_summary, exact_details = score_exact_sessions(
        rows=round_scored,
        policy=policy,
        prob_key="app_prob",
        truth_by_session=truth,
        selected_rows_meta=manifest_rows,
    )
    heldout_summary = score_single_exact(
        rows=heldout_scored,
        policy=policy,
        prob_key="app_prob",
        truth_ms=heldout_truth,
    )
    manual_rows = manual_fp_status_rows(
        cases=manual_fp_cases(t0073_dir),
        round_scored=round_scored,
        policy=policy,
    )
    manual_summary = manual_fp_summary(manual_rows)
    t0074_row = selected_t0074_policy_row(t0074_dir)

    max_sklearn_app = max_numeric(parity_rows, "abs_diff_sklearn_app")
    max_t0072_app = max_numeric(parity_rows, "abs_diff_t0072_app")
    max_t0072_sklearn = max_numeric(parity_rows, "abs_diff_t0072_sklearn")
    export_status = "pass" if max_sklearn_app <= 1e-12 else "check_probability_drift"

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ticket": "T0075-fable-extra-trees-app-export-parity",
        "app_model_out": str(app_model_out),
        "export_status": export_status,
        "classifier_id": SELECTED_CLASSIFIER_ID,
        "classifier_label": SELECTED_CLASSIFIER_LABEL,
        "threshold": SELECTED_THRESHOLD,
        "dedupe_ms": SELECTED_DEDUPE_MS,
        "round_a_rows": len(round_rows),
        "heldout_rows": len(heldout_rows),
        "features": len(features),
        "tree_count": len(estimator.estimators_),
        "total_nodes": exported["metadata"]["total_nodes"],
        "max_abs_diff_sklearn_app": max_sklearn_app,
        "max_abs_diff_t0072_app": max_t0072_app,
        "max_abs_diff_t0072_sklearn": max_t0072_sklearn,
        "final_fit_replay": replay_total,
        "exact_summary": exact_summary,
        "heldout_summary": heldout_summary,
        "manual_fp_summary": manual_summary,
        "t0074_oof_safety_baseline": t0074_row,
        "active_fable_model_changed": False,
        "live_runtime_changed": False,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "t0075_probability_parity_rows.csv", parity_rows)
    write_csv(out_dir / "t0075_final_fit_round_a_block_replay.csv", round_blocks)
    write_csv(out_dir / "t0075_final_fit_round_a_by_scenario.csv", scenario_rows)
    write_csv(out_dir / "t0075_final_fit_round_a_total.csv", total_rows)
    write_csv(out_dir / "t0075_exact_details.csv", exact_details)
    write_csv(out_dir / "t0075_manual_fp_status.csv", manual_rows)
    (out_dir / "t0075_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "t0075_report.md").write_text(
        render_report(
            summary=summary,
            app_model_out=app_model_out,
            parity_rows=parity_rows,
            replay_total=replay_total,
            replay_scenarios=scenario_rows,
            exact_summary=exact_summary,
            heldout_summary=heldout_summary,
            manual_summary=manual_summary,
            manual_rows=manual_rows,
            t0074_row=t0074_row,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
