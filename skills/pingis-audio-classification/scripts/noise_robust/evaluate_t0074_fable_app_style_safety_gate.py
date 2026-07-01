#!/usr/bin/env python3
"""T0074 app-style threshold/safety replay for the T0072 Fable candidate.

Evaluation-only. This script reads existing T0072 probability outputs and the
T0073 manual bad-case review, then sweeps app-style probability thresholds plus
smart dedupe. It does not train a new model, export app JSON, change runtime
behavior, build an APK, or install anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
for _path in (str(SCRIPT_DIR), str(SCRIPTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from evaluate_t0067_peak_gate_replay import match_predictions, write_csv  # noqa: E402
from evaluate_t0069_peak_fable_hybrid_replay import HELDOUT_SESSION_ID, finite_float, intish  # noqa: E402
from evaluate_t0070_peak_candidate_classifier_veto import (  # noqa: E402
    PolicySpec,
    accepted_after_dedupe,
    md_table,
    read_csv_dicts,
    summarize_round_a,
)
from evaluate_t0072_round_a_reviewed_classifier_replay import (  # noqa: E402
    DEFAULT_T0063_LABELS,
    DEFAULT_T0071_DIR,
    enrich_total_scenario_fields,
    heldout_truth_ms,
    load_manifest,
    truth_by_session,
)

DEFAULT_T0072_DIR = ROOT / "data/audio/models/evaluations/t0072_round_a_reviewed_classifier_replay"
DEFAULT_T0073_DIR = ROOT / "data/audio/models/evaluations/t0073_fable_candidate_bad_case_export_prep"
DEFAULT_OUT_DIR = ROOT / "data/audio/models/evaluations/t0074_fable_app_style_parity_safety_gate"

SELECTED_CLASSIFIER_ID = "extra_leaf4"
SELECTED_CLASSIFIER_LABEL = "ExtraTrees leaf4"
ROUND_PROB_KEY = "oof_prob"
HELDOUT_PROB_KEY = "clf_prob"
MATCH_TOLERANCES_MS = (140.0, 250.0)
MANUAL_CASE_NEARBY_MS = 140.0


def threshold_grid() -> list[float]:
    return [
        0.50,
        0.525,
        0.55,
        0.56,
        0.565,
        0.57,
        0.573,
        0.575,
        0.58,
        0.59,
        0.60,
        0.625,
        0.65,
        0.675,
        0.70,
        0.725,
        0.75,
    ]


def dedupe_grid() -> list[float]:
    return [220.0, 240.0, 300.0]


def selected_oof_rows(t0072_dir: Path) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in read_csv_dicts(t0072_dir / "t0072_oof_predictions.csv")
        if row.get("classifier_id") == SELECTED_CLASSIFIER_ID
    ]


def selected_heldout_rows(t0072_dir: Path) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in read_csv_dicts(t0072_dir / "t0072_final_predictions.csv")
        if row.get("classifier_id") == SELECTED_CLASSIFIER_ID and row.get("session_id") == HELDOUT_SESSION_ID
    ]


def rows_by_session(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row.get("session_id", ""))].append(row)
    for values in out.values():
        values.sort(key=lambda row: finite_float(row.get("time_ms"), 0.0))
    return dict(out)


def accepted_by_session(rows: list[dict[str, Any]], prob_key: str, threshold: float, dedupe_ms: float) -> dict[str, list[dict[str, Any]]]:
    return {
        sid: accepted_after_dedupe(session_rows, prob_key, threshold, dedupe_ms)
        for sid, session_rows in rows_by_session(rows).items()
    }


def accepted_id(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row.get("session_id", "")), intish(row.get("candidate_index")))


def round_a_block_rows(
    *,
    rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, str]],
    policy: PolicySpec,
    prob_key: str,
) -> list[dict[str, Any]]:
    by_sid = rows_by_session(rows)
    out: list[dict[str, Any]] = []
    for meta in manifest_rows:
        sid = meta["session_id"]
        session_rows = by_sid.get(sid, [])
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


def exact_metrics(
    *,
    rows: list[dict[str, Any]],
    truth: dict[str, list[float]],
    policy: PolicySpec,
    prob_key: str,
) -> dict[str, Any]:
    by_sid = rows_by_session(rows)
    out: dict[str, Any] = {
        "exact_truth": sum(len(values) for values in truth.values()),
        "exact_positive_counted": 0,
    }
    totals = {tol: {"tp": 0, "fp": 0, "missed": 0} for tol in MATCH_TOLERANCES_MS}
    for sid, truth_ms in truth.items():
        counted = accepted_after_dedupe(by_sid.get(sid, []), prob_key, policy.threshold, policy.dedupe_ms)
        pred_ms = [finite_float(row.get("time_ms"), 0.0) for row in counted]
        out["exact_positive_counted"] += len(pred_ms)
        for tol in MATCH_TOLERANCES_MS:
            matched = match_predictions(pred_ms, truth_ms, tol)
            totals[tol]["tp"] += matched["tp"]
            totals[tol]["fp"] += matched["fp"]
            totals[tol]["missed"] += matched["missed"]
    for tol in MATCH_TOLERANCES_MS:
        key = int(tol)
        tp = totals[tol]["tp"]
        fp = totals[tol]["fp"]
        missed = totals[tol]["missed"]
        out[f"exact_tp_{key}ms"] = tp
        out[f"exact_positive_fp_{key}ms"] = fp
        out[f"exact_missed_{key}ms"] = missed
        out[f"exact_recall_{key}ms"] = tp / out["exact_truth"] if out["exact_truth"] else 0.0
    return out


def heldout_metrics(
    *,
    rows: list[dict[str, Any]],
    truth_ms: list[float],
    policy: PolicySpec,
    prob_key: str,
) -> dict[str, Any]:
    counted = accepted_after_dedupe(rows, prob_key, policy.threshold, policy.dedupe_ms)
    pred_ms = [finite_float(row.get("time_ms"), 0.0) for row in counted]
    out: dict[str, Any] = {
        "heldout_truth": len(truth_ms),
        "heldout_candidates": len(rows),
        "heldout_counted": len(pred_ms),
    }
    for tol in MATCH_TOLERANCES_MS:
        key = int(tol)
        matched = match_predictions(pred_ms, truth_ms, tol)
        precision = matched["tp"] / (matched["tp"] + matched["fp"]) if (matched["tp"] + matched["fp"]) else 0.0
        recall = matched["tp"] / len(truth_ms) if truth_ms else 0.0
        out[f"heldout_tp_{key}ms"] = matched["tp"]
        out[f"heldout_fp_{key}ms"] = matched["fp"]
        out[f"heldout_missed_{key}ms"] = matched["missed"]
        out[f"heldout_precision_{key}ms"] = precision
        out[f"heldout_recall_{key}ms"] = recall
    return out


def manual_fp_cases(t0073_dir: Path) -> list[dict[str, str]]:
    rows = read_csv_dicts(t0073_dir / "t0073_bad_cases.csv")
    return [row for row in rows if row.get("case_id", "").startswith("round_fp_")]


def manual_fp_status_rows(
    *,
    policy: PolicySpec,
    accepted: dict[str, list[dict[str, Any]]],
    cases: list[dict[str, str]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    accepted_ids = {accepted_id(row) for rows in accepted.values() for row in rows}
    for case in cases:
        sid = case.get("session_id", "")
        candidate_index = intish(case.get("candidate_index"))
        candidate_time_ms = finite_float(case.get("candidate_time_ms"), 0.0)
        session_accepted = accepted.get(sid, [])
        exact_counted = (sid, candidate_index) in accepted_ids
        nearby_rows = [
            row
            for row in session_accepted
            if abs(finite_float(row.get("time_ms"), 0.0) - candidate_time_ms) <= MANUAL_CASE_NEARBY_MS
        ]
        nearby_counted = bool(nearby_rows)
        nearest = min(
            nearby_rows,
            key=lambda row: abs(finite_float(row.get("time_ms"), 0.0) - candidate_time_ms),
            default=None,
        )
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
                "candidate_time_ms": candidate_time_ms,
                "candidate_prob": finite_float(case.get("candidate_prob"), 0.0),
                "exact_candidate_counted": int(exact_counted),
                "nearby_counted_140ms": int(nearby_counted),
                "case_still_counted": int(exact_counted or nearby_counted),
                "nearest_counted_candidate_index": intish(nearest.get("candidate_index")) if nearest else "",
                "nearest_counted_time_ms": round(finite_float(nearest.get("time_ms"), 0.0), 3) if nearest else "",
                "nearest_counted_prob": round(finite_float(nearest.get(ROUND_PROB_KEY), 0.0), 6) if nearest else "",
            }
        )
    return out


def manual_fp_summary(status_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {
        "manual_fp_counted": sum(intish(row.get("case_still_counted")) for row in status_rows),
        "acceptable_bounce_like_counted": sum(
            intish(row.get("case_still_counted"))
            for row in status_rows
            if row.get("manual_review") == "acceptable_bounce_like"
        ),
        "rejected_unsafe_counted": sum(
            intish(row.get("case_still_counted"))
            for row in status_rows
            if row.get("manual_review") == "reject_unsafe_false_positive"
        ),
    }
    for row in status_rows:
        out[f"{row.get('case_id')}_counted"] = intish(row.get("case_still_counted"))
    return out


def enrich_policy_totals(policy_rows: list[dict[str, Any]]) -> None:
    for row in policy_rows:
        pos_tp = intish(row.get("exact_tp_140ms"))
        pos_fp = intish(row.get("exact_positive_fp_140ms"))
        neg_fp = intish(row.get("negative_false_counts"))
        precision = pos_tp / (pos_tp + pos_fp + neg_fp) if (pos_tp + pos_fp + neg_fp) else 0.0
        row["precision_including_negatives_140ms"] = precision
        row["total_error_plus_negatives"] = intish(row.get("positive_abs_count_error")) + neg_fp


def choose_recommendation(policy_rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = next(
        (
            row
            for row in policy_rows
            if abs(finite_float(row.get("threshold"), 0.0) - 0.5) < 1e-9
            and abs(finite_float(row.get("dedupe_ms"), 0.0) - 220.0) < 1e-9
        ),
        {},
    )
    safe_220 = [
        row
        for row in policy_rows
        if intish(row.get("rejected_unsafe_counted")) == 0 and abs(finite_float(row.get("dedupe_ms"), 0.0) - 220.0) < 1e-9
    ]
    safe_all = [row for row in policy_rows if intish(row.get("rejected_unsafe_counted")) == 0]

    def rank(row: dict[str, Any]) -> tuple[int, int, int, int, float]:
        threshold = finite_float(row.get("threshold"), 0.0)
        return (
            intish(row.get("positive_abs_count_error")),
            intish(row.get("negative_false_counts")),
            intish(row.get("heldout_missed_140ms")),
            -intish(row.get("background_counted")),
            abs(threshold - 0.575),
        )

    best_safe = min(safe_220 or safe_all, key=rank) if (safe_220 or safe_all) else {}
    positive_loss_vs_selected = intish(best_safe.get("positive_abs_count_error")) - intish(selected.get("positive_abs_count_error"))
    heldout_loss_vs_selected = intish(best_safe.get("heldout_missed_140ms")) - intish(selected.get("heldout_missed_140ms"))
    if not best_safe:
        recommendation = "no_threshold_policy_rejects_manual_false_positives"
    elif positive_loss_vs_selected <= 10 and heldout_loss_vs_selected <= 5:
        recommendation = "promote_threshold_candidate_to_app_style_export_parity"
    else:
        recommendation = "threshold_rejects_manual_false_positives_but_tradeoff_needs_review"
    return {
        "recommendation": recommendation,
        "selected_original_pipeline_id": selected.get("pipeline_id", ""),
        "selected_original_label": selected.get("pipeline_label", ""),
        "recommended_pipeline_id": best_safe.get("pipeline_id", ""),
        "recommended_label": best_safe.get("pipeline_label", ""),
        "recommended_threshold": best_safe.get("threshold", ""),
        "recommended_dedupe_ms": best_safe.get("dedupe_ms", ""),
        "positive_abs_error_selected": intish(selected.get("positive_abs_count_error")),
        "positive_abs_error_recommended": intish(best_safe.get("positive_abs_count_error")),
        "positive_abs_error_delta": positive_loss_vs_selected if best_safe else "",
        "negative_false_counts_selected": intish(selected.get("negative_false_counts")),
        "negative_false_counts_recommended": intish(best_safe.get("negative_false_counts")),
        "rejected_unsafe_counted_selected": intish(selected.get("rejected_unsafe_counted")),
        "rejected_unsafe_counted_recommended": intish(best_safe.get("rejected_unsafe_counted")),
        "heldout_counted_selected": intish(selected.get("heldout_counted")),
        "heldout_counted_recommended": intish(best_safe.get("heldout_counted")),
        "heldout_missed_delta": heldout_loss_vs_selected if best_safe else "",
    }


def scenario_rows_for_policy(scenario_rows: list[dict[str, Any]], pipeline_id: str) -> list[dict[str, Any]]:
    return [row for row in scenario_rows if row.get("pipeline_id") == pipeline_id]


def render_report(
    *,
    summary: dict[str, Any],
    recommendation: dict[str, Any],
    policy_rows: list[dict[str, Any]],
    scenario_rows: list[dict[str, Any]],
    manual_rows: list[dict[str, Any]],
) -> str:
    selected_id = recommendation.get("selected_original_pipeline_id", "")
    recommended_id = recommendation.get("recommended_pipeline_id", "")
    focus_ids = {selected_id, recommended_id}
    focus_rows = [row for row in policy_rows if row.get("pipeline_id") in focus_ids]
    threshold_220 = [row for row in policy_rows if abs(finite_float(row.get("dedupe_ms"), 0.0) - 220.0) < 1e-9]
    safe_rows = sorted(
        [row for row in policy_rows if intish(row.get("rejected_unsafe_counted")) == 0],
        key=lambda row: (
            intish(row.get("positive_abs_count_error")),
            intish(row.get("negative_false_counts")),
            intish(row.get("heldout_missed_140ms")),
            abs(finite_float(row.get("threshold"), 0.0) - 0.575),
        ),
    )
    selected_manual = [row for row in manual_rows if row.get("pipeline_id") in focus_ids]
    recommended_scenarios = scenario_rows_for_policy(scenario_rows, recommended_id)

    lines = [
        "# T0074 Fable App-Style Safety Gate Replay",
        "",
        f"Generated at: `{summary['generated_at']}`",
        "",
        "## Recommendation",
        "",
        f"- Recommendation: `{recommendation['recommendation']}`",
        f"- Original selected policy: `{recommendation.get('selected_original_label')}`",
        f"- Recommended policy: `{recommendation.get('recommended_label') or 'none'}`",
        f"- Positive abs error: `{recommendation.get('positive_abs_error_selected')}` -> `{recommendation.get('positive_abs_error_recommended')}`",
        f"- Hard-negative false counts: `{recommendation.get('negative_false_counts_selected')}` -> `{recommendation.get('negative_false_counts_recommended')}`",
        f"- Rejected manual false positives counted: `{recommendation.get('rejected_unsafe_counted_selected')}` -> `{recommendation.get('rejected_unsafe_counted_recommended')}`",
        f"- Held-out C2 counted: `{recommendation.get('heldout_counted_selected')}` -> `{recommendation.get('heldout_counted_recommended')}`",
        "",
        "This is evaluation-only. No model JSON, app runtime, APK, camera, cloud/API, or AWS behavior changed.",
        "",
        "## Original vs Recommended",
        "",
        *md_table(
            focus_rows,
            [
                "pipeline_label",
                "positive_expected",
                "positive_counted",
                "positive_abs_count_error",
                "negative_false_counts",
                "background_counted",
                "speaking_counted",
                "fast_counted",
                "talking_only_false_counts",
                "racket_handling_false_counts",
                "floor_table_other_false_counts",
                "rejected_unsafe_counted",
                "heldout_counted",
                "heldout_missed_140ms",
            ],
            [
                "Policy",
                "Pos Exp",
                "Pos Count",
                "Pos Abs Err",
                "Neg FP",
                "BG",
                "Speaking",
                "Fast",
                "Talk FP",
                "Handling FP",
                "Impact FP",
                "Rejected FP Counted",
                "C2 Count",
                "C2 Miss",
            ],
        ),
        "",
        "## 220 ms Threshold Sweep",
        "",
        *md_table(
            threshold_220,
            [
                "threshold",
                "positive_counted",
                "positive_abs_count_error",
                "negative_false_counts",
                "background_counted",
                "speaking_counted",
                "floor_table_other_false_counts",
                "rejected_unsafe_counted",
                "acceptable_bounce_like_counted",
                "heldout_counted",
                "heldout_missed_140ms",
            ],
            [
                "Thr",
                "Pos Count",
                "Pos Abs Err",
                "Neg FP",
                "BG",
                "Speaking",
                "Impact FP",
                "Rejected FP Counted",
                "Accepted Bounce-like Counted",
                "C2 Count",
                "C2 Miss",
            ],
        ),
        "",
        "## Best Safe Policies",
        "",
        *md_table(
            safe_rows,
            [
                "pipeline_label",
                "positive_counted",
                "positive_abs_count_error",
                "negative_false_counts",
                "background_counted",
                "speaking_counted",
                "heldout_counted",
                "heldout_missed_140ms",
            ],
            ["Policy", "Pos Count", "Pos Abs Err", "Neg FP", "BG", "Speaking", "C2 Count", "C2 Miss"],
            limit=12,
        ),
        "",
        "## Manual False-Positive Status",
        "",
        *md_table(
            selected_manual,
            [
                "pipeline_label",
                "case_id",
                "manual_review",
                "candidate_prob",
                "case_still_counted",
                "nearest_counted_candidate_index",
                "nearest_counted_prob",
            ],
            ["Policy", "Case", "Manual Review", "Prob", "Still Counted", "Nearest Counted Idx", "Nearest Counted Prob"],
        ),
        "",
        "## Recommended Policy By Scenario",
        "",
        *md_table(
            recommended_scenarios,
            ["scenario_title", "expected_contacts", "candidate_count", "counted", "count_error"],
            ["Scenario", "Expected", "Candidates", "Counted", "Error"],
        ),
        "",
        "## Outputs",
        "",
        "- `t0074_policy_sweep.csv`",
        "- `t0074_scenario_summary.csv`",
        "- `t0074_manual_fp_status.csv`",
        "- `t0074_summary.json`",
        "- `t0074_report.md`",
    ]
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0071_dir = Path(args.t0071_dir)
    t0072_dir = Path(args.t0072_dir)
    t0073_dir = Path(args.t0073_dir)
    manifest_rows = load_manifest(t0071_dir)
    truth = truth_by_session(t0071_dir)
    heldout_truth = heldout_truth_ms(Path(args.heldout_labels))
    oof_rows = selected_oof_rows(t0072_dir)
    heldout_rows = selected_heldout_rows(t0072_dir)
    manual_cases = manual_fp_cases(t0073_dir)

    policy_rows: list[dict[str, Any]] = []
    all_scenario_rows: list[dict[str, Any]] = []
    all_manual_rows: list[dict[str, Any]] = []
    for dedupe_ms in dedupe_grid():
        for threshold in threshold_grid():
            policy = PolicySpec(SELECTED_CLASSIFIER_ID, SELECTED_CLASSIFIER_LABEL, threshold, dedupe_ms)
            block_rows = round_a_block_rows(rows=oof_rows, manifest_rows=manifest_rows, policy=policy, prob_key=ROUND_PROB_KEY)
            scenario_rows, total_rows = summarize_round_a(block_rows)
            enrich_total_scenario_fields(scenario_rows, total_rows)
            exact = exact_metrics(rows=oof_rows, truth=truth, policy=policy, prob_key=ROUND_PROB_KEY)
            heldout = heldout_metrics(rows=heldout_rows, truth_ms=heldout_truth, policy=policy, prob_key=HELDOUT_PROB_KEY)
            accepted = accepted_by_session(oof_rows, ROUND_PROB_KEY, threshold, dedupe_ms)
            manual_status = manual_fp_status_rows(policy=policy, accepted=accepted, cases=manual_cases)
            manual_summary = manual_fp_summary(manual_status)
            if not total_rows:
                continue
            row = {
                **total_rows[0],
                **exact,
                **heldout,
                **manual_summary,
            }
            policy_rows.append(row)
            all_scenario_rows.extend(scenario_rows)
            all_manual_rows.extend(manual_status)

    enrich_policy_totals(policy_rows)
    recommendation = choose_recommendation(policy_rows)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ticket": "T0074-fable-app-style-parity-and-safety-gate",
        "classifier_id": SELECTED_CLASSIFIER_ID,
        "classifier_label": SELECTED_CLASSIFIER_LABEL,
        "round_a_oof_rows": len(oof_rows),
        "heldout_rows": len(heldout_rows),
        "reviewed_positive_labels": sum(len(values) for values in truth.values()),
        "heldout_truth_labels": len(heldout_truth),
        "manual_fp_cases": len(manual_cases),
        "thresholds": threshold_grid(),
        "dedupe_windows_ms": dedupe_grid(),
        "policies_evaluated": len(policy_rows),
        "recommendation": recommendation,
    }

    write_csv(out_dir / "t0074_policy_sweep.csv", policy_rows)
    write_csv(out_dir / "t0074_scenario_summary.csv", all_scenario_rows)
    write_csv(out_dir / "t0074_manual_fp_status.csv", all_manual_rows)
    (out_dir / "t0074_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report = render_report(
        summary=summary,
        recommendation=recommendation,
        policy_rows=policy_rows,
        scenario_rows=all_scenario_rows,
        manual_rows=all_manual_rows,
    )
    (out_dir / "t0074_report.md").write_text(report, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t0071-dir", default=str(DEFAULT_T0071_DIR))
    parser.add_argument("--t0072-dir", default=str(DEFAULT_T0072_DIR))
    parser.add_argument("--t0073-dir", default=str(DEFAULT_T0073_DIR))
    parser.add_argument("--heldout-labels", default=str(DEFAULT_T0063_LABELS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary["recommendation"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
