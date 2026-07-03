#!/usr/bin/env python3
"""Evaluate PCEN-rich post-Hybrid filters for STIGA Hybrid 2.0.

T0129 proved that a learned post-Hybrid veto can remove many hard negatives
offline, but the first phone check was too strict on real bounces. This ticket
keeps the same training frame and asks a narrower question:

  Can PCEN/noise-robust features remove hard negatives at a much higher-recall
  operating point than the previous learned veto?

The script intentionally does not touch app code. It exports a local candidate
JSON only for a variant that is worth future STIGA QA.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_t0129_hybrid_hard_negative_veto import (
    DEFAULT_CONTACT,
    DEFAULT_ROWS,
    DEFAULT_ROOT,
    DEFAULT_SURFACE,
    TrainConfig,
    add_live_rf_predictions,
    build_training_matrices,
    dedupe_mask,
    export_final_model,
    hybrid_accept_mask,
    load_json,
    metrics_for_keep,
    run_grouped_oof,
    value_counts_dict,
)


DEFAULT_OUT = DEFAULT_ROOT / "data/audio/models/evaluations/t0130_pcen_hybrid_post_filter"


@dataclass(frozen=True)
class VariantResult:
    name: str
    feature_columns: list[str]
    train_positions: np.ndarray
    matrix: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    selected_config: TrainConfig
    selected_reason: str
    selected_oof_full: np.ndarray
    sweep: pd.DataFrame
    metrics: dict[str, int]


def unique_columns(columns: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for column in columns:
        if column not in seen:
            out.append(column)
            seen.add(column)
    return out


def existing_columns(rows: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in unique_columns(columns) if column in rows.columns]


def variant_feature_sets(
    rows: pd.DataFrame,
    contact_model: dict[str, Any],
    surface_model: dict[str, Any],
) -> dict[str, list[str]]:
    rf_feature_columns = ["feat_" + name for name in contact_model["feature_names"]]
    probability_columns = (
        [f"contact_prob_{label}" for label in contact_model["labels"]]
        + [f"surface_prob_{label}" for label in surface_model["labels"]]
        + ["contact_confidence", "surface_confidence"]
    )

    pcen_columns = [
        "feat_nr_pcen_max",
        "feat_nr_pcen_mean",
        "feat_nr_pcen_std",
    ]
    pcen_context_columns = [
        "feat_nr_snr_db_est",
        "feat_nr_bg_rms_db",
        "feat_nr_flux_onset",
        "feat_nr_bg_flatness",
        "feat_nr_impact_flatness",
        "feat_nr_bp_attack_ms",
        "feat_nr_bp_decay50_ms",
        "feat_nr_bp_crest",
        "feat_nr_bp_peak_ratio",
        "feat_nr_post_decay_db_50ms",
        "feat_nr_post_decay_db_100ms",
        "feat_nr_bp_peak_db",
    ]
    nr_band_delta_columns = [column for column in rows.columns if column.startswith("feat_nr_band_delta_")]
    time_domain_columns = [column for column in rows.columns if column.startswith("td_")]
    spectral_window_columns = [column for column in rows.columns if column.startswith("sp_")]

    rf_prob = existing_columns(rows, rf_feature_columns + probability_columns)
    pcen = existing_columns(rows, pcen_columns)
    pcen_context = existing_columns(rows, pcen_columns + pcen_context_columns + nr_band_delta_columns)
    transient_spectral = existing_columns(rows, time_domain_columns + spectral_window_columns)

    return {
        "rf_prob_t0129_style": rf_prob,
        "pcen_only": pcen,
        "pcen_context_only": pcen_context,
        "rf_prob_plus_pcen": rf_prob + pcen_context,
        "rf_prob_plus_pcen_transient_spectral": rf_prob + pcen_context + transient_spectral,
    }


def choose_high_recall_config(sweep: pd.DataFrame) -> tuple[pd.Series, str]:
    for max_loss in (0, 3, 5, 10, 25):
        candidates = sweep[sweep["tp_loss"] <= max_loss]
        if not candidates.empty:
            ordered = candidates.sort_values(
                ["fp_drop", "recall_vs_hybrid", "threshold"],
                ascending=[False, False, False],
            )
            return ordered.iloc[0], f"max_tp_loss_{max_loss}"

    ordered = sweep.sort_values(["score", "recall_vs_hybrid"], ascending=[False, False])
    return ordered.iloc[0], "fallback_weighted_score"


def evaluate_variant(
    rows: pd.DataFrame,
    hybrid_accept: np.ndarray,
    feature_columns: list[str],
    *,
    merge_ms: int,
    folds: int,
) -> VariantResult:
    train_positions, matrix, labels, groups = build_training_matrices(rows, hybrid_accept, feature_columns)
    base_keep = dedupe_mask(rows, hybrid_accept, merge_ms)
    base_metrics = metrics_for_keep(rows, base_keep)

    configs = [
        TrainConfig(neg_weight=neg_weight, l2=l2, threshold=threshold, epochs=700, lr=0.025)
        for neg_weight in (0.75, 1.0, 1.5, 2.0, 3.0)
        for l2 in (0.002, 0.008, 0.02, 0.06)
        for threshold in np.linspace(0.05, 0.95, 19)
    ]

    records: list[dict[str, Any]] = []
    oof_cache: dict[tuple[float, float], np.ndarray] = {}
    selected_row: pd.Series | None = None
    selected_reason = ""
    selected_oof_full: np.ndarray | None = None
    selected_config: TrainConfig | None = None

    for config in configs:
        cache_key = (config.neg_weight, config.l2)
        if cache_key not in oof_cache:
            oof_cache[cache_key] = run_grouped_oof(matrix, labels, groups, config, folds=folds)
        oof_full = np.full(len(rows), np.nan, dtype=np.float64)
        oof_full[train_positions] = oof_cache[cache_key]

        accepted = hybrid_accept & (oof_full >= config.threshold)
        keep = dedupe_mask(rows, accepted, merge_ms)
        metrics = metrics_for_keep(rows, keep)
        recall = metrics["tp"] / base_metrics["tp"] if base_metrics["tp"] else 0.0
        tp_loss = base_metrics["tp"] - metrics["tp"]
        fp_drop = base_metrics["fp"] - metrics["fp"]
        score = fp_drop - max(0.0, 0.995 - recall) * 20000.0 - max(0, tp_loss - 5) * 400.0
        records.append(
            {
                "neg_weight": config.neg_weight,
                "l2": config.l2,
                "threshold": float(config.threshold),
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "tp_loss": tp_loss,
                "fp_drop": fp_drop,
                "recall_vs_hybrid": recall,
                "score": score,
            }
        )

    sweep = pd.DataFrame(records)
    selected_row, selected_reason = choose_high_recall_config(sweep)
    selected_config = TrainConfig(
        neg_weight=float(selected_row["neg_weight"]),
        l2=float(selected_row["l2"]),
        threshold=float(selected_row["threshold"]),
    )
    selected_oof = run_grouped_oof(matrix, labels, groups, selected_config, folds=folds)
    selected_oof_full = np.full(len(rows), np.nan, dtype=np.float64)
    selected_oof_full[train_positions] = selected_oof
    selected_accept = hybrid_accept & (selected_oof_full >= selected_config.threshold)
    selected_keep = dedupe_mask(rows, selected_accept, merge_ms)
    selected_metrics = metrics_for_keep(rows, selected_keep)

    return VariantResult(
        name="",
        feature_columns=feature_columns,
        train_positions=train_positions,
        matrix=matrix,
        labels=labels,
        groups=groups,
        selected_config=selected_config,
        selected_reason=selected_reason,
        selected_oof_full=selected_oof_full,
        sweep=sweep.sort_values(["tp_loss", "fp_drop"], ascending=[True, False]),
        metrics=selected_metrics,
    )


def count_by_group(rows: pd.DataFrame, keep: np.ndarray, *, label_value: int) -> dict[str, int]:
    return value_counts_dict(rows[(rows["label"].astype(int) == label_value) & keep]["eval_group"], 40)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values: list[str] = []
        for column in frame.columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--contact-model", type=Path, default=DEFAULT_CONTACT)
    parser.add_argument("--surface-model", type=Path, default=DEFAULT_SURFACE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--hybrid-contact-threshold", type=float, default=0.25)
    parser.add_argument("--hybrid-surface-veto", type=float, default=0.75)
    parser.add_argument("--merge-ms", type=int, default=180)
    parser.add_argument("--folds", type=int, default=8)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = pd.read_csv(args.candidate_rows)
    if "domain_session_id" not in rows.columns:
        rows["domain_session_id"] = rows["session_id"].astype(str)
    rows["domain_session_id"] = rows["domain_session_id"].fillna(rows["session_id"]).astype(str)
    rows["eval_group"] = rows["eval_group"].fillna(rows.get("scenario_title", "unknown"))

    contact_model = load_json(args.contact_model)
    surface_model = load_json(args.surface_model)
    rows = add_live_rf_predictions(rows, contact_model, surface_model)

    hybrid_accept = hybrid_accept_mask(
        rows,
        confidence_threshold=args.hybrid_contact_threshold,
        surface_veto_confidence=args.hybrid_surface_veto,
    )
    base_keep = dedupe_mask(rows, hybrid_accept, args.merge_ms)
    base_metrics = metrics_for_keep(rows, base_keep)

    variants = variant_feature_sets(rows, contact_model, surface_model)
    results: list[VariantResult] = []
    summary_records: list[dict[str, Any]] = []

    for name, columns in variants.items():
        if not columns:
            continue
        print(f"Evaluating {name} ({len(columns)} features)")
        result = evaluate_variant(
            rows,
            hybrid_accept,
            columns,
            merge_ms=args.merge_ms,
            folds=args.folds,
        )
        result = replace(result, name=name)
        results.append(result)
        result.sweep.to_csv(args.out_dir / f"t0130_{name}_sweep.csv", index=False)
        summary_records.append(
            {
                "variant": name,
                "feature_count": len(columns),
                "selection": result.selected_reason,
                "neg_weight": result.selected_config.neg_weight,
                "l2": result.selected_config.l2,
                "threshold": result.selected_config.threshold,
                "tp": result.metrics["tp"],
                "fp": result.metrics["fp"],
                "tp_loss": base_metrics["tp"] - result.metrics["tp"],
                "fp_drop": base_metrics["fp"] - result.metrics["fp"],
                "recall_vs_hybrid": result.metrics["tp"] / base_metrics["tp"] if base_metrics["tp"] else 0.0,
            }
        )

    summary = pd.DataFrame(summary_records).sort_values(["tp_loss", "fp_drop"], ascending=[True, False])
    summary.to_csv(args.out_dir / "t0130_variant_summary.csv", index=False)

    candidate_rows = summary[summary["tp_loss"] <= 5].copy()
    if candidate_rows.empty:
        candidate_rows = summary[summary["tp_loss"] <= 10].copy()
    if candidate_rows.empty:
        candidate_rows = summary.copy()
    best_variant_name = str(candidate_rows.sort_values(["fp_drop", "recall_vs_hybrid"], ascending=[False, False]).iloc[0]["variant"])
    best_result = next(result for result in results if result.name == best_variant_name)

    accepted = hybrid_accept & (best_result.selected_oof_full >= best_result.selected_config.threshold)
    keep = dedupe_mask(rows, accepted, args.merge_ms)
    rows_with_probs = rows.copy()
    rows_with_probs["t0130_variant"] = best_result.name
    rows_with_probs["t0130_oof_hard_positive_probability"] = best_result.selected_oof_full
    rows_with_probs["t0130_oof_accept"] = accepted
    rows_with_probs["t0130_oof_counted_after_dedupe"] = keep
    rows_with_probs.to_csv(args.out_dir / "t0130_best_oof_predictions.csv", index=False)

    model_path = args.out_dir / f"hybrid_pcen_post_filter_t0130_{best_result.name}.json"
    metadata = {
        "ticket_id": "T0130-pcen-hybrid-post-filter-audit",
        "runtime_status": "offline_candidate_only_not_ported",
        "trained_from": str(args.candidate_rows).replace("\\", "/"),
        "variant": best_result.name,
        "selection_reason": best_result.selected_reason,
        "base_hybrid": {
            "contact_threshold": args.hybrid_contact_threshold,
            "surface_veto_confidence": args.hybrid_surface_veto,
            "merge_ms": args.merge_ms,
        },
        "grouped_oof_folds": args.folds,
        "selected_config": {
            "neg_weight": best_result.selected_config.neg_weight,
            "l2": best_result.selected_config.l2,
            "threshold": best_result.selected_config.threshold,
        },
        "oof_metrics": {
            "base_hybrid": base_metrics,
            "selected_variant": best_result.metrics,
            "tp_loss": base_metrics["tp"] - best_result.metrics["tp"],
            "fp_drop": base_metrics["fp"] - best_result.metrics["fp"],
            "recall_vs_hybrid": best_result.metrics["tp"] / base_metrics["tp"] if base_metrics["tp"] else 0.0,
        },
    }
    export_final_model(
        model_path,
        best_result.feature_columns,
        best_result.matrix,
        best_result.labels,
        best_result.selected_config,
        metadata,
    )

    base_fp = count_by_group(rows, base_keep, label_value=0)
    selected_fp = count_by_group(rows, keep, label_value=0)
    selected_tp = count_by_group(rows, keep, label_value=1)
    lost_tp_rows = rows[(rows["label"].astype(int) == 1) & base_keep & ~keep]
    removed_fp_rows = rows[(rows["label"].astype(int) == 0) & base_keep & ~keep]

    report = [
        "# T0130 PCEN Hybrid Post-Filter Audit",
        "",
        "Purpose: test whether PCEN/noise-robust features can create a safer post-Hybrid hard-negative filter than the failed live T0129 veto.",
        "",
        "## Base Hybrid",
        "",
        f"- Plain Hybrid after dedupe: TP `{base_metrics['tp']}`, FP `{base_metrics['fp']}`.",
        "",
        "## Variant Summary",
        "",
        markdown_table(summary),
        "",
        "## Selected Offline Candidate",
        "",
        f"- Variant: `{best_result.name}`.",
        f"- Selection rule: `{best_result.selected_reason}`.",
        f"- Feature count: `{len(best_result.feature_columns)}`.",
        f"- Threshold: `{best_result.selected_config.threshold:.2f}`.",
        f"- Selected TP/FP: `{best_result.metrics['tp']}/{best_result.metrics['fp']}`.",
        f"- TP loss vs Hybrid: `{base_metrics['tp'] - best_result.metrics['tp']}`.",
        f"- FP drop vs Hybrid: `{base_metrics['fp'] - best_result.metrics['fp']}`.",
        f"- Recall vs Hybrid: `{best_result.metrics['tp'] / base_metrics['tp']:.4f}`.",
        "",
        "## False Positives By Group",
        "",
        "### Plain Hybrid",
        "",
        json.dumps(base_fp, indent=2),
        "",
        "### Selected Candidate Remaining",
        "",
        json.dumps(selected_fp, indent=2),
        "",
        "### Removed By Selected Candidate",
        "",
        json.dumps(value_counts_dict(removed_fp_rows["eval_group"], 40), indent=2),
        "",
        "## True Positives By Group",
        "",
        "### Selected Candidate Counted",
        "",
        json.dumps(selected_tp, indent=2),
        "",
        "### Lost From Plain Hybrid",
        "",
        json.dumps(value_counts_dict(lost_tp_rows["eval_group"], 40), indent=2),
        "",
        "## Interpretation",
        "",
        "This is an offline audit only. PCEN by itself is not enough: `pcen_only` and `pcen_context_only` preserved recall but removed only a handful of false positives. The strongest row combines RF probabilities, PCEN/noise-robust features, and `td_*`/`sp_*` transient/spectral columns. That row is not a drop-in STIGA port unless the app runtime computes the same transient/spectral features with parity.",
        "",
        "Recommendation: do not port the exported JSON blindly. The next implementation ticket should either add and parity-test the missing `td_*`/`sp_*` feature extractor in STIGA, or use a gentler already-portable RF/probability post-filter as a guarded comparison.",
        "",
        "## Outputs",
        "",
        f"- Summary CSV: `{args.out_dir / 't0130_variant_summary.csv'}`",
        f"- Best OOF predictions: `{args.out_dir / 't0130_best_oof_predictions.csv'}`",
        f"- Local candidate JSON: `{model_path}`",
    ]
    (args.out_dir / "t0130_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print("T0130 PCEN audit complete")
    print(f"Base Hybrid TP/FP: {base_metrics['tp']}/{base_metrics['fp']}")
    print(summary.to_string(index=False))
    print(f"Selected: {best_result.name} -> TP/FP {best_result.metrics['tp']}/{best_result.metrics['fp']}")
    print(f"Report: {args.out_dir / 't0130_report.md'}")


if __name__ == "__main__":
    main()
