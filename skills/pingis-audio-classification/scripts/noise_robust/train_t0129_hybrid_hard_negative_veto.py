#!/usr/bin/env python3
"""Train the T0129 learned hard-negative veto for STIGA Hybrid.

This script intentionally avoids sklearn so the workflow still runs on the
current Windows machine where importing sklearn is unreliable. It uses:

1. Existing app RF JSONs to reproduce STIGA Hybrid candidate acceptance.
2. A small logistic model trained only on Hybrid-accepted rows.
3. Grouped out-of-fold validation by session/domain to avoid same-clip leakage.

The exported JSON is portable to TypeScript:

  raw clip -> existing 62 audio features
           -> existing contact/surface RF probabilities
           -> this logistic hard-positive probability
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ROWS = DEFAULT_ROOT / "data/audio/models/evaluations/t0104e_live_positive_candidate_loop/t0104e_candidate_rows.csv"
DEFAULT_CONTACT = DEFAULT_ROOT / "apps/collector/src/models/audio_contact_model.json"
DEFAULT_SURFACE = DEFAULT_ROOT / "apps/collector/src/models/audio_model.json"
DEFAULT_OUT = DEFAULT_ROOT / "data/audio/models/evaluations/t0129_hybrid_hard_negative_veto_model"

VETO_LABELS = {"floor_bounce", "noise", "table_bounce"}
POSITIVE_LABEL = "hard_positive"
NEGATIVE_LABEL = "hard_negative"


@dataclass(frozen=True)
class TrainConfig:
    neg_weight: float
    l2: float
    threshold: float
    epochs: int = 700
    lr: float = 0.025


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_leaf(node: list[float], n_classes: int) -> bool:
    if len(node) != n_classes:
        return len(node) != 4
    total = 0.0
    for value in node:
        if value < 0 or value > 1:
            return False
        total += value
    return abs(total - 1.0) < 0.01


def traverse_rf_tree(tree: list[list[float]], scaled_features: np.ndarray, n_classes: int) -> np.ndarray:
    idx = 0
    while not is_leaf(tree[idx], n_classes):
        node = tree[idx]
        feature_idx = int(node[0])
        threshold = float(node[1])
        idx = int(node[2] if scaled_features[feature_idx] <= threshold else node[3])
    return np.asarray(tree[idx], dtype=np.float64)


def rf_predict(model: dict[str, Any], row: dict[str, Any]) -> tuple[str, float, dict[str, float]]:
    labels = model["labels"]
    names = model["feature_names"]
    means = np.asarray(model["scaler_mean"], dtype=np.float64)
    stds = np.asarray(model["scaler_std"], dtype=np.float64)

    raw_values: list[float] = []
    for name in names:
        value = row.get("feat_" + name, row.get(name, 0.0))
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0
        if not math.isfinite(numeric):
            numeric = 0.0
        raw_values.append(numeric)

    raw = np.asarray(raw_values, dtype=np.float64)
    scaled = (raw - means) / np.where(stds == 0, 1.0, stds)

    sums = np.zeros(len(labels), dtype=np.float64)
    for tree in model["trees"]:
        sums += traverse_rf_tree(tree, scaled, len(labels))
    probabilities = sums / max(1, len(model["trees"]))
    max_idx = int(np.argmax(probabilities))
    by_label = {labels[i]: float(probabilities[i]) for i in range(len(labels))}
    return labels[max_idx], float(probabilities[max_idx]), by_label


def sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40, 40)))


def normalize_fit(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.nanmean(matrix, axis=0)
    stds = np.nanstd(matrix, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    stds = np.where(np.isfinite(stds) & (stds >= 1e-6), stds, 1.0)
    normalized = (np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0) - means) / stds
    return means, stds, normalized


def normalize_apply(matrix: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    return (np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0) - means) / stds


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    l2: float,
    lr: float,
    epochs: int,
) -> tuple[np.ndarray, float]:
    n_features = features.shape[1]
    coef = np.zeros(n_features, dtype=np.float64)
    bias = 0.0

    coef_m = np.zeros(n_features, dtype=np.float64)
    coef_v = np.zeros(n_features, dtype=np.float64)
    bias_m = 0.0
    bias_v = 0.0
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    weight_sum = float(weights.sum())

    for step in range(1, epochs + 1):
        predicted = sigmoid(features @ coef + bias)
        error = (predicted - labels) * weights / weight_sum
        grad_coef = features.T @ error + l2 * coef
        grad_bias = float(error.sum())

        coef_m = beta1 * coef_m + (1.0 - beta1) * grad_coef
        coef_v = beta2 * coef_v + (1.0 - beta2) * (grad_coef * grad_coef)
        bias_m = beta1 * bias_m + (1.0 - beta1) * grad_bias
        bias_v = beta2 * bias_v + (1.0 - beta2) * (grad_bias * grad_bias)

        coef -= lr * (coef_m / (1.0 - beta1**step)) / (np.sqrt(coef_v / (1.0 - beta2**step)) + eps)
        bias -= lr * (bias_m / (1.0 - beta1**step)) / (math.sqrt(bias_v / (1.0 - beta2**step)) + eps)

    return coef, float(bias)


def add_live_rf_predictions(
    rows: pd.DataFrame,
    contact_model: dict[str, Any],
    surface_model: dict[str, Any],
) -> pd.DataFrame:
    records = rows.to_dict("records")
    contact_labels: list[str] = []
    contact_confidences: list[float] = []
    contact_probs: list[dict[str, float]] = []
    surface_labels: list[str] = []
    surface_confidences: list[float] = []
    surface_probs: list[dict[str, float]] = []

    for record in records:
        contact_label, contact_confidence, contact_by_label = rf_predict(contact_model, record)
        surface_label, surface_confidence, surface_by_label = rf_predict(surface_model, record)
        contact_labels.append(contact_label)
        contact_confidences.append(contact_confidence)
        contact_probs.append(contact_by_label)
        surface_labels.append(surface_label)
        surface_confidences.append(surface_confidence)
        surface_probs.append(surface_by_label)

    out = rows.copy()
    out["contact_label_live"] = contact_labels
    out["contact_confidence"] = contact_confidences
    out["surface_label_live"] = surface_labels
    out["surface_confidence"] = surface_confidences

    for label in contact_model["labels"]:
        out[f"contact_prob_{label}"] = [probs.get(label, 0.0) for probs in contact_probs]
    for label in surface_model["labels"]:
        out[f"surface_prob_{label}"] = [probs.get(label, 0.0) for probs in surface_probs]
    return out


def hybrid_accept_mask(rows: pd.DataFrame, confidence_threshold: float, surface_veto_confidence: float) -> np.ndarray:
    return (
        rows["contact_label_live"].eq("racket_contact")
        & (rows["contact_confidence"] >= confidence_threshold)
        & ~(
            rows["surface_label_live"].isin(VETO_LABELS)
            & (rows["surface_confidence"] >= surface_veto_confidence)
        )
    ).to_numpy()


def dedupe_mask(rows: pd.DataFrame, accept: np.ndarray, merge_ms: int) -> np.ndarray:
    keep = np.zeros(len(rows), dtype=bool)
    accepted_positions = np.where(accept)[0]
    if accepted_positions.size == 0:
        return keep

    accepted = rows.iloc[accepted_positions].sort_values(["domain_session_id", "time_ms"])
    for _, group in accepted.groupby("domain_session_id", sort=False):
        last_kept = -1e18
        for row_idx, row in group.iterrows():
            t = float(row["time_ms"])
            if t - last_kept >= merge_ms:
                keep[rows.index.get_loc(row_idx)] = True
                last_kept = t
    return keep


def fold_for_group(group_id: str, folds: int) -> int:
    digest = hashlib.md5(group_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds


def metrics_for_keep(rows: pd.DataFrame, keep: np.ndarray) -> dict[str, int]:
    labels = rows["label"].astype(int).to_numpy()
    return {
        "tp": int(((labels == 1) & keep).sum()),
        "fp": int(((labels == 0) & keep).sum()),
        "kept": int(keep.sum()),
    }


def value_counts_dict(series: pd.Series, limit: int = 20) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts().head(limit).items()}


def build_training_matrices(
    rows: pd.DataFrame,
    hybrid_accept: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_positions = np.where(hybrid_accept)[0]
    matrix = rows.iloc[train_positions][feature_names].astype(float).to_numpy()
    labels = (rows.iloc[train_positions]["label"].astype(int).to_numpy() == 1).astype(float)
    groups = rows.iloc[train_positions]["domain_session_id"].astype(str).to_numpy()
    return train_positions, matrix, labels, groups


def run_grouped_oof(
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    config: TrainConfig,
    *,
    folds: int,
) -> np.ndarray:
    fold_ids = np.asarray([fold_for_group(group, folds) for group in groups])
    oof = np.full(len(labels), np.nan, dtype=np.float64)

    for fold in range(folds):
        train_mask = fold_ids != fold
        test_mask = fold_ids == fold
        if not test_mask.any() or len(set(labels[train_mask])) < 2:
            continue

        means, stds, normalized_train = normalize_fit(matrix[train_mask])
        normalized_test = normalize_apply(matrix[test_mask], means, stds)
        sample_weights = np.where(labels[train_mask] > 0.5, 1.0, config.neg_weight)
        coef, bias = fit_logistic(
            normalized_train,
            labels[train_mask],
            sample_weights,
            l2=config.l2,
            lr=config.lr,
            epochs=config.epochs,
        )
        oof[test_mask] = sigmoid(normalized_test @ coef + bias)

    if not np.isfinite(oof).all():
        missing = int((~np.isfinite(oof)).sum())
        raise RuntimeError(f"OOF training left {missing} rows without predictions")
    return oof


def sweep_configs(
    rows: pd.DataFrame,
    hybrid_accept: np.ndarray,
    train_positions: np.ndarray,
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    merge_ms: int,
    folds: int,
) -> tuple[TrainConfig, pd.DataFrame, np.ndarray]:
    base_keep = dedupe_mask(rows, hybrid_accept, merge_ms)
    base_metrics = metrics_for_keep(rows, base_keep)

    configs = [
        TrainConfig(neg_weight=neg_weight, l2=l2, threshold=threshold)
        for neg_weight in (0.75, 1.0, 1.5, 2.0, 3.0)
        for l2 in (0.002, 0.008, 0.02, 0.06)
        for threshold in np.linspace(0.05, 0.95, 19)
    ]

    rows_out: list[dict[str, Any]] = []
    best_config: TrainConfig | None = None
    best_oof_full: np.ndarray | None = None
    best_score = -1e18

    # OOF probabilities are identical for each neg_weight/l2 pair, so cache them.
    oof_cache: dict[tuple[float, float], np.ndarray] = {}
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
        fp_drop = base_metrics["fp"] - metrics["fp"]
        tp_loss = base_metrics["tp"] - metrics["tp"]
        score = fp_drop - max(0.0, 0.96 - recall) * 7000.0 - max(0, tp_loss - 25) * 100.0

        record = {
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
        rows_out.append(record)
        if score > best_score:
            best_score = score
            best_config = config
            best_oof_full = oof_full

    if best_config is None or best_oof_full is None:
        raise RuntimeError("No valid config found")
    return best_config, pd.DataFrame(rows_out).sort_values("score", ascending=False), best_oof_full


def export_final_model(
    out_path: Path,
    feature_names: list[str],
    matrix: np.ndarray,
    labels: np.ndarray,
    config: TrainConfig,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    means, stds, normalized = normalize_fit(matrix)
    sample_weights = np.where(labels > 0.5, 1.0, config.neg_weight)
    coef, bias = fit_logistic(
        normalized,
        labels,
        sample_weights,
        l2=config.l2,
        lr=config.lr,
        epochs=config.epochs,
    )

    model = {
        "model_type": "logistic_binary_v1",
        "labels": [NEGATIVE_LABEL, POSITIVE_LABEL],
        "positive_label": POSITIVE_LABEL,
        "feature_names": feature_names,
        "scaler_mean": [float(v) for v in means],
        "scaler_std": [float(v) for v in stds],
        "weights": [float(v) for v in coef],
        "bias": float(bias),
        "threshold": float(config.threshold),
        "metadata": metadata,
    }
    out_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    return model


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

    feature_names = (
        list(contact_model["feature_names"])
        + [f"contact_prob_{label}" for label in contact_model["labels"]]
        + [f"surface_prob_{label}" for label in surface_model["labels"]]
        + ["contact_confidence", "surface_confidence"]
    )
    feature_columns = ["feat_" + name if name in contact_model["feature_names"] else name for name in feature_names]
    train_positions, matrix, labels, groups = build_training_matrices(rows, hybrid_accept, feature_columns)

    best_config, sweep, best_oof_full = sweep_configs(
        rows,
        hybrid_accept,
        train_positions,
        matrix,
        labels,
        groups,
        merge_ms=args.merge_ms,
        folds=args.folds,
    )

    learned_accept = hybrid_accept & (best_oof_full >= best_config.threshold)
    learned_keep = dedupe_mask(rows, learned_accept, args.merge_ms)
    learned_metrics = metrics_for_keep(rows, learned_keep)

    base_fp_rows = rows[(rows["label"].astype(int) == 0) & base_keep]
    learned_fp_rows = rows[(rows["label"].astype(int) == 0) & learned_keep]
    lost_tp_rows = rows[(rows["label"].astype(int) == 1) & base_keep & ~learned_keep]

    model_path = args.out_dir / "hybrid_hard_negative_veto_t0129_logreg.json"
    metadata = {
        "ticket_id": "T0129-hybrid-hard-negative-veto-model",
        "trained_from": str(args.candidate_rows).replace("\\", "/"),
        "base_hybrid": {
            "contact_threshold": args.hybrid_contact_threshold,
            "surface_veto_confidence": args.hybrid_surface_veto,
            "merge_ms": args.merge_ms,
        },
        "training_rows": int(len(labels)),
        "training_positive_rows": int(labels.sum()),
        "training_negative_rows": int((1.0 - labels).sum()),
        "grouped_oof_folds": args.folds,
        "selected_config": {
            "neg_weight": best_config.neg_weight,
            "l2": best_config.l2,
            "threshold": best_config.threshold,
            "epochs": best_config.epochs,
            "lr": best_config.lr,
        },
        "oof_metrics": {
            "base_hybrid": base_metrics,
            "learned_veto": learned_metrics,
            "tp_loss": base_metrics["tp"] - learned_metrics["tp"],
            "fp_drop": base_metrics["fp"] - learned_metrics["fp"],
            "recall_vs_hybrid": learned_metrics["tp"] / base_metrics["tp"] if base_metrics["tp"] else 0.0,
        },
    }
    export_final_model(model_path, feature_names, matrix, labels, best_config, metadata)

    rows_with_probs = rows.copy()
    rows_with_probs["t0129_oof_hard_positive_probability"] = best_oof_full
    rows_with_probs["t0129_oof_accept"] = learned_accept
    rows_with_probs["t0129_oof_counted_after_dedupe"] = learned_keep
    rows_with_probs.to_csv(args.out_dir / "t0129_candidate_oof_predictions.csv", index=False)
    sweep.to_csv(args.out_dir / "t0129_threshold_sweep.csv", index=False)

    report = [
        "# T0129 Hybrid Hard-Negative Veto Model",
        "",
        "Purpose: keep plain STIGA Hybrid unchanged, then add a learned post-Hybrid veto for the guarded Hybrid 2.0 option.",
        "",
        "## Base vs Learned Veto",
        "",
        f"- Plain Hybrid after dedupe: TP `{base_metrics['tp']}`, FP `{base_metrics['fp']}`.",
        f"- Learned veto after dedupe: TP `{learned_metrics['tp']}`, FP `{learned_metrics['fp']}`.",
        f"- True-positive loss: `{base_metrics['tp'] - learned_metrics['tp']}`.",
        f"- False-positive reduction: `{base_metrics['fp'] - learned_metrics['fp']}`.",
        f"- Recall vs Hybrid: `{learned_metrics['tp'] / base_metrics['tp']:.4f}`.",
        "",
        "## Selected Config",
        "",
        f"- Negative sample weight: `{best_config.neg_weight}`.",
        f"- L2: `{best_config.l2}`.",
        f"- Hard-positive threshold: `{best_config.threshold:.2f}`.",
        f"- Folds: `{args.folds}` grouped by `domain_session_id`.",
        "",
        "## False Positives Remaining",
        "",
        json.dumps(value_counts_dict(learned_fp_rows["eval_group"], 30), indent=2),
        "",
        "## False Positives Removed From Plain Hybrid",
        "",
        json.dumps(value_counts_dict(base_fp_rows[~base_fp_rows.index.isin(learned_fp_rows.index)]["eval_group"], 30), indent=2),
        "",
        "## True Positives Lost",
        "",
        json.dumps(value_counts_dict(lost_tp_rows["eval_group"], 30), indent=2),
        "",
        "## Outputs",
        "",
        f"- Model JSON: `{model_path}`",
        f"- OOF rows: `{args.out_dir / 't0129_candidate_oof_predictions.csv'}`",
        f"- Sweep CSV: `{args.out_dir / 't0129_threshold_sweep.csv'}`",
        "",
        "## Runtime Shape",
        "",
        "Existing Hybrid still decides first. Hybrid 2.0 should call this model only for clips Hybrid already accepts.",
        "If `hard_positive_probability` is below the exported threshold, reject as a hard negative; otherwise keep Hybrid's count.",
    ]
    (args.out_dir / "t0129_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print("T0129 learned veto complete")
    print(f"Base Hybrid TP/FP: {base_metrics['tp']}/{base_metrics['fp']}")
    print(f"Learned veto TP/FP: {learned_metrics['tp']}/{learned_metrics['fp']}")
    print(f"Model: {model_path}")
    print(f"Report: {args.out_dir / 't0129_report.md'}")


if __name__ == "__main__":
    main()
