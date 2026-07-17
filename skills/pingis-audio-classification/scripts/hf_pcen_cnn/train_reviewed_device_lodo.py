"""Train on reviewed HF candidates and evaluate one unseen phone at a time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from .audit_review_pack import greedy_time_match
from .candidate_labels import FOUR_CLASSES
from .fable_baseline import FableTimingConfig, replay_deployed_fable_timing
from .train_ablation import (
    CLASS_TO_INDEX,
    _choose_grouped_split,
    _predict,
    _train_one,
)


RACKET_LABEL = "racket_bounce"
TRAINABLE_DISPOSITIONS = ("positive", "hard_negative")


def _trainable_mask(metadata: pd.DataFrame) -> np.ndarray:
    labels = metadata["label"].map(CLASS_TO_INDEX)
    return (
        metadata["cache_complete"].astype(bool)
        & metadata["disposition"].isin(TRAINABLE_DISPOSITIONS)
        & labels.notna()
    ).to_numpy()


def review_fold_indexes(
    metadata: pd.DataFrame,
    held_device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return reviewed train and held-device inference rows without leakage."""
    trainable = _trainable_mask(metadata)
    held = metadata["device_id"].astype(str).eq(str(held_device)).to_numpy()
    return np.flatnonzero(trainable & ~held), np.flatnonzero(held)


def _binary_racket_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, object]:
    truth = labels == CLASS_TO_INDEX[RACKET_LABEL]
    predicted = predictions == CLASS_TO_INDEX[RACKET_LABEL]
    return {
        "rows": int(len(labels)),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(
            truth, predicted, labels=[False, True]
        ).tolist(),
    }


def _candidate_rows_for_timing(
    metadata: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (_, row), prediction, probability in zip(
        metadata.iterrows(), predictions, probabilities, strict=True
    ):
        rows.append(
            {
                "onset_ms": float(row["onset_ms"]),
                "frame_rms": float(row["hf_rms"]),
                "background_db": float(row["background_db"]),
                "predicted_label": FOUR_CLASSES[int(prediction)],
                "confidence": float(np.max(probability)),
            }
        )
    return rows


def _fable_rows_for_timing(metadata: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {
            "onset_ms": float(row["onset_ms"]),
            "frame_rms": float(row["hf_rms"]),
            "background_db": float(row["background_db"]),
            "predicted_label": str(row["fable_predicted_label"]),
            "confidence": float(row["fable_confidence"]),
        }
        for _, row in metadata.iterrows()
    ]


def _truth_times(session_rows: pd.DataFrame) -> list[float]:
    positives = session_rows.loc[
        session_rows["disposition"].eq("positive"),
        ["matched_event_index", "reviewed_event_ms"],
    ].dropna()
    if positives.empty:
        return []
    return [
        float(value)
        for value in (
            positives.sort_values("matched_event_index")
            .drop_duplicates("matched_event_index")["reviewed_event_ms"]
            .tolist()
        )
    ]


def _score_counting(
    metadata: pd.DataFrame,
    candidate_rows: list[dict[str, object]],
    *,
    tolerance_ms: float,
    timing_config: FableTimingConfig = FableTimingConfig(),
) -> dict[str, object]:
    if len(metadata) != len(candidate_rows):
        raise ValueError("Metadata and prediction rows must have the same length")
    evaluated = metadata.copy().reset_index(drop=True)
    evaluated["timing_row"] = candidate_rows
    sessions: list[dict[str, object]] = []
    for session_id, group in evaluated.groupby("session_id", sort=True):
        rows = list(group["timing_row"])
        decisions = replay_deployed_fable_timing(rows, timing_config)
        counted_ms = [
            decision.onset_ms for decision in decisions if decision.counted
        ]
        truth_ms = _truth_times(group)
        matches, matched_truth = greedy_time_match(
            counted_ms, truth_ms, tolerance_ms
        )
        tp = len(matches)
        sessions.append(
            {
                "session_id": str(session_id),
                "device_id": str(group.iloc[0]["device_id"]),
                "scenario_id": str(group.iloc[0]["scenario_id"]),
                "truth": len(truth_ms),
                "counted": len(counted_ms),
                "tp": tp,
                "fp": len(counted_ms) - tp,
                "fn": len(truth_ms) - len(matched_truth),
                "absolute_count_error": abs(len(counted_ms) - len(truth_ms)),
            }
        )
    tp = sum(int(row["tp"]) for row in sessions)
    fp = sum(int(row["fp"]) for row in sessions)
    fn = sum(int(row["fn"]) for row in sessions)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "sessions": sessions,
        "aggregate": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": (
                2.0 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            ),
            "mean_absolute_count_error": float(
                np.mean([row["absolute_count_error"] for row in sessions])
            ),
        },
    }


def _oracle_threshold_diagnostic(
    metadata: pd.DataFrame,
    candidate_rows: list[dict[str, object]],
    *,
    tolerance_ms: float,
) -> dict[str, object]:
    """Return a held-label oracle upper bound, never a deployable threshold."""
    best: tuple[float, float, float, dict[str, object]] | None = None
    for threshold in np.linspace(0.0, 0.99, 100):
        config = FableTimingConfig(
            quiet_confidence=float(threshold),
            loud_confidence=float(threshold),
            fast_rebound_confidence=float(threshold),
        )
        report = _score_counting(
            metadata,
            candidate_rows,
            tolerance_ms=tolerance_ms,
            timing_config=config,
        )
        aggregate = report["aggregate"]
        rank = (
            float(aggregate["f1"]),
            -float(aggregate["mean_absolute_count_error"]),
            float(threshold),
        )
        if best is None or rank > best[:3]:
            best = (*rank, report)
    if best is None:
        raise AssertionError("Oracle threshold sweep produced no result")
    return {
        "warning": (
            "Diagnostic upper bound only: the threshold was selected using "
            "held-device labels and must not be deployed."
        ),
        "threshold": best[2],
        "counting": best[3],
    }


def _aggregate_count_reports(folds: list[dict[str, object]], key: str) -> dict[str, object]:
    sessions = [
        session
        for fold in folds
        for session in fold[key]["sessions"]
    ]
    tp = sum(int(row["tp"]) for row in sessions)
    fp = sum(int(row["fp"]) for row in sessions)
    fn = sum(int(row["fn"]) for row in sessions)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "sessions": sessions,
        "aggregate": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": (
                2.0 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            ),
            "mean_absolute_count_error": float(
                np.mean([row["absolute_count_error"] for row in sessions])
            ),
        },
    }


def run_reviewed_device_lodo(
    base_cache: Path,
    review_cache: Path,
    output_dir: Path,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    tolerance_ms: float,
) -> dict[str, object]:
    """Run strict three-phone holdout folds against the frozen Fable path."""
    base_metadata = pd.read_csv(base_cache / "metadata.csv")
    review_metadata = pd.read_csv(review_cache / "metadata.csv")
    required_review_columns = {
        "background_db",
        "fable_predicted_label",
        "fable_confidence",
        "reviewed_event_ms",
    }
    missing = sorted(required_review_columns - set(review_metadata.columns))
    if missing:
        raise ValueError(f"Review cache is missing required columns: {missing}")

    base_label_values = base_metadata["label"].map(CLASS_TO_INDEX)
    base_labels = base_label_values.fillna(-1).to_numpy(dtype=np.int64)
    base_trainable = _trainable_mask(base_metadata)
    colleague = base_metadata["source"].eq("legacy:colleague_legacy").to_numpy()
    development_indexes = np.flatnonzero(base_trainable & colleague)
    groups = base_metadata["base_session_id"].astype(str).to_numpy()
    base_train_indexes, base_val_indexes = _choose_grouped_split(
        base_labels, groups, development_indexes, seed
    )

    base_features = np.load(base_cache / "logmel.npy", mmap_mode="r")
    review_features = np.load(review_cache / "logmel.npy", mmap_mode="r")
    selected_base_indexes = np.concatenate([base_train_indexes, base_val_indexes])
    combined_features = np.concatenate(
        [
            np.asarray(base_features[selected_base_indexes], dtype=np.float16),
            np.asarray(review_features, dtype=np.float16),
        ],
        axis=0,
    )
    review_labels = (
        review_metadata["label"].map(CLASS_TO_INDEX).fillna(-1).to_numpy(dtype=np.int64)
    )
    combined_labels = np.concatenate(
        [base_labels[selected_base_indexes], review_labels]
    )
    base_train_local = np.arange(len(base_train_indexes), dtype=np.int64)
    base_val_local = np.arange(
        len(base_train_indexes), len(selected_base_indexes), dtype=np.int64
    )
    review_offset = len(selected_base_indexes)
    devices = sorted(review_metadata["device_id"].astype(str).unique())
    if len(devices) < 2:
        raise ValueError("At least two reviewed devices are required")

    output_dir.mkdir(parents=True, exist_ok=True)
    folds: list[dict[str, object]] = []
    candidate_predictions: list[pd.DataFrame] = []
    for fold_number, held_device in enumerate(devices):
        review_train, held_all = review_fold_indexes(review_metadata, held_device)
        held_trainable = held_all[_trainable_mask(review_metadata)[held_all]]
        train_indexes = np.concatenate(
            [base_train_local, review_offset + review_train]
        )
        external_indexes = review_offset + held_trainable
        model, training_report, _, external_probabilities = _train_one(
            f"logmel_holdout_{held_device}",
            combined_features,
            combined_labels,
            train_indexes,
            base_val_local,
            external_indexes,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed + fold_number,
        )
        torch.save(
            {"state_dict": model.state_dict(), "classes": FOUR_CLASSES},
            output_dir / f"logmel_holdout_{fold_number + 1}.pt",
        )
        held_predictions = np.argmax(external_probabilities, axis=1)
        binary_metrics = _binary_racket_metrics(
            review_labels[held_trainable], held_predictions
        )

        dummy_labels = combined_labels.copy()
        dummy_labels[dummy_labels < 0] = 0
        held_all_local = review_offset + held_all
        all_predictions, all_probabilities = _predict(
            model,
            combined_features,
            dummy_labels,
            held_all_local,
            batch_size,
        )
        held_metadata = review_metadata.iloc[held_all].reset_index(drop=True)
        cnn_timing_rows = _candidate_rows_for_timing(
            held_metadata, all_predictions, all_probabilities
        )
        cnn_counting = _score_counting(
            held_metadata,
            cnn_timing_rows,
            tolerance_ms=tolerance_ms,
        )
        cnn_oracle = _oracle_threshold_diagnostic(
            held_metadata,
            cnn_timing_rows,
            tolerance_ms=tolerance_ms,
        )
        fable_counting = _score_counting(
            held_metadata,
            _fable_rows_for_timing(held_metadata),
            tolerance_ms=tolerance_ms,
        )
        train_devices = sorted(
            review_metadata.iloc[review_train]["device_id"].astype(str).unique()
        )
        if held_device in train_devices:
            raise AssertionError("Held device leaked into the reviewed training rows")
        folds.append(
            {
                "held_device": held_device,
                "review_train_devices": train_devices,
                "review_train_rows": int(len(review_train)),
                "held_inference_rows": int(len(held_all)),
                "held_trainable_rows": int(len(held_trainable)),
                "candidate_racket_binary": binary_metrics,
                "cnn_counting": cnn_counting,
                "cnn_oracle_threshold_diagnostic": cnn_oracle,
                "frozen_fable_counting": fable_counting,
                "training": training_report,
            }
        )
        prediction_frame = held_metadata.copy()
        prediction_frame["held_device"] = held_device
        prediction_frame["cnn_predicted_label"] = [
            FOUR_CLASSES[index] for index in all_predictions
        ]
        prediction_frame["cnn_confidence"] = np.max(all_probabilities, axis=1)
        for class_index, class_name in enumerate(FOUR_CLASSES):
            prediction_frame[f"cnn_prob_{class_name}"] = all_probabilities[:, class_index]
        candidate_predictions.append(prediction_frame)

    report: dict[str, object] = {
        "experiment": "reviewed_hf_candidate_logmel_cnn_device_holdout",
        "classes": list(FOUR_CLASSES),
        "seed": seed,
        "split_policy": {
            "base_training_source": "legacy:colleague_legacy only",
            "base_validation": "grouped session fold",
            "reviewed_training": "two of three physical phones per fold",
            "external_evaluation": "third physical phone, entirely unseen",
            "excluded": (
                "all old Motorola and STIGA native rows plus ambiguous reviewed "
                "candidates are excluded from training"
            ),
            "limitation": (
                "The legacy colleague corpus has no physical-device identity. "
                "This is strict holdout for the three reviewed phones, not a claim "
                "that every historical recorder is device-disjoint."
            ),
        },
        "row_counts": {
            "base_train": int(len(base_train_indexes)),
            "base_validation": int(len(base_val_indexes)),
            "review_candidates": int(len(review_metadata)),
        },
        "folds": folds,
        "cnn_counting": _aggregate_count_reports(folds, "cnn_counting"),
        "frozen_fable_counting": _aggregate_count_reports(
            folds, "frozen_fable_counting"
        ),
    }
    report["comparison"] = {
        "cnn_minus_fable_f1": (
            report["cnn_counting"]["aggregate"]["f1"]
            - report["frozen_fable_counting"]["aggregate"]["f1"]
        ),
        "cnn_minus_fable_mae": (
            report["cnn_counting"]["aggregate"]["mean_absolute_count_error"]
            - report["frozen_fable_counting"]["aggregate"][
                "mean_absolute_count_error"
            ]
        ),
    }
    pd.concat(candidate_predictions, ignore_index=True).to_csv(
        output_dir / "candidate_predictions.csv", index=False
    )
    (output_dir / "device_lodo_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--review-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance-ms", type=float, default=140.0)
    args = parser.parse_args()
    report = run_reviewed_device_lodo(
        args.base_cache,
        args.review_cache,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        tolerance_ms=args.tolerance_ms,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
