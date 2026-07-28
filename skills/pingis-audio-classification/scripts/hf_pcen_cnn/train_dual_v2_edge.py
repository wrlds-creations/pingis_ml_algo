"""Retrain Dual V2 with reviewed edge-racket contacts and compare it to Dual V1."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from .audit_review_pack import greedy_time_match
from .candidate_labels import FOUR_CLASSES
from .evaluate_cross_day_cnn import MODEL_FACTORIES
from .evaluate_final_holdout import _score_model, _train_cnn_fixed
from .fable_baseline import replay_deployed_fable_timing
from .train_ablation import CLASS_TO_INDEX, _metric_block
from .train_round_lodo import (
    NO_CONFIDENCE_GATE,
    RACKET_INDEX,
    TRAINABLE_DISPOSITIONS,
    _predict_cnn,
    _thresholded_rows,
)


MODEL_NAME = "dual_residual"
MODEL_VERSION = "dual_residual_v2_edge_racket"
DEFAULT_THRESHOLD = 0.775
DEFAULT_EPOCHS = 13
DEFAULT_SEED = 20260727


class ConcatenatedFeature:
    """Expose multiple memory-mapped feature arrays as one indexed sequence."""

    def __init__(self, arrays: Sequence[np.ndarray]) -> None:
        if not arrays:
            raise ValueError("At least one feature array is required")
        trailing_shape = tuple(arrays[0].shape[1:])
        if any(tuple(array.shape[1:]) != trailing_shape for array in arrays):
            raise ValueError("Feature arrays do not share a trailing shape")
        self.arrays = tuple(arrays)
        self.offsets = np.cumsum([0, *(len(array) for array in arrays)])
        self.shape = (int(self.offsets[-1]), *trailing_shape)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, index: int) -> np.ndarray:
        normalized = int(index)
        if normalized < 0:
            normalized += len(self)
        if normalized < 0 or normalized >= len(self):
            raise IndexError(index)
        array_index = int(np.searchsorted(self.offsets, normalized, side="right") - 1)
        return self.arrays[array_index][normalized - int(self.offsets[array_index])]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_torch_artifact(path: Path) -> dict[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_cache(cache_dir: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    metadata = pd.read_csv(cache_dir / "metadata.csv")
    logmel = np.load(cache_dir / "logmel.npy", mmap_mode="r")
    pcen = np.load(cache_dir / "pcen.npy", mmap_mode="r")
    if len(metadata) != len(logmel) or len(metadata) != len(pcen):
        raise ValueError(f"Cache row mismatch in {cache_dir}")
    if not metadata["cache_complete"].astype(bool).all():
        raise ValueError(f"Incomplete feature rows in {cache_dir}")
    return metadata, logmel, pcen


def _labels(metadata: pd.DataFrame) -> np.ndarray:
    return (
        metadata["label"]
        .map(CLASS_TO_INDEX)
        .fillna(-1)
        .to_numpy(dtype=np.int64)
    )


def _trainable(metadata: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    return (
        metadata["cache_complete"].astype(bool).to_numpy()
        & metadata["disposition"].isin(TRAINABLE_DISPOSITIONS).to_numpy()
        & (labels >= 0)
    )


def _edge_truth_times(group: pd.DataFrame) -> list[float]:
    racket = group.loc[
        group["disposition"].eq("positive")
        & group["label"].eq("racket_bounce"),
        ["matched_event_index", "reviewed_event_ms"],
    ].dropna()
    if racket.empty:
        return []
    return [
        float(value)
        for value in (
            racket.sort_values("matched_event_index")
            .drop_duplicates("matched_event_index")["reviewed_event_ms"]
            .tolist()
        )
    ]


def _score_edge_counting(
    metadata: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    tolerance_ms: float,
) -> dict[str, object]:
    evaluated = metadata.reset_index(drop=True).copy()
    evaluated["timing_row"] = _thresholded_rows(
        evaluated,
        probabilities,
        threshold,
    )
    sessions: list[dict[str, object]] = []
    for session_id, group in evaluated.groupby("session_id", sort=True):
        decisions = replay_deployed_fable_timing(
            list(group["timing_row"]),
            NO_CONFIDENCE_GATE,
        )
        counted_ms = [decision.onset_ms for decision in decisions if decision.counted]
        truth_ms = _edge_truth_times(group)
        matches, matched_truth = greedy_time_match(counted_ms, truth_ms, tolerance_ms)
        tp = len(matches)
        sessions.append(
            {
                "session_id": str(session_id),
                "take_id": str(group.iloc[0]["take_id"]),
                "device_alias": str(group.iloc[0]["device_alias"]),
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


def _threshold_binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    truth = labels == RACKET_INDEX
    predicted = probabilities[:, RACKET_INDEX] >= threshold
    return {
        "rows": int(len(labels)),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(
            truth,
            predicted,
            labels=[False, True],
        ).tolist(),
    }


def _diagnostic_breakdown(
    metadata: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> list[dict[str, object]]:
    evaluated = metadata.reset_index(drop=True).copy()
    evaluated["racket_probability"] = probabilities[:, RACKET_INDEX]
    reviewed = evaluated.loc[
        evaluated["disposition"].eq("positive") & evaluated["label"].notna()
    ].copy()
    reviewed["truth_racket"] = reviewed["label"].eq("racket_bounce")
    reviewed["predicted_racket"] = reviewed["racket_probability"].ge(threshold)
    columns = ["take_id", "device_alias", "racket_id", "rubber_face", "impact_zone"]
    output: list[dict[str, object]] = []
    for keys, group in reviewed.groupby(columns, dropna=False, sort=True):
        truth = group["truth_racket"].to_numpy(dtype=bool)
        predicted = group["predicted_racket"].to_numpy(dtype=bool)
        output.append(
            {
                **{
                    column: (
                        None if pd.isna(value) else str(value)
                    )
                    for column, value in zip(columns, keys, strict=True)
                },
                "rows": int(len(group)),
                "racket_truth": int(truth.sum()),
                "predicted_racket": int(predicted.sum()),
                "racket_recall": (
                    float((predicted & truth).sum() / truth.sum())
                    if truth.sum()
                    else None
                ),
                "table_false_positive_rate": (
                    float(predicted[~truth].mean()) if (~truth).sum() else None
                ),
                "mean_racket_probability": float(group["racket_probability"].mean()),
            }
        )
    return output


def _score_edge_diagnostic(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    tolerance_ms: float,
) -> dict[str, object]:
    trainable = _trainable(metadata, labels)
    indexes = np.flatnonzero(trainable)
    reviewed_labels = labels[indexes]
    reviewed_probabilities = probabilities[indexes]
    return {
        "candidate_four_class": _metric_block(
            reviewed_labels,
            reviewed_probabilities.argmax(axis=1),
        ),
        "candidate_racket_binary_fixed_threshold": _threshold_binary_metrics(
            reviewed_labels,
            reviewed_probabilities,
            threshold,
        ),
        "counting": _score_edge_counting(
            metadata,
            probabilities,
            threshold,
            tolerance_ms,
        ),
        "breakdown": _diagnostic_breakdown(metadata, probabilities, threshold),
    }


def _prediction_frame(
    metadata: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    model_version: str,
    threshold: float,
) -> pd.DataFrame:
    columns = [
        "session_id",
        "take_id",
        "device_alias",
        "racket_id",
        "rubber_face",
        "impact_zone",
        "onset_ms",
        "disposition",
        "label",
        "matched_event_index",
        "reviewed_event_ms",
    ]
    available = [column for column in columns if column in metadata.columns]
    output = metadata[available].reset_index(drop=True).copy()
    output["model"] = model_version
    for class_index, class_name in enumerate(FOUR_CLASSES):
        output[f"prob_{class_name}"] = probabilities[:, class_index]
    output["threshold"] = threshold
    output["predicted_racket"] = probabilities[:, RACKET_INDEX] >= threshold
    return output


def train_dual_v2_edge(
    base_cache: Path,
    edge_cache: Path,
    frozen_artifact_path: Path,
    output_dir: Path,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    threshold: float,
    tolerance_ms: float,
) -> dict[str, object]:
    base_metadata, base_logmel, base_pcen = _load_cache(base_cache)
    edge_metadata, edge_logmel, edge_pcen = _load_cache(edge_cache)
    if set(edge_metadata["dataset_split"].astype(str)) != {"train", "diagnostic"}:
        raise ValueError("Edge cache must contain train and diagnostic splits")

    base_labels = _labels(base_metadata)
    edge_labels = _labels(edge_metadata)
    combined_metadata = pd.concat(
        (
            base_metadata.assign(data_origin="original_round"),
            edge_metadata.assign(data_origin="edge_round"),
        ),
        ignore_index=True,
        sort=False,
    )
    combined_labels = np.concatenate((base_labels, edge_labels))
    base_rows = len(base_metadata)
    base_trainable = _trainable(base_metadata, base_labels)
    edge_trainable = _trainable(edge_metadata, edge_labels)
    base_train = base_metadata["dataset_split"].astype(str).eq("train").to_numpy()
    edge_train = edge_metadata["dataset_split"].astype(str).eq("train").to_numpy()
    train_indexes = np.concatenate(
        (
            np.flatnonzero(base_train & base_trainable),
            base_rows + np.flatnonzero(edge_train & edge_trainable),
        )
    )
    if not len(train_indexes):
        raise ValueError("No trainable rows")

    combined_features = (
        ConcatenatedFeature((base_logmel, edge_logmel)),
        ConcatenatedFeature((base_pcen, edge_pcen)),
    )
    model, losses = _train_cnn_fixed(
        MODEL_FACTORIES[MODEL_NAME](),
        combined_features,
        combined_labels,
        train_indexes,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        augment=True,
    )

    frozen_artifact = _load_torch_artifact(frozen_artifact_path)
    frozen_model = MODEL_FACTORIES[MODEL_NAME]()
    frozen_model.load_state_dict(frozen_artifact["state_dict"])
    frozen_model.eval()

    base_holdout = base_metadata["dataset_split"].astype(str).eq("holdout").to_numpy()
    base_holdout_all = np.flatnonzero(base_holdout)
    base_holdout_trainable = np.flatnonzero(base_holdout & base_trainable)
    edge_diagnostic = edge_metadata["dataset_split"].astype(str).eq("diagnostic").to_numpy()
    edge_diagnostic_all = np.flatnonzero(edge_diagnostic)

    frozen_holdout_probabilities = _predict_cnn(
        frozen_model,
        (base_logmel, base_pcen),
        base_labels,
        base_holdout_all,
        batch_size,
    )
    v2_holdout_probabilities = _predict_cnn(
        model,
        (base_logmel, base_pcen),
        base_labels,
        base_holdout_all,
        batch_size,
    )
    frozen_edge_probabilities = _predict_cnn(
        frozen_model,
        (edge_logmel, edge_pcen),
        edge_labels,
        edge_diagnostic_all,
        batch_size,
    )
    v2_edge_probabilities = _predict_cnn(
        model,
        (edge_logmel, edge_pcen),
        edge_labels,
        edge_diagnostic_all,
        batch_size,
    )
    diagnostic_metadata = edge_metadata.iloc[edge_diagnostic_all].reset_index(drop=True)
    diagnostic_labels = edge_labels[edge_diagnostic_all]

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "dual_residual_v2_edge_racket.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": FOUR_CLASSES,
            "feature_names": ("logmel", "pcen"),
            "threshold": threshold,
            "fixed_epochs": epochs,
            "seed": seed,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "training_policy": (
                "Fresh fit on original Train plus reviewed edge-round T01/T02; "
                "T03/T04 diagnostic only; original final holdout unchanged."
            ),
        },
        artifact_path,
    )

    edge_contract_path = edge_cache.parent / "candidates.contract.json"
    report: dict[str, object] = {
        "experiment": MODEL_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "policy": {
            "threshold_tuning": False,
            "fixed_threshold": threshold,
            "fixed_epochs": epochs,
            "seed": seed,
            "tolerance_ms": tolerance_ms,
            "original_holdout_modified": False,
            "edge_training_takes": ["T01", "T02"],
            "edge_diagnostic_takes": ["T03", "T04"],
        },
        "data": {
            "base_cache": str(base_cache.resolve()),
            "base_metadata_sha256": _sha256(base_cache / "metadata.csv"),
            "edge_cache": str(edge_cache.resolve()),
            "edge_metadata_sha256": _sha256(edge_cache / "metadata.csv"),
            "edge_contract": str(edge_contract_path.resolve()),
            "edge_contract_sha256": _sha256(edge_contract_path),
            "combined_rows": int(len(combined_metadata)),
            "training_rows": int(len(train_indexes)),
            "training_class_counts": {
                FOUR_CLASSES[class_index]: int(
                    (combined_labels[train_indexes] == class_index).sum()
                )
                for class_index in range(len(FOUR_CLASSES))
            },
            "added_edge_training_rows": int((edge_train & edge_trainable).sum()),
            "edge_diagnostic_rows": int(len(edge_diagnostic_all)),
        },
        "artifact": {
            "path": str(artifact_path.resolve()),
            "sha256": _sha256(artifact_path),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "losses": losses,
        },
        "frozen_v1": {
            "artifact": str(frozen_artifact_path.resolve()),
            "artifact_sha256": _sha256(frozen_artifact_path),
            "original_holdout": _score_model(
                base_metadata,
                base_labels,
                base_holdout_all,
                base_holdout_trainable,
                frozen_holdout_probabilities,
                threshold,
                tolerance_ms=tolerance_ms,
            ),
            "edge_diagnostic": _score_edge_diagnostic(
                diagnostic_metadata,
                diagnostic_labels,
                frozen_edge_probabilities,
                threshold,
                tolerance_ms,
            ),
        },
        "dual_v2": {
            "original_holdout": _score_model(
                base_metadata,
                base_labels,
                base_holdout_all,
                base_holdout_trainable,
                v2_holdout_probabilities,
                threshold,
                tolerance_ms=tolerance_ms,
            ),
            "edge_diagnostic": _score_edge_diagnostic(
                diagnostic_metadata,
                diagnostic_labels,
                v2_edge_probabilities,
                threshold,
                tolerance_ms,
            ),
        },
    }
    report_path = output_dir / "dual_v2_edge_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    predictions = pd.concat(
        (
            _prediction_frame(
                diagnostic_metadata,
                frozen_edge_probabilities,
                model_version="dual_residual_v1_frozen",
                threshold=threshold,
            ),
            _prediction_frame(
                diagnostic_metadata,
                v2_edge_probabilities,
                model_version=MODEL_VERSION,
                threshold=threshold,
            ),
        ),
        ignore_index=True,
    )
    predictions.to_csv(output_dir / "edge_diagnostic_predictions.csv", index=False)
    return report


def _summary(report: dict[str, object]) -> dict[str, object]:
    output: dict[str, object] = {
        "artifact": report["artifact"]["path"],
        "training_rows": report["data"]["training_rows"],
        "added_edge_training_rows": report["data"]["added_edge_training_rows"],
        "models": {},
    }
    for key in ("frozen_v1", "dual_v2"):
        result = report[key]
        output["models"][key] = {
            "edge_candidate": result["edge_diagnostic"][
                "candidate_racket_binary_fixed_threshold"
            ],
            "edge_counting": result["edge_diagnostic"]["counting"]["aggregate"],
            "holdout_candidate_f1": result["original_holdout"][
                "candidate_racket_binary"
            ]["f1"],
            "holdout_counting": result["original_holdout"]["counting"]["aggregate"],
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--edge-cache", type=Path, required=True)
    parser.add_argument("--frozen-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--tolerance-ms", type=float, default=140.0)
    args = parser.parse_args()
    report = train_dual_v2_edge(
        args.base_cache,
        args.edge_cache,
        args.frozen_artifact,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        threshold=args.threshold,
        tolerance_ms=args.tolerance_ms,
    )
    print(json.dumps(_summary(report), indent=2))


if __name__ == "__main__":
    main()
