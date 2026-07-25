"""Compare strong four-class candidates with leave-one-device-out evaluation."""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .candidate_labels import FOUR_CLASSES
from .fable_baseline import FableTimingConfig
from .model import BounceCandidateCnn, ResidualBounceCandidateCnn
from .train_ablation import CLASS_TO_INDEX
from .train_reviewed_device_lodo import (
    _aggregate_count_reports,
    _fable_rows_for_timing,
    _score_counting,
)


RACKET_INDEX = CLASS_TO_INDEX["racket_bounce"]
NOISE_INDEX = CLASS_TO_INDEX["voice_music_noise"]
TRAINABLE_DISPOSITIONS = frozenset({"positive", "hard_negative"})
TABULAR_MODELS = (
    "extra_trees",
    "histgb",
    "extra_trees_binary",
    "histgb_binary",
)
FABLE_PROBABILITY_COLUMNS = (
    "fable_prob_racket_bounce",
    "fable_prob_table_bounce",
    "fable_prob_floor_bounce",
    "fable_prob_noise",
)
NO_CONFIDENCE_GATE = FableTimingConfig(
    quiet_confidence=0.0,
    loud_confidence=0.0,
    fast_rebound_confidence=0.0,
)


@dataclass(frozen=True)
class CnnSpec:
    name: str
    feature_names: tuple[str, ...]
    factory: Callable[[], nn.Module]
    augment: bool


class MultiFrontendDataset(Dataset):
    def __init__(
        self,
        features: tuple[np.ndarray, ...],
        labels: np.ndarray,
        indexes: np.ndarray,
    ) -> None:
        self.features = features
        self.labels = labels
        self.indexes = np.asarray(indexes, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(self.indexes[item])
        channels = [
            np.array(feature[index], dtype=np.float32, copy=True)
            for feature in self.features
        ]
        return (
            torch.from_numpy(np.stack(channels, axis=0)),
            torch.tensor(int(self.labels[index]), dtype=torch.long),
        )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=len(FOUR_CLASSES)).astype(np.float64)
    weights = counts.sum() / (len(FOUR_CLASSES) * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def _augment_spectrograms(inputs: torch.Tensor) -> torch.Tensor:
    output = inputs.clone()
    batch, channels, frequencies, frames = output.shape
    output += torch.randn_like(output) * 0.015
    for row in range(batch):
        if torch.rand(()) < 0.55:
            width = int(torch.randint(1, 7, ()).item())
            start = int(torch.randint(0, max(1, frequencies - width + 1), ()).item())
            output[row, :, start : start + width, :] = 0.0
        if torch.rand(()) < 0.55:
            width = int(torch.randint(1, 7, ()).item())
            start = int(torch.randint(0, max(1, frames - width + 1), ()).item())
            output[row, :, :, start : start + width] = 0.0
        if channels > 1 and torch.rand(()) < 0.10:
            channel = int(torch.randint(0, channels, ()).item())
            output[row, channel] = 0.0
    return output


@torch.no_grad()
def _predict_cnn(
    model: nn.Module,
    features: tuple[np.ndarray, ...],
    labels: np.ndarray,
    indexes: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(
        MultiFrontendDataset(features, labels, indexes),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model.eval()
    probabilities: list[np.ndarray] = []
    for inputs, _ in loader:
        probabilities.append(torch.softmax(model(inputs), dim=1).cpu().numpy())
    return np.concatenate(probabilities)


def _train_cnn(
    spec: CnnSpec,
    features: tuple[np.ndarray, ...],
    labels: np.ndarray,
    train_indexes: np.ndarray,
    validation_indexes: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[nn.Module, dict[str, object]]:
    _seed_everything(seed)
    model = spec.factory()
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(labels[train_indexes]))
    loader = DataLoader(
        MultiFrontendDataset(features, labels, train_indexes),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_score = -1.0
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for inputs, targets in loader:
            if spec.augment:
                inputs = _augment_spectrograms(inputs)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        probabilities = _predict_cnn(
            model,
            features,
            labels,
            validation_indexes,
            batch_size,
        )
        predictions = probabilities.argmax(axis=1)
        macro_f1 = float(
            f1_score(
                labels[validation_indexes],
                predictions,
                average="macro",
                zero_division=0,
            )
        )
        racket_truth = labels[validation_indexes] == RACKET_INDEX
        racket_prediction = predictions == RACKET_INDEX
        racket_f1 = float(
            f1_score(racket_truth, racket_prediction, zero_division=0)
        )
        score = macro_f1 + racket_f1
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_macro_f1": macro_f1,
                "validation_racket_f1": racket_f1,
            }
        )
        print(
            f"{spec.name} epoch {epoch:02d}: loss={np.mean(losses):.4f} "
            f"macro_f1={macro_f1:.4f} racket_f1={racket_f1:.4f}",
            flush=True,
        )
        if score > best_score + 1e-4:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= 5:
                break
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "history": history,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def _feature_matrix(metadata: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    feature_names = sorted(
        column for column in metadata.columns if column.startswith("fable_feature_")
    )
    values = metadata[feature_names].to_numpy(dtype=np.float64)
    fable_probability_names = list(FABLE_PROBABILITY_COLUMNS)
    missing_probability_names = [
        name for name in fable_probability_names if name not in metadata.columns
    ]
    if missing_probability_names:
        raise ValueError(
            "Missing frozen-Fable probability columns: "
            + ", ".join(missing_probability_names)
        )
    fable_values = metadata[fable_probability_names].to_numpy(dtype=np.float64)
    fable_confidence = (
        metadata["fable_confidence"].to_numpy(dtype=np.float64).reshape(-1, 1)
    )
    eps = 1e-12
    derived = np.column_stack(
        (
            np.log10(np.maximum(metadata["hf_rms"].to_numpy(float), eps)),
            np.log10(np.maximum(metadata["background_rms"].to_numpy(float), eps)),
            np.log10(np.maximum(metadata["full_band_peak"].to_numpy(float), eps)),
            metadata["hf_rms"].to_numpy(float)
            / np.maximum(metadata["background_rms"].to_numpy(float), eps),
            metadata["hf_rms"].to_numpy(float)
            / np.maximum(metadata["threshold"].to_numpy(float), eps),
        )
    )
    derived_names = [
        "gate_log_hf_rms",
        "gate_log_background_rms",
        "gate_log_full_band_peak",
        "gate_hf_background_ratio",
        "gate_threshold_ratio",
    ]
    matrix = np.column_stack((values, fable_values, fable_confidence, derived))
    matrix = np.where(np.isfinite(matrix), matrix, 0.0)
    return (
        matrix,
        feature_names
        + fable_probability_names
        + ["fable_confidence"]
        + derived_names,
    )


def _binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    truth = labels == RACKET_INDEX
    predictions = probabilities.argmax(axis=1) == RACKET_INDEX
    return {
        "rows": int(len(labels)),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(
            truth,
            predictions,
            labels=[False, True],
        ).tolist(),
    }


def _thresholded_rows(
    metadata: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> list[dict[str, object]]:
    non_racket_indexes = [index for index in range(len(FOUR_CLASSES)) if index != RACKET_INDEX]
    rows: list[dict[str, object]] = []
    for (_, row), probability in zip(metadata.iterrows(), probabilities, strict=True):
        racket_probability = float(probability[RACKET_INDEX])
        if racket_probability >= threshold:
            label = "racket_bounce"
        else:
            label = FOUR_CLASSES[
                max(non_racket_indexes, key=lambda index: float(probability[index]))
            ]
        rows.append(
            {
                "onset_ms": float(row["onset_ms"]),
                "frame_rms": float(row["hf_rms"]),
                "background_db": float(row["background_db"]),
                "predicted_label": label,
                "confidence": 1.0,
                "racket_probability": racket_probability,
            }
        )
    return rows


def _select_count_threshold(
    metadata: pd.DataFrame,
    probabilities: np.ndarray,
    tolerance_ms: float,
) -> tuple[float, dict[str, object]]:
    best: tuple[tuple[float, float, float, float], float, dict[str, object]] | None = None
    for threshold in np.linspace(0.15, 0.95, 33):
        report = _score_counting(
            metadata,
            _thresholded_rows(metadata, probabilities, float(threshold)),
            tolerance_ms=tolerance_ms,
            timing_config=NO_CONFIDENCE_GATE,
        )
        aggregate = report["aggregate"]
        rank = (
            float(aggregate["f1"]),
            -float(aggregate["mean_absolute_count_error"]),
            float(aggregate["precision"]),
            -abs(float(threshold) - 0.5),
        )
        if best is None or rank > best[0]:
            best = (rank, float(threshold), report)
    if best is None:
        raise AssertionError("Threshold sweep produced no result")
    return best[1], best[2]


def _fit_tabular(
    name: str,
    features: np.ndarray,
    labels: np.ndarray,
    train_indexes: np.ndarray,
    seed: int,
) -> object:
    family = name.removesuffix("_binary")
    binary_target = name.endswith("_binary")
    target = (labels == RACKET_INDEX).astype(np.int64) if binary_target else labels
    if family == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    elif family == "histgb":
        model = HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=240,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.1,
            class_weight="balanced",
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown tabular model: {name}")
    return model.fit(features[train_indexes], target[train_indexes])


def _model_probabilities(
    model: object,
    features: np.ndarray,
    indexes: np.ndarray,
    *,
    binary_target: bool,
) -> np.ndarray:
    raw = np.asarray(model.predict_proba(features[indexes]), dtype=np.float64)
    output = np.zeros((len(indexes), len(FOUR_CLASSES)), dtype=np.float64)
    if binary_target:
        class_indexes = {int(label): index for index, label in enumerate(model.classes_)}
        racket_probabilities = raw[:, class_indexes[1]]
        output[:, RACKET_INDEX] = racket_probabilities
        output[:, NOISE_INDEX] = 1.0 - racket_probabilities
        return output
    for source_index, class_index in enumerate(model.classes_):
        output[:, int(class_index)] = raw[:, source_index]
    return output


def _choose_round_grouped_split(
    labels: np.ndarray,
    groups: np.ndarray,
    indexes: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep rare classes in fitting while creating a useful binary validation split."""
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    overall = np.bincount(labels[indexes], minlength=len(FOUR_CLASSES)) / len(indexes)
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for train_local, validation_local in splitter.split(
        indexes,
        labels[indexes],
        groups[indexes],
    ):
        train_indexes = indexes[train_local]
        validation_indexes = indexes[validation_local]
        train_counts = np.bincount(
            labels[train_indexes],
            minlength=len(FOUR_CLASSES),
        )
        if np.any(train_counts == 0):
            continue
        validation_labels = labels[validation_indexes]
        validation_has_racket = np.any(validation_labels == RACKET_INDEX)
        validation_has_non_racket = np.any(validation_labels != RACKET_INDEX)
        if not validation_has_racket or not validation_has_non_racket:
            continue
        validation_distribution = np.bincount(
            validation_labels,
            minlength=len(FOUR_CLASSES),
        ) / len(validation_indexes)
        distribution_error = float(np.abs(validation_distribution - overall).sum())
        size_error = abs(len(validation_indexes) / len(indexes) - 0.2)
        score = distribution_error + size_error
        if best is None or score < best[0]:
            best = (score, train_indexes, validation_indexes)
    if best is None:
        raise RuntimeError(
            "Could not create a grouped split with every class in fitting and "
            "both racket/non-racket candidates in validation"
        )
    return best[1], best[2]


def _fold_indexes(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    held_device: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_split = metadata["dataset_split"].astype(str).eq("train").to_numpy()
    trainable = (
        metadata["cache_complete"].astype(bool).to_numpy()
        & metadata["disposition"].isin(TRAINABLE_DISPOSITIONS).to_numpy()
        & (labels >= 0)
    )
    held = metadata["device_id"].astype(str).eq(held_device).to_numpy()
    development = np.flatnonzero(train_split & trainable & ~held)
    groups = metadata["physical_event_group_id"].astype(str).to_numpy()
    train_indexes, validation_trainable = _choose_round_grouped_split(
        labels,
        groups,
        development,
        seed,
    )
    validation_groups = set(groups[validation_trainable])
    validation_all = np.flatnonzero(
        train_split
        & ~held
        & metadata["physical_event_group_id"].astype(str).isin(validation_groups).to_numpy()
    )
    held_all = np.flatnonzero(train_split & held)
    held_trainable = held_all[trainable[held_all]]
    if set(groups[train_indexes]) & validation_groups:
        raise AssertionError("Physical take leaked between training and validation")
    return train_indexes, validation_trainable, validation_all, held_all


def _aggregate_binary(folds: list[dict[str, object]]) -> dict[str, float | int]:
    matrices = np.asarray(
        [fold["candidate_racket_binary"]["confusion_matrix"] for fold in folds],
        dtype=np.int64,
    ).sum(axis=0)
    tn, fp, fn, tp = matrices.ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
    }


def run_round_lodo(
    cache_dir: Path,
    output_dir: Path,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    tolerance_ms: float,
    include_cnns: bool = True,
) -> dict[str, object]:
    metadata = pd.read_csv(cache_dir / "metadata.csv")
    if set(metadata["dataset_split"].astype(str)) != {"train", "holdout"}:
        raise ValueError("Expected explicit train and holdout rows")
    labels = (
        metadata["label"].map(CLASS_TO_INDEX).fillna(-1).to_numpy(dtype=np.int64)
    )
    devices = sorted(metadata["device_id"].astype(str).unique())
    if len(devices) != 3:
        raise ValueError(f"Expected exactly three physical devices, got {devices}")

    frontends = (
        {
            "logmel": np.load(cache_dir / "logmel.npy", mmap_mode="r"),
            "pcen": np.load(cache_dir / "pcen.npy", mmap_mode="r"),
        }
        if include_cnns
        else {}
    )
    cnn_specs = (
        CnnSpec(
            "logmel_compact",
            ("logmel",),
            lambda: BounceCandidateCnn(num_classes=len(FOUR_CLASSES)),
            False,
        ),
        CnnSpec(
            "pcen_compact",
            ("pcen",),
            lambda: BounceCandidateCnn(num_classes=len(FOUR_CLASSES)),
            False,
        ),
        CnnSpec(
            "dual_residual",
            ("logmel", "pcen"),
            lambda: ResidualBounceCandidateCnn(
                num_classes=len(FOUR_CLASSES),
                input_channels=2,
            ),
            True,
        ),
    )
    tabular_features, tabular_feature_names = _feature_matrix(metadata)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict[str, object]] = {
        name: {"folds": []}
        for name in ("frozen_fable", *TABULAR_MODELS)
    }
    if include_cnns:
        reports.update({spec.name: {"folds": []} for spec in cnn_specs})

    for fold_number, held_device in enumerate(devices, start=1):
        train_indexes, validation_trainable, validation_all, held_all = _fold_indexes(
            metadata,
            labels,
            held_device,
            seed + fold_number,
        )
        held_trainable = held_all[
            metadata.iloc[held_all]["disposition"]
            .isin(TRAINABLE_DISPOSITIONS)
            .to_numpy()
        ]
        validation_metadata = metadata.iloc[validation_all].reset_index(drop=True)
        held_metadata = metadata.iloc[held_all].reset_index(drop=True)
        fold_common = {
            "held_device": held_device,
            "train_rows": int(len(train_indexes)),
            "validation_trainable_rows": int(len(validation_trainable)),
            "validation_candidate_rows": int(len(validation_all)),
            "held_trainable_rows": int(len(held_trainable)),
            "held_candidate_rows": int(len(held_all)),
            "validation_take_ids": sorted(
                validation_metadata["take_id"].astype(str).unique().tolist()
            ),
        }

        fable_counting = _score_counting(
            held_metadata,
            _fable_rows_for_timing(held_metadata),
            tolerance_ms=tolerance_ms,
        )
        reports["frozen_fable"]["folds"].append(
            {**fold_common, "counting": fable_counting}
        )

        for model_name in TABULAR_MODELS:
            binary_target = model_name.endswith("_binary")
            print(f"Training {model_name} with held device {held_device}", flush=True)
            model = _fit_tabular(
                model_name,
                tabular_features,
                labels,
                train_indexes,
                seed + fold_number,
            )
            validation_probabilities = _model_probabilities(
                model,
                tabular_features,
                validation_all,
                binary_target=binary_target,
            )
            threshold, validation_counting = _select_count_threshold(
                validation_metadata,
                validation_probabilities,
                tolerance_ms,
            )
            held_probabilities = _model_probabilities(
                model,
                tabular_features,
                held_all,
                binary_target=binary_target,
            )
            held_trainable_probabilities = _model_probabilities(
                model,
                tabular_features,
                held_trainable,
                binary_target=binary_target,
            )
            held_counting = _score_counting(
                held_metadata,
                _thresholded_rows(held_metadata, held_probabilities, threshold),
                tolerance_ms=tolerance_ms,
                timing_config=NO_CONFIDENCE_GATE,
            )
            joblib.dump(
                {
                    "model": model,
                    "feature_names": tabular_feature_names,
                    "classes": FOUR_CLASSES,
                    "binary_target": binary_target,
                    "threshold": threshold,
                },
                output_dir / f"{model_name}_holdout_{fold_number}.joblib",
            )
            reports[model_name]["folds"].append(
                {
                    **fold_common,
                    "threshold": threshold,
                    "validation_counting": validation_counting,
                    "candidate_racket_binary": _binary_metrics(
                        labels[held_trainable],
                        held_trainable_probabilities,
                    ),
                    "counting": held_counting,
                }
            )

        for spec in cnn_specs if include_cnns else ():
            print(f"Training {spec.name} with held device {held_device}", flush=True)
            selected_features = tuple(frontends[name] for name in spec.feature_names)
            model, training = _train_cnn(
                spec,
                selected_features,
                labels,
                train_indexes,
                validation_trainable,
                epochs=epochs,
                batch_size=batch_size,
                seed=seed + fold_number,
            )
            validation_probabilities = _predict_cnn(
                model,
                selected_features,
                labels,
                validation_all,
                batch_size,
            )
            threshold, validation_counting = _select_count_threshold(
                validation_metadata,
                validation_probabilities,
                tolerance_ms,
            )
            held_probabilities = _predict_cnn(
                model,
                selected_features,
                labels,
                held_all,
                batch_size,
            )
            trainable_local = np.flatnonzero(
                held_metadata["disposition"].isin(TRAINABLE_DISPOSITIONS).to_numpy()
            )
            held_counting = _score_counting(
                held_metadata,
                _thresholded_rows(held_metadata, held_probabilities, threshold),
                tolerance_ms=tolerance_ms,
                timing_config=NO_CONFIDENCE_GATE,
            )
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": FOUR_CLASSES,
                    "threshold": threshold,
                    "feature_names": spec.feature_names,
                },
                output_dir / f"{spec.name}_holdout_{fold_number}.pt",
            )
            reports[spec.name]["folds"].append(
                {
                    **fold_common,
                    "threshold": threshold,
                    "validation_counting": validation_counting,
                    "candidate_racket_binary": _binary_metrics(
                        labels[held_trainable],
                        held_probabilities[trainable_local],
                    ),
                    "counting": held_counting,
                    "training": training,
                }
            )

    leaderboard: list[dict[str, object]] = []
    for model_name, model_report in reports.items():
        folds = model_report["folds"]
        counting = _aggregate_count_reports(folds, "counting")
        model_report["counting"] = counting
        if model_name != "frozen_fable":
            model_report["candidate_racket_binary"] = _aggregate_binary(folds)
        aggregate = counting["aggregate"]
        leaderboard.append(
            {
                "model": model_name,
                "counting_f1": aggregate["f1"],
                "counting_precision": aggregate["precision"],
                "counting_recall": aggregate["recall"],
                "mean_absolute_count_error": aggregate[
                    "mean_absolute_count_error"
                ],
                "candidate_racket_f1": (
                    model_report.get("candidate_racket_binary", {}).get("f1")
                ),
            }
        )
    leaderboard.sort(
        key=lambda row: (
            -float(row["counting_f1"]),
            float(row["mean_absolute_count_error"]),
            row["model"],
        )
    )
    report: dict[str, object] = {
        "experiment": "reviewed_round_train_only_device_lodo",
        "classes": list(FOUR_CLASSES),
        "seed": seed,
        "selection_policy": {
            "eligible_rows": "dataset_split=train only",
            "held_device": "one stable install ID excluded from all fitting",
            "validation": (
                "physical take groups held out from both training devices; "
                "threshold selected on those groups only"
            ),
            "final_holdout": (
                "dataset_split=holdout rows are present in the cache but never "
                "selected by this script"
            ),
        },
        "tabular_features": tabular_feature_names,
        "include_cnns": include_cnns,
        "leaderboard": leaderboard,
        "models": reports,
    }
    (output_dir / "lodo_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance-ms", type=float, default=140.0)
    parser.add_argument(
        "--tabular-only",
        action="store_true",
        help="Run frozen Fable and tabular stacking candidates without CNN training.",
    )
    args = parser.parse_args()
    report = run_round_lodo(
        args.cache_dir,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        tolerance_ms=args.tolerance_ms,
        include_cnns=not args.tabular_only,
    )
    print(json.dumps({"leaderboard": report["leaderboard"]}, indent=2))


if __name__ == "__main__":
    main()
