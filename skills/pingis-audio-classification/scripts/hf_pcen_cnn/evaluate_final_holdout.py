"""Fit frozen finalists on Train data and score the final holdout exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from torch import nn
from torch.utils.data import DataLoader

from .candidate_labels import FOUR_CLASSES
from .evaluate_cross_day_cnn import MODEL_FACTORIES
from .train_ablation import CLASS_TO_INDEX, _metric_block
from .train_reviewed_device_lodo import (
    _fable_rows_for_timing,
    _score_counting,
)
from .train_round_lodo import (
    NO_CONFIDENCE_GATE,
    RACKET_INDEX,
    TRAINABLE_DISPOSITIONS,
    MultiFrontendDataset,
    _augment_spectrograms,
    _binary_metrics,
    _class_weights,
    _feature_matrix,
    _model_probabilities,
    _predict_cnn,
    _seed_everything,
    _thresholded_rows,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _train_cnn_fixed(
    model: nn.Module,
    features: tuple[np.ndarray, ...],
    labels: np.ndarray,
    train_indexes: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    augment: bool,
) -> tuple[nn.Module, list[float]]:
    _seed_everything(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=8e-4,
        weight_decay=2e-4,
    )
    criterion = nn.CrossEntropyLoss(
        weight=_class_weights(labels[train_indexes])
    )
    loader = DataLoader(
        MultiFrontendDataset(features, labels, train_indexes),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    losses: list[float] = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for inputs, targets in loader:
            if augment:
                inputs = _augment_spectrograms(inputs)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        mean_loss = float(np.mean(epoch_losses))
        losses.append(mean_loss)
        print(
            f"final epoch {epoch:02d}/{epochs:02d}: loss={mean_loss:.4f}",
            flush=True,
        )
    return model, losses


def _score_model(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    holdout_all: np.ndarray,
    holdout_trainable: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    *,
    tolerance_ms: float,
) -> dict[str, object]:
    holdout_metadata = metadata.iloc[holdout_all].reset_index(drop=True)
    trainable_mask = np.isin(holdout_all, holdout_trainable)
    trainable_labels = labels[holdout_all][trainable_mask]
    trainable_probabilities = probabilities[trainable_mask]
    predictions = trainable_probabilities.argmax(axis=1)
    return {
        "candidate_four_class": _metric_block(
            trainable_labels,
            predictions,
        ),
        "candidate_racket_binary": _binary_metrics(
            trainable_labels,
            trainable_probabilities,
        ),
        "counting": _score_counting(
            holdout_metadata,
            _thresholded_rows(
                holdout_metadata,
                probabilities,
                threshold,
            ),
            tolerance_ms=tolerance_ms,
            timing_config=NO_CONFIDENCE_GATE,
        ),
    }


def evaluate_final_holdout(
    cache_dir: Path,
    finalist_plan_path: Path,
    output_dir: Path,
    *,
    tolerance_ms: float,
    batch_size: int,
) -> dict[str, object]:
    report_path = output_dir / "final_holdout_report.json"
    if report_path.exists():
        raise FileExistsError(
            f"Refusing to rescore sealed holdout: {report_path}"
        )
    plan = json.loads(finalist_plan_path.read_text(encoding="utf-8"))
    metadata_path = cache_dir / "metadata.csv"
    metadata = pd.read_csv(metadata_path)
    labels = (
        metadata["label"]
        .map(CLASS_TO_INDEX)
        .fillna(-1)
        .to_numpy(dtype=np.int64)
    )
    cache_complete = metadata["cache_complete"].astype(bool).to_numpy()
    trainable = (
        cache_complete
        & metadata["disposition"].isin(TRAINABLE_DISPOSITIONS).to_numpy()
        & (labels >= 0)
    )
    train_split = metadata["dataset_split"].astype(str).eq("train").to_numpy()
    holdout_split = (
        metadata["dataset_split"].astype(str).eq("holdout").to_numpy()
    )
    train_indexes = np.flatnonzero(train_split & trainable)
    holdout_trainable = np.flatnonzero(holdout_split & trainable)
    holdout_all = np.flatnonzero(holdout_split)
    if not len(train_indexes) or not len(holdout_trainable):
        raise ValueError("Train or final-holdout rows are empty")

    frontends = {
        "logmel": np.load(cache_dir / "logmel.npy", mmap_mode="r"),
        "pcen": np.load(cache_dir / "pcen.npy", mmap_mode="r"),
    }
    tabular_features, tabular_feature_names = _feature_matrix(metadata)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    reports: dict[str, dict[str, object]] = {}
    prediction_rows: list[pd.DataFrame] = []
    for candidate_number, candidate in enumerate(plan["candidates"], start=1):
        model_name = str(candidate["model"])
        threshold = float(candidate["threshold"])
        print(f"Fitting frozen finalist {model_name}", flush=True)
        if candidate["family"] == "cnn":
            feature_names = tuple(candidate["feature_names"])
            selected_frontends = tuple(
                frontends[name]
                for name in feature_names
            )
            model, losses = _train_cnn_fixed(
                MODEL_FACTORIES[model_name](),
                selected_frontends,
                labels,
                train_indexes,
                epochs=int(candidate["fixed_epochs"]),
                batch_size=batch_size,
                seed=int(plan["seed"]) + candidate_number,
                augment=bool(candidate["augment"]),
            )
            probabilities = _predict_cnn(
                model,
                selected_frontends,
                labels,
                holdout_all,
                batch_size,
            )
            artifact_path = artifacts_dir / f"{model_name}.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": FOUR_CLASSES,
                    "feature_names": feature_names,
                    "threshold": threshold,
                    "fixed_epochs": int(candidate["fixed_epochs"]),
                    "seed": int(plan["seed"]) + candidate_number,
                },
                artifact_path,
            )
            training = {
                "fixed_epochs": int(candidate["fixed_epochs"]),
                "losses": losses,
                "parameter_count": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                ),
            }
        elif candidate["family"] == "histgb":
            if list(candidate["feature_names"]) != tabular_feature_names:
                raise ValueError(
                    f"Tabular feature contract mismatch for {model_name}"
                )
            model = HistGradientBoostingClassifier(
                **candidate["config"],
                class_weight="balanced",
                random_state=int(plan["seed"]) + candidate_number,
            )
            model.fit(
                tabular_features[train_indexes],
                labels[train_indexes],
            )
            probabilities = _model_probabilities(
                model,
                tabular_features,
                holdout_all,
                binary_target=False,
            )
            artifact_path = artifacts_dir / f"{model_name}.joblib"
            joblib.dump(
                {
                    "model": model,
                    "classes": FOUR_CLASSES,
                    "feature_names": tabular_feature_names,
                    "threshold": threshold,
                    "config": candidate["config"],
                    "seed": int(plan["seed"]) + candidate_number,
                },
                artifact_path,
            )
            training = {"config": candidate["config"]}
        else:
            raise ValueError(
                f"Unsupported finalist family: {candidate['family']}"
            )

        reports[model_name] = {
            "family": candidate["family"],
            "threshold": threshold,
            "artifact": str(artifact_path.resolve()),
            "artifact_sha256": _sha256(artifact_path),
            "training": training,
            **_score_model(
                metadata,
                labels,
                holdout_all,
                holdout_trainable,
                probabilities,
                threshold,
                tolerance_ms=tolerance_ms,
            ),
        }

        predictions = metadata.iloc[holdout_all][
            [
                "session_id",
                "device_id",
                "device_alias",
                "take_id",
                "scenario_id",
                "onset_ms",
                "disposition",
                "label",
            ]
        ].copy()
        predictions["model"] = model_name
        for class_index, class_name in enumerate(FOUR_CLASSES):
            predictions[f"prob_{class_name}"] = probabilities[:, class_index]
        predictions["threshold"] = threshold
        predictions["predicted_racket"] = (
            probabilities[:, RACKET_INDEX] >= threshold
        )
        prediction_rows.append(predictions)

    holdout_metadata = metadata.iloc[holdout_all].reset_index(drop=True)
    fable_counting = _score_counting(
        holdout_metadata,
        _fable_rows_for_timing(holdout_metadata),
        tolerance_ms=tolerance_ms,
    )
    reports["frozen_fable"] = {
        "family": "baseline",
        "counting": fable_counting,
    }

    leaderboard = []
    for model_name, model_report in reports.items():
        aggregate = model_report["counting"]["aggregate"]
        leaderboard.append(
            {
                "model": model_name,
                "counting_f1": aggregate["f1"],
                "counting_precision": aggregate["precision"],
                "counting_recall": aggregate["recall"],
                "mean_absolute_count_error": aggregate[
                    "mean_absolute_count_error"
                ],
                "candidate_macro_f1": (
                    model_report.get("candidate_four_class", {}).get(
                        "macro_f1"
                    )
                ),
                "candidate_racket_f1": (
                    model_report.get("candidate_racket_binary", {}).get("f1")
                ),
            }
        )
    leaderboard.sort(
        key=lambda row: (
            -float(row["counting_f1"]),
            float(row["mean_absolute_count_error"]),
            str(row["model"]),
        )
    )
    result: dict[str, object] = {
        "experiment": "frozen_final_holdout",
        "scored_at_utc": datetime.now(UTC).isoformat(),
        "finalist_plan": str(finalist_plan_path.resolve()),
        "finalist_plan_sha256": _sha256(finalist_plan_path),
        "metadata": str(metadata_path.resolve()),
        "metadata_sha256": _sha256(metadata_path),
        "policy": {
            "threshold_tuning_on_holdout": False,
            "hyperparameter_tuning_on_holdout": False,
            "rescore_guard": str(report_path.resolve()),
            "tolerance_ms": tolerance_ms,
        },
        "data": {
            "trainable_train_rows": int(len(train_indexes)),
            "trainable_holdout_rows": int(len(holdout_trainable)),
            "all_holdout_candidate_rows": int(len(holdout_all)),
            "holdout_take_ids": sorted(
                holdout_metadata["take_id"].astype(str).unique().tolist()
            ),
            "holdout_devices": sorted(
                holdout_metadata["device_id"].astype(str).unique().tolist()
            ),
        },
        "leaderboard": leaderboard,
        "models": reports,
    }
    report_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        output_dir / "final_holdout_candidate_predictions.csv",
        index=False,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--finalist-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tolerance-ms", type=float, default=140.0)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    report = evaluate_final_holdout(
        args.cache_dir,
        args.finalist_plan,
        args.output_dir,
        tolerance_ms=args.tolerance_ms,
        batch_size=args.batch_size,
    )
    print(json.dumps({"leaderboard": report["leaderboard"]}, indent=2))


if __name__ == "__main__":
    main()
