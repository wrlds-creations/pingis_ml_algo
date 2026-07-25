"""Evaluate saved LODO CNN checkpoints on an earlier reviewed phone round."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn

from .candidate_labels import FOUR_CLASSES
from .evaluate_cross_day_stack import _device_contract, _device_key
from .model import BounceCandidateCnn, ResidualBounceCandidateCnn
from .train_ablation import CLASS_TO_INDEX
from .train_reviewed_device_lodo import (
    _aggregate_count_reports,
    _score_counting,
)
from .train_round_lodo import (
    NO_CONFIDENCE_GATE,
    RACKET_INDEX,
    _predict_cnn,
    _thresholded_rows,
)


MODEL_FACTORIES: dict[str, Callable[[], nn.Module]] = {
    "logmel_compact": lambda: BounceCandidateCnn(
        num_classes=len(FOUR_CLASSES)
    ),
    "pcen_compact": lambda: BounceCandidateCnn(
        num_classes=len(FOUR_CLASSES)
    ),
    "dual_residual": lambda: ResidualBounceCandidateCnn(
        num_classes=len(FOUR_CLASSES),
        input_channels=2,
    ),
}


def _load_frontends(
    cache_dir: Path,
    feature_names: tuple[str, ...],
) -> tuple[np.ndarray, ...]:
    unsupported = sorted(set(feature_names) - {"logmel", "pcen"})
    if unsupported:
        raise ValueError(f"Unsupported CNN frontends: {unsupported}")
    return tuple(
        np.load(cache_dir / f"{name}.npy", mmap_mode="r")
        for name in feature_names
    )


def _checkpoint_fold_by_device(
    model_report: dict[str, object],
) -> dict[str, int]:
    return {
        str(fold["held_device"]): fold_number
        for fold_number, fold in enumerate(model_report["folds"], start=1)
    }


def evaluate_cross_day_cnn(
    train_cache: Path,
    external_cache: Path,
    lodo_dir: Path,
    output_dir: Path,
    *,
    model_names: tuple[str, ...],
    tolerance_ms: float,
    batch_size: int,
) -> dict[str, object]:
    train_metadata = pd.read_csv(train_cache / "metadata.csv")
    external = pd.read_csv(external_cache / "metadata.csv").reset_index(
        drop=True
    )
    labels = (
        external["label"]
        .map(CLASS_TO_INDEX)
        .fillna(-1)
        .to_numpy(dtype=np.int64)
    )
    train_contract = _device_contract(train_metadata)
    lodo_report = json.loads(
        (lodo_dir / "lodo_report.json").read_text(encoding="utf-8")
    )

    external_to_train_device: dict[str, str] = {}
    for external_device, group in external.groupby("device_id", sort=True):
        key = _device_key(group.iloc[0])
        if key not in train_contract:
            raise ValueError(
                f"No Train-round physical-device match for external device {key}"
            )
        external_to_train_device[str(external_device)] = train_contract[key]

    output_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict[str, object]] = {}
    prediction_rows: list[pd.DataFrame] = []
    for model_name in model_names:
        if model_name not in MODEL_FACTORIES:
            raise ValueError(f"Unsupported CNN model: {model_name}")
        if model_name not in lodo_report["models"]:
            raise ValueError(f"Missing LODO report for CNN model: {model_name}")
        model_report = lodo_report["models"][model_name]
        fold_by_device = _checkpoint_fold_by_device(model_report)
        scoring_folds: list[dict[str, object]] = []

        for external_device, group in external.groupby(
            "device_id",
            sort=True,
        ):
            train_device = external_to_train_device[str(external_device)]
            fold_number = fold_by_device[train_device]
            checkpoint_path = (
                lodo_dir / f"{model_name}_holdout_{fold_number}.pt"
            )
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            feature_names = tuple(checkpoint["feature_names"])
            frontends = _load_frontends(external_cache, feature_names)
            model = MODEL_FACTORIES[model_name]()
            model.load_state_dict(checkpoint["state_dict"])

            indexes = group.index.to_numpy(dtype=np.int64)
            probabilities = _predict_cnn(
                model,
                frontends,
                labels,
                indexes,
                batch_size,
            )
            device_metadata = group.reset_index(drop=True)
            threshold = float(checkpoint["threshold"])
            counting = _score_counting(
                device_metadata,
                _thresholded_rows(
                    device_metadata,
                    probabilities,
                    threshold,
                ),
                tolerance_ms=tolerance_ms,
                timing_config=NO_CONFIDENCE_GATE,
            )
            scoring_folds.append(
                {
                    "held_device": train_device,
                    "external_device": str(external_device),
                    "checkpoint": checkpoint_path.name,
                    "feature_names": list(feature_names),
                    "threshold": threshold,
                    "counting": counting,
                }
            )

            predictions = group[
                [
                    "session_id",
                    "device_id",
                    "device_alias",
                    "scenario_id",
                    "onset_ms",
                    "disposition",
                    "label",
                ]
            ].copy()
            predictions["model"] = model_name
            predictions["racket_probability"] = probabilities[:, RACKET_INDEX]
            predictions["threshold"] = threshold
            predictions["predicted_racket"] = (
                predictions["racket_probability"] >= threshold
            )
            prediction_rows.append(predictions)

        reports[model_name] = {
            "folds": scoring_folds,
            "counting": _aggregate_count_reports(scoring_folds, "counting"),
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
        "experiment": "cross_day_cnn_physical_device_holdout",
        "train_cache": str(train_cache.resolve()),
        "external_cache": str(external_cache.resolve()),
        "lodo_dir": str(lodo_dir.resolve()),
        "device_mapping": external_to_train_device,
        "selection_warning": (
            "This earlier reviewed round is a development check. It may guide "
            "candidate selection, but the explicit final holdout remains sealed."
        ),
        "leaderboard": leaderboard,
        "models": reports,
    }
    (output_dir / "cross_day_cnn_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if prediction_rows:
        pd.concat(prediction_rows, ignore_index=True).to_csv(
            output_dir / "cross_day_cnn_candidate_predictions.csv",
            index=False,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--external-cache", type=Path, required=True)
    parser.add_argument("--lodo-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_FACTORIES),
        default=["dual_residual"],
    )
    parser.add_argument("--tolerance-ms", type=float, default=140.0)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    report = evaluate_cross_day_cnn(
        args.train_cache,
        args.external_cache,
        args.lodo_dir,
        args.output_dir,
        model_names=tuple(args.models),
        tolerance_ms=args.tolerance_ms,
        batch_size=args.batch_size,
    )
    print(json.dumps({"leaderboard": report["leaderboard"]}, indent=2))


if __name__ == "__main__":
    main()
