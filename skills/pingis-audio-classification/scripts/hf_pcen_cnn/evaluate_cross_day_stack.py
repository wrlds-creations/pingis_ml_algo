"""Evaluate Train-only LODO stack models on an earlier reviewed phone round."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .train_reviewed_device_lodo import (
    _aggregate_count_reports,
    _fable_rows_for_timing,
    _score_counting,
)
from .train_round_lodo import (
    NO_CONFIDENCE_GATE,
    RACKET_INDEX,
    _feature_matrix,
    _model_probabilities,
    _thresholded_rows,
)


DEVICE_KEY_COLUMNS = (
    "device_manufacturer",
    "device_model",
    "device_platform",
)


def _device_key(row: pd.Series) -> tuple[str, ...]:
    return tuple(str(row[column]).strip().lower() for column in DEVICE_KEY_COLUMNS)


def _device_contract(metadata: pd.DataFrame) -> dict[tuple[str, ...], str]:
    contract: dict[tuple[str, ...], str] = {}
    for device_id, group in metadata.groupby("device_id", sort=True):
        key = _device_key(group.iloc[0])
        if key in contract:
            raise ValueError(f"Duplicate physical-device key: {key}")
        contract[key] = str(device_id)
    return contract


def evaluate_cross_day_stack(
    train_cache: Path,
    lodo_dir: Path,
    external_manifest: Path,
    output_dir: Path,
    *,
    tolerance_ms: float,
) -> dict[str, object]:
    train_metadata = pd.read_csv(train_cache / "metadata.csv")
    external = pd.read_csv(external_manifest).reset_index(drop=True)
    feature_matrix, feature_names = _feature_matrix(external)
    train_contract = _device_contract(train_metadata)
    report = json.loads((lodo_dir / "lodo_report.json").read_text(encoding="utf-8"))

    external_to_train_device: dict[str, str] = {}
    for external_device, group in external.groupby("device_id", sort=True):
        key = _device_key(group.iloc[0])
        if key not in train_contract:
            raise ValueError(
                f"No Train-round physical-device match for external device {key}"
            )
        external_to_train_device[str(external_device)] = train_contract[key]

    reports: dict[str, dict[str, object]] = {}
    fable_folds = []
    for external_device, group in external.groupby("device_id", sort=True):
        device_metadata = group.reset_index(drop=True)
        fable_folds.append(
            {
                "held_device": external_to_train_device[str(external_device)],
                "external_device": str(external_device),
                "counting": _score_counting(
                    device_metadata,
                    _fable_rows_for_timing(device_metadata),
                    tolerance_ms=tolerance_ms,
                ),
            }
        )
    reports["frozen_fable"] = {
        "folds": fable_folds,
        "counting": _aggregate_count_reports(fable_folds, "counting"),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows: list[pd.DataFrame] = []
    artifact_models = [
        model_name
        for model_name, model_report in report["models"].items()
        if model_name != "frozen_fable"
        and model_report.get("folds")
        and all(
            (
                lodo_dir
                / f"{model_name}_holdout_{fold_number}.joblib"
            ).exists()
            for fold_number in range(1, len(model_report["folds"]) + 1)
        )
    ]
    for model_name in artifact_models:
        model_folds = report["models"][model_name]["folds"]
        fold_number_by_device = {
            str(fold["held_device"]): index
            for index, fold in enumerate(model_folds, start=1)
        }
        scoring_folds: list[dict[str, object]] = []
        for external_device, group in external.groupby("device_id", sort=True):
            train_device = external_to_train_device[str(external_device)]
            fold_number = fold_number_by_device[train_device]
            artifact = joblib.load(
                lodo_dir / f"{model_name}_holdout_{fold_number}.joblib"
            )
            if list(artifact["feature_names"]) != feature_names:
                raise ValueError(f"Feature contract mismatch for {model_name}")
            indexes = group.index.to_numpy(dtype=np.int64)
            probabilities = _model_probabilities(
                artifact["model"],
                feature_matrix,
                indexes,
                binary_target=bool(artifact.get("binary_target", False)),
            )
            device_metadata = group.reset_index(drop=True)
            threshold = float(artifact["threshold"])
            counting = _score_counting(
                device_metadata,
                _thresholded_rows(device_metadata, probabilities, threshold),
                tolerance_ms=tolerance_ms,
                timing_config=NO_CONFIDENCE_GATE,
            )
            scoring_folds.append(
                {
                    "held_device": train_device,
                    "external_device": str(external_device),
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
        "experiment": "cross_day_physical_device_holdout",
        "train_cache": str(train_cache.resolve()),
        "lodo_dir": str(lodo_dir.resolve()),
        "external_manifest": str(external_manifest.resolve()),
        "device_mapping": external_to_train_device,
        "selection_warning": (
            "This earlier reviewed round is a development check. It may guide "
            "candidate selection, but the explicit final holdout remains sealed."
        ),
        "leaderboard": leaderboard,
        "models": reports,
    }
    (output_dir / "cross_day_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if prediction_rows:
        pd.concat(prediction_rows, ignore_index=True).to_csv(
            output_dir / "cross_day_candidate_predictions.csv",
            index=False,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--lodo-dir", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tolerance-ms", type=float, default=140.0)
    args = parser.parse_args()
    report = evaluate_cross_day_stack(
        args.train_cache,
        args.lodo_dir,
        args.external_manifest,
        args.output_dir,
        tolerance_ms=args.tolerance_ms,
    )
    print(json.dumps({"leaderboard": report["leaderboard"]}, indent=2))


if __name__ == "__main__":
    main()
