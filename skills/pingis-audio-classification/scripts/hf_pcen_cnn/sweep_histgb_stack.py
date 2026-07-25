"""Sweep regularized four-class HistGB Fable-stack candidates with device LODO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .candidate_labels import FOUR_CLASSES
from .train_ablation import CLASS_TO_INDEX
from .train_reviewed_device_lodo import (
    _aggregate_count_reports,
    _fable_rows_for_timing,
    _score_counting,
)
from .train_round_lodo import (
    NO_CONFIDENCE_GATE,
    TRAINABLE_DISPOSITIONS,
    _aggregate_binary,
    _binary_metrics,
    _feature_matrix,
    _fold_indexes,
    _model_probabilities,
    _select_count_threshold,
    _thresholded_rows,
)


RACKET_INDEX = CLASS_TO_INDEX["racket_bounce"]

HISTGB_CONFIGS: dict[str, dict[str, float | int]] = {
    "histgb_stack_reference": {
        "learning_rate": 0.06,
        "max_iter": 240,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.1,
    },
    "histgb_stack_regularized": {
        "learning_rate": 0.05,
        "max_iter": 300,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 30,
        "l2_regularization": 0.3,
    },
    "histgb_stack_compact": {
        "learning_rate": 0.05,
        "max_iter": 280,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 12,
        "l2_regularization": 0.2,
    },
    "histgb_stack_flexible": {
        "learning_rate": 0.04,
        "max_iter": 320,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 12,
        "l2_regularization": 0.5,
    },
}


def _fit_histgb(
    config: dict[str, float | int],
    features: np.ndarray,
    labels: np.ndarray,
    indexes: np.ndarray,
    seed: int,
) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        **config,
        class_weight="balanced",
        random_state=seed,
    )
    return model.fit(features[indexes], labels[indexes])


def run_histgb_sweep(
    cache_dir: Path,
    output_dir: Path,
    *,
    seed: int,
    tolerance_ms: float,
) -> dict[str, object]:
    metadata = pd.read_csv(cache_dir / "metadata.csv")
    labels = (
        metadata["label"].map(CLASS_TO_INDEX).fillna(-1).to_numpy(dtype=np.int64)
    )
    devices = sorted(metadata["device_id"].astype(str).unique())
    if len(devices) != 3:
        raise ValueError(f"Expected exactly three physical devices, got {devices}")
    features, feature_names = _feature_matrix(metadata)
    output_dir.mkdir(parents=True, exist_ok=True)

    reports: dict[str, dict[str, object]] = {
        "frozen_fable": {"folds": []},
        **{name: {"folds": []} for name in HISTGB_CONFIGS},
    }
    for fold_number, held_device in enumerate(devices, start=1):
        train_indexes, _, validation_all, held_all = _fold_indexes(
            metadata,
            labels,
            held_device,
            seed + fold_number,
        )
        validation_metadata = metadata.iloc[validation_all].reset_index(drop=True)
        held_metadata = metadata.iloc[held_all].reset_index(drop=True)
        held_trainable_local = np.flatnonzero(
            held_metadata["disposition"].isin(TRAINABLE_DISPOSITIONS).to_numpy()
        )
        held_trainable = held_all[held_trainable_local]
        fold_common = {
            "held_device": held_device,
            "train_rows": int(len(train_indexes)),
            "validation_candidate_rows": int(len(validation_all)),
            "held_candidate_rows": int(len(held_all)),
            "validation_take_ids": sorted(
                validation_metadata["take_id"].astype(str).unique().tolist()
            ),
        }
        reports["frozen_fable"]["folds"].append(
            {
                **fold_common,
                "counting": _score_counting(
                    held_metadata,
                    _fable_rows_for_timing(held_metadata),
                    tolerance_ms=tolerance_ms,
                ),
            }
        )

        for model_name, config in HISTGB_CONFIGS.items():
            print(
                f"Training {model_name} with held device {held_device}",
                flush=True,
            )
            model = _fit_histgb(
                config,
                features,
                labels,
                train_indexes,
                seed + fold_number,
            )
            validation_probabilities = _model_probabilities(
                model,
                features,
                validation_all,
                binary_target=False,
            )
            threshold, validation_counting = _select_count_threshold(
                validation_metadata,
                validation_probabilities,
                tolerance_ms,
            )
            held_probabilities = _model_probabilities(
                model,
                features,
                held_all,
                binary_target=False,
            )
            held_counting = _score_counting(
                held_metadata,
                _thresholded_rows(
                    held_metadata,
                    held_probabilities,
                    threshold,
                ),
                tolerance_ms=tolerance_ms,
                timing_config=NO_CONFIDENCE_GATE,
            )
            joblib.dump(
                {
                    "model": model,
                    "feature_names": feature_names,
                    "classes": FOUR_CLASSES,
                    "binary_target": False,
                    "threshold": threshold,
                    "config": config,
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
                        held_probabilities[held_trainable_local],
                    ),
                    "counting": held_counting,
                }
            )

    leaderboard: list[dict[str, object]] = []
    for model_name, model_report in reports.items():
        folds = model_report["folds"]
        counting = _aggregate_count_reports(folds, "counting")
        model_report["counting"] = counting
        if model_name != "frozen_fable":
            model_report["candidate_racket_binary"] = _aggregate_binary(folds)
            model_report["config"] = HISTGB_CONFIGS[model_name]
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
            str(row["model"]),
        )
    )
    report: dict[str, object] = {
        "experiment": "reviewed_round_histgb_stack_sweep",
        "classes": list(FOUR_CLASSES),
        "seed": seed,
        "selection_policy": {
            "eligible_rows": "dataset_split=train only",
            "held_device": "one stable install ID excluded from all fitting",
            "validation": "physical take groups from the other devices",
            "final_holdout": "never selected or scored by this script",
        },
        "tabular_features": feature_names,
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance-ms", type=float, default=140.0)
    args = parser.parse_args()
    report = run_histgb_sweep(
        args.cache_dir,
        args.output_dir,
        seed=args.seed,
        tolerance_ms=args.tolerance_ms,
    )
    print(json.dumps({"leaderboard": report["leaderboard"]}, indent=2))


if __name__ == "__main__":
    main()
