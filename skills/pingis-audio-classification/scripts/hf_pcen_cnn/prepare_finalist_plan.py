"""Freeze finalist model contracts before opening the explicit final holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


CNN_FINALISTS = ("dual_residual", "pcen_compact")
HISTGB_FINALISTS = (
    "histgb_stack_reference",
    "histgb_stack_flexible",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _median_threshold(model_report: dict[str, object]) -> float:
    thresholds = [
        float(fold["threshold"])
        for fold in model_report["folds"]
    ]
    return float(np.median(thresholds))


def _median_best_epoch(model_report: dict[str, object]) -> int:
    epochs = [
        int(fold["training"]["best_epoch"])
        for fold in model_report["folds"]
    ]
    return int(np.median(epochs))


def prepare_finalist_plan(
    cnn_lodo_report: Path,
    histgb_lodo_report: Path,
    output_path: Path,
    *,
    seed: int,
) -> dict[str, object]:
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to replace frozen finalist plan: {output_path}"
        )
    cnn_report = json.loads(cnn_lodo_report.read_text(encoding="utf-8"))
    histgb_report = json.loads(histgb_lodo_report.read_text(encoding="utf-8"))

    candidates: list[dict[str, object]] = []
    for model_name in CNN_FINALISTS:
        model_report = cnn_report["models"][model_name]
        feature_names = (
            ["logmel", "pcen"]
            if model_name == "dual_residual"
            else ["pcen"]
        )
        candidates.append(
            {
                "model": model_name,
                "family": "cnn",
                "classes": list(cnn_report["classes"]),
                "feature_names": feature_names,
                "fixed_epochs": _median_best_epoch(model_report),
                "threshold": _median_threshold(model_report),
                "augment": model_name == "dual_residual",
                "selection_metrics": next(
                    row
                    for row in cnn_report["leaderboard"]
                    if row["model"] == model_name
                ),
                "fold_thresholds": [
                    float(fold["threshold"])
                    for fold in model_report["folds"]
                ],
                "fold_best_epochs": [
                    int(fold["training"]["best_epoch"])
                    for fold in model_report["folds"]
                ],
            }
        )

    for model_name in HISTGB_FINALISTS:
        model_report = histgb_report["models"][model_name]
        candidates.append(
            {
                "model": model_name,
                "family": "histgb",
                "classes": list(histgb_report["classes"]),
                "feature_names": list(histgb_report["tabular_features"]),
                "threshold": _median_threshold(model_report),
                "config": model_report["config"],
                "selection_metrics": next(
                    row
                    for row in histgb_report["leaderboard"]
                    if row["model"] == model_name
                ),
                "fold_thresholds": [
                    float(fold["threshold"])
                    for fold in model_report["folds"]
                ],
            }
        )

    plan: dict[str, object] = {
        "plan_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "seed": seed,
        "selection_policy": {
            "model_selection_data": (
                "Train-only physical-device LODO plus the earlier independent "
                "reviewed phone round"
            ),
            "threshold": "median of the three Train-only LODO fold thresholds",
            "cnn_epochs": "median of the three Train-only LODO best epochs",
            "final_holdout": (
                "sealed until this plan exists; evaluate all frozen candidates "
                "once without threshold or hyperparameter changes"
            ),
            "production_constraint": "four-class models only",
        },
        "evidence": {
            "cnn_lodo_report": str(cnn_lodo_report.resolve()),
            "cnn_lodo_report_sha256": _sha256(cnn_lodo_report),
            "histgb_lodo_report": str(histgb_lodo_report.resolve()),
            "histgb_lodo_report_sha256": _sha256(histgb_lodo_report),
        },
        "candidates": candidates,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnn-lodo-report", type=Path, required=True)
    parser.add_argument("--histgb-lodo-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()
    plan = prepare_finalist_plan(
        args.cnn_lodo_report,
        args.histgb_lodo_report,
        args.output,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "candidates": [
                    {
                        "model": candidate["model"],
                        "threshold": candidate["threshold"],
                        "fixed_epochs": candidate.get("fixed_epochs"),
                    }
                    for candidate in plan["candidates"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
