"""Evaluate candidate classification and final Fable-style bounce counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from .timing import replay_fable_timing


RACKET_LABEL = "racket_bounce"


def _binary_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth,
        predicted,
        average="binary",
        zero_division=0,
    )
    return {
        "rows": int(len(truth)),
        "positives": int(truth.sum()),
        "predicted_positives": int(predicted.sum()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def choose_threshold(rows: pd.DataFrame) -> tuple[float, list[dict]]:
    trainable = rows["disposition"].isin(("positive", "hard_negative")) & rows["label"].notna()
    frame = rows.loc[trainable]
    truth = frame["label"].eq(RACKET_LABEL).to_numpy()
    predicted_label = frame["predicted_label"].eq(RACKET_LABEL).to_numpy()
    probabilities = frame["prob_racket_bounce"].to_numpy(dtype=np.float64)
    sweep: list[dict] = []
    for threshold in np.arange(0.05, 0.951, 0.05):
        predicted = predicted_label & (probabilities >= threshold)
        metrics = _binary_metrics(truth, predicted)
        metrics["threshold"] = float(round(threshold, 2))
        sweep.append(metrics)
    winner = max(sweep, key=lambda item: (item["f1"], item["recall"], item["precision"]))
    return float(winner["threshold"]), sweep


def _score_split(rows: pd.DataFrame, threshold: float) -> dict:
    trainable = rows["disposition"].isin(("positive", "hard_negative")) & rows["label"].notna()
    frame = rows.loc[trainable]
    truth = frame["label"].eq(RACKET_LABEL).to_numpy()
    predicted = (
        frame["predicted_label"].eq(RACKET_LABEL).to_numpy()
        & (frame["prob_racket_bounce"].to_numpy(dtype=np.float64) >= threshold)
    )
    return _binary_metrics(truth, predicted)


def _count_session(frame: pd.DataFrame, threshold: float) -> int:
    candidates = frame[[
        "onset_ms",
        "hf_rms",
        "predicted_label",
        "prob_racket_bounce",
    ]].to_dict("records")
    return sum(
        decision.counted
        for decision in replay_fable_timing(
            candidates, probability_threshold=threshold
        )
    )


def _session_count_rows(rows: pd.DataFrame, threshold: float) -> pd.DataFrame:
    output: list[dict] = []
    for session_id, frame in rows.groupby("session_id", sort=True):
        counted = _count_session(frame, threshold)
        expected_values = frame["expected_bounce_count"].dropna().unique()
        expected = int(expected_values[0]) if len(expected_values) == 1 else None
        reviewed_racket = int(
            ((frame["disposition"] == "positive") & (frame["label"] == RACKET_LABEL)).sum()
        )
        output.append(
            {
                "session_id": session_id,
                "source": str(frame["source"].iloc[0]),
                "device_id": str(frame["device_id"].iloc[0]),
                "scenario_id": str(frame["scenario_id"].iloc[0]),
                "split": str(frame["split"].iloc[0]),
                "candidate_count": int(len(frame)),
                "counted": counted,
                "expected_count": expected,
                "reviewed_racket_candidates": reviewed_racket,
            }
        )
    return pd.DataFrame(output)


def _count_summary(session_rows: pd.DataFrame, truth_column: str) -> dict:
    selected = session_rows[session_rows[truth_column].notna()].copy()
    if selected.empty:
        return {"sessions": 0}
    truth = selected[truth_column].to_numpy(dtype=np.float64)
    counted = selected["counted"].to_numpy(dtype=np.float64)
    error = counted - truth
    return {
        "sessions": int(len(selected)),
        "expected_total": int(truth.sum()),
        "counted_total": int(counted.sum()),
        "mae_per_session": float(np.abs(error).mean()),
        "mean_signed_error": float(error.mean()),
        "exact_session_count": int((error == 0).sum()),
    }


def evaluate_frontend(
    prediction_path: Path,
    split_path: Path,
    output_dir: Path,
) -> dict:
    predictions = pd.read_csv(prediction_path)
    split = pd.read_csv(split_path, usecols=["cache_row", "split"])
    rows = predictions.merge(split, on="cache_row", how="left", validate="one_to_one")
    validation = rows[rows["split"] == "validation"]
    threshold, sweep = choose_threshold(validation)
    sessions = _session_count_rows(rows, threshold)
    output_dir.mkdir(parents=True, exist_ok=True)
    frontend = prediction_path.name.removesuffix("_candidate_predictions.csv")
    sessions.to_csv(output_dir / f"{frontend}_session_counts.csv", index=False)

    count_only = sessions[sessions["expected_count"].notna()]
    explicit_negative = count_only[count_only["expected_count"] == 0]
    reviewed = sessions[sessions["reviewed_racket_candidates"] > 0].copy()
    reviewed["reviewed_truth"] = reviewed["reviewed_racket_candidates"]
    report = {
        "frontend": frontend,
        "selected_threshold": threshold,
        "threshold_selection": "maximum binary racket F1 on grouped validation candidates",
        "threshold_sweep": sweep,
        "candidate_metrics": {
            split_name: _score_split(rows[rows["split"] == split_name], threshold)
            for split_name in ("train", "validation", "external_device")
        },
        "final_count_metrics": {
            "count_only_sessions": _count_summary(count_only, "expected_count"),
            "explicit_negative_sessions": _count_summary(
                explicit_negative, "expected_count"
            ),
            "reviewed_racket_candidates_proxy": _count_summary(
                reviewed, "reviewed_truth"
            ),
        },
        "limitations": [
            "Reviewed count truth uses matched racket candidates as a proxy; partial reviews can understate truth.",
            "STIGA positive sessions provide expected counts but no timestamps, so they are evaluation-only.",
            "The replay uses HF onset RMS for timing strength and a fixed validation-tuned probability threshold.",
        ],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    split_path = args.ablation_dir / "split.csv"
    report = {
        frontend: evaluate_frontend(
            args.ablation_dir / f"{frontend}_candidate_predictions.csv",
            split_path,
            args.output_dir,
        )
        for frontend in ("logmel", "pcen")
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "count_evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
