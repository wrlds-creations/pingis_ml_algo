"""
Export the selected playing-retro audio RandomForest as a separate app JSON.

This exporter is intentionally separate from `export_model_json.py` so the
playing-retro model cannot overwrite the Collector live `audio_model.json`.

Run:
  python skills/pingis-audio-classification/scripts/export_playing_retro_audio_model_json.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib

from export_model_json import export_tree

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = (
    ROOT_DIR
    / "data"
    / "audio"
    / "models"
    / "playing_retro_candidates"
    / "playing_retro_audio_rf_v2026_06_04_t0026_multi_window_context"
)
DEFAULT_OUT_PATH = (
    ROOT_DIR
    / "apps"
    / "collector"
    / "src"
    / "models"
    / "playing_retro_audio_model.json"
)

WINDOWS = [
    {"name": "tight", "before_ms": 60, "after_ms": 140},
    {"name": "normal", "before_ms": 100, "after_ms": 200},
    {"name": "wide", "before_ms": 160, "after_ms": 320},
]

CONTEXT_FEATURE_NAMES = [
    "ctx_is_saved_candidate",
    "ctx_candidate_count_log",
    "ctx_candidate_index_norm",
    "ctx_has_prev_candidate",
    "ctx_has_next_candidate",
    "ctx_prev_gap_1000",
    "ctx_next_gap_1000",
    "ctx_nearest_gap_1000",
    "ctx_density_150ms",
    "ctx_density_300ms",
    "ctx_density_600ms",
]

REVIEW_THRESHOLDS = {
    "racket_contact": 0.0,
    "table_bounce": 0.45,
    "same_label_dedupe_ms": 80,
    "source_ticket": "T0027",
}

TRUTH_DERIVED_FORBIDDEN_PREFIXES = (
    "truth_",
    "matched_truth_",
    "nearest_truth_",
)

TRUTH_DERIVED_FORBIDDEN_NAMES = {
    "close_event_bucket",
    "neighbor_sequence",
    "candidate_to_truth_offset_ms",
}


def assert_exportable_features(feature_names: list[str]) -> None:
    forbidden = [
        feature
        for feature in feature_names
        if feature in TRUTH_DERIVED_FORBIDDEN_NAMES
        or any(feature.startswith(prefix) for prefix in TRUTH_DERIVED_FORBIDDEN_PREFIXES)
    ]
    if forbidden:
        raise ValueError(f"Truth-derived feature names are not exportable: {forbidden[:10]}")

    for window in WINDOWS:
        prefix = f"{window['name']}_"
        count = sum(feature.startswith(prefix) for feature in feature_names)
        if count != 62:
            raise ValueError(f"Expected 62 features for {window['name']}, found {count}")

    missing_context = [feature for feature in CONTEXT_FEATURE_NAMES if feature not in feature_names]
    if missing_context:
        raise ValueError(f"Missing context features: {missing_context}")


def build_model_json(model_dir: Path, model_version: str, train_dataset: str) -> dict[str, Any]:
    classifier = joblib.load(model_dir / "playing_retro_audio_rf_classifier.pkl")
    scaler = joblib.load(model_dir / "playing_retro_audio_feature_scaler.pkl")
    label_encoder = joblib.load(model_dir / "playing_retro_audio_label_encoder.pkl")
    feature_names = joblib.load(model_dir / "playing_retro_audio_feature_cols.pkl")

    assert_exportable_features(feature_names)

    total_nodes = sum(tree.tree_.node_count for tree in classifier.estimators_)
    labels = label_encoder.classes_.tolist()
    return {
        "metadata": {
            "model_version": model_version,
            "train_dataset": train_dataset,
            "feature_version": "playing_retro_audio_features_t0009_multi_window_context_v1",
            "model_type": "random_forest_playing_retro_audio",
            "app_model_role": "spel_retro_audio_review_only",
            "selected_variant": "multi_window_context_racket_weighted",
            "classes": labels,
            "tree_count": len(classifier.estimators_),
            "total_nodes": total_nodes,
            "windows": WINDOWS,
            "context_feature_names": CONTEXT_FEATURE_NAMES,
            "review_thresholds": REVIEW_THRESHOLDS,
            "sample_rate_hz": 22050,
            "normal_audio_model_unchanged": True,
        },
        "labels": labels,
        "feature_names": feature_names,
        "scaler_mean": [round(float(value), 8) for value in scaler.mean_],
        "scaler_std": [round(float(value), 8) for value in scaler.scale_],
        "trees": [export_tree(tree) for tree in classifier.estimators_],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export playing-retro audio RF to app JSON.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--out-path", default=str(DEFAULT_OUT_PATH))
    parser.add_argument(
        "--model-version",
        default="playing_retro_audio_rf_v2026_06_04_t0026_multi_window_context",
    )
    parser.add_argument(
        "--train-dataset",
        default=(
            "T0026 multi-window/context rows from 18 reviewed playing sessions, "
            "including audio_session_2026-06-03_005 and audio_session_2026-06-04_001"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    out_path = Path(args.out_path)
    model = build_model_json(model_dir, args.model_version, args.train_dataset)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")

    print(f"Model directory: {model_dir}")
    print(f"Exported: {out_path}")
    print(f"Size: {out_path.stat().st_size / 1024:.0f} KB")
    print(f"Labels: {model['labels']}")
    print(f"Features: {len(model['feature_names'])}")
    print(f"Trees: {model['metadata']['tree_count']} nodes={model['metadata']['total_nodes']}")


if __name__ == "__main__":
    main()
