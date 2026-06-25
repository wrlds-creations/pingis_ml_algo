"""
Validate the playing-retro app model export against the selected joblib model.

Run:
  python skills/pingis-audio-classification/scripts/validate_playing_retro_audio_app_export.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = (
    ROOT_DIR
    / "data"
    / "audio"
    / "models"
    / "playing_retro_candidates"
    / "playing_retro_audio_rf_v2026_06_04_t0030_multi_window_context"
)
DEFAULT_APP_MODEL = (
    ROOT_DIR
    / "apps"
    / "collector"
    / "src"
    / "models"
    / "playing_retro_audio_model.json"
)

REQUIRED_WINDOWS = {
    "tight": {"before_ms": 60, "after_ms": 140},
    "normal": {"before_ms": 100, "after_ms": 200},
    "wide": {"before_ms": 160, "after_ms": 320},
}

REQUIRED_CONTEXT_FEATURES = [
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

EXPECTED_MODEL_VERSION = "playing_retro_audio_rf_v2026_06_04_t0030_multi_window_context"
REQUIRED_REVIEW_THRESHOLDS = {
    "racket_contact": 0.0,
    "table_bounce": 0.0,
    "same_label_dedupe_ms": 80,
}

FORBIDDEN_FEATURE_FRAGMENTS = [
    "close_event_bucket",
    "neighbor_sequence",
    "matched_truth",
    "nearest_truth",
    "truth_nearest",
    "candidate_to_truth",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate playing-retro app model export.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--app-model", default=str(DEFAULT_APP_MODEL))
    parser.add_argument("--expected-model-version", default=EXPECTED_MODEL_VERSION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    app_model_path = Path(args.app_model)

    expected_features = joblib.load(model_dir / "playing_retro_audio_feature_cols.pkl")
    label_encoder = joblib.load(model_dir / "playing_retro_audio_label_encoder.pkl")
    app_model = json.loads(app_model_path.read_text(encoding="utf-8"))

    feature_names = app_model["feature_names"]
    if feature_names != expected_features:
        for index, (expected, actual) in enumerate(zip(expected_features, feature_names)):
            if expected != actual:
                raise AssertionError(
                    f"Feature mismatch at index {index}: expected {expected}, got {actual}"
                )
        raise AssertionError(
            f"Feature length mismatch: expected {len(expected_features)}, got {len(feature_names)}"
        )

    if app_model["labels"] != label_encoder.classes_.tolist():
        raise AssertionError(
            f"Label mismatch: expected {label_encoder.classes_.tolist()}, got {app_model['labels']}"
        )

    metadata = app_model.get("metadata", {})
    if metadata.get("app_model_role") != "spel_retro_audio_review_only":
        raise AssertionError("playing-retro export must be marked review-only")
    if metadata.get("normal_audio_model_unchanged") is not True:
        raise AssertionError("export metadata must state normal audio model is unchanged")
    if metadata.get("model_version") != args.expected_model_version:
        raise AssertionError(
            f"Model version mismatch: expected {args.expected_model_version}, "
            f"got {metadata.get('model_version')}"
        )

    review_thresholds = metadata.get("review_thresholds", {})
    for key, expected in REQUIRED_REVIEW_THRESHOLDS.items():
        actual = review_thresholds.get(key)
        if actual != expected:
            raise AssertionError(f"Review threshold mismatch for {key}: expected {expected}, got {actual}")

    windows = {item["name"]: item for item in metadata.get("windows", [])}
    for name, expected in REQUIRED_WINDOWS.items():
        actual = windows.get(name)
        if actual is None:
            raise AssertionError(f"Missing window metadata: {name}")
        if actual["before_ms"] != expected["before_ms"] or actual["after_ms"] != expected["after_ms"]:
            raise AssertionError(f"Window metadata mismatch for {name}: {actual}")
        count = sum(feature.startswith(f"{name}_") for feature in feature_names)
        if count != 62:
            raise AssertionError(f"Expected 62 {name} features, got {count}")

    missing_context = [feature for feature in REQUIRED_CONTEXT_FEATURES if feature not in feature_names]
    if missing_context:
        raise AssertionError(f"Missing context features: {missing_context}")

    forbidden = [
        feature
        for feature in feature_names
        if any(fragment in feature for fragment in FORBIDDEN_FEATURE_FRAGMENTS)
    ]
    if forbidden:
        raise AssertionError(f"Truth-derived features leaked into app export: {forbidden}")

    if len(app_model["scaler_mean"]) != len(feature_names):
        raise AssertionError("scaler_mean length does not match feature_names")
    if len(app_model["scaler_std"]) != len(feature_names):
        raise AssertionError("scaler_std length does not match feature_names")
    if len(app_model["trees"]) != metadata.get("tree_count"):
        raise AssertionError("tree_count metadata does not match exported trees")

    print("OK playing-retro app export parity")
    print(f"model={app_model_path}")
    print(f"features={len(feature_names)} labels={app_model['labels']} trees={len(app_model['trees'])}")


if __name__ == "__main__":
    main()
