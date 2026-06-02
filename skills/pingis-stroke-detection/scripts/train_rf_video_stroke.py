"""
Train the first forehand/backhand video-stroke RandomForest model.

Usage:
  python skills/pingis-stroke-detection/scripts/train_rf_video_stroke.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = ROOT_DIR / "data" / "video" / "processed" / "video_stroke_dataset.csv"
DEFAULT_MODEL_DIR = ROOT_DIR / "data" / "video" / "models"

NON_FEATURE_COLUMNS = {
    "session_id",
    "player_name",
    "handedness",
    "camera_facing",
    "camera_angle",
    "camera_side",
    "video_filename",
    "take_index",
    "marker_id",
    "timestamp_ms",
    "stroke_type",
    "feature_spec",
}

TARGET_STROKE_TYPES = ["forehand", "backhand", "unknown"]


def build_classifier() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def split_rows(df: pd.DataFrame):
    if "session_id" in df.columns and df["session_id"].nunique() >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_index, test_index = next(splitter.split(df, df["stroke_type"], groups=df["session_id"]))
        return df.iloc[train_index], df.iloc[test_index], "grouped_by_session"

    train_df, test_df = train_test_split(
        df,
        test_size=0.25,
        random_state=42,
        stratify=df["stroke_type"],
    )
    return train_df, test_df, "stratified_random"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train video stroke forehand/backhand RF.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--min-rows", type=int, default=20)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    model_dir = Path(args.model_dir)
    if not dataset.exists():
        raise SystemExit(f"Dataset not found: {dataset}. Run preprocess_video_strokes.py first.")

    df = pd.read_csv(dataset)
    df = df[df["stroke_type"].isin(TARGET_STROKE_TYPES)].copy()
    if len(df) < args.min_rows:
        raise SystemExit(f"Only {len(df)} rows. Need at least {args.min_rows} for a first model.")
    if df["stroke_type"].nunique() < 2:
        raise SystemExit("Need at least two stroke classes.")

    feature_cols = [column for column in df.columns if column not in NON_FEATURE_COLUMNS]
    train_df, test_df, split_mode = split_rows(df)

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_df["stroke_type"])
    y_test = encoder.transform(test_df["stroke_type"])
    x_train = train_df[feature_cols].astype(float).values
    x_test = test_df[feature_cols].astype(float).values

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    classifier = build_classifier()
    classifier.fit(x_train_scaled, y_train)
    y_pred = classifier.predict(x_test_scaled)

    print(f"Rows: {len(df)}")
    print(f"Split: {split_mode} train={len(train_df)} test={len(test_df)}")
    print(df["stroke_type"].value_counts().to_string())
    print()
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    print(f"Macro F1: {f1_score(y_test, y_pred, average='macro'):.3f}")

    final_encoder = LabelEncoder()
    y_all = final_encoder.fit_transform(df["stroke_type"])
    x_all = df[feature_cols].astype(float).values
    final_scaler = StandardScaler()
    x_all_scaled = final_scaler.fit_transform(x_all)
    final_classifier = build_classifier()
    final_classifier.fit(x_all_scaled, y_all)

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_classifier, model_dir / "video_stroke_rf_classifier.pkl")
    joblib.dump(final_scaler, model_dir / "video_stroke_feature_scaler.pkl")
    joblib.dump(final_encoder, model_dir / "video_stroke_label_encoder.pkl")
    joblib.dump(feature_cols, model_dir / "video_stroke_feature_cols.pkl")
    print(f"Saved final all-data model artifacts to {model_dir}")


if __name__ == "__main__":
    main()
