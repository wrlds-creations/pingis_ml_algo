"""
train_rf_bounce_imu.py

Train a first RandomForest for bounce-contact motion using reviewed
audio + IMU synchronized takes.

Run:
  python skills/pingis-stroke-detection/scripts/train_rf_bounce_imu.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[3]
DATASET = ROOT_DIR / "data" / "imu" / "processed" / "bounce_imu_dataset.csv"
MODEL_DIR = ROOT_DIR / "data" / "models"
META_COLS = {
    "label",
    "session_id",
    "group_id",
    "scenario_id",
    "scenario",
    "bounce_context",
    "calibration_status",
    "background_condition",
    "take_index",
    "marker_id",
    "marker_label",
    "review_status",
    "contact_kind",
    "not_racket_kind",
    "bounce_side",
    "marker_take_ts_ms",
    "source_file",
}


def load_dataset():
    if not DATASET.exists():
        print(f"Dataset not found: {DATASET}")
        print("Run preprocess_bounce_imu.py first.")
        sys.exit(1)

    df = pd.read_csv(DATASET)
    if len(df) < 20:
        print(f"Only {len(df)} rows found. Need more synced bounce IMU data first.")
        sys.exit(1)

    if df["label"].nunique() < 2:
        print("Bounce IMU dataset has only one class:")
        print(df["label"].value_counts().to_string())
        print("Need reviewed not-bounce IMU windows before training a binary bounce-contact model.")
        sys.exit(0)

    feature_cols = [
        column
        for column in df.columns
        if column not in META_COLS and pd.api.types.is_numeric_dtype(df[column])
    ]
    if not feature_cols:
        print("No numeric IMU feature columns found.")
        sys.exit(1)
    X = df[feature_cols].values.astype(float)
    groups = df["group_id"].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(df["label"].values)
    return df, X, y, groups, feature_cols, encoder


def grouped_cv_score(X, y, groups) -> float:
    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))
    if n_splits < 2:
        return float("nan")

    scores: list[float] = []
    splitter = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in splitter.split(X, y, groups):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        clf = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train, y[train_idx])
        pred = clf.predict(X_test)
        scores.append(f1_score(y[test_idx], pred, average="macro"))
    return float(np.mean(scores))


def main() -> None:
    df, X, y, groups, feature_cols, encoder = load_dataset()

    print(f"Loaded {len(df)} rows")
    print(df["label"].value_counts().to_string())

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])

    clf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y[train_idx])

    test_pred = clf.predict(X_test)
    cv_f1 = grouped_cv_score(X, y, groups)

    print(f"\nGrouped CV F1 macro: {cv_f1:.3f}" if not np.isnan(cv_f1) else "\nGrouped CV F1 macro: n/a")
    print("\nGrouped test report")
    print(
        classification_report(
            y[test_idx],
            test_pred,
            labels=np.arange(len(encoder.classes_)),
            target_names=encoder.classes_,
            zero_division=0,
        )
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_DIR / "bounce_rf_classifier.pkl")
    joblib.dump(scaler, MODEL_DIR / "bounce_feature_scaler.pkl")
    joblib.dump(encoder, MODEL_DIR / "bounce_label_encoder.pkl")
    with (MODEL_DIR / "bounce_feature_cols.json").open("w", encoding="utf-8") as handle:
        json.dump(feature_cols, handle, indent=2)

    print("\nSaved:")
    print(f"- {MODEL_DIR / 'bounce_rf_classifier.pkl'}")
    print(f"- {MODEL_DIR / 'bounce_feature_scaler.pkl'}")
    print(f"- {MODEL_DIR / 'bounce_label_encoder.pkl'}")
    print(f"- {MODEL_DIR / 'bounce_feature_cols.json'}")


if __name__ == "__main__":
    main()
