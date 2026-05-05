"""
preprocess.py — Window extraction and feature engineering

Reads all session JSON files from data/raw/, extracts 800ms windows
centered on each labeled event, computes features, and writes dataset.csv.

Usage:
    python skills/pingis-stroke-detection/scripts/preprocess.py

Output:
    data/processed/dataset.csv
"""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

SAMPLE_RATE = 50          # Hz (approximate; BERG AirHive emits ~50 samples/sec)
WINDOW_MS = 800           # ms — total window around each event
WINDOW_SAMPLES = 40       # WINDOW_MS / (1000 / SAMPLE_RATE)
DT_S = WINDOW_MS / WINDOW_SAMPLES / 1000.0
CHANNELS = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_FILE = OUT_DIR / "dataset.csv"


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(window: np.ndarray) -> dict:
    """
    Extract time-domain features from a (WINDOW_SAMPLES, 6) array.

    Columns: accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
    Returns a flat dict of feature_name -> float.
    """
    features = {}

    for i, ch in enumerate(CHANNELS):
        x = window[:, i].astype(float)
        features[f"{ch}_mean"] = float(np.mean(x))
        features[f"{ch}_std"] = float(np.std(x))
        features[f"{ch}_min"] = float(np.min(x))
        features[f"{ch}_max"] = float(np.max(x))
        features[f"{ch}_ptp"] = float(np.ptp(x))                    # peak-to-peak
        features[f"{ch}_rms"] = float(np.sqrt(np.mean(x ** 2)))
        if ch in {"accel_x", "gyro_x"}:
            midpoint = max(1, len(x) // 2)
            pre = x[:midpoint]
            post = x[midpoint:]
            pre_mean = float(np.mean(pre))
            post_mean = float(np.mean(post)) if len(post) else pre_mean
            features[f"{ch}_signed_integral"] = float(np.sum(x) * DT_S)
            features[f"{ch}_pre_mean"] = pre_mean
            features[f"{ch}_post_mean"] = post_mean
            features[f"{ch}_post_minus_pre"] = float(post_mean - pre_mean)
            features[f"{ch}_positive_peak"] = float(np.max(x))
            features[f"{ch}_negative_peak"] = float(np.min(x))

    # Magnitude features (orientation-independent)
    accel = window[:, 0:3].astype(float)
    gyro = window[:, 3:6].astype(float)

    accel_mag = np.linalg.norm(accel, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)

    features["accel_mag_mean"] = float(np.mean(accel_mag))
    features["accel_mag_peak"] = float(np.max(accel_mag))
    features["accel_mag_rms"] = float(np.sqrt(np.mean(accel_mag ** 2)))
    features["gyro_mag_std"] = float(np.std(gyro_mag))
    features["gyro_mag_peak"] = float(np.max(gyro_mag))

    return features


def add_handedness_normalized_x_features(features: dict, handedness: str) -> None:
    sign = -1.0 if handedness == "left" else 1.0
    features["handedness_sign"] = sign
    for channel in ("accel_x", "gyro_x"):
        for suffix in ("mean", "signed_integral", "pre_mean", "post_mean", "post_minus_pre"):
            key = f"{channel}_{suffix}"
            if key in features:
                features[f"{key}_hand_norm"] = float(features[key] * sign)


def extract_window(samples: list[dict], center_idx: int) -> np.ndarray | None:
    """
    Extract WINDOW_SAMPLES samples centered on center_idx.
    Returns (WINDOW_SAMPLES, 6) array or None if not enough samples.
    """
    half = WINDOW_SAMPLES // 2
    start = center_idx - half
    end = center_idx + half

    if start < 0 or end > len(samples):
        return None

    rows = []
    for s in samples[start:end]:
        try:
            rows.append([
                s["accel_x"], s["accel_y"], s["accel_z"],
                s["gyro_x"], s["gyro_y"], s["gyro_z"],
            ])
        except KeyError as e:
            print(f"  Warning: missing field {e} in sample, skipping event")
            return None

    return np.array(rows, dtype=float)


# ── Session loading ───────────────────────────────────────────────────────────

def find_event_center(samples: list[dict], event_ts_ms: int) -> int:
    """
    Find the sample index closest to the labeled event timestamp.
    """
    ts_values = [s["ts_ms"] for s in samples]
    diffs = [abs(ts - event_ts_ms) for ts in ts_values]
    return int(np.argmin(diffs))


def load_session(filepath: Path) -> tuple[list[dict], dict]:
    """
    Load a session file. Supports two formats:

    1. New format (app v1.0+):
       { "session_meta": { "player_name", "handedness", "calibration_accel",
                           "calibration_gyro_bias", ... },
         "events": [{label, stroke_type, samples: [...]}, ...] }

    2. Old format (array):
       [{label, stroke_type, samples: [...]}, ...]

    Returns (events_list, meta_dict).
    meta_dict has keys: player_name, handedness, calibration_accel, calibration_gyro_bias.
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    empty_meta = {
        "player_name": "unknown",
        "handedness": "unknown",
        "calibration_accel": {"x": 0, "y": 0, "z": 0},
        "calibration_gyro_bias": {"x": 0, "y": 0, "z": 0},
    }

    if isinstance(data, dict) and "session_meta" in data:
        # New format
        meta = data.get("session_meta", {})
        return data.get("events", []), {
            "player_name": meta.get("player_name", "unknown"),
            "handedness": meta.get("handedness", "unknown"),
            "calibration_accel": meta.get("calibration_accel", {"x": 0, "y": 0, "z": 0}),
            "calibration_gyro_bias": meta.get("calibration_gyro_bias", {"x": 0, "y": 0, "z": 0}),
        }
    elif isinstance(data, list):
        # Old format — no metadata
        return data, empty_meta
    elif isinstance(data, dict):
        # Single event object (old)
        return [data], empty_meta
    else:
        print(f"  Warning: unexpected format in {filepath.name}")
        return [], empty_meta


def apply_calibration(samples: list[dict], cal_accel: dict) -> list[dict]:
    """
    Subtract gravity baseline from accel channels.
    Returns new list of dicts with calibrated accel values.
    """
    ox, oy, oz = cal_accel.get("x", 0), cal_accel.get("y", 0), cal_accel.get("z", 0)
    if ox == 0 and oy == 0 and oz == 0:
        return samples  # No calibration data — return as-is
    result = []
    for s in samples:
        c = dict(s)
        c["accel_x"] = s.get("accel_x", 0) - ox
        c["accel_y"] = s.get("accel_y", 0) - oy
        c["accel_z"] = s.get("accel_z", 0) - oz
        result.append(c)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    session_files = sorted(RAW_DIR.glob("*.json"))
    if not session_files:
        print(f"No session files found in {RAW_DIR}/")
        print("Record some training data first using the DataCollectionScreen app.")
        return

    print(f"Found {len(session_files)} session file(s)")

    all_rows = []

    for filepath in session_files:
        print(f"\nProcessing: {filepath.name}")
        events, meta = load_session(filepath)

        player_name = meta["player_name"]
        handedness = meta["handedness"]
        cal_accel = meta["calibration_accel"]

        if handedness != "unknown":
            print(f"  Player: {player_name} ({handedness}-handed)")

        for event in events:
            label = event.get("label")
            stroke_type = event.get("stroke_type", "unknown")
            raw_samples = event.get("samples", [])

            if not label or not raw_samples:
                print(f"  Skipping event: missing label or samples")
                continue

            if len(raw_samples) < WINDOW_SAMPLES:
                print(f"  Skipping {label}: only {len(raw_samples)} samples (need {WINDOW_SAMPLES})")
                continue

            # Apply calibration correction (removes gravity baseline)
            samples = apply_calibration(raw_samples, cal_accel)

            # Use middle of samples as the event center
            center_idx = len(samples) // 2

            window = extract_window(samples, center_idx)
            if window is None:
                print(f"  Skipping {label}: could not extract window")
                continue

            features = extract_features(window)
            add_handedness_normalized_x_features(features, handedness)
            features["label"] = label
            features["stroke_type"] = stroke_type
            features["player_name"] = player_name
            features["handedness"] = handedness
            all_rows.append(features)

    if not all_rows:
        print("\nNo valid events extracted. Check your session files.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_FILE, index=False)

    print(f"\n✓ Saved {len(df)} rows to {OUT_FILE}")
    print("\nLabel distribution:")
    print(df["label"].value_counts().to_string())
    print("\nFirst few rows (feature columns only):")
    meta_cols = ("label", "stroke_type", "player_name", "handedness")
    feature_cols = [c for c in df.columns if c not in meta_cols]
    print(df[feature_cols].head(3).to_string())


if __name__ == "__main__":
    main()
