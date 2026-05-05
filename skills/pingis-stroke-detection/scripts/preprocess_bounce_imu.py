"""
preprocess_bounce_imu.py

Build a binary IMU dataset for bounce-contact motion from synchronized
audio + IMU collection sessions.

Ground truth comes from reviewed audio markers:
- racket_contact -> bounce_contact_motion
- not_racket_contact -> not_bounce_contact
- ignore -> skipped

Only reviewed takes with saved IMU recordings are included.

Run:
  python skills/pingis-stroke-detection/scripts/preprocess_bounce_imu.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT_DIR / "data" / "audio" / "raw"
OUT_DIR = ROOT_DIR / "data" / "imu" / "processed"
OUT_FILE = OUT_DIR / "bounce_imu_dataset.csv"

WINDOW_PRE_MS = 180
WINDOW_POST_MS = 220
MIN_WINDOW_SAMPLES = 8

CHANNELS = [
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_samples(samples: list[dict], calibration_profile: dict | None) -> list[dict]:
    if not calibration_profile:
        return samples

    gravity = calibration_profile.get("gravity", {})
    gyro_bias = calibration_profile.get("gyro_bias", {})
    gx = float(gravity.get("x", 0.0))
    gy = float(gravity.get("y", 0.0))
    gz = float(gravity.get("z", 0.0))
    bgx = float(gyro_bias.get("x", 0.0))
    bgy = float(gyro_bias.get("y", 0.0))
    bgz = float(gyro_bias.get("z", 0.0))

    normalized: list[dict] = []
    for sample in samples:
        normalized.append(
            {
                **sample,
                "accel_x": float(sample["accel_x"]) - gx,
                "accel_y": float(sample["accel_y"]) - gy,
                "accel_z": float(sample["accel_z"]) - gz,
                "gyro_x": float(sample["gyro_x"]) - bgx,
                "gyro_y": float(sample["gyro_y"]) - bgy,
                "gyro_z": float(sample["gyro_z"]) - bgz,
            }
        )
    return normalized


def sample_take_ts_ms(sample: dict, take_start_ms: int) -> int:
    if sample.get("take_ts_ms") is not None:
        return int(sample["take_ts_ms"])
    if sample.get("received_at_ms") is not None:
        return max(0, int(sample["received_at_ms"]) - take_start_ms)

    ts_ms = int(sample.get("ts_ms", 0))
    if ts_ms > 1_000_000_000_000:
        return max(0, ts_ms - take_start_ms)
    return ts_ms


def extract_window(samples: list[dict], center_take_ts_ms: int, take_start_ms: int) -> list[dict]:
    start_ms = center_take_ts_ms - WINDOW_PRE_MS
    end_ms = center_take_ts_ms + WINDOW_POST_MS
    window: list[dict] = []
    for sample in samples:
        ts_ms = sample_take_ts_ms(sample, take_start_ms)
        if start_ms <= ts_ms <= end_ms:
            window.append({**sample, "sample_take_ts_ms": ts_ms})
    return window


def add_axis_features(features: dict, values: np.ndarray, prefix: str) -> None:
    features[f"{prefix}_mean"] = float(np.mean(values))
    features[f"{prefix}_std"] = float(np.std(values))
    features[f"{prefix}_min"] = float(np.min(values))
    features[f"{prefix}_max"] = float(np.max(values))
    features[f"{prefix}_ptp"] = float(np.ptp(values))
    features[f"{prefix}_rms"] = float(np.sqrt(np.mean(values ** 2)))
    if values.size > 1:
        diffs = np.diff(values)
        features[f"{prefix}_diff_abs_mean"] = float(np.mean(np.abs(diffs)))
        features[f"{prefix}_diff_abs_max"] = float(np.max(np.abs(diffs)))
    else:
        features[f"{prefix}_diff_abs_mean"] = 0.0
        features[f"{prefix}_diff_abs_max"] = 0.0


def extract_features(window: list[dict]) -> dict:
    matrix = np.array([[float(sample[channel]) for channel in CHANNELS] for sample in window], dtype=float)
    features: dict[str, float] = {}

    for index, channel in enumerate(CHANNELS):
        add_axis_features(features, matrix[:, index], channel)

    accel = matrix[:, 0:3]
    gyro = matrix[:, 3:6]
    accel_mag = np.linalg.norm(accel, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)

    add_axis_features(features, accel_mag, "accel_mag")
    add_axis_features(features, gyro_mag, "gyro_mag")
    features["window_samples"] = float(matrix.shape[0])
    features["window_duration_ms"] = float(
        int(window[-1]["sample_take_ts_ms"]) - int(window[0]["sample_take_ts_ms"])
    )
    return features


def marker_to_label(final_label: str) -> str | None:
    if final_label == "racket_contact":
        return "bounce_contact_motion"
    if final_label == "not_racket_contact":
        return "not_bounce_contact"
    return None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    session_files = sorted(RAW_DIR.glob("audio_session_*.json"))
    if not session_files:
        print(f"No audio session files found in {RAW_DIR}")
        return

    rows: list[dict] = []
    skipped_sessions = 0
    skipped_events = 0

    for session_path in session_files:
        session = load_json(session_path)
        session_meta = session.get("session_meta", {})
        if session_meta.get("collection_mode") != "guided_scenarios_audio_imu":
            skipped_sessions += 1
            continue

        calibration_profile = session.get("calibration_profile")
        session_id = session_path.stem

        for event in session.get("events", []):
            event_scenario = str(event.get("scenario") or "")
            if not event_scenario and str(event.get("scenario_id", "")).startswith("racket_bounce"):
                event_scenario = "racket_bouncing"
            if event_scenario != "racket_bouncing":
                skipped_events += 1
                continue

            review = event.get("review") or {}
            imu_recording = event.get("imu_recording") or {}
            if not review.get("completed_at"):
                skipped_events += 1
                continue
            raw_samples = imu_recording.get("samples") or []
            if not raw_samples:
                skipped_events += 1
                continue

            normalized_samples = normalize_samples(raw_samples, calibration_profile)
            take_group = f"{session_id}:{event['scenario_id']}:{event['take_index']}"
            take_start_ms = int(imu_recording.get("started_at_ms", 0))
            imu_partial = bool(imu_recording.get("partial", False))
            imu_disconnected = bool(imu_recording.get("disconnected", False))
            imu_sample_count = int(imu_recording.get("sample_count") or len(raw_samples))
            imu_hz_estimate = float(imu_recording.get("sample_hz_estimate") or 0.0)

            for marker in review.get("markers", []):
                review_status = str(marker.get("review_status") or "confirmed")
                if review_status in {"pending", "deleted", "filtered"}:
                    continue
                target_label = marker_to_label(marker.get("final_label", "ignore"))
                if target_label is None:
                    continue

                center_take_ts_ms = int(marker["timestamp_ms"])
                window = extract_window(normalized_samples, center_take_ts_ms, take_start_ms)
                if len(window) < MIN_WINDOW_SAMPLES:
                    continue

                row = extract_features(window)
                row["label"] = target_label
                row["session_id"] = session_id
                row["group_id"] = take_group
                row["scenario_id"] = event["scenario_id"]
                row["scenario"] = event_scenario
                row["bounce_context"] = event.get("bounce_context", "")
                row["calibration_status"] = event.get(
                    "calibration_status",
                    session_meta.get("calibration_status", ""),
                )
                row["background_condition"] = event["background_condition"]
                row["take_index"] = int(event["take_index"])
                row["marker_id"] = marker["id"]
                row["marker_label"] = marker["final_label"]
                row["review_status"] = review_status
                row["contact_kind"] = marker.get("contact_kind", "")
                row["not_racket_kind"] = marker.get("not_racket_kind", "")
                row["bounce_side"] = marker.get("bounce_side", "unknown")
                row["marker_take_ts_ms"] = center_take_ts_ms
                row["imu_sample_count"] = imu_sample_count
                row["imu_hz_estimate"] = imu_hz_estimate
                row["imu_partial"] = imu_partial
                row["imu_disconnected"] = imu_disconnected
                row["source_file"] = event["wav_filename"]
                rows.append(row)

    if not rows:
        print("No reviewed audio+IMU rows found yet. Record and review synced takes first.")
        print(f"Skipped sessions without synced IMU mode: {skipped_sessions}")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUT_FILE, index=False)

    print(f"Saved {len(df)} rows to {OUT_FILE}")
    print("Label distribution:")
    print(df["label"].value_counts().to_string())
    print(f"Skipped sessions without synced IMU mode: {skipped_sessions}")
    print(f"Skipped events without reviewed IMU data: {skipped_events}")


if __name__ == "__main__":
    main()
