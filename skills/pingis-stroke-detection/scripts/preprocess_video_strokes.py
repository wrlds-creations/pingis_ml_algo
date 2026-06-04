"""
Build a forehand/backhand video-stroke feature dataset from reviewed video sessions.

Input:
  data/video/raw/video_stroke_session_YYYY-MM-DD_NNN.json
  data/audio/raw/audio_session_YYYY-MM-DD_NNN.json with audio_video_pose events
  matching MP4 files either next to the JSON or in a folder with the same stem.

Output:
  data/video/processed/video_stroke_dataset.csv

Usage:
  python skills/pingis-stroke-detection/scripts/preprocess_video_strokes.py
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.request
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import pandas as pd

try:
    import mediapipe as mp
    from mediapipe.tasks.python.core import base_options as base_options_module
    from mediapipe.tasks.python.vision import pose_landmarker
    from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module
except ImportError as exc:
    raise SystemExit("mediapipe is required. Run: pip install mediapipe opencv-python") from exc


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = ROOT_DIR / "data" / "video" / "raw"
DEFAULT_AUDIO_RAW_DIR = ROOT_DIR / "data" / "audio" / "raw"
DEFAULT_OUT_CSV = ROOT_DIR / "data" / "video" / "processed" / "video_stroke_dataset.csv"
DEFAULT_LANDMARK_DIR = ROOT_DIR / "data" / "video" / "processed" / "landmarks"
DEFAULT_POSE_MODEL = ROOT_DIR / "data" / "video" / "models" / "pose_landmarker_lite.task"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
FEATURE_SPEC = "video_stroke_features_v1"
WINDOW_PRE_MS = 700
WINDOW_POST_MS = 500
MIN_FRAMES = 4
MIN_AVG_VISIBILITY = 0.35
LANDMARK_SAMPLE_FPS = 15
EXCLUDED_AUDIO_VIDEO_SESSIONS = {
    "audio_session_2026-05-26_002",
    "audio_session_2026-05-26_003",
    "audio_session_2026-05-26_004",
}

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16

FEATURE_NAMES = [
    "frame_count",
    "avg_visibility",
    "wrist_x_mean",
    "wrist_x_std",
    "wrist_x_min",
    "wrist_x_max",
    "wrist_x_delta",
    "wrist_x_ptp",
    "wrist_y_mean",
    "wrist_y_std",
    "wrist_y_min",
    "wrist_y_max",
    "wrist_y_delta",
    "wrist_y_ptp",
    "elbow_x_mean",
    "elbow_x_std",
    "elbow_x_delta",
    "elbow_x_ptp",
    "elbow_y_mean",
    "elbow_y_std",
    "elbow_y_delta",
    "elbow_y_ptp",
    "wrist_speed_mean",
    "wrist_speed_max",
    "elbow_angle_mean",
    "elbow_angle_min",
    "elbow_angle_max",
    "elbow_angle_delta",
    "wrist_above_shoulder_ratio",
    "wrist_cross_body_ratio",
]


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / len(values))


def delta(values: list[float]) -> float:
    return 0.0 if len(values) < 2 else values[-1] - values[0]


def point_to_point(values: list[float]) -> float:
    return 0.0 if not values else max(values) - min(values)


def ratio(values: list[float], threshold_fn) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if threshold_fn(value)) / len(values)


def angle_degrees(shoulder_x: float, shoulder_y: float, elbow_x: float, elbow_y: float, wrist_x: float, wrist_y: float) -> float:
    upper_x = shoulder_x - elbow_x
    upper_y = shoulder_y - elbow_y
    lower_x = wrist_x - elbow_x
    lower_y = wrist_y - elbow_y
    upper_length = math.hypot(upper_x, upper_y)
    lower_length = math.hypot(lower_x, lower_y)
    if upper_length <= 0 or lower_length <= 0:
        return 0.0
    cosine = max(-1.0, min(1.0, (upper_x * lower_x + upper_y * lower_y) / (upper_length * lower_length)))
    return math.degrees(math.acos(cosine))


def landmark_to_dict(landmark: Any, landmark_type: int) -> dict[str, float]:
    return {
        "type": landmark_type,
        "x": float(landmark.x),
        "y": float(landmark.y),
        "z": float(landmark.z),
        "visibility": float(getattr(landmark, "visibility", 1.0)),
    }


def ensure_pose_model(model_path: Path) -> None:
    if model_path.exists():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pose model to {model_path}")
    urllib.request.urlretrieve(POSE_MODEL_URL, model_path)


def extract_landmarks(video_path: Path, sample_fps: int, cache_path: Path | None, pose_model_path: Path) -> list[dict[str, Any]]:
    if cache_path and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    ensure_pose_model(pose_model_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30
    frame_step = max(1, round(source_fps / max(1, sample_fps)))
    frames: list[dict[str, Any]] = []
    base_options = base_options_module.BaseOptions(model_asset_path=str(pose_model_path))
    options = pose_landmarker.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=running_mode_module.VisionTaskRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with pose_landmarker.PoseLandmarker.create_from_options(options) as landmarker:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_step != 0:
                frame_index += 1
                continue
            timestamp_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect_for_video(image, int(timestamp_ms))
            landmarks = []
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = [
                    landmark_to_dict(landmark, landmark_index)
                    for landmark_index, landmark in enumerate(result.pose_landmarks[0])
                ]
            frames.append({
                "timestamp_ms": float(timestamp_ms),
                "pose_detected": bool(landmarks),
                "landmarks": landmarks,
            })
            frame_index += 1

    capture.release()
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(frames), encoding="utf-8")
    return frames


def normalized_samples(frames: list[dict[str, Any]], marker_ms: float, handedness: str) -> list[dict[str, float]]:
    start_ms = marker_ms - WINDOW_PRE_MS
    end_ms = marker_ms + WINDOW_POST_MS
    hand_shoulder_type = RIGHT_SHOULDER if handedness == "right" else LEFT_SHOULDER
    hand_elbow_type = RIGHT_ELBOW if handedness == "right" else LEFT_ELBOW
    hand_wrist_type = RIGHT_WRIST if handedness == "right" else LEFT_WRIST

    samples: list[dict[str, float]] = []
    for frame in frames:
        timestamp_ms = float(frame["timestamp_ms"])
        if timestamp_ms < start_ms or timestamp_ms > end_ms or not frame.get("pose_detected"):
            continue
        landmarks = {int(landmark["type"]): landmark for landmark in frame["landmarks"]}
        left_shoulder = landmarks.get(LEFT_SHOULDER)
        right_shoulder = landmarks.get(RIGHT_SHOULDER)
        hand_shoulder = landmarks.get(hand_shoulder_type)
        hand_elbow = landmarks.get(hand_elbow_type)
        hand_wrist = landmarks.get(hand_wrist_type)
        if not all([left_shoulder, right_shoulder, hand_shoulder, hand_elbow, hand_wrist]):
            continue

        shoulder_width = math.hypot(right_shoulder["x"] - left_shoulder["x"], right_shoulder["y"] - left_shoulder["y"])
        if shoulder_width < 0.04:
            continue
        center_x = (left_shoulder["x"] + right_shoulder["x"]) / 2
        center_y = (left_shoulder["y"] + right_shoulder["y"]) / 2
        visibility = mean([
            left_shoulder["visibility"],
            right_shoulder["visibility"],
            hand_shoulder["visibility"],
            hand_elbow["visibility"],
            hand_wrist["visibility"],
        ])
        if visibility < 0.2:
            continue

        shoulder_x = (hand_shoulder["x"] - center_x) / shoulder_width
        shoulder_y = (hand_shoulder["y"] - center_y) / shoulder_width
        elbow_x = (hand_elbow["x"] - center_x) / shoulder_width
        elbow_y = (hand_elbow["y"] - center_y) / shoulder_width
        wrist_x = (hand_wrist["x"] - center_x) / shoulder_width
        wrist_y = (hand_wrist["y"] - center_y) / shoulder_width

        samples.append({
            "timestamp_ms": timestamp_ms,
            "visibility": visibility,
            "wrist_x": wrist_x,
            "wrist_y": wrist_y,
            "elbow_x": elbow_x,
            "elbow_y": elbow_y,
            "shoulder_y": shoulder_y,
            "elbow_angle": angle_degrees(shoulder_x, shoulder_y, elbow_x, elbow_y, wrist_x, wrist_y),
        })

    return sorted(samples, key=lambda sample: sample["timestamp_ms"])


def build_features(frames: list[dict[str, Any]], marker_ms: float, handedness: str) -> dict[str, float] | None:
    samples = normalized_samples(frames, marker_ms, handedness)
    if len(samples) < MIN_FRAMES:
        return None
    avg_visibility = mean(sample["visibility"] for sample in samples)
    if avg_visibility < MIN_AVG_VISIBILITY:
        return None

    wrist_x = [sample["wrist_x"] for sample in samples]
    wrist_y = [sample["wrist_y"] for sample in samples]
    elbow_x = [sample["elbow_x"] for sample in samples]
    elbow_y = [sample["elbow_y"] for sample in samples]
    elbow_angles = [sample["elbow_angle"] for sample in samples]
    wrist_speeds: list[float] = []
    for sample_index in range(1, len(samples)):
        previous = samples[sample_index - 1]
        current = samples[sample_index]
        delta_ms = current["timestamp_ms"] - previous["timestamp_ms"]
        if delta_ms <= 0:
            continue
        wrist_speeds.append(math.hypot(current["wrist_x"] - previous["wrist_x"], current["wrist_y"] - previous["wrist_y"]) / delta_ms * 1000)

    handedness_sign = 1 if handedness == "right" else -1
    return {
        "frame_count": float(len(samples)),
        "avg_visibility": float(avg_visibility),
        "wrist_x_mean": mean(wrist_x),
        "wrist_x_std": std(wrist_x),
        "wrist_x_min": min(wrist_x),
        "wrist_x_max": max(wrist_x),
        "wrist_x_delta": delta(wrist_x),
        "wrist_x_ptp": point_to_point(wrist_x),
        "wrist_y_mean": mean(wrist_y),
        "wrist_y_std": std(wrist_y),
        "wrist_y_min": min(wrist_y),
        "wrist_y_max": max(wrist_y),
        "wrist_y_delta": delta(wrist_y),
        "wrist_y_ptp": point_to_point(wrist_y),
        "elbow_x_mean": mean(elbow_x),
        "elbow_x_std": std(elbow_x),
        "elbow_x_delta": delta(elbow_x),
        "elbow_x_ptp": point_to_point(elbow_x),
        "elbow_y_mean": mean(elbow_y),
        "elbow_y_std": std(elbow_y),
        "elbow_y_delta": delta(elbow_y),
        "elbow_y_ptp": point_to_point(elbow_y),
        "wrist_speed_mean": mean(wrist_speeds) if wrist_speeds else 0.0,
        "wrist_speed_max": max(wrist_speeds) if wrist_speeds else 0.0,
        "elbow_angle_mean": mean(elbow_angles),
        "elbow_angle_min": min(elbow_angles),
        "elbow_angle_max": max(elbow_angles),
        "elbow_angle_delta": delta(elbow_angles),
        "wrist_above_shoulder_ratio": ratio([sample["wrist_y"] - sample["shoulder_y"] for sample in samples], lambda value: value < 0),
        "wrist_cross_body_ratio": ratio(wrist_x, lambda value: value * handedness_sign < 0),
    }


def find_video_path(session_path: Path, video_filename: str) -> Path | None:
    candidates = [
        session_path.parent / video_filename,
        session_path.parent / session_path.stem / video_filename,
    ]
    return next((candidate for candidate in candidates if candidate.exists()), None)


def preprocess(raw_dir: Path, out_csv: Path, landmark_dir: Path, pose_model_path: Path, sample_fps: int) -> None:
    rows: list[dict[str, Any]] = []
    session_paths = sorted(raw_dir.rglob("video_stroke_session_*.json"))
    audio_session_paths = sorted(raw_dir.rglob("audio_session_*.json"))
    if raw_dir == DEFAULT_RAW_DIR and DEFAULT_AUDIO_RAW_DIR.exists():
        audio_session_paths.extend(sorted(DEFAULT_AUDIO_RAW_DIR.rglob("audio_session_*.json")))
    if not session_paths and not audio_session_paths:
        raise SystemExit(f"No video stroke or audio+video pose sessions found under {raw_dir}")

    for session_path in session_paths:
        session = json.loads(session_path.read_text(encoding="utf-8"))
        meta = session.get("session_meta", {})
        handedness = meta.get("handedness", "right")
        for take in session.get("takes", []):
            video_filename = take.get("video_filename")
            if not video_filename:
                continue
            video_path = find_video_path(session_path, video_filename)
            if not video_path:
                print(f"Missing video for {session_path.name}: {video_filename}")
                continue
            cache_path = landmark_dir / session_path.stem / f"{Path(video_filename).stem}.pose.json"
            frames = extract_landmarks(video_path, sample_fps, cache_path, pose_model_path)
            for marker in take.get("markers", []):
                if marker.get("review_status") != "confirmed":
                    continue
                stroke_type = marker.get("stroke_type")
                if stroke_type not in {"forehand", "backhand", "unknown"}:
                    continue
                features = build_features(frames, float(marker["timestamp_ms"]), handedness)
                if not features:
                    continue
                rows.append({
                    "session_id": session_path.stem,
                    "player_name": meta.get("player_name", ""),
                    "handedness": handedness,
                    "camera_facing": meta.get("camera_facing", ""),
                    "camera_angle": meta.get("camera_angle", ""),
                    "camera_side": meta.get("camera_side", ""),
                    "video_filename": video_filename,
                    "take_index": take.get("take_index"),
                    "marker_id": marker.get("id"),
                    "timestamp_ms": marker.get("timestamp_ms"),
                    "stroke_type": stroke_type,
                    "feature_spec": FEATURE_SPEC,
                    **features,
                })

    for session_path in sorted(set(audio_session_paths)):
        if session_path.stem in EXCLUDED_AUDIO_VIDEO_SESSIONS:
            print(f"Skipping diagnostic-only audio/video session: {session_path.stem}")
            continue
        session = json.loads(session_path.read_text(encoding="utf-8"))
        meta = session.get("session_meta", {})
        for event in session.get("events", []):
            if event.get("recording_mode") != "audio_video_pose" and event.get("collection_type") != "audio_video_pose":
                continue
            video_recording = event.get("video_recording") or {}
            video_filename = video_recording.get("video_filename")
            if not video_filename:
                continue
            video_path = find_video_path(session_path, video_filename)
            if not video_path:
                print(f"Missing audio+video pose video for {session_path.name}: {video_filename}")
                continue
            handedness = event.get("player_handedness") or meta.get("handedness", "right")
            cache_path = landmark_dir / session_path.stem / f"{Path(video_filename).stem}.pose.json"
            frames = extract_landmarks(video_path, sample_fps, cache_path, pose_model_path)
            review = event.get("review") or {}
            if not review.get("completed_at"):
                continue
            for marker in review.get("markers", []):
                review_status = marker.get("review_status") or ""
                motion_label = marker.get("motion_label")
                class_label = marker.get("class_label")
                is_pose_negative = (
                    review_status == "ignored"
                    and motion_label == "unknown"
                    and (
                        marker.get("linked_pose_candidate_id")
                        or marker.get("motion_confidence") is not None
                        or marker.get("event_type") in {"motion", "ignore"}
                    )
                )
                if review_status not in {"confirmed", "edited"} and not is_pose_negative:
                    continue
                stroke_type = "unknown" if is_pose_negative else (
                    motion_label if motion_label in {"forehand", "backhand"} else None
                )
                if not stroke_type and class_label == "forehand_hit":
                    stroke_type = "forehand"
                if not stroke_type and class_label == "backhand_hit":
                    stroke_type = "backhand"
                if stroke_type not in {"forehand", "backhand", "unknown"}:
                    continue
                features = build_features(frames, float(marker["timestamp_ms"]), handedness)
                if not features:
                    continue
                rows.append({
                    "session_id": session_path.stem,
                    "player_name": meta.get("player_name") or meta.get("recorder_name", ""),
                    "handedness": handedness,
                    "camera_facing": event.get("camera_facing", ""),
                    "camera_angle": event.get("camera_angle") or meta.get("camera_angle") or "front_oblique",
                    "camera_side": event.get("camera_side") or meta.get("camera_side", ""),
                    "video_filename": video_filename,
                    "take_index": event.get("take_index"),
                    "marker_id": marker.get("id"),
                    "timestamp_ms": marker.get("timestamp_ms"),
                    "stroke_type": stroke_type,
                    "feature_spec": FEATURE_SPEC,
                    **features,
                })

    if not rows:
        raise SystemExit("No trainable video stroke markers found.")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {len(rows)} rows to {out_csv}")
    print(pd.Series([row["stroke_type"] for row in rows]).value_counts().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess reviewed video stroke sessions.")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--out-csv", default=str(DEFAULT_OUT_CSV))
    parser.add_argument("--landmark-dir", default=str(DEFAULT_LANDMARK_DIR))
    parser.add_argument("--pose-model", default=str(DEFAULT_POSE_MODEL))
    parser.add_argument("--sample-fps", type=int, default=LANDMARK_SAMPLE_FPS)
    args = parser.parse_args()
    preprocess(Path(args.raw_dir), Path(args.out_csv), Path(args.landmark_dir), Path(args.pose_model), args.sample_fps)


if __name__ == "__main__":
    main()
