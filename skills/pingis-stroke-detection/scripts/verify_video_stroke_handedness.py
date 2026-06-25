"""Compare video stroke predictions under left vs right handedness.

This diagnoses the common failure mode where the profile hand does not match
the racket hand in the video. It extracts MediaPipe pose frames, estimates
which wrist moves more, then runs the app-exported video stroke model at the
session marker timestamps with both handedness assumptions.

Example:
  python skills/pingis-stroke-detection/scripts/verify_video_stroke_handedness.py \
      data/video/raw/diag_0611/media/video_stroke_session_2026-06-11_001.mp4 \
      data/video/raw/diag_0611/video_stroke_session_2026-06-11_001.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_video_stroke_v2 as v2  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_APP_MODEL = ROOT_DIR / "apps" / "collector" / "src" / "models" / "video_stroke_model.json"
DEFAULT_POSE_MODEL = ROOT_DIR / "data" / "video" / "models" / "pose_landmarker_lite.task"


def extract_marker_times(session_json: Path, take_index: int) -> list[float]:
    session = json.loads(session_json.read_text(encoding="utf-8"))
    takes = session.get("takes") or []
    if take_index >= len(takes):
        raise ValueError(f"take_index {take_index} is outside {len(takes)} takes")
    markers = takes[take_index].get("markers") or []
    return [float(m["timestamp_ms"]) for m in markers if "timestamp_ms" in m]


def extract_pose_frames(video_path: Path, pose_model: Path, pose_fps: float) -> list[dict[str, Any]]:
    import cv2  # noqa: PLC0415
    import mediapipe as mp  # noqa: PLC0415
    from mediapipe.tasks.python.core import base_options as bo  # noqa: PLC0415
    from mediapipe.tasks.python.vision import pose_landmarker as plm  # noqa: PLC0415
    from mediapipe.tasks.python.vision.core import vision_task_running_mode as rm  # noqa: PLC0415

    options = plm.PoseLandmarkerOptions(
        base_options=bo.BaseOptions(model_asset_path=str(pose_model)),
        running_mode=rm.VisionTaskRunningMode.VIDEO,
    )
    frames: list[dict[str, Any]] = []
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / pose_fps))
    with plm.PoseLandmarker.create_from_options(options) as landmarker:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                result = landmarker.detect_for_video(image, ts_ms)
                if result.pose_landmarks:
                    frames.append(
                        {
                            "timestamp_ms": float(ts_ms),
                            "pose_detected": True,
                            "landmarks": [
                                {
                                    "type": landmark_type,
                                    "x": landmark.x,
                                    "y": landmark.y,
                                    "z": landmark.z,
                                    "visibility": landmark.visibility,
                                }
                                for landmark_type, landmark in enumerate(result.pose_landmarks[0])
                            ],
                        }
                    )
            idx += 1
    cap.release()
    return frames


def wrist_travel(frames: list[dict[str, Any]], wrist_type: int) -> float:
    points: list[tuple[float, float]] = []
    for frame in frames:
        landmarks = {int(p["type"]): p for p in frame.get("landmarks", [])}
        if wrist_type in landmarks:
            point = landmarks[wrist_type]
            points.append((float(point["x"]), float(point["y"])))
    return float(
        sum(abs(points[i][0] - points[i - 1][0]) + abs(points[i][1] - points[i - 1][1])
            for i in range(1, len(points)))
    )


def load_app_model(model_path: Path) -> dict[str, Any]:
    return json.loads(model_path.read_text(encoding="utf-8"))


def predict_label(model: dict[str, Any], features: dict[str, float]) -> tuple[str, float]:
    names = model["feature_names"]
    mean = np.array(model["scaler_mean"], dtype=float)
    std = np.array(model["scaler_std"], dtype=float)
    labels = model["labels"]
    x = np.array([float(features.get(name, 0.0)) for name in names], dtype=float)
    xs = (x - mean) / np.where(std == 0, 1, std)
    acc = np.zeros(len(labels), dtype=float)
    for tree in model["trees"]:
        node = tree[0]
        while not (
            len(node) == len(labels)
            and all(0 <= value <= 1 for value in node)
            and abs(sum(node) - 1) < 0.01
        ):
            node = tree[node[2] if xs[int(node[0])] <= node[1] else node[3]]
        acc += np.array(node, dtype=float)
    probabilities = acc / len(model["trees"])
    index = int(np.argmax(probabilities))
    return str(labels[index]), float(probabilities[index])


def predictions_for_handedness(
    frames: list[dict[str, Any]],
    marker_times: list[float],
    model: dict[str, Any],
    handedness: str,
) -> list[str]:
    predictions: list[str] = []
    for timestamp_ms in marker_times:
        features = v2.extract_v2_features(frames, float(timestamp_ms), handedness)
        if features is None:
            predictions.append("(no frames)")
            continue
        app_features = {k: value for k, value in features.items() if k not in v2.APP_EXCLUDED_FEATURES}
        label, _confidence = predict_label(model, app_features)
        predictions.append(label)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("session_json", type=Path)
    parser.add_argument("--take-index", type=int, default=0)
    parser.add_argument("--app-model", type=Path, default=DEFAULT_APP_MODEL)
    parser.add_argument("--pose-model", type=Path, default=DEFAULT_POSE_MODEL)
    parser.add_argument("--pose-fps", type=float, default=15.0)
    args = parser.parse_args()

    marker_times = extract_marker_times(args.session_json, args.take_index)
    print(f"markers: {len(marker_times)}")

    frames = extract_pose_frames(args.video, args.pose_model, args.pose_fps)
    print(f"pose frames: {len(frames)}")

    left_travel = wrist_travel(frames, 15)
    right_travel = wrist_travel(frames, 16)
    detected = "right" if right_travel > left_travel else "left"
    print(f"left wrist travel: {left_travel:.1f} | right: {right_travel:.1f} -> detected={detected}")

    model = load_app_model(args.app_model)
    for handedness in ("left", "right"):
        labels = predictions_for_handedness(frames, marker_times, model, handedness)
        print(f"handedness={handedness}: {dict(Counter(labels))}")


if __name__ == "__main__":
    main()
