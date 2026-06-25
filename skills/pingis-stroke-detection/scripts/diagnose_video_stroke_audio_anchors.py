"""Run video-stroke model diagnostics at Fable/audio anchor timestamps.

This answers: if the audio side finds N anchors, how many would pass the
video-stroke feature/model gate with offline MediaPipe pose? If Python gets
many confident forehand/backhand hits while the app gets almost none, suspect
the app pose stream, camera rotation handling, or timestamp alignment.

Example:
  python skills/pingis-stroke-detection/scripts/diagnose_video_stroke_audio_anchors.py \
      data/video/raw/diag_0611/media/video_stroke_session_2026-06-11_001.mp4 \
      --handedness right
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path(__file__).resolve().parents[3]
AUDIO_NOISE_ROBUST_DIR = (
    ROOT_DIR / "skills" / "pingis-audio-classification" / "scripts" / "noise_robust"
)
AUDIO_SCRIPTS_DIR = ROOT_DIR / "skills" / "pingis-audio-classification" / "scripts"
for _path in (str(SCRIPT_DIR), str(AUDIO_NOISE_ROBUST_DIR), str(AUDIO_SCRIPTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import train_video_stroke_v2 as v2  # noqa: E402
from compare_fable_anchor_gates import app_find_audio_peaks  # noqa: E402
from preprocess_audio import load_audio  # noqa: E402
from verify_video_stroke_handedness import (  # noqa: E402
    DEFAULT_APP_MODEL,
    DEFAULT_POSE_MODEL,
    extract_pose_frames,
    load_app_model,
    predict_label,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--handedness", choices=["left", "right"], default="right")
    parser.add_argument("--confidence", type=float, default=0.58)
    parser.add_argument("--app-model", type=Path, default=DEFAULT_APP_MODEL)
    parser.add_argument("--pose-model", type=Path, default=DEFAULT_POSE_MODEL)
    parser.add_argument("--pose-fps", type=float, default=15.0)
    parser.add_argument("--anchor-gap-ms", type=float, default=350.0)
    parser.add_argument("--max-anchors", type=int, default=220)
    args = parser.parse_args()

    y, sr = load_audio(str(args.video))
    peaks = app_find_audio_peaks(
        y,
        sr,
        frame_ms=10.0,
        min_rms_floor=0.004,
        median_mult=4.0,
        p75_mult=1.8,
        max_mult=0.16,
        local_radius=2,
        anchor_gap_ms=args.anchor_gap_ms,
        max_anchors=args.max_anchors,
    )
    anchors = [float(ts) for ts, _rms in peaks["anchors"]]
    print(
        f"audio anchors: {len(anchors)} | threshold={peaks['threshold']:.5f} | "
        f"gap={args.anchor_gap_ms:.0f} ms"
    )

    frames = extract_pose_frames(args.video, args.pose_model, args.pose_fps)
    print(f"pose frames: {len(frames)}")

    model = load_app_model(args.app_model)
    labels: Counter[str] = Counter()
    confident_strokes = 0
    no_frames = 0
    for timestamp_ms in anchors:
        features = v2.extract_v2_features(frames, timestamp_ms, args.handedness)
        if features is None:
            no_frames += 1
            continue
        app_features = {k: value for k, value in features.items() if k not in v2.APP_EXCLUDED_FEATURES}
        label, confidence = predict_label(model, app_features)
        labels[label] += 1
        if label in ("forehand", "backhand") and confidence >= args.confidence:
            confident_strokes += 1

    print(f"predictions at anchors: {dict(labels)}")
    print(f"anchors without pose-window features: {no_frames}")
    print(
        f"forehand/backhand with confidence >= {args.confidence:.2f}: "
        f"{confident_strokes} / {len(anchors)}"
    )


if __name__ == "__main__":
    main()
