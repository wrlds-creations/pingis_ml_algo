"""Dump fixture för TS-paritetskontroll av video-stroke v2.

Tar 30 markörer från två sessioner (en FH-tung, en BH-tung), beräknar
Python-referensens v2-features (utan z) + appmodellens förväntade
sannolikheter, och sparar pose-frames per video.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import train_video_stroke_v2 as v2

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "video" / "models" / "video_stroke_v2" / "ts_parity_fixture.json"
APP_MODEL = ROOT / "apps" / "collector" / "src" / "models" / "video_stroke_model.json"

meta = pd.read_csv(v2.DATASET_CSV)
meta = meta[meta["stroke_type"].isin(v2.CLASSES)]
pick = meta[meta["session_id"].isin(["audio_session_2026-05-22_001", "audio_session_2026-05-22_003"])]
pick = pick.groupby("session_id").head(15)

model = json.loads(APP_MODEL.read_text(encoding="utf-8"))
feature_names = model["feature_names"]
mean = np.array(model["scaler_mean"])
std = np.array(model["scaler_std"])
labels = model["labels"]


def predict(features: dict) -> dict:
    x = np.array([float(features.get(n, 0.0)) for n in feature_names])
    xs = (x - mean) / np.where(std == 0, 1, std)
    n_classes = len(labels)
    acc = np.zeros(n_classes)
    for tree in model["trees"]:
        node = tree[0]
        while not (len(node) == n_classes and all(0 <= v <= 1 for v in node) and abs(sum(node) - 1) < 0.01):
            node = tree[node[2] if xs[int(node[0])] <= node[1] else node[3]]
        acc += np.array(node)
    p = acc / len(model["trees"])
    return {label: float(v) for label, v in zip(labels, p)}


frames_by_video: dict[str, list] = {}
samples = []
for _, r in pick.iterrows():
    stem = Path(str(r["video_filename"])).stem
    key = f"{r['session_id']}/{stem}"
    pose_path = v2.LANDMARK_DIR / str(r["session_id"]) / f"{stem}.pose.json"
    if not pose_path.exists():
        continue
    if key not in frames_by_video:
        frames_by_video[key] = json.loads(pose_path.read_text(encoding="utf-8"))
    feats = v2.extract_v2_features(frames_by_video[key], float(r["timestamp_ms"]), str(r["handedness"]))
    if feats is None:
        continue
    feats = {k: v for k, v in feats.items() if k not in v2.APP_EXCLUDED_FEATURES}
    samples.append({
        "video": key,
        "marker_ms": float(r["timestamp_ms"]),
        "handedness": str(r["handedness"]),
        "py_features": {k: float(val) for k, val in feats.items()},
        "py_proba": predict(feats),
    })

OUT.write_text(json.dumps({"frames": frames_by_video, "samples": samples}), encoding="utf-8")
print(f"Wrote {OUT} with {len(samples)} samples, {len(frames_by_video)} videos")
