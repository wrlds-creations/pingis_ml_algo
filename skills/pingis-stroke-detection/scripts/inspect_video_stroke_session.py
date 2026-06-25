"""Print a compact summary of a video stroke collection session JSON.

Useful before debugging Fable/video handoff issues: it shows take structure,
marker label counts, pose-analysis counts, and small samples of the saved
objects without opening the React Native app.

Example:
  python skills/pingis-stroke-detection/scripts/inspect_video_stroke_session.py \
      data/video/raw/diag_0611/video_stroke_session_2026-06-11_001.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def preview(value: Any, max_chars: int) -> str:
    text = json.dumps(value, indent=1, ensure_ascii=False)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def label_counts(markers: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for marker in markers:
        label = (
            marker.get("motion_label")
            or marker.get("class_label")
            or marker.get("final_label")
            or marker.get("label")
            or "(none)"
        )
        counts[str(label)] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_json", type=Path)
    parser.add_argument("--preview-chars", type=int, default=500)
    args = parser.parse_args()

    data = json.loads(args.session_json.read_text(encoding="utf-8"))
    print("session_meta:")
    print(preview(data.get("session_meta", {}), args.preview_chars))

    for index, take in enumerate(data.get("takes", [])):
        print("=" * 78)
        print(f"take {index}: {take.get('video_filename')}")
        print(f"keys: {sorted(take.keys())}")

        markers = take.get("markers") or []
        print(f"markers: {len(markers)}")
        if markers:
            print(f"marker labels: {dict(label_counts(markers))}")
            print("marker sample:")
            print(preview(markers[0], args.preview_chars))

        pose_analysis = take.get("pose_analysis") or take.get("video_pose_candidates") or []
        print(f"pose_analysis: {len(pose_analysis)}")
        if pose_analysis:
            predictions = Counter(str(p.get("predicted_stroke_type")) for p in pose_analysis)
            print(f"predictions: {dict(predictions)}")
            print("pose sample:")
            print(preview(pose_analysis[0], args.preview_chars))


if __name__ == "__main__":
    main()
