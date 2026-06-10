"""
dump_fable_parity_clips.py

Dumps real gate-trigger clips from VAL sessions plus the Python-side
expected features and model probabilities, as input to the Node TS-parity
harness (check_fable_ts_parity.js).

Output: data/audio/processed/noise_robust/fable_clip_parity_fixture.json
  { clips: [ { id, session, wav, onset_sample, frame_rms,
               samples: [6615 float32 values],
               py_features: {83 name->value},
               py_proba: {label->prob} } ] }
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nr_config  # noqa: E402
import nr_features  # noqa: E402
from preprocess_audio import load_audio  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parents[4]
INVENTORY = ROOT_DIR / "data" / "audio" / "processed" / "audio_inventory_2026_06_10.json"
OUT = ROOT_DIR / "data" / "audio" / "processed" / "noise_robust" / "fable_clip_parity_fixture.json"
MODEL_DIR = ROOT_DIR / "data" / "audio" / "models" / "noise_robust_v3"

SESSIONS = [
    "audio_session_2026-05-06_007",  # quiet
    "audio_session_2026-05-12_004",  # music_high
    "audio_session_2026-05-11_005",  # speech
    "audio_session_2026-05-13_008",  # mixed dense
]
PER_WAV = 6


def main() -> None:
    inventory = {s["session_id"]: s for s in json.loads(INVENTORY.read_text(encoding="utf-8"))["sessions"]}
    clf = joblib.load(MODEL_DIR / "nr_histgb_all83.pkl")
    scaler = joblib.load(MODEL_DIR / "nr_scaler_all83.pkl")
    feature_cols = list(joblib.load(MODEL_DIR / "nr_feature_cols_all83.pkl"))
    labels = nr_config.CLASSES

    clips = []
    for session_id in SESSIONS:
        record = inventory[session_id]
        media_dir = ROOT_DIR / record["media_dir"]
        for event in record["events"]:
            wav_name = event.get("wav_filename")
            if not wav_name:
                continue
            wav_path = media_dir / wav_name
            if not wav_path.exists():
                continue
            y, sr = load_audio(str(wav_path))
            triggers = nr_features.simulate_gate(
                y, sr, onset_ratio=1.5, retrigger_ms=120, abs_min_rms=0.0015,
                mode="bandpass", spectral_gate=False,
            )
            if not triggers:
                continue
            keep = np.linspace(0, len(triggers) - 1, min(PER_WAV, len(triggers))).astype(int)
            for ti in sorted(set(int(i) for i in keep)):
                trig = triggers[ti]
                clip = nr_features.extract_live_clip(y, int(trig["onset_sample"]))
                feats = nr_features.extract_all_features(clip, sr)
                x = np.array([[float(feats.get(name, 0.0)) for name in feature_cols]], dtype=np.float64)
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                xs = scaler.transform(x)
                proba = clf.predict_proba(xs)[0]
                clips.append({
                    "id": f"{session_id}:{wav_name}:{ti}",
                    "session": session_id,
                    "wav": wav_name,
                    "onset_sample": int(trig["onset_sample"]),
                    "frame_rms": float(trig["frame_rms"]),
                    "samples": [float(v) for v in clip],
                    "py_features": {name: float(feats[name]) for name in feature_cols},
                    "py_proba": {label: float(p) for label, p in zip(labels, proba)},
                })

    OUT.write_text(json.dumps({"feature_names": feature_cols, "clips": clips}), encoding="utf-8")
    print(f"Wrote {OUT} with {len(clips)} clips")


if __name__ == "__main__":
    main()
