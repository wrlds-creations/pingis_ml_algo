---
name: pingis-audio-classification
description: Work on the pingis audio collection, review, preprocessing, training, model export, live racket-contact detection, table/floor/noise veto behavior, and scripts under this skill.
---

# Skill: pingis-audio-classification

## Trigger
Load this skill when working on:
- Audio recording or bounce sound classification for ping pong
- `AudioCollectionScreen` or audio session JSON files
- Any file in `skills/pingis-audio-classification/scripts/`
- Noise filtering or bounce-type detection from microphone input

## Purpose
Collect labeled audio clips of ping pong ball bounces and train a machine learning
model that can distinguish between:
- `racket_bounce` — ball hitting racket rubber
- `table_bounce` — ball bouncing on an actual ping-pong table or table-like playing surface relevant to STIGA Smart Pingis, not an arbitrary desk/table
- `floor_bounce` — ball bouncing on the floor
- `noise` — background sounds (shouting, applause, conversation)

## Architecture Overview

```
Android Mic (44100 Hz)
    │
[AudioCollectionScreen.tsx]  ← 1-second m4a clips, tap-triggered
    │
audio_session_DATE_NNN.json + audio_session_DATE_NNN/*.m4a
    │
[preprocess_audio.py]        ← librosa MFCC + spectral features (35 total)
    │
data/audio/processed/audio_dataset.csv
    │
[train_rf_audio.py]          ← RandomForest (200 trees)
    │
data/audio/models/
  audio_rf_classifier.pkl
  audio_feature_scaler.pkl
  audio_label_encoder.pkl
    │
[serve_api_audio.py]         ← Flask API on port 5001
  POST /predict_audio         ← base64-encoded m4a → label + confidence
```

## Key Files

| File | Purpose |
|------|---------|
| `apps/collector/src/AudioCollectionScreen.tsx` | Android recording UI |
| `apps/collector/src/types.ts` | AudioLabel, AudioEvent, AudioSessionFile types |
| `scripts/preprocess_audio.py` | Feature extraction (35 features per clip) |
| `scripts/train_rf_audio.py` | Model training + evaluation |
| `scripts/serve_api_audio.py` | Inference API (port 5001) |

## Data Format

Each session produces:
```
data/audio/raw/
  audio_session_2026-04-07_001.json       ← metadata + event list
  audio_session_2026-04-07_001/           ← clip folder
    racket_bounce_000.m4a
    table_bounce_000.m4a
    ...
```

Session JSON schema:
```json
{
  "session_meta": {
    "recorder_name": "Erik",
    "session_date": "2026-04-07T...",
    "app_version": "1.0",
    "clip_duration_ms": 1000
  },
  "events": [
    {
      "label": "racket_bounce",
      "recorded_at": "2026-04-07T...",
      "wav_filename": "racket_bounce_000.m4a",
      "duration_ms": 1000
    }
  ]
}
```

## Feature Set (35 features)
- MFCCs 0-12: mean + std (26 features) — timbral texture differs by surface
- Spectral centroid: mean + std (2) — energy distribution
- Spectral rolloff: mean + std (2) — high-frequency drop-off
- Zero-crossing rate: mean + std (2) — noise has much higher ZCR than impacts
- RMS energy: mean + std (2) — relative loudness of transient
- Onset strength max (1) — sharpest transient peak in the clip

## Quick Start

```bash
pip install -r skills/pingis-audio-classification/requirements.txt

# After collecting sessions with the app:
python skills/pingis-audio-classification/scripts/preprocess_audio.py
python skills/pingis-audio-classification/scripts/train_rf_audio.py

# Start inference API:
python skills/pingis-audio-classification/scripts/serve_api_audio.py
```

## Collecting Data

Aim for **30+ clips per class**. Recording tips:
- Hold phone ~1 m from the table for consistent distance
- Record in the actual play environment (gym, basement, etc.)
- For `noise`: record actual game noise — crowd, shouts, music
- Vary the shot speed/angle to improve model generalization
