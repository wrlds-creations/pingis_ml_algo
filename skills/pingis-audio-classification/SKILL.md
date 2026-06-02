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
| `scripts/build_playing_retro_candidate_report.py` | Candidate-centered `spel_retro_audio` peak report before training |
| `scripts/train_playing_retro_audio.py` | Local `spel_retro_audio` candidate training/evaluation, separate from app export |
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
python skills/pingis-audio-classification/scripts/build_playing_retro_candidate_report.py
python skills/pingis-audio-classification/scripts/train_playing_retro_audio.py
python skills/pingis-audio-classification/scripts/train_rf_audio.py

# Start inference API:
python skills/pingis-audio-classification/scripts/serve_api_audio.py
```

## New Audio Training Intake

Before pulling or training on a new audio file/session, ask Love for missing
metadata instead of guessing. Do not treat a new file as broadly trainable until
the answers are clear enough to tag and evaluate it correctly.

Required questions:
- Is this session trainable ground truth, diagnostic-only, or holdout/replay-only?
- Which bucket should it evaluate: ordinary vertical racket bounce, dense
  racket+table playing, Stiga-office/Tomas-style hard stroke sounds, table,
  floor, noise/speech/music, or another bucket?
- Was the impact style ordinary up/down bounce, normal play, hard stroke-like
  contact, fast dense sequence, or mixed?
- What environment/background applies: room, table/surface, background noise,
  phone/mic placement, device, and recorder/player when known?
- Should this session be allowed to influence the ordinary bounce detector, or
  only the domain-specific bucket until it passes cross-bucket replay?

Training rules:
- Tag sessions with explicit scenario/domain metadata before preprocessing.
- Keep diagnostic and unfinished review sessions excluded from training.
- Report metrics by bucket, not only aggregate accuracy or macro F1.
- Promotion requires no unacceptable regression on ordinary bounce, even when a
  Stiga/Tomas or dense-playing bucket improves.

## Playing Retro Audio

`spel_retro_audio` is a separate post-recording playing-mode audio path. It is
not the live `studs_live` model and must not be exported into Collector app JSON
without a dedicated ticket.

Current local workflow:
- `build_playing_retro_candidate_report.py` matches saved app candidates and
  replay peaks against reviewed racket/table truth for diagnostics.
- `train_playing_retro_audio.py` trains a local RandomForest candidate from all
  matchable saved app candidate peaks plus manually reviewed missed markers.
- Unmatched app candidates become `non_target` rows with lower sample weight.
- Replay-generated peaks stay diagnostic in T0005 and are not multiplied into
  training rows by timing config.
- Ordinary up/down bounce is evaluated as a separate regression slice; it is not
  mixed into dense playing metrics.

## Collecting Data

Aim for **30+ clips per class**. Recording tips:
- Hold phone ~1 m from the table for consistent distance
- Record in the actual play environment (gym, basement, etc.)
- For `noise`: record actual game noise — crowd, shouts, music
- Vary the shot speed/angle to improve model generalization
