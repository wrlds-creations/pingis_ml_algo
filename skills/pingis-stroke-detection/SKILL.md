---
name: pingis-stroke-detection
description: Work on pingis video-stroke detection, reviewed Video FH/BH data, pose preprocessing, RandomForest training, model export, and scripts under this skill.
---

# Skill: pingis-stroke-detection

## Trigger

Load this skill when working on:
- `Video FH/BH` data, review, preprocessing, training, or export
- `Ljud + video ML` motion rows after audio/video review
- video-stroke model debugging for forehand, backhand, or unknown motion
- any file in `skills/pingis-stroke-detection/scripts/`

## Current Scope

This skill is audio/video-only. Legacy AirHive/IMU work is retired from the active product scope and should not be restarted without a new explicit decision and ticket.

Video stroke rows are separate from audio rows. Use reviewed or corrected motion markers as training truth; auto-candidates are diagnostic until Love confirms or corrects them.

## Active Pipeline

```text
Reviewed video-stroke session JSON
    |
    v
preprocess_video_strokes.py
    |
    v
train_rf_video_stroke.py
    |
    v
export_video_stroke_model_json.py
    |
    v
Collector video_stroke_model.json
```

## Key Files

- `scripts/preprocess_video_strokes.py` - builds video-stroke training rows from reviewed sessions
- `scripts/train_rf_video_stroke.py` - trains the RandomForest video-stroke model
- `scripts/export_video_stroke_model_json.py` - exports the trained model into app-readable JSON

## Working Rules

- Keep `audio_model`, `audio_contact_model`, and `video_stroke_model` separate.
- Do not train video rows from unreviewed auto-candidates.
- Keep `unknown` motion available for opponent/no-visible-player/no-clear-stroke cases.
- Preserve `camera_side`, selected-player handedness, and 15 fps pose assumptions in reports and evaluation notes.
- Update `PROJECT_CONTEXT.md`, `REPO_CURRENT_STATE.md`, and `ITERATION_LOG.md` when video-model facts or build/device state changes.
