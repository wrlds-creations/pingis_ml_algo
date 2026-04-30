# Pingis ML App

This repository contains the WRLDS pingis ML app and local training pipeline.

The project goal is to build a mobile workflow that can collect trustworthy pingis audio/video/IMU data, review labels on-device, train local ML models, export model artifacts into the React Native app, and validate live behavior on a physical Android device.

## Current Focus

Audio comes first.

The active goal is a stable racket-contact detector:

- Detect `racket_contact` reliably.
- Reject floor, table, speech, music, desk sounds, claps, clicks, and other hard negatives.
- Avoid double-counting one physical contact.
- Work for both straight-up racket bounces and forehand/backhand-style racket contacts.

IMU work exists but is paused until audio is stable enough to fuse with swing evidence.

## Read First

Before coding:

1. Read `AGENTS.md`.
2. Read `PROJECT_CONTEXT.md`.
3. Read `DECISIONS.md`.
4. For ML/audio/bounce/IMU work, read `ITERATION_LOG.md`.
5. Load the relevant local skill under `skills/`.

## Repository Shape

- `apps/collector/`: React Native Android collector/test app.
- `skills/pingis-audio-classification/`: audio preprocessing, training, export scripts, and workflow notes.
- `skills/pingis-stroke-detection/`: IMU/stroke preprocessing, training, export scripts, and workflow notes.
- `skills/berg-airhive-imu-3d-view/`: AirHive BLE protocol/orientation reference.
- `data/`: local raw/processed training data, intentionally gitignored.
- `PROJECT_CONTEXT.md`: stable project facts.
- `DECISIONS.md`: durable architecture/product decisions.
- `ITERATION_LOG.md`: detailed model, data, build, and device feedback history.

## Validation

Root WRLDS template validation:

```bash
npm run validate
```

Collector commands live under `apps/collector/`. Use the latest working Android build notes in `ITERATION_LOG.md` before trusting a test APK.

## Data And Model Loop

1. Collect takes on the Motorola.
2. Review markers in the app.
3. Pull session JSON/media into local raw data.
4. Run deterministic preprocessing scripts.
5. Train binary contact and 4-class audio models.
6. Export JSON model artifacts into the app.
7. Build/install a test APK.
8. Validate live behavior and update `ITERATION_LOG.md`.

## Branching

Do not push directly to `main`. Work on a feature branch and merge through GitHub review.
