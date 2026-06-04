# Pingis ML App

This repository contains the WRLDS pingis ML app and local training pipeline.

The project goal is to build a mobile workflow that can collect trustworthy pingis audio/video data, review labels on-device, train local ML models, export model artifacts into the React Native app, and validate live behavior on a physical Android device.

## Current Focus

Audio comes first.

The active goal is a stable racket-contact detector:

- Detect `racket_contact` reliably.
- Reject floor, table, speech, music, desk sounds, claps, clicks, and other hard negatives.
- Avoid double-counting one physical contact.
- Work for both straight-up racket bounces and forehand/backhand-style racket contacts.

External IMU/AirHive sensor work is no longer active scope. The current product direction is audio first, then video-supported retro analysis.

## Read First

Before coding:

1. Read `AGENTS.md`.
2. Read `PROJECT_CONTEXT.md`.
3. Read `DECISIONS.md`.
4. Read `REPO_CURRENT_STATE.md`.
5. Read the active ticket in `CODEX_TASK.md`.
6. For ML/audio/bounce/video work, read `ITERATION_LOG.md`.
7. Load the relevant local skill under `skills/`.

## Repository Shape

- `apps/collector/`: React Native Android collector/test app.
- `skills/pingis-audio-classification/`: audio preprocessing, training, export scripts, and workflow notes.
- `skills/pingis-stroke-detection/`: video-stroke preprocessing, training, export scripts, and workflow notes.
- `data/`: local raw/processed training data, intentionally gitignored.
- `PROJECT_CONTEXT.md`: stable project facts.
- `DECISIONS.md`: durable architecture/product decisions.
- `CODEX_TASK.md`: one active scoped ticket.
- `REPO_CURRENT_STATE.md`: latest repo snapshot, completed tickets, validation status, and next ticket.
- `FOLLOWUPS.md`: out-of-scope issues and future tickets.
- `ITERATION_LOG.md`: detailed model, data, build, and device feedback history.

## Ticket Workflow

Work one ticket at a time. Each ticket defines the goal, allowed areas, non-goals, acceptance criteria, manual verification, and automated validation. Out-of-scope findings go to `FOLLOWUPS.md`; completed ticket state and next steps go to `REPO_CURRENT_STATE.md`.

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
5. Train and evaluate scoped audio/video model candidates.
6. Export JSON model artifacts into the app only after replay gates pass.
7. Build/install a test APK when the ticket explicitly allows it.
8. Validate live or retro behavior and update `ITERATION_LOG.md`.

## Branching

Do not push directly to `main`. Work on a feature branch and merge through GitHub review.
