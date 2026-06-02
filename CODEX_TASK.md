# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0009`

## Branch

`codex/t0009-playing-retro-audio-review-integration`

## Goal

Integrate the validated T0007/T0008 `spel_retro_audio` model family behind a separate post-recording Review retro path, with app-side feature extraction matching the Python tight/normal/wide/context pipeline, without changing `studs_live` or the normal Collector `audio_model.json` path.

## Dependencies

- T0001 ticket workflow adoption is complete.
- T0002 documentation refresh removed IMU/AirHive from active project scope.
- T0003 cleanup removed retired IMU/AirHive workflow docs and skill scripts from the active repo workflow.
- T0004 generated candidate-centered playing-retro diagnostics locally from saved app candidates plus replay peaks.
- T0005 trained local candidate `playing_retro_audio_rf_v2026_06_02_app_candidates_100_200` from 4,028 candidate-centered rows across 16 reviewed playing sessions.
- T0006 selected local one-window candidate `playing_retro_audio_rf_v2026_06_02_safe_racket_weighted`, but the safe gain was too small for app integration.
- T0007 selected local multi-window/context candidate `playing_retro_audio_rf_v2026_06_02_multi_window_context` / `multi_window_context_racket_weighted`.
- T0008 cross-session validation passed across `audio_session_2026-05-28_002`, `audio_session_2026-05-29_001`, and `audio_session_2026-05-29_002`: selected T0007 racket recall was `0.910`, `0.939`, and `0.896`, table recall was `0.932`, `0.958`, and `0.933`, and non-target recall was `0.894`, `0.859`, and `0.833`.
- Current audio source-of-truth lives in `PROJECT_CONTEXT.md`, `DECISIONS.md`, and `ITERATION_LOG.md`.
- Existing audio scripts and replay behavior live under `skills/pingis-audio-classification/scripts/`.

## Allowed areas

- `skills/pingis-audio-classification/scripts/`
- `skills/pingis-audio-classification/SKILL.md`
- `apps/collector/src/AudioTakeReviewScreen.tsx`
- `apps/collector/src/audioReview.ts`
- `apps/collector/src/types.ts`
- New clearly named `apps/collector/src/*playingRetroAudio*` helper/model files if needed
- `data/audio/models/evaluations/`
- `data/audio/models/playing_retro_candidates/`
- `PROJECT_CONTEXT.md`
- `DECISIONS.md`
- `ITERATION_LOG.md`
- `REPO_CURRENT_STATE.md`
- `FOLLOWUPS.md`
- `CODEX_TASK.md`

## Do not touch

- `apps/collector/src/models/audio_model.json`
- `apps/collector/src/models/audio_contact_model.json`
- Existing `studs_live` app behavior or live detector thresholds
- APK/build artifacts unless Love explicitly asks for a build
- Video-stroke model files
- Raw reviewed session JSON labels, except for metadata fixes explicitly approved by Love

## Requirements

- Start from the T0007/T0008 reports, prediction CSVs, and selected local candidate.
- Define the app-side model/export shape for `spel_retro_audio` separately from normal `audio_model.json`.
- Implement or stage feature extraction so Review retro can compute the same tight `-60/+140 ms`, normal `-100/+200 ms`, wide `-160/+320 ms`, and candidate-context features used by Python.
- Keep truth-derived fields such as `close_event_bucket` and `neighbor_sequence` out of app inference features.
- Add an app/local parity check where practical: the same candidate timestamp should produce the same feature names/order as the Python model expects.
- Keep all new behavior behind a separate playing-retro path; normal Review candidates and `studs_live` must keep using their existing model/config.
- Do not build or install an APK unless Love explicitly asks.

## Non-Goals

- No `studs_live` promotion.
- No ordinary up/down bounce model change.
- No video/FH-BH fusion yet.
- No broad UI redesign.
- No app release build unless explicitly requested.

## Acceptance criteria

- The app/review code has a clearly separate `spel_retro_audio` path or the ticket documents why a prerequisite export/parity step is still needed.
- App-side features and Python feature order are traceable and testable.
- Existing `audio_model.json` and `audio_contact_model.json` are unchanged.
- The ticket leaves a deterministic command or test for feature/export parity.

## Manual verification

- Inspect a few T0008 prediction rows and confirm the app-side path would classify the same candidate timestamps, not human truth timestamps.
- Confirm normal audio review and live bounce paths do not use the new `spel_retro_audio` model.

## Automated validation

- Run targeted TypeScript validation if app code changes.
- Run any new parity/export check.
- Run `npm run validate` if source-of-truth workflow files changed.
- Run targeted Python syntax checks for changed Python scripts.

## Completion Report Expected

Codex should report:

- Summary of changes
- Files changed
- Commands run
- Model/report outputs generated
- Validation results
- Manual verification performed or still needed
- Docs updated or still needed
- Risks or unresolved questions
- Follow-up tickets for out-of-scope work
