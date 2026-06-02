# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0011`

## Branch

`codex/t0011-playing-retro-audio-review-ui-apk`

## Goal

Wire the T0009/T0010 `spel_retro_audio` path into a visible, separate post-recording Review flow and decide/build the first APK only after the Review behavior is useful enough to test on Motorola.

## Dependencies

- T0009 exported separate app model `apps/collector/src/models/playing_retro_audio_model.json`.
- T0009 added opt-in app helper `apps/collector/src/playingRetroAudio.ts`.
- T0010 replay passed on saved Tomas/Stiga Review candidates: 643 saved candidates, accuracy `0.978`, racket recall `0.987`, table recall `0.988`, non-target recall `0.948`.
- T0010 also showed 15 reviewed missed markers in the target sessions are not classifiable by saved-candidate replay because no saved app candidate exists at those timestamps.
- Existing live/current app audio still uses `apps/collector/src/models/audio_model.json` through the existing normal path.

## Allowed Areas

- `apps/collector/src/playingRetroAudio.ts`
- `apps/collector/src/AudioTakeReviewScreen.tsx`
- `apps/collector/src/audioReview.ts`
- `apps/collector/src/types.ts`
- New clearly named `apps/collector/src/*playingRetroAudio*` helper files
- New clearly named local replay/check scripts under `skills/pingis-audio-classification/scripts/`
- Android APK/build files only if Love explicitly asks to build after the Review path is ready
- `PROJECT_CONTEXT.md`
- `DECISIONS.md`
- `ITERATION_LOG.md`
- `REPO_CURRENT_STATE.md`
- `FOLLOWUPS.md`
- `CODEX_TASK.md`

## Do Not Touch

- `apps/collector/src/models/audio_model.json`
- `apps/collector/src/models/audio_contact_model.json`
- Existing `studs_live` app behavior or live detector thresholds
- Video-stroke model files
- Raw reviewed session JSON labels, except for metadata fixes explicitly approved by Love

## Requirements

- Keep `spel_retro_audio` visibly separate from normal Review candidates and live bounce.
- Decide whether the first APK should only reclassify saved candidates or also generate/surface additional retro candidates for missed markers.
- If adding candidate generation, keep it post-recording only and report how many new candidates it surfaces near reviewed missed markers.
- Preserve normal Review behavior unless the separate retro path is explicitly invoked.
- Keep candidate context based on app candidate timestamps or explicitly documented generated retro candidate timestamps, never human truth timestamps.
- Keep truth-derived fields such as `close_event_bucket` and `neighbor_sequence` out of inference features.
- Do not build or install an APK until Love gives explicit go-ahead after the Review behavior is implemented and validated.

## Non-Goals

- No `studs_live` promotion.
- No ordinary up/down bounce model change.
- No video/FH-BH fusion yet.
- No broad UI redesign.

## Acceptance Criteria

- Review has a clearly separate way to run or show `spel_retro_audio` output.
- Normal Review and live bounce paths still use their existing models/configs by default.
- The path can be tested against the T0010 Tomas/Stiga sessions before APK build.
- Validation includes TypeScript and T0010 replay/parity commands.
- APK build/install is either completed with Love's explicit approval or intentionally deferred with a clear reason.

## Manual Verification

- Confirm normal Review candidates still render as before when the new retro path is not invoked.
- Confirm `rfInference.ts` still imports only `audio_model.json` for normal audio.
- Confirm the retro path does not modify reviewed marker truth automatically.

## Automated Validation

- Run `python skills/pingis-audio-classification/scripts/validate_playing_retro_audio_app_export.py`.
- Run `python skills/pingis-audio-classification/scripts/replay_playing_retro_audio_app_export.py`.
- Run targeted Python syntax checks for changed Python scripts.
- Run `cd apps/collector && npx tsc --noEmit` if app code changes.
- Run `npm run validate` if source-of-truth workflow files changed.

## Completion Report Expected

Codex should report:

- Summary of changes
- Files changed
- Commands run
- Replay/session metrics generated
- Validation results
- Manual verification performed or still needed
- Docs updated or still needed
- APK build/install status
- Risks or unresolved questions
- Follow-up tickets for out-of-scope work
