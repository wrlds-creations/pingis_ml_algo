# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0010`

## Branch

`codex/t0010-playing-retro-audio-review-replay-ui`

## Goal

Wire the T0009 `spel_retro_audio` export/runtime into a controlled post-recording Review replay or UI path, compare it against reviewed Tomas/Stiga sessions, and keep it fully separate from `studs_live` and normal `audio_model.json`.

## Dependencies

- T0008 cross-session validation passed for selected T0007 variant `multi_window_context_racket_weighted`.
- T0009 exported `apps/collector/src/models/playing_retro_audio_model.json`.
- T0009 added `apps/collector/src/playingRetroAudio.ts` with tight `-60/+140 ms`, normal `-100/+200 ms`, wide `-160/+320 ms`, and 11 non-leaky candidate-context features.
- T0009 parity command passed: `python skills/pingis-audio-classification/scripts/validate_playing_retro_audio_app_export.py`.
- Existing live/current app audio still uses `apps/collector/src/models/audio_model.json` through the existing normal path.

## Allowed Areas

- `apps/collector/src/playingRetroAudio.ts`
- `apps/collector/src/AudioTakeReviewScreen.tsx`
- `apps/collector/src/audioReview.ts`
- `apps/collector/src/types.ts`
- New clearly named `apps/collector/src/*playingRetroAudio*` helper files
- New clearly named local replay/check scripts under `skills/pingis-audio-classification/scripts/`
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
- APK/build artifacts unless Love explicitly asks for a build
- Video-stroke model files
- Raw reviewed session JSON labels, except for metadata fixes explicitly approved by Love

## Requirements

- Add a controlled way to run `spel_retro_audio` on Review candidates after recording/import, either as a local app replay script or a clearly separate Review UI action.
- Compare output against reviewed candidates/markers for at least `audio_session_2026-05-28_002`, `audio_session_2026-05-29_001`, and `audio_session_2026-05-29_002`.
- Report racket/table/non-target behavior and close-gap behavior, not only aggregate accuracy.
- Keep candidate context based on app candidate timestamps, not human truth timestamps.
- Keep truth-derived fields such as `close_event_bucket` and `neighbor_sequence` out of app inference features.
- Make normal Review and live bounce paths continue using their existing models/configs unless the separate path is explicitly invoked.
- Do not build or install an APK unless Love explicitly asks.

## Non-Goals

- No `studs_live` promotion.
- No ordinary up/down bounce model change.
- No video/FH-BH fusion yet.
- No broad UI redesign.
- No release APK unless explicitly requested.

## Acceptance Criteria

- A deterministic command or app path can run the T0009 model on saved Review candidates.
- Replay/UI output identifies which candidates are classified as racket, table, or non-target by the separate `spel_retro_audio` model.
- The comparison includes the three Tomas/Stiga sessions used in T0008.
- Existing `audio_model.json` and `audio_contact_model.json` are unchanged.
- Validation commands are documented in `REPO_CURRENT_STATE.md`.

## Manual Verification

- Confirm normal Review candidates still render as before when the new retro path is not invoked.
- Confirm `rfInference.ts` still imports only `audio_model.json` for normal audio.
- Confirm the new retro output uses candidate timestamps from saved model candidates, not moved/confirmed human marker timestamps.

## Automated Validation

- Run `python skills/pingis-audio-classification/scripts/validate_playing_retro_audio_app_export.py`.
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
- Risks or unresolved questions
- Follow-up tickets for out-of-scope work
