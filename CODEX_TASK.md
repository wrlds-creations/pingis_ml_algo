# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0041-live-bounce-side-backhand-bias`

## Branch

`claude/audio-noise-robust-racket-bounce`

## Status

`Completed`

## Goal

Diagnose and fix the current `Studs FH/BH LIVE` / `Video studs FH/BH` behavior where red-forehand racket bounces are being suggested almost entirely as backhand. Keep the fix limited to runtime side mapping/guarding and debug visibility; do not retrain or replace model artifacts in this ticket.

## Dependencies

- Current installed Motorola APK from 2026-06-13 contains Fable v5 and `bounce_side_v2_2026_06_11_underangle`.
- Latest device debug dumps from 2026-06-17 show `bounce_side_v2_2026_06_11_underangle` predicting almost all counted events as `backhand`, even when crops visibly include strong red racket evidence.
- `node skills/pingis-stroke-detection/scripts/check_bounce_side_ts_parity.js` passes, so the TypeScript runtime matches the exported model on its fixture.

## Allowed Areas

- `apps/collector/src/bounceSideInference.ts`
- `apps/collector/src/BounceSideLiveScreen.tsx`
- `apps/collector/src/VideoOnlyStrokeCollectionScreen.tsx`
- `CODEX_TASK.md`
- `ITERATION_LOG.md`
- `REPO_CURRENT_STATE.md`
- Local ignored debug artifacts under `data/video/raw/live_sidedebug/`

## Do Not Touch

- `apps/collector/src/models/bounce_side_model.json`
- `apps/collector/src/models/fable_audio_model.json`
- `apps/collector/src/models/video_stroke_model.json`
- `audio_model.json`, `audio_contact_model.json`, or `playing_retro_audio_model.json`
- Raw reviewed labels
- Training scripts or model retraining
- AWS or backend resources

## Requirements

- Keep Fable audio detection unchanged.
- Preserve the existing model and parity fixture.
- Add a runtime color-evidence guard using the existing side features so visible red/black racket evidence can override an obviously wrong side suggestion.
- Respect the user-selected forehand color: red forehand maps visible red to forehand; black forehand maps visible red to backhand.
- Avoid confident wrong auto-suggestions when color evidence is ambiguous.
- Include debug fields that make future live dumps explainable: raw model label/probability, visible color decision, red/dark evidence, and decision source.

## Non-Goals

- No model retraining.
- No app model JSON export.
- No changes to Fable audio model behavior.
- No changes to video stroke FH/BH motion model.
- No broad UI redesign.

## Acceptance Criteria

- Today's 2026-06-17 live debug crops no longer collapse to all backhand under the runtime side resolver.
- TypeScript validation passes.
- Existing bounce-side TypeScript parity check still passes.
- If APK build/install is run, record build/install result in the handoff.

## Completion Notes

- Pulled and inspected the latest 2026-06-17 live side debug dumps from Motorola.
- Confirmed Fable audio was producing racket-bounce events; the failure was side resolution after the audio anchor.
- Confirmed Collector TypeScript parity against the exported bounce-side model still passes, so the runtime model implementation is not the source of the backhand collapse.
- Added a visible-color resolver that lets clear red/black racket evidence override the model side when it contradicts the selected forehand color.
- Added explainability fields to live debug rows: raw side/confidence, visible color/confidence, red/dark totals, and decision source.
- Updated `Video studs FH/BH` auto-suggestions so ambiguous color evidence becomes `unknown` instead of a confident wrong side.

## Validation

- `cd apps/collector && npx tsc --noEmit`
- `node skills/pingis-stroke-detection/scripts/check_bounce_side_ts_parity.js`
- Root `npm run validate`
- `git diff --check` passed with existing Windows line-ending warnings only.
- Forced release bundle generation and `.\gradlew.bat assembleRelease` passed.
- APK installed on Motorola `ZY22L6NDHV` at `2026-06-17 16:29:12`; SHA256 `BE0C3F21473A44AF99979A38DF55B19FCBB50C5A755EBB502A3D6D3CBA5A30D2`.
