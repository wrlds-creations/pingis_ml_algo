# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0108-bounce-audio-test-preserve-typed-config`

## Branch

`codex/t0057-fable-auto-improvement-loop`

## Status

`Complete`

## Goal

Keep typed `Bounce audio test` threshold values stable while switching models, so Love can compare T0103/T0104E/RMS+Fable without retyping `p` and noise-veto values each time.

## Dependencies

- T0103 is already the default guarded model behind `Bounce audio test` only.
- T0104E is already bundled as a diagnostic switch-only candidate.
- `RMS+Fable` is already bundled as a diagnostic baseline and ignores typed `p`/noise-veto values.
- Love reported that model switching currently resets `p` to `0.575`, which slows down same-setting comparisons.
- Raw/generated `data/` remains ignored and must not be committed.

## Allowed Areas

- `CODEX_TASK.md`
- `PROJECT_CONTEXT.md`
- `REPO_CURRENT_STATE.md`
- `ITERATION_LOG.md`
- `apps/collector/src/BounceAudioTestScreen.tsx`
- validation/status commands

## Do Not Touch

- Do not merge to `main`.
- Do not push.
- Do not delete local or device data.
- Do not revert tracked or user changes.
- Do not replace or promote production Fable/studs/camera behavior.
- Do not change `audio_model.json`, `audio_contact_model.json`, `fable_audio_model.json`, `bounce_side_model.json`, T0103/T0104E JSON, or native peak-gate defaults.
- Do not move raw/generated data into git.

## Requirements

- Preserve the current typed `p threshold` and `noise veto` values when switching between T0103 and T0104E.
- Preserve the typed values while temporarily switching to RMS+Fable, even though RMS+Fable ignores them.
- Do not reset threshold fields to per-model defaults on every selector tap.
- Keep selected-model freeze-on-`START` behavior unchanged.
- Keep saved debug JSON config behavior unchanged.

## Non-Goals

- No production promotion.
- No APK release build unless explicitly needed; debug install is acceptable if validation passes and a phone is connected.
- No camera/racket-side changes.
- No new training data pull or labeling.

## Acceptance Criteria

- TypeScript validation passes for the Collector app.
- Root validation passes.
- Switching model options no longer overwrites typed `p`/noise-veto values.
- T0103/T0104E still start with the currently typed config.
- RMS+Fable still ignores typed config but does not erase it.

## Completion Notes

- Changed `Bounce audio test` model switching so it preserves the current text in `p threshold` and `noise veto`.
- Switching T0103/T0104E now keeps the same typed values for the next `START`.
- Switching to RMS+Fable still ignores the typed values, but no longer erases them for when Love switches back.
- If the typed values are invalid, model switching preserves the text and tells Love to fix it before `START` instead of silently resetting to defaults.
- Installed/launched the debug app on Motorola `ZY22KSPF5W`.

## Validation

- `cd apps/collector && npx tsc --noEmit`
- `npm run validate`
- `git diff --check`
  - passed with existing Windows LF-to-CRLF warnings only.
- `.\install-android-dev.ps1`
  - installed/launched `com.collectorapp` on Motorola `ZY22KSPF5W`;
  - smoke: `pidof com.collectorapp` returned `26873`;
  - package `lastUpdateTime=2026-07-01 19:41:07`.
