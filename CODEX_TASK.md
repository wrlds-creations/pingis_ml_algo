# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0107-bounce-audio-test-rms-fable-baseline`

## Branch

`codex/t0057-fable-auto-improvement-loop`

## Status

`Complete`

## Goal

Add the original RMS/native gate + Fable counter path as a third selectable baseline inside `Bounce audio test`, so Love can compare T0103, T0104E, and the older Fable behavior in the same UI.

## Dependencies

- T0103 is already the default guarded model behind `Bounce audio test` only.
- T0104E is already bundled as a diagnostic switch-only candidate.
- `Fable-algoritm` already has the older RMS/native gate + Fable counter flow.
- Love explicitly asked to compare the original RMS + Fable model in the same `Bounce audio test` UI.
- Raw/generated `data/` remains ignored and must not be committed.

## Allowed Areas

- `CODEX_TASK.md`
- `PROJECT_CONTEXT.md`
- `DECISIONS.md`
- `REPO_CURRENT_STATE.md`
- `ITERATION_LOG.md`
- `apps/collector/src/BounceAudioTestScreen.tsx`
- `apps/collector/src/bounceAudioTestEngine.ts`
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

- Keep T0103 as the default `Bounce audio test` model.
- Add a visible third selector option for the original RMS + Fable baseline.
- For RMS + Fable, use the original native gate settings from `Fable-algoritm`: bandpass gate, retrigger `120 ms`, abs min RMS `0.0015`, and the existing `FableCounter` logic.
- The typed `p threshold` and `noise veto` controls do not need to affect RMS + Fable; the UI should make that clear.
- Save the selected baseline/runtime metadata in `bounce_audio_test_debug` JSON.
- Keep T0103/T0104E typed threshold/noise-veto behavior intact.

## Non-Goals

- No production promotion.
- No APK release build unless explicitly needed; debug install is acceptable if validation passes and a phone is connected.
- No camera/racket-side changes.
- No new training data pull or labeling.

## Acceptance Criteria

- TypeScript validation passes for the Collector app.
- Root validation passes.
- `Bounce audio test` can choose T0103, T0104E, or RMS + Fable before starting a test.
- T0103 and T0104E still use peak gate + ExtraTrees with typed config.
- RMS + Fable uses the original RMS/native gate and Fable counter, and saved debug JSON identifies it clearly.

## Completion Notes

- Added `RMS+Fable` as a third `Bounce audio test` selector option.
- Reused the existing Fable runtime pieces for the baseline:
  - native RMS/bandpass gate setup;
  - retrigger `120 ms`;
  - abs min RMS `0.0015`;
  - existing `FableCounter` confidence/window logic with the same live overrides as `Fable-algoritm`.
- Kept T0103 as the default selector choice.
- Kept T0103/T0104E peak-gate + ExtraTrees behavior and typed threshold/noise-veto controls intact.
- Made the `p threshold` and `noise veto` fields disabled/ignored for RMS+Fable and explained that in the UI.
- Saved selected runtime mode plus either peak-gate config or RMS+Fable gate config in `bounce_audio_test_debug` JSON.
- Installed/launched the debug app on Motorola `ZY22KSPF5W`.

## Validation

- `cd apps/collector && npx tsc --noEmit`
- `npm run validate`
- `git diff --check`
  - passed with existing Windows LF-to-CRLF warnings only.
- `.\install-android-dev.ps1`
  - installed/launched `com.collectorapp` on Motorola `ZY22KSPF5W`;
  - smoke: `pidof com.collectorapp` returned `25499`;
  - package `lastUpdateTime=2026-07-01 19:32:24`.
