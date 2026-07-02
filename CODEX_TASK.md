# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0127-bounce-audio-soft-peak-test-option`

## Branch

`codex/t0057-fable-auto-improvement-loop`

## Status

`Completed`

## Goal

Add a guarded `Bounce audio test` runtime option that keeps the current T0104E classifier stack but lowers only the peak-candidate absolute floor from `0.08` to `0.03`, so Love can test whether the measured `soft_peak_abs003` recovery helps on real phones.

## Dependencies

- T0121 showed the failed calibrated transient phone run is partly candidate-gate limited and partly second-layer limited.
- T0122 restored failed adaptive/transient/calibrated app experiments out of the worktree.
- T0123 showed `soft_peak_abs003` can recover most missed true bounces but should not be counted directly.
- T0124/T0125 showed soft peak plus existing T0104E improves full-flow recall compared with current peak while still leaving second-layer misses.
- T0126 showed removing T0104E and using Fable probability alone is not the best current architecture.

## Allowed Areas

- `CODEX_TASK.md`
- `PROJECT_CONTEXT.md`
- `REPO_CURRENT_STATE.md`
- `ITERATION_LOG.md`
- `DECISIONS.md`
- `apps/collector/src/bounceAudioTestEngine.ts`
- `apps/collector/src/BounceAudioTestScreen.tsx`
- `apps/collector/src/NativeAudioStream.ts`
- `apps/collector/android/app/src/main/java/com/collectorapp/AudioStreamModule.kt`

## Do Not Touch

- Do not delete raw/generated `data/`.
- Do not merge to `main`.
- Do not push unless explicitly requested.
- Do not delete local or device data.
- Do not replace or promote production Fable/studs/camera behavior.
- Do not move raw/generated data into git.
- Do not train or export a new model.
- Do not change model JSON files.
- Do not change the production/default `Fable-algoritm`, `Studsdetektor`, `Studs FH/BH LIVE`, or camera behavior.

## Requirements

- Add a new selector option in `Bounce audio test`, tentatively `T0104E Soft`.
- Keep the existing fixed `T0104E` option available for A/B comparison.
- Use the same T0104E JSON, Fable feature extraction, threshold/noise-veto, decision delay, and smart dedupe.
- For the soft option, use native peak gate settings matching the measured soft peak row:
  - raw absolute envelope;
  - smoothing `3 ms`;
  - min gap `220 ms`;
  - background window `500 ms`;
  - background exclusion `60 ms`;
  - absolute minimum `0.03`;
  - ratio minimum `2.0`;
  - z minimum `0.0`.
- Make `T0104E Soft` the default diagnostic option for the next phone test.
- Show the active gate/floor in the UI and save it in `bounce_audio_test_debug` JSON.
- Keep `RMS+Fable` comparison behavior unchanged.

## Non-Goals

- No model export/retrain.
- No production/default Fable, studs, or camera behavior change.
- No cloud/API/AWS changes.
- No deletion of local analysis/audio files.
- No claim that this is production-ready before phone validation.

## Acceptance Criteria

- `Bounce audio test` shows a selectable `T0104E Soft` option.
- Starting that option passes the soft peak gate parameters to native audio streaming.
- Debug JSON records the selected model, runtime config, and active soft peak gate config.
- TypeScript validation, Android/Kotlin validation, root validation, and `git diff --check` pass or blockers are documented.

## Completion Notes

- Added `T0104E Soft` as a separate default option in `Bounce audio test`.
- The soft option keeps the same T0104E model JSON, Fable-derived feature stack, typed defaults `p=0.25` and Fable-noise veto `0.98`, `500 ms` decision delay, and smart dedupe.
- The only algorithmic runtime difference is native peak gate absolute floor `0.03` instead of the fixed `0.08`, matching the offline `soft_peak_abs003` row.
- Fixed `T0104E`, old `T0103`, and `RMS+Fable` remain selectable for A/B comparison.
- The model selector now wraps into two rows so four options fit on phone screens.
- Saved `bounce_audio_test_debug` JSON records the active peak gate config, and native candidate debug rows report `gate_id=peak_fast_soft_abs003` when the soft floor is active.
- No model JSON, production Fable/studs/camera behavior, raw/generated data, merge, push, or production promotion changed.
- Quick debug/Metro install passed on connected Android `EHT0219B01004275`; package `lastUpdateTime=2026-07-03 00:17:00`, app PID `12134`.

## Validation

- `cd apps/collector && npx tsc --noEmit`
- `cd apps/collector/android && .\gradlew.bat :app:compileDebugKotlin --no-daemon --console plain`
- `npm run validate`
- `git diff --check` passed with existing LF/CRLF warnings only.
- `.\install-android-dev.ps1`
