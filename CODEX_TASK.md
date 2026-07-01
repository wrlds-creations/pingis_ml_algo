# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0106-bounce-audio-test-model-switcher`

## Branch

`codex/t0057-fable-auto-improvement-loop`

## Status

`Complete`

## Goal

Let `Bounce audio test` switch between the current guarded T0103 model and the newly exported T0104E diagnostic candidate, so Love can compare both directly on the Motorola without changing production/default audio behavior.

## Dependencies

- T0103 is already exported and installed behind `Bounce audio test` only.
- T0104E was evaluated offline as a near-miss candidate, not production-ready.
- Love explicitly asked to include T0104E as a switchable test model despite the earlier no-promotion decision.
- Raw/generated `data/` remains ignored and must not be committed.

## Allowed Areas

- `CODEX_TASK.md`
- `PROJECT_CONTEXT.md`
- `DECISIONS.md`
- `REPO_CURRENT_STATE.md`
- `ITERATION_LOG.md`
- `apps/collector/src/BounceAudioTestScreen.tsx`
- `apps/collector/src/bounceAudioTestEngine.ts`
- `apps/collector/src/models/fable_extra_trees_candidate_t0104e.json`
- `skills/pingis-audio-classification/scripts/noise_robust/evaluate_t0104e_live_positive_candidate_loop.py`
- ignored local evaluation artifacts under `data/audio/`
- validation/status commands

## Do Not Touch

- Do not merge to `main`.
- Do not push.
- Do not delete local or device data.
- Do not revert tracked or user changes.
- Do not replace or promote production Fable/studs/camera behavior.
- Do not change `audio_model.json`, `audio_contact_model.json`, `fable_audio_model.json`, `bounce_side_model.json`, or native peak-gate defaults.
- Do not move raw/generated data into git.

## Requirements

- Export T0104E as a diagnostic app JSON artifact with metadata that makes clear it is test-only.
- Keep T0103 as the default `Bounce audio test` model.
- Add a visible model selector to `Bounce audio test` with T0103 and T0104E options.
- Freeze the selected model and typed threshold/noise-veto config when `START` is pressed.
- Save selected model metadata/config in `bounce_audio_test_debug` JSON.
- Keep existing T0103 typed threshold/noise-veto behavior intact.

## Non-Goals

- No production promotion.
- No APK release build unless explicitly needed.
- No camera/racket-side changes.
- No new training data pull or labeling.

## Acceptance Criteria

- TypeScript validation passes for the Collector app.
- The T0104E export command runs and creates the app model JSON.
- Root validation passes.
- `Bounce audio test` can choose either T0103 or T0104E before starting a test.

## Completion Notes

- Extended `evaluate_t0104e_live_positive_candidate_loop.py` with a guarded `--export-app-model` path.
- Exported `apps/collector/src/models/fable_extra_trees_candidate_t0104e.json`.
- Added model switching to `Bounce audio test`:
  - default remains `T0103`;
  - second option is `T0104E`;
  - switching resets threshold/noise-veto fields to the selected model defaults, while typed values still freeze at `START`;
  - saved debug JSON includes selected model id/title/metadata and active config.
- Kept production Fable, studs, camera, native peak-gate defaults, and T0103 default behavior unchanged.
- Installed/launched the debug app on Motorola `ZY22KSPF5W`.

## Validation

- `python -m py_compile skills\pingis-audio-classification\scripts\noise_robust\evaluate_t0104e_live_positive_candidate_loop.py`
- `python skills\pingis-audio-classification\scripts\noise_robust\evaluate_t0104e_live_positive_candidate_loop.py --export-app-model`
  - exported T0104E app model;
  - Python/app JSON parity max diff: `5.55e-16`.
- `cd apps/collector && npx tsc --noEmit`
- `npm run validate`
- `git diff --check`
  - passed with existing Windows LF-to-CRLF warnings only.
- `.\install-android-dev.ps1`
  - installed/launched `com.collectorapp` on Motorola `ZY22KSPF5W`;
  - smoke: `pidof com.collectorapp` returned `21612`;
  - package `lastUpdateTime=2026-07-01 18:39:11`.
