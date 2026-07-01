# Worktree Cleanup Plan

Date: `2026-07-01`
Branch: `codex/t0057-fable-auto-improvement-loop`
Ticket: `T0105C-local-cleanup-commits`

## Current Status

This started as an inventory-only cleanup plan. Love later explicitly asked Codex to tidy, stage, and commit locally.

T0105C has now created local commits on `codex/t0057-fable-auto-improvement-loop`:

- `16f1e01 chore(audio): add fable reliability tooling`
- `0fbee48 feat(collector): add bounce audio diagnostics`
- `61d532e feat(collector): add live racket color tracker`
- a final docs/source-of-truth cleanup commit

This is still not a merge-to-main decision.

## Corrected Main-Branch Policy

Do not merge this branch to `main` before the current audio model/runtime works on device.

Cleanup can continue locally on this feature branch. A local cleanup commit is not a merge-to-main decision.

## Dirty Tree Outcome

- The source files from the dirty tree were grouped into local commits.
- Raw/generated files under `data/` remain ignored and were not committed.
- The large app-side candidate artifact `apps/collector/src/models/fable_extra_trees_candidate_t0075.json` is committed only as part of the guarded `Bounce audio test` diagnostic path, not as a production promotion.
- Any useful raw WAV/JSON/review data still needs a separate data handoff outside git.

## Cleanup Buckets

### 1. Source-Of-Truth Docs

Files:

- `CODEX_TASK.md`
- `DECISIONS.md`
- `ITERATION_LOG.md`
- `PROJECT_CONTEXT.md`
- `REPO_CURRENT_STATE.md`
- `WORKTREE_CLEANUP_PLAN.md`

What this is:

- Project memory from the long audio iteration.
- The active cleanup ticket.
- Decision and iteration context for T0044-T0102D.

Recommended local action:

- Keep this as a docs bucket, but review carefully before committing because `CODEX_TASK.md` is an active-ticket file and may keep changing.
- If committing locally, commit docs separately from app/runtime changes.
- Do not merge docs to `main` as part of this mixed branch until the model/runtime validation story is ready.

Validation before commit:

- `npm run validate`
- `git diff --check`

### 2. Fable Data Recorder And Boundary Collection UI

Files:

- `apps/collector/src/FableTrainingRecorderScreen.tsx`
- `apps/collector/App.tsx`
- `apps/collector/src/SetupScreen.tsx`

Related source-of-truth context:

- T0064 added `Fable data recorder`.
- T0102 added boundary categories and metadata for new hard-negative collection.

What this is:

- A user-facing app data entry that records WAV plus metadata.
- Save/discard flow with scenario, expected count, unclear flag, and boundary metadata.

Recommended local action:

- Treat this as a likely keeper because it supports the next required data collection.
- Keep it separate from `Bounce audio test` and camera tracker work if possible.
- Do not merge it to `main` yet unless we deliberately decide the recorder should ship independently before the detector is validated.

Validation before commit:

- `cd apps/collector && npx tsc --noEmit`
- `npm run validate`
- App smoke only if preparing for install.

### 3. Bounce Audio Test Entry And T0075 Candidate Runtime

Files:

- `apps/collector/src/BounceAudioTestScreen.tsx`
- `apps/collector/src/bounceAudioTestEngine.ts`
- `apps/collector/src/models/fable_extra_trees_candidate_t0075.json`
- `apps/collector/src/NativeAudioStream.ts`
- `apps/collector/src/rfRuntime.ts`
- `apps/collector/android/app/src/main/java/com/collectorapp/AudioStreamModule.kt`
- shared route wiring in `apps/collector/App.tsx`
- shared setup entry wiring in `apps/collector/src/SetupScreen.tsx`

What this is:

- Separate `Bounce audio test` entry.
- Peak candidate gate in native audio streaming.
- T0075 ExtraTrees classifier runtime using raw probabilities.
- Tunable threshold and Fable-noise veto inputs.
- Save/discard debug JSON/WAV behavior.

Recommended local action:

- Keep only if the team wants the diagnostic app entry in source history.
- Keep separate from the data recorder because this is not production behavior and depends on the large candidate JSON.
- Before any local commit of the model JSON, confirm that a 2 MB app artifact is acceptable in the repo. It is source-controlled app behavior, unlike ignored raw data, but it is still bulky.
- Do not merge to `main` before device validation.

Validation before commit:

- `cd apps/collector && npx tsc --noEmit`
- `cd apps/collector/android && .\gradlew.bat :app:compileDebugKotlin`
- `npm run validate`
- Optional device smoke only when testing this screen again.

### 4. Studs FH/BH LIVE Racket Color Tracker

Files:

- `apps/collector/android/app/src/main/java/com/collectorapp/BounceSideLiveModule.kt`
- `apps/collector/src/BounceSideLiveScreen.tsx`
- `apps/collector/src/NativeBounceSideLive.ts`

What this is:

- Native continuous racket red/black color and shape tracker.
- `onBounceSideRacketTrack` event and `getRacketTrack(targetTimeMs)` API.
- Green live overlay box and confidence-gated side resolution in `Studs FH/BH LIVE`.

Recommended local action:

- Keep as prototype source only if we still want the no-training live tracker available.
- Keep separate from audio work. It has different risk, validation, and product meaning.
- Do not mix this with audio-model commits.
- Do not merge to `main` before separate red/black/racket-absent validation.

Validation before commit:

- `cd apps/collector && npx tsc --noEmit`
- `cd apps/collector/android && .\gradlew.bat :app:compileDebugKotlin`
- Manual device check with red/black racket only if promoting.

### 5. Fable-Algroritm Continuous Debug Additions

Files:

- `apps/collector/src/FableLiveScreen.tsx`
- portions of `apps/collector/src/NativeAudioStream.ts`
- portions of `apps/collector/android/app/src/main/java/com/collectorapp/AudioStreamModule.kt`

What this is:

- Continuous debug WAV support and richer native debug payloads for the old `Fable-algoritm` screen.
- Some of this overlaps technically with the `Bounce audio test` native audio work.

Recommended local action:

- Split only if practical. `AudioStreamModule.kt` currently contains both debug-WAV and peak-gate changes, so it may need a manual review before staging by bucket.
- If splitting is too risky, keep this local until the full audio runtime path is validated.

Validation before commit:

- `cd apps/collector && npx tsc --noEmit`
- `cd apps/collector/android && .\gradlew.bat :app:compileDebugKotlin`

### 6. Local Audio Evaluation And Review Scripts

Files:

- `skills/pingis-audio-classification/scripts/noise_robust/evaluate_fable_audio_reliability_t0044.py`
- `skills/pingis-audio-classification/scripts/noise_robust/evaluate_t0049_speech_veto_candidate.py`
- `skills/pingis-audio-classification/scripts/noise_robust/evaluate_t0050_fable_targeted_round.py`
- `skills/pingis-audio-classification/scripts/noise_robust/evaluate_t0052_continuous_fable_round.py`
- `skills/pingis-audio-classification/scripts/noise_robust/evaluate_t0056_fable_candidate_retrain_replay.py`
- `skills/pingis-audio-classification/scripts/noise_robust/evaluate_t0063_t0060_current_only_replay.py`
- `skills/pingis-audio-classification/scripts/noise_robust/evaluate_t0067_peak_gate_replay.py`
- `skills/pingis-audio-classification/scripts/noise_robust/evaluate_t0068_rms_vs_peak_gate_audit.py`
- `skills/pingis-audio-classification/scripts/noise_robust/evaluate_t0069_peak_fable_hybrid_replay.py`
- `skills/pingis-audio-classification/scripts/noise_robust/evaluate_t0070_peak_candidate_classifier_veto.py`
- `skills/pingis-audio-classification/scripts/noise_robust/evaluate_t0072_round_a_reviewed_classifier_replay.py`
- `skills/pingis-audio-classification/scripts/noise_robust/evaluate_t0074_fable_app_style_safety_gate.py`
- `skills/pingis-audio-classification/scripts/noise_robust/evaluate_t0079_live_failure_candidate.py`
- `skills/pingis-audio-classification/scripts/noise_robust/export_t0075_fable_extra_trees_app_parity.py`
- `skills/pingis-audio-classification/scripts/noise_robust/ingest_t0055_fable_review_labels.py`
- `skills/pingis-audio-classification/scripts/noise_robust/ingest_t0063_t0060_heldout_labels.py`
- `skills/pingis-audio-classification/scripts/noise_robust/ingest_t0071_round_a_review_labels.py`
- `skills/pingis-audio-classification/scripts/noise_robust/prepare_t0071_round_a_scenario_labels.py`
- `skills/pingis-audio-classification/scripts/noise_robust/prepare_t0073_fable_bad_case_pack.py`
- `skills/pingis-audio-classification/scripts/noise_robust/prepare_t0102_boundary_recorder_pack.py`
- `skills/pingis-audio-classification/scripts/noise_robust/run_t0057_fable_auto_improvement_loop.py`
- `skills/pingis-audio-classification/scripts/noise_robust/serve_t0053_trigger_review_ui.py`

What this is:

- The local audit, labeling UI, replay, ingest, export-parity, and boundary-pull toolchain built during the audio reliability work.
- These are source scripts, not raw data.

Recommended local action:

- Keep the scripts that are reusable for current and next work:
  - `serve_t0053_trigger_review_ui.py`
  - `prepare_t0102_boundary_recorder_pack.py`
  - the T0071/T0073/T0074/T0075/T0079 replay/export scripts that explain current decisions.
- Consider whether earlier one-off scripts T0044-T0056 should be committed or archived locally. They document history, but some may be less reusable.
- If committing locally, commit script tooling separately from app code.
- Do not treat script commits as permission to merge this branch to `main`.

Validation before commit:

- `python -m py_compile` for all staged Python scripts.
- Run at least one smoke command for the currently important script, likely `prepare_t0102_boundary_recorder_pack.py`.
- `npm run validate`

### 7. Ignored Raw And Generated Data

Files:

- Anything under `data/` is ignored by `.gitignore`.
- Examples include pulled phone WAV/JSON files, review UI outputs, model evaluation CSV/JSON/MD, generated review pages, snippets, and local model artifacts.

What this is:

- Evidence and training material, often large and potentially private.
- Necessary for reproducing exact ML work, but not suitable for ordinary git commits.

Recommended action:

- Do not commit these files into the normal repo.
- If the team needs to share them, create a separate data handoff:
  - a manifest listing exact folders/files, counts, labels, and purpose;
  - a zip/archive or cloud storage upload outside git;
  - a short README explaining which ticket produced the data and whether it is trainable, holdout, diagnostic, or generated.
- Keep source scripts in git so another person can regenerate outputs when they receive the data package.

## Actual Local Commit Sequence

Love explicitly approved local commits. This is not a `main` merge sequence.

1. `16f1e01 chore(audio): add fable reliability tooling`
   - Python review, ingest, replay, audit, and export-parity scripts.
2. `0fbee48 feat(collector): add bounce audio diagnostics`
   - `Fable data recorder`, `Fable-algoritm` continuous debug support, `Bounce audio test`, native peak gate/debug, runtime helpers, and T0075 candidate JSON.
3. `61d532e feat(collector): add live racket color tracker`
   - `Studs FH/BH LIVE` native/API/screen tracker prototype.
4. `docs: record cleanup handoff state`
   - Source-of-truth docs plus cleanup plans.

The sequence can be adjusted, but app route wiring in `App.tsx` and `SetupScreen.tsx` is shared by buckets 2 and 3, so staging those files may require manual hunks.

## Main Merge Gate

Do not merge this branch to `main` until:

- the selected audio runtime has passed fresh Motorola validation;
- debug JSON/WAV for failures has been pulled and reviewed;
- counts are reported by scenario, including positives and hard negatives;
- app entries are classified as product, diagnostic, or hidden;
- TypeScript, Android, root validation, and diff check pass for the final staged source;
- Love explicitly approves the merge/PR direction.

## Remaining Decisions Before Any Main Merge

- Where should raw/generated data be handed off outside git so other teammates can reproduce model work?
- Should the diagnostic `Bounce audio test` screen remain visible, hidden, or removed before any product-facing merge?
- Should the `Fable data recorder` be split onto a clean branch if we want to ship collection support before the detector is validated?
- Should the no-training `Studs FH/BH LIVE` color tracker remain as a prototype or be replaced by a trained detector later?

## Immediate Next Step

Keep working locally on validation and data. Prioritize T0102 boundary data and the next measured candidate loop; do not merge this branch to `main` yet.
