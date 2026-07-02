# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0122-discard-calibrated-transient-app-experiment`

## Branch

`codex/t0057-fable-auto-improvement-loop`

## Status

`Completed`

## Goal

Clean up the dirty worktree after the T0115/T0117/T0118 adaptive/transient/calibrated `Bounce audio test` experiments failed the T0119/T0121 saved-label check. Keep the reusable review helper commit, discard the experimental runtime code, and record the next direction.

## Dependencies

- T0120 committed the reusable full-WAV review helper.
- T0121 showed the calibrated transient phone run counted only `18/30` true bounces at `140 ms`, with `10` candidate-gate misses and `2` unmatched counted candidates.
- Love agreed the unstaged adaptive/transient/calibrated app code should be treated as experimental and cleaned up rather than staged.

## Allowed Areas

- `CODEX_TASK.md`
- `PROJECT_CONTEXT.md`
- `REPO_CURRENT_STATE.md`
- `ITERATION_LOG.md`
- `DECISIONS.md`
- Restore only these experimental app/runtime files back to `HEAD`:
  - `apps/collector/android/app/src/main/java/com/collectorapp/AudioStreamModule.kt`
  - `apps/collector/src/BounceAudioTestScreen.tsx`
  - `apps/collector/src/NativeAudioStream.ts`
  - `apps/collector/src/bounceAudioTestEngine.ts`

## Do Not Touch

- Do not delete raw/generated `data/`.
- Do not merge to `main`.
- Do not push unless explicitly requested.
- Do not delete local or device data.
- Do not replace or promote production Fable/studs/camera behavior.
- Do not move raw/generated data into git.
- Do not train or export a model.
- Do not discard the committed T0120 review helper.
- Do not use broad destructive cleanup such as `git reset --hard`.

## Requirements

- Restore only the four experimental app/runtime files listed above.
- Verify the remaining dirty worktree is docs-only.
- Record that the adaptive/transient/calibrated gate app changes are abandoned as current code, but their evidence remains useful.
- Mark the recommended next direction as a new measured candidate-generation approach, likely buffered/sliding-window PCM or hybrid recovery, not another hardcoded peak floor tweak.

## Non-Goals

- No new app code.
- No model export/retrain.
- No production/default Fable, studs, or camera behavior change.
- No APK/reinstall.
- No cloud/API/AWS changes.
- No deletion of local analysis/audio files.

## Acceptance Criteria

- The four experimental app/runtime files are restored to `HEAD`.
- Docs clearly say T0115/T0117/T0118 are historical experiments that were reverted from the current worktree.
- Current repo state recommends the next audio direction without implying the calibrated transient mode is still active.
- Root validation and `git diff --check` pass or blockers are documented.

## Completion Notes

- Restored the four experimental app/runtime files to `HEAD`.
- The current working tree is docs-only after the restore.
- The current committed app code keeps `Bounce audio test` to the pre-experiment selector set: `T0103`, `T0104E`, and `RMS+Fable`; the adaptive/transient/calibrated native gate methods are no longer in the worktree.
- T0115/T0117/T0118 remain documented as historical diagnostics, but are not the current repo state.
- The T0121 evidence remains the basis for the cleanup: the miss is mostly candidate-generation loss, so the next direction should be a measured buffered/sliding-window PCM or hybrid recovery experiment.
- No APK, model export, production behavior, push, merge, raw-data deletion, or raw-data git state changed.

## Validation

- `npm run validate` passed.
- `git diff --check` passed with existing Windows LF-to-CRLF warnings only.
