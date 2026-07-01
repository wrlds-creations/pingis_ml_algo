# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0110-t0104e-loud-music-negative-feedback`

## Branch

`codex/t0057-fable-auto-improvement-loop`

## Status

`Complete`

## Goal

Record Love's fresh guarded phone-test feedback that T0104E with `p=0.25` and noise veto `0.98` false-counts loud background music without bounce, while quieter background volume behaves well.

## Dependencies

- T0109 evaluated T0104E `p=0.25`, noise veto `0.98` as a guarded phone-test setting.
- Love tested a negative loud-music case on the Motorola app and reported false counts only when music was loud.
- Raw/generated `data/` remains ignored and must not be committed.

## Allowed Areas

- `CODEX_TASK.md`
- `PROJECT_CONTEXT.md`
- `DECISIONS.md`
- `REPO_CURRENT_STATE.md`
- `ITERATION_LOG.md`
- validation/status commands

## Do Not Touch

- Do not merge to `main`.
- Do not push.
- Do not delete local or device data.
- Do not revert tracked or user changes.
- Do not replace or promote production Fable/studs/camera behavior.
- Do not change `audio_model.json`, `audio_contact_model.json`, `fable_audio_model.json`, `bounce_side_model.json`, T0103/T0104E JSON, native peak-gate defaults, or app runtime code.
- Do not move raw/generated data into git.

## Requirements

- Record the live feedback clearly in source-of-truth docs.
- Keep T0104E `p=0.25`, noise veto `0.98` as diagnostic only.
- Call out loud background music as a confirmed hard negative to collect/label or gate against.

## Non-Goals

- No app code change.
- No model export/retrain.
- No APK install.
- No camera/racket-side changes.
- No new data pull or labeling.

## Acceptance Criteria

- Root validation passes.
- `git diff --check` passes.
- Device feedback is recorded in `PROJECT_CONTEXT.md`, `DECISIONS.md`, `REPO_CURRENT_STATE.md`, and `ITERATION_LOG.md`.
- Final answer explains what the feedback means and what to do next.

## Completion Notes

- Recorded Love's live feedback: T0104E `p=0.25`, noise veto `0.98` false-counts loud background music without bounces, but is OK when the volume is reduced.
- Updated the recommendation to collect/save loud-music-only negatives and treat loud music as outside the current reliable operating envelope.
- No code, model, APK, raw data, merge, or push changed.

## Validation

- `npm run validate`
- `git diff --check`
  - passed with existing Windows LF-to-CRLF warnings only.
