# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0103A-bounce-audio-boundary-test-tags`

## Branch

`codex/t0057-fable-auto-improvement-loop`

## Status

`Completed`

## Goal

Add and clarify the missing T0102/T0103 boundary scenario tags so Love can label fresh validation runs with the same buckets used for the boundary training/evaluation loop.

## Dependencies

- T0103 exported and installed the guarded `Bounce audio test` candidate.
- The installed screen has generic scenarios but is missing at least `Far/soft racket bounce + background`, which Love needs for validation.
- Raw/generated data under `data/` remains ignored and must not be committed.

## Allowed Areas

- `CODEX_TASK.md`
- `REPO_CURRENT_STATE.md`
- `PROJECT_CONTEXT.md` if confirmed project facts change
- `DECISIONS.md` if a meaningful decision changes
- `ITERATION_LOG.md`
- `skills/pingis-audio-classification/scripts/noise_robust/`
- ignored local review/ingest/evaluation/training artifacts under `data/audio/`
- validation/status commands
- guarded app scenario metadata under `apps/collector/`, limited to boundary label text in `Bounce audio test` and `Fable data recorder`

## Do Not Touch

- Do not merge to `main`.
- Do not push.
- Do not delete local or device data.
- Do not revert tracked or user changes.
- Do not replace current production Fable behavior unless a later ticket explicitly promotes the candidate.
- Do not move raw/generated data into git.
- Do not change the T0103 model or counting thresholds.
- Do not touch camera, Roboflow, cloud/API credentials, backend resources, or AWS resources.

## Requirements

- Add boundary positive tags to `Bounce audio test`: `Far/soft racket bounce + background` and `High racket bounce + background`.
- Add boundary negative tags for cleaner false-count validation: background-only, talking/counting+background, racket-handling+background, catch/after-sound, and ball-like impact near phone.
- Keep the legacy internal ID `soft_high_racket_bounce_background` compatible with existing saved files, but show it to users as `High racket bounce + background`.
- Keep existing tags and T0103 runtime behavior unchanged.
- Validate TypeScript and reinstall the debug app.

## Non-Goals

- No merge to `main`.
- No push.
- No raw/generated data commit.
- No production model replacement.
- No camera/racket-side work.
- No cloud/API/AWS work.
- No app install if offline validation is still clearly weak.

## Acceptance Criteria

- The `Bounce audio test` popup offers the missing boundary tags.
- Existing scenario tags remain available.
- T0103 model JSON, threshold, dedupe, classifier, and production app flows are unchanged.
- Source-of-truth docs/logs note this small test-UI metadata fix.

## Completion Notes

Added the missing boundary scenario metadata to the separate `Bounce audio test` save popup:

- `Far/soft racket bounce + background`
- `High racket bounce + background` using the existing internal ID `soft_high_racket_bounce_background`
- `Background sound only, no bounce`
- `Talking/counting + background, no bounce`
- `Racket handling + background, no bounce`
- `Catch/after-sound, no racket`
- `Ball-like impact near phone, no racket`

This only changes the tags available when saving `Bounce audio test` / `Fable data recorder` recordings. It does not change the T0103 candidate model, decision threshold, dedupe, decision delay, production Fable flow, studs/camera flows, or raw/generated data.

## Validation

- `cd apps/collector && npx tsc --noEmit` passed.
- `npm run validate` passed.
- `git diff --check` passed with only LF/CRLF warnings before docs completion.
- `.\install-android-dev.ps1` was run again after the wording clarification, and install smoke passed on Motorola `ZY22KSPF5W`: `com.collectorapp` is running, and package `lastUpdateTime=2026-07-01 15:28:50`.
