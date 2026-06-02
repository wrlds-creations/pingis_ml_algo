# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0008`

## Branch

`codex/t0008-playing-retro-audio-cross-session-validation`

## Goal

Validate the T0007 `spel_retro_audio` multi-window/context candidate across dense playing sessions before any Collector app integration, so we know whether the large `audio_session_2026-05-29_002` gain generalizes or only fits one Tomas/Stiga holdout.

## Dependencies

- T0001 ticket workflow adoption is complete.
- T0002 documentation refresh removed IMU/AirHive from active project scope.
- T0003 cleanup removed retired IMU/AirHive workflow docs and skill scripts from the active repo workflow.
- T0004 generated candidate-centered playing-retro diagnostics locally from saved app candidates plus replay peaks.
- T0005 trained local candidate `playing_retro_audio_rf_v2026_06_02_app_candidates_100_200` from 4,028 candidate-centered rows across 16 reviewed playing sessions.
- T0006 selected local one-window candidate `playing_retro_audio_rf_v2026_06_02_safe_racket_weighted`, but the safe gain was too small for app integration.
- T0007 selected local multi-window/context candidate `playing_retro_audio_rf_v2026_06_02_multi_window_context` / `multi_window_context_racket_weighted`.
- T0007 holdout on `audio_session_2026-05-29_002` reached `0.908` accuracy, `0.896` racket recall, `0.933` table recall, and `0.833` non-target recall, versus T0006 `0.771`, `0.623`, `0.933`, and `0.625`.
- Current audio source-of-truth lives in `PROJECT_CONTEXT.md`, `DECISIONS.md`, and `ITERATION_LOG.md`.
- Existing audio scripts and replay behavior live under `skills/pingis-audio-classification/scripts/`.

## Allowed areas

- `skills/pingis-audio-classification/scripts/`
- `skills/pingis-audio-classification/SKILL.md`
- `data/audio/processed/`
- `data/audio/models/evaluations/`
- `data/audio/models/playing_retro_candidates/`
- `PROJECT_CONTEXT.md`
- `DECISIONS.md`
- `ITERATION_LOG.md`
- `REPO_CURRENT_STATE.md`
- `FOLLOWUPS.md`
- `CODEX_TASK.md`

## Do not touch

- `apps/collector/src/models/audio_model.json`
- `apps/collector/src/models/audio_contact_model.json`
- APK/build artifacts
- `studs_live` app behavior or live detector thresholds
- Video-stroke model files
- Raw reviewed session JSON labels, except for metadata fixes explicitly approved by Love

## Requirements

- Start from the T0007 script, report, holdout prediction CSV, and selected local candidate.
- Add deterministic cross-session validation for the same multi-window/context feature family.
- At minimum evaluate alternate dense-playing holdouts for `audio_session_2026-05-28_002`, `audio_session_2026-05-29_001`, and `audio_session_2026-05-29_002`.
- Compare T0007 selected behavior against T0005 and T0006 references for each holdout where reference metrics exist or can be recomputed fairly.
- Report racket recall, table recall, non-target recall, wrong-class racket/table errors, and close-event buckets per holdout session.
- Keep ordinary up/down bounce regression separate and clearly mark any fallback metric as advisory if exact raw timestamps are unavailable.
- Do not export the candidate into Collector app model JSON in this ticket.
- Do not build or install an APK in this ticket.

## Non-Goals

- No Collector UI integration yet.
- No video/FH-BH work yet.
- No changes to ordinary `studs_live` promotion rules.
- No app model export or APK build.

## Acceptance criteria

- A deterministic command or report validates T0007-style multi-window/context features across multiple dense-playing holdout sessions.
- The report makes clear whether `playing_retro_audio_rf_v2026_06_02_multi_window_context` is a good general candidate, a Tomas-backhand-specific candidate, or needs another feature/training change.
- Ordinary bounce regression remains separate from dense playing metrics.
- Candidate remains local unless a later ticket explicitly approves app export.

## Manual verification

- Inspect per-session T0008 errors for the Tomas/Stiga sessions, especially close table-to-racket and racket-to-table gaps under 120 ms.
- Confirm T0008 does not treat ordinary up/down bounce as the same promotion bucket as Tomas/Stiga dense play.

## Automated validation

- Run the new or updated cross-session evaluation command.
- Run `npm run validate` if source-of-truth workflow files changed.
- Run a targeted Python syntax check for any changed Python script.

## Completion Report Expected

Codex should report:

- Summary of changes
- Files changed
- Commands run
- Model/report outputs generated
- Validation results
- Manual verification performed or still needed
- Docs updated or still needed
- Risks or unresolved questions
- Follow-up tickets for out-of-scope work
