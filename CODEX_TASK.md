# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0007`

## Branch

`codex/t0007-playing-retro-audio-multi-window-context`

## Goal

Build the next `spel_retro_audio` improvement step around real multi-window and candidate-context features, because T0006 only produced a small safe racket-recall gain and is not enough for app integration.

## Dependencies

- T0001 ticket workflow adoption is complete.
- T0002 documentation refresh removed IMU/AirHive from active project scope.
- T0003 cleanup removed retired IMU/AirHive workflow docs and skill scripts from the active repo workflow.
- T0004 generated `data/audio/processed/playing_retro_candidate_peak_rows.csv`, `playing_retro_candidate_peak_summary.csv`, and `playing_retro_candidate_peak_report.md` locally.
- T0005 trained local candidate `playing_retro_audio_rf_v2026_06_02_app_candidates_100_200` from 4,028 candidate-centered rows across 16 reviewed playing sessions.
- T0005 holdout on `audio_session_2026-05-29_002` reached `0.759` accuracy versus old app prediction `0.682`, but racket recall was only `0.604`, so it is not ready for app export.
- T0006 compared focused one-window variants and selected local candidate `playing_retro_audio_rf_v2026_06_02_safe_racket_weighted`.
- T0006 selected variant improved holdout racket recall from `0.604` to `0.623`, table recall from `0.924` to `0.933`, and kept non-target recall at `0.625`; this is useful but too small for app integration.
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

- Start from the T0006 variant report, holdout prediction CSV, and selected local candidate.
- Add or test multi-window feature extraction per candidate peak, at minimum tight `-60/+140 ms` plus normal `-100/+200 ms`; include a broader context window only if runtime stays practical.
- Add non-leaky candidate-context features only if they can be computed during retro inference, such as previous/next candidate gap and candidate-sequence position. Do not use truth-derived `close_event_bucket` or `neighbor_sequence` as model features.
- Compare multi-window/context variants against both T0005 baseline and T0006 `safe_racket_weighted`.
- Evaluate every variant by dense bucket/session and separately against ordinary up/down bounce regression sessions.
- Do not export the candidate into Collector app model JSON in this ticket.
- Do not build or install an APK in this ticket.

## Non-Goals

- No Collector UI integration yet.
- No video/FH-BH work yet.
- No changes to ordinary `studs_live` promotion rules.
- No app model export or APK build.

## Acceptance criteria

- A deterministic command or report compares multi-window/context variants against T0005 and T0006 baselines.
- Evaluation reports per-session and per-bucket racket/table recall, false positives, wrong-class table/racket errors, and close-event performance.
- Ordinary bounce regression results are reported separately and must not be mixed into dense playing aggregate metrics.
- Candidate remains local unless a later ticket explicitly approves app export.

## Manual verification

- Inspect `audio_session_2026-05-29_002` candidate errors before and after any multi-window/context variant.
- Confirm T0007 does not treat ordinary up/down bounce as the same promotion bucket as Tomas/Stiga dense play.

## Automated validation

- Run the new or updated error-analysis/training/evaluation command.
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
