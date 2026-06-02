# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0006`

## Branch

`codex/t0006-improve-playing-retro-audio-candidate`

## Goal

Improve the `spel_retro_audio` candidate before any app integration by analyzing T0005 holdout errors and testing focused candidate/window/context variants that raise racket recall without making table or non-target behavior worse.

## Dependencies

- T0001 ticket workflow adoption is complete.
- T0002 documentation refresh removed IMU/AirHive from active project scope.
- T0003 cleanup removed retired IMU/AirHive workflow docs and skill scripts from the active repo workflow.
- T0004 generated `data/audio/processed/playing_retro_candidate_peak_rows.csv`, `playing_retro_candidate_peak_summary.csv`, and `playing_retro_candidate_peak_report.md` locally.
- T0005 trained local candidate `playing_retro_audio_rf_v2026_06_02_app_candidates_100_200` from 4,028 candidate-centered rows across 16 reviewed playing sessions.
- T0005 holdout on `audio_session_2026-05-29_002` reached `0.759` accuracy versus old app prediction `0.682`, but racket recall was only `0.604`, so it is not ready for app export.
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

- Use the T0005 candidate dataset and evaluation report as the baseline.
- Keep the T0005 training choice fixed unless the error analysis proves that a different row policy is necessary.
- Start from the T0005 dataset, evaluation CSV, and local model report.
- Inspect holdout errors by `source_rule`, close-event bucket, and confusion class, especially `wrong_class_racket_as_table`, `manual_missed_marker`, and sub-120 ms gaps.
- Try focused candidate variants only when the error analysis points to a concrete hypothesis.
- Evaluate every variant by dense bucket/session and separately against ordinary up/down bounce regression sessions.
- Do not export the candidate into Collector app model JSON in this ticket.
- Do not build or install an APK in this ticket.

## Non-Goals

- No Collector UI integration yet.
- No video/FH-BH work yet.
- No changes to ordinary `studs_live` promotion rules.
- No app model export or APK build.

## Acceptance criteria

- A deterministic command or report identifies the largest T0005 error causes and compares any tested variants against the T0005 baseline.
- Evaluation reports per-session and per-bucket racket/table recall, false positives, wrong-class table/racket errors, and close-event performance.
- Ordinary bounce regression results are reported separately and must not be mixed into dense playing aggregate metrics.
- Candidate remains local unless a later ticket explicitly approves app export.

## Manual verification

- Inspect `audio_session_2026-05-29_002` candidate errors before and after any variant.
- Confirm T0006 does not treat ordinary up/down bounce as the same promotion bucket as Tomas/Stiga dense play.

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
