# CODEX_TASK.md

Use one active ticket per implementation pass. The active ticket for this
branch is the approved GitHub issue below.

## Ticket ID

`GH-6-dual-v2-edge-racket`

## Branch

`codex/gh-6-dual-v2-edge-racket`

## Status

`Completed`

## Goal

Retrain a versioned V2 `dual_residual` bounce classifier that improves recall
for ball contacts near the racket edge without weakening the frozen model's
cross-device noise resistance.

## Approved Issue

- https://github.com/wrlds-creations/pingis_ml_algo/issues/6
- Approved directly by the user on 2026-07-28.

## Dependencies

- The frozen Dual baseline and reproducible training/evaluation pipeline from
  issue #4.
- The completed edge-racket recording and review round
  `CJ-20260727-01`.
- `RECORDING_PLAN_EDGE_RACKET.md` as the source of truth for corrections,
  split policy, and evaluation boundaries.

## Allowed Areas

- `skills/pingis-audio-classification/scripts/hf_pcen_cnn/`
- `data/rounds-CJ-20260727-01-labeled/` derived manifests and ignored outputs
- `CODEX_TASK.md`
- `DECISIONS.md`
- `ITERATION_LOG.md`
- `RECORDING_PLAN_EDGE_RACKET.md`
- focused tests and validation configuration required by the pipeline

## Do Not Touch

- Raw WAV or source session JSON data
- STIGA application/runtime source
- Production model defaults
- AWS or backend resources
- The sealed final holdout's membership or labels

## Requirements

- Initialize a fresh `dual_residual` model and train from scratch.
- Use the original Train split plus reviewed T01/T02 from
  `CJ-20260727-01`.
- Correct T01 iPhone/Motorola target metadata only in a derived manifest.
- Keep T03/T04 diagnostic-only and exclude the original final holdout from
  training and threshold selection.
- Compare V2 with the unchanged frozen Dual baseline.
- Report edge-racket and center-racket recall, table and speech/noise false
  positives, per-device metrics, count error, and sealed-holdout metrics.
- Keep generated checkpoints, features, and reports ignored.

## Non-Goals

- No STIGA export, runtime promotion, or production-default change.
- No HF-gate retuning unless the reviewed evidence shows gate misses.
- No mutation of raw recordings or sealed evaluation membership.

## Acceptance Criteria

- Intake audit confirms complete reviewed T01-T04 device coverage.
- The expanded manifest and training command are reproducible.
- Evaluation separates Train-derived validation, T03/T04 diagnostics, and the
  untouched final holdout.
- The final report records both improvements and regressions against frozen
  Dual and makes an evidence-based promotion recommendation.

## Validation

- Focused dataset, training, and evaluation tests
- Python compile checks
- `npm run validate`
- `git diff --check`

## Outcome

- Built a reproducible 9,838-row edge-round manifest and complete
  log-mel/PCEN feature cache from `CJ-20260727-01`.
- Trained `dual_residual_v2_edge_racket` from scratch on 16,343 rows, including
  1,020 reviewed T01/T02 additions.
- Improved same-day T03/T04 counting F1 from `0.6247` to `0.8269` and original
  final-holdout counting F1 from `0.7220` to `0.7483`.
- Preserved the frozen `0.775` threshold and original holdout membership.
- Left STIGA runtime and production model defaults unchanged. V2 remains a QA
  candidate pending an independent edge-contact round, especially for racket D
  black top-edge contacts.
