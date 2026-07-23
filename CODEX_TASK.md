# CODEX_TASK.md

Use one active ticket per implementation pass. The active ticket for this branch
is the approved GitHub issue below.

## Ticket ID

`GH-5-simplify-43-take-recording-plan`

## Branch

`codex/gh-5-simplify-recording-plan`

## Status

`Completed`

## Goal

Simplify the cross-device recording instructions so the phones capture only
stable round/take identity, target class, automatic provenance, and exceptions,
while retaining all 43 planned takes and moving experiment metadata to a
deterministic post-processing manifest.

## Approved Issue

- https://github.com/wrlds-creations/pingis_ml_algo/issues/5
- Approved directly by the user on 2026-07-22.

## Dependencies

- The current recording plan exists on the unmerged issue #4 branch.
- This branch is based on `codex/gh-4-hf-four-class-cnn`.
- STIGA recorder implementation is owned by a separate STIGA app issue.

## Allowed Areas

- `RECORDING_PLAN.md`
- `CODEX_TASK.md`
- `DECISIONS.md`
- `ITERATION_LOG.md`

## Do Not Touch

- App/runtime source
- Model artifacts, training code, or evaluation outputs
- Raw or processed recordings
- AWS or backend resources

## Requirements

- Preserve T01-T43, all seven blocks, and three-phone simultaneous capture.
- Separate the pilot from the official round.
- Keep every phone flat, including Block 6.
- Use one shared Round ID and stable Take IDs.
- Move scenario, position, split, expected-count, and label details to
  post-processing.
- Correct T30, identify T39 catch/after-sound coverage, and require T40 with a
  speech-playback fallback.
- Keep the physical-event holdout grouping deterministic.

## Non-Goals

- No model or app change.
- No reduction in take count.
- No raw data generation.

## Acceptance Criteria

- The plan contains contiguous T01-T43 and totals 43 takes per phone / 129 files.
- Operators enter only round/take identity, binary target, and exceptions.
- Every take has one binary target and deterministic post-processing metadata.
- The pilot cannot consume official T01.
- Block 6 remains present and required.

## Validation

- Manual take/target/holdout audit
- `npm run validate`
- `git diff --check`

## Completion Notes

- Preserved all seven blocks and contiguous T01-T43: 43 recordings per phone / 129 total.
- Separated the pilot as a different Round ID at T00, then reset the official round to T01.
- Removed per-phone scenario, position, orientation, split, distance, racket, background, and expected-count entry from the recording workflow.
- Kept every phone flat, including Block 6, without removing T37-T41.
- Corrected T30 to mixed household impacts, tagged T39 as catch/after-sound coverage, and made T40 mandatory with recorded speech as fallback.
- Defined deterministic post-processing joins, physical-event grouping, fixed holdout assignment, timestamp review, sync-clap exclusion, and quality flags.
- Fixed Music 1/Music 2 to downloaded local 60-second excerpts and documented
  no-loop/no-ad playback.
- Added NIOSH Sound Level Meter settings and round-level targets for music,
  normal counting, and raised speech while preserving real cross-device gain
  differences for post-processing.
- Verified 43 unique contiguous take rows with no missing IDs.
- `npm run validate` and `git diff --check` passed.
