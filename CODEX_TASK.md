# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0104D-t0104b-positive-label-ingest`

## Branch

`codex/t0057-fable-auto-improvement-loop`

## Status

`Completed`

## Goal

Ingest Love's saved exact labels from the ten T0104B positive review pages, all confirmed expected `30`, and produce corrected label/coverage artifacts for the next candidate-training loop.

## Dependencies

- T0104B prepared positive review pages on `8783`-`8792`.
- Love saved labels for all ten pages and confirmed every clip should be expected `30`.
- T0104C showed threshold lowering alone is diagnostic only, not safe enough to promote.
- Raw/generated data under `data/` remains ignored and must not be committed.

## Allowed Areas

- `CODEX_TASK.md`
- `REPO_CURRENT_STATE.md`
- `PROJECT_CONTEXT.md` if confirmed project facts change
- `DECISIONS.md` if a meaningful decision changes
- `ITERATION_LOG.md`
- `skills/pingis-audio-classification/scripts/noise_robust/`
- ignored local raw/evaluation artifacts under `data/audio/`
- validation/status commands

## Do Not Touch

- Do not merge to `main`.
- Do not push.
- Do not delete local or device data.
- Do not revert tracked or user changes.
- Do not replace or promote the T0103 model.
- Do not change app thresholds, model JSON, native audio runtime, production Fable behavior, studs/camera behavior, cloud/API credentials, backend resources, or AWS resources.
- Do not move raw/generated data into git.

## Requirements

- Read all ten T0104B review-label JSON files.
- Enforce Love's confirmed expected count of `30` for every clip.
- Produce reviewed positive label CSV, nearest peak/candidate match CSV, per-session summary, summary JSON, and markdown report.
- Report candidate coverage within `140 ms` and `250 ms`, manual additions/deletions, and scenario-level weak spots.
- Update source-of-truth docs/logs with the ingest result and next recommended model loop.

## Non-Goals

- No model retraining/export in this ticket.
- No app runtime change.
- No APK install.
- No raw/generated data commit.
- No merge or push.
- No camera/racket-side work.
- Do not change the T0103 candidate or `Bounce audio test` defaults.

## Acceptance Criteria

- All ten saved T0104B pages are ingested with expected count `30`.
- The report shows exact label totals and peak/candidate coverage by scenario.
- The next training/evaluation step is clear.

## Completion Notes

Added `ingest_t0104b_positive_review_labels.py` to ingest the ten saved T0104B review pages with Love-confirmed expected count `30` for every clip.

Generated ignored local artifacts under `data/audio/models/evaluations/t0104d_t0104b_positive_label_ingest/`:

- `t0104d_reviewed_positive_labels.csv`
- `t0104d_nearest_peak_matches.csv`
- `t0104d_session_summary.csv`
- `t0104d_scenario_summary.csv`
- `t0104d_summary.json`
- `t0104d_report.md`

Key result:

| Scenario | Labels | Current App Count | Peak Candidates | Within 140 ms | Within 250 ms |
|---|---:|---:|---:|---:|---:|
| Normal racket bounce | `60` | `38` | `60` | `60` | `60` |
| Fast racket bounce | `60` | `46` | `60` | `60` | `60` |
| Racket bounce + speaking/counting | `60` | `47` | `64` | `60` | `60` |
| Racket bounce + background sound | `60` | `34` | `79` | `59` | `60` |
| Far/soft racket bounce + background | `60` | `4` | `89` | `49` | `55` |

Total reviewed positives: `300/300`; strict peak-candidate coverage: `288/300` within `140 ms`, `295/300` within `250 ms`.

Conclusion: T0104B labels are clean and useful for the next candidate-training loop. Far/soft + background still has some candidate-generation weakness, but the main problem for most fresh positives is classifier/scoring rather than missing peaks.

## Validation

- `python -m py_compile skills/pingis-audio-classification/scripts/noise_robust/ingest_t0104b_positive_review_labels.py` passed.
- `python skills/pingis-audio-classification/scripts/noise_robust/ingest_t0104b_positive_review_labels.py` passed.
