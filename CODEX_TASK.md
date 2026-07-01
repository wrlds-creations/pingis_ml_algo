# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0104B-positive-review-pages`

## Branch

`codex/t0057-fable-auto-improvement-loop`

## Status

`Completed`

## Goal

Prepare exact-label review pages for the remaining useful T0104 positive `Bounce audio test` validation WAVs so Love can correct/save racket-contact timestamps before the next model loop.

## Dependencies

- T0104 pulled and summarized the fresh `Bounce audio test` debug runs.
- T0104A resolved the slow/high ambiguity: use `8781=20`, exclude `8782`.
- Love requested review pages for:
  - Normal racket bounce
  - Fast racket bounce
  - Racket bounce + speaking/counting
  - Racket bounce + background sound
  - Far/soft racket + background
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

- Locate the ten T0104 positive WAV/JSON pairs for the five requested scenario groups.
- Generate peak-prefilled manual review labels on separate local URLs.
- Use waveform peak candidates as gray read-only references and green editable draft racket-contact labels.
- Prefill up to each clip's expected count, normally `30`, so Love can delete false drafts, drag mistimed labels, or add missing contacts.
- Start and smoke-check local review servers on separate ports.
- Update source-of-truth docs/logs with the prepared URLs and review instructions.

## Non-Goals

- No model retraining/export.
- No app runtime change.
- No APK install.
- No raw/generated data commit.
- No merge or push.
- No camera/racket-side work.

## Acceptance Criteria

- Ten review pages are prepared from the requested T0104 positive WAVs.
- Review pages load through `/api/session` and expose editable green labels.
- Love receives the URLs and clear save instructions.

## Completion Notes

Added `prepare_t0104b_positive_review_pages.py` to prepare exact-label review pages for the remaining useful T0104 positives:

| URL | Scenario | Peaks | Draft Labels |
|---|---|---:|---:|
| `http://127.0.0.1:8783/` | Normal racket bounce | `30` | `30` |
| `http://127.0.0.1:8784/` | Normal racket bounce | `30` | `30` |
| `http://127.0.0.1:8785/` | Fast racket bounce | `30` | `30` |
| `http://127.0.0.1:8786/` | Fast racket bounce | `30` | `30` |
| `http://127.0.0.1:8787/` | Racket bounce + speaking/counting | `31` | `30` |
| `http://127.0.0.1:8788/` | Racket bounce + speaking/counting | `33` | `30` |
| `http://127.0.0.1:8789/` | Racket bounce + background sound | `30` | `30` |
| `http://127.0.0.1:8790/` | Racket bounce + background sound | `49` | `30` |
| `http://127.0.0.1:8791/` | Far/soft racket bounce + background | `45` | `30` |
| `http://127.0.0.1:8792/` | Far/soft racket bounce + background | `44` | `30` |

Generated ignored local artifacts under `data/audio/models/evaluations/t0104b_positive_review_pages/`.

Started and smoke-checked all ten local review pages. Each page is `manual_only=true`, expected `30`, with gray waveform peak candidates and editable green draft racket-contact labels.

## Validation

- `python -m py_compile skills/pingis-audio-classification/scripts/noise_robust/prepare_t0104b_positive_review_pages.py` passed.
- `python skills/pingis-audio-classification/scripts/noise_robust/prepare_t0104b_positive_review_pages.py --force --port-start 8783` passed.
- `/api/session` smoke checks passed for ports `8783` through `8792`.
