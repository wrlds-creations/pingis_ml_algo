# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0104A-slow-high-expected-count-review-pages`

## Branch

`codex/t0057-fable-auto-improvement-loop`

## Status

`Completed`

## Goal

Prepare exact-label review pages for the two ambiguous T0104 `Slow/high racket bounce` validation WAVs so Love can resolve whether the real count was `20` or `30`.

## Dependencies

- T0104 pulled and summarized the fresh `Bounce audio test` debug runs.
- Two `Slow/high racket bounce` sessions were saved in app metadata as expected `20`, while Love initially reported `30` and then noted uncertainty.
- The exact count needs human waveform/audio review before using these runs as validation labels.
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

- Locate the two T0104 slow/high WAV/JSON pairs.
- Generate peak-prefilled manual review labels on separate local URLs.
- Use waveform peak candidates as gray read-only references and green editable draft racket-contact labels.
- Prefill up to `30` draft labels so Love can delete extras if the actual count was `20`, or add missing labels if needed.
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

- Two review pages are prepared from the T0104 slow/high WAVs.
- Review pages load through `/api/session` and expose editable green labels.
- Love receives the URLs and clear save instructions.

## Completion Notes

Added `prepare_t0104a_slow_high_review_pages.py` to prepare exact-label review pages for the two ambiguous T0104 slow/high runs:

- `bounce_audio_test_session_2026-07-01T13-37-11-083Z`: saved expected `20`, app count `12`, WAV duration `21.900s`, waveform peak candidates/draft labels `20`.
- `bounce_audio_test_session_2026-07-01T13-38-19-066Z`: saved expected `20`, app count `5`, WAV duration `22.449s`, waveform peak candidates/draft labels `24`.

The review pages intentionally display expected `30` because the question is whether the real count was `20` or `30`. Green labels are editable draft labels from `peak_fast_balanced`; if Love hears only `20`, delete extras, and if Love hears more than the draft labels, add missing contacts.

Generated ignored local artifacts under `data/audio/models/evaluations/t0104a_slow_high_expected_count_review/`.

Started and smoke-checked local review pages:

- `http://127.0.0.1:8781/` with `20` gray candidates and `20` green draft labels.
- `http://127.0.0.1:8782/` with `24` gray candidates and `24` green draft labels.

## Validation

- `python -m py_compile skills/pingis-audio-classification/scripts/noise_robust/prepare_t0104a_slow_high_review_pages.py` passed.
- `python skills/pingis-audio-classification/scripts/noise_robust/prepare_t0104a_slow_high_review_pages.py --force --port-start 8781` passed.
- `/api/session` smoke checks passed for `8781` and `8782`.
