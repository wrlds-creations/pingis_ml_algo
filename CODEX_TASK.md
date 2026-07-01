# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0104-bounce-audio-test-live-validation-pull`

## Branch

`codex/t0057-fable-auto-improvement-loop`

## Status

`Completed`

## Goal

Pull and analyze Love's fresh Motorola `Bounce audio test` validation runs for the guarded T0103 candidate. Determine whether the live failures are mainly candidate-gate misses, final classifier/threshold misses, duplicate/dedupe behavior, or hard-negative false counts.

## Dependencies

- T0103 exported and installed the guarded `Bounce audio test` candidate.
- T0103A clarified the validation tags and reinstalled the app.
- Love tested two runs each for:
  - Normal racket bounce, expected `30`
  - Slow/high racket bounce, expected `30`
  - Fast racket bounce, expected `30`
  - Racket bounce + speaking/counting, expected `30`
  - Racket bounce + background sound, expected `30`
  - Far/soft racket + background, expected `30`
  - Talking only, expected `0`
  - Racket handling only, expected `0`
- Love observed that some runs showed roughly `30` detected/logged candidates but only about `2` counted bounces, suggesting probability/threshold rejection may dominate.
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

- Pull the latest `Download/pingis_sessions/bounce_audio_test_debug` files from Motorola into a new ignored local T0104 raw folder.
- Identify the fresh validation runs and map them to Love's provided scenario/order information when possible.
- Produce a concise report with:
  - scenario, expected count, app count, candidate count;
  - counted vs rejected candidates;
  - rejection reasons and probability distribution;
  - evidence for whether the bottleneck is peak candidate generation, final classifier probability/threshold, dedupe, or false positives.
- Keep app behavior unchanged during this audit.
- Update source-of-truth docs/logs with the result.

## Non-Goals

- No model retraining/export.
- No app runtime change.
- No APK install unless a later ticket explicitly changes runtime behavior.
- No raw/generated data commit.
- No merge or push.
- No camera/racket-side work.

## Acceptance Criteria

- Fresh T0104 debug files are pulled locally and summarized.
- The report explains the observed "many detected, few counted" behavior with concrete counts/probabilities from the debug JSON.
- Next recommendation is concrete: threshold tuning, classifier retrain, gate work, or additional exact labeling.

## Completion Notes

Pulled `38` files from Motorola `Download/pingis_sessions/bounce_audio_test_debug` into ignored local raw data under `data/audio/raw/t0104_bounce_audio_test_live_validation/`.

Added `summarize_t0104_bounce_audio_test_validation.py` and generated ignored report artifacts under `data/audio/models/evaluations/t0104_bounce_audio_test_live_validation/`.

Key result:

- Fresh analyzed sessions: `16`.
- Love-reported positive expected/count: `186/360`.
- Positive peak candidates: `389`.
- Positive low-probability rejections: `203`.
- Talking-only and racket-handling-only negatives: `0` counted from `287` peak candidates.
- Dedupe and Fable-noise veto were not material in these runs; the dominant miss reason is `below_threshold`.
- Far/soft + background is the clearest failure: `4/60` counted at T0103 `p>=0.575`, despite `88` peak candidates.
- The two slow/high runs were saved in app metadata as expected `20`, but Love reported `30`; the report uses Love's `30` as the real expected count and marks those rows.

Conclusion: your "the app sees candidates but the percent is too low" read is correct for the weak positive runs. T0103 should not be promoted as-is. The next useful work is exact-labeling the weak live positive runs or training a next candidate that uses this pull as validation while preserving older hard-negative safety.

## Validation

- `python -m py_compile skills/pingis-audio-classification/scripts/noise_robust/summarize_t0104_bounce_audio_test_validation.py` passed.
- `python skills/pingis-audio-classification/scripts/noise_robust/summarize_t0104_bounce_audio_test_validation.py` passed and wrote ignored T0104 CSV/JSON/MD report artifacts.
