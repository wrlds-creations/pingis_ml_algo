# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0104E-live-positive-candidate-loop`

## Branch

`codex/t0057-fable-auto-improvement-loop`

## Status

`Complete`

## Goal

Train/evaluate a guarded audio candidate loop using the newly ingested T0104D live positives, while preserving the existing boundary/Round A hard-negative safety gates.

## Dependencies

- T0104D ingested `300/300` exact labels from T0104B, with `288/300` within `140 ms` and `295/300` within `250 ms`.
- T0104C showed threshold lowering alone is diagnostic only, not safe enough to promote.
- T0103 candidate-loop artifacts and older boundary/Round A safety rows are available locally.
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

- Build or reuse app-style candidate rows for T0104D positive labels.
- Train/evaluate candidate policies using T0104D positives plus existing T0103 boundary positives/negatives and older Round A/T0073 safety rows.
- Compare against the installed T0103 default and the diagnostic threshold-only settings.
- Report live-positive recall, boundary false counts, Round A hard-negative false counts, and scenario-level failures.
- Do not export or install unless a candidate clearly passes the offline safety gate and a separate install ticket is opened.

## Non-Goals

- No app model export.
- No app runtime change.
- No APK install.
- No raw/generated data commit.
- No merge or push.
- No camera/racket-side work.
- Do not change the T0103 candidate or `Bounce audio test` defaults.

## Acceptance Criteria

- Candidate-loop report is generated from T0104D positives and safety rows.
- The report states whether any candidate is worth an APK test.
- If no candidate passes, the failure mode and next data/model need are clear.

## Completion Notes

- Added `evaluate_t0104e_live_positive_candidate_loop.py` as an offline-only candidate loop.
- Built live app-style candidate rows from the installed T0103 debug JSON feature vectors and the corrected T0104 labels:
  - T0104D: `300/300` reviewed positives across normal, fast, speaking/counting, background, and far/soft+background.
  - T0104A: included only the first confirmed slow/high run (`20` labels), excluded the unclear second slow/high run.
  - T0104 fresh negatives: talking-only and racket-handling-only sessions included as expected-zero safety rows.
- Generated ignored outputs under `data/audio/models/evaluations/t0104e_live_positive_candidate_loop/`.
- Best newly trained candidate is only a near miss:
  - `extra_leaf2_t0104e_base_t0075_live_recall_safety_thr0p575_dedupe180_vetoNone`
  - Live positives: `274/320` (`0.8562` recall)
  - Fresh live negatives: `4` false counts
  - T0103 boundary negatives: `2` false counts
  - Round A hard negatives: `8` false counts
- Current T0103 app model with a typed diagnostic setting `p>=0.30`, no Fable-noise veto, dedupe `180 ms` is the better next manual phone-test/data-collection baseline on fresh T0104:
  - Fresh live positives: `270/320` (`0.8438` recall)
  - Fresh live negatives: `0` false counts
  - Its boundary/Round A rows in the T0104E app-baseline table are final-fit/in-sample context only, not a promotion safety result.
  - The older T0104C out-of-fold safety warning still applies: `p>=0.30/no-veto` had `28` boundary false counts and `70` Round A hard-negative false counts in the T0103 OOF sweep.
- No model JSON was exported, no runtime/app change was made, and no APK was installed.

## Validation

- `python -m py_compile skills\pingis-audio-classification\scripts\noise_robust\evaluate_t0104e_live_positive_candidate_loop.py`
- `python skills\pingis-audio-classification\scripts\noise_robust\evaluate_t0104e_live_positive_candidate_loop.py`
