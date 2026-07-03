# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0129-hybrid-hard-negative-veto-model`

## Branch

`codex/t0057-fable-auto-improvement-loop`

## Status

`Completed`

## Goal

Train and evaluate a learned hard-negative veto that runs only after the existing STIGA `Hybrid` bounce detector has already accepted a clip. Keep the existing Hybrid detector unchanged, and replace the failed hand-written `band_energy_mid` Hybrid 2.0 veto with a learned `hard_positive` / `hard_negative` model if grouped validation shows it preserves bounce recall while removing hard false positives.

## Dependencies

- T0127 made soft peak available for app-side candidate generation tests.
- STIGA `Hybrid` currently works better on the user's Android/iPhone tests than the failed `Hybrid 2.0` hand rule.
- The failed `Hybrid 2.0` rule used a direct `band_energy_mid` threshold and rejected real bounces in live testing.
- The latest hard-negative dataset contains racket handling, talking/counting, floor/table/other impacts, background noise, and recent multi-phone positives.

## Allowed Areas

- `CODEX_TASK.md`
- `PROJECT_CONTEXT.md`
- `REPO_CURRENT_STATE.md`
- `ITERATION_LOG.md`
- `DECISIONS.md`
- `skills/pingis-audio-classification/scripts/noise_robust/`
- `data/audio/models/evaluations/t0129_hybrid_hard_negative_veto_model/` as ignored local evaluation output
- If validation is acceptable, `C:\Development\Wrlds\android\stiga-app-v2` files needed to port the learned veto into the guarded `Hybrid 2.0` test option

## Do Not Touch

- Do not delete raw/generated `data/`.
- Do not merge to `main`.
- Do not replace the current working STIGA `Hybrid` mode.
- Do not add a direct `band_energy_mid` or other single-feature hard-coded veto.
- Do not promote the learned veto to production/default behavior before phone validation.
- Do not change camera/rubber-side behavior.
- Do not use cloud APIs or AWS.

## Requirements

- Train the new layer only on rows that the existing STIGA `Hybrid` stack would accept.
- Use available app-portable inputs:
  - existing 62 RF audio features from the clip;
  - existing binary contact RF probabilities/confidence;
  - existing 4-class surface RF probabilities/confidence.
- Export a compact JSON model that can run in the STIGA app without Python/sklearn.
- Evaluate with grouped out-of-fold validation so one recording/session does not train and test on itself.
- Report comparison against plain Hybrid:
  - Hybrid true-positive count and false-positive count after app-style dedupe;
  - learned-veto true-positive count and false-positive count after app-style dedupe;
  - false-positive reduction by hard-negative bucket;
  - true-positive losses by positive bucket.
- Port only if the learned veto is clearly better than the failed hand rule and does not materially break normal/far/fast/background bounce positives.

## Non-Goals

- No new peak picker, sliding-window, or candidate-generation change.
- No replacement of the existing Hybrid detector.
- No native app release promotion.
- No new raw data collection requirement for this ticket.

## Acceptance Criteria

- A reproducible training/evaluation script exists for the learned Hybrid hard-negative veto.
- The script exports a small portable model JSON and report under the ignored T0129 evaluation folder.
- If the metrics are acceptable, STIGA `Hybrid 2.0` uses the learned veto behind existing `Hybrid`, while plain `Hybrid` remains unchanged.
- Validation commands are run where practical, or blockers are documented.

## Completion Notes

- Added `skills/pingis-audio-classification/scripts/noise_robust/train_t0129_hybrid_hard_negative_veto.py`.
- The trainer avoids sklearn and fits a portable `logistic_binary_v1` hard-positive / hard-negative model with numpy.
- Training rows are only rows accepted by the existing STIGA Hybrid stack.
- Inputs are app-portable: the existing 62 RF audio features, binary contact RF probabilities/confidence, and 4-class surface RF probabilities/confidence.
- Grouped out-of-fold validation by `domain_session_id` selected negative weight `1.5`, L2 `0.002`, and hard-positive threshold `0.45`.
- Plain Hybrid after dedupe scored TP/FP `1518/791`.
- Learned post-Hybrid veto after dedupe scored TP/FP `1493/37`, losing `25` true positives while removing `754` false positives.
- Exported ignored local artifacts under `data/audio/models/evaluations/t0129_hybrid_hard_negative_veto_model/`.
- Ported the exported JSON into the guarded STIGA `Hybrid 2.0` QA option on sibling branch `codex/t0419-studsboll-learned-hybrid-veto`.
- Plain STIGA `Hybrid` remains unchanged.

## Validation

- `python -m py_compile skills/pingis-audio-classification/scripts/noise_robust/train_t0129_hybrid_hard_negative_veto.py`
- `python skills/pingis-audio-classification/scripts/noise_robust/train_t0129_hybrid_hard_negative_veto.py`
