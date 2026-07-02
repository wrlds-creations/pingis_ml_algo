# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0123-sliding-window-pcm-candidate-audit`

## Branch

`codex/t0057-fable-auto-improvement-loop`

## Status

`Completed`

## Goal

Evaluate, offline only, whether a buffered/sliding-window PCM candidate generator can recover the bounces missed by the current peak-gate path without obviously exploding talking/handling false candidates.

## Dependencies

- T0121 showed the calibrated transient phone run is mostly candidate-gate limited.
- T0122 restored the failed adaptive/transient app experiment out of the worktree.
- Existing reviewed labels are available from T0121 and T0104D/T0104B positive review pages.
- Existing expected-zero T0104 sessions are available for first-pass hard-negative pressure.

## Allowed Areas

- `CODEX_TASK.md`
- `PROJECT_CONTEXT.md`
- `REPO_CURRENT_STATE.md`
- `ITERATION_LOG.md`
- `DECISIONS.md`
- `skills/pingis-audio-classification/scripts/noise_robust/`
- ignored local outputs under `data/audio/models/evaluations/t0123_sliding_window_pcm_candidate_audit/`

## Do Not Touch

- Do not delete raw/generated `data/`.
- Do not merge to `main`.
- Do not push unless explicitly requested.
- Do not delete local or device data.
- Do not replace or promote production Fable/studs/camera behavior.
- Do not move raw/generated data into git.
- Do not train or export a model.
- Do not change app/runtime/model JSON files.
- Do not install an APK.

## Requirements

- Add an evaluation-only script that loads existing WAVs and reviewed labels.
- Compare current fixed peak gate, soft peak gate, and several sliding-window PCM candidate configs.
- Report true-label coverage at `140 ms` and `250 ms`, unmatched candidates on positives, and candidate counts on expected-zero negative sessions.
- Use T0121 as the critical miss case and T0104D positives/T0104 negatives as broader sanity checks.

## Non-Goals

- No new app code.
- No model export/retrain.
- No production/default Fable, studs, or camera behavior change.
- No APK/reinstall.
- No cloud/API/AWS changes.
- No deletion of local analysis/audio files.
- No claim that a config is production-ready from this audit alone.

## Acceptance Criteria

- Script runs from the repo root and writes ignored CSV/JSON/MD outputs.
- Report clearly says whether sliding-window candidates improve T0121 and broader positive coverage versus peak gate.
- Report also shows expected-zero negative candidate load.
- Root validation and `git diff --check` pass or blockers are documented.

## Completion Notes

- Added `evaluate_t0123_sliding_window_pcm_candidate_audit.py`, an offline-only evaluator for current fixed peak gate, softer peak gates, and sliding-window PCM candidate generators.
- Wrote ignored audit outputs under `data/audio/models/evaluations/t0123_sliding_window_pcm_candidate_audit/`.
- Best positive recall row was `soft_peak_abs003`: `319/330` truth labels matched within `140 ms` (`96.7%`) with `319` expected-zero negative candidates.
- Best sliding-window row was `sw_raw_abs030_r2_z4`: `318/330` within `140 ms` (`96.4%`) with `311` expected-zero negative candidates.
- Current fixed peak reference `current_peak_abs008` was `288/330` within `140 ms` (`87.3%`) with `294` expected-zero negative candidates.
- On the critical T0119/T0121 speaking/counting miss case, current fixed peak found `0/30`, while soft peak and high-pass sliding-window variants found `30/30`.
- Conclusion: sliding-window/soft PCM candidate generation is promising as a recovery layer, but not as a standalone counter because expected-zero candidate load is still high. The next ticket should pair recovered candidates with a classifier/veto before any app/runtime promotion.

## Validation

- `python -m py_compile skills/pingis-audio-classification/scripts/noise_robust/evaluate_t0123_sliding_window_pcm_candidate_audit.py`
- `python skills/pingis-audio-classification/scripts/noise_robust/evaluate_t0123_sliding_window_pcm_candidate_audit.py`
- `npm run validate`
- `git diff --check` passed with existing LF/CRLF warnings only.
