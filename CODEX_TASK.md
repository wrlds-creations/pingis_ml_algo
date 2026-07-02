# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0124-sliding-window-full-flow-replay`

## Branch

`codex/t0057-fable-auto-improvement-loop`

## Status

`Completed`

## Goal

Evaluate, offline only, whether replacing the `Bounce audio test` peak candidate timestamps with softer peak or sliding-window PCM candidate timestamps improves the full current T0104E flow:

`Audio WAV -> candidate timestamps -> existing live clip extraction -> Fable features/model -> T0104E ExtraTrees -> threshold/noise veto/smart dedupe -> scored count`

## Dependencies

- T0121 showed the failed calibrated transient phone run is partly candidate-gate limited and partly second-layer limited.
- T0122 restored failed app/runtime experiments out of the worktree.
- T0123 showed soft/sliding PCM candidate generation can recover most missed true bounces, but produces too many raw candidates to count directly.
- Existing reviewed labels are available from T0121 and T0104D/T0104B positive review pages.
- Existing expected-zero T0104 sessions are available for hard-negative pressure.

## Allowed Areas

- `CODEX_TASK.md`
- `PROJECT_CONTEXT.md`
- `REPO_CURRENT_STATE.md`
- `ITERATION_LOG.md`
- `DECISIONS.md`
- `skills/pingis-audio-classification/scripts/noise_robust/`
- ignored local outputs under `data/audio/models/evaluations/t0124_sliding_window_full_flow_replay/`

## Do Not Touch

- Do not delete raw/generated `data/`.
- Do not merge to `main`.
- Do not push unless explicitly requested.
- Do not delete local or device data.
- Do not replace or promote production Fable/studs/camera behavior.
- Do not move raw/generated data into git.
- Do not train or export a new model.
- Do not change app/runtime/model JSON files.
- Do not install an APK.

## Requirements

- Add an evaluation-only replay script that loads existing WAVs and reviewed labels.
- Compare current fixed peak, soft peak, and selected sliding-window PCM candidate timestamp methods.
- For each candidate method, run the existing app-style second layer:
  - event-centered live clip extraction;
  - Fable feature extraction and current `fable_audio_model.json`;
  - exported `fable_extra_trees_candidate_t0104e.json`;
  - raw ExtraTrees probabilities;
  - threshold / Fable-noise-veto / smart dedupe.
- Report positive true counts, misses, false/unmatched counts, and expected-zero negative false counts.
- Include at least the current app test setting `p=0.25`, Fable-noise veto `0.98`, and a strict reference policy.

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
- Report clearly compares full-flow counts for current peak, soft peak, and sliding-window candidates.
- Report identifies whether sliding-window timestamps are better after the existing classifier/veto/dedupe stack, not just before it.
- Root validation and `git diff --check` pass or blockers are documented.

## Completion Notes

- Added `evaluate_t0124_sliding_window_full_flow_replay.py`, an offline evaluator that runs candidate timestamps through live clip extraction, Fable features/model, exported T0104E ExtraTrees JSON, threshold/noise-veto, and smart dedupe.
- Wrote ignored outputs under `data/audio/models/evaluations/t0124_sliding_window_full_flow_replay/`.
- Under the current favored diagnostic setting `p=0.25`, Fable-noise veto `0.98`, and dedupe `180 ms`, best sliding-window row `sw_raw_abs030_r2_z4` scored `288/330` true positives (`87.3%`) with `3` expected-zero negative false counts and `3` positive unmatched counts.
- The current peak reference `current_peak_abs008` scored `271/330` true positives (`82.1%`) with the same `3` expected-zero negative false counts and `1` positive unmatched count.
- The softer peak row `soft_peak_abs003` scored `285/330` true positives (`86.4%`) with `4` expected-zero negative false counts.
- On the critical T0119 speaking/counting clip, current peak produced only `1` candidate and counted `0/30`; soft peak counted `13/30`; the best raw sliding row counted `10/30`. This means recovered timestamps help, but the existing T0104E/Fable decision layer still rejects many hard bounces.
- Conclusion: replacing the peak picker with sliding-window candidates is promising in full-flow replay, but it is not ready for app promotion without a stronger second-layer/veto or policy tuning because negative false counts remain and T0119 recall is still weak.

## Validation

- `python -m py_compile skills/pingis-audio-classification/scripts/noise_robust/evaluate_t0124_sliding_window_full_flow_replay.py`
- `python skills/pingis-audio-classification/scripts/noise_robust/evaluate_t0124_sliding_window_full_flow_replay.py`
- `npm run validate`
- `git diff --check` passed with existing LF/CRLF warnings only.
