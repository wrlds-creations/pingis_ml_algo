# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0104C-p020-no-veto-safety-sweep`

## Branch

`codex/t0057-fable-auto-improvement-loop`

## Status

`Completed`

## Goal

Evaluate Love's live-observed `Bounce audio test` setting, `p>=0.20` with Fable-noise veto effectively disabled (`1.00`), against the already pulled/evaluable T0103/T0104 audio data before considering it good enough.

## Dependencies

- T0103 exported the guarded boundary-trained candidate behind the separate `Bounce audio test` entry.
- T0104 pulled and summarized the fresh `Bounce audio test` debug runs, with T0104A resolving the slow/high ambiguity.
- Love observed that typing `p=0.20` and noise veto `1.00` felt good on-device and asked whether the already pulled WAV/evaluation data can be swept before more labeling.
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

- Evaluate `threshold=0.20`, Fable-noise veto disabled/no-veto, against fresh corrected T0104 live debug summaries.
- Evaluate the same setting against the existing T0103 boundary/Round A safety policy sweep.
- Compare with safer thresholds including current installed default `0.575`.
- Produce a concise report that separates fresh Motorola results from older boundary/Round A safety.
- Make a recommendation on whether `0.20/no-veto` is already good enough or only a useful diagnostic.
- Update source-of-truth docs/logs with the result.

## Non-Goals

- No model retraining/export.
- No app runtime change.
- No APK install.
- No raw/generated data commit.
- No merge or push.
- No camera/racket-side work.
- Do not ingest unfinished T0104B review labels in this ticket.

## Acceptance Criteria

- T0104 live block-level `p>=0.20` counts are summarized.
- T0103 boundary/Round A safety sweep rows for `p>=0.20/no-veto` are summarized.
- The report clearly states whether the setting is safe enough to promote or should remain diagnostic.

## Completion Notes

Added `summarize_t0104c_p020_no_veto_sweep.py` to summarize Love's live-observed `p>=0.20` / no-veto setting from already pulled/evaluable artifacts.

Generated ignored local artifacts under `data/audio/models/evaluations/t0104c_p020_no_veto_safety_sweep/`.

Key result:

| Slice | p>=0.20/no-veto | p>=0.30/no-veto | default p>=0.575/no-veto |
|---|---:|---:|---:|
| Fresh T0104 positives | `303/320` | `284/320` | `181/320` |
| Fresh T0104 negative false counts | `3` | `0` | `0` |
| T0103 boundary positives | `242/269` | `239/269` | `225/269` |
| T0103 boundary negative false counts | `56` | `28` | `2` |
| Round A positives | `960/960` | `958/960` | `944/960` |
| Round A hard-negative false counts | `121` | `70` | `4` |

Conclusion: `p>=0.20/no-veto` explains why Love's quick phone test felt good, but it is not safe enough to promote. Keep it diagnostic/manual only; next work should ingest T0104B exact labels and train/evaluate a better candidate.

## Validation

- `python -m py_compile skills/pingis-audio-classification/scripts/noise_robust/summarize_t0104c_p020_no_veto_sweep.py` passed.
- `python skills/pingis-audio-classification/scripts/noise_robust/summarize_t0104c_p020_no_veto_sweep.py` passed.
