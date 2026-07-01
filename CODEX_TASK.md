# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0109-t0104e-p025-veto098-sweep`

## Branch

`codex/t0057-fable-auto-improvement-loop`

## Status

`Complete`

## Goal

Evaluate Love's preferred `Bounce audio test` setting for T0104E: `p threshold = 0.25`, Fable noise veto `0.98`, smart dedupe `180 ms`, using the already pulled/labeled local artifacts.

## Dependencies

- T0104E is already bundled as a diagnostic switch-only candidate in `Bounce audio test`.
- T0108 preserves typed `p`/noise-veto values across model switches.
- Existing T0104/T0104D live-positive labels, T0103 boundary rows, Round A rows, and noisy target rows are available locally.
- Raw/generated `data/` remains ignored and must not be committed.

## Allowed Areas

- `CODEX_TASK.md`
- `PROJECT_CONTEXT.md`
- `DECISIONS.md`
- `REPO_CURRENT_STATE.md`
- `ITERATION_LOG.md`
- ignored evaluation outputs under `data/audio/models/evaluations/t0109_t0104e_p025_veto098_sweep/`
- validation/status commands

## Do Not Touch

- Do not merge to `main`.
- Do not push.
- Do not delete local or device data.
- Do not revert tracked or user changes.
- Do not replace or promote production Fable/studs/camera behavior.
- Do not change `audio_model.json`, `audio_contact_model.json`, `fable_audio_model.json`, `bounce_side_model.json`, T0103/T0104E JSON, native peak-gate defaults, or app runtime code.
- Do not move raw/generated data into git.

## Requirements

- Score the exported T0104E app JSON with `threshold=0.25`, `noise_veto=0.98`, and `dedupe=180 ms`.
- Also score the matching T0104E out-of-fold selected-model predictions to separate app-runtime optimism from promotion risk.
- Report live positives, fresh live negatives, boundary safety, Round A safety, noisy target rows, and scenario breakdown.
- Keep the result diagnostic only unless fresh phone validation says otherwise.

## Non-Goals

- No app code change.
- No model export/retrain.
- No APK install.
- No camera/racket-side changes.
- No new data pull or labeling.

## Acceptance Criteria

- Root validation passes.
- `git diff --check` passes.
- Ignored local outputs record the exact T0104E `0.25/0.98` sweep.
- Final answer explains both the app-finalfit result and the less-optimistic OOF risk.

## Completion Notes

- Generated ignored outputs under `data/audio/models/evaluations/t0109_t0104e_p025_veto098_sweep/`.
- Exported T0104E app JSON final-fit replay at `p=0.25`, `noise_veto=0.98`, `dedupe=180` scored:
  - live positives `286/320`;
  - fresh live negative false counts `2`;
  - boundary negative false counts `1`;
  - Round A hard-negative false counts `0`;
  - noisy-target negative false counts `1`.
- Matching T0104E selected OOF replay at the same setting scored:
  - live positives `283/320`;
  - fresh live negative false counts `32`;
  - boundary negative false counts `26`;
  - Round A hard-negative false counts `79`;
  - noisy-target negative false counts `44`.
- Interpretation: this is worth a guarded phone test because the exported app model likely explains why Love likes the setting, but the OOF safety row blocks promotion/defaulting.

## Validation

- Inline Python evaluation using existing T0104E helper functions and exported T0104E app JSON
  - wrote `t0109_summary_rows.csv`, `t0109_summary.json`, and `t0109_nearby_grid.csv` under ignored `data/audio/models/evaluations/t0109_t0104e_p025_veto098_sweep/`.
- `npm run validate`
- `git diff --check`
  - passed with existing Windows LF-to-CRLF warnings only.
