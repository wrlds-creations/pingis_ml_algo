# Repo Current State

Use this file as the living snapshot of what actually exists in the repository. Update it after completed tickets, audits, meaningful dependency changes, validation changes, workflow changes, model/data/build state changes, or next-ticket changes.

## Snapshot

- Date: `2026-06-02`
- Current branch: `codex/video-stroke-test`
- Current phase: `Audio/video stabilization with playing-retro audio next`
- Current status: `T0004 candidate-centered playing-retro audio report built; next active ticket is training/evaluating from those rows`
- Current ticket: `T0005`
- Last completed ticket: `T0004`
- Recommended next ticket: `T0005-train-playing-retro-audio-candidates`

## Current Structure

```text
apps/collector/                         React Native Android collector/test app
data/audio/raw/                         local pulled audio session JSON/media, gitignored data source
data/audio/processed/                   local generated datasets and reports
data/audio/models/                      local trained model artifacts and evaluation reports
skills/pingis-audio-classification/     audio preprocess, replay, train, export workflow
skills/pingis-stroke-detection/         video stroke preprocess/train/export workflow
PROJECT_CONTEXT.md                      confirmed project facts and current model context
DECISIONS.md                            durable decision log
ITERATION_LOG.md                        detailed ML/data/build/device history
CODEX_TASK.md                           single active ticket
FOLLOWUPS.md                            out-of-scope issues and future tickets
```

## Known Validation Commands

| Command | Purpose | Last Known Result |
|---|---|---|
| `npm run validate` | Root WRLDS template, skill, and AWS metadata validation | `Passed 2026-06-02 after T0004 report` |
| `cd apps/collector && npx tsc --noEmit` | Collector TypeScript validation | `Passed 2026-06-01 after review pin cleanup` |
| `python skills/pingis-audio-classification/scripts/build_playing_retro_candidate_report.py` | T0004 candidate-centered playing-retro audio report | `Passed 2026-06-02 on Tomas 05-28/05-29 sessions` |
| `python -m py_compile <script>` | Targeted Python syntax check for changed scripts | `Run per ticket when Python changes` |

## Completed Tickets

| Ticket | Summary | Completed On | Notes |
|---|---|---|---|
| `T0001` | Adopt stricter WRLDS ticket workflow for this repo | `2026-06-02` | Added `REPO_CURRENT_STATE.md` and `FOLLOWUPS.md`, upgraded `CODEX_TASK.md`, and updated root workflow instructions |
| `T0002` | Refresh docs for audio/video-only scope | `2026-06-02` | Removed IMU/AirHive from active project context and marked sensor specs/workflows as legacy |
| `T0003` | Remove retired IMU/AirHive workflow files | `2026-06-02` | Deleted obsolete AirHive skills/specs and converted stroke skill to video-only workflow |
| `T0004` | Build candidate-centered `spel_retro_audio` report/dataset step | `2026-06-02` | Added deterministic report script and generated local row/summary outputs from saved app candidates plus replay peaks |

## Current Ticket

| Ticket | Goal | Status | Notes |
|---|---|---|---|
| `T0005` | Train and evaluate `spel_retro_audio` candidates from candidate-centered rows | `Ready` | Use T0004 row/summary outputs; no app export or APK until replay gates are explicitly accepted |

## Confirmed Next Tickets

| Ticket | Goal | Notes |
|---|---|---|
| `T0006` | Integrate `spel_retro_audio` behind a separate Review retro path | Only after T0005 passes replay gates |
| `T0007` | Revisit video-assisted FH/BH fusion after audio retro is stable | Keep video paused until audio is useful |

## Dependencies

| Dependency | Purpose | Notes |
|---|---|---|
| `apps/collector` | Android app and bundled model runtime | Do not change app artifacts unless ticket allows it |
| `skills/pingis-audio-classification` | Audio ML/replay workflow | Primary area for T0004/T0005 |
| `data/audio/raw` | Local reviewed session source data | Keep out of git; do not alter labels without approval |
| `data/audio/processed` | Generated datasets/reports | Local artifacts may be regenerated |

## Validation Status

- Build: `Not run for T0004; no app build needed`
- Tests: `npm run validate`, T0004 full report command, and targeted Python syntax checks passed on 2026-06-02
- Lint: `git diff --check` passed for T0004 on 2026-06-02
- Manual verification: `T0004 report generated 4317 row-level records and 21 summary rows; audio_session_2026-05-29_002 has close-event rows across under-80 ms, 80-119 ms, 120-179 ms, and 180-300 ms buckets`

## Known Issues Summary

- Current working tree already contains many pre-existing app/model/doc changes on `codex/video-stroke-test`; ticket work must avoid staging or reverting unrelated files.
- `studs_live`, `spel_retro_audio`, and `video_stroke_retro` need separate ticket scopes to avoid model/config bleed.
- IMU/AirHive workflow docs and skill scripts were removed from active scope; remaining app labels/code paths should be treated as legacy unless a ticket explicitly removes or renames them.
- `audio_session_2026-05-29_002` improves dense Tomas backhand replay with a local candidate but is not safe to promote to ordinary bounce.
- T0004 app-saved candidate rows count all saved model candidates, including hidden analysis-only candidates, so T0005 must decide whether training rows use all matchable peaks or only review-relevant/accepted peaks.

## Open Questions

- Should future ticket branches be created from current dirty `codex/video-stroke-test` or from a cleaned approved base?
- What exact replay thresholds define "good enough" for `spel_retro_audio` promotion into Review?
- Should legacy IMU app surfaces be removed immediately or left as hidden/internal historical code until the next UI cleanup?
