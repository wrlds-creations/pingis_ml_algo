# Repo Current State

Use this file as the living snapshot of what actually exists in the repository. Update it after completed tickets, audits, meaningful dependency changes, validation changes, workflow changes, model/data/build state changes, or next-ticket changes.

## Snapshot

- Date: `2026-06-02`
- Current branch: `codex/t0005-train-playing-retro-audio-candidates`
- Current phase: `Audio-first playing-retro stabilization before app integration`
- Current status: `T0005 local spel_retro_audio candidate trained and evaluated; next active ticket is holdout error analysis and focused improvement`
- Current ticket: `T0006`
- Last completed ticket: `T0005`
- Recommended next ticket: `T0006-improve-playing-retro-audio-candidate`

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
| `python skills/pingis-audio-classification/scripts/train_playing_retro_audio.py` | T0005 local `spel_retro_audio` candidate train/eval | `Passed 2026-06-02; 4,028 rows, holdout accuracy 0.759` |
| `python -m py_compile <script>` | Targeted Python syntax check for changed scripts | `Run per ticket when Python changes` |

## Completed Tickets

| Ticket | Summary | Completed On | Notes |
|---|---|---|---|
| `T0001` | Adopt stricter WRLDS ticket workflow for this repo | `2026-06-02` | Added `REPO_CURRENT_STATE.md` and `FOLLOWUPS.md`, upgraded `CODEX_TASK.md`, and updated root workflow instructions |
| `T0002` | Refresh docs for audio/video-only scope | `2026-06-02` | Removed IMU/AirHive from active project context and marked sensor specs/workflows as legacy |
| `T0003` | Remove retired IMU/AirHive workflow files | `2026-06-02` | Deleted obsolete AirHive skills/specs and converted stroke skill to video-only workflow |
| `T0004` | Build candidate-centered `spel_retro_audio` report/dataset step | `2026-06-02` | Added deterministic report script and generated local row/summary outputs from saved app candidates plus replay peaks |
| `T0005` | Train and evaluate first local `spel_retro_audio` candidate | `2026-06-02` | Trained `playing_retro_audio_rf_v2026_06_02_app_candidates_100_200` from all matchable saved app candidates plus manually missed reviewed markers; no app export/APK |

## Current Ticket

| Ticket | Goal | Status | Notes |
|---|---|---|---|
| `T0006` | Improve `spel_retro_audio` candidate by holdout error analysis and focused variants | `Ready` | T0005 improved old app holdout accuracy but racket recall is still too low for app integration |

## Confirmed Next Tickets

| Ticket | Goal | Notes |
|---|---|---|
| `T0007` | Integrate `spel_retro_audio` behind a separate Review retro path | Only after candidate recall/false-positive gates are accepted |
| `T0008` | Revisit video-assisted FH/BH fusion after audio retro is stable | Keep video paused until audio is useful |

## Dependencies

| Dependency | Purpose | Notes |
|---|---|---|
| `apps/collector` | Android app and bundled model runtime | Do not change app artifacts unless ticket allows it |
| `skills/pingis-audio-classification` | Audio ML/replay workflow | Primary area for T0004-T0006 |
| `data/audio/raw` | Local reviewed session source data | Keep out of git; do not alter labels without approval |
| `data/audio/processed` | Generated datasets/reports | Local artifacts may be regenerated |

## Validation Status

- Build: `Not run for T0005; no app build needed`
- Tests: `npm run validate`, T0004 full report command, T0005 train/eval command, and targeted Python syntax checks passed on 2026-06-02
- Lint: `git diff --check` passed for T0005 on 2026-06-02
- Manual verification: `T0005 report generated 4,028 candidate-centered training rows across 16 playing sessions. Holdout audio_session_2026-05-29_002: candidate accuracy 0.759 versus old app prediction 0.682; racket recall 0.604, table recall 0.924, non-target recall 0.625. Ordinary regression slice: 10,353 rows at 0.714, reported separately and not promoted to studs_live.`

## Known Issues Summary

- Current working tree already contains many pre-existing app/model/doc changes on `codex/video-stroke-test`; ticket work must avoid staging or reverting unrelated files.
- `studs_live`, `spel_retro_audio`, and `video_stroke_retro` need separate ticket scopes to avoid model/config bleed.
- IMU/AirHive workflow docs and skill scripts were removed from active scope; remaining app labels/code paths should be treated as legacy unless a ticket explicitly removes or renames them.
- `audio_session_2026-05-29_002` improves over old app prediction with the T0005 local candidate but racket recall remains too low for app integration.
- T0005 trains from all matchable saved app candidates plus manually missed reviewed markers; replay-generated T0004 candidates remain diagnostic, not training rows.

## Open Questions

- Should future ticket branches be created from current dirty `codex/video-stroke-test` or from a cleaned approved base?
- What exact recall/false-positive replay thresholds define "good enough" for `spel_retro_audio` promotion into Review?
- Should legacy IMU app surfaces be removed immediately or left as hidden/internal historical code until the next UI cleanup?
