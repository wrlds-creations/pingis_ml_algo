# Decisions

This file is the source of truth for meaningful project decisions. Use `ITERATION_LOG.md` for detailed model metrics, build hashes, data rounds, and device feedback.

## Decision Log

| Date | Decision | Rationale | Decided By | Impact | Revisit Trigger |
|---|---|---|---|---|---|
| 2026-05-05 | Limit Playing review to FH hit, BH hit, and table bounce | Long match reviews need a compact label set focused on the events Love actually marks, while audio confidence can still filter auto-candidates | Love + Codex | Playing markers use `forehand_hit`, `backhand_hit`, or `table_bounce`; low-confidence auto-candidates can be stored as filtered and excluded from training | Playing review needs floor/noise labels again or a trained IMU stroke model can pre-label FH/BH safely |
| 2026-05-05 | Move `Playing` under `Audio plus IMU` | Free longer review-first capture is an audio+movement scenario, not a separate top-level data mode | Love + Codex | Startsida Data shows `Ljudinsamling` and `Audio plus IMU`; audio+IMU events carry high-level `scenario` and optional `bounce_context` metadata | Playing needs a fundamentally different product entrypoint or audio/video-only free recording becomes user-facing again |
| 2026-05-05 | Make Audio plus IMU pose calibration optional after table baseline | Raw IMU should still be collectible when FH/BH poses are not useful or would slow down collection | Love + Codex | Calibration stores `captured` vs `partial`; FH/BH poses are helper metadata, not playing stroke labels | A trained side classifier requires pose calibration as a hard prerequisite |
| 2026-05-05 | Limit guided Audio plus IMU hard negatives to audio-only collection | Table/floor/noise do not need IMU in v1, while racket-bounce side coverage does | Love + Codex | Audio plus IMU stores racket-bounce FH-side, BH-side, mixed, and playing; hard negatives stay in `Ljudinsamling` | IMU becomes useful for non-racket audio classes |
| 2026-05-05 | Separate audio-only, audio+IMU, and free-recording data modes | Audio-only cannot truthfully label FH/BH, while synced IMU/free recording need richer reviewed event labels | Love + Codex | Ljud-insamling collects generic sound classes; Studs audio + IMU keeps FH/BH prompts; Fri inspelning records long review-first takes with structured marker labels | A later product flow proves one unified collection mode is simpler without hurting label quality |
| 2026-05-05 | Hide legacy `Datainsamling` from the main Data section | The card leads to the older IMU-only `DataCollectionScreen`, while current user-facing collection is `Ljud-insamling` and `Studs audio + IMU` | Love + Codex | Legacy route/code remains available internally, but the startsida only shows relevant collection flows | A debug/developer menu is added or legacy IMU-only collection becomes user-facing again |
| 2026-05-04 | Use a visible and audible sync event for review video calibration | `Date.now()` between separate audio/video recorders is useful as a default but too coarse for easy manual review alignment | Love + Codex | Collection prompts a clap/tap at take start; Review detects the early audio sync spike and computes `video_sync_offset_ms` from the selected video frame | Video recording can safely include its own audio track, or measured drift shows a constant offset is not enough |
| 2026-05-04 | Use an immersive, current-session recording flow | Collection should feel like a camera flow and should not expose stale pending samples during new data collection | Love + Codex | Android hides system bars; collection uses countdown, fixed controls, and current-session pending review only | Device testing shows immersive mode or hidden old queue makes collection harder |
| 2026-05-04 | Target 150 Hz raw IMU collection when stable | 50 Hz was a baseline assumption, not a proven limit; higher raw resolution can be downsampled later | Love + Codex | Takes record `target_hz`, measured sample rate, interval stats, quality flag, and AirHive `sensor_ts` without blindly writing BLE config | BLE/device stability is worse at higher rates or AirHive confirms a lower hardware limit |
| 2026-05-04 | Make human-reviewed markers the primary contact-training truth | Legacy data can bootstrap and compare variants, but the algorithm must not create its own final labels | Love + Codex | Default contact preprocessing uses reviewed markers; legacy variants are opt-in via `human_reviewed`, `legacy_hybrid`, and `bootstrap` | Reviewed-only metrics fail to catch up after enough reviewed coverage |
| 2026-05-04 | Treat FH/BH racket-bounce as metadata, not audio classes | The contact sound should stay `racket_contact`; FH/BH coverage and side belong in prompts and IMU/debug metadata | Love + Codex | Collection presets split FH/BH/mixed for coverage, but audio labels remain binary/multiclass surface labels | Evidence shows FH/BH produces reliably different audio that improves count quality |
| 2026-05-04 | Separate bounce IMU from stroke IMU | Repeated racket bounces and real in-play forehand/backhand strokes are different ML problems | Love + Codex | Bounce IMU windows use reviewed audio markers; stroke IMU gets its own x-axis feature engineering | A later fusion model needs a shared representation after both datasets are stable |
| 2026-05-04 | Add live contact grouping after native onset and JS merge | Native retrigger suppresses raw onsets, while JS grouping suppresses duplicate qualified contacts from one physical contact | Love + Codex | Debug rows expose `group_id`, `best_candidate`, and `ignored_duplicate`; counted contacts are capped per group | Device validation shows grouping drops true fast contacts too aggressively |
| 2026-04-22 | Keep audio as the primary contact truth | Reviewed audio data is the strongest current signal for racket contact | Love + Codex | Audio model remains live count engine | Audio fails stable device validation after targeted data and live-merge fixes |
| 2026-04-22 | Do not replace audio with IMU | Bounce motion and real stroke motion are separate problems | Love + Codex | IMU work stays additive/removable | IMU evidence clearly outperforms audio in controlled tests |
| 2026-04-22 | Use one reviewed audio timeline as label truth | Keeps labels consistent across audio-only and synced audio+IMU collection | Love + Codex | Review markers supervise current audio and future synchronized data | Review UI becomes unreliable or a new sensor requires separate labels |
| 2026-04-22 | Use `0.5-1.0 s` spacing for base bounce collection | Cleaner clips are easier to review and train initially | Love + Codex | Fast double-bounce scenarios are deferred | Base detector becomes stable enough for fast-contact data |
| 2026-04-22 | Include legacy 4-class data in binary contact training via `all_legacy` | It improved binary contact behavior compared with stricter current-only datasets | Love + Codex | `all_legacy` is current best binary strategy | A reviewed-only dataset exceeds it across scenario-level metrics |
| 2026-04-23 | Keep training the 4-class model alongside the binary model | Table/floor/noise separation is useful for veto/debug and future behavior | Love + Codex | Two audio models remain active in app artifacts | 4-class model stops adding veto/debug value |
| 2026-04-24 | Pause IMU work and prioritize audio stability | Duplicate counts, missed angled contacts, and floor false positives are audio blockers | Love + Codex | New work should target audio data, live merge, and evaluation first | Audio meets agreed live validation thresholds |
| 2026-04-24 | Treat review video as labeling support, not model input | Video helps humans review but would create a new ML problem before audio is stable | Love + Codex | Training pipeline continues to use WAV/features and review markers | Audio review cannot be made trustworthy without video-derived labels |
| 2026-04-30 | Adopt the WRLDS template workflow in this repo | Future AI agents need project context, decision history, skills, and validation in-repo | Love + Codex | `PROJECT_CONTEXT.md`, `DECISIONS.md`, template skills, references, and validation scripts become part of the project | Template workflow blocks practical project work |

## Active Constraints

| Constraint | Source | Impact | Revisit Trigger |
|---|---|---|---|
| Do not push directly to `main` | WRLDS workflow | Work must happen on a feature branch and merge through PR/review | Explicit repository policy change |
| Do not commit unless explicitly asked | WRLDS workflow | Agents can edit locally but need user intent before committing | Explicit user request to commit/push |
| Update `ITERATION_LOG.md` after model/data/build feedback | Project workflow | ML state must survive handoff between AI agents | A better model registry replaces it |
| Keep raw data out of git | Data size/privacy | `/data/` and large local exports stay local | A proper data storage/sync system is introduced |

## Deferred Decisions

| Decision | Why Deferred | Needed By | Owner |
|---|---|---|---|
| Whether to use AWS/backend sync | Current loop is local device-to-computer | Before multi-user or cloud data collection | Love + WRLDS |
| Whether to promote IMU fusion into live count | Audio is not stable enough yet | After audio live validation passes | Love + Codex |
| Whether to change model family beyond RandomForest | Current blockers are data coverage and live event logic first | If targeted data + merge fixes do not solve live failures | Codex |

## Reversed Decisions

| Date | Reversed Decision | Replacement Decision | Rationale | Decided By |
|---|---|---|---|---|
| TBD | TBD | TBD | TBD | TBD |
