# Decisions

This file is the source of truth for meaningful project decisions. Use `ITERATION_LOG.md` for detailed model metrics, build hashes, data rounds, and device feedback.

## Decision Log

| Date | Decision | Rationale | Decided By | Impact | Revisit Trigger |
|---|---|---|---|---|---|
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
