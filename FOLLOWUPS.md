# Followups

Use this file for known issues, deferred improvements, and out-of-scope findings. Codex should add items here instead of fixing unrelated work during a scoped ticket. Every followup should point back to the source ticket or decision that exposed it.

## Open Followups

| ID | Source Ticket | Type | Description | Priority | Owner | Status |
|---|---|---|---|---|---|---|
| `F0001` | `T0001` | `Workflow` | Decide whether to create future ticket branches from the current dirty `codex/video-stroke-test` branch or first establish a clean approved base. | `Medium` | `Love + Codex` | `Open` |
| `F0002` | `T0001` | `Planning` | Define explicit replay pass/fail gates for `spel_retro_audio` before any app model export. | `High` | `Love + Codex` | `Open` |
| `F0003` | `T0002/T0003` | `Cleanup` | Remove, hide, or rename legacy `Audio plus IMU` app surfaces and any sensor-specific user-facing text; docs and skills are cleaned, but app code cleanup is still separate. | `High` | `Love + Codex` | `Open` |
| `F0006` | `T0006` | `Model` | Build true multi-window and non-leaky candidate-context features for `spel_retro_audio`; one-window weighting only improved racket recall to 0.623 and is not enough for app integration. | `High` | `Codex` | `Open` |

## Resolved Followups

| ID | Resolved In | Resolution | Date |
|---|---|---|---|
| `F0004` | `T0005` | T0005 trains from all matchable saved app candidate peaks plus manually reviewed missed markers; replay-generated peaks stay diagnostic and are not multiplied into training rows. | `2026-06-02` |
| `F0005` | `T0006` | T0006 improved holdout racket recall from 0.604 to 0.623 with `safe_racket_weighted`, but kept it local because the gain is too small for app integration. | `2026-06-02` |
