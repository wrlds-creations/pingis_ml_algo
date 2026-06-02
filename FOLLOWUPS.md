# Followups

Use this file for known issues, deferred improvements, and out-of-scope findings. Codex should add items here instead of fixing unrelated work during a scoped ticket. Every followup should point back to the source ticket or decision that exposed it.

## Open Followups

| ID | Source Ticket | Type | Description | Priority | Owner | Status |
|---|---|---|---|---|---|---|
| `F0001` | `T0001` | `Workflow` | Decide whether to create future ticket branches from the current dirty `codex/video-stroke-test` branch or first establish a clean approved base. | `Medium` | `Love + Codex` | `Open` |
| `F0002` | `T0001` | `Planning` | Define explicit replay pass/fail gates for `spel_retro_audio` before any app model export. | `High` | `Love + Codex` | `Open` |
| `F0003` | `T0002/T0003` | `Cleanup` | Remove, hide, or rename legacy `Audio plus IMU` app surfaces and any sensor-specific user-facing text; docs and skills are cleaned, but app code cleanup is still separate. | `High` | `Love + Codex` | `Open` |
| `F0007` | `T0007` | `Model` | Run cross-session or leave-one-session-out validation for the T0007 multi-window/context candidate before app integration, especially across `audio_session_2026-05-28_002`, `audio_session_2026-05-29_001`, and `audio_session_2026-05-29_002`. | `High` | `Codex` | `Open` |

## Resolved Followups

| ID | Resolved In | Resolution | Date |
|---|---|---|---|
| `F0004` | `T0005` | T0005 trains from all matchable saved app candidate peaks plus manually reviewed missed markers; replay-generated peaks stay diagnostic and are not multiplied into training rows. | `2026-06-02` |
| `F0005` | `T0006` | T0006 improved holdout racket recall from 0.604 to 0.623 with `safe_racket_weighted`, but kept it local because the gain is too small for app integration. | `2026-06-02` |
| `F0006` | `T0007` | T0007 added true tight/normal/wide WAV windows plus non-leaky candidate-context features and selected local candidate `playing_retro_audio_rf_v2026_06_02_multi_window_context`; app integration remains deferred until cross-session validation. | `2026-06-02` |
