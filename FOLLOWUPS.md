# Followups

Use this file for known issues, deferred improvements, and out-of-scope findings. Codex should add items here instead of fixing unrelated work during a scoped ticket. Every followup should point back to the source ticket or decision that exposed it.

## Open Followups

| ID | Source Ticket | Type | Description | Priority | Owner | Status |
|---|---|---|---|---|---|---|
| `F0001` | `T0001` | `Workflow` | Decide whether to create future ticket branches from the current dirty `codex/video-stroke-test` branch or first establish a clean approved base. | `Medium` | `Love + Codex` | `Open` |
| `F0013` | `T0025` | `Model/Training` | Complete the 2026-06-04 playing-retro improvement sequence: T0028 export/build/install after T0027 selected T0026 safely, and T0029 candidate/peak recovery only if later reviewed clips show true candidate-generation gaps remain material. T0025 audit found classification/threshold misses, T0026 trained the local 06-04 candidate, and T0027 replay selected racket threshold `0.0`, table threshold `0.45`, and 80 ms same-label dedupe. | `High` | `Codex` | `T0028 next` |

## Resolved Followups

| ID | Resolved In | Resolution | Date |
|---|---|---|---|
| `F0004` | `T0005` | T0005 trains from all matchable saved app candidate peaks plus manually reviewed missed markers; replay-generated peaks stay diagnostic and are not multiplied into training rows. | `2026-06-02` |
| `F0005` | `T0006` | T0006 improved holdout racket recall from 0.604 to 0.623 with `safe_racket_weighted`, but kept it local because the gain is too small for app integration. | `2026-06-02` |
| `F0006` | `T0007` | T0007 added true tight/normal/wide WAV windows plus non-leaky candidate-context features and selected local candidate `playing_retro_audio_rf_v2026_06_02_multi_window_context`; app integration remains deferred until cross-session validation. | `2026-06-02` |
| `F0007` | `T0008` | T0008 validated the T0007 multi-window/context candidate across `audio_session_2026-05-28_002`, `audio_session_2026-05-29_001`, and `audio_session_2026-05-29_002`; selected variant passes the local cross-session gate and can proceed to a separate Review retro integration ticket. | `2026-06-02` |
| `F0003` | `T0012C` | Removed legacy `Ljudinsamling` and `Audio plus IMU` setup/router entry points from the installed current-worktree APK, changed the old generic review title to `Ljudreview`, and replaced remaining bundled `Audio plus IMU` helper text with neutral legacy wording. | `2026-06-02` |
| `F0002` | `T0015` | T0015 defined the T0016 pass gate: 0 wrong-class near missed truth, 0 duplicate near already matched truth, 0 visible false positives, and at least 6 correct recovered missed truths. | `2026-06-03` |
| `F0008` | `T0015` | T0015 swept racket/table confidence and nearest-saved-gap gates separately. It kept racket conservative because no safe tested gate recovered extra racket hits; table remains the only visible recovery gain for the first APK test. | `2026-06-03` |
| `F0009` | `T0020` | T0020 audited `audio_session_2026-06-03_005` and added 80 ms same-label-only duplicate suppression for generated playing-retro review markers. Replay changed visible target predictions from 204 to 197 and false positives from 11 to 4 while keeping TP 186, wrong-class 7, and missed 19 unchanged. | `2026-06-03` |
| `F0010` | `T0020` | T0020 confirmed the blue outline is the linked audio/motion marker state, not a deleted or locked marker, and added that explanation to the playing-retro model info dialog in the installed APK. | `2026-06-03` |
| `F0011` | `T0021` | T0021 produced a row-level miss/correction analysis for `audio_session_2026-06-03_005`: 13/20 manual additions were near candidates classified as `non_target`, 2 were wrong-class, 2 were true candidate-generation gaps, 1 was hidden by recovery gate, 1 was timing/dense sequence, and 1 was visible but unlinked/deleted. | `2026-06-03` |
| `F0012` | `T0024` | T0022 retrained `spel_retro_audio` from `audio_session_2026-06-03_005` plus historical playing-dense data, T0023 replay selected racket threshold `0.0` and table threshold `0.5`, and T0024 exported/built/installed the improved review-only model without changing `studs_live`, `audio_model.json`, or `audio_contact_model.json`. | `2026-06-04` |
