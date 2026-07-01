# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0103-boundary-label-candidate-phone-gate`

## Branch

`codex/t0057-fable-auto-improvement-loop`

## Status

`Completed`

## Goal

Ingest Love's remaining saved T0102 boundary labels, train and evaluate an improved bounce-audio candidate against the new boundary positives/negatives plus existing safety sets, and prepare a phone-testable app path only if offline metrics are strong enough to justify Motorola testing.

## Dependencies

- T0102E prepared nine positive review pages for fresh boundary recordings.
- T0102F ingested the first three `Far/soft racket bounce + background` pages (`8770`-`8772`) and found only `66/90` labels covered by the current peak-candidate path.
- Love confirmed the remaining saved review pages and expected counts:
  - `8773`: expected `30`
  - `8774`: expected `29`
  - `8775`: expected `30`
  - `8776`: expected `30`
  - `8777`: expected `30`
  - `8778`: expected `30`
- Raw/generated data under `data/` remains ignored and must not be committed.

## Allowed Areas

- `CODEX_TASK.md`
- `REPO_CURRENT_STATE.md`
- `PROJECT_CONTEXT.md` if confirmed project facts change
- `DECISIONS.md` if a meaningful decision changes
- `ITERATION_LOG.md`
- `skills/pingis-audio-classification/scripts/noise_robust/`
- ignored local review/ingest/evaluation/training artifacts under `data/audio/`
- validation/status commands
- guarded app test-path files under `apps/collector/` only after offline validation passes and the change remains isolated from current production/Fable behavior

## Do Not Touch

- Do not merge to `main`.
- Do not push.
- Do not delete local or device data.
- Do not revert tracked or user changes.
- Do not replace current production Fable behavior unless a later ticket explicitly promotes the candidate.
- Do not move raw/generated data into git.
- Do not install an APK until offline metrics justify a fresh phone test.
- Do not touch camera, Roboflow, cloud/API credentials, backend resources, or AWS resources.

## Requirements

- Ingest all nine T0102E positive review pages, honoring Love's `8774` expected-count override of `29`.
- Verify saved labels, expected counts, draft/deleted/manual behavior, and nearest candidate coverage.
- Assemble the new boundary positives and fresh expected-zero negatives into a reproducible ignored local evaluation set.
- Sweep candidate-generation and classifier/veto options with metrics split by scenario.
- Compare against the current `Bounce audio test` candidate behavior and prior safety sets where available.
- Only export/wire a phone-test candidate if the candidate materially improves soft/noisy positive recall without unacceptable false counts on talking/background/handling/impact negatives.
- Update source-of-truth docs/logs with the ingest, metrics, decision, and next recommended action.

## Non-Goals

- No merge to `main`.
- No push.
- No raw/generated data commit.
- No production model replacement.
- No camera/racket-side work.
- No cloud/API/AWS work.
- No app install if offline validation is still clearly weak.

## Acceptance Criteria

- All confirmed saved labels are ingested or any blocking label issue is reported with exact session/file details.
- A report summarizes target positive coverage, expected-zero false counts, app-current baseline, and best candidate behavior.
- If a candidate is worthwhile for phone testing, the app has a separate guarded test path and can be installed; otherwise the report clearly says why it is not worth testing yet.
- Source-of-truth docs are updated with validation commands and the recommended next step.

## Completion Notes

Completed. All nine T0102E/T0102F positive review pages were ingested with Love's `8774` expected-count override (`29`), producing `269/269` reviewed racket labels. Candidate coverage with the current `peak_fast_balanced` gate is `242/269` within `140 ms` and `250 ms`; the main remaining coverage weakness is still far/soft background bounces.

The current T0075-style `Bounce audio test` baseline on the boundary pack was `209/269` positives with `33` boundary negative false counts. The selected T0103 candidate, `extra_leaf4_t0103_base_t0075_boundary_recall_safety_thr0p575_dedupe180_vetonoveto`, improved to `225/269` positives with `2` boundary negative false counts, while retaining Round A `944/960` positives with `4` hard-negative false counts. This is strong enough for guarded Motorola testing, but not production promotion.

Exported `apps/collector/src/models/fable_extra_trees_candidate_t0103.json` and wired only the separate `Bounce audio test` entry to it. Defaults now come from the T0103 model metadata: threshold `0.575`, smart dedupe `180 ms`, and Fable-noise veto disabled. Existing `Fable-algoritm`, `Studsdetektor`, `Studs FH/BH LIVE`, production Fable JSON, camera tracker, and normal app flows remain unchanged.

## Validation

- `python -m py_compile skills\pingis-audio-classification\scripts\noise_robust\ingest_t0102_boundary_review_labels.py`
- `python -m py_compile skills\pingis-audio-classification\scripts\noise_robust\evaluate_t0103_boundary_candidate_loop.py`
- `python skills\pingis-audio-classification\scripts\noise_robust\ingest_t0102_boundary_review_labels.py --out-dir data\audio\models\evaluations\t0103_boundary_label_candidate_phone_gate\full_label_ingest_2026_07_01 --expected-override fable_training_audio_2026-07-01T10-39-26-712Z=29`
- `python skills\pingis-audio-classification\scripts\noise_robust\evaluate_t0103_boundary_candidate_loop.py --reuse-existing --export-app-model`
- `cd apps/collector && npx tsc --noEmit`
- `cd apps/collector/android && .\gradlew.bat :app:compileDebugKotlin -PreactNativeArchitectures=arm64-v8a`
- `npm run validate`
- `git diff --check`
- `.\install-android-dev.ps1` installed and launched the debug APK on Motorola `ZY22KSPF5W`
