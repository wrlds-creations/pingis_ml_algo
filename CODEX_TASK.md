# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0130-pcen-hybrid-post-filter-audit`

## Branch

`codex/t0057-fable-auto-improvement-loop`

## Status

`Completed`

## Goal

Evaluate whether PCEN-derived audio features can create a safer post-Hybrid hard-negative filter for the STIGA `Hybrid 2.0` QA path. Keep the existing STIGA `Hybrid` detector unchanged, and do not port another live veto until the offline evidence is strong enough to justify phone testing.

## Dependencies

- T0129 trained a learned post-Hybrid hard-negative veto, but live phone testing showed it was too strict: Hybrid 2.0 saw candidates yet dropped most clear bounces.
- The candidate row dataset already contains noise-robust/PCEN-style columns such as `feat_nr_pcen_max`, `feat_nr_pcen_mean`, and `feat_nr_pcen_std`.
- Plain STIGA `Hybrid` is currently the recall baseline because it worked best across the user's Android/iPhone checks.

## Allowed Areas

- `CODEX_TASK.md`
- `PROJECT_CONTEXT.md`
- `REPO_CURRENT_STATE.md`
- `ITERATION_LOG.md`
- `DECISIONS.md`
- `skills/pingis-audio-classification/scripts/noise_robust/`
- `data/audio/models/evaluations/t0130_pcen_hybrid_post_filter/` as ignored local evaluation output

## Do Not Touch

- Do not delete raw/generated `data/`.
- Do not merge to `main`.
- Do not replace the current working STIGA `Hybrid` mode.
- Do not port a new veto to `stiga-app-v2` unless the offline result is clearly worth a phone test.
- Do not add a single-feature hard-coded veto.
- Do not change camera/rubber-side behavior.
- Do not use cloud APIs or AWS.

## Requirements

- Train/evaluate only on rows that the existing STIGA `Hybrid` stack would accept.
- Compare at least:
  - T0129-style RF/probability feature baseline;
  - PCEN-only features;
  - RF/probability features plus PCEN features;
  - broader noise-robust PCEN/transient/spectral feature groups if present in the row CSV.
- Use grouped out-of-fold validation by session/domain.
- Report plain Hybrid TP/FP and candidate post-filter TP/FP after app-style dedupe.
- Prefer a high-recall veto setting: almost never reject true bounces offline, even if it removes fewer false positives.
- Export a local JSON only if a PCEN variant is worth future app QA.

## Non-Goals

- No new peak picker or candidate-generation change.
- No replacement of existing Hybrid.
- No Android/iOS/TestFlight build.
- No model promotion into production/default app behavior.

## Acceptance Criteria

- A reproducible PCEN post-filter audit script exists.
- The script writes a report comparing PCEN variants against plain Hybrid and the T0129 feature baseline.
- The report states whether PCEN is worth porting to STIGA for another Hybrid 2.0 phone test.
- Source-of-truth docs are updated with the conclusion.

## Completion Notes

- Added `train_t0130_pcen_hybrid_post_filter.py`.
- Ran grouped OOF sweeps on Hybrid-accepted rows for:
  - T0129-style RF/probability baseline;
  - PCEN-only;
  - PCEN context/noise-robust features;
  - RF/probability plus PCEN;
  - RF/probability plus PCEN plus transient/spectral columns.
- Plain Hybrid baseline after dedupe: `1518` TP / `791` FP.
- PCEN-only kept all true positives but removed only `9` false positives.
- PCEN context kept all true positives but removed only `5` false positives.
- Best offline row was `rf_prob_plus_pcen_transient_spectral`: `1515` TP / `190` FP, losing `3` true positives and removing `601` false positives.
- That best row depends on `td_*` and `sp_*` transient/spectral columns that the current STIGA runtime does not compute, so the exported ignored JSON is not a safe drop-in app port.
- Recommendation: do not port T0130 directly. Next ticket should either implement/parity-test the missing transient/spectral feature extractor in STIGA, or test a gentler already-portable RF/probability post-filter as a guarded comparison.

## Validation

- `python -m py_compile skills\pingis-audio-classification\scripts\noise_robust\train_t0130_pcen_hybrid_post_filter.py`
- `python skills\pingis-audio-classification\scripts\noise_robust\train_t0130_pcen_hybrid_post_filter.py`
- `git diff --check`
- `npm run validate`
