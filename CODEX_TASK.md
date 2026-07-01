# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0105C-local-cleanup-commits`

## Branch

`codex/t0057-fable-auto-improvement-loop`

## Status

`Completed`

## Goal

Tidy the current dirty feature branch by validating, staging, and committing accumulated source changes into sensible local commits, without merging to `main` or committing raw/generated data.

## Dependencies

- T0105 created `WORKTREE_CLEANUP_PLAN.md` and grouped the dirty tree.
- T0105B corrected the policy: do not merge this branch to `main` before validating the current model/runtime.
- Love explicitly asked Codex to set a goal, tidy the worktree, stage, and commit, using Codex judgment.
- Raw/generated data under `data/` remains ignored and must not be committed.

## Allowed Areas

- `CODEX_TASK.md`
- `WORKTREE_CLEANUP_PLAN.md`
- `SAFE_MERGE_PREP.md`
- `REPO_CURRENT_STATE.md`
- `PROJECT_CONTEXT.md`
- `DECISIONS.md`
- `ITERATION_LOG.md`
- app/source files already dirty in this branch
- audio scripts already dirty/untracked in this branch
- validation commands
- git staging and local commits

## Do Not Touch

- Do not merge or push.
- Do not delete local or device data.
- Do not revert tracked changes.
- Do not broaden app/model behavior beyond the existing dirty worktree.
- Do not train/export a model or install an APK.
- Do not move raw/generated data into git.
- Do not touch Roboflow/cloud/API credentials, backend resources, or AWS resources.

## Requirements

- Validate Python audio scripts.
- Validate Collector TypeScript and Android Kotlin/native code if practical.
- Stage and commit source changes in logical local commits.
- Keep ignored `data/` out of git.
- Update source-of-truth docs after commits.
- Stop only if a real user decision is needed.

## Non-Goals

- No merge to `main`.
- No push.
- No raw/generated data commit.
- No raw-data sharing solution beyond documenting the policy.
- No judgment that every existing untracked file should be committed.

## Acceptance Criteria

- Dirty source files are grouped into local commits.
- Validation results are recorded.
- `git status --short` is materially cleaner, with raw/generated ignored data still untracked/ignored.
- Final response lists commits and any remaining dirt.

## Completion Notes

- Created local source commits on feature branch `codex/t0057-fable-auto-improvement-loop`:
  - `16f1e01 chore(audio): add fable reliability tooling`
  - `0fbee48 feat(collector): add bounce audio diagnostics`
  - `61d532e feat(collector): add live racket color tracker`
- Kept ignored/raw/generated `data/` out of git.
- Did not merge, push, delete local/device data, train/export a model, install an APK, or touch AWS/cloud/API resources.
- Updated source-of-truth docs and cleanup plans to record that the branch is locally tidied but still not approved for `main`.

## Validation

- `python -m py_compile` on the changed/untracked audio `noise_robust` Python scripts passed before committing script tooling.
- `cd apps\collector && npx tsc --noEmit` passed.
- `cd apps\collector\android && .\gradlew.bat :app:compileDebugKotlin` passed.
- `npm run validate` passed before and after the final docs commit.
- `git diff --check` passed, with existing Windows LF-to-CRLF warnings only before the final docs commit and clean output after it.
