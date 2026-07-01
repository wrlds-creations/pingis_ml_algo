# No-Main-Merge Cleanup Plan

Date: `2026-07-01`
Branch: `codex/t0057-fable-auto-improvement-loop`
Ticket: `T0105C-local-cleanup-commits`

## Corrected Policy

Do not merge the current branch to `main` before the current audio model/runtime works on device.

This applies even to changes that look "safe" in isolation, because the branch is currently a mixed experimental workspace. A clean `main` matters more than preserving partial work there early.

## What We Can Still Do Locally

We can clean the branch without merging it:

1. Keep a local inventory of dirty files and data.
2. Split work into local commits on this feature branch if Love explicitly asks.
3. Package ignored audio data outside git with a manifest.
4. Continue model/runtime validation on the phone.
5. Prepare a later PR only after the model/runtime story is measured and clear.

Love explicitly approved local cleanup commits in T0105C, and those commits have now been created on this feature branch. A local commit is not the same as merging to `main`.

## Hold All Main Merges Until Validation

Hold these out of `main`:

- `Bounce audio test` runtime and UI.
- T0075 JSON/runtime and any candidate model artifact.
- Native peak-gate audio path.
- `Fable-algoritm` continuous debug changes if they are tangled with unvalidated native audio changes.
- `Studs FH/BH LIVE` color tracker changes.
- `Fable data recorder` app changes unless we deliberately decide the recorder itself should ship independently.
- Raw/generated `data/` folders.

## Local Cleanup Sequence Completed

T0105C created these local commits on the feature branch:

1. `16f1e01 chore(audio): add fable reliability tooling`
2. `0fbee48 feat(collector): add bounce audio diagnostics`
3. `61d532e feat(collector): add live racket color tracker`
4. `docs: record cleanup handoff state`

These commits are for local branch hygiene and handoff only. They do not approve any app/runtime/model behavior for `main`.

## Main-Merge Gate

Before any PR/merge to `main`, require:

- A clear selected runtime path.
- Fresh Motorola validation counts for normal, high/slow, fast, messy/kid-style, speaking/counting, background sound, talking-only, racket-handling-only, and floor/table/other impact.
- Pulled debug JSON/WAV review for bad cases.
- A report showing recall and false counts by scenario.
- Agreement on what app entries are product, diagnostic, or hidden.
- TypeScript and Android validation for app/runtime changes.
- Root validation and diff check.

## Data Handoff

Ignored `data/` stays out of git. To share it with teammates:

- create a manifest with source folders, counts, label state, train/holdout/diagnostic role, and privacy note;
- package the exact WAV/JSON/review artifacts outside git;
- keep scripts in the repo so outputs can be regenerated from the shared data package.

## Current Recommendation

Continue improving and validating on this feature branch. Do not merge to `main` yet.

The next model-focused work remains: collect/review the T0102 boundary data, train/evaluate a new candidate only after that data exists, and keep T0074/T0073 safety gates as the promotion baseline.

## Explicit Boundary

T0105C did stage and commit local source/docs after explicit approval. It did not push, merge, delete, revert, move raw data, train/export a model, install an APK, or touch cloud/API/AWS resources.
