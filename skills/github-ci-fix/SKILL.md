---
name: github-ci-fix
description: Debug and fix failing CI, GitHub Actions failures, broken tests, lint failures, or build failures by inspecting logs, identifying root cause, making the smallest safe fix, running relevant local validation, and summarizing what failed, what changed, and what remains.
---

# GitHub CI Fix

Use this skill when CI, GitHub Actions, tests, lint, or builds fail.

## Workflow

1. Identify the failing workflow, job, command, and first meaningful error.
2. Inspect logs before editing.
3. Reproduce locally when practical.
4. Identify the root cause instead of patching symptoms.
5. Make the smallest safe fix.
6. Run the narrowest relevant validation first, then broader validation if needed.
7. Leave unrelated failures untouched and call them out.

## Output Summary

Summarize:

- What failed
- Root cause
- Files changed
- Validation run
- Remaining failures or risks
