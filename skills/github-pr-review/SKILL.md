---
name: github-pr-review
description: Address GitHub PR review comments, requested changes, and issue comments by grouping feedback by theme, fixing straightforward issues, asking before architectural changes, and summarizing resolved and unresolved items.
---

# GitHub PR Review

Use this skill when a pull request has review comments, requested changes, or related issue comments.

## Workflow

1. Collect actionable comments and requested changes.
2. Group comments by theme, such as correctness, tests, UX, architecture, docs, or style.
3. Fix straightforward issues directly.
4. Ask before making architectural or product-scope changes.
5. Run relevant validation.
6. Summarize resolved and unresolved items.

## Review Discipline

- Preserve the intent of reviewers' comments.
- Avoid broad refactors unless the review specifically requires them.
- Do not resolve comments by hiding behavior changes.
- Call out comments that need product or architecture decisions.

## Output Summary

Summarize:

- Resolved review themes
- Unresolved questions
- Files changed
- Validation run
- Suggested reply or follow-up
