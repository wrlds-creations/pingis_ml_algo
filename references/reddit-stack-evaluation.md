# Codex Stack Evaluation

This summarizes guidance from the Codex stack discussion for the WRLDS template.

## Use Now

- Official Codex app or CLI
- `AGENTS.md`
- `PROJECT_CONTEXT.md`
- `DECISIONS.md`
- Small focused skills
- `github-ci-fix` style workflow
- `github-pr-review` style workflow
- `release-notes` workflow
- AWS governance skill

## Evaluate Later

- `agent-orchestrator` for parallel worktrees
- `graphify` for large unfamiliar repositories
- `ccusage` for token monitoring
- Composio or connect-style tools for controlled external app access
- `cc-switch` only if switching between coding agents matters
- `caveman` only as optional low-token mode, not default

## WRLDS Position

Start with source-of-truth files, focused skills, and deterministic validation scripts. Add orchestration or external-control tooling only when the workflow has proven repetition and clear value.
