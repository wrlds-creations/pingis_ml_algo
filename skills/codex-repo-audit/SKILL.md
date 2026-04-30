---
name: codex-repo-audit
description: Audit a repository for Codex-readiness by checking AGENTS.md, PROJECT_CONTEXT.md, DECISIONS.md, command documentation, validation commands, AWS_RESOURCES.md when AWS is used, skills for reusable workflows, README setup instructions, documented assumptions, and open questions.
---

# Codex Repo Audit

Use this skill to evaluate whether a repository is ready for WRLDS Codex work.

## Checklist

Check whether:

- `AGENTS.md` exists.
- `PROJECT_CONTEXT.md` exists.
- `DECISIONS.md` exists.
- Package scripts or project commands are documented.
- Validation commands are known.
- `AWS_RESOURCES.md` exists if AWS is used.
- `skills/` exists if the project has reusable workflows.
- `README.md` has setup instructions.
- Obvious project assumptions are documented.
- Missing project information is listed as open questions.

## Output

Report:

- Ready items
- Gaps
- Risks
- Recommended file updates
- Questions that block reliable Codex work

Do not make broad edits unless the user asks for the audit findings to be applied.
