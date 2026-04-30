---
name: project-intake
description: Initialize or refresh WRLDS project context by reading PROJECT_CONTEXT.md, identifying missing project facts, asking the smallest blocking question set, updating PROJECT_CONTEXT.md and DECISIONS.md after confirmation, and producing a short setup plan. Use at the start of a new project, milestone, repository audit, or when project context is incomplete.
---

# Project Intake

Use this skill to turn an under-specified WRLDS project into a clear working context without overwhelming the project owner with the full question bank.

## Workflow

1. Read `PROJECT_CONTEXT.md`.
2. Read `DECISIONS.md`.
3. Check the root question bank at `../../references/project-intake-questions.md` only for the categories relevant to the current phase.
4. Identify missing fields that block the current task or phase.
5. Ask only the smallest useful set of focused questions.
6. After the user confirms answers, update `PROJECT_CONTEXT.md`.
7. Add entries to `DECISIONS.md` only when the user makes a meaningful decision.
8. Produce a short setup plan with immediate next steps, validation, and open questions.

## Question Discipline

- Do not ask the entire question bank at once.
- Prefer 1 to 5 questions per pass.
- Ask for facts before opinions when facts unblock implementation.
- Offer options with tradeoffs when the project needs a decision.
- Leave unknowns as `TBD` until confirmed.

## Update Rules

- Write confirmed project facts into the most specific `PROJECT_CONTEXT.md` section.
- Add unresolved blockers to `PROJECT_CONTEXT.md` under `Open Questions`.
- Log decisions in `DECISIONS.md` with rationale, impact, and revisit trigger.
- Do not invent client, billing, AWS, data, or platform details.

## Setup Plan Output

After context is filled enough for the phase, summarize:

- Confirmed project identity and phase
- Known stack and environments
- Decisions made
- Remaining blockers
- Recommended next implementation step
