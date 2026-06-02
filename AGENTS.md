# WRLDS Codex Workflow

This repo uses the WRLDS Codex workflow. Keep it practical: understand the project, work in scoped tickets, document decisions, and handle AWS carefully.

## Required Reading Order

Start every implementation ticket by reading:

1. Read `PROJECT_CONTEXT.md`.
2. Read `DECISIONS.md`.
3. Read `REPO_CURRENT_STATE.md`.
4. Read the active ticket in `CODEX_TASK.md`.
5. Check local `skills/` for a matching domain or workflow skill.
6. For audio, bounce detection, ML model, collector, review, or video stroke work, read `ITERATION_LOG.md`.

Use project files as the source of truth. Do not rely on chat history when the repo has confirmed context.

## Source Of Truth

- Use `PROJECT_CONTEXT.md` for confirmed project facts, constraints, commands, environments, and open questions.
- Use `DECISIONS.md` for meaningful decisions, rationale, impact, and revisit triggers.
- Use `REPO_CURRENT_STATE.md` for the latest repository snapshot, completed tickets, validation status, and next recommended ticket.
- Use `CODEX_TASK.md` for the single active ticket, allowed areas, non-goals, and acceptance criteria.
- Use `FOLLOWUPS.md` for out-of-scope issues, deferred improvements, and future tickets.
- Use `ITERATION_LOG.md` for detailed model metrics, data pulls, app builds, device feedback, and ML handoff history.
- Use `AWS_RESOURCES.md` for AWS resources created, changed, deleted, or materially affecting cost, security, data, deployment, or ownership.

## Working Rules

- Ask focused questions only when missing information blocks the task.
- Work on one active ticket only.
- Do not broaden scope beyond the active ticket.
- Do not touch files outside `CODEX_TASK.md` allowed areas unless the user explicitly approves it.
- Work in a feature branch. Do not push directly to `main`.
- Do not commit unless explicitly asked.
- Prefer small, reviewable diffs.
- When confirmed project facts change, update `PROJECT_CONTEXT.md`.
- When a meaningful decision is made, update `DECISIONS.md`.
- When a ticket is completed, update `REPO_CURRENT_STATE.md` if repo structure, commands, dependencies, validation status, or next steps changed.
- When Codex notices out-of-scope work, record it in `FOLLOWUPS.md` instead of fixing it automatically.
- When a model, data pull, app build, or device feedback round changes state, update `ITERATION_LOG.md`.
- When a workflow becomes reusable, suggest creating or updating a skill.

## Branch And Commit Workflow

- Use one dedicated branch per ticket unless the user explicitly says otherwise.
- Name ticket branches with the `codex/` prefix, for example `codex/t0001-short-description`.
- Create each ticket branch from the current approved base, preferably `main`.
- Stage and commit only files that belong to the current ticket.
- Leave unrelated local assets, deliverables, and user changes unstaged unless the ticket explicitly includes them.
- Push ticket branches only when explicitly requested.

## AWS Work

Before creating, changing, deploying, or deleting AWS resources:

1. Read `AWS_RESOURCES.md`.
2. Use `skills/aws-project-infrastructure/`.
3. Confirm client, project, environment, owner, repository, tags, data classification, exportability, and cost center.
4. Update `AWS_RESOURCES.md` when AWS changes.

## Handoff

Summarize changed files, commands run, validation performed, manual verification, docs updated, risks or open questions, follow-up tickets, and the recommended next step.
