# WRLDS Codex Agreement

This repo uses the WRLDS Codex workflow. Keep it practical: understand the project, work in branches, document decisions, and handle AWS carefully.

## Before Coding

1. Read `PROJECT_CONTEXT.md`.
2. Read `DECISIONS.md`.
3. Check local `skills/` for a matching workflow.
4. For audio, bounce detection, ML model, collector, review, or AirHive IMU work, read `ITERATION_LOG.md`.

Use project files as the source of truth. Do not rely on chat history when the repo has confirmed context.

## Working Rules

- Ask focused questions only when missing information blocks the task.
- Work in a feature branch. Do not push directly to `main`.
- Do not commit unless explicitly asked.
- Prefer small, reviewable diffs.
- When confirmed project facts change, update `PROJECT_CONTEXT.md`.
- When a meaningful decision is made, update `DECISIONS.md`.
- When a model, data pull, app build, or device feedback round changes state, update `ITERATION_LOG.md`.
- When a workflow becomes reusable, suggest creating or updating a skill.

## AWS Work

Before creating, changing, deploying, or deleting AWS resources:

1. Read `AWS_RESOURCES.md`.
2. Use `skills/aws-project-infrastructure/`.
3. Confirm client, project, environment, owner, repository, tags, data classification, exportability, and cost center.
4. Update `AWS_RESOURCES.md` when AWS changes.

## Handoff

Summarize changed files, validation run, risks or open questions, and the recommended next step.
