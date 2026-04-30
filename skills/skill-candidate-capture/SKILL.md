---
name: skill-candidate-capture
description: Detect when a solved workflow should become a reusable WRLDS skill, then propose a skill name, trigger description, SKILL.md contents, references, scripts, assets, and whether it should be global or project-specific. Use after repeated fixes, reusable scripts, fragile manual workflows, domain-specific debugging, or recurring client/project patterns.
---

# Skill Candidate Capture

Use this skill when a task reveals reusable workflow knowledge worth preserving.

## Trigger Conditions

Suggest a skill when:

- The same type of problem has been solved more than once.
- A domain-specific workflow has emerged.
- A fragile manual process was clarified.
- A reusable script was created.
- A debugging process produced reusable knowledge.
- A customer or project pattern is likely to recur.

Do not interrupt for trivial one-off tasks.

## Prompt To User

Ask:

> This looks reusable. Should I create a new skill or update an existing one?

## Proposal Contents

Suggest:

- Skill name
- Trigger description
- What belongs in `SKILL.md`
- What belongs in `references/`
- What belongs in `scripts/`
- What belongs in `assets/`, if anything
- Whether the skill should be global or project-specific

## Naming Guidance

- Use lowercase, hyphen-separated names.
- Prefer domain plus workflow, such as `aws-project-infrastructure`.
- Avoid vague names like `helpers`, `workflow`, or `debugging`.
- Avoid project-specific names unless the knowledge is truly project-specific.
