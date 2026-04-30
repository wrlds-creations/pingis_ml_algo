---
name: skill-creator
description: Create or update Codex skills for reusable WRLDS workflows, using concise SKILL.md frontmatter for routing, references for detailed knowledge, scripts for deterministic repeatable logic, assets for reusable output resources, agents/openai.yaml metadata, lowercase hyphenated names, and validation guidance.
---

# Skill Creator

Use this skill to create or update a Codex skill for a reusable WRLDS workflow.

## Core Rules

- Use "Codex" or "AI coding agent", not a vendor-specific assistant name.
- Treat `SKILL.md` frontmatter `name` and `description` as the routing mechanism.
- Keep `SKILL.md` concise and procedural.
- Put detailed knowledge in `references/`.
- Put deterministic repeatable logic in `scripts/`.
- Put templates or output assets in `assets/`.
- Require `agents/openai.yaml` for every skill.
- Prefer lowercase, hyphen-separated, specific names.
- Avoid vague names.

## Skill Shape

```text
skills/{skill-name}/
  SKILL.md
  agents/openai.yaml
  references/
  scripts/
  assets/
```

Only include resource folders that the skill actually needs, unless a project template explicitly requires placeholders.

## SKILL.md Frontmatter

Use only:

```yaml
---
name: domain-workflow
description: Specific description of what the skill does and when Codex should use it.
---
```

The description should include trigger contexts because the body is loaded only after routing.

## agents/openai.yaml

Create:

```yaml
interface:
  display_name: "Human Name"
  short_description: "Short UI description"
  default_prompt: "Use $skill-name to ..."
```

Keep this aligned with `SKILL.md`.

## Validation

Before handoff:

- Confirm folder name matches frontmatter `name`.
- Confirm `description` is specific.
- Confirm all referenced files exist.
- Confirm `agents/openai.yaml` exists.
- Remove accidental vendor-specific wording unless intentionally documented.
- Run the project skill validation script if available.

Read `references/skill-structure.md` for detailed authoring guidance and use `scripts/check-skill.js` for a single-skill sanity check when useful.
