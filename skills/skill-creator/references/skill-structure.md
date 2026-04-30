# Skill Structure

Use progressive disclosure:

1. `SKILL.md` frontmatter routes the task.
2. `SKILL.md` body gives the core workflow.
3. `references/` holds details that Codex loads only when needed.
4. `scripts/` holds deterministic logic.
5. `assets/` holds reusable templates and output resources.

## Good Skill Names

- `aws-project-infrastructure`
- `berg-airhive-ble-imu`
- `github-ci-fix`
- `react-native-ui-system`

## Weak Skill Names

- `aws`
- `debugging`
- `workflow`
- `helpers`
- `client-stuff`

## Description Checklist

A good description says:

- What the skill does
- When to use it
- Important trigger words or contexts
- What makes it different from nearby skills

## Reference Checklist

Move content to references when it is:

- Long
- Detailed
- Domain-specific
- Needed only for certain branches of the workflow

## Script Checklist

Move logic to scripts when it is:

- Repeated
- Fragile
- Easy to validate deterministically
- Likely to be copied incorrectly if rewritten each time
