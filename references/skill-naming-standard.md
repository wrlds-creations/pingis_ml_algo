# Skill Naming Standard

Skill names should be easy to route, easy to scan, and specific enough to avoid accidental triggering.

## Rules

- Use lowercase names.
- Use hyphen-separated words.
- Do not add a `skill` suffix.
- Prefer domain plus workflow.
- Avoid vague names.
- Avoid project-specific names unless the knowledge is truly project-specific.
- Keep folder name and `SKILL.md` frontmatter `name` identical.

## Good Names

- `aws-project-infrastructure`
- `react-native-ui-system`
- `pingis-audio-classification`
- `github-ci-fix`
- `project-intake`

## Bad Names

- `aws`
- `ui`
- `workflow`
- `debug`
- `client-skill`
- `my-project`

## Project-Specific Names

Use project-specific names only when the workflow includes proprietary data formats, model pipelines, or recurring client-specific process. `pingis-audio-classification` is acceptable because it preserves validated project data and model workflow knowledge.
