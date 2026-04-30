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
- `berg-airhive-ble-imu`
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

Use project-specific names only when the workflow includes proprietary protocol details, hardware contracts, or recurring client-specific process. `berg-airhive-ble-imu` is acceptable because it preserves validated hardware integration knowledge.
