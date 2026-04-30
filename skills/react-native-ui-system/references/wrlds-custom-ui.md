# WRLDS Custom UI Reference

Use this reference when the project needs a custom WRLDS-owned React Native UI instead of a packaged component library. This is adapted from the older premium UI skeleton, but it is not the default choice.

## When Custom UI Fits

- The project has a premium or brand-specific visual identity.
- The UI needs motion, bespoke controls, or domain-specific interaction patterns.
- A Figma or brand guide exists and library defaults would fight it.
- The client values differentiation more than fastest implementation.

## Baseline Patterns

- Centralize theme tokens for color, spacing, type, radii, and shadows.
- Keep light and dark mode explicit instead of hard-coding one mode.
- Use a `makeStyles(theme)` or equivalent factory when the codebase uses React Native StyleSheet.
- Use accessible touch targets, readable contrast, and clear focus/disabled states.
- Keep navigation, app state, and theme providers separated.

## Useful Component Ideas

- Pressable cards with visual feedback.
- Animated button scale feedback for high-value actions.
- Status colors for success, warning, error, and neutral states.
- Dashboard, auth, list, detail, and settings screen patterns.
- Icon-led actions using the project's icon library.

## Avoid Carrying Over Blindly

- Do not force dark mode unless the project requires it.
- Do not assume React Native CLI over Expo.
- Do not assume Android immersive mode.
- Do not use a brown/gold palette unless it belongs to the brand.
- Do not add navigation or icon dependencies without explaining why.

## Acceptance Checks

- The visual system matches the brand or product category.
- Text and controls fit on small mobile screens.
- Light/dark behavior is deliberate.
- Components are reusable without hiding project-specific assumptions.
- The decision is logged in `DECISIONS.md`.
