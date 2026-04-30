---
name: react-native-ui-system
description: Choose and guide the right UI system for a React Native or cross-platform app, comparing gluestack-ui, Tamagui, React Native Paper, UI Kitten, and custom WRLDS UI. Use when starting or refactoring a mobile UI, choosing a component library, aligning with Figma or brand guides, deciding between speed and long-term design-system ownership, or planning dark mode, accessibility, and web-plus-mobile support.
---

# React Native UI System

Use this as a decision skill. Do not assume every WRLDS app should use the same UI library.

## Required Questions

Ask the focused subset needed for the current decision:

- Expo or React Native CLI?
- Mobile only or web plus mobile?
- Do we need to own copied component code?
- Is there a Figma file or brand guide?
- Is speed or long-term design-system ownership more important?
- Should the UI be Material, premium custom, or brand-specific?
- Is accessibility a requirement?
- Does the app need dark mode, light mode, or both?

## Options To Evaluate

- `gluestack-ui`: good for copy-paste NativeWind components across web and mobile when owning component code matters.
- `Tamagui`: good for shared React and React Native design systems, especially monorepos and serious cross-platform reuse.
- `React Native Paper`: good for stable Material Design components and fast conventional app UI.
- `UI Kitten`: good for themed mobile apps and multi-brand visual systems.
- Custom WRLDS UI: good when the project needs a unique premium visual identity or brand-specific interaction model.

Read `references/wrlds-custom-ui.md` before recommending custom WRLDS UI.

## Decision Output

Summarize:

- Recommended UI approach
- Why it fits this project
- Tradeoffs
- Required packages or design assets
- Risks and validation needs
- Any decision to record in `DECISIONS.md`

## Implementation Guidance

- Match the project context and existing codebase.
- Prefer library defaults for commodity app patterns.
- Prefer custom WRLDS UI only when brand or experience quality justifies ownership cost.
- Preserve accessibility, theming, and platform behavior as explicit requirements.
- If a UI decision is made, update `DECISIONS.md`.
