# UI Library Selection

Use this guide with the `react-native-ui-system` skill when choosing a UI approach for React Native or cross-platform projects.

| Option | Best For | Tradeoffs |
|---|---|---|
| `gluestack-ui` | Copy-paste NativeWind components across web and mobile; projects that want to own component code | More ownership and maintenance; style discipline matters |
| `Tamagui` | Shared React and React Native design systems; monorepos; serious cross-platform reuse | More setup complexity; requires team buy-in |
| `React Native Paper` | Fast stable Material Design mobile UI | Material look may not fit premium or brand-specific products |
| `UI Kitten` | Themed mobile apps and multi-brand UI | Mobile-oriented; design language may need customization |
| Custom WRLDS UI | Premium custom identity, bespoke interactions, or strong brand direction | Highest ownership cost; requires clear design decisions |

## Decision Factors

- Expo or React Native CLI
- Mobile only or web plus mobile
- Need to own copied component code
- Figma or brand guide availability
- Speed versus long-term design-system ownership
- Material, premium custom, or brand-specific direction
- Accessibility requirements
- Light mode, dark mode, or both

## Decision Record

When a UI approach is chosen, record it in `DECISIONS.md` with rationale, impact, and revisit trigger.
