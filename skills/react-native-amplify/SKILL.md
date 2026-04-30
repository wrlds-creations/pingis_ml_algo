---
name: react-native-amplify
description: Add or review AWS Amplify for React Native or Expo apps in a WRLDS-governed way, including deciding whether Amplify is appropriate, choosing Gen 2 for new app-adjacent auth/data/storage, using Gen 1 only for existing aws-exports.js projects, and coordinating tags, environments, CI/CD, and AWS_RESOURCES.md with aws-project-infrastructure.
---

# React Native Amplify

Use this skill only when a React Native or Expo project is considering or already using AWS Amplify.

## First Questions

Ask the focused subset needed:

- Is this a new project or an existing project?
- Is the app Expo or React Native CLI?
- Is Amplify actually the right choice for auth, data, and storage?
- Does the project already contain `aws-exports.js`?
- Which environments are required?
- Are required WRLDS AWS tags and project metadata known?

## Generation Choice

- Use Amplify Gen 2 for new app-adjacent auth, data, and storage when Amplify is the right fit.
- Use Amplify Gen 1 only for existing projects that already use `aws-exports.js`.
- Do not migrate Gen 1 to Gen 2 unless the user explicitly asks and the migration risk is reviewed.

## AWS Governance Gate

Before creating AWS resources:

1. Use `aws-project-infrastructure`.
2. Confirm required WRLDS AWS metadata.
3. Confirm environments and deployment model.
4. Update `AWS_RESOURCES.md` when resources are created, changed, or deleted.

## Amplify Namespace Preflight

Before the first Amplify sandbox or deploy:

- Verify `package.json` `name` matches the intended project slug.
- Treat the package name as part of the infrastructure namespace when using Amplify-generated stacks.
- Remember that Amplify sandbox stack names may derive from `package.json` `name` and the sandbox identifier.
- Changing the package name later can create a new sandbox stack and new Cognito/auth outputs.
- Fix namespace issues before creating real users, customer data, or durable app data.

## References

- Gen 2: `references/gen2-patterns.md`
- Gen 1: `references/gen1-legacy.md`
- Auth: `references/auth-patterns.md`
- Storage: `references/storage-patterns.md`
- App context: `assets/AppContext.tsx`

## Implementation Principles

- Keep Amplify client configuration near app entry.
- Keep auth, data, and storage logic in a small service or context boundary.
- Avoid generated nested GraphQL queries in Gen 1 when auth rules make them fragile.
- Treat `amplify_outputs.json` as configuration, not a secret.
- Do not add dependencies until the project confirms Amplify and the app runtime.

## Output Summary

Summarize:

- Whether Amplify is recommended
- Gen 2 or Gen 1 rationale
- Required metadata still missing
- Files changed
- AWS resources and inventory updates
- Validation run or still needed
