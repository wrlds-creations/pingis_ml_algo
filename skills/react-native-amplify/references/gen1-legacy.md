# Amplify Gen 1 Legacy

Use Gen 1 only when the project already has `aws-exports.js` or an established Amplify Gen 1 backend.

## Signals

- `src/aws-exports.js` exists.
- `amplify/` contains Gen 1 backend category files.
- The project uses generated GraphQL operations in `src/graphql/`.

## Safe Workflow

1. Run `amplify status` before changing backend categories.
2. Avoid direct edits to generated backend files unless the Amplify workflow requires them.
3. Prefer custom flattened GraphQL queries for owner-protected models.
4. Review auth rules before changing models.
5. Update `AWS_RESOURCES.md` after resource changes.

## Common Patterns

- Email sign-in through Cognito User Pool.
- Owner-based GraphQL auth.
- S3 storage with `guest`, `protected`, or `private` access levels.
- Custom queries that avoid nested relations when auth rules block reads.

## Caution

Do not run backend-changing Amplify commands until WRLDS AWS metadata, target environment, and deployment expectations are confirmed.
