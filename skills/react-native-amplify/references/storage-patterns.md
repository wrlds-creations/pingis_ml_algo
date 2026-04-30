# Amplify Storage Patterns

Use Amplify Storage when the app needs S3-backed upload and retrieval tied to the app auth model.

## Imports

```typescript
import { getUrl, uploadData } from 'aws-amplify/storage';
```

## Access Levels

| Level | Use Case |
|---|---|
| `guest` | Shared authenticated content or public-like app assets |
| `protected` | Owner writes, other authenticated users read |
| `private` | Owner-only content |

## Key Naming

Use a stable folder strategy:

```text
users/{userId}/{timestamp}-{filename}
items/{itemId}/{timestamp}-{filename}
uploads/{environment}/{entity}/{id}
```

## Governance

- Decide data classification before upload paths are finalized.
- Record storage buckets in `AWS_RESOURCES.md`.
- Confirm exportability requirements for client-owned data.
