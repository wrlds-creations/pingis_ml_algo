# Amplify Gen 2 Patterns

Use Amplify Gen 2 for new app-adjacent auth, data, and storage when the project needs managed AWS-backed application features and accepts Amplify conventions.

## Typical Files

- `amplify/backend.ts`
- `amplify/auth/resource.ts`
- `amplify/data/resource.ts`
- `amplify/storage/resource.ts`, when storage is needed
- `amplify_outputs.json`

## Basic Data Pattern

```typescript
import { type ClientSchema, a, defineData } from '@aws-amplify/backend';

const schema = a.schema({
  Item: a.model({
    name: a.string().required(),
    ownerId: a.id(),
  }).authorization(allow => [
    allow.owner(),
  ]),
});

export type Schema = ClientSchema<typeof schema>;

export const data = defineData({
  schema,
  authorizationModes: {
    defaultAuthorizationMode: 'userPool',
  },
});
```

## Client Pattern

```typescript
import { generateClient } from 'aws-amplify/api';
import type { Schema } from '../amplify/data/resource';

const client = generateClient<Schema>({ authMode: 'userPool' });

const { data: items } = await client.models.Item.list();
```

## Gotchas

- `amplify_outputs.json` is safe to commit when it contains endpoint configuration only.
- Sandbox must be running during local development.
- Schema changes may require sandbox reload.
- Tags and environment governance still belong to `aws-project-infrastructure`.
