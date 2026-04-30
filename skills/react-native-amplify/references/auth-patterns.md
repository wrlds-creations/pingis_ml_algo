# Amplify Auth Patterns

Use these patterns after the project confirms Amplify and the app runtime.

## Imports

```typescript
import {
  confirmSignUp,
  fetchUserAttributes,
  getCurrentUser,
  signIn,
  signOut,
  signUp,
} from 'aws-amplify/auth';
import { Hub } from 'aws-amplify/utils';
```

## Core Flow

- On launch, call `getCurrentUser()` and `fetchUserAttributes()`.
- On sign-up, collect only required attributes.
- On confirmation, route users back to sign-in or into the app after confirmed sign-in.
- Listen to `Hub` auth events and clean up the listener.
- Handle the "already signed in" edge case by signing out before retrying only when that matches the intended UX.

## Profile Sync

For apps with a database profile:

1. Trust Cognito as the authentication source.
2. Fetch or create the app profile after successful auth.
3. Keep profile sync failure non-fatal unless the app requires profile data to operate.
4. Document role and permission decisions in `PROJECT_CONTEXT.md`.
