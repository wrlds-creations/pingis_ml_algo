# AWS CDK Patterns

Prefer AWS CDK TypeScript for general WRLDS infrastructure when the project does not have an existing IaC standard.

## Defaults

- Keep stacks environment-aware.
- Centralize required WRLDS tags.
- Use least-privilege IAM policies.
- Keep production deploys behind explicit approval.
- Run `cdk synth` or equivalent validation before handoff.
- Use `cdk diff` for review when credentials and context are available.

## Tagging Pattern

Apply tags at app or stack scope where possible:

```typescript
import { Tags } from 'aws-cdk-lib';

const requiredTags = {
  'WRLDS:Client': client,
  'WRLDS:Project': project,
  'WRLDS:Environment': environment,
  'WRLDS:Owner': owner,
  'WRLDS:Repository': repository,
  'WRLDS:ManagedBy': 'cdk',
  'WRLDS:DataClassification': dataClassification,
  'WRLDS:Exportable': exportable,
  'WRLDS:CostCenter': costCenter,
  'WRLDS:CreatedBy': createdBy,
};

for (const [key, value] of Object.entries(requiredTags)) {
  Tags.of(stack).add(key, value);
}
```

## Review Questions

- Is this resource project-owned or shared?
- Which environment owns this resource?
- What data classification applies?
- Can data be exported if the client asks?
- What happens when the project is archived?
