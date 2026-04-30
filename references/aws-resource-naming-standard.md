# AWS Resource Naming Standard

Use predictable AWS resource names so ownership, project, service, and environment are visible without opening tags.

## Pattern

```text
{client}-{project}-{service}-{env}
```

## Fields

- `client`: lowercase client or owner slug, such as `jumpyard`, `stiga`, or `wrlds`
- `project`: lowercase project slug
- `service`: short AWS service or workload name, such as `api`, `web`, `assets`, `events`, `auth`, `data`, `lambda`, or `cdn`
- `env`: environment slug, such as `dev`, `staging`, `prod`, `preview`, `test`, or `shared`

## Examples

| Client | Project | Service | Environment | Name |
|---|---|---|---|---|
| JumpYard | booking | api | staging | `jumpyard-booking-api-staging` |
| JumpYard | booking | assets | prod | `jumpyard-booking-assets-prod` |
| STIGA | connected product | data | dev | `stiga-connected-product-data-dev` |
| STIGA | connected product | auth | prod | `stiga-connected-product-auth-prod` |
| WRLDS internal | internal tooling | events | dev | `wrlds-internal-tooling-events-dev` |
| WRLDS internal | codex template | validation | shared | `wrlds-codex-template-validation-shared` |

## Globally Unique Names

Some AWS resource names must be globally unique across AWS, not just unique inside an account or region. S3 bucket names are the most common example.

For globally unique resources, extend the pattern with a stable uniqueness suffix:

```text
{client}-{project}-{service}-{env}-{region-or-account-suffix}
```

Examples:

- `jumpyard-booking-assets-prod-eu-north-1`
- `stiga-connected-product-exports-staging-123456789012`
- `wrlds-internal-tooling-artifacts-dev-eu-west-1`

## Generated And Managed Names

- Prefer human-readable names where it is safe and supported.
- Do not force custom physical names for Amplify-managed or framework-managed resources when it increases replacement, drift, or maintenance risk.
- When physical resource names are generated, use CloudFormation stack names, required WRLDS tags, and `AWS_RESOURCES.md` as the source of truth.

## Rules

- Keep names lowercase.
- Use hyphens between words.
- Avoid personal names.
- Avoid secrets, customer data, or ticket numbers in resource names.
- Keep the resource name aligned with required WRLDS tags.
- Prefer stable names for durable resources and generated names for ephemeral preview resources.
- Record created resources in `AWS_RESOURCES.md`.
