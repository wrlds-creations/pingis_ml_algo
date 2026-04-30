# AWS Tagging

Apply required WRLDS tags to every project-managed AWS resource that supports tags. If a resource does not support tags directly, tag the parent construct or document the exception in `AWS_RESOURCES.md`.

## Required Tags

| Tag | Meaning | Example |
|---|---|---|
| `WRLDS:Client` | Client or internal owner | `JumpYard` |
| `WRLDS:Project` | Project name | `jumpyard-app` |
| `WRLDS:Environment` | Environment | `dev`, `staging`, `prod` |
| `WRLDS:Owner` | Responsible person or team | `WRLDS` |
| `WRLDS:Repository` | Source repository | `wrlds-creations/example` |
| `WRLDS:ManagedBy` | Management method | `cdk`, `amplify-gen2`, `console`, `terraform` |
| `WRLDS:DataClassification` | Data sensitivity | `public`, `internal`, `confidential`, `restricted` |
| `WRLDS:Exportable` | Whether data must be exportable | `true`, `false` |
| `WRLDS:CostCenter` | Billing identifier | `WRLDS-internal` |
| `WRLDS:CreatedBy` | Creator or automation identity | `codex`, `github-actions`, `love` |

## Enforcement

- Ask for missing required metadata before resource creation.
- Add tags in IaC, not only after deployment.
- Include tag validation in review where practical.
- Record untaggable resources and tagging exceptions in `AWS_RESOURCES.md`.
