# AWS Tagging Standard

Every AWS resource managed by a WRLDS project must include required WRLDS tags when the resource supports tagging. Record unsupported or inherited tagging exceptions in `AWS_RESOURCES.md`.

## Required Tags

| Tag | Required | Example |
|---|---|---|
| `WRLDS:Client` | Yes | `JumpYard` |
| `WRLDS:Project` | Yes | `jumpyard-booking` |
| `WRLDS:Environment` | Yes | `staging` |
| `WRLDS:Owner` | Yes | `WRLDS` |
| `WRLDS:Repository` | Yes | `wrlds-creations/jumpyard-booking` |
| `WRLDS:ManagedBy` | Yes | `cdk` |
| `WRLDS:DataClassification` | Yes | `confidential` |
| `WRLDS:Exportable` | Yes | `true` |
| `WRLDS:CostCenter` | Yes | `JumpYard-2026` |
| `WRLDS:CreatedBy` | Yes | `github-actions` |

## Cost Allocation Tags

Activate key WRLDS tags as AWS Cost Allocation Tags for billing visibility. Recommended billing tags:

- `WRLDS:Client`
- `WRLDS:Project`
- `WRLDS:Environment`
- `WRLDS:CostCenter`
- `WRLDS:ManagedBy`

## Allowed Values

### `WRLDS:Environment`

- `dev`: local-adjacent or active development resources
- `test`: automated or manual test resources
- `staging`: pre-production validation resources
- `prod`: production resources
- `preview`: short-lived pull request or branch preview resources
- `shared`: shared resources used by multiple environments

### `WRLDS:ManagedBy`

- `cdk`
- `amplify-gen2`
- `amplify-gen1`
- `cloudformation`
- `terraform`
- `github-actions`
- `console`
- `manual`
- `external`

Use `console` or `manual` only when the resource is intentionally not yet codified, then document the reason in `AWS_RESOURCES.md`.

### `WRLDS:DataClassification`

- `public`: safe for public access
- `internal`: WRLDS or client internal operational data
- `confidential`: sensitive business, customer, or project data
- `restricted`: regulated, highly sensitive, or tightly controlled data

### `WRLDS:Exportable`

- `true`: data must be exportable for client handoff, migration, audit, or deletion workflows
- `false`: data is operational, derived, temporary, or not intended for export

### `WRLDS:CreatedBy`

- `codex`
- `github-actions`
- `wrlds-cli`
- `project-owner`
- `manual`
- `external`

## Example: JumpYard

| Tag | Value |
|---|---|
| `WRLDS:Client` | `JumpYard` |
| `WRLDS:Project` | `jumpyard-app` |
| `WRLDS:Environment` | `prod` |
| `WRLDS:Owner` | `WRLDS` |
| `WRLDS:Repository` | `wrlds-creations/jumpyard-app` |
| `WRLDS:ManagedBy` | `cdk` |
| `WRLDS:DataClassification` | `confidential` |
| `WRLDS:Exportable` | `true` |
| `WRLDS:CostCenter` | `JumpYard` |
| `WRLDS:CreatedBy` | `github-actions` |

## Example: STIGA

| Tag | Value |
|---|---|
| `WRLDS:Client` | `STIGA` |
| `WRLDS:Project` | `stiga-connected-product` |
| `WRLDS:Environment` | `staging` |
| `WRLDS:Owner` | `WRLDS` |
| `WRLDS:Repository` | `wrlds-creations/stiga-connected-product` |
| `WRLDS:ManagedBy` | `amplify-gen2` |
| `WRLDS:DataClassification` | `internal` |
| `WRLDS:Exportable` | `true` |
| `WRLDS:CostCenter` | `STIGA` |
| `WRLDS:CreatedBy` | `codex` |

## Example: WRLDS Internal

| Tag | Value |
|---|---|
| `WRLDS:Client` | `WRLDS` |
| `WRLDS:Project` | `wrlds-internal-tooling` |
| `WRLDS:Environment` | `dev` |
| `WRLDS:Owner` | `WRLDS` |
| `WRLDS:Repository` | `wrlds-creations/wrlds-internal-tooling` |
| `WRLDS:ManagedBy` | `cdk` |
| `WRLDS:DataClassification` | `internal` |
| `WRLDS:Exportable` | `false` |
| `WRLDS:CostCenter` | `WRLDS-internal` |
| `WRLDS:CreatedBy` | `manual` |
