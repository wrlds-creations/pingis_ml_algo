---
name: aws-project-infrastructure
description: Govern WRLDS AWS infrastructure work, including creating, changing, reviewing, deploying, tagging, or deleting AWS resources such as S3, Lambda, DynamoDB, RDS, Cognito, AppSync, API Gateway, CloudWatch, EventBridge, IAM, Amplify, Secrets Manager, SQS, SNS, CloudFront, Route 53, CDK, CloudFormation, and GitHub Actions deploy workflows. Requires WRLDS metadata, tagging, CI/CD safety, and AWS_RESOURCES.md updates.
---

# AWS Project Infrastructure

Use this skill before creating, changing, reviewing, deploying, tagging, or deleting AWS resources.

## Required Reads

1. Read `PROJECT_CONTEXT.md`.
2. Read `DECISIONS.md`.
3. Read `AWS_RESOURCES.md`.
4. Read the relevant reference:
   - Tagging: `references/tagging.md`
   - GitHub Actions OIDC: `references/github-actions-oidc.md`
   - Resource inventory: `references/resource-inventory.md`
   - CDK patterns: `references/aws-cdk-patterns.md`

## Required Metadata

Before creating AWS resources, confirm these fields:

- `WRLDS:Client`
- `WRLDS:Project`
- `WRLDS:Environment`
- `WRLDS:Owner`
- `WRLDS:Repository`
- `WRLDS:ManagedBy`
- `WRLDS:DataClassification`
- `WRLDS:Exportable`
- `WRLDS:CostCenter`
- `WRLDS:CreatedBy`

If any required metadata is missing, ask only for the missing fields.

## First Deploy Preflight

Before the first AWS deploy for a project or environment:

1. Verify the active AWS account with `aws sts get-caller-identity`.
2. Verify the target AWS region.
3. Verify the intended environment, such as `dev`, `staging`, or `prod`.
4. Verify required WRLDS tag values before synth/deploy.
5. Verify project and repository metadata match the target repo.
6. If using Amplify, CDK-generated names, or another generated stack naming system, verify the app/package namespace before creating resources.

## Default Principles

- Prefer Infrastructure as Code.
- Prefer AWS CDK TypeScript for general infrastructure.
- Use Amplify Gen 2 for app-adjacent auth, data, and storage when appropriate.
- Use GitHub Actions with OIDC for deploys where possible.
- Do not use long-lived AWS access keys unless explicitly approved.
- PRs should validate, synth, or diff infrastructure but not deploy production.
- Production deploys should require manual approval, `workflow_dispatch`, or a release-based process.
- Manual AWS Console changes are allowed for inspection, debugging, or emergency fixes, but must be codified afterward.

## Inventory Rule

Update `AWS_RESOURCES.md` whenever infrastructure is created, changed, deleted, imported, or discovered to materially affect cost, security, data, deployment, or ownership.

## Output Summary

For AWS work, summarize:

- Metadata confirmed or still missing
- Resources created, changed, deleted, or reviewed
- IaC files changed
- `AWS_RESOURCES.md` updates
- Validation run, such as synth, diff, tests, or dry run
- Deployment risk and next approval step
