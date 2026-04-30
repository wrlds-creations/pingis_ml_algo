# GitHub Actions OIDC

Use GitHub Actions OpenID Connect for AWS deployments when possible. Avoid long-lived AWS access keys by default.

## Pattern

1. Create an AWS IAM role trusted by GitHub's OIDC provider.
2. Restrict the trust policy to the intended repository, branch, environment, or workflow.
3. Grant least-privilege permissions for the deployment action.
4. Use `aws-actions/configure-aws-credentials` with `role-to-assume`.
5. Keep PR workflows to validation, synth, diff, lint, or tests.
6. Require manual approval, `workflow_dispatch`, environment protection, or release events for production deploys.

## Review Checklist

- No static AWS credentials in repository secrets unless explicitly approved.
- Role trust policy is scoped to the repo and deployment path.
- Production deployment requires a human-controlled gate.
- Workflow output includes enough information to review changed resources.
- `AWS_RESOURCES.md` is updated when resources are added, changed, or removed.
