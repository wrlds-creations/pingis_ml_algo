# Resource Inventory

`AWS_RESOURCES.md` tracks AWS resources that this project owns or materially depends on.

## Include Resources When They Affect

- Cost
- Security
- Data ownership
- Deployment
- Operational reliability
- Client ownership or exportability

## Inventory Fields

- Resource name
- AWS service
- Client
- Project
- Environment
- Region
- Managed by
- Repository
- Data classification
- Exportable
- Cost center
- Notes

## Update Timing

- During planning, add proposed resources if they influence decisions.
- During implementation, update actual resource names and management details.
- During deletion or replacement, move resources to the deleted/replaced section.
- During audits, record discovered unmanaged resources and whether they should be codified.
