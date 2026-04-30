# Project Intake Questions

Use this as a question bank. Do not ask all questions at once. Ask the smallest set needed to unblock the current phase.

## Business Goal

- What problem should this project solve?
- What outcome makes the project successful?
- What is the current phase: prototype, MVP, production, audit, or maintenance?
- What risks matter most right now?

## Client And Billing

- Who is the client?
- What billing entity or cost center should be used?
- Who owns decisions on the client side?
- Who owns delivery on the WRLDS side?

## Users

- Who are the primary users?
- Which user roles exist?
- Which actions require admin or owner permissions?
- Are any users internal-only?

## Platform

- Which platforms are required now?
- Which platforms are future-only?
- Are there device, browser, OS, or hardware constraints?
- Is offline behavior required?

## Frontend

- Is there an existing UI?
- Is there a Figma file or brand guide?
- Is speed or long-term design-system ownership more important?
- Are accessibility, dark mode, or localization required?

## Backend

- Does the project need a backend?
- What API style fits: REST, GraphQL, realtime, files, events, or batch?
- Are there background jobs?
- Does the backend need local development support?

## AWS

- Which AWS account and region should be used?
- Which environments are required?
- Who owns AWS costs?
- What WRLDS tags apply?
- Should infrastructure use CDK, Amplify, CloudFormation, Terraform, or another standard?

## Data

- What are the main entities?
- Who owns the data?
- Is any data personal, confidential, restricted, or regulated?
- Does the client require export or deletion workflows?
- What retention policy applies?

## Auth

- Is authentication required?
- Which sign-in methods are required?
- Which roles and permissions exist?
- Is SSO, Cognito, or another identity provider required?

## Hardware And BLE

- Is hardware involved?
- Which sensors, firmware versions, or protocols matter?
- Is BLE scanning, pairing, streaming, or background behavior required?
- What calibration or device validation is needed?

## Integrations

- Which external APIs or systems are required?
- Are webhooks required?
- How are credentials stored?
- Are sandbox and production credentials separate?

## Analytics

- Which analytics tools should be used?
- Which events matter?
- Are there privacy or consent requirements?
- Who needs reports?

## Deployment

- How should staging deploy?
- How should production deploy?
- Who approves production?
- Are release notes or versioning required?

## Commercial Constraints

- Are there fixed budget or cost limits?
- Are there timeline constraints?
- Is there a handoff, export, or client ownership requirement?
- Are there vendor restrictions?

## Timeline

- What milestone is next?
- What date matters?
- What is blocking the next phase?

## Unknowns

- Which facts are still `TBD`?
- Which unknowns block implementation?
- Which unknowns can safely wait?
