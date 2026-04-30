#!/usr/bin/env node
const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');

const requiredTags = [
  'WRLDS:Client',
  'WRLDS:Project',
  'WRLDS:Environment',
  'WRLDS:Owner',
  'WRLDS:Repository',
  'WRLDS:ManagedBy',
  'WRLDS:DataClassification',
  'WRLDS:Exportable',
  'WRLDS:CostCenter',
  'WRLDS:CreatedBy',
];

const filesToCheck = [
  'AWS_RESOURCES.md',
  'references/aws-tagging-standard.md',
  'references/aws-cicd-standard.md',
  'skills/aws-project-infrastructure/SKILL.md',
  'skills/aws-project-infrastructure/references/tagging.md',
];

let failures = 0;

function fail(message) {
  failures += 1;
  console.error(`FAIL: ${message}`);
}

for (const relativePath of filesToCheck) {
  const fullPath = path.join(root, relativePath);
  if (!fs.existsSync(fullPath)) {
    fail(`missing ${relativePath}`);
    continue;
  }

  const text = fs.readFileSync(fullPath, 'utf8');
  for (const tag of requiredTags) {
    if (!text.includes(tag)) {
      fail(`${relativePath} missing ${tag}`);
    }
  }
}

if (failures > 0) {
  console.error(`AWS tag validation failed with ${failures} issue(s).`);
  process.exit(1);
}

console.log('AWS tag validation passed.');
