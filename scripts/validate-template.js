#!/usr/bin/env node
const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');

const coreFiles = [
  'package.json',
  'README.md',
  'AGENTS.md',
  'PROJECT_CONTEXT.md',
  'DECISIONS.md',
  'AWS_RESOURCES.md',
  'references/aws-tagging-standard.md',
  'references/aws-resource-naming-standard.md',
  'references/aws-cicd-standard.md',
  'references/skill-naming-standard.md',
  'scripts/validate-template.js',
  'scripts/validate-skills.js',
  'scripts/validate-aws-tags.js',
];

const optionalFiles = [
  'TEST_PLAN.md',
  'CODEX_TASK.md',
  '.github/pull_request_template.md',
  'references/project-intake-questions.md',
  'references/ui-library-selection.md',
  'references/reddit-stack-evaluation.md',
  'references/codex-working-model.md',
];

const coreSkills = [
  'project-intake',
  'skill-creator',
  'skill-candidate-capture',
  'aws-project-infrastructure',
];

const optionalSkills = [
  'react-native-ui-system',
  'react-native-amplify',
  'berg-airhive-ble-imu',
  'github-ci-fix',
  'github-pr-review',
  'release-notes',
  'codex-repo-audit',
];

const forbiddenRootAssumptions = [
  /android apk/i,
  /android-only/i,
  /ios-only/i,
  /testflight-only/i,
  /react native-only/i,
  /amplify-only/i,
  /mobile-only/i,
];

let failures = 0;

function fail(message) {
  failures += 1;
  console.error(`FAIL: ${message}`);
}

function exists(relativePath) {
  return fs.existsSync(path.join(root, relativePath));
}

for (const file of coreFiles) {
  if (!exists(file)) fail(`missing ${file}`);
}

for (const file of optionalFiles) {
  const fullPath = path.join(root, file);
  if (fs.existsSync(fullPath) && fs.statSync(fullPath).isDirectory()) {
    fail(`optional path should be a file, not directory: ${file}`);
  }
}

function validateSkillShape(skill, required) {
  if (!exists(path.join('skills', skill, 'SKILL.md'))) {
    if (required) fail(`missing skills/${skill}/SKILL.md`);
    return;
  }
  if (!exists(path.join('skills', skill, 'agents', 'openai.yaml'))) {
    fail(`missing skills/${skill}/agents/openai.yaml`);
  }
}

for (const skill of coreSkills) {
  validateSkillShape(skill, true);
}

for (const skill of optionalSkills) {
  if (exists(path.join('skills', skill))) validateSkillShape(skill, false);
}

const agentsPath = path.join(root, 'AGENTS.md');
if (fs.existsSync(agentsPath)) {
  const agentsText = fs.readFileSync(agentsPath, 'utf8');
  for (const pattern of forbiddenRootAssumptions) {
    if (pattern.test(agentsText)) {
      fail(`AGENTS.md contains root-level project assumption matching ${pattern}`);
    }
  }

  for (const requiredPhrase of [
    'PROJECT_CONTEXT.md',
    'DECISIONS.md',
    'AWS_RESOURCES.md',
    'aws-project-infrastructure',
    'Do not push directly to `main`',
    'Do not commit unless explicitly asked',
  ]) {
    if (!agentsText.includes(requiredPhrase)) {
      fail(`AGENTS.md missing required phrase: ${requiredPhrase}`);
    }
  }
}

if (failures > 0) {
  console.error(`Template validation failed with ${failures} issue(s).`);
  process.exit(1);
}

console.log('Template validation passed.');
