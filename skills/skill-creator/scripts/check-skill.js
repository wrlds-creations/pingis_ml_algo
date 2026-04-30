#!/usr/bin/env node
const fs = require('fs');
const path = require('path');

const skillDir = process.argv[2];

if (!skillDir) {
  console.error('Usage: node check-skill.js <skill-directory>');
  process.exit(1);
}

const resolved = path.resolve(skillDir);
const skillPath = path.join(resolved, 'SKILL.md');
const agentPath = path.join(resolved, 'agents', 'openai.yaml');

function fail(message) {
  console.error(`FAIL: ${message}`);
  process.exitCode = 1;
}

if (!fs.existsSync(skillPath)) {
  fail('SKILL.md is missing');
} else {
  const text = fs.readFileSync(skillPath, 'utf8');
  const match = text.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  if (!match) {
    fail('SKILL.md frontmatter is missing');
  } else {
    const name = (match[1].match(/^name:\s*(.+)$/m) || [])[1];
    const description = (match[1].match(/^description:\s*(.+)$/m) || [])[1];
    if (!name) fail('frontmatter name is missing');
    if (!description) fail('frontmatter description is missing');
    if (name && name.trim().replace(/^["']|["']$/g, '') !== path.basename(resolved)) {
      fail('folder name does not match frontmatter name');
    }
  }
}

if (!fs.existsSync(agentPath)) {
  fail('agents/openai.yaml is missing');
}

if (!process.exitCode) {
  console.log(`OK: ${path.basename(resolved)}`);
}
