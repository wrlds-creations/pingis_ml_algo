#!/usr/bin/env node
const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const skillsDir = path.join(root, 'skills');
let failures = 0;

function fail(message) {
  failures += 1;
  console.error(`FAIL: ${message}`);
}

function readText(filePath) {
  return fs.readFileSync(filePath, 'utf8');
}

function parseFrontmatter(text) {
  const match = text.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  if (!match) return null;
  const data = {};
  for (const line of match[1].split(/\r?\n/)) {
    const field = line.match(/^([A-Za-z0-9_-]+):\s*(.*)$/);
    if (field) {
      data[field[1]] = field[2].trim().replace(/^["']|["']$/g, '');
    }
  }
  return data;
}

function walkFiles(dir) {
  if (!fs.existsSync(dir)) return [];
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) files.push(...walkFiles(fullPath));
    else files.push(fullPath);
  }
  return files;
}

function isExternalLink(link) {
  return /^(https?:|mailto:|#)/i.test(link);
}

function resolveReferencePath(skillDir, currentDir, reference) {
  const clean = reference.split('#')[0].trim();
  if (!clean || clean.endsWith('/')) return null;

  if (/^(references|scripts|assets)\//.test(clean)) {
    return path.join(skillDir, clean);
  }

  if (/^\.\.?\//.test(clean)) {
    return path.resolve(currentDir, clean);
  }

  return path.resolve(currentDir, clean);
}

function validateReferencedFiles(skillDir, skillName, filePath) {
  const text = readText(filePath);
  const currentDir = path.dirname(filePath);
  const markdownLinkPattern = /\]\(([^)]+)\)/g;
  const backtickPathPattern = /`((?:references|scripts|assets)\/[^`]+|\.\.?\/[^`]+)`/g;
  let match;

  while ((match = markdownLinkPattern.exec(text)) !== null) {
    const reference = match[1];
    if (isExternalLink(reference)) continue;
    const resolved = resolveReferencePath(skillDir, currentDir, reference);
    if (resolved && !fs.existsSync(resolved)) {
      fail(`${skillName} references missing file ${reference} from ${path.relative(skillDir, filePath)}`);
    }
  }

  while ((match = backtickPathPattern.exec(text)) !== null) {
    const reference = match[1];
    const resolved = resolveReferencePath(skillDir, currentDir, reference);
    if (resolved && !fs.existsSync(resolved)) {
      fail(`${skillName} references missing file ${reference} from ${path.relative(skillDir, filePath)}`);
    }
  }
}

if (!fs.existsSync(skillsDir)) {
  fail('skills directory is missing');
} else {
  const skills = fs.readdirSync(skillsDir, { withFileTypes: true }).filter(entry => entry.isDirectory());

  for (const skill of skills) {
    const skillName = skill.name;
    const skillDir = path.join(skillsDir, skillName);
    const skillPath = path.join(skillDir, 'SKILL.md');
    const agentPath = path.join(skillDir, 'agents', 'openai.yaml');

    if (!fs.existsSync(skillPath)) {
      fail(`${skillName} missing SKILL.md`);
      continue;
    }

    const text = readText(skillPath);
    const frontmatter = parseFrontmatter(text);

    if (!frontmatter) {
      fail(`${skillName} missing YAML frontmatter`);
    } else {
      if (!frontmatter.name) fail(`${skillName} missing frontmatter name`);
      if (!frontmatter.description) fail(`${skillName} missing frontmatter description`);
      if (frontmatter.name && frontmatter.name !== skillName) {
        fail(`${skillName} folder name does not match frontmatter name ${frontmatter.name}`);
      }
      if (frontmatter.description && frontmatter.description.split(/\s+/).length < 12) {
        fail(`${skillName} description is too generic`);
      }
    }

    if (!fs.existsSync(agentPath)) {
      fail(`${skillName} missing agents/openai.yaml`);
    }

    for (const markdownPath of walkFiles(skillDir).filter(filePath => /\.md$/.test(filePath))) {
      validateReferencedFiles(skillDir, skillName, markdownPath);
    }

    const allText = walkFiles(skillDir)
      .filter(filePath => /\.(md|yaml|yml|js|ts|tsx)$/.test(filePath))
      .map(readText)
      .join('\n');

    if (/\bClaude\b/i.test(allText)) {
      fail(`${skillName} contains Claude-specific wording`);
    }
  }
}

if (failures > 0) {
  console.error(`Skill validation failed with ${failures} issue(s).`);
  process.exit(1);
}

console.log('Skill validation passed.');
