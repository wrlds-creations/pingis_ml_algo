#!/usr/bin/env node
/**
 * report_feature_parity.js
 *
 * Per-feature max relativ skillnad TS vs Python över alla klipp i
 * fable_clip_parity_fixture.json. Underlag för approximations-stabilt
 * featureurval (v5): features med stor divergens i appens extraktor
 * utesluts ur träningen så att app == referens.
 */
const fs = require('fs');
const path = require('path');

const SCRIPT_DIR = __dirname;
const REPO_ROOT = path.resolve(SCRIPT_DIR, '..', '..', '..', '..');
const CLIP_FIXTURE = path.join(REPO_ROOT, 'data', 'audio', 'processed', 'noise_robust', 'fable_clip_parity_fixture.json');
const OUT = path.join(REPO_ROOT, 'data', 'audio', 'processed', 'noise_robust', 'feature_parity_report.json');

// Återanvänd parity-harnessens TS-laddare
const harness = path.join(SCRIPT_DIR, 'check_fable_ts_parity.js');
const harnessSrc = fs.readFileSync(harness, 'utf8');
// kör harnessens modul-laddning genom att importera nrFeatures på samma sätt
const vm = require('vm');
const ts = require(path.join(REPO_ROOT, 'apps', 'collector', 'node_modules', 'typescript', 'lib', 'typescript.js'));
const SRC = path.join(REPO_ROOT, 'apps', 'collector', 'src');

const moduleCache = new Map();
function loadTsModule(tsPath) {
  const resolved = tsPath.endsWith('.ts') ? tsPath : `${tsPath}.ts`;
  if (moduleCache.has(resolved)) return moduleCache.get(resolved);
  const source = fs.readFileSync(resolved, 'utf8');
  const js = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
    fileName: resolved,
  }).outputText;
  const mod = { exports: {} };
  const localRequire = (spec) => {
    if (spec.startsWith('.')) return loadTsModule(path.resolve(path.dirname(resolved), spec));
    if (spec.endsWith('.json')) return JSON.parse(fs.readFileSync(path.resolve(path.dirname(resolved), spec), 'utf8'));
    return require(spec);
  };
  vm.runInThisContext(`(function (exports, require, module, __filename, __dirname) {${js}\n})`)(
    mod.exports, localRequire, mod, resolved, path.dirname(resolved),
  );
  moduleCache.set(resolved, mod.exports);
  return mod.exports;
}

const nrFeatures = loadTsModule(path.join(SRC, 'nrFeatures'));
const fixture = JSON.parse(fs.readFileSync(CLIP_FIXTURE, 'utf8'));

const worstRel = {};
const worstAbs = {};
for (const clip of fixture.clips) {
  const pcm = Float32Array.from(clip.samples);
  const tsFeats = nrFeatures.extractFableFeatures(pcm);
  for (const name of fixture.feature_names) {
    const py = clip.py_features[name];
    const tsv = tsFeats[name];
    const abs = Math.abs((tsv ?? 0) - py);
    const rel = abs / Math.max(Math.abs(py), 1e-9);
    if (!(name in worstRel) || rel > worstRel[name]) worstRel[name] = rel;
    if (!(name in worstAbs) || abs > worstAbs[name]) worstAbs[name] = abs;
  }
}

const sorted = Object.entries(worstRel).sort((a, b) => b[1] - a[1]);
console.log('Värsta 20 (max rel diff över', fixture.clips.length, 'klipp):');
for (const [name, rel] of sorted.slice(0, 20)) console.log(`  ${name}: ${rel.toExponential(2)}`);
fs.writeFileSync(OUT, JSON.stringify({
  max_rel_diff: Object.fromEntries(sorted),
  max_abs_diff: worstAbs,
}, null, 1));
console.log('Wrote', OUT);
