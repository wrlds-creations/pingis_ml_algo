#!/usr/bin/env node
/**
 * Parity: appens bounceSideInference.ts (features + trädvandring) mot
 * Python-referensen, på riktiga 64x64-crops från holdout-sessionen.
 * Kör först: python .../classify_bounce_side.py --export-app
 */

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..');
const SRC = path.join(REPO_ROOT, 'apps', 'collector', 'src');
const ts = require(path.join(REPO_ROOT, 'apps', 'collector', 'node_modules', 'typescript', 'lib', 'typescript.js'));
const FIXTURE = path.join(REPO_ROOT, 'data', 'video', 'models', 'bounce_side_v1', 'bounce_side_ts_parity_fixture.json');

function loadTsModule(tsPath) {
  const compiled = ts.transpileModule(fs.readFileSync(tsPath, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2019, esModuleInterop: true, resolveJsonModule: true },
  }).outputText;
  const sandbox = {
    exports: {}, module: { exports: {} }, console, Float64Array, Uint8Array, Math, Number, Object, Array, Error,
    require: (spec) => JSON.parse(fs.readFileSync(path.resolve(path.dirname(tsPath), spec), 'utf8')),
  };
  vm.createContext(sandbox);
  vm.runInContext(compiled, sandbox, { filename: tsPath });
  return Object.keys(sandbox.module.exports).length ? sandbox.module.exports : sandbox.exports;
}

const inference = loadTsModule(path.join(SRC, 'bounceSideInference.ts'));
const fixture = JSON.parse(fs.readFileSync(FIXTURE, 'utf8'));

let maxFeatDiff = 0;
let worstFeat = '';
let agree = 0;
let maxProbDiff = 0;
for (const sample of fixture.samples) {
  const rgb = Uint8Array.from(sample.rgb64);
  const feats = inference.bounceSideFeatures(rgb, sample.roi_source);
  for (const [name, py] of Object.entries(sample.py_features)) {
    const diff = Math.abs((feats[name] ?? NaN) - py);
    if (!(diff <= maxFeatDiff)) { maxFeatDiff = diff; worstFeat = name; }
  }
  const pred = inference.predictBounceSide(feats);
  const pyBest = Object.entries(sample.py_proba).sort((a, b) => b[1] - a[1])[0][0];
  if (pred.label === pyBest) agree++;
  for (const [label, p] of Object.entries(sample.py_proba)) {
    maxProbDiff = Math.max(maxProbDiff, Math.abs((pred.probabilities[label] ?? 0) - p));
  }
}
console.log(`Features: max abs diff ${maxFeatDiff.toExponential(2)} (${worstFeat})`);
console.log(`Prediktion: argmax-överensstämmelse ${agree}/${fixture.samples.length}, max probdiff ${maxProbDiff.toFixed(4)}`);
if (maxFeatDiff > 1e-3) { console.error('PARITY FAIL: features'); process.exit(1); }
if (agree < fixture.samples.length - 1) { console.error('PARITY FAIL: argmax'); process.exit(1); }
console.log('PARITY OK');
