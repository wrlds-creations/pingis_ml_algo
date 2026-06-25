#!/usr/bin/env node
/**
 * Parity check: appens buildVideoStrokeFeatures (v2-delen) + exporterade
 * video_stroke_model.json mot Python-referensen, på riktiga pose-serier.
 * Kör först dump-steget:
 *   python skills/pingis-stroke-detection/scripts/dump_stroke_v2_parity_fixture.py
 * sen:
 *   node skills/pingis-stroke-detection/scripts/check_stroke_v2_ts_parity.js
 */

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..');
const SRC = path.join(REPO_ROOT, 'apps', 'collector', 'src');
const ts = require(path.join(REPO_ROOT, 'apps', 'collector', 'node_modules', 'typescript', 'lib', 'typescript.js'));
const FIXTURE = path.join(REPO_ROOT, 'data', 'video', 'models', 'video_stroke_v2', 'ts_parity_fixture.json');

const cache = new Map();
function loadTsModule(tsPath) {
  const key = path.resolve(tsPath);
  if (cache.has(key)) return cache.get(key);
  const compiled = ts.transpileModule(fs.readFileSync(tsPath, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2019, esModuleInterop: true, resolveJsonModule: true },
  }).outputText;
  const sandbox = {
    exports: {}, module: { exports: {} }, console, Float32Array, Float64Array, Math, Number, Object, Array, Map, Set, Error, Boolean, Infinity, NaN, isFinite,
    require: (spec) => {
      if (spec.endsWith('.json')) return JSON.parse(fs.readFileSync(path.resolve(path.dirname(tsPath), spec), 'utf8'));
      return loadTsModule(path.resolve(path.dirname(tsPath), spec) + '.ts');
    },
  };
  vm.createContext(sandbox);
  vm.runInContext(compiled, sandbox, { filename: tsPath });
  const out = Object.keys(sandbox.module.exports).length ? sandbox.module.exports : sandbox.exports;
  cache.set(key, out);
  return out;
}

const featuresMod = loadTsModule(path.join(SRC, 'videoStrokeFeatures.ts'));
const inferenceMod = loadTsModule(path.join(SRC, 'videoStrokeInference.ts'));
const fixture = JSON.parse(fs.readFileSync(FIXTURE, 'utf8'));

let maxFeatDiff = 0;
let worstFeat = '';
let agree = 0;
let maxProbDiff = 0;
for (const sample of fixture.samples) {
  const result = featuresMod.buildVideoStrokeFeatures(fixture.frames[sample.video], sample.marker_ms, sample.handedness);
  if (!result) { console.log(`  MISS: inga features för ${sample.marker_ms}`); continue; }
  for (const [name, py] of Object.entries(sample.py_features)) {
    const tsv = result.features[name];
    const diff = Math.abs((tsv ?? NaN) - py) / Math.max(1, Math.abs(py));
    if (!(diff <= maxFeatDiff)) { maxFeatDiff = diff; worstFeat = name; }
  }
  const pred = inferenceMod.predictVideoStroke(result.features);
  const pyBest = Object.entries(sample.py_proba).sort((a, b) => b[1] - a[1])[0][0];
  if ((pred.raw_label ?? pred.label) === pyBest) agree++;
  for (const [label, p] of Object.entries(sample.py_proba)) {
    maxProbDiff = Math.max(maxProbDiff, Math.abs((pred.probabilities[label] ?? 0) - p));
  }
}
console.log(`Features: max rel diff ${maxFeatDiff.toExponential(2)} (${worstFeat})`);
console.log(`Modell: argmax-överensstämmelse ${agree}/${fixture.samples.length}, max probdiff ${maxProbDiff.toFixed(4)}`);
if (maxFeatDiff > 1e-6) { console.error('PARITY FAIL: featurediff för stor'); process.exit(1); }
if (agree < fixture.samples.length * 0.97) { console.error('PARITY FAIL: argmax'); process.exit(1); }
console.log('PARITY OK');
