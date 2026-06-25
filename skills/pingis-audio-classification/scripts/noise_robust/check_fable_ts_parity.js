#!/usr/bin/env node
/**
 * check_fable_ts_parity.js
 *
 * Validates the Collector TypeScript Fable pipeline against the Python
 * reference on real clips (T0033 lesson: never promote app code without
 * app-feature/output parity evidence).
 *
 * Checks:
 *  1. Model-only: fable_model_parity_fixture.json scaled vectors through
 *     hgbRuntime.fablePredictFromScaled vs sklearn predict_proba  -> exact.
 *  2. Features: fable_clip_parity_fixture.json raw clips through
 *     nrFeatures.extractFableFeatures vs Python extract_all_features
 *     -> nr_ features near-exact (sosfiltfilt/PCEN are float64 on both
 *     sides); base62 approximate (the app's existing extractor uses a
 *     symmetric Hann etc. - same approximation production already runs).
 *  3. End-to-end: TS features -> TS model vs Python features -> Python
 *     model: argmax agreement + probability deltas.
 *
 * Run:
 *   node skills/pingis-audio-classification/scripts/noise_robust/check_fable_ts_parity.js
 */

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const SCRIPT_DIR = __dirname;
const REPO_ROOT = path.resolve(SCRIPT_DIR, '..', '..', '..', '..');
const SRC = path.join(REPO_ROOT, 'apps', 'collector', 'src');
const TYPESCRIPT_JS = path.join(REPO_ROOT, 'apps', 'collector', 'node_modules', 'typescript', 'lib', 'typescript.js');
const MODEL_FIXTURE = path.join(REPO_ROOT, 'data', 'audio', 'processed', 'noise_robust', 'fable_model_parity_fixture.json');
const CLIP_FIXTURE = path.join(REPO_ROOT, 'data', 'audio', 'processed', 'noise_robust', 'fable_clip_parity_fixture.json');
const MODEL_JSON = path.join(SRC, 'models', 'fable_audio_model.json');

const ts = require(TYPESCRIPT_JS);

const moduleCache = new Map();

function loadTsModule(tsPath) {
  const key = path.resolve(tsPath);
  if (moduleCache.has(key)) return moduleCache.get(key);
  const source = fs.readFileSync(tsPath, 'utf8');
  const compiled = ts.transpileModule(source, {
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2019,
      esModuleInterop: true,
      resolveJsonModule: true,
    },
  }).outputText;
  const sandbox = {
    exports: {},
    module: { exports: {} },
    console,
    Float32Array,
    Float64Array,
    Int32Array,
    Math,
    Number,
    Object,
    Array,
    Set,
    Error,
    Infinity,
    NaN,
    isFinite,
    require: (spec) => {
      if (spec.endsWith('.json')) {
        const jsonPath = path.resolve(path.dirname(tsPath), spec);
        return JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
      }
      const target = path.resolve(path.dirname(tsPath), spec) + '.ts';
      return loadTsModule(target);
    },
  };
  vm.createContext(sandbox);
  vm.runInContext(compiled, sandbox, { filename: tsPath });
  const exportsObj = Object.keys(sandbox.module.exports).length
    ? sandbox.module.exports
    : sandbox.exports;
  moduleCache.set(key, exportsObj);
  return exportsObj;
}

function fail(message) {
  console.error('PARITY FAIL: ' + message);
  process.exitCode = 1;
}

// ── 1. Model-only parity ─────────────────────────────────────────────────────

const hgb = loadTsModule(path.join(SRC, 'hgbRuntime.ts'));
const modelFixture = JSON.parse(fs.readFileSync(MODEL_FIXTURE, 'utf8'));
const modelJson = JSON.parse(fs.readFileSync(MODEL_JSON, 'utf8'));
const labels = modelJson.labels;

let maxModelDiff = 0;
for (let i = 0; i < modelFixture.x_scaled.length; i++) {
  const scaled = Float64Array.from(modelFixture.x_scaled[i]);
  const pred = hgb.fablePredictFromScaled(scaled);
  for (let c = 0; c < labels.length; c++) {
    const diff = Math.abs(pred.probabilities[labels[c]] - modelFixture.expected_proba[i][c]);
    if (diff > maxModelDiff) maxModelDiff = diff;
  }
}
console.log(`1. Model-only parity over ${modelFixture.x_scaled.length} scaled vectors: max |TS - sklearn| = ${maxModelDiff.toExponential(2)}`);
if (maxModelDiff > 1e-9) fail('model-only probabilities diverge (tree walk or softmax bug)');

// ── 2 + 3. Feature & end-to-end parity ───────────────────────────────────────

const nr = loadTsModule(path.join(SRC, 'nrFeatures.ts'));
const clipFixture = JSON.parse(fs.readFileSync(CLIP_FIXTURE, 'utf8'));
const featureNames = clipFixture.feature_names;
const nrNames = featureNames.filter((n) => n.startsWith('nr_'));
const baseNames = featureNames.filter((n) => !n.startsWith('nr_'));

const featDiff = {};
for (const name of featureNames) featDiff[name] = 0;

let argmaxAgree = 0;
let maxProbaDiff = 0;
const probaDiffs = [];

for (const clip of clipFixture.clips) {
  const pcm = Float32Array.from(clip.samples);
  const tsFeats = nr.extractFableFeatures(pcm);

  for (const name of featureNames) {
    const py = clip.py_features[name];
    const tsv = tsFeats[name];
    const scale = Math.max(1, Math.abs(py));
    const diff = Math.abs(tsv - py) / scale;
    if (diff > featDiff[name]) featDiff[name] = diff;
  }

  const tsPred = hgb.fablePredict(tsFeats);
  const pyBest = Object.entries(clip.py_proba).sort((a, b) => b[1] - a[1])[0][0];
  if (tsPred.label === pyBest) argmaxAgree++;
  let probaDiff = 0;
  for (const label of labels) {
    probaDiff = Math.max(probaDiff, Math.abs(tsPred.probabilities[label] - clip.py_proba[label]));
  }
  probaDiffs.push(probaDiff);
  if (probaDiff > maxProbaDiff) maxProbaDiff = probaDiff;
}

const worstNr = nrNames.map((n) => [n, featDiff[n]]).sort((a, b) => b[1] - a[1]).slice(0, 5);
const worstBase = baseNames.map((n) => [n, featDiff[n]]).sort((a, b) => b[1] - a[1]).slice(0, 5);
console.log(`2. Feature parity over ${clipFixture.clips.length} real clips (max relative diff):`);
console.log('   worst nr_ features:  ' + worstNr.map(([n, d]) => `${n}=${d.toExponential(2)}`).join(', '));
console.log('   worst base features: ' + worstBase.map(([n, d]) => `${n}=${d.toExponential(2)}`).join(', '));

const maxNrDiff = Math.max(...nrNames.map((n) => featDiff[n]));
if (maxNrDiff > 1e-4) fail(`nr_ feature parity too loose (max rel diff ${maxNrDiff.toExponential(2)} > 1e-4) - port bug likely`);

probaDiffs.sort((a, b) => a - b);
const p50 = probaDiffs[Math.floor(probaDiffs.length / 2)];
const p95 = probaDiffs[Math.floor(probaDiffs.length * 0.95)];
console.log(`3. End-to-end over ${clipFixture.clips.length} clips: argmax agreement ${argmaxAgree}/${clipFixture.clips.length}, prob diff p50=${p50.toFixed(4)} p95=${p95.toFixed(4)} max=${maxProbaDiff.toFixed(4)}`);
if (argmaxAgree / clipFixture.clips.length < 0.92) {
  fail('end-to-end argmax agreement below 92% - base62 approximation drifted too far');
}

if (process.exitCode !== 1) console.log('PARITY OK');
