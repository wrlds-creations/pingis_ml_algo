#!/usr/bin/env node
/** Quick Node benchmark of extractFableFeatures (proxy for relative speedup). */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..', '..');
const SRC = path.join(REPO_ROOT, 'apps', 'collector', 'src');
const ts = require(path.join(REPO_ROOT, 'apps', 'collector', 'node_modules', 'typescript', 'lib', 'typescript.js'));

const cache = new Map();
function loadTsModule(tsPath) {
  const key = path.resolve(tsPath);
  if (cache.has(key)) return cache.get(key);
  const compiled = ts.transpileModule(fs.readFileSync(tsPath, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2019, esModuleInterop: true, resolveJsonModule: true },
  }).outputText;
  const sandbox = {
    exports: {}, module: { exports: {} }, console, Float32Array, Float64Array, Int32Array, Math, Number, Object, Array, Set, Error, Infinity, NaN, isFinite, Date,
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

const nr = loadTsModule(path.join(SRC, 'nrFeatures.ts'));
const fixture = JSON.parse(fs.readFileSync(path.join(REPO_ROOT, 'data', 'audio', 'processed', 'noise_robust', 'fable_clip_parity_fixture.json'), 'utf8'));
const clips = fixture.clips.map(c => Float32Array.from(c.samples));

// warmup
for (let i = 0; i < 5; i++) nr.extractFableFeatures(clips[i % clips.length]);

const times = [];
for (const clip of clips) {
  const t0 = process.hrtime.bigint();
  nr.extractFableFeatures(clip);
  const t1 = process.hrtime.bigint();
  times.push(Number(t1 - t0) / 1e6);
}
times.sort((a, b) => a - b);
console.log(`extractFableFeatures over ${clips.length} clips: p50=${times[Math.floor(times.length / 2)].toFixed(1)} ms  p95=${times[Math.floor(times.length * 0.95)].toFixed(1)} ms`);
