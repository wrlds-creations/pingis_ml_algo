#!/usr/bin/env node
/**
 * Build a playing-retro dataset with the exact Collector TypeScript audio
 * feature extractor and JSON RandomForest runtime.
 *
 * This is intentionally a Node script because T0032 showed that the Python
 * librosa-style feature surface can diverge from the app's actual output.
 */

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const SCRIPT_DIR = __dirname;
const REPO_ROOT = path.resolve(SCRIPT_DIR, '..', '..', '..');
const INPUT_CSV = path.join(
  REPO_ROOT,
  'data',
  'audio',
  'processed',
  'playing_retro_audio_multi_window_dataset_t0030_2026_06_04_006.csv',
);
const OUTPUT_CSV = path.join(
  REPO_ROOT,
  'data',
  'audio',
  'processed',
  'playing_retro_audio_app_feature_dataset_t0033.csv',
);
const RAW_DIR = path.join(REPO_ROOT, 'data', 'audio', 'raw');
const MODEL_JSON = path.join(REPO_ROOT, 'apps', 'collector', 'src', 'models', 'playing_retro_audio_model.json');
const AUDIO_FEATURES_TS = path.join(REPO_ROOT, 'apps', 'collector', 'src', 'audioFeatures.ts');
const RF_RUNTIME_TS = path.join(REPO_ROOT, 'apps', 'collector', 'src', 'rfRuntime.ts');
const TYPESCRIPT_JS = path.join(REPO_ROOT, 'apps', 'collector', 'node_modules', 'typescript', 'lib', 'typescript.js');

function parseArgs(argv) {
  const args = {
    input: INPUT_CSV,
    output: OUTPUT_CSV,
    sessions: [],
    limit: 0,
  };
  for (let index = 2; index < argv.length; index++) {
    const arg = argv[index];
    if (arg === '--input') args.input = argv[++index];
    else if (arg === '--output') args.output = argv[++index];
    else if (arg === '--session') args.sessions.push(argv[++index]);
    else if (arg === '--limit') args.limit = Number(argv[++index]);
    else if (arg === '--help') {
      console.log([
        'Usage: node build_playing_retro_audio_app_feature_dataset_t0033.js [options]',
        '',
        'Options:',
        '  --input <csv>       Source multi-window dataset CSV',
        '  --output <csv>      Output app-feature dataset CSV',
        '  --session <id>      Keep only one session; can be repeated',
        '  --limit <n>         Keep only first n rows after session filtering',
      ].join('\n'));
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  return args;
}

function loadTranspiledTs(tsPath) {
  if (!fs.existsSync(TYPESCRIPT_JS)) {
    throw new Error(`Missing TypeScript compiler at ${TYPESCRIPT_JS}`);
  }
  const ts = require(TYPESCRIPT_JS);
  const source = fs.readFileSync(tsPath, 'utf8');
  const compiled = ts.transpileModule(source, {
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2019,
      esModuleInterop: true,
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
  };
  vm.createContext(sandbox);
  vm.runInContext(compiled, sandbox, { filename: tsPath });
  return Object.keys(sandbox.module.exports).length ? sandbox.module.exports : sandbox.exports;
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = '';
  let quoted = false;
  for (let index = 0; index < text.length; index++) {
    const char = text[index];
    if (quoted) {
      if (char === '"') {
        if (text[index + 1] === '"') {
          field += '"';
          index++;
        } else {
          quoted = false;
        }
      } else {
        field += char;
      }
      continue;
    }
    if (char === '"') {
      quoted = true;
    } else if (char === ',') {
      row.push(field);
      field = '';
    } else if (char === '\n') {
      row.push(field);
      rows.push(row);
      row = [];
      field = '';
    } else if (char !== '\r') {
      field += char;
    }
  }
  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  const headers = rows.shift() || [];
  return {
    headers,
    rows: rows
      .filter(values => values.length > 1 || (values[0] || '') !== '')
      .map(values => {
        const record = {};
        headers.forEach((header, index) => {
          record[header] = values[index] ?? '';
        });
        return record;
      }),
  };
}

function csvEscape(value) {
  if (value === null || value === undefined) return '';
  const text = typeof value === 'number' ? String(value) : String(value);
  if (text.includes('"') || text.includes(',') || text.includes('\n') || text.includes('\r')) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

function writeCsv(filePath, headers, rows) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const lines = [headers.map(csvEscape).join(',')];
  for (const row of rows) {
    lines.push(headers.map(header => csvEscape(row[header])).join(','));
  }
  fs.writeFileSync(filePath, `${lines.join('\n')}\n`, 'utf8');
}

function readWavMonoFloat32(filePath) {
  const buffer = fs.readFileSync(filePath);
  if (buffer.toString('ascii', 0, 4) !== 'RIFF' || buffer.toString('ascii', 8, 12) !== 'WAVE') {
    throw new Error(`Not a RIFF/WAVE file: ${filePath}`);
  }
  let offset = 12;
  let fmt = null;
  let dataOffset = -1;
  let dataSize = 0;
  while (offset + 8 <= buffer.length) {
    const chunkId = buffer.toString('ascii', offset, offset + 4);
    const chunkSize = buffer.readUInt32LE(offset + 4);
    const chunkStart = offset + 8;
    if (chunkId === 'fmt ') {
      fmt = {
        audioFormat: buffer.readUInt16LE(chunkStart),
        channels: buffer.readUInt16LE(chunkStart + 2),
        sampleRate: buffer.readUInt32LE(chunkStart + 4),
        bitsPerSample: buffer.readUInt16LE(chunkStart + 14),
      };
    } else if (chunkId === 'data') {
      dataOffset = chunkStart;
      dataSize = chunkSize;
    }
    offset = chunkStart + chunkSize + (chunkSize % 2);
  }
  if (!fmt || dataOffset < 0) {
    throw new Error(`Missing fmt/data chunk in ${filePath}`);
  }
  const bytesPerSample = fmt.bitsPerSample / 8;
  const frameCount = Math.floor(dataSize / (bytesPerSample * fmt.channels));
  const samples = new Float32Array(frameCount);
  for (let frame = 0; frame < frameCount; frame++) {
    let sum = 0;
    for (let channel = 0; channel < fmt.channels; channel++) {
      const sampleOffset = dataOffset + (frame * fmt.channels + channel) * bytesPerSample;
      let value;
      if (fmt.audioFormat === 1 && fmt.bitsPerSample === 16) {
        value = buffer.readInt16LE(sampleOffset) / 32768;
      } else if (fmt.audioFormat === 1 && fmt.bitsPerSample === 24) {
        const raw = buffer.readUIntLE(sampleOffset, 3);
        const signed = raw & 0x800000 ? raw | 0xff000000 : raw;
        value = signed / 8388608;
      } else if (fmt.audioFormat === 1 && fmt.bitsPerSample === 32) {
        value = buffer.readInt32LE(sampleOffset) / 2147483648;
      } else if (fmt.audioFormat === 3 && fmt.bitsPerSample === 32) {
        value = buffer.readFloatLE(sampleOffset);
      } else {
        throw new Error(`Unsupported WAV format ${fmt.audioFormat}/${fmt.bitsPerSample} in ${filePath}`);
      }
      sum += value;
    }
    samples[frame] = sum / fmt.channels;
  }
  return { sampleRate: fmt.sampleRate, samples };
}

function findRecursiveByBasename(dir, basename) {
  if (!fs.existsSync(dir)) return null;
  const stack = [dir];
  while (stack.length) {
    const current = stack.pop();
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const fullPath = path.join(current, entry.name);
      if (entry.isDirectory()) stack.push(fullPath);
      else if (entry.name === basename) return fullPath;
    }
  }
  return null;
}

function makeWavResolver() {
  const sessionCache = new Map();
  return function resolveWavPath(row) {
    const sessionId = row.session_id;
    const eventIndex = Number(row.event_index || 0);
    if (!sessionCache.has(sessionId)) {
      const sessionPath = path.join(RAW_DIR, `${sessionId}.json`);
      const session = JSON.parse(fs.readFileSync(sessionPath, 'utf8'));
      sessionCache.set(sessionId, session);
    }
    const session = sessionCache.get(sessionId);
    const event = (session.events || [])[eventIndex] || {};
    const wavName = row.wav_filename || event.wav_filename || (event.audio || {}).wav_filename;
    if (!wavName) {
      throw new Error(`No wav filename for ${sessionId} event ${eventIndex}`);
    }
    const candidates = [
      path.join(RAW_DIR, sessionId, wavName),
      path.join(RAW_DIR, wavName),
    ];
    for (const candidate of candidates) {
      if (fs.existsSync(candidate)) return candidate;
    }
    const recursive = findRecursiveByBasename(path.join(RAW_DIR, sessionId), path.basename(wavName));
    if (recursive) return recursive;
    throw new Error(`Could not resolve WAV ${wavName} for ${sessionId} event ${eventIndex}`);
  };
}

function extractWindow(samples, sampleRate, anchorMs, beforeMs, afterMs) {
  const length = Math.round(((beforeMs + afterMs) / 1000) * sampleRate);
  const clip = new Float32Array(length);
  const anchorSample = Math.round((anchorMs / 1000) * sampleRate);
  const beforeSamples = Math.round((beforeMs / 1000) * sampleRate);
  const afterSamples = Math.round((afterMs / 1000) * sampleRate);
  const start = anchorSample - beforeSamples;
  const end = anchorSample + afterSamples;
  const srcStart = Math.max(0, start);
  const srcEnd = Math.min(samples.length, end);
  const dstStart = srcStart - start;
  if (srcEnd > srcStart) {
    clip.set(samples.subarray(srcStart, srcEnd), dstStart);
  }
  return clip;
}

function clippedGap(value) {
  return value === null ? 1.0 : Math.min(value, 1000) / 1000;
}

function contextFeatures(anchorMs, timestampsInput, isSavedCandidate) {
  const timestamps = Array.from(new Set(timestampsInput.map(value => Math.round(Number(value)))))
    .filter(value => Number.isFinite(value))
    .sort((left, right) => left - right);
  const roundedAnchor = Math.round(anchorMs);
  const prevGaps = timestamps.filter(timestamp => timestamp < roundedAnchor).map(timestamp => roundedAnchor - timestamp);
  const nextGaps = timestamps.filter(timestamp => timestamp > roundedAnchor).map(timestamp => timestamp - roundedAnchor);
  const prevGap = prevGaps.length > 0 ? Math.min(...prevGaps) : null;
  const nextGap = nextGaps.length > 0 ? Math.min(...nextGaps) : null;
  const nearestGap = Math.min(...[prevGap, nextGap].filter(value => value !== null));
  const hasNearestGap = Number.isFinite(nearestGap);
  let nearestIndex = 0;
  if (timestamps.length > 0) {
    let nearestDistance = Number.POSITIVE_INFINITY;
    timestamps.forEach((timestamp, index) => {
      const distance = Math.abs(roundedAnchor - timestamp);
      if (distance < nearestDistance) {
        nearestDistance = distance;
        nearestIndex = index;
      }
    });
  }
  const count = timestamps.length;
  return {
    ctx_is_saved_candidate: isSavedCandidate ? 1.0 : 0.0,
    ctx_candidate_count_log: Math.log1p(count),
    ctx_candidate_index_norm: count > 0 ? nearestIndex / Math.max(1, count - 1) : 0.0,
    ctx_has_prev_candidate: prevGap !== null ? 1.0 : 0.0,
    ctx_has_next_candidate: nextGap !== null ? 1.0 : 0.0,
    ctx_prev_gap_1000: clippedGap(prevGap),
    ctx_next_gap_1000: clippedGap(nextGap),
    ctx_nearest_gap_1000: clippedGap(hasNearestGap ? nearestGap : null),
    ctx_density_150ms: timestamps.filter(timestamp => Math.abs(roundedAnchor - timestamp) <= 150).length,
    ctx_density_300ms: timestamps.filter(timestamp => Math.abs(roundedAnchor - timestamp) <= 300).length,
    ctx_density_600ms: timestamps.filter(timestamp => Math.abs(roundedAnchor - timestamp) <= 600).length,
  };
}

function candidateGroupKey(row) {
  return `${row.session_id}\u0000${row.event_index || 0}\u0000${row.source_config || ''}`;
}

function buildCandidateTimestamps(rows) {
  const result = new Map();
  for (const row of rows) {
    if ((row.row_type || '') !== 'candidate') continue;
    const anchor = Number(row.anchor_ms);
    if (!Number.isFinite(anchor)) continue;
    const key = candidateGroupKey(row);
    if (!result.has(key)) result.set(key, []);
    result.get(key).push(Math.round(anchor));
  }
  for (const [key, values] of result.entries()) {
    result.set(key, Array.from(new Set(values)).sort((left, right) => left - right));
  }
  return result;
}

function buildFeatureRow(row, audio, windows, extractFeatures, timestampsByEvent) {
  const features = {};
  const anchorMs = Number(row.anchor_ms);
  if (!Number.isFinite(anchorMs)) {
    throw new Error(`Bad anchor_ms for ${row.session_id}/${row.candidate_id}`);
  }
  for (const windowSpec of windows) {
    const clip = extractWindow(audio.samples, audio.sampleRate, anchorMs, windowSpec.before_ms, windowSpec.after_ms);
    const rawFeatures = extractFeatures(clip);
    for (const [key, value] of Object.entries(rawFeatures)) {
      features[`${windowSpec.name}_${key}`] = value;
    }
  }
  Object.assign(
    features,
    contextFeatures(anchorMs, timestampsByEvent.get(candidateGroupKey(row)) || [], (row.row_type || '') === 'candidate'),
  );
  return features;
}

function main() {
  const args = parseArgs(process.argv);
  const model = JSON.parse(fs.readFileSync(MODEL_JSON, 'utf8'));
  const { extractFeatures } = loadTranspiledTs(AUDIO_FEATURES_TS);
  const { predictWithRfModel } = loadTranspiledTs(RF_RUNTIME_TS);
  const parsed = parseCsv(fs.readFileSync(args.input, 'utf8'));
  const sessionFilter = new Set(args.sessions);
  let rows = parsed.rows;
  if (sessionFilter.size > 0) {
    rows = rows.filter(row => sessionFilter.has(row.session_id));
  }
  if (args.limit > 0) rows = rows.slice(0, args.limit);
  if (rows.length === 0) throw new Error('No rows selected.');

  const featureNames = model.feature_names;
  const featureNameSet = new Set(featureNames);
  const metaHeaders = parsed.headers.filter(header => !featureNameSet.has(header));
  const predictionHeaders = [
    't0033_app_prediction',
    't0033_app_confidence',
    ...model.labels.map(label => `t0033_app_probability_${label}`),
    't0033_app_feature_source',
  ];
  const headers = [...featureNames, ...metaHeaders, ...predictionHeaders];
  const timestampsByEvent = buildCandidateTimestamps(rows);
  const resolveWavPath = makeWavResolver();
  const audioCache = new Map();
  const outputRows = [];

  rows.forEach((row, index) => {
    const wavPath = resolveWavPath(row);
    if (!audioCache.has(wavPath)) {
      const audio = readWavMonoFloat32(wavPath);
      if (Math.round(audio.sampleRate) !== Math.round(model.metadata.sample_rate_hz)) {
        throw new Error(`Expected ${model.metadata.sample_rate_hz} Hz, got ${audio.sampleRate} in ${wavPath}`);
      }
      audioCache.set(wavPath, audio);
    }
    const features = buildFeatureRow(
      row,
      audioCache.get(wavPath),
      model.metadata.windows,
      extractFeatures,
      timestampsByEvent,
    );
    const missing = featureNames.filter(featureName => features[featureName] === undefined);
    if (missing.length > 0) {
      throw new Error(`Missing app features for ${row.session_id}/${row.candidate_id}: ${missing.slice(0, 10).join(', ')}`);
    }
    const prediction = predictWithRfModel(model, features);
    const out = {};
    for (const featureName of featureNames) out[featureName] = features[featureName];
    for (const header of metaHeaders) out[header] = row[header] ?? '';
    out.t0033_app_prediction = prediction.label;
    out.t0033_app_confidence = prediction.confidence;
    for (const label of model.labels) {
      out[`t0033_app_probability_${label}`] = prediction.probabilities[label] ?? '';
    }
    out.t0033_app_feature_source = 'collector_audioFeatures_ts_and_rfRuntime_ts';
    outputRows.push(out);
    if ((index + 1) % 100 === 0 || index + 1 === rows.length) {
      console.error(`app-feature rows ${index + 1}/${rows.length}`);
    }
  });

  writeCsv(args.output, headers, outputRows);
  console.log(JSON.stringify({
    input: path.relative(REPO_ROOT, args.input),
    output: path.relative(REPO_ROOT, args.output),
    rows: outputRows.length,
    sessions: Array.from(new Set(outputRows.map(row => row.session_id))).sort(),
    model_version: model.metadata.model_version,
    feature_count: featureNames.length,
  }, null, 2));
}

main();
