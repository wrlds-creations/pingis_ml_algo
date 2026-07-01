import { extractFableFeatures } from './nrFeatures';
import { fablePredict, type FablePrediction } from './hgbRuntime';
import { bounceHeightMeters } from './fableEngine';
import { predictWithRfModelRaw, type RfJsonModel, type RfPrediction } from './rfRuntime';
import type { NativeAudioOnsetDebug } from './NativeAudioStream';
import t0103ModelJson from './models/fable_extra_trees_candidate_t0103.json';
import t0104eModelJson from './models/fable_extra_trees_candidate_t0104e.json';

export interface CandidateModel extends RfJsonModel {
  metadata?: {
    model_version?: string;
    source_ticket?: string;
    model_type?: string;
    candidate_gate?: string;
    selected_threshold?: number;
    smart_dedupe_ms?: number;
    positive_label?: string;
    runtime_status?: string;
    [key: string]: unknown;
  };
}

export type BounceAudioTestModelMetadata = NonNullable<CandidateModel['metadata']>;

export interface BounceAudioTestModelOption {
  id: string;
  title: string;
  shortTitle: string;
  subtitle: string;
  model: CandidateModel;
}

const T0103_MODEL = t0103ModelJson as unknown as CandidateModel;
const T0104E_MODEL = t0104eModelJson as unknown as CandidateModel;

export const BOUNCE_AUDIO_TEST_MODEL_OPTIONS: BounceAudioTestModelOption[] = [
  {
    id: 't0103',
    title: 'T0103 current',
    shortTitle: 'T0103',
    subtitle: 'current guarded test model',
    model: T0103_MODEL,
  },
  {
    id: 't0104e',
    title: 'T0104E candidate',
    shortTitle: 'T0104E',
    subtitle: 'new diagnostic near-miss',
    model: T0104E_MODEL,
  },
];

export const BOUNCE_AUDIO_TEST_DEFAULT_MODEL_ID = 't0103';

export function getBounceAudioTestModelOption(modelId: string): BounceAudioTestModelOption {
  return BOUNCE_AUDIO_TEST_MODEL_OPTIONS.find(option => option.id === modelId)
    ?? BOUNCE_AUDIO_TEST_MODEL_OPTIONS[0];
}

const DEFAULT_MODEL_OPTION = getBounceAudioTestModelOption(BOUNCE_AUDIO_TEST_DEFAULT_MODEL_ID);

export const BOUNCE_AUDIO_TEST_MODEL_VERSION =
  DEFAULT_MODEL_OPTION.model.metadata?.model_version ?? 'fable_extra_trees_candidate_t0103';

export interface BounceAudioTestRuntimeConfig {
  threshold: number;
  fableNoiseVetoThreshold: number;
}

export interface BounceAudioTestDecisionConfig {
  positiveLabel: string;
  threshold: number;
  fableNoiseVetoThreshold: number;
  smartDedupeMs: number;
  decisionDelayMs: number;
  staleMs: number;
}

function metadataNumber(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

export const BOUNCE_AUDIO_TEST_DEFAULT_RUNTIME_CONFIG: BounceAudioTestRuntimeConfig = {
  threshold: metadataNumber(DEFAULT_MODEL_OPTION.model.metadata?.selected_threshold, 0.575),
  fableNoiseVetoThreshold: metadataNumber(DEFAULT_MODEL_OPTION.model.metadata?.fable_noise_veto_threshold, 1.0),
};

function defaultRuntimeConfigForModel(model: CandidateModel): BounceAudioTestRuntimeConfig {
  return {
    threshold: metadataNumber(model.metadata?.selected_threshold, 0.575),
    fableNoiseVetoThreshold: metadataNumber(model.metadata?.fable_noise_veto_threshold, 1.0),
  };
}

function decisionConfigForModel(
  model: CandidateModel,
  runtimeConfig: BounceAudioTestRuntimeConfig,
): BounceAudioTestDecisionConfig {
  return {
    positiveLabel: model.metadata?.positive_label ?? 'racket_bounce',
    threshold: runtimeConfig.threshold,
    fableNoiseVetoThreshold: runtimeConfig.fableNoiseVetoThreshold,
    smartDedupeMs: metadataNumber(model.metadata?.smart_dedupe_ms, 180),
    decisionDelayMs: 500,
    staleMs: 2500,
  };
}

export function defaultRuntimeConfigForModelId(modelId: string): BounceAudioTestRuntimeConfig {
  return defaultRuntimeConfigForModel(getBounceAudioTestModelOption(modelId).model);
}

export function decisionConfigForModelId(
  modelId: string,
  runtimeConfig = defaultRuntimeConfigForModelId(modelId),
): BounceAudioTestDecisionConfig {
  return decisionConfigForModel(getBounceAudioTestModelOption(modelId).model, runtimeConfig);
}

export const BOUNCE_AUDIO_TEST_CONFIG: BounceAudioTestDecisionConfig = {
  ...decisionConfigForModel(DEFAULT_MODEL_OPTION.model, BOUNCE_AUDIO_TEST_DEFAULT_RUNTIME_CONFIG),
  decisionDelayMs: 500,
  staleMs: 2500,
};

export const BOUNCE_AUDIO_TEST_PEAK_GATE_CONFIG = {
  gateId: 'peak_fast_balanced',
  envelope: 'raw_abs',
  smoothingMs: 3,
  minGapMs: 220,
  backgroundWindowMs: 500,
  backgroundExcludeBeforePeakMs: 60,
  absoluteMinimum: 0.08,
  ratioMinimum: 2.0,
  zMinimum: 0.0,
} as const;

const FAR_GAP_MS = 99999;
const HEIGHT_MIN_GAP_MS = 250;
const HEIGHT_MAX_GAP_MS = 1500;

export type BounceAudioCandidateDecision =
  | 'pending_delay'
  | 'classified_low_probability'
  | 'rejected_fable_noise_veto'
  | 'accepted_pending_dedupe'
  | 'deduped_lower_probability'
  | 'counted'
  | 'stale_backlog'
  | 'js_error';

export interface BounceAudioDebugReason {
  code: string;
  severity: 'info' | 'warning' | 'blocker';
  message: string;
  value?: number;
  threshold?: number;
}

export interface BounceAudioFeatureDiagnostic {
  feature: string;
  value: number;
  model_z?: number;
}

export interface BounceAudioDebugExplanation {
  summary: string;
  score: number;
  threshold: number;
  margin: number;
  fable_racket_probability?: number;
  fable_noise_probability?: number;
  fable_noise_veto_threshold?: number;
  tree_positive_mass?: number;
  reasons: BounceAudioDebugReason[];
  feature_diagnostics: BounceAudioFeatureDiagnostic[];
}

export interface BounceAudioCandidateRow {
  id: number;
  index: number;
  received_at_ms: number;
  native_onset_time_ms: number;
  native_onset_pos?: number;
  age_ms?: number;
  native_debug?: NativeAudioOnsetDebug;
  frame_rms: number;
  bg_rms: number;
  peak_value: number;
  peak_ratio: number;
  peak_z: number;
  prev_gap_ms?: number;
  next_gap_ms?: number;
  neighbor_count_500ms?: number;
  fable_label?: string;
  fable_confidence?: number;
  fable_probabilities?: Record<string, number>;
  classifier_label?: string;
  classifier_confidence?: number;
  classifier_probability?: number;
  classifier_probabilities?: Record<string, number>;
  counted: boolean;
  decision: BounceAudioCandidateDecision;
  reject_reason?: string;
  debug_explanation?: BounceAudioDebugExplanation;
  feature_ms?: number;
  predict_ms?: number;
  bounce_gap_ms?: number;
  bounce_height_m?: number;
  audio_b64?: string;
  feature_vector?: Record<string, number>;
}

export interface BounceAudioCandidateInput {
  pcm: Float32Array;
  audioB64?: string;
  nativeDebug?: NativeAudioOnsetDebug;
  receivedAtMs: number;
}

export interface BounceAudioFlushResult {
  newlyCounted: BounceAudioCandidateRow[];
  rowsChanged: boolean;
}

function finiteNumber(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function probabilityOrFallback(value: unknown, fallback: number): number {
  const parsed = finiteNumber(value, fallback);
  if (parsed < 0) return 0;
  if (parsed > 1) return 1;
  return parsed;
}

function normalizeRuntimeConfig(
  config?: Partial<BounceAudioTestRuntimeConfig>,
  model = DEFAULT_MODEL_OPTION.model,
): BounceAudioTestRuntimeConfig {
  const defaults = defaultRuntimeConfigForModel(model);
  return {
    threshold: probabilityOrFallback(
      config?.threshold,
      defaults.threshold,
    ),
    fableNoiseVetoThreshold: probabilityOrFallback(
      config?.fableNoiseVetoThreshold,
      defaults.fableNoiseVetoThreshold,
    ),
  };
}

function nativeOnsetTime(nativeDebug: NativeAudioOnsetDebug | undefined, receivedAtMs: number) {
  return finiteNumber(nativeDebug?.onset_time_ms, receivedAtMs);
}

function positiveProbability(prediction: RfPrediction, positiveLabel: string): number {
  return finiteNumber(
    prediction.probabilities[positiveLabel],
    prediction.label === positiveLabel ? prediction.confidence : 0,
  );
}

function isHighConfidenceFableNoise(
  prediction: FablePrediction,
  config: BounceAudioTestRuntimeConfig,
): boolean {
  if (config.fableNoiseVetoThreshold >= 1) return false;
  return prediction.label === 'noise'
    && prediction.confidence >= config.fableNoiseVetoThreshold;
}

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function percentagePoints(value: number): string {
  return `${Math.abs(value * 100).toFixed(1)}pp`;
}

function compactNumber(value: number): string {
  if (!Number.isFinite(value)) return '0';
  if (Math.abs(value) >= 100) return value.toFixed(0);
  if (Math.abs(value) >= 10) return value.toFixed(1);
  if (Math.abs(value) >= 1) return value.toFixed(2);
  return value.toFixed(3);
}

function modelZ(model: CandidateModel, featureName: string, value: number): number | undefined {
  const index = model.feature_names.indexOf(featureName);
  if (index < 0) return undefined;
  const std = finiteNumber(model.scaler_std[index], 1);
  const denominator = std === 0 ? 1 : std;
  return (value - finiteNumber(model.scaler_mean[index])) / denominator;
}

function diagnosticFeature(model: CandidateModel, featureName: string, value: number): BounceAudioFeatureDiagnostic {
  const z = modelZ(model, featureName, value);
  return {
    feature: featureName,
    value,
    ...(z === undefined ? {} : { model_z: z }),
  };
}

function buildDebugExplanation(
  row: BounceAudioCandidateRow,
  fablePrediction: FablePrediction,
  probability: number,
  config: BounceAudioTestRuntimeConfig,
  model: CandidateModel,
  decisionConfig: BounceAudioTestDecisionConfig,
): BounceAudioDebugExplanation {
  const threshold = config.threshold;
  const margin = probability - threshold;
  const fableRacketProbability = finiteNumber(fablePrediction.probabilities.racket_bounce);
  const fableNoiseProbability = finiteNumber(fablePrediction.probabilities.noise);
  const reasons: BounceAudioDebugReason[] = [];

  if (isHighConfidenceFableNoise(fablePrediction, config)) {
    reasons.push({
      code: 'fable_noise_veto',
      severity: 'blocker',
      message: `Veto: first layer says noise (${percent(fablePrediction.confidence)} confidence).`,
      value: fablePrediction.confidence,
      threshold: config.fableNoiseVetoThreshold,
    });
  }

  if (margin < 0) {
    reasons.push({
      code: 'below_threshold',
      severity: 'blocker',
      message: `Score ${percent(probability)} is ${percentagePoints(margin)} below ${percent(threshold)} threshold.`,
      value: probability,
      threshold,
    });
  } else {
    reasons.push({
      code: 'above_threshold',
      severity: 'info',
      message: `Score ${percent(probability)} is ${percentagePoints(margin)} above ${percent(threshold)} threshold.`,
      value: probability,
      threshold,
    });
  }

  if (fablePrediction.label !== decisionConfig.positiveLabel) {
    reasons.push({
      code: 'fable_non_racket',
      severity: 'warning',
      message: `First layer says ${fablePrediction.label} (${percent(fablePrediction.confidence)} confidence).`,
      value: fablePrediction.confidence,
    });
  } else if (fableRacketProbability >= 0.9 && margin < 0) {
    reasons.push({
      code: 'fable_agrees_classifier_rejects',
      severity: 'info',
      message: `First layer strongly says racket (${percent(fableRacketProbability)}), but second layer still rejected it.`,
      value: fableRacketProbability,
    });
  } else if (fableRacketProbability < 0.75) {
    reasons.push({
      code: 'fable_racket_prob_low',
      severity: 'warning',
      message: `First-layer racket probability is only ${percent(fableRacketProbability)}.`,
      value: fableRacketProbability,
      threshold: 0.75,
    });
  }

  if (row.peak_value < 0.12) {
    reasons.push({
      code: 'very_soft_peak',
      severity: 'warning',
      message: `Peak is very soft (${compactNumber(row.peak_value)}).`,
      value: row.peak_value,
      threshold: 0.12,
    });
  } else if (row.peak_value < 0.18) {
    reasons.push({
      code: 'soft_peak',
      severity: 'info',
      message: `Peak is soft (${compactNumber(row.peak_value)}).`,
      value: row.peak_value,
      threshold: 0.18,
    });
  }

  if (row.frame_rms < 0.04) {
    reasons.push({
      code: 'very_low_frame_rms',
      severity: 'warning',
      message: `Impact window energy is very low (${compactNumber(row.frame_rms)} RMS).`,
      value: row.frame_rms,
      threshold: 0.04,
    });
  } else if (row.frame_rms < 0.06) {
    reasons.push({
      code: 'low_frame_rms',
      severity: 'info',
      message: `Impact window energy is low (${compactNumber(row.frame_rms)} RMS).`,
      value: row.frame_rms,
      threshold: 0.06,
    });
  }

  if (row.bg_rms > 0.006) {
    reasons.push({
      code: 'high_background_rms',
      severity: 'warning',
      message: `Background level is high (${compactNumber(row.bg_rms)} RMS).`,
      value: row.bg_rms,
      threshold: 0.006,
    });
  }

  if (row.peak_ratio < 20) {
    reasons.push({
      code: 'low_peak_contrast',
      severity: 'warning',
      message: `Peak contrast over background is low (${compactNumber(row.peak_ratio)}x).`,
      value: row.peak_ratio,
      threshold: 20,
    });
  }

  if (row.peak_z < 80) {
    reasons.push({
      code: 'low_peak_z',
      severity: 'info',
      message: `Peak z-score is low (${compactNumber(row.peak_z)}).`,
      value: row.peak_z,
      threshold: 80,
    });
  }

  const nearestGap = Math.min(row.prev_gap_ms ?? FAR_GAP_MS, row.next_gap_ms ?? FAR_GAP_MS);
  if ((row.neighbor_count_500ms ?? 0) > 0 && nearestGap < 300) {
    reasons.push({
      code: 'nearby_candidate',
      severity: 'info',
      message: `Another peak is nearby (${compactNumber(nearestGap)} ms), so dedupe/timing may matter.`,
      value: nearestGap,
      threshold: 300,
    });
  }

  const firstReason = reasons.find(reason => reason.severity === 'blocker')
    ?? reasons.find(reason => reason.severity === 'warning')
    ?? reasons[0];
  const supportReason = reasons.find(reason => reason.code === 'soft_peak' || reason.code === 'very_soft_peak' || reason.code === 'low_frame_rms' || reason.code === 'very_low_frame_rms');
  const summary = supportReason && firstReason.code === 'below_threshold'
    ? `${firstReason.message} ${supportReason.message}`
    : firstReason.message;

  return {
    summary,
    score: probability,
    threshold,
    margin,
    fable_racket_probability: fableRacketProbability,
    fable_noise_probability: fableNoiseProbability,
    fable_noise_veto_threshold: config.fableNoiseVetoThreshold,
    tree_positive_mass: probability * model.trees.length,
    reasons,
    feature_diagnostics: [
      diagnosticFeature(model, 'peak_value', row.peak_value),
      diagnosticFeature(model, 'frame_rms', row.frame_rms),
      diagnosticFeature(model, 'bg_rms', row.bg_rms),
      diagnosticFeature(model, 'peak_ratio', row.peak_ratio),
      diagnosticFeature(model, 'peak_z', row.peak_z),
      diagnosticFeature(model, 'prob_racket_bounce', fableRacketProbability),
      diagnosticFeature(model, 'prob_noise', finiteNumber(fablePrediction.probabilities.noise)),
      diagnosticFeature(model, 'prob_floor_bounce', finiteNumber(fablePrediction.probabilities.floor_bounce)),
      diagnosticFeature(model, 'prob_table_bounce', finiteNumber(fablePrediction.probabilities.table_bounce)),
    ],
  };
}

function fableModelFlags(prediction: FablePrediction): Record<string, number> {
  return {
    model_is_racket: prediction.label === 'racket_bounce' ? 1 : 0,
    model_is_noise: prediction.label === 'noise' ? 1 : 0,
    model_is_floor: prediction.label === 'floor_bounce' ? 1 : 0,
    model_is_table: prediction.label === 'table_bounce' ? 1 : 0,
  };
}

export function buildBounceAudioFeatureVector(
  row: BounceAudioCandidateRow,
  fableFeatures: Record<string, number>,
  fablePrediction: FablePrediction,
  model: CandidateModel = DEFAULT_MODEL_OPTION.model,
): Record<string, number> {
  const vector: Record<string, number> = {
    frame_rms: row.frame_rms,
    bg_rms: row.bg_rms,
    peak_value: row.peak_value,
    peak_ratio: row.peak_ratio,
    peak_z: row.peak_z,
    prev_gap_ms: row.prev_gap_ms ?? FAR_GAP_MS,
    next_gap_ms: row.next_gap_ms ?? FAR_GAP_MS,
    neighbor_count_500ms: row.neighbor_count_500ms ?? 0,
    prob_racket_bounce: finiteNumber(fablePrediction.probabilities.racket_bounce),
    prob_noise: finiteNumber(fablePrediction.probabilities.noise),
    prob_floor_bounce: finiteNumber(fablePrediction.probabilities.floor_bounce),
    prob_table_bounce: finiteNumber(fablePrediction.probabilities.table_bounce),
    model_confidence: fablePrediction.confidence,
    ...fableModelFlags(fablePrediction),
  };

  for (const [name, value] of Object.entries(fableFeatures)) {
    vector[`feat_${name}`] = finiteNumber(value);
  }

  // Keep the vector aligned with the exported artifact even if a future feature
  // extractor emits fewer keys than the model expects.
  const aligned: Record<string, number> = {};
  for (const name of model.feature_names) aligned[name] = finiteNumber(vector[name]);
  return aligned;
}

export class BounceAudioTestEngine {
  private rows: BounceAudioCandidateRow[] = [];
  private pcmById = new Map<number, Float32Array>();
  private nextId = 1;
  private emittedCountedIds = new Set<number>();
  private lastEmittedCountedOnsetMs: number | null = null;
  private modelOption = DEFAULT_MODEL_OPTION;
  private runtimeConfig = normalizeRuntimeConfig(undefined, DEFAULT_MODEL_OPTION.model);
  private decisionConfig = decisionConfigForModel(DEFAULT_MODEL_OPTION.model, this.runtimeConfig);

  constructor(config?: Partial<BounceAudioTestRuntimeConfig>, modelId = BOUNCE_AUDIO_TEST_DEFAULT_MODEL_ID) {
    this.setModelOption(modelId, config);
  }

  setModelOption(modelId: string, config?: Partial<BounceAudioTestRuntimeConfig>) {
    this.modelOption = getBounceAudioTestModelOption(modelId);
    this.runtimeConfig = normalizeRuntimeConfig(config, this.modelOption.model);
    this.decisionConfig = decisionConfigForModel(this.modelOption.model, this.runtimeConfig);
    return {
      modelOption: this.getModelOption(),
      runtimeConfig: this.getRuntimeConfig(),
      decisionConfig: this.getDecisionConfig(),
    };
  }

  setRuntimeConfig(config: Partial<BounceAudioTestRuntimeConfig>): BounceAudioTestRuntimeConfig {
    this.runtimeConfig = normalizeRuntimeConfig(config, this.modelOption.model);
    this.decisionConfig = decisionConfigForModel(this.modelOption.model, this.runtimeConfig);
    return this.getRuntimeConfig();
  }

  getRuntimeConfig(): BounceAudioTestRuntimeConfig {
    return { ...this.runtimeConfig };
  }

  getModelOption(): BounceAudioTestModelOption {
    return this.modelOption;
  }

  getDecisionConfig(): BounceAudioTestDecisionConfig {
    return { ...this.decisionConfig };
  }

  getModelMetadata(): BounceAudioTestModelMetadata {
    return this.modelOption.model.metadata ?? {};
  }

  reset(): void {
    this.rows = [];
    this.pcmById.clear();
    this.nextId = 1;
    this.emittedCountedIds.clear();
    this.lastEmittedCountedOnsetMs = null;
  }

  addCandidate(input: BounceAudioCandidateInput): BounceAudioCandidateRow {
    const id = this.nextId++;
    const nativeDebug = input.nativeDebug;
    const row: BounceAudioCandidateRow = {
      id,
      index: id,
      received_at_ms: input.receivedAtMs,
      native_onset_time_ms: nativeOnsetTime(nativeDebug, input.receivedAtMs),
      native_onset_pos: nativeDebug?.onset_pos,
      native_debug: nativeDebug,
      frame_rms: finiteNumber(nativeDebug?.rms),
      bg_rms: finiteNumber(nativeDebug?.background_rms),
      peak_value: finiteNumber(nativeDebug?.peak_value, finiteNumber(nativeDebug?.rms)),
      peak_ratio: finiteNumber(nativeDebug?.peak_ratio, finiteNumber(nativeDebug?.onset_ratio)),
      peak_z: finiteNumber(nativeDebug?.peak_z),
      counted: false,
      decision: 'pending_delay',
      ...(input.audioB64 ? { audio_b64: input.audioB64 } : {}),
    };
    this.rows.push(row);
    this.rows.sort((a, b) => a.native_onset_time_ms - b.native_onset_time_ms);
    this.pcmById.set(id, input.pcm);
    return row;
  }

  flush(nowMs = Date.now(), final = false): BounceAudioFlushResult {
    const matureCutoffMs = final ? Number.POSITIVE_INFINITY : nowMs - this.decisionConfig.decisionDelayMs;
    let rowsChanged = false;

    for (const row of this.rows) {
      row.age_ms = Math.max(0, nowMs - row.native_onset_time_ms);
      if (row.decision !== 'pending_delay') continue;
      if (row.native_onset_time_ms > matureCutoffMs) continue;
      rowsChanged = this.classifyRow(row, nowMs) || rowsChanged;
    }

    const newlyCounted = this.applySmartDedupe(matureCutoffMs, final);
    if (newlyCounted.length > 0) rowsChanged = true;
    return { newlyCounted, rowsChanged };
  }

  getRows(): BounceAudioCandidateRow[] {
    return this.rows.map(row => ({ ...row, native_debug: row.native_debug ? { ...row.native_debug } : undefined }));
  }

  getCounts() {
    const counted = this.rows.filter(row => row.counted).length;
    const lowProbability = this.rows.filter(row => row.decision === 'classified_low_probability').length;
    const fableNoiseVetoed = this.rows.filter(row => row.decision === 'rejected_fable_noise_veto').length;
    const deduped = this.rows.filter(row => row.decision === 'deduped_lower_probability').length;
    const acceptedPending = this.rows.filter(row => row.decision === 'accepted_pending_dedupe').length;
    const pending = this.rows.filter(row => row.decision === 'pending_delay').length;
    const errors = this.rows.filter(row => row.decision === 'js_error').length;
    return {
      native_candidates: this.rows.length,
      classified: this.rows.filter(row => row.classifier_probability !== undefined).length,
      counted,
      low_probability: lowProbability,
      fable_noise_vetoed: fableNoiseVetoed,
      deduped,
      accepted_pending_dedupe: acceptedPending,
      pending,
      errors,
    };
  }

  private updateTimingFeatures(): void {
    const sorted = this.rows;
    for (let i = 0; i < sorted.length; i++) {
      const row = sorted[i];
      const timeMs = row.native_onset_time_ms;
      row.prev_gap_ms = i > 0 ? timeMs - sorted[i - 1].native_onset_time_ms : FAR_GAP_MS;
      row.next_gap_ms = i + 1 < sorted.length ? sorted[i + 1].native_onset_time_ms - timeMs : FAR_GAP_MS;
      row.neighbor_count_500ms = sorted.reduce((count, other) => {
        const delta = Math.abs(other.native_onset_time_ms - timeMs);
        return delta > 0 && delta <= 500 ? count + 1 : count;
      }, 0);
    }
  }

  private classifyRow(row: BounceAudioCandidateRow, nowMs: number): boolean {
    if (nowMs - row.native_onset_time_ms > this.decisionConfig.staleMs) {
      row.decision = 'stale_backlog';
      row.reject_reason = 'stale_backlog';
      return true;
    }

    this.updateTimingFeatures();
    const pcm = this.pcmById.get(row.id);
    if (!pcm) {
      row.decision = 'js_error';
      row.reject_reason = 'missing_pcm';
      return true;
    }

    try {
      const featureStart = Date.now();
      const fableFeatures = extractFableFeatures(pcm);
      const featureEnd = Date.now();
      const fablePrediction = fablePredict(fableFeatures);
      const vector = buildBounceAudioFeatureVector(row, fableFeatures, fablePrediction, this.modelOption.model);
      const rfStart = Date.now();
      const prediction = predictWithRfModelRaw(this.modelOption.model, vector);
      const rfEnd = Date.now();
      const probability = positiveProbability(prediction, this.decisionConfig.positiveLabel);
      const runtimeConfig = this.runtimeConfig;

      row.feature_ms = featureEnd - featureStart;
      row.predict_ms = rfEnd - rfStart;
      row.fable_label = fablePrediction.label;
      row.fable_confidence = fablePrediction.confidence;
      row.fable_probabilities = fablePrediction.probabilities;
      row.classifier_label = prediction.label;
      row.classifier_confidence = prediction.confidence;
      row.classifier_probability = probability;
      row.classifier_probabilities = prediction.probabilities;
      row.feature_vector = vector;
      row.debug_explanation = buildDebugExplanation(
        row,
        fablePrediction,
        probability,
        runtimeConfig,
        this.modelOption.model,
        this.decisionConfig,
      );
      if (isHighConfidenceFableNoise(fablePrediction, runtimeConfig)) {
        row.decision = 'rejected_fable_noise_veto';
        row.reject_reason = 'fable_noise_veto';
      } else if (probability >= runtimeConfig.threshold) {
        row.decision = 'accepted_pending_dedupe';
        row.reject_reason = undefined;
      } else {
        row.decision = 'classified_low_probability';
        row.reject_reason = 'below_threshold';
      }
      this.pcmById.delete(row.id);
      return true;
    } catch (err) {
      row.decision = 'js_error';
      row.reject_reason = `js_error:${String(err).slice(0, 120)}`;
      this.pcmById.delete(row.id);
      return true;
    }
  }

  private applySmartDedupe(matureCutoffMs: number, final: boolean): BounceAudioCandidateRow[] {
    const threshold = this.runtimeConfig.threshold;
    const smartDedupeMs = this.decisionConfig.smartDedupeMs;
    const accepted = this.rows
      .filter(row => (row.classifier_probability ?? 0) >= threshold)
      .filter(row => row.decision !== 'rejected_fable_noise_veto')
      .sort((a, b) => a.native_onset_time_ms - b.native_onset_time_ms);

    for (const row of accepted) {
      if (!row.counted && row.decision !== 'deduped_lower_probability') {
        row.decision = 'accepted_pending_dedupe';
        row.reject_reason = undefined;
      }
    }

    const newlyCounted: BounceAudioCandidateRow[] = [];
    let cluster: BounceAudioCandidateRow[] = [];
    const flushCluster = (items: BounceAudioCandidateRow[]) => {
      if (items.length === 0) return;
      const lastTime = items[items.length - 1].native_onset_time_ms;
      const clusterIsClosed = final || lastTime <= matureCutoffMs - smartDedupeMs;
      if (!clusterIsClosed) return;
      const winner = items.reduce((best, row) => {
        const bestProb = best.classifier_probability ?? 0;
        const rowProb = row.classifier_probability ?? 0;
        if (rowProb > bestProb) return row;
        if (rowProb === bestProb && row.peak_value > best.peak_value) return row;
        return best;
      }, items[0]);
      for (const row of items) {
        if (row.id === winner.id) {
          row.counted = true;
          row.decision = 'counted';
          row.reject_reason = undefined;
          if (!this.emittedCountedIds.has(row.id)) {
            const previousTime = this.lastEmittedCountedOnsetMs;
            if (previousTime !== null) {
              const gapMs = row.native_onset_time_ms - previousTime;
              if (gapMs >= HEIGHT_MIN_GAP_MS && gapMs <= HEIGHT_MAX_GAP_MS) {
                row.bounce_gap_ms = gapMs;
                row.bounce_height_m = bounceHeightMeters(gapMs);
              }
            }
            this.lastEmittedCountedOnsetMs = row.native_onset_time_ms;
            this.emittedCountedIds.add(row.id);
            newlyCounted.push(row);
          }
        } else {
          row.counted = false;
          row.decision = 'deduped_lower_probability';
          row.reject_reason = 'deduped_lower_probability';
        }
      }
    };

    for (const row of accepted) {
      const last = cluster[cluster.length - 1];
      if (last && row.native_onset_time_ms - last.native_onset_time_ms > smartDedupeMs) {
        flushCluster(cluster);
        cluster = [];
      }
      cluster.push(row);
    }
    flushCluster(cluster);
    return newlyCounted;
  }
}

export const BOUNCE_AUDIO_TEST_MODEL_METADATA = DEFAULT_MODEL_OPTION.model.metadata ?? {};
