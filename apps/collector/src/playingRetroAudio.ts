import { extractFeatures } from './audioFeatures';
import { detectDenseAudioReviewCandidatePeaks, type AudioReviewCandidatePeak } from './audioReview';
import playingRetroAudioModelJson from './models/playing_retro_audio_model.json';
import { predictWithRfModel, type RfJsonModel, type RfPrediction } from './rfRuntime';
import type {
  AudioContactKind,
  AudioLabel,
  AudioModelCandidate,
  AudioNotRacketKind,
  AudioReviewClassLabel,
  AudioReviewEventType,
  AudioReviewLabel,
} from './types';

export const PLAYING_RETRO_AUDIO_SAMPLE_RATE = 22050;
export const PLAYING_RETRO_AUDIO_MODEL_ROLE = 'spel_retro_audio_review_only';

export type PlayingRetroAudioLabel = 'non_target' | 'racket_contact' | 'table_bounce';

export interface PlayingRetroAudioWindowSpec {
  name: 'tight' | 'normal' | 'wide';
  before_ms: number;
  after_ms: number;
}

export interface PlayingRetroAudioMetadata {
  model_version: string;
  feature_version: string;
  app_model_role: typeof PLAYING_RETRO_AUDIO_MODEL_ROLE;
  selected_variant: string;
  sample_rate_hz: number;
  windows: PlayingRetroAudioWindowSpec[];
  context_feature_names: string[];
  review_thresholds?: {
    racket_contact?: number;
    table_bounce?: number;
    same_label_dedupe_ms?: number;
    source_ticket?: string;
  };
}

type PlayingRetroAudioModel = RfJsonModel & {
  metadata: PlayingRetroAudioMetadata;
};

export interface PlayingRetroAudioPrediction {
  label: PlayingRetroAudioLabel;
  confidence: number;
  probabilities: Record<string, number>;
  model_version: string;
  feature_version: string;
}

export interface PlayingRetroAudioReviewCandidate extends AudioModelCandidate {
  source_candidate_id: string;
  playing_retro_prediction: PlayingRetroAudioPrediction;
  playing_retro_candidate_source?: 'saved_candidate' | 'recovery_candidate';
  playing_retro_recovery_score?: number;
  playing_retro_nearest_saved_gap_ms?: number | null;
}

export interface PlayingRetroAudioAnalysisResult {
  model_version: string;
  feature_version: string;
  feature_count: number;
  candidate_count: number;
  saved_candidate_count: number;
  recovery_candidate_count: number;
  visible_recovery_candidate_count: number;
  candidates: PlayingRetroAudioReviewCandidate[];
}

const MODEL = playingRetroAudioModelJson as PlayingRetroAudioModel;
const WINDOWS = MODEL.metadata.windows;
const REVIEW_THRESHOLDS = {
  racket_contact: MODEL.metadata.review_thresholds?.racket_contact ?? 0.0,
  table_bounce: MODEL.metadata.review_thresholds?.table_bounce ?? 0.5,
};
const RECOVERY_MIN_GAP_FROM_KNOWN_MS = 32;
const RECOVERY_MIN_GAP_FROM_RECOVERY_MS = 48;
const RECOVERY_MAX_GAP_FROM_SAVED_MS = 520;
const RECOVERY_DENSE_CANDIDATE_GAP_MS = 28;
const RECOVERY_MAX_CANDIDATES = 220;
const RECOVERY_RACKET_MIN_CONFIDENCE = 0.8;
const RECOVERY_TABLE_MIN_CONFIDENCE = 0.54;
const RECOVERY_RACKET_MIN_SAVED_GAP_MS = 120;
const RECOVERY_TABLE_MIN_SAVED_GAP_MS = 60;

function assertTargetSampleRate(sampleRate: number): void {
  if (Math.round(sampleRate) !== PLAYING_RETRO_AUDIO_SAMPLE_RATE) {
    throw new Error(
      `playingRetroAudio expects ${PLAYING_RETRO_AUDIO_SAMPLE_RATE} Hz PCM, got ${sampleRate}`,
    );
  }
}

function extractWindow(
  samples: Float32Array,
  sampleRate: number,
  anchorMs: number,
  beforeMs: number,
  afterMs: number,
): Float32Array {
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

function prefixedFeatures(features: Record<string, number>, prefix: string): Record<string, number> {
  const result: Record<string, number> = {};
  for (const [key, value] of Object.entries(features)) {
    result[`${prefix}_${key}`] = value;
  }
  return result;
}

function uniqueSortedTimestamps(timestamps: number[]): number[] {
  return Array.from(new Set(timestamps.map(timestamp => Math.round(timestamp)))).sort((a, b) => a - b);
}

function nearestGapMs(anchorMs: number, timestamps: number[]): number | null {
  if (timestamps.length === 0) return null;
  let nearest = Number.POSITIVE_INFINITY;
  for (const timestamp of timestamps) {
    nearest = Math.min(nearest, Math.abs(Math.round(anchorMs) - Math.round(timestamp)));
  }
  return Number.isFinite(nearest) ? nearest : null;
}

function isFarEnough(anchorMs: number, timestamps: number[], minGapMs: number): boolean {
  const gap = nearestGapMs(anchorMs, timestamps);
  return gap === null || gap >= minGapMs;
}

function clippedGap(value: number | null): number {
  return value === null ? 1.0 : Math.min(value, 1000) / 1000;
}

function buildPlayingRetroAudioContextFeaturesFromSorted(
  anchorMs: number,
  timestamps: number[],
  isSavedCandidate = true,
): Record<string, number> {
  const roundedAnchor = Math.round(anchorMs);
  const prevGaps = timestamps
    .filter(timestamp => timestamp < roundedAnchor)
    .map(timestamp => roundedAnchor - timestamp);
  const nextGaps = timestamps
    .filter(timestamp => timestamp > roundedAnchor)
    .map(timestamp => timestamp - roundedAnchor);
  const prevGap = prevGaps.length > 0 ? Math.min(...prevGaps) : null;
  const nextGap = nextGaps.length > 0 ? Math.min(...nextGaps) : null;
  const nearestGap = Math.min(...[prevGap, nextGap].filter((gap): gap is number => gap !== null));
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

export function buildPlayingRetroAudioContextFeatures(
  anchorMs: number,
  candidateTimestampsMs: number[],
  isSavedCandidate = true,
): Record<string, number> {
  return buildPlayingRetroAudioContextFeaturesFromSorted(
    anchorMs,
    uniqueSortedTimestamps(candidateTimestampsMs),
    isSavedCandidate,
  );
}

export function buildPlayingRetroAudioFeatureVector(
  samples: Float32Array,
  sampleRate: number,
  anchorMs: number,
  candidateTimestampsMs: number[],
  isSavedCandidate = true,
  candidateTimestampsAlreadySorted = false,
): Record<string, number> {
  assertTargetSampleRate(sampleRate);
  const features: Record<string, number> = {};
  for (const windowSpec of WINDOWS) {
    const clip = extractWindow(
      samples,
      sampleRate,
      anchorMs,
      windowSpec.before_ms,
      windowSpec.after_ms,
    );
    Object.assign(features, prefixedFeatures(extractFeatures(clip), windowSpec.name));
  }
  Object.assign(
    features,
    buildPlayingRetroAudioContextFeaturesFromSorted(
      anchorMs,
      candidateTimestampsAlreadySorted ? candidateTimestampsMs : uniqueSortedTimestamps(candidateTimestampsMs),
      isSavedCandidate,
    ),
  );
  return features;
}

export function missingPlayingRetroAudioFeatureNames(features: Record<string, number>): string[] {
  return MODEL.feature_names.filter(featureName => features[featureName] === undefined);
}

export function predictPlayingRetroAudioFeatures(
  features: Record<string, number>,
): PlayingRetroAudioPrediction {
  const missing = missingPlayingRetroAudioFeatureNames(features);
  if (missing.length > 0) {
    throw new Error(`playingRetroAudio missing model features: ${missing.slice(0, 10).join(', ')}`);
  }
  const prediction: RfPrediction = predictWithRfModel(MODEL, features);
  return {
    label: prediction.label as PlayingRetroAudioLabel,
    confidence: prediction.confidence,
    probabilities: prediction.probabilities,
    model_version: MODEL.metadata.model_version,
    feature_version: MODEL.metadata.feature_version,
  };
}

export function predictPlayingRetroAudioAt(
  samples: Float32Array,
  sampleRate: number,
  anchorMs: number,
  candidateTimestampsMs: number[],
  isSavedCandidate = true,
  candidateTimestampsAlreadySorted = false,
): PlayingRetroAudioPrediction {
  return predictPlayingRetroAudioFeatures(
    buildPlayingRetroAudioFeatureVector(
      samples,
      sampleRate,
      anchorMs,
      candidateTimestampsMs,
      isSavedCandidate,
      candidateTimestampsAlreadySorted,
    ),
  );
}

function predictionClearsReviewThresholds(
  prediction: PlayingRetroAudioPrediction,
  options: { recovery?: boolean; nearestSavedGapMs?: number | null } = {},
): boolean {
  if (
    prediction.label === 'racket_contact' &&
    prediction.confidence < REVIEW_THRESHOLDS.racket_contact
  ) {
    return false;
  }
  if (
    prediction.label === 'table_bounce' &&
    prediction.confidence < REVIEW_THRESHOLDS.table_bounce
  ) {
    return false;
  }
  if (!options.recovery) return true;
  const nearestSavedGapMs = options.nearestSavedGapMs ?? null;
  if (prediction.label === 'racket_contact') {
    return prediction.confidence >= RECOVERY_RACKET_MIN_CONFIDENCE &&
      (nearestSavedGapMs === null || nearestSavedGapMs >= RECOVERY_RACKET_MIN_SAVED_GAP_MS);
  }
  if (prediction.label === 'table_bounce') {
    return prediction.confidence >= RECOVERY_TABLE_MIN_CONFIDENCE &&
      (nearestSavedGapMs === null || nearestSavedGapMs >= RECOVERY_TABLE_MIN_SAVED_GAP_MS);
  }
  return true;
}

function reviewFieldsForPrediction(
  prediction: PlayingRetroAudioPrediction,
  options: { recovery?: boolean; nearestSavedGapMs?: number | null } = {},
): {
  review_relevant: boolean;
  suggested_label: AudioReviewLabel;
  event_type: AudioReviewEventType;
  class_label: AudioReviewClassLabel;
  contact_kind?: AudioContactKind;
  not_racket_kind?: AudioNotRacketKind;
  surface_label?: AudioLabel;
} {
  if (!predictionClearsReviewThresholds(prediction, options)) {
    return {
      review_relevant: false,
      suggested_label: 'ignore',
      event_type: 'ignore',
      class_label: 'ignore',
    };
  }
  if (prediction.label === 'racket_contact') {
    return {
      review_relevant: true,
      suggested_label: 'racket_contact',
      event_type: 'racket_hit',
      class_label: 'racket_bounce',
      contact_kind: 'racket_bounce',
    };
  }
  if (prediction.label === 'table_bounce') {
    return {
      review_relevant: true,
      suggested_label: 'not_racket_contact',
      event_type: 'bounce',
      class_label: 'table_bounce',
      not_racket_kind: 'table_bounce',
      surface_label: 'table_bounce',
    };
  }
  return {
    review_relevant: false,
    suggested_label: 'ignore',
    event_type: 'ignore',
    class_label: 'ignore',
  };
}

export function buildPlayingRetroAudioReviewCandidate(
  candidate: AudioModelCandidate,
  prediction: PlayingRetroAudioPrediction,
  options: {
    source?: 'saved_candidate' | 'recovery_candidate';
    recoveryScore?: number;
    nearestSavedGapMs?: number | null;
  } = {},
): PlayingRetroAudioReviewCandidate {
  const source = options.source ?? 'saved_candidate';
  const reviewFields = reviewFieldsForPrediction(prediction, {
    recovery: source === 'recovery_candidate',
    nearestSavedGapMs: options.nearestSavedGapMs,
  });
  return {
    id: source === 'recovery_candidate' ? candidate.id : `playing_retro_${candidate.id}`,
    timestamp_ms: candidate.timestamp_ms,
    ...reviewFields,
    contact_confidence: prediction.label === 'racket_contact' ? prediction.confidence : undefined,
    surface_confidence: prediction.label === 'table_bounce' ? prediction.confidence : undefined,
    detection_config_id: MODEL.metadata.model_version,
    source_candidate_id: candidate.id,
    playing_retro_prediction: prediction,
    playing_retro_candidate_source: source,
    playing_retro_recovery_score: options.recoveryScore,
    playing_retro_nearest_saved_gap_ms: options.nearestSavedGapMs,
  };
}

function candidateFromRecoveryPeak(
  peak: AudioReviewCandidatePeak,
  index: number,
  nearestSavedGapMs: number | null,
): AudioModelCandidate & { playing_retro_nearest_saved_gap_ms?: number | null } {
  const timestampMs = Math.round(peak.refined_timestamp_ms);
  return {
    id: `playing_retro_recovery_${index}_${timestampMs}`,
    timestamp_ms: timestampMs,
    review_relevant: false,
    suggested_label: 'ignore',
    event_type: 'ignore',
    class_label: 'ignore',
    detection_config_id: `${MODEL.metadata.model_version}:recovery_v1`,
    playing_retro_nearest_saved_gap_ms: nearestSavedGapMs,
  };
}

export function findPlayingRetroAudioRecoveryCandidates(
  samples: Float32Array,
  sampleRate: number,
  savedCandidates: AudioModelCandidate[],
  blockedTimestampsMs: number[] = [],
): AudioModelCandidate[] {
  assertTargetSampleRate(sampleRate);
  const savedTimestamps = uniqueSortedTimestamps(savedCandidates.map(candidate => candidate.timestamp_ms));
  const knownTimestamps = uniqueSortedTimestamps([...savedTimestamps, ...blockedTimestampsMs]);
  const peaks = detectDenseAudioReviewCandidatePeaks(samples, sampleRate, {
    minCandidateGapMs: RECOVERY_DENSE_CANDIDATE_GAP_MS,
  })
    .map((peak, index) => ({ ...peak, index, refined_timestamp_ms: Math.round(peak.refined_timestamp_ms) }))
    .filter(peak => isFarEnough(peak.refined_timestamp_ms, knownTimestamps, RECOVERY_MIN_GAP_FROM_KNOWN_MS))
    .filter(peak => {
      const savedGap = nearestGapMs(peak.refined_timestamp_ms, savedTimestamps);
      return savedGap === null || savedGap <= RECOVERY_MAX_GAP_FROM_SAVED_MS;
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, RECOVERY_MAX_CANDIDATES);

  const selected: typeof peaks = [];
  for (const peak of peaks) {
    if (isFarEnough(peak.refined_timestamp_ms, selected.map(item => item.refined_timestamp_ms), RECOVERY_MIN_GAP_FROM_RECOVERY_MS)) {
      selected.push(peak);
    }
  }

  return selected
    .sort((a, b) => a.refined_timestamp_ms - b.refined_timestamp_ms)
    .map((peak, index) => ({
      ...candidateFromRecoveryPeak(peak, index, nearestGapMs(peak.refined_timestamp_ms, savedTimestamps)),
      playing_retro_recovery_score: peak.score,
    } as AudioModelCandidate));
}

export function analyzePlayingRetroAudioCandidates(
  samples: Float32Array,
  sampleRate: number,
  candidates: AudioModelCandidate[],
  options: {
    recoverMissingCandidates?: boolean;
    blockedTimestampsMs?: number[];
    onProfileEvent?: (phase: string, detail?: string, durationMs?: number) => void;
  } = {},
): PlayingRetroAudioAnalysisResult {
  const recoveryStartedAt = Date.now();
  const recoveryCandidates = options.recoverMissingCandidates
    ? findPlayingRetroAudioRecoveryCandidates(
      samples,
      sampleRate,
      candidates,
      options.blockedTimestampsMs ?? [],
    )
    : [];
  options.onProfileEvent?.(
    'recovery_candidates_ready',
    `saved=${candidates.length} recovery=${recoveryCandidates.length}`,
    Date.now() - recoveryStartedAt,
  );
  const allCandidateInputs = [...candidates, ...recoveryCandidates];
  const candidateTimestamps = uniqueSortedTimestamps(allCandidateInputs.map(candidate => candidate.timestamp_ms));
  const predictionStartedAt = Date.now();
  const retroCandidates = allCandidateInputs.map(candidate => {
    const source = candidate.id.startsWith('playing_retro_recovery_') ? 'recovery_candidate' : 'saved_candidate';
    const prediction = predictPlayingRetroAudioAt(
      samples,
      sampleRate,
      candidate.timestamp_ms,
      candidateTimestamps,
      source === 'saved_candidate',
      true,
    );
    return buildPlayingRetroAudioReviewCandidate(candidate, prediction, {
      source,
      recoveryScore: (candidate as AudioModelCandidate & { playing_retro_recovery_score?: number }).playing_retro_recovery_score,
      nearestSavedGapMs: (
        candidate as AudioModelCandidate & { playing_retro_nearest_saved_gap_ms?: number | null }
      ).playing_retro_nearest_saved_gap_ms,
    });
  });
  options.onProfileEvent?.(
    'rf_predictions_ready',
    `classified=${retroCandidates.length}`,
    Date.now() - predictionStartedAt,
  );
  const visibleRecoveryCount = retroCandidates.filter(candidate => (
    candidate.playing_retro_candidate_source === 'recovery_candidate' && candidate.review_relevant
  )).length;
  return {
    model_version: MODEL.metadata.model_version,
    feature_version: MODEL.metadata.feature_version,
    feature_count: MODEL.feature_names.length,
    candidate_count: retroCandidates.length,
    saved_candidate_count: candidates.length,
    recovery_candidate_count: recoveryCandidates.length,
    visible_recovery_candidate_count: visibleRecoveryCount,
    candidates: retroCandidates,
  };
}

export function getPlayingRetroAudioModelMetadata(): PlayingRetroAudioMetadata {
  return MODEL.metadata;
}
