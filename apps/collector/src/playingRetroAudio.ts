import { extractFeatures } from './audioFeatures';
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
}

export interface PlayingRetroAudioAnalysisResult {
  model_version: string;
  feature_version: string;
  feature_count: number;
  candidate_count: number;
  candidates: PlayingRetroAudioReviewCandidate[];
}

const MODEL = playingRetroAudioModelJson as PlayingRetroAudioModel;
const WINDOWS = MODEL.metadata.windows;

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

function clippedGap(value: number | null): number {
  return value === null ? 1.0 : Math.min(value, 1000) / 1000;
}

export function buildPlayingRetroAudioContextFeatures(
  anchorMs: number,
  candidateTimestampsMs: number[],
  isSavedCandidate = true,
): Record<string, number> {
  const timestamps = uniqueSortedTimestamps(candidateTimestampsMs);
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

export function buildPlayingRetroAudioFeatureVector(
  samples: Float32Array,
  sampleRate: number,
  anchorMs: number,
  candidateTimestampsMs: number[],
  isSavedCandidate = true,
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
    buildPlayingRetroAudioContextFeatures(anchorMs, candidateTimestampsMs, isSavedCandidate),
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
): PlayingRetroAudioPrediction {
  return predictPlayingRetroAudioFeatures(
    buildPlayingRetroAudioFeatureVector(
      samples,
      sampleRate,
      anchorMs,
      candidateTimestampsMs,
      isSavedCandidate,
    ),
  );
}

function reviewFieldsForPrediction(prediction: PlayingRetroAudioPrediction): {
  review_relevant: boolean;
  suggested_label: AudioReviewLabel;
  event_type: AudioReviewEventType;
  class_label: AudioReviewClassLabel;
  contact_kind?: AudioContactKind;
  not_racket_kind?: AudioNotRacketKind;
  surface_label?: AudioLabel;
} {
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
): PlayingRetroAudioReviewCandidate {
  const reviewFields = reviewFieldsForPrediction(prediction);
  return {
    id: `playing_retro_${candidate.id}`,
    timestamp_ms: candidate.timestamp_ms,
    ...reviewFields,
    contact_confidence: prediction.label === 'racket_contact' ? prediction.confidence : undefined,
    surface_confidence: prediction.label === 'table_bounce' ? prediction.confidence : undefined,
    detection_config_id: MODEL.metadata.model_version,
    source_candidate_id: candidate.id,
    playing_retro_prediction: prediction,
  };
}

export function analyzePlayingRetroAudioCandidates(
  samples: Float32Array,
  sampleRate: number,
  candidates: AudioModelCandidate[],
): PlayingRetroAudioAnalysisResult {
  const candidateTimestamps = uniqueSortedTimestamps(candidates.map(candidate => candidate.timestamp_ms));
  const retroCandidates = candidates.map(candidate => {
    const prediction = predictPlayingRetroAudioAt(
      samples,
      sampleRate,
      candidate.timestamp_ms,
      candidateTimestamps,
      true,
    );
    return buildPlayingRetroAudioReviewCandidate(candidate, prediction);
  });
  return {
    model_version: MODEL.metadata.model_version,
    feature_version: MODEL.metadata.feature_version,
    feature_count: MODEL.feature_names.length,
    candidate_count: candidates.length,
    candidates: retroCandidates,
  };
}

export function getPlayingRetroAudioModelMetadata(): PlayingRetroAudioMetadata {
  return MODEL.metadata;
}
