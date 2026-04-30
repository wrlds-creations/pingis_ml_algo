import { extractFeatures } from './audioFeatures';
import { predictAudioContact } from './audioContactInference';
import { rfPredict } from './rfInference';
import type { AudioDetectionEvent } from './types';

const SURFACE_VETO_CONFIDENCE = 0.75;

interface ContactDecisionParams {
  detectedAtMs: number;
  pcm: Float32Array;
  confidenceThreshold: number;
  dedupMs: number;
  lastQualifiedTsMs?: number;
}

export function detectAudioContact({
  detectedAtMs,
  pcm,
  confidenceThreshold,
  dedupMs,
  lastQualifiedTsMs,
}: ContactDecisionParams): AudioDetectionEvent {
  const features = extractFeatures(pcm);
  const prediction = predictAudioContact(features);
  const surfacePrediction = rfPredict(features);

  let qualified = true;
  let ignoredReason: AudioDetectionEvent['ignored_reason'];

  if (prediction.label !== 'racket_contact') {
    qualified = false;
    ignoredReason = 'not_racket_contact';
  } else if (
    (surfacePrediction.label === 'floor_bounce' || surfacePrediction.label === 'table_bounce') &&
    surfacePrediction.confidence >= SURFACE_VETO_CONFIDENCE
  ) {
    qualified = false;
    ignoredReason = 'surface_veto';
  } else if (prediction.confidence < confidenceThreshold) {
    qualified = false;
    ignoredReason = 'low_confidence';
  } else if (lastQualifiedTsMs && detectedAtMs - lastQualifiedTsMs < dedupMs) {
    qualified = false;
    ignoredReason = 'dedup';
  }

  return {
    detected_at: new Date(detectedAtMs).toISOString(),
    ts_ms: detectedAtMs,
    label: prediction.label as AudioDetectionEvent['label'],
    confidence: prediction.confidence,
    probabilities: prediction.probabilities,
    surface_label: surfacePrediction.label as AudioDetectionEvent['surface_label'],
    surface_confidence: surfacePrediction.confidence,
    surface_probabilities: surfacePrediction.probabilities,
    qualified,
    ignored_reason: ignoredReason,
  };
}
