import { extractFeatures } from './audioFeatures';
import { DEFAULT_AUDIO_DETECTION_CONFIG } from './audioDetectionConfig';
import { predictAudioContact } from './audioContactInference';
import { rfPredict } from './rfInference';
import type { AudioContactLabel, AudioDetectionConfigSnapshot, AudioDetectionEvent, AudioDetectionMode } from './types';

const SURFACE_VETO_CONFIDENCE = 0.75;

interface ContactDecisionParams {
  detectedAtMs: number;
  pcm: Float32Array;
  confidenceThreshold?: number;
  dedupMs?: number;
  lastQualifiedTsMs?: number;
  surfaceVetoConfidence?: number;
  detectionMode?: AudioDetectionMode;
  config?: AudioDetectionConfigSnapshot;
}

export function detectAudioContact({
  detectedAtMs,
  pcm,
  confidenceThreshold,
  dedupMs,
  lastQualifiedTsMs,
  surfaceVetoConfidence,
  detectionMode,
  config,
}: ContactDecisionParams): AudioDetectionEvent {
  const activeConfig = config ?? DEFAULT_AUDIO_DETECTION_CONFIG;
  const activeDetectionMode = detectionMode ?? activeConfig.detection_mode;
  const activeConfidenceThreshold = confidenceThreshold ?? activeConfig.contact_confidence_min;
  const activeDedupMs = dedupMs ?? activeConfig.merge_window_ms;
  const activeSurfaceVetoConfidence = surfaceVetoConfidence ?? activeConfig.surface_veto_confidence ?? SURFACE_VETO_CONFIDENCE;
  const features = extractFeatures(pcm);
  const binaryPrediction = predictAudioContact(features);
  const surfacePrediction = rfPredict(features);
  const surfaceSaysRacket = surfacePrediction.label === 'racket_bounce';
  const foldedSurfaceProbabilities: Record<string, number> = {
    racket_contact: surfacePrediction.probabilities.racket_bounce ?? 0,
    not_racket_contact: (
      (surfacePrediction.probabilities.table_bounce ?? 0) +
      (surfacePrediction.probabilities.floor_bounce ?? 0) +
      (surfacePrediction.probabilities.noise ?? 0)
    ),
  };
  const prediction = activeDetectionMode === 'four_class_only'
    ? {
        label: (surfaceSaysRacket ? 'racket_contact' : 'not_racket_contact') as AudioContactLabel,
        confidence: surfacePrediction.confidence,
        probabilities: foldedSurfaceProbabilities,
      }
    : {
        label: binaryPrediction.label as AudioContactLabel,
        confidence: binaryPrediction.confidence,
        probabilities: binaryPrediction.probabilities,
      };

  let qualified = true;
  let ignoredReason: AudioDetectionEvent['ignored_reason'];

  if (prediction.label !== 'racket_contact') {
    qualified = false;
    ignoredReason = 'not_racket_contact';
  } else if (
    activeDetectionMode === 'hybrid' &&
    (surfacePrediction.label === 'floor_bounce' ||
      surfacePrediction.label === 'table_bounce' ||
      surfacePrediction.label === 'noise') &&
    surfacePrediction.confidence >= activeSurfaceVetoConfidence
  ) {
    qualified = false;
    ignoredReason = 'surface_veto';
  } else if (prediction.confidence < activeConfidenceThreshold) {
    qualified = false;
    ignoredReason = 'low_confidence';
  } else if (lastQualifiedTsMs && detectedAtMs - lastQualifiedTsMs < activeDedupMs) {
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
