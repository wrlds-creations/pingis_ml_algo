import { extractFeatures } from './audioFeatures';
import { DEFAULT_AUDIO_DETECTION_CONFIG } from './audioDetectionConfig';
import { predictAudioContact } from './audioContactInference';
import { predictWithLogisticBinaryModel, type LogisticBinaryJsonModel } from './logisticRuntime';
import contactModelJson from './models/audio_contact_model.json';
import surfaceModelJson from './models/audio_model.json';
import hybridVetoModelJson from './models/hybrid_hard_negative_veto_t0129_logreg.json';
import { rfPredict } from './rfInference';
import { predictWithRfModelRaw, type RfJsonModel, type RfPrediction } from './rfRuntime';
import type {
  AudioContactLabel,
  AudioDetectionConfigSnapshot,
  AudioDetectionEvent,
  AudioDetectionMode,
} from './types';

const SURFACE_VETO_CONFIDENCE = 0.75;
const CONTACT_MODEL = contactModelJson as RfJsonModel;
const SURFACE_MODEL = surfaceModelJson as RfJsonModel;
const HYBRID_VETO_MODEL = hybridVetoModelJson as unknown as LogisticBinaryJsonModel;
const HYBRID22_DEFAULT_CONTACT_THRESHOLD = 0.25;
const HYBRID22_DEFAULT_VETO_THRESHOLD = 0.01;
const HYBRID22_DEFAULT_BYPASS_THRESHOLD = 0.61;
const VETO_SURFACE_LABELS = new Set(['floor_bounce', 'table_bounce', 'noise']);

interface ContactDecisionParams {
  detectedAtMs: number;
  pcm: Float32Array;
  confidenceThreshold?: number;
  dedupMs?: number;
  lastQualifiedTsMs?: number;
  surfaceVetoConfidence?: number;
  detectionMode?: AudioDetectionMode;
  config?: AudioDetectionConfigSnapshot;
  hybrid22ContactThreshold?: number;
  hybrid22VetoThreshold?: number;
  hybrid22VetoBypassEnabled?: boolean;
  hybrid22VetoBypassThreshold?: number;
}

function clampProbability(value: number | undefined, fallback: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) return fallback;
  return Math.max(0, Math.min(1, value));
}

function roundPrediction(prediction: RfPrediction): RfPrediction {
  return {
    label: prediction.label,
    confidence: Math.round(prediction.confidence * 1000) / 1000,
    probabilities: Object.fromEntries(
      Object.entries(prediction.probabilities).map(([label, probability]) => [
        label,
        Math.round(probability * 1000) / 1000,
      ]),
    ),
  };
}

function buildHybridVetoFeatures(
  features: Record<string, number>,
  contactPrediction: RfPrediction,
  surfacePrediction: RfPrediction,
) {
  const vetoFeatures: Record<string, number> = { ...features };
  for (const [label, probability] of Object.entries(contactPrediction.probabilities)) {
    vetoFeatures[`contact_prob_${label}`] = probability;
  }
  for (const [label, probability] of Object.entries(surfacePrediction.probabilities)) {
    vetoFeatures[`surface_prob_${label}`] = probability;
  }
  vetoFeatures.contact_confidence = contactPrediction.confidence;
  vetoFeatures.surface_confidence = surfacePrediction.confidence;
  return vetoFeatures;
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
  hybrid22ContactThreshold,
  hybrid22VetoThreshold,
  hybrid22VetoBypassEnabled,
  hybrid22VetoBypassThreshold,
}: ContactDecisionParams): AudioDetectionEvent {
  const activeConfig = config ?? DEFAULT_AUDIO_DETECTION_CONFIG;
  const activeDetectionMode = detectionMode ?? activeConfig.detection_mode;
  const activeConfidenceThreshold = activeDetectionMode === 'hybrid22'
    ? clampProbability(
        hybrid22ContactThreshold ?? confidenceThreshold,
        HYBRID22_DEFAULT_CONTACT_THRESHOLD,
      )
    : confidenceThreshold ?? activeConfig.contact_confidence_min;
  const activeDedupMs = dedupMs ?? activeConfig.merge_window_ms;
  const activeSurfaceVetoConfidence = surfaceVetoConfidence ??
    activeConfig.surface_veto_confidence ??
    SURFACE_VETO_CONFIDENCE;
  const features = extractFeatures(pcm);
  const binaryPrediction = predictAudioContact(features);
  const surfacePrediction = rfPredict(features);
  const rawContactPrediction = activeDetectionMode === 'hybrid22'
    ? predictWithRfModelRaw(CONTACT_MODEL, features)
    : undefined;
  const rawSurfacePrediction = activeDetectionMode === 'hybrid22'
    ? predictWithRfModelRaw(SURFACE_MODEL, features)
    : undefined;
  const racketContactProbability = rawContactPrediction?.probabilities.racket_contact ?? 0;
  const roundedContactPrediction = rawContactPrediction ? roundPrediction(rawContactPrediction) : undefined;
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
    : activeDetectionMode === 'hybrid22'
      ? {
          label: (
            racketContactProbability >= activeConfidenceThreshold
              ? 'racket_contact'
              : 'not_racket_contact'
          ) as AudioContactLabel,
          confidence: racketContactProbability,
          probabilities: roundedContactPrediction?.probabilities ?? {},
        }
      : {
          label: binaryPrediction.label as AudioContactLabel,
          confidence: binaryPrediction.confidence,
          probabilities: binaryPrediction.probabilities,
        };

  let qualified = true;
  let ignoredReason: AudioDetectionEvent['ignored_reason'];
  let hybridVetoBypassed = false;
  let hybridVetoBypassThreshold: number | undefined;
  let hybridVetoProbability: number | undefined;
  let hybridVetoThreshold: number | undefined;

  if (prediction.label !== 'racket_contact') {
    qualified = false;
    ignoredReason = 'not_racket_contact';
  } else if (
    (activeDetectionMode === 'hybrid' || activeDetectionMode === 'hybrid22') &&
    VETO_SURFACE_LABELS.has(surfacePrediction.label) &&
    surfacePrediction.confidence >= activeSurfaceVetoConfidence
  ) {
    qualified = false;
    ignoredReason = 'surface_veto';
  } else if (prediction.confidence < activeConfidenceThreshold) {
    qualified = false;
    ignoredReason = 'low_confidence';
  } else if (activeDetectionMode === 'hybrid22') {
    if (!rawContactPrediction || !rawSurfacePrediction) {
      qualified = false;
      ignoredReason = 'hybrid22_veto_unavailable';
    } else {
      hybridVetoBypassThreshold = clampProbability(
        hybrid22VetoBypassThreshold,
        HYBRID22_DEFAULT_BYPASS_THRESHOLD,
      );
      hybridVetoBypassed = Boolean(hybrid22VetoBypassEnabled) &&
        racketContactProbability > hybridVetoBypassThreshold;
      if (!hybridVetoBypassed) {
        const vetoPrediction = predictWithLogisticBinaryModel(
          HYBRID_VETO_MODEL,
          buildHybridVetoFeatures(features, rawContactPrediction, rawSurfacePrediction),
          clampProbability(hybrid22VetoThreshold, HYBRID22_DEFAULT_VETO_THRESHOLD),
        );
        hybridVetoProbability = vetoPrediction.probability;
        hybridVetoThreshold = vetoPrediction.threshold;
        if (vetoPrediction.label !== HYBRID_VETO_MODEL.positive_label) {
          qualified = false;
          ignoredReason = 'hybrid22_hard_negative_veto';
        }
      }
    }
  }

  if (qualified && lastQualifiedTsMs && detectedAtMs - lastQualifiedTsMs < activeDedupMs) {
    qualified = false;
    ignoredReason = 'dedup';
  }

  return {
    detected_at: new Date(detectedAtMs).toISOString(),
    ts_ms: detectedAtMs,
    label: prediction.label as AudioDetectionEvent['label'],
    confidence: prediction.confidence,
    probabilities: prediction.probabilities,
    contact_threshold: activeDetectionMode === 'hybrid' || activeDetectionMode === 'hybrid22'
      ? activeConfidenceThreshold
      : undefined,
    surface_label: surfacePrediction.label as AudioDetectionEvent['surface_label'],
    surface_confidence: surfacePrediction.confidence,
    surface_probabilities: surfacePrediction.probabilities,
    hybrid_veto_probability: hybridVetoProbability,
    hybrid_veto_threshold: hybridVetoThreshold,
    hybrid_veto_bypassed: hybridVetoBypassed || undefined,
    hybrid_veto_bypass_threshold: hybridVetoBypassThreshold,
    mid_band_energy: features.band_energy_mid,
    qualified,
    ignored_reason: ignoredReason,
  };
}
