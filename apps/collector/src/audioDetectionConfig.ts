import type {
  AudioDetectionConfigSnapshot,
  AudioDetectionMode,
  AudioDetectionSensitivity,
} from './types';

export const COLLECTOR_BOUNCE_BASELINE_ID =
  'collector_bounce_live_v2026_05_28_tomas_stiga_candidate_normal_4class_220_80_220';

const MODEL_VERSIONS = {
  bundle_id: COLLECTOR_BOUNCE_BASELINE_ID,
  live_config_id: 'normal_four_class_100_200_retrigger220_group80_merge220_v1',
  audio_contact_model: 'collector_audio_contact_v2026_05_12_stable_debug',
  audio_model: 'collector_audio_4class_v2026_05_28_tomas_stiga_C_hybrid_window_candidate',
};

const onsetThresholdForRatio = (ratio: number) =>
  0.005 + ((ratio - 1.5) / (5.0 - 1.5)) * (0.15 - 0.005);

export const AUDIO_DETECTION_CONFIGS: Record<AudioDetectionSensitivity, AudioDetectionConfigSnapshot> = {
  strict: {
    config_id: 'strict_four_class_100_200_retrigger220_group80_merge280_v1',
    sensitivity: 'strict',
    detection_mode: 'four_class_only',
    contact_confidence_min: 0.76,
    surface_veto_confidence: 0.65,
    merge_window_ms: 280,
    onset_threshold: onsetThresholdForRatio(2.5),
    model_versions: MODEL_VERSIONS,
  },
  normal: {
    config_id: 'normal_four_class_100_200_retrigger220_group80_merge220_v1',
    sensitivity: 'normal',
    detection_mode: 'four_class_only',
    contact_confidence_min: 0.65,
    surface_veto_confidence: 0.75,
    merge_window_ms: 220,
    onset_threshold: onsetThresholdForRatio(1.5),
    model_versions: MODEL_VERSIONS,
  },
  sensitive: {
    config_id: 'sensitive_four_class_100_200_retrigger220_group80_merge180_v1',
    sensitivity: 'sensitive',
    detection_mode: 'four_class_only',
    contact_confidence_min: 0.5,
    surface_veto_confidence: 0.82,
    merge_window_ms: 180,
    onset_threshold: onsetThresholdForRatio(1.5),
    model_versions: MODEL_VERSIONS,
  },
};

export const DEFAULT_AUDIO_DETECTION_CONFIG = AUDIO_DETECTION_CONFIGS.normal;

export function getAudioDetectionConfig(
  sensitivity: AudioDetectionSensitivity = 'normal',
  detectionMode: AudioDetectionMode = 'four_class_only',
): AudioDetectionConfigSnapshot {
  const base = AUDIO_DETECTION_CONFIGS[sensitivity] ?? DEFAULT_AUDIO_DETECTION_CONFIG;
  return {
    ...base,
    config_id: `${sensitivity}_${detectionMode}_100_200_retrigger220_group80_merge${base.merge_window_ms}_v1`,
    detection_mode: detectionMode,
    model_versions: { ...base.model_versions },
  };
}

export function getDefaultAudioDetectionConfigSnapshot(): AudioDetectionConfigSnapshot {
  return getAudioDetectionConfig('normal', 'four_class_only');
}

export function detectionConfigTitle(config?: AudioDetectionConfigSnapshot): string {
  const snapshot = config ?? DEFAULT_AUDIO_DETECTION_CONFIG;
  const sensitivity = snapshot.sensitivity === 'strict'
    ? 'Strikt'
    : snapshot.sensitivity === 'sensitive'
      ? 'Känslig'
      : 'Normal';
  const mode = snapshot.detection_mode === 'four_class_only'
    ? '4-klass'
    : snapshot.detection_mode === 'binary_only'
      ? 'Binär'
      : 'Hybrid';
  return `${sensitivity} / ${mode}`;
}
