// Shared types for the Pingis Collector app.

export interface Vector3 {
  x: number;
  y: number;
  z: number;
}

export interface ImuSample {
  accel_x: number;
  accel_y: number;
  accel_z: number;
  gyro_x: number;
  gyro_y: number;
  gyro_z: number;
  mag_x: number;
  mag_y: number;
  mag_z: number;
  ts_ms: number;
}

export interface PlayerSetup {
  name: string;
  handedness: 'right' | 'left';
}

export interface TableCalibration {
  gravity: Vector3;
  gyro_bias: Vector3;
  captured_at: string;
}

export interface BounceSideCalibration {
  side: 'forehand' | 'backhand';
  pose_accel: Vector3;
  captured_at: string;
}

export interface CalibrationProfile {
  calibration_id: string;
  captured_at: string;
  gravity: Vector3;
  gyro_bias: Vector3;
  table: TableCalibration;
  bounce_sides?: {
    forehand: BounceSideCalibration;
    backhand: BounceSideCalibration;
  };
}

export type CalibrationData = CalibrationProfile;
export type CalibrationMode = 'table_only' | 'bounce_sides';

export interface LabeledEvent {
  label: 'hit' | 'swing_miss' | 'idle';
  stroke_type: 'forehand' | 'backhand' | 'unknown';
  recorded_at: string;
  samples: ImuSample[];
}

export interface SessionFile {
  session_meta: {
    player_name: string;
    handedness: 'right' | 'left';
    calibration_accel: Vector3;
    calibration_gyro_bias: Vector3;
    calibration_id?: string;
    session_date: string;
    app_version: string;
  };
  events: LabeledEvent[];
}

// ---- Audio bounce detection ----

export type AudioLabel = 'racket_bounce' | 'table_bounce' | 'floor_bounce' | 'noise';
export type AudioBackgroundCondition = 'quiet' | 'speech' | 'music_low' | 'music_mid' | 'desk';
export type AudioContactLabel = 'racket_contact' | 'not_racket_contact';
export type AudioReviewLabel = AudioContactLabel | 'ignore';
export type AudioReviewSource = 'auto' | 'manual';
export type AudioReviewAnchorRule = 'attack_start';
export type AudioScenarioId =
  | 'racket_quiet'
  | 'racket_counting'
  | 'racket_music_low'
  | 'racket_music_mid'
  | 'speech_only'
  | 'desk_keyboard_only'
  | 'music_low_only'
  | 'music_mid_only'
  | 'table_quiet'
  | 'floor_quiet';

export interface AudioReviewMarker {
  id: string;
  timestamp_ms: number;
  source: AudioReviewSource;
  suggested_label: AudioReviewLabel;
  final_label: AudioReviewLabel;
}

export interface AudioTakeReview {
  required: boolean;
  anchor_rule: AudioReviewAnchorRule;
  completed_at?: string;
  markers: AudioReviewMarker[];
}

export interface AudioImuRecording {
  started_at_ms: number;
  ended_at_ms: number;
  sample_hz_estimate: number;
  sample_count: number;
  samples: ImuSample[];
}

export interface AudioVideoRecording {
  video_filename: string;
  started_at_ms: number;
  ended_at_ms: number;
  duration_ms: number;
  audio_origin_in_video_ms: number;
}

export interface AudioEvent {
  label: AudioLabel;
  recorded_at: string;
  wav_filename: string;
  duration_ms: number;
  scenario_id: AudioScenarioId;
  background_condition: AudioBackgroundCondition;
  take_index: number;
  target_duration_s: number;
  review?: AudioTakeReview;
  imu_recording?: AudioImuRecording;
  video_recording?: AudioVideoRecording;
}

export interface AudioCollectionScenarioSummary {
  scenario_id: AudioScenarioId;
  label: AudioLabel;
  target_takes: number;
  completed_takes: number;
  remaining_takes: number;
}

export interface AudioCollectionSummary {
  total_scenarios: number;
  completed_scenarios: number;
  total_takes: number;
  completed_takes: number;
  remaining_takes: number;
  pending_review_takes: number;
  reviewed_takes: number;
  auto_saved_takes: number;
}

export interface AudioSessionFile {
  session_meta: {
    recorder_name: string;
    player_name?: string;
    handedness?: 'right' | 'left';
    session_date: string;
    app_version: string;
    clip_duration_ms: number;
    collection_mode: 'guided_scenarios' | 'guided_scenarios_audio_imu';
    target_duration_s: number;
    planned_takes: number;
    calibration_id?: string;
  };
  calibration_profile?: CalibrationData;
  events: AudioEvent[];
}

// ---- Test modes ----

export type TestMode = 'bounce_free' | 'bounce_alternating' | 'stroke_debug';
export type BouncePresetId = 'B0' | 'B1' | 'B2';
export type StrokePresetId = 'S0' | 'S1' | 'S2';

export type BounceSide = 'forehand' | 'backhand' | 'uncertain';
export type StrokeLabel = 'hit' | 'swing_miss' | 'idle';
export type StrokeCombinedLabel = 'fh_hit' | 'bh_hit' | 'fh_miss' | 'bh_miss' | 'idle';

export interface BounceSettings {
  audioThreshold: number;
  audioConfidence: number;
  audioDedupMs: number;
  motionWindowMs: number;
  motionGyroThreshold: number;
  motionAccelThreshold: number;
  orientationSampleWindowMs: number;
  orientationDeadzone: number;
}

export interface BounceMotionMetrics {
  gyro_peak: number;
  accel_peak: number;
  accel_ptp: number;
}

export interface StrokeSettings {
  motionGyroThreshold: number;
  motionAccelThreshold: number;
  modelThreshold: number;
  imuDedupMs: number;
  audioDebugThreshold: number;
  audioDebugConfidence: number;
}

export interface StrokeMotionMetrics {
  gyro_peak: number;
  gyro_std: number;
  accel_peak: number;
  accel_ptp: number;
}

export interface AudioDetectionEvent {
  detected_at: string;
  ts_ms: number;
  label: AudioContactLabel;
  confidence: number;
  probabilities: Record<string, number>;
  surface_label?: AudioLabel;
  surface_confidence?: number;
  surface_probabilities?: Record<string, number>;
  qualified: boolean;
  ignored_reason?: 'not_racket_contact' | 'low_confidence' | 'dedup' | 'surface_veto';
}

export interface BounceSideEvent {
  detected_at: string;
  ts_ms: number;
  side: BounceSide;
  orientation: Vector3;
  forehand_score: number;
  backhand_score: number;
}

export interface BounceContactEvent {
  detected_at: string;
  ts_ms: number;
  mode: 'bounce_free' | 'bounce_alternating';
  audio_label: AudioContactLabel;
  audio_confidence: number;
  surface_label?: AudioLabel;
  surface_confidence?: number;
  motion_gate_open: boolean;
  motion_metrics: BounceMotionMetrics;
  side: BounceSide;
  orientation: Vector3;
  forehand_score: number;
  backhand_score: number;
  counted: boolean;
  total_after: number;
  alternation_after: number;
  ignored_reason?: 'not_racket_contact' | 'low_confidence' | 'dedup' | 'surface_veto';
}

export interface StrokeInferenceEvent {
  detected_at: string;
  ts_ms: number;
  label: StrokeCombinedLabel;
  motion_gate_open: boolean;
  motion_metrics: StrokeMotionMetrics;
  hit_label: StrokeLabel;
  hit_confidence: number;
  hit_probabilities: Record<string, number>;
  stroke_side: BounceSide;
  stroke_confidence: number;
  stroke_probabilities: Record<string, number>;
  counted: boolean;
  ignored_reason?: 'idle_motion_gate' | 'model_low_confidence' | 'dedup' | 'side_uncertain';
  audio_support?: {
    label: AudioContactLabel;
    confidence: number;
    delta_ms: number;
  } | null;
}

export interface BounceTestSessionFile {
  session_meta: {
    player_name: string;
    handedness: 'right' | 'left';
    mode: 'bounce_free' | 'bounce_alternating';
    session_date: string;
    duration_ms: number;
    app_version: string;
  };
  calibration_profile: CalibrationProfile;
  calibration_summary: {
    table_ready: boolean;
    bounce_sides_ready: boolean;
  };
  preset_id: BouncePresetId;
  settings: BounceSettings;
  samples: ImuSample[];
  audio_events: AudioDetectionEvent[];
  bounce_side_events: BounceSideEvent[];
  bounce_contacts: BounceContactEvent[];
  summary: {
    total_count: number;
    forehand_count: number;
    backhand_count: number;
    uncertain_count: number;
    alternation_count: number;
  };
}

export interface StrokeTestSessionFile {
  session_meta: {
    player_name: string;
    handedness: 'right' | 'left';
    mode: 'stroke_debug';
    session_date: string;
    duration_ms: number;
    app_version: string;
  };
  calibration_profile: CalibrationProfile;
  calibration_summary: {
    table_ready: boolean;
    bounce_sides_ready: boolean;
  };
  preset_id: StrokePresetId;
  settings: StrokeSettings;
  samples: ImuSample[];
  audio_events: AudioDetectionEvent[];
  stroke_events: StrokeInferenceEvent[];
  summary: {
    fh_hit_count: number;
    bh_hit_count: number;
    fh_miss_count: number;
    bh_miss_count: number;
    idle_gated_windows: number;
    motion_windows: number;
    counted_events: number;
  };
}
