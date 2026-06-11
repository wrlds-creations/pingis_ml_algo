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
  received_at_ms?: number;
  take_ts_ms?: number;
  sensor_ts?: number;
  ts_ms: number;
}

export interface PlayerSetup {
  name: string;
  handedness: 'right' | 'left';
}

// ---- Video stroke classification ----

export type VideoStrokeType = 'forehand' | 'backhand';
export type VideoStrokeMarkerType = VideoStrokeType | 'unknown';
export type VideoStrokePredictionLabel = VideoStrokeType | 'uncertain';
export type VideoStrokeCameraFacing = 'front' | 'back';
export type VideoStrokeCameraAngle = 'front_oblique';
export type VideoStrokeCameraSide = 'player_left' | 'player_right' | 'center_front' | 'unknown';
export type VideoStrokeReviewStatus = 'pending' | 'reviewed';
export type VideoStrokeAnalysisStatus = 'ok' | 'uncertain' | 'model_missing' | 'insufficient_pose';
export type VideoStrokeRecordingMode = 'fixed_30s' | 'continuous_segment' | 'long_source_segment' | 'imported_source';

export interface VideoStrokeMarker {
  id: string;
  timestamp_ms: number;
  stroke_type: VideoStrokeMarkerType;
  source: 'manual' | 'model' | 'audio_peak';
  review_status: 'confirmed' | 'suggested';
  created_at: string;
}

export interface VideoPoseLandmark {
  type: number;
  x: number;
  y: number;
  z: number;
  visibility: number;
}

export interface VideoPoseFrame {
  timestamp_ms: number;
  pose_detected: boolean;
  landmarks: VideoPoseLandmark[];
}

export interface VideoPoseExtractionResult {
  video_path: string;
  sample_fps: number;
  duration_ms: number;
  frame_count: number;
  frames: VideoPoseFrame[];
}

export interface VideoStrokePoseAnalysis {
  marker_id: string;
  timestamp_ms: number;
  predicted_stroke_type: VideoStrokePredictionLabel;
  confidence: number;
  probabilities: Record<string, number>;
  model_version: string;
  feature_spec: 'video_stroke_features_v1' | 'video_stroke_features_v2';
  status: VideoStrokeAnalysisStatus;
}

export interface VideoStrokeTake {
  video_filename: string;
  duration_ms: number;
  take_index: number;
  recording_mode?: VideoStrokeRecordingMode;
  segment_duration_ms?: number;
  source_video_filename?: string;
  waveform_audio_filename?: string;
  imported_source_uri?: string;
  imported_at?: string;
  source_start_ms?: number;
  source_end_ms?: number;
  review_status: VideoStrokeReviewStatus;
  markers: VideoStrokeMarker[];
  pose_analysis?: VideoStrokePoseAnalysis[];
}

export interface VideoStrokeSessionFile {
  session_meta: {
    player_name: string;
    handedness: 'right' | 'left';
    camera_facing: VideoStrokeCameraFacing;
    camera_angle: VideoStrokeCameraAngle;
    camera_side?: VideoStrokeCameraSide;
    camera_source?: string;
    collection_type?: 'video_stroke' | 'video_pose_only' | 'video_bounce_side_snapshot';
    pose_sample_fps?: number;
    waveform_audio_filename?: string;
    app_version: string;
    created_at: string;
  };
  takes: VideoStrokeTake[];
}

export interface VideoBounceSideMarker {
  id: string;
  timestamp_ms: number;
  bounce_side: VideoStrokeMarkerType;
  source: 'manual' | 'audio_peak';
  review_status: 'confirmed';
  created_at: string;
  audio_peak_score?: number;
  snapshot_window_ms?: {
    pre_ms: number;
    post_ms: number;
  };
}

export interface VideoBounceSideSessionFile {
  session_meta: {
    player_name: string;
    handedness: 'right' | 'left';
    camera_facing: VideoStrokeCameraFacing;
    camera_angle: VideoStrokeCameraAngle;
    camera_side?: VideoStrokeCameraSide;
    camera_source?: string;
    collection_type: 'video_bounce_side_snapshot';
    label_schema: 'video_bounce_side_v1';
    anchor_source: 'audio_peak';
    snapshot_window_ms: {
      pre_ms: number;
      post_ms: number;
    };
    waveform_audio_filename?: string;
    app_version: string;
    created_at: string;
  };
  takes: Array<{
    video_filename: string;
    duration_ms: number;
    take_index: number;
    recording_mode: 'imported_source';
    review_status: VideoStrokeReviewStatus;
    markers: VideoBounceSideMarker[];
    source_video_filename?: string;
    waveform_audio_filename?: string;
    imported_source_uri?: string;
    imported_at?: string;
  }>;
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

export type AudioLabel = 'racket_bounce' | 'table_bounce' | 'floor_bounce' | 'noise' | 'unlabeled';
export type AudioBackgroundCondition = 'quiet' | 'speech' | 'music_low' | 'music_mid' | 'music_high' | 'desk' | 'mixed' | 'impact';
export type AudioContactLabel = 'racket_contact' | 'not_racket_contact';
export type AudioReviewLabel = AudioContactLabel | 'ignore';
export type AudioReviewSource = 'auto' | 'manual';
export type AudioReviewAnchorRule = 'attack_start';
export type AudioReviewMotionLabel = 'forehand' | 'backhand' | 'unknown' | 'none';
export type AudioVideoSyncSource = 'manual' | 'auto_peak';
export type AudioDetectionSensitivity = 'strict' | 'normal' | 'sensitive';
export type AudioDetectionMode = 'hybrid' | 'four_class_only' | 'binary_only';
export type AudioDetectionIgnoredReason =
  | 'not_racket_contact'
  | 'low_confidence'
  | 'dedup'
  | 'same_label_duplicate'
  | 'surface_veto'
  | 'group_duplicate'
  | 'not_preset_relevant';
export type AudioContactKind = 'racket_bounce';
export type AudioReviewEventType = 'racket_hit' | 'bounce' | 'noise' | 'motion' | 'ignore';
export type AudioReviewClassLabel =
  | 'racket_bounce'
  | 'forehand'
  | 'backhand'
  | 'forehand_hit'
  | 'backhand_hit'
  | 'no_bounce_motion'
  | 'table_bounce'
  | 'floor_bounce'
  | 'catch_after_sound'
  | 'voice_music_noise'
  | 'other_impact'
  | 'ignore';
export type AudioNotRacketKind =
  | 'table_bounce'
  | 'floor_bounce'
  | 'catch_after_sound'
  | 'voice_music_noise'
  | 'other_impact';
export type AudioReviewStatus = 'pending' | 'confirmed' | 'edited' | 'ignored' | 'deleted' | 'filtered';
export type AudioReviewBounceSide = 'forehand' | 'backhand' | 'unknown';
export type AudioRecordingScenario = 'audio_sound' | 'racket_bouncing' | 'playing';
export type AudioBounceContext = 'forehand_side' | 'backhand_side' | 'mixed';
export type AudioCalibrationStatus = 'captured' | 'partial' | 'skipped';
export type AudioScenarioId =
  | 'imported_audio'
  | 'free_recording'
  | 'racket_bounce_fh'
  | 'racket_bounce_bh'
  | 'racket_bounce_mixed'
  | 'racket_motion_no_bounce'
  | 'table_bounce'
  | 'table_noisy'
  | 'floor_bounce'
  | 'floor_noisy'
  | 'catch_after_sound'
  | 'speech_music_noise'
  | 'racket_quiet'
    | 'racket_speech'
    | 'racket_counting'
    | 'racket_music'
    | 'racket_music_low'
    | 'racket_music_mid'
  | 'racket_other_bounces'
  | 'racket_fast'
  | 'playing_dense_audio'
  | 'playing_dense_imu'
  | 'other_bounce_noise'
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
  linked_candidate_id?: string;
  linked_pose_candidate_id?: string;
  source_audio_marker_id?: string;
  suggested_label: AudioReviewLabel;
  final_label: AudioReviewLabel;
  event_type?: AudioReviewEventType;
  class_label?: AudioReviewClassLabel;
  motion_label?: AudioReviewMotionLabel;
  motion_confidence?: number;
  contact_kind?: AudioContactKind;
  not_racket_kind?: AudioNotRacketKind;
  bounce_side?: AudioReviewBounceSide;
  review_status?: AudioReviewStatus;
  contact_confidence?: number;
  surface_label?: AudioLabel;
  surface_confidence?: number;
}

export interface AudioDetectionConfigSnapshot {
  config_id: string;
  sensitivity: AudioDetectionSensitivity;
  detection_mode: AudioDetectionMode;
  contact_confidence_min: number;
  surface_veto_confidence: number;
  merge_window_ms: number;
  onset_threshold: number;
  model_versions: {
    bundle_id: string;
    live_config_id: string;
    audio_contact_model: string;
    audio_model: string;
  };
}

export interface AudioModelCandidate {
  id: string;
  timestamp_ms: number;
  review_relevant: boolean;
  suggested_label?: AudioReviewLabel;
  event_type?: AudioReviewEventType;
  class_label?: AudioReviewClassLabel;
  contact_kind?: AudioContactKind;
  not_racket_kind?: AudioNotRacketKind;
  bounce_side?: AudioReviewBounceSide;
  contact_confidence?: number;
  surface_label?: AudioLabel;
  surface_confidence?: number;
  detection_mode?: AudioDetectionMode;
  detection_config_id?: string;
  ignored_reason?: AudioDetectionIgnoredReason;
}

export interface VideoPoseCandidate {
  id: string;
  timestamp_ms: number;
  source_audio_marker_id?: string;
  predicted_stroke_type: VideoStrokePredictionLabel;
  confidence: number;
  probabilities: Record<string, number>;
  model_version: string;
  feature_spec: 'video_stroke_features_v1' | 'video_stroke_features_v2';
  status: VideoStrokeAnalysisStatus;
  wrist_speed_max?: number;
  review_relevant: boolean;
}

export interface AudioTakeReview {
  required: boolean;
  anchor_rule: AudioReviewAnchorRule;
  review_stage?: 'audio' | 'motion' | 'complete';
  audio_completed_at?: string;
  motion_completed_at?: string;
  completed_at?: string;
  markers: AudioReviewMarker[];
}

export interface AudioTakeReviewSaveOptions {
  completion?: 'audio' | 'complete';
}

export interface AudioImuRecording {
  started_at_ms: number;
  ended_at_ms: number;
  target_hz?: number;
  sample_hz_estimate: number;
  sample_count: number;
  sample_interval_min_ms?: number;
  sample_interval_avg_ms?: number;
  sample_interval_max_ms?: number;
  quality_flag?: 'target_150_met' | 'below_target' | 'unstable' | 'partial';
  disconnected?: boolean;
  partial?: boolean;
  samples: ImuSample[];
}

export interface AudioVideoRecording {
  video_filename: string;
  started_at_ms: number;
  ended_at_ms: number;
  duration_ms: number;
  audio_origin_in_video_ms: number;
  video_sync_offset_ms?: number;
  video_sync_anchor_audio_ms?: number;
  video_sync_anchor_video_ms?: number;
  video_sync_source?: AudioVideoSyncSource;
}

export interface AudioVideoSyncMetadata {
  video_sync_offset_ms: number;
  video_sync_anchor_audio_ms?: number;
  video_sync_anchor_video_ms?: number;
  video_sync_source?: AudioVideoSyncSource;
}

export interface AudioEvent {
  label: AudioLabel;
  recorded_at: string;
  created_at?: string;
  wav_filename: string;
  duration_ms: number;
  scenario_id: AudioScenarioId;
  background_condition: AudioBackgroundCondition;
  take_index: number;
  target_duration_s: number;
  recording_mode?: 'guided_audio_only' | 'guided_audio_imu' | 'audio_imu' | 'free_recording' | 'audio_video_pose' | 'audio_video_pose_import' | 'imported_audio';
  collection_type?: 'audio_only' | 'audio_only_import' | 'audio_video_only' | 'audio_video_imu' | 'audio_video_pose';
  scenario?: AudioRecordingScenario;
  bounce_context?: AudioBounceContext;
  calibration_status?: AudioCalibrationStatus;
  has_audio?: boolean;
  has_video?: boolean;
  has_imu?: boolean;
  imported_source_filename?: string;
  imported_source_uri?: string;
  imported_at?: string;
  source_wav_filename?: string;
  source_video_filename?: string;
  source_start_ms?: number;
  source_end_ms?: number;
  player_handedness?: 'right' | 'left';
  camera_facing?: VideoStrokeCameraFacing;
  detection_config_snapshot?: AudioDetectionConfigSnapshot;
  model_candidates?: AudioModelCandidate[];
  video_pose_candidates?: VideoPoseCandidate[];
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
    collection_mode: 'guided_scenarios' | 'guided_scenarios_audio_imu' | 'free_recording' | 'audio_video_pose';
    recording_mode?: 'guided_audio_only' | 'guided_audio_imu' | 'audio_imu' | 'free_recording' | 'audio_video_pose' | 'audio_video_pose_import' | 'imported_audio';
    collection_type?: 'audio_only' | 'audio_only_import' | 'audio_video_only' | 'audio_video_imu' | 'audio_video_pose';
    scenarios?: AudioRecordingScenario[];
    calibration_status?: AudioCalibrationStatus;
    detection_config_snapshot?: AudioDetectionConfigSnapshot;
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
  audioRetriggerMs: number;
  audioGroupWindowMs: number;
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
  group_id?: number;
  group_status?: 'best_candidate' | 'ignored_duplicate' | 'standalone';
  qualified: boolean;
  ignored_reason?: AudioDetectionIgnoredReason;
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
  group_id?: number;
  group_status?: 'best_candidate' | 'ignored_duplicate' | 'standalone';
  counted: boolean;
  total_after: number;
  alternation_after: number;
  ignored_reason?: AudioDetectionIgnoredReason;
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
