import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  PanResponder,
  Pressable,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  useWindowDimensions,
} from 'react-native';
import AudioRecorderPlayer from 'react-native-audio-recorder-player';
import RNFS from 'react-native-fs';
import Video, { type VideoRef } from 'react-native-video';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import {
  REVIEW_PRE_MS,
  buildMarkerZoomWaveformWindow,
  buildSuggestedReviewData,
  buildWaveformBins,
  createManualMarker,
  decodeWavFile,
  detectAudioSyncPoint,
  snapMarkerToAttack,
  writePreviewClip,
  writeTakePlaybackClip,
  type AudioSyncPoint,
} from './audioReview';
import { VideoPose } from './NativeVideoPose';
import {
  detectionConfigTitle,
  getAudioDetectionConfig,
  getDefaultAudioDetectionConfigSnapshot,
} from './audioDetectionConfig';
import {
  analyzePlayingRetroAudioCandidates,
  getPlayingRetroAudioModelMetadata,
  type PlayingRetroAudioAnalysisResult,
  type PlayingRetroAudioReviewCandidate,
} from './playingRetroAudio';
import { ReviewOrientation } from './ReviewOrientation';
import { buildVideoStrokeFeatures, detectRacketHandedness, VIDEO_STROKE_FEATURE_SPEC } from './videoStrokeFeatures';
import { hasTrainedVideoStrokeModel, predictVideoStroke, videoStrokeModelVersion } from './videoStrokeInference';
import type {
  AudioContactKind,
  AudioEvent,
  AudioDetectionConfigSnapshot,
  AudioDetectionMode,
  AudioDetectionSensitivity,
  AudioModelCandidate,
  AudioNotRacketKind,
  AudioReviewBounceSide,
  AudioReviewClassLabel,
  AudioReviewEventType,
  AudioReviewLabel,
  AudioReviewMotionLabel,
  AudioReviewMarker,
  AudioTakeReviewSaveOptions,
  AudioVideoSyncMetadata,
  AudioVideoSyncSource,
  ImuSample,
  VideoPoseCandidate,
} from './types';

interface Props {
  event: AudioEvent;
  filePath: string;
  videoFilePath?: string;
  onSave: (
    markers: AudioReviewMarker[],
    videoSyncMetadata?: AudioVideoSyncMetadata,
    modelCandidates?: AudioModelCandidate[],
    detectionConfigSnapshot?: AudioDetectionConfigSnapshot,
    videoPoseCandidates?: VideoPoseCandidate[],
    saveOptions?: AudioTakeReviewSaveOptions,
  ) => Promise<void> | void;
  onDiscard: () => Promise<void> | void;
  onBack: () => void;
}

type PlaybackMode = 'idle' | 'playing_full_take' | 'playing_preview' | 'paused_full_take';
type PlaybackRate = 1 | 0.5 | 0.25;
type TimelineZoomLevel = 1 | 2 | 4 | 8 | 12 | 16;
type TimelineInteractionMode = 'idle' | 'playing' | 'scrubbing' | 'autoScrollingWhileScrubbing';
type PlayingConfidenceFilterId = 'all' | 'medium' | 'safe';
type ImuSeriesKey = 'accel_x' | 'accel_y' | 'accel_z' | 'gyro_x' | 'gyro_y' | 'gyro_z';
type AudioVideoReviewStage = 'audio' | 'motion';

interface ReviewStartProfileEntry {
  phase: string;
  elapsedMs: number;
  durationMs?: number;
  detail?: string;
}

interface ReviewStartProfile {
  key: string;
  startedAtMs: number;
  entries: ReviewStartProfileEntry[];
}

const REVIEW_UI_REVISION = 'Simple Review UI | attack_start | r12-synced-timeline';
const NUDGE_STEP_MS = 10;
const LARGE_NUDGE_STEP_MS = 20;
const EXTRA_LARGE_NUDGE_STEP_MS = 50;
const PLAYBACK_SUBSCRIPTION_SEC = 0.1;
const OVERVIEW_BAR_COUNT = 260;
const DETAIL_PRE_MS = 120;
const DETAIL_POST_MS = 120;
const OVERVIEW_PRE_MS = 1800;
const OVERVIEW_POST_MS = 1800;
const TIMELINE_ZOOM_LEVELS: TimelineZoomLevel[] = [1, 2, 4, 8, 12, 16];
const IMU_PLOT_HEIGHT = 66;
const VIDEO_SEEK_THROTTLE_MS = 90;
const TIMELINE_EDGE_PX = 44;
const TIMELINE_DRAG_THRESHOLD_PX = 4;
const TIMELINE_EDGE_SCROLL_INTERVAL_MS = 55;
const TIMELINE_EDGE_SCROLL_FRACTION = 0.035;
const VIDEO_FRAME_STEP_MS = 33;
const PLAYHEAD_LONG_PRESS_MS = 2000;
const PLAYING_CONFIDENCE_FILTERS: Array<{ id: PlayingConfidenceFilterId; title: string; minConfidence: number }> = [
  { id: 'all', title: 'Alla', minConfidence: 0 },
  { id: 'medium', title: 'Medium', minConfidence: 0.65 },
  { id: 'safe', title: 'Säkra', minConfidence: 0.82 },
];
const REVIEW_SENSITIVITY_OPTIONS: AudioDetectionSensitivity[] = ['strict', 'normal', 'sensitive'];
const REVIEW_DETECTION_MODE_OPTIONS: AudioDetectionMode[] = ['hybrid', 'binary_only', 'four_class_only'];
const PRESERVED_MARKER_DUPLICATE_GAP_MS = 120;
const PLAYING_RETRO_SAME_LABEL_DUPLICATE_GAP_MS = 80;
const WAVEFORM_MIN_BIN_COUNT = 160;
const WAVEFORM_MAX_BAR_HEIGHT = 92;
const VIDEO_POSE_SAMPLE_FPS = 15;
const VIDEO_POSE_SCAN_STEP_MS = 100;
const VIDEO_POSE_MIN_CONFIDENCE = 0.58;
const VIDEO_POSE_MIN_GAP_MS = 750;
const VIDEO_POSE_AUDIO_MERGE_MS = 250;
const VIDEO_POSE_MIN_WRIST_SPEED = 1.15;
const VIDEO_POSE_MIN_FRAME_COUNT = 6;
const VIDEO_POSE_MIN_AVG_VISIBILITY = 0.45;
const VIDEO_POSE_MIN_WRIST_TRAVEL = 0.25;
const VIDEO_POSE_MIN_LATERAL_TRAVEL = 0.1;
const VIDEO_POSE_MIN_ELBOW_TRAVEL = 0.04;
const AUDIO_VIDEO_POSE_MIN_AUDIO_CONFIDENCE = 0.65;
const REVIEW_START_PROFILE_LOG_PREFIX = '[review-start-profile]';

function audioVideoPoseReviewConfidence(config?: AudioDetectionConfigSnapshot): number {
  return config?.contact_confidence_min ?? AUDIO_VIDEO_POSE_MIN_AUDIO_CONFIDENCE;
}

function reviewStartProfileEntryText(entry: ReviewStartProfileEntry): string {
  const durationText = typeof entry.durationMs === 'number' ? `/${entry.durationMs}ms` : '';
  return `${entry.phase} ${entry.elapsedMs}ms${durationText}`;
}

function reviewStartProfileSummary(entries: ReviewStartProfileEntry[]): string {
  const latest = entries[entries.length - 1];
  const recent = entries.slice(-3).map(reviewStartProfileEntryText).join(' | ');
  return latest ? `${latest.elapsedMs} ms · ${recent}` : '0 ms';
}

const IMU_SERIES_COLORS = {
  x: '#2ecc71',
  y: '#ff4d4d',
  z: '#4a9eff',
};

interface LineSegment {
  left: number;
  top: number;
  width: number;
  rotateDeg: number;
}

interface QuickLabelPrompt {
  markerId: string;
  timestampMs: number;
  layer?: ReviewTimelineLayer;
}

type ReviewTimelineLayer = 'audio' | 'motion';

function labelText(label: AudioReviewLabel) {
  switch (label) {
    case 'racket_contact':
      return 'Racketträff';
    case 'not_racket_contact':
      return 'Inte racket';
    case 'ignore':
      return 'Ignorera';
  }
}

function labelColor(label: AudioReviewLabel) {
  switch (label) {
    case 'racket_contact':
      return '#2ecc71';
    case 'not_racket_contact':
      return '#ff9f43';
    case 'ignore':
      return '#888';
  }
}

function classColor(classLabel?: string, fallbackLabel?: AudioReviewLabel) {
  if (fallbackLabel === 'ignore' || classLabel === 'ignore') return '#888';
  if (
    fallbackLabel === 'racket_contact' ||
    classLabel === 'racket_bounce' ||
    classLabel === 'forehand' ||
    classLabel === 'backhand' ||
    classLabel === 'forehand_hit' ||
    classLabel === 'backhand_hit'
  ) {
    return '#2ecc71';
  }
  if (classLabel === 'table_bounce') return '#4a9eff';
  if (classLabel === 'floor_bounce') return '#ffc02f';
  if (classLabel === 'noise' || classLabel === 'voice_music_noise') return '#ff5a4f';
  if (classLabel === 'catch_after_sound' || classLabel === 'other_impact') return '#b06cff';
  if (classLabel === 'no_bounce_motion') return '#ffb04f';
  return fallbackLabel ? labelColor(fallbackLabel) : '#f5c76d';
}

function markerColor(marker: AudioReviewMarker) {
  if (marker.event_type === 'motion') {
    if (marker.motion_label === 'forehand') return '#35c7ff';
    if (marker.motion_label === 'backhand') return '#7db7ff';
  }
  return classColor(
    marker.class_label ?? marker.not_racket_kind ?? marker.contact_kind ?? marker.surface_label,
    marker.final_label,
  );
}

function candidateColor(candidate: AudioModelCandidate) {
  return classColor(
    candidate.class_label ?? candidate.not_racket_kind ?? candidate.contact_kind ?? candidate.surface_label,
    candidate.suggested_label,
  );
}

function poseCandidateColor(candidate: VideoPoseCandidate) {
  if (candidate.predicted_stroke_type === 'forehand') return '#35c7ff';
  if (candidate.predicted_stroke_type === 'backhand') return '#7db7ff';
  return '#888';
}

function isConcretePoseStrokeCandidate(candidate: VideoPoseCandidate): boolean {
  return candidate.review_relevant &&
    candidate.status === 'ok' &&
    (candidate.predicted_stroke_type === 'forehand' || candidate.predicted_stroke_type === 'backhand') &&
    candidate.confidence >= VIDEO_POSE_MIN_CONFIDENCE;
}

function shouldShowPoseCandidatePin(candidate: VideoPoseCandidate): boolean {
  return isConcretePoseStrokeCandidate(candidate);
}

function poseCandidateReviewStatusText(candidates: VideoPoseCandidate[], anchoredToAudio: boolean): string {
  const visibleCount = candidates.filter(isConcretePoseStrokeCandidate).length;
  const hiddenCount = candidates.length - visibleCount;
  const sourceText = anchoredToAudio ? 'från racketträffar' : 'i hela videon';
  const hiddenText = hiddenCount > 0
    ? ` ${hiddenCount} oklara/no-target-kandidater sparas som analys men visas inte.`
    : '';
  return `Pose skapade ${visibleCount} FH/BH-förslag ${sourceText}. ${Math.round(VIDEO_POSE_MIN_CONFIDENCE * 100)}% krävs.${hiddenText}`;
}

function shouldShowCandidatePin(candidate: AudioModelCandidate) {
  return candidate.review_relevant;
}

interface ReviewLabelChoice {
  id: string;
  title: string;
  final_label: AudioReviewLabel;
  event_type: AudioReviewEventType;
  class_label: AudioReviewClassLabel;
  motion_label?: AudioReviewMotionLabel;
  contact_kind?: AudioContactKind;
  not_racket_kind?: AudioNotRacketKind;
  bounce_side?: AudioReviewBounceSide;
  color: string;
}

interface ReviewMotionChoice {
  id: string;
  title: string;
  motion_label: AudioReviewMotionLabel;
  color: string;
}

const BASE_REVIEW_LABEL_CHOICES: ReviewLabelChoice[] = [
  {
    id: 'racket_bounce',
    title: 'Racketträff',
    final_label: 'racket_contact',
    event_type: 'racket_hit',
    class_label: 'racket_bounce',
    contact_kind: 'racket_bounce',
    bounce_side: 'unknown',
    color: '#2ecc71',
  },
  {
    id: 'table_bounce',
    title: 'Bordsstuds',
    final_label: 'not_racket_contact',
    event_type: 'bounce',
    class_label: 'table_bounce',
    not_racket_kind: 'table_bounce',
    color: '#4a9eff',
  },
  {
    id: 'floor_bounce',
    title: 'Golvstuds',
    final_label: 'not_racket_contact',
    event_type: 'bounce',
    class_label: 'floor_bounce',
    not_racket_kind: 'floor_bounce',
    color: '#ffc02f',
  },
  {
    id: 'voice_music_noise',
    title: 'Brus',
    final_label: 'not_racket_contact',
    event_type: 'noise',
    class_label: 'voice_music_noise',
    not_racket_kind: 'voice_music_noise',
    color: '#ff7f66',
  },
  {
    id: 'other_impact',
    title: 'Annat',
    final_label: 'not_racket_contact',
    event_type: 'noise',
    class_label: 'other_impact',
    not_racket_kind: 'other_impact',
    color: '#aeb4be',
  },
  {
    id: 'ignore',
    title: 'Ignorera',
    final_label: 'ignore',
    event_type: 'ignore',
    class_label: 'ignore',
    color: '#888',
  },
];

const RACKET_BOUNCING_REVIEW_LABEL_CHOICES: ReviewLabelChoice[] = [
  {
    id: 'racket_bounce_forehand_side',
    title: 'Racketträff FH-sida',
    final_label: 'racket_contact',
    event_type: 'racket_hit',
    class_label: 'forehand',
    contact_kind: 'racket_bounce',
    bounce_side: 'forehand',
    color: '#2ecc71',
  },
  {
    id: 'racket_bounce_backhand_side',
    title: 'Racketträff BH-sida',
    final_label: 'racket_contact',
    event_type: 'racket_hit',
    class_label: 'backhand',
    contact_kind: 'racket_bounce',
    bounce_side: 'backhand',
    color: '#5fd18b',
  },
  {
    id: 'no_bounce_motion',
    title: 'Inte studs',
    final_label: 'not_racket_contact',
    event_type: 'noise',
    class_label: 'no_bounce_motion',
    color: '#ffb04f',
  },
  {
    id: 'racket_bouncing_ignore',
    title: 'Ignorera',
    final_label: 'ignore',
    event_type: 'ignore',
    class_label: 'ignore',
    color: '#888',
  },
];

const PLAYING_REVIEW_LABEL_CHOICES: ReviewLabelChoice[] = [
  {
    id: 'forehand_hit',
    title: 'Racketträff forehand',
    final_label: 'racket_contact',
    event_type: 'racket_hit',
    class_label: 'forehand_hit',
    motion_label: 'forehand',
    contact_kind: 'racket_bounce',
    bounce_side: 'forehand',
    color: '#2ecc71',
  },
  {
    id: 'backhand_hit',
    title: 'Racketträff backhand',
    final_label: 'racket_contact',
    event_type: 'racket_hit',
    class_label: 'backhand_hit',
    motion_label: 'backhand',
    contact_kind: 'racket_bounce',
    bounce_side: 'backhand',
    color: '#5fd18b',
  },
  {
    id: 'playing_table_bounce',
    title: 'Bordsstuds',
    final_label: 'not_racket_contact',
    event_type: 'bounce',
    class_label: 'table_bounce',
    not_racket_kind: 'table_bounce',
    color: '#4a9eff',
  },
];

const AUDIO_VIDEO_POSE_AUDIO_LABEL_CHOICES: ReviewLabelChoice[] = [
  {
    id: 'audio_pose_racket_hit',
    title: 'Ljud: Racketträff',
    final_label: 'racket_contact',
    event_type: 'racket_hit',
    class_label: 'racket_bounce',
    contact_kind: 'racket_bounce',
    bounce_side: 'unknown',
    color: '#2ecc71',
  },
  {
    id: 'audio_pose_table_bounce',
    title: 'Ljud: Bordsstuds',
    final_label: 'not_racket_contact',
    event_type: 'bounce',
    class_label: 'table_bounce',
    not_racket_kind: 'table_bounce',
    color: '#4a9eff',
  },
  {
    id: 'audio_pose_ignore',
    title: 'Ljud: Ignorera',
    final_label: 'ignore',
    event_type: 'ignore',
    class_label: 'ignore',
    color: '#888',
  },
];

const AUDIO_VIDEO_POSE_MOTION_LABEL_CHOICES: ReviewMotionChoice[] = [
  { id: 'motion_forehand', title: 'Rörelse: Forehand', motion_label: 'forehand', color: '#35c7ff' },
  { id: 'motion_backhand', title: 'Rörelse: Backhand', motion_label: 'backhand', color: '#7db7ff' },
  { id: 'motion_unknown', title: 'Rörelse: Oklart', motion_label: 'unknown', color: '#aeb4be' },
];

const AUDIO_VIDEO_POSE_REVIEW_LABEL_CHOICES = AUDIO_VIDEO_POSE_AUDIO_LABEL_CHOICES;

function markerMatchesChoice(marker: AudioReviewMarker | null, choice: ReviewLabelChoice): boolean {
  if (!marker) return false;
  if (choice.event_type === 'motion') {
    return marker.event_type === 'motion' && marker.motion_label === choice.motion_label;
  }
  if (marker.final_label !== choice.final_label) return false;
  if (choice.final_label === 'racket_contact') {
    return (marker.class_label ?? marker.contact_kind ?? 'racket_bounce') === choice.class_label;
  }
  if (choice.final_label === 'not_racket_contact') {
    return (marker.class_label ?? marker.not_racket_kind ?? 'other_impact') === choice.class_label;
  }
  if (marker.event_type === 'motion') return false;
  return choice.final_label === 'ignore';
}

function hasConcreteMotionLabel(motionLabel?: AudioReviewMotionLabel): motionLabel is 'forehand' | 'backhand' {
  return motionLabel === 'forehand' || motionLabel === 'backhand';
}

function markerMatchesAudioPoseAudioChoice(marker: AudioReviewMarker | null, choice: ReviewLabelChoice): boolean {
  if (!marker || marker.final_label !== choice.final_label) return false;
  if (choice.final_label === 'ignore') return true;
  if (choice.final_label === 'racket_contact') {
    return marker.contact_kind === 'racket_bounce' ||
      ['racket_bounce', 'forehand', 'backhand', 'forehand_hit', 'backhand_hit'].includes(marker.class_label ?? '');
  }
  if (choice.final_label === 'not_racket_contact') {
    return (marker.class_label ?? marker.not_racket_kind) === choice.class_label;
  }
  return false;
}

function markerMatchesAudioPoseMotionChoice(marker: AudioReviewMarker | null, choice: ReviewMotionChoice): boolean {
  if (!marker) return false;
  if (choice.motion_label === 'unknown') {
    return !hasConcreteMotionLabel(marker.motion_label);
  }
  return marker.motion_label === choice.motion_label;
}

function audioPoseClassLabelFor(marker: AudioReviewMarker, choice: ReviewLabelChoice): AudioReviewClassLabel {
  if (choice.final_label === 'ignore' && hasConcreteMotionLabel(marker.motion_label)) {
    return marker.motion_label;
  }
  return choice.class_label;
}

function audioPoseEventTypeFor(finalLabel: AudioReviewLabel, eventType: AudioReviewEventType, motionLabel?: AudioReviewMotionLabel) {
  if (finalLabel === 'ignore' && hasConcreteMotionLabel(motionLabel)) return 'motion';
  return eventType;
}

function statusForAudioPoseMarker(marker: AudioReviewMarker, next: AudioReviewMarker): AudioReviewMarker['review_status'] {
  if (next.final_label === 'ignore' && !hasConcreteMotionLabel(next.motion_label)) return 'ignored';
  return marker.source === 'auto' &&
    marker.suggested_label === next.final_label &&
    marker.final_label === next.final_label &&
    marker.motion_label === next.motion_label
    ? 'confirmed'
    : 'edited';
}

function applyAudioPoseAudioChoice(marker: AudioReviewMarker, choice: ReviewLabelChoice): AudioReviewMarker {
  const next: AudioReviewMarker = {
    ...marker,
    final_label: choice.final_label,
    event_type: audioPoseEventTypeFor(choice.final_label, choice.event_type, marker.motion_label),
    class_label: audioPoseClassLabelFor(marker, choice),
    contact_kind: choice.contact_kind,
    not_racket_kind: choice.not_racket_kind,
    bounce_side: choice.final_label === 'racket_contact' && hasConcreteMotionLabel(marker.motion_label)
      ? marker.motion_label
      : choice.bounce_side,
  };
  return {
    ...next,
    review_status: statusForAudioPoseMarker(marker, next),
  };
}

function applyAudioPoseMotionChoice(marker: AudioReviewMarker, choice: ReviewMotionChoice): AudioReviewMarker {
  const concreteMotionLabel = hasConcreteMotionLabel(choice.motion_label) ? choice.motion_label : undefined;
  const concreteMotion = Boolean(concreteMotionLabel);
  const finalLabel = marker.final_label;
  const next: AudioReviewMarker = {
    ...marker,
    motion_label: choice.motion_label,
    motion_confidence: concreteMotion ? marker.motion_confidence : undefined,
    event_type: finalLabel === 'ignore'
      ? (concreteMotion ? 'motion' : 'ignore')
      : marker.event_type,
    class_label: finalLabel === 'ignore'
      ? (concreteMotionLabel ?? 'ignore')
      : finalLabel === 'racket_contact'
        ? 'racket_bounce'
        : marker.class_label,
    bounce_side: finalLabel === 'racket_contact'
      ? (concreteMotionLabel ?? 'unknown')
      : marker.bounce_side,
  };
  return {
    ...next,
    review_status: statusForAudioPoseMarker(marker, next),
  };
}

function audioMarkerDetailText(marker: AudioReviewMarker): string {
  const classLabel = marker.class_label ?? marker.contact_kind ?? marker.not_racket_kind;
  if (classLabel === 'forehand_hit') return 'Racketträff forehand';
  if (classLabel === 'backhand_hit') return 'Racketträff backhand';
  if (classLabel === 'forehand') return 'Racketträff FH-sida';
  if (classLabel === 'backhand') return 'Racketträff BH-sida';
  if (classLabel === 'no_bounce_motion') return 'Inte studs';
  if (classLabel === 'racket_bounce') return 'Racketträff';
  if (classLabel === 'table_bounce') return 'Bordsstuds';
  if (classLabel === 'floor_bounce') return 'Golvstuds';
  if (classLabel === 'voice_music_noise') return 'Brus';
  if (classLabel === 'catch_after_sound') return 'Fång/efterljud';
  if (classLabel === 'other_impact') return 'Annat ljud';
  if (classLabel === 'ignore' || marker.final_label === 'ignore') return 'Ignorerad';
  return labelText(marker.final_label);
}

function motionMarkerDetailText(marker: AudioReviewMarker): string | null {
  if (marker.motion_label === 'forehand') return 'Forehand-rörelse';
  if (marker.motion_label === 'backhand') return 'Backhand-rörelse';
  if (marker.motion_label === 'unknown') return 'Rörelse oklar';
  return null;
}

function markerDetailText(marker: AudioReviewMarker | null): string {
  if (!marker) return 'Ingen marker vald';
  const audioText = audioMarkerDetailText(marker);
  const motionText = motionMarkerDetailText(marker);
  if (marker.final_label === 'ignore') return motionText ?? audioText;
  return motionText ? `${audioText} + ${motionText}` : audioText;
}

function isMotionLayerMarker(marker: AudioReviewMarker): boolean {
  return marker.event_type === 'motion' || (
    marker.final_label === 'ignore' &&
    hasConcreteMotionLabel(marker.motion_label)
  );
}

function isAudioLayerMarker(marker: AudioReviewMarker): boolean {
  return !isMotionLayerMarker(marker);
}

function markerLayer(marker: AudioReviewMarker): ReviewTimelineLayer {
  return isMotionLayerMarker(marker) ? 'motion' : 'audio';
}

function markerConfidence(marker: AudioReviewMarker): number {
  if (marker.motion_confidence && !marker.contact_confidence && !marker.surface_confidence) {
    return marker.motion_confidence;
  }
  if (marker.final_label === 'not_racket_contact' && marker.class_label === 'table_bounce') {
    return marker.surface_confidence ?? marker.contact_confidence ?? 0;
  }
  return Math.max(marker.contact_confidence ?? 0, marker.surface_confidence ?? 0, marker.motion_confidence ?? 0);
}

function markerConfidenceValue(marker: AudioReviewMarker): number | undefined {
  if (isMotionLayerMarker(marker)) return marker.motion_confidence;
  if (marker.final_label === 'racket_contact') return marker.contact_confidence ?? markerConfidence(marker);
  if (marker.final_label === 'not_racket_contact') return marker.surface_confidence ?? marker.contact_confidence ?? markerConfidence(marker);
  return markerConfidence(marker);
}

function markerConfidenceText(marker: AudioReviewMarker | null): string | null {
  if (!marker) return null;
  const confidence = markerConfidenceValue(marker);
  if (typeof confidence !== 'number' || confidence <= 0) return null;
  return `${markerDetailText(marker)} ${Math.round(confidence * 100)}%`;
}

function shouldAlwaysShowMarker(marker: AudioReviewMarker): boolean {
  const status = marker.review_status ?? 'pending';
  return (
    marker.source === 'manual' ||
    Boolean(marker.linked_pose_candidate_id) ||
    status === 'confirmed' ||
    status === 'edited' ||
    status === 'ignored'
  );
}

function markerPassesPlayingFilter(marker: AudioReviewMarker, minConfidence: number): boolean {
  if (shouldAlwaysShowMarker(marker)) return true;
  if (marker.source !== 'auto') return true;
  return markerConfidence(marker) >= minConfidence;
}

function shouldPreserveMarkerWhenConfigChanges(marker: AudioReviewMarker): boolean {
  const status = marker.review_status ?? (marker.source === 'manual' ? 'edited' : 'pending');
  return (
    marker.source === 'manual' ||
    status === 'confirmed' ||
    status === 'edited' ||
    status === 'ignored' ||
    status === 'deleted'
  );
}

function sensitivityTitle(sensitivity: AudioDetectionSensitivity): string {
  if (sensitivity === 'strict') return 'Strikt';
  if (sensitivity === 'sensitive') return 'Känslig';
  return 'Normal';
}

function detectionModeTitle(mode: AudioDetectionMode): string {
  if (mode === 'four_class_only') return '4-klass';
  if (mode === 'binary_only') return 'Binär';
  return 'Hybrid';
}

function shouldDropPlayingAutoMarkerOnSave(
  marker: AudioReviewMarker,
  minConfidence: number,
): boolean {
  if (marker.source !== 'auto') return false;
  const status = marker.review_status ?? 'pending';
  if (status === 'filtered') return true;
  return status === 'pending' && !markerPassesPlayingFilter(marker, minConfidence);
}

function modelCandidatesFromMarkers(
  markers: AudioReviewMarker[],
  detectionConfig: AudioDetectionConfigSnapshot,
): AudioModelCandidate[] {
  return markers
    .filter(marker => marker.source === 'auto' && isAudioLayerMarker(marker))
    .map((marker, index) => ({
      id: marker.linked_candidate_id ?? `candidate_from_marker_${index}_${Math.round(marker.timestamp_ms)}`,
      timestamp_ms: marker.timestamp_ms,
      review_relevant: true,
      suggested_label: marker.suggested_label,
      event_type: marker.event_type,
      class_label: marker.class_label,
      contact_kind: marker.contact_kind,
      not_racket_kind: marker.not_racket_kind,
      bounce_side: marker.bounce_side,
      contact_confidence: marker.contact_confidence,
      surface_label: marker.surface_label,
      surface_confidence: marker.surface_confidence,
      detection_mode: detectionConfig.detection_mode,
      detection_config_id: detectionConfig.config_id,
    }));
}

function shouldKeepExistingReviewMarkersForPlayingRetro(markers: AudioReviewMarker[]): boolean {
  return markers.some(marker => {
    const status = marker.review_status ?? 'pending';
    return marker.source === 'manual' ||
      status === 'confirmed' ||
      status === 'edited' ||
      status === 'ignored' ||
      status === 'deleted';
  });
}

function markerFromPlayingRetroCandidate(
  candidate: PlayingRetroAudioReviewCandidate,
  index: number,
  durationMs: number,
): AudioReviewMarker | null {
  if (!candidate.review_relevant || !candidate.suggested_label || !candidate.event_type || !candidate.class_label) {
    return null;
  }
  return {
    id: `auto_playing_retro_${index}_${candidate.source_candidate_id}_${Math.round(candidate.timestamp_ms)}`,
    timestamp_ms: clampTimestamp(candidate.timestamp_ms, durationMs),
    source: 'auto',
    linked_candidate_id: candidate.id,
    suggested_label: candidate.suggested_label,
    final_label: candidate.suggested_label,
    event_type: candidate.event_type,
    class_label: candidate.class_label,
    contact_kind: candidate.contact_kind,
    not_racket_kind: candidate.not_racket_kind,
    bounce_side: candidate.suggested_label === 'racket_contact' ? 'unknown' : candidate.bounce_side,
    review_status: 'pending',
    contact_confidence: candidate.contact_confidence,
    surface_label: candidate.surface_label,
    surface_confidence: candidate.surface_confidence,
  };
}

function playingRetroMarkerKind(marker: AudioReviewMarker): 'racket' | 'table' | null {
  if (
    marker.final_label === 'racket_contact' ||
    marker.event_type === 'racket_hit' ||
    marker.contact_kind === 'racket_bounce' ||
    marker.class_label === 'racket_bounce'
  ) {
    return 'racket';
  }
  if (
    marker.final_label === 'not_racket_contact' ||
    marker.event_type === 'bounce' ||
    marker.not_racket_kind === 'table_bounce' ||
    marker.class_label === 'table_bounce' ||
    marker.surface_label === 'table_bounce'
  ) {
    return 'table';
  }
  return null;
}

function dedupePlayingRetroSameLabelMarkers(markers: AudioReviewMarker[]): AudioReviewMarker[] {
  const kept: AudioReviewMarker[] = [];
  for (const marker of sortMarkers(markers)) {
    const markerKind = playingRetroMarkerKind(marker);
    let duplicateIndex = -1;
    if (markerKind) {
      for (let index = kept.length - 1; index >= 0; index -= 1) {
        const existing = kept[index];
        if (Math.abs(existing.timestamp_ms - marker.timestamp_ms) > PLAYING_RETRO_SAME_LABEL_DUPLICATE_GAP_MS) {
          break;
        }
        if (playingRetroMarkerKind(existing) === markerKind) {
          duplicateIndex = index;
          break;
        }
      }
    }
    if (duplicateIndex >= 0) {
      kept[duplicateIndex] = marker;
    } else {
      kept.push(marker);
    }
  }
  return sortMarkers(kept);
}

function markersFromPlayingRetroAnalysis(
  candidates: PlayingRetroAudioReviewCandidate[],
  durationMs: number,
): AudioReviewMarker[] {
  return dedupePlayingRetroSameLabelMarkers(candidates
    .map((candidate, index) => markerFromPlayingRetroCandidate(candidate, index, durationMs))
    .filter((marker): marker is AudioReviewMarker => marker !== null));
}

interface PlayingRetroPrimaryReviewSummary {
  markerCount: number;
  reviewCandidateCount: number;
  reviewRacketCount: number;
  reviewTableCount: number;
  rawCandidateCount: number;
  savedCandidateCount: number;
  recoveryCandidateCount: number;
  visibleRecoveryCandidateCount: number;
  rawRacketCount: number;
  rawTableCount: number;
  rawNonTargetCount: number;
  hiddenTargetCount: number;
}

function summarizePlayingRetroPrimaryReview(
  analysis: PlayingRetroAudioAnalysisResult,
  markerCount: number,
): PlayingRetroPrimaryReviewSummary {
  const reviewCandidates = analysis.candidates.filter(candidate => candidate.review_relevant);
  const reviewRacketCount = reviewCandidates.filter(candidate => (
    candidate.playing_retro_prediction.label === 'racket_contact'
  )).length;
  const reviewTableCount = reviewCandidates.filter(candidate => (
    candidate.playing_retro_prediction.label === 'table_bounce'
  )).length;
  const rawRacketCount = analysis.candidates.filter(candidate => (
    candidate.playing_retro_prediction.label === 'racket_contact'
  )).length;
  const rawTableCount = analysis.candidates.filter(candidate => (
    candidate.playing_retro_prediction.label === 'table_bounce'
  )).length;
  const rawTargetCount = rawRacketCount + rawTableCount;
  const reviewTargetCount = reviewRacketCount + reviewTableCount;
  return {
    markerCount,
    reviewCandidateCount: reviewCandidates.length,
    reviewRacketCount,
    reviewTableCount,
    rawCandidateCount: analysis.candidates.length,
    savedCandidateCount: analysis.saved_candidate_count,
    recoveryCandidateCount: analysis.recovery_candidate_count,
    visibleRecoveryCandidateCount: analysis.visible_recovery_candidate_count,
    rawRacketCount,
    rawTableCount,
    rawNonTargetCount: Math.max(0, analysis.candidates.length - rawTargetCount),
    hiddenTargetCount: Math.max(0, rawTargetCount - reviewTargetCount),
  };
}

function playingRetroPrimaryStatusText(summary: PlayingRetroPrimaryReviewSummary): string {
  const mismatchText = summary.markerCount !== summary.reviewCandidateCount
    ? `, ${summary.reviewCandidateCount} reviewkandidater`
    : '';
  const recoveryText = summary.recoveryCandidateCount > 0
    ? `, recovery ${summary.visibleRecoveryCandidateCount}/${summary.recoveryCandidateCount} visas`
    : '';
  const hiddenTargetText = summary.hiddenTargetCount > 0
    ? `, ${summary.hiddenTargetCount} target filtrerade`
    : '';
  return [
    `Spel-retro aktiv: ${summary.markerCount} markers att granska (${summary.reviewRacketCount} racket, ${summary.reviewTableCount} bord${mismatchText}).`,
    `Rådata: ${summary.rawCandidateCount} kandidater (${summary.rawRacketCount} racket, ${summary.rawTableCount} bord, ${summary.rawNonTargetCount} ej target), ${summary.savedCandidateCount} sparade${recoveryText}${hiddenTargetText}.`,
  ].join(' ');
}

function applyReviewLabelChoice(marker: AudioReviewMarker, choice: ReviewLabelChoice): AudioReviewMarker {
  const wasSuggested = (
    marker.suggested_label === choice.final_label &&
    marker.final_label === choice.final_label &&
    marker.class_label === choice.class_label &&
    marker.motion_label === choice.motion_label
  );
  return {
    ...marker,
    final_label: choice.final_label,
    event_type: choice.event_type,
    class_label: choice.class_label,
    motion_label: choice.motion_label,
    contact_kind: choice.contact_kind,
    not_racket_kind: choice.not_racket_kind,
    bounce_side: choice.bounce_side,
    review_status: choice.event_type === 'motion'
      ? (wasSuggested ? 'confirmed' : 'edited')
      : choice.final_label === 'ignore'
      ? 'ignored'
      : wasSuggested
        ? 'confirmed'
        : 'edited',
  };
}

function sortMarkers(markers: AudioReviewMarker[]) {
  return [...markers].sort((a, b) => a.timestamp_ms - b.timestamp_ms);
}

function clampTimestamp(timestampMs: number, durationMs: number) {
  return Math.max(0, Math.min(durationMs, Math.round(timestampMs)));
}

function createAudioVideoPoseMotionMarker(
  timestampMs: number,
  durationMs: number,
  motionLabel: 'forehand' | 'backhand' | 'unknown' = 'unknown',
  candidate?: VideoPoseCandidate,
): AudioReviewMarker {
  return {
    id: candidate?.id ? `pose_marker_${candidate.id}` : `manual_motion_${Date.now()}_${Math.round(timestampMs)}`,
    timestamp_ms: clampTimestamp(timestampMs, durationMs),
    source: candidate ? 'auto' : 'manual',
    linked_pose_candidate_id: candidate?.id,
    source_audio_marker_id: candidate?.source_audio_marker_id,
    suggested_label: 'ignore',
    final_label: 'ignore',
    event_type: 'motion',
    class_label: motionLabel === 'unknown' ? 'ignore' : motionLabel,
    motion_label: motionLabel,
    motion_confidence: candidate?.confidence,
    review_status: candidate ? 'pending' : 'pending',
  };
}

function splitAudioVideoPoseMarkers(markers: AudioReviewMarker[], durationMs: number): AudioReviewMarker[] {
  const splitMarkers: AudioReviewMarker[] = [];
  for (const marker of markers) {
    const motionLabel = hasConcreteMotionLabel(marker.motion_label) ? marker.motion_label : undefined;
    const hasAudioLayer = marker.final_label !== 'ignore' || marker.event_type !== 'motion';
    const hasMotionLayer = Boolean(motionLabel || marker.linked_pose_candidate_id);
    if (hasAudioLayer && hasMotionLayer) {
      const audioMarker: AudioReviewMarker = {
        ...marker,
        id: marker.id,
        linked_pose_candidate_id: undefined,
        source_audio_marker_id: undefined,
        motion_label: undefined,
        motion_confidence: undefined,
        class_label: marker.final_label === 'racket_contact'
          ? 'racket_bounce'
          : marker.class_label,
        bounce_side: marker.final_label === 'racket_contact' ? 'unknown' : marker.bounce_side,
      };
      const motionMarker: AudioReviewMarker = {
        id: `${marker.id}_motion`,
        timestamp_ms: clampTimestamp(marker.timestamp_ms, durationMs),
        source: marker.source,
        linked_pose_candidate_id: marker.linked_pose_candidate_id,
        source_audio_marker_id: marker.source_audio_marker_id,
        suggested_label: 'ignore',
        final_label: 'ignore',
        event_type: 'motion',
        class_label: motionLabel ?? 'ignore',
        motion_label: motionLabel ?? 'unknown',
        motion_confidence: marker.motion_confidence,
        review_status: marker.review_status ?? 'pending',
      };
      splitMarkers.push(audioMarker, motionMarker);
    } else {
      splitMarkers.push(marker);
    }
  }
  return sortMarkers(splitMarkers);
}

function normalizeAudioVideoPoseMarkerForSave(marker: AudioReviewMarker): AudioReviewMarker {
  if (isMotionLayerMarker(marker)) {
    return {
      id: marker.id,
      timestamp_ms: marker.timestamp_ms,
      source: marker.source,
      linked_pose_candidate_id: marker.linked_pose_candidate_id,
      source_audio_marker_id: marker.source_audio_marker_id,
      suggested_label: 'ignore',
      final_label: 'ignore',
      event_type: 'motion',
      class_label: hasConcreteMotionLabel(marker.motion_label) ? marker.motion_label : 'ignore',
      motion_label: marker.motion_label ?? 'unknown',
      motion_confidence: marker.motion_confidence,
      review_status: marker.review_status,
    };
  }
  return {
    ...marker,
    linked_pose_candidate_id: undefined,
    source_audio_marker_id: undefined,
    motion_label: undefined,
    motion_confidence: undefined,
    class_label: marker.final_label === 'racket_contact' ? 'racket_bounce' : marker.class_label,
    bounce_side: marker.final_label === 'racket_contact' ? 'unknown' : marker.bounce_side,
  };
}

function shouldDropLowConfidenceAudioVideoPoseMarker(
  marker: AudioReviewMarker,
  minConfidence: number,
): boolean {
  const status = marker.review_status ?? 'pending';
  return (
    isAudioLayerMarker(marker) &&
    marker.source === 'auto' &&
    status === 'pending' &&
    marker.final_label === 'racket_contact' &&
    (marker.contact_confidence ?? 0) < minConfidence
  );
}

function passesPoseMotionGate(featureResult: ReturnType<typeof buildVideoStrokeFeatures>): boolean {
  if (!featureResult) return false;
  const features = featureResult.features;
  const wristTravel = Math.hypot(features.wrist_x_ptp, features.wrist_y_ptp);
  const elbowTravel = Math.hypot(features.elbow_x_ptp, features.elbow_y_ptp);
  const lateralTravel = Math.max(Math.abs(features.wrist_x_delta), features.wrist_x_ptp);
  return (
    featureResult.frame_count >= VIDEO_POSE_MIN_FRAME_COUNT &&
    featureResult.avg_visibility >= VIDEO_POSE_MIN_AVG_VISIBILITY &&
    features.wrist_speed_max >= VIDEO_POSE_MIN_WRIST_SPEED &&
    wristTravel >= VIDEO_POSE_MIN_WRIST_TRAVEL &&
    lateralTravel >= VIDEO_POSE_MIN_LATERAL_TRAVEL &&
    elbowTravel >= VIDEO_POSE_MIN_ELBOW_TRAVEL
  );
}

function hasUsableAnchoredPoseWindow(featureResult: ReturnType<typeof buildVideoStrokeFeatures>): boolean {
  return Boolean(
    featureResult &&
    featureResult.frame_count >= VIDEO_POSE_MIN_FRAME_COUNT &&
    featureResult.avg_visibility >= VIDEO_POSE_MIN_AVG_VISIBILITY,
  );
}

async function buildVideoPoseCandidatesForReview(
  videoFilePath: string,
  durationMs: number,
  handedness: 'right' | 'left',
  anchors?: Array<{ id: string; timestamp_ms: number }>,
): Promise<VideoPoseCandidate[]> {
  if (!hasTrainedVideoStrokeModel()) {
    return [];
  }

  const pose = await VideoPose.extractPose(videoFilePath, VIDEO_POSE_SAMPLE_FPS);
  const scanDurationMs = Math.max(0, pose.duration_ms || durationMs);
  // Spelaren i videon kan ha annan hänthet än profilen (importerad video av
  // t.ex. Tomas). Racketarmen detekteras ur rörelsen - fel hand ger fel arm
  // + spegelvänd x-axel och systematiskt fel FH/BH.
  const playerHandedness = detectRacketHandedness(pose.frames, handedness).handedness;
  if (anchors?.length) {
    return anchors.map(anchor => {
      const timestampMs = clampTimestamp(anchor.timestamp_ms, scanDurationMs || durationMs);
      const baseCandidate = {
        id: `pose_for_${anchor.id}`,
        timestamp_ms: timestampMs,
        source_audio_marker_id: anchor.id,
        probabilities: {},
        model_version: videoStrokeModelVersion(),
        feature_spec: VIDEO_STROKE_FEATURE_SPEC,
        wrist_speed_max: 0,
        review_relevant: false,
      } satisfies Omit<VideoPoseCandidate, 'predicted_stroke_type' | 'confidence' | 'status'>;
      const featureResult = buildVideoStrokeFeatures(pose.frames, timestampMs, playerHandedness);
      if (!featureResult || !hasUsableAnchoredPoseWindow(featureResult)) {
        return {
          ...baseCandidate,
          predicted_stroke_type: 'uncertain',
          confidence: 0,
          status: 'insufficient_pose',
        };
      }
      const prediction = predictVideoStroke(featureResult.features);
      const wristSpeed = featureResult.features.wrist_speed_max;
      if (
        prediction.status !== 'ok' ||
        (prediction.label !== 'forehand' && prediction.label !== 'backhand') ||
        prediction.confidence < VIDEO_POSE_MIN_CONFIDENCE
      ) {
        return {
          ...baseCandidate,
          predicted_stroke_type: 'uncertain',
          confidence: prediction.confidence ?? 0,
          probabilities: prediction.probabilities ?? {},
          model_version: prediction.model_version ?? videoStrokeModelVersion(),
          status: prediction.status === 'ok' ? 'uncertain' : prediction.status,
          wrist_speed_max: wristSpeed,
          review_relevant: false,
        };
      }
      return {
        ...baseCandidate,
        predicted_stroke_type: prediction.label,
        confidence: prediction.confidence,
        probabilities: prediction.probabilities,
        model_version: prediction.model_version,
        status: 'ok',
        wrist_speed_max: wristSpeed,
        review_relevant: true,
      };
    });
  }

  const candidates: Array<VideoPoseCandidate & { score: number }> = [];
  for (
    let timestampMs = 700;
    timestampMs <= Math.max(700, scanDurationMs - 500);
    timestampMs += VIDEO_POSE_SCAN_STEP_MS
  ) {
    const featureResult = buildVideoStrokeFeatures(pose.frames, timestampMs, playerHandedness);
    if (!featureResult) continue;
    if (!passesPoseMotionGate(featureResult)) continue;
    const prediction = predictVideoStroke(featureResult.features);
    if (prediction.status !== 'ok' || (prediction.label !== 'forehand' && prediction.label !== 'backhand')) continue;
    const wristSpeed = featureResult.features.wrist_speed_max;
    if (prediction.confidence < VIDEO_POSE_MIN_CONFIDENCE) continue;
    candidates.push({
      id: `pose_${Math.round(timestampMs)}`,
      timestamp_ms: Math.round(timestampMs),
      predicted_stroke_type: prediction.label,
      confidence: prediction.confidence,
      probabilities: prediction.probabilities,
      model_version: prediction.model_version,
      feature_spec: VIDEO_STROKE_FEATURE_SPEC,
      status: 'ok',
      wrist_speed_max: wristSpeed,
      review_relevant: true,
      score: prediction.confidence + Math.min(1, wristSpeed / 16) * 0.15,
    });
  }

  const picked: Array<VideoPoseCandidate & { score: number }> = [];
  for (const candidate of candidates.sort((left, right) => right.score - left.score)) {
    if (picked.every(existing => Math.abs(existing.timestamp_ms - candidate.timestamp_ms) >= VIDEO_POSE_MIN_GAP_MS)) {
      picked.push(candidate);
    }
  }
  return picked
    .sort((left, right) => left.timestamp_ms - right.timestamp_ms)
    .map(({ score: _score, ...candidate }) => candidate);
}

function addPoseCandidatesAsMotionMarkers(
  audioMarkers: AudioReviewMarker[],
  poseCandidates: VideoPoseCandidate[],
  durationMs: number,
): AudioReviewMarker[] {
  const nextMarkers = audioMarkers.map(marker => ({ ...marker }));
  const linkedPoseIds = new Set(nextMarkers
    .map(marker => marker.linked_pose_candidate_id)
    .filter((candidateId): candidateId is string => Boolean(candidateId)));

  for (const candidate of poseCandidates) {
    if (linkedPoseIds.has(candidate.id)) continue;
    if (!isConcretePoseStrokeCandidate(candidate)) continue;
    const motionLabel = candidate.predicted_stroke_type === 'forehand' || candidate.predicted_stroke_type === 'backhand'
      ? candidate.predicted_stroke_type
      : 'unknown';
    nextMarkers.push(createAudioVideoPoseMotionMarker(
      candidate.timestamp_ms,
      durationMs,
      motionLabel,
      candidate,
    ));
    linkedPoseIds.add(candidate.id);
  }

  return sortMarkers(nextMarkers);
}

function formatMs(ms: number) {
  const safe = Math.max(0, Math.round(ms));
  const seconds = Math.floor(safe / 1000);
  const centiseconds = Math.floor((safe % 1000) / 10);
  return `${seconds}.${String(centiseconds).padStart(2, '0')}s`;
}

function formatTimelineMs(ms: number) {
  const safe = Math.max(0, Math.round(ms));
  const minutes = Math.floor(safe / 60000);
  const seconds = Math.floor((safe % 60000) / 1000);
  const centiseconds = Math.floor((safe % 1000) / 10);
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.${String(centiseconds).padStart(2, '0')}`;
}

function formatTimelineShort(ms: number) {
  const safe = Math.max(0, Math.round(ms));
  const minutes = Math.floor(safe / 60000);
  const seconds = Math.floor((safe % 60000) / 1000);
  const tenths = Math.floor((safe % 1000) / 100);
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.${tenths}`;
}

function ratioToLeft(timestampMs: number, startMs: number, endMs: number, width: number) {
  if (endMs <= startMs) return 0;
  const ratio = (timestampMs - startMs) / (endMs - startMs);
  return Math.max(0, Math.min(width, ratio * width));
}

function isTimestampVisible(timestampMs: number, startMs: number, endMs: number) {
  return timestampMs >= startMs && timestampMs <= endMs;
}

function centeredTimelineStart(playheadMs: number, durationMs: number, spanMs: number) {
  const maxStart = Math.max(0, durationMs - spanMs);
  return Math.max(0, Math.min(maxStart, playheadMs - spanMs / 2));
}

function followedTimelineStart(playheadMs: number, currentStartMs: number, durationMs: number, spanMs: number) {
  if (durationMs <= 0 || spanMs >= durationMs) return 0;
  const maxStart = Math.max(0, durationMs - spanMs);
  const currentEndMs = currentStartMs + spanMs;
  const marginMs = Math.max(80, spanMs * 0.18);
  if (playheadMs < currentStartMs + marginMs) {
    return Math.max(0, Math.min(maxStart, playheadMs - marginMs));
  }
  if (playheadMs > currentEndMs - marginMs) {
    return Math.max(0, Math.min(maxStart, playheadMs - spanMs + marginMs));
  }
  return Math.max(0, Math.min(maxStart, currentStartMs));
}

function imuSampleTimeMs(sample: ImuSample, recordingStartedAtMs?: number) {
  if (typeof sample.take_ts_ms === 'number' && Number.isFinite(sample.take_ts_ms)) {
    return sample.take_ts_ms;
  }
  if (
    typeof sample.received_at_ms === 'number' &&
    Number.isFinite(sample.received_at_ms) &&
    typeof recordingStartedAtMs === 'number'
  ) {
    return sample.received_at_ms - recordingStartedAtMs;
  }
  if (
    typeof sample.ts_ms === 'number' &&
    Number.isFinite(sample.ts_ms) &&
    typeof recordingStartedAtMs === 'number' &&
    sample.ts_ms > recordingStartedAtMs
  ) {
    return sample.ts_ms - recordingStartedAtMs;
  }
  return Number.isFinite(sample.ts_ms) ? sample.ts_ms : 0;
}

function buildImuLineSegments(
  samples: ImuSample[],
  key: ImuSeriesKey,
  startMs: number,
  endMs: number,
  width: number,
  height: number,
  recordingStartedAtMs?: number,
): LineSegment[] {
  if (samples.length < 2 || width <= 0 || endMs <= startMs) return [];

  const visible = samples
    .map(sample => ({
      t: imuSampleTimeMs(sample, recordingStartedAtMs),
      v: sample[key],
    }))
    .filter(point => point.t >= startMs && point.t <= endMs && Number.isFinite(point.v));
  if (visible.length < 2) return [];

  const maxPoints = Math.max(24, Math.min(120, Math.floor(width / 3)));
  const step = Math.max(1, Math.ceil(visible.length / maxPoints));
  const points = visible.filter((_, index) => index % step === 0);
  if (points[points.length - 1] !== visible[visible.length - 1]) {
    points.push(visible[visible.length - 1]);
  }
  const maxAbs = Math.max(1, ...points.map(point => Math.abs(point.v)));

  const toX = (t: number) => ratioToLeft(t, startMs, endMs, width);
  const toY = (v: number) => (height / 2) - (v / maxAbs) * (height * 0.42);
  const segments: LineSegment[] = [];
  for (let index = 1; index < points.length; index++) {
    const prev = points[index - 1];
    const next = points[index];
    const x1 = toX(prev.t);
    const y1 = toY(prev.v);
    const x2 = toX(next.t);
    const y2 = toY(next.v);
    const dx = x2 - x1;
    const dy = y2 - y1;
    const length = Math.sqrt(dx * dx + dy * dy);
    if (length < 1) continue;
    segments.push({
      left: x1 + dx / 2 - length / 2,
      top: y1 + dy / 2 - 1,
      width: length,
      rotateDeg: Math.atan2(dy, dx) * (180 / Math.PI),
    });
  }
  return segments;
}

function mapAudioPlayheadToVideoMs(
  event: AudioEvent,
  audioMs: number,
  audioDurationMs: number,
  videoSyncOffsetMs = 0,
) {
  const video = event.video_recording;
  if (!video) return 0;
  if (audioDurationMs <= 0 || video.duration_ms <= 0) return 0;
  const audioOriginInVideoMs = typeof video.audio_origin_in_video_ms === 'number' &&
    Number.isFinite(video.audio_origin_in_video_ms)
    ? video.audio_origin_in_video_ms
    : undefined;
  if (typeof audioOriginInVideoMs === 'number') {
    return clampTimestamp(audioOriginInVideoMs + videoSyncOffsetMs + audioMs, video.duration_ms);
  }
  const normalized = Math.max(0, Math.min(1, audioMs / audioDurationMs));
  return clampTimestamp(Math.round(normalized * video.duration_ms) + videoSyncOffsetMs, video.duration_ms);
}

function baseVideoMsForAudio(event: AudioEvent, audioMs: number, audioDurationMs: number) {
  const video = event.video_recording;
  if (!video) return 0;
  if (audioDurationMs <= 0 || video.duration_ms <= 0) return 0;
  const audioOriginInVideoMs = typeof video.audio_origin_in_video_ms === 'number' &&
    Number.isFinite(video.audio_origin_in_video_ms)
    ? video.audio_origin_in_video_ms
    : undefined;
  if (typeof audioOriginInVideoMs === 'number') {
    return audioOriginInVideoMs + audioMs;
  }
  const normalized = Math.max(0, Math.min(1, audioMs / audioDurationMs));
  return Math.round(normalized * video.duration_ms);
}

function offsetFromVideoSyncAnchor(
  event: AudioEvent,
  audioMs: number,
  audioDurationMs: number,
  selectedVideoMs: number,
) {
  return Math.round(selectedVideoMs - baseVideoMsForAudio(event, audioMs, audioDurationMs));
}

function syncPointFromAudioMs(timestampMs: number, durationMs: number, confidence = 1): AudioSyncPoint {
  const clampedTimestampMs = clampTimestamp(timestampMs, durationMs);
  return {
    timestamp_ms: clampedTimestampMs,
    score: 0,
    confidence,
    window_start_ms: Math.max(0, clampedTimestampMs - 80),
    window_end_ms: Math.min(durationMs, clampedTimestampMs + 80),
  };
}

interface ImuLinePlotProps {
  samples: ImuSample[];
  keys: Array<{ key: ImuSeriesKey; axis: 'x' | 'y' | 'z'; color: string }>;
  startMs: number;
  endMs: number;
  playheadMs: number;
  markers: AudioReviewMarker[];
  selectedMarkerId: string | null;
  width: number;
  recordingStartedAtMs?: number;
  onSelectMarker: (marker: AudioReviewMarker) => void;
}

function ImuLinePlot({
  samples,
  keys,
  startMs,
  endMs,
  playheadMs,
  markers,
  selectedMarkerId,
  width,
  recordingStartedAtMs,
  onSelectMarker,
}: ImuLinePlotProps) {
  const plotWidth = Math.max(1, width);
  const playheadLeft = ratioToLeft(playheadMs, startMs, endMs, plotWidth);
  const visibleMarkers = markers.filter(marker => isTimestampVisible(marker.timestamp_ms, startMs, endMs));

  return (
    <View style={styles.imuPlot}>
      <View style={styles.imuZeroLine} />
      {keys.map(series => buildImuLineSegments(
        samples,
        series.key,
        startMs,
        endMs,
        plotWidth,
        IMU_PLOT_HEIGHT,
        recordingStartedAtMs,
      ).map((segment, index) => (
        <View
          key={`${series.key}-${index}`}
          style={[
            styles.imuLineSegment,
            {
              left: segment.left,
              top: segment.top,
              width: segment.width,
              backgroundColor: series.color,
              transform: [{ rotateZ: `${segment.rotateDeg}deg` }],
            },
          ]}
        />
      )))}
      {visibleMarkers.map(marker => {
        const left = ratioToLeft(marker.timestamp_ms, startMs, endMs, plotWidth);
        const selected = marker.id === selectedMarkerId;
        return (
          <TouchableOpacity
            key={`imu-marker-${marker.id}`}
            style={[styles.imuMarkerHitbox, { left: Math.max(0, left - 11) }]}
            onPress={() => onSelectMarker(marker)}
          >
            <View
              style={[
                styles.imuMarkerLine,
                { backgroundColor: markerColor(marker) },
                selected && styles.imuMarkerLineActive,
              ]}
            />
          </TouchableOpacity>
        );
      })}
      <View style={[styles.imuPlayhead, { left: Math.max(0, playheadLeft) }]} />
    </View>
  );
}

export function AudioTakeReviewScreen({ event, filePath, videoFilePath, onSave, onDiscard, onBack }: Props) {
  const { width: windowWidth, height: windowHeight } = useWindowDimensions();
  const insets = useSafeAreaInsets();
  const playerRef = useRef(new AudioRecorderPlayer());
  const videoRef = useRef<VideoRef>(null);
  const decodedRef = useRef<{ sampleRate: number; samples: Float32Array } | null>(null);
  const tempPlaybackPathRef = useRef<string | null>(null);
  const playbackModeRef = useRef<PlaybackMode>('idle');
  const playbackRateRef = useRef<PlaybackRate>(1);
  const playbackSourceStartMsRef = useRef(0);
  const latestAudioCallbackPositionMsRef = useRef(0);
  const latestAudioCallbackWallClockMsRef = useRef(0);
  const latestVideoMsRef = useRef(0);
  const latestVideoProgressWallClockMsRef = useRef(0);
  const pausedAtAudioMsRef = useRef<number | null>(null);
  const overviewDragStartMsRef = useRef(0);
  const overviewMarkerDragStartMsRef = useRef(0);
  const detailMarkerDragStartMsRef = useRef(0);
  const playbackPositionRef = useRef(0);
  const timelineWindowStartRef = useRef(0);
  const timelineWindowSpanRef = useRef(1);
  const timelineSurfaceRef = useRef<View>(null);
  const videoProgressTrackRef = useRef<View>(null);
  const lastVideoSeekAtRef = useRef(0);
  const videoScrubPendingMsRef = useRef<number | null>(null);
  const edgeScrollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const edgeScrollDirectionRef = useRef<-1 | 0 | 1>(0);
  const timelineModeRef = useRef<TimelineInteractionMode>('idle');
  const syncCalibrationActiveRef = useRef(false);
  const scrubStartWindowMsRef = useRef(0);
  const scrubStartPlayheadMsRef = useRef(0);
  const scrubTimelineLayoutRef = useRef({ pageX: 0, width: 0 });
  const videoProgressLayoutRef = useRef({ pageX: 0, width: 0 });
  const scrubFingerOffsetPxRef = useRef(0);
  const latestScrubMsRef = useRef(0);
  const latestScrubPageXRef = useRef(0);
  const playheadLongPressTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const playheadLongPressStartPageXRef = useRef(0);
  const playheadLongPressHandledRef = useRef(false);
  const createMarkerFromPlayheadRef = useRef<(layer?: ReviewTimelineLayer) => void>(() => {});
  const videoPoseScanInFlightRef = useRef(false);
  const videoPoseScanCompletedRef = useRef((event.video_pose_candidates?.length ?? 0) > 0);
  const videoPoseAutoMarkersSeededRef = useRef(false);
  const videoPoseDeferredForAudioRef = useRef(false);
  const playingRetroPrimaryAppliedRef = useRef(false);
  const reviewStartProfileRef = useRef<ReviewStartProfile | null>(null);

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [durationMs, setDurationMs] = useState(event.duration_ms);
  const [overviewBins, setOverviewBins] = useState<number[]>([]);
  const [markers, setMarkers] = useState<AudioReviewMarker[]>(event.review?.markers ?? []);
  const [modelCandidates, setModelCandidates] = useState<AudioModelCandidate[]>(event.model_candidates ?? []);
  const [videoPoseCandidates, setVideoPoseCandidates] = useState<VideoPoseCandidate[]>(event.video_pose_candidates ?? []);
  const [playingRetroAnalysis, setPlayingRetroAnalysis] = useState<PlayingRetroAudioAnalysisResult | null>(null);
  const [playingRetroRunning, setPlayingRetroRunning] = useState(false);
  const [playingRetroStatus, setPlayingRetroStatus] = useState<string | null>(null);
  const [reviewStartProfileText, setReviewStartProfileText] = useState<string | null>(null);
  const [poseAnalysisStatus, setPoseAnalysisStatus] = useState<string | null>(null);
  const [poseAnalysisRunning, setPoseAnalysisRunning] = useState(false);
  const [audioVideoReviewStage, setAudioVideoReviewStage] = useState<AudioVideoReviewStage>(() => (
    event.collection_type === 'audio_video_pose' && event.review?.audio_completed_at && !event.review?.completed_at
      ? 'motion'
      : 'audio'
  ));
  const [detectionConfigSnapshot, setDetectionConfigSnapshot] = useState<AudioDetectionConfigSnapshot>(
    event.detection_config_snapshot ?? getDefaultAudioDetectionConfigSnapshot(),
  );
  const [selectedMarkerId, setSelectedMarkerId] = useState<string | null>(event.review?.markers?.[0]?.id ?? null);
  const [playbackPositionMs, setPlaybackPositionMs] = useState(0);
  const [playbackMode, setPlaybackMode] = useState<PlaybackMode>('idle');
  const [playbackRate, setPlaybackRate] = useState<PlaybackRate>(1);
  const [timelineMode, setTimelineMode] = useState<TimelineInteractionMode>('idle');
  const [videoSyncOffsetMs, setVideoSyncOffsetMs] = useState(event.video_recording?.video_sync_offset_ms ?? 0);
  const [audioSyncPoint, setAudioSyncPoint] = useState<AudioSyncPoint | null>(null);
  const [syncCandidateVideoMs, setSyncCandidateVideoMs] = useState<number | null>(null);
  const [videoSyncSource, setVideoSyncSource] = useState<AudioVideoSyncSource>(
    event.video_recording?.video_sync_source ?? 'auto_peak',
  );
  const [syncCalibrationActive, setSyncCalibrationActive] = useState(false);
  const [videoSyncExpanded, setVideoSyncExpanded] = useState(!event.video_recording?.video_sync_offset_ms);
  const [videoReady, setVideoReady] = useState(false);
  const [videoBuffering, setVideoBuffering] = useState(false);
  const [videoPreparingPlayback, setVideoPreparingPlayback] = useState(false);
  const [playingConfidenceFilter, setPlayingConfidenceFilter] = useState<PlayingConfidenceFilterId>('safe');
  const [overviewWidth, setOverviewWidth] = useState(0);
  const [detailWidth, setDetailWidth] = useState(0);
  const [videoProgressWidth, setVideoProgressWidth] = useState(0);
  const [imuPlotWidth, setImuPlotWidth] = useState(0);
  const [zoomLevel, setZoomLevel] = useState<TimelineZoomLevel>(1);
  const [timelineWindowStartMs, setTimelineWindowStartMs] = useState(0);
  const [videoNaturalSize, setVideoNaturalSize] = useState<{
    width: number;
    height: number;
    orientation?: 'landscape' | 'portrait';
  } | null>(null);
  const [quickLabelPrompt, setQuickLabelPrompt] = useState<QuickLabelPrompt | null>(null);
  const overviewBinCount = useMemo(
    () => Math.max(WAVEFORM_MIN_BIN_COUNT, Math.floor(overviewWidth || 320)),
    [overviewWidth],
  );
  const timelineWindowDurationMs = useMemo(
    () => Math.max(500, durationMs > 0 ? durationMs / zoomLevel : 0),
    [durationMs, zoomLevel],
  );
  const safeTimelineWindowStartMs = Math.max(
    0,
    Math.min(timelineWindowStartMs, Math.max(0, durationMs - timelineWindowDurationMs)),
  );
  const timelineWindowEndMs = Math.min(durationMs, safeTimelineWindowStartMs + timelineWindowDurationMs);
  const timelineWindowSpanMs = Math.max(1, timelineWindowEndMs - safeTimelineWindowStartMs);

  const allOrderedMarkers = useMemo(() => sortMarkers(markers), [markers]);
  const visibleMarkerCandidateIds = useMemo(() => new Set(
    allOrderedMarkers
      .filter(marker => {
        const status = marker.review_status ?? 'pending';
        return status !== 'deleted' && status !== 'filtered' && marker.linked_candidate_id;
      })
      .map(marker => marker.linked_candidate_id as string),
  ), [allOrderedMarkers]);
  const visiblePoseCandidateIds = useMemo(() => new Set(
    allOrderedMarkers
      .filter(marker => {
        const status = marker.review_status ?? 'pending';
        return status !== 'deleted' && status !== 'filtered' && marker.linked_pose_candidate_id;
      })
      .map(marker => marker.linked_pose_candidate_id as string),
  ), [allOrderedMarkers]);
  const isAudioVideoPoseReview = event.recording_mode === 'audio_video_pose' || event.collection_type === 'audio_video_pose';
  const isAudioVideoPoseAudioStage = isAudioVideoPoseReview && audioVideoReviewStage === 'audio';
  const isAudioVideoPoseMotionStage = isAudioVideoPoseReview && audioVideoReviewStage === 'motion';
  const isPlayingReview = event.scenario === 'playing' || event.scenario_id === 'free_recording';
  const usesPlayingRetroPrimaryReview = isPlayingReview && isAudioVideoPoseAudioStage;
  const isRacketBouncingReview = (
    event.scenario === 'racket_bouncing' ||
    event.scenario_id === 'racket_motion_no_bounce' ||
    event.scenario_id.startsWith('racket_bounce')
  );
  const allowsMarkerlessNoBounceSave = event.scenario_id === 'racket_motion_no_bounce';
  const isAudioOnlyReview = event.scenario === 'audio_sound' || event.recording_mode === 'guided_audio_only' || event.recording_mode === 'imported_audio';
  const startReviewStartProfile = useCallback((phase: string, detail?: string) => {
    const startedAtMs = Date.now();
    const key = `${event.scenario_id || event.scenario || 'audio'}:${event.take_index ?? 0}:${event.recorded_at ?? event.created_at ?? filePath}`;
    const firstEntry: ReviewStartProfileEntry = {
      phase,
      elapsedMs: 0,
      detail,
    };
    reviewStartProfileRef.current = {
      key,
      startedAtMs,
      entries: [firstEntry],
    };
    setReviewStartProfileText(reviewStartProfileSummary([firstEntry]));
    console.log(`${REVIEW_START_PROFILE_LOG_PREFIX} ${key} +0ms ${phase}${detail ? ` | ${detail}` : ''}`);
  }, [event.created_at, event.recorded_at, event.scenario, event.scenario_id, event.take_index, filePath]);
  const markReviewStartProfile = useCallback((phase: string, detail?: string, durationMs?: number) => {
    const profile = reviewStartProfileRef.current;
    if (!profile) return;
    const elapsedMs = Date.now() - profile.startedAtMs;
    const entry: ReviewStartProfileEntry = {
      phase,
      elapsedMs,
      durationMs,
      detail,
    };
    profile.entries.push(entry);
    setReviewStartProfileText(reviewStartProfileSummary(profile.entries));
    const durationText = typeof durationMs === 'number' ? ` duration=${durationMs}ms` : '';
    console.log(`${REVIEW_START_PROFILE_LOG_PREFIX} ${profile.key} +${elapsedMs}ms ${phase}${durationText}${detail ? ` | ${detail}` : ''}`);
  }, []);
  const quickLabelChoices = useMemo(() => {
    if (isAudioVideoPoseReview) return AUDIO_VIDEO_POSE_REVIEW_LABEL_CHOICES;
    if (isPlayingReview) return PLAYING_REVIEW_LABEL_CHOICES;
    if (isRacketBouncingReview) return RACKET_BOUNCING_REVIEW_LABEL_CHOICES;
    if (isAudioOnlyReview) return BASE_REVIEW_LABEL_CHOICES;
    return [];
  }, [isAudioOnlyReview, isAudioVideoPoseReview, isPlayingReview, isRacketBouncingReview]);
  const supportsQuickLabels = quickLabelChoices.length > 0;
  const handleDetectionConfigChange = useCallback((
    nextSensitivity: AudioDetectionSensitivity,
    nextDetectionMode: AudioDetectionMode,
  ) => {
    const nextConfig = getAudioDetectionConfig(nextSensitivity, nextDetectionMode);
    const decoded = decodedRef.current;
    setDetectionConfigSnapshot(nextConfig);
    setPlayingRetroAnalysis(null);
    setPlayingRetroStatus(null);
    setQuickLabelPrompt(null);

    if (!decoded) {
      setModelCandidates([]);
      return;
    }

    const reviewData = buildSuggestedReviewData(
      decoded.samples,
      decoded.sampleRate,
      event.scenario_id,
      nextConfig,
      isAudioVideoPoseReview
        ? { min_contact_confidence: audioVideoPoseReviewConfidence(nextConfig) }
        : undefined,
    );
    const preservedMarkers = markers.filter(marker => (
      shouldPreserveMarkerWhenConfigChanges(marker) ||
      (isAudioVideoPoseReview && isMotionLayerMarker(marker))
    ));
    const preservedCandidateIds = new Set(
      preservedMarkers
        .map(marker => marker.linked_candidate_id)
        .filter((candidateId): candidateId is string => Boolean(candidateId)),
    );
    const preservedCandidates = modelCandidates.filter(candidate => preservedCandidateIds.has(candidate.id));
    const nextAutoMarkers = reviewData.markers
      .filter(autoMarker => !preservedMarkers.some(marker => (
        Math.abs(marker.timestamp_ms - autoMarker.timestamp_ms) <= PRESERVED_MARKER_DUPLICATE_GAP_MS
      )))
      .map((autoMarker, index) => ({
        ...autoMarker,
        id: `${autoMarker.id}_${nextConfig.config_id}_${index}`,
      }));
    const mergedMarkers = sortMarkers([...preservedMarkers, ...nextAutoMarkers]);

    setMarkers(mergedMarkers);
    setModelCandidates([...preservedCandidates, ...reviewData.model_candidates]);
    setSelectedMarkerId(previousSelectedId => (
      previousSelectedId && mergedMarkers.some(marker => marker.id === previousSelectedId)
        ? previousSelectedId
      : mergedMarkers[0]?.id ?? null
    ));
  }, [event.scenario_id, isAudioVideoPoseReview, markers, modelCandidates]);
  const handleShowModelInfo = useCallback(() => {
    if (usesPlayingRetroPrimaryReview) {
      Alert.alert(
        'Spel-retro audio',
        [
          'Denna ljudreview anvander spel-retro som primar klassare for de ljudkandidater appen redan hittat.',
          'Retro skapar vanliga editbara reviewforslag: rackettraff, bordsstuds eller dolda ej-target-kandidater.',
          'T0014 skannar dessutom efter extra tata peaks som saknas bland sparade kandidater och visar bara starka racket/bord-recoveryforslag.',
          'Bla ram betyder att markern ar lankad mellan ljud- och rorelselagret. Det ar inte en raderad marker.',
          'T0020 tar bort tata dubletter av samma ljudtyp inom 80 ms, men behaller tata racket+bord-par.',
          'Studsdetektor och vanlig upp/ner-studs ar oforandrade.',
        ].join('\n\n'),
      );
      return;
    }
    const contactThreshold = Math.round(audioVideoPoseReviewConfidence(detectionConfigSnapshot) * 100);
    const configContactThreshold = Math.round(detectionConfigSnapshot.contact_confidence_min * 100);
    const surfaceThreshold = Math.round(detectionConfigSnapshot.surface_veto_confidence * 100);
    Alert.alert(
      detectionConfigTitle(detectionConfigSnapshot),
      [
        `Normal styr hur aggressivt appen skapar ljudkandidater. Den här konfigurationen har rackettröskel ${configContactThreshold}% och merge-window ${detectionConfigSnapshot.merge_window_ms} ms.`,
        '4-klass betyder att varje ljudpeak klassas som racket, bord, golv eller brus/noise.',
        `I Ljud + video ML blir auto-racket bara review-markers om de är minst ${contactThreshold}%.`,
        `Bord/golv/noise visas som ytförslag när 4-klassmodellen är tydlig nog. Surface-veto i denna config är ${surfaceThreshold}%.`,
        'Auto-förslag är bara förslag tills du bekräftar, ändrar, ignorerar eller tar bort dem.',
      ].join('\n\n'),
    );
  }, [detectionConfigSnapshot, usesPlayingRetroPrimaryReview]);
  const handleRunPlayingRetroAudio = useCallback(() => {
    const decoded = decodedRef.current;
    if (!decoded) {
      Alert.alert('Spel-retro audio', 'Waveform Ã¤r inte laddad Ã¤n.');
      return;
    }
    if (modelCandidates.length === 0) {
      Alert.alert('Spel-retro audio', 'Det finns inga sparade ljudkandidater att reklassificera.');
      return;
    }

    setPlayingRetroRunning(true);
    setPlayingRetroStatus('Analyserar sparade kandidater...');
    setTimeout(() => {
      try {
        const analysis = analyzePlayingRetroAudioCandidates(
          decoded.samples,
          decoded.sampleRate,
          modelCandidates,
        );
        const racket = analysis.candidates.filter(candidate => (
          candidate.playing_retro_prediction.label === 'racket_contact'
        )).length;
        const table = analysis.candidates.filter(candidate => (
          candidate.playing_retro_prediction.label === 'table_bounce'
        )).length;
        const nonTarget = analysis.candidates.length - racket - table;
        setPlayingRetroAnalysis(analysis);
        setPlayingRetroStatus(`Klart: ${racket} racket, ${table} bord, ${nonTarget} ej target.`);
      } catch (error) {
        setPlayingRetroStatus('Kunde inte kÃ¶ra spel-retro.');
        Alert.alert('Spel-retro audio', String(error));
      } finally {
        setPlayingRetroRunning(false);
      }
    }, 0);
  }, [modelCandidates]);
  const activePlayingConfidenceFilter = useMemo(
    () => PLAYING_CONFIDENCE_FILTERS.find(filter => filter.id === playingConfidenceFilter) ?? PLAYING_CONFIDENCE_FILTERS[2],
    [playingConfidenceFilter],
  );
  const playingRetroModelMetadata = useMemo(() => getPlayingRetroAudioModelMetadata(), []);
  const playingRetroSummary = useMemo(() => {
    const candidates = playingRetroAnalysis?.candidates ?? [];
    const racket = candidates.filter(candidate => candidate.playing_retro_prediction.label === 'racket_contact').length;
    const table = candidates.filter(candidate => candidate.playing_retro_prediction.label === 'table_bounce').length;
    const nonTarget = candidates.filter(candidate => candidate.playing_retro_prediction.label === 'non_target').length;
    const targets = racket + table;
    return {
      total: candidates.length,
      racket,
      table,
      nonTarget,
      targets,
    };
  }, [playingRetroAnalysis]);
  const orderedMarkers = useMemo(() => {
    const nonDeletedMarkers = allOrderedMarkers.filter(marker => {
      const status = marker.review_status ?? 'pending';
      return status !== 'deleted' && status !== 'filtered';
    });
    if (!isPlayingReview || isAudioVideoPoseReview) return nonDeletedMarkers;
    return nonDeletedMarkers.filter(marker => markerPassesPlayingFilter(
      marker,
      activePlayingConfidenceFilter.minConfidence,
    ));
  }, [activePlayingConfidenceFilter.minConfidence, allOrderedMarkers, isAudioVideoPoseReview, isPlayingReview]);
  const playingAutoMarkerStats = useMemo(() => {
    const autoMarkers = allOrderedMarkers.filter(marker => (
      marker.source === 'auto' &&
      (marker.review_status ?? 'pending') !== 'deleted' &&
      (marker.review_status ?? 'pending') !== 'filtered'
    ));
    const visibleAutoMarkers = orderedMarkers.filter(marker => (
      marker.source === 'auto' &&
      (marker.review_status ?? 'pending') !== 'deleted'
    ));
    return {
      total: autoMarkers.length,
      visible: visibleAutoMarkers.length,
    };
  }, [allOrderedMarkers, orderedMarkers]);
  const selectedMarker = useMemo(
    () => orderedMarkers.find(marker => marker.id === selectedMarkerId) ?? null,
    [orderedMarkers, selectedMarkerId],
  );
  const audioTimelineMarkers = useMemo(
    () => orderedMarkers.filter(isAudioLayerMarker),
    [orderedMarkers],
  );
  const motionTimelineMarkers = useMemo(
    () => orderedMarkers.filter(isMotionLayerMarker),
    [orderedMarkers],
  );
  const activeStageMarkers = useMemo(() => {
    if (isAudioVideoPoseMotionStage) return motionTimelineMarkers;
    if (isAudioVideoPoseAudioStage) return audioTimelineMarkers;
    return orderedMarkers;
  }, [
    audioTimelineMarkers,
    isAudioVideoPoseAudioStage,
    isAudioVideoPoseMotionStage,
    motionTimelineMarkers,
    orderedMarkers,
  ]);
  const selectedMarkerLayer = selectedMarker ? markerLayer(selectedMarker) : null;
  const selectedAudioMarker = selectedMarkerLayer === 'audio' && !isAudioVideoPoseMotionStage ? selectedMarker : null;
  const selectedMotionMarker = selectedMarkerLayer === 'motion' && !isAudioVideoPoseAudioStage ? selectedMarker : null;
  const linkedAudioMarkerIds = useMemo(() => {
    const linkedIds = new Set<string>();
    for (const audioMarker of audioTimelineMarkers) {
      if (motionTimelineMarkers.some(motionMarker => (
        Math.abs(motionMarker.timestamp_ms - audioMarker.timestamp_ms) <= VIDEO_POSE_AUDIO_MERGE_MS
      ))) {
        linkedIds.add(audioMarker.id);
      }
    }
    return linkedIds;
  }, [audioTimelineMarkers, motionTimelineMarkers]);
  const linkedMotionMarkerIds = useMemo(() => {
    const linkedIds = new Set<string>();
    for (const motionMarker of motionTimelineMarkers) {
      if (audioTimelineMarkers.some(audioMarker => (
        Math.abs(audioMarker.timestamp_ms - motionMarker.timestamp_ms) <= VIDEO_POSE_AUDIO_MERGE_MS
      ))) {
        linkedIds.add(motionMarker.id);
      }
    }
    return linkedIds;
  }, [audioTimelineMarkers, motionTimelineMarkers]);
  const selectedMarkerIndex = useMemo(
    () => selectedMarker ? orderedMarkers.findIndex(marker => marker.id === selectedMarker.id) : -1,
    [orderedMarkers, selectedMarker],
  );
  const selectedStageMarkerIndex = useMemo(
    () => selectedMarker ? activeStageMarkers.findIndex(marker => marker.id === selectedMarker.id) : -1,
    [activeStageMarkers, selectedMarker],
  );
  const reviewLabelChoices = useMemo(
    () => {
      if (isAudioVideoPoseReview) return AUDIO_VIDEO_POSE_REVIEW_LABEL_CHOICES;
      if (isPlayingReview) return PLAYING_REVIEW_LABEL_CHOICES;
      if (isRacketBouncingReview) return RACKET_BOUNCING_REVIEW_LABEL_CHOICES;
      return BASE_REVIEW_LABEL_CHOICES;
    },
    [isAudioVideoPoseReview, isPlayingReview, isRacketBouncingReview],
  );

  const detailFocusMs = playbackMode === 'playing_full_take'
    ? playbackPositionMs
    : selectedMarker?.timestamp_ms ?? playbackPositionMs;

  const detailWindow = useMemo(() => {
    if (!decodedRef.current) {
      return { bins: [], start_ms: 0, end_ms: 0, focus_ms: 0, peak_ms: 0 };
    }
    return buildMarkerZoomWaveformWindow(
      decodedRef.current.samples,
      decodedRef.current.sampleRate,
      detailFocusMs,
      Math.max(280, Math.round((detailWidth || 720) / 2)),
      DETAIL_PRE_MS,
      DETAIL_POST_MS,
    );
  }, [detailFocusMs, detailWidth]);
  const overviewFocusMs = playbackPositionMs;
  const overviewWindow = useMemo(() => {
    if (!decodedRef.current) {
      return { bins: overviewBins, start_ms: 0, end_ms: durationMs };
    }
    const totalDurationMs = durationMs;
    if (totalDurationMs <= 0) {
      return { bins: [], start_ms: 0, end_ms: 0 };
    }

    const desiredSpanMs = OVERVIEW_PRE_MS + OVERVIEW_POST_MS;
    const safeSpanMs = Math.min(totalDurationMs, desiredSpanMs);
    let startMs = Math.max(0, overviewFocusMs - OVERVIEW_PRE_MS);
    let endMs = Math.min(totalDurationMs, startMs + safeSpanMs);
    startMs = Math.max(0, endMs - safeSpanMs);

    const startSample = Math.max(0, Math.floor((startMs / 1000) * decodedRef.current.sampleRate));
    const endSample = Math.min(
      decodedRef.current.samples.length,
      Math.ceil((endMs / 1000) * decodedRef.current.sampleRate),
    );

    return {
      bins: buildWaveformBins(decodedRef.current.samples.slice(startSample, endSample), overviewBinCount),
      start_ms: Math.round(startMs),
      end_ms: Math.round(endMs),
    };
  }, [durationMs, overviewBinCount, overviewBins, overviewFocusMs]);
  const timelineBins = useMemo(() => {
    if (!decodedRef.current) return overviewBins;
    const startSample = Math.max(
      0,
      Math.floor((safeTimelineWindowStartMs / 1000) * decodedRef.current.sampleRate),
    );
    const endSample = Math.min(
      decodedRef.current.samples.length,
      Math.ceil((timelineWindowEndMs / 1000) * decodedRef.current.sampleRate),
    );
    return buildWaveformBins(decodedRef.current.samples.slice(startSample, endSample), overviewBinCount);
  }, [overviewBinCount, overviewBins, safeTimelineWindowStartMs, timelineWindowEndMs]);
  const hasVideo = Boolean(videoFilePath && event.video_recording);
  const setTimelineInteractionMode = useCallback((nextMode: TimelineInteractionMode) => {
    timelineModeRef.current = nextMode;
    setTimelineMode(nextMode);
  }, []);

  const setSyncCalibrationMode = useCallback((active: boolean) => {
    syncCalibrationActiveRef.current = active;
    setSyncCalibrationActive(active);
  }, []);

  const stopCurrentPlayback = useCallback(async (nextMode: PlaybackMode = 'idle') => {
    playbackModeRef.current = nextMode;
    setPlaybackMode(nextMode);
    if (nextMode !== 'paused_full_take') {
      pausedAtAudioMsRef.current = null;
    }

    const tempPath = tempPlaybackPathRef.current;
    tempPlaybackPathRef.current = null;

    await playerRef.current.stopPlayer().catch(() => {});

    if (tempPath) {
      await RNFS.unlink(tempPath).catch(() => {});
    }
  }, []);

  const updateTimelineWindowForPlayhead = useCallback((nextTimestampMs: number) => {
    if (
      timelineModeRef.current === 'scrubbing' ||
      timelineModeRef.current === 'autoScrollingWhileScrubbing' ||
      syncCalibrationActiveRef.current
    ) {
      return;
    }
    const clampedMs = clampTimestamp(nextTimestampMs, durationMs);
    if (zoomLevel <= 1) {
      setTimelineWindowStartMs(0);
      return;
    }
    setTimelineWindowStartMs(prev => followedTimelineStart(
      clampedMs,
      prev,
      durationMs,
      timelineWindowDurationMs,
    ));
  }, [durationMs, timelineWindowDurationMs, zoomLevel]);

  const seekVideoToAudioMsWithOffset = useCallback((
    nextTimestampMs: number,
    nextVideoSyncOffsetMs: number,
    force = false,
  ) => {
    if (!hasVideo) return;
    const nowMs = Date.now();
    if (!force && nowMs - lastVideoSeekAtRef.current < VIDEO_SEEK_THROTTLE_MS) return;
    lastVideoSeekAtRef.current = nowMs;
    videoRef.current?.seek(mapAudioPlayheadToVideoMs(event, nextTimestampMs, durationMs, nextVideoSyncOffsetMs) / 1000);
  }, [durationMs, event, hasVideo]);

  const seekVideoToAudioMs = useCallback((nextTimestampMs: number, force = false) => {
    seekVideoToAudioMsWithOffset(nextTimestampMs, videoSyncOffsetMs, force);
  }, [seekVideoToAudioMsWithOffset, videoSyncOffsetMs]);

  const seekToMs = useCallback((
    nextTimestampMs: number,
    options: { seekVideo?: boolean; throttleVideo?: boolean } = {},
  ) => {
    const clampedMs = clampTimestamp(nextTimestampMs, durationMs);
    playbackPositionRef.current = clampedMs;
    if (playbackModeRef.current === 'paused_full_take') {
      pausedAtAudioMsRef.current = clampedMs;
    }
    setPlaybackPositionMs(clampedMs);
    updateTimelineWindowForPlayhead(clampedMs);
    if (
      options.seekVideo !== false &&
      hasVideo &&
      playbackModeRef.current !== 'playing_full_take' &&
      playbackModeRef.current !== 'playing_preview'
    ) {
      seekVideoToAudioMs(clampedMs, options.throttleVideo === false);
    }
  }, [durationMs, hasVideo, seekVideoToAudioMs, updateTimelineWindowForPlayhead]);

  const setPlayheadMs = useCallback((nextTimestampMs: number) => {
    seekToMs(nextTimestampMs);
  }, [seekToMs]);

  const playerPositionMsForAudioMs = useCallback((audioMs: number) => {
    const playbackRateValue = playbackRateRef.current || 1;
    const relativeAudioMs = Math.max(0, audioMs - playbackSourceStartMsRef.current);
    return Math.round(relativeAudioMs / playbackRateValue);
  }, []);

  const estimateCurrentAudioMsFromClock = useCallback(() => {
    const mode = playbackModeRef.current;
    if (mode !== 'playing_full_take' && mode !== 'playing_preview') {
      return clampTimestamp(playbackPositionRef.current, durationMs);
    }

    const latestCallbackMs = latestAudioCallbackPositionMsRef.current || playbackPositionRef.current;
    const latestCallbackWallClockMs = latestAudioCallbackWallClockMsRef.current;
    if (!latestCallbackWallClockMs) {
      return clampTimestamp(latestCallbackMs, durationMs);
    }

    const elapsedWallClockMs = Math.max(0, Date.now() - latestCallbackWallClockMs);
    return clampTimestamp(latestCallbackMs + elapsedWallClockMs * playbackRateRef.current, durationMs);
  }, [durationMs]);

  const lockPlaybackPositionToAudioMs = useCallback((nextTimestampMs: number, seekVideo = true) => {
    const clampedMs = clampTimestamp(nextTimestampMs, durationMs);
    playbackPositionRef.current = clampedMs;
    pausedAtAudioMsRef.current = clampedMs;
    latestAudioCallbackPositionMsRef.current = clampedMs;
    latestAudioCallbackWallClockMsRef.current = Date.now();
    setPlaybackPositionMs(clampedMs);
    updateTimelineWindowForPlayhead(clampedMs);
    if (seekVideo) {
      seekVideoToAudioMs(clampedMs, true);
    }
    return clampedMs;
  }, [durationMs, seekVideoToAudioMs, updateTimelineWindowForPlayhead]);

  const setManualVideoSyncOffset = useCallback((nextOffsetMs: number) => {
    const clampedOffsetMs = Math.max(-3000, Math.min(3000, Math.round(nextOffsetMs)));
    setVideoSyncOffsetMs(clampedOffsetMs);
    seekVideoToAudioMsWithOffset(playbackPositionRef.current, clampedOffsetMs, true);
  }, [seekVideoToAudioMsWithOffset]);

  const nudgeVideoSyncOffset = useCallback((deltaMs: number) => {
    setManualVideoSyncOffset(videoSyncOffsetMs + deltaMs);
  }, [setManualVideoSyncOffset, videoSyncOffsetMs]);

  const updateSelectedMarkerTimestamp = useCallback((nextTimestampMs: number) => {
    if (!selectedMarkerId) return;
    setMarkers(prev => sortMarkers(
      prev.map(marker => (
        marker.id === selectedMarkerId
          ? {
              ...marker,
              timestamp_ms: clampTimestamp(nextTimestampMs, durationMs),
              review_status: marker.final_label === 'ignore' ? 'ignored' : 'edited',
            }
          : marker
      )),
    ));
  }, [durationMs, selectedMarkerId]);

  const stopTimelineEdgeScroll = useCallback(() => {
    if (edgeScrollTimerRef.current) {
      clearInterval(edgeScrollTimerRef.current);
      edgeScrollTimerRef.current = null;
    }
    edgeScrollDirectionRef.current = 0;
  }, []);

  const stopForScrubIfNeeded = useCallback(() => {
    if (playbackModeRef.current === 'playing_full_take') {
      void stopCurrentPlayback('idle');
    } else if (playbackModeRef.current === 'playing_preview') {
      void stopCurrentPlayback('idle');
    }
  }, [stopCurrentPlayback]);

  const seekVideoOnlyToMs = useCallback((nextVideoMs: number) => {
    if (!event.video_recording) return;
    const clampedVideoMs = clampTimestamp(nextVideoMs, event.video_recording.duration_ms);
    setSyncCandidateVideoMs(clampedVideoMs);
    videoRef.current?.seek(clampedVideoMs / 1000);
  }, [event.video_recording]);

  const setAudioSyncHere = useCallback(() => {
    if (!event.video_recording) return;
    stopForScrubIfNeeded();
    stopTimelineEdgeScroll();
    const syncAudioMs = clampTimestamp(playbackPositionRef.current, durationMs);
    const nextSyncPoint = syncPointFromAudioMs(syncAudioMs, durationMs, 1);
    const nextVideoMs = mapAudioPlayheadToVideoMs(event, syncAudioMs, durationMs, videoSyncOffsetMs);
    setAudioSyncPoint(nextSyncPoint);
    setVideoSyncSource('manual');
    setVideoSyncExpanded(true);
    setSyncCalibrationMode(true);
    playbackPositionRef.current = syncAudioMs;
    latestScrubMsRef.current = syncAudioMs;
    setPlaybackPositionMs(syncAudioMs);
    updateTimelineWindowForPlayhead(syncAudioMs);
    seekVideoOnlyToMs(nextVideoMs);
  }, [
    durationMs,
    event,
    seekVideoOnlyToMs,
    setSyncCalibrationMode,
    stopForScrubIfNeeded,
    stopTimelineEdgeScroll,
    updateTimelineWindowForPlayhead,
    videoSyncOffsetMs,
  ]);

  const startSyncCalibration = useCallback(() => {
    if (!audioSyncPoint || !event.video_recording) return;
    stopForScrubIfNeeded();
    stopTimelineEdgeScroll();
    setSyncCalibrationMode(true);
    setVideoSyncExpanded(true);
    const syncAudioMs = clampTimestamp(audioSyncPoint.timestamp_ms, durationMs);
    const currentCandidateMs = syncCandidateVideoMs ?? mapAudioPlayheadToVideoMs(
      event,
      syncAudioMs,
      durationMs,
      videoSyncOffsetMs,
    );
    playbackPositionRef.current = syncAudioMs;
    latestScrubMsRef.current = syncAudioMs;
    setPlaybackPositionMs(syncAudioMs);
    updateTimelineWindowForPlayhead(syncAudioMs);
    seekVideoOnlyToMs(currentCandidateMs);
  }, [
    audioSyncPoint,
    durationMs,
    event,
    seekVideoOnlyToMs,
    setSyncCalibrationMode,
    stopForScrubIfNeeded,
    stopTimelineEdgeScroll,
    syncCandidateVideoMs,
    updateTimelineWindowForPlayhead,
    videoSyncOffsetMs,
  ]);

  const adjustSyncCandidateVideo = useCallback((deltaMs: number) => {
    if (!audioSyncPoint || !event.video_recording) {
      nudgeVideoSyncOffset(deltaMs);
      return;
    }
    const syncAudioMs = clampTimestamp(audioSyncPoint.timestamp_ms, durationMs);
    const baseCandidateMs = syncCandidateVideoMs ?? mapAudioPlayheadToVideoMs(
      event,
      syncAudioMs,
      durationMs,
      videoSyncOffsetMs,
    );
    if (!syncCalibrationActiveRef.current) {
      setSyncCalibrationMode(true);
    }
    setVideoSyncExpanded(true);
    playbackPositionRef.current = syncAudioMs;
    latestScrubMsRef.current = syncAudioMs;
    setPlaybackPositionMs(syncAudioMs);
    updateTimelineWindowForPlayhead(syncAudioMs);
    seekVideoOnlyToMs(baseCandidateMs + deltaMs);
  }, [
    audioSyncPoint,
    durationMs,
    event,
    nudgeVideoSyncOffset,
    seekVideoOnlyToMs,
    setSyncCalibrationMode,
    syncCandidateVideoMs,
    updateTimelineWindowForPlayhead,
    videoSyncOffsetMs,
  ]);

  const applySyncCandidateHere = useCallback(() => {
    if (!audioSyncPoint || !event.video_recording || syncCandidateVideoMs === null) return;
    const syncAudioMs = clampTimestamp(audioSyncPoint.timestamp_ms, durationMs);
    const nextOffsetMs = offsetFromVideoSyncAnchor(event, syncAudioMs, durationMs, syncCandidateVideoMs);
    playbackPositionRef.current = syncAudioMs;
    latestScrubMsRef.current = syncAudioMs;
    setPlaybackPositionMs(syncAudioMs);
    updateTimelineWindowForPlayhead(syncAudioMs);
    setSyncCalibrationMode(false);
    setManualVideoSyncOffset(nextOffsetMs);
    setVideoSyncSource('manual');
    setVideoSyncExpanded(false);
  }, [
    audioSyncPoint,
    durationMs,
    event,
    setManualVideoSyncOffset,
    setSyncCalibrationMode,
    syncCandidateVideoMs,
    updateTimelineWindowForPlayhead,
  ]);

  const resetGuidedVideoSync = useCallback(() => {
    setSyncCalibrationMode(false);
    setManualVideoSyncOffset(0);
    setVideoSyncSource('auto_peak');
    setVideoSyncExpanded(true);
    if (audioSyncPoint && event.video_recording) {
      const resetCandidateMs = mapAudioPlayheadToVideoMs(event, audioSyncPoint.timestamp_ms, durationMs, 0);
      setSyncCandidateVideoMs(resetCandidateMs);
    }
  }, [audioSyncPoint, durationMs, event, setManualVideoSyncOffset, setSyncCalibrationMode]);

  const handleSelectMarker = useCallback((marker: AudioReviewMarker) => {
    setQuickLabelPrompt(null);
    setSelectedMarkerId(marker.id);
    setPlayheadMs(marker.timestamp_ms);
  }, [setPlayheadMs]);

  const handleSelectPoseCandidate = useCallback((candidate: VideoPoseCandidate) => {
    setQuickLabelPrompt(null);
    const linkedMarker = orderedMarkers.find(marker => marker.linked_pose_candidate_id === candidate.id);
    if (linkedMarker) {
      handleSelectMarker(linkedMarker);
      return;
    }
    setSelectedMarkerId(null);
    setPlayheadMs(candidate.timestamp_ms);
  }, [handleSelectMarker, orderedMarkers, setPlayheadMs]);

  const handleJumpToMarker = useCallback((direction: -1 | 1) => {
    if (activeStageMarkers.length === 0) return;
    if (selectedStageMarkerIndex < 0) {
      handleSelectMarker(direction > 0 ? activeStageMarkers[0] : activeStageMarkers[activeStageMarkers.length - 1]);
      return;
    }
    const nextIndex = selectedStageMarkerIndex + direction;
    if (nextIndex < 0 || nextIndex >= activeStageMarkers.length) return;
    handleSelectMarker(activeStageMarkers[nextIndex]);
  }, [activeStageMarkers, handleSelectMarker, selectedStageMarkerIndex]);

  const measureTimelineSurface = useCallback((onMeasured?: () => void) => {
    timelineSurfaceRef.current?.measureInWindow((pageX, _pageY, width) => {
      if (width > 0) {
        scrubTimelineLayoutRef.current = { pageX, width };
        setOverviewWidth(width);
      }
      onMeasured?.();
    });
  }, []);

  const measureVideoProgressTrack = useCallback((onMeasured?: () => void) => {
    videoProgressTrackRef.current?.measureInWindow((pageX, _pageY, width) => {
      if (width > 0) {
        videoProgressLayoutRef.current = { pageX, width };
        setVideoProgressWidth(width);
      }
      onMeasured?.();
    });
  }, []);

  const setScrubPlaybackPosition = useCallback((nextTimestampMs: number, forceVideoSeek = false) => {
    const clampedMs = clampTimestamp(nextTimestampMs, durationMs);
    latestScrubMsRef.current = clampedMs;
    playbackPositionRef.current = clampedMs;
    setPlaybackPositionMs(clampedMs);
    seekVideoToAudioMs(clampedMs, forceVideoSeek);
  }, [durationMs, seekVideoToAudioMs]);

  const applyTimelineScrubFromPageX = useCallback((pageX: number, forceVideoSeek = false) => {
    const layout = scrubTimelineLayoutRef.current;
    if (layout.width <= 0 || layout.pageX <= 0 || durationMs <= 0) return;
    latestScrubPageXRef.current = pageX;
    const localX = Math.max(
      0,
      Math.min(layout.width, pageX - layout.pageX - scrubFingerOffsetPxRef.current),
    );
    const ratio = localX / layout.width;
    const nextTimestampMs = timelineWindowStartRef.current + ratio * timelineWindowSpanRef.current;
    setScrubPlaybackPosition(nextTimestampMs, forceVideoSeek);
  }, [durationMs, setScrubPlaybackPosition]);

  const startTimelineEdgeScroll = useCallback((direction: -1 | 0 | 1) => {
    if (zoomLevel <= 1 || direction === 0 || durationMs <= timelineWindowDurationMs) {
      stopTimelineEdgeScroll();
      return;
    }
    if (edgeScrollDirectionRef.current === direction && edgeScrollTimerRef.current) return;
    stopTimelineEdgeScroll();
    edgeScrollDirectionRef.current = direction;
    setTimelineInteractionMode('autoScrollingWhileScrubbing');
    edgeScrollTimerRef.current = setInterval(() => {
      if (
        timelineModeRef.current !== 'scrubbing' &&
        timelineModeRef.current !== 'autoScrollingWhileScrubbing'
      ) {
        stopTimelineEdgeScroll();
        return;
      }
      const maxStartMs = Math.max(0, durationMs - timelineWindowSpanRef.current);
      const deltaMs = Math.min(
        900,
        Math.max(24, timelineWindowSpanRef.current * TIMELINE_EDGE_SCROLL_FRACTION),
      ) * direction;
      const nextWindowStartMs = Math.max(
        0,
        Math.min(maxStartMs, timelineWindowStartRef.current + deltaMs),
      );
      if (nextWindowStartMs !== timelineWindowStartRef.current) {
        timelineWindowStartRef.current = nextWindowStartMs;
        setTimelineWindowStartMs(nextWindowStartMs);
      }
      applyTimelineScrubFromPageX(latestScrubPageXRef.current, false);
    }, TIMELINE_EDGE_SCROLL_INTERVAL_MS);
  }, [
    applyTimelineScrubFromPageX,
    durationMs,
    setTimelineInteractionMode,
    stopTimelineEdgeScroll,
    timelineWindowDurationMs,
    zoomLevel,
  ]);

  const updateTimelineEdgeScrollFromPageX = useCallback((pageX: number) => {
    const layout = scrubTimelineLayoutRef.current;
    if (layout.width <= 0 || layout.pageX <= 0 || zoomLevel <= 1) {
      stopTimelineEdgeScroll();
      return;
    }
    const localX = pageX - layout.pageX - scrubFingerOffsetPxRef.current;
    if (localX <= TIMELINE_EDGE_PX) {
      startTimelineEdgeScroll(-1);
    } else if (localX >= layout.width - TIMELINE_EDGE_PX) {
      startTimelineEdgeScroll(1);
    } else {
      stopTimelineEdgeScroll();
      if (timelineModeRef.current === 'autoScrollingWhileScrubbing') {
        setTimelineInteractionMode('scrubbing');
      }
    }
  }, [setTimelineInteractionMode, startTimelineEdgeScroll, stopTimelineEdgeScroll, zoomLevel]);

  const beginTimelineScrub = useCallback((pageX: number) => {
    setTimelineInteractionMode('scrubbing');
    stopTimelineEdgeScroll();
    stopForScrubIfNeeded();
    latestScrubPageXRef.current = pageX;
    latestScrubMsRef.current = playbackPositionRef.current;
    scrubStartWindowMsRef.current = timelineWindowStartRef.current;
    scrubStartPlayheadMsRef.current = playbackPositionRef.current;
    if (scrubTimelineLayoutRef.current.width > 0 && scrubTimelineLayoutRef.current.pageX > 0) {
      const layout = scrubTimelineLayoutRef.current;
      const playheadLeft = ratioToLeft(
        playbackPositionRef.current,
        timelineWindowStartRef.current,
        timelineWindowStartRef.current + timelineWindowSpanRef.current,
        layout.width,
      );
      scrubFingerOffsetPxRef.current = pageX - (layout.pageX + playheadLeft);
    }
    measureTimelineSurface(() => {
      const layout = scrubTimelineLayoutRef.current;
      const playheadLeft = ratioToLeft(
        playbackPositionRef.current,
        timelineWindowStartRef.current,
        timelineWindowStartRef.current + timelineWindowSpanRef.current,
        layout.width,
      );
      scrubFingerOffsetPxRef.current = pageX - (layout.pageX + playheadLeft);
    });
  }, [measureTimelineSurface, setTimelineInteractionMode, stopForScrubIfNeeded, stopTimelineEdgeScroll]);

  const endTimelineScrub = useCallback(() => {
    stopTimelineEdgeScroll();
    setTimelineInteractionMode('idle');
    seekVideoToAudioMs(latestScrubMsRef.current, true);
  }, [seekVideoToAudioMs, setTimelineInteractionMode, stopTimelineEdgeScroll]);

  const clearPlayheadLongPress = useCallback(() => {
    if (playheadLongPressTimerRef.current) {
      clearTimeout(playheadLongPressTimerRef.current);
      playheadLongPressTimerRef.current = null;
    }
  }, []);

  const startPlayheadLongPress = useCallback((pageX: number, layer: ReviewTimelineLayer = 'audio') => {
    clearPlayheadLongPress();
    playheadLongPressStartPageXRef.current = pageX;
    if (!supportsQuickLabels) return;
    if (playheadLongPressHandledRef.current) return;
    playheadLongPressTimerRef.current = setTimeout(() => {
      playheadLongPressTimerRef.current = null;
      playheadLongPressHandledRef.current = true;
      createMarkerFromPlayheadRef.current(layer);
    }, PLAYHEAD_LONG_PRESS_MS);
  }, [clearPlayheadLongPress, supportsQuickLabels]);

  const createTimelinePlayheadResponder = useCallback((layer: ReviewTimelineLayer) => PanResponder.create({
    onStartShouldSetPanResponder: () => true,
    onMoveShouldSetPanResponder: (_, gestureState) => (
      Math.abs(gestureState.dx) > TIMELINE_DRAG_THRESHOLD_PX ||
      Math.abs(gestureState.dy) > TIMELINE_DRAG_THRESHOLD_PX
    ),
    onPanResponderGrant: eventData => {
      playheadLongPressHandledRef.current = false;
      startPlayheadLongPress(eventData.nativeEvent.pageX, layer);
      beginTimelineScrub(eventData.nativeEvent.pageX);
    },
    onPanResponderMove: eventData => {
      if (timelineModeRef.current !== 'scrubbing' && timelineModeRef.current !== 'autoScrollingWhileScrubbing') return;
      const pageX = eventData.nativeEvent.pageX;
      const movedAwayFromPressStart = Math.abs(pageX - playheadLongPressStartPageXRef.current) > TIMELINE_DRAG_THRESHOLD_PX;
      if (movedAwayFromPressStart) {
        clearPlayheadLongPress();
      }
      applyTimelineScrubFromPageX(pageX, false);
      updateTimelineEdgeScrollFromPageX(pageX);
      if (
        movedAwayFromPressStart &&
        timelineModeRef.current === 'scrubbing' &&
        !playheadLongPressHandledRef.current
      ) {
        startPlayheadLongPress(pageX, layer);
      }
    },
    onPanResponderRelease: () => {
      clearPlayheadLongPress();
      playheadLongPressHandledRef.current = false;
      endTimelineScrub();
    },
    onPanResponderTerminate: () => {
      clearPlayheadLongPress();
      playheadLongPressHandledRef.current = false;
      endTimelineScrub();
    },
  }), [
    applyTimelineScrubFromPageX,
    beginTimelineScrub,
    clearPlayheadLongPress,
    endTimelineScrub,
    startPlayheadLongPress,
    updateTimelineEdgeScrollFromPageX,
  ]);

  const overviewPlayheadResponder = useMemo(
    () => createTimelinePlayheadResponder('audio'),
    [createTimelinePlayheadResponder],
  );

  const motionPlayheadResponder = useMemo(
    () => createTimelinePlayheadResponder('motion'),
    [createTimelinePlayheadResponder],
  );

  const overviewMarkerResponder = useMemo(() => PanResponder.create({
    onStartShouldSetPanResponder: () => Boolean(selectedMarker),
    onMoveShouldSetPanResponder: (_, gestureState) => Boolean(selectedMarker) && Math.abs(gestureState.dx) > 2,
    onPanResponderGrant: () => {
      if (!selectedMarker) return;
      stopForScrubIfNeeded();
      overviewMarkerDragStartMsRef.current = selectedMarker.timestamp_ms;
    },
    onPanResponderMove: (_, gestureState) => {
      if (!selectedMarker || overviewWidth <= 0) return;
      const msPerPx = timelineWindowSpanMs / overviewWidth;
      const nextTimestampMs = overviewMarkerDragStartMsRef.current + gestureState.dx * msPerPx;
      updateSelectedMarkerTimestamp(nextTimestampMs);
      setPlayheadMs(nextTimestampMs);
    },
  }), [overviewWidth, selectedMarker, setPlayheadMs, stopForScrubIfNeeded, timelineWindowSpanMs, updateSelectedMarkerTimestamp]);

  const detailMarkerResponder = useMemo(() => PanResponder.create({
    onStartShouldSetPanResponder: () => Boolean(selectedMarker),
    onMoveShouldSetPanResponder: (_, gestureState) => Boolean(selectedMarker) && Math.abs(gestureState.dx) > 1,
    onPanResponderGrant: () => {
      if (!selectedMarker) return;
      stopForScrubIfNeeded();
      detailMarkerDragStartMsRef.current = selectedMarker.timestamp_ms;
    },
    onPanResponderMove: (_, gestureState) => {
      if (!selectedMarker || detailWidth <= 0) return;
      const detailDurationMs = Math.max(1, detailWindow.end_ms - detailWindow.start_ms);
      const msPerPx = detailDurationMs / detailWidth;
      const nextTimestampMs = detailMarkerDragStartMsRef.current + gestureState.dx * msPerPx;
      updateSelectedMarkerTimestamp(nextTimestampMs);
      setPlayheadMs(Math.max(0, nextTimestampMs - REVIEW_PRE_MS));
    },
  }), [detailWidth, detailWindow.end_ms, detailWindow.start_ms, selectedMarker, setPlayheadMs, stopForScrubIfNeeded, updateSelectedMarkerTimestamp]);

  useEffect(() => {
    playbackModeRef.current = playbackMode;
  }, [playbackMode]);

  useEffect(() => {
    playbackPositionRef.current = playbackPositionMs;
  }, [playbackPositionMs]);

  useEffect(() => {
    timelineWindowStartRef.current = safeTimelineWindowStartMs;
  }, [safeTimelineWindowStartMs]);

  useEffect(() => {
    timelineWindowSpanRef.current = timelineWindowSpanMs;
  }, [timelineWindowSpanMs]);

  useEffect(() => {
    playbackRateRef.current = playbackRate;
  }, [playbackRate]);

  useEffect(() => {
    if (!quickLabelPrompt) return;
    if (!allOrderedMarkers.some(marker => marker.id === quickLabelPrompt.markerId)) {
      setQuickLabelPrompt(null);
    }
  }, [allOrderedMarkers, quickLabelPrompt]);

  useEffect(() => {
    playingRetroPrimaryAppliedRef.current = false;
    videoPoseDeferredForAudioRef.current = false;
    setPlayingRetroAnalysis(null);
    setPlayingRetroStatus(null);
  }, [event.recorded_at, event.take_index, filePath]);

  useEffect(() => {
    if (orderedMarkers.length === 0) {
      setSelectedMarkerId(null);
      return;
    }
    if (!selectedMarkerId || !orderedMarkers.some(marker => marker.id === selectedMarkerId)) {
      setSelectedMarkerId(orderedMarkers[0].id);
    }
  }, [orderedMarkers, selectedMarkerId]);

  useEffect(() => {
    if (!isAudioVideoPoseReview) return;
    const stageMarkers = audioVideoReviewStage === 'motion'
      ? motionTimelineMarkers
      : audioTimelineMarkers;
    if (selectedMarkerId && stageMarkers.some(marker => marker.id === selectedMarkerId)) return;
    setSelectedMarkerId(stageMarkers[0]?.id ?? null);
  }, [
    audioTimelineMarkers,
    audioVideoReviewStage,
    isAudioVideoPoseReview,
    motionTimelineMarkers,
    selectedMarkerId,
  ]);

  useEffect(() => {
    if (!hasVideo || !videoRef.current) return;
    if (playbackMode === 'playing_full_take' || playbackMode === 'playing_preview') return;
    if (
      timelineModeRef.current === 'scrubbing' ||
      timelineModeRef.current === 'autoScrollingWhileScrubbing'
    ) {
      return;
    }
    seekVideoToAudioMs(playbackPositionMs);
  }, [hasVideo, playbackMode, playbackPositionMs, seekVideoToAudioMs]);

  useEffect(() => {
    let cancelled = false;
    if (usesPlayingRetroPrimaryReview) {
      startReviewStartProfile('load_start', `savedCandidates=${event.model_candidates?.length ?? 0}`);
    } else {
      reviewStartProfileRef.current = null;
      setReviewStartProfileText(null);
    }
    setLoading(true);

    decodeWavFile(filePath)
      .then(async decoded => {
        if (cancelled) return;
        markReviewStartProfile(
          'wav_decoded',
          `duration=${Math.round(decoded.durationMs)}ms samples=${decoded.samples.length}`,
        );
        decodedRef.current = { sampleRate: decoded.sampleRate, samples: decoded.samples };
        setDurationMs(decoded.durationMs);
        const waveformStartedAt = Date.now();
        const nextOverviewBins = buildWaveformBins(decoded.samples, overviewBinCount);
        markReviewStartProfile(
          'waveform_bins_ready',
          `bins=${nextOverviewBins.length}`,
          Date.now() - waveformStartedAt,
        );
        setOverviewBins(nextOverviewBins);
        const syncStartedAt = Date.now();
        const detectedSyncPoint = detectAudioSyncPoint(decoded.samples, decoded.sampleRate);
        markReviewStartProfile(
          'audio_sync_checked',
          detectedSyncPoint ? `sync=${Math.round(detectedSyncPoint.timestamp_ms)}ms` : 'sync=none',
          Date.now() - syncStartedAt,
        );
        const savedVideoSyncOffsetMs = event.video_recording?.video_sync_offset_ms ?? 0;
        const savedSyncAnchorAudioMs = event.video_recording?.video_sync_anchor_audio_ms;
        const savedSyncAnchorVideoMs = event.video_recording?.video_sync_anchor_video_ms;
        const savedSyncSource = event.video_recording?.video_sync_source;
        const syncPoint = typeof savedSyncAnchorAudioMs === 'number'
          ? syncPointFromAudioMs(savedSyncAnchorAudioMs, decoded.durationMs, savedSyncSource === 'manual' ? 1 : 0.9)
          : detectedSyncPoint;
        const savedDetectionConfig = event.detection_config_snapshot ?? getDefaultAudioDetectionConfigSnapshot();

        const hasCompletedReview = Boolean(event.review?.completed_at || event.review?.audio_completed_at);
        const hasSavedReviewMarkers = Boolean(
          event.review?.markers &&
            (event.review.markers.length > 0 || hasCompletedReview),
        );
        const reviewDataStartedAt = Date.now();
        const reviewData = hasSavedReviewMarkers
          ? null
          : buildSuggestedReviewData(
              decoded.samples,
              decoded.sampleRate,
              event.scenario_id,
              savedDetectionConfig,
              isAudioVideoPoseReview
                ? { min_contact_confidence: audioVideoPoseReviewConfidence(savedDetectionConfig) }
                : undefined,
            );
        markReviewStartProfile(
          hasSavedReviewMarkers ? 'saved_review_loaded' : 'suggested_review_data_ready',
          hasSavedReviewMarkers
            ? `markers=${event.review?.markers?.length ?? 0}`
            : `markers=${reviewData?.markers.length ?? 0} candidates=${reviewData?.model_candidates.length ?? 0}`,
          Date.now() - reviewDataStartedAt,
        );
        const reviewedMarkers = event.review?.markers?.map(marker => ({
          ...marker,
          review_status: marker.review_status ?? 'confirmed',
        }));
        const seedOnlyPlayingRetroCandidates = usesPlayingRetroPrimaryReview && !hasSavedReviewMarkers;
        let nextMarkers = hasSavedReviewMarkers
          ? isAudioVideoPoseReview
            ? splitAudioVideoPoseMarkers(reviewedMarkers ?? [], decoded.durationMs)
            : sortMarkers(reviewedMarkers ?? [])
          : isAudioVideoPoseMotionStage && markers.length > 0
            ? sortMarkers(markers)
          : seedOnlyPlayingRetroCandidates
            ? []
          : reviewData?.markers ?? [];
        if (isAudioVideoPoseReview) {
          const minAudioConfidence = audioVideoPoseReviewConfidence(savedDetectionConfig);
          nextMarkers = nextMarkers.filter(marker => !shouldDropLowConfidenceAudioVideoPoseMarker(marker, minAudioConfidence));
        }
        const nextModelCandidates = event.model_candidates?.length
          ? event.model_candidates
          : reviewData?.model_candidates ?? modelCandidatesFromMarkers(nextMarkers, savedDetectionConfig);
        const nextVideoPoseCandidates = event.video_pose_candidates ?? [];
        if (isAudioVideoPoseReview && videoFilePath) {
          if (!nextMarkers.some(isMotionLayerMarker) && nextVideoPoseCandidates.length > 0) {
            nextMarkers = addPoseCandidatesAsMotionMarkers(
              nextMarkers.filter(isAudioLayerMarker),
              nextVideoPoseCandidates,
              decoded.durationMs,
            );
            videoPoseAutoMarkersSeededRef.current = true;
          }
          if (nextVideoPoseCandidates.length > 0) {
            setPoseAnalysisStatus(poseCandidateReviewStatusText(nextVideoPoseCandidates, false));
          } else if (!hasTrainedVideoStrokeModel()) {
            setPoseAnalysisStatus(`Ingen posemodell exporterad (${videoStrokeModelVersion()}).`);
          } else {
            setPoseAnalysisStatus(isAudioVideoPoseAudioStage
              ? 'Förbereder rörelseanalys i bakgrunden på hela videon...'
              : 'Kör rörelseanalys på hela videon...');
          }
        } else {
          setPoseAnalysisStatus(null);
        }
        const preservePlayingRetroPrimaryMarkers = seedOnlyPlayingRetroCandidates && playingRetroPrimaryAppliedRef.current;
        if (!preservePlayingRetroPrimaryMarkers) {
          setMarkers(nextMarkers);
          setModelCandidates(nextModelCandidates);
        }
        setVideoPoseCandidates(nextVideoPoseCandidates);
        setDetectionConfigSnapshot(reviewData?.detection_config_snapshot ?? savedDetectionConfig);
        const nextStageMarkers = isAudioVideoPoseMotionStage
          ? nextMarkers.filter(isMotionLayerMarker)
          : isAudioVideoPoseAudioStage
            ? nextMarkers.filter(isAudioLayerMarker)
            : nextMarkers;
        if (!preservePlayingRetroPrimaryMarkers) {
          setSelectedMarkerId(nextStageMarkers[0]?.id ?? null);
        }
        playbackPositionRef.current = 0;
        latestAudioCallbackPositionMsRef.current = 0;
        latestAudioCallbackWallClockMsRef.current = 0;
        pausedAtAudioMsRef.current = null;
        setPlaybackPositionMs(0);
        setTimelineWindowStartMs(0);
        setVideoSyncOffsetMs(savedVideoSyncOffsetMs);
        setVideoSyncExpanded(!savedVideoSyncOffsetMs);
        setVideoSyncSource(savedSyncSource ?? (typeof savedSyncAnchorAudioMs === 'number' ? 'manual' : 'auto_peak'));
        setAudioSyncPoint(syncPoint);
        setSyncCandidateVideoMs(syncPoint && event.video_recording
          ? typeof savedSyncAnchorVideoMs === 'number'
            ? clampTimestamp(savedSyncAnchorVideoMs, event.video_recording.duration_ms)
            : mapAudioPlayheadToVideoMs(event, syncPoint.timestamp_ms, decoded.durationMs, savedVideoSyncOffsetMs)
          : null);
        setSyncCalibrationMode(false);
        markReviewStartProfile(
          'audio_state_seeded',
          `markers=${nextMarkers.length} candidates=${nextModelCandidates.length} poseCandidates=${nextVideoPoseCandidates.length}`,
        );
      })
      .catch(error => {
        markReviewStartProfile('load_failed', String(error));
        Alert.alert('Granskningsfel', `Kunde inte läsa tagningen: ${String(error)}`);
      })
      .finally(() => {
        if (!cancelled) {
          markReviewStartProfile('loading_finished');
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [
    event,
    event.model_candidates,
    event.player_handedness,
    event.review?.markers,
    event.scenario_id,
    event.video_pose_candidates,
    filePath,
    isAudioVideoPoseAudioStage,
    isAudioVideoPoseMotionStage,
    isAudioVideoPoseReview,
    markReviewStartProfile,
    overviewBinCount,
    setSyncCalibrationMode,
    startReviewStartProfile,
    usesPlayingRetroPrimaryReview,
    videoFilePath,
  ]);

  useEffect(() => {
    if (!usesPlayingRetroPrimaryReview || loading || playingRetroPrimaryAppliedRef.current) return;
    const decoded = decodedRef.current;
    if (!decoded) return;
    if (modelCandidates.length === 0) {
      setPlayingRetroStatus('Spel-retro: inga ljudkandidater att klassa.');
      return;
    }
    if (shouldKeepExistingReviewMarkersForPlayingRetro(allOrderedMarkers.filter(isAudioLayerMarker))) {
      setPlayingRetroStatus('Spel-retro: befintliga granskade ljudmarkers bevaras.');
      return;
    }

    playingRetroPrimaryAppliedRef.current = true;
    setPlayingRetroRunning(true);
    setPlayingRetroStatus('Spel-retro klassar ljudkandidater...');
    markReviewStartProfile('playing_retro_primary_start', `savedCandidates=${modelCandidates.length}`);
    const timer = setTimeout(() => {
      try {
        markReviewStartProfile('playing_retro_timer_start');
        const retroStartedAt = Date.now();
        const analysis = analyzePlayingRetroAudioCandidates(
          decoded.samples,
          decoded.sampleRate,
          modelCandidates,
          {
            recoverMissingCandidates: true,
            blockedTimestampsMs: allOrderedMarkers.filter(isAudioLayerMarker).map(marker => marker.timestamp_ms),
            onProfileEvent: (phase, detail, durationMs) => {
              markReviewStartProfile(`playing_retro_${phase}`, detail, durationMs);
            },
          },
        );
        markReviewStartProfile(
          'playing_retro_analysis_ready',
          `raw=${analysis.candidate_count} saved=${analysis.saved_candidate_count} recovery=${analysis.recovery_candidate_count} visibleRecovery=${analysis.visible_recovery_candidate_count}`,
          Date.now() - retroStartedAt,
        );
        const retroMarkers = markersFromPlayingRetroAnalysis(analysis.candidates, durationMs);
        const motionMarkersToKeep = allOrderedMarkers.filter(isMotionLayerMarker);
        const mergedMarkers = sortMarkers([...retroMarkers, ...motionMarkersToKeep]);
        const primarySummary = summarizePlayingRetroPrimaryReview(analysis, retroMarkers.length);
        setPlayingRetroAnalysis(analysis);
        setPlayingRetroStatus(playingRetroPrimaryStatusText(primarySummary));
        setModelCandidates(analysis.candidates);
        setMarkers(mergedMarkers);
        setSelectedMarkerId(previousSelectedId => (
          previousSelectedId && mergedMarkers.some(marker => marker.id === previousSelectedId)
            ? previousSelectedId
            : retroMarkers[0]?.id ?? motionMarkersToKeep[0]?.id ?? null
        ));
        markReviewStartProfile('playing_retro_markers_ready', `markers=${retroMarkers.length}`);
      } catch (error) {
        playingRetroPrimaryAppliedRef.current = false;
        markReviewStartProfile('playing_retro_failed', String(error));
        setPlayingRetroStatus(`Spel-retro misslyckades: ${String(error)}`);
      } finally {
        markReviewStartProfile('playing_retro_primary_done');
        setPlayingRetroRunning(false);
      }
    }, 0);

    return () => clearTimeout(timer);
  }, [
    allOrderedMarkers,
    durationMs,
    loading,
    markReviewStartProfile,
    modelCandidates,
    usesPlayingRetroPrimaryReview,
  ]);

  useEffect(() => {
    if (!isAudioVideoPoseReview || !videoFilePath || loading || durationMs <= 0) return;
    const hasMotionMarkers = allOrderedMarkers.some(isMotionLayerMarker);
    if (videoPoseCandidates.length > 0) {
      if (!hasMotionMarkers && !videoPoseAutoMarkersSeededRef.current) {
        setMarkers(prev => {
          const splitMarkers = splitAudioVideoPoseMarkers(prev, durationMs);
          if (splitMarkers.some(isMotionLayerMarker)) return splitMarkers;
          videoPoseAutoMarkersSeededRef.current = true;
          return addPoseCandidatesAsMotionMarkers(
            splitMarkers.filter(isAudioLayerMarker),
            videoPoseCandidates,
            durationMs,
          );
        });
      }
      setPoseAnalysisStatus(poseCandidateReviewStatusText(videoPoseCandidates, false));
      return;
    }
    const shouldDelayPoseUntilAudioMarkersReady = usesPlayingRetroPrimaryReview &&
      isAudioVideoPoseAudioStage &&
      !playingRetroAnalysis &&
      modelCandidates.length > 0 &&
      !shouldKeepExistingReviewMarkersForPlayingRetro(allOrderedMarkers.filter(isAudioLayerMarker));
    if (shouldDelayPoseUntilAudioMarkersReady) {
      setPoseAnalysisStatus('Rörelseanalys väntar tills ljudmarkörer är klara...');
      if (!videoPoseDeferredForAudioRef.current) {
        videoPoseDeferredForAudioRef.current = true;
        markReviewStartProfile('pose_scan_deferred', 'waiting_for_playing_retro_audio_markers');
      }
      return;
    }
    if (hasMotionMarkers || videoPoseScanInFlightRef.current || videoPoseScanCompletedRef.current) return;
    if (!hasTrainedVideoStrokeModel()) {
      setPoseAnalysisStatus(`Ingen posemodell exporterad (${videoStrokeModelVersion()}).`);
      return;
    }

    let cancelled = false;
    videoPoseScanInFlightRef.current = true;
    setPoseAnalysisRunning(true);
    markReviewStartProfile('pose_scan_start', isAudioVideoPoseAudioStage ? 'background_during_audio_stage' : 'motion_stage');
    const poseStartedAt = Date.now();
    setPoseAnalysisStatus(isAudioVideoPoseAudioStage
      ? 'Förbereder rörelseanalys i bakgrunden på hela videon...'
      : 'Kör rörelseanalys på hela videon...');
    buildVideoPoseCandidatesForReview(
      videoFilePath,
      durationMs,
      event.player_handedness ?? 'left',
    )
      .then(candidates => {
        if (cancelled) return;
        markReviewStartProfile('pose_scan_done', `candidates=${candidates.length}`, Date.now() - poseStartedAt);
        videoPoseScanCompletedRef.current = true;
        setVideoPoseCandidates(candidates);
        setPoseAnalysisStatus(poseCandidateReviewStatusText(candidates, false));
        setMarkers(prev => {
          const splitMarkers = splitAudioVideoPoseMarkers(prev, durationMs);
          if (splitMarkers.some(isMotionLayerMarker)) return splitMarkers;
          videoPoseAutoMarkersSeededRef.current = true;
          return addPoseCandidatesAsMotionMarkers(
            splitMarkers.filter(isAudioLayerMarker),
            candidates,
            durationMs,
          );
        });
      })
      .catch(error => {
        if (!cancelled) {
          markReviewStartProfile('pose_scan_failed', String(error), Date.now() - poseStartedAt);
          videoPoseScanCompletedRef.current = true;
          setPoseAnalysisStatus(`Poseanalys misslyckades: ${String(error)}`);
        }
      })
      .finally(() => {
        videoPoseScanInFlightRef.current = false;
        setPoseAnalysisRunning(false);
        markReviewStartProfile('pose_scan_finished');
      });

    return () => {
      cancelled = true;
    };
  }, [
    allOrderedMarkers,
    durationMs,
    event.player_handedness,
    isAudioVideoPoseReview,
    isAudioVideoPoseAudioStage,
    loading,
    markReviewStartProfile,
    modelCandidates.length,
    playingRetroAnalysis,
    usesPlayingRetroPrimaryReview,
    videoFilePath,
    videoPoseCandidates,
  ]);

  useEffect(() => {
    void ReviewOrientation.lockPortrait();
    return () => {
      clearPlayheadLongPress();
      stopTimelineEdgeScroll();
      void stopCurrentPlayback('idle');
      void ReviewOrientation.unlock();
    };
  }, [clearPlayheadLongPress, stopCurrentPlayback, stopTimelineEdgeScroll]);

  useEffect(() => {
    playerRef.current.setSubscriptionDuration(PLAYBACK_SUBSCRIPTION_SEC).catch(() => {});
    playerRef.current.addPlayBackListener(eventData => {
      const currentPosition = Math.round(eventData.currentPosition);
      const duration = Math.round(eventData.duration);

      const timelineIsUserControlled =
        timelineModeRef.current === 'scrubbing' ||
        timelineModeRef.current === 'autoScrollingWhileScrubbing';

      if (
        !timelineIsUserControlled &&
        (playbackModeRef.current === 'playing_full_take' || playbackModeRef.current === 'playing_preview')
      ) {
        const mappedMs = playbackSourceStartMsRef.current + currentPosition * playbackRateRef.current;
        const clampedMs = clampTimestamp(mappedMs, durationMs);
        latestAudioCallbackPositionMsRef.current = clampedMs;
        latestAudioCallbackWallClockMsRef.current = Date.now();
        playbackPositionRef.current = clampedMs;
        setPlaybackPositionMs(clampedMs);
        updateTimelineWindowForPlayhead(clampedMs);
      }

      if (duration > 0 && currentPosition >= duration - 40) {
        if (timelineModeRef.current === 'playing') {
          setTimelineInteractionMode('idle');
        }
        void stopCurrentPlayback('idle');
      }
    });

    return () => {
      playerRef.current.removePlayBackListener();
      void stopCurrentPlayback('idle');
    };
  }, [durationMs, setTimelineInteractionMode, stopCurrentPlayback, updateTimelineWindowForPlayhead]);

  const timestampFromTimelineLocation = useCallback((locationX: number) => {
    if (overviewWidth <= 0 || durationMs <= 0) return null;
    const ratio = Math.max(0, Math.min(1, locationX / overviewWidth));
    return clampTimestamp(safeTimelineWindowStartMs + ratio * timelineWindowSpanMs, durationMs);
  }, [durationMs, overviewWidth, safeTimelineWindowStartMs, timelineWindowSpanMs]);

  const createReviewMarkerAtTimestamp = useCallback((
    timestampMs: number,
    layer: ReviewTimelineLayer = 'audio',
  ) => {
    if (isAudioVideoPoseReview && layer === 'motion') {
      return createAudioVideoPoseMotionMarker(timestampMs, durationMs);
    }
    const baseMarker = createManualMarker(clampTimestamp(timestampMs, durationMs), event.scenario_id);
    return supportsQuickLabels
      ? {
          ...baseMarker,
          review_status: 'pending' as const,
          event_type: undefined,
          class_label: undefined,
          contact_kind: undefined,
          not_racket_kind: undefined,
          bounce_side: undefined,
        }
      : baseMarker;
  }, [durationMs, event.scenario_id, isAudioVideoPoseReview, supportsQuickLabels]);

  const handleCreateMarkerAtTimestamp = useCallback((
    timestampMs: number,
    layer: ReviewTimelineLayer = 'audio',
  ) => {
    const marker = createReviewMarkerAtTimestamp(timestampMs, layer);
    stopForScrubIfNeeded();
    setMarkers(prev => sortMarkers([...prev, marker]));
    setSelectedMarkerId(marker.id);
    setPlayheadMs(marker.timestamp_ms);
    setQuickLabelPrompt(supportsQuickLabels && !isAudioVideoPoseReview
      ? { markerId: marker.id, timestampMs: marker.timestamp_ms, layer }
      : null);
  }, [createReviewMarkerAtTimestamp, isAudioVideoPoseReview, setPlayheadMs, stopForScrubIfNeeded, supportsQuickLabels]);

  useEffect(() => {
    createMarkerFromPlayheadRef.current = (layer: ReviewTimelineLayer = 'audio') => {
      handleCreateMarkerAtTimestamp(playbackPositionRef.current, layer);
    };
  }, [handleCreateMarkerAtTimestamp]);

  const handleFullOverviewPress = useCallback((locationX: number) => {
    if (
      timelineModeRef.current === 'scrubbing' ||
      timelineModeRef.current === 'autoScrollingWhileScrubbing'
    ) {
      return;
    }
    const nextTimestampMs = timestampFromTimelineLocation(locationX);
    if (nextTimestampMs === null) return;

    setQuickLabelPrompt(null);
    stopForScrubIfNeeded();
    setPlayheadMs(nextTimestampMs);
  }, [
    setPlayheadMs,
    stopForScrubIfNeeded,
    timestampFromTimelineLocation,
  ]);

  const handleZoomChange = useCallback((nextZoom: TimelineZoomLevel) => {
    setZoomLevel(nextZoom);
    const nextSpanMs = Math.max(500, durationMs > 0 ? durationMs / nextZoom : 0);
    setTimelineWindowStartMs(centeredTimelineStart(playbackPositionMs, durationMs, nextSpanMs));
  }, [durationMs, playbackPositionMs]);

  const applyVideoProgressScrubFromPageX = useCallback((pageX: number, forceVideoSeek = false) => {
    const layout = videoProgressLayoutRef.current;
    const width = layout.width || videoProgressWidth;
    if (width <= 0 || layout.pageX <= 0 || durationMs <= 0) return;
    const localX = Math.max(0, Math.min(width, pageX - layout.pageX));
    const nextTimestampMs = (localX / width) * durationMs;
    videoScrubPendingMsRef.current = nextTimestampMs;
    seekToMs(nextTimestampMs, { seekVideo: true, throttleVideo: !forceVideoSeek });
  }, [durationMs, seekToMs, videoProgressWidth]);

  const beginVideoProgressScrub = useCallback((pageX: number) => {
    setTimelineInteractionMode('scrubbing');
    stopTimelineEdgeScroll();
    stopForScrubIfNeeded();
    measureVideoProgressTrack(() => applyVideoProgressScrubFromPageX(pageX, false));
  }, [
    applyVideoProgressScrubFromPageX,
    measureVideoProgressTrack,
    setTimelineInteractionMode,
    stopForScrubIfNeeded,
    stopTimelineEdgeScroll,
  ]);

  const endVideoProgressScrub = useCallback(() => {
    const finalTimestampMs = videoScrubPendingMsRef.current;
    if (finalTimestampMs !== null) {
      seekToMs(finalTimestampMs, { seekVideo: true, throttleVideo: false });
      videoScrubPendingMsRef.current = null;
    }
    setTimelineInteractionMode('idle');
  }, [seekToMs, setTimelineInteractionMode]);

  const videoProgressScrubResponder = useMemo(() => PanResponder.create({
    onStartShouldSetPanResponder: () => true,
    onMoveShouldSetPanResponder: (_, gestureState) => (
      Math.abs(gestureState.dx) > TIMELINE_DRAG_THRESHOLD_PX ||
      Math.abs(gestureState.dy) > TIMELINE_DRAG_THRESHOLD_PX
    ),
    onPanResponderGrant: eventData => {
      beginVideoProgressScrub(eventData.nativeEvent.pageX);
    },
    onPanResponderMove: eventData => {
      applyVideoProgressScrubFromPageX(eventData.nativeEvent.pageX, false);
    },
    onPanResponderRelease: endVideoProgressScrub,
    onPanResponderTerminate: endVideoProgressScrub,
  }), [applyVideoProgressScrubFromPageX, beginVideoProgressScrub, endVideoProgressScrub]);

  const handleDetailPress = useCallback((locationX: number) => {
    stopForScrubIfNeeded();
    if (detailWidth <= 0) return;
    const ratio = locationX / detailWidth;
    const nextTimestampMs = detailWindow.start_ms + ratio * (detailWindow.end_ms - detailWindow.start_ms);
    setPlayheadMs(nextTimestampMs);
  }, [detailWidth, detailWindow.end_ms, detailWindow.start_ms, setPlayheadMs, stopForScrubIfNeeded]);

  const startPlaybackAtAudioMs = useCallback(async (startTimestampMs: number) => {
    if (!decodedRef.current) return;

    const startAudioMs = clampTimestamp(startTimestampMs, durationMs);
    await stopCurrentPlayback('idle');
    setVideoPreparingPlayback(hasVideo && (!videoReady || videoBuffering));
    if (hasVideo) {
      videoRef.current?.seek(mapAudioPlayheadToVideoMs(event, startAudioMs, durationMs, videoSyncOffsetMs) / 1000);
    }
    const activePlaybackRate = playbackRateRef.current;
    const playOriginalTake = activePlaybackRate === 1;
    const playbackPath = playOriginalTake
      ? filePath
      : await writeTakePlaybackClip(
          decodedRef.current.samples,
          decodedRef.current.sampleRate,
          startAudioMs,
          activePlaybackRate,
        );
    tempPlaybackPathRef.current = playOriginalTake ? null : playbackPath;
    playbackSourceStartMsRef.current = playOriginalTake ? 0 : startAudioMs;
    playbackPositionRef.current = startAudioMs;
    pausedAtAudioMsRef.current = null;
    setPlaybackPositionMs(startAudioMs);
    updateTimelineWindowForPlayhead(startAudioMs);
    await playerRef.current.startPlayer(playbackPath);
    if (playOriginalTake && startAudioMs > 0) {
      await playerRef.current.seekToPlayer(startAudioMs);
    }
    latestAudioCallbackPositionMsRef.current = startAudioMs;
    latestAudioCallbackWallClockMsRef.current = Date.now();
    playbackModeRef.current = 'playing_full_take';
    setPlaybackMode('playing_full_take');
    setTimelineInteractionMode('playing');
    setVideoPreparingPlayback(false);
  }, [
    durationMs,
    event,
    filePath,
    hasVideo,
    setTimelineInteractionMode,
    stopCurrentPlayback,
    updateTimelineWindowForPlayhead,
    videoBuffering,
    videoReady,
    videoSyncOffsetMs,
  ]);

  const pauseAtCurrentAudioTime = useCallback(async () => {
    const mode = playbackModeRef.current;
    if (mode !== 'playing_full_take') {
      setVideoPreparingPlayback(false);
      await stopCurrentPlayback('idle');
      setTimelineInteractionMode('idle');
      return;
    }

    const pauseAudioMs = lockPlaybackPositionToAudioMs(estimateCurrentAudioMsFromClock(), true);
    playbackModeRef.current = 'paused_full_take';
    setPlaybackMode('paused_full_take');
    setTimelineInteractionMode('idle');
    setVideoPreparingPlayback(false);

    try {
      await playerRef.current.pausePlayer();
    } catch {
      await playerRef.current.stopPlayer().catch(() => {});
    }

    lockPlaybackPositionToAudioMs(pauseAudioMs, true);
  }, [
    estimateCurrentAudioMsFromClock,
    lockPlaybackPositionToAudioMs,
    setTimelineInteractionMode,
    stopCurrentPlayback,
  ]);

  const resumeFromPausedAudioTime = useCallback(async () => {
    const resumeAudioMs = clampTimestamp(pausedAtAudioMsRef.current ?? playbackPositionRef.current, durationMs);

    try {
      setVideoPreparingPlayback(hasVideo && (!videoReady || videoBuffering));
      if (hasVideo) {
        videoRef.current?.seek(mapAudioPlayheadToVideoMs(event, resumeAudioMs, durationMs, videoSyncOffsetMs) / 1000);
      }
      await playerRef.current.seekToPlayer(playerPositionMsForAudioMs(resumeAudioMs));
      playbackPositionRef.current = resumeAudioMs;
      latestAudioCallbackPositionMsRef.current = resumeAudioMs;
      latestAudioCallbackWallClockMsRef.current = Date.now();
      setPlaybackPositionMs(resumeAudioMs);
      updateTimelineWindowForPlayhead(resumeAudioMs);
      playbackModeRef.current = 'playing_full_take';
      setPlaybackMode('playing_full_take');
      setTimelineInteractionMode('playing');
      await playerRef.current.resumePlayer();
      pausedAtAudioMsRef.current = null;
      setVideoPreparingPlayback(false);
    } catch {
      setVideoPreparingPlayback(false);
      await startPlaybackAtAudioMs(resumeAudioMs);
    }
  }, [
    durationMs,
    event,
    hasVideo,
    playerPositionMsForAudioMs,
    setTimelineInteractionMode,
    startPlaybackAtAudioMs,
    updateTimelineWindowForPlayhead,
    videoBuffering,
    videoReady,
    videoSyncOffsetMs,
  ]);

  const handlePlayFromHere = useCallback(async () => {
    try {
      if (playbackModeRef.current === 'paused_full_take') {
        await resumeFromPausedAudioTime();
        return;
      }
      await startPlaybackAtAudioMs(playbackPositionRef.current);
    } catch (error) {
      setVideoPreparingPlayback(false);
      await stopCurrentPlayback('idle');
      Alert.alert('Uppspelningsfel', `Kunde inte spela från vald position: ${String(error)}`);
    }
  }, [resumeFromPausedAudioTime, startPlaybackAtAudioMs, stopCurrentPlayback]);

  const handlePause = useCallback(async () => {
    if (playbackModeRef.current === 'playing_full_take') {
      await pauseAtCurrentAudioTime();
      return;
    }
    setVideoPreparingPlayback(false);
    await stopCurrentPlayback('idle');
    setTimelineInteractionMode('idle');
  }, [pauseAtCurrentAudioTime, setTimelineInteractionMode, stopCurrentPlayback]);

  const handleToggleVideoPlayback = useCallback(() => {
    if (playbackModeRef.current === 'playing_full_take') {
      void pauseAtCurrentAudioTime();
    } else if (playbackModeRef.current === 'playing_preview') {
      void handlePause();
    } else {
      void handlePlayFromHere();
    }
  }, [handlePause, handlePlayFromHere, pauseAtCurrentAudioTime]);

  const handlePlaySelectedMarker = useCallback(async () => {
    if (!selectedMarker || !decodedRef.current) return;

    try {
      await stopCurrentPlayback('idle');
      const previewStartMs = Math.max(0, selectedMarker.timestamp_ms - REVIEW_PRE_MS);
      if (hasVideo) {
        videoRef.current?.seek(mapAudioPlayheadToVideoMs(event, previewStartMs, durationMs, videoSyncOffsetMs) / 1000);
      }
      const previewPath = await writePreviewClip(
        decodedRef.current.samples,
        decodedRef.current.sampleRate,
        selectedMarker.timestamp_ms,
        playbackRate,
      );
      tempPlaybackPathRef.current = previewPath;
      playbackSourceStartMsRef.current = previewStartMs;
      playbackModeRef.current = 'playing_preview';
      setPlaybackMode('playing_preview');
      setPlayheadMs(previewStartMs);
      setTimelineInteractionMode('playing');
      await playerRef.current.startPlayer(previewPath);
    } catch (error) {
      await stopCurrentPlayback('idle');
      Alert.alert('Förhandsvisningsfel', `Kunde inte spela marker-förhandsvisningen: ${String(error)}`);
    }
  }, [durationMs, event, hasVideo, playbackRate, selectedMarker, setPlayheadMs, setTimelineInteractionMode, stopCurrentPlayback, videoSyncOffsetMs]);

  const handleSetPlaybackRate = useCallback((nextRate: PlaybackRate) => {
    if (playbackModeRef.current === 'playing_full_take' || playbackModeRef.current === 'playing_preview') {
      void handlePause();
    }
    setPlaybackRate(nextRate);
    playbackRateRef.current = nextRate;
  }, [handlePause]);

  const updateSelectedMarker = useCallback((updater: (marker: AudioReviewMarker) => AudioReviewMarker) => {
    if (!selectedMarkerId) return;
    setMarkers(prev => sortMarkers(prev.map(marker => (
      marker.id === selectedMarkerId ? updater(marker) : marker
    ))));
  }, [selectedMarkerId]);

  const handleAddMarkerHere = useCallback(() => {
    const marker = createReviewMarkerAtTimestamp(playbackPositionMs);
    setMarkers(prev => sortMarkers([...prev, marker]));
    setSelectedMarkerId(marker.id);
    setQuickLabelPrompt(supportsQuickLabels ? { markerId: marker.id, timestampMs: marker.timestamp_ms } : null);
  }, [createReviewMarkerAtTimestamp, playbackPositionMs, supportsQuickLabels]);

  const handleAddActiveStageMarkerHere = useCallback(() => {
    if (!isAudioVideoPoseReview) {
      handleAddMarkerHere();
      return;
    }
    handleCreateMarkerAtTimestamp(
      playbackPositionMs,
      isAudioVideoPoseMotionStage ? 'motion' : 'audio',
    );
  }, [
    handleAddMarkerHere,
    handleCreateMarkerAtTimestamp,
    isAudioVideoPoseMotionStage,
    isAudioVideoPoseReview,
    playbackPositionMs,
  ]);

  const handleQuickLabelChoice = useCallback((choice: ReviewLabelChoice) => {
    const prompt = quickLabelPrompt;
    if (!prompt) return;
    setMarkers(prev => sortMarkers(prev.map(marker => (
      marker.id === prompt.markerId
        ? isAudioVideoPoseReview
          ? applyAudioPoseAudioChoice(marker, choice)
          : applyReviewLabelChoice(marker, choice)
        : marker
    ))));
    setSelectedMarkerId(prompt.markerId);
    setPlayheadMs(prompt.timestampMs);
    setQuickLabelPrompt(null);
  }, [isAudioVideoPoseReview, quickLabelPrompt, setPlayheadMs]);

  const approvableAutoMarkers = useMemo(() => activeStageMarkers.filter(marker => {
    const status = marker.review_status ?? 'pending';
    if (marker.source !== 'auto' || status !== 'pending') return false;
    if (isAudioVideoPoseMotionStage) return isMotionLayerMarker(marker);
    if (isAudioVideoPoseAudioStage) return isAudioLayerMarker(marker) && marker.final_label !== 'ignore';
    return !isPlayingReview &&
      marker.final_label === marker.suggested_label &&
      marker.final_label !== 'ignore';
  }), [activeStageMarkers, isAudioVideoPoseAudioStage, isAudioVideoPoseMotionStage, isPlayingReview]);

  const handleApproveAllMarkers = useCallback(() => {
    if (approvableAutoMarkers.length === 0) return;

    Alert.alert(
      'Godkänn alla slag?',
      `Bekräfta ${approvableAutoMarkers.length} auto-förslag i den här tagningen.`,
      [
        { text: 'Avbryt', style: 'cancel' },
        {
          text: 'Godkänn alla',
          onPress: () => {
            const approvableIds = new Set(approvableAutoMarkers.map(marker => marker.id));
            setMarkers(prev => sortMarkers(prev.map(marker => (
              approvableIds.has(marker.id)
                ? {
                    ...marker,
                    review_status: isMotionLayerMarker(marker) && !hasConcreteMotionLabel(marker.motion_label)
                      ? 'ignored'
                      : 'confirmed',
                  }
                : marker
            ))));
          },
        },
      ],
    );
  }, [approvableAutoMarkers]);

  const handleDeleteSelectedMarker = useCallback(() => {
    if (!selectedMarker) return;
    const nextMarkers = activeStageMarkers.filter(marker => marker.id !== selectedMarker.id);
    setMarkers(prev => sortMarkers(prev.filter(marker => marker.id !== selectedMarker.id)));
    setQuickLabelPrompt(null);
    const nextSelected = nextMarkers[Math.min(selectedStageMarkerIndex, nextMarkers.length - 1)] ?? null;
    setSelectedMarkerId(nextSelected?.id ?? null);
    if (nextSelected) {
      setPlayheadMs(nextSelected.timestamp_ms);
    }
  }, [activeStageMarkers, selectedMarker, selectedStageMarkerIndex, setPlayheadMs]);

  const handleNudgeMarker = useCallback((deltaMs: number) => {
    if (!selectedMarker) return;
    const nextTimestampMs = clampTimestamp(selectedMarker.timestamp_ms + deltaMs, durationMs);
    updateSelectedMarker(marker => ({
      ...marker,
      timestamp_ms: nextTimestampMs,
      review_status: isMotionLayerMarker(marker)
        ? 'edited'
        : marker.final_label === 'ignore'
          ? 'ignored'
          : 'edited',
    }));
    setQuickLabelPrompt(prev => (
      prev?.markerId === selectedMarker.id
        ? { ...prev, timestampMs: nextTimestampMs }
        : prev
    ));
    setPlayheadMs(nextTimestampMs);
  }, [durationMs, selectedMarker, setPlayheadMs, updateSelectedMarker]);

  const handleSnapSelectedMarker = useCallback(() => {
    if (!selectedMarker || !decodedRef.current) return;
    const snappedTimestampMs = snapMarkerToAttack(
      decodedRef.current.samples,
      decodedRef.current.sampleRate,
      selectedMarker.timestamp_ms,
    );
    updateSelectedMarker(marker => ({
      ...marker,
      timestamp_ms: snappedTimestampMs,
      review_status: isMotionLayerMarker(marker)
        ? 'edited'
        : marker.final_label === 'ignore'
          ? 'ignored'
          : 'edited',
    }));
    setPlayheadMs(snappedTimestampMs);
  }, [selectedMarker, setPlayheadMs, updateSelectedMarker]);

  const buildCurrentVideoSyncMetadata = useCallback((): AudioVideoSyncMetadata | undefined => {
    if (!hasVideo) return undefined;
    return {
      video_sync_offset_ms: videoSyncOffsetMs,
      video_sync_anchor_audio_ms: audioSyncPoint?.timestamp_ms,
      video_sync_anchor_video_ms: syncCandidateVideoMs ?? undefined,
      video_sync_source: audioSyncPoint ? videoSyncSource : undefined,
    };
  }, [audioSyncPoint, hasVideo, syncCandidateVideoMs, videoSyncOffsetMs, videoSyncSource]);

  const handleReturnToAudioReview = useCallback(() => {
    setAudioVideoReviewStage('audio');
    setPoseAnalysisStatus('Ändra ljudreview och tryck Klar med ljud för att köra om rörelseförslag.');
    const firstAudioMarker = audioTimelineMarkers[0];
    setSelectedMarkerId(firstAudioMarker?.id ?? null);
  }, [audioTimelineMarkers]);

  const handleSkipAudioReview = useCallback(() => {
    if (!isAudioVideoPoseReview || !videoFilePath) return;
    Alert.alert(
      'Hoppa över ljud?',
      'Då går review direkt till rörelseanalys på hela videon. Använd detta när videon ska märkas utan ljudvåg.',
      [
        { text: 'Avbryt', style: 'cancel' },
        {
          text: 'Gå till video',
          onPress: () => {
            setSaving(true);
            const motionMarkersToKeep = allOrderedMarkers.filter(isMotionLayerMarker);
            setMarkers(motionMarkersToKeep);
            setSelectedMarkerId(motionMarkersToKeep[0]?.id ?? null);
            Promise.resolve(onSave(
              [],
              buildCurrentVideoSyncMetadata(),
              modelCandidates,
              detectionConfigSnapshot,
              videoPoseCandidates,
              { completion: 'audio' },
            ))
              .then(() => {
                setAudioVideoReviewStage('motion');
                setPoseAnalysisStatus('Ljud hoppades över. Kör rörelseanalys på hela videon...');
              })
              .catch((error: unknown) => {
                Alert.alert('Kunde inte hoppa över ljud', String(error));
              })
              .finally(() => setSaving(false));
          },
        },
      ],
    );
  }, [
    allOrderedMarkers,
    buildCurrentVideoSyncMetadata,
    detectionConfigSnapshot,
    isAudioVideoPoseReview,
    modelCandidates,
    onSave,
    videoPoseCandidates,
    videoFilePath,
  ]);

  const handleCompleteAudioReview = useCallback(async () => {
    if (!isAudioVideoPoseReview) return;
    if (!videoFilePath) {
      Alert.alert('Video saknas', 'Rörelsereview kräver en sparad video.');
      return;
    }
    const pendingAudioMarkers = audioTimelineMarkers.filter(marker => (marker.review_status ?? 'pending') === 'pending');
    if (pendingAudioMarkers.length > 0) {
      Alert.alert(
        'Ljudreview ofullständig',
        `Hantera ${pendingAudioMarkers.length} ljudmarkrar innan du går vidare.`,
      );
      return;
    }
    if (audioTimelineMarkers.length === 0) {
      Alert.alert('Inga ljudmarkers', 'Lägg till minst en racketträff, bordsstuds eller ignorera-marker först.');
      return;
    }

    setSaving(true);
    try {
      const audioMarkersToSave = allOrderedMarkers
        .filter(isAudioLayerMarker)
        .map(marker => normalizeAudioVideoPoseMarkerForSave({
          ...marker,
          timestamp_ms: clampTimestamp(marker.timestamp_ms, durationMs),
        }));
      const motionMarkersToKeep = allOrderedMarkers
        .filter(isMotionLayerMarker)
        .map(marker => normalizeAudioVideoPoseMarkerForSave({
          ...marker,
          timestamp_ms: clampTimestamp(marker.timestamp_ms, durationMs),
        }));
      setMarkers(sortMarkers([...audioMarkersToSave, ...motionMarkersToKeep]));
      await onSave(
        audioMarkersToSave,
        buildCurrentVideoSyncMetadata(),
        modelCandidates,
        detectionConfigSnapshot,
        videoPoseCandidates,
        { completion: 'audio' },
      );
      setAudioVideoReviewStage('motion');
      setSelectedMarkerId(motionMarkersToKeep[0]?.id ?? null);
      setPoseAnalysisStatus(videoPoseCandidates.length > 0
        ? poseCandidateReviewStatusText(videoPoseCandidates, false)
        : 'Kör rörelseanalys på hela videon...');
    } finally {
      setSaving(false);
    }
  }, [
    allOrderedMarkers,
    audioTimelineMarkers,
    buildCurrentVideoSyncMetadata,
    detectionConfigSnapshot,
    durationMs,
    isAudioVideoPoseReview,
    modelCandidates,
    onSave,
    videoPoseCandidates,
    videoFilePath,
  ]);

  const handleSave = useCallback(async () => {
    if (isAudioVideoPoseAudioStage) {
      await handleCompleteAudioReview();
      return;
    }

    if (isAudioVideoPoseMotionStage && poseAnalysisRunning) {
      Alert.alert('Rörelseanalys körs', 'Vänta tills videon är färdiganalyserad innan du sparar rörelsereviewen.');
      return;
    }

    if (!isAudioVideoPoseMotionStage && orderedMarkers.length === 0 && !allowsMarkerlessNoBounceSave) {
      Alert.alert('Inga markers', 'Lägg till minst en marker eller kassera tagningen.');
      return;
    }

    const pendingMarkers = orderedMarkers.filter(marker => (marker.review_status ?? 'pending') === 'pending');
    if (pendingMarkers.length > 0) {
      Alert.alert(
        'Granskning ofullständig',
        `Hantera ${pendingMarkers.length} auto-markrar innan du sparar. Bekräfta, ändra, ignorera eller ta bort varje marker.`,
      );
      return;
    }

    setSaving(true);
    try {
      const baseMarkersToSave = allOrderedMarkers
        .filter(marker => !(
          isPlayingReview &&
          !isAudioVideoPoseReview &&
          shouldDropPlayingAutoMarkerOnSave(marker, activePlayingConfidenceFilter.minConfidence)
        ))
        .map(marker => ({
          ...marker,
          timestamp_ms: clampTimestamp(marker.timestamp_ms, durationMs),
        }));
      const markersToSave = isAudioVideoPoseReview
        ? splitAudioVideoPoseMarkers(baseMarkersToSave, durationMs).map(normalizeAudioVideoPoseMarkerForSave)
        : baseMarkersToSave;
      const videoSyncMetadata = buildCurrentVideoSyncMetadata();
      await onSave(markersToSave.map(marker => ({
        ...marker,
        timestamp_ms: clampTimestamp(marker.timestamp_ms, durationMs),
      })), videoSyncMetadata, modelCandidates, detectionConfigSnapshot, videoPoseCandidates, { completion: 'complete' });
    } finally {
      setSaving(false);
    }
  }, [
    activePlayingConfidenceFilter.minConfidence,
    allowsMarkerlessNoBounceSave,
    allOrderedMarkers,
    buildCurrentVideoSyncMetadata,
    durationMs,
    handleCompleteAudioReview,
    isAudioVideoPoseAudioStage,
    isAudioVideoPoseMotionStage,
    isAudioVideoPoseReview,
    isPlayingReview,
    modelCandidates,
    onSave,
    orderedMarkers,
    poseAnalysisRunning,
    detectionConfigSnapshot,
    videoPoseCandidates,
  ]);

  const handleDiscard = useCallback(() => {
    Alert.alert(
      'Kassera tagning',
      'Det tar bort tagningen från sessionen och raderar WAV-filen och granskningsvideon.',
      [
        { text: 'Avbryt', style: 'cancel' },
        {
          text: 'Kassera',
          style: 'destructive',
          onPress: () => {
            onDiscard();
          },
        },
      ],
    );
  }, [onDiscard]);

  const markerCounts = useMemo(() => ({
    racket: orderedMarkers.filter(marker => marker.final_label === 'racket_contact').length,
    notRacket: orderedMarkers.filter(marker => marker.final_label === 'not_racket_contact').length,
    ignore: orderedMarkers.filter(marker => marker.final_label === 'ignore').length,
    pending: orderedMarkers.filter(marker => (marker.review_status ?? 'pending') === 'pending').length,
  }), [orderedMarkers]);
  const audioVideoStagePendingCount = activeStageMarkers.filter(marker => (
    (marker.review_status ?? 'pending') === 'pending'
  )).length;
  const activeStageTitle = isAudioVideoPoseMotionStage ? 'Rörelse' : 'Ljud';
  const activeStageMarkerCount = activeStageMarkers.length;
  const playingRetroPrimaryPreservesExistingReview = usesPlayingRetroPrimaryReview &&
    shouldKeepExistingReviewMarkersForPlayingRetro(audioTimelineMarkers);
  const playingRetroPrimaryWaitingForMarkers = usesPlayingRetroPrimaryReview &&
    !playingRetroAnalysis &&
    !playingRetroPrimaryPreservesExistingReview &&
    (playingRetroRunning || modelCandidates.length > 0);
  const reviewTruthSummary = useMemo(() => {
    const liveMarkers = allOrderedMarkers.filter(marker => {
      const status = marker.review_status ?? 'pending';
      return status !== 'deleted' && status !== 'filtered';
    });
    const linkedAutoCandidateIds = new Set(liveMarkers
      .filter(marker => marker.source === 'auto' && marker.linked_candidate_id)
      .map(marker => marker.linked_candidate_id as string));
    const totalCandidates = modelCandidates.length;
    const modelFound = modelCandidates.filter(candidate => candidate.review_relevant).length;
    const added = liveMarkers.filter(marker => marker.source === 'manual').length;
    const changedAuto = liveMarkers.filter(marker => (
      marker.source === 'auto' &&
      (
        marker.final_label !== marker.suggested_label ||
        marker.review_status === 'ignored' ||
        marker.review_status === 'edited'
      )
    )).length;
    const removed = modelCandidates.filter(candidate => (
      candidate.review_relevant && !linkedAutoCandidateIds.has(candidate.id)
    )).length;
    const approved = liveMarkers.filter(marker => (
      (marker.review_status === 'confirmed' || marker.review_status === 'edited') &&
      marker.final_label !== 'ignore'
    )).length;
    const motionApproved = liveMarkers.filter(marker => (
      (marker.review_status === 'confirmed' || marker.review_status === 'edited') &&
      hasConcreteMotionLabel(marker.motion_label)
    )).length;
    return {
      totalCandidates,
      modelFound,
      added,
      removedOrChanged: removed + changedAuto,
      approved,
      motionApproved,
    };
  }, [allOrderedMarkers, modelCandidates]);
  const playingRetroPrimaryReviewSummary = useMemo(() => (
    usesPlayingRetroPrimaryReview && playingRetroAnalysis
      ? summarizePlayingRetroPrimaryReview(playingRetroAnalysis, audioTimelineMarkers.length)
      : null
  ), [audioTimelineMarkers.length, playingRetroAnalysis, usesPlayingRetroPrimaryReview]);
  const algorithmMetaText = useMemo(() => {
    if (usesPlayingRetroPrimaryReview && playingRetroPrimaryWaitingForMarkers) {
      return [
        `Peak-kandidater ${reviewTruthSummary.totalCandidates}`,
        'spel-retro skapar review-markers',
        `Love lade till ${reviewTruthSummary.added}`,
        `ändrade/tog bort ${reviewTruthSummary.removedOrChanged}`,
      ].join(' · ');
    }
    if (usesPlayingRetroPrimaryReview && playingRetroPrimaryReviewSummary) {
      return [
        `Review-markers ${playingRetroPrimaryReviewSummary.markerCount}`,
        `kandidater ${playingRetroPrimaryReviewSummary.rawCandidateCount}`,
        `Love lade till ${reviewTruthSummary.added}`,
        `ändrade/tog bort ${reviewTruthSummary.removedOrChanged}`,
      ].join(' · ');
    }
    return [
      `Kandidater ${reviewTruthSummary.totalCandidates}`,
      `review ${reviewTruthSummary.modelFound}`,
      `Love lade till ${reviewTruthSummary.added}`,
      `ändrade/tog bort ${reviewTruthSummary.removedOrChanged}`,
    ].join(' · ');
  }, [
    playingRetroPrimaryReviewSummary,
    playingRetroPrimaryWaitingForMarkers,
    reviewTruthSummary,
    usesPlayingRetroPrimaryReview,
  ]);
  const algorithmSaveHint = useMemo(() => {
    if (usesPlayingRetroPrimaryReview) {
      if (reviewTruthSummary.approved > 0 || reviewTruthSummary.motionApproved > 0) {
        return `Save tränar på ${reviewTruthSummary.approved} ljudmarkers och ${reviewTruthSummary.motionApproved} rörelsemarkers. Kandidater sparas bara för analys.`;
      }
      return 'Bekräfta eller ändra markers först. Råa kandidater sparas bara för analys.';
    }
    return isAudioVideoPoseReview
      ? `Save tränar på ${reviewTruthSummary.approved} ljudmarkers och ${reviewTruthSummary.motionApproved} rörelsemarkers. Kandidater sparas bara för analys.`
      : `Save tränar på ${reviewTruthSummary.approved} mänskligt godkända/ändrade markers. Kandidater sparas bara för analys.`;
  }, [isAudioVideoPoseReview, reviewTruthSummary, usesPlayingRetroPrimaryReview]);
  const canSave = isAudioVideoPoseReview
    ? !playingRetroPrimaryWaitingForMarkers && audioVideoStagePendingCount === 0 && (
        isAudioVideoPoseMotionStage
          ? !poseAnalysisRunning
          : audioTimelineMarkers.length > 0
      )
    : markerCounts.pending === 0 && (orderedMarkers.length > 0 || allowsMarkerlessNoBounceSave);
  const saveButtonText = saving
    ? 'Sparar...'
    : isAudioVideoPoseAudioStage
      ? playingRetroPrimaryWaitingForMarkers
        ? 'Vänta på spel-retro'
        : audioVideoStagePendingCount > 0
        ? `Hantera ${audioVideoStagePendingCount} ljudmarkrar`
        : audioTimelineMarkers.length > 0
          ? 'Klar med ljud'
          : 'Lägg till ljudmarker'
    : isAudioVideoPoseMotionStage
      ? poseAnalysisRunning
        ? 'Analyserar rörelse...'
        : audioVideoStagePendingCount > 0
        ? `Hantera ${audioVideoStagePendingCount} rörelsemarkrar`
        : 'Klar med rörelse'
    : canSave
      ? allowsMarkerlessNoBounceSave && orderedMarkers.length === 0
        ? 'Spara utan studs'
        : 'Spara tagning'
      : markerCounts.pending > 0
        ? `Hantera ${markerCounts.pending} markrar`
        : 'Lägg till marker';

  const timelinePlayheadLeft = ratioToLeft(
    playbackPositionMs,
    safeTimelineWindowStartMs,
    timelineWindowEndMs,
    overviewWidth,
  );

  const videoFrameWidth = Math.max(260, windowWidth - 32);
  const videoAspectRatio = useMemo(() => {
    if (!videoNaturalSize?.width || !videoNaturalSize?.height) return 16 / 9;
    return Math.max(0.35, Math.min(2.4, videoNaturalSize.width / videoNaturalSize.height));
  }, [videoNaturalSize]);
  const videoHeight = Math.max(
    205,
    Math.min(
      Math.round(windowHeight * 0.58),
      Math.round(videoFrameWidth / videoAspectRatio),
    ),
  );
  const previewImuSamples = event.imu_recording?.samples ?? [];
  const playbackActive = playbackMode === 'playing_full_take' || playbackMode === 'playing_preview';
  const syncPointVisible = Boolean(audioSyncPoint && isTimestampVisible(
    audioSyncPoint.timestamp_ms,
    safeTimelineWindowStartMs,
    timelineWindowEndMs,
  ));
  const syncPointLeft = audioSyncPoint
    ? ratioToLeft(audioSyncPoint.timestamp_ms, safeTimelineWindowStartMs, timelineWindowEndMs, overviewWidth)
    : 0;
  const videoSyncOffsetLabel = `${videoSyncOffsetMs > 0 ? '+' : ''}${videoSyncOffsetMs} ms`;
  const videoSyncStatusTitle = videoSyncOffsetMs !== 0 && !videoSyncExpanded
    ? 'Video synkad'
    : 'Synka video';
  const syncCandidateLabel = syncCandidateVideoMs !== null
    ? formatTimelineShort(syncCandidateVideoMs)
    : 'ej vald';
  const syncSourceLabel = videoSyncSource === 'manual' ? 'manuell ljudpunkt' : 'auto-förslag';
  const reviewTimelineLayer: ReviewTimelineLayer = isAudioVideoPoseMotionStage ? 'motion' : 'audio';
  const reviewTimelineMarkers = isAudioVideoPoseMotionStage
    ? motionTimelineMarkers
    : isAudioVideoPoseReview
      ? audioTimelineMarkers
      : orderedMarkers;
  const reviewTimelinePlayheadResponder = isAudioVideoPoseMotionStage
    ? motionPlayheadResponder
    : overviewPlayheadResponder;
  const showPlayingRetroPreparation = isAudioVideoPoseAudioStage && playingRetroPrimaryWaitingForMarkers;
  const showFullScreenPreparation = loading || showPlayingRetroPreparation;
  const loadingTitle = showPlayingRetroPreparation
    ? 'Analyserar spel-retro audio'
    : 'Förbereder review';
  const loadingSubtitle = showPlayingRetroPreparation
    ? 'Appen hittar ljudpeakar, klassar dem med retro-modellen och skapar editable racket/bord-markers.'
    : 'Laddar ljudvåg, video och kandidater.';

  return (
    <View style={styles.root}>
      <StatusBar hidden barStyle="light-content" backgroundColor="#0d0d0d" />
      {showFullScreenPreparation && hasVideo && videoFilePath && (
        <View style={styles.preloadVideoMount} pointerEvents="none">
          <Video
            source={{ uri: `file://${videoFilePath}` }}
            style={styles.preloadVideo}
            paused
            muted
            controls={false}
            resizeMode="contain"
            progressUpdateInterval={500}
            onLoadStart={() => {
              setVideoReady(false);
              setVideoBuffering(true);
            }}
            onLoad={() => {
              setVideoBuffering(false);
            }}
            onReadyForDisplay={() => {
              setVideoReady(true);
            }}
          />
        </View>
      )}
      {showFullScreenPreparation ? (
        <View style={styles.loadingBox}>
          <TouchableOpacity onPress={onBack} style={styles.loadingBackBtn}>
            <Text style={styles.loadingBackTxt}>Tillbaka</Text>
          </TouchableOpacity>
          <ActivityIndicator color="#f5c76d" />
          <Text style={styles.loadingTitle}>{loadingTitle}</Text>
          <Text style={styles.loadingTxt}>{loadingSubtitle}</Text>
          {showPlayingRetroPreparation && (
            <View style={styles.retroPrepCard}>
              <Text style={styles.retroPrepLine}>1. Råa peak-kandidater: {reviewTruthSummary.totalCandidates}</Text>
              <Text style={styles.retroPrepLine}>2. Multi-window/context ML klassar racket, bord och ej target.</Text>
              <Text style={styles.retroPrepLine}>3. Bara racket + bord blir markers i ljudvågen.</Text>
              <Text style={styles.retroPrepMeta}>{playingRetroStatus ?? 'Skapar review-listan...'}</Text>
              {reviewStartProfileText && (
                <Text style={styles.retroPrepTiming}>Timing: {reviewStartProfileText}</Text>
              )}
            </View>
          )}
        </View>
      ) : (
        <ScrollView
          style={styles.reviewScroll}
          contentContainerStyle={[
            styles.reviewContent,
            {
              paddingTop: Math.max(insets.top, 16),
              paddingBottom: Math.max(insets.bottom + 96, 120),
            },
          ]}
          showsVerticalScrollIndicator={false}
        >
          <View style={styles.reviewHeader}>
            <TouchableOpacity onPress={onBack} style={styles.reviewBackBtn}>
              <Text style={styles.reviewBackTxt}>Tillbaka</Text>
            </TouchableOpacity>
            <View style={styles.reviewHeaderCopy}>
              <Text style={styles.reviewTitle}>Granska tagning</Text>
              <Text style={styles.reviewSubtitle}>Tagning {event.take_index} | {formatTimelineShort(durationMs)}</Text>
            </View>
            <TouchableOpacity
              style={styles.menuBtn}
              onPress={() => Alert.alert(REVIEW_UI_REVISION, 'Placera markern på första tydliga träffen. Video är bara stöd för granskning.')}
            >
              <Text style={styles.menuTxt}>...</Text>
            </TouchableOpacity>
          </View>

          {hasVideo && (
            <View style={styles.videoSyncPanel}>
              <View style={styles.videoSyncHeader}>
                <View style={styles.videoSyncHeaderCopy}>
                  <Text style={styles.videoSyncTitle}>{videoSyncStatusTitle}</Text>
                  <Text style={styles.videoSyncMeta}>
                    {audioSyncPoint
                      ? `Sync-punkt ${formatTimelineShort(audioSyncPoint!.timestamp_ms)} | video ${syncCandidateLabel} | ${syncSourceLabel}`
                      : 'Ingen tydlig sync-spik hittad i början.'}
                  </Text>
                </View>
                <View style={styles.videoSyncHeaderActions}>
                  <Text style={styles.videoSyncOffsetPill}>{videoSyncOffsetLabel}</Text>
                  <TouchableOpacity
                    style={styles.videoSyncToggleBtn}
                    onPress={() => setVideoSyncExpanded(prev => !prev)}
                  >
                    <Text style={styles.videoSyncToggleTxt}>
                      {videoSyncExpanded ? 'Dölj' : 'Justera synk'}
                    </Text>
                  </TouchableOpacity>
                </View>
              </View>
              {videoSyncExpanded && (
                <>
                  <Text style={styles.videoSyncHelp}>
                    Dra playhead till rätt klapp i ljudet, tryck Sätt ljudsync här, stega videon tills klappen syns och tryck Synka här.
                  </Text>
                  <View style={styles.videoSyncControls}>
                    <TouchableOpacity style={styles.videoSyncApplyBtn} onPress={setAudioSyncHere}>
                      <Text style={styles.videoSyncApplyTxt}>Sätt ljudsync här</Text>
                    </TouchableOpacity>
                    <TouchableOpacity
                      style={[styles.videoSyncBtn, !audioSyncPoint && styles.disabledBtn]}
                      onPress={startSyncCalibration}
                      disabled={!audioSyncPoint}
                    >
                      <Text style={styles.videoSyncBtnTxt}>Gå till sync</Text>
                    </TouchableOpacity>
                    <TouchableOpacity
                      style={[styles.videoSyncBtn, !audioSyncPoint && styles.disabledBtn]}
                      onPress={() => adjustSyncCandidateVideo(-VIDEO_FRAME_STEP_MS)}
                      disabled={!audioSyncPoint}
                    >
                      <Text style={styles.videoSyncBtnTxt}>-1 frame</Text>
                    </TouchableOpacity>
                    <TouchableOpacity
                      style={[styles.videoSyncBtn, !audioSyncPoint && styles.disabledBtn]}
                      onPress={() => adjustSyncCandidateVideo(VIDEO_FRAME_STEP_MS)}
                      disabled={!audioSyncPoint}
                    >
                      <Text style={styles.videoSyncBtnTxt}>+1 frame</Text>
                    </TouchableOpacity>
                    <TouchableOpacity style={styles.videoSyncBtn} onPress={() => adjustSyncCandidateVideo(-100)}>
                      <Text style={styles.videoSyncBtnTxt}>Video tidigare</Text>
                    </TouchableOpacity>
                    <TouchableOpacity style={styles.videoSyncBtn} onPress={() => adjustSyncCandidateVideo(100)}>
                      <Text style={styles.videoSyncBtnTxt}>Video senare</Text>
                    </TouchableOpacity>
                    <TouchableOpacity
                      style={[styles.videoSyncApplyBtn, (!audioSyncPoint || syncCandidateVideoMs === null) && styles.disabledBtn]}
                      onPress={applySyncCandidateHere}
                      disabled={!audioSyncPoint || syncCandidateVideoMs === null}
                    >
                      <Text style={styles.videoSyncApplyTxt}>Synka här</Text>
                    </TouchableOpacity>
                    <TouchableOpacity style={styles.videoSyncResetBtn} onPress={resetGuidedVideoSync}>
                      <Text style={styles.videoSyncResetTxt}>Återställ</Text>
                    </TouchableOpacity>
                  </View>
                  {syncCalibrationActive && (
                    <Text style={styles.videoSyncActiveTxt}>Kalibrering aktiv: video stegas utan att flytta audio-markern.</Text>
                  )}
                </>
              )}
            </View>
          )}

          <View style={[styles.heroVideoFrame, { height: videoHeight }]}>
            {hasVideo && videoFilePath ? (
              <Video
                ref={videoRef}
                source={{ uri: `file://${videoFilePath}` }}
                style={styles.videoPlayer}
                paused={!playbackActive}
                rate={playbackRate}
                muted
                repeat={false}
                controls={false}
                resizeMode="contain"
                progressUpdateInterval={50}
                bufferConfig={{
                  minBufferMs: 250,
                  maxBufferMs: 1500,
                  bufferForPlaybackMs: 250,
                  bufferForPlaybackAfterRebufferMs: 500,
                }}
                onLoadStart={() => {
                  setVideoReady(false);
                  setVideoBuffering(true);
                }}
                onLoad={(data: any) => {
                  setVideoBuffering(false);
                  const naturalSize = data?.naturalSize;
                  const width = Number(naturalSize?.width ?? 0);
                  const height = Number(naturalSize?.height ?? 0);
                  const orientation = naturalSize?.orientation === 'portrait' ? 'portrait' : 'landscape';
                  if (width > 0 && height > 0) {
                    const normalizedWidth = orientation === 'portrait' && width > height ? height : width;
                    const normalizedHeight = orientation === 'portrait' && width > height ? width : height;
                    setVideoNaturalSize({
                      width: normalizedWidth,
                      height: normalizedHeight,
                      orientation,
                    });
                  }
                }}
                onReadyForDisplay={() => {
                  setVideoReady(true);
                  setVideoPreparingPlayback(false);
                }}
                onBuffer={({ isBuffering }) => {
                  setVideoBuffering(Boolean(isBuffering));
                }}
                onProgress={(progress: any) => {
                  latestVideoMsRef.current = Math.max(0, Math.round(Number(progress?.currentTime ?? 0) * 1000));
                  latestVideoProgressWallClockMsRef.current = Date.now();
                }}
                onError={error => {
                  setVideoReady(false);
                  setVideoBuffering(false);
                  Alert.alert('Videofel', `Kunde inte läsa granskningsvideon: ${String(error)}`);
                }}
              />
            ) : (
              <View style={styles.noVideoSurface}>
                <Text style={styles.noVideoTitle}>Endast ljud</Text>
                <Text style={styles.noVideoSub}>Video saknas, men audio-review fungerar.</Text>
              </View>
            )}
            {hasVideo && (
              <Pressable
                style={styles.videoTapLayer}
                onPressIn={handleToggleVideoPlayback}
              />
            )}
            {hasVideo && (videoPreparingPlayback || (!videoReady && videoBuffering)) && (
              <View style={styles.videoPreparingBadge}>
                <ActivityIndicator color="#fff" size="small" />
                <Text style={styles.videoPreparingTxt}>Förbereder video...</Text>
              </View>
            )}
            <View style={styles.rateChip}>
              <TouchableOpacity
                onPress={() => {
                  const nextRate = playbackRate === 1 ? 0.5 : playbackRate === 0.5 ? 0.25 : 1;
                  handleSetPlaybackRate(nextRate);
                }}
              >
                <Text style={styles.rateChipTxt}>{playbackRate.toFixed(1)}x</Text>
              </TouchableOpacity>
            </View>
            <View style={styles.videoControls}>
              <TouchableOpacity style={styles.videoPlayBtn} onPress={handleToggleVideoPlayback}>
                <Text style={styles.videoPlayTxt}>{playbackActive ? 'Paus' : 'Spela'}</Text>
              </TouchableOpacity>
              <Text style={styles.videoTimeTxt}>{formatTimelineShort(playbackPositionMs)}</Text>
              <View
                ref={videoProgressTrackRef}
                style={styles.videoProgressTrack}
                onLayout={eventData => {
                  const width = eventData.nativeEvent.layout.width;
                  setVideoProgressWidth(width);
                  videoProgressLayoutRef.current = { ...videoProgressLayoutRef.current, width };
                  requestAnimationFrame(() => measureVideoProgressTrack());
                }}
                {...videoProgressScrubResponder.panHandlers}
              >
                <View
                  style={[
                    styles.videoProgressFill,
                    { width: `${durationMs > 0 ? Math.min(100, (playbackPositionMs / durationMs) * 100) : 0}%` },
                  ]}
                />
                <View
                  style={[
                    styles.videoProgressKnob,
                    { left: `${durationMs > 0 ? Math.min(100, (playbackPositionMs / durationMs) * 100) : 0}%` },
                  ]}
                />
              </View>
              <Text style={styles.videoTimeTxt}>{formatTimelineShort(durationMs)}</Text>
            </View>
          </View>

          {false && hasVideo && audioSyncPoint && (
            <View style={styles.videoSyncPanel}>
              <View style={styles.videoSyncHeader}>
                <View style={styles.videoSyncHeaderCopy}>
                  <Text style={styles.videoSyncTitle}>{videoSyncStatusTitle}</Text>
                <Text style={styles.videoSyncMeta}>
                    {audioSyncPoint
                      ? `Sync-punkt ${formatTimelineShort(audioSyncPoint!.timestamp_ms)} | video ${syncCandidateLabel}`
                      : 'Ingen tydlig sync-spik hittad i början.'}
                </Text>
                </View>
                <View style={styles.videoSyncHeaderActions}>
                  <Text style={styles.videoSyncOffsetPill}>{videoSyncOffsetLabel}</Text>
                  <TouchableOpacity
                    style={styles.videoSyncToggleBtn}
                    onPress={() => setVideoSyncExpanded(prev => !prev)}
                  >
                    <Text style={styles.videoSyncToggleTxt}>
                      {videoSyncExpanded ? 'Dölj' : 'Justera synk'}
                    </Text>
                  </TouchableOpacity>
                </View>
              </View>
              {videoSyncExpanded && (
                <>
              <Text style={styles.videoSyncHelp}>
                Klappa eller tappa i början. Gå till sync-punkten, stega videon tills klappen syns och tryck Synka här.
              </Text>
              <View style={styles.videoSyncControls}>
                <TouchableOpacity
                  style={[styles.videoSyncBtn, !audioSyncPoint && styles.disabledBtn]}
                  onPress={startSyncCalibration}
                  disabled={!audioSyncPoint}
                >
                  <Text style={styles.videoSyncBtnTxt}>Gå till sync</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.videoSyncBtn, !audioSyncPoint && styles.disabledBtn]}
                  onPress={() => adjustSyncCandidateVideo(-VIDEO_FRAME_STEP_MS)}
                  disabled={!audioSyncPoint}
                >
                  <Text style={styles.videoSyncBtnTxt}>-1 frame</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.videoSyncBtn, !audioSyncPoint && styles.disabledBtn]}
                  onPress={() => adjustSyncCandidateVideo(VIDEO_FRAME_STEP_MS)}
                  disabled={!audioSyncPoint}
                >
                  <Text style={styles.videoSyncBtnTxt}>+1 frame</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.videoSyncBtn} onPress={() => adjustSyncCandidateVideo(-100)}>
                  <Text style={styles.videoSyncBtnTxt}>Video tidigare</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.videoSyncBtn} onPress={() => adjustSyncCandidateVideo(100)}>
                  <Text style={styles.videoSyncBtnTxt}>Video senare</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.videoSyncApplyBtn, (!audioSyncPoint || syncCandidateVideoMs === null) && styles.disabledBtn]}
                  onPress={applySyncCandidateHere}
                  disabled={!audioSyncPoint || syncCandidateVideoMs === null}
                >
                  <Text style={styles.videoSyncApplyTxt}>Synka här</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.videoSyncResetBtn} onPress={resetGuidedVideoSync}>
                  <Text style={styles.videoSyncResetTxt}>Återställ</Text>
                </TouchableOpacity>
              </View>
              {syncCalibrationActive && (
                <Text style={styles.videoSyncActiveTxt}>Kalibrering aktiv: video stegas utan att flytta audio-markern.</Text>
              )}
                </>
              )}
            </View>
          )}

          <View style={[styles.markerPanel, styles.hidden]}>
            <View style={styles.markerTopRow}>
              <View>
                <Text style={styles.markerPanelTitle}>Marker {selectedMarkerIndex >= 0 ? selectedMarkerIndex + 1 : 0} av {orderedMarkers.length}</Text>
                <Text style={styles.markerPanelSub}>
                  {formatTimelineShort(selectedMarker?.timestamp_ms ?? playbackPositionMs)} · {markerDetailText(selectedMarker)}
                </Text>
                {selectedMarker?.source === 'auto' && (
                  <Text style={styles.markerConfidenceTxt}>
                    Säkerhet {Math.round(markerConfidence(selectedMarker) * 100)}%
                  </Text>
                )}
              </View>
              <View style={styles.markerControlsRow}>
                <TouchableOpacity
                  style={[styles.roundControlBtn, selectedMarkerIndex <= 0 && styles.disabledBtn]}
                  onPress={() => handleJumpToMarker(-1)}
                  disabled={selectedMarkerIndex <= 0}
                >
                  <Text style={styles.roundControlTxt}>Föreg.</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.centerPlayBtn} onPress={handlePlaySelectedMarker}>
                  <Text style={styles.centerPlayTxt}>Spela</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.roundControlBtn, selectedMarkerIndex >= orderedMarkers.length - 1 && styles.disabledBtn]}
                  onPress={() => handleJumpToMarker(1)}
                  disabled={selectedMarkerIndex >= orderedMarkers.length - 1}
                >
                  <Text style={styles.roundControlTxt}>Nästa</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.addMarkerBtn} onPress={handleAddMarkerHere}>
                  <Text style={styles.addMarkerTxt}>+ Lägg till</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[
                    styles.approveAllBtn,
                    isPlayingReview && styles.hidden,
                    approvableAutoMarkers.length === 0 && styles.disabledBtn,
                  ]}
                  onPress={handleApproveAllMarkers}
                  disabled={isPlayingReview || approvableAutoMarkers.length === 0}
                >
                  <Text style={styles.approveAllTxt}>Godkänn alla</Text>
                </TouchableOpacity>
              </View>
            </View>

            {isPlayingReview && (
              <View style={styles.playingFilterPanel}>
                <Text style={styles.playingFilterMeta}>
                  Visar {playingAutoMarkerStats.visible} av {playingAutoMarkerStats.total} auto-detekterade
                </Text>
                <View style={styles.playingFilterRow}>
                  {PLAYING_CONFIDENCE_FILTERS.map(filter => (
                    <TouchableOpacity
                      key={filter.id}
                      style={[
                        styles.playingFilterBtn,
                        playingConfidenceFilter === filter.id && styles.playingFilterBtnActive,
                      ]}
                      onPress={() => setPlayingConfidenceFilter(filter.id)}
                    >
                      <Text
                        style={[
                          styles.playingFilterTxt,
                          playingConfidenceFilter === filter.id && styles.playingFilterTxtActive,
                        ]}
                      >
                        {filter.title}
                      </Text>
                    </TouchableOpacity>
                  ))}
                </View>
              </View>
            )}

            <View style={styles.reviewLayerSummary}>
              <Text style={styles.reviewLayerTitle}>{detectionConfigTitle(detectionConfigSnapshot)}</Text>
              <Text style={styles.reviewLayerText}>
                Kandidater {reviewTruthSummary.totalCandidates} · visas {reviewTruthSummary.modelFound} · tillagda {reviewTruthSummary.added} · ändrade/borttagna {reviewTruthSummary.removedOrChanged}
              </Text>
            </View>

            {isAudioVideoPoseReview ? (
              <View style={styles.audioPoseLayerPanel}>
                <Text style={styles.audioPoseLayerTitle}>Ljud</Text>
                <View style={styles.labelSegment}>
                  {AUDIO_VIDEO_POSE_AUDIO_LABEL_CHOICES.map(choice => {
                    const active = markerMatchesAudioPoseAudioChoice(selectedMarker, choice);
                    return (
                      <TouchableOpacity
                        key={choice.id}
                        style={[
                          styles.segmentBtn,
                          active && { backgroundColor: choice.color, borderColor: choice.color },
                        ]}
                        disabled={!selectedMarker}
                        onPress={() => {
                          setQuickLabelPrompt(null);
                          updateSelectedMarker(marker => applyAudioPoseAudioChoice(marker, choice));
                        }}
                      >
                        <Text style={[styles.segmentTxt, active && styles.segmentTxtActive]}>{choice.title}</Text>
                      </TouchableOpacity>
                    );
                  })}
                </View>
                <Text style={styles.audioPoseLayerTitle}>Rörelse</Text>
                <View style={styles.labelSegment}>
                  {AUDIO_VIDEO_POSE_MOTION_LABEL_CHOICES.map(choice => {
                    const active = markerMatchesAudioPoseMotionChoice(selectedMarker, choice);
                    return (
                      <TouchableOpacity
                        key={choice.id}
                        style={[
                          styles.segmentBtn,
                          active && { backgroundColor: choice.color, borderColor: choice.color },
                        ]}
                        disabled={!selectedMarker}
                        onPress={() => {
                          setQuickLabelPrompt(null);
                          updateSelectedMarker(marker => applyAudioPoseMotionChoice(marker, choice));
                        }}
                      >
                        <Text style={[styles.segmentTxt, active && styles.segmentTxtActive]}>{choice.title}</Text>
                      </TouchableOpacity>
                    );
                  })}
                </View>
              </View>
            ) : (
              <View style={[styles.labelSegment, isPlayingReview && styles.playingLabelSegment]}>
                {reviewLabelChoices.map(choice => {
                  const active = markerMatchesChoice(selectedMarker, choice);
                  return (
                    <TouchableOpacity
                      key={choice.id}
                      style={[
                        styles.segmentBtn,
                        isPlayingReview && styles.playingSegmentBtn,
                        active && {
                          backgroundColor: choice.color,
                          borderColor: choice.color,
                        },
                      ]}
                      disabled={!selectedMarker}
                      onPress={() => {
                        setQuickLabelPrompt(null);
                        updateSelectedMarker(marker => applyReviewLabelChoice(marker, choice));
                      }}
                    >
                      <Text style={[styles.segmentTxt, active && styles.segmentTxtActive]}>{choice.title}</Text>
                    </TouchableOpacity>
                  );
                })}
              </View>
            )}

            <View style={styles.markerUtilityRow}>
              <TouchableOpacity style={styles.utilityBtn} onPress={handleSnapSelectedMarker} disabled={!selectedMarker}>
                <Text style={styles.utilityTxt}>Snappa</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.utilityBtn} onPress={() => handleNudgeMarker(-EXTRA_LARGE_NUDGE_STEP_MS)} disabled={!selectedMarker}>
                <Text style={styles.utilityTxt}>-50 ms</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.utilityBtn} onPress={() => handleNudgeMarker(-LARGE_NUDGE_STEP_MS)} disabled={!selectedMarker}>
                <Text style={styles.utilityTxt}>-20 ms</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.utilityBtn} onPress={() => handleNudgeMarker(-NUDGE_STEP_MS)} disabled={!selectedMarker}>
                <Text style={styles.utilityTxt}>-10 ms</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.utilityBtn} onPress={() => handleNudgeMarker(NUDGE_STEP_MS)} disabled={!selectedMarker}>
                <Text style={styles.utilityTxt}>+10 ms</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.utilityBtn} onPress={() => handleNudgeMarker(LARGE_NUDGE_STEP_MS)} disabled={!selectedMarker}>
                <Text style={styles.utilityTxt}>+20 ms</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.utilityBtn} onPress={() => handleNudgeMarker(EXTRA_LARGE_NUDGE_STEP_MS)} disabled={!selectedMarker}>
                <Text style={styles.utilityTxt}>+50 ms</Text>
              </TouchableOpacity>
              <TouchableOpacity style={[styles.utilityBtn, styles.utilityDeleteBtn]} onPress={handleDeleteSelectedMarker} disabled={!selectedMarker}>
                <Text style={styles.utilityDeleteTxt}>Ta bort</Text>
              </TouchableOpacity>
            </View>
          </View>

          <View style={styles.waveformCard}>
            <View style={styles.timelineHeaderRow}>
              <Text style={styles.timelineTitle}>
                {isAudioVideoPoseAudioStage
                  ? 'Steg 1: ljudreview'
                  : isAudioVideoPoseMotionStage
                    ? 'Steg 2: rörelsereview'
                    : 'Ljudreview'}
              </Text>
              <View style={styles.zoomControlsRow}>
                {TIMELINE_ZOOM_LEVELS.map(level => (
                  <TouchableOpacity
                    key={`zoom-${level}`}
                    style={[styles.zoomBtn, zoomLevel === level && styles.zoomBtnActive]}
                    onPress={() => handleZoomChange(level)}
                  >
                    <Text style={[styles.zoomBtnTxt, zoomLevel === level && styles.zoomBtnTxtActive]}>{level}x</Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>
            <>
            <View style={styles.timeTicksRow}>
              {[0, 0.25, 0.5, 0.75, 1].map(ratio => (
                <Text key={`tick-${ratio}`} style={styles.tickTxt}>
                  {formatTimelineShort(safeTimelineWindowStartMs + timelineWindowSpanMs * ratio)}
                </Text>
              ))}
            </View>
            <View
              ref={timelineSurfaceRef}
              style={styles.fullWaveformSurface}
              onLayout={eventData => {
                const width = eventData.nativeEvent.layout.width;
                setOverviewWidth(width);
                scrubTimelineLayoutRef.current = { ...scrubTimelineLayoutRef.current, width };
                requestAnimationFrame(() => measureTimelineSurface());
              }}
            >
              <Pressable
                style={StyleSheet.absoluteFill}
                onPress={eventData => handleFullOverviewPress(eventData.nativeEvent.locationX)}
                onLongPress={eventData => {
                  const timestampMs = timestampFromTimelineLocation(eventData.nativeEvent.locationX);
                  if (timestampMs !== null) {
                    handleCreateMarkerAtTimestamp(timestampMs, reviewTimelineLayer);
                  }
                }}
                delayLongPress={PLAYHEAD_LONG_PRESS_MS}
              />
              <View style={styles.fullWaveformRow}>
                {timelineBins.map((bin, index) => {
                  const visualAmplitude = Math.pow(Math.max(0, Math.min(1, bin)), 0.72);
                  return (
                    <View
                      key={`full-bin-${index}`}
                      style={[
                        styles.fullWaveBar,
                        {
                          height: Math.max(2, visualAmplitude * WAVEFORM_MAX_BAR_HEIGHT),
                          opacity: 0.34 + visualAmplitude * 0.66,
                        },
                      ]}
                    />
                  );
                })}
              </View>
              {!isAudioVideoPoseMotionStage && modelCandidates.filter(candidate => (
                !visibleMarkerCandidateIds.has(candidate.id) &&
                shouldShowCandidatePin(candidate) &&
                isTimestampVisible(candidate.timestamp_ms, safeTimelineWindowStartMs, timelineWindowEndMs)
              )).map(candidate => {
                const left = ratioToLeft(
                  candidate.timestamp_ms,
                  safeTimelineWindowStartMs,
                  timelineWindowEndMs,
                  overviewWidth,
                );
                return (
                  <View
                    key={`candidate-pin-${candidate.id}`}
                    style={[
                      styles.candidatePin,
                      {
                        left: Math.max(0, left - 5),
                        backgroundColor: candidateColor(candidate),
                        opacity: candidate.review_relevant ? 0.58 : 0.42,
                      },
                    ]}
                  />
                );
              })}
              {!usesPlayingRetroPrimaryReview && !isAudioVideoPoseMotionStage && playingRetroAnalysis?.candidates.filter(candidate => (
                candidate.review_relevant &&
                isTimestampVisible(candidate.timestamp_ms, safeTimelineWindowStartMs, timelineWindowEndMs)
              )).map(candidate => {
                const left = ratioToLeft(
                  candidate.timestamp_ms,
                  safeTimelineWindowStartMs,
                  timelineWindowEndMs,
                  overviewWidth,
                );
                return (
                  <View
                    key={`playing-retro-pin-${candidate.id}`}
                    style={[
                      styles.playingRetroCandidatePin,
                      {
                        left: Math.max(0, left - 6),
                        backgroundColor: candidateColor(candidate),
                        opacity: Math.max(0.46, candidate.playing_retro_prediction.confidence),
                      },
                    ]}
                  />
                );
              })}
              {(!isAudioVideoPoseReview || isAudioVideoPoseMotionStage) && videoPoseCandidates.filter(candidate => (
                !visiblePoseCandidateIds.has(candidate.id) &&
                shouldShowPoseCandidatePin(candidate) &&
                isTimestampVisible(candidate.timestamp_ms, safeTimelineWindowStartMs, timelineWindowEndMs)
              )).map(candidate => {
                const left = ratioToLeft(
                  candidate.timestamp_ms,
                  safeTimelineWindowStartMs,
                  timelineWindowEndMs,
                  overviewWidth,
                );
                return (
                  <View
                    key={`pose-candidate-pin-${candidate.id}`}
                    style={[
                      styles.poseCandidatePin,
                      {
                        left: Math.max(0, left - 5),
                        backgroundColor: poseCandidateColor(candidate),
                      },
                    ]}
                  />
                );
              })}
              {reviewTimelineMarkers.filter(marker => isTimestampVisible(
                marker.timestamp_ms,
                safeTimelineWindowStartMs,
                timelineWindowEndMs,
              )).map(marker => {
                const left = ratioToLeft(
                  marker.timestamp_ms,
                  safeTimelineWindowStartMs,
                  timelineWindowEndMs,
                  overviewWidth,
                );
                const isSelected = marker.id === selectedMarkerId;
                return (
                  <TouchableOpacity
                    key={`full-marker-${marker.id}`}
                    style={[styles.fullMarker, { left: Math.max(0, left - 12) }]}
                    onPress={() => handleSelectMarker(marker)}
                    {...(isSelected ? overviewMarkerResponder.panHandlers : {})}
                  >
                    <View
                      style={[
                        styles.fullMarkerPin,
                        { backgroundColor: markerColor(marker) },
                        isAudioVideoPoseReview &&
                          (isAudioVideoPoseMotionStage ? linkedMotionMarkerIds : linkedAudioMarkerIds).has(marker.id) &&
                          styles.fullMarkerPinLinked,
                        isSelected && styles.fullMarkerPinActive,
                      ]}
                    />
                  </TouchableOpacity>
                );
              })}
              {syncPointVisible && audioSyncPoint && (
                <View style={[styles.syncPointMarker, { left: Math.max(0, syncPointLeft - 1) }]}>
                  <Text style={styles.syncPointLabel}>SYNC</Text>
                  <View style={styles.syncPointLine} />
                </View>
              )}
              <View
                style={[styles.fullPlayheadHitbox, { left: Math.max(0, timelinePlayheadLeft - TIMELINE_EDGE_PX / 2) }]}
                {...reviewTimelinePlayheadResponder.panHandlers}
              >
                <View
                  style={[
                    styles.fullPlayheadLine,
                    (timelineMode === 'scrubbing' || timelineMode === 'autoScrollingWhileScrubbing') && styles.fullPlayheadLineActive,
                  ]}
                />
                <View
                  style={[
                    styles.fullPlayheadKnob,
                    (timelineMode === 'scrubbing' || timelineMode === 'autoScrollingWhileScrubbing') && styles.fullPlayheadKnobActive,
                  ]}
                />
              </View>
            </View>
            <Text style={styles.waveHelpTxt}>
              Dra playhead till träffen. Håll på den vita markeringen i 2 sekunder för att skapa marker.
            </Text>

            {(!isAudioVideoPoseReview || isAudioVideoPoseMotionStage) && activeStageMarkerCount > 0 && (
              <View style={styles.timelineMarkerNav}>
                <TouchableOpacity
                  style={[styles.timelineMarkerNavBtn, selectedStageMarkerIndex === 0 && styles.disabledBtn]}
                  onPress={() => handleJumpToMarker(-1)}
                  disabled={selectedStageMarkerIndex === 0}
                >
                  <Text style={styles.timelineMarkerNavTxt}>Föregående</Text>
                </TouchableOpacity>
                <View style={styles.timelineMarkerNavCenter}>
                  <Text style={styles.timelineMarkerNavTitle}>
                    Marker {selectedStageMarkerIndex >= 0 ? selectedStageMarkerIndex + 1 : 0} av {activeStageMarkerCount}
                  </Text>
                  <Text style={styles.timelineMarkerNavSub}>
                    {selectedMarker ? formatTimelineShort(selectedMarker.timestamp_ms) : formatTimelineShort(playbackPositionMs)}
                  </Text>
                </View>
                <TouchableOpacity
                  style={[
                    styles.timelineMarkerNavBtn,
                    selectedStageMarkerIndex >= activeStageMarkerCount - 1 && styles.disabledBtn,
                  ]}
                  onPress={() => handleJumpToMarker(1)}
                  disabled={selectedStageMarkerIndex >= activeStageMarkerCount - 1}
                >
                  <Text style={styles.timelineMarkerNavTxt}>Nästa</Text>
                </TouchableOpacity>
              </View>
            )}

            {allowsMarkerlessNoBounceSave && orderedMarkers.length === 0 && (
              <Text style={styles.waveHelpTxt}>
                Denna tagning kan sparas utan markers. Hela IMU-sekvensen används som negativ no-bounce-data.
              </Text>
            )}

            {quickLabelPrompt && supportsQuickLabels && !isAudioVideoPoseReview && (
              <View style={styles.quickLabelPanel}>
                <View style={styles.quickLabelCopy}>
                  <Text style={styles.quickLabelTitle}>Ny marker {formatTimelineShort(quickLabelPrompt.timestampMs)}</Text>
                  <Text style={styles.quickLabelSub}>Välj typ direkt</Text>
                </View>
                <View style={styles.quickLabelChoices}>
                  {quickLabelChoices.map(choice => (
                    <TouchableOpacity
                      key={`quick-${choice.id}`}
                      style={[styles.quickLabelBtn, { borderColor: choice.color }]}
                      onPress={() => handleQuickLabelChoice(choice)}
                    >
                      <Text style={styles.quickLabelBtnTxt}>{choice.title}</Text>
                    </TouchableOpacity>
                  ))}
                  <TouchableOpacity style={styles.quickLabelCancelBtn} onPress={() => setQuickLabelPrompt(null)}>
                    <Text style={styles.quickLabelCancelTxt}>Senare</Text>
                  </TouchableOpacity>
                </View>
                <View style={styles.quickLabelTools}>
                  <TouchableOpacity style={styles.utilityBtn} onPress={() => handleNudgeMarker(-EXTRA_LARGE_NUDGE_STEP_MS)} disabled={!selectedMarker}>
                    <Text style={styles.utilityTxt}>-50 ms</Text>
                  </TouchableOpacity>
                  <TouchableOpacity style={styles.utilityBtn} onPress={() => handleNudgeMarker(-LARGE_NUDGE_STEP_MS)} disabled={!selectedMarker}>
                    <Text style={styles.utilityTxt}>-20 ms</Text>
                  </TouchableOpacity>
                  <TouchableOpacity style={styles.utilityBtn} onPress={() => handleNudgeMarker(-NUDGE_STEP_MS)} disabled={!selectedMarker}>
                    <Text style={styles.utilityTxt}>-10 ms</Text>
                  </TouchableOpacity>
                  <TouchableOpacity style={styles.utilityBtn} onPress={() => handleNudgeMarker(NUDGE_STEP_MS)} disabled={!selectedMarker}>
                    <Text style={styles.utilityTxt}>+10 ms</Text>
                  </TouchableOpacity>
                  <TouchableOpacity style={styles.utilityBtn} onPress={() => handleNudgeMarker(LARGE_NUDGE_STEP_MS)} disabled={!selectedMarker}>
                    <Text style={styles.utilityTxt}>+20 ms</Text>
                  </TouchableOpacity>
                  <TouchableOpacity style={styles.utilityBtn} onPress={() => handleNudgeMarker(EXTRA_LARGE_NUDGE_STEP_MS)} disabled={!selectedMarker}>
                    <Text style={styles.utilityTxt}>+50 ms</Text>
                  </TouchableOpacity>
                  <TouchableOpacity style={[styles.utilityBtn, styles.utilityDeleteBtn]} onPress={handleDeleteSelectedMarker} disabled={!selectedMarker}>
                    <Text style={styles.utilityDeleteTxt}>Ta bort</Text>
                  </TouchableOpacity>
                </View>
              </View>
            )}

            {isAudioVideoPoseReview && selectedAudioMarker && (
              <View style={styles.layerEditorPanel}>
                <View style={styles.layerEditorHeader}>
                  <Text style={styles.layerEditorTitle}>Ljud {formatTimelineShort(selectedAudioMarker.timestamp_ms)}</Text>
                  <Text style={styles.layerEditorMeta}>
                    {markerConfidenceText(selectedAudioMarker) ?? markerDetailText(selectedAudioMarker)}
                  </Text>
                </View>
                <View style={styles.labelSegment}>
                  {AUDIO_VIDEO_POSE_AUDIO_LABEL_CHOICES.map(choice => {
                    const active = markerMatchesAudioPoseAudioChoice(selectedAudioMarker, choice);
                    return (
                      <TouchableOpacity
                        key={`inline-audio-${choice.id}`}
                        style={[
                          styles.segmentBtn,
                          active && { backgroundColor: choice.color, borderColor: choice.color },
                        ]}
                        onPress={() => {
                          setQuickLabelPrompt(null);
                          updateSelectedMarker(marker => applyAudioPoseAudioChoice(marker, choice));
                        }}
                      >
                        <Text style={[styles.segmentTxt, active && styles.segmentTxtActive]}>{choice.title}</Text>
                      </TouchableOpacity>
                    );
                  })}
                </View>
                <View style={styles.markerUtilityRow}>
                  <TouchableOpacity style={styles.utilityBtn} onPress={handlePlaySelectedMarker}>
                    <Text style={styles.utilityTxt}>Spela</Text>
                  </TouchableOpacity>
                  <TouchableOpacity style={styles.utilityBtn} onPress={handleSnapSelectedMarker}>
                    <Text style={styles.utilityTxt}>Snappa</Text>
                  </TouchableOpacity>
                  <TouchableOpacity style={styles.utilityBtn} onPress={() => handleNudgeMarker(-EXTRA_LARGE_NUDGE_STEP_MS)}>
                    <Text style={styles.utilityTxt}>-50 ms</Text>
                  </TouchableOpacity>
                  <TouchableOpacity style={styles.utilityBtn} onPress={() => handleNudgeMarker(-LARGE_NUDGE_STEP_MS)}>
                    <Text style={styles.utilityTxt}>-20 ms</Text>
                  </TouchableOpacity>
                  <TouchableOpacity style={styles.utilityBtn} onPress={() => handleNudgeMarker(-NUDGE_STEP_MS)}>
                    <Text style={styles.utilityTxt}>-10 ms</Text>
                  </TouchableOpacity>
                  <TouchableOpacity style={styles.utilityBtn} onPress={() => handleNudgeMarker(NUDGE_STEP_MS)}>
                    <Text style={styles.utilityTxt}>+10 ms</Text>
                  </TouchableOpacity>
                  <TouchableOpacity style={styles.utilityBtn} onPress={() => handleNudgeMarker(LARGE_NUDGE_STEP_MS)}>
                    <Text style={styles.utilityTxt}>+20 ms</Text>
                  </TouchableOpacity>
                  <TouchableOpacity style={styles.utilityBtn} onPress={() => handleNudgeMarker(EXTRA_LARGE_NUDGE_STEP_MS)}>
                    <Text style={styles.utilityTxt}>+50 ms</Text>
                  </TouchableOpacity>
                  <TouchableOpacity style={[styles.utilityBtn, styles.utilityDeleteBtn]} onPress={handleDeleteSelectedMarker}>
                    <Text style={styles.utilityDeleteTxt}>Ta bort</Text>
                  </TouchableOpacity>
                </View>
              </View>
            )}
            </>

            {isAudioVideoPoseReview ? (
              isAudioVideoPoseMotionStage ? (
              <View style={styles.motionTimelinePanel}>
                <View style={styles.motionTimelineTitleRow}>
                  <Text style={styles.motionTimelineTitle}>Rörelse</Text>
                  <Text style={styles.motionTimelineMeta}>
                    {poseAnalysisRunning ? 'Analyserar...' : `${motionTimelineMarkers.length} markers`}
                  </Text>
                </View>
                {poseAnalysisStatus && (
                  <Text style={styles.motionStatus}>{poseAnalysisStatus}</Text>
                )}
                <Text style={styles.motionHelpText}>
                  Video analyserar hela klippet. Justera FH/BH/Oklart direkt i tidslinjen ovan.
                </Text>
                {motionTimelineMarkers.length === 0 && !poseAnalysisRunning && (
                  <Text style={styles.motionEmpty}>Inga tydliga FH/BH-markers. Långtryck på linjen om du vill lägga till Forehand, Backhand eller Oklart manuellt.</Text>
                )}
                {selectedMotionMarker && (
                  <View style={styles.layerEditorPanel}>
                    <View style={styles.layerEditorHeader}>
                      <Text style={styles.layerEditorTitle}>Rörelse {formatTimelineShort(selectedMotionMarker.timestamp_ms)}</Text>
                      <Text style={styles.layerEditorMeta}>
                        {markerConfidenceText(selectedMotionMarker) ?? markerDetailText(selectedMotionMarker)}
                      </Text>
                    </View>
                    <View style={styles.labelSegment}>
                      {AUDIO_VIDEO_POSE_MOTION_LABEL_CHOICES.map(choice => {
                        const active = markerMatchesAudioPoseMotionChoice(selectedMotionMarker, choice);
                        return (
                          <TouchableOpacity
                            key={`inline-motion-${choice.id}`}
                            style={[
                              styles.segmentBtn,
                              active && { backgroundColor: choice.color, borderColor: choice.color },
                            ]}
                            onPress={() => {
                              setQuickLabelPrompt(null);
                              updateSelectedMarker(marker => applyAudioPoseMotionChoice(marker, choice));
                            }}
                          >
                            <Text style={[styles.segmentTxt, active && styles.segmentTxtActive]}>{choice.title}</Text>
                          </TouchableOpacity>
                        );
                      })}
                    </View>
                    <View style={styles.markerUtilityRow}>
                      <TouchableOpacity style={styles.utilityBtn} onPress={handlePlaySelectedMarker}>
                        <Text style={styles.utilityTxt}>Spela</Text>
                      </TouchableOpacity>
                      <TouchableOpacity style={styles.utilityBtn} onPress={() => handleNudgeMarker(-EXTRA_LARGE_NUDGE_STEP_MS)}>
                        <Text style={styles.utilityTxt}>-50 ms</Text>
                      </TouchableOpacity>
                      <TouchableOpacity style={styles.utilityBtn} onPress={() => handleNudgeMarker(-LARGE_NUDGE_STEP_MS)}>
                        <Text style={styles.utilityTxt}>-20 ms</Text>
                      </TouchableOpacity>
                      <TouchableOpacity style={styles.utilityBtn} onPress={() => handleNudgeMarker(-NUDGE_STEP_MS)}>
                        <Text style={styles.utilityTxt}>-10 ms</Text>
                      </TouchableOpacity>
                      <TouchableOpacity style={styles.utilityBtn} onPress={() => handleNudgeMarker(NUDGE_STEP_MS)}>
                        <Text style={styles.utilityTxt}>+10 ms</Text>
                      </TouchableOpacity>
                      <TouchableOpacity style={styles.utilityBtn} onPress={() => handleNudgeMarker(LARGE_NUDGE_STEP_MS)}>
                        <Text style={styles.utilityTxt}>+20 ms</Text>
                      </TouchableOpacity>
                      <TouchableOpacity style={styles.utilityBtn} onPress={() => handleNudgeMarker(EXTRA_LARGE_NUDGE_STEP_MS)}>
                        <Text style={styles.utilityTxt}>+50 ms</Text>
                      </TouchableOpacity>
                      <TouchableOpacity style={[styles.utilityBtn, styles.utilityDeleteBtn]} onPress={handleDeleteSelectedMarker}>
                        <Text style={styles.utilityDeleteTxt}>Ta bort</Text>
                      </TouchableOpacity>
                    </View>
                  </View>
                )}
              </View>
              ) : null
            ) : (
            <View style={styles.imuPanel}>
              <View style={styles.imuTitleRow}>
                <Text style={styles.imuTitle}>IMU-förhandsvisning</Text>
                <Text style={styles.imuSyncedTxt}>synkad</Text>
              </View>
              {previewImuSamples.length > 0 ? (
                <>
                  <View style={styles.imuRow}>
                    <View style={styles.imuLegend}>
                      <Text style={styles.imuLegendTitle}>Accel</Text>
                      <Text style={styles.imuLegendSub}>m/s2</Text>
                      <Text style={styles.imuAxisLegend}>
                        <Text style={{ color: IMU_SERIES_COLORS.x }}>X </Text>
                        <Text style={{ color: IMU_SERIES_COLORS.y }}>Y </Text>
                        <Text style={{ color: IMU_SERIES_COLORS.z }}>Z</Text>
                      </Text>
                    </View>
                    <View
                      style={styles.imuPlotWrap}
                      onLayout={eventData => setImuPlotWidth(eventData.nativeEvent.layout.width)}
                    >
                      <ImuLinePlot
                        samples={previewImuSamples}
                        keys={[
                          { key: 'accel_x', axis: 'x', color: IMU_SERIES_COLORS.x },
                          { key: 'accel_y', axis: 'y', color: IMU_SERIES_COLORS.y },
                          { key: 'accel_z', axis: 'z', color: IMU_SERIES_COLORS.z },
                        ]}
                        startMs={safeTimelineWindowStartMs}
                        endMs={timelineWindowEndMs}
                        playheadMs={playbackPositionMs}
                        markers={orderedMarkers}
                        selectedMarkerId={selectedMarkerId}
                        width={imuPlotWidth}
                        recordingStartedAtMs={event.imu_recording?.started_at_ms}
                        onSelectMarker={handleSelectMarker}
                      />
                    </View>
                  </View>
                  <View style={styles.imuRow}>
                    <View style={styles.imuLegend}>
                      <Text style={styles.imuLegendTitle}>Gyro</Text>
                      <Text style={styles.imuLegendSub}>deg/s</Text>
                      <Text style={styles.imuAxisLegend}>
                        <Text style={{ color: IMU_SERIES_COLORS.x }}>X </Text>
                        <Text style={{ color: IMU_SERIES_COLORS.y }}>Y </Text>
                        <Text style={{ color: IMU_SERIES_COLORS.z }}>Z</Text>
                      </Text>
                    </View>
                    <View style={styles.imuPlotWrap}>
                      <ImuLinePlot
                        samples={previewImuSamples}
                        keys={[
                          { key: 'gyro_x', axis: 'x', color: IMU_SERIES_COLORS.x },
                          { key: 'gyro_y', axis: 'y', color: IMU_SERIES_COLORS.y },
                          { key: 'gyro_z', axis: 'z', color: IMU_SERIES_COLORS.z },
                        ]}
                        startMs={safeTimelineWindowStartMs}
                        endMs={timelineWindowEndMs}
                        playheadMs={playbackPositionMs}
                        markers={orderedMarkers}
                        selectedMarkerId={selectedMarkerId}
                        width={imuPlotWidth}
                        recordingStartedAtMs={event.imu_recording?.started_at_ms}
                        onSelectMarker={handleSelectMarker}
                      />
                    </View>
                  </View>
                </>
              ) : (
                <Text style={styles.imuEmpty}>Ingen IMU sparad för denna tagning.</Text>
              )}
            </View>
            )}
          </View>

          {!isAudioVideoPoseReview && (
          <View style={styles.markerPanel}>
            <View style={styles.markerTopRow}>
              <View>
                <Text style={styles.markerPanelTitle}>Marker {selectedMarkerIndex >= 0 ? selectedMarkerIndex + 1 : 0} av {orderedMarkers.length}</Text>
                <Text style={styles.markerPanelSub}>
                  {formatTimelineShort(selectedMarker?.timestamp_ms ?? playbackPositionMs)} · {markerDetailText(selectedMarker)}
                </Text>
                {selectedMarker?.source === 'auto' && (
                  <Text style={styles.markerConfidenceTxt}>
                    Säkerhet {Math.round(markerConfidence(selectedMarker) * 100)}%
                  </Text>
                )}
              </View>
              <View style={styles.markerControlsRow}>
                <TouchableOpacity
                  style={[styles.roundControlBtn, selectedMarkerIndex <= 0 && styles.disabledBtn]}
                  onPress={() => handleJumpToMarker(-1)}
                  disabled={selectedMarkerIndex <= 0}
                >
                  <Text style={styles.roundControlTxt}>Föreg.</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.centerPlayBtn} onPress={handlePlaySelectedMarker}>
                  <Text style={styles.centerPlayTxt}>Spela</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.roundControlBtn, selectedMarkerIndex >= orderedMarkers.length - 1 && styles.disabledBtn]}
                  onPress={() => handleJumpToMarker(1)}
                  disabled={selectedMarkerIndex >= orderedMarkers.length - 1}
                >
                  <Text style={styles.roundControlTxt}>Nästa</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.addMarkerBtn} onPress={handleAddMarkerHere}>
                  <Text style={styles.addMarkerTxt}>+ Lägg till</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[
                    styles.approveAllBtn,
                    isPlayingReview && styles.hidden,
                    approvableAutoMarkers.length === 0 && styles.disabledBtn,
                  ]}
                  onPress={handleApproveAllMarkers}
                  disabled={isPlayingReview || approvableAutoMarkers.length === 0}
                >
                  <Text style={styles.approveAllTxt}>Godkänn alla</Text>
                </TouchableOpacity>
              </View>
            </View>

            {isPlayingReview && (
              <View style={styles.playingFilterPanel}>
                <Text style={styles.playingFilterMeta}>
                  Visar {playingAutoMarkerStats.visible} av {playingAutoMarkerStats.total} auto-detekterade
                </Text>
                <View style={styles.playingFilterRow}>
                  {PLAYING_CONFIDENCE_FILTERS.map(filter => (
                    <TouchableOpacity
                      key={filter.id}
                      style={[
                        styles.playingFilterBtn,
                        playingConfidenceFilter === filter.id && styles.playingFilterBtnActive,
                      ]}
                      onPress={() => setPlayingConfidenceFilter(filter.id)}
                    >
                      <Text
                        style={[
                          styles.playingFilterTxt,
                          playingConfidenceFilter === filter.id && styles.playingFilterTxtActive,
                        ]}
                      >
                        {filter.title}
                      </Text>
                    </TouchableOpacity>
                  ))}
                </View>
              </View>
            )}

            {isAudioVideoPoseReview ? (
              <View style={styles.audioPoseLayerPanel}>
                <Text style={styles.audioPoseLayerTitle}>Ljud</Text>
                <View style={styles.labelSegment}>
                  {AUDIO_VIDEO_POSE_AUDIO_LABEL_CHOICES.map(choice => {
                    const active = markerMatchesAudioPoseAudioChoice(selectedMarker, choice);
                    return (
                      <TouchableOpacity
                        key={choice.id}
                        style={[
                          styles.segmentBtn,
                          active && { backgroundColor: choice.color, borderColor: choice.color },
                        ]}
                        disabled={!selectedMarker}
                        onPress={() => {
                          setQuickLabelPrompt(null);
                          updateSelectedMarker(marker => applyAudioPoseAudioChoice(marker, choice));
                        }}
                      >
                        <Text style={[styles.segmentTxt, active && styles.segmentTxtActive]}>{choice.title}</Text>
                      </TouchableOpacity>
                    );
                  })}
                </View>
                <Text style={styles.audioPoseLayerTitle}>Rörelse</Text>
                <View style={styles.labelSegment}>
                  {AUDIO_VIDEO_POSE_MOTION_LABEL_CHOICES.map(choice => {
                    const active = markerMatchesAudioPoseMotionChoice(selectedMarker, choice);
                    return (
                      <TouchableOpacity
                        key={choice.id}
                        style={[
                          styles.segmentBtn,
                          active && { backgroundColor: choice.color, borderColor: choice.color },
                        ]}
                        disabled={!selectedMarker}
                        onPress={() => {
                          setQuickLabelPrompt(null);
                          updateSelectedMarker(marker => applyAudioPoseMotionChoice(marker, choice));
                        }}
                      >
                        <Text style={[styles.segmentTxt, active && styles.segmentTxtActive]}>{choice.title}</Text>
                      </TouchableOpacity>
                    );
                  })}
                </View>
              </View>
            ) : (
              <View style={[styles.labelSegment, isPlayingReview && styles.playingLabelSegment]}>
                {reviewLabelChoices.map(choice => {
                  const active = markerMatchesChoice(selectedMarker, choice);
                  return (
                    <TouchableOpacity
                      key={choice.id}
                      style={[
                        styles.segmentBtn,
                        isPlayingReview && styles.playingSegmentBtn,
                        active && {
                          backgroundColor: choice.color,
                          borderColor: choice.color,
                        },
                      ]}
                      disabled={!selectedMarker}
                      onPress={() => {
                        setQuickLabelPrompt(null);
                        updateSelectedMarker(marker => applyReviewLabelChoice(marker, choice));
                      }}
                    >
                      <Text style={[styles.segmentTxt, active && styles.segmentTxtActive]}>{choice.title}</Text>
                    </TouchableOpacity>
                  );
                })}
              </View>
            )}

            <View style={styles.markerUtilityRow}>
              <TouchableOpacity style={styles.utilityBtn} onPress={handleSnapSelectedMarker} disabled={!selectedMarker}>
                <Text style={styles.utilityTxt}>Snappa</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.utilityBtn} onPress={() => handleNudgeMarker(-EXTRA_LARGE_NUDGE_STEP_MS)} disabled={!selectedMarker}>
                <Text style={styles.utilityTxt}>-50 ms</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.utilityBtn} onPress={() => handleNudgeMarker(-LARGE_NUDGE_STEP_MS)} disabled={!selectedMarker}>
                <Text style={styles.utilityTxt}>-20 ms</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.utilityBtn} onPress={() => handleNudgeMarker(-NUDGE_STEP_MS)} disabled={!selectedMarker}>
                <Text style={styles.utilityTxt}>-10 ms</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.utilityBtn} onPress={() => handleNudgeMarker(NUDGE_STEP_MS)} disabled={!selectedMarker}>
                <Text style={styles.utilityTxt}>+10 ms</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.utilityBtn} onPress={() => handleNudgeMarker(LARGE_NUDGE_STEP_MS)} disabled={!selectedMarker}>
                <Text style={styles.utilityTxt}>+20 ms</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.utilityBtn} onPress={() => handleNudgeMarker(EXTRA_LARGE_NUDGE_STEP_MS)} disabled={!selectedMarker}>
                <Text style={styles.utilityTxt}>+50 ms</Text>
              </TouchableOpacity>
              <TouchableOpacity style={[styles.utilityBtn, styles.utilityDeleteBtn]} onPress={handleDeleteSelectedMarker} disabled={!selectedMarker}>
                <Text style={styles.utilityDeleteTxt}>Ta bort</Text>
              </TouchableOpacity>
            </View>
          </View>
          )}

          {isAudioVideoPoseReview && (
          <View style={styles.stageControlPanel}>
            <View style={styles.stageControlTopRow}>
              <View style={styles.stageControlCopy}>
                <Text style={styles.markerPanelTitle}>
                  {playingRetroPrimaryWaitingForMarkers
                    ? 'Spel-retro analyserar'
                    : `${activeStageTitle} ${selectedStageMarkerIndex >= 0 ? selectedStageMarkerIndex + 1 : 0} av ${activeStageMarkerCount}`}
                </Text>
                <Text style={styles.markerPanelSub}>
                  {playingRetroPrimaryWaitingForMarkers
                    ? 'Gamla ljudförslag visas inte medan retro skapar review-listan.'
                    : selectedMarker
                    ? `${formatTimelineShort(selectedMarker.timestamp_ms)} · ${markerDetailText(selectedMarker)}`
                    : `${formatTimelineShort(playbackPositionMs)} · ingen marker vald`}
                </Text>
                {selectedMarker?.source === 'auto' && markerConfidenceText(selectedMarker) && (
                  <Text style={styles.markerConfidenceTxt}>{markerConfidenceText(selectedMarker)}</Text>
                )}
                <Text style={styles.stageControlMeta}>
                  {playingRetroPrimaryWaitingForMarkers
                    ? 'Vänta tills retro-markers finns i ljudvågen.'
                    : audioVideoStagePendingCount > 0
                    ? `${audioVideoStagePendingCount} pending auto-förslag kvar i detta steg.`
                    : 'Alla förslag i detta steg är hanterade.'}
                </Text>
              </View>
              <TouchableOpacity
                style={[styles.approveAllBtn, approvableAutoMarkers.length === 0 && styles.disabledBtn]}
                onPress={handleApproveAllMarkers}
                disabled={approvableAutoMarkers.length === 0}
              >
                <Text style={styles.approveAllTxt}>Godkänn alla</Text>
              </TouchableOpacity>
            </View>
            <View style={styles.markerControlsRow}>
              <TouchableOpacity
                style={[styles.roundControlBtn, selectedStageMarkerIndex <= 0 && styles.disabledBtn]}
                onPress={() => handleJumpToMarker(-1)}
                disabled={selectedStageMarkerIndex <= 0}
              >
                <Text style={styles.roundControlTxt}>Föreg.</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.centerPlayBtn, !selectedMarker && styles.disabledBtn]}
                onPress={handlePlaySelectedMarker}
                disabled={!selectedMarker}
              >
                <Text style={styles.centerPlayTxt}>Spela</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[
                  styles.roundControlBtn,
                  (selectedStageMarkerIndex < 0 || selectedStageMarkerIndex >= activeStageMarkerCount - 1) && styles.disabledBtn,
                ]}
                onPress={() => handleJumpToMarker(1)}
                disabled={selectedStageMarkerIndex < 0 || selectedStageMarkerIndex >= activeStageMarkerCount - 1}
              >
                <Text style={styles.roundControlTxt}>Nästa</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.addMarkerBtn} onPress={handleAddActiveStageMarkerHere}>
                <Text style={styles.addMarkerTxt}>+ Lägg till</Text>
              </TouchableOpacity>
            </View>
          </View>
          )}

          <View style={styles.algorithmPanel}>
            <View style={styles.algorithmHeaderRow}>
              <View style={styles.algorithmTitleRow}>
                <Text style={styles.algorithmTitle}>
                  {usesPlayingRetroPrimaryReview ? 'Spel-retro audio' : detectionConfigTitle(detectionConfigSnapshot)}
                </Text>
                <TouchableOpacity style={styles.algorithmInfoBtn} onPress={handleShowModelInfo}>
                  <Text style={styles.algorithmInfoTxt}>i</Text>
                </TouchableOpacity>
              </View>
              <Text style={styles.algorithmMeta}>
                {algorithmMetaText}
              </Text>
            </View>
            {usesPlayingRetroPrimaryReview ? (
              <Text style={styles.algorithmThresholdHint}>
                {playingRetroStatus ?? `Auto: ${playingRetroModelMetadata.selected_variant} (${playingRetroModelMetadata.windows.length} fonster).`}
              </Text>
            ) : isAudioVideoPoseReview && (
              <Text style={styles.algorithmThresholdHint}>
                Racketförslag kräver minst {Math.round(audioVideoPoseReviewConfidence(detectionConfigSnapshot) * 100)}% · merge-window {detectionConfigSnapshot.merge_window_ms} ms.
              </Text>
            )}
            {usesPlayingRetroPrimaryReview && reviewStartProfileText && (
              <Text style={styles.reviewStartTimingText}>
                Timing: {reviewStartProfileText}
              </Text>
            )}
            <Text style={styles.algorithmSaveHint}>
              {algorithmSaveHint}
            </Text>
            {isPlayingReview && !isAudioVideoPoseMotionStage && !usesPlayingRetroPrimaryReview && (
              <View style={styles.playingRetroPanel}>
                <View style={styles.playingRetroHeaderRow}>
                  <View style={styles.playingRetroTitleBlock}>
                    <Text style={styles.playingRetroTitle}>Spel-retro audio</Text>
                    <Text style={styles.playingRetroMeta}>
                      {playingRetroModelMetadata.selected_variant} · {playingRetroModelMetadata.windows.length} fönster
                    </Text>
                  </View>
                  <TouchableOpacity
                    style={[
                      styles.playingRetroRunBtn,
                      (playingRetroRunning || loading || modelCandidates.length === 0) && styles.playingRetroRunBtnDisabled,
                    ]}
                    onPress={handleRunPlayingRetroAudio}
                    disabled={playingRetroRunning || loading || modelCandidates.length === 0}
                  >
                    <Text style={styles.playingRetroRunTxt}>
                      {playingRetroRunning ? 'Kör...' : 'Kör retro'}
                    </Text>
                  </TouchableOpacity>
                </View>
                <Text style={styles.playingRetroHint}>
                  Separat post-review test. Skapar inte markers och påverkar inte Save.
                </Text>
                <View style={styles.playingRetroStatsRow}>
                  <View style={styles.playingRetroStat}>
                    <Text style={styles.playingRetroStatNumber}>{playingRetroSummary.racket}</Text>
                    <Text style={styles.playingRetroStatLabel}>Racket</Text>
                  </View>
                  <View style={styles.playingRetroStat}>
                    <Text style={styles.playingRetroStatNumber}>{playingRetroSummary.table}</Text>
                    <Text style={styles.playingRetroStatLabel}>Bord</Text>
                  </View>
                  <View style={styles.playingRetroStat}>
                    <Text style={styles.playingRetroStatNumber}>{playingRetroSummary.nonTarget}</Text>
                    <Text style={styles.playingRetroStatLabel}>Ej target</Text>
                  </View>
                  <View style={styles.playingRetroStat}>
                    <Text style={styles.playingRetroStatNumber}>{playingRetroSummary.total || modelCandidates.length}</Text>
                    <Text style={styles.playingRetroStatLabel}>Kandidater</Text>
                  </View>
                </View>
                <Text style={styles.playingRetroStatus}>
                  {playingRetroStatus ?? 'Kör för att visa retro-racket/bord som extra pinnar ovanför waveformen.'}
                </Text>
              </View>
            )}
            {!usesPlayingRetroPrimaryReview && (
              <>
                <Text style={styles.algorithmLabel}>Känslighet</Text>
                <View style={styles.algorithmOptionRow}>
                  {REVIEW_SENSITIVITY_OPTIONS.map(option => {
                    const active = detectionConfigSnapshot.sensitivity === option;
                    return (
                      <TouchableOpacity
                        key={`review-sensitivity-${option}`}
                        style={[styles.algorithmOptionBtn, active && styles.algorithmOptionBtnActive]}
                        onPress={() => handleDetectionConfigChange(option, detectionConfigSnapshot.detection_mode)}
                        disabled={loading}
                      >
                        <Text style={[styles.algorithmOptionTxt, active && styles.algorithmOptionTxtActive]}>
                          {sensitivityTitle(option)}
                        </Text>
                      </TouchableOpacity>
                    );
                  })}
                </View>
                <Text style={styles.algorithmLabel}>Modelläge</Text>
                <View style={styles.algorithmOptionRow}>
                  {REVIEW_DETECTION_MODE_OPTIONS.map(option => {
                    const active = detectionConfigSnapshot.detection_mode === option;
                    return (
                      <TouchableOpacity
                        key={`review-mode-${option}`}
                        style={[styles.algorithmOptionBtn, active && styles.algorithmOptionBtnActive]}
                        onPress={() => handleDetectionConfigChange(detectionConfigSnapshot.sensitivity, option)}
                        disabled={loading}
                      >
                        <Text style={[styles.algorithmOptionTxt, active && styles.algorithmOptionTxtActive]}>
                          {detectionModeTitle(option)}
                        </Text>
                      </TouchableOpacity>
                    );
                  })}
                </View>
              </>
            )}
          </View>

          <View style={styles.footerActions}>
            {isAudioVideoPoseMotionStage && (
              <TouchableOpacity style={[styles.footerBtn, styles.discardBtn]} onPress={handleReturnToAudioReview}>
                <Text style={styles.discardBtnTxt}>Tillbaka till ljud</Text>
              </TouchableOpacity>
            )}
            {isAudioVideoPoseAudioStage && (
              <TouchableOpacity
                style={[styles.footerBtn, styles.discardBtn]}
                onPress={handleSkipAudioReview}
                disabled={saving}
              >
                <Text style={styles.discardBtnTxt}>Hoppa över ljud</Text>
              </TouchableOpacity>
            )}
            <TouchableOpacity
              style={[styles.footerBtn, styles.saveBtn, (!canSave || saving) && styles.saveBtnDisabled]}
              onPress={handleSave}
              disabled={saving || !canSave}
            >
              <Text style={[styles.saveBtnTxt, !canSave && styles.saveBtnTxtDisabled]}>
                {saveButtonText}
              </Text>
            </TouchableOpacity>
            <TouchableOpacity style={[styles.footerBtn, styles.discardBtn]} onPress={handleDiscard}>
              <Text style={styles.discardBtnTxt}>Kassera</Text>
            </TouchableOpacity>
          </View>
        </ScrollView>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: '#0d0d0d',
  },
  hidden: { display: 'none' },
  preloadVideoMount: {
    position: 'absolute',
    width: 2,
    height: 2,
    opacity: 0.01,
    overflow: 'hidden',
  },
  preloadVideo: { width: 2, height: 2 },
  reviewScroll: {
    flex: 1,
  },
  reviewContent: {
    paddingHorizontal: 16,
    gap: 14,
  },
  reviewHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    minHeight: 54,
  },
  reviewBackBtn: {
    minWidth: 78,
    height: 40,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#2d3238',
    backgroundColor: '#101417',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 10,
  },
  reviewBackTxt: { color: '#dce2eb', fontSize: 13, fontWeight: '800' },
  reviewHeaderCopy: { flex: 1 },
  reviewTitle: { color: '#fff', fontSize: 21, fontWeight: '900' },
  reviewSubtitle: { color: '#aeb4be', fontSize: 14, marginTop: 3 },
  menuBtn: {
    width: 34,
    height: 44,
    alignItems: 'center',
    justifyContent: 'center',
  },
  menuTxt: { color: '#fff', fontSize: 22, fontWeight: '900' },
  heroVideoFrame: {
    borderRadius: 16,
    overflow: 'hidden',
    backgroundColor: '#050505',
  },
  noVideoSurface: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#101417',
  },
  noVideoTitle: { color: '#fff', fontSize: 20, fontWeight: '900' },
  noVideoSub: { color: '#aeb4be', fontSize: 13, marginTop: 6 },
  rateChip: {
    position: 'absolute',
    top: 12,
    left: 12,
    zIndex: 2,
    elevation: 2,
    borderRadius: 11,
    backgroundColor: 'rgba(0,0,0,0.56)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.22)',
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  rateChipTxt: { color: '#fff', fontSize: 16, fontWeight: '800' },
  videoControls: {
    position: 'absolute',
    left: 14,
    right: 14,
    bottom: 12,
    zIndex: 2,
    elevation: 2,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  videoPlayBtn: {
    minWidth: 54,
    height: 36,
    paddingHorizontal: 8,
    borderRadius: 12,
    backgroundColor: 'rgba(0,0,0,0.45)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  videoPlayTxt: { color: '#fff', fontSize: 12, fontWeight: '900' },
  videoTimeTxt: { color: '#e8e8e8', fontSize: 13, fontWeight: '700' },
  videoProgressTrack: {
    flex: 1,
    height: 18,
    borderRadius: 9,
    backgroundColor: 'rgba(20,20,20,0.78)',
    justifyContent: 'center',
  },
  videoProgressFill: {
    position: 'absolute',
    left: 0,
    top: 6,
    height: 5,
    borderRadius: 4,
    backgroundColor: '#2ecc71',
  },
  videoProgressKnob: {
    position: 'absolute',
    top: 1,
    width: 17,
    height: 17,
    marginLeft: -8,
    borderRadius: 9,
    backgroundColor: '#fff',
  },
  videoPreparingBadge: {
    position: 'absolute',
    top: 12,
    right: 12,
    zIndex: 3,
    elevation: 3,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    borderRadius: 12,
    backgroundColor: 'rgba(0,0,0,0.62)',
    paddingHorizontal: 10,
    paddingVertical: 7,
  },
  videoPreparingTxt: { color: '#fff', fontSize: 11, fontWeight: '800' },
  videoTapLayer: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 1,
    elevation: 1,
  },
  videoSyncPanel: {
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#242a30',
    backgroundColor: '#101316',
    padding: 10,
    gap: 8,
  },
  videoSyncHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: 10,
  },
  videoSyncHeaderCopy: { flex: 1, minWidth: 0 },
  videoSyncHeaderActions: { alignItems: 'flex-end', gap: 6 },
  videoSyncTitle: { color: '#fff', fontSize: 12, fontWeight: '900' },
  videoSyncMeta: { color: '#aeb4be', fontSize: 11, fontWeight: '700', marginTop: 2 },
  videoSyncOffsetPill: {
    color: '#2ee678',
    fontSize: 11,
    fontWeight: '900',
    backgroundColor: '#11271a',
    borderRadius: 999,
    paddingHorizontal: 8,
    paddingVertical: 5,
  },
  videoSyncToggleBtn: {
    borderRadius: 9,
    backgroundColor: '#1a2026',
    borderWidth: 1,
    borderColor: '#2d353d',
    paddingHorizontal: 8,
    paddingVertical: 5,
  },
  videoSyncToggleTxt: { color: '#dce2eb', fontSize: 10, fontWeight: '900' },
  videoSyncHelp: { color: '#cbd2dc', fontSize: 11, lineHeight: 16 },
  videoSyncControls: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  videoSyncBtn: {
    minHeight: 30,
    borderRadius: 9,
    backgroundColor: '#1a2026',
    borderWidth: 1,
    borderColor: '#2d353d',
    paddingHorizontal: 9,
    alignItems: 'center',
    justifyContent: 'center',
  },
  videoSyncBtnTxt: { color: '#dce2eb', fontSize: 11, fontWeight: '800' },
  videoSyncApplyBtn: {
    minHeight: 30,
    borderRadius: 9,
    backgroundColor: '#145c2a',
    borderWidth: 1,
    borderColor: '#269d4c',
    paddingHorizontal: 10,
    alignItems: 'center',
    justifyContent: 'center',
  },
  videoSyncApplyTxt: { color: '#fff', fontSize: 11, fontWeight: '900' },
  videoSyncResetBtn: {
    minHeight: 30,
    borderRadius: 9,
    backgroundColor: '#15251b',
    borderWidth: 1,
    borderColor: '#275c38',
    paddingHorizontal: 10,
    alignItems: 'center',
    justifyContent: 'center',
  },
  videoSyncResetTxt: { color: '#2ee678', fontSize: 11, fontWeight: '900' },
  videoSyncActiveTxt: { color: '#f5c76d', fontSize: 11, fontWeight: '800' },
  markerPanel: {
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#22272d',
    backgroundColor: '#111417',
    padding: 10,
    gap: 10,
  },
  stageControlPanel: {
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#25303a',
    backgroundColor: '#101417',
    padding: 10,
    gap: 10,
  },
  stageControlTopRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    gap: 10,
  },
  stageControlCopy: { flex: 1 },
  stageControlMeta: { color: '#7f8993', fontSize: 11, fontWeight: '800', marginTop: 4 },
  markerTopRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 10,
    flexWrap: 'wrap',
  },
  markerPanelTitle: { color: '#fff', fontSize: 16, fontWeight: '900' },
  markerPanelSub: { color: '#a9b0ba', fontSize: 12, marginTop: 2 },
  markerConfidenceTxt: { color: '#f5c76d', fontSize: 11, fontWeight: '800', marginTop: 2 },
  markerControlsRow: { flexDirection: 'row', alignItems: 'center', gap: 6, flexWrap: 'wrap' },
  timelineMarkerNav: {
    marginTop: 10,
    padding: 8,
    borderRadius: 12,
    backgroundColor: '#11161a',
    borderWidth: 1,
    borderColor: '#252c33',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 8,
  },
  timelineMarkerNavBtn: {
    minWidth: 92,
    height: 36,
    borderRadius: 10,
    backgroundColor: '#202225',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 10,
  },
  timelineMarkerNavTxt: { color: '#fff', fontSize: 11, fontWeight: '900' },
  timelineMarkerNavCenter: { flex: 1, alignItems: 'center' },
  timelineMarkerNavTitle: { color: '#fff', fontSize: 12, fontWeight: '900' },
  timelineMarkerNavSub: { color: '#9ca4ad', fontSize: 11, marginTop: 2, fontWeight: '700' },
  roundControlBtn: {
    width: 44,
    height: 38,
    borderRadius: 12,
    backgroundColor: '#202225',
    alignItems: 'center',
    justifyContent: 'center',
  },
  roundControlTxt: { color: '#fff', fontSize: 11, fontWeight: '900' },
  centerPlayBtn: {
    width: 70,
    height: 38,
    borderRadius: 12,
    backgroundColor: '#145c2a',
    borderWidth: 1,
    borderColor: '#269d4c',
    alignItems: 'center',
    justifyContent: 'center',
  },
  centerPlayTxt: { color: '#fff', fontSize: 11, fontWeight: '900' },
  addMarkerBtn: {
    minHeight: 38,
    borderRadius: 12,
    backgroundColor: '#222426',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 12,
  },
  addMarkerTxt: { color: '#fff', fontSize: 13, fontWeight: '900' },
  approveAllBtn: {
    minHeight: 38,
    borderRadius: 12,
    backgroundColor: '#12351f',
    borderWidth: 1,
    borderColor: '#2ecc71',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 12,
  },
  approveAllTxt: { color: '#2ee678', fontSize: 12, fontWeight: '900' },
  playingFilterPanel: {
    borderRadius: 12,
    backgroundColor: '#0c0d0f',
    borderWidth: 1,
    borderColor: '#24282e',
    padding: 6,
    gap: 6,
  },
  playingFilterMeta: { color: '#aeb4be', fontSize: 11, fontWeight: '800' },
  playingFilterRow: { flexDirection: 'row', gap: 6 },
  playingFilterBtn: {
    minHeight: 28,
    flex: 1,
    borderRadius: 9,
    borderWidth: 1,
    borderColor: '#2c333a',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 8,
  },
  playingFilterBtnActive: { borderColor: '#2ecc71', backgroundColor: '#12351f' },
  playingFilterTxt: { color: '#aeb4be', fontSize: 11, fontWeight: '900' },
  playingFilterTxtActive: { color: '#2ee678' },
  reviewLayerSummary: {
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#24303a',
    backgroundColor: '#0b1115',
    paddingHorizontal: 10,
    paddingVertical: 8,
    gap: 2,
  },
  reviewLayerTitle: { color: '#2ee678', fontSize: 12, fontWeight: '900' },
  reviewLayerText: { color: '#aeb4be', fontSize: 11, fontWeight: '700' },
  audioPoseLayerPanel: {
    gap: 7,
  },
  audioPoseLayerTitle: {
    color: '#727b86',
    fontSize: 10,
    fontWeight: '900',
    letterSpacing: 1.1,
    textTransform: 'uppercase',
  },
  algorithmPanel: {
    borderRadius: 14,
    borderWidth: 1,
    borderColor: '#24303a',
    backgroundColor: '#0b1115',
    padding: 10,
    gap: 7,
  },
  algorithmHeaderRow: {
    gap: 2,
  },
  algorithmTitleRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  algorithmTitle: { color: '#2ee678', fontSize: 12, fontWeight: '900' },
  algorithmInfoBtn: {
    width: 22,
    height: 22,
    borderRadius: 11,
    borderWidth: 1,
    borderColor: '#2ecc71',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#12351f',
  },
  algorithmInfoTxt: { color: '#2ee678', fontSize: 12, fontWeight: '900' },
  algorithmMeta: { color: '#aeb4be', fontSize: 11, fontWeight: '700' },
  algorithmThresholdHint: { color: '#f5c76d', fontSize: 11, lineHeight: 15, fontWeight: '800' },
  reviewStartTimingText: { color: '#7f8993', fontSize: 10, lineHeight: 14, fontWeight: '800' },
  algorithmSaveHint: { color: '#7f8993', fontSize: 10, lineHeight: 14, fontWeight: '700' },
  playingRetroPanel: {
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#2a4a5f',
    backgroundColor: '#0d1720',
    padding: 9,
    gap: 7,
  },
  playingRetroHeaderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 8,
  },
  playingRetroTitleBlock: { flex: 1, gap: 2 },
  playingRetroTitle: { color: '#cfeeff', fontSize: 12, fontWeight: '900' },
  playingRetroMeta: { color: '#8fb4c8', fontSize: 10, fontWeight: '800' },
  playingRetroRunBtn: {
    minHeight: 30,
    borderRadius: 9,
    backgroundColor: '#194d64',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 10,
  },
  playingRetroRunBtnDisabled: { opacity: 0.45 },
  playingRetroRunTxt: { color: '#e8f8ff', fontSize: 11, fontWeight: '900' },
  playingRetroHint: { color: '#9db4c0', fontSize: 10, lineHeight: 14, fontWeight: '700' },
  playingRetroStatsRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  playingRetroStat: {
    flexGrow: 1,
    minWidth: 70,
    borderRadius: 9,
    borderWidth: 1,
    borderColor: '#244355',
    backgroundColor: '#101f29',
    paddingHorizontal: 8,
    paddingVertical: 6,
  },
  playingRetroStatNumber: { color: '#fff', fontSize: 15, fontWeight: '900' },
  playingRetroStatLabel: { color: '#9db4c0', fontSize: 10, fontWeight: '800' },
  playingRetroStatus: { color: '#cfeeff', fontSize: 10, lineHeight: 14, fontWeight: '800' },
  algorithmLabel: {
    color: '#727b86',
    fontSize: 10,
    fontWeight: '900',
    letterSpacing: 1.2,
    textTransform: 'uppercase',
  },
  algorithmOptionRow: { flexDirection: 'row', gap: 6 },
  algorithmOptionBtn: {
    flex: 1,
    minHeight: 30,
    borderRadius: 9,
    borderWidth: 1,
    borderColor: '#2c333a',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 8,
  },
  algorithmOptionBtnActive: { borderColor: '#2ecc71', backgroundColor: '#12351f' },
  algorithmOptionTxt: { color: '#aeb4be', fontSize: 11, fontWeight: '900' },
  algorithmOptionTxtActive: { color: '#2ee678' },
  labelSegment: {
    minHeight: 42,
    borderRadius: 13,
    borderWidth: 1,
    borderColor: '#24282e',
    backgroundColor: '#0c0d0f',
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
    padding: 6,
  },
  playingLabelSegment: { minHeight: 34, padding: 4, gap: 4 },
  segmentBtn: {
    minHeight: 32,
    minWidth: 92,
    flexGrow: 1,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#24282e',
    paddingHorizontal: 8,
  },
  playingSegmentBtn: { minHeight: 28, minWidth: 86, paddingHorizontal: 6 },
  segmentTxt: { color: '#aeb4be', fontSize: 13, fontWeight: '900' },
  segmentTxtActive: { color: '#fff' },
  markerUtilityRow: { flexDirection: 'row', gap: 8, flexWrap: 'wrap' },
  utilityBtn: {
    borderRadius: 10,
    backgroundColor: '#1c2024',
    paddingHorizontal: 10,
    paddingVertical: 8,
  },
  utilityTxt: { color: '#cbd2dc', fontSize: 12, fontWeight: '800' },
  utilityDeleteBtn: { backgroundColor: '#2a1111' },
  utilityDeleteTxt: { color: '#ff8989', fontSize: 12, fontWeight: '800' },
  waveformCard: {
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#242a30',
    backgroundColor: '#101316',
    padding: 14,
    gap: 10,
  },
  timelineHeaderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 10,
    flexWrap: 'wrap',
  },
  timelineTitle: { color: '#fff', fontSize: 14, fontWeight: '900' },
  zoomControlsRow: { flexDirection: 'row', alignItems: 'center', gap: 6, flexWrap: 'wrap' },
  zoomBtn: {
    minWidth: 34,
    height: 30,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#2c333a',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#15191d',
  },
  zoomBtnActive: { borderColor: '#2ecc71', backgroundColor: '#11351d' },
  zoomBtnTxt: { color: '#aeb4be', fontSize: 12, fontWeight: '900' },
  zoomBtnTxtActive: { color: '#2ee678' },
  imuTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 12,
    paddingVertical: 10,
  },
  timeTicksRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  tickTxt: { color: '#9aa2ad', fontSize: 11 },
  fullWaveformSurface: {
    height: 116,
    borderWidth: 1,
    borderColor: '#262d33',
    backgroundColor: '#0c0e10',
    overflow: 'hidden',
  },
  fullWaveformRow: {
    ...StyleSheet.absoluteFillObject,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 0,
    paddingHorizontal: 3,
  },
  fullWaveBar: {
    flex: 1,
    minWidth: 1,
    borderRadius: 2,
    backgroundColor: '#39d961',
    alignSelf: 'center',
  },
  fullMarker: {
    position: 'absolute',
    top: 20,
    width: 28,
    height: 28,
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 4,
    elevation: 4,
  },
  candidatePin: {
    position: 'absolute',
    top: 29,
    width: 10,
    height: 10,
    borderRadius: 5,
    zIndex: 2,
  },
  playingRetroCandidatePin: {
    position: 'absolute',
    top: 8,
    width: 12,
    height: 12,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#e8f8ff',
    zIndex: 3,
  },
  poseCandidatePin: {
    position: 'absolute',
    top: 13,
    width: 10,
    height: 10,
    borderRadius: 5,
    zIndex: 2,
    opacity: 0.72,
  },
  fullMarkerPin: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  fullMarkerPinActive: {
    borderWidth: 2,
    borderColor: '#ffffff',
    width: 14,
    height: 14,
    borderRadius: 7,
  },
  fullMarkerPinLinked: {
    borderWidth: 2,
    borderColor: '#35c7ff',
    width: 14,
    height: 14,
    borderRadius: 7,
  },
  syncPointMarker: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    width: 2,
    zIndex: 3,
    elevation: 3,
    alignItems: 'center',
  },
  syncPointLabel: {
    color: '#f5c76d',
    fontSize: 9,
    fontWeight: '900',
    backgroundColor: 'rgba(0,0,0,0.7)',
    borderRadius: 5,
    paddingHorizontal: 4,
    paddingVertical: 2,
    marginTop: 4,
  },
  syncPointLine: {
    flex: 1,
    width: 2,
    backgroundColor: '#f5c76d',
    opacity: 0.92,
  },
  fullPlayheadHitbox: {
    position: 'absolute',
    top: -10,
    bottom: 0,
    width: TIMELINE_EDGE_PX,
    alignItems: 'center',
    zIndex: 5,
    elevation: 5,
  },
  fullPlayheadLine: {
    position: 'absolute',
    top: 10,
    bottom: 0,
    width: 2,
    backgroundColor: '#fff',
  },
  fullPlayheadLineActive: {
    width: 3,
    backgroundColor: '#f8fff9',
  },
  fullPlayheadKnob: {
    position: 'absolute',
    top: 0,
    width: 18,
    height: 18,
    borderRadius: 9,
    backgroundColor: '#fff',
  },
  fullPlayheadKnobActive: {
    top: -3,
    width: 24,
    height: 24,
    borderRadius: 12,
    borderWidth: 3,
    borderColor: '#2ecc71',
  },
  waveHelpTxt: { color: '#aeb4be', fontSize: 12 },
  quickLabelPanel: {
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#2ecc71',
    backgroundColor: '#0f2117',
    padding: 8,
    gap: 8,
  },
  quickLabelCopy: {
    flexDirection: 'row',
    alignItems: 'baseline',
    justifyContent: 'space-between',
    gap: 8,
  },
  quickLabelTitle: { color: '#fff', fontSize: 12, fontWeight: '900' },
  quickLabelSub: { color: '#9ee7b5', fontSize: 11, fontWeight: '800' },
  quickLabelChoices: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  quickLabelTools: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  quickLabelBtn: {
    minHeight: 30,
    borderRadius: 10,
    borderWidth: 1,
    backgroundColor: '#15191d',
    paddingHorizontal: 9,
    alignItems: 'center',
    justifyContent: 'center',
  },
  quickLabelBtnTxt: { color: '#fff', fontSize: 11, fontWeight: '900' },
  quickLabelCancelBtn: {
    minHeight: 30,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#2c333a',
    backgroundColor: '#101316',
    paddingHorizontal: 9,
    alignItems: 'center',
    justifyContent: 'center',
  },
  quickLabelCancelTxt: { color: '#aeb4be', fontSize: 11, fontWeight: '900' },
  layerEditorPanel: {
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#24303a',
    backgroundColor: '#0b1115',
    padding: 8,
    gap: 8,
  },
  layerEditorHeader: {
    flexDirection: 'row',
    alignItems: 'baseline',
    justifyContent: 'space-between',
    gap: 8,
    flexWrap: 'wrap',
  },
  layerEditorTitle: { color: '#fff', fontSize: 12, fontWeight: '900' },
  layerEditorMeta: { color: '#aeb4be', fontSize: 11, fontWeight: '800' },
  imuCard: {
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#242a30',
    backgroundColor: '#101316',
    padding: 12,
  },
  segmentedTabs: {
    minHeight: 48,
    borderRadius: 14,
    backgroundColor: '#222426',
    flexDirection: 'row',
    padding: 3,
    marginBottom: 10,
  },
  tabBtn: {
    flex: 1,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  tabBtnActive: { backgroundColor: '#12481f' },
  tabTxt: { color: '#9da4ae', fontSize: 15, fontWeight: '800' },
  tabTxtActive: { color: '#2ee678', fontSize: 15, fontWeight: '900' },
  imuPanel: {
    borderRadius: 12,
    backgroundColor: '#151719',
    overflow: 'hidden',
    marginTop: 2,
  },
  motionPanel: {
    borderRadius: 12,
    backgroundColor: '#151719',
    overflow: 'hidden',
    marginTop: 2,
  },
  motionTimelinePanel: {
    borderRadius: 12,
    backgroundColor: '#151719',
    overflow: 'hidden',
    marginTop: 2,
  },
  motionTimelineTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 12,
    paddingVertical: 10,
  },
  motionTimelineTitle: { color: '#dce2eb', fontSize: 14, fontWeight: '900' },
  motionTimelineMeta: { color: '#2ee678', fontSize: 11, fontWeight: '900' },
  motionTimelineSurface: {
    height: 82,
    borderTopWidth: 1,
    borderTopColor: '#2a3036',
    backgroundColor: '#0f1113',
    overflow: 'hidden',
  },
  motionTimelineBaseLine: {
    position: 'absolute',
    left: 8,
    right: 8,
    top: 41,
    height: 2,
    borderRadius: 2,
    backgroundColor: '#2a3036',
  },
  motionCandidateHitbox: {
    position: 'absolute',
    top: 12,
    width: 24,
    height: 24,
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 2,
    elevation: 2,
  },
  motionCandidatePin: {
    width: 9,
    height: 9,
    borderRadius: 5,
    opacity: 0.58,
  },
  motionMarkerHitbox: {
    position: 'absolute',
    top: 29,
    width: 28,
    height: 28,
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 4,
    elevation: 4,
  },
  motionMarkerPin: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  motionMarkerPinLinked: {
    width: 16,
    height: 16,
    borderRadius: 8,
    borderWidth: 2,
    borderColor: '#2ee678',
  },
  motionMarkerPinActive: {
    width: 18,
    height: 18,
    borderRadius: 9,
    borderWidth: 3,
    borderColor: '#fff',
  },
  motionPlayheadHitbox: {
    position: 'absolute',
    top: -8,
    bottom: 0,
    width: TIMELINE_EDGE_PX,
    alignItems: 'center',
    zIndex: 5,
    elevation: 5,
  },
  motionPlayheadLine: {
    position: 'absolute',
    top: 8,
    bottom: 0,
    width: 2,
    backgroundColor: '#fff',
  },
  motionPlayheadLineActive: {
    width: 3,
    backgroundColor: '#f8fff9',
  },
  motionPlayheadKnob: {
    position: 'absolute',
    top: 22,
    width: 18,
    height: 18,
    borderRadius: 9,
    backgroundColor: '#fff',
  },
  motionPlayheadKnobActive: {
    top: 19,
    width: 24,
    height: 24,
    borderRadius: 12,
    borderWidth: 3,
    borderColor: '#2ecc71',
  },
  motionTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 12,
    paddingVertical: 10,
  },
  motionTitle: { color: '#dce2eb', fontSize: 14, fontWeight: '900' },
  motionMeta: { color: '#2ee678', fontSize: 11, fontWeight: '900' },
  motionStatus: {
    color: '#aeb4be',
    fontSize: 12,
    fontWeight: '700',
    paddingHorizontal: 12,
    paddingBottom: 8,
  },
  motionHelpText: {
    color: '#7f8791',
    fontSize: 11,
    fontWeight: '700',
    lineHeight: 15,
    paddingHorizontal: 12,
    paddingBottom: 10,
  },
  motionList: {
    borderTopWidth: 1,
    borderTopColor: '#2a3036',
  },
  motionRow: {
    minHeight: 48,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#242a30',
  },
  motionRowActive: { backgroundColor: '#102019' },
  motionDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  motionCopy: { flex: 1, minWidth: 0 },
  motionRowTitle: { color: '#fff', fontSize: 13, fontWeight: '900' },
  motionRowMeta: { color: '#9aa2ad', fontSize: 11, fontWeight: '700', marginTop: 2 },
  motionEmpty: { color: '#8e949d', fontSize: 13, padding: 14 },
  imuTitle: { color: '#aeb4be', fontSize: 14, fontWeight: '800' },
  imuSyncedTxt: { color: '#2ee678', fontSize: 11, fontWeight: '900' },
  imuRow: {
    height: 78,
    flexDirection: 'row',
    borderTopWidth: 1,
    borderTopColor: '#2a3036',
  },
  imuLegend: {
    width: 92,
    borderRightWidth: 1,
    borderRightColor: '#2a3036',
    justifyContent: 'center',
    paddingHorizontal: 10,
  },
  imuLegendTitle: { color: '#f0f0f0', fontSize: 13, fontWeight: '800' },
  imuLegendSub: { color: '#9aa2ad', fontSize: 12, marginTop: 3 },
  imuAxisLegend: { fontSize: 11, fontWeight: '900', marginTop: 4 },
  imuPlotWrap: {
    flex: 1,
    height: IMU_PLOT_HEIGHT,
    alignSelf: 'center',
  },
  imuPlot: {
    flex: 1,
    position: 'relative',
    overflow: 'hidden',
    backgroundColor: '#0f1113',
  },
  imuZeroLine: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: '50%',
    height: 1,
    backgroundColor: '#283038',
  },
  imuLineSegment: {
    position: 'absolute',
    height: 2,
    borderRadius: 2,
  },
  imuMarkerHitbox: {
    position: 'absolute',
    top: 0,
    width: 22,
    height: IMU_PLOT_HEIGHT,
    alignItems: 'center',
  },
  imuMarkerLine: {
    width: 2,
    height: IMU_PLOT_HEIGHT,
    opacity: 0.75,
  },
  imuMarkerLineActive: {
    width: 4,
    opacity: 1,
  },
  imuPlayhead: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    width: 2,
    backgroundColor: '#fff',
  },
  imuFooter: { color: '#aeb4be', fontSize: 13, padding: 12 },
  imuEmpty: { color: '#8e949d', fontSize: 13, padding: 14 },
  disabledBtn: { opacity: 0.45 },
  screen: {
    flex: 1,
    paddingHorizontal: 16,
    paddingTop: 0,
    paddingBottom: 12,
  },
  loadingBox: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 12,
    paddingHorizontal: 28,
  },
  loadingBackBtn: {
    position: 'absolute',
    top: 48,
    left: 18,
    minWidth: 78,
    height: 40,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#2d3238',
    backgroundColor: '#101417',
    alignItems: 'center',
    justifyContent: 'center',
  },
  loadingBackTxt: { color: '#dce2eb', fontSize: 13, fontWeight: '900' },
  loadingTitle: { color: '#fff', fontSize: 20, fontWeight: '900', textAlign: 'center' },
  loadingTxt: { color: '#aaa', fontSize: 13, textAlign: 'center', lineHeight: 18 },
  retroPrepCard: {
    width: '100%',
    maxWidth: 420,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#24303a',
    backgroundColor: '#0b1115',
    padding: 14,
    gap: 8,
    marginTop: 4,
  },
  retroPrepLine: { color: '#dce2eb', fontSize: 12, fontWeight: '800', lineHeight: 17 },
  retroPrepMeta: { color: '#f5c76d', fontSize: 12, fontWeight: '900', lineHeight: 17, marginTop: 2 },
  retroPrepTiming: { color: '#7f8993', fontSize: 10, fontWeight: '800', lineHeight: 14 },
  header: {
    flexDirection: 'column',
    alignItems: 'flex-start',
    gap: 6,
    marginBottom: 8,
  },
  headerMain: { flex: 1 },
  backBtn: { paddingBottom: 2 },
  backTxt: { color: '#4a9eff', fontSize: 14, fontWeight: '700' },
  headerTitle: { color: '#fff', fontSize: 22, fontWeight: '800' },
  headerSub: { color: '#8a8a8a', fontSize: 11, marginTop: 2 },
  headerInfo: {
    width: '100%',
    alignItems: 'flex-start',
  },
  revisionTxt: { color: '#f5c76d', fontSize: 12, fontWeight: '800' },
  headerHint: { color: '#7c7c7c', fontSize: 10, marginTop: 2, textAlign: 'left', lineHeight: 14 },
  topSection: {
    borderRadius: 16,
    backgroundColor: '#111',
    padding: 12,
    marginBottom: 10,
  },
  topRow: {
    flexDirection: 'row',
    gap: 10,
    flex: 1,
  },
  videoColumn: {
    alignItems: 'center',
    gap: 6,
  },
  controlColumn: {
    flex: 1,
  },
  controlColumnSolo: {
    flex: 1,
  },
  card: {
    backgroundColor: '#111',
    borderRadius: 16,
    padding: 12,
  },
  editorCard: {
    flex: 1,
    backgroundColor: '#111',
    borderRadius: 16,
    padding: 12,
    minHeight: 0,
  },
  sectionLabel: { color: '#777', fontSize: 9, letterSpacing: 1.8, marginBottom: 6 },
  helperTxt: { color: '#8e8e8e', fontSize: 11, lineHeight: 15, marginBottom: 8 },
  videoFramePortrait: {
    borderRadius: 12,
    overflow: 'hidden',
    backgroundColor: '#050505',
  },
  videoPlayer: {
    width: '100%',
    height: '100%',
    backgroundColor: '#050505',
  },
  videoMeta: {
    color: '#6f6f6f',
    fontSize: 10,
    fontFamily: 'monospace',
    textAlign: 'center',
  },
  controlsBlock: {
    gap: 6,
  },
  overviewMini: {
    backgroundColor: '#0d0d0d',
    borderRadius: 14,
    paddingHorizontal: 10,
    paddingVertical: 8,
    overflow: 'hidden',
    minHeight: 112,
    marginBottom: 10,
  },
  detailSurface: {
    flex: 1,
    backgroundColor: '#0d0d0d',
    borderRadius: 14,
    paddingHorizontal: 10,
    paddingVertical: 8,
    overflow: 'hidden',
    minHeight: 72,
  },
  waveformRow: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 1,
  },
  overviewWaveformRow: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 0,
  },
  overviewBar: {
    width: 2,
    borderRadius: 2,
    backgroundColor: '#4a9eff',
    alignSelf: 'center',
  },
  detailBar: {
    borderRadius: 2,
    backgroundColor: '#4a9eff',
    alignSelf: 'center',
  },
  overviewMarkerHitbox: {
    position: 'absolute',
    top: 6,
    bottom: 6,
    width: 32,
    alignItems: 'center',
    justifyContent: 'flex-start',
    borderWidth: 1,
    borderRadius: 10,
    zIndex: 4,
    elevation: 4,
  },
  markerStem: { width: 3, flex: 1, marginTop: 4 },
  markerDot: { width: 10, height: 10, borderRadius: 5, marginBottom: 6 },
  playheadHitbox: {
    position: 'absolute',
    top: 6,
    bottom: 6,
    width: 44,
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 5,
    elevation: 5,
  },
  playheadLine: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    width: 2,
    backgroundColor: '#ffffff',
  },
  playheadKnob: {
    position: 'absolute',
    top: -2,
    width: 16,
    height: 16,
    borderRadius: 8,
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: '#111',
  },
  detailPlayheadLine: {
    position: 'absolute',
    top: 6,
    bottom: 6,
    width: 2,
    backgroundColor: '#ffffff',
  },
  detailMarkerHitbox: {
    position: 'absolute',
    top: 6,
    bottom: 6,
    width: 56,
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 4,
    elevation: 4,
  },
  detailMarkerLine: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    width: 3,
    borderRadius: 2,
  },
  peakGuide: {
    position: 'absolute',
    top: 6,
    bottom: 6,
    width: 2,
    backgroundColor: '#545454',
    opacity: 0.8,
  },
  windowMeta: { color: '#7a7a7a', fontSize: 10, marginTop: 6, lineHeight: 14 },
  rateRow: { flexDirection: 'row', gap: 6, flexWrap: 'wrap', marginBottom: 4 },
  rateBtn: {
    borderRadius: 9,
    borderWidth: 1,
    borderColor: '#282828',
    backgroundColor: '#171717',
    paddingVertical: 6,
    paddingHorizontal: 10,
  },
  rateBtnActive: {
    borderColor: '#2ecc71',
    backgroundColor: '#0d2d1a',
  },
  rateTxt: { color: '#bdbdbd', fontSize: 11, fontWeight: '700' },
  rateTxtActive: { color: '#2ecc71' },
  buttonRow: { flexDirection: 'row', gap: 6, flexWrap: 'wrap', marginTop: 4 },
  controlBtn: {
    borderRadius: 10,
    backgroundColor: '#1a1a1a',
    paddingVertical: 7,
    paddingHorizontal: 9,
  },
  primaryControlBtn: { backgroundColor: '#0d2d1a' },
  controlTxt: { color: '#e0e0e0', fontSize: 11, fontWeight: '700' },
  primaryControlTxt: { color: '#2ecc71', fontSize: 11, fontWeight: '800' },
  statusTxt: { color: '#8a8a8a', fontSize: 10, marginTop: 4 },
  markerTitle: { color: '#fff', fontSize: 14, fontWeight: '800' },
  markerMeta: { color: '#8a8a8a', fontSize: 10, lineHeight: 13, marginTop: 2 },
  countMeta: { color: '#6f6f6f', fontSize: 10, lineHeight: 13, marginTop: 4 },
  labelRow: { flexDirection: 'row', gap: 6, flexWrap: 'wrap', marginTop: 4 },
  labelBtn: {
    borderRadius: 9,
    borderWidth: 1,
    borderColor: '#262626',
    paddingVertical: 7,
    paddingHorizontal: 9,
    backgroundColor: '#0d0d0d',
  },
  labelBtnTxt: { color: '#d0d0d0', fontWeight: '700', fontSize: 11 },
  confirmBtn: { backgroundColor: '#102616' },
  confirmTxt: { color: '#7ee39e', fontSize: 11, fontWeight: '700' },
  snapBtn: { backgroundColor: '#17253c' },
  snapTxt: { color: '#a8c2ff', fontSize: 11, fontWeight: '700' },
  deleteBtn: { backgroundColor: '#2d0d0d' },
  deleteTxt: { color: '#ff9f9f', fontSize: 11, fontWeight: '700' },
  disabledTxt: { color: '#666' },
  emptyTxt: { color: '#777', fontSize: 11, lineHeight: 15 },
  helpRow: {
    paddingHorizontal: 4,
    paddingVertical: 6,
  },
  helpRowText: {
    color: '#757575',
    fontSize: 10,
    lineHeight: 13,
  },
  footerActions: {
    flexDirection: 'row',
    gap: 10,
    marginTop: 4,
  },
  footerBtn: {
    flex: 1,
    borderRadius: 13,
    paddingVertical: 13,
    alignItems: 'center',
  },
  saveBtn: { backgroundColor: '#0d2d0d' },
  saveBtnDisabled: { backgroundColor: '#161616' },
  saveBtnTxt: { color: '#2ecc71', fontSize: 14, fontWeight: '800' },
  saveBtnTxtDisabled: { color: '#777' },
  discardBtn: { backgroundColor: '#2d0d0d' },
  discardBtnTxt: { color: '#ff7f7f', fontSize: 14, fontWeight: '800' },
});
