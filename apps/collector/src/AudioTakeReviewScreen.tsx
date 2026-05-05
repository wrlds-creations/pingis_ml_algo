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
  buildSuggestedReviewMarkers,
  buildWaveformBins,
  createManualMarker,
  decodeWavFile,
  detectAudioSyncPoint,
  snapMarkerToAttack,
  writePreviewClip,
  writeTakePlaybackClip,
  type AudioSyncPoint,
} from './audioReview';
import { ReviewOrientation } from './ReviewOrientation';
import type {
  AudioContactKind,
  AudioEvent,
  AudioNotRacketKind,
  AudioReviewBounceSide,
  AudioReviewClassLabel,
  AudioReviewEventType,
  AudioReviewLabel,
  AudioReviewMarker,
  ImuSample,
} from './types';

interface Props {
  event: AudioEvent;
  filePath: string;
  videoFilePath?: string;
  onSave: (markers: AudioReviewMarker[], videoSyncOffsetMs?: number) => Promise<void> | void;
  onDiscard: () => Promise<void> | void;
  onBack: () => void;
}

type PlaybackMode = 'idle' | 'playing_full_take' | 'playing_preview' | 'paused_full_take';
type PlaybackRate = 1 | 0.5 | 0.25;
type TimelineZoomLevel = 1 | 2 | 4 | 8 | 12 | 16;
type TimelineInteractionMode = 'idle' | 'playing' | 'scrubbing' | 'autoScrollingWhileScrubbing';
type PlayingConfidenceFilterId = 'all' | 'medium' | 'safe';
type ImuSeriesKey = 'accel_x' | 'accel_y' | 'accel_z' | 'gyro_x' | 'gyro_y' | 'gyro_z';

const REVIEW_UI_REVISION = 'Simple Review UI | attack_start | r12-synced-timeline';
const NUDGE_STEP_MS = 10;
const LARGE_NUDGE_STEP_MS = 20;
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
}

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

interface ReviewLabelChoice {
  id: string;
  title: string;
  final_label: AudioReviewLabel;
  event_type: AudioReviewEventType;
  class_label: AudioReviewClassLabel;
  contact_kind?: AudioContactKind;
  not_racket_kind?: AudioNotRacketKind;
  bounce_side?: AudioReviewBounceSide;
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

const IMU_REVIEW_LABEL_CHOICES: ReviewLabelChoice[] = [
  {
    id: 'forehand',
    title: 'Racketträff forehand',
    final_label: 'racket_contact',
    event_type: 'racket_hit',
    class_label: 'forehand',
    contact_kind: 'racket_bounce',
    bounce_side: 'forehand',
    color: '#2ecc71',
  },
  {
    id: 'backhand',
    title: 'Racketträff backhand',
    final_label: 'racket_contact',
    event_type: 'racket_hit',
    class_label: 'backhand',
    contact_kind: 'racket_bounce',
    bounce_side: 'backhand',
    color: '#5fd18b',
  },
];

const PLAYING_REVIEW_LABEL_CHOICES: ReviewLabelChoice[] = [
  {
    id: 'forehand_hit',
    title: 'Racketträff forehand',
    final_label: 'racket_contact',
    event_type: 'racket_hit',
    class_label: 'forehand_hit',
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

function markerMatchesChoice(marker: AudioReviewMarker | null, choice: ReviewLabelChoice): boolean {
  if (!marker) return false;
  if (marker.final_label !== choice.final_label) return false;
  if (choice.final_label === 'racket_contact') {
    return (marker.class_label ?? marker.contact_kind ?? 'racket_bounce') === choice.class_label;
  }
  if (choice.final_label === 'not_racket_contact') {
    return (marker.class_label ?? marker.not_racket_kind ?? 'other_impact') === choice.class_label;
  }
  return choice.final_label === 'ignore';
}

function markerDetailText(marker: AudioReviewMarker | null): string {
  if (!marker) return 'Ingen marker vald';
  const classLabel = marker.class_label ?? marker.contact_kind ?? marker.not_racket_kind;
  if (classLabel === 'forehand_hit') return 'Racketträff forehand';
  if (classLabel === 'backhand_hit') return 'Racketträff backhand';
  if (classLabel === 'forehand') return 'Racketträff forehand';
  if (classLabel === 'backhand') return 'Racketträff backhand';
  if (classLabel === 'racket_bounce') return 'Racketträff';
  if (classLabel === 'table_bounce') return 'Bordsstuds';
  if (classLabel === 'floor_bounce') return 'Golvstuds';
  if (classLabel === 'voice_music_noise') return 'Brus';
  if (classLabel === 'catch_after_sound') return 'Fång/efterljud';
  if (classLabel === 'other_impact') return 'Annat ljud';
  if (classLabel === 'ignore' || marker.final_label === 'ignore') return 'Ignorerad';
  return labelText(marker.final_label);
}

function markerConfidence(marker: AudioReviewMarker): number {
  if (marker.final_label === 'not_racket_contact' && marker.class_label === 'table_bounce') {
    return marker.surface_confidence ?? marker.contact_confidence ?? 0;
  }
  return marker.contact_confidence ?? marker.surface_confidence ?? 0;
}

function shouldAlwaysShowMarker(marker: AudioReviewMarker): boolean {
  const status = marker.review_status ?? 'pending';
  return (
    marker.source === 'manual' ||
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

function shouldDropPlayingAutoMarkerOnSave(
  marker: AudioReviewMarker,
  minConfidence: number,
): boolean {
  if (marker.source !== 'auto') return false;
  const status = marker.review_status ?? 'pending';
  if (status === 'filtered') return true;
  return status === 'pending' && !markerPassesPlayingFilter(marker, minConfidence);
}

function applyReviewLabelChoice(marker: AudioReviewMarker, choice: ReviewLabelChoice): AudioReviewMarker {
  const wasSuggested = (
    marker.suggested_label === choice.final_label &&
    marker.final_label === choice.final_label &&
    marker.class_label === choice.class_label
  );
  return {
    ...marker,
    final_label: choice.final_label,
    event_type: choice.event_type,
    class_label: choice.class_label,
    contact_kind: choice.contact_kind,
    not_racket_kind: choice.not_racket_kind,
    bounce_side: choice.bounce_side,
    review_status: choice.final_label === 'ignore'
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
                { backgroundColor: selected ? labelColor(marker.final_label) : '#f5c76d' },
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
  const createMarkerFromPlayheadRef = useRef<() => void>(() => {});

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [durationMs, setDurationMs] = useState(event.duration_ms);
  const [overviewBins, setOverviewBins] = useState<number[]>([]);
  const [markers, setMarkers] = useState<AudioReviewMarker[]>(event.review?.markers ?? []);
  const [selectedMarkerId, setSelectedMarkerId] = useState<string | null>(event.review?.markers?.[0]?.id ?? null);
  const [playbackPositionMs, setPlaybackPositionMs] = useState(0);
  const [playbackMode, setPlaybackMode] = useState<PlaybackMode>('idle');
  const [playbackRate, setPlaybackRate] = useState<PlaybackRate>(1);
  const [timelineMode, setTimelineMode] = useState<TimelineInteractionMode>('idle');
  const [videoSyncOffsetMs, setVideoSyncOffsetMs] = useState(event.video_recording?.video_sync_offset_ms ?? 0);
  const [audioSyncPoint, setAudioSyncPoint] = useState<AudioSyncPoint | null>(null);
  const [syncCandidateVideoMs, setSyncCandidateVideoMs] = useState<number | null>(null);
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
    () => Math.max(64, Math.floor((overviewWidth || 240) / 2)),
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
  const isPlayingReview = event.scenario === 'playing' || event.scenario_id === 'free_recording';
  const isAudioOnlyReview = event.scenario === 'audio_sound' || event.recording_mode === 'guided_audio_only';
  const quickLabelChoices = useMemo(() => {
    if (isPlayingReview) return PLAYING_REVIEW_LABEL_CHOICES;
    if (isAudioOnlyReview) return BASE_REVIEW_LABEL_CHOICES;
    return [];
  }, [isAudioOnlyReview, isPlayingReview]);
  const supportsQuickLabels = quickLabelChoices.length > 0;
  const activePlayingConfidenceFilter = useMemo(
    () => PLAYING_CONFIDENCE_FILTERS.find(filter => filter.id === playingConfidenceFilter) ?? PLAYING_CONFIDENCE_FILTERS[2],
    [playingConfidenceFilter],
  );
  const orderedMarkers = useMemo(() => {
    const nonDeletedMarkers = allOrderedMarkers.filter(marker => {
      const status = marker.review_status ?? 'pending';
      return status !== 'deleted' && status !== 'filtered';
    });
    if (!isPlayingReview) return nonDeletedMarkers;
    return nonDeletedMarkers.filter(marker => markerPassesPlayingFilter(
      marker,
      activePlayingConfidenceFilter.minConfidence,
    ));
  }, [activePlayingConfidenceFilter.minConfidence, allOrderedMarkers, isPlayingReview]);
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
  const selectedMarkerIndex = useMemo(
    () => selectedMarker ? orderedMarkers.findIndex(marker => marker.id === selectedMarker.id) : -1,
    [orderedMarkers, selectedMarker],
  );
  const reviewLabelChoices = useMemo(
    () => isPlayingReview
      ? PLAYING_REVIEW_LABEL_CHOICES
      : BASE_REVIEW_LABEL_CHOICES,
    [isPlayingReview],
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
      void stopCurrentPlayback('paused_full_take');
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

  const handleJumpToMarker = useCallback((direction: -1 | 1) => {
    if (selectedMarkerIndex < 0) return;
    const nextIndex = selectedMarkerIndex + direction;
    if (nextIndex < 0 || nextIndex >= orderedMarkers.length) return;
    handleSelectMarker(orderedMarkers[nextIndex]);
  }, [handleSelectMarker, orderedMarkers, selectedMarkerIndex]);

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

  const startPlayheadLongPress = useCallback((pageX: number) => {
    clearPlayheadLongPress();
    playheadLongPressStartPageXRef.current = pageX;
    if (!supportsQuickLabels) return;
    if (playheadLongPressHandledRef.current) return;
    playheadLongPressTimerRef.current = setTimeout(() => {
      playheadLongPressTimerRef.current = null;
      playheadLongPressHandledRef.current = true;
      createMarkerFromPlayheadRef.current();
    }, PLAYHEAD_LONG_PRESS_MS);
  }, [clearPlayheadLongPress, supportsQuickLabels]);

  const overviewPlayheadResponder = useMemo(() => PanResponder.create({
    onStartShouldSetPanResponder: () => true,
    onMoveShouldSetPanResponder: (_, gestureState) => (
      Math.abs(gestureState.dx) > TIMELINE_DRAG_THRESHOLD_PX ||
      Math.abs(gestureState.dy) > TIMELINE_DRAG_THRESHOLD_PX
    ),
    onPanResponderGrant: eventData => {
      playheadLongPressHandledRef.current = false;
      startPlayheadLongPress(eventData.nativeEvent.pageX);
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
        startPlayheadLongPress(pageX);
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
    if (orderedMarkers.length === 0) {
      setSelectedMarkerId(null);
      return;
    }
    if (!selectedMarkerId || !orderedMarkers.some(marker => marker.id === selectedMarkerId)) {
      setSelectedMarkerId(orderedMarkers[0].id);
    }
  }, [orderedMarkers, selectedMarkerId]);

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
    setLoading(true);

    decodeWavFile(filePath)
      .then(decoded => {
        if (cancelled) return;
        decodedRef.current = { sampleRate: decoded.sampleRate, samples: decoded.samples };
        setDurationMs(decoded.durationMs);
        setOverviewBins(buildWaveformBins(decoded.samples, overviewBinCount));
        const syncPoint = detectAudioSyncPoint(decoded.samples, decoded.sampleRate);
        const savedVideoSyncOffsetMs = event.video_recording?.video_sync_offset_ms ?? 0;

        const nextMarkers = event.review?.markers?.length
          ? sortMarkers(event.review.markers.map(marker => ({
              ...marker,
              review_status: marker.review_status ?? 'confirmed',
            })))
          : buildSuggestedReviewMarkers(decoded.samples, decoded.sampleRate, event.scenario_id);
        setMarkers(nextMarkers);
        setSelectedMarkerId(nextMarkers[0]?.id ?? null);
        playbackPositionRef.current = 0;
        setPlaybackPositionMs(0);
        setTimelineWindowStartMs(0);
        setVideoSyncOffsetMs(savedVideoSyncOffsetMs);
        setVideoSyncExpanded(!savedVideoSyncOffsetMs);
        setAudioSyncPoint(syncPoint);
        setSyncCandidateVideoMs(syncPoint && event.video_recording
          ? mapAudioPlayheadToVideoMs(event, syncPoint.timestamp_ms, decoded.durationMs, savedVideoSyncOffsetMs)
          : null);
        setSyncCalibrationMode(false);
      })
      .catch(error => {
        Alert.alert('Granskningsfel', `Kunde inte läsa tagningen: ${String(error)}`);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [event, event.review?.markers, event.scenario_id, filePath, overviewBinCount, setSyncCalibrationMode]);

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

  const createReviewMarkerAtTimestamp = useCallback((timestampMs: number) => {
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
  }, [durationMs, event.scenario_id, supportsQuickLabels]);

  const handleCreateMarkerAtTimestamp = useCallback((timestampMs: number) => {
    const marker = createReviewMarkerAtTimestamp(timestampMs);
    stopForScrubIfNeeded();
    setMarkers(prev => sortMarkers([...prev, marker]));
    setSelectedMarkerId(marker.id);
    setPlayheadMs(marker.timestamp_ms);
    setQuickLabelPrompt(supportsQuickLabels ? { markerId: marker.id, timestampMs: marker.timestamp_ms } : null);
  }, [createReviewMarkerAtTimestamp, setPlayheadMs, stopForScrubIfNeeded, supportsQuickLabels]);

  useEffect(() => {
    createMarkerFromPlayheadRef.current = () => {
      handleCreateMarkerAtTimestamp(playbackPositionRef.current);
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

  const handlePlayFromHere = useCallback(async () => {
    if (!decodedRef.current) return;

    try {
      await stopCurrentPlayback('idle');
      setVideoPreparingPlayback(hasVideo && (!videoReady || videoBuffering));
      if (hasVideo) {
        videoRef.current?.seek(mapAudioPlayheadToVideoMs(event, playbackPositionMs, durationMs, videoSyncOffsetMs) / 1000);
      }
      const playOriginalTake = playbackRate === 1;
      const playbackPath = playOriginalTake
        ? filePath
        : await writeTakePlaybackClip(
            decodedRef.current.samples,
            decodedRef.current.sampleRate,
            playbackPositionMs,
            playbackRate,
          );
      tempPlaybackPathRef.current = playOriginalTake ? null : playbackPath;
      playbackSourceStartMsRef.current = playOriginalTake ? 0 : playbackPositionMs;
      playbackModeRef.current = 'playing_full_take';
      setPlaybackMode('playing_full_take');
      setTimelineInteractionMode('playing');
      await playerRef.current.startPlayer(playbackPath);
      if (playOriginalTake && playbackPositionMs > 0) {
        await playerRef.current.seekToPlayer(playbackPositionMs);
      }
      setVideoPreparingPlayback(false);
    } catch (error) {
      setVideoPreparingPlayback(false);
      await stopCurrentPlayback('idle');
      Alert.alert('Uppspelningsfel', `Kunde inte spela från vald position: ${String(error)}`);
    }
  }, [
    durationMs,
    event,
    filePath,
    hasVideo,
    playbackPositionMs,
    playbackRate,
    setTimelineInteractionMode,
    stopCurrentPlayback,
    videoBuffering,
    videoReady,
    videoSyncOffsetMs,
  ]);

  const handlePause = useCallback(async () => {
    const nextMode = playbackModeRef.current === 'playing_full_take' ? 'paused_full_take' : 'idle';
    setVideoPreparingPlayback(false);
    await stopCurrentPlayback(nextMode);
    setTimelineInteractionMode('idle');
  }, [setTimelineInteractionMode, stopCurrentPlayback]);

  const handleToggleVideoPlayback = useCallback(() => {
    if (
      playbackModeRef.current === 'playing_full_take' ||
      playbackModeRef.current === 'playing_preview'
    ) {
      void handlePause();
    } else {
      void handlePlayFromHere();
    }
  }, [handlePause, handlePlayFromHere]);

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
    setPlaybackRate(nextRate);
    playbackRateRef.current = nextRate;
    if (playbackModeRef.current === 'playing_full_take' || playbackModeRef.current === 'playing_preview') {
      void handlePause();
    }
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

  const handleQuickLabelChoice = useCallback((choice: ReviewLabelChoice) => {
    const prompt = quickLabelPrompt;
    if (!prompt) return;
    setMarkers(prev => sortMarkers(prev.map(marker => (
      marker.id === prompt.markerId
        ? applyReviewLabelChoice(marker, choice)
        : marker
    ))));
    setSelectedMarkerId(prompt.markerId);
    setPlayheadMs(prompt.timestampMs);
    setQuickLabelPrompt(null);
  }, [quickLabelPrompt, setPlayheadMs]);

  const approvableAutoMarkers = useMemo(() => orderedMarkers.filter(marker => (
    !isPlayingReview &&
    marker.source === 'auto' &&
    (marker.review_status ?? 'pending') === 'pending' &&
    marker.final_label === marker.suggested_label &&
    marker.final_label !== 'ignore'
  )), [isPlayingReview, orderedMarkers]);

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
                ? { ...marker, review_status: 'confirmed' }
                : marker
            ))));
          },
        },
      ],
    );
  }, [approvableAutoMarkers]);

  const handleDeleteSelectedMarker = useCallback(() => {
    if (!selectedMarker) return;
    const nextMarkers = orderedMarkers.filter(marker => marker.id !== selectedMarker.id);
    setMarkers(prev => sortMarkers(prev.filter(marker => marker.id !== selectedMarker.id)));
    setQuickLabelPrompt(null);
    const nextSelected = nextMarkers[Math.min(selectedMarkerIndex, nextMarkers.length - 1)] ?? null;
    setSelectedMarkerId(nextSelected?.id ?? null);
    if (nextSelected) {
      setPlayheadMs(nextSelected.timestamp_ms);
    }
  }, [orderedMarkers, selectedMarker, selectedMarkerIndex, setPlayheadMs]);

  const handleNudgeMarker = useCallback((deltaMs: number) => {
    if (!selectedMarker) return;
    const nextTimestampMs = clampTimestamp(selectedMarker.timestamp_ms + deltaMs, durationMs);
    updateSelectedMarker(marker => ({
      ...marker,
      timestamp_ms: nextTimestampMs,
      review_status: marker.final_label === 'ignore' ? 'ignored' : 'edited',
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
      review_status: marker.final_label === 'ignore' ? 'ignored' : 'edited',
    }));
    setPlayheadMs(snappedTimestampMs);
  }, [selectedMarker, setPlayheadMs, updateSelectedMarker]);

  const handleSave = useCallback(async () => {
    if (orderedMarkers.length === 0) {
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
      const markersToSave = allOrderedMarkers
        .filter(marker => !(
          isPlayingReview &&
          shouldDropPlayingAutoMarkerOnSave(marker, activePlayingConfidenceFilter.minConfidence)
        ))
        .map(marker => ({
          ...marker,
          timestamp_ms: clampTimestamp(marker.timestamp_ms, durationMs),
        }));
      await onSave(markersToSave.map(marker => ({
        ...marker,
        timestamp_ms: clampTimestamp(marker.timestamp_ms, durationMs),
      })), hasVideo ? videoSyncOffsetMs : undefined);
    } finally {
      setSaving(false);
    }
  }, [
    activePlayingConfidenceFilter.minConfidence,
    allOrderedMarkers,
    durationMs,
    hasVideo,
    isPlayingReview,
    onSave,
    orderedMarkers,
    videoSyncOffsetMs,
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
  const canSave = markerCounts.pending === 0 && orderedMarkers.length > 0;

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

  return (
    <View style={styles.root}>
      <StatusBar hidden barStyle="light-content" backgroundColor="#0d0d0d" />
      {loading && hasVideo && videoFilePath && (
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
      {loading ? (
        <View style={styles.loadingBox}>
          <ActivityIndicator color="#f5c76d" />
          <Text style={styles.loadingTxt}>Förbereder waveform och markers...</Text>
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
                progressUpdateInterval={250}
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
                onPress={handleToggleVideoPlayback}
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
              <TouchableOpacity style={styles.videoPlayBtn} onPress={playbackActive ? handlePause : handlePlayFromHere}>
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

            <View style={styles.markerUtilityRow}>
              <TouchableOpacity style={styles.utilityBtn} onPress={handleSnapSelectedMarker} disabled={!selectedMarker}>
                <Text style={styles.utilityTxt}>Snappa</Text>
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
              <TouchableOpacity style={[styles.utilityBtn, styles.utilityDeleteBtn]} onPress={handleDeleteSelectedMarker} disabled={!selectedMarker}>
                <Text style={styles.utilityDeleteTxt}>Ta bort</Text>
              </TouchableOpacity>
            </View>
          </View>

          <View style={styles.waveformCard}>
            <View style={styles.timelineHeaderRow}>
              <Text style={styles.timelineTitle}>Audio + IMU</Text>
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
              />
              <View style={styles.fullWaveformRow}>
                {timelineBins.map((bin, index) => (
                  <View
                    key={`full-bin-${index}`}
                    style={[
                      styles.fullWaveBar,
                      {
                        height: Math.max(3, bin * 50),
                        opacity: 0.38 + bin * 0.62,
                      },
                    ]}
                  />
                ))}
              </View>
              {orderedMarkers.filter(marker => isTimestampVisible(
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
                    <View style={[styles.fullMarkerPin, { backgroundColor: isSelected ? labelColor(marker.final_label) : '#f5c76d' }]} />
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
                {...overviewPlayheadResponder.panHandlers}
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

            {quickLabelPrompt && supportsQuickLabels && (
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
                  <TouchableOpacity style={[styles.utilityBtn, styles.utilityDeleteBtn]} onPress={handleDeleteSelectedMarker} disabled={!selectedMarker}>
                    <Text style={styles.utilityDeleteTxt}>Ta bort</Text>
                  </TouchableOpacity>
                </View>
              </View>
            )}

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
          </View>

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

            <View style={styles.markerUtilityRow}>
              <TouchableOpacity style={styles.utilityBtn} onPress={handleSnapSelectedMarker} disabled={!selectedMarker}>
                <Text style={styles.utilityTxt}>Snappa</Text>
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
              <TouchableOpacity style={[styles.utilityBtn, styles.utilityDeleteBtn]} onPress={handleDeleteSelectedMarker} disabled={!selectedMarker}>
                <Text style={styles.utilityDeleteTxt}>Ta bort</Text>
              </TouchableOpacity>
            </View>
          </View>

          <View style={styles.footerActions}>
            <TouchableOpacity
              style={[styles.footerBtn, styles.saveBtn, (!canSave || saving) && styles.saveBtnDisabled]}
              onPress={handleSave}
              disabled={saving || !canSave}
            >
              <Text style={[styles.saveBtnTxt, !canSave && styles.saveBtnTxtDisabled]}>
                {saving ? 'Sparar...' : canSave ? 'Spara tagning' : `Hantera ${markerCounts.pending} markrar`}
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
    top: 22,
    width: 24,
    height: 70,
    alignItems: 'center',
    zIndex: 4,
    elevation: 4,
  },
  fullMarkerPin: {
    width: 10,
    height: 10,
    borderRadius: 5,
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
  },
  loadingTxt: { color: '#aaa', fontSize: 13 },
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
