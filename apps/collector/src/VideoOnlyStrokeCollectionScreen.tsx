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
  useWindowDimensions,
  View,
} from 'react-native';
import RNFS from 'react-native-fs';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import Video, { type VideoRef } from 'react-native-video';
import { AudioCapture } from './NativeAudioCapture';
import { VideoSegment } from './NativeVideoSegment';
import { VideoPose } from './NativeVideoPose';
import { buildWaveformBins, decodeWavFile, snapMarkerToAttack, type DecodedWavFile } from './audioReview';
import type {
  PlayerSetup,
  VideoBounceSideSessionFile,
  VideoStrokeCameraFacing,
  VideoStrokeCameraSide,
  VideoStrokeMarker,
  VideoStrokeMarkerType,
  VideoStrokePoseAnalysis,
  VideoStrokeSessionFile,
} from './types';
import { buildVideoStrokeFeatures, detectRacketHandedness, VIDEO_STROKE_FEATURE_SPEC } from './videoStrokeFeatures';
import { detectGateOnsets, extractFableFeatures } from './nrFeatures';
import { fablePredict } from './hgbRuntime';
import { bounceSideFeatures, predictBounceSide } from './bounceSideInference';
import { hasTrainedVideoStrokeModel, predictVideoStroke, videoStrokeModelVersion } from './videoStrokeInference';

const APP_VERSION = 'collector-video-pose-only-v5';
const BOUNCE_SIDE_APP_VERSION = 'collector-video-bounce-side-t0037';
const VIDEO_STROKE_STORAGE_FOLDER = `${RNFS.DownloadDirectoryPath}/pingis_video_stroke_sessions`;
const VIDEO_BOUNCE_SIDE_STORAGE_FOLDER = `${RNFS.DownloadDirectoryPath}/pingis_video_bounce_side_sessions`;
const SAMPLE_FPS = 15;
const SCAN_STEP_MS = 100;
const WINDOW_PRE_MS = 700;
const WINDOW_POST_MS = 500;
const MIN_CONFIDENCE = 0.58;
const MIN_UNKNOWN_CONFIDENCE = 0.72;
const MIN_GAP_MS = 750;
const MIN_FRAME_COUNT = 6;
const MIN_AVG_VISIBILITY = 0.45;
const MIN_WRIST_SPEED = 1.15;
const MIN_WRIST_TRAVEL = 0.25;
const MIN_LATERAL_TRAVEL = 0.1;
const MIN_ELBOW_TRAVEL = 0.04;
const TIMELINE_ZOOM_LEVELS = [1, 2, 4, 8, 12, 16] as const;
const WAVEFORM_MIN_BIN_COUNT = 120;
const WAVEFORM_MAX_BAR_HEIGHT = 92;
const TIMELINE_EDGE_PX = 44;
const TIMELINE_DRAG_THRESHOLD_PX = 4;
const TIMELINE_EDGE_SCROLL_INTERVAL_MS = 55;
const TIMELINE_EDGE_SCROLL_FRACTION = 0.035;
const PLAYHEAD_LONG_PRESS_MS = 2000;
const NUDGE_STEP_MS = 10;
const LARGE_NUDGE_STEP_MS = 20;
const EXTRA_LARGE_NUDGE_STEP_MS = 50;
const BOUNCE_SIDE_SNAPSHOT_PRE_MS = 80;
const BOUNCE_SIDE_SNAPSHOT_POST_MS = 80;
const BOUNCE_SIDE_PEAK_FRAME_MS = 10;
const BOUNCE_SIDE_PEAK_MIN_GAP_MS = 180;
const BOUNCE_SIDE_PEAK_MAX_MARKERS = 260;

type VideoReviewMode = 'stroke' | 'bounce_side';

interface ImportedVideoState {
  sessionId: string;
  sessionDir: string;
  sessionJsonPath: string;
  videoFilename: string;
  videoPath: string;
  durationMs: number;
  importedSourceFilename?: string;
  importedSourceUri?: string;
  importedAt: string;
  rotation?: number;
  sizeBytes?: number;
  waveformAudioFilename?: string;
  waveformAudioPath?: string;
  waveformDurationMs?: number;
}

type ReviewMarker = VideoStrokeMarker & {
  deleted?: boolean;
  anchor_source?: 'audio_peak';
  audio_peak_score?: number;
  snapshot_window_ms?: {
    pre_ms: number;
    post_ms: number;
  };
};

type TimelineWaveform = DecodedWavFile & {
  filename: string;
  path: string;
};

interface QuickLabelPrompt {
  markerId: string;
  timestampMs: number;
}

interface AnalyzedStrokeCandidate {
  marker: ReviewMarker;
  analysis: VideoStrokePoseAnalysis;
  wristSpeedMax: number;
  score: number;
}

interface Props {
  setup: PlayerSetup;
  mode?: VideoReviewMode;
  onDone: () => void;
}

function dateStamp(): string {
  return new Date().toISOString().slice(0, 10);
}

function pad3(value: number): string {
  return String(value).padStart(3, '0');
}

function formatMs(ms: number): string {
  const safeMs = Math.max(0, Math.round(ms));
  const seconds = Math.floor(safeMs / 1000);
  const hundredths = Math.floor((safeMs % 1000) / 10);
  return `${seconds}.${String(hundredths).padStart(2, '0')}s`;
}

function clampTimestamp(value: number, durationMs: number): number {
  return Math.max(0, Math.min(Math.max(0, durationMs), Math.round(value)));
}

function markerLabel(label: VideoStrokeMarkerType, mode: VideoReviewMode = 'stroke'): string {
  if (mode === 'bounce_side') {
    if (label === 'forehand') return 'FH-sida';
    if (label === 'backhand') return 'BH-sida';
    return 'Oklart';
  }
  if (label === 'forehand') return 'Forehand';
  if (label === 'backhand') return 'Backhand';
  return 'Oklart';
}

function markerColor(label: VideoStrokeMarkerType): string {
  if (label === 'forehand') return '#35c7ff';
  if (label === 'backhand') return '#7db7ff';
  return '#888';
}

function ratioToLeft(timestampMs: number, startMs: number, endMs: number, width: number): number {
  if (width <= 0 || endMs <= startMs) return 0;
  const ratio = (timestampMs - startMs) / (endMs - startMs);
  return Math.max(0, Math.min(width, ratio * width));
}

function cameraSideLabel(side: VideoStrokeCameraSide): string {
  if (side === 'player_left') return 'Kamera vänster';
  if (side === 'player_right') return 'Kamera höger';
  if (side === 'center_front') return 'Rakt fram';
  return 'Okänd';
}

function percentile(sortedValues: number[], ratio: number): number {
  if (sortedValues.length === 0) return 0;
  const index = Math.max(0, Math.min(sortedValues.length - 1, Math.floor((sortedValues.length - 1) * ratio)));
  return sortedValues[index];
}

interface AudioPeak {
  timestamp_ms: number;
  rms: number;
  score: number;
}

/** Hitta ljudpeakar (bollträffar) i en avkodad WAV. Delas av studs-läget
 *  (FH-/BH-sida-ankare) och slagläget (ljudankrad poseanalys). */
function findAudioPeaks(
  waveform: DecodedWavFile,
  minGapMs: number,
  maxPeaks: number,
): AudioPeak[] {
  const { samples, sampleRate } = waveform;
  if (samples.length === 0 || sampleRate <= 0) return [];

  const frameSamples = Math.max(32, Math.round((sampleRate * BOUNCE_SIDE_PEAK_FRAME_MS) / 1000));
  const frames: Array<{ timestamp_ms: number; rms: number }> = [];
  for (let start = 0; start + frameSamples <= samples.length; start += frameSamples) {
    let sum = 0;
    for (let sampleIndex = start; sampleIndex < start + frameSamples; sampleIndex += 1) {
      sum += samples[sampleIndex] * samples[sampleIndex];
    }
    const rms = Math.sqrt(sum / frameSamples);
    frames.push({
      timestamp_ms: Math.round(((start + frameSamples / 2) / sampleRate) * 1000),
      rms,
    });
  }

  if (frames.length === 0) return [];
  const rmsValues = frames.map(frame => frame.rms).sort((left, right) => left - right);
  const maxRms = rmsValues[rmsValues.length - 1] ?? 0;
  if (maxRms <= 0) return [];
  const medianRms = percentile(rmsValues, 0.5);
  const p75Rms = percentile(rmsValues, 0.75);
  const threshold = Math.max(0.004, medianRms * 4.0, p75Rms * 1.8, maxRms * 0.16);
  const localRadius = 2;
  const candidates: Array<{ timestamp_ms: number; rms: number }> = [];

  for (let frameIndex = localRadius; frameIndex < frames.length - localRadius; frameIndex += 1) {
    const frame = frames[frameIndex];
    if (frame.rms < threshold) continue;
    let isLocalMax = true;
    for (let offset = -localRadius; offset <= localRadius; offset += 1) {
      if (offset !== 0 && frames[frameIndex + offset].rms > frame.rms) {
        isLocalMax = false;
        break;
      }
    }
    if (isLocalMax) candidates.push(frame);
  }

  const picked: Array<{ timestamp_ms: number; rms: number }> = [];
  for (const candidate of candidates.sort((left, right) => right.rms - left.rms)) {
    if (picked.length >= maxPeaks) break;
    if (picked.every(existing => Math.abs(existing.timestamp_ms - candidate.timestamp_ms) >= minGapMs)) {
      picked.push(candidate);
    }
  }

  return picked
    .sort((left, right) => left.timestamp_ms - right.timestamp_ms)
    .map(candidate => ({
      timestamp_ms: candidate.timestamp_ms,
      rms: candidate.rms,
      score: Math.round((candidate.rms / maxRms) * 1000) / 1000,
    }));
}

function buildAudioPeakBounceMarkers(waveform: DecodedWavFile, durationMs: number): ReviewMarker[] {
  const picked = findAudioPeaks(waveform, BOUNCE_SIDE_PEAK_MIN_GAP_MS, BOUNCE_SIDE_PEAK_MAX_MARKERS);
  const maxRms = picked.reduce((currentMax, peak) => Math.max(currentMax, peak.rms), 0);
  return picked
    .map((candidate, index) => ({
      id: `audio_peak_bounce_${String(index + 1).padStart(3, '0')}_${candidate.timestamp_ms}`,
      timestamp_ms: clampTimestamp(candidate.timestamp_ms, durationMs),
      stroke_type: 'unknown',
      source: 'audio_peak',
      review_status: 'suggested',
      created_at: new Date().toISOString(),
      anchor_source: 'audio_peak',
      audio_peak_score: maxRms > 0 ? candidate.score : 0,
      snapshot_window_ms: {
        pre_ms: BOUNCE_SIDE_SNAPSHOT_PRE_MS,
        post_ms: BOUNCE_SIDE_SNAPSHOT_POST_MS,
      },
    }));
}

async function nextSessionId(storageFolder: string, sessionPrefix: string): Promise<string> {
  await RNFS.mkdir(storageFolder);
  const prefix = `${sessionPrefix}_${dateStamp()}_`;
  const files = await RNFS.readDir(storageFolder).catch(() => []);
  const maxIndex = files.reduce((currentMax, file) => {
    const match = file.name.match(new RegExp(`^${prefix.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(\\d{3})\\.json$`));
    if (!match) return currentMax;
    return Math.max(currentMax, Number(match[1]) || 0);
  }, 0);
  return `${prefix}${pad3(maxIndex + 1)}`;
}

function passesVideoOnlyMotionGate(featureResult: ReturnType<typeof buildVideoStrokeFeatures>): boolean {
  if (!featureResult) return false;
  const features = featureResult.features;
  const wristTravel = Math.hypot(features.wrist_x_ptp, features.wrist_y_ptp);
  const elbowTravel = Math.hypot(features.elbow_x_ptp, features.elbow_y_ptp);
  const lateralTravel = Math.max(Math.abs(features.wrist_x_delta), features.wrist_x_ptp);
  return (
    featureResult.frame_count >= MIN_FRAME_COUNT &&
    featureResult.avg_visibility >= MIN_AVG_VISIBILITY &&
    features.wrist_speed_max >= MIN_WRIST_SPEED &&
    wristTravel >= MIN_WRIST_TRAVEL &&
    lateralTravel >= MIN_LATERAL_TRAVEL &&
    elbowTravel >= MIN_ELBOW_TRAVEL
  );
}

// Ljudankrat slagläge: varje riktigt slag träffar en boll = en ljudpeak.
// Blindskanning både missade slag (28 av 60+ i Tomas-testet) och hittade
// på slag i tysta ögonblick; ankare gör båda felklasserna strukturellt
// omöjliga (tyst ögonblick = inget ankare = inget förslag).
const STROKE_ANCHOR_MIN_GAP_MS = 350;
const STROKE_ANCHOR_MAX_PEAKS = 220;
const STROKE_ANCHOR_DEDUPE_MS = 600;
const STROKE_ANCHOR_MIN_PEAKS = 3;
// Posefönster runt varje ankare: slagfönstret är -700/+500 ms, plus marginal.
const STROKE_ANCHOR_POSE_PRE_MS = 800;
const STROKE_ANCHOR_POSE_POST_MS = 600;
// Ankrat läge behöver lägre konfidens än blindskanningen: ankaret i sig är
// stark evidens (bollträff) och spök-FP:er kan inte uppstå i tysta partier.
// Vid riktiga bollträff-ankare på Tomas-videon klarade bara 21/75 slag 0.58
// medan 61/75 var korrekt klassade FH/BH; 0.45 gav 41 på mobilen (ML Kit-
// pose ger systematiskt lägre konfidens än träningens MediaPipe-pose).
// 0.38 är strax över 3-klassers slumpnivå - fel kan bara bli fel SIDA på
// ett riktigt slag, och förslagen granskas alltid manuellt.
const STROKE_ANCHOR_MIN_CONFIDENCE = 0.38;

export interface StrokeAnalysisDiagnostics {
  mode: 'audio_anchored' | 'scan';
  anchors: number;
  pose_frames: number;
  detected_hand: string;
  drops: {
    no_features: number;
    low_visibility: number;
    not_strokeish: number;
    deduped: number;
  };
  /** Ankare som föll på slag-aktighetsregeln, med sannolikheter - facit för
   *  nästa tröskel-/modelliteration. */
  not_strokeish_detail: Array<{ ts_ms: number; p_fh: number; p_bh: number; p_unknown: number }>;
}

async function analyzeVideoOnlyStrokes(
  videoPath: string,
  durationMs: number,
  handedness: PlayerSetup['handedness'],
  waveform: DecodedWavFile | null,
): Promise<{
  markers: ReviewMarker[];
  poseAnalysis: VideoStrokePoseAnalysis[];
  status: string;
  diagnostics: StrokeAnalysisDiagnostics | null;
}> {
  if (!hasTrainedVideoStrokeModel()) {
    return {
      markers: [],
      poseAnalysis: [],
      status: `Ingen videomodell exporterad (${videoStrokeModelVersion()}).`,
      diagnostics: null,
    };
  }

  // Ankare FÖRE pose: med ljudankare behöver pose bara köras i fönster
  // kring bollträffarna (stor hastighetsvinst i nativemodulen).
  // Adaptiv gate i stället för global tröskel: en enda stark smäll i
  // klippet får inte dränka de vanliga bollträffarna.
  const anchors = waveform
    ? detectGateOnsets(waveform.samples, waveform.sampleRate, STROKE_ANCHOR_MIN_GAP_MS)
        .slice(0, STROKE_ANCHOR_MAX_PEAKS)
    : [];
  const anchored = anchors.length >= STROKE_ANCHOR_MIN_PEAKS;
  const pose = anchored
    ? await VideoPose.extractPoseInWindows(
        videoPath,
        SAMPLE_FPS,
        anchors.flatMap(anchor => [
          Math.max(0, anchor.timestamp_ms - STROKE_ANCHOR_POSE_PRE_MS),
          anchor.timestamp_ms + STROKE_ANCHOR_POSE_POST_MS,
        ]),
      )
    : await VideoPose.extractPose(videoPath, SAMPLE_FPS);
  const scanDurationMs = Math.max(0, Math.round(pose.duration_ms || durationMs));
  const candidates: AnalyzedStrokeCandidate[] = [];

  // Spelhanden i videon kan skilja sig från profilens hänthet (importerad
  // video av annan spelare). Racketarmen detekteras ur rörelsen; fel hand
  // ger fel arm + spegelvänd x-axel och systematiskt fel FH/BH.
  const handDetection = detectRacketHandedness(pose.frames, handedness);
  const playerHandedness = handDetection.handedness;
  const handLabel = playerHandedness === 'right' ? 'höger' : 'vänster';
  const handSource = handDetection.source === 'auto' ? 'auto' : 'profil';

  if (anchored) {
    interface AnchoredCandidate {
      timestampMs: number;
      strokeType: 'forehand' | 'backhand';
      prediction: ReturnType<typeof predictVideoStroke>;
      wristSpeedMax: number;
    }
    const diagnostics: StrokeAnalysisDiagnostics = {
      mode: 'audio_anchored',
      anchors: anchors.length,
      pose_frames: pose.frames.length,
      detected_hand: `${handLabel} (${handSource})`,
      drops: { no_features: 0, low_visibility: 0, not_strokeish: 0, deduped: 0 },
      not_strokeish_detail: [],
    };
    const anchoredCandidates: AnchoredCandidate[] = [];
    for (const anchor of anchors) {
      const timestampMs = clampTimestamp(anchor.timestamp_ms, scanDurationMs || durationMs);
      const featureResult = buildVideoStrokeFeatures(pose.frames, timestampMs, playerHandedness);
      // Mjukare gate än blindskanningen: ankaret är redan stark evidens och
      // modellens unknown-klass avvisar bordsstuds-ankare utan sving.
      if (!featureResult) {
        diagnostics.drops.no_features += 1;
        continue;
      }
      if (featureResult.avg_visibility < MIN_AVG_VISIBILITY) {
        diagnostics.drops.low_visibility += 1;
        continue;
      }
      const prediction = predictVideoStroke(featureResult.features);
      // Slag-aktighet i stället för argmax: mobilens pose (ML Kit) ger
      // flackare sannolikheter än träningens (MediaPipe), så "unknown" kan
      // vinna knappt på riktiga slag. P(FH)+P(BH) >= 0.5 = det är ett slag;
      // sidan avgörs av den större. Bordsstuds-ankare utan sving har
      // unknown-sannolikhet klart över 0.5 och avvisas fortfarande.
      // Tröskel 0.40 från device-diagnostik 2026-06-11 (76 ankare): äkta
      // slag som föll bort hade P(FH)+P(BH) 0.40-0.48 i tydlig slagrytm,
      // medan bordsstuds-ankare låg 0.25-0.36 (unknown 0.65-0.75).
      const pForehand = prediction.probabilities.forehand ?? 0;
      const pBackhand = prediction.probabilities.backhand ?? 0;
      if (pForehand + pBackhand < 0.4) {
        diagnostics.drops.not_strokeish += 1;
        diagnostics.not_strokeish_detail.push({
          ts_ms: Math.round(timestampMs),
          p_fh: Math.round(pForehand * 1000) / 1000,
          p_bh: Math.round(pBackhand * 1000) / 1000,
          p_unknown: Math.round((prediction.probabilities.unknown ?? 0) * 1000) / 1000,
        });
        continue;
      }
      const side: 'forehand' | 'backhand' = pForehand >= pBackhand ? 'forehand' : 'backhand';
      anchoredCandidates.push({
        timestampMs,
        strokeType: side,
        prediction: { ...prediction, confidence: Math.max(pForehand, pBackhand) },
        wristSpeedMax: featureResult.features.wrist_speed_max,
      });
    }
    // En boll kan ge två peakar nära varandra (slaget + bordsstudsen) vars
    // posefönster överlappar samma sving: behåll den säkraste per slag.
    const deduped: AnchoredCandidate[] = [];
    for (const candidate of anchoredCandidates.sort((a, b) => b.prediction.confidence - a.prediction.confidence)) {
      if (deduped.every(existing =>
        existing.strokeType !== candidate.strokeType ||
        Math.abs(existing.timestampMs - candidate.timestampMs) >= STROKE_ANCHOR_DEDUPE_MS
      )) {
        deduped.push(candidate);
      }
    }
    deduped.sort((a, b) => a.timestampMs - b.timestampMs);
    const markers: ReviewMarker[] = [];
    const poseAnalysis: VideoStrokePoseAnalysis[] = [];
    for (const candidate of deduped) {
      const markerId = `auto_pose_${Math.round(candidate.timestampMs)}`;
      markers.push({
        id: markerId,
        timestamp_ms: Math.round(candidate.timestampMs),
        stroke_type: candidate.strokeType,
        source: 'model',
        review_status: 'suggested',
        created_at: new Date().toISOString(),
      });
      poseAnalysis.push({
        marker_id: markerId,
        timestamp_ms: Math.round(candidate.timestampMs),
        predicted_stroke_type: candidate.strokeType,
        confidence: candidate.prediction.confidence,
        probabilities: candidate.prediction.probabilities,
        model_version: candidate.prediction.model_version,
        feature_spec: VIDEO_STROKE_FEATURE_SPEC,
        status: 'ok',
      });
    }
    diagnostics.drops.deduped = anchoredCandidates.length - deduped.length;
    const fhCount = markers.filter(marker => marker.stroke_type === 'forehand').length;
    const bhCount = markers.filter(marker => marker.stroke_type === 'backhand').length;
    const dropSummary = `bortfall: ${diagnostics.drops.not_strokeish} ej slag, ${diagnostics.drops.low_visibility} låg synlighet, ${diagnostics.drops.no_features} få frames, ${diagnostics.drops.deduped} dubbletter`;
    return {
      markers,
      poseAnalysis,
      status: `Ljudankrad pose: ${anchors.length} bollträffar -> ${markers.length} slag (${fhCount} FH, ${bhCount} BH). ${dropSummary}. Spelhand: ${handLabel} (${handSource}).`,
      diagnostics,
    };
  }

  for (
    let timestampMs = WINDOW_PRE_MS;
    timestampMs <= Math.max(WINDOW_PRE_MS, scanDurationMs - WINDOW_POST_MS);
    timestampMs += SCAN_STEP_MS
  ) {
    const featureResult = buildVideoStrokeFeatures(pose.frames, timestampMs, playerHandedness);
    if (!passesVideoOnlyMotionGate(featureResult)) continue;
    const prediction = predictVideoStroke(featureResult!.features);
    const rawLabel = prediction.raw_label ?? prediction.label;
    const wristSpeedMax = featureResult!.features.wrist_speed_max;

    let strokeType: VideoStrokeMarkerType | null = null;
    if (
      prediction.status === 'ok' &&
      (prediction.label === 'forehand' || prediction.label === 'backhand') &&
      prediction.confidence >= MIN_CONFIDENCE
    ) {
      strokeType = prediction.label;
    } else if (rawLabel === 'unknown' && prediction.confidence >= MIN_UNKNOWN_CONFIDENCE) {
      strokeType = 'unknown';
    }
    if (!strokeType) continue;

    const roundedTimestamp = clampTimestamp(timestampMs, scanDurationMs);
    const markerId = `auto_pose_${roundedTimestamp}`;
    const analysis: VideoStrokePoseAnalysis = {
      marker_id: markerId,
      timestamp_ms: roundedTimestamp,
      predicted_stroke_type: strokeType === 'unknown' ? 'uncertain' : strokeType,
      confidence: prediction.confidence,
      probabilities: prediction.probabilities,
      model_version: prediction.model_version,
      feature_spec: VIDEO_STROKE_FEATURE_SPEC,
      status: strokeType === 'unknown' ? 'uncertain' : 'ok',
    };
    candidates.push({
      marker: {
        id: markerId,
        timestamp_ms: roundedTimestamp,
        stroke_type: strokeType,
        source: 'model',
        review_status: 'suggested',
        created_at: new Date().toISOString(),
      },
      analysis,
      wristSpeedMax,
      score: prediction.confidence + Math.min(1, wristSpeedMax / 8) * 0.2 - (strokeType === 'unknown' ? 0.12 : 0),
    });
  }

  const picked: AnalyzedStrokeCandidate[] = [];
  for (const candidate of candidates.sort((left, right) => right.score - left.score)) {
    if (picked.every(existing => Math.abs(existing.marker.timestamp_ms - candidate.marker.timestamp_ms) >= MIN_GAP_MS)) {
      picked.push(candidate);
    }
  }
  const sorted = picked.sort((left, right) => left.marker.timestamp_ms - right.marker.timestamp_ms);
  const markers = sorted.map(candidate => candidate.marker);
  const poseAnalysis = sorted.map(candidate => candidate.analysis);
  const concreteCount = markers.filter(marker => marker.stroke_type !== 'unknown').length;

  return {
    markers,
    poseAnalysis,
    status: `Pose ${SAMPLE_FPS} fps (utan ljudankare): ${markers.length} förslag (${concreteCount} FH/BH). Spelhand: ${handLabel} (${handSource}).`,
    diagnostics: {
      mode: 'scan',
      anchors: 0,
      pose_frames: pose.frames.length,
      detected_hand: `${handLabel} (${handSource})`,
      drops: { no_features: 0, low_visibility: 0, not_strokeish: 0, deduped: 0 },
      not_strokeish_detail: [],
    },
  };
}

export function VideoOnlyStrokeCollectionScreen({ setup, mode = 'stroke', onDone }: Props) {
  const { width: windowWidth, height: windowHeight } = useWindowDimensions();
  const insets = useSafeAreaInsets();
  const isBounceSideMode = mode === 'bounce_side';
  const modeConfig = useMemo(() => (
    isBounceSideMode
      ? {
          appVersion: BOUNCE_SIDE_APP_VERSION,
          storageFolder: VIDEO_BOUNCE_SIDE_STORAGE_FOLDER,
          sessionPrefix: 'video_bounce_side_session',
          title: 'Video studs FH/BH',
          subtitlePrefix: 'Studs FH/BH',
          helpTitle: 'Video studs FH/BH',
          helpText: 'Importera video med ljud. Appen föreslår studsankare från ljudpeakar; märk varje anchor som FH-sida, BH-sida eller Oklart.',
          timelineTitle: 'Studs-side review',
          algorithmTitle: 'Video studs FH/BH',
          saveHint: 'Save tränar bara på bekräftade FH-sida/BH-sida/Oklart-markers. Ljudpeakar används bara som tidsankare.',
        }
      : {
          appVersion: APP_VERSION,
          storageFolder: VIDEO_STROKE_STORAGE_FOLDER,
          sessionPrefix: 'video_stroke_session',
          title: 'Video FH/BH',
          subtitlePrefix: 'Video FH/BH',
          helpTitle: 'Video FH/BH',
          helpText: 'Samma review som ljud, men labels är Forehand, Backhand och Oklart. WAV visas bara som tidsstöd.',
          timelineTitle: 'FH/BH-review',
          algorithmTitle: 'Video FH/BH',
          saveHint: 'Save tränar bara på bekräftade FH/BH/Oklart-markers. WAV används bara för review-timing.',
        }
  ), [isBounceSideMode]);
  const videoRef = useRef<VideoRef>(null);
  const timelineSurfaceRef = useRef<View>(null);
  const strokePreviewTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const strokePreviewEndMsRef = useRef<number | null>(null);
  const markerDragStartMsRef = useRef(0);
  const playbackPositionRef = useRef(0);
  const timelineIsDraggingRef = useRef(false);
  const timelineWindowStartRef = useRef(0);
  const timelineWindowSpanRef = useRef(1);
  const timelineLayoutRef = useRef({ pageX: 0, width: 0 });
  const scrubFingerOffsetPxRef = useRef(0);
  const latestScrubPageXRef = useRef(0);
  const edgeScrollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const edgeScrollDirectionRef = useRef<-1 | 0 | 1>(0);
  const playheadLongPressTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const playheadLongPressHandledRef = useRef(false);
  const playheadLongPressStartPageXRef = useRef(0);
  const [cameraSide, setCameraSide] = useState<VideoStrokeCameraSide>('player_left');
  // Vilken färg spelarens FOREHANDSIDA har. Sidomodellen känner igen
  // färgsidan (tränad på Loves racket: forehand = röd); väljer spelaren
  // 'black' byts förslagens FH/BH-etiketter - ingen omträning behövs.
  const [forehandColor, setForehandColor] = useState<'red' | 'black'>('red');
  const forehandColorRef = useRef<'red' | 'black'>('red');
  forehandColorRef.current = forehandColor;
  const [cameraFacing, setCameraFacing] = useState<VideoStrokeCameraFacing>('front');
  const [video, setVideo] = useState<ImportedVideoState | null>(null);
  const [waveform, setWaveform] = useState<TimelineWaveform | null>(null);
  const [markers, setMarkers] = useState<ReviewMarker[]>([]);
  const [poseAnalysis, setPoseAnalysis] = useState<VideoStrokePoseAnalysis[]>([]);
  const [analysisDiagnostics, setAnalysisDiagnostics] = useState<StrokeAnalysisDiagnostics | null>(null);
  const [selectedMarkerId, setSelectedMarkerId] = useState<string | null>(null);
  const [playbackMs, setPlaybackMs] = useState(0);
  const [paused, setPaused] = useState(true);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [timelineZoom, setTimelineZoom] = useState<(typeof TIMELINE_ZOOM_LEVELS)[number]>(1);
  const [timelineIsDragging, setTimelineIsDragging] = useState(false);
  const [timelineWidth, setTimelineWidth] = useState(1);
  const [timelineWindowStartMs, setTimelineWindowStartMs] = useState(0);
  const [videoProgressWidth, setVideoProgressWidth] = useState(1);
  const [videoNaturalSize, setVideoNaturalSize] = useState<{ width: number; height: number } | null>(null);
  const [quickLabelPrompt, setQuickLabelPrompt] = useState<QuickLabelPrompt | null>(null);
  const [isImporting, setIsImporting] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [status, setStatus] = useState<string | null>(null);

  const activeMarkers = useMemo(
    () => markers.filter(marker => !marker.deleted).sort((left, right) => left.timestamp_ms - right.timestamp_ms),
    [markers],
  );
  const selectedMarker = activeMarkers.find(marker => marker.id === selectedMarkerId) ?? activeMarkers[0] ?? null;
  const selectedMarkerIndex = selectedMarker ? activeMarkers.findIndex(marker => marker.id === selectedMarker.id) : -1;
  const pendingCount = activeMarkers.filter(marker => marker.review_status !== 'confirmed').length;
  const approvableAutoMarkers = isBounceSideMode
    ? []
    : activeMarkers.filter(marker => marker.review_status !== 'confirmed');
  const durationMs = video?.durationMs ?? 0;
  const visibleDurationMs = durationMs > 0 ? Math.max(1000, durationMs / timelineZoom) : 0;
  const visibleWindowStartMs = Math.max(
    0,
    Math.min(timelineWindowStartMs, Math.max(0, durationMs - visibleDurationMs)),
  );
  const visibleWindowEndMs = Math.min(durationMs, visibleWindowStartMs + visibleDurationMs);
  const visibleWindowSpanMs = Math.max(1, visibleWindowEndMs - visibleWindowStartMs);
  const visibleMarkers = activeMarkers.filter(marker =>
    marker.timestamp_ms >= visibleWindowStartMs && marker.timestamp_ms <= visibleWindowEndMs,
  );
  const waveformBins = useMemo(() => {
    if (!waveform || visibleWindowSpanMs <= 0) return [];
    const startSample = Math.max(0, Math.floor((visibleWindowStartMs / 1000) * waveform.sampleRate));
    const endSample = Math.min(
      waveform.samples.length,
      Math.ceil((visibleWindowEndMs / 1000) * waveform.sampleRate),
    );
    if (endSample <= startSample) return [];
    return buildWaveformBins(
      waveform.samples.slice(startSample, endSample),
      Math.max(WAVEFORM_MIN_BIN_COUNT, Math.floor(timelineWidth || 320)),
    );
  }, [timelineWidth, visibleWindowEndMs, visibleWindowSpanMs, visibleWindowStartMs, waveform]);
  const timelinePlayheadLeft = ratioToLeft(
    playbackMs,
    visibleWindowStartMs,
    visibleWindowEndMs,
    timelineWidth,
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
  const reviewedCount = activeMarkers.filter(marker => marker.review_status === 'confirmed').length;
  const autoCount = activeMarkers.filter(marker => marker.source === 'model' || marker.source === 'audio_peak').length;
  const manualCount = activeMarkers.filter(marker => marker.source === 'manual').length;

  const clearStrokePreview = useCallback(() => {
    if (strokePreviewTimerRef.current) {
      clearTimeout(strokePreviewTimerRef.current);
      strokePreviewTimerRef.current = null;
    }
    strokePreviewEndMsRef.current = null;
  }, []);

  const seekTo = useCallback((timestampMs: number) => {
    const clamped = clampTimestamp(timestampMs, durationMs);
    setPlaybackMs(clamped);
    if (!timelineIsDraggingRef.current && durationMs > visibleWindowSpanMs) {
      setTimelineWindowStartMs(currentStart => {
        if (timelineZoom <= 1) return 0;
        const marginMs = visibleWindowSpanMs * 0.18;
        const currentEnd = currentStart + visibleWindowSpanMs;
        if (clamped >= currentStart + marginMs && clamped <= currentEnd - marginMs) {
          return currentStart;
        }
        return Math.max(0, Math.min(durationMs - visibleWindowSpanMs, clamped - visibleWindowSpanMs / 2));
      });
    }
    videoRef.current?.seek(clamped / 1000);
  }, [durationMs, timelineZoom, visibleWindowSpanMs]);

  const timestampFromTimelineX = useCallback((locationX: number) => {
    if (!video || timelineWidth <= 0 || visibleWindowSpanMs <= 0) return playbackMs;
    return clampTimestamp(
      visibleWindowStartMs + (locationX / timelineWidth) * visibleWindowSpanMs,
      video.durationMs,
    );
  }, [playbackMs, timelineWidth, video, visibleWindowSpanMs, visibleWindowStartMs]);

  const measureTimelineSurface = useCallback((onMeasured?: () => void) => {
    timelineSurfaceRef.current?.measureInWindow((pageX, _pageY, width) => {
      if (width > 0) {
        timelineLayoutRef.current = { pageX, width };
        setTimelineWidth(width);
      }
      onMeasured?.();
    });
  }, []);

  const stopTimelineEdgeScroll = useCallback(() => {
    if (edgeScrollTimerRef.current) {
      clearInterval(edgeScrollTimerRef.current);
      edgeScrollTimerRef.current = null;
    }
    edgeScrollDirectionRef.current = 0;
  }, []);

  const scrubToPageX = useCallback((pageX: number) => {
    const layout = timelineLayoutRef.current;
    if (!video || layout.width <= 0 || durationMs <= 0) return;
    latestScrubPageXRef.current = pageX;
    const localX = Math.max(
      0,
      Math.min(layout.width, pageX - layout.pageX - scrubFingerOffsetPxRef.current),
    );
    const timestampMs = timelineWindowStartRef.current + (localX / layout.width) * timelineWindowSpanRef.current;
    const clamped = clampTimestamp(timestampMs, durationMs);
    playbackPositionRef.current = clamped;
    setPlaybackMs(clamped);
    videoRef.current?.seek(clamped / 1000);
  }, [durationMs, video]);

  const startTimelineEdgeScroll = useCallback((direction: -1 | 0 | 1) => {
    if (timelineZoom <= 1 || direction === 0 || durationMs <= timelineWindowSpanRef.current) {
      stopTimelineEdgeScroll();
      return;
    }
    if (edgeScrollDirectionRef.current === direction && edgeScrollTimerRef.current) return;
    stopTimelineEdgeScroll();
    edgeScrollDirectionRef.current = direction;
    edgeScrollTimerRef.current = setInterval(() => {
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
      scrubToPageX(latestScrubPageXRef.current);
    }, TIMELINE_EDGE_SCROLL_INTERVAL_MS);
  }, [durationMs, scrubToPageX, stopTimelineEdgeScroll, timelineZoom]);

  const updateTimelineEdgeScrollFromPageX = useCallback((pageX: number) => {
    const layout = timelineLayoutRef.current;
    if (layout.width <= 0 || timelineZoom <= 1) {
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
    }
  }, [startTimelineEdgeScroll, stopTimelineEdgeScroll, timelineZoom]);

  const beginTimelineScrub = useCallback((pageX: number) => {
    setPaused(true);
    timelineIsDraggingRef.current = true;
    setTimelineIsDragging(true);
    clearStrokePreview();
    stopTimelineEdgeScroll();
    latestScrubPageXRef.current = pageX;
    if (timelineLayoutRef.current.width > 0) {
      const layout = timelineLayoutRef.current;
      const playheadLeft = ratioToLeft(
        playbackPositionRef.current,
        timelineWindowStartRef.current,
        timelineWindowStartRef.current + timelineWindowSpanRef.current,
        layout.width,
      );
      scrubFingerOffsetPxRef.current = pageX - (layout.pageX + playheadLeft);
    }
    measureTimelineSurface(() => {
      const layout = timelineLayoutRef.current;
      const playheadLeft = ratioToLeft(
        playbackPositionRef.current,
        timelineWindowStartRef.current,
        timelineWindowStartRef.current + timelineWindowSpanRef.current,
        layout.width,
      );
      scrubFingerOffsetPxRef.current = pageX - (layout.pageX + playheadLeft);
      scrubToPageX(pageX);
    });
  }, [clearStrokePreview, measureTimelineSurface, scrubToPageX, stopTimelineEdgeScroll]);

  const endTimelineScrub = useCallback(() => {
    stopTimelineEdgeScroll();
    timelineIsDraggingRef.current = false;
    setTimelineIsDragging(false);
    seekTo(playbackPositionRef.current);
  }, [seekTo, stopTimelineEdgeScroll]);

  const handleTimelineZoomChange = useCallback((zoomLevel: (typeof TIMELINE_ZOOM_LEVELS)[number]) => {
    setTimelineZoom(zoomLevel);
    const nextSpanMs = Math.max(1000, durationMs > 0 ? durationMs / zoomLevel : 0);
    setTimelineWindowStartMs(Math.max(0, Math.min(
      Math.max(0, durationMs - nextSpanMs),
      playbackPositionRef.current - nextSpanMs / 2,
    )));
  }, [durationMs]);

  useEffect(() => () => clearStrokePreview(), [clearStrokePreview]);

  useEffect(() => {
    playbackPositionRef.current = playbackMs;
  }, [playbackMs]);

  useEffect(() => {
    timelineWindowStartRef.current = visibleWindowStartMs;
  }, [visibleWindowStartMs]);

  useEffect(() => {
    timelineWindowSpanRef.current = visibleWindowSpanMs;
  }, [visibleWindowSpanMs]);

  useEffect(() => () => stopTimelineEdgeScroll(), [stopTimelineEdgeScroll]);

  useEffect(() => {
    if (!quickLabelPrompt) return;
    if (!activeMarkers.some(marker => marker.id === quickLabelPrompt.markerId)) {
      setQuickLabelPrompt(null);
    }
  }, [activeMarkers, quickLabelPrompt]);

  const selectMarker = useCallback((markerId: string) => {
    const marker = activeMarkers.find(item => item.id === markerId);
    if (!marker) return;
    setSelectedMarkerId(marker.id);
    setPaused(true);
    clearStrokePreview();
    seekTo(marker.timestamp_ms);
  }, [activeMarkers, clearStrokePreview, seekTo]);

  const toggleFullPlayback = useCallback(() => {
    clearStrokePreview();
    setPaused(current => !current);
  }, [clearStrokePreview]);

  const importAndAnalyzeVideo = useCallback(async () => {
    if (isImporting || isAnalyzing || isSaving) return;
    setIsImporting(true);
    setStatus('Välj en video...');
    setWaveform(null);
    try {
      const sessionId = await nextSessionId(modeConfig.storageFolder, modeConfig.sessionPrefix);
      const sessionDir = `${modeConfig.storageFolder}/${sessionId}`;
      await RNFS.mkdir(sessionDir);
      const videoFilename = `${sessionId}.mp4`;
      const videoPath = `${sessionDir}/${videoFilename}`;
      const imported = await VideoSegment.importVideoFile(videoPath);
      const importedAt = new Date().toISOString();
      let nextVideo: ImportedVideoState = {
        sessionId,
        sessionDir,
        sessionJsonPath: `${modeConfig.storageFolder}/${sessionId}.json`,
        videoFilename,
        videoPath: imported.outputPath || videoPath,
        durationMs: Math.max(0, Math.round(Number(imported.durationMs ?? 0))),
        importedSourceFilename: imported.displayName,
        importedSourceUri: imported.sourceUri,
        importedAt,
        rotation: Number(imported.rotation ?? 0),
        sizeBytes: Number(imported.sizeBytes ?? 0),
      };
      setStatus('Video importerad. Extraherar WAV för ljudvåg...');
      const wavFilename = `${sessionId}.wav`;
      const wavPath = `${sessionDir}/${wavFilename}`;
      let decodedWaveform: DecodedWavFile | null = null;
      try {
        const extractedAudio = await AudioCapture.extractAudioFromVideoFile(nextVideo.videoPath, wavPath);
        const audioPath = extractedAudio.outputPath || wavPath;
        const decoded = await decodeWavFile(audioPath);
        decodedWaveform = decoded;
        await RNFS.scanFile(audioPath).catch(() => {});
        setWaveform({
          ...decoded,
          filename: wavFilename,
          path: audioPath,
        });
        nextVideo = {
          ...nextVideo,
          waveformAudioFilename: wavFilename,
          waveformAudioPath: audioPath,
          waveformDurationMs: decoded.durationMs,
        };
      } catch {
        setWaveform(null);
      }
      setVideo(nextVideo);
      setMarkers([]);
      setPoseAnalysis([]);
      setAnalysisDiagnostics(null);
      setSelectedMarkerId(null);
      setPlaybackMs(0);
      setTimelineWindowStartMs(0);
      setPaused(true);
      setStatus(isBounceSideMode ? 'Video importerad. Letar ljudankare...' : 'Video importerad. Kör poseanalys...');
      if (isBounceSideMode) {
        // Ankare från den ADAPTIVA gaten (samma princip som live-detektorn):
        // en global tröskel relativ klippets max dödade alla studsar så fort
        // klippet innehöll en enda stark smäll (2026-06-11_002: 2 av ~60).
        const gateOnsets = decodedWaveform
          ? detectGateOnsets(decodedWaveform.samples, decodedWaveform.sampleRate, BOUNCE_SIDE_PEAK_MIN_GAP_MS)
              .slice(0, BOUNCE_SIDE_PEAK_MAX_MARKERS)
          : [];
        const peakMarkers: ReviewMarker[] = gateOnsets.map((onset, index) => ({
          id: `audio_peak_bounce_${String(index + 1).padStart(3, '0')}_${onset.timestamp_ms}`,
          timestamp_ms: clampTimestamp(onset.timestamp_ms, nextVideo.durationMs),
          stroke_type: 'unknown',
          source: 'audio_peak',
          review_status: 'suggested',
          created_at: new Date().toISOString(),
          anchor_source: 'audio_peak',
          audio_peak_score: Math.round(onset.rms * 1000) / 1000,
          snapshot_window_ms: {
            pre_ms: BOUNCE_SIDE_SNAPSHOT_PRE_MS,
            post_ms: BOUNCE_SIDE_SNAPSHOT_POST_MS,
          },
        }));
        // Fable-filtret behåller BOLLTRÄFFAR (racket ELLER bord): med mobilen
        // liggande på bordet leds studsljudet genom skivan och klassas ofta
        // som bordsstuds - sidan avgörs ändå av videon. Bara röst/klapp/
        // tystnad ska bort.
        let rackedMarkers = peakMarkers;
        let filteredAway = 0;
        if (decodedWaveform && peakMarkers.length > 0) {
          setStatus(`Hittade ${peakMarkers.length} ljudtoppar. Filtrerar bollträffar med Fable-modellen...`);
          const { samples, sampleRate } = decodedWaveform;
          const kept: ReviewMarker[] = [];
          for (let index = 0; index < peakMarkers.length; index += 1) {
            const marker = peakMarkers[index];
            const onsetSample = Math.round((marker.timestamp_ms / 1000) * sampleRate);
            const clip = new Float32Array(6615);
            const start = onsetSample - 2205; // 100 ms före, som live-detektorn
            for (let i = 0; i < 6615; i += 1) {
              const j = start + i;
              clip[i] = j >= 0 && j < samples.length ? samples[j] : 0;
            }
            const prediction = fablePredict(extractFableFeatures(clip));
            const pBall = (prediction.probabilities.racket_bounce ?? 0)
              + (prediction.probabilities.table_bounce ?? 0);
            if (pBall >= 0.5) {
              kept.push(marker);
            }
            if (index % 8 === 7) {
              // Släpp fram UI-tråden under filtreringen.
              await new Promise<void>(resolve => setTimeout(resolve, 0));
            }
          }
          filteredAway = peakMarkers.length - kept.length;
          rackedMarkers = kept;
        }
        // FH-/BH-sidoförslag per ankare: träfframe -> handleds-ROI ->
        // sidomodellen (0.96 på orörd holdout). Förslagen är 'suggested'
        // tills de bekräftas/rättas - de blir aldrig träningsfacit av sig
        // själva.
        let sideSuggested = 0;
        if (rackedMarkers.length > 0) {
          try {
            setStatus(`${rackedMarkers.length} racketstudsar. Föreslår FH-/BH-sida från videon...`);
            const crops = await VideoPose.extractBounceSideCrops(
              nextVideo.videoPath,
              rackedMarkers.map(marker => marker.timestamp_ms),
            );
            // Spara appens egna crops till sessionen: tillsammans med de
            // granskade markörerna blir de träningsdata i appens EGEN
            // bilddomän (pose-motor, färgrymd, skalning) - då kan PC/mobil-
            // skillnader i bildvägen inte skapa osynliga modellgap.
            try {
              await RNFS.writeFile(
                `${sessionDir}/${sessionId}.crops.json`,
                JSON.stringify({ session_id: sessionId, crops }),
                'utf8',
              );
            } catch {}
            const cropByTs = new Map(crops.map(crop => [Math.round(crop.timestamp_ms), crop]));
            rackedMarkers = rackedMarkers.map(marker => {
              const crop = cropByTs.get(Math.round(marker.timestamp_ms));
              if (!crop) return marker;
              const binary = atob(crop.rgb_b64);
              const rgb = new Uint8Array(binary.length);
              for (let i = 0; i < binary.length; i += 1) rgb[i] = binary.charCodeAt(i);
              const prediction = predictBounceSide(bounceSideFeatures(rgb, crop.roi_source));
              sideSuggested += 1;
              // Modellen känner igen FÄRGSIDAN (tränad: forehand = röd).
              // Spelare med svart forehandsida får etiketterna växlade.
              const side = forehandColorRef.current === 'black'
                ? (prediction.label === 'forehand' ? 'backhand' as const : 'forehand' as const)
                : prediction.label;
              return { ...marker, stroke_type: side };
            });
          } catch {
            // Sidoförslag är "nice to have" - ankarna fungerar utan dem.
          }
        }
        setMarkers(rackedMarkers);
        setPoseAnalysis([]);
        setSelectedMarkerId(rackedMarkers[0]?.id ?? null);
        setStatus(
          rackedMarkers.length > 0
            ? `${rackedMarkers.length} racketstudsar (Fable filtrerade bort ${filteredAway} av ${peakMarkers.length} ljudtoppar), ${sideSuggested} med FH/BH-sidoförslag. Godkänn eller rätta varje studs.`
            : 'Inga racketstudsar hittades i ljudet. Lägg till markers manuellt.',
        );
        await RNFS.scanFile(nextVideo.videoPath).catch(() => {});
        return;
      }
      setIsAnalyzing(true);
      const analyzed = await analyzeVideoOnlyStrokes(
        nextVideo.videoPath,
        nextVideo.durationMs,
        setup.handedness,
        decodedWaveform,
      );
      setMarkers(analyzed.markers);
      setPoseAnalysis(analyzed.poseAnalysis);
      setAnalysisDiagnostics(analyzed.diagnostics);
      setSelectedMarkerId(analyzed.markers[0]?.id ?? null);
      setStatus(analyzed.status);
      await RNFS.scanFile(nextVideo.videoPath).catch(() => {});
    } catch (error: any) {
      const message = String(error?.message ?? error ?? '');
      if (String(error?.code ?? '').includes('IMPORT_CANCELLED') || message.includes('cancel')) {
        setStatus('Videoimport avbruten.');
      } else {
        Alert.alert('Videoimport', `Kunde inte importera/analysera videon: ${message || 'okänt fel'}`);
        setStatus('Import eller analys misslyckades.');
      }
    } finally {
      setIsImporting(false);
      setIsAnalyzing(false);
    }
  }, [isAnalyzing, isBounceSideMode, isImporting, isSaving, modeConfig.sessionPrefix, modeConfig.storageFolder, setup.handedness]);

  const addMarkerAtTimestamp = useCallback((
    strokeType: VideoStrokeMarkerType,
    requestedTimestampMs = playbackMs,
    reviewStatus: ReviewMarker['review_status'] = 'confirmed',
  ) => {
    const timestampMs = clampTimestamp(requestedTimestampMs, durationMs);
    const marker: ReviewMarker = {
      id: `manual_pose_${Date.now()}_${timestampMs}`,
      timestamp_ms: timestampMs,
      stroke_type: strokeType,
      source: 'manual',
      review_status: reviewStatus,
      created_at: new Date().toISOString(),
    };
    setMarkers(current => [...current, marker]);
    setSelectedMarkerId(marker.id);
    setPaused(true);
    clearStrokePreview();
    return marker;
  }, [clearStrokePreview, durationMs, playbackMs]);

  const handleAddMarkerHere = useCallback(() => {
    const marker = addMarkerAtTimestamp('unknown', playbackMs, 'suggested');
    setQuickLabelPrompt({ markerId: marker.id, timestampMs: marker.timestamp_ms });
  }, [addMarkerAtTimestamp, playbackMs]);

  const handleQuickLabelChoice = useCallback((strokeType: VideoStrokeMarkerType) => {
    if (!quickLabelPrompt) return;
    setMarkers(current => current.map(marker => marker.id === quickLabelPrompt.markerId ? {
      ...marker,
      stroke_type: strokeType,
      source: 'manual',
      review_status: 'confirmed',
    } : marker));
    setSelectedMarkerId(quickLabelPrompt.markerId);
    setQuickLabelPrompt(null);
  }, [quickLabelPrompt]);

  const updateSelectedMarker = useCallback((updates: Partial<ReviewMarker>) => {
    if (!selectedMarker) return;
    setMarkers(current => current.map(marker => marker.id === selectedMarker.id ? { ...marker, ...updates } : marker));
  }, [selectedMarker]);

  const setSelectedLabel = useCallback((strokeType: VideoStrokeMarkerType) => {
    setQuickLabelPrompt(null);
    updateSelectedMarker({
      stroke_type: strokeType,
      source: 'manual',
      review_status: 'confirmed',
    });
  }, [updateSelectedMarker]);

  const moveSelected = useCallback((deltaMs: number) => {
    if (!selectedMarker) return;
    const timestampMs = clampTimestamp(selectedMarker.timestamp_ms + deltaMs, durationMs);
    updateSelectedMarker({
      timestamp_ms: timestampMs,
      source: 'manual',
      review_status: 'confirmed',
    });
    seekTo(timestampMs);
  }, [durationMs, seekTo, selectedMarker, updateSelectedMarker]);

  const snapSelectedMarker = useCallback(() => {
    if (!selectedMarker || !waveform) return;
    const timestampMs = snapMarkerToAttack(waveform.samples, waveform.sampleRate, selectedMarker.timestamp_ms);
    updateSelectedMarker({
      timestamp_ms: timestampMs,
      source: 'manual',
      review_status: 'confirmed',
    });
    seekTo(timestampMs);
  }, [seekTo, selectedMarker, updateSelectedMarker, waveform]);

  const deleteSelected = useCallback(() => {
    if (!selectedMarker) return;
    setMarkers(current => current.map(marker => marker.id === selectedMarker.id ? { ...marker, deleted: true } : marker));
    const remaining = activeMarkers.filter(marker => marker.id !== selectedMarker.id);
    setSelectedMarkerId(remaining[0]?.id ?? null);
  }, [activeMarkers, selectedMarker]);

  const goToNeighbor = useCallback((direction: -1 | 1) => {
    if (activeMarkers.length === 0) return;
    const currentIndex = Math.max(0, activeMarkers.findIndex(marker => marker.id === selectedMarker?.id));
    const nextIndex = Math.max(0, Math.min(activeMarkers.length - 1, currentIndex + direction));
    selectMarker(activeMarkers[nextIndex].id);
  }, [activeMarkers, selectMarker, selectedMarker]);

  const approveAll = useCallback(() => {
    setMarkers(current => current.map(marker => marker.deleted ? marker : {
      ...marker,
      source: 'manual',
      review_status: 'confirmed',
    }));
  }, []);

  const handleTimelinePress = useCallback((pageX: number) => {
    if (!video) return;
    const layout = timelineLayoutRef.current;
    if (layout.width <= 0) {
      measureTimelineSurface();
      return;
    }
    const timestampMs = timestampFromTimelineX(pageX - layout.pageX);
    setPaused(true);
    clearStrokePreview();
    seekTo(timestampMs);
  }, [clearStrokePreview, measureTimelineSurface, seekTo, timestampFromTimelineX, video]);

  const promptAddMarkerAtTimestamp = useCallback((timestampMs: number) => {
    setPaused(true);
    clearStrokePreview();
    seekTo(timestampMs);
    const marker = addMarkerAtTimestamp('unknown', timestampMs, 'suggested');
    setQuickLabelPrompt({ markerId: marker.id, timestampMs: marker.timestamp_ms });
  }, [addMarkerAtTimestamp, clearStrokePreview, seekTo]);

  const clearPlayheadLongPress = useCallback(() => {
    if (playheadLongPressTimerRef.current) {
      clearTimeout(playheadLongPressTimerRef.current);
      playheadLongPressTimerRef.current = null;
    }
  }, []);

  useEffect(() => () => clearPlayheadLongPress(), [clearPlayheadLongPress]);

  const playheadDragResponder = useMemo(() => PanResponder.create({
    onStartShouldSetPanResponder: () => true,
    onMoveShouldSetPanResponder: (_, gestureState) => (
      Math.abs(gestureState.dx) > TIMELINE_DRAG_THRESHOLD_PX ||
      Math.abs(gestureState.dy) > TIMELINE_DRAG_THRESHOLD_PX
    ),
    onPanResponderGrant: event => {
      playheadLongPressHandledRef.current = false;
      playheadLongPressStartPageXRef.current = event.nativeEvent.pageX;
      beginTimelineScrub(event.nativeEvent.pageX);
      playheadLongPressTimerRef.current = setTimeout(() => {
        playheadLongPressTimerRef.current = null;
        playheadLongPressHandledRef.current = true;
        promptAddMarkerAtTimestamp(playbackPositionRef.current);
      }, PLAYHEAD_LONG_PRESS_MS);
    },
    onPanResponderMove: event => {
      const movedAway = Math.abs(event.nativeEvent.pageX - playheadLongPressStartPageXRef.current) > TIMELINE_DRAG_THRESHOLD_PX;
      if (movedAway) clearPlayheadLongPress();
      scrubToPageX(event.nativeEvent.pageX);
      updateTimelineEdgeScrollFromPageX(event.nativeEvent.pageX);
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
    beginTimelineScrub,
    clearPlayheadLongPress,
    endTimelineScrub,
    promptAddMarkerAtTimestamp,
    scrubToPageX,
    updateTimelineEdgeScrollFromPageX,
  ]);

  const markerDragResponder = useMemo(() => PanResponder.create({
    onStartShouldSetPanResponder: () => Boolean(selectedMarker),
    onMoveShouldSetPanResponder: (_, gestureState) => Boolean(selectedMarker) && Math.abs(gestureState.dx) > 2,
    onPanResponderGrant: () => {
      if (!selectedMarker) return;
      setPaused(true);
      clearStrokePreview();
      markerDragStartMsRef.current = selectedMarker.timestamp_ms;
    },
    onPanResponderMove: (_, gestureState) => {
      if (!selectedMarker || timelineWidth <= 0 || visibleWindowSpanMs <= 0) return;
      const msPerPx = visibleWindowSpanMs / timelineWidth;
      const timestampMs = clampTimestamp(markerDragStartMsRef.current + gestureState.dx * msPerPx, durationMs);
      setMarkers(current => current.map(marker => marker.id === selectedMarker.id ? {
        ...marker,
        timestamp_ms: timestampMs,
        source: 'manual',
        review_status: 'confirmed',
      } : marker));
      seekTo(timestampMs);
    },
  }), [clearStrokePreview, durationMs, seekTo, selectedMarker, timelineWidth, visibleWindowSpanMs]);

  const videoProgressResponder = useMemo(() => PanResponder.create({
    onStartShouldSetPanResponder: () => Boolean(video),
    onMoveShouldSetPanResponder: (_, gestureState) => (
      Math.abs(gestureState.dx) > TIMELINE_DRAG_THRESHOLD_PX ||
      Math.abs(gestureState.dy) > TIMELINE_DRAG_THRESHOLD_PX
    ),
    onPanResponderGrant: event => {
      if (!video || videoProgressWidth <= 0) return;
      setPaused(true);
      clearStrokePreview();
      seekTo((event.nativeEvent.locationX / videoProgressWidth) * durationMs);
    },
    onPanResponderMove: event => {
      if (!video || videoProgressWidth <= 0) return;
      seekTo((event.nativeEvent.locationX / videoProgressWidth) * durationMs);
    },
  }), [clearStrokePreview, durationMs, seekTo, video, videoProgressWidth]);

  const playSelectedStroke = useCallback(() => {
    if (!selectedMarker || !video) return;
    clearStrokePreview();
    const previewPreMs = isBounceSideMode ? BOUNCE_SIDE_SNAPSHOT_PRE_MS : WINDOW_PRE_MS;
    const previewPostMs = isBounceSideMode ? BOUNCE_SIDE_SNAPSHOT_POST_MS : WINDOW_POST_MS;
    const startMs = clampTimestamp(selectedMarker.timestamp_ms - previewPreMs, video.durationMs);
    const endMs = clampTimestamp(selectedMarker.timestamp_ms + previewPostMs, video.durationMs);
    const playbackWindowMs = Math.max(250, endMs - startMs);
    strokePreviewEndMsRef.current = endMs;
    setPaused(false);
    seekTo(startMs);
    strokePreviewTimerRef.current = setTimeout(() => {
      setPaused(true);
      seekTo(endMs);
      clearStrokePreview();
    }, playbackWindowMs / Math.max(0.1, playbackRate) + 140);
  }, [clearStrokePreview, isBounceSideMode, playbackRate, seekTo, selectedMarker, video]);

  const writeSession = useCallback(async (confirmPending: boolean) => {
    if (!video) return;
    setIsSaving(true);
    try {
      if (isBounceSideMode) {
        const reviewedMarkers = activeMarkers
          .filter(marker => confirmPending || marker.review_status === 'confirmed')
          .map(marker => ({
            id: marker.id,
            timestamp_ms: marker.timestamp_ms,
            bounce_side: marker.stroke_type,
            source: marker.source === 'audio_peak' ? ('audio_peak' as const) : ('manual' as const),
            // Ärlig status: bara individuellt granskade markörer blir
            // 'confirmed'; "Godkänn alla" ger 'bulk_confirmed' som
            // träningspipelinen INTE accepterar som facit.
            review_status: marker.review_status === 'confirmed'
              ? ('confirmed' as const)
              : ('bulk_confirmed' as const),
            created_at: marker.created_at,
            audio_peak_score: marker.audio_peak_score,
            snapshot_window_ms: marker.snapshot_window_ms ?? {
              pre_ms: BOUNCE_SIDE_SNAPSHOT_PRE_MS,
              post_ms: BOUNCE_SIDE_SNAPSHOT_POST_MS,
            },
          }))
          .sort((left, right) => left.timestamp_ms - right.timestamp_ms);
        const session: VideoBounceSideSessionFile = {
          session_meta: {
            player_name: setup.name,
            handedness: setup.handedness,
            camera_facing: cameraFacing,
            camera_angle: 'front_oblique',
            camera_side: cameraSide,
            camera_source: cameraFacing === 'front' ? 'front_camera_or_imported_selfie' : 'back_camera_or_imported',
            collection_type: 'video_bounce_side_snapshot',
            label_schema: 'video_bounce_side_v1',
            anchor_source: 'audio_peak',
            racket_forehand_color: forehandColor,
            snapshot_window_ms: {
              pre_ms: BOUNCE_SIDE_SNAPSHOT_PRE_MS,
              post_ms: BOUNCE_SIDE_SNAPSHOT_POST_MS,
            },
            waveform_audio_filename: video.waveformAudioFilename,
            app_version: modeConfig.appVersion,
            created_at: video.importedAt,
          },
          takes: [{
            video_filename: video.videoFilename,
            duration_ms: video.durationMs,
            take_index: 1,
            recording_mode: 'imported_source',
            review_status: 'reviewed',
            markers: reviewedMarkers,
            source_video_filename: video.importedSourceFilename,
            waveform_audio_filename: video.waveformAudioFilename,
            imported_source_uri: video.importedSourceUri,
            imported_at: video.importedAt,
          }],
        };
        await RNFS.writeFile(video.sessionJsonPath, JSON.stringify(session, null, 2), 'utf8');
        await RNFS.scanFile(video.sessionJsonPath).catch(() => {});
        setMarkers(current => current.map(marker => marker.deleted ? marker : {
          ...marker,
          source: 'manual',
          review_status: 'confirmed',
        }));
        setStatus(`Sparat ${reviewedMarkers.length} studs-side-markeringar till ${video.sessionId}.json`);
        Alert.alert('Video studs review sparad', `Sparade ${reviewedMarkers.length} markeringar.\n${video.sessionJsonPath}`);
        return;
      }
      const reviewedMarkers = activeMarkers
        .filter(marker => confirmPending || marker.review_status === 'confirmed')
        .map(marker => ({
          id: marker.id,
          timestamp_ms: marker.timestamp_ms,
          stroke_type: marker.stroke_type,
          source: 'manual' as const,
          // Ärlig status: "Godkänn alla" får aldrig se ut som individuell
          // granskning i filen (träningen kräver confirmed/edited).
          review_status: marker.review_status === 'confirmed'
            ? ('confirmed' as const)
            : ('bulk_confirmed' as const),
          created_at: marker.created_at,
        }))
        .sort((left, right) => left.timestamp_ms - right.timestamp_ms);
      const session: VideoStrokeSessionFile = {
        session_meta: {
          player_name: setup.name,
          handedness: setup.handedness,
          camera_facing: cameraFacing,
          camera_angle: 'front_oblique',
          camera_side: cameraSide,
          camera_source: cameraFacing === 'front' ? 'front_camera_or_imported_selfie' : 'back_camera_or_imported',
          collection_type: 'video_pose_only',
          pose_sample_fps: SAMPLE_FPS,
          waveform_audio_filename: video.waveformAudioFilename,
          app_version: modeConfig.appVersion,
          created_at: video.importedAt,
        },
        takes: [{
          video_filename: video.videoFilename,
          duration_ms: video.durationMs,
          take_index: 1,
          recording_mode: 'imported_source',
          review_status: 'reviewed',
          markers: reviewedMarkers,
          pose_analysis: poseAnalysis,
          analysis_diagnostics: analysisDiagnostics,
          source_video_filename: video.importedSourceFilename,
          waveform_audio_filename: video.waveformAudioFilename,
          imported_source_uri: video.importedSourceUri,
          imported_at: video.importedAt,
        }],
      };
      await RNFS.writeFile(video.sessionJsonPath, JSON.stringify(session, null, 2), 'utf8');
      await RNFS.scanFile(video.sessionJsonPath).catch(() => {});
      setMarkers(current => current.map(marker => marker.deleted ? marker : {
        ...marker,
        source: 'manual',
        review_status: 'confirmed',
      }));
      setStatus(`Sparat ${reviewedMarkers.length} markeringar till ${video.sessionId}.json`);
      Alert.alert('Video review sparad', `Sparade ${reviewedMarkers.length} markeringar.\n${video.sessionJsonPath}`);
    } catch (error) {
      Alert.alert('Sparfel', `Kunde inte spara video-sessionen: ${String(error)}`);
    } finally {
      setIsSaving(false);
    }
  }, [
    activeMarkers,
    cameraFacing,
    cameraSide,
    isBounceSideMode,
    modeConfig.appVersion,
    poseAnalysis,
    setup.handedness,
    setup.name,
    video,
  ]);

  const saveReview = useCallback(() => {
    if (!video) return;
    if (pendingCount > 0) {
      if (isBounceSideMode) {
        Alert.alert(
          'Det finns omarkta ljudankare',
          `${pendingCount} ljudankare ar inte bekraftade. Vill du spara bara de markeringar du har labelat?`,
          [
            { text: 'Avbryt', style: 'cancel' },
            { text: 'Spara labelade', onPress: () => void writeSession(false) },
          ],
        );
        return;
      }
      Alert.alert(
        'Det finns förslag kvar',
        `${pendingCount} förslag är inte godkända. Vill du godkänna dem och spara?`,
        [
          { text: 'Avbryt', style: 'cancel' },
          { text: 'Spara godkända', onPress: () => void writeSession(false) },
          { text: 'Godkänn alla', onPress: () => void writeSession(true) },
        ],
      );
      return;
    }
    void writeSession(false);
  }, [isBounceSideMode, pendingCount, video, writeSession]);

  const selectedAnalysis = selectedMarker
    ? poseAnalysis.find(analysis => analysis.marker_id === selectedMarker.id)
    : null;

  return (
    <View style={styles.root}>
      <StatusBar hidden barStyle="light-content" backgroundColor="#0d0d0d" />
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
          <TouchableOpacity onPress={onDone} style={styles.reviewBackBtn}>
            <Text style={styles.reviewBackTxt}>Tillbaka</Text>
          </TouchableOpacity>
          <View style={styles.reviewHeaderCopy}>
            <Text style={styles.reviewTitle}>{modeConfig.title}</Text>
            <Text style={styles.reviewSubtitle}>
              {modeConfig.subtitlePrefix} {video ? `| ${formatMs(durationMs)}` : '| importera MP4'}
            </Text>
          </View>
          <TouchableOpacity
            style={styles.menuBtn}
            onPress={() => Alert.alert(modeConfig.helpTitle, modeConfig.helpText)}
          >
            <Text style={styles.menuTxt}>...</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.algorithmPanel}>
          {isBounceSideMode && (
            <>
              <Text style={styles.algorithmLabel}>Forehandsidans färg (ditt racket)</Text>
              <View style={styles.labelSegment}>
                {(['red', 'black'] as const).map(color => {
                  const active = forehandColor === color;
                  return (
                    <TouchableOpacity
                      key={color}
                      style={[styles.segmentBtn, active && { backgroundColor: '#12351f', borderColor: '#2ecc71' }]}
                      onPress={() => setForehandColor(color)}
                    >
                      <Text style={[styles.segmentTxt, active && styles.segmentTxtActive]}>
                        {color === 'red' ? 'Röd' : 'Svart'}
                      </Text>
                    </TouchableOpacity>
                  );
                })}
              </View>
            </>
          )}
          <Text style={styles.algorithmLabel}>Kameraposition</Text>
          <View style={styles.labelSegment}>
            {(['player_left', 'player_right', 'center_front'] as VideoStrokeCameraSide[]).map(side => {
              const active = cameraSide === side;
              return (
                <TouchableOpacity
                  key={side}
                  style={[styles.segmentBtn, active && { backgroundColor: '#12351f', borderColor: '#2ecc71' }]}
                  onPress={() => setCameraSide(side)}
                >
                  <Text style={[styles.segmentTxt, active && styles.segmentTxtActive]}>{cameraSideLabel(side)}</Text>
                </TouchableOpacity>
              );
            })}
          </View>
          <Text style={styles.algorithmLabel}>Kamera</Text>
          <View style={styles.labelSegment}>
            {(['front', 'back'] as VideoStrokeCameraFacing[]).map(facing => {
              const active = cameraFacing === facing;
              return (
                <TouchableOpacity
                  key={facing}
                  style={[styles.segmentBtn, active && { backgroundColor: '#12351f', borderColor: '#2ecc71' }]}
                  onPress={() => setCameraFacing(facing)}
                >
                  <Text style={[styles.segmentTxt, active && styles.segmentTxtActive]}>
                    {facing === 'front' ? 'Front/selfie' : 'Backkamera'}
                  </Text>
                </TouchableOpacity>
              );
            })}
          </View>
          <TouchableOpacity
            style={[styles.importBtn, (isImporting || isAnalyzing || isSaving) && styles.disabledBtn]}
            onPress={importAndAnalyzeVideo}
            disabled={isImporting || isAnalyzing || isSaving}
          >
            <Text style={styles.importBtnTxt}>{video ? 'Importera ny video' : 'Importera MP4/video'}</Text>
          </TouchableOpacity>
          {status && <Text style={styles.algorithmSaveHint}>{status}</Text>}
          {(isImporting || isAnalyzing) && (
            <View style={styles.loadingRow}>
              <ActivityIndicator color="#2ecc71" />
              <Text style={styles.loadingTxt}>{isAnalyzing ? 'Analyserar pose...' : 'Importerar...'}</Text>
            </View>
          )}
        </View>

        {video && (
          <>
            <View style={[styles.heroVideoFrame, { height: videoHeight }]}>
              <Video
                ref={videoRef}
                source={{ uri: `file://${video.videoPath}` }}
                style={styles.videoPlayer}
                paused={paused}
                rate={playbackRate}
                muted={false}
                volume={1}
                repeat={false}
                controls={false}
                resizeMode="contain"
                progressUpdateInterval={50}
                onLoad={(data: any) => {
                  const naturalSize = data?.naturalSize;
                  const width = Number(naturalSize?.width ?? 0);
                  const height = Number(naturalSize?.height ?? 0);
                  if (width > 0 && height > 0) setVideoNaturalSize({ width, height });
                  seekTo(playbackMs);
                }}
                onProgress={(progress: any) => {
                  const currentMs = Math.max(0, Math.round(Number(progress?.currentTime ?? 0) * 1000));
                  playbackPositionRef.current = currentMs;
                  setPlaybackMs(currentMs);
                  if (!timelineIsDraggingRef.current && timelineZoom > 1 && durationMs > visibleWindowSpanMs) {
                    setTimelineWindowStartMs(currentStart => {
                      const marginMs = visibleWindowSpanMs * 0.18;
                      const currentEnd = currentStart + visibleWindowSpanMs;
                      if (currentMs >= currentStart + marginMs && currentMs <= currentEnd - marginMs) {
                        return currentStart;
                      }
                      return Math.max(0, Math.min(durationMs - visibleWindowSpanMs, currentMs - visibleWindowSpanMs / 2));
                    });
                  }
                  if (strokePreviewEndMsRef.current !== null && currentMs >= strokePreviewEndMsRef.current) {
                    setPaused(true);
                    clearStrokePreview();
                  }
                }}
                onError={error => Alert.alert('Videofel', `Kunde inte spela videon: ${String(error)}`)}
              />
              <Pressable style={styles.videoTapLayer} onPress={toggleFullPlayback} />
              <View style={styles.rateChip}>
                <TouchableOpacity
                  onPress={() => setPlaybackRate(current => current === 1 ? 0.5 : current === 0.5 ? 0.25 : 1)}
                >
                  <Text style={styles.rateChipTxt}>{playbackRate.toFixed(1)}x</Text>
                </TouchableOpacity>
              </View>
              <View style={styles.videoControls}>
                <TouchableOpacity style={styles.videoPlayBtn} onPress={toggleFullPlayback}>
                  <Text style={styles.videoPlayTxt}>{paused ? 'Spela' : 'Paus'}</Text>
                </TouchableOpacity>
                <Text style={styles.videoTimeTxt}>{formatMs(playbackMs)}</Text>
                <View
                  style={styles.videoProgressTrack}
                  onLayout={event => setVideoProgressWidth(Math.max(1, event.nativeEvent.layout.width))}
                  {...videoProgressResponder.panHandlers}
                >
                  <View
                    style={[
                      styles.videoProgressFill,
                      { width: `${durationMs > 0 ? Math.min(100, (playbackMs / durationMs) * 100) : 0}%` },
                    ]}
                  />
                  <View
                    style={[
                      styles.videoProgressKnob,
                      { left: `${durationMs > 0 ? Math.min(100, (playbackMs / durationMs) * 100) : 0}%` },
                    ]}
                  />
                </View>
                <Text style={styles.videoTimeTxt}>{formatMs(durationMs)}</Text>
              </View>
            </View>

            <View style={styles.waveformCard}>
              <View style={styles.timelineHeaderRow}>
                <Text style={styles.timelineTitle}>{modeConfig.timelineTitle}</Text>
                <View style={styles.zoomControlsRow}>
                  {TIMELINE_ZOOM_LEVELS.map(zoomLevel => (
                    <TouchableOpacity
                      key={zoomLevel}
                      style={[styles.zoomBtn, timelineZoom === zoomLevel && styles.zoomBtnActive]}
                      onPress={() => handleTimelineZoomChange(zoomLevel)}
                    >
                      <Text style={[styles.zoomBtnTxt, timelineZoom === zoomLevel && styles.zoomBtnTxtActive]}>{zoomLevel}x</Text>
                    </TouchableOpacity>
                  ))}
                </View>
              </View>
              <View style={styles.timeTicksRow}>
                {[0, 0.25, 0.5, 0.75, 1].map(ratio => (
                  <Text key={`tick-${ratio}`} style={styles.tickTxt}>
                    {formatMs(visibleWindowStartMs + visibleWindowSpanMs * ratio)}
                  </Text>
                ))}
              </View>
              <View
                ref={timelineSurfaceRef}
                style={styles.fullWaveformSurface}
                onLayout={event => {
                  const width = Math.max(1, event.nativeEvent.layout.width);
                  setTimelineWidth(width);
                  timelineLayoutRef.current = { ...timelineLayoutRef.current, width };
                  requestAnimationFrame(() => measureTimelineSurface());
                }}
              >
                <Pressable
                  style={StyleSheet.absoluteFill}
                  onPress={event => handleTimelinePress(event.nativeEvent.pageX)}
                  onLongPress={event => {
                    const layout = timelineLayoutRef.current;
                    promptAddMarkerAtTimestamp(timestampFromTimelineX(event.nativeEvent.pageX - layout.pageX));
                  }}
                  delayLongPress={PLAYHEAD_LONG_PRESS_MS}
                />
                <View pointerEvents="none" style={styles.fullWaveformRow}>
                  {waveformBins.length > 0 ? waveformBins.map((amplitude, index) => {
                    const visualAmplitude = Math.pow(Math.max(0, Math.min(1, amplitude)), 0.72);
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
                  }) : (
                    <Text style={styles.waveformEmptyTxt}>Ingen ljudvåg i filen</Text>
                  )}
                </View>
                {visibleMarkers.map(marker => {
                  const left = ratioToLeft(marker.timestamp_ms, visibleWindowStartMs, visibleWindowEndMs, timelineWidth);
                  const isSelected = selectedMarker?.id === marker.id;
                  return (
                    <TouchableOpacity
                      key={`full-marker-${marker.id}`}
                      style={[styles.fullMarker, { left: Math.max(0, left - 12) }]}
                      onPress={() => selectMarker(marker.id)}
                      {...(isSelected ? markerDragResponder.panHandlers : {})}
                    >
                      <View
                        style={[
                          styles.fullMarkerPin,
                          { backgroundColor: markerColor(marker.stroke_type) },
                          isSelected && styles.fullMarkerPinActive,
                        ]}
                      />
                    </TouchableOpacity>
                  );
                })}
                <View
                  style={[styles.fullPlayheadHitbox, { left: Math.max(0, timelinePlayheadLeft - TIMELINE_EDGE_PX / 2) }]}
                  {...playheadDragResponder.panHandlers}
                >
                  <View style={[styles.fullPlayheadLine, timelineIsDragging && styles.fullPlayheadLineActive]} />
                  <View style={[styles.fullPlayheadKnob, timelineIsDragging && styles.fullPlayheadKnobActive]} />
                </View>
              </View>
              <Text style={styles.waveHelpTxt}>
                Dra playhead till träffen. Håll på den vita markeringen i 2 sekunder för att skapa marker.
              </Text>

              {activeMarkers.length > 0 && (
                <View style={styles.timelineMarkerNav}>
                  <TouchableOpacity
                    style={[styles.timelineMarkerNavBtn, selectedMarkerIndex <= 0 && styles.disabledBtn]}
                    onPress={() => goToNeighbor(-1)}
                    disabled={selectedMarkerIndex <= 0}
                  >
                    <Text style={styles.timelineMarkerNavTxt}>Föregående</Text>
                  </TouchableOpacity>
                  <View style={styles.timelineMarkerNavCenter}>
                    <Text style={styles.timelineMarkerNavTitle}>
                      Marker {selectedMarkerIndex >= 0 ? selectedMarkerIndex + 1 : 0} av {activeMarkers.length}
                    </Text>
                    <Text style={styles.timelineMarkerNavSub}>
                      {selectedMarker ? formatMs(selectedMarker.timestamp_ms) : formatMs(playbackMs)}
                    </Text>
                  </View>
                  <TouchableOpacity
                    style={[
                      styles.timelineMarkerNavBtn,
                      selectedMarkerIndex >= activeMarkers.length - 1 && styles.disabledBtn,
                    ]}
                    onPress={() => goToNeighbor(1)}
                    disabled={selectedMarkerIndex >= activeMarkers.length - 1}
                  >
                    <Text style={styles.timelineMarkerNavTxt}>Nästa</Text>
                  </TouchableOpacity>
                </View>
              )}

              {quickLabelPrompt && (
                <View style={styles.quickLabelPanel}>
                  <View style={styles.quickLabelCopy}>
                    <Text style={styles.quickLabelTitle}>Ny marker {formatMs(quickLabelPrompt.timestampMs)}</Text>
                    <Text style={styles.quickLabelSub}>Välj typ direkt</Text>
                  </View>
                  <View style={styles.quickLabelChoices}>
                    {(['forehand', 'backhand', 'unknown'] as VideoStrokeMarkerType[]).map(label => (
                      <TouchableOpacity
                        key={`quick-${label}`}
                        style={[styles.quickLabelBtn, { borderColor: markerColor(label) }]}
                        onPress={() => handleQuickLabelChoice(label)}
                      >
                        <Text style={styles.quickLabelBtnTxt}>{markerLabel(label, mode)}</Text>
                      </TouchableOpacity>
                    ))}
                    <TouchableOpacity style={styles.quickLabelCancelBtn} onPress={() => setQuickLabelPrompt(null)}>
                      <Text style={styles.quickLabelCancelTxt}>Senare</Text>
                    </TouchableOpacity>
                  </View>
                  <View style={styles.quickLabelTools}>
                    <TouchableOpacity style={styles.utilityBtn} onPress={() => moveSelected(-EXTRA_LARGE_NUDGE_STEP_MS)} disabled={!selectedMarker}>
                      <Text style={styles.utilityTxt}>-50 ms</Text>
                    </TouchableOpacity>
                    <TouchableOpacity style={styles.utilityBtn} onPress={() => moveSelected(-LARGE_NUDGE_STEP_MS)} disabled={!selectedMarker}>
                      <Text style={styles.utilityTxt}>-20 ms</Text>
                    </TouchableOpacity>
                    <TouchableOpacity style={styles.utilityBtn} onPress={() => moveSelected(-NUDGE_STEP_MS)} disabled={!selectedMarker}>
                      <Text style={styles.utilityTxt}>-10 ms</Text>
                    </TouchableOpacity>
                    <TouchableOpacity style={styles.utilityBtn} onPress={() => moveSelected(NUDGE_STEP_MS)} disabled={!selectedMarker}>
                      <Text style={styles.utilityTxt}>+10 ms</Text>
                    </TouchableOpacity>
                    <TouchableOpacity style={styles.utilityBtn} onPress={() => moveSelected(LARGE_NUDGE_STEP_MS)} disabled={!selectedMarker}>
                      <Text style={styles.utilityTxt}>+20 ms</Text>
                    </TouchableOpacity>
                    <TouchableOpacity style={styles.utilityBtn} onPress={() => moveSelected(EXTRA_LARGE_NUDGE_STEP_MS)} disabled={!selectedMarker}>
                      <Text style={styles.utilityTxt}>+50 ms</Text>
                    </TouchableOpacity>
                    <TouchableOpacity style={[styles.utilityBtn, styles.utilityDeleteBtn]} onPress={deleteSelected} disabled={!selectedMarker}>
                      <Text style={styles.utilityDeleteTxt}>Ta bort</Text>
                    </TouchableOpacity>
                  </View>
                </View>
              )}
            </View>

            <View style={styles.markerPanel}>
              <View style={styles.markerTopRow}>
                <View>
                  <Text style={styles.markerPanelTitle}>Marker {selectedMarkerIndex >= 0 ? selectedMarkerIndex + 1 : 0} av {activeMarkers.length}</Text>
                  <Text style={styles.markerPanelSub}>
                    {selectedMarker
                      ? `${formatMs(selectedMarker.timestamp_ms)} · ${markerLabel(selectedMarker.stroke_type, mode)}`
                      : `${formatMs(playbackMs)} · ingen marker vald`}
                  </Text>
                  {selectedMarker?.source === 'model' && selectedAnalysis && (
                    <Text style={styles.markerConfidenceTxt}>
                      Säkerhet {Math.round(selectedAnalysis.confidence * 100)}%
                    </Text>
                  )}
                </View>
                <View style={styles.markerControlsRow}>
                  <TouchableOpacity
                    style={[styles.roundControlBtn, selectedMarkerIndex <= 0 && styles.disabledBtn]}
                    onPress={() => goToNeighbor(-1)}
                    disabled={selectedMarkerIndex <= 0}
                  >
                    <Text style={styles.roundControlTxt}>Föreg.</Text>
                  </TouchableOpacity>
                  <TouchableOpacity
                    style={[styles.centerPlayBtn, !selectedMarker && styles.disabledBtn]}
                    onPress={playSelectedStroke}
                    disabled={!selectedMarker}
                  >
                    <Text style={styles.centerPlayTxt}>Spela</Text>
                  </TouchableOpacity>
                  <TouchableOpacity
                    style={[styles.roundControlBtn, selectedMarkerIndex >= activeMarkers.length - 1 && styles.disabledBtn]}
                    onPress={() => goToNeighbor(1)}
                    disabled={selectedMarkerIndex >= activeMarkers.length - 1}
                  >
                    <Text style={styles.roundControlTxt}>Nästa</Text>
                  </TouchableOpacity>
                  <TouchableOpacity style={styles.addMarkerBtn} onPress={handleAddMarkerHere}>
                    <Text style={styles.addMarkerTxt}>+ Lägg till</Text>
                  </TouchableOpacity>
                  <TouchableOpacity
                    style={[styles.approveAllBtn, approvableAutoMarkers.length === 0 && styles.disabledBtn]}
                    onPress={approveAll}
                    disabled={approvableAutoMarkers.length === 0}
                  >
                    <Text style={styles.approveAllTxt}>Godkänn alla</Text>
                  </TouchableOpacity>
                </View>
              </View>

              <View style={styles.labelSegment}>
                {(['forehand', 'backhand', 'unknown'] as VideoStrokeMarkerType[]).map(label => {
                  const active = selectedMarker?.stroke_type === label;
                  return (
                    <TouchableOpacity
                      key={label}
                      style={[
                        styles.segmentBtn,
                        active && { backgroundColor: markerColor(label), borderColor: markerColor(label) },
                      ]}
                      disabled={!selectedMarker}
                      onPress={() => setSelectedLabel(label)}
                    >
                      <Text style={[styles.segmentTxt, active && styles.segmentTxtActive]}>{markerLabel(label, mode)}</Text>
                    </TouchableOpacity>
                  );
                })}
              </View>

              <View style={styles.markerUtilityRow}>
                <TouchableOpacity style={styles.utilityBtn} onPress={snapSelectedMarker} disabled={!selectedMarker || !waveform}>
                  <Text style={styles.utilityTxt}>Snappa</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.utilityBtn} onPress={() => moveSelected(-EXTRA_LARGE_NUDGE_STEP_MS)} disabled={!selectedMarker}>
                  <Text style={styles.utilityTxt}>-50 ms</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.utilityBtn} onPress={() => moveSelected(-LARGE_NUDGE_STEP_MS)} disabled={!selectedMarker}>
                  <Text style={styles.utilityTxt}>-20 ms</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.utilityBtn} onPress={() => moveSelected(-NUDGE_STEP_MS)} disabled={!selectedMarker}>
                  <Text style={styles.utilityTxt}>-10 ms</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.utilityBtn} onPress={() => moveSelected(NUDGE_STEP_MS)} disabled={!selectedMarker}>
                  <Text style={styles.utilityTxt}>+10 ms</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.utilityBtn} onPress={() => moveSelected(LARGE_NUDGE_STEP_MS)} disabled={!selectedMarker}>
                  <Text style={styles.utilityTxt}>+20 ms</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.utilityBtn} onPress={() => moveSelected(EXTRA_LARGE_NUDGE_STEP_MS)} disabled={!selectedMarker}>
                  <Text style={styles.utilityTxt}>+50 ms</Text>
                </TouchableOpacity>
                <TouchableOpacity style={[styles.utilityBtn, styles.utilityDeleteBtn]} onPress={deleteSelected} disabled={!selectedMarker}>
                  <Text style={styles.utilityDeleteTxt}>Ta bort</Text>
                </TouchableOpacity>
              </View>
            </View>

            <View style={styles.algorithmPanel}>
              <View style={styles.algorithmHeaderRow}>
                <View style={styles.algorithmTitleRow}>
                  <Text style={styles.algorithmTitle}>{modeConfig.algorithmTitle}</Text>
                </View>
                <Text style={styles.algorithmMeta}>
                  Kandidater {autoCount} · review {reviewedCount} · manuella {manualCount} · pending {pendingCount}
                </Text>
              </View>
              <Text style={styles.algorithmSaveHint}>
                {modeConfig.saveHint}
              </Text>
            </View>

            <View style={styles.footerActions}>
              <TouchableOpacity
                style={[styles.footerBtn, styles.saveBtn, isSaving && styles.saveBtnDisabled]}
                onPress={saveReview}
                disabled={isSaving}
              >
                <Text style={styles.saveBtnTxt}>{isSaving ? 'Sparar...' : 'Spara review'}</Text>
              </TouchableOpacity>
              <TouchableOpacity style={[styles.footerBtn, styles.discardBtn]} onPress={onDone}>
                <Text style={styles.discardBtnTxt}>Kassera</Text>
              </TouchableOpacity>
            </View>
          </>
        )}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: '#0d0d0d',
  },
  disabledBtn: { opacity: 0.45 },
  reviewScroll: { flex: 1 },
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
  videoPlayer: {
    width: '100%',
    height: '100%',
    backgroundColor: '#050505',
  },
  videoTapLayer: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 1,
    elevation: 1,
  },
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
  videoTimeTxt: { color: '#e8e8e8', fontSize: 13, fontWeight: '700', fontVariant: ['tabular-nums'] },
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
  algorithmMeta: { color: '#aeb4be', fontSize: 11, fontWeight: '700' },
  algorithmSaveHint: { color: '#7f8993', fontSize: 10, lineHeight: 14, fontWeight: '700' },
  algorithmLabel: {
    color: '#727b86',
    fontSize: 10,
    fontWeight: '900',
    letterSpacing: 1.2,
    textTransform: 'uppercase',
  },
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
  timeTicksRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  tickTxt: { color: '#9aa2ad', fontSize: 11, fontVariant: ['tabular-nums'] },
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
  waveformEmptyTxt: {
    color: '#606873',
    fontSize: 12,
    fontWeight: '800',
    textAlign: 'center',
    width: '100%',
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
  importBtn: {
    minHeight: 38,
    borderRadius: 12,
    backgroundColor: '#145c2a',
    borderWidth: 1,
    borderColor: '#269d4c',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 12,
  },
  importBtnTxt: { color: '#fff', fontSize: 13, fontWeight: '900' },
  loadingRow: { flexDirection: 'row', alignItems: 'center', gap: 10, marginTop: 2 },
  loadingTxt: { color: '#bbb', fontSize: 12, fontWeight: '800' },
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
  discardBtn: { backgroundColor: '#2d0d0d' },
  discardBtnTxt: { color: '#ff7f7f', fontSize: 14, fontWeight: '800' },
});
