/**
 * Studs FH/BH LIVE
 *
 * Camera starts immediately for aiming. START only starts the audio bounce
 * counter. The original and v2 modes use the continuous racket tracker for
 * side decisions; v3 keeps tracker metadata for debug but decides side from the
 * wrist-crop model.
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  PermissionsAndroid, Platform, StyleSheet, StatusBar, Text, TouchableOpacity, View,
} from 'react-native';
import RNFS from 'react-native-fs';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { decodeBase64PCM } from './NativeAudioCapture';
import {
  AudioStream,
  AudioStreamEmitter,
  type NativeAudioBounceEvent,
  type NativeAudioOnsetDebug,
} from './NativeAudioStream';
import {
  BounceSideCameraView,
  BounceSideLive,
  BounceSideLiveEmitter,
  type BounceSideRacketTrack,
} from './NativeBounceSideLive';
import { FableCounter } from './fableEngine';
import {
  bounceSideFeatures,
  predictBounceSide,
  resolveBounceSide,
  BOUNCE_SIDE_MODEL_VERSION,
} from './bounceSideInference';
import { detectAudioContact } from './audioContactEngine';
import { getAudioDetectionConfig } from './audioDetectionConfig';
import type { PlayerSetup } from './types';

const ONSET_THRESHOLD = 0.005;
const RETRIGGER_MS = 120;
const ABS_MIN_RMS = 0.0015;
const HYBRID_AUDIO_CONFIG = getAudioDetectionConfig('normal', 'hybrid');
const HYBRID_ONSET_THRESHOLD = HYBRID_AUDIO_CONFIG.onset_threshold;
const HYBRID_RETRIGGER_MS = 220;
const HYBRID_ABS_MIN_RMS = ABS_MIN_RMS;
const HYBRID_CONTACT_THRESHOLD = HYBRID_AUDIO_CONFIG.contact_confidence_min;
const HYBRID_SURFACE_VETO_CONFIDENCE = HYBRID_AUDIO_CONFIG.surface_veto_confidence ?? 0.75;
const HYBRID_DEDUP_MS = HYBRID_AUDIO_CONFIG.merge_window_ms;
const HYBRID22_ONSET_THRESHOLD = 0.005;
const HYBRID22_RETRIGGER_MS = 220;
const HYBRID22_ABS_MIN_RMS = 0.003;
const HYBRID22_CONTACT_THRESHOLD = 0.25;
const HYBRID22_SURFACE_VETO_CONFIDENCE = 0.75;
const HYBRID22_VETO_THRESHOLD = 0.01;
const HYBRID22_VETO_BYPASS_ENABLED = true;
const HYBRID22_VETO_BYPASS_THRESHOLD = 0.61;
const HYBRID22_DEDUP_MS = 180;
const SIDE_MIN_CONFIDENCE = 0.6;
const TRACK_MAX_DELAY_MS = 500;
const TRACK_MIN_CONFIDENCE = 0.95;
const TRACK_VISIBLE_MIN_CONFIDENCE = 0.95;
const TRACK_EVENT_NAME = 'onBounceSideRacketTrack';
const TRACKER_VERSION = 'color_shape_tracker_v3_2026_06_26';

type LiveSide = 'forehand' | 'backhand' | 'uncertain';
type ForehandColor = 'red' | 'black';
type AudioTriggerMode = 'fable' | 'hybrid' | 'hybrid22';
type SideDecisionMode = 'tracker' | 'wrist_crop';

interface LiveDebugEvent {
  onset_time_ms: number;
  side: LiveSide;
  confidence: number;
  decision_source: string;
  tracker_version: string;
  track_tracked: boolean;
  track_label: string;
  track_color: string;
  track_confidence: number;
  track_source: string;
  track_frame_delay_ms: number;
  track_x: number;
  track_y: number;
  track_width: number;
  track_height: number;
  track_red_score: number;
  track_dark_score: number;
  track_area_ratio: number;
  track_fill_ratio: number;
  raw_side?: string;
  raw_confidence?: number;
  probabilities?: Record<string, number>;
  visible_color?: string;
  color_confidence?: number;
  red_total?: number;
  dark_total?: number;
  roi_source?: string;
  crop_frame_delay_ms?: number;
  crop_error?: string;
  audio_label: string;
  audio_confidence: number;
  audio_contact_threshold?: number;
  audio_surface_label?: string;
  audio_surface_confidence?: number;
  audio_hybrid_veto_probability?: number;
  audio_hybrid_veto_threshold?: number;
  audio_hybrid_veto_bypassed?: boolean;
  audio_hybrid_veto_bypass_threshold?: number;
  rgb_b64?: string;
}

interface LiveAudioCandidate {
  onset_time_ms: number;
  frame_rms: number;
  counted: boolean;
  reject_reason?: string;
  audio_label?: string;
  audio_confidence?: number;
  audio_contact_threshold?: number;
  audio_surface_label?: string;
  audio_surface_confidence?: number;
  audio_hybrid_veto_probability?: number;
  audio_hybrid_veto_threshold?: number;
  audio_hybrid_veto_bypassed?: boolean;
  audio_hybrid_veto_bypass_threshold?: number;
  bg_mode?: string;
}

interface Props {
  setup: PlayerSetup;
  onDone: () => void;
  audioTriggerMode?: AudioTriggerMode;
  sideDecisionMode?: SideDecisionMode;
}

function parseNativeEvent(event: NativeAudioBounceEvent): {
  audioB64?: string;
  nativeDebug?: NativeAudioOnsetDebug;
} {
  if (typeof event === 'string') return { audioB64: event };
  return { audioB64: event.audio_b64 ?? undefined, nativeDebug: event.native_debug };
}

function lostTrack(): BounceSideRacketTrack {
  return {
    tracked: false,
    label: 'lost',
    color: 'uncertain',
    confidence: 0,
    x: 0,
    y: 0,
    width: 0,
    height: 0,
    timestamp_ms: 0,
    age_ms: 0,
    frame_delay_ms: 0,
    source: 'lost',
    red_score: 0,
    dark_score: 0,
    area_ratio: 0,
    fill_ratio: 0,
  };
}

function sideFromColor(color: 'red' | 'black', forehandColor: ForehandColor): LiveSide {
  if (forehandColor === 'red') return color === 'red' ? 'forehand' : 'backhand';
  return color === 'black' ? 'forehand' : 'backhand';
}

function resolveTrackSide(track: BounceSideRacketTrack | null, forehandColor: ForehandColor): {
  side: LiveSide;
  confidence: number;
  decisionSource: string;
} {
  if (!track || !track.tracked) {
    return { side: 'uncertain', confidence: 0, decisionSource: 'tracker_lost' };
  }
  if (Math.abs(track.frame_delay_ms) > TRACK_MAX_DELAY_MS) {
    return { side: 'uncertain', confidence: track.confidence, decisionSource: 'tracker_stale' };
  }
  if (track.confidence < TRACK_MIN_CONFIDENCE) {
    return { side: 'uncertain', confidence: track.confidence, decisionSource: 'tracker_low_confidence' };
  }
  if (track.label === 'racket-red') {
    return { side: sideFromColor('red', forehandColor), confidence: track.confidence, decisionSource: 'tracker_color' };
  }
  if (track.label === 'racket-black') {
    return { side: sideFromColor('black', forehandColor), confidence: track.confidence, decisionSource: 'tracker_color' };
  }
  return { side: 'uncertain', confidence: track.confidence, decisionSource: 'tracker_generic' };
}

function visibleTrack(track: BounceSideRacketTrack | null): BounceSideRacketTrack | null {
  if (!track || !track.tracked || track.width <= 0 || track.height <= 0) return null;
  if (track.confidence < TRACK_VISIBLE_MIN_CONFIDENCE) return null;
  return track;
}

export function BounceSideLiveScreen({
  setup,
  onDone,
  audioTriggerMode = 'fable',
  sideDecisionMode = 'tracker',
}: Props) {
  const insets = useSafeAreaInsets();
  const isHybrid = audioTriggerMode === 'hybrid';
  const isHybrid22 = audioTriggerMode === 'hybrid22';
  const usesAudioContactEngine = isHybrid || isHybrid22;
  const usesWristCropSide = sideDecisionMode === 'wrist_crop';
  const [isRunning, setIsRunning] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [cameraViewReady, setCameraViewReady] = useState(false);
  const [fhCount, setFhCount] = useState(0);
  const [bhCount, setBhCount] = useState(0);
  const [uncertainCount, setUncertainCount] = useState(0);
  const [lastSide, setLastSide] = useState<{ side: LiveSide; confidence: number; source: string } | null>(null);
  const [latestTrack, setLatestTrack] = useState<BounceSideRacketTrack | null>(null);
  const [forehandColor, setForehandColor] = useState<ForehandColor>('red');
  const [audioOnlyMode, setAudioOnlyMode] = useState(false);
  const [statusText, setStatusText] = useState('Startar kamera...');

  const counterRef = useRef(new FableCounter({ loudBgDb: -36, loudConfidence: 0.85 }));
  const hybridLastQualifiedTsRef = useRef<number | undefined>(undefined);
  const forehandColorRef = useRef<ForehandColor>('red');
  const cameraViewReadyRef = useRef(false);
  const cameraStartedRef = useRef(false);
  const busyRef = useRef(false);
  const debugEventsRef = useRef<LiveDebugEvent[]>([]);
  const audioCandidatesRef = useRef<LiveAudioCandidate[]>([]);
  forehandColorRef.current = forehandColor;

  const writeDebugDump = useCallback(() => {
    const events = debugEventsRef.current;
    const audioCandidates = audioCandidatesRef.current;
    if (events.length === 0 && audioCandidates.length === 0) return;
    const path = `${RNFS.DownloadDirectoryPath}/pingis_live_sidedebug_${Date.now()}.json`;
    const payload = {
      model: BOUNCE_SIDE_MODEL_VERSION,
      tracker: TRACKER_VERSION,
      audio_trigger_mode: audioTriggerMode,
      side_decision_mode: sideDecisionMode,
      audio_only_count_mode: audioOnlyMode,
      audio_trigger_config: isHybrid22 ? {
        contact_threshold: HYBRID22_CONTACT_THRESHOLD,
        surface_veto_confidence: HYBRID22_SURFACE_VETO_CONFIDENCE,
        hybrid_veto_threshold: HYBRID22_VETO_THRESHOLD,
        hybrid_veto_bypass_enabled: HYBRID22_VETO_BYPASS_ENABLED,
        hybrid_veto_bypass_threshold: HYBRID22_VETO_BYPASS_THRESHOLD,
        dedup_ms: HYBRID22_DEDUP_MS,
        native_gate: {
          onset_threshold: HYBRID22_ONSET_THRESHOLD,
          retrigger_ms: HYBRID22_RETRIGGER_MS,
          mode: 'broadband',
          spectral_gate: true,
          abs_min_rms: HYBRID22_ABS_MIN_RMS,
        },
      } : isHybrid ? {
        detection_mode: 'hybrid',
        contact_threshold: HYBRID_CONTACT_THRESHOLD,
        surface_veto_confidence: HYBRID_SURFACE_VETO_CONFIDENCE,
        dedup_ms: HYBRID_DEDUP_MS,
        native_gate: {
          onset_threshold: HYBRID_ONSET_THRESHOLD,
          retrigger_ms: HYBRID_RETRIGGER_MS,
          mode: 'bandpass',
          spectral_gate: false,
          abs_min_rms: HYBRID_ABS_MIN_RMS,
        },
      } : {
        onset_threshold: ONSET_THRESHOLD,
        retrigger_ms: RETRIGGER_MS,
        mode: 'bandpass',
        spectral_gate: false,
        abs_min_rms: ABS_MIN_RMS,
      },
      setup,
      forehand_color: forehandColorRef.current,
      events,
      audio_candidates: audioCandidates,
    };
    RNFS.writeFile(path, JSON.stringify(payload), 'utf8')
      .then(() => RNFS.scanFile(path))
      .catch(() => {});
    debugEventsRef.current = [];
    audioCandidatesRef.current = [];
  }, [audioOnlyMode, audioTriggerMode, isHybrid, isHybrid22, setup, sideDecisionMode]);

  const startCameraForAiming = useCallback(async () => {
    if (cameraStartedRef.current) {
      setCameraReady(true);
      return true;
    }
    if (!cameraViewReadyRef.current) {
      setStatusText('Startar kameravy...');
      return false;
    }
    if (Platform.OS === 'android') {
      const granted = await PermissionsAndroid.request(PermissionsAndroid.PERMISSIONS.CAMERA);
      if (granted !== 'granted') {
        setStatusText('Kameratillstand kravs.');
        return false;
      }
    }
    await BounceSideLive.startCamera(true);
    cameraStartedRef.current = true;
    setCameraReady(true);
    setStatusText('Rikta kameran sa att racketen syns. Tryck STARTA nar du vill rakna.');
    return true;
  }, []);

  const stopCameraPreview = useCallback(() => {
    void BounceSideLive.stopCamera().catch(() => {});
    cameraStartedRef.current = false;
    setCameraReady(false);
  }, []);

  const restartCameraForAiming = useCallback(async () => {
    await BounceSideLive.stopCamera().catch(() => {});
    cameraStartedRef.current = false;
    setCameraReady(false);
    return startCameraForAiming();
  }, [startCameraForAiming]);

  const stopCounting = useCallback((message = 'Stoppad. Kameran ar kvar for riktning.') => {
    AudioStream.stopStreaming();
    setIsRunning(false);
    writeDebugDump();
    setStatusText(message);
  }, [writeDebugDump]);

  const stopAll = useCallback(() => {
    AudioStream.stopStreaming();
    void BounceSideLive.stopCamera();
    cameraStartedRef.current = false;
    setCameraReady(false);
    setIsRunning(false);
    writeDebugDump();
  }, [writeDebugDump]);

  const toggle = useCallback(() => {
    if (isRunning) {
      stopCounting();
      return;
    }
    void (async () => {
      try {
        if (Platform.OS === 'android') {
          const granted = await PermissionsAndroid.request(PermissionsAndroid.PERMISSIONS.RECORD_AUDIO);
          if (granted !== 'granted') {
            setStatusText('Mikrofontillstand kravs.');
            return;
          }
        }
        const cameraOk = audioOnlyMode ? true : await startCameraForAiming();
        if (!cameraOk) return;
        counterRef.current.reset();
        debugEventsRef.current = [];
        audioCandidatesRef.current = [];
        hybridLastQualifiedTsRef.current = undefined;
        setFhCount(0);
        setBhCount(0);
        setUncertainCount(0);
        setLastSide(null);
        await AudioStream.startStreaming(
          isHybrid22 ? HYBRID22_ONSET_THRESHOLD : isHybrid ? HYBRID_ONSET_THRESHOLD : ONSET_THRESHOLD,
        );
        await AudioStream.setRetriggerMs(
          isHybrid22 ? HYBRID22_RETRIGGER_MS : isHybrid ? HYBRID_RETRIGGER_MS : RETRIGGER_MS,
        );
        if (isHybrid22) {
          await AudioStream.setGateConfig('broadband', true, HYBRID22_ABS_MIN_RMS);
        } else if (isHybrid) {
          await AudioStream.setGateConfig('bandpass', false, HYBRID_ABS_MIN_RMS);
        } else {
          await AudioStream.setGateConfig('bandpass', false, ABS_MIN_RMS);
        }
        setIsRunning(true);
        setStatusText(audioOnlyMode
          ? 'Endast ljud: godkanda ljudstudsar raknas som OSAKER.'
          : isHybrid22
            ? (usesWristCropSide
                ? 'Hybrid 2.2 lyssnar. FH/BH avgors med handledscrop efter varje studs.'
                : 'Hybrid 2.2 lyssnar och kameran avgor FH/BH. Studsa bollen pa racketen!')
            : isHybrid
              ? 'Hybrid lyssnar. FH/BH avgors med handledscrop efter varje studs.'
              : 'Lyssnar och tittar. Studsa bollen pa racketen!');
      } catch (error) {
        setStatusText(`Kunde inte starta: ${String((error as Error)?.message ?? error)}`);
      }
    })();
  }, [audioOnlyMode, isHybrid, isHybrid22, isRunning, startCameraForAiming, stopCounting, usesWristCropSide]);

  useEffect(() => {
    if (!cameraViewReady || audioOnlyMode) return;
    void startCameraForAiming().catch(error => {
      setStatusText(`Kunde inte starta kamera: ${String((error as Error)?.message ?? error)}`);
    });
  }, [audioOnlyMode, cameraViewReady, startCameraForAiming]);

  useEffect(() => {
    return () => {
      stopAll();
    };
  }, [stopAll]);

  useEffect(() => {
    const sub = BounceSideLiveEmitter.addListener(TRACK_EVENT_NAME, (track: BounceSideRacketTrack) => {
      setLatestTrack(track);
    });
    return () => sub.remove();
  }, []);

  useEffect(() => {
    if (!isRunning) return undefined;

    const sub = AudioStreamEmitter.addListener('onBounceDetected', (event: NativeAudioBounceEvent) => {
      const { audioB64, nativeDebug } = parseNativeEvent(event);
      if (!audioB64 || busyRef.current) return;
      busyRef.current = true;
      void (async () => {
        try {
          const onsetTimeMs = nativeDebug?.onset_time_ms ?? Date.now();
          const frameRms = nativeDebug?.rms ?? 0;
          const pcm = decodeBase64PCM(audioB64);
          const audioDecision = usesAudioContactEngine ? (() => {
            const contactThreshold = isHybrid22 ? HYBRID22_CONTACT_THRESHOLD : HYBRID_CONTACT_THRESHOLD;
            const surfaceVetoConfidence = isHybrid22
              ? HYBRID22_SURFACE_VETO_CONFIDENCE
              : HYBRID_SURFACE_VETO_CONFIDENCE;
            const dedupMs = isHybrid22 ? HYBRID22_DEDUP_MS : HYBRID_DEDUP_MS;
            const decision = detectAudioContact({
              detectedAtMs: onsetTimeMs,
              pcm,
              confidenceThreshold: contactThreshold,
              dedupMs,
              lastQualifiedTsMs: hybridLastQualifiedTsRef.current,
              surfaceVetoConfidence,
              detectionMode: audioTriggerMode === 'hybrid22' ? 'hybrid22' : 'hybrid',
              config: audioTriggerMode === 'hybrid' ? HYBRID_AUDIO_CONFIG : undefined,
              hybrid22ContactThreshold: HYBRID22_CONTACT_THRESHOLD,
              hybrid22VetoThreshold: HYBRID22_VETO_THRESHOLD,
              hybrid22VetoBypassEnabled: HYBRID22_VETO_BYPASS_ENABLED,
              hybrid22VetoBypassThreshold: HYBRID22_VETO_BYPASS_THRESHOLD,
            });
            if (decision.qualified) {
              hybridLastQualifiedTsRef.current = onsetTimeMs;
            }
            return {
              counted: decision.qualified,
              rejectReason: decision.ignored_reason,
              label: decision.label,
              confidence: decision.confidence,
              contactThreshold: decision.contact_threshold,
              surfaceLabel: decision.surface_label,
              surfaceConfidence: decision.surface_confidence,
              hybridVetoProbability: decision.hybrid_veto_probability,
              hybridVetoThreshold: decision.hybrid_veto_threshold,
              hybridVetoBypassed: decision.hybrid_veto_bypassed,
              hybridVetoBypassThreshold: decision.hybrid_veto_bypass_threshold,
              bgMode: undefined,
            };
          })() : (() => {
            const result = counterRef.current.process(pcm, onsetTimeMs, frameRms, Date.now());
            return {
              counted: result.counted,
              rejectReason: result.rejectReason,
              label: result.prediction?.label,
              confidence: result.prediction?.confidence,
              contactThreshold: undefined,
              surfaceLabel: undefined,
              surfaceConfidence: undefined,
              hybridVetoProbability: undefined,
              hybridVetoThreshold: undefined,
              hybridVetoBypassed: undefined,
              hybridVetoBypassThreshold: undefined,
              bgMode: result.bgMode,
            };
          })();
          if (audioCandidatesRef.current.length < 600) {
            audioCandidatesRef.current.push({
              onset_time_ms: onsetTimeMs,
              frame_rms: frameRms,
              counted: audioDecision.counted,
              reject_reason: audioDecision.rejectReason,
              audio_label: audioDecision.label,
              audio_confidence: audioDecision.confidence,
              audio_contact_threshold: audioDecision.contactThreshold,
              audio_surface_label: audioDecision.surfaceLabel,
              audio_surface_confidence: audioDecision.surfaceConfidence,
              audio_hybrid_veto_probability: audioDecision.hybridVetoProbability,
              audio_hybrid_veto_threshold: audioDecision.hybridVetoThreshold,
              audio_hybrid_veto_bypassed: audioDecision.hybridVetoBypassed,
              audio_hybrid_veto_bypass_threshold: audioDecision.hybridVetoBypassThreshold,
              bg_mode: audioDecision.bgMode,
            });
          }
          if (!audioDecision.counted) return;

          if (audioOnlyMode) {
            const track = lostTrack();
            const resolved = {
              side: 'uncertain' as LiveSide,
              confidence: audioDecision.confidence ?? 0,
              decisionSource: 'audio_only',
            };
            setUncertainCount(n => n + 1);
            setLastSide({ side: resolved.side, confidence: resolved.confidence, source: resolved.decisionSource });

            if (debugEventsRef.current.length < 300) {
              debugEventsRef.current.push({
                onset_time_ms: onsetTimeMs,
                side: resolved.side,
                confidence: resolved.confidence,
                decision_source: resolved.decisionSource,
                tracker_version: TRACKER_VERSION,
                track_tracked: track.tracked,
                track_label: track.label,
                track_color: track.color,
                track_confidence: track.confidence,
                track_source: track.source,
                track_frame_delay_ms: track.frame_delay_ms,
                track_x: track.x,
                track_y: track.y,
                track_width: track.width,
                track_height: track.height,
                track_red_score: track.red_score,
                track_dark_score: track.dark_score,
                track_area_ratio: track.area_ratio,
                track_fill_ratio: track.fill_ratio,
                audio_label: audioDecision.label ?? 'unknown',
                audio_confidence: audioDecision.confidence ?? 0,
                audio_contact_threshold: audioDecision.contactThreshold,
                audio_surface_label: audioDecision.surfaceLabel,
                audio_surface_confidence: audioDecision.surfaceConfidence,
                audio_hybrid_veto_probability: audioDecision.hybridVetoProbability,
                audio_hybrid_veto_threshold: audioDecision.hybridVetoThreshold,
                audio_hybrid_veto_bypassed: audioDecision.hybridVetoBypassed,
                audio_hybrid_veto_bypass_threshold: audioDecision.hybridVetoBypassThreshold,
              });
            }
            return;
          }

          const track = await BounceSideLive.getRacketTrack(onsetTimeMs).catch(() => lostTrack());
          let resolved = resolveTrackSide(track, forehandColorRef.current);
          const cropDebug: Partial<LiveDebugEvent> = {};

          try {
            const crop = await BounceSideLive.captureCrop(onsetTimeMs);
            const binary = atob(crop.rgb_b64);
            const rgb = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i += 1) rgb[i] = binary.charCodeAt(i);
            const features = bounceSideFeatures(rgb, crop.roi_source);
            const prediction = predictBounceSide(features);
            const cropResolved = resolveBounceSide(
              features,
              prediction,
              forehandColorRef.current,
              SIDE_MIN_CONFIDENCE,
            );
            cropDebug.raw_side = cropResolved.rawLabel;
            cropDebug.raw_confidence = cropResolved.rawConfidence;
            cropDebug.probabilities = prediction.probabilities;
            cropDebug.visible_color = cropResolved.visibleColor;
            cropDebug.color_confidence = cropResolved.colorConfidence;
            cropDebug.red_total = cropResolved.redTotal;
            cropDebug.dark_total = cropResolved.darkTotal;
            cropDebug.roi_source = crop.roi_source;
            cropDebug.crop_frame_delay_ms = crop.frame_delay_ms;
            cropDebug.rgb_b64 = crop.rgb_b64;
            if (usesWristCropSide) {
              resolved = {
                side: cropResolved.side,
                confidence: cropResolved.confidence,
                decisionSource: `wrist_crop_${cropResolved.decisionSource}`,
              };
            }
          } catch (error) {
            cropDebug.crop_error = String((error as Error)?.message ?? error);
            if (usesWristCropSide) {
              resolved = { side: 'uncertain', confidence: 0, decisionSource: 'wrist_crop_error' };
            }
          }

          if (resolved.side === 'forehand') setFhCount(n => n + 1);
          else if (resolved.side === 'backhand') setBhCount(n => n + 1);
          else setUncertainCount(n => n + 1);
          setLastSide({ side: resolved.side, confidence: resolved.confidence, source: resolved.decisionSource });

          const debugEvent: LiveDebugEvent = {
            onset_time_ms: onsetTimeMs,
            side: resolved.side,
            confidence: resolved.confidence,
            decision_source: resolved.decisionSource,
            tracker_version: TRACKER_VERSION,
            track_tracked: track.tracked,
            track_label: track.label,
            track_color: track.color,
            track_confidence: track.confidence,
            track_source: track.source,
            track_frame_delay_ms: track.frame_delay_ms,
            track_x: track.x,
            track_y: track.y,
            track_width: track.width,
            track_height: track.height,
            track_red_score: track.red_score,
            track_dark_score: track.dark_score,
            track_area_ratio: track.area_ratio,
            track_fill_ratio: track.fill_ratio,
            audio_label: audioDecision.label ?? 'unknown',
            audio_confidence: audioDecision.confidence ?? 0,
            audio_contact_threshold: audioDecision.contactThreshold,
            audio_surface_label: audioDecision.surfaceLabel,
            audio_surface_confidence: audioDecision.surfaceConfidence,
            audio_hybrid_veto_probability: audioDecision.hybridVetoProbability,
            audio_hybrid_veto_threshold: audioDecision.hybridVetoThreshold,
            audio_hybrid_veto_bypassed: audioDecision.hybridVetoBypassed,
            audio_hybrid_veto_bypass_threshold: audioDecision.hybridVetoBypassThreshold,
            ...cropDebug,
          };

          if (debugEventsRef.current.length < 300) {
            debugEventsRef.current.push(debugEvent);
          }
        } finally {
          busyRef.current = false;
        }
      })();
    });

    return () => sub.remove();
  }, [audioOnlyMode, audioTriggerMode, isHybrid22, isRunning, usesAudioContactEngine, usesWristCropSide]);

  const shouldShowTrackerOverlay = !usesWristCropSide;
  const shownTrack = shouldShowTrackerOverlay ? visibleTrack(latestTrack) : null;

  return (
    <View style={[styles.root, { paddingTop: insets.top }]}>
      <StatusBar barStyle="light-content" backgroundColor="#000" />
      <View style={styles.header}>
        <TouchableOpacity onPress={() => { stopAll(); onDone(); }}>
          <Text style={styles.back}>{'<'} Tillbaka</Text>
        </TouchableOpacity>
        <Text style={styles.title}>
          {isHybrid22
            ? (usesWristCropSide ? 'Studs FH/BH LIVE v3' : 'Studs FH/BH LIVE v2')
            : isHybrid && usesWristCropSide
              ? 'Studs FH/BH LIVE v4'
              : 'Studs FH/BH LIVE'}
        </Text>
        <Text style={styles.subtitle}>
          {isHybrid22
            ? (usesWristCropSide ? 'Hybrid 2.2 audio + wrist crop side' : `Hybrid 2.2 audio + ${TRACKER_VERSION}`)
            : isHybrid && usesWristCropSide
              ? 'Hybrid audio + wrist crop side'
            : TRACKER_VERSION}
        </Text>
      </View>

      <View style={styles.cameraWrap}>
        <BounceSideCameraView
          style={styles.camera}
          collapsable={false}
          onLayout={() => {
            cameraViewReadyRef.current = true;
            setCameraViewReady(true);
          }}
        />
        {shouldShowTrackerOverlay ? (
          shownTrack ? (
            <View
              pointerEvents="none"
              style={[
                styles.trackBox,
                {
                  left: `${shownTrack.x * 100}%`,
                  top: `${shownTrack.y * 100}%`,
                  width: `${shownTrack.width * 100}%`,
                  height: `${shownTrack.height * 100}%`,
                },
              ]}
            >
              <Text style={styles.trackLabel} numberOfLines={1}>
                {shownTrack.label} {(shownTrack.confidence * 100).toFixed(0)}%
              </Text>
            </View>
          ) : (
            <View pointerEvents="none" style={styles.trackerBadge}>
              <Text style={styles.trackerBadgeText}>{cameraReady ? 'racket lost' : 'camera starting'}</Text>
            </View>
          )
        ) : null}
        {lastSide ? (
          <View style={[
            styles.sideBadge,
            lastSide.side === 'forehand' ? styles.badgeFh : lastSide.side === 'backhand' ? styles.badgeBh : styles.badgeUncertain,
          ]}>
            <Text style={styles.sideBadgeTxt}>
              {lastSide.side === 'forehand' ? 'FOREHAND' : lastSide.side === 'backhand' ? 'BACKHAND' : 'OSAKER'} {(lastSide.confidence * 100).toFixed(0)}%
            </Text>
          </View>
        ) : null}
      </View>

      <View style={styles.countRow}>
        <View style={styles.countBox}>
          <Text style={[styles.countValue, { color: '#35c7ff' }]}>{fhCount}</Text>
          <Text style={styles.countLabel}>FOREHAND</Text>
        </View>
        <View style={styles.countBox}>
          <Text style={[styles.countValue, { color: '#f1c40f' }]}>{bhCount}</Text>
          <Text style={styles.countLabel}>BACKHAND</Text>
        </View>
        <View style={styles.countBox}>
          <Text style={[styles.countValue, { color: '#888' }]}>{uncertainCount}</Text>
          <Text style={styles.countLabel}>OSAKER</Text>
        </View>
      </View>

      <View style={styles.colorRow}>
        <Text style={styles.colorLabel}>Forehandsidans farg:</Text>
        {(['red', 'black'] as const).map(color => {
          const active = forehandColor === color;
          return (
            <TouchableOpacity
              key={color}
              style={[styles.colorBtn, active && styles.colorBtnActive]}
              onPress={() => setForehandColor(color)}
            >
              <Text style={[styles.colorTxt, active && styles.colorTxtActive]}>
                {color === 'red' ? 'Rod' : 'Svart'}
              </Text>
            </TouchableOpacity>
          );
        })}
      </View>

      <TouchableOpacity
        style={[
          styles.audioOnlyToggle,
          audioOnlyMode && styles.audioOnlyToggleActive,
          isRunning && styles.audioOnlyToggleDisabled,
        ]}
        onPress={() => {
          if (!isRunning) {
            setAudioOnlyMode(value => {
              const nextValue = !value;
              if (nextValue) {
                stopCameraPreview();
                setStatusText('Audio only: kameran anvands inte for rakning. Tryck STARTA.');
              } else {
                void restartCameraForAiming().catch(error => {
                  setStatusText(`Kunde inte starta kamera: ${String((error as Error)?.message ?? error)}`);
                });
              }
              return nextValue;
            });
          }
        }}
        disabled={isRunning}
      >
        <Text style={[styles.audioOnlyTitle, audioOnlyMode && styles.audioOnlyTitleActive]}>
          {audioOnlyMode ? 'Audio only: ON' : 'Audio only: OFF'}
        </Text>
        <Text style={styles.audioOnlyHint}>
          {isRunning
            ? 'Stoppa for att andra'
            : audioOnlyMode
              ? 'Skippar kamera/FH-BH och raknar allt som OSAKER'
              : 'Anvander kamera/FH-BH efter ljudstuds'}
        </Text>
      </TouchableOpacity>

      <TouchableOpacity
        style={[
          styles.toggle,
          isRunning ? styles.toggleStop : styles.toggleStart,
          !isRunning && !audioOnlyMode && !cameraReady && styles.toggleDisabled,
        ]}
        onPress={toggle}
        disabled={!isRunning && !audioOnlyMode && !cameraReady}
      >
        <Text style={styles.toggleText}>{isRunning ? 'STOPPA' : 'STARTA'}</Text>
      </TouchableOpacity>

      <Text style={styles.statusText}>{statusText}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#000' },
  header: { paddingHorizontal: 16, paddingBottom: 6 },
  back: { color: '#4a9eff', fontSize: 16, paddingVertical: 6 },
  title: { color: '#fff', fontSize: 24, fontWeight: '700' },
  subtitle: { color: '#555', fontSize: 11 },
  cameraWrap: { flex: 1, margin: 12, borderRadius: 14, overflow: 'hidden', backgroundColor: '#111' },
  camera: { flex: 1 },
  trackBox: {
    position: 'absolute',
    borderWidth: 3,
    borderColor: '#31f06a',
    backgroundColor: 'rgba(49,240,106,0.08)',
    minWidth: 42,
    minHeight: 32,
  },
  trackLabel: {
    position: 'absolute',
    top: -26,
    left: -3,
    backgroundColor: '#31f06a',
    color: '#001b08',
    fontSize: 12,
    fontWeight: '800',
    minWidth: 116,
    paddingHorizontal: 6,
    paddingVertical: 3,
  },
  trackerBadge: {
    position: 'absolute',
    bottom: 12,
    left: 12,
    backgroundColor: 'rgba(0,0,0,0.62)',
    borderWidth: 1,
    borderColor: '#333',
    paddingHorizontal: 10,
    paddingVertical: 6,
  },
  trackerBadgeText: { color: '#aaa', fontSize: 12, fontWeight: '700' },
  sideBadge: { position: 'absolute', top: 12, alignSelf: 'center', paddingHorizontal: 18, paddingVertical: 8, borderRadius: 20 },
  badgeFh: { backgroundColor: 'rgba(53,199,255,0.85)' },
  badgeBh: { backgroundColor: 'rgba(241,196,15,0.85)' },
  badgeUncertain: { backgroundColor: 'rgba(150,150,150,0.85)' },
  sideBadgeTxt: { color: '#000', fontWeight: '800', fontSize: 16 },
  countRow: { flexDirection: 'row', gap: 12, paddingHorizontal: 12 },
  countBox: { flex: 1, alignItems: 'center', backgroundColor: '#101010', borderRadius: 12, paddingVertical: 10 },
  countValue: { fontSize: 44, fontWeight: '800' },
  countLabel: { color: '#888', fontSize: 12, letterSpacing: 2 },
  colorRow: { flexDirection: 'row', alignItems: 'center', gap: 8, paddingHorizontal: 16, paddingTop: 10 },
  colorLabel: { color: '#888', fontSize: 13 },
  colorBtn: { paddingHorizontal: 14, paddingVertical: 6, borderRadius: 8, borderWidth: 1, borderColor: '#333', backgroundColor: '#111' },
  colorBtnActive: { borderColor: '#2ecc71', backgroundColor: '#12351f' },
  colorTxt: { color: '#888', fontSize: 13 },
  colorTxtActive: { color: '#fff', fontWeight: '700' },
  audioOnlyToggle: {
    marginHorizontal: 16,
    marginTop: 10,
    borderWidth: 1,
    borderColor: '#333',
    backgroundColor: '#101010',
    borderRadius: 10,
    paddingHorizontal: 14,
    paddingVertical: 9,
  },
  audioOnlyToggleActive: { borderColor: '#31f06a', backgroundColor: '#102b19' },
  audioOnlyToggleDisabled: { opacity: 0.65 },
  audioOnlyTitle: { color: '#aaa', fontSize: 14, fontWeight: '800' },
  audioOnlyTitleActive: { color: '#31f06a' },
  audioOnlyHint: { color: '#777', fontSize: 11, marginTop: 2 },
  toggle: { marginHorizontal: 24, marginVertical: 10, paddingVertical: 14, borderRadius: 10, alignItems: 'center' },
  toggleStart: { backgroundColor: '#1d6f42' },
  toggleStop: { backgroundColor: '#8e2b2b' },
  toggleDisabled: { backgroundColor: '#333' },
  toggleText: { color: '#fff', fontSize: 18, fontWeight: '800' },
  statusText: { color: '#666', fontSize: 12, textAlign: 'center', paddingHorizontal: 16, paddingBottom: 12 },
});
