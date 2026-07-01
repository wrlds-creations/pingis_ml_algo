/**
 * Studs FH/BH LIVE
 *
 * Camera starts immediately for aiming. The visible racket tracker draws a box
 * continuously; START only starts the audio bounce counter. Fable audio remains
 * the bounce trigger, and the tracker color decides FH/BH when it is fresh.
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
import type { PlayerSetup } from './types';

const ONSET_THRESHOLD = 0.005;
const RETRIGGER_MS = 120;
const ABS_MIN_RMS = 0.0015;
const SIDE_MIN_CONFIDENCE = 0.6;
const TRACK_MAX_DELAY_MS = 500;
const TRACK_MIN_CONFIDENCE = 0.95;
const TRACK_VISIBLE_MIN_CONFIDENCE = 0.95;
const TRACK_EVENT_NAME = 'onBounceSideRacketTrack';
const TRACKER_VERSION = 'color_shape_tracker_v3_2026_06_26';

type LiveSide = 'forehand' | 'backhand' | 'uncertain';
type ForehandColor = 'red' | 'black';

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
  rgb_b64?: string;
}

interface LiveAudioCandidate {
  onset_time_ms: number;
  frame_rms: number;
  counted: boolean;
  reject_reason?: string;
  audio_label?: string;
  audio_confidence?: number;
  bg_mode?: string;
}

interface Props { setup: PlayerSetup; onDone: () => void; }

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

export function BounceSideLiveScreen({ setup, onDone }: Props) {
  const insets = useSafeAreaInsets();
  const [isRunning, setIsRunning] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [cameraViewReady, setCameraViewReady] = useState(false);
  const [fhCount, setFhCount] = useState(0);
  const [bhCount, setBhCount] = useState(0);
  const [uncertainCount, setUncertainCount] = useState(0);
  const [lastSide, setLastSide] = useState<{ side: LiveSide; confidence: number; source: string } | null>(null);
  const [latestTrack, setLatestTrack] = useState<BounceSideRacketTrack | null>(null);
  const [forehandColor, setForehandColor] = useState<ForehandColor>('red');
  const [statusText, setStatusText] = useState('Startar kamera...');

  const counterRef = useRef(new FableCounter({ loudBgDb: -36, loudConfidence: 0.85 }));
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
  }, [setup]);

  const startCameraForAiming = useCallback(async () => {
    if (cameraStartedRef.current) return true;
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
        const cameraOk = await startCameraForAiming();
        if (!cameraOk) return;
        counterRef.current.reset();
        debugEventsRef.current = [];
        audioCandidatesRef.current = [];
        setFhCount(0);
        setBhCount(0);
        setUncertainCount(0);
        setLastSide(null);
        await AudioStream.startStreaming(ONSET_THRESHOLD);
        await AudioStream.setRetriggerMs(RETRIGGER_MS);
        await AudioStream.setGateConfig('bandpass', false, ABS_MIN_RMS);
        setIsRunning(true);
        setStatusText('Lyssnar och tittar. Studsa bollen pa racketen!');
      } catch (error) {
        setStatusText(`Kunde inte starta: ${String((error as Error)?.message ?? error)}`);
      }
    })();
  }, [isRunning, startCameraForAiming, stopCounting]);

  useEffect(() => {
    if (!cameraViewReady) return;
    void startCameraForAiming().catch(error => {
      setStatusText(`Kunde inte starta kamera: ${String((error as Error)?.message ?? error)}`);
    });
  }, [cameraViewReady, startCameraForAiming]);

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
          const result = counterRef.current.process(pcm, onsetTimeMs, frameRms, Date.now());
          if (audioCandidatesRef.current.length < 600) {
            audioCandidatesRef.current.push({
              onset_time_ms: onsetTimeMs,
              frame_rms: frameRms,
              counted: result.counted,
              reject_reason: result.rejectReason,
              audio_label: result.prediction?.label,
              audio_confidence: result.prediction?.confidence,
              bg_mode: result.bgMode,
            });
          }
          if (!result.counted) return;

          const track = await BounceSideLive.getRacketTrack(onsetTimeMs).catch(() => lostTrack());
          const resolved = resolveTrackSide(track, forehandColorRef.current);
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
            audio_label: result.prediction?.label ?? 'unknown',
            audio_confidence: result.prediction?.confidence ?? 0,
          };

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
            debugEvent.raw_side = cropResolved.rawLabel;
            debugEvent.raw_confidence = cropResolved.rawConfidence;
            debugEvent.probabilities = prediction.probabilities;
            debugEvent.visible_color = cropResolved.visibleColor;
            debugEvent.color_confidence = cropResolved.colorConfidence;
            debugEvent.red_total = cropResolved.redTotal;
            debugEvent.dark_total = cropResolved.darkTotal;
            debugEvent.roi_source = crop.roi_source;
            debugEvent.crop_frame_delay_ms = crop.frame_delay_ms;
            debugEvent.rgb_b64 = crop.rgb_b64;
          } catch (error) {
            debugEvent.crop_error = String((error as Error)?.message ?? error);
          }

          if (debugEventsRef.current.length < 300) {
            debugEventsRef.current.push(debugEvent);
          }
        } finally {
          busyRef.current = false;
        }
      })();
    });

    return () => sub.remove();
  }, [isRunning]);

  const shownTrack = visibleTrack(latestTrack);

  return (
    <View style={[styles.root, { paddingTop: insets.top }]}>
      <StatusBar barStyle="light-content" backgroundColor="#000" />
      <View style={styles.header}>
        <TouchableOpacity onPress={() => { stopAll(); onDone(); }}>
          <Text style={styles.back}>{'<'} Tillbaka</Text>
        </TouchableOpacity>
        <Text style={styles.title}>Studs FH/BH LIVE</Text>
        <Text style={styles.subtitle}>{TRACKER_VERSION}</Text>
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
        {shownTrack ? (
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
        )}
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
        style={[styles.toggle, isRunning ? styles.toggleStop : styles.toggleStart, !cameraReady && styles.toggleDisabled]}
        onPress={toggle}
        disabled={!cameraReady && !isRunning}
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
  toggle: { marginHorizontal: 24, marginVertical: 10, paddingVertical: 14, borderRadius: 10, alignItems: 'center' },
  toggleStart: { backgroundColor: '#1d6f42' },
  toggleStop: { backgroundColor: '#8e2b2b' },
  toggleDisabled: { backgroundColor: '#333' },
  toggleText: { color: '#fff', fontSize: 18, fontWeight: '800' },
  statusText: { color: '#666', fontSize: 12, textAlign: 'center', paddingHorizontal: 16, paddingBottom: 12 },
});
