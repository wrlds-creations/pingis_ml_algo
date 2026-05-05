import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Alert,
  PermissionsAndroid,
  Platform,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import type { BleError, Characteristic, Device } from 'react-native-ble-plx';
import RNFS from 'react-native-fs';
import { AudioStream, AudioStreamEmitter } from './NativeAudioStream';
import { decodeBase64PCM } from './NativeAudioCapture';
import { detectAudioContact } from './audioContactEngine';
import {
  ACCEL_UUID,
  ACCEL_UUID_ALT,
  GYRO_UUID,
  MAG_UUID,
  SERVICE_UUID,
  formatClock,
  parsePacket,
} from './airhive';
import { DebugSlider } from './DebugSlider';
import {
  buildStrokeWindow,
  extractStrokeFeatures,
  extractStrokeMotionMetrics,
} from './strokeFeatures';
import { predictStroke } from './strokeInference';
import type {
  AudioDetectionEvent,
  CalibrationData,
  ImuSample,
  PlayerSetup,
  StrokeCombinedLabel,
  StrokeInferenceEvent,
  StrokeMotionMetrics,
  StrokePresetId,
  StrokeSettings,
  StrokeTestSessionFile,
} from './types';

const APP_VERSION = '1.6';
const EXPORT_DIR = `${RNFS.ExternalStorageDirectoryPath}/Download/pingis_sessions`;
const WINDOW_SAMPLES = 40;
const STEP_MS = 200;
const AUDIO_MATCH_MS = 250;
const AUDIO_DEDUP_MS = 180;

const STROKE_PRESETS: Record<StrokePresetId, { label: string; settings: StrokeSettings }> = {
  S0: {
    label: 'S0 Default',
    settings: {
      motionGyroThreshold: 150,
      motionAccelThreshold: 500,
      modelThreshold: 0.75,
      imuDedupMs: 500,
      audioDebugThreshold: 0.02,
      audioDebugConfidence: 0.65,
    },
  },
  S1: {
    label: 'S1 Sensitive',
    settings: {
      motionGyroThreshold: 120,
      motionAccelThreshold: 350,
      modelThreshold: 0.65,
      imuDedupMs: 420,
      audioDebugThreshold: 0.015,
      audioDebugConfidence: 0.55,
    },
  },
  S2: {
    label: 'S2 Strict',
    settings: {
      motionGyroThreshold: 180,
      motionAccelThreshold: 700,
      modelThreshold: 0.8,
      imuDedupMs: 600,
      audioDebugThreshold: 0.03,
      audioDebugConfidence: 0.75,
    },
  },
};

interface Props {
  setup: PlayerSetup;
  calibration: CalibrationData;
  device: Device;
  onDone: () => void;
}

const EMPTY_MOTION_METRICS: StrokeMotionMetrics = {
  gyro_peak: 0,
  gyro_std: 0,
  accel_peak: 0,
  accel_ptp: 0,
};

async function nextExportPath(): Promise<string> {
  const date = new Date().toISOString().slice(0, 10);
  let counter = 1;
  let filePath = `${EXPORT_DIR}/stroke_debug_${date}_${String(counter).padStart(3, '0')}.json`;
  while (await RNFS.exists(filePath)) {
    counter += 1;
    filePath = `${EXPORT_DIR}/stroke_debug_${date}_${String(counter).padStart(3, '0')}.json`;
  }
  return filePath;
}

function labelName(label: StrokeCombinedLabel) {
  switch (label) {
    case 'fh_hit':
      return 'FH HIT';
    case 'bh_hit':
      return 'BH HIT';
    case 'fh_miss':
      return 'FH MISS';
    case 'bh_miss':
      return 'BH MISS';
    default:
      return 'IDLE';
  }
}

function audioLabelName(label: AudioDetectionEvent['label']) {
  switch (label) {
    case 'racket_contact':
      return 'CONTACT';
    case 'not_racket_contact':
      return 'IGNORE';
  }
}

function combineStrokeLabel(
  hitLabel: 'hit' | 'swing_miss',
  side: 'forehand' | 'backhand',
): Exclude<StrokeCombinedLabel, 'idle'> {
  if (hitLabel === 'hit') {
    return side === 'forehand' ? 'fh_hit' : 'bh_hit';
  }
  return side === 'forehand' ? 'fh_miss' : 'bh_miss';
}

function roundedMotionMetrics(metrics: StrokeMotionMetrics): StrokeMotionMetrics {
  return {
    gyro_peak: Number(metrics.gyro_peak.toFixed(1)),
    gyro_std: Number(metrics.gyro_std.toFixed(1)),
    accel_peak: Number(metrics.accel_peak.toFixed(1)),
    accel_ptp: Number(metrics.accel_ptp.toFixed(1)),
  };
}

function formatTopProbabilities(probabilities: Record<string, number>) {
  const entries = Object.entries(probabilities)
    .sort((left, right) => right[1] - left[1])
    .slice(0, 2);

  if (entries.length === 0) return 'n/a';

  return entries
    .map(([label, probability]) => `${label} ${Math.round(probability * 100)}%`)
    .join(' / ');
}

function ignoredReasonText(reason: StrokeInferenceEvent['ignored_reason']) {
  switch (reason) {
    case 'idle_motion_gate':
      return 'idle_motion_gate';
    case 'model_low_confidence':
      return 'model_low_confidence';
    case 'side_uncertain':
      return 'side_uncertain';
    case 'dedup':
      return 'dedup';
    default:
      return 'counted';
  }
}

export function StrokeTestScreen({ setup, calibration, device, onDone }: Props) {
  const [presetId, setPresetId] = useState<StrokePresetId>('S0');
  const [settings, setSettings] = useState<StrokeSettings>(STROKE_PRESETS.S0.settings);
  const [isRunning, setIsRunning] = useState(false);
  const [isConnected, setIsConnected] = useState(true);
  const [micGranted, setMicGranted] = useState(Platform.OS !== 'android');
  const [sampleHz, setSampleHz] = useState(0);
  const [showIgnored, setShowIgnored] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);
  const [latestLabel, setLatestLabel] = useState('IDLE');
  const [motionGateStatus, setMotionGateStatus] = useState<'OPEN' | 'CLOSED'>('CLOSED');
  const [latestMotionMetrics, setLatestMotionMetrics] = useState<StrokeMotionMetrics>(EMPTY_MOTION_METRICS);

  const [fhHitCount, setFhHitCount] = useState(0);
  const [bhHitCount, setBhHitCount] = useState(0);
  const [fhMissCount, setFhMissCount] = useState(0);
  const [bhMissCount, setBhMissCount] = useState(0);
  const [idleGatedWindows, setIdleGatedWindows] = useState(0);
  const [motionWindows, setMotionWindows] = useState(0);
  const [countedEvents, setCountedEvents] = useState(0);

  const [recentStrokeEvents, setRecentStrokeEvents] = useState<StrokeInferenceEvent[]>([]);
  const [recentAudioEvents, setRecentAudioEvents] = useState<AudioDetectionEvent[]>([]);

  const sessionStartRef = useRef<number | null>(null);
  const isRunningRef = useRef(false);
  const settingsRef = useRef(settings);
  const sampleCountRef = useRef(0);
  const hzTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastInferenceAtRef = useRef(0);
  const lastCountedAtRef = useRef(0);
  const lastAudioTsRef = useRef<Partial<Record<AudioDetectionEvent['label'], number>>>({});

  const latestRef = useRef({
    accel: { x: 0, y: 0, z: 0 },
    gyro: { x: 0, y: 0, z: 0 },
    mag: { x: 0, y: 0, z: 0 },
  });

  const sampleLogRef = useRef<ImuSample[]>([]);
  const audioLogRef = useRef<AudioDetectionEvent[]>([]);
  const strokeLogRef = useRef<StrokeInferenceEvent[]>([]);

  useEffect(() => {
    isRunningRef.current = isRunning;
  }, [isRunning]);

  useEffect(() => {
    settingsRef.current = settings;
    if (isRunning) {
      AudioStream.setThreshold(settings.audioDebugThreshold).catch(() => {});
    }
  }, [isRunning, settings]);

  const requestMicPermission = useCallback(async () => {
    if (Platform.OS !== 'android') {
      setMicGranted(true);
      return true;
    }

    const result = await PermissionsAndroid.request(
      PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
      {
        title: 'Microphone access',
        message: 'Stroke debug uses the microphone only as optional audio context.',
        buttonPositive: 'OK',
      },
    );
    const granted = result === PermissionsAndroid.RESULTS.GRANTED;
    setMicGranted(granted);
    return granted;
  }, []);

  useEffect(() => {
    requestMicPermission().catch(() => {});
  }, [requestMicPermission]);

  const resetRun = useCallback(() => {
    sessionStartRef.current = null;
    sampleLogRef.current = [];
    audioLogRef.current = [];
    strokeLogRef.current = [];
    lastInferenceAtRef.current = 0;
    lastCountedAtRef.current = 0;
    lastAudioTsRef.current = {};
    setFhHitCount(0);
    setBhHitCount(0);
    setFhMissCount(0);
    setBhMissCount(0);
    setIdleGatedWindows(0);
    setMotionWindows(0);
    setCountedEvents(0);
    setRecentStrokeEvents([]);
    setRecentAudioEvents([]);
    setLatestLabel('IDLE');
    setMotionGateStatus('CLOSED');
    setLatestMotionMetrics(EMPTY_MOTION_METRICS);
    setFeedback(null);
  }, []);

  const requestAudioSupport = useCallback((tsMs: number) => {
    let best: AudioDetectionEvent | null = null;
    let bestDelta = Number.MAX_SAFE_INTEGER;
    for (const event of audioLogRef.current) {
      if (!event.qualified || event.label !== 'racket_contact') continue;
      const delta = Math.abs(event.ts_ms - tsMs);
      if (delta <= AUDIO_MATCH_MS && delta < bestDelta) {
        best = event;
        bestDelta = delta;
      }
    }

    if (!best) return null;
    return {
      label: best.label,
      confidence: best.confidence,
      delta_ms: bestDelta,
    };
  }, []);

  const appendStrokeEvent = useCallback((event: StrokeInferenceEvent) => {
    strokeLogRef.current.push(event);
    setRecentStrokeEvents(prev => [event, ...prev].slice(0, 14));
    setLatestLabel(labelName(event.label));
    setMotionGateStatus(event.motion_gate_open ? 'OPEN' : 'CLOSED');
    setLatestMotionMetrics(event.motion_metrics);

    if (!event.motion_gate_open) {
      setIdleGatedWindows(prev => prev + 1);
      return;
    }

    setMotionWindows(prev => prev + 1);
    if (!event.counted) return;

    setCountedEvents(prev => prev + 1);
    if (event.label === 'fh_hit') setFhHitCount(prev => prev + 1);
    if (event.label === 'bh_hit') setBhHitCount(prev => prev + 1);
    if (event.label === 'fh_miss') setFhMissCount(prev => prev + 1);
    if (event.label === 'bh_miss') setBhMissCount(prev => prev + 1);
  }, []);

  const appendAudioEvent = useCallback((event: AudioDetectionEvent) => {
    audioLogRef.current.push(event);
    setRecentAudioEvents(prev => [event, ...prev].slice(0, 14));
  }, []);

  const inferCurrentWindow = useCallback((now: number) => {
    const window = buildStrokeWindow(sampleLogRef.current);
    if (!window || window.length < WINDOW_SAMPLES) return;

    const motionMetrics = roundedMotionMetrics(extractStrokeMotionMetrics(window));
    const motionGateOpen =
      motionMetrics.gyro_peak >= settingsRef.current.motionGyroThreshold ||
      motionMetrics.accel_peak >= settingsRef.current.motionAccelThreshold;
    const audioSupport = requestAudioSupport(now);

    if (!motionGateOpen) {
      appendStrokeEvent({
        detected_at: new Date(now).toISOString(),
        ts_ms: now,
        label: 'idle',
        motion_gate_open: false,
        motion_metrics: motionMetrics,
        hit_label: 'idle',
        hit_confidence: 0,
        hit_probabilities: {},
        stroke_side: 'uncertain',
        stroke_confidence: 0,
        stroke_probabilities: {},
        counted: false,
        ignored_reason: 'idle_motion_gate',
        audio_support: audioSupport,
      });
      return;
    }

    const features = extractStrokeFeatures(window);
    const prediction = predictStroke(features);

    let label: StrokeCombinedLabel = 'idle';
    let counted = false;
    let ignoredReason: StrokeInferenceEvent['ignored_reason'];

    if (prediction.hit_confidence < settingsRef.current.modelThreshold) {
      ignoredReason = 'model_low_confidence';
    } else if (
      prediction.stroke_side === 'uncertain' ||
      prediction.stroke_confidence < settingsRef.current.modelThreshold
    ) {
      ignoredReason = 'side_uncertain';
    } else {
      label = combineStrokeLabel(
        prediction.hit_label === 'hit' ? 'hit' : 'swing_miss',
        prediction.stroke_side,
      );

      if (now - lastCountedAtRef.current < settingsRef.current.imuDedupMs) {
        ignoredReason = 'dedup';
      } else {
        counted = true;
        lastCountedAtRef.current = now;
      }
    }

    const event: StrokeInferenceEvent = {
      detected_at: new Date(now).toISOString(),
      ts_ms: now,
      label,
      motion_gate_open: true,
      motion_metrics: motionMetrics,
      hit_label: prediction.hit_label,
      hit_confidence: prediction.hit_confidence,
      hit_probabilities: prediction.hit_probabilities,
      stroke_side: prediction.stroke_side,
      stroke_confidence: prediction.stroke_confidence,
      stroke_probabilities: prediction.stroke_probabilities,
      counted,
      ignored_reason: counted ? undefined : ignoredReason,
      audio_support: audioSupport,
    };

    appendStrokeEvent(event);
    if (counted) {
      setFeedback(
        `Gate OPEN · ${labelName(label)} · hit ${Math.round(event.hit_confidence * 100)}% · side ${Math.round(event.stroke_confidence * 100)}%`,
      );
    } else if (ignoredReason) {
      setFeedback(`Gate OPEN · ${ignoredReasonText(ignoredReason)}`);
    }
  }, [appendStrokeEvent, requestAudioSupport]);

  const handleAudioDetected = useCallback((audioB64: string) => {
    if (!isRunningRef.current) return;

    const detectedAt = Date.now();
    try {
      const pcm = decodeBase64PCM(audioB64);
      const event = detectAudioContact({
        detectedAtMs: detectedAt,
        pcm,
        confidenceThreshold: settingsRef.current.audioDebugConfidence,
        dedupMs: AUDIO_DEDUP_MS,
        lastQualifiedTsMs: lastAudioTsRef.current.racket_contact,
      });

      if (event.qualified) {
        lastAudioTsRef.current.racket_contact = detectedAt;
      }

      appendAudioEvent(event);
    } catch (error: any) {
      setFeedback(`Audio debug failed: ${error?.message ?? 'unknown error'}`);
    }
  }, [appendAudioEvent]);

  const handleNotification = useCallback(
    (_error: BleError | null, characteristic: Characteristic | null) => {
      if (!characteristic?.value || !characteristic.uuid) return;
      const parsed = parsePacket(characteristic.uuid, characteristic.value);
      if (!parsed) return;

      const latest = latestRef.current;
      if (parsed.type === 'accel') latest.accel = parsed;
      else if (parsed.type === 'gyro') latest.gyro = parsed;
      else latest.mag = parsed;

      sampleCountRef.current += 1;
      if (!isRunningRef.current) return;

      const now = Date.now();
      const takeTsMs = sessionStartRef.current === null ? 0 : Math.max(0, now - sessionStartRef.current);
      sampleLogRef.current.push({
        accel_x: latest.accel.x - calibration.gravity.x,
        accel_y: latest.accel.y - calibration.gravity.y,
        accel_z: latest.accel.z - calibration.gravity.z,
        gyro_x: latest.gyro.x,
        gyro_y: latest.gyro.y,
        gyro_z: latest.gyro.z,
        mag_x: latest.mag.x,
        mag_y: latest.mag.y,
        mag_z: latest.mag.z,
        ts_ms: now,
        received_at_ms: now,
        take_ts_ms: takeTsMs,
        sensor_ts: parsed.sensor_ts,
      });

      if (now - lastInferenceAtRef.current < STEP_MS) return;
      lastInferenceAtRef.current = now;
      inferCurrentWindow(now);
    },
    [calibration.gravity.x, calibration.gravity.y, calibration.gravity.z, inferCurrentWindow],
  );

  useEffect(() => {
    for (const uuid of [ACCEL_UUID, ACCEL_UUID_ALT, GYRO_UUID, MAG_UUID]) {
      try {
        device.monitorCharacteristicForService(SERVICE_UUID, uuid, handleNotification);
      } catch (_) {}
    }

    let lastCount = 0;
    hzTimerRef.current = setInterval(() => {
      setSampleHz(sampleCountRef.current - lastCount);
      lastCount = sampleCountRef.current;
    }, 1000);

    const disconnectSub = device.onDisconnected(() => {
      setIsConnected(false);
      setFeedback('Sensor disconnected.');
      if (hzTimerRef.current) clearInterval(hzTimerRef.current);
    });

    return () => {
      if (hzTimerRef.current) clearInterval(hzTimerRef.current);
      disconnectSub.remove();
    };
  }, [device, handleNotification]);

  useEffect(() => {
    const sub = AudioStreamEmitter.addListener('onBounceDetected', handleAudioDetected);
    return () => {
      sub.remove();
      AudioStream.stopStreaming().catch(() => {});
    };
  }, [handleAudioDetected]);

  const startRun = useCallback(async () => {
    if (!isConnected) {
      Alert.alert('No sensor', 'Reconnect the AirHive sensor before starting.');
      return;
    }
    if (!(await requestMicPermission())) {
      Alert.alert('Microphone missing', 'Grant microphone access before starting.');
      return;
    }

    resetRun();
    sessionStartRef.current = Date.now();
    try {
      await AudioStream.startStreaming(settingsRef.current.audioDebugThreshold);
      setIsRunning(true);
      setFeedback('Stroke debug running.');
    } catch (error: any) {
      setFeedback(`Could not start audio debug: ${error?.message ?? 'unknown error'}`);
    }
  }, [isConnected, requestMicPermission, resetRun]);

  const stopRun = useCallback(async () => {
    try {
      await AudioStream.stopStreaming();
    } catch (_) {}
    setIsRunning(false);
    setFeedback('Stroke debug stopped.');
  }, []);

  const saveSession = useCallback(async () => {
    if (isRunning) {
      Alert.alert('Stop first', 'Stop the current pass before saving.');
      return;
    }
    if (sessionStartRef.current === null || strokeLogRef.current.length === 0) {
      Alert.alert('No data', 'Run at least one pass before saving.');
      return;
    }

    try {
      await RNFS.mkdir(EXPORT_DIR);
      const filePath = await nextExportPath();
      const session: StrokeTestSessionFile = {
        session_meta: {
          player_name: setup.name,
          handedness: setup.handedness,
          mode: 'stroke_debug',
          session_date: new Date(sessionStartRef.current).toISOString(),
          duration_ms: Date.now() - sessionStartRef.current,
          app_version: APP_VERSION,
        },
        calibration_profile: calibration,
        calibration_summary: {
          table_ready: true,
          bounce_sides_ready: !!calibration.bounce_sides,
        },
        preset_id: presetId,
        settings: settingsRef.current,
        samples: sampleLogRef.current,
        audio_events: audioLogRef.current,
        stroke_events: strokeLogRef.current,
        summary: {
          fh_hit_count: fhHitCount,
          bh_hit_count: bhHitCount,
          fh_miss_count: fhMissCount,
          bh_miss_count: bhMissCount,
          idle_gated_windows: idleGatedWindows,
          motion_windows: motionWindows,
          counted_events: countedEvents,
        },
      };

      await RNFS.writeFile(filePath, JSON.stringify(session, null, 2), 'utf8');
      try { await RNFS.scanFile(filePath); } catch (_) {}

      Alert.alert('Export saved', `File: Download/pingis_sessions/${filePath.split('/').pop()}`);
      setFeedback(`Export saved: ${filePath.split('/').pop()}`);
    } catch (error: any) {
      Alert.alert('Error', `Could not save session: ${error?.message ?? 'unknown error'}`);
    }
  }, [
    bhHitCount,
    bhMissCount,
    calibration,
    countedEvents,
    fhHitCount,
    fhMissCount,
    idleGatedWindows,
    isRunning,
    motionWindows,
    presetId,
    setup.handedness,
    setup.name,
  ]);

  const leaveScreen = useCallback(async () => {
    try { await AudioStream.stopStreaming(); } catch (_) {}
    try { await device.cancelConnection(); } catch (_) {}
    onDone();
  }, [device, onDone]);

  const visibleAudioEvents = showIgnored
    ? recentAudioEvents
    : recentAudioEvents.filter(event => event.qualified);

  return (
    <ScrollView style={styles.root} contentContainerStyle={styles.content}>
      <StatusBar barStyle="light-content" backgroundColor="#0d0d0d" />

      <View style={styles.header}>
        <TouchableOpacity onPress={leaveScreen} style={styles.backBtn}>
          <Text style={styles.backTxt}>Back</Text>
        </TouchableOpacity>
        <View style={styles.headerMain}>
          <Text style={styles.headerTitle}>Stroke debug</Text>
          <Text style={styles.headerSub}>
            {setup.name} · {setup.handedness === 'right' ? 'Right' : 'Left'} hand · {presetId} · debug only
          </Text>
        </View>
      </View>

      <View style={styles.infoCard}>
        <Text style={styles.infoTitle}>Debug mode, not score</Text>
        <Text style={styles.infoText}>
          This screen is for motion-gate tuning and data capture. Stillness should stay at 0 counted events.
        </Text>
      </View>

      <View style={styles.heroCard}>
        <Text style={styles.heroLabel}>LATEST DEBUG LABEL</Text>
        <Text style={styles.heroValue}>{latestLabel}</Text>
        <Text style={styles.heroMeta}>
          Sensor {isConnected ? `${sampleHz} Hz` : 'offline'} · Mic {micGranted ? 'ready' : 'missing'}
        </Text>
        <View style={styles.motionBadgeRow}>
          <View style={[styles.motionBadge, motionGateStatus === 'OPEN' ? styles.motionBadgeOpen : styles.motionBadgeClosed]}>
            <Text style={[styles.motionBadgeTxt, motionGateStatus === 'OPEN' ? styles.motionBadgeTxtOpen : styles.motionBadgeTxtClosed]}>
              MOTION GATE {motionGateStatus}
            </Text>
          </View>
        </View>
        <Text style={styles.heroMetrics}>
          Gyro peak {latestMotionMetrics.gyro_peak} / {settings.motionGyroThreshold} · Accel peak {latestMotionMetrics.accel_peak} / {settings.motionAccelThreshold}
        </Text>
      </View>

      <View style={styles.toggleRow}>
        <TouchableOpacity
          style={[styles.toggleBtn, isRunning ? styles.toggleStop : styles.toggleStart]}
          onPress={isRunning ? stopRun : startRun}
        >
          <Text style={[styles.toggleTxt, isRunning ? styles.toggleTxtStop : styles.toggleTxtStart]}>
            {isRunning ? 'STOP' : 'START'}
          </Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.toggleBtn, styles.secondaryBtn]} onPress={saveSession}>
          <Text style={styles.secondaryTxt}>SAVE EXPORT</Text>
        </TouchableOpacity>
      </View>

      {feedback && <Text style={styles.feedback}>{feedback}</Text>}

      <View style={styles.summaryCard}>
        <Text style={styles.sectionLabel}>COUNTED EVENTS</Text>
        <View style={styles.summaryGrid}>
          <Metric label="FH HIT" value={fhHitCount} accent="#2ecc71" />
          <Metric label="BH HIT" value={bhHitCount} accent="#4a9eff" />
          <Metric label="FH MISS" value={fhMissCount} accent="#ffb36b" />
          <Metric label="BH MISS" value={bhMissCount} accent="#ff7f7f" />
        </View>
        <Text style={styles.summaryHint}>
          Idle gated: {idleGatedWindows} · Motion windows: {motionWindows} · Counted: {countedEvents}
        </Text>
      </View>

      <View style={styles.presetCard}>
        <Text style={styles.sectionLabel}>PRESETS</Text>
        <View style={styles.presetRow}>
          {(Object.keys(STROKE_PRESETS) as StrokePresetId[]).map(id => (
            <TouchableOpacity
              key={id}
              style={[styles.presetBtn, presetId === id && styles.presetBtnOn]}
              onPress={() => {
                setPresetId(id);
                setSettings(STROKE_PRESETS[id].settings);
              }}
              disabled={isRunning}
            >
              <Text style={[styles.presetTxt, presetId === id && styles.presetTxtOn]}>{id}</Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      <View style={styles.settingsCard}>
        <Text style={styles.sectionLabel}>LIVE SETTINGS</Text>
        <DebugSlider
          label="Motion gyro threshold"
          value={settings.motionGyroThreshold}
          min={40}
          max={320}
          step={10}
          onChange={value => setSettings(prev => ({ ...prev, motionGyroThreshold: value }))}
          valueFormatter={value => `${Math.round(value)}`}
          leftHint="40"
          rightHint="320"
        />
        <DebugSlider
          label="Motion accel threshold"
          value={settings.motionAccelThreshold}
          min={150}
          max={1400}
          step={50}
          onChange={value => setSettings(prev => ({ ...prev, motionAccelThreshold: value }))}
          valueFormatter={value => `${Math.round(value)}`}
          leftHint="150"
          rightHint="1400"
        />
        <DebugSlider
          label="Model confidence"
          value={settings.modelThreshold}
          min={0.5}
          max={0.95}
          step={0.05}
          onChange={value => setSettings(prev => ({ ...prev, modelThreshold: value }))}
          valueFormatter={value => `${Math.round(value * 100)}%`}
          leftHint="50%"
          rightHint="95%"
        />
        <DebugSlider
          label="IMU dedup"
          value={settings.imuDedupMs}
          min={250}
          max={900}
          step={50}
          onChange={value => setSettings(prev => ({ ...prev, imuDedupMs: value }))}
          valueFormatter={value => `${Math.round(value)} ms`}
          leftHint="250"
          rightHint="900"
        />
        <DebugSlider
          label="Audio onset"
          value={settings.audioDebugThreshold}
          min={0.005}
          max={0.06}
          step={0.005}
          onChange={value => setSettings(prev => ({ ...prev, audioDebugThreshold: value }))}
          valueFormatter={value => value.toFixed(3)}
          leftHint="0.005"
          rightHint="0.060"
        />
        <DebugSlider
          label="Audio confidence"
          value={settings.audioDebugConfidence}
          min={0.4}
          max={0.9}
          step={0.05}
          onChange={value => setSettings(prev => ({ ...prev, audioDebugConfidence: value }))}
          valueFormatter={value => `${Math.round(value * 100)}%`}
          leftHint="40%"
          rightHint="90%"
        />
      </View>

      <View style={styles.debugCard}>
        <Text style={styles.sectionLabel}>STROKE DEBUG</Text>
        {recentStrokeEvents.length === 0 && <Text style={styles.emptyTxt}>No stroke windows yet.</Text>}
        {recentStrokeEvents.map((event, index) => (
          <Text key={`${event.ts_ms}-${index}`} style={styles.debugRow}>
            {formatClock(event.ts_ms)} · gate {event.motion_gate_open ? 'OPEN' : 'CLOSED'} ·
            {' '}gyro {event.motion_metrics.gyro_peak} · accel {event.motion_metrics.accel_peak} ·
            {' '}{labelName(event.label)} · H {formatTopProbabilities(event.hit_probabilities)} ·
            {' '}S {formatTopProbabilities(event.stroke_probabilities)} ·
            {' '}{event.counted ? 'counted' : ignoredReasonText(event.ignored_reason)}
          </Text>
        ))}
      </View>

      <View style={styles.debugCard}>
        <View style={styles.debugHeader}>
          <Text style={styles.sectionLabel}>AUDIO DEBUG</Text>
          <TouchableOpacity onPress={() => setShowIgnored(prev => !prev)}>
            <Text style={styles.linkTxt}>{showIgnored ? 'Hide ignored' : 'Show ignored'}</Text>
          </TouchableOpacity>
        </View>
        {visibleAudioEvents.length === 0 && <Text style={styles.emptyTxt}>No audio events in current view.</Text>}
        {visibleAudioEvents.map((event, index) => (
          <Text key={`${event.ts_ms}-${index}`} style={styles.debugRow}>
            {formatClock(event.ts_ms)} · {audioLabelName(event.label)} · {Math.round(event.confidence * 100)}%
            {' '}· {event.qualified ? 'ok' : event.ignored_reason}
          </Text>
        ))}
      </View>

      <View style={styles.fileCard}>
        <Text style={styles.fileTitle}>Test loop</Text>
        <Text style={styles.fileText}>
          1. Pick a preset and start a pass.{'\n'}
          2. Verify stillness first, then try miss and hit motions.{'\n'}
          3. Save the export to Downloads/pingis_sessions.{'\n'}
          4. Fill the matching testcase rows in TEST_PLAN.md.
        </Text>
      </View>
    </ScrollView>
  );
}

function Metric({ label, value, accent }: { label: string; value: number; accent: string }) {
  return (
    <View style={styles.metricItem}>
      <Text style={[styles.metricValue, { color: accent }]}>{value}</Text>
      <Text style={styles.metricLabel}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#0d0d0d' },
  content: { padding: 20, paddingBottom: 48 },
  header: { flexDirection: 'row', alignItems: 'center', gap: 12, marginBottom: 18 },
  backBtn: { paddingVertical: 6, paddingRight: 8 },
  backTxt: { color: '#4a9eff', fontSize: 14 },
  headerMain: { flex: 1 },
  headerTitle: { color: '#fff', fontSize: 24, fontWeight: '800' },
  headerSub: { color: '#777', fontSize: 12, marginTop: 2 },
  infoCard: {
    backgroundColor: '#111',
    borderRadius: 16,
    padding: 16,
    borderWidth: 1,
    borderColor: '#1b3557',
    marginBottom: 16,
  },
  infoTitle: { color: '#4a9eff', fontSize: 14, fontWeight: '700', marginBottom: 6 },
  infoText: { color: '#95a4b5', fontSize: 12, lineHeight: 18 },
  heroCard: {
    backgroundColor: '#111',
    borderRadius: 18,
    padding: 18,
    alignItems: 'center',
  },
  heroLabel: { color: '#666', fontSize: 10, letterSpacing: 2 },
  heroValue: { color: '#f5c76d', fontSize: 38, fontWeight: '900', marginTop: 8 },
  heroMeta: { color: '#888', fontSize: 12, marginTop: 8 },
  motionBadgeRow: { marginTop: 10 },
  motionBadge: {
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  motionBadgeOpen: { backgroundColor: '#0d2d1a' },
  motionBadgeClosed: { backgroundColor: '#2a1812' },
  motionBadgeTxt: { fontSize: 11, fontWeight: '800', letterSpacing: 1.2 },
  motionBadgeTxtOpen: { color: '#2ecc71' },
  motionBadgeTxtClosed: { color: '#ffb36b' },
  heroMetrics: { color: '#888', fontSize: 12, marginTop: 10, textAlign: 'center' },
  toggleRow: { flexDirection: 'row', gap: 12, marginTop: 14 },
  toggleBtn: {
    flex: 1,
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: 'center',
  },
  toggleStart: { backgroundColor: '#0d2d1a' },
  toggleStop: { backgroundColor: '#2d0d0d' },
  toggleTxt: { fontSize: 14, fontWeight: '800', letterSpacing: 1 },
  toggleTxtStart: { color: '#2ecc71' },
  toggleTxtStop: { color: '#ff7f7f' },
  secondaryBtn: { backgroundColor: '#0d1f33' },
  secondaryTxt: { color: '#4a9eff', fontSize: 14, fontWeight: '800', letterSpacing: 1 },
  feedback: { color: '#f5c76d', fontSize: 13, textAlign: 'center', marginTop: 10 },
  summaryCard: {
    marginTop: 16,
    backgroundColor: '#111',
    borderRadius: 16,
    padding: 16,
  },
  sectionLabel: { color: '#777', fontSize: 10, letterSpacing: 2, marginBottom: 12 },
  summaryGrid: { flexDirection: 'row', justifyContent: 'space-between', gap: 10, flexWrap: 'wrap' },
  metricItem: {
    width: '48%',
    backgroundColor: '#171717',
    borderRadius: 12,
    paddingVertical: 16,
    alignItems: 'center',
    marginBottom: 10,
  },
  metricValue: { fontSize: 28, fontWeight: '800', fontFamily: 'monospace' },
  metricLabel: { color: '#666', fontSize: 10, marginTop: 4, letterSpacing: 1.5 },
  summaryHint: { color: '#888', fontSize: 12, lineHeight: 18 },
  presetCard: {
    marginTop: 16,
    backgroundColor: '#111',
    borderRadius: 16,
    padding: 16,
  },
  presetRow: { flexDirection: 'row', gap: 10 },
  presetBtn: {
    flex: 1,
    backgroundColor: '#171717',
    borderRadius: 10,
    paddingVertical: 12,
    alignItems: 'center',
  },
  presetBtnOn: { backgroundColor: '#2b220d', borderWidth: 1, borderColor: '#f5c76d' },
  presetTxt: { color: '#777', fontSize: 12, fontWeight: '700' },
  presetTxtOn: { color: '#f5c76d' },
  settingsCard: {
    marginTop: 16,
    backgroundColor: '#111',
    borderRadius: 16,
    padding: 16,
  },
  debugCard: {
    marginTop: 16,
    backgroundColor: '#111',
    borderRadius: 16,
    padding: 16,
  },
  debugHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  emptyTxt: { color: '#666', fontSize: 12 },
  debugRow: {
    color: '#c9c9c9',
    fontSize: 12,
    marginBottom: 8,
    fontFamily: 'monospace',
  },
  linkTxt: { color: '#f5c76d', fontSize: 12 },
  fileCard: {
    marginTop: 16,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#242424',
    padding: 16,
  },
  fileTitle: { color: '#fff', fontSize: 15, fontWeight: '700', marginBottom: 8 },
  fileText: { color: '#888', fontSize: 12, lineHeight: 18 },
});
