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
import { getAudioDetectionConfig } from './audioDetectionConfig';
import { detectAudioContact } from './audioContactEngine';
import {
  ACCEL_UUID,
  ACCEL_UUID_ALT,
  GYRO_UUID,
  MAG_UUID,
  SERVICE_UUID,
  averageVector,
  dot,
  formatClock,
  normalize,
  parsePacket,
} from './airhive';
import { DebugSlider } from './DebugSlider';
import type {
  AudioDetectionEvent,
  BounceContactEvent,
  BounceMotionMetrics,
  BouncePresetId,
  BounceSettings,
  BounceSide,
  BounceSideEvent,
  BounceTestSessionFile,
  CalibrationData,
  ImuSample,
  PlayerSetup,
  Vector3,
} from './types';

const APP_VERSION = '1.7';
const EXPORT_DIR = `${RNFS.ExternalStorageDirectoryPath}/Download/pingis_sessions`;

const DEFAULT_BOUNCE_PRESET_ID: BouncePresetId = 'B0';
const DEFAULT_BOUNCE_PRESET_LABEL = 'B0 Default';
const DEFAULT_AUDIO_CONFIG = getAudioDetectionConfig('normal', 'four_class_only');
const DEFAULT_BOUNCE_SETTINGS: BounceSettings = {
  audioThreshold: DEFAULT_AUDIO_CONFIG.onset_threshold,
  audioRetriggerMs: 220,
  audioGroupWindowMs: 80,
  audioConfidence: DEFAULT_AUDIO_CONFIG.contact_confidence_min,
  audioDedupMs: DEFAULT_AUDIO_CONFIG.merge_window_ms,
  motionWindowMs: 240,
  motionGyroThreshold: 45,
  motionAccelThreshold: 180,
  orientationSampleWindowMs: 160,
  orientationDeadzone: 0.1,
};

interface Props {
  setup: PlayerSetup;
  calibration: CalibrationData;
  device: Device;
  mode: 'bounce_free' | 'bounce_alternating';
  onDone: () => void;
}

function nextFileName(prefix: string, date: string, counter: number): string {
  return `${EXPORT_DIR}/${prefix}_${date}_${String(counter).padStart(3, '0')}.json`;
}

async function nextExportPath(prefix: string): Promise<string> {
  const date = new Date().toISOString().slice(0, 10);
  let counter = 1;
  let path = nextFileName(prefix, date, counter);
  while (await RNFS.exists(path)) {
    counter += 1;
    path = nextFileName(prefix, date, counter);
  }
  return path;
}

function modeLabel(mode: 'bounce_free' | 'bounce_alternating') {
  return mode === 'bounce_free' ? 'Studs fritt' : 'Studs vaxla sida';
}

function sideLabel(side: BounceSide) {
  switch (side) {
    case 'forehand':
      return 'FH-side';
    case 'backhand':
      return 'BH-side';
    default:
      return 'Uncertain';
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

function surfaceLabelName(label?: AudioDetectionEvent['surface_label']) {
  switch (label) {
    case 'racket_bounce':
      return 'RACKET';
    case 'table_bounce':
      return 'TABLE';
    case 'floor_bounce':
      return 'FLOOR';
    case 'noise':
      return 'NOISE';
    default:
      return '-';
  }
}

function reasonLabel(reason?: AudioDetectionEvent['ignored_reason'] | BounceContactEvent['ignored_reason']) {
  if (reason === 'dedup') {
    return 'merge_window';
  }
  return reason ?? 'counted';
}

function classifyBounceSide(
  orientation: Vector3,
  calibration: CalibrationData,
  deadzone: number,
): { side: BounceSide; forehandScore: number; backhandScore: number } {
  if (!calibration.bounce_sides) {
    return { side: 'uncertain', forehandScore: 0, backhandScore: 0 };
  }

  const current = normalize(orientation);
  const forehand = normalize(calibration.bounce_sides.forehand.pose_accel);
  const backhand = normalize(calibration.bounce_sides.backhand.pose_accel);

  const forehandScore = dot(current, forehand);
  const backhandScore = dot(current, backhand);
  if (Math.abs(forehandScore - backhandScore) < deadzone) {
    return { side: 'uncertain', forehandScore, backhandScore };
  }

  return {
    side: forehandScore >= backhandScore ? 'forehand' : 'backhand',
    forehandScore,
    backhandScore,
  };
}

function findOrientationAround(
  samples: ImuSample[],
  tsMs: number,
  windowMs: number,
): Vector3 | null {
  const halfWindow = windowMs / 2;
  const relevant = samples.filter(sample => Math.abs(sample.ts_ms - tsMs) <= halfWindow);
  if (relevant.length === 0) return null;
  return averageVector(
    relevant.map(sample => ({
      x: sample.accel_x,
      y: sample.accel_y,
      z: sample.accel_z,
    })),
  );
}

function emptyMotionMetrics(): BounceMotionMetrics {
  return {
    gyro_peak: 0,
    accel_peak: 0,
    accel_ptp: 0,
  };
}

function roundMotionMetrics(metrics: BounceMotionMetrics): BounceMotionMetrics {
  return {
    gyro_peak: Number(metrics.gyro_peak.toFixed(1)),
    accel_peak: Number(metrics.accel_peak.toFixed(1)),
    accel_ptp: Number(metrics.accel_ptp.toFixed(1)),
  };
}

function findBounceMotionMetricsAround(
  samples: ImuSample[],
  tsMs: number,
  windowMs: number,
): BounceMotionMetrics {
  const halfWindow = windowMs / 2;
  const relevant = samples.filter(sample => Math.abs(sample.ts_ms - tsMs) <= halfWindow);
  if (relevant.length === 0) return emptyMotionMetrics();

  const accelMagnitude = relevant.map(sample =>
    Math.sqrt(sample.accel_x ** 2 + sample.accel_y ** 2 + sample.accel_z ** 2),
  );
  const gyroMagnitude = relevant.map(sample =>
    Math.sqrt(sample.gyro_x ** 2 + sample.gyro_y ** 2 + sample.gyro_z ** 2),
  );

  const accelPeak = Math.max(...accelMagnitude);
  const accelMin = Math.min(...accelMagnitude);

  return {
    gyro_peak: Math.max(...gyroMagnitude),
    accel_peak: accelPeak,
    accel_ptp: accelPeak - accelMin,
  };
}

export function BounceTestScreen({ setup, calibration, device, mode, onDone }: Props) {
  const presetId = DEFAULT_BOUNCE_PRESET_ID;
  const [settings, setSettings] = useState<BounceSettings>(DEFAULT_BOUNCE_SETTINGS);
  const [isRunning, setIsRunning] = useState(false);
  const [isConnected, setIsConnected] = useState(true);
  const [micGranted, setMicGranted] = useState(Platform.OS !== 'android');
  const [sampleHz, setSampleHz] = useState(0);
  const [showIgnored, setShowIgnored] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);

  const [totalCount, setTotalCount] = useState(0);
  const [forehandCount, setForehandCount] = useState(0);
  const [backhandCount, setBackhandCount] = useState(0);
  const [uncertainCount, setUncertainCount] = useState(0);
  const [alternationCount, setAlternationCount] = useState(0);
  const [motionClosedCount, setMotionClosedCount] = useState(0);

  const [recentAudioEvents, setRecentAudioEvents] = useState<AudioDetectionEvent[]>([]);
  const [recentContacts, setRecentContacts] = useState<BounceContactEvent[]>([]);

  const sampleCountRef = useRef(0);
  const hzTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const sessionStartRef = useRef<number | null>(null);
  const isRunningRef = useRef(false);
  const settingsRef = useRef(settings);
  const lastQualifiedLabelTsRef = useRef<Partial<Record<AudioDetectionEvent['label'], number>>>({});
  const lastConfirmedSideRef = useRef<BounceSide | null>(null);
  const contactGroupRef = useRef<{ id: number; startedAtMs: number } | null>(null);
  const groupCounterRef = useRef(0);

  const latestRef = useRef({
    accel: { x: 0, y: 0, z: 0 },
    gyro: { x: 0, y: 0, z: 0 },
    mag: { x: 0, y: 0, z: 0 },
  });

  const sampleLogRef = useRef<ImuSample[]>([]);
  const audioLogRef = useRef<AudioDetectionEvent[]>([]);
  const sideLogRef = useRef<BounceSideEvent[]>([]);
  const contactLogRef = useRef<BounceContactEvent[]>([]);

  useEffect(() => {
    isRunningRef.current = isRunning;
  }, [isRunning]);

  useEffect(() => {
    settingsRef.current = settings;
    if (isRunning) {
      AudioStream.setThreshold(settings.audioThreshold).catch(() => {});
      AudioStream.setRetriggerMs(settings.audioRetriggerMs).catch(() => {});
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
        message: 'Studs-test needs microphone access for racket bounce detection.',
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

  const syncSummary = useCallback(() => {
    const contacts = contactLogRef.current.filter(contact => contact.counted);
    setTotalCount(contacts.length);
    setForehandCount(contacts.filter(contact => contact.side === 'forehand').length);
    setBackhandCount(contacts.filter(contact => contact.side === 'backhand').length);
    setUncertainCount(contacts.filter(contact => contact.side === 'uncertain').length);
    setAlternationCount(contacts.length > 0 ? contacts[contacts.length - 1].alternation_after : 0);
    setMotionClosedCount(
      contactLogRef.current.filter(contact => contact.motion_gate_open === false).length,
    );
  }, []);

  const resetRun = useCallback(() => {
    sessionStartRef.current = null;
    sampleLogRef.current = [];
    audioLogRef.current = [];
    sideLogRef.current = [];
    contactLogRef.current = [];
    lastQualifiedLabelTsRef.current = {};
    lastConfirmedSideRef.current = null;
    contactGroupRef.current = null;
    groupCounterRef.current = 0;
    setTotalCount(0);
    setForehandCount(0);
    setBackhandCount(0);
    setUncertainCount(0);
    setAlternationCount(0);
    setMotionClosedCount(0);
    setRecentAudioEvents([]);
    setRecentContacts([]);
    setFeedback(null);
  }, []);

  const appendAudioEvent = useCallback((event: AudioDetectionEvent) => {
    audioLogRef.current.push(event);
    setRecentAudioEvents(prev => [event, ...prev].slice(0, 14));
  }, []);

  const appendContact = useCallback((event: BounceContactEvent) => {
    contactLogRef.current.push(event);
    setRecentContacts(prev => [event, ...prev].slice(0, 14));
    syncSummary();
  }, [syncSummary]);

  const handleAudioDetected = useCallback((audioB64: string) => {
    if (!isRunningRef.current) return;

    const detectedAt = Date.now();

    try {
      const pcm = decodeBase64PCM(audioB64);
      const audioEvent = detectAudioContact({
        detectedAtMs: detectedAt,
        pcm,
        confidenceThreshold: settingsRef.current.audioConfidence,
        dedupMs: settingsRef.current.audioDedupMs,
        lastQualifiedTsMs: lastQualifiedLabelTsRef.current.racket_contact,
      });

      const activeGroup = contactGroupRef.current;
      const inActiveGroup = !!activeGroup && detectedAt - activeGroup.startedAtMs <= settingsRef.current.audioGroupWindowMs;
      if (audioEvent.qualified && inActiveGroup) {
        audioEvent.qualified = false;
        audioEvent.ignored_reason = 'group_duplicate';
        audioEvent.group_id = activeGroup.id;
        audioEvent.group_status = 'ignored_duplicate';
      } else if (audioEvent.qualified) {
        groupCounterRef.current += 1;
        contactGroupRef.current = { id: groupCounterRef.current, startedAtMs: detectedAt };
        audioEvent.group_id = groupCounterRef.current;
        audioEvent.group_status = 'best_candidate';
      } else if (inActiveGroup) {
        audioEvent.group_id = activeGroup.id;
        audioEvent.group_status = 'ignored_duplicate';
      } else {
        audioEvent.group_status = 'standalone';
      }

      if (audioEvent.qualified) {
        lastQualifiedLabelTsRef.current.racket_contact = detectedAt;
      }

      appendAudioEvent(audioEvent);

      const motionMetrics = roundMotionMetrics(
        findBounceMotionMetricsAround(
          sampleLogRef.current,
          detectedAt,
          settingsRef.current.motionWindowMs,
        ),
      );
      const motionGateOpen =
        motionMetrics.gyro_peak >= settingsRef.current.motionGyroThreshold ||
        motionMetrics.accel_ptp >= settingsRef.current.motionAccelThreshold;

      let side: BounceSide = 'uncertain';
      let contactOrientation: Vector3 = { x: 0, y: 0, z: 0 };
      let forehandScore = 0;
      let backhandScore = 0;
      let counted = false;
      let alternationAfter = contactLogRef.current.length > 0
        ? contactLogRef.current[contactLogRef.current.length - 1].alternation_after
        : 0;
      let totalAfter = contactLogRef.current.filter(contact => contact.counted).length;
      let contactIgnoredReason: BounceContactEvent['ignored_reason'] = audioEvent.ignored_reason;

      if (audioEvent.qualified) {
        const orientation = findOrientationAround(
          sampleLogRef.current,
          detectedAt,
          settingsRef.current.orientationSampleWindowMs,
        );

        if (orientation) {
          const classified = classifyBounceSide(
            orientation,
            calibration,
            settingsRef.current.orientationDeadzone,
          );
          side = classified.side;
          contactOrientation = normalize(orientation);
          forehandScore = Number(classified.forehandScore.toFixed(3));
          backhandScore = Number(classified.backhandScore.toFixed(3));
          sideLogRef.current.push({
            detected_at: audioEvent.detected_at,
            ts_ms: detectedAt,
            side,
            orientation: contactOrientation,
            forehand_score: forehandScore,
            backhand_score: backhandScore,
          });
        } else {
          sideLogRef.current.push({
            detected_at: audioEvent.detected_at,
            ts_ms: detectedAt,
            side: 'uncertain',
            orientation: { x: 0, y: 0, z: 0 },
            forehand_score: 0,
            backhand_score: 0,
          });
        }

        counted = true;
        totalAfter = contactLogRef.current.filter(contact => contact.counted).length + 1;
        if (mode === 'bounce_alternating' && side !== 'uncertain') {
          const previousSide = lastConfirmedSideRef.current;
          if (previousSide && previousSide !== side) {
            alternationAfter += 1;
          }
          lastConfirmedSideRef.current = side;
        } else if (side !== 'uncertain') {
          lastConfirmedSideRef.current = side;
        }
      }

      const contactEvent: BounceContactEvent = {
        detected_at: audioEvent.detected_at,
        ts_ms: detectedAt,
        mode,
        audio_label: audioEvent.label,
        audio_confidence: audioEvent.confidence,
        surface_label: audioEvent.surface_label,
        surface_confidence: audioEvent.surface_confidence,
        motion_gate_open: motionGateOpen,
        motion_metrics: motionMetrics,
        side,
        orientation: contactOrientation,
        forehand_score: forehandScore,
        backhand_score: backhandScore,
        group_id: audioEvent.group_id,
        group_status: audioEvent.group_status,
        counted,
        total_after: totalAfter,
        alternation_after: alternationAfter,
        ignored_reason: counted ? undefined : contactIgnoredReason,
      };

      appendContact(contactEvent);
      if (counted) {
        setFeedback(
          `${modeLabel(mode)}: ${sideLabel(side)} ${Math.round(contactEvent.audio_confidence * 100)}%`,
        );
      }
    } catch (error: any) {
      setFeedback(`Audio decode failed: ${error?.message ?? 'unknown error'}`);
    }
  }, [
    appendAudioEvent,
    appendContact,
    calibration,
    mode,
    syncSummary,
  ]);

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

      const receivedAtMs = Date.now();
      const takeTsMs = sessionStartRef.current === null ? 0 : Math.max(0, receivedAtMs - sessionStartRef.current);
      sampleLogRef.current.push({
        accel_x: latest.accel.x,
        accel_y: latest.accel.y,
        accel_z: latest.accel.z,
        gyro_x: latest.gyro.x,
        gyro_y: latest.gyro.y,
        gyro_z: latest.gyro.z,
        mag_x: latest.mag.x,
        mag_y: latest.mag.y,
        mag_z: latest.mag.z,
        ts_ms: receivedAtMs,
        received_at_ms: receivedAtMs,
        take_ts_ms: takeTsMs,
        sensor_ts: parsed.sensor_ts,
      });
    },
    [],
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
      await AudioStream.setRetriggerMs(settingsRef.current.audioRetriggerMs);
      await AudioStream.startStreaming(settingsRef.current.audioThreshold);
      setIsRunning(true);
      setFeedback(`${modeLabel(mode)} running.`);
    } catch (error: any) {
      setFeedback(`Could not start audio stream: ${error?.message ?? 'unknown error'}`);
    }
  }, [isConnected, mode, requestMicPermission, resetRun]);

  const stopRun = useCallback(async () => {
    try {
      await AudioStream.stopStreaming();
    } catch (_) {}
    setIsRunning(false);
    setFeedback(`${modeLabel(mode)} stopped.`);
  }, [mode]);

  const saveSession = useCallback(async () => {
    if (isRunning) {
      Alert.alert('Stop first', 'Stop the current pass before saving.');
      return;
    }
    if (sessionStartRef.current === null || audioLogRef.current.length === 0) {
      Alert.alert('No data', 'Run at least one pass before saving.');
      return;
    }

    try {
      await RNFS.mkdir(EXPORT_DIR);
      const filePath = await nextExportPath(mode === 'bounce_free' ? 'bounce_free' : 'bounce_alternating');
      const session: BounceTestSessionFile = {
        session_meta: {
          player_name: setup.name,
          handedness: setup.handedness,
          mode,
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
        bounce_side_events: sideLogRef.current,
        bounce_contacts: contactLogRef.current,
        summary: {
          total_count: totalCount,
          forehand_count: forehandCount,
          backhand_count: backhandCount,
          uncertain_count: uncertainCount,
          alternation_count: alternationCount,
        },
      };

      await RNFS.writeFile(filePath, JSON.stringify(session, null, 2), 'utf8');
      try { await RNFS.scanFile(filePath); } catch (_) {}

      Alert.alert(
        'Export saved',
        `File: Download/pingis_sessions/${filePath.split('/').pop()}`,
      );
      setFeedback(`Export saved: ${filePath.split('/').pop()}`);
    } catch (error: any) {
      Alert.alert('Error', `Could not save session: ${error?.message ?? 'unknown error'}`);
    }
  }, [
    alternationCount,
    calibration,
    forehandCount,
    isRunning,
    mode,
    presetId,
    setup.handedness,
    setup.name,
    totalCount,
    uncertainCount,
    backhandCount,
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
          <Text style={styles.headerTitle}>{modeLabel(mode)}</Text>
          <Text style={styles.headerSub}>
            {setup.name} · {setup.handedness === 'right' ? 'Right' : 'Left'} hand · {DEFAULT_BOUNCE_PRESET_LABEL}
          </Text>
        </View>
      </View>

      <View style={styles.statusCard}>
        <View>
          <Text style={styles.statusLabel}>STATUS</Text>
          <Text style={styles.statusValue}>{isRunning ? 'RUNNING' : 'STOPPED'}</Text>
          <Text style={styles.statusMeta}>
            Sensor {isConnected ? `${sampleHz} Hz` : 'offline'} · Mic {micGranted ? 'ready' : 'missing'}
          </Text>
        </View>
        <TouchableOpacity
          style={[styles.toggleBtn, isRunning ? styles.toggleStop : styles.toggleStart]}
          onPress={isRunning ? stopRun : startRun}
          activeOpacity={0.7}
        >
          <Text style={[styles.toggleTxt, isRunning ? styles.toggleTxtStop : styles.toggleTxtStart]}>
            {isRunning ? 'STOP' : 'START'}
          </Text>
        </TouchableOpacity>
      </View>

      {feedback && <Text style={styles.feedback}>{feedback}</Text>}

      <View style={styles.summaryCard}>
        <Text style={styles.sectionLabel}>COUNTS</Text>
        <View style={styles.summaryGrid}>
          <Metric label="TOTAL" value={totalCount} accent="#f5c76d" />
          <Metric label="FH" value={forehandCount} accent="#2ecc71" />
          <Metric label="BH" value={backhandCount} accent="#4a9eff" />
          <Metric label="UNC" value={uncertainCount} accent="#b6b6b6" />
        </View>
        <Text style={styles.summaryHint}>Motion gate closed (debug): {motionClosedCount}</Text>
        {mode === 'bounce_alternating' && (
          <Text style={styles.summaryHint}>Alternation score: {alternationCount}</Text>
        )}
      </View>

      <View style={styles.presetCard}>
        <Text style={styles.sectionLabel}>ACTIVE PRESET</Text>
        <Text style={styles.presetStaticTitle}>{DEFAULT_BOUNCE_PRESET_LABEL}</Text>
        <Text style={styles.presetStaticText}>
          Normal testing in this iteration uses only B0. Tune the live sliders if needed, but do
          not preset-sweep B1 or B2 in the main workflow.
        </Text>
      </View>

      <View style={styles.settingsCard}>
        <Text style={styles.sectionLabel}>LIVE SETTINGS</Text>
        <DebugSlider
          label="Audio onset"
          value={settings.audioThreshold}
          min={0.005}
          max={0.08}
          step={0.005}
          onChange={value => setSettings(prev => ({ ...prev, audioThreshold: value }))}
          valueFormatter={value => value.toFixed(3)}
          leftHint="0.005"
          rightHint="0.080"
        />
        <DebugSlider
          label="Retrigger window"
          value={settings.audioRetriggerMs}
          min={0}
          max={420}
          step={20}
          onChange={value => setSettings(prev => ({ ...prev, audioRetriggerMs: value }))}
          valueFormatter={value => `${Math.round(value)} ms`}
          leftHint="0"
          rightHint="420"
        />
        <DebugSlider
          label="Group window"
          value={settings.audioGroupWindowMs}
          min={0}
          max={240}
          step={20}
          onChange={value => setSettings(prev => ({ ...prev, audioGroupWindowMs: value }))}
          valueFormatter={value => `${Math.round(value)} ms`}
          leftHint="0"
          rightHint="240"
        />
        <DebugSlider
          label="Audio confidence"
          value={settings.audioConfidence}
          min={0.4}
          max={0.9}
          step={0.05}
          onChange={value => setSettings(prev => ({ ...prev, audioConfidence: value }))}
          valueFormatter={value => `${Math.round(value * 100)}%`}
          leftHint="40%"
          rightHint="90%"
        />
        <DebugSlider
          label="Merge window"
          value={settings.audioDedupMs}
          min={80}
          max={360}
          step={20}
          onChange={value => setSettings(prev => ({ ...prev, audioDedupMs: value }))}
          valueFormatter={value => `${Math.round(value)} ms`}
          leftHint="80"
          rightHint="360"
        />
        <DebugSlider
          label="Motion window"
          value={settings.motionWindowMs}
          min={160}
          max={320}
          step={20}
          onChange={value => setSettings(prev => ({ ...prev, motionWindowMs: value }))}
          valueFormatter={value => `${Math.round(value)} ms`}
          leftHint="160"
          rightHint="320"
        />
        <DebugSlider
          label="Motion gyro threshold"
          value={settings.motionGyroThreshold}
          min={20}
          max={120}
          step={5}
          onChange={value => setSettings(prev => ({ ...prev, motionGyroThreshold: value }))}
          valueFormatter={value => `${Math.round(value)}`}
          leftHint="20"
          rightHint="120"
        />
        <DebugSlider
          label="Motion accel threshold"
          value={settings.motionAccelThreshold}
          min={80}
          max={360}
          step={10}
          onChange={value => setSettings(prev => ({ ...prev, motionAccelThreshold: value }))}
          valueFormatter={value => `${Math.round(value)}`}
          leftHint="80"
          rightHint="360"
        />
      </View>

      <View style={styles.settingsCard}>
        <View style={styles.debugHeader}>
          <Text style={styles.sectionLabel}>ADVANCED SIDE SETTINGS</Text>
          <TouchableOpacity onPress={() => setShowAdvanced(prev => !prev)}>
            <Text style={styles.linkTxt}>{showAdvanced ? 'Hide advanced' : 'Show advanced'}</Text>
          </TouchableOpacity>
        </View>
        {showAdvanced && (
          <>
            <DebugSlider
              label="Orientation window"
              value={settings.orientationSampleWindowMs}
              min={80}
              max={320}
              step={20}
              onChange={value => setSettings(prev => ({ ...prev, orientationSampleWindowMs: value }))}
              valueFormatter={value => `${Math.round(value)} ms`}
              leftHint="80"
              rightHint="320"
            />
            <DebugSlider
              label="Side deadzone"
              value={settings.orientationDeadzone}
              min={0.02}
              max={0.25}
              step={0.01}
              onChange={value => setSettings(prev => ({ ...prev, orientationDeadzone: value }))}
              valueFormatter={value => value.toFixed(2)}
              leftHint="strict"
              rightHint="lenient"
            />
          </>
        )}
      </View>

      <View style={styles.actionRow}>
        <TouchableOpacity style={[styles.secondaryBtn, styles.saveBtn]} onPress={saveSession}>
          <Text style={styles.saveBtnTxt}>Save export</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.secondaryBtn, styles.resetBtn]} onPress={resetRun}>
          <Text style={styles.resetBtnTxt}>Reset run</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.debugCard}>
        <View style={styles.debugHeader}>
          <Text style={styles.sectionLabel}>CONTACT DEBUG</Text>
        </View>
        {recentContacts.length === 0 && <Text style={styles.emptyTxt}>No bounce contacts yet.</Text>}
        {recentContacts.map((contact, index) => (
          <Text key={`${contact.ts_ms}-${index}`} style={styles.debugRow}>
            {formatClock(contact.ts_ms)} · {contact.counted ? 'counted' : reasonLabel(contact.ignored_reason)}
            {' | '}grp {contact.group_id ?? '-'} {contact.group_status ?? 'standalone'}
            {' | '}bin {audioLabelName(contact.audio_label)} {Math.round(contact.audio_confidence * 100)}%
            {' | '}surf {surfaceLabelName(contact.surface_label)} {Math.round((contact.surface_confidence ?? 0) * 100)}%
            {' | '}side {sideLabel(contact.side)} {'|'} gate {contact.motion_gate_open ? 'OPEN' : 'CLOSED'}
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
            {formatClock(event.ts_ms)} · bin {audioLabelName(event.label)} {Math.round(event.confidence * 100)}%
            {' | '}grp {event.group_id ?? '-'} {event.group_status ?? 'standalone'}
            {' | '}surf {surfaceLabelName(event.surface_label)} {Math.round((event.surface_confidence ?? 0) * 100)}%
            {' | '}{event.qualified ? 'qualified' : reasonLabel(event.ignored_reason)}
          </Text>
        ))}
      </View>

      <View style={styles.fileCard}>
        <Text style={styles.fileTitle}>Test loop</Text>
        <Text style={styles.fileText}>
          1. Run B0 as the only normal preset in this iteration.{'\n'}
          2. Prioritize Studs fritt and use Studs vaxla sida only as regression.{'\n'}
          3. Bounce on the racket and watch TOTAL / FH / BH / UNC.{'\n'}
          4. Save the export to Downloads/pingis_sessions and fill the matching row in TEST_PLAN.md.
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
  statusCard: {
    backgroundColor: '#111',
    borderRadius: 16,
    padding: 16,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  statusLabel: { color: '#666', fontSize: 10, letterSpacing: 2, marginBottom: 6 },
  statusValue: { color: '#fff', fontSize: 22, fontWeight: '800' },
  statusMeta: { color: '#888', fontSize: 12, marginTop: 4 },
  toggleBtn: { borderRadius: 12, paddingHorizontal: 18, paddingVertical: 14 },
  toggleStart: { backgroundColor: '#0d2d1a' },
  toggleStop: { backgroundColor: '#2d0d0d' },
  toggleTxt: { fontSize: 14, fontWeight: '800', letterSpacing: 1 },
  toggleTxtStart: { color: '#2ecc71' },
  toggleTxtStop: { color: '#ff7f7f' },
  feedback: { color: '#f5c76d', fontSize: 13, textAlign: 'center', marginTop: 10 },
  summaryCard: {
    marginTop: 16,
    backgroundColor: '#111',
    borderRadius: 16,
    padding: 16,
  },
  sectionLabel: { color: '#777', fontSize: 10, letterSpacing: 2, marginBottom: 12 },
  summaryGrid: { flexDirection: 'row', justifyContent: 'space-between', gap: 10 },
  metricItem: {
    flex: 1,
    backgroundColor: '#171717',
    borderRadius: 12,
    paddingVertical: 16,
    alignItems: 'center',
  },
  metricValue: { fontSize: 30, fontWeight: '800', fontFamily: 'monospace' },
  metricLabel: { color: '#666', fontSize: 10, marginTop: 4, letterSpacing: 1.5 },
  summaryHint: { color: '#888', fontSize: 12, marginTop: 12 },
  presetCard: {
    marginTop: 16,
    backgroundColor: '#111',
    borderRadius: 16,
    padding: 16,
  },
  presetStaticTitle: { color: '#f5c76d', fontSize: 18, fontWeight: '800', marginBottom: 6 },
  presetStaticText: { color: '#888', fontSize: 12, lineHeight: 18 },
  settingsCard: {
    marginTop: 16,
    backgroundColor: '#111',
    borderRadius: 16,
    padding: 16,
  },
  actionRow: { flexDirection: 'row', gap: 12, marginTop: 14 },
  secondaryBtn: {
    flex: 1,
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: 'center',
  },
  saveBtn: { backgroundColor: '#0d2d1a' },
  saveBtnTxt: { color: '#2ecc71', fontWeight: '700' },
  resetBtn: { backgroundColor: '#2b220d' },
  resetBtnTxt: { color: '#f5c76d', fontWeight: '700' },
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
