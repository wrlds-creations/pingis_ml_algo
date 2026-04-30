import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
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
import { Camera, useCameraDevice, useCameraPermission, useVideoOutput, type Recorder } from 'react-native-vision-camera';
import {
  AUDIO_CAPTURE_STOPPED_EVENT,
  AudioCapture,
  AudioCaptureEmitter,
  type AudioCaptureStoppedEvent,
} from './NativeAudioCapture';
import { AudioTakeReviewScreen } from './AudioTakeReviewScreen';
import { requiresAudioReview } from './audioReview';
import {
  ACCEL_UUID,
  ACCEL_UUID_ALT,
  GYRO_UUID,
  MAG_UUID,
  SERVICE_UUID,
  parsePacket,
} from './airhive';
import type {
  AudioBackgroundCondition,
  AudioCollectionScenarioSummary,
  AudioCollectionSummary,
  AudioEvent,
  AudioImuRecording,
  AudioVideoRecording,
  AudioLabel,
  AudioReviewMarker,
  AudioScenarioId,
  AudioSessionFile,
  CalibrationData,
  ImuSample,
  PlayerSetup,
} from './types';

const APP_VERSION = '1.7';
const TARGET_DURATION_S = 30;
const TARGET_DURATION_MS = TARGET_DURATION_S * 1000;
const WATCHDOG_STOP_MS = TARGET_DURATION_MS + 5000;
const VIDEO_STOP_TIMEOUT_MS = 4000;
const VIDEO_FINALIZE_TIMEOUT_MS = 5000;
const VIDEO_RECORDER_COOLDOWN_MS = 750;
const SESSION_DIR = `${RNFS.ExternalStorageDirectoryPath}/Download/pingis_sessions`;
type CameraFacing = 'front' | 'back';

interface AudioScenarioDefinition {
  id: AudioScenarioId;
  title: string;
  prompt: string;
  label: AudioLabel;
  background_condition: AudioBackgroundCondition;
  target_takes: number;
  color: string;
  bg: string;
}

interface PendingReviewItem {
  sessionJsonPath: string;
  sessionDir: string;
  eventIndex: number;
  event: AudioEvent;
}

const AUDIO_SCENARIOS: AudioScenarioDefinition[] = [
  {
    id: 'racket_quiet',
    title: 'Racket quiet',
    prompt: 'Studsa pa racket i lugn miljo.',
    label: 'racket_bounce',
    background_condition: 'quiet',
    target_takes: 3,
    color: '#2ecc71',
    bg: '#0d2d1a',
  },
  {
    id: 'racket_counting',
    title: 'Racket counting',
    prompt: 'Studsa pa racket medan du raknar hogt.',
    label: 'racket_bounce',
    background_condition: 'speech',
    target_takes: 3,
    color: '#5fd18b',
    bg: '#123720',
  },
  {
    id: 'racket_music_low',
    title: 'Racket music low',
    prompt: 'Studsa pa racket med lag musik i bakgrunden.',
    label: 'racket_bounce',
    background_condition: 'music_low',
    target_takes: 2,
    color: '#78d6a6',
    bg: '#153c27',
  },
  {
    id: 'racket_music_mid',
    title: 'Racket music mid',
    prompt: 'Studsa pa racket med tydligare musik i bakgrunden.',
    label: 'racket_bounce',
    background_condition: 'music_mid',
    target_takes: 2,
    color: '#98dfbd',
    bg: '#18432d',
  },
  {
    id: 'speech_only',
    title: 'Speech only',
    prompt: 'Ingen boll. Rakna eller prata i 30 sekunder.',
    label: 'noise',
    background_condition: 'speech',
    target_takes: 2,
    color: '#e74c3c',
    bg: '#2d0d0d',
  },
  {
    id: 'desk_keyboard_only',
    title: 'Desk keyboard only',
    prompt: 'Ingen boll. Tangentbord eller skrivbordsljud i 30 sekunder.',
    label: 'noise',
    background_condition: 'desk',
    target_takes: 2,
    color: '#ff7f66',
    bg: '#33130f',
  },
  {
    id: 'music_low_only',
    title: 'Music low only',
    prompt: 'Ingen boll. Lag musik i 30 sekunder.',
    label: 'noise',
    background_condition: 'music_low',
    target_takes: 1,
    color: '#ff9e7d',
    bg: '#382019',
  },
  {
    id: 'music_mid_only',
    title: 'Music mid only',
    prompt: 'Ingen boll. Tydligare musik i 30 sekunder.',
    label: 'noise',
    background_condition: 'music_mid',
    target_takes: 1,
    color: '#ffc09f',
    bg: '#3a241e',
  },
  {
    id: 'table_quiet',
    title: 'Table quiet',
    prompt: 'Studsa pa bordet i lugn miljo.',
    label: 'table_bounce',
    background_condition: 'quiet',
    target_takes: 3,
    color: '#4a9eff',
    bg: '#0d1f33',
  },
  {
    id: 'floor_quiet',
    title: 'Floor quiet',
    prompt: 'Studsa pa golvet i lugn miljo.',
    label: 'floor_bounce',
    background_condition: 'quiet',
    target_takes: 3,
    color: '#e67e22',
    bg: '#2d1a00',
  },
];

function formatDuration(ms: number): string {
  const s = Math.max(0, Math.ceil(ms / 1000));
  return `${s}s`;
}

function jsonPathFromSessionDir(sessionDir: string) {
  const sessionName = sessionDir.split('/').pop();
  return `${SESSION_DIR}/${sessionName}.json`;
}

async function nextSessionDir(date: string): Promise<string> {
  let n = 1;
  let dir: string;
  do {
    dir = `${SESSION_DIR}/audio_session_${date}_${String(n).padStart(3, '0')}`;
    n += 1;
  } while (await RNFS.exists(dir) || await RNFS.exists(jsonPathFromSessionDir(dir)));
  return dir;
}

function withTimeout<T>(promise: Promise<T>, timeoutMs: number, message: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error(message)), timeoutMs);
    promise.then(
      value => {
        clearTimeout(timeout);
        resolve(value);
      },
      error => {
        clearTimeout(timeout);
        reject(error);
      },
    );
  });
}

function sleep(ms: number) {
  return new Promise<void>(resolve => {
    setTimeout(() => resolve(), ms);
  });
}

function isPendingReview(event: AudioEvent): boolean {
  return requiresAudioReview(event.scenario_id) && !event.review?.completed_at;
}

function buildScenarioSummaries(events: AudioEvent[]): AudioCollectionScenarioSummary[] {
  return AUDIO_SCENARIOS.map(scenario => {
    const completed = events.filter(event => event.scenario_id === scenario.id).length;
    return {
      scenario_id: scenario.id,
      label: scenario.label,
      target_takes: scenario.target_takes,
      completed_takes: completed,
      remaining_takes: Math.max(0, scenario.target_takes - completed),
    };
  });
}

function buildCollectionSummary(
  scenarioSummaries: AudioCollectionScenarioSummary[],
  events: AudioEvent[],
): AudioCollectionSummary {
  const totalScenarios = scenarioSummaries.length;
  const completedScenarios = scenarioSummaries.filter(item => item.remaining_takes === 0).length;
  const totalTakes = scenarioSummaries.reduce((sum, item) => sum + item.target_takes, 0);
  const completedTakes = scenarioSummaries.reduce((sum, item) => sum + item.completed_takes, 0);
  return {
    total_scenarios: totalScenarios,
    completed_scenarios: completedScenarios,
    total_takes: totalTakes,
    completed_takes: completedTakes,
    remaining_takes: Math.max(0, totalTakes - completedTakes),
    pending_review_takes: events.filter(isPendingReview).length,
    reviewed_takes: events.filter(event => !!event.review?.completed_at && event.review?.required).length,
    auto_saved_takes: events.filter(event => event.review?.required === false).length,
  };
}

function findNextIncompleteScenario(events: AudioEvent[]): AudioScenarioId | null {
  const summaries = buildScenarioSummaries(events);
  const next = summaries.find(item => item.remaining_takes > 0);
  return next?.scenario_id ?? null;
}

function buildSessionFile(
  setup: PlayerSetup,
  events: AudioEvent[],
  sessionDate: string,
  mode: 'audio_only' | 'audio_imu',
  calibration?: CalibrationData,
): AudioSessionFile {
  return {
    session_meta: {
      recorder_name: setup.name,
      player_name: setup.name,
      handedness: setup.handedness,
      session_date: sessionDate,
      app_version: APP_VERSION,
      clip_duration_ms: 0,
      collection_mode: mode === 'audio_imu' ? 'guided_scenarios_audio_imu' : 'guided_scenarios',
      target_duration_s: TARGET_DURATION_S,
      planned_takes: AUDIO_SCENARIOS.reduce((sum, scenario) => sum + scenario.target_takes, 0),
      calibration_id: calibration?.calibration_id,
    },
    calibration_profile: calibration,
    events,
  };
}

async function readSessionFile(jsonPath: string): Promise<AudioSessionFile | null> {
  if (!(await RNFS.exists(jsonPath))) return null;
  const contents = await RNFS.readFile(jsonPath, 'utf8');
  return JSON.parse(contents) as AudioSessionFile;
}

async function writeSessionFile(jsonPath: string, sessionData: AudioSessionFile) {
  await RNFS.writeFile(jsonPath, JSON.stringify(sessionData, null, 2), 'utf8');
  try { await RNFS.scanFile(jsonPath); } catch (_) {}
}

async function buildPendingReviewQueue(): Promise<PendingReviewItem[]> {
  if (!(await RNFS.exists(SESSION_DIR))) return [];
  const entries = await RNFS.readDir(SESSION_DIR);
  const jsonFiles = entries.filter(
    entry => entry.isFile() && /^audio_session_\d{4}-\d{2}-\d{2}_\d{3}\.json$/i.test(entry.name),
  );

  const items: PendingReviewItem[] = [];
  for (const file of jsonFiles) {
    try {
      const session = await readSessionFile(file.path);
      if (!session) continue;
      const sessionDir = `${SESSION_DIR}/${file.name.replace(/\.json$/i, '')}`;
      session.events.forEach((event, index) => {
        if (isPendingReview(event)) {
          items.push({
            sessionJsonPath: file.path,
            sessionDir,
            eventIndex: index,
            event,
          });
        }
      });
    } catch (_) {
      // ignore malformed legacy files in the queue view
    }
  }

  items.sort((a, b) => a.event.recorded_at.localeCompare(b.event.recorded_at));
  return items;
}

interface Props {
  setup: PlayerSetup;
  mode?: 'audio_only' | 'audio_imu';
  calibration?: CalibrationData;
  device?: Device;
  onDone: () => void;
}

export function AudioCollectionScreen({
  setup,
  mode = 'audio_only',
  calibration,
  device,
  onDone,
}: Props) {
  const isAudioImuMode = mode === 'audio_imu' && !!device && !!calibration;
  const { hasPermission: hasCameraPermission, requestPermission: requestCameraPermission } = useCameraPermission();
  const [preferredCameraFacing, setPreferredCameraFacing] = useState<CameraFacing>('front');
  const frontCameraDevice = useCameraDevice('front');
  const backCameraDevice = useCameraDevice('back');
  const cameraDevice = preferredCameraFacing === 'front'
    ? (frontCameraDevice ?? backCameraDevice)
    : (backCameraDevice ?? frontCameraDevice);
  const effectiveCameraFacing: CameraFacing | null = cameraDevice
    ? (cameraDevice.id === frontCameraDevice?.id ? 'front' : 'back')
    : null;
  const videoOutput = useVideoOutput({ enableAudio: false });
  const [selectedScenarioId, setSelectedScenarioId] = useState<AudioScenarioId>('racket_quiet');
  const [isRecording, setIsRecording] = useState(false);
  const [isStartingRecording, setIsStartingRecording] = useState(false);
  const [isSensorConnected, setIsSensorConnected] = useState(true);
  const [sampleHz, setSampleHz] = useState(0);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [remainingMs, setRemainingMs] = useState(TARGET_DURATION_MS);
  const [events, setEvents] = useState<AudioEvent[]>([]);
  const [permissionGranted, setPermission] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);
  const [pendingReviews, setPendingReviews] = useState<PendingReviewItem[]>([]);
  const [reviewTarget, setReviewTarget] = useState<PendingReviewItem | null>(null);

  const sessionDirRef = useRef<string | null>(null);
  const sessionJsonPathRef = useRef<string | null>(null);
  const sessionDateRef = useRef<string>(new Date().toISOString());
  const isRecordingRef = useRef(false);
  const isStartingRecordingRef = useRef(false);
  const startTimeRef = useRef<number>(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const stopTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const eventsRef = useRef<AudioEvent[]>([]);
  const sampleCountRef = useRef(0);
  const hzTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const latestSensorRef = useRef({
    accel: { x: 0, y: 0, z: 0 },
    gyro: { x: 0, y: 0, z: 0 },
    mag: { x: 0, y: 0, z: 0 },
  });
  const currentTakeImuRef = useRef<AudioImuRecording | null>(null);
  const videoFinalizeBarrierRef = useRef<Promise<void>>(Promise.resolve());
  const videoCooldownUntilRef = useRef(0);
  const currentTakeVideoRef = useRef<{
    recorder: Recorder;
    filename: string;
    targetPath: string;
    started_at_ms: number;
    tempPath?: string;
    finishPromise: Promise<string>;
    resolveFinish: (path: string) => void;
    rejectFinish: (error: Error) => void;
  } | null>(null);
  const activeTakeRef = useRef<{
    scenario_id: AudioScenarioId;
    filename: string;
    filePath: string;
    videoFilename?: string;
    videoFilePath?: string;
    take_index: number;
  } | null>(null);
  const finalizingStopRef = useRef(false);

  const scenarioSummaries = useMemo(() => buildScenarioSummaries(events), [events]);
  const collectionSummary = useMemo(
    () => buildCollectionSummary(scenarioSummaries, events),
    [events, scenarioSummaries],
  );
  const selectedScenario = AUDIO_SCENARIOS.find(item => item.id === selectedScenarioId) ?? AUDIO_SCENARIOS[0];
  const selectedSummary =
    scenarioSummaries.find(item => item.scenario_id === selectedScenarioId) ?? scenarioSummaries[0];
  const canRecord =
    permissionGranted &&
    hasCameraPermission &&
    !isRecording &&
    !isStartingRecording &&
    !!cameraDevice &&
    cameraReady &&
    (!isAudioImuMode || isSensorConnected) &&
    !!selectedScenario &&
    (selectedSummary?.remaining_takes ?? 0) > 0 &&
    reviewTarget === null;

  useEffect(() => {
    setCameraReady(false);
  }, [cameraDevice?.id]);

  useEffect(() => {
    eventsRef.current = events;
  }, [events]);

  useEffect(() => {
    isRecordingRef.current = isRecording;
  }, [isRecording]);

  useEffect(() => {
    isStartingRecordingRef.current = isStartingRecording;
  }, [isStartingRecording]);

  const handleSensorNotification = useCallback(
    (_error: BleError | null, characteristic: Characteristic | null) => {
      if (!characteristic?.value || !characteristic.uuid) return;
      const parsed = parsePacket(characteristic.uuid, characteristic.value);
      if (!parsed) return;

      const latest = latestSensorRef.current;
      if (parsed.type === 'accel') latest.accel = parsed;
      else if (parsed.type === 'gyro') latest.gyro = parsed;
      else latest.mag = parsed;

      sampleCountRef.current += 1;

      if (!isRecordingRef.current || !currentTakeImuRef.current) return;

      const sample: ImuSample = {
        accel_x: latest.accel.x,
        accel_y: latest.accel.y,
        accel_z: latest.accel.z,
        gyro_x: latest.gyro.x,
        gyro_y: latest.gyro.y,
        gyro_z: latest.gyro.z,
        mag_x: latest.mag.x,
        mag_y: latest.mag.y,
        mag_z: latest.mag.z,
        ts_ms: Date.now(),
      };
      currentTakeImuRef.current.samples.push(sample);
    },
    [],
  );

  useEffect(() => {
    if (!isAudioImuMode || !device) {
      setIsSensorConnected(true);
      setSampleHz(0);
      return;
    }

    setIsSensorConnected(true);
    sampleCountRef.current = 0;

    for (const uuid of [ACCEL_UUID, ACCEL_UUID_ALT, GYRO_UUID, MAG_UUID]) {
      try {
        device.monitorCharacteristicForService(SERVICE_UUID, uuid, handleSensorNotification);
      } catch (_) {}
    }

    let lastCount = 0;
    hzTimerRef.current = setInterval(() => {
      setSampleHz(sampleCountRef.current - lastCount);
      lastCount = sampleCountRef.current;
    }, 1000);

    const disconnectSub = device.onDisconnected(() => {
      setIsSensorConnected(false);
      setSampleHz(0);
      setFeedback('AirHive disconnected.');
      if (hzTimerRef.current) {
        clearInterval(hzTimerRef.current);
        hzTimerRef.current = null;
      }
    });

    return () => {
      if (hzTimerRef.current) {
        clearInterval(hzTimerRef.current);
        hzTimerRef.current = null;
      }
      disconnectSub.remove();
      try { device.cancelConnection(); } catch (_) {}
    };
  }, [device, handleSensorNotification, isAudioImuMode]);

  const refreshPendingReviews = useCallback(async () => {
    const queue = await buildPendingReviewQueue();
    setPendingReviews(queue);
  }, []);

  const prepareNewSession = useCallback(async () => {
    await RNFS.mkdir(SESSION_DIR);
    const date = new Date().toISOString().slice(0, 10);
    const sessionDir = await nextSessionDir(date);
    const jsonPath = jsonPathFromSessionDir(sessionDir);
    sessionDirRef.current = sessionDir;
    sessionJsonPathRef.current = jsonPath;
    sessionDateRef.current = new Date().toISOString();
    await RNFS.mkdir(sessionDir);
    await writeSessionFile(
      jsonPath,
      buildSessionFile(setup, [], sessionDateRef.current, mode, calibration),
    );
  }, [calibration, mode, setup]);

  const persistCurrentSession = useCallback(async (nextEvents: AudioEvent[]) => {
    if (!sessionJsonPathRef.current) return;
    await writeSessionFile(
      sessionJsonPathRef.current,
      buildSessionFile(setup, nextEvents, sessionDateRef.current, mode, calibration),
    );
  }, [calibration, mode, setup]);

  const clearTimers = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (stopTimeoutRef.current) {
      clearTimeout(stopTimeoutRef.current);
      stopTimeoutRef.current = null;
    }
  }, []);

  useEffect(() => {
    (async () => {
      if (Platform.OS === 'android') {
        const result = await PermissionsAndroid.request(
          PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
          {
            title: 'Mikrofonatkomst',
            message: 'Appen behover mikrofonen for att samla in ljuddata.',
            buttonPositive: 'OK',
          },
        );
        if (result !== PermissionsAndroid.RESULTS.GRANTED) {
          Alert.alert('Tillstand saknas', 'Mikrofontillstand behovs for ljudinsamlingen.');
          return;
        }
      }

      if (!hasCameraPermission) {
        const granted = await requestCameraPermission();
        if (!granted) {
          Alert.alert('Tillstand saknas', 'Kameratillstand behovs for video-review i ljudinsamlingen.');
          return;
        }
      }

      setPermission(true);
      await prepareNewSession();
      await refreshPendingReviews();
    })().catch(error => {
      Alert.alert('Fel', `Kunde inte forbereda ljudinsamlingen: ${String(error)}`);
    });

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (stopTimeoutRef.current) clearTimeout(stopTimeoutRef.current);
    };
  }, [hasCameraPermission, prepareNewSession, refreshPendingReviews, requestCameraPermission]);

  const removeCurrentEventByIndex = useCallback(async (eventIndex: number, eventToRemove: AudioEvent) => {
    const nextEvents = eventsRef.current.filter((_, index) => index !== eventIndex);
    eventsRef.current = nextEvents;
    setEvents(nextEvents);
    await persistCurrentSession(nextEvents);

    if (sessionDirRef.current) {
      const wavPath = `${sessionDirRef.current}/${eventToRemove.wav_filename}`;
      if (await RNFS.exists(wavPath)) {
        await RNFS.unlink(wavPath).catch(() => {});
      }
      if (eventToRemove.video_recording?.video_filename) {
        const videoPath = `${sessionDirRef.current}/${eventToRemove.video_recording.video_filename}`;
        if (await RNFS.exists(videoPath)) {
          await RNFS.unlink(videoPath).catch(() => {});
        }
      }
    }
    setSelectedScenarioId(eventToRemove.scenario_id);
  }, [persistCurrentSession]);

  const finalizeVideoRecording = useCallback(async (): Promise<AudioVideoRecording | undefined> => {
    const currentVideo = currentTakeVideoRef.current;
    if (!currentVideo) return undefined;

    try {
      let tempPath: string;
      try {
        tempPath = await withTimeout(
          currentVideo.finishPromise,
          VIDEO_FINALIZE_TIMEOUT_MS,
          'Timed out waiting for review video to finalize.',
        );
      } catch (error) {
        const fallbackPaths = [
          currentVideo.tempPath,
          currentVideo.recorder.filePath,
          currentVideo.targetPath,
        ].filter((path): path is string => Boolean(path));
        const existingPaths = await Promise.all(
          fallbackPaths.map(async path => ((await RNFS.exists(path)) ? path : null)),
        );
        const fallbackPath = existingPaths.find((path): path is string => path !== null);
        if (!fallbackPath) {
          throw error;
        }
        tempPath = fallbackPath;
        setFeedback('Video finalize was slow. Using the saved video file fallback for review.');
      }

      if (tempPath !== currentVideo.targetPath) {
        if (await RNFS.exists(currentVideo.targetPath)) {
          await RNFS.unlink(currentVideo.targetPath).catch(() => {});
        }
        await RNFS.moveFile(tempPath, currentVideo.targetPath);
      }

      const endedAtMs = Date.now();
      return {
        video_filename: currentVideo.filename,
        started_at_ms: currentVideo.started_at_ms,
        ended_at_ms: endedAtMs,
        duration_ms: Math.max(0, endedAtMs - currentVideo.started_at_ms),
        audio_origin_in_video_ms: Math.max(0, startTimeRef.current - currentVideo.started_at_ms),
      };
    } catch (error) {
      setFeedback(`Video review disabled for this take: ${String(error)}`);
      return undefined;
    } finally {
      currentTakeVideoRef.current = null;
      videoCooldownUntilRef.current = Date.now() + VIDEO_RECORDER_COOLDOWN_MS;
    }
  }, []);

  const stopActiveVideoRecording = useCallback(async () => {
    const currentVideo = currentTakeVideoRef.current;
    if (!currentVideo?.recorder?.isRecording) return;
    await withTimeout(
      currentVideo.recorder.stopRecording(),
      VIDEO_STOP_TIMEOUT_MS,
      'Timed out while stopping review video.',
    );
  }, []);

  const finalizeRecordingStop = useCallback(async (
    durationMs: number,
    autoStopped: boolean,
    outputPath?: string,
  ) => {
    const activeTake = activeTakeRef.current;
    if (!activeTake) return;
    if (outputPath && outputPath !== activeTake.filePath) return;

    const scenario = AUDIO_SCENARIOS.find(item => item.id === activeTake.scenario_id);
    if (!scenario) {
      throw new Error(`Unknown scenario: ${activeTake.scenario_id}`);
    }

    const requiresReview = requiresAudioReview(scenario.id);
    const videoRecording = await finalizeVideoRecording();
    const imuRecording = currentTakeImuRef.current
      ? {
          ...currentTakeImuRef.current,
          ended_at_ms: startTimeRef.current + durationMs,
          sample_count: currentTakeImuRef.current.samples.length,
          sample_hz_estimate:
            durationMs > 0
              ? Number(((currentTakeImuRef.current.samples.length * 1000) / durationMs).toFixed(1))
              : 0,
        }
      : undefined;
    const event: AudioEvent = {
      label: scenario.label,
      recorded_at: new Date(startTimeRef.current).toISOString(),
      wav_filename: activeTake.filename,
      duration_ms: durationMs,
      scenario_id: scenario.id,
      background_condition: scenario.background_condition,
      take_index: activeTake.take_index,
      target_duration_s: TARGET_DURATION_S,
      review: {
        required: requiresReview,
        anchor_rule: 'attack_start',
        completed_at: requiresReview ? undefined : new Date().toISOString(),
        markers: [],
      },
      imu_recording: imuRecording,
      video_recording: videoRecording,
    };

    const nextEvents = [...eventsRef.current, event];
    eventsRef.current = nextEvents;
    setEvents(nextEvents);
    await persistCurrentSession(nextEvents);

    const nextSummary = buildScenarioSummaries(nextEvents).find(
      item => item.scenario_id === scenario.id,
    );
    if (nextSummary?.remaining_takes === 0) {
      const nextScenarioId = findNextIncompleteScenario(nextEvents);
      if (nextScenarioId) {
        setSelectedScenarioId(nextScenarioId);
      }
    }

    if (requiresReview && sessionJsonPathRef.current && sessionDirRef.current) {
      setReviewTarget({
        sessionJsonPath: sessionJsonPathRef.current,
        sessionDir: sessionDirRef.current,
        eventIndex: nextEvents.length - 1,
        event,
      });
      setFeedback(`Review pending: ${scenario.title} take ${activeTake.take_index}`);
    } else {
      setFeedback(
        `${autoStopped ? 'Auto-stopped' : 'Stopped'} ${scenario.title} take ${activeTake.take_index}/${scenario.target_takes}`,
      );
      await refreshPendingReviews();
    }

    activeTakeRef.current = null;
    currentTakeImuRef.current = null;
    setIsRecording(false);
    setElapsedMs(0);
    setRemainingMs(TARGET_DURATION_MS);
  }, [finalizeVideoRecording, persistCurrentSession, refreshPendingReviews]);

  const stopRecording = useCallback(async (autoStopped: boolean) => {
    const activeTake = activeTakeRef.current;
    if (!isRecording || !activeTake || finalizingStopRef.current) return;

    finalizingStopRef.current = true;
    clearTimers();
    setFeedback(autoStopped ? 'Auto-stop reached. Finalizing take...' : 'Stopping take and preparing review...');

    try {
      const durationMs = await AudioCapture.stopSession() as number;
      await stopActiveVideoRecording().catch(() => {});
      await finalizeRecordingStop(durationMs, autoStopped, activeTake.filePath);
      return;
    } catch (error: any) {
      setFeedback(`Fel vid stopp: ${error?.message ?? 'unknown error'}`);
    } finally {
      activeTakeRef.current = null;
      currentTakeImuRef.current = null;
      setIsRecording(false);
      setElapsedMs(0);
      setRemainingMs(TARGET_DURATION_MS);
      finalizingStopRef.current = false;
    }
  }, [clearTimers, finalizeRecordingStop, isRecording, stopActiveVideoRecording]);

  useEffect(() => {
    const sub = AudioCaptureEmitter.addListener(
      AUDIO_CAPTURE_STOPPED_EVENT,
      (payload: AudioCaptureStoppedEvent) => {
        const activeTake = activeTakeRef.current;
        if (!activeTake || finalizingStopRef.current) return;
        if (payload.outputPath !== activeTake.filePath) return;

        finalizingStopRef.current = true;
        clearTimers();
        setFeedback('Auto-stop reached. Finalizing take...');
        void stopActiveVideoRecording()
          .catch(() => {})
          .then(() => finalizeRecordingStop(Math.round(payload.durationMs), true, payload.outputPath))
          .catch((error: any) => {
            setFeedback(`Fel vid auto-stop: ${error?.message ?? 'unknown error'}`);
            activeTakeRef.current = null;
            currentTakeImuRef.current = null;
            setIsRecording(false);
            setElapsedMs(0);
            setRemainingMs(TARGET_DURATION_MS);
          })
          .finally(() => {
            finalizingStopRef.current = false;
          });
      },
    );

    return () => {
      sub.remove();
    };
  }, [clearTimers, finalizeRecordingStop, stopActiveVideoRecording]);

  const startRecording = useCallback(async () => {
    if (!permissionGranted || isRecording || isStartingRecordingRef.current || !selectedScenario) return;
    if (!hasCameraPermission || !cameraDevice || !cameraReady) {
      Alert.alert('Camera not ready', 'Wait for the camera preview before starting the take.');
      return;
    }
    if (isAudioImuMode && !isSensorConnected) {
      Alert.alert('Sensor disconnected', 'Reconnect and recalibrate AirHive before recording a synced take.');
      return;
    }
    if ((selectedSummary?.remaining_takes ?? 0) <= 0) {
      Alert.alert('Scenario klart', 'Valt scenario har redan natt malet. Valj nasta scenario.');
      return;
    }
    if (!sessionDirRef.current || !sessionJsonPathRef.current) {
      await prepareNewSession();
    }

    isStartingRecordingRef.current = true;
    setIsStartingRecording(true);
    setFeedback('Starting take...');

    try {
      await withTimeout(
        videoFinalizeBarrierRef.current,
        VIDEO_FINALIZE_TIMEOUT_MS + 1000,
        'Previous review video is still finalizing.',
      );
      const waitMs = Math.max(0, videoCooldownUntilRef.current - Date.now());
      if (waitMs > 0) {
        setFeedback('Waiting for camera recorder to become ready...');
        await sleep(waitMs);
      }

      const takeIndex = (selectedSummary?.completed_takes ?? 0) + 1;
      const filename = `${selectedScenario.id}_${String(takeIndex).padStart(3, '0')}.wav`;
      const filePath = `${sessionDirRef.current}/${filename}`;
      const videoFilename = `${selectedScenario.id}_${String(takeIndex).padStart(3, '0')}.mp4`;
      const videoPath = `${sessionDirRef.current}/${videoFilename}`;
      const recorder = await videoOutput.createRecorder({});
      let resolveFinish!: (path: string) => void;
      let rejectFinish!: (error: Error) => void;
      const finishPromise = new Promise<string>((resolve, reject) => {
        resolveFinish = resolve;
        rejectFinish = reject;
      });
      videoFinalizeBarrierRef.current = finishPromise.then(() => undefined).catch(() => undefined);
      const videoStartedAtMs = Date.now();
      currentTakeVideoRef.current = {
        recorder,
        filename: videoFilename,
        targetPath: videoPath,
        started_at_ms: videoStartedAtMs,
        finishPromise,
        resolveFinish,
        rejectFinish,
      };
      await recorder.startRecording(
        (tempPath: string) => {
          if (currentTakeVideoRef.current?.filename === videoFilename) {
            currentTakeVideoRef.current.tempPath = tempPath;
          }
          resolveFinish(tempPath);
        },
        (error: Error) => {
          rejectFinish(error);
        },
      );
      await AudioCapture.startSession(filePath, TARGET_DURATION_MS);

      activeTakeRef.current = {
        scenario_id: selectedScenario.id,
        filename,
        filePath,
        videoFilename,
        videoFilePath: videoPath,
        take_index: takeIndex,
      };
      startTimeRef.current = Date.now();
      currentTakeImuRef.current = isAudioImuMode ? {
        started_at_ms: startTimeRef.current,
        ended_at_ms: startTimeRef.current,
        sample_hz_estimate: 0,
        sample_count: 0,
        samples: [],
      } : null;
      setElapsedMs(0);
      setRemainingMs(TARGET_DURATION_MS);
      setIsRecording(true);
      setFeedback(
        `${selectedScenario.title} | take ${takeIndex}/${selectedScenario.target_takes} running`,
      );
      isStartingRecordingRef.current = false;
      setIsStartingRecording(false);

      timerRef.current = setInterval(() => {
        const elapsed = Date.now() - startTimeRef.current;
        setElapsedMs(elapsed);
        setRemainingMs(Math.max(0, TARGET_DURATION_MS - elapsed));
      }, 200);

      stopTimeoutRef.current = setTimeout(() => {
        stopRecording(true).catch(() => {});
      }, WATCHDOG_STOP_MS);
    } catch (error: any) {
      const videoRecording = currentTakeVideoRef.current;
      currentTakeVideoRef.current = null;
      if (videoRecording?.recorder?.isRecording) {
        await videoRecording.recorder.cancelRecording().catch(() => {});
      }
      await withTimeout(
        videoFinalizeBarrierRef.current.catch(() => undefined),
        VIDEO_FINALIZE_TIMEOUT_MS,
        'Recorder cleanup timeout after start failure.',
      ).catch(() => {});
      videoCooldownUntilRef.current = Date.now() + VIDEO_RECORDER_COOLDOWN_MS;
      setFeedback(`Fel vid start: ${error?.message ?? 'unknown error'}`);
      isStartingRecordingRef.current = false;
      setIsStartingRecording(false);
    }
  }, [
    cameraDevice,
    hasCameraPermission,
    cameraReady,
    isAudioImuMode,
    isRecording,
    isSensorConnected,
    permissionGranted,
    prepareNewSession,
    selectedScenario,
    selectedSummary,
    stopRecording,
    videoOutput,
  ]);

  const undoLastTake = useCallback(async () => {
    if (isRecording) return;
    if (eventsRef.current.length === 0) {
      Alert.alert('Ingen data', 'Det finns ingen tagning att ta bort.');
      return;
    }

    const removed = eventsRef.current[eventsRef.current.length - 1];
    await removeCurrentEventByIndex(eventsRef.current.length - 1, removed);
    setFeedback(`Tog bort senaste tagningen: ${removed.scenario_id} #${removed.take_index}`);
    await refreshPendingReviews();
  }, [isRecording, refreshPendingReviews, removeCurrentEventByIndex]);

  const resetSession = useCallback(async () => {
    if (isRecording) return;
    await prepareNewSession();
    eventsRef.current = [];
    setEvents([]);
    setSelectedScenarioId('racket_quiet');
    setFeedback('Ny session skapad.');
  }, [isRecording, prepareNewSession]);

  const openNextPendingReview = useCallback(() => {
    if (pendingReviews.length === 0) {
      Alert.alert('Ingen review-ko', 'Det finns inga vÃ¤ntande takes att reviewa.');
      return;
    }
    setReviewTarget(pendingReviews[0]);
  }, [pendingReviews]);

  const saveReview = useCallback(async (markers: AudioReviewMarker[]) => {
    const target = reviewTarget;
    if (!target) return;

    const session = await readSessionFile(target.sessionJsonPath);
    if (!session || !session.events[target.eventIndex]) {
      throw new Error('Review target missing on disk.');
    }

    const nextEvents = [...session.events];
    nextEvents[target.eventIndex] = {
      ...nextEvents[target.eventIndex],
      review: {
        required: true,
        anchor_rule: 'attack_start',
        completed_at: new Date().toISOString(),
        markers: [...markers].sort((a, b) => a.timestamp_ms - b.timestamp_ms),
      },
    };

    const nextSession = {
      ...session,
      events: nextEvents,
    };
    await writeSessionFile(target.sessionJsonPath, nextSession);

    if (target.sessionJsonPath === sessionJsonPathRef.current) {
      eventsRef.current = nextEvents;
      setEvents(nextEvents);
    }

    setReviewTarget(null);
    setFeedback(`Review sparad: ${target.event.scenario_id} #${target.event.take_index}`);
    await refreshPendingReviews();
  }, [refreshPendingReviews, reviewTarget]);

  const discardReview = useCallback(async () => {
    const target = reviewTarget;
    if (!target) return;

    if (target.sessionJsonPath === sessionJsonPathRef.current) {
      await removeCurrentEventByIndex(target.eventIndex, target.event);
    } else {
      const session = await readSessionFile(target.sessionJsonPath);
      if (session && session.events[target.eventIndex]) {
        const nextEvents = session.events.filter((_, index) => index !== target.eventIndex);
        await writeSessionFile(target.sessionJsonPath, { ...session, events: nextEvents });
      }
      const wavPath = `${target.sessionDir}/${target.event.wav_filename}`;
      if (await RNFS.exists(wavPath)) {
        await RNFS.unlink(wavPath).catch(() => {});
      }
      if (target.event.video_recording?.video_filename) {
        const videoPath = `${target.sessionDir}/${target.event.video_recording.video_filename}`;
        if (await RNFS.exists(videoPath)) {
          await RNFS.unlink(videoPath).catch(() => {});
        }
      }
    }

    setReviewTarget(null);
    setFeedback(`Tagning kastad: ${target.event.scenario_id} #${target.event.take_index}`);
    await refreshPendingReviews();
  }, [refreshPendingReviews, removeCurrentEventByIndex, reviewTarget]);

  if (reviewTarget) {
    return (
      <AudioTakeReviewScreen
        event={reviewTarget.event}
        filePath={`${reviewTarget.sessionDir}/${reviewTarget.event.wav_filename}`}
        videoFilePath={reviewTarget.event.video_recording?.video_filename
          ? `${reviewTarget.sessionDir}/${reviewTarget.event.video_recording.video_filename}`
          : undefined}
        onSave={saveReview}
        onDiscard={discardReview}
        onBack={() => setReviewTarget(null)}
      />
    );
  }

  return (
    <ScrollView style={styles.root} contentContainerStyle={styles.content}>
      <StatusBar barStyle="light-content" backgroundColor="#0d0d0d" />

      <View style={styles.header}>
        <View>
          <Text style={styles.playerName}>{setup.name}</Text>
          <Text style={styles.playerMeta}>
            {isAudioImuMode
              ? 'Guided audio + IMU collection | review audio as usual and save synced AirHive motion'
              : 'Guided audio collection | review noisy takes into racket_contact / not_racket_contact'}
          </Text>
        </View>
        {isRecording && (
          <View style={styles.recBadge}>
            <Text style={styles.recBadgeTxt}>REC {formatDuration(remainingMs)} left</Text>
          </View>
        )}
      </View>

      {!permissionGranted && (
        <View style={styles.warnBox}>
          <Text style={styles.warnTxt}>Mikrofontillstand saknas. Bevilja tillstand for att samla ljuddata.</Text>
        </View>
      )}

      {isAudioImuMode && (
        <View style={[styles.progressCard, !isSensorConnected && styles.warnBox]}>
          <Text style={styles.sectionLabel}>SYNCED IMU</Text>
          <Text style={styles.progressMain}>
            {isSensorConnected ? `${sampleHz} Hz` : 'Sensor disconnected'}
          </Text>
          <Text style={styles.progressSub}>
            Table calibration: {calibration?.calibration_id ?? 'missing'} | Audio review remains the source of truth.
          </Text>
        </View>
      )}

      {feedback && <Text style={styles.feedbackTxt}>{feedback}</Text>}

      <View style={styles.progressCard}>
        <Text style={styles.sectionLabel}>CURRENT SESSION</Text>
        <Text style={styles.progressMain}>
          {collectionSummary.completed_takes}/{collectionSummary.total_takes} takes klara
        </Text>
        <Text style={styles.progressSub}>
          Reviewed: {collectionSummary.reviewed_takes} | Auto saved: {collectionSummary.auto_saved_takes}
        </Text>
        <Text style={styles.progressSub}>
          Pending in current session: {collectionSummary.pending_review_takes}
        </Text>
      </View>

      <View style={styles.progressCard}>
        <Text style={styles.sectionLabel}>REVIEW QUEUE</Text>
        <Text style={styles.progressMain}>{pendingReviews.length}</Text>
        <Text style={styles.progressSub}>
          Pending takes across saved sessions. Use this queue to review older takes before retraining.
        </Text>
        <TouchableOpacity
          style={[styles.secondaryBtn, styles.queueBtn, pendingReviews.length === 0 && styles.disabledBtn]}
          onPress={openNextPendingReview}
          disabled={pendingReviews.length === 0}
        >
          <Text style={styles.secondaryBtnTxt}>Review next pending</Text>
        </TouchableOpacity>
        {pendingReviews.slice(0, 5).map(item => (
          <Text key={`${item.sessionJsonPath}-${item.eventIndex}`} style={styles.queueRow}>
            {item.event.scenario_id} | take {item.event.take_index} | {item.sessionJsonPath.split('/').pop()}
          </Text>
        ))}
      </View>

      <View style={styles.currentCard}>
        <Text style={styles.sectionLabel}>ACTIVE SCENARIO</Text>
        <Text style={styles.currentTitle}>{selectedScenario.title}</Text>
        <Text style={styles.currentPrompt}>{selectedScenario.prompt}</Text>
        <Text style={styles.currentMeta}>
          Label: {selectedScenario.label} | Remaining: {selectedSummary?.remaining_takes ?? 0} / {selectedScenario.target_takes}
        </Text>
      </View>

      <View style={styles.previewCard}>
        <Text style={styles.sectionLabel}>REVIEW VIDEO</Text>
        <Text style={styles.previewHelp}>
          Video is only for easier review. WAV audio is still the training source.
        </Text>
        <View style={styles.cameraFacingRow}>
          {(['front', 'back'] as CameraFacing[]).map(facing => {
            const available = facing === 'front' ? !!frontCameraDevice : !!backCameraDevice;
            const active = preferredCameraFacing === facing;
            return (
              <TouchableOpacity
                key={facing}
                style={[
                  styles.cameraFacingBtn,
                  active && styles.cameraFacingBtnActive,
                  !available && styles.disabledBtn,
                ]}
                onPress={() => {
                  if (available) {
                    setPreferredCameraFacing(facing);
                  }
                }}
                disabled={!available || isRecording}
                activeOpacity={0.8}
              >
                <Text style={[styles.cameraFacingTxt, active && styles.cameraFacingTxtActive]}>
                  {facing === 'front' ? 'Front camera' : 'Back camera'}
                </Text>
              </TouchableOpacity>
            );
          })}
        </View>
        <Text style={styles.previewMeta}>
          Active camera: {effectiveCameraFacing ?? 'none'} | Auto-stop should open review directly after ~{TARGET_DURATION_S}s.
        </Text>
        {hasCameraPermission && cameraDevice ? (
          <View style={styles.cameraFrame}>
            <Camera
              key={cameraDevice.id}
              style={styles.cameraPreview}
              device={cameraDevice}
              isActive
              outputs={[videoOutput]}
              resizeMode="contain"
              onConfigured={() => setCameraReady(true)}
              onStarted={() => setCameraReady(true)}
              onStopped={() => setCameraReady(false)}
              onError={error => {
                setCameraReady(false);
                setFeedback(`Camera error: ${error.message}`);
              }}
            />
            <View style={styles.cameraOverlay}>
              <Text style={styles.cameraOverlayTxt}>
                {cameraReady ? (isRecording ? 'VIDEO REC ON' : 'VIDEO READY') : 'PREPARING CAMERA'}
              </Text>
            </View>
          </View>
        ) : (
          <View style={styles.warnBox}>
            <Text style={styles.warnTxt}>Camera permission missing or no camera is available on this device.</Text>
          </View>
        )}
      </View>

      <View style={styles.infoCard}>
        <Text style={styles.infoTitle}>Why these scenarios exist</Text>
        <Text style={styles.infoText}>
          racket_counting and racket_music_* create true racket-contact positives inside speech and music noise.{'\n\n'}
          speech_only, music_*_only, desk_keyboard_only, table_quiet, and floor_quiet create negatives from the same noise families.{'\n\n'}
          Review turns each transient into a binary training clip: Racket = racket_contact, Not racket = not_racket_contact, Ignore = skipped.{'\n\n'}
          {isAudioImuMode
            ? 'This mode also saves synchronized AirHive IMU during the same take so a future bounce-motion model can be trained from the same reviewed markers.'
            : 'Keep about 0.5-1.0 seconds between bounces in this round. Do not collect fast double contacts yet.'}
          {isAudioImuMode ? `\n\nKeep about 0.5-1.0 seconds between bounces in this round. Do not collect fast double contacts yet.` : ''}
        </Text>
      </View>

      <Text style={styles.sectionLabel}>SCENARIOS</Text>
      <View style={styles.scenarioList}>
        {AUDIO_SCENARIOS.map(scenario => {
          const summary = scenarioSummaries.find(item => item.scenario_id === scenario.id)!;
          const isSelected = selectedScenarioId === scenario.id;
          const isDone = summary.remaining_takes === 0;
          const reviewRequired = requiresAudioReview(scenario.id);

          return (
            <TouchableOpacity
              key={scenario.id}
              style={[
                styles.scenarioCard,
                { backgroundColor: scenario.bg },
                isSelected && { borderColor: scenario.color, borderWidth: 2 },
                isDone && styles.scenarioDone,
              ]}
              activeOpacity={0.8}
              disabled={isRecording || isStartingRecording}
              onPress={() => setSelectedScenarioId(scenario.id)}
            >
              <View style={styles.scenarioHeader}>
                <Text style={[styles.scenarioTitle, { color: scenario.color }]}>{scenario.title}</Text>
                <Text style={styles.scenarioCount}>
                  {summary.completed_takes}/{scenario.target_takes}
                </Text>
              </View>
              <Text style={styles.scenarioPrompt}>{scenario.prompt}</Text>
              <Text style={styles.scenarioMeta}>
                {scenario.label} | {scenario.background_condition} | {reviewRequired ? 'review after take' : 'auto save'}
              </Text>
            </TouchableOpacity>
          );
        })}
      </View>

      {!isRecording ? (
        <TouchableOpacity
          style={[styles.recordBtn, !canRecord && styles.recordBtnOff]}
          onPress={startRecording}
          disabled={!canRecord}
          activeOpacity={0.75}
        >
          <Text style={[styles.recordBtnTxt, !canRecord && styles.recordBtnTxtOff]}>
            {isStartingRecording ? 'STARTING TAKE...' : 'START 30S TAKE'}
          </Text>
          <Text style={styles.recordBtnSub}>
            {(selectedSummary?.remaining_takes ?? 0) > 0
              ? `${selectedScenario.title} | take ${(selectedSummary?.completed_takes ?? 0) + 1}/${selectedScenario.target_takes} | keep 0.5-1.0s between bounces`
              : 'Scenario already complete. Pick another scenario or reset the current session.'}
          </Text>
        </TouchableOpacity>
      ) : (
        <TouchableOpacity
          style={[styles.recordBtn, styles.stopBtn]}
          onPress={() => stopRecording(false)}
          activeOpacity={0.75}
        >
          <Text style={styles.stopBtnTxt}>STOP NOW</Text>
          <Text style={styles.recordBtnSub}>
            {selectedScenario.title} | elapsed {formatDuration(elapsedMs)} | native stop around {TARGET_DURATION_S}s
          </Text>
        </TouchableOpacity>
      )}

      <View style={styles.actionRow}>
        <TouchableOpacity style={[styles.secondaryBtn, styles.undoBtn]} onPress={() => undoLastTake().catch(() => {})}>
          <Text style={styles.secondaryBtnTxt}>Undo last take</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.secondaryBtn, styles.resetBtn]} onPress={() => resetSession().catch(() => {})}>
          <Text style={styles.secondaryBtnTxt}>New session</Text>
        </TouchableOpacity>
      </View>

      <TouchableOpacity
        style={[styles.secondaryBtn, styles.finishBtn]}
        onPress={() => {
          if (isAudioImuMode && device) {
            try { device.cancelConnection(); } catch (_) {}
          }
          onDone();
        }}
      >
        <Text style={styles.secondaryBtnTxt}>Back to setup</Text>
      </TouchableOpacity>

      <View style={styles.recentCard}>
        <Text style={styles.sectionLabel}>LATEST TAKES</Text>
        {events.length === 0 && <Text style={styles.emptyTxt}>No takes recorded in the current session yet.</Text>}
        {events.slice(-8).reverse().map((event, index) => (
          <Text key={`${event.scenario_id}-${event.take_index}-${index}`} style={styles.recentRow}>
            {event.scenario_id} | take {event.take_index} | {formatDuration(event.duration_ms)} |{' '}
            {event.review?.completed_at ? 'reviewed' : event.review?.required ? 'pending review' : 'auto saved'}
          </Text>
        ))}
      </View>

      <View style={styles.infoCard}>
        <Text style={styles.infoTitle}>Workflow</Text>
        <Text style={styles.infoText}>
          1. Record a 30 second take.{'\n'}
          2. Video is recorded alongside WAV for easier review.{'\n'}
          3. Racket and noise scenarios open review directly after the take.{'\n'}
          4. Table and floor scenarios save automatically.{'\n'}
          5. Use Review next pending to revisit older sessions before retraining.{'\n'}
          6. The model learns directly from reviewed transient clips. There is no source separation step.
        </Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#0d0d0d' },
  content: { padding: 20, paddingBottom: 48 },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  playerName: { color: '#fff', fontSize: 20, fontWeight: '700' },
  playerMeta: { color: '#777', fontSize: 12, marginTop: 2 },
  recBadge: {
    backgroundColor: '#2d0d0d',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  recBadgeTxt: { color: '#ff7f7f', fontSize: 13, fontWeight: '700' },
  warnBox: { backgroundColor: '#2d1a00', borderRadius: 10, padding: 12, marginBottom: 12 },
  warnTxt: { color: '#e67e22', fontSize: 13 },
  feedbackTxt: { color: '#4a9eff', fontSize: 13, textAlign: 'center', marginBottom: 10 },
  sectionLabel: { color: '#666', fontSize: 10, letterSpacing: 2, marginBottom: 10, marginTop: 16 },
  progressCard: { backgroundColor: '#111', borderRadius: 14, padding: 16, marginBottom: 12 },
  progressMain: { color: '#fff', fontSize: 24, fontWeight: '800' },
  progressSub: { color: '#888', fontSize: 12, marginTop: 6, lineHeight: 18 },
  currentCard: { backgroundColor: '#111', borderRadius: 14, padding: 16, marginTop: 4 },
  currentTitle: { color: '#f5c76d', fontSize: 18, fontWeight: '800' },
  currentPrompt: { color: '#d2d2d2', fontSize: 13, marginTop: 8, lineHeight: 18 },
  currentMeta: { color: '#777', fontSize: 12, marginTop: 8 },
  previewCard: { backgroundColor: '#111', borderRadius: 14, padding: 16, marginTop: 12 },
  previewHelp: { color: '#888', fontSize: 12, marginBottom: 10, lineHeight: 18 },
  previewMeta: { color: '#6f6f6f', fontSize: 11, marginBottom: 10, lineHeight: 16 },
  cameraFacingRow: { flexDirection: 'row', gap: 8, flexWrap: 'wrap', marginBottom: 10 },
  cameraFacingBtn: {
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#2a2a2a',
    backgroundColor: '#161616',
    paddingVertical: 10,
    paddingHorizontal: 12,
  },
  cameraFacingBtnActive: {
    borderColor: '#2ecc71',
    backgroundColor: '#0d2d1a',
  },
  cameraFacingTxt: { color: '#cfcfcf', fontSize: 12, fontWeight: '700' },
  cameraFacingTxtActive: { color: '#2ecc71' },
  cameraFrame: { borderRadius: 14, overflow: 'hidden', backgroundColor: '#050505', height: 220, justifyContent: 'center' },
  cameraPreview: { ...StyleSheet.absoluteFillObject },
  cameraOverlay: {
    position: 'absolute',
    top: 10,
    right: 10,
    backgroundColor: 'rgba(0,0,0,0.55)',
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 6,
  },
  cameraOverlayTxt: { color: '#fff', fontSize: 11, fontWeight: '700' },
  scenarioList: { gap: 10 },
  scenarioCard: {
    borderRadius: 14,
    padding: 14,
    borderWidth: 1,
    borderColor: '#222',
  },
  scenarioDone: { opacity: 0.75 },
  scenarioHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 6,
  },
  scenarioTitle: { fontSize: 16, fontWeight: '800' },
  scenarioCount: { color: '#fff', fontSize: 14, fontWeight: '700' },
  scenarioPrompt: { color: '#c6c6c6', fontSize: 12, lineHeight: 18 },
  scenarioMeta: { color: '#777', fontSize: 11, marginTop: 8 },
  recordBtn: {
    borderRadius: 14,
    padding: 22,
    marginTop: 16,
    alignItems: 'center',
    backgroundColor: '#0d2d0d',
  },
  recordBtnOff: { backgroundColor: '#151515' },
  recordBtnTxt: { color: '#2ecc71', fontSize: 20, fontWeight: '800', letterSpacing: 1.5 },
  recordBtnTxtOff: { color: '#404040' },
  recordBtnSub: { color: '#888', fontSize: 12, marginTop: 8, textAlign: 'center', lineHeight: 18 },
  stopBtn: { backgroundColor: '#2d0d0d' },
  stopBtnTxt: { color: '#ff7f7f', fontSize: 20, fontWeight: '800', letterSpacing: 1.5 },
  actionRow: { flexDirection: 'row', gap: 12, marginTop: 12 },
  secondaryBtn: {
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: 'center',
    justifyContent: 'center',
  },
  undoBtn: { flex: 1, backgroundColor: '#2b220d' },
  resetBtn: { flex: 1, backgroundColor: '#2d0d0d' },
  queueBtn: { backgroundColor: '#0d1f33', marginTop: 8 },
  finishBtn: { backgroundColor: '#111', borderWidth: 1, borderColor: '#242424', marginTop: 12 },
  disabledBtn: { opacity: 0.45 },
  secondaryBtnTxt: { color: '#f0f0f0', fontWeight: '700' },
  queueRow: { color: '#d0d0d0', fontSize: 11, marginTop: 8, fontFamily: 'monospace' },
  recentCard: { backgroundColor: '#111', borderRadius: 14, padding: 16, marginTop: 16 },
  emptyTxt: { color: '#666', fontSize: 12 },
  recentRow: { color: '#d0d0d0', fontSize: 11, marginBottom: 6, fontFamily: 'monospace' },
  infoCard: {
    marginTop: 16,
    borderWidth: 1,
    borderColor: '#242424',
    borderRadius: 14,
    padding: 16,
  },
  infoTitle: { color: '#fff', fontSize: 15, fontWeight: '700', marginBottom: 8 },
  infoText: { color: '#888', fontSize: 12, lineHeight: 18 },
});
