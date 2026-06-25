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
  Vibration,
  View,
} from 'react-native';
import type { BleError, Characteristic, Device } from 'react-native-ble-plx';
import RNFS from 'react-native-fs';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Camera, useCameraDevice, useCameraPermission, useVideoOutput, type Recorder } from 'react-native-vision-camera';
import {
  AUDIO_CAPTURE_STOPPED_EVENT,
  AudioCapture,
  AudioCaptureEmitter,
  type ImportedAudioFile,
  type AudioCaptureStoppedEvent,
} from './NativeAudioCapture';
import { AudioTakeReviewScreen } from './AudioTakeReviewScreen';
import { getDefaultAudioDetectionConfigSnapshot } from './audioDetectionConfig';
import { requiresAudioReview } from './audioReview';
import { VideoSegment } from './NativeVideoSegment';
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
  AudioBounceContext,
  AudioCalibrationStatus,
  AudioImuRecording,
  AudioVideoRecording,
  AudioLabel,
  AudioModelCandidate,
  AudioRecordingScenario,
  AudioDetectionConfigSnapshot,
  AudioReviewMarker,
  AudioTakeReviewSaveOptions,
  AudioScenarioId,
  AudioSessionFile,
  AudioVideoSyncMetadata,
  CalibrationData,
  ImuSample,
  PlayerSetup,
  VideoPoseCandidate,
} from './types';

const APP_VERSION = '1.7';
const TARGET_DURATION_S = 30;
const TARGET_DURATION_MS = TARGET_DURATION_S * 1000;
const COUNTDOWN_S = 3;
const IMU_TARGET_HZ = 150;
const SYNC_CUE_MS = 3000;
const SYNC_CUE_TEXT = 'Synka: klappa en gång framför kameran.';
const WATCHDOG_STOP_MS = TARGET_DURATION_MS + 5000;
const VIDEO_STOP_TIMEOUT_MS = 4000;
const VIDEO_FINALIZE_TIMEOUT_MS = 5000;
const VIDEO_RECORDER_COOLDOWN_MS = 750;
const SESSION_DIR = `${RNFS.ExternalStorageDirectoryPath}/Download/pingis_sessions`;
const PROGRESS_RING_SEGMENTS = 40;
const PROGRESS_RING_SIZE = 62;
const PROGRESS_RING_RADIUS = 25;
type CameraFacing = 'front' | 'back';
type RecordingPhase = 'ready' | 'countdown' | 'recording' | 'finalizing' | 'done';
type AudioCollectionMode = 'audio_only' | 'audio_imu' | 'free_recording' | 'audio_video_pose';
type MusicLevel = 'music_low' | 'music_mid' | 'music_high';

const MUSIC_LEVEL_OPTIONS: Array<{ id: MusicLevel; title: string }> = [
  { id: 'music_low', title: 'Låg' },
  { id: 'music_mid', title: 'Medel' },
  { id: 'music_high', title: 'Hög' },
];

interface AudioScenarioDefinition {
  id: AudioScenarioId;
  title: string;
  prompt: string;
  label: AudioLabel;
  background_condition: AudioBackgroundCondition;
  target_takes: number;
  color: string;
  bg: string;
  scenario: AudioRecordingScenario;
  bounce_context?: AudioBounceContext;
}

interface PendingReviewItem {
  sessionJsonPath: string;
  sessionDir: string;
  eventIndex: number;
  event: AudioEvent;
}

const AUDIO_ONLY_SCENARIOS: AudioScenarioDefinition[] = [
  {
    id: 'racket_quiet',
    title: 'Racket lugnt',
    prompt: 'Studsa bollen på racket i lugn miljö. Sikta på tydliga racketljud utan extra bakgrund.',
    label: 'racket_bounce',
    background_condition: 'quiet',
    target_takes: 3,
    color: '#2ecc71',
    bg: '#0d2d1a',
    scenario: 'audio_sound',
  },
  {
    id: 'racket_speech',
    title: 'Racket + prat',
    prompt: 'Studsa bollen på racket medan någon pratar eller räknar i bakgrunden.',
    label: 'racket_bounce',
    background_condition: 'speech',
    target_takes: 3,
    color: '#52d884',
    bg: '#123720',
    scenario: 'audio_sound',
  },
  {
    id: 'racket_music',
    title: 'Racket + musik',
    prompt: 'Studsa bollen på racket med musik i bakgrunden. Välj ljudnivå före start.',
    label: 'racket_bounce',
    background_condition: 'music_mid',
    target_takes: 3,
    color: '#7ee39f',
    bg: '#173f26',
    scenario: 'audio_sound',
  },
  {
    id: 'racket_other_bounces',
    title: 'Racket + andra studs',
    prompt: 'Studsa bollen på racket medan andra pingis-, bords- eller golvstudsar hörs i bakgrunden.',
    label: 'racket_bounce',
    background_condition: 'mixed',
    target_takes: 3,
    color: '#9be66d',
    bg: '#1c3314',
    scenario: 'audio_sound',
  },
  {
    id: 'racket_fast',
    title: 'Racket snabbt',
    prompt: 'Studsa i normalt/snabbt tempo, ungefär 40 racketträffar på 25 sekunder. Granska varje tydlig racketträff.',
    label: 'racket_bounce',
    background_condition: 'mixed',
    target_takes: 3,
    color: '#c0f06d',
    bg: '#243814',
    scenario: 'audio_sound',
  },
  {
    id: 'playing_dense_audio',
    title: 'Spel: racket + bord',
    prompt: 'Spela en tät sekvens där bordsstuds och racketträff kommer nära varandra. Märk både racket och bord i Review.',
    label: 'unlabeled',
    background_condition: 'mixed',
    target_takes: 3,
    color: '#35c7ff',
    bg: '#0d2633',
    scenario: 'audio_sound',
  },
  {
    id: 'table_bounce',
    title: 'Bordsstuds lugnt',
    prompt: 'Studsa bollen på ett pingisbord eller pingisbordsliknande spelyta. Inte annan bordsyta.',
    label: 'table_bounce',
    background_condition: 'quiet',
    target_takes: 3,
    color: '#4a9eff',
    bg: '#0d1f33',
    scenario: 'audio_sound',
  },
  {
    id: 'table_noisy',
    title: 'Bordsstuds stökigt',
    prompt: 'Studsa bollen på pingisbord medan prat, musik eller andra studs hörs i bakgrunden.',
    label: 'table_bounce',
    background_condition: 'mixed',
    target_takes: 3,
    color: '#72b6ff',
    bg: '#102842',
    scenario: 'audio_sound',
  },
  {
    id: 'floor_bounce',
    title: 'Golvstuds lugnt',
    prompt: 'Studsa bollen på golvet. Granska varje tydlig golvkontakt som inte racket.',
    label: 'floor_bounce',
    background_condition: 'quiet',
    target_takes: 3,
    color: '#e67e22',
    bg: '#2d1a00',
    scenario: 'audio_sound',
  },
  {
    id: 'floor_noisy',
    title: 'Golvstuds stökigt',
    prompt: 'Studsa bollen på golvet medan prat, musik eller andra studs hörs i bakgrunden.',
    label: 'floor_bounce',
    background_condition: 'mixed',
    target_takes: 3,
    color: '#f2a33c',
    bg: '#332000',
    scenario: 'audio_sound',
  },
  {
    id: 'other_bounce_noise',
    title: 'Brus/negativt',
    prompt: 'Spela in prat, musik, fångljud, klapp, steg eller andra studs utan din racketkontakt.',
    label: 'noise',
    background_condition: 'impact',
    target_takes: 3,
    color: '#ffc09f',
    bg: '#3a241e',
    scenario: 'audio_sound',
  },
];

const AUDIO_IMU_SCENARIOS: AudioScenarioDefinition[] = [
  {
    id: 'racket_bounce_fh',
    title: 'Forehand-sida',
    prompt: 'Kontrollerad racketstuds på forehand-sidan. Detta är studs-sida, inte forehand-slag i spel.',
    label: 'racket_bounce',
    background_condition: 'quiet',
    target_takes: 3,
    color: '#2ecc71',
    bg: '#0d2d1a',
    scenario: 'racket_bouncing',
    bounce_context: 'forehand_side',
  },
  {
    id: 'racket_bounce_bh',
    title: 'Backhand-sida',
    prompt: 'Kontrollerad racketstuds på backhand-sidan. Detta är studs-sida, inte backhand-slag i spel.',
    label: 'racket_bounce',
    background_condition: 'quiet',
    target_takes: 3,
    color: '#5fd18b',
    bg: '#123720',
    scenario: 'racket_bouncing',
    bounce_context: 'backhand_side',
  },
  {
    id: 'racket_motion_no_bounce',
    title: 'Racketrörelse utan studs',
    prompt: 'Rör racketarm och handled naturligt utan boll på racket. Variera grepp, handledsvridning, lyft/sänk, fram/bak och tempo.',
    label: 'unlabeled',
    background_condition: 'mixed',
    target_takes: 3,
    color: '#ffb04f',
    bg: '#33200b',
    scenario: 'racket_bouncing',
  },
  {
    id: 'playing_dense_imu',
    title: 'Playing: racket + bord',
    prompt: 'Spela fritt med täta bordsstudsar och racketträffar. Samlar ljud, video och optional IMU; märk händelser i Review.',
    label: 'unlabeled',
    background_condition: 'mixed',
    target_takes: 1,
    color: '#b06cff',
    bg: '#211333',
    scenario: 'playing',
  },
];

const FREE_RECORDING_SCENARIOS: AudioScenarioDefinition[] = [
  {
    id: 'free_recording',
    title: 'Playing',
    prompt: 'Spela in en längre sekvens med video, ljud och optional IMU. Märk händelser i efterhand.',
    label: 'unlabeled',
    background_condition: 'mixed',
    target_takes: 1,
    color: '#b06cff',
    bg: '#211333',
    scenario: 'playing',
  },
];

const AUDIO_VIDEO_POSE_SCENARIOS: AudioScenarioDefinition[] = [
  {
    id: 'free_recording',
    title: 'Ljud + video ML',
    prompt: 'Spela in en längre pingissekvens eller importera MP4. Review börjar med ljud och går sedan vidare till rörelseförslag.',
    label: 'unlabeled',
    background_condition: 'mixed',
    target_takes: 999,
    color: '#35c7ff',
    bg: '#0d2633',
    scenario: 'playing',
  },
];

interface ScenarioGroupDefinition {
  id: 'racket' | 'table' | 'floor' | 'noise' | 'free' | 'playing';
  title: string;
  icon: string;
  color: string;
  scenarioIds: AudioScenarioId[];
}

const AUDIO_ONLY_GROUPS: ScenarioGroupDefinition[] = [
  {
    id: 'racket',
    title: 'Racket',
    icon: 'R',
    color: '#2ee678',
    scenarioIds: ['racket_quiet', 'racket_speech', 'racket_music', 'racket_other_bounces', 'racket_fast'],
  },
  { id: 'playing', title: 'Spel', icon: 'S', color: '#35c7ff', scenarioIds: ['playing_dense_audio'] },
  { id: 'table', title: 'Bord', icon: 'B', color: '#4a9eff', scenarioIds: ['table_bounce', 'table_noisy'] },
  { id: 'floor', title: 'Golv', icon: 'G', color: '#ffc02f', scenarioIds: ['floor_bounce', 'floor_noisy'] },
  { id: 'noise', title: 'Brus', icon: 'N', color: '#ff5a4f', scenarioIds: ['other_bounce_noise'] },
];

const AUDIO_IMU_GROUPS: ScenarioGroupDefinition[] = [
  {
    id: 'racket',
    title: 'Racketstuds',
    icon: 'R',
    color: '#2ee678',
    scenarioIds: ['racket_bounce_fh', 'racket_bounce_bh', 'racket_motion_no_bounce'],
  },
  { id: 'playing', title: 'Playing', icon: 'P', color: '#b06cff', scenarioIds: ['playing_dense_imu'] },
];

const FREE_RECORDING_GROUPS: ScenarioGroupDefinition[] = [
  { id: 'free', title: 'Fri', icon: 'F', color: '#b06cff', scenarioIds: ['free_recording'] },
];

const AUDIO_VIDEO_POSE_GROUPS: ScenarioGroupDefinition[] = [
  { id: 'playing', title: 'ML', icon: 'M', color: '#35c7ff', scenarioIds: ['free_recording'] },
];

function scenariosForMode(mode: AudioCollectionMode): AudioScenarioDefinition[] {
  if (mode === 'audio_video_pose') return AUDIO_VIDEO_POSE_SCENARIOS;
  if (mode === 'audio_imu') return AUDIO_IMU_SCENARIOS;
  if (mode === 'free_recording') return FREE_RECORDING_SCENARIOS;
  return AUDIO_ONLY_SCENARIOS;
}

function scenarioGroupsForMode(mode: AudioCollectionMode): ScenarioGroupDefinition[] {
  if (mode === 'audio_video_pose') return AUDIO_VIDEO_POSE_GROUPS;
  if (mode === 'audio_imu') return AUDIO_IMU_GROUPS;
  if (mode === 'free_recording') return FREE_RECORDING_GROUPS;
  return AUDIO_ONLY_GROUPS;
}

function initialScenarioIdForMode(mode: AudioCollectionMode): AudioScenarioId {
  if (mode === 'audio_video_pose') return 'free_recording';
  if (mode === 'audio_imu') return 'racket_bounce_fh';
  if (mode === 'free_recording') return 'free_recording';
  return 'racket_quiet';
}

function formatDuration(ms: number): string {
  const s = Math.max(0, Math.ceil(ms / 1000));
  return `${s}s`;
}

function musicLevelTitle(level: MusicLevel): string {
  return MUSIC_LEVEL_OPTIONS.find(option => option.id === level)?.title ?? 'Medel';
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
  return requiresAudioReview(event.scenario_id) && event.review?.required !== false && !event.review?.completed_at;
}

function computeImuQuality(samples: ImuSample[], durationMs: number, partial?: boolean): Partial<AudioImuRecording> {
  const sampleCount = samples.length;
  const sampleHzEstimate = durationMs > 0
    ? Number(((sampleCount * 1000) / durationMs).toFixed(1))
    : 0;
  const intervals = samples
    .map(sample => sample.take_ts_ms ?? sample.ts_ms)
    .slice(1)
    .map((tsMs, index) => tsMs - (samples[index].take_ts_ms ?? samples[index].ts_ms))
    .filter(interval => Number.isFinite(interval) && interval >= 0);

  const sampleIntervalMinMs = intervals.length > 0 ? Number(Math.min(...intervals).toFixed(1)) : undefined;
  const sampleIntervalMaxMs = intervals.length > 0 ? Number(Math.max(...intervals).toFixed(1)) : undefined;
  const sampleIntervalAvgMs = intervals.length > 0
    ? Number((intervals.reduce((sum, interval) => sum + interval, 0) / intervals.length).toFixed(1))
    : undefined;

  let qualityFlag: AudioImuRecording['quality_flag'] = 'unstable';
  if (partial) {
    qualityFlag = 'partial';
  } else if (sampleHzEstimate >= IMU_TARGET_HZ * 0.9 && (sampleIntervalMaxMs ?? 0) <= 40) {
    qualityFlag = 'target_150_met';
  } else if (sampleHzEstimate >= 50) {
    qualityFlag = 'below_target';
  }

  return {
    target_hz: IMU_TARGET_HZ,
    sample_hz_estimate: sampleHzEstimate,
    sample_count: sampleCount,
    sample_interval_min_ms: sampleIntervalMinMs,
    sample_interval_avg_ms: sampleIntervalAvgMs,
    sample_interval_max_ms: sampleIntervalMaxMs,
    quality_flag: qualityFlag,
  };
}

function buildScenarioSummaries(
  events: AudioEvent[],
  scenarios: AudioScenarioDefinition[],
): AudioCollectionScenarioSummary[] {
  return scenarios.map(scenario => {
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

function findNextIncompleteScenario(
  events: AudioEvent[],
  scenarios: AudioScenarioDefinition[],
): AudioScenarioId | null {
  const summaries = buildScenarioSummaries(events, scenarios);
  const next = summaries.find(item => item.remaining_takes > 0);
  return next?.scenario_id ?? null;
}

function calibrationStatusFor(calibration: CalibrationData | undefined, recordsImu: boolean): AudioCalibrationStatus {
  if (!recordsImu) return 'skipped';
  return calibration?.bounce_sides ? 'captured' : 'partial';
}

function buildSessionFile(
  setup: PlayerSetup,
  events: AudioEvent[],
  sessionDate: string,
  mode: AudioCollectionMode,
  calibration?: CalibrationData,
  scenarios = scenariosForMode(mode),
): AudioSessionFile {
  const collectionType = mode === 'audio_video_pose'
    ? 'audio_video_pose'
    : mode === 'audio_imu'
    ? 'audio_video_imu'
    : mode === 'free_recording' && calibration
      ? 'audio_video_imu'
      : 'audio_video_only';

  return {
    session_meta: {
      recorder_name: setup.name,
      player_name: setup.name,
      handedness: setup.handedness,
      session_date: sessionDate,
      app_version: APP_VERSION,
      clip_duration_ms: 0,
      collection_mode: mode === 'audio_video_pose'
        ? 'audio_video_pose'
        : mode === 'free_recording'
        ? 'free_recording'
        : mode === 'audio_imu'
          ? 'guided_scenarios_audio_imu'
          : 'guided_scenarios',
      recording_mode: mode === 'audio_video_pose'
        ? 'audio_video_pose'
        : mode === 'free_recording'
        ? 'free_recording'
        : mode === 'audio_imu'
          ? 'audio_imu'
          : 'guided_audio_only',
      collection_type: collectionType,
      scenarios: Array.from(new Set(scenarios.map(scenario => scenario.scenario))),
      calibration_status: calibrationStatusFor(calibration, collectionType === 'audio_video_imu'),
      detection_config_snapshot: getDefaultAudioDetectionConfigSnapshot(),
      target_duration_s: mode === 'audio_video_pose' || mode === 'free_recording' || scenarios.some(scenario => scenario.scenario === 'playing')
        ? 0
        : TARGET_DURATION_S,
      planned_takes: scenarios.reduce((sum, scenario) => sum + scenario.target_takes, 0),
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

function ProgressRing({ progressPct }: { progressPct: number }) {
  const clampedPct = Math.max(0, Math.min(100, progressPct));
  const activeSegments = Math.round((clampedPct / 100) * PROGRESS_RING_SEGMENTS);

  return (
    <View style={styles.progressRing}>
      {Array.from({ length: PROGRESS_RING_SEGMENTS }, (_, index) => {
        const angleDeg = (index / PROGRESS_RING_SEGMENTS) * 360;
        const angleRad = ((angleDeg - 90) * Math.PI) / 180;
        const left = PROGRESS_RING_SIZE / 2 + Math.cos(angleRad) * PROGRESS_RING_RADIUS - 1.5;
        const top = PROGRESS_RING_SIZE / 2 + Math.sin(angleRad) * PROGRESS_RING_RADIUS - 4;
        return (
          <View
            key={`progress-segment-${index}`}
            style={[
              styles.progressRingSegment,
              {
                left,
                top,
                transform: [{ rotateZ: `${angleDeg}deg` }],
              },
              index < activeSegments && styles.progressRingSegmentActive,
            ]}
          />
        );
      })}
      <View style={styles.progressRingInner}>
        <Text style={styles.progressRingTxt}>{clampedPct}%</Text>
      </View>
    </View>
  );
}

interface Props {
  setup: PlayerSetup;
  mode?: AudioCollectionMode;
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
  const insets = useSafeAreaInsets();
  const scenarioDefinitions = useMemo(() => scenariosForMode(mode), [mode]);
  const scenarioGroups = useMemo(() => scenarioGroupsForMode(mode), [mode]);
  const isAudioVideoPoseMode = mode === 'audio_video_pose';
  const recordsImu = (mode === 'audio_imu' || mode === 'free_recording') && !!device && !!calibration;
  const calibrationStatus = calibrationStatusFor(calibration, recordsImu);
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
  const [selectedScenarioId, setSelectedScenarioId] = useState<AudioScenarioId>(initialScenarioIdForMode(mode));
  const [selectedMusicLevel, setSelectedMusicLevel] = useState<MusicLevel>('music_mid');
  const [recordingPhase, setRecordingPhase] = useState<RecordingPhase>('ready');
  const [countdownValue, setCountdownValue] = useState(COUNTDOWN_S);
  const [isRecording, setIsRecording] = useState(false);
  const [isStartingRecording, setIsStartingRecording] = useState(false);
  const [isImportingAudio, setIsImportingAudio] = useState(false);
  const [isSensorConnected, setIsSensorConnected] = useState(true);
  const [sampleHz, setSampleHz] = useState(0);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [remainingMs, setRemainingMs] = useState(TARGET_DURATION_MS);
  const [events, setEvents] = useState<AudioEvent[]>([]);
  const [permissionGranted, setPermission] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);
  const [pendingReviews, setPendingReviews] = useState<PendingReviewItem[]>([]);
  const [reviewQueueVisible, setReviewQueueVisible] = useState(false);
  const [reviewTarget, setReviewTarget] = useState<PendingReviewItem | null>(null);

  const sessionDirRef = useRef<string | null>(null);
  const sessionJsonPathRef = useRef<string | null>(null);
  const sessionDateRef = useRef<string>(new Date().toISOString());
  const isRecordingRef = useRef(false);
  const isStartingRecordingRef = useRef(false);
  const startTimeRef = useRef<number>(0);
  const countdownTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
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

  const scenarioSummaries = useMemo(
    () => buildScenarioSummaries(events, scenarioDefinitions),
    [events, scenarioDefinitions],
  );
  const collectionSummary = useMemo(
    () => buildCollectionSummary(scenarioSummaries, events),
    [events, scenarioSummaries],
  );
  const selectedScenario = scenarioDefinitions.find(item => item.id === selectedScenarioId) ?? scenarioDefinitions[0];
  const selectedBackgroundCondition: AudioBackgroundCondition = selectedScenario?.id === 'racket_music'
    ? selectedMusicLevel
    : selectedScenario?.background_condition ?? 'quiet';
  const isFreeRecording = isAudioVideoPoseMode || mode === 'free_recording' || selectedScenario?.scenario === 'playing';
  const isBusyWithTake = isRecording || isStartingRecording || isImportingAudio || recordingPhase === 'countdown' || recordingPhase === 'finalizing';
  const selectedSummary =
    scenarioSummaries.find(item => item.scenario_id === selectedScenarioId) ?? scenarioSummaries[0];
  const canRecord =
    permissionGranted &&
    hasCameraPermission &&
    !isRecording &&
    !isStartingRecording &&
    !isImportingAudio &&
    !!cameraDevice &&
    cameraReady &&
    (!recordsImu || isSensorConnected) &&
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

      const receivedAtMs = Date.now();
      const takeTsMs = Math.max(0, receivedAtMs - startTimeRef.current);
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
        received_at_ms: receivedAtMs,
        take_ts_ms: takeTsMs,
        sensor_ts: parsed.sensor_ts,
        ts_ms: takeTsMs,
      };
      currentTakeImuRef.current.samples.push(sample);
    },
    [],
  );

  useEffect(() => {
    if (!recordsImu || !device) {
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
      setFeedback('AirHive frånkopplad.');
      if (currentTakeImuRef.current) {
        currentTakeImuRef.current.disconnected = true;
        currentTakeImuRef.current.partial = true;
      }
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
  }, [device, handleSensorNotification, recordsImu]);

  const refreshPendingReviews = useCallback(async () => {
    const sessionJsonPath = sessionJsonPathRef.current;
    const sessionDir = sessionDirRef.current;
    const queue: PendingReviewItem[] = sessionJsonPath && sessionDir
      ? eventsRef.current
          .map((event, eventIndex) => ({ sessionJsonPath, sessionDir, eventIndex, event }))
          .filter(item => isPendingReview(item.event))
      : [];
    setPendingReviews(queue);
    if (queue.length === 0) {
      setReviewQueueVisible(false);
    }
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
      buildSessionFile(setup, [], sessionDateRef.current, mode, calibration, scenarioDefinitions),
    );
  }, [calibration, mode, scenarioDefinitions, setup]);

  const persistCurrentSession = useCallback(async (nextEvents: AudioEvent[]) => {
    if (!sessionJsonPathRef.current) return;
    await writeSessionFile(
      sessionJsonPathRef.current,
      buildSessionFile(setup, nextEvents, sessionDateRef.current, mode, calibration, scenarioDefinitions),
    );
  }, [calibration, mode, scenarioDefinitions, setup]);

  const clearTimers = useCallback(() => {
    if (countdownTimerRef.current) {
      clearInterval(countdownTimerRef.current);
      countdownTimerRef.current = null;
    }
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
            message: 'Appen behöver mikrofonen för att samla in ljuddata.',
            buttonPositive: 'OK',
          },
        );
        if (result !== PermissionsAndroid.RESULTS.GRANTED) {
          Alert.alert('Tillstånd saknas', 'Mikrofontillstånd behövs för ljudinsamlingen.');
          return;
        }
      }

      if (!hasCameraPermission) {
        const granted = await requestCameraPermission();
        if (!granted) {
          Alert.alert('Tillstånd saknas', 'Kameratillstånd behövs för video-review i ljudinsamlingen.');
          return;
        }
      }

      setPermission(true);
      await prepareNewSession();
      await refreshPendingReviews();
    })().catch(error => {
      Alert.alert('Fel', `Kunde inte förbereda ljudinsamlingen: ${String(error)}`);
    });

    return () => {
      if (countdownTimerRef.current) clearInterval(countdownTimerRef.current);
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
        setFeedback('Videon tog lång tid att färdigställa. Använder sparad videofil för granskning.');
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
      setFeedback(`Videogranskning avstängd för tagningen: ${String(error)}`);
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

    const scenario = scenarioDefinitions.find(item => item.id === activeTake.scenario_id);
    if (!scenario) {
      throw new Error(`Unknown scenario: ${activeTake.scenario_id}`);
    }

    const requiresReview = requiresAudioReview(scenario.id);
    const videoRecording = await finalizeVideoRecording();
    const imuRecording = currentTakeImuRef.current
      ? {
          ...currentTakeImuRef.current,
          ended_at_ms: startTimeRef.current + durationMs,
          ...computeImuQuality(
            currentTakeImuRef.current.samples,
            durationMs,
            currentTakeImuRef.current.partial || currentTakeImuRef.current.disconnected,
          ),
        }
      : undefined;
    const recordedAtIso = new Date(startTimeRef.current).toISOString();

    if (isAudioVideoPoseMode) {
      const sessionDir = sessionDirRef.current;
      if (!sessionDir) {
        throw new Error('Session directory missing while saving audio+video pose take.');
      }

      const takeIndex = activeTake.take_index;
      const event: AudioEvent = {
        label: 'unlabeled',
        recorded_at: recordedAtIso,
        created_at: recordedAtIso,
        wav_filename: activeTake.filename,
        duration_ms: durationMs,
        scenario_id: 'free_recording',
        background_condition: 'mixed',
        take_index: takeIndex,
        target_duration_s: 0,
        recording_mode: 'audio_video_pose',
        collection_type: 'audio_video_pose',
        scenario: 'playing',
        bounce_context: 'mixed',
        calibration_status: 'skipped',
        has_audio: true,
        has_video: Boolean(videoRecording),
        has_imu: false,
        player_handedness: setup.handedness,
        camera_facing: effectiveCameraFacing ?? undefined,
        detection_config_snapshot: getDefaultAudioDetectionConfigSnapshot(),
        review: {
          required: true,
          anchor_rule: 'attack_start',
          review_stage: 'audio',
          markers: [],
        },
        video_recording: videoRecording,
      };

      const nextEvents = [...eventsRef.current, event];
      eventsRef.current = nextEvents;
      setEvents(nextEvents);
      await persistCurrentSession(nextEvents);

      if (sessionJsonPathRef.current) {
        setRecordingPhase('done');
        setReviewTarget({
          sessionJsonPath: sessionJsonPathRef.current,
          sessionDir,
          eventIndex: nextEvents.length - 1,
          event,
        });
      }
      setFeedback('Ljud+video sparad som en hel tagning. Börja med ljudreview.');
      await refreshPendingReviews();
      return;
    }

    const event: AudioEvent = {
      label: scenario.label,
      recorded_at: recordedAtIso,
      created_at: recordedAtIso,
      wav_filename: activeTake.filename,
      duration_ms: durationMs,
      scenario_id: scenario.id,
      background_condition: scenario.id === 'racket_music' ? selectedMusicLevel : scenario.background_condition,
      take_index: activeTake.take_index,
      target_duration_s: isFreeRecording ? 0 : TARGET_DURATION_S,
      recording_mode: isFreeRecording
        ? (mode === 'audio_imu' ? 'audio_imu' : 'free_recording')
        : recordsImu
          ? 'audio_imu'
          : 'guided_audio_only',
      collection_type: recordsImu ? 'audio_video_imu' : 'audio_video_only',
      scenario: scenario.scenario,
      bounce_context: scenario.bounce_context,
      calibration_status: calibrationStatus,
      has_audio: true,
      has_video: !!videoRecording,
      has_imu: !!imuRecording,
      detection_config_snapshot: getDefaultAudioDetectionConfigSnapshot(),
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

    const nextSummary = buildScenarioSummaries(nextEvents, scenarioDefinitions).find(
      item => item.scenario_id === scenario.id,
    );
    if (nextSummary?.remaining_takes === 0) {
      const nextScenarioId = findNextIncompleteScenario(nextEvents, scenarioDefinitions);
      if (nextScenarioId) {
        setSelectedScenarioId(nextScenarioId);
      }
    }

    if (requiresReview && sessionJsonPathRef.current && sessionDirRef.current) {
      setRecordingPhase('done');
      setReviewTarget({
        sessionJsonPath: sessionJsonPathRef.current,
        sessionDir: sessionDirRef.current,
        eventIndex: nextEvents.length - 1,
        event,
      });
      setFeedback(`Granskning väntar: ${scenario.title} tagning ${activeTake.take_index}`);
    } else {
      setRecordingPhase('done');
      setFeedback(
        `${autoStopped ? 'Auto-stopp' : 'Stoppad'} ${scenario.title} tagning ${activeTake.take_index}/${scenario.target_takes}`,
      );
      await refreshPendingReviews();
    }

    activeTakeRef.current = null;
    currentTakeImuRef.current = null;
    setIsRecording(false);
    setElapsedMs(0);
    setRemainingMs(isFreeRecording ? 0 : TARGET_DURATION_MS);
  }, [
    calibrationStatus,
    effectiveCameraFacing,
    finalizeVideoRecording,
    isAudioVideoPoseMode,
    isFreeRecording,
    mode,
    persistCurrentSession,
    recordsImu,
    refreshPendingReviews,
    scenarioDefinitions,
    selectedMusicLevel,
    setup.handedness,
  ]);

  const stopRecording = useCallback(async (autoStopped: boolean) => {
    const activeTake = activeTakeRef.current;
    if (!isRecording || !activeTake || finalizingStopRef.current) return;

    finalizingStopRef.current = true;
    clearTimers();
    setRecordingPhase('finalizing');
    Vibration.vibrate(90);
    setFeedback(autoStopped ? 'Auto-stopp nådd. Färdigställer tagning...' : 'Stoppar tagning och förbereder granskning...');

    try {
      const durationMs = await AudioCapture.stopSession() as number;
      await stopActiveVideoRecording().catch(() => {});
      await finalizeRecordingStop(durationMs, autoStopped, activeTake.filePath);
      return;
    } catch (error: any) {
      setFeedback(`Fel vid stopp: ${error?.message ?? 'unknown error'}`);
      setRecordingPhase('ready');
    } finally {
      activeTakeRef.current = null;
      currentTakeImuRef.current = null;
      setIsRecording(false);
      setElapsedMs(0);
      setRemainingMs(isFreeRecording ? 0 : TARGET_DURATION_MS);
      finalizingStopRef.current = false;
    }
  }, [clearTimers, finalizeRecordingStop, isFreeRecording, isRecording, stopActiveVideoRecording]);

  useEffect(() => {
    const sub = AudioCaptureEmitter.addListener(
      AUDIO_CAPTURE_STOPPED_EVENT,
      (payload: AudioCaptureStoppedEvent) => {
        const activeTake = activeTakeRef.current;
        if (!activeTake || finalizingStopRef.current) return;
        if (payload.outputPath !== activeTake.filePath) return;

        finalizingStopRef.current = true;
        clearTimers();
        setRecordingPhase('finalizing');
        Vibration.vibrate(90);
        setFeedback('Auto-stopp nådd. Färdigställer tagning...');
        void stopActiveVideoRecording()
          .catch(() => {})
          .then(() => finalizeRecordingStop(Math.round(payload.durationMs), true, payload.outputPath))
          .catch((error: any) => {
            setFeedback(`Fel vid auto-stop: ${error?.message ?? 'unknown error'}`);
            setRecordingPhase('ready');
            activeTakeRef.current = null;
            currentTakeImuRef.current = null;
            setIsRecording(false);
            setElapsedMs(0);
            setRemainingMs(isFreeRecording ? 0 : TARGET_DURATION_MS);
          })
          .finally(() => {
            finalizingStopRef.current = false;
          });
      },
    );

    return () => {
      sub.remove();
    };
  }, [clearTimers, finalizeRecordingStop, isFreeRecording, stopActiveVideoRecording]);

  const startRecording = useCallback(async () => {
    if (!permissionGranted || isRecording || isStartingRecordingRef.current || !selectedScenario) return;
    if (!hasCameraPermission || !cameraDevice || !cameraReady) {
      Alert.alert('Kamera inte redo', 'Vänta på kameraförhandsvisningen innan du startar tagningen.');
      return;
    }
    if (recordsImu && !isSensorConnected) {
      Alert.alert('Sensor frånkopplad', 'Anslut och kalibrera AirHive igen innan du spelar in en synkad tagning.');
      return;
    }
    if ((selectedSummary?.remaining_takes ?? 0) <= 0) {
      Alert.alert('Scenario klart', 'Valt scenario har redan nått målet. Välj nästa scenario.');
      return;
    }
    if (!sessionDirRef.current || !sessionJsonPathRef.current) {
      await prepareNewSession();
    }

    isStartingRecordingRef.current = true;
    setIsStartingRecording(true);
    setFeedback('Startar tagning...');

    try {
      await withTimeout(
        videoFinalizeBarrierRef.current,
        VIDEO_FINALIZE_TIMEOUT_MS + 1000,
        'Previous review video is still finalizing.',
      );
      const waitMs = Math.max(0, videoCooldownUntilRef.current - Date.now());
      if (waitMs > 0) {
        setFeedback('Väntar på att kameran blir redo...');
        await sleep(waitMs);
      }

      const takeIndex = (selectedSummary?.completed_takes ?? 0) + 1;
      const sourcePrefix = isAudioVideoPoseMode ? 'audio_video_pose_source' : selectedScenario.id;
      const filename = `${sourcePrefix}_${String(takeIndex).padStart(3, '0')}.wav`;
      const filePath = `${sessionDirRef.current}/${filename}`;
      const videoFilename = `${sourcePrefix}_${String(takeIndex).padStart(3, '0')}.mp4`;
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
      const targetDurationMs = isFreeRecording ? 0 : TARGET_DURATION_MS;
      await AudioCapture.startSession(filePath, targetDurationMs);

      activeTakeRef.current = {
        scenario_id: selectedScenario.id,
        filename,
        filePath,
        videoFilename,
        videoFilePath: videoPath,
        take_index: takeIndex,
      };
      startTimeRef.current = Date.now();
      currentTakeImuRef.current = recordsImu ? {
        started_at_ms: startTimeRef.current,
        ended_at_ms: startTimeRef.current,
        target_hz: IMU_TARGET_HZ,
        sample_hz_estimate: 0,
        sample_count: 0,
        quality_flag: 'unstable',
        disconnected: false,
        partial: false,
        samples: [],
      } : null;
      setElapsedMs(0);
      setRemainingMs(targetDurationMs);
      setIsRecording(true);
      setRecordingPhase('recording');
      Vibration.vibrate(80);
      setFeedback(
        isFreeRecording
          ? 'Sync först: klappa en gång framför kameran. Stoppa när sekvensen är klar.'
          : `Sync först: klappa en gång framför kameran. Sedan ${selectedScenario.title}.`,
      );
      isStartingRecordingRef.current = false;
      setIsStartingRecording(false);

      timerRef.current = setInterval(() => {
        const elapsed = Date.now() - startTimeRef.current;
        setElapsedMs(elapsed);
        setRemainingMs(isFreeRecording ? 0 : Math.max(0, TARGET_DURATION_MS - elapsed));
      }, 200);

      if (!isFreeRecording) {
        stopTimeoutRef.current = setTimeout(() => {
          stopRecording(true).catch(() => {});
        }, WATCHDOG_STOP_MS);
      }
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
      setRecordingPhase('ready');
      isStartingRecordingRef.current = false;
      setIsStartingRecording(false);
    }
  }, [
    cameraDevice,
    hasCameraPermission,
    cameraReady,
    isFreeRecording,
    isAudioVideoPoseMode,
    isRecording,
    isSensorConnected,
    permissionGranted,
    prepareNewSession,
    recordsImu,
    selectedScenario,
    selectedMusicLevel,
    selectedSummary,
    stopRecording,
    videoOutput,
  ]);

  const startCountdownAndRecording = useCallback(() => {
    if (!canRecord) return;
    clearTimers();
    let nextValue = COUNTDOWN_S;
    setCountdownValue(nextValue);
    setRecordingPhase('countdown');
    setFeedback('Gor dig redo.');

    countdownTimerRef.current = setInterval(() => {
      nextValue -= 1;
      if (nextValue <= 0) {
        if (countdownTimerRef.current) {
          clearInterval(countdownTimerRef.current);
          countdownTimerRef.current = null;
        }
        setCountdownValue(0);
        startRecording().catch((error: any) => {
          setFeedback(`Fel vid start: ${error?.message ?? 'unknown error'}`);
          setRecordingPhase('ready');
        });
        return;
      }
      setCountdownValue(nextValue);
    }, 1000);
  }, [canRecord, clearTimers, startRecording]);

  const undoLastTake = useCallback(async () => {
    if (isRecording || isStartingRecording || recordingPhase === 'countdown' || recordingPhase === 'finalizing') return;
    if (eventsRef.current.length === 0) {
      Alert.alert('Ingen data', 'Det finns ingen tagning att ta bort.');
      return;
    }

    const removed = eventsRef.current[eventsRef.current.length - 1];
    await removeCurrentEventByIndex(eventsRef.current.length - 1, removed);
    setFeedback(`Tog bort senaste tagningen: ${removed.scenario_id} #${removed.take_index}`);
    await refreshPendingReviews();
  }, [isRecording, isStartingRecording, recordingPhase, refreshPendingReviews, removeCurrentEventByIndex]);

  const resetSession = useCallback(async () => {
    if (isRecording || isStartingRecording || isImportingAudio || recordingPhase === 'countdown' || recordingPhase === 'finalizing') return;
    clearTimers();
    await prepareNewSession();
    eventsRef.current = [];
    setEvents([]);
    setReviewQueueVisible(false);
    setSelectedScenarioId(initialScenarioIdForMode(mode));
    setRecordingPhase('ready');
    setFeedback('Ny session skapad.');
  }, [clearTimers, isImportingAudio, isRecording, isStartingRecording, mode, prepareNewSession, recordingPhase]);

  const importAudioFileForReview = useCallback(async () => {
    if (mode !== 'audio_only' || isBusyWithTake) return;
    if (!sessionDirRef.current || !sessionJsonPathRef.current) {
      await prepareNewSession();
    }
    const sessionDir = sessionDirRef.current;
    const sessionJsonPath = sessionJsonPathRef.current;
    if (!sessionDir || !sessionJsonPath) {
      Alert.alert('Importfel', 'Kunde inte skapa sessionsmapp för ljudimport.');
      return;
    }

    setIsImportingAudio(true);
    setFeedback('Välj en ljudfil från iPhone eller Android...');

    try {
      const takeIndex = eventsRef.current.filter(event => event.scenario_id === 'imported_audio').length + 1;
      const filename = `imported_audio_${String(takeIndex).padStart(3, '0')}.wav`;
      const filePath = `${sessionDir}/${filename}`;
      const imported = await AudioCapture.importAudioFile(filePath) as ImportedAudioFile;
      const nowIso = new Date().toISOString();
      const durationMs = Math.max(0, Math.round(Number(imported.durationMs ?? 0)));
      const event: AudioEvent = {
        label: 'unlabeled',
        recorded_at: nowIso,
        created_at: nowIso,
        imported_at: nowIso,
        wav_filename: filename,
        duration_ms: durationMs,
        scenario_id: 'imported_audio',
        background_condition: 'mixed',
        take_index: takeIndex,
        target_duration_s: 0,
        recording_mode: 'imported_audio',
        collection_type: 'audio_only_import',
        scenario: 'audio_sound',
        has_audio: true,
        has_video: false,
        has_imu: false,
        imported_source_filename: imported.displayName,
        imported_source_uri: imported.sourceUri,
        detection_config_snapshot: getDefaultAudioDetectionConfigSnapshot(),
        review: {
          required: true,
          anchor_rule: 'attack_start',
          markers: [],
        },
      };

      await RNFS.scanFile(filePath).catch(() => {});
      const nextEvents = [...eventsRef.current, event];
      eventsRef.current = nextEvents;
      setEvents(nextEvents);
      await persistCurrentSession(nextEvents);
      setReviewTarget({
        sessionJsonPath,
        sessionDir,
        eventIndex: nextEvents.length - 1,
        event,
      });
      setFeedback(`Importerade ljudfil: ${imported.displayName ?? filename}`);
    } catch (error: any) {
      const message = String(error?.message ?? error ?? '');
      if (String(error?.code ?? '').includes('IMPORT_CANCELLED') || message.includes('cancel')) {
        setFeedback('Ljudimport avbruten.');
      } else {
        Alert.alert('Importfel', `Kunde inte importera ljudfilen: ${message || 'okänt fel'}`);
        setFeedback('Importen misslyckades.');
      }
    } finally {
      setIsImportingAudio(false);
      await refreshPendingReviews();
    }
  }, [isBusyWithTake, mode, persistCurrentSession, prepareNewSession, refreshPendingReviews]);

  const importAudioVideoFileForReview = useCallback(async () => {
    if (!isAudioVideoPoseMode || isBusyWithTake) return;
    if (!sessionDirRef.current || !sessionJsonPathRef.current) {
      await prepareNewSession();
    }
    const sessionDir = sessionDirRef.current;
    const sessionJsonPath = sessionJsonPathRef.current;
    if (!sessionDir || !sessionJsonPath) {
      Alert.alert('Importfel', 'Kunde inte skapa sessionsmapp för videoimport.');
      return;
    }

    setIsImportingAudio(true);
    setFeedback('Välj en MP4/video från telefonen...');

    const takeIndex = eventsRef.current.filter(event => event.collection_type === 'audio_video_pose').length + 1;
    const prefix = `audio_video_pose_import_${String(takeIndex).padStart(3, '0')}`;
    const videoFilename = `${prefix}.mp4`;
    const wavFilename = `${prefix}.wav`;
    const videoPath = `${sessionDir}/${videoFilename}`;
    const wavPath = `${sessionDir}/${wavFilename}`;

    try {
      const importedVideo = await VideoSegment.importVideoFile(videoPath);
      setFeedback('Video importerad. Extraherar ljudspår...');
      const importedAudio = await AudioCapture.extractAudioFromVideoFile(
        importedVideo.outputPath || videoPath,
        wavPath,
      ) as ImportedAudioFile;
      const nowIso = new Date().toISOString();
      const durationMs = Math.max(0, Math.round(Number(importedAudio.durationMs ?? importedVideo.durationMs ?? 0)));
      const videoDurationMs = Math.max(0, Math.round(Number(importedVideo.durationMs ?? durationMs)));
      const event: AudioEvent = {
        label: 'unlabeled',
        recorded_at: nowIso,
        created_at: nowIso,
        imported_at: nowIso,
        wav_filename: wavFilename,
        duration_ms: durationMs,
        scenario_id: 'free_recording',
        background_condition: 'mixed',
        take_index: takeIndex,
        target_duration_s: 0,
        recording_mode: 'audio_video_pose_import',
        collection_type: 'audio_video_pose',
        scenario: 'playing',
        bounce_context: 'mixed',
        calibration_status: 'skipped',
        has_audio: true,
        has_video: true,
        has_imu: false,
        imported_source_filename: importedVideo.displayName,
        imported_source_uri: importedVideo.sourceUri,
        player_handedness: setup.handedness,
        camera_facing: effectiveCameraFacing ?? undefined,
        detection_config_snapshot: getDefaultAudioDetectionConfigSnapshot(),
        review: {
          required: true,
          anchor_rule: 'attack_start',
          review_stage: 'audio',
          markers: [],
        },
        video_recording: {
          video_filename: videoFilename,
          started_at_ms: Date.now(),
          ended_at_ms: Date.now() + videoDurationMs,
          duration_ms: videoDurationMs || durationMs,
          audio_origin_in_video_ms: 0,
        },
      };

      await Promise.all([
        RNFS.scanFile(videoPath).catch(() => {}),
        RNFS.scanFile(wavPath).catch(() => {}),
      ]);
      const nextEvents = [...eventsRef.current, event];
      eventsRef.current = nextEvents;
      setEvents(nextEvents);
      await persistCurrentSession(nextEvents);
      setReviewTarget({
        sessionJsonPath,
        sessionDir,
        eventIndex: nextEvents.length - 1,
        event,
      });
      setFeedback(`Importerade ${importedVideo.displayName ?? videoFilename}. Börja med ljudreview.`);
    } catch (error: any) {
      await Promise.all([
        RNFS.exists(videoPath).then(exists => exists ? RNFS.unlink(videoPath).catch(() => {}) : undefined),
        RNFS.exists(wavPath).then(exists => exists ? RNFS.unlink(wavPath).catch(() => {}) : undefined),
      ]).catch(() => {});
      const message = String(error?.message ?? error ?? '');
      if (String(error?.code ?? '').includes('IMPORT_CANCELLED') || message.includes('cancel')) {
        setFeedback('Videoimport avbruten.');
      } else if (message.includes('No audio track')) {
        Alert.alert('Importfel', 'Videon saknar ljudspår. Välj en MP4 med inspelat ljud.');
        setFeedback('Videoimporten saknade ljudspår.');
      } else {
        Alert.alert('Importfel', `Kunde inte importera videon: ${message || 'okänt fel'}`);
        setFeedback('Videoimporten misslyckades.');
      }
    } finally {
      setIsImportingAudio(false);
      await refreshPendingReviews();
    }
  }, [
    effectiveCameraFacing,
    isAudioVideoPoseMode,
    isBusyWithTake,
    persistCurrentSession,
    prepareNewSession,
    refreshPendingReviews,
    setup.handedness,
  ]);

  const openPendingReview = useCallback(async (item: PendingReviewItem) => {
    const session = await readSessionFile(item.sessionJsonPath);
    const eventFromDisk = session?.events[item.eventIndex]
      ?? session?.events.find(event => event.wav_filename === item.event.wav_filename);

    if (!session || !eventFromDisk) {
      Alert.alert('Kan inte öppna granskning', 'Tagningen saknar metadata på disk. Starta en ny tagning eller kasta den från kön.');
      return;
    }

    const eventIndex = session.events.findIndex(event => event.wav_filename === eventFromDisk.wav_filename);
    const wavPath = `${item.sessionDir}/${eventFromDisk.wav_filename}`;
    if (!(await RNFS.exists(wavPath))) {
      Alert.alert('Kan inte öppna granskning', `WAV-filen saknas: ${eventFromDisk.wav_filename}`);
      return;
    }

    setReviewQueueVisible(false);
    setReviewTarget({
      ...item,
      eventIndex: eventIndex >= 0 ? eventIndex : item.eventIndex,
      event: eventFromDisk,
    });
  }, []);

  const openNextPendingReview = useCallback(() => {
    if (pendingReviews.length === 0) {
      Alert.alert('Ingen granskningskö', 'Det finns inga väntande tagningar att granska.');
      return;
    }
    if (pendingReviews.length === 1) {
      openPendingReview(pendingReviews[0]).catch(error => {
        Alert.alert('Kan inte öppna granskning', String(error));
      });
      return;
    }
    setReviewQueueVisible(true);
  }, [openPendingReview, pendingReviews]);

  const saveReview = useCallback(async (
    markers: AudioReviewMarker[],
    videoSyncMetadata?: AudioVideoSyncMetadata,
    modelCandidates?: AudioModelCandidate[],
    detectionConfigSnapshot?: AudioDetectionConfigSnapshot,
    videoPoseCandidates?: VideoPoseCandidate[],
    saveOptions?: AudioTakeReviewSaveOptions,
  ) => {
    const target = reviewTarget;
    if (!target) return;

    const session = await readSessionFile(target.sessionJsonPath);
    if (!session || !session.events[target.eventIndex]) {
      throw new Error('Review target missing on disk.');
    }

    const nextEvents = [...session.events];
    const eventToSave = nextEvents[target.eventIndex];
    const nextVideoRecording = eventToSave.video_recording && videoSyncMetadata
      ? {
          ...eventToSave.video_recording,
          ...videoSyncMetadata,
        }
      : eventToSave.video_recording;
    const nowIso = new Date().toISOString();
    const isAudioVideoPoseReview = eventToSave.collection_type === 'audio_video_pose';
    const isPartialAudioReview = isAudioVideoPoseReview && saveOptions?.completion === 'audio';
    const nextReview = {
      required: true,
      anchor_rule: 'attack_start' as const,
      review_stage: isAudioVideoPoseReview
        ? (isPartialAudioReview ? 'motion' as const : 'complete' as const)
        : eventToSave.review?.review_stage,
      audio_completed_at: isAudioVideoPoseReview
        ? (eventToSave.review?.audio_completed_at ?? nowIso)
        : eventToSave.review?.audio_completed_at,
      motion_completed_at: isAudioVideoPoseReview && !isPartialAudioReview
        ? nowIso
        : eventToSave.review?.motion_completed_at,
      completed_at: isPartialAudioReview ? undefined : nowIso,
      markers: [...markers].sort((a, b) => a.timestamp_ms - b.timestamp_ms),
    };
    nextEvents[target.eventIndex] = {
      ...eventToSave,
      video_recording: nextVideoRecording,
      detection_config_snapshot: detectionConfigSnapshot ?? eventToSave.detection_config_snapshot,
      model_candidates: modelCandidates ?? eventToSave.model_candidates,
      video_pose_candidates: videoPoseCandidates ?? eventToSave.video_pose_candidates,
      review: nextReview,
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

    if (isPartialAudioReview) {
      setReviewTarget({
        ...target,
        event: nextEvents[target.eventIndex],
      });
      setFeedback('Ljudreview sparad. Rörelsereview startar från racketträffarna.');
    } else {
      setReviewTarget(null);
      setFeedback(`Granskning sparad: ${target.event.scenario_id} #${target.event.take_index}`);
    }
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
        onBack={() => {
          setReviewTarget(null);
          refreshPendingReviews().catch(() => {});
        }}
      />
    );
  }

  const startDisabled = !canRecord || recordingPhase === 'countdown' || recordingPhase === 'finalizing';
  const statusLabel = recordingPhase === 'countdown'
    ? 'Startar'
    : recordingPhase === 'recording'
      ? 'Spelar in'
      : recordingPhase === 'finalizing'
        ? 'Sparar'
        : recordingPhase === 'done'
          ? 'Klar'
          : 'Redo';
  const mainTimerLabel = recordingPhase === 'countdown'
    ? String(countdownValue)
    : isRecording
      ? (isFreeRecording ? formatDuration(elapsedMs) : formatDuration(remainingMs))
      : (isFreeRecording ? 'Fri' : `${TARGET_DURATION_S}s`);
  const showSyncCue = recordingPhase === 'recording' && elapsedMs < SYNC_CUE_MS;
  const instruction = recordingPhase === 'countdown'
    ? 'Håll racket redo.'
    : recordingPhase === 'recording'
      ? showSyncCue
        ? SYNC_CUE_TEXT
        : isFreeRecording
          ? 'Spela in sekvensen. Stoppa när du är klar och märk händelser i review.'
          : selectedScenario.prompt
      : recordingPhase === 'finalizing'
        ? `Sparar ljud, video${recordsImu ? ' och IMU' : ''}.`
        : recordingPhase === 'done'
          ? 'Klar. Fortsätt med granskning eller nästa tagning.'
          : selectedScenario.prompt;
  const nextTakeIndex = (selectedSummary?.completed_takes ?? 0) + 1;
  const selectedRemaining = selectedSummary?.remaining_takes ?? 0;
  const progressPct = collectionSummary.total_takes > 0
    ? Math.round((collectionSummary.completed_takes / collectionSummary.total_takes) * 100)
    : 0;
  const selectedGroup = scenarioGroups.find(group => group.scenarioIds.includes(selectedScenarioId)) ?? scenarioGroups[0];
  const groupScenarioIds = selectedGroup.scenarioIds;
  const selectedGroupScenarioIndex = Math.max(0, groupScenarioIds.indexOf(selectedScenarioId));
  const latestTake = events[events.length - 1];
  const selectScenarioGroup = (group: ScenarioGroupDefinition) => {
    if (isBusyWithTake) return;
    const firstOpenScenarioId = group.scenarioIds.find(id => {
      const summary = scenarioSummaries.find(item => item.scenario_id === id);
      return (summary?.remaining_takes ?? 0) > 0;
    });
    setSelectedScenarioId(firstOpenScenarioId ?? group.scenarioIds[0]);
    setRecordingPhase('ready');
  };
  const showNextScenarioInGroup = () => {
    if (isBusyWithTake || groupScenarioIds.length <= 1) return;
    const nextId = groupScenarioIds[(selectedGroupScenarioIndex + 1) % groupScenarioIds.length];
    setSelectedScenarioId(nextId);
    setRecordingPhase('ready');
  };
  const handleBackPress = () => {
    if (recordingPhase === 'countdown') {
      clearTimers();
      setCountdownValue(COUNTDOWN_S);
      setRecordingPhase('ready');
      setFeedback(null);
      onDone();
      return;
    }

    if (isRecording) {
      Alert.alert(
        'Inspelning pågår',
        'Stoppa tagningen och gå till granskning innan du lämnar insamlingen.',
        [
          { text: 'Avbryt', style: 'cancel' },
          {
            text: 'Stoppa tagning',
            style: 'destructive',
            onPress: () => {
              stopRecording(false).catch(() => {});
            },
          },
        ],
      );
      return;
    }

    if (isStartingRecording || isImportingAudio || recordingPhase === 'finalizing') {
      Alert.alert('Vänta lite', 'Tagningen startar, importeras eller sparas just nu.');
      return;
    }

    if (recordsImu && device) {
      try { device.cancelConnection(); } catch (_) {}
    }
    onDone();
  };

  return (
    <View style={styles.root}>
      <StatusBar hidden barStyle="light-content" backgroundColor="#0d0d0d" />
      <ScrollView
        style={styles.collectionScroll}
        contentContainerStyle={[
          styles.dashboardContent,
          {
            paddingTop: Math.max(insets.top, 10),
            paddingBottom: Math.max(insets.bottom + 96, 120),
          },
        ]}
        scrollIndicatorInsets={{ bottom: Math.max(insets.bottom + 96, 120) }}
        showsVerticalScrollIndicator={false}
      >

        <View style={styles.dashboardHeader}>
          <TouchableOpacity
            style={styles.collectionBackBtn}
            onPress={handleBackPress}
            accessibilityRole="button"
            accessibilityLabel="Tillbaka"
          >
            <Text style={styles.collectionBackTxt}>Tillbaka</Text>
          </TouchableOpacity>
          <View style={styles.dashboardTitleBlock}>
            <Text style={styles.dashboardTitle}>Datainsamling</Text>
            <Text style={styles.dashboardSubtitle}>
              {isAudioVideoPoseMode
                ? 'Ljud + video ML: en hel tagning, först ljudreview och sedan rörelsereview.'
                : isFreeRecording
                ? 'Playing: spela längre sekvenser och märk händelser i efterhand.'
                : mode === 'audio_imu'
                  ? 'Äldre synkad ljudinsamling: racketstuds eller playing.'
                  : 'Äldre ljudinsamling: racket, bord, golv och brus.'}
            </Text>
          </View>
          <TouchableOpacity
            style={styles.infoBtn}
            onPress={() => Alert.alert(
              'Info',
              isAudioVideoPoseMode
                ? 'Ljudtruth och rörelsetruth sparas som separata rader. Pose körs efter ljudreview.'
                : 'Video är bara stöd för granskning. Ljud och granskade markers är träningsfacit.',
            )}
          >
            <Text style={styles.infoBtnTxt}>Info</Text>
          </TouchableOpacity>
        </View>

      <View style={styles.dashboardCard}>
        <View style={styles.cardHeaderRow}>
          <Text style={styles.cardLabel}>SESSION</Text>
          <Text style={styles.cardActionTxt}>Översikt</Text>
        </View>
        <View style={styles.sessionStatsRow}>
          <View style={styles.progressBlock}>
            <ProgressRing progressPct={progressPct} />
            <Text style={styles.mutedSmall}>Session framsteg</Text>
          </View>
          <View style={styles.statDivider} />
          <View style={styles.statBlock}>
            <Text style={styles.statMain}>
              {isAudioVideoPoseMode ? events.length : `${collectionSummary.completed_takes}/${collectionSummary.total_takes}`}
            </Text>
            <Text style={styles.mutedSmall}>tag</Text>
          </View>
          <View style={styles.statDivider} />
          <View style={styles.statBlock}>
            <Text style={styles.statMain}>{collectionSummary.reviewed_takes}</Text>
            <Text style={styles.mutedSmall}>granskade</Text>
          </View>
          <View style={styles.statDivider} />
          <TouchableOpacity
            style={[
              styles.statBlock,
              styles.queueStatBlock,
              pendingReviews.length > 0 && styles.queueStatBlockActive,
            ]}
            onPress={openNextPendingReview}
            disabled={pendingReviews.length === 0 || isBusyWithTake}
            activeOpacity={0.82}
          >
            <Text style={styles.statMain}>{pendingReviews.length}</Text>
            <Text style={styles.mutedSmall}>i kö</Text>
          </TouchableOpacity>
        </View>
        <Text style={styles.sessionHint}>
          En sessionfil sparar alla tagningar som events med eget scenario_id.
        </Text>
      </View>

      <View style={styles.dashboardCard}>
        <Text style={styles.cardLabel}>AKTIVT SCENARIO</Text>
        {mode !== 'free_recording' && !isAudioVideoPoseMode && (
          <View style={styles.groupTabs}>
            {scenarioGroups.map(group => {
              const active = selectedGroup.id === group.id;
              return (
                <TouchableOpacity
                  key={group.id}
                  style={[styles.groupTab, active && { borderColor: group.color, backgroundColor: '#0f2418' }]}
                  disabled={isBusyWithTake}
                  onPress={() => selectScenarioGroup(group)}
                >
                  <Text style={[styles.groupTabIcon, active && { color: group.color }]}>{group.icon}</Text>
                  <Text style={[styles.groupTabTxt, active && { color: '#fff' }]}>{group.title}</Text>
                </TouchableOpacity>
              );
            })}
          </View>
        )}

        <TouchableOpacity
          style={[styles.activeScenarioCard, { borderColor: selectedScenario.color }]}
          activeOpacity={0.85}
          disabled={isBusyWithTake}
          onPress={showNextScenarioInGroup}
        >
          <View style={[styles.scenarioRoundIcon, { backgroundColor: selectedScenario.bg }]}>
            <Text style={[styles.scenarioRoundTxt, { color: selectedScenario.color }]}>{selectedGroup.icon}</Text>
          </View>
          <View style={styles.scenarioMainCopy}>
            <Text style={styles.activeScenarioTitle}>{selectedScenario.title}</Text>
            <Text style={styles.activeScenarioPrompt}>{selectedScenario.prompt}</Text>
            <View style={styles.tagRow}>
              <Text style={styles.tagPill}>{selectedScenario.label}</Text>
              <Text style={styles.tagPill}>
                {selectedScenario.id === 'racket_music'
                  ? `musik ${musicLevelTitle(selectedMusicLevel).toLowerCase()}`
                  : selectedBackgroundCondition}
              </Text>
              <Text style={styles.tagPill}>{isAudioVideoPoseMode ? 'audio + video + pose' : recordsImu ? 'synkad ljuddata' : 'video + ljud'}</Text>
              <Text style={styles.tagPill}>{selectedScenario.scenario}</Text>
              <Text style={styles.tagPill}>granska efter tagning</Text>
            </View>
            {selectedScenario.id === 'racket_music' && (
              <View style={styles.musicLevelRow}>
                {MUSIC_LEVEL_OPTIONS.map(option => {
                  const active = selectedMusicLevel === option.id;
                  return (
                    <TouchableOpacity
                      key={`music-level-${option.id}`}
                      style={[styles.musicLevelBtn, active && styles.musicLevelBtnActive]}
                      disabled={isBusyWithTake}
                      onPress={() => setSelectedMusicLevel(option.id)}
                    >
                      <Text style={[styles.musicLevelTxt, active && styles.musicLevelTxtActive]}>{option.title}</Text>
                    </TouchableOpacity>
                  );
                })}
              </View>
            )}
            {mode === 'audio_imu' && selectedScenario.scenario === 'racket_bouncing' && (
              <Text style={styles.scenarioModeHint}>
                {selectedScenario.id === 'racket_motion_no_bounce'
                  ? 'Negativ IMU-data: naturlig racketarmsrörelse utan bollkontakt. Märk som Inte studs i Review.'
                  : 'Racketstuds är kontrollerad studs på racket. Forehand/backhand här betyder studs-sida, inte spel-slag.'}
              </Text>
            )}
            {mode === 'audio_imu' && selectedScenario.scenario === 'playing' && (
              <Text style={styles.scenarioModeHint}>
                Playing kräver ingen förvald label. Markera forehand, backhand och bordsstudsar i Review.
              </Text>
            )}
            {isAudioVideoPoseMode && (
              <Text style={styles.scenarioModeHint}>
                Efter stopp sparas en hel WAV + MP4. Review 1 märker ljud, Review 2 kör pose runt bekräftade racketträffar.
              </Text>
            )}
          </View>
          <View style={styles.remainingBlock}>
            <Text style={styles.remainingMain}>
              {isAudioVideoPoseMode ? `${events.length}` : `${selectedSummary?.completed_takes ?? 0}/${selectedScenario.target_takes}`}
            </Text>
            <Text style={styles.mutedSmall}>{isFreeRecording ? 'sekvens' : 'tag kvar'}</Text>
            {groupScenarioIds.length > 1 && <Text style={styles.changeScenarioTxt}>Byt</Text>}
          </View>
        </TouchableOpacity>
        {groupScenarioIds.length > 1 && <View style={styles.dotRow}>
          {groupScenarioIds.map((id, index) => (
            <View
              key={id}
              style={[styles.dot, index === selectedGroupScenarioIndex && styles.dotActive]}
            />
          ))}
        </View>}
      </View>

      <View style={styles.dashboardCard}>
        <Text style={styles.cardLabel}>KAMERA</Text>
        <View style={styles.cameraPanelRow}>
          <View style={styles.cameraButtonColumn}>
            {(['front', 'back'] as CameraFacing[]).map(facing => {
              const available = facing === 'front' ? !!frontCameraDevice : !!backCameraDevice;
              const active = preferredCameraFacing === facing;
              return (
                <TouchableOpacity
                  key={facing}
                  style={[styles.cameraChoiceBtn, active && styles.cameraChoiceBtnActive, !available && styles.disabledBtn]}
                  disabled={!available || isBusyWithTake}
                  onPress={() => {
                    if (available) setPreferredCameraFacing(facing);
                  }}
                >
                  <Text style={[styles.cameraChoiceTxt, active && styles.cameraChoiceTxtActive]}>
                    {facing === 'front' ? 'Fram' : 'Bak'}
                  </Text>
                </TouchableOpacity>
              );
            })}
          </View>
          <View style={styles.dashboardCameraFrame}>
            {hasCameraPermission && cameraDevice ? (
              <Camera
                key={cameraDevice.id}
                style={styles.captureCamera}
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
            ) : (
              <View style={styles.captureFallback}>
                <Text style={styles.warnTxt}>Kamera saknas.</Text>
              </View>
            )}
            {(recordingPhase === 'countdown' || recordingPhase === 'recording' || recordingPhase === 'finalizing') && (
              <View style={styles.cameraCountdownOverlay}>
                <Text style={styles.cameraCountdownTxt}>{mainTimerLabel}</Text>
                {showSyncCue && <Text style={styles.cameraSyncCueTxt}>{SYNC_CUE_TEXT}</Text>}
              </View>
            )}
          </View>
          <View style={styles.cameraStatusColumn}>
            <Text style={[styles.readyTxt, recordingPhase === 'recording' && styles.recordingTxt]}>
              {statusLabel}
            </Text>
            <Text style={styles.cameraMeta}>Aktiv kamera: {effectiveCameraFacing === 'back' ? 'Bak' : 'Fram'}</Text>
            <Text style={styles.cameraMeta}>
              {isFreeRecording
                ? (isRecording ? `${formatDuration(elapsedMs)} inspelat` : 'Manuell stopp')
                : (isRecording ? `${formatDuration(remainingMs)} kvar` : `Auto-stop efter ${TARGET_DURATION_S}s`)}
            </Text>
            {recordsImu && (
              <Text style={styles.cameraMeta}>AirHive {isSensorConnected ? `${sampleHz} Hz` : 'frånkopplad'} | mål 150</Text>
            )}
            {isFreeRecording && !recordsImu && (
              <Text style={styles.cameraMeta}>{isAudioVideoPoseMode ? 'Pose körs i review' : 'IMU saknas: sparar video + ljud'}</Text>
            )}
            {recordsImu && (
              <Text style={styles.cameraMeta}>
                Kalibrering: {calibrationStatus === 'captured' ? 'poser sparade' : 'posekalibrering skippad'}
              </Text>
            )}
          </View>
        </View>
      </View>

      {!!feedback && <Text style={styles.captureFeedback}>{feedback}</Text>}
      {!permissionGranted && <Text style={styles.captureWarning}>Mikrofontillstånd saknas.</Text>}

      {isRecording ? (
        <TouchableOpacity
          style={[styles.dashboardStartBtn, styles.dashboardStopBtn]}
          onPress={() => stopRecording(false)}
          activeOpacity={0.82}
        >
          <Text style={styles.dashboardStopTxt}>STOPPA TAGNING</Text>
          <Text style={[styles.dashboardStartSub, styles.dashboardStopSub]}>
            {isFreeRecording ? `${formatDuration(elapsedMs)} inspelat` : `${formatDuration(remainingMs)} kvar`}
          </Text>
        </TouchableOpacity>
      ) : (
        <TouchableOpacity
          style={[styles.dashboardStartBtn, startDisabled && styles.dashboardStartDisabled]}
          onPress={startCountdownAndRecording}
          disabled={startDisabled}
          activeOpacity={0.82}
        >
          <Text style={[styles.dashboardStartTxt, startDisabled && styles.dashboardStartTxtDisabled]}>
            {recordingPhase === 'countdown'
              ? String(countdownValue)
              : recordingPhase === 'finalizing'
                ? 'SPARAR...'
                : selectedRemaining > 0
                  ? (isAudioVideoPoseMode ? 'STARTA LJUD + VIDEO ML' : isFreeRecording ? 'STARTA PLAYING' : 'STARTA 30 S TAGNING')
                  : 'SCENARIO KLART'}
          </Text>
          <Text style={[styles.dashboardStartSub, startDisabled && styles.dashboardStartSubDisabled]}>{instruction}</Text>
        </TouchableOpacity>
      )}

      {mode === 'audio_only' && (
        <TouchableOpacity
          style={[styles.importAudioBtn, isBusyWithTake && styles.disabledBtn]}
          onPress={() => importAudioFileForReview().catch(error => {
            Alert.alert('Importfel', String(error));
          })}
          disabled={isBusyWithTake}
          activeOpacity={0.82}
        >
          <Text style={styles.importAudioTitle}>
            {isImportingAudio ? 'IMPORTERAR LJUD...' : 'IMPORTERA LJUD'}
          </Text>
          <Text style={styles.importAudioSub}>
            Välj en iPhone-inspelning. Appen konverterar till WAV och öppnar audio-review utan video.
          </Text>
        </TouchableOpacity>
      )}

      {isAudioVideoPoseMode && (
        <TouchableOpacity
          style={[styles.importAudioBtn, isBusyWithTake && styles.disabledBtn]}
          onPress={() => importAudioVideoFileForReview().catch(error => {
            Alert.alert('Importfel', String(error));
          })}
          disabled={isBusyWithTake}
          activeOpacity={0.82}
        >
          <Text style={styles.importAudioTitle}>
            {isImportingAudio ? 'IMPORTERAR VIDEO...' : 'IMPORTERA VIDEO'}
          </Text>
          <Text style={styles.importAudioSub}>
            Välj en MP4 från iPhone/Drive. Appen extraherar ljud och öppnar tvåstegsreview.
          </Text>
        </TouchableOpacity>
      )}

      <View style={styles.actionCardsRow}>
        <TouchableOpacity
          style={[styles.actionCard, pendingReviews.length === 0 && styles.disabledBtn]}
          onPress={openNextPendingReview}
          disabled={pendingReviews.length === 0 || isBusyWithTake}
        >
          <Text style={styles.actionCardTxt}>Granskningskö {pendingReviews.length}</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.actionCard, isBusyWithTake && styles.disabledBtn]}
          onPress={() => resetSession().catch(() => {})}
          disabled={isBusyWithTake}
        >
          <Text style={styles.actionCardTxt}>Ny session</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.actionCard, (events.length === 0 || isBusyWithTake) && styles.disabledBtn]}
          onPress={() => undoLastTake().catch(() => {})}
          disabled={events.length === 0 || isBusyWithTake}
        >
          <Text style={styles.actionCardTxt}>Ångra senaste</Text>
        </TouchableOpacity>
      </View>

      {reviewQueueVisible && pendingReviews.length > 0 && (
        <View style={styles.reviewQueuePanel}>
          <View style={styles.cardHeaderRow}>
            <Text style={styles.cardLabel}>GRANSKNINGSKÖ</Text>
            <Text style={styles.queuePanelMeta}>{pendingReviews.length} väntar</Text>
          </View>
          {pendingReviews.map(item => (
            <TouchableOpacity
              key={`${item.sessionJsonPath}-${item.eventIndex}`}
              style={styles.reviewQueueRow}
              onPress={() => openPendingReview(item).catch(error => {
                Alert.alert('Kan inte öppna granskning', String(error));
              })}
              disabled={isBusyWithTake}
              activeOpacity={0.82}
            >
              <View style={styles.reviewQueueMain}>
                <Text style={styles.reviewQueueTitle}>{item.event.scenario_id}</Text>
                <Text style={styles.reviewQueueSub}>
                  Tagning {item.event.take_index} · {Math.round(item.event.duration_ms / 1000)} s
                </Text>
              </View>
              <Text style={styles.reviewQueueAction}>Granska</Text>
            </TouchableOpacity>
          ))}
        </View>
      )}

      <View style={styles.dashboardCard}>
        <View style={styles.cardHeaderRow}>
          <Text style={styles.cardLabel}>SENASTE TAG</Text>
          <TouchableOpacity
            onPress={openNextPendingReview}
            disabled={pendingReviews.length === 0 || isBusyWithTake}
            activeOpacity={0.82}
          >
            <Text style={[styles.linkTxt, pendingReviews.length === 0 && styles.linkTxtDisabled]}>
              Visa alla
            </Text>
          </TouchableOpacity>
        </View>
        <Text style={styles.latestTxt}>
          {latestTake
            ? `${latestTake.scenario_id} | tagning ${latestTake.take_index} | ${latestTake.review?.completed_at ? 'granskad' : 'väntar på granskning'}`
            : 'Inga tag ännu i denna session.'}
        </Text>
      </View>

      <TouchableOpacity
        style={styles.workflowCard}
        onPress={() => {
          if (recordsImu && device) {
            try { device.cancelConnection(); } catch (_) {}
          }
          onDone();
        }}
      >
        <Text style={styles.workflowTitle}>Workflow</Text>
        <Text style={styles.workflowSub}>Setup och översikt</Text>
      </TouchableOpacity>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#0d0d0d' },
  collectionScroll: { flex: 1 },
  content: { padding: 20, paddingBottom: 48 },
  dashboardContent: {
    flexGrow: 1,
    paddingHorizontal: 14,
    gap: 12,
  },
  dashboardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingTop: 6,
    paddingBottom: 6,
  },
  collectionBackBtn: {
    minWidth: 76,
    height: 38,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#101417',
    borderWidth: 1,
    borderColor: '#2d3238',
  },
  collectionBackTxt: { color: '#dce2eb', fontSize: 13, fontWeight: '800' },
  dashboardTitleBlock: { flex: 1, minWidth: 0 },
  dashboardTitle: { color: '#fff', fontSize: 25, fontWeight: '900' },
  dashboardSubtitle: { color: '#a8adb7', fontSize: 13, marginTop: 1 },
  infoBtn: {
    minHeight: 34,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#2d3238',
    paddingHorizontal: 12,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#101417',
  },
  infoBtnTxt: { color: '#b8bec8', fontSize: 12, fontWeight: '800' },
  dashboardCard: {
    borderRadius: 14,
    borderWidth: 1,
    borderColor: '#2a2f34',
    backgroundColor: '#101417',
    padding: 14,
  },
  cardHeaderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  cardLabel: { color: '#a8adb7', fontSize: 11, fontWeight: '900' },
  cardActionTxt: { color: '#7ee39e', fontSize: 12, fontWeight: '800' },
  sessionStatsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    marginTop: 14,
  },
  progressBlock: { width: 92, alignItems: 'center', gap: 7 },
  progressRing: {
    width: 62,
    height: 62,
    borderRadius: 31,
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
    backgroundColor: '#101417',
  },
  progressRingInner: {
    width: 46,
    height: 46,
    borderRadius: 23,
    backgroundColor: '#0d0d0d',
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: '#252b31',
  },
  progressRingSegment: {
    position: 'absolute',
    width: 3,
    height: 8,
    borderRadius: 2,
    backgroundColor: '#343c44',
  },
  progressRingSegmentActive: {
    backgroundColor: '#2ee678',
  },
  progressRingTxt: { color: '#fff', fontSize: 15, fontWeight: '900' },
  mutedSmall: { color: '#9aa0aa', fontSize: 11, lineHeight: 16 },
  sessionHint: { color: '#7f8792', fontSize: 10, lineHeight: 14, marginTop: 10 },
  statDivider: { width: 1, height: 64, backgroundColor: '#30363d' },
  statBlock: { flex: 1, alignItems: 'center' },
  queueStatBlock: {
    minHeight: 54,
    borderRadius: 12,
    justifyContent: 'center',
  },
  queueStatBlockActive: {
    backgroundColor: '#111d17',
    borderWidth: 1,
    borderColor: '#244d34',
  },
  statMain: { color: '#fff', fontSize: 22, fontWeight: '900' },
  groupTabs: { flexDirection: 'row', gap: 7, marginTop: 12, marginBottom: 12 },
  groupTab: {
    flex: 1,
    minHeight: 40,
    borderRadius: 11,
    borderWidth: 1,
    borderColor: '#282e34',
    backgroundColor: '#11161a',
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    gap: 7,
    paddingHorizontal: 8,
  },
  groupTabIcon: { color: '#9da4ae', fontSize: 12, fontWeight: '900' },
  groupTabTxt: { color: '#b4bac4', fontSize: 12, fontWeight: '800' },
  activeScenarioCard: {
    minHeight: 104,
    borderRadius: 14,
    borderWidth: 2,
    backgroundColor: '#0b1713',
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    gap: 12,
  },
  scenarioRoundIcon: {
    width: 52,
    height: 52,
    borderRadius: 26,
    alignItems: 'center',
    justifyContent: 'center',
  },
  scenarioRoundTxt: { fontSize: 19, fontWeight: '900' },
  scenarioMainCopy: { flex: 1, minWidth: 0 },
  activeScenarioTitle: { color: '#fff', fontSize: 18, fontWeight: '900' },
  activeScenarioPrompt: { color: '#c0c6ce', fontSize: 12, marginTop: 3, lineHeight: 17 },
  scenarioModeHint: { color: '#9aa0aa', fontSize: 11, lineHeight: 15, marginTop: 6 },
  tagRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 5, marginTop: 7 },
  tagPill: {
    color: '#7ee39e',
    backgroundColor: '#182126',
    borderRadius: 6,
    paddingHorizontal: 7,
    paddingVertical: 3,
    fontSize: 10,
    fontWeight: '700',
  },
  musicLevelRow: { flexDirection: 'row', gap: 6, marginTop: 8 },
  musicLevelBtn: {
    minHeight: 30,
    minWidth: 56,
    borderRadius: 9,
    borderWidth: 1,
    borderColor: '#27313a',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#10161b',
    paddingHorizontal: 10,
  },
  musicLevelBtnActive: { borderColor: '#2ecc71', backgroundColor: '#12351f' },
  musicLevelTxt: { color: '#aeb4be', fontSize: 11, fontWeight: '900' },
  musicLevelTxtActive: { color: '#2ee678' },
  remainingBlock: { minWidth: 68, alignItems: 'center' },
  remainingMain: { color: '#2ee678', fontSize: 23, fontWeight: '900' },
  changeScenarioTxt: { color: '#7ee39e', fontSize: 12, fontWeight: '900', marginTop: 4 },
  dotRow: { flexDirection: 'row', justifyContent: 'center', gap: 10, marginTop: 10 },
  dot: { width: 9, height: 9, borderRadius: 5, backgroundColor: '#40474f' },
  dotActive: { backgroundColor: '#2ee678' },
  cameraPanelRow: { flexDirection: 'row', alignItems: 'center', gap: 10, marginTop: 10 },
  cameraButtonColumn: { width: 84, gap: 8 },
  cameraChoiceBtn: {
    minHeight: 42,
    borderRadius: 11,
    borderWidth: 1,
    borderColor: '#30363d',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#11161a',
  },
  cameraChoiceBtnActive: { borderColor: '#2ee678', backgroundColor: '#0d2418' },
  cameraChoiceTxt: { color: '#aeb5bf', fontSize: 13, fontWeight: '800' },
  cameraChoiceTxtActive: { color: '#2ee678' },
  dashboardCameraFrame: {
    flex: 1,
    aspectRatio: 1.24,
    borderRadius: 8,
    overflow: 'hidden',
    backgroundColor: '#050505',
  },
  cameraCountdownOverlay: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(0,0,0,0.38)',
    paddingHorizontal: 14,
  },
  cameraCountdownTxt: { color: '#fff', fontSize: 42, fontWeight: '900' },
  cameraSyncCueTxt: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '900',
    lineHeight: 17,
    marginTop: 8,
    textAlign: 'center',
    backgroundColor: 'rgba(0,0,0,0.6)',
    borderRadius: 10,
    paddingHorizontal: 10,
    paddingVertical: 6,
  },
  cameraStatusColumn: { width: 118, gap: 7 },
  readyTxt: { color: '#2ee678', fontSize: 15, fontWeight: '900' },
  recordingTxt: { color: '#ff7f7f' },
  cameraMeta: { color: '#a9b0ba', fontSize: 11, lineHeight: 16 },
  dashboardStartBtn: {
    minHeight: 74,
    borderRadius: 14,
    borderWidth: 2,
    borderColor: '#1fbf63',
    backgroundColor: '#2edb78',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 14,
  },
  dashboardStopBtn: { backgroundColor: '#2d0d0d', borderColor: '#ff7f7f' },
  dashboardStartDisabled: { backgroundColor: '#161b1f', borderColor: '#2a3036' },
  dashboardStartTxt: { color: '#03120a', fontSize: 18, fontWeight: '900' },
  dashboardStartTxtDisabled: { color: '#5e6670' },
  dashboardStopTxt: { color: '#ff7f7f', fontSize: 18, fontWeight: '900' },
  dashboardStartSub: { color: '#173322', fontSize: 12, marginTop: 6, textAlign: 'center', fontWeight: '700' },
  dashboardStartSubDisabled: { color: '#777f88' },
  dashboardStopSub: { color: '#f1b5b5' },
  importAudioBtn: {
    minHeight: 58,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#4a9eff',
    backgroundColor: '#0f1d2b',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 14,
    paddingVertical: 10,
  },
  importAudioTitle: { color: '#dcecff', fontSize: 14, fontWeight: '900' },
  importAudioSub: { color: '#aebed0', fontSize: 11, marginTop: 4, textAlign: 'center', fontWeight: '700' },
  actionCardsRow: { flexDirection: 'row', gap: 9 },
  actionCard: {
    flex: 1,
    minHeight: 52,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#2b3137',
    backgroundColor: '#101417',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 8,
  },
  actionCardTxt: { color: '#c7cdd6', fontSize: 11, fontWeight: '800', textAlign: 'center' },
  linkTxt: { color: '#2ee678', fontSize: 12, fontWeight: '900' },
  linkTxtDisabled: { color: '#5f686f' },
  reviewQueuePanel: {
    borderRadius: 14,
    borderWidth: 1,
    borderColor: '#2a2f34',
    backgroundColor: '#101417',
    padding: 12,
    gap: 8,
  },
  queuePanelMeta: { color: '#9aa0aa', fontSize: 11, fontWeight: '800' },
  reviewQueueRow: {
    minHeight: 50,
    borderRadius: 11,
    borderWidth: 1,
    borderColor: '#283039',
    backgroundColor: '#0d1114',
    paddingHorizontal: 12,
    paddingVertical: 9,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 10,
  },
  reviewQueueMain: { flex: 1, minWidth: 0 },
  reviewQueueTitle: { color: '#fff', fontSize: 13, fontWeight: '900' },
  reviewQueueSub: { color: '#9aa0aa', fontSize: 11, marginTop: 3 },
  reviewQueueAction: { color: '#2ee678', fontSize: 12, fontWeight: '900' },
  latestTxt: { color: '#aeb5bf', fontSize: 12, marginTop: 10, lineHeight: 18 },
  workflowCard: {
    minHeight: 58,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#2b3137',
    backgroundColor: '#101820',
    paddingHorizontal: 16,
    justifyContent: 'center',
  },
  workflowTitle: { color: '#cbd2dc', fontSize: 13, fontWeight: '900' },
  workflowSub: { color: '#87909a', fontSize: 11, marginTop: 3 },
  captureHeader: {
    minHeight: 46,
    paddingHorizontal: 14,
    paddingBottom: 8,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
  },
  captureHeaderMain: { flex: 1 },
  statusPill: {
    borderRadius: 999,
    backgroundColor: '#191919',
    borderWidth: 1,
    borderColor: '#2d2d2d',
    paddingHorizontal: 12,
    paddingVertical: 7,
  },
  statusPillRec: {
    borderColor: '#ff7f7f',
    backgroundColor: '#2d0d0d',
  },
  statusPillTxt: { color: '#f5f5f5', fontSize: 12, fontWeight: '800' },
  scenarioStrip: {
    maxHeight: 56,
    flexGrow: 0,
  },
  scenarioStripContent: {
    paddingHorizontal: 12,
    paddingBottom: 8,
    gap: 8,
  },
  scenarioPill: {
    minWidth: 132,
    borderRadius: 8,
    borderWidth: 1,
    paddingHorizontal: 11,
    paddingVertical: 8,
    justifyContent: 'center',
  },
  scenarioPillDone: { opacity: 0.55 },
  scenarioPillTitle: { color: '#d7d7d7', fontSize: 12, fontWeight: '800' },
  scenarioPillMeta: { color: '#8a8a8a', fontSize: 11, marginTop: 2 },
  captureStage: {
    flex: 1,
    marginHorizontal: 10,
    borderRadius: 10,
    overflow: 'hidden',
    backgroundColor: '#050505',
    minHeight: 320,
  },
  captureCamera: { ...StyleSheet.absoluteFillObject },
  captureFallback: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  captureShade: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0,0,0,0.12)',
  },
  captureTopOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    padding: 14,
    backgroundColor: 'rgba(0,0,0,0.48)',
  },
  captureInstruction: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '800',
    lineHeight: 20,
  },
  captureMeta: { color: '#c9c9c9', fontSize: 12, marginTop: 4 },
  captureCenterOverlay: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: '38%',
    alignItems: 'center',
  },
  captureTimer: {
    color: '#fff',
    fontSize: 72,
    fontWeight: '900',
    textShadowColor: 'rgba(0,0,0,0.7)',
    textShadowRadius: 12,
  },
  captureTimerSub: {
    color: '#f0f0f0',
    fontSize: 13,
    fontWeight: '800',
    marginTop: 2,
    backgroundColor: 'rgba(0,0,0,0.42)',
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 5,
  },
  captureBottomOverlay: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    padding: 12,
    backgroundColor: 'rgba(0,0,0,0.48)',
  },
  cameraFacingRowCompact: { flexDirection: 'row', gap: 8, marginBottom: 8 },
  cameraFacingBtnCompact: {
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#2a2a2a',
    backgroundColor: '#161616',
    paddingVertical: 8,
    paddingHorizontal: 12,
  },
  captureSignal: { color: '#d0d0d0', fontSize: 11, fontWeight: '700' },
  captureFeedback: {
    color: '#4a9eff',
    fontSize: 12,
    textAlign: 'center',
    paddingHorizontal: 14,
    paddingTop: 7,
  },
  captureWarning: {
    color: '#e67e22',
    fontSize: 12,
    textAlign: 'center',
    paddingTop: 6,
  },
  controlDock: {
    paddingHorizontal: 12,
    paddingTop: 8,
    gap: 8,
  },
  primaryRecordBtn: {
    borderRadius: 10,
    minHeight: 64,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#0d2d0d',
    paddingHorizontal: 16,
    paddingVertical: 10,
  },
  primaryStopBtn: { backgroundColor: '#2d0d0d' },
  primaryRecordTxt: { color: '#2ecc71', fontSize: 23, fontWeight: '900' },
  primaryStopTxt: { color: '#ff7f7f', fontSize: 23, fontWeight: '900' },
  primaryRecordSub: { color: '#8f8f8f', fontSize: 12, marginTop: 3, textAlign: 'center' },
  compactActionRow: { flexDirection: 'row', gap: 8 },
  compactBtn: {
    flex: 1,
    minHeight: 42,
    borderRadius: 8,
    backgroundColor: '#151515',
    borderWidth: 1,
    borderColor: '#262626',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 6,
  },
  compactBtnTxt: { color: '#f1f1f1', fontSize: 11, fontWeight: '800', textAlign: 'center' },
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
