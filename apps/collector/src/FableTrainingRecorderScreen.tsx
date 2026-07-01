import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Alert,
  Modal,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import RNFS from 'react-native-fs';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { AudioStream } from './NativeAudioStream';
import type { PlayerSetup } from './types';

const APP_VERSION = '1.7';
const SAMPLE_RATE_HZ = 22050;
const RECORDING_THRESHOLD = 0.005;
const RECORDING_DIR = `${RNFS.ExternalStorageDirectoryPath}/Download/pingis_sessions/fable_training_audio`;

type ScenarioPolarity = 'positive' | 'negative' | 'mixed' | 'unclear';

interface RecorderScenario {
  id: string;
  title: string;
  short: string;
  howTo: string;
  countHint: string;
  polarity: ScenarioPolarity;
  boundaryBucket: string;
  collectionGoal: string;
  trainingRoleHint?: string;
}

const SCENARIOS: RecorderScenario[] = [
  {
    id: 'normal_racket_bounce',
    title: 'Normal racket bounce',
    short: 'Normal up/down practice in a quiet-ish room.',
    howTo: 'Put the phone in the normal training position. Bounce the ball on the racket at comfortable height and tempo. Avoid talking during this one.',
    countHint: 'Count every real ball-to-racket contact.',
    polarity: 'positive',
    boundaryBucket: 'ordinary_racket_positive',
    collectionGoal: 'general_positive_coverage',
  },
  {
    id: 'slow_high_racket_bounce',
    title: 'Slow/high racket bounce',
    short: 'Higher bounces with longer gaps.',
    howTo: 'Bounce higher than usual so there is more time between contacts. Let the ball travel naturally; do not rush to keep the app happy.',
    countHint: 'Count every real ball-to-racket contact, even if the gaps are long.',
    polarity: 'positive',
    boundaryBucket: 'slow_high_racket_positive',
    collectionGoal: 'general_positive_coverage',
  },
  {
    id: 'fast_racket_bounce',
    title: 'Fast racket bounce',
    short: 'Dense, quick racket contacts.',
    howTo: 'Bounce at a fast tempo with short gaps between contacts. Keep it realistic rather than perfectly robotic.',
    countHint: 'Count every real contact. If it becomes too fast to know, mark the count unclear.',
    polarity: 'positive',
    boundaryBucket: 'fast_racket_positive',
    collectionGoal: 'general_positive_coverage',
  },
  {
    id: 'messy_kid_style_racket_bounce',
    title: 'Messy/kid-style racket bounce',
    short: 'Uneven practice with misses, catches, and restarts.',
    howTo: 'Practice like a child at home: uneven height, catches, restarts, occasional misses. Only actual ball-to-racket contacts should count.',
    countHint: 'Count only real racket contacts. Do not count catches, misses, or ball drops.',
    polarity: 'positive',
    boundaryBucket: 'messy_racket_positive',
    collectionGoal: 'general_positive_coverage',
  },
  {
    id: 'racket_bounce_speaking_counting',
    title: 'Racket bounce + speaking/counting',
    short: 'Bounce while someone talks or you count out loud.',
    howTo: 'Bounce normally while you or someone nearby speaks, laughs, or counts out loud. This is one of the most important hard cases.',
    countHint: 'Count real racket contacts, not spoken numbers.',
    polarity: 'positive',
    boundaryBucket: 'speech_plus_racket_positive',
    collectionGoal: 'general_positive_coverage',
  },
  {
    id: 'racket_bounce_background_sound',
    title: 'Racket bounce + background sound',
    short: 'Bounce with TV, kid sounds, music, or room noise.',
    howTo: 'Play realistic background sound while bouncing. Keep it normal-home noisy, not extremely loud chaos.',
    countHint: 'Count real racket contacts. Background sounds are not contacts.',
    polarity: 'positive',
    boundaryBucket: 'background_plus_racket_positive',
    collectionGoal: 'general_positive_coverage',
  },
  {
    id: 'far_soft_racket_bounce_background',
    title: 'Far/soft racket bounce + background',
    short: 'Softer contacts or racket farther from the phone, with noise.',
    howTo: 'Put the phone in the normal place, then bounce a little farther away or softer than usual while realistic background sound is playing.',
    countHint: 'Count every real racket contact. This is a high-priority recovery case.',
    polarity: 'positive',
    boundaryBucket: 'far_soft_background_racket_positive',
    collectionGoal: 't0102_boundary_positive_recovery',
  },
  {
    id: 'soft_high_racket_bounce_background',
    title: 'Soft/high racket bounce + background',
    short: 'Higher slower bounces where the contact is not very loud.',
    howTo: 'Bounce higher and slower while background sound is present. Let some contacts be soft, like normal home practice.',
    countHint: 'Count every real racket contact, even when the peak is small in the waveform.',
    polarity: 'positive',
    boundaryBucket: 'soft_high_background_racket_positive',
    collectionGoal: 't0102_boundary_positive_recovery',
  },
  {
    id: 'talking_only_no_bounce',
    title: 'Talking only, no bounce',
    short: 'Speech/laughter/counting with no ball or racket contact.',
    howTo: 'Talk, count, laugh, or read out loud near the phone. Do not hit the ball, racket, table, or floor.',
    countHint: 'Expected racket contacts should be 0.',
    polarity: 'negative',
    boundaryBucket: 'speech_only_negative',
    collectionGoal: 'general_hard_negative_coverage',
  },
  {
    id: 'racket_handling_no_bounce',
    title: 'Racket handling, no bounce',
    short: 'Move, grip, rotate, or tap the racket without ball contact.',
    howTo: 'Hold and move the racket like a child preparing to practice. Grip it, rotate it, brush it, or lightly handle it, but do not let the ball hit the racket.',
    countHint: 'Expected racket contacts should be 0.',
    polarity: 'negative',
    boundaryBucket: 'racket_handling_negative',
    collectionGoal: 'general_hard_negative_coverage',
  },
  {
    id: 'floor_table_other_impact_no_racket',
    title: 'Floor/table/other impact, no racket',
    short: 'Ball or objects hit something that is not the racket.',
    howTo: 'Create realistic non-racket impacts: ball on floor, ball on table-like surface, desk taps, chair/room sounds. Do not hit the racket.',
    countHint: 'Expected racket contacts should be 0.',
    polarity: 'negative',
    boundaryBucket: 'floor_table_other_impact_negative',
    collectionGoal: 'general_hard_negative_coverage',
  },
  {
    id: 'background_sound_only_no_bounce',
    title: 'Background sound only, no bounce',
    short: 'TV, kid sounds, music, or room noise with no impacts.',
    howTo: 'Play the same background sounds used during noisy practice, but do not bounce the ball or touch the racket.',
    countHint: 'Expected racket contacts should be 0.',
    polarity: 'negative',
    boundaryBucket: 'background_only_negative',
    collectionGoal: 't0102_boundary_hard_negative',
  },
  {
    id: 'talking_counting_background_no_bounce',
    title: 'Talking/counting + background, no bounce',
    short: 'Speech plus background sound with no ball or racket contact.',
    howTo: 'Talk, count, or laugh while background sound is playing. Do not hit the racket, ball, table, or floor.',
    countHint: 'Expected racket contacts should be 0.',
    polarity: 'negative',
    boundaryBucket: 'speech_background_negative',
    collectionGoal: 't0102_boundary_hard_negative',
  },
  {
    id: 'racket_handling_background_no_bounce',
    title: 'Racket handling + background, no bounce',
    short: 'Move or grip the racket near noise, but no ball contact.',
    howTo: 'Move, rotate, grip, brush, or adjust the racket while background sound is playing. Do not let the ball hit the racket.',
    countHint: 'Expected racket contacts should be 0.',
    polarity: 'negative',
    boundaryBucket: 'racket_handling_background_negative',
    collectionGoal: 't0102_boundary_hard_negative',
  },
  {
    id: 'catch_after_sound_no_racket',
    title: 'Catch/after-sound, no racket',
    short: 'Ball catch, scrape, drop, or after-sound without racket contact.',
    howTo: 'Create realistic practice cleanup sounds: catch the ball, let it drop, scrape lightly, or make after-sounds without a racket bounce.',
    countHint: 'Expected racket contacts should be 0. If a real racket contact sneaks in, discard or mark unclear.',
    polarity: 'negative',
    boundaryBucket: 'catch_after_sound_negative',
    collectionGoal: 't0102_boundary_hard_negative',
  },
  {
    id: 'ambiguous_ball_like_impact_near_phone_no_racket',
    title: 'Ball-like impact near phone, no racket',
    short: 'Sharp ball-like impacts close to the phone, but not racket hits.',
    howTo: 'Make controlled ball-like sounds near the phone using floor, wall, book, box, desk, or hand catch. Do not hit the racket.',
    countHint: 'Expected racket contacts should be 0. This is a high-priority safety case.',
    polarity: 'negative',
    boundaryBucket: 'ambiguous_ball_like_near_phone_negative',
    collectionGoal: 't0102_boundary_hard_negative',
  },
  {
    id: 'other_unclear',
    title: 'Other/unclear',
    short: 'Use when the recording does not fit the other buckets.',
    howTo: 'Use this for a useful but unusual situation, or if you are not sure what happened. It can still help diagnostics after listening.',
    countHint: 'Use unclear unless you are confident in the exact racket-contact count.',
    polarity: 'unclear',
    boundaryBucket: 'other_unclear',
    collectionGoal: 'diagnostic_unclear',
  },
];

interface Props {
  setup: PlayerSetup;
  onDone: () => void;
}

interface RecordingPaths {
  sessionId: string;
  wavFilename: string;
  jsonFilename: string;
  wavPath: string;
  jsonPath: string;
}

interface PendingRecording {
  paths: RecordingPaths;
  startedAtIso: string;
  stoppedAtIso: string;
  startedAtMs: number;
  stoppedAtMs: number;
  durationMs: number;
}

function buildRecordingPaths(startedAtIso: string): RecordingPaths {
  const sessionId = `fable_training_audio_${startedAtIso.replace(/[:.]/g, '-')}`;
  const wavFilename = `${sessionId}.wav`;
  const jsonFilename = `${sessionId}.json`;
  return {
    sessionId,
    wavFilename,
    jsonFilename,
    wavPath: `${RECORDING_DIR}/${wavFilename}`,
    jsonPath: `${RECORDING_DIR}/${jsonFilename}`,
  };
}

function formatDuration(ms: number) {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${String(seconds).padStart(2, '0')}`;
}

function scenarioById(id: string) {
  return SCENARIOS.find(scenario => scenario.id === id) ?? SCENARIOS[0];
}

export function FableTrainingRecorderScreen({ setup, onDone }: Props) {
  const insets = useSafeAreaInsets();
  const startedAtIsoRef = useRef<string | null>(null);
  const startedAtMsRef = useRef<number | null>(null);
  const pathsRef = useRef<RecordingPaths | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [pendingRecording, setPendingRecording] = useState<PendingRecording | null>(null);
  const [selectedScenarioId, setSelectedScenarioId] = useState(SCENARIOS[0].id);
  const [expectedCount, setExpectedCount] = useState('');
  const [countUnclear, setCountUnclear] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [status, setStatus] = useState('Ready to record targeted audio.');
  const [savedPath, setSavedPath] = useState<string | null>(null);

  const selectedScenario = useMemo(
    () => scenarioById(selectedScenarioId),
    [selectedScenarioId],
  );

  useEffect(() => {
    if (!isRecording) return;
    const interval = setInterval(() => {
      const startedAtMs = startedAtMsRef.current;
      if (startedAtMs !== null) setElapsedMs(Date.now() - startedAtMs);
    }, 250);
    return () => clearInterval(interval);
  }, [isRecording]);

  useEffect(() => () => {
    if (isRecording) {
      AudioStream.stopStreaming().catch(() => {});
      AudioStream.setDebugRecordingPath(null).catch(() => {});
    }
  }, [isRecording]);

  const startRecording = useCallback(() => {
    if (isRecording || pendingRecording) return;
    const startedAtIso = new Date().toISOString();
    const paths = buildRecordingPaths(startedAtIso);
    startedAtIsoRef.current = startedAtIso;
    startedAtMsRef.current = Date.now();
    pathsRef.current = paths;
    setElapsedMs(0);
    setSavedPath(null);
    setStatus('Recording...');

    void (async () => {
      try {
        await RNFS.mkdir(RECORDING_DIR);
        await AudioStream.setDebugRecordingPath(paths.wavPath);
        await AudioStream.startStreaming(RECORDING_THRESHOLD);
        setIsRecording(true);
      } catch (err) {
        setStatus(`Could not start recording: ${String(err).slice(0, 120)}`);
        startedAtIsoRef.current = null;
        startedAtMsRef.current = null;
        pathsRef.current = null;
        try { await AudioStream.setDebugRecordingPath(null); } catch (_) {}
      }
    })();
  }, [isRecording, pendingRecording]);

  const stopRecording = useCallback(() => {
    if (!isRecording) return;
    void (async () => {
      const stoppedAtIso = new Date().toISOString();
      const stoppedAtMs = Date.now();
      const startedAtMs = startedAtMsRef.current ?? stoppedAtMs;
      const durationMs = Math.max(0, stoppedAtMs - startedAtMs);
      const paths = pathsRef.current;
      const startedAtIso = startedAtIsoRef.current ?? stoppedAtIso;

      try {
        await AudioStream.stopStreaming();
      } catch (_) {
        // Keep the WAV path visible; the native side may already have closed it.
      }
      try { await AudioStream.setDebugRecordingPath(null); } catch (_) {}

      setIsRecording(false);
      setElapsedMs(durationMs);
      if (!paths) {
        setStatus('Recording stopped, but file path was missing.');
        return;
      }
      setSelectedScenarioId(SCENARIOS[0].id);
      setExpectedCount('');
      setCountUnclear(false);
      setShowHelp(false);
      setPendingRecording({
        paths,
        startedAtIso,
        stoppedAtIso,
        startedAtMs,
        stoppedAtMs,
        durationMs,
      });
      setStatus('Recording stopped. Tag it before saving.');
    })();
  }, [isRecording]);

  const discardPending = useCallback(() => {
    if (!pendingRecording) return;
    void (async () => {
      try {
        const exists = await RNFS.exists(pendingRecording.paths.wavPath);
        if (exists) await RNFS.unlink(pendingRecording.paths.wavPath);
      } catch (_) {
        // Discard is best-effort; the next pull can ignore orphaned files.
      }
      setPendingRecording(null);
      setStatus('Recording discarded.');
    })();
  }, [pendingRecording]);

  const savePending = useCallback(() => {
    if (!pendingRecording) return;
    const parsedCount = Number.parseInt(expectedCount.trim(), 10);
    if (!countUnclear && (!Number.isFinite(parsedCount) || parsedCount < 0)) {
      Alert.alert('Expected count', 'Enter a number, or choose Unclear.');
      return;
    }

    const scenario = selectedScenario;
    if (!countUnclear && scenario.polarity === 'positive' && parsedCount === 0) {
      Alert.alert('Expected count', 'For a racket-bounce recording, enter the real count above 0, choose Unclear, or discard the recording.');
      return;
    }
    if (!countUnclear && scenario.polarity === 'negative' && parsedCount !== 0) {
      Alert.alert('Expected count', 'For a no-bounce recording, use 0, choose Unclear, or pick a racket-bounce sound type.');
      return;
    }

    const expectedRacketContacts = countUnclear ? null : parsedCount;
    const trainingRoleHint = scenario.trainingRoleHint ?? (
      scenario.polarity === 'negative'
        ? 'negative_interval_candidate'
        : scenario.polarity === 'positive'
          ? 'positive_needs_exact_timestamp_review'
          : 'diagnostic_needs_review'
    );
    const metadata = {
      type: 'fable_training_audio_recording',
      schema_version: 1,
      collection_type: 'fable_training_audio_v1',
      app_version: APP_VERSION,
      player: setup,
      session_id: pendingRecording.paths.sessionId,
      created_at: new Date().toISOString(),
      started_at: pendingRecording.startedAtIso,
      stopped_at: pendingRecording.stoppedAtIso,
      duration_ms: pendingRecording.durationMs,
      scenario: {
        id: scenario.id,
        title: scenario.title,
        polarity: scenario.polarity,
        short: scenario.short,
        how_to: scenario.howTo,
        count_hint: scenario.countHint,
        boundary_bucket: scenario.boundaryBucket,
        collection_goal: scenario.collectionGoal,
        training_role_hint: trainingRoleHint,
      },
      expected_racket_contacts: expectedRacketContacts,
      expected_count_unclear: countUnclear,
      training_role_hint: trainingRoleHint,
      collection_tags: [
        scenario.polarity,
        scenario.boundaryBucket,
        scenario.collectionGoal,
      ],
      audio: {
        wav_filename: pendingRecording.paths.wavFilename,
        wav_path: pendingRecording.paths.wavPath,
        sample_rate_hz: SAMPLE_RATE_HZ,
        format: 'pcm_s16le_mono_wav',
        source: 'AudioStreamModule mic stream',
      },
    };

    void (async () => {
      try {
        await RNFS.writeFile(pendingRecording.paths.jsonPath, JSON.stringify(metadata, null, 2), 'utf8');
        setSavedPath(`WAV: ${pendingRecording.paths.wavPath}\nJSON: ${pendingRecording.paths.jsonPath}`);
        setPendingRecording(null);
        setStatus('Saved. Pull this pair from fable_training_audio.');
      } catch (err) {
        setStatus(`Could not save metadata: ${String(err).slice(0, 120)}`);
      }
    })();
  }, [countUnclear, expectedCount, pendingRecording, selectedScenario, setup]);

  const selectScenario = useCallback((id: string) => {
    const nextScenario = scenarioById(id);
    setSelectedScenarioId(id);
    if (nextScenario.polarity === 'negative') {
      setExpectedCount('0');
      setCountUnclear(false);
    } else if (nextScenario.polarity === 'unclear') {
      setCountUnclear(true);
      setExpectedCount('');
    } else {
      setCountUnclear(false);
      setExpectedCount(value => value === '0' ? '' : value);
    }
  }, []);

  const handleBack = useCallback(() => {
    if (isRecording) {
      Alert.alert('Recording active', 'Stop or discard the current recording before leaving.');
      return;
    }
    if (pendingRecording) {
      Alert.alert('Unsaved recording', 'Save or discard this recording before leaving.');
      return;
    }
    onDone();
  }, [isRecording, onDone, pendingRecording]);

  return (
    <View style={[styles.root, { paddingTop: insets.top }]}>
      <StatusBar barStyle="light-content" backgroundColor="#000" />
      <View style={styles.header}>
        <TouchableOpacity onPress={handleBack}>
          <Text style={styles.back}>{'< Tillbaka'}</Text>
        </TouchableOpacity>
        <Text style={styles.title}>Fable data recorder</Text>
        <Text style={styles.subtitle}>Targeted WAV intake for bounce-model training</Text>
      </View>

      <View style={styles.timerBox}>
        <Text style={styles.timer}>{formatDuration(isRecording ? elapsedMs : pendingRecording?.durationMs ?? elapsedMs)}</Text>
        <Text style={styles.status}>{status}</Text>
      </View>

      <View style={styles.controls}>
        <TouchableOpacity
          style={[styles.primaryButton, isRecording ? styles.stopButton : styles.startButton, pendingRecording && styles.disabledButton]}
          onPress={isRecording ? stopRecording : startRecording}
          disabled={Boolean(pendingRecording)}
        >
          <Text style={styles.primaryButtonText}>{isRecording ? 'STOP' : 'START'}</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.infoPanel}>
        <Text style={styles.infoTitle}>What to do</Text>
        <Text style={styles.infoText}>
          Record one focused block. After stopping, choose the closest sound type, enter the
          real racket-contact count or mark it unclear, then save only useful recordings.
        </Text>
      </View>

      {savedPath ? (
        <View style={styles.savedPanel}>
          <Text style={styles.savedTitle}>Saved</Text>
          <Text style={styles.savedText}>{savedPath}</Text>
        </View>
      ) : null}

      <Modal visible={Boolean(pendingRecording)} transparent animationType="slide">
        <View style={styles.modalBackdrop}>
          <View style={styles.modalCard}>
            <View style={styles.modalHeader}>
              <View>
                <Text style={styles.modalTitle}>Tag recording</Text>
                <Text style={styles.modalSubtitle}>
                  {pendingRecording ? formatDuration(pendingRecording.durationMs) : '0:00'} captured
                </Text>
              </View>
              <TouchableOpacity
                style={[styles.infoIconButton, showHelp && styles.infoIconButtonOn]}
                onPress={() => setShowHelp(value => !value)}
              >
                <Text style={[styles.infoIconText, showHelp && styles.infoIconTextOn]}>i</Text>
              </TouchableOpacity>
            </View>

            {showHelp ? (
              <ScrollView style={styles.helpBox}>
                {SCENARIOS.map(scenario => (
                  <View key={scenario.id} style={styles.helpItem}>
                    <Text style={styles.helpTitle}>{scenario.title}</Text>
                    <Text style={styles.helpText}>{scenario.howTo}</Text>
                    <Text style={styles.helpHint}>{scenario.countHint}</Text>
                  </View>
                ))}
              </ScrollView>
            ) : (
              <ScrollView style={styles.scenarioList}>
                {SCENARIOS.map(scenario => {
                  const active = scenario.id === selectedScenarioId;
                  return (
                    <TouchableOpacity
                      key={scenario.id}
                      style={[styles.scenarioButton, active && styles.scenarioButtonOn]}
                      onPress={() => selectScenario(scenario.id)}
                    >
                      <Text style={[styles.scenarioTitle, active && styles.scenarioTitleOn]}>{scenario.title}</Text>
                      <Text style={styles.scenarioShort}>{scenario.short}</Text>
                    </TouchableOpacity>
                  );
                })}
              </ScrollView>
            )}

            <View style={styles.countSection}>
              <Text style={styles.countLabel}>Expected racket contacts</Text>
              <View style={styles.countRow}>
                <TextInput
                  style={[styles.countInput, countUnclear && styles.countInputOff]}
                  value={expectedCount}
                  onChangeText={value => {
                    setExpectedCount(value.replace(/[^0-9]/g, ''));
                    setCountUnclear(false);
                  }}
                  editable={!countUnclear}
                  keyboardType="number-pad"
                  placeholder="0"
                  placeholderTextColor="#555"
                />
                <TouchableOpacity
                  style={[styles.unclearButton, countUnclear && styles.unclearButtonOn]}
                  onPress={() => setCountUnclear(value => !value)}
                >
                  <Text style={[styles.unclearText, countUnclear && styles.unclearTextOn]}>Unclear</Text>
                </TouchableOpacity>
              </View>
              <Text style={styles.countHint}>{selectedScenario.countHint}</Text>
            </View>

            <View style={styles.modalActions}>
              <TouchableOpacity style={styles.discardButton} onPress={discardPending}>
                <Text style={styles.discardText}>Discard</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.saveButton} onPress={savePending}>
                <Text style={styles.saveText}>Save audio</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#000' },
  header: { paddingHorizontal: 16, paddingBottom: 8 },
  back: { color: '#4a9eff', fontSize: 16, paddingVertical: 6 },
  title: { color: '#fff', fontSize: 24, fontWeight: '800' },
  subtitle: { color: '#777', fontSize: 12, marginTop: 2 },
  timerBox: { alignItems: 'center', paddingVertical: 34 },
  timer: { color: '#fff', fontSize: 64, fontWeight: '800', fontVariant: ['tabular-nums'] },
  status: { color: '#999', fontSize: 13, marginTop: 8, textAlign: 'center', paddingHorizontal: 20 },
  controls: { paddingHorizontal: 24 },
  primaryButton: { borderRadius: 8, paddingVertical: 16, alignItems: 'center' },
  startButton: { backgroundColor: '#1d6f42' },
  stopButton: { backgroundColor: '#8e2b2b' },
  disabledButton: { backgroundColor: '#222' },
  primaryButtonText: { color: '#fff', fontSize: 18, fontWeight: '900' },
  infoPanel: {
    margin: 16,
    padding: 14,
    borderRadius: 8,
    backgroundColor: '#111',
    borderWidth: 1,
    borderColor: '#222',
  },
  infoTitle: { color: '#fff', fontSize: 14, fontWeight: '800', marginBottom: 6 },
  infoText: { color: '#aaa', fontSize: 13, lineHeight: 19 },
  savedPanel: {
    marginHorizontal: 16,
    padding: 12,
    borderRadius: 8,
    backgroundColor: '#0d1f33',
    borderWidth: 1,
    borderColor: '#224b76',
  },
  savedTitle: { color: '#77bdff', fontSize: 13, fontWeight: '800', marginBottom: 4 },
  savedText: { color: '#9bcfff', fontSize: 10, lineHeight: 15 },
  modalBackdrop: {
    flex: 1,
    justifyContent: 'flex-end',
    backgroundColor: 'rgba(0,0,0,0.78)',
  },
  modalCard: {
    maxHeight: '92%',
    backgroundColor: '#0b0b0b',
    borderTopLeftRadius: 8,
    borderTopRightRadius: 8,
    borderWidth: 1,
    borderColor: '#222',
    padding: 16,
  },
  modalHeader: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 },
  modalTitle: { color: '#fff', fontSize: 22, fontWeight: '900' },
  modalSubtitle: { color: '#888', fontSize: 12, marginTop: 2 },
  infoIconButton: {
    width: 32,
    height: 32,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#444',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#141414',
  },
  infoIconButtonOn: { borderColor: '#4a9eff', backgroundColor: '#0d1f33' },
  infoIconText: { color: '#aaa', fontSize: 16, fontWeight: '900' },
  infoIconTextOn: { color: '#77bdff' },
  scenarioList: { maxHeight: 310 },
  scenarioButton: {
    minHeight: 66,
    padding: 12,
    borderRadius: 8,
    backgroundColor: '#111',
    borderWidth: 1,
    borderColor: '#222',
    marginBottom: 8,
  },
  scenarioButtonOn: { borderColor: '#2ecc71', backgroundColor: '#0d2d1a' },
  scenarioTitle: { color: '#f5f5f5', fontSize: 14, fontWeight: '800' },
  scenarioTitleOn: { color: '#76f0a0' },
  scenarioShort: { color: '#999', fontSize: 12, lineHeight: 17, marginTop: 4 },
  helpBox: {
    maxHeight: 310,
    borderRadius: 8,
    backgroundColor: '#101010',
    borderWidth: 1,
    borderColor: '#252525',
    paddingHorizontal: 12,
  },
  helpItem: { paddingVertical: 12, borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: '#2a2a2a' },
  helpTitle: { color: '#fff', fontSize: 14, fontWeight: '900', marginBottom: 5 },
  helpText: { color: '#bdbdbd', fontSize: 12, lineHeight: 18 },
  helpHint: { color: '#77bdff', fontSize: 12, lineHeight: 17, marginTop: 5 },
  countSection: { paddingTop: 12 },
  countLabel: { color: '#ccc', fontSize: 13, fontWeight: '800', marginBottom: 8 },
  countRow: { flexDirection: 'row', gap: 10 },
  countInput: {
    flex: 1,
    height: 48,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#333',
    backgroundColor: '#151515',
    color: '#fff',
    fontSize: 18,
    paddingHorizontal: 14,
  },
  countInputOff: { color: '#555', backgroundColor: '#101010' },
  unclearButton: {
    minWidth: 112,
    height: 48,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#333',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#151515',
  },
  unclearButtonOn: { borderColor: '#f1c40f', backgroundColor: '#332500' },
  unclearText: { color: '#aaa', fontSize: 13, fontWeight: '800' },
  unclearTextOn: { color: '#ffe58a' },
  countHint: { color: '#777', fontSize: 12, lineHeight: 17, marginTop: 8 },
  modalActions: { flexDirection: 'row', gap: 10, paddingTop: 16 },
  discardButton: {
    flex: 1,
    height: 48,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#241111',
    borderWidth: 1,
    borderColor: '#5b2828',
  },
  discardText: { color: '#ff8a8a', fontSize: 14, fontWeight: '900' },
  saveButton: {
    flex: 1,
    height: 48,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#0f321d',
    borderWidth: 1,
    borderColor: '#267d45',
  },
  saveText: { color: '#76f0a0', fontSize: 14, fontWeight: '900' },
});
