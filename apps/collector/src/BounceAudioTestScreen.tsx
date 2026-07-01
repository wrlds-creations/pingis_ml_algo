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
import { decodeBase64PCM } from './NativeAudioCapture';
import {
  AudioStream,
  AudioStreamEmitter,
  type NativeAudioBounceEvent,
  type NativeAudioOnsetDebug,
} from './NativeAudioStream';
import {
  BOUNCE_AUDIO_TEST_CONFIG,
  BOUNCE_AUDIO_TEST_DEFAULT_RUNTIME_CONFIG,
  BOUNCE_AUDIO_TEST_DEFAULT_MODEL_ID,
  BOUNCE_AUDIO_TEST_MODEL_OPTIONS,
  BOUNCE_AUDIO_TEST_MODEL_VERSION,
  BOUNCE_AUDIO_TEST_PEAK_GATE_CONFIG,
  BOUNCE_AUDIO_TEST_RMS_FABLE_GATE_CONFIG,
  BounceAudioTestEngine,
  defaultRuntimeConfigForModelId,
  decisionConfigForModelId,
  getBounceAudioTestModelMetadata,
  getBounceAudioTestModelOption,
  modelOptionUsesTypedRuntimeConfig,
  type BounceAudioTestDecisionConfig,
  type BounceAudioTestModelOption,
  type BounceAudioCandidateRow,
  type BounceAudioTestRuntimeConfig,
} from './bounceAudioTestEngine';
import type { PlayerSetup } from './types';

const TEST_ONSET_THRESHOLD = 0.005;
const DEBUG_DIR = `${RNFS.ExternalStorageDirectoryPath}/Download/pingis_sessions/bounce_audio_test_debug`;
const SAMPLE_RATE_HZ = 22050;
const MAX_AUDIO_DEBUG_CLIPS = 180;

interface Props {
  setup: PlayerSetup;
  onDone: () => void;
}

interface DebugSessionPaths {
  sessionId: string;
  jsonPath: string;
  wavPath: string;
  wavFilename: string;
}

interface RecentRow {
  ts: string;
  probability?: number;
  decision: string;
  counted: boolean;
  peak?: number;
  explanation?: string;
}

interface TestScenario {
  id: string;
  title: string;
  polarity: 'positive' | 'negative' | 'mixed' | 'unclear';
}

interface PendingDebugSession {
  paths: DebugSessionPaths;
  startedAtIso: string;
  stoppedAtIso: string;
}

const TEST_SCENARIOS: TestScenario[] = [
  { id: 'normal_racket_bounce', title: 'Normal racket bounce', polarity: 'positive' },
  { id: 'slow_high_racket_bounce', title: 'Slow/high racket bounce', polarity: 'positive' },
  { id: 'fast_racket_bounce', title: 'Fast racket bounce', polarity: 'positive' },
  { id: 'messy_kid_style_racket_bounce', title: 'Messy/kid-style racket bounce', polarity: 'positive' },
  { id: 'racket_bounce_speaking_counting', title: 'Racket bounce + speaking/counting', polarity: 'positive' },
  { id: 'racket_bounce_background_sound', title: 'Racket bounce + background sound', polarity: 'positive' },
  { id: 'far_soft_racket_bounce_background', title: 'Far/soft racket bounce + background', polarity: 'positive' },
  { id: 'soft_high_racket_bounce_background', title: 'High racket bounce + background', polarity: 'positive' },
  { id: 'talking_only_no_bounce', title: 'Talking only, no bounce', polarity: 'negative' },
  { id: 'racket_handling_no_bounce', title: 'Racket handling, no bounce', polarity: 'negative' },
  { id: 'floor_table_other_impact_no_racket', title: 'Floor/table/other impact, no racket', polarity: 'negative' },
  { id: 'background_sound_only_no_bounce', title: 'Background sound only, no bounce', polarity: 'negative' },
  { id: 'talking_counting_background_no_bounce', title: 'Talking/counting + background, no bounce', polarity: 'negative' },
  { id: 'racket_handling_background_no_bounce', title: 'Racket handling + background, no bounce', polarity: 'negative' },
  { id: 'catch_after_sound_no_racket', title: 'Catch/after-sound, no racket', polarity: 'negative' },
  { id: 'ambiguous_ball_like_impact_near_phone_no_racket', title: 'Ball-like impact near phone, no racket', polarity: 'negative' },
  { id: 'other_unclear', title: 'Other/unclear', polarity: 'unclear' },
];

function parseNativeEvent(event: NativeAudioBounceEvent): {
  audioB64?: string;
  nativeDebug?: NativeAudioOnsetDebug;
} {
  if (typeof event === 'string') return { audioB64: event };
  return { audioB64: event.audio_b64 ?? undefined, nativeDebug: event.native_debug };
}

function buildDebugSessionPaths(startedAtIso: string): DebugSessionPaths {
  const sessionId = `bounce_audio_test_session_${startedAtIso.replace(/[:.]/g, '-')}`;
  const wavFilename = `${sessionId}.wav`;
  return {
    sessionId,
    jsonPath: `${DEBUG_DIR}/${sessionId}.json`,
    wavPath: `${DEBUG_DIR}/${wavFilename}`,
    wavFilename,
  };
}

function displayTime(ms: number) {
  return new Date(ms).toLocaleTimeString('sv-SE', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

function recentRows(rows: BounceAudioCandidateRow[]): RecentRow[] {
  return rows
    .slice()
    .sort((a, b) => b.native_onset_time_ms - a.native_onset_time_ms)
    .slice(0, 30)
    .map(row => ({
      ts: displayTime(row.native_onset_time_ms),
      probability: row.classifier_probability,
      decision: row.decision,
      counted: row.counted,
      peak: row.peak_value,
      explanation: row.debug_explanation?.summary,
    }));
}

function cappedDebugRows(rows: BounceAudioCandidateRow[]) {
  return rows.map((row, index) => {
    if (index < MAX_AUDIO_DEBUG_CLIPS) return row;
    const { audio_b64: _audioB64, ...rest } = row;
    return rest;
  });
}

function scenarioById(id: string) {
  return TEST_SCENARIOS.find(scenario => scenario.id === id) ?? TEST_SCENARIOS[0];
}

function parseProbabilityInput(text: string): number | null {
  const cleaned = text.trim().replace('%', '').replace(',', '.');
  if (!cleaned) return null;
  const value = Number(cleaned);
  if (!Number.isFinite(value)) return null;
  const probability = value > 1 ? value / 100 : value;
  if (probability < 0 || probability > 1) return null;
  return probability;
}

function formatProbabilityInput(value: number): string {
  if (value >= 0.995) return '1.00';
  return value.toFixed(3).replace(/0+$/, '').replace(/\.$/, '');
}

function formatPercent(value: number): string {
  const percent = value * 100;
  return `${Number.isInteger(percent) ? percent.toFixed(0) : percent.toFixed(1)}%`;
}

export function BounceAudioTestScreen({ setup, onDone }: Props) {
  const insets = useSafeAreaInsets();
  const engineRef = useRef(new BounceAudioTestEngine());
  const startedAtRef = useRef<string | null>(null);
  const pathsRef = useRef<DebugSessionPaths | null>(null);
  const defaultModelOption = getBounceAudioTestModelOption(BOUNCE_AUDIO_TEST_DEFAULT_MODEL_ID);
  const activeRuntimeConfigRef = useRef<BounceAudioTestRuntimeConfig>({
    ...BOUNCE_AUDIO_TEST_DEFAULT_RUNTIME_CONFIG,
  });
  const activeDecisionConfigRef = useRef<BounceAudioTestDecisionConfig>({
    ...BOUNCE_AUDIO_TEST_CONFIG,
  });
  const activeModelOptionRef = useRef<BounceAudioTestModelOption>(defaultModelOption);
  const [isListening, setIsListening] = useState(false);
  const [hitCount, setHitCount] = useState(0);
  const [candidateCount, setCandidateCount] = useState(0);
  const [classifiedCount, setClassifiedCount] = useState(0);
  const [pendingCount, setPendingCount] = useState(0);
  const [lastCounted, setLastCounted] = useState<BounceAudioCandidateRow | null>(null);
  const [lastResult, setLastResult] = useState<BounceAudioCandidateRow | null>(null);
  const [recent, setRecent] = useState<RecentRow[]>([]);
  const [savedDebugPath, setSavedDebugPath] = useState<string | null>(null);
  const [status, setStatus] = useState('Ready.');
  const [pendingDebugSession, setPendingDebugSession] = useState<PendingDebugSession | null>(null);
  const [selectedModelId, setSelectedModelId] = useState(BOUNCE_AUDIO_TEST_DEFAULT_MODEL_ID);
  const selectedModelOption = getBounceAudioTestModelOption(selectedModelId);
  const selectedModelUsesTypedConfig = modelOptionUsesTypedRuntimeConfig(selectedModelOption);
  const [activeModelOption, setActiveModelOption] = useState<BounceAudioTestModelOption>(defaultModelOption);
  const [selectedScenarioId, setSelectedScenarioId] = useState(TEST_SCENARIOS[0].id);
  const [expectedCount, setExpectedCount] = useState('');
  const [countUnclear, setCountUnclear] = useState(false);
  const [thresholdText, setThresholdText] = useState(
    formatProbabilityInput(BOUNCE_AUDIO_TEST_DEFAULT_RUNTIME_CONFIG.threshold),
  );
  const [noiseVetoText, setNoiseVetoText] = useState(
    formatProbabilityInput(BOUNCE_AUDIO_TEST_DEFAULT_RUNTIME_CONFIG.fableNoiseVetoThreshold),
  );
  const [activeRuntimeConfig, setActiveRuntimeConfig] = useState<BounceAudioTestRuntimeConfig>({
    ...BOUNCE_AUDIO_TEST_DEFAULT_RUNTIME_CONFIG,
  });

  const typedRuntimeConfig = useMemo(() => {
    if (!selectedModelUsesTypedConfig) return defaultRuntimeConfigForModelId(selectedModelId);
    const threshold = parseProbabilityInput(thresholdText);
    const fableNoiseVetoThreshold = parseProbabilityInput(noiseVetoText);
    if (threshold === null || fableNoiseVetoThreshold === null) return null;
    return { threshold, fableNoiseVetoThreshold };
  }, [noiseVetoText, selectedModelId, selectedModelUsesTypedConfig, thresholdText]);

  const configError = useMemo(() => {
    if (!selectedModelUsesTypedConfig) return null;
    const threshold = parseProbabilityInput(thresholdText);
    if (threshold === null) return 'p threshold must be 0-1 or 0-100%.';
    const veto = parseProbabilityInput(noiseVetoText);
    if (veto === null) return 'Noise veto must be 0-1 or 0-100%.';
    return null;
  }, [noiseVetoText, selectedModelUsesTypedConfig, thresholdText]);

  const syncSnapshot = useCallback(() => {
    const rows = engineRef.current.getRows();
    const counts = engineRef.current.getCounts();
    setHitCount(counts.counted);
    setCandidateCount(counts.native_candidates);
    setClassifiedCount(counts.classified);
    setPendingCount(counts.pending + counts.accepted_pending_dedupe);
    const latestClassified = rows
      .slice()
      .reverse()
      .find(row => row.classifier_probability !== undefined || row.decision !== 'pending_delay');
    if (latestClassified) setLastResult(latestClassified);
    setRecent(recentRows(rows));
  }, []);

  const flushEngine = useCallback((final = false) => {
    const result = engineRef.current.flush(Date.now(), final);
    if (result.newlyCounted.length > 0) {
      setLastCounted(result.newlyCounted[result.newlyCounted.length - 1]);
    }
    if (result.rowsChanged || result.newlyCounted.length > 0 || final) syncSnapshot();
  }, [syncSnapshot]);

  const saveDebugSession = useCallback(async (pending: PendingDebugSession) => {
    const parsedCount = Number.parseInt(expectedCount.trim(), 10);
    if (!countUnclear && (!Number.isFinite(parsedCount) || parsedCount < 0)) {
      Alert.alert('Expected count', 'Enter a number, or choose Unclear.');
      return;
    }
    const scenario = scenarioById(selectedScenarioId);
    const paths = pending.paths;
    const rows = engineRef.current.getRows();
    const counts = engineRef.current.getCounts();
    const runtimeConfig = activeRuntimeConfigRef.current;
    const decisionConfig = activeDecisionConfigRef.current;
    const modelOption = activeModelOptionRef.current;
    const modelMetadata = engineRef.current.getModelMetadata();
    const payload = {
      type: 'bounce_audio_test_debug_session',
      schema_version: 1,
      model_version: modelMetadata.model_version ?? modelOption.id,
      model_selector: {
        id: modelOption.id,
        title: modelOption.title,
        short_title: modelOption.shortTitle,
        subtitle: modelOption.subtitle,
        runtime_mode: modelOption.runtimeMode,
      },
      model_metadata: modelMetadata,
      player: setup,
      created_at: new Date().toISOString(),
      started_at: pending.startedAtIso,
      stopped_at: pending.stoppedAtIso,
      review: {
        scenario,
        expected_racket_contacts: countUnclear ? null : parsedCount,
        expected_count_unclear: countUnclear,
        app_count_at_stop: counts.counted,
      },
      continuous_audio: {
        wav_filename: paths.wavFilename,
        wav_path: paths.wavPath,
        sample_rate_hz: SAMPLE_RATE_HZ,
        format: 'pcm_s16le_mono_wav',
        source: 'AudioStreamModule mic stream',
      },
      peak_gate_config: modelOption.runtimeMode === 'peak_extra_trees' ? BOUNCE_AUDIO_TEST_PEAK_GATE_CONFIG : null,
      rms_fable_gate_config: modelOption.runtimeMode === 'rms_fable' ? BOUNCE_AUDIO_TEST_RMS_FABLE_GATE_CONFIG : null,
      decision_config: {
        ...decisionConfig,
        threshold: runtimeConfig.threshold,
        fableNoiseVetoThreshold: runtimeConfig.fableNoiseVetoThreshold,
        source: modelOptionUsesTypedRuntimeConfig(modelOption)
          ? 'typed_bounce_audio_test_ui'
          : 'original_rms_fable_counter',
        selected_model_id: modelOption.id,
        selected_model_title: modelOption.title,
        threshold_input_text: thresholdText,
        fable_noise_veto_input_text: noiseVetoText,
        typed_thresholds_used: modelOptionUsesTypedRuntimeConfig(modelOption),
      },
      counts,
      candidates: cappedDebugRows(rows),
    };
    try {
      await RNFS.mkdir(DEBUG_DIR);
      await RNFS.writeFile(paths.jsonPath, JSON.stringify(payload, null, 2), 'utf8');
      setSavedDebugPath(`JSON: ${paths.jsonPath}\nWAV: ${paths.wavPath}`);
      setStatus('Saved debug JSON/WAV.');
      setPendingDebugSession(null);
    } catch (err) {
      setSavedDebugPath(null);
      setStatus(`Could not save debug: ${String(err).slice(0, 120)}`);
    }
  }, [countUnclear, expectedCount, noiseVetoText, selectedScenarioId, setup, thresholdText]);

  const start = useCallback(() => {
    if (isListening || pendingDebugSession) return;
    if (!typedRuntimeConfig) {
      setStatus(configError ?? 'Enter valid probability config before starting.');
      return;
    }
    const startedAtIso = new Date().toISOString();
    const paths = buildDebugSessionPaths(startedAtIso);
    const modelSelection = engineRef.current.setModelOption(selectedModelId, typedRuntimeConfig);
    activeRuntimeConfigRef.current = modelSelection.runtimeConfig;
    activeDecisionConfigRef.current = modelSelection.decisionConfig;
    activeModelOptionRef.current = modelSelection.modelOption;
    setActiveRuntimeConfig(modelSelection.runtimeConfig);
    setActiveModelOption(modelSelection.modelOption);
    engineRef.current.reset();
    startedAtRef.current = startedAtIso;
    pathsRef.current = paths;
    setHitCount(0);
    setCandidateCount(0);
    setClassifiedCount(0);
    setPendingCount(0);
    setLastCounted(null);
    setLastResult(null);
    setRecent([]);
    setSavedDebugPath(null);
    setSelectedScenarioId(TEST_SCENARIOS[0].id);
    setExpectedCount('');
    setCountUnclear(false);
    setStatus(modelSelection.modelOption.runtimeMode === 'rms_fable'
      ? 'Starting RMS + Fable stream...'
      : 'Starting peak-gate stream...');

    void (async () => {
      try {
        await RNFS.mkdir(DEBUG_DIR);
        await AudioStream.setDebugRecordingPath(paths.wavPath);
        await AudioStream.startStreaming(TEST_ONSET_THRESHOLD);
        if (modelSelection.modelOption.runtimeMode === 'rms_fable') {
          await AudioStream.setRetriggerMs(BOUNCE_AUDIO_TEST_RMS_FABLE_GATE_CONFIG.retriggerMs);
          await AudioStream.setGateConfig(
            BOUNCE_AUDIO_TEST_RMS_FABLE_GATE_CONFIG.gateMode,
            BOUNCE_AUDIO_TEST_RMS_FABLE_GATE_CONFIG.spectralGate,
            BOUNCE_AUDIO_TEST_RMS_FABLE_GATE_CONFIG.absoluteMinimumRms,
          );
        } else {
          await AudioStream.setPeakGateConfig(
            true,
            BOUNCE_AUDIO_TEST_PEAK_GATE_CONFIG.smoothingMs,
            BOUNCE_AUDIO_TEST_PEAK_GATE_CONFIG.minGapMs,
            BOUNCE_AUDIO_TEST_PEAK_GATE_CONFIG.backgroundWindowMs,
            BOUNCE_AUDIO_TEST_PEAK_GATE_CONFIG.backgroundExcludeBeforePeakMs,
            BOUNCE_AUDIO_TEST_PEAK_GATE_CONFIG.absoluteMinimum,
            BOUNCE_AUDIO_TEST_PEAK_GATE_CONFIG.ratioMinimum,
            BOUNCE_AUDIO_TEST_PEAK_GATE_CONFIG.zMinimum,
          );
        }
        setIsListening(true);
        setStatus(modelSelection.modelOption.runtimeMode === 'rms_fable'
          ? 'Listening with RMS+Fable original gate/counter. p and veto fields are ignored.'
          : `Listening with ${modelSelection.modelOption.shortTitle}, p>=${formatPercent(modelSelection.runtimeConfig.threshold)}, noise veto>=${formatPercent(modelSelection.runtimeConfig.fableNoiseVetoThreshold)}.`);
      } catch (err) {
        setStatus(`Could not start: ${String(err).slice(0, 120)}`);
        try { await AudioStream.stopStreaming(); } catch (_) {}
        try { await AudioStream.setDebugRecordingPath(null); } catch (_) {}
      }
    })();
  }, [configError, isListening, pendingDebugSession, selectedModelId, typedRuntimeConfig]);

  const stop = useCallback(() => {
    if (!isListening) return;
    void (async () => {
      setStatus('Stopping...');
      try { await AudioStream.stopStreaming(); } catch (_) {}
      try {
        await AudioStream.setPeakGateConfig(false, 3, 220, 500, 60, 0.08, 2.0, 0.0);
      } catch (_) {}
      setIsListening(false);
      flushEngine(true);
      setSelectedScenarioId(TEST_SCENARIOS[0].id);
      setExpectedCount(String(engineRef.current.getCounts().counted));
      setCountUnclear(false);
      setPendingDebugSession({
        paths: pathsRef.current ?? buildDebugSessionPaths(new Date().toISOString()),
        startedAtIso: startedAtRef.current ?? new Date().toISOString(),
        stoppedAtIso: new Date().toISOString(),
      });
      setStatus('Stopped. Choose sound type, expected count, then save or discard.');
    })();
  }, [flushEngine, isListening]);

  const discardPending = useCallback(() => {
    if (!pendingDebugSession) return;
    void (async () => {
      try {
        const wavExists = await RNFS.exists(pendingDebugSession.paths.wavPath);
        if (wavExists) await RNFS.unlink(pendingDebugSession.paths.wavPath);
      } catch (_) {}
      try {
        const jsonExists = await RNFS.exists(pendingDebugSession.paths.jsonPath);
        if (jsonExists) await RNFS.unlink(pendingDebugSession.paths.jsonPath);
      } catch (_) {}
      setPendingDebugSession(null);
      setSavedDebugPath(null);
      setStatus('Discarded this test recording.');
    })();
  }, [pendingDebugSession]);

  useEffect(() => {
    if (!isListening) return;
    const sub = AudioStreamEmitter.addListener('onBounceDetected', (event: NativeAudioBounceEvent) => {
      const receivedAtMs = Date.now();
      const { audioB64, nativeDebug } = parseNativeEvent(event);
      if (!audioB64) {
        setStatus(nativeDebug?.native_reject_reason ?? 'Native rejected a candidate.');
        return;
      }
      try {
        engineRef.current.addCandidate({
          pcm: decodeBase64PCM(audioB64),
          audioB64,
          nativeDebug,
          receivedAtMs,
        });
        syncSnapshot();
        flushEngine(false);
      } catch (err) {
        setStatus(`Candidate error: ${String(err).slice(0, 120)}`);
      }
    });
    const interval = setInterval(() => flushEngine(false), 100);
    return () => {
      clearInterval(interval);
      sub.remove();
    };
  }, [flushEngine, isListening, syncSnapshot]);

  useEffect(() => () => {
    if (isListening) {
      AudioStream.stopStreaming().catch(() => {});
      AudioStream.setDebugRecordingPath(null).catch(() => {});
    }
  }, [isListening]);

  const toggle = isListening ? stop : start;
  const lastProbability = lastResult?.classifier_probability;
  const lastExplanation = lastResult?.debug_explanation?.summary;
  const lastCountedText = lastCounted
    ? `${displayTime(lastCounted.native_onset_time_ms)} p=${((lastCounted.classifier_probability ?? 0) * 100).toFixed(1)}%`
    : 'None yet';
  const displayedRuntimeConfig = isListening || pendingDebugSession
    ? activeRuntimeConfig
    : typedRuntimeConfig ?? activeRuntimeConfig;
  const canEditConfig = !isListening && !pendingDebugSession;
  const startDisabled = !isListening && (Boolean(pendingDebugSession) || typedRuntimeConfig === null);
  const displayedModelOption = isListening || pendingDebugSession ? activeModelOption : selectedModelOption;
  const displayedModelMetadata = getBounceAudioTestModelMetadata(displayedModelOption);
  const displayedModelUsesTypedConfig = modelOptionUsesTypedRuntimeConfig(displayedModelOption);
  const displayedCandidateLabel = displayedModelOption.runtimeMode === 'rms_fable' ? 'RMS candidates' : 'peak candidates';
  const displayedDecisionConfig = isListening || pendingDebugSession
    ? activeDecisionConfigRef.current
    : decisionConfigForModelId(displayedModelOption.id, displayedRuntimeConfig);
  const selectModel = useCallback((modelId: string) => {
    if (!canEditConfig) return;
    const threshold = parseProbabilityInput(thresholdText);
    const fableNoiseVetoThreshold = parseProbabilityInput(noiseVetoText);
    const preservedConfig = threshold === null || fableNoiseVetoThreshold === null
      ? undefined
      : { threshold, fableNoiseVetoThreshold };
    const modelSelection = engineRef.current.setModelOption(modelId, preservedConfig);
    activeRuntimeConfigRef.current = modelSelection.runtimeConfig;
    activeDecisionConfigRef.current = modelSelection.decisionConfig;
    activeModelOptionRef.current = modelSelection.modelOption;
    setSelectedModelId(modelSelection.modelOption.id);
    setActiveModelOption(modelSelection.modelOption);
    setActiveRuntimeConfig(modelSelection.runtimeConfig);
    setStatus(preservedConfig
      ? `Selected ${modelSelection.modelOption.title}. Keeping typed p/noise values.`
      : `Selected ${modelSelection.modelOption.title}. Fix typed p/noise values before START.`);
  }, [canEditConfig, noiseVetoText, thresholdText]);

  return (
    <View style={[styles.root, { paddingTop: insets.top }]}>
      <StatusBar barStyle="light-content" backgroundColor="#000" />
      <View style={styles.header}>
        <TouchableOpacity onPress={() => {
          if (!isListening) {
            onDone();
            return;
          }
          void (async () => {
            stop();
            setTimeout(onDone, 250);
          })();
        }}>
          <Text style={styles.back}>Back</Text>
        </TouchableOpacity>
        <Text style={styles.title}>Bounce audio test</Text>
        <Text style={styles.subtitle}>
          {displayedModelMetadata.model_version ?? BOUNCE_AUDIO_TEST_MODEL_VERSION}
        </Text>
      </View>

      <View style={styles.counterBox}>
        <Text style={styles.counterValue}>{hitCount}</Text>
        <Text style={styles.counterLabel}>counted racket bounces</Text>
        <Text style={styles.eventMeta}>
          {candidateCount} {displayedCandidateLabel} · {classifiedCount} classified · {pendingCount} waiting
        </Text>
        {lastCounted?.bounce_height_m !== undefined && lastCounted.bounce_gap_ms !== undefined ? (
          <Text style={styles.heightText}>
            height {(lastCounted.bounce_height_m * 100).toFixed(0)} cm · gap {lastCounted.bounce_gap_ms.toFixed(0)} ms
          </Text>
        ) : null}
      </View>

      <View style={styles.lastBox}>
        <Text style={styles.lastLabel}>
          Last counted: {lastCountedText}
        </Text>
        <Text style={styles.probRow}>
          Last candidate: {lastProbability === undefined ? '-' : `${(lastProbability * 100).toFixed(1)}%`} · {lastResult?.decision ?? '-'}
        </Text>
        {lastExplanation ? (
          <Text style={styles.reasonRow}>
            Why: {lastExplanation}
          </Text>
        ) : null}
      </View>

      <View style={styles.modelPanel}>
        <Text style={styles.configLabel}>model</Text>
        <View style={styles.modelButtons}>
          {BOUNCE_AUDIO_TEST_MODEL_OPTIONS.map(option => {
            const selected = option.id === selectedModelId;
            const modelDefaults = defaultRuntimeConfigForModelId(option.id);
            const usesTypedConfig = modelOptionUsesTypedRuntimeConfig(option);
            return (
              <TouchableOpacity
                key={option.id}
                style={[
                  styles.modelButton,
                  selected && styles.modelButtonOn,
                  !canEditConfig && !selected && styles.modelButtonDisabled,
                ]}
                onPress={() => selectModel(option.id)}
                activeOpacity={0.75}
                disabled={!canEditConfig}
              >
                <Text style={[styles.modelButtonTitle, selected && styles.modelButtonTitleOn]}>
                  {option.shortTitle}
                </Text>
                <Text style={[styles.modelButtonMeta, selected && styles.modelButtonMetaOn]}>
                  {usesTypedConfig ? `p ${formatPercent(modelDefaults.threshold)}` : 'original gate'}
                </Text>
              </TouchableOpacity>
            );
          })}
        </View>
      </View>

      <View style={styles.configPanel}>
        <View style={styles.configInputGroup}>
          <Text style={styles.configLabel}>p threshold</Text>
          <TextInput
            style={[
              styles.configInput,
              !selectedModelUsesTypedConfig && styles.configInputDisabled,
              selectedModelUsesTypedConfig && parseProbabilityInput(thresholdText) === null && styles.configInputError,
            ]}
            value={thresholdText}
            onChangeText={setThresholdText}
            editable={canEditConfig && selectedModelUsesTypedConfig}
            keyboardType="default"
            placeholder={formatProbabilityInput(BOUNCE_AUDIO_TEST_DEFAULT_RUNTIME_CONFIG.threshold)}
            placeholderTextColor="#555"
            selectTextOnFocus
          />
        </View>
        <View style={styles.configInputGroup}>
          <Text style={styles.configLabel}>noise veto</Text>
          <TextInput
            style={[
              styles.configInput,
              !selectedModelUsesTypedConfig && styles.configInputDisabled,
              selectedModelUsesTypedConfig && parseProbabilityInput(noiseVetoText) === null && styles.configInputError,
            ]}
            value={noiseVetoText}
            onChangeText={setNoiseVetoText}
            editable={canEditConfig && selectedModelUsesTypedConfig}
            keyboardType="default"
            placeholder={formatProbabilityInput(BOUNCE_AUDIO_TEST_DEFAULT_RUNTIME_CONFIG.fableNoiseVetoThreshold)}
            placeholderTextColor="#555"
            selectTextOnFocus
          />
        </View>
      </View>
      <Text style={styles.configHint}>
        {selectedModelUsesTypedConfig
          ? 'Type decimals or percents: 0.575, 57.5%, or 100% to disable noise veto.'
          : 'RMS+Fable uses the original Fable counter. p threshold and noise veto are ignored for this option.'}
      </Text>
      {configError && canEditConfig ? (
        <Text style={styles.configError}>{configError}</Text>
      ) : null}

      <TouchableOpacity
        style={[
          styles.toggle,
          isListening ? styles.toggleStop : styles.toggleStart,
          startDisabled ? styles.toggleDisabled : null,
        ]}
        onPress={toggle}
        activeOpacity={0.75}
        disabled={startDisabled}
      >
        <Text style={styles.toggleText}>{isListening ? 'STOP' : pendingDebugSession ? 'TAG RECORDING' : 'START'}</Text>
      </TouchableOpacity>

      <Text style={styles.configLine}>
        {displayedModelUsesTypedConfig
          ? `${displayedModelOption.shortTitle} TEST | Peak gate raw abs 3 ms | p>=${formatPercent(displayedRuntimeConfig.threshold)} | Fable noise veto ${displayedRuntimeConfig.fableNoiseVetoThreshold >= 1 ? 'off' : `>=${formatPercent(displayedRuntimeConfig.fableNoiseVetoThreshold)}`} | dedupe ${displayedDecisionConfig.smartDedupeMs} ms | delay ${displayedDecisionConfig.decisionDelayMs} ms`
          : `${displayedModelOption.shortTitle} TEST | RMS bandpass gate | retrigger ${BOUNCE_AUDIO_TEST_RMS_FABLE_GATE_CONFIG.retriggerMs} ms | abs RMS ${BOUNCE_AUDIO_TEST_RMS_FABLE_GATE_CONFIG.absoluteMinimumRms} | original Fable confidence/count logic`}
      </Text>
      <Text style={styles.warningLine}>
        {'Still diagnostic. Selected model and active runtime are saved in each JSON.'}
      </Text>
      <Text style={styles.status}>{status}</Text>
      {savedDebugPath ? <Text style={styles.savedPath}>{savedDebugPath}</Text> : null}

      <ScrollView style={styles.eventList}>
        {recent.map((row, index) => (
          <View key={`${row.ts}-${index}`} style={styles.eventRow}>
            <Text style={[styles.eventText, row.counted ? styles.eventCounted : styles.eventIgnored]}>
              {row.ts}  {row.probability === undefined ? 'pending' : `p=${(row.probability * 100).toFixed(1)}%`}  {row.counted ? 'counted' : row.decision}
            </Text>
            {row.explanation ? (
              <Text style={styles.eventReason}>{row.explanation}</Text>
            ) : null}
          </View>
        ))}
      </ScrollView>

      <Modal
        visible={pendingDebugSession !== null}
        transparent
        animationType="slide"
        onRequestClose={() => {}}
      >
        <View style={styles.modalBackdrop}>
          <View style={styles.modalPanel}>
            <Text style={styles.modalTitle}>Save this test?</Text>
            <Text style={styles.modalSubtitle}>
              Choose the closest sound type and enter the real racket-contact count.
            </Text>

            <ScrollView style={styles.scenarioList}>
              {TEST_SCENARIOS.map(scenario => (
                <TouchableOpacity
                  key={scenario.id}
                  style={[
                    styles.scenarioButton,
                    selectedScenarioId === scenario.id && styles.scenarioButtonOn,
                  ]}
                  onPress={() => setSelectedScenarioId(scenario.id)}
                  activeOpacity={0.75}
                >
                  <Text
                    style={[
                      styles.scenarioText,
                      selectedScenarioId === scenario.id && styles.scenarioTextOn,
                    ]}
                  >
                    {scenario.title}
                  </Text>
                </TouchableOpacity>
              ))}
            </ScrollView>

            <View style={styles.countRow}>
              <TextInput
                style={[styles.countInput, countUnclear && styles.countInputDisabled]}
                value={expectedCount}
                onChangeText={setExpectedCount}
                editable={!countUnclear}
                keyboardType="number-pad"
                placeholder="Expected count"
                placeholderTextColor="#666"
              />
              <TouchableOpacity
                style={[styles.unclearButton, countUnclear && styles.unclearButtonOn]}
                onPress={() => setCountUnclear(value => !value)}
              >
                <Text style={[styles.unclearText, countUnclear && styles.unclearTextOn]}>
                  Unclear
                </Text>
              </TouchableOpacity>
            </View>

            <Text style={styles.modalMeta}>
              App counted {engineRef.current.getCounts().counted}. Save only useful tests.
            </Text>

            <View style={styles.modalActions}>
              <TouchableOpacity style={styles.discardButton} onPress={discardPending}>
                <Text style={styles.discardText}>Discard</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.saveButton}
                onPress={() => pendingDebugSession && void saveDebugSession(pendingDebugSession)}
              >
                <Text style={styles.saveText}>Save labels</Text>
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
  subtitle: { color: '#555', fontSize: 11 },
  counterBox: { alignItems: 'center', paddingVertical: 12 },
  counterValue: { color: '#2ecc71', fontSize: 72, fontWeight: '800' },
  counterLabel: { color: '#aaa', fontSize: 14 },
  eventMeta: { color: '#666', fontSize: 12, marginTop: 4 },
  heightText: { color: '#f1c40f', fontSize: 15, fontWeight: '700', marginTop: 4 },
  lastBox: { alignItems: 'center', paddingVertical: 6, paddingHorizontal: 16 },
  lastLabel: { color: '#ddd', fontSize: 16, fontWeight: '700' },
  probRow: { color: '#888', fontSize: 13, marginTop: 3, fontVariant: ['tabular-nums'] },
  reasonRow: { color: '#aaa', fontSize: 12, marginTop: 4, lineHeight: 16, textAlign: 'center' },
  toggle: { marginHorizontal: 24, marginVertical: 10, paddingVertical: 14, borderRadius: 10, alignItems: 'center' },
  toggleStart: { backgroundColor: '#1d6f42' },
  toggleStop: { backgroundColor: '#8e2b2b' },
  toggleDisabled: { backgroundColor: '#333' },
  toggleText: { color: '#fff', fontSize: 18, fontWeight: '800' },
  modelPanel: { marginHorizontal: 24, marginTop: 4 },
  modelButtons: { flexDirection: 'row', gap: 8 },
  modelButton: {
    flex: 1,
    borderWidth: 1,
    borderColor: '#333',
    borderRadius: 8,
    backgroundColor: '#101010',
    paddingVertical: 9,
    paddingHorizontal: 10,
    minHeight: 54,
  },
  modelButtonOn: { borderColor: '#4a9eff', backgroundColor: '#0d1f33' },
  modelButtonDisabled: { opacity: 0.45 },
  modelButtonTitle: { color: '#aaa', fontSize: 13, fontWeight: '800' },
  modelButtonTitleOn: { color: '#fff' },
  modelButtonMeta: { color: '#666', fontSize: 10, marginTop: 3 },
  modelButtonMetaOn: { color: '#9dccff' },
  configPanel: {
    flexDirection: 'row',
    gap: 10,
    marginHorizontal: 24,
    marginTop: 4,
  },
  configInputGroup: { flex: 1 },
  configLabel: { color: '#888', fontSize: 11, fontWeight: '700', marginBottom: 4 },
  configInput: {
    borderWidth: 1,
    borderColor: '#333',
    borderRadius: 8,
    backgroundColor: '#0b0b0b',
    color: '#fff',
    paddingHorizontal: 10,
    minHeight: 38,
    fontSize: 15,
    fontVariant: ['tabular-nums'],
  },
  configInputDisabled: { color: '#666', backgroundColor: '#111' },
  configInputError: { borderColor: '#9b3333' },
  configHint: { color: '#666', fontSize: 10, textAlign: 'center', marginHorizontal: 18, marginTop: 5 },
  configError: { color: '#ff9a9a', fontSize: 11, textAlign: 'center', marginHorizontal: 18, marginTop: 4 },
  configLine: { color: '#666', fontSize: 11, textAlign: 'center', marginHorizontal: 18, marginBottom: 4 },
  warningLine: { color: '#c99a2e', fontSize: 11, textAlign: 'center', marginHorizontal: 18, marginBottom: 4 },
  status: { color: '#4a9eff', fontSize: 11, textAlign: 'center', marginHorizontal: 18, marginBottom: 6 },
  savedPath: { color: '#4a9eff', fontSize: 10, paddingHorizontal: 16, marginBottom: 4 },
  eventList: { flex: 1, paddingHorizontal: 16 },
  eventRow: { paddingVertical: 5, borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: '#1a1a1a' },
  eventText: { fontSize: 13, fontVariant: ['tabular-nums'] },
  eventCounted: { color: '#2ecc71' },
  eventIgnored: { color: '#777' },
  eventReason: { color: '#666', fontSize: 11, lineHeight: 15, marginTop: 2 },
  modalBackdrop: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.78)',
    justifyContent: 'flex-end',
  },
  modalPanel: {
    backgroundColor: '#111',
    borderTopLeftRadius: 18,
    borderTopRightRadius: 18,
    borderWidth: 1,
    borderColor: '#333',
    padding: 18,
    maxHeight: '88%',
  },
  modalTitle: { color: '#fff', fontSize: 22, fontWeight: '800', marginBottom: 4 },
  modalSubtitle: { color: '#aaa', fontSize: 13, lineHeight: 18, marginBottom: 12 },
  scenarioList: { maxHeight: 270, marginBottom: 12 },
  scenarioButton: {
    borderWidth: 1,
    borderColor: '#333',
    backgroundColor: '#171717',
    borderRadius: 10,
    paddingVertical: 10,
    paddingHorizontal: 12,
    marginBottom: 7,
  },
  scenarioButtonOn: { borderColor: '#4a9eff', backgroundColor: '#0d1f33' },
  scenarioText: { color: '#aaa', fontSize: 13, fontWeight: '700' },
  scenarioTextOn: { color: '#fff' },
  countRow: { flexDirection: 'row', gap: 10, marginBottom: 10 },
  countInput: {
    flex: 1,
    borderWidth: 1,
    borderColor: '#333',
    borderRadius: 10,
    backgroundColor: '#0b0b0b',
    color: '#fff',
    paddingHorizontal: 12,
    minHeight: 46,
    fontSize: 16,
  },
  countInputDisabled: { color: '#555' },
  unclearButton: {
    borderWidth: 1,
    borderColor: '#333',
    borderRadius: 10,
    paddingHorizontal: 14,
    alignItems: 'center',
    justifyContent: 'center',
    minWidth: 94,
  },
  unclearButtonOn: { borderColor: '#4a9eff', backgroundColor: '#0d1f33' },
  unclearText: { color: '#999', fontWeight: '800' },
  unclearTextOn: { color: '#fff' },
  modalMeta: { color: '#777', fontSize: 12, marginBottom: 12 },
  modalActions: { flexDirection: 'row', gap: 10 },
  discardButton: {
    flex: 1,
    borderRadius: 10,
    paddingVertical: 13,
    alignItems: 'center',
    backgroundColor: '#2b1a1a',
    borderWidth: 1,
    borderColor: '#5a2d2d',
  },
  saveButton: {
    flex: 1,
    borderRadius: 10,
    paddingVertical: 13,
    alignItems: 'center',
    backgroundColor: '#1d6f42',
  },
  discardText: { color: '#ffb0b0', fontSize: 15, fontWeight: '800' },
  saveText: { color: '#fff', fontSize: 15, fontWeight: '800' },
});
