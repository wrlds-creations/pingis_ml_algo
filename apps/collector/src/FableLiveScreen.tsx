/**
 * FableLiveScreen.tsx
 *
 * "Fable-algoritm" — separat live-testläge för den brusrobusta
 * racketstuds-detektorn (noise_robust v3, HistGB all83):
 *
 *   - Bandpassad onset-gate 1.5–7 kHz (musik ligger lågfrekvent)
 *   - Ingen hård spektralgate (modellen sköter avvisningen)
 *   - Retrigger 120 ms, abs-min-RMS 0.0015
 *   - 83 features (62 befintliga + 21 nr_) + HistGB med softmax
 *   - Räknelogik: konfidens >= 0.5, merge 120 ms, grupp 80 ms,
 *     eko-gate 300 ms / 0.6x RMS
 *
 * Rör INTE Studsdetektor/övriga lägen: egen modell (fable_audio_model.json),
 * egen motor (fableEngine.ts) och gate-konfig som återställs av native vid
 * varje startStreaming.
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  View, Text, TouchableOpacity, ScrollView,
  StyleSheet, StatusBar,
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
import { FableCounter, FABLE_DEFAULT_CONFIG, type FableDetectionResult } from './fableEngine';
import { FABLE_MODEL_VERSION } from './hgbRuntime';
import type { PlayerSetup } from './types';

const FABLE_ONSET_THRESHOLD = 0.005; // -> native onset-ratio 1.5
const FABLE_RETRIGGER_MS = 120;
const FABLE_ABS_MIN_RMS = 0.0015;
const FABLE_DEBUG_DIR = `${RNFS.ExternalStorageDirectoryPath}/Download/pingis_sessions/fable_live_debug`;

interface Props { setup: PlayerSetup; onDone: () => void; }

interface FableEventRow {
  ts: string;
  label: string;
  confidence: number;
  counted: boolean;
  reason: string;
}

interface FableDebugEvent {
  index: number;
  received_at_ms: number;
  native_onset_time_ms?: number;
  native_rms?: number;
  native_background_rms?: number;
  model_label?: string;
  model_confidence?: number;
  model_probabilities?: Record<string, number>;
  bg_mode?: string;
  bg_rms_db?: number;
  counted: boolean;
  reject_reason: string;
  feature_ms?: number;
  predict_ms?: number;
}

function parseNativeEvent(event: NativeAudioBounceEvent): {
  audioB64?: string;
  nativeDebug?: NativeAudioOnsetDebug;
} {
  if (typeof event === 'string') return { audioB64: event };
  return { audioB64: event.audio_b64 ?? undefined, nativeDebug: event.native_debug };
}

function labelName(label?: string) {
  switch (label) {
    case 'racket_bounce': return 'RACKET';
    case 'table_bounce': return 'BORD';
    case 'floor_bounce': return 'GOLV';
    case 'noise': return 'BRUS';
    default: return '-';
  }
}

export function FableLiveScreen({ setup, onDone }: Props) {
  const insets = useSafeAreaInsets();
  const [isListening, setIsListening] = useState(false);
  const [hitCount, setHitCount] = useState(0);
  const [eventCount, setEventCount] = useState(0);
  const [staleCount, setStaleCount] = useState(0);
  const [lastHeight, setLastHeight] = useState<{ heightM: number; gapMs: number; avgM: number } | null>(null);
  const heightsRef = useRef<number[]>([]);
  const [lastResult, setLastResult] = useState<{ label: string; conf: number; reason: string; probs: Record<string, number>; bgMode?: string } | null>(null);
  const [recentEvents, setRecentEvents] = useState<FableEventRow[]>([]);
  const [latencyMs, setLatencyMs] = useState<{ p50: number; max: number } | null>(null);
  const [savedDebugPath, setSavedDebugPath] = useState<string | null>(null);

  const counterRef = useRef(new FableCounter());
  const latenciesRef = useRef<number[]>([]);
  const debugEventsRef = useRef<FableDebugEvent[]>([]);
  const startedAtRef = useRef<string | null>(null);

  const saveDebugSession = useCallback(async () => {
    if (debugEventsRef.current.length === 0) return;
    const stoppedAt = new Date().toISOString();
    const path = `${FABLE_DEBUG_DIR}/fable_live_session_${stoppedAt.replace(/[:.]/g, '-')}.json`;
    const payload = {
      type: 'fable_live_debug_session',
      model_version: FABLE_MODEL_VERSION,
      player: setup,
      started_at: startedAtRef.current ?? stoppedAt,
      stopped_at: stoppedAt,
      engine_config: FABLE_DEFAULT_CONFIG,
      gate_config: {
        mode: 'bandpass',
        spectral_gate: false,
        abs_min_rms: FABLE_ABS_MIN_RMS,
        onset_threshold: FABLE_ONSET_THRESHOLD,
        retrigger_ms: FABLE_RETRIGGER_MS,
      },
      counts: {
        native_candidates: debugEventsRef.current.length,
        counted: debugEventsRef.current.filter(e => e.counted).length,
      },
      events: debugEventsRef.current,
    };
    try {
      await RNFS.mkdir(FABLE_DEBUG_DIR);
      await RNFS.writeFile(path, JSON.stringify(payload, null, 2), 'utf8');
      setSavedDebugPath(path);
    } catch {
      setSavedDebugPath('Kunde inte spara fable-debug.');
    }
  }, [setup]);

  const toggle = useCallback(() => {
    if (isListening) {
      AudioStream.stopStreaming();
      setIsListening(false);
      void saveDebugSession();
      return;
    }
    counterRef.current.reset();
    latenciesRef.current = [];
    debugEventsRef.current = [];
    startedAtRef.current = new Date().toISOString();
    setHitCount(0);
    setEventCount(0);
    setStaleCount(0);
    setLastHeight(null);
    heightsRef.current = [];
    setLastResult(null);
    setRecentEvents([]);
    setSavedDebugPath(null);
    void (async () => {
      await AudioStream.startStreaming(FABLE_ONSET_THRESHOLD);
      // startStreaming återställer gate-konfig till gamla beteendet,
      // så Fable-inställningarna måste sättas EFTER start.
      await AudioStream.setRetriggerMs(FABLE_RETRIGGER_MS);
      await AudioStream.setGateConfig('bandpass', false, FABLE_ABS_MIN_RMS);
      setIsListening(true);
    })();
  }, [isListening, saveDebugSession]);

  useEffect(() => {
    if (!isListening) return;

    const sub = AudioStreamEmitter.addListener('onBounceDetected', (event: NativeAudioBounceEvent) => {
      setEventCount(n => n + 1);
      const { audioB64, nativeDebug } = parseNativeEvent(event);
      const receivedAtMs = Date.now();
      const ts = new Date().toLocaleTimeString('sv-SE', { hour: '2-digit', minute: '2-digit', second: '2-digit' });

      if (!audioB64) {
        debugEventsRef.current.push({
          index: debugEventsRef.current.length + 1,
          received_at_ms: receivedAtMs,
          native_onset_time_ms: nativeDebug?.onset_time_ms,
          native_rms: nativeDebug?.rms,
          native_background_rms: nativeDebug?.background_rms,
          counted: false,
          reject_reason: nativeDebug?.native_reject_reason ?? 'native_reject',
        });
        return;
      }

      try {
        const onsetTimeMs = nativeDebug?.onset_time_ms ?? receivedAtMs;
        const frameRms = nativeDebug?.rms ?? 0;
        const pcm = decodeBase64PCM(audioB64);
        const result: FableDetectionResult = counterRef.current.process(pcm, onsetTimeMs, frameRms, receivedAtMs);

        if (result.prediction) {
          latenciesRef.current.push(result.featureMs + result.predictMs);
          if (latenciesRef.current.length % 5 === 0) {
            const sorted = [...latenciesRef.current].sort((a, b) => a - b);
            setLatencyMs({
              p50: sorted[Math.floor(sorted.length / 2)],
              max: sorted[sorted.length - 1],
            });
          }
        }
        if (result.rejectReason === 'stale_backlog') setStaleCount(n => n + 1);

        const reason = result.counted ? 'räknad' : (result.rejectReason ?? 'okänd');
        if (result.counted) setHitCount(n => n + 1);
        if (result.counted && result.bounceHeightM !== undefined && result.bounceGapMs !== undefined) {
          heightsRef.current.push(result.bounceHeightM);
          const avg = heightsRef.current.reduce((a, b) => a + b, 0) / heightsRef.current.length;
          setLastHeight({ heightM: result.bounceHeightM, gapMs: result.bounceGapMs, avgM: avg });
        }
        if (result.prediction) {
          setLastResult({
            label: result.prediction.label,
            conf: result.prediction.confidence,
            reason,
            probs: result.prediction.probabilities,
            bgMode: result.bgMode,
          });
        }
        setRecentEvents(prev => [{
          ts,
          label: result.prediction?.label ?? '-',
          confidence: result.prediction?.confidence ?? 0,
          counted: result.counted,
          reason,
        }, ...prev.slice(0, 24)]);
        debugEventsRef.current.push({
          index: debugEventsRef.current.length + 1,
          received_at_ms: receivedAtMs,
          native_onset_time_ms: nativeDebug?.onset_time_ms,
          native_rms: nativeDebug?.rms,
          native_background_rms: nativeDebug?.background_rms,
          model_label: result.prediction?.label,
          model_confidence: result.prediction?.confidence,
          model_probabilities: result.prediction?.probabilities,
          bg_mode: result.bgMode,
          bg_rms_db: result.bgRmsDb,
          counted: result.counted,
          reject_reason: result.counted ? '' : (result.rejectReason ?? 'unknown'),
          feature_ms: result.featureMs,
          predict_ms: result.predictMs,
        });
      } catch (err) {
        debugEventsRef.current.push({
          index: debugEventsRef.current.length + 1,
          received_at_ms: receivedAtMs,
          counted: false,
          reject_reason: `js_error:${String(err).slice(0, 120)}`,
        });
      }
    });

    return () => sub.remove();
  }, [isListening]);

  useEffect(() => () => { if (isListening) AudioStream.stopStreaming(); }, [isListening]);

  const probs = lastResult?.probs ?? {};

  return (
    <View style={[styles.root, { paddingTop: insets.top }]}>
      <StatusBar barStyle="light-content" backgroundColor="#000" />
      <View style={styles.header}>
        <TouchableOpacity onPress={() => { if (isListening) { AudioStream.stopStreaming(); } onDone(); }}>
          <Text style={styles.back}>‹ Tillbaka</Text>
        </TouchableOpacity>
        <Text style={styles.title}>Fable-algoritm</Text>
        <Text style={styles.subtitle}>{FABLE_MODEL_VERSION}</Text>
      </View>

      <View style={styles.counterBox}>
        <Text style={styles.counterValue}>{hitCount}</Text>
        <Text style={styles.counterLabel}>räknade racketstudsar</Text>
        <Text style={styles.eventMeta}>
          {eventCount} kandidater från gaten{staleCount > 0 ? ` · ${staleCount} släppta (kö)` : ''}
        </Text>
        {latencyMs ? (
          <Text style={styles.eventMeta}>
            JS-latens p50 {latencyMs.p50.toFixed(0)} ms / max {latencyMs.max.toFixed(0)} ms
          </Text>
        ) : null}
        {lastHeight ? (
          <Text style={styles.heightText}>
            studshöjd {(lastHeight.heightM * 100).toFixed(0)} cm ({lastHeight.gapMs.toFixed(0)} ms) · snitt {(lastHeight.avgM * 100).toFixed(0)} cm
          </Text>
        ) : null}
      </View>

      {lastResult ? (
        <View style={styles.lastBox}>
          <Text style={[styles.lastLabel, lastResult.label === 'racket_bounce' ? styles.lastRacket : styles.lastOther]}>
            {labelName(lastResult.label)} {(lastResult.conf * 100).toFixed(0)}% — {lastResult.reason}
            {lastResult.bgMode ? (lastResult.bgMode === 'loud' ? ' · LJUDLIG MILJÖ' : ' · tyst') : ''}
          </Text>
          <Text style={styles.probRow}>
            R {(probs.racket_bounce ?? 0).toFixed(2)}  B {(probs.table_bounce ?? 0).toFixed(2)}  G {(probs.floor_bounce ?? 0).toFixed(2)}  N {(probs.noise ?? 0).toFixed(2)}
          </Text>
        </View>
      ) : null}

      <TouchableOpacity style={[styles.toggle, isListening ? styles.toggleStop : styles.toggleStart]} onPress={toggle}>
        <Text style={styles.toggleText}>{isListening ? 'STOPPA' : 'STARTA'}</Text>
      </TouchableOpacity>

      <Text style={styles.configLine}>
        Gate: bandpass 1.5–7 kHz · ingen spektralgate · retrigger 120 ms{'\n'}
        Modell: HistGB 83 features · adaptiv konfidens 0.65 tyst / 0.90 ljudligt · eko-gate 300 ms/0.6
      </Text>

      {savedDebugPath ? <Text style={styles.savedPath}>Debug: {savedDebugPath}</Text> : null}

      <ScrollView style={styles.eventList}>
        {recentEvents.map((row, i) => (
          <View key={`${row.ts}-${i}`} style={styles.eventRow}>
            <Text style={[styles.eventText, row.counted ? styles.eventCounted : styles.eventIgnored]}>
              {row.ts}  {labelName(row.label)} {(row.confidence * 100).toFixed(0)}%  {row.counted ? '✓' : row.reason}
            </Text>
          </View>
        ))}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#000' },
  header: { paddingHorizontal: 16, paddingBottom: 8 },
  back: { color: '#4a9eff', fontSize: 16, paddingVertical: 6 },
  title: { color: '#fff', fontSize: 24, fontWeight: '700' },
  subtitle: { color: '#555', fontSize: 11 },
  counterBox: { alignItems: 'center', paddingVertical: 12 },
  counterValue: { color: '#2ecc71', fontSize: 72, fontWeight: '800' },
  counterLabel: { color: '#aaa', fontSize: 14 },
  eventMeta: { color: '#555', fontSize: 12, marginTop: 2 },
  heightText: { color: '#f1c40f', fontSize: 15, fontWeight: '700', marginTop: 4 },
  lastBox: { alignItems: 'center', paddingVertical: 6 },
  lastLabel: { fontSize: 18, fontWeight: '700' },
  lastRacket: { color: '#2ecc71' },
  lastOther: { color: '#e67e22' },
  probRow: { color: '#888', fontSize: 13, marginTop: 2, fontVariant: ['tabular-nums'] },
  toggle: { marginHorizontal: 24, marginVertical: 10, paddingVertical: 14, borderRadius: 10, alignItems: 'center' },
  toggleStart: { backgroundColor: '#1d6f42' },
  toggleStop: { backgroundColor: '#8e2b2b' },
  toggleText: { color: '#fff', fontSize: 18, fontWeight: '800' },
  configLine: { color: '#555', fontSize: 11, textAlign: 'center', marginBottom: 6 },
  savedPath: { color: '#4a9eff', fontSize: 10, paddingHorizontal: 16, marginBottom: 4 },
  eventList: { flex: 1, paddingHorizontal: 16 },
  eventRow: { paddingVertical: 3, borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: '#1a1a1a' },
  eventText: { fontSize: 13, fontVariant: ['tabular-nums'] },
  eventCounted: { color: '#2ecc71' },
  eventIgnored: { color: '#777' },
});
