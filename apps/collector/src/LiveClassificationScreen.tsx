/**
 * LiveClassificationScreen.tsx
 *
 * Studs-detektor som körs helt on-device utan nätverksanrop.
 *
 * Flöde:
 *   AudioStreamModule (Kotlin) kör AudioRecord kontinuerligt med en
 *   ring-buffer. När ett energispiket detekteras extraheras ett 1s-klipp
 *   (300ms pre + 700ms post onset) och skickas som händelsen
 *   "onBounceDetected". JS tar emot, kör MFCC + RF och visar resultatet.
 *
 * Varje studs triggar oberoende — klarar 3+ studsar/sek utan att missa något.
 */

import React, {
  useState, useRef, useCallback, useEffect,
} from 'react';
import {
  View, Text, TouchableOpacity, ScrollView,
  StyleSheet, StatusBar, PanResponder,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { decodeBase64PCM }                      from './NativeAudioCapture';
import { AudioStream, AudioStreamEmitter }      from './NativeAudioStream';
import { extractFeatures }                      from './audioFeatures';
import { rfPredict }                            from './rfInference';
import type { PlayerSetup }                     from './types';

// ── Konstanter ─────────────────────────────────────────────────────────────────

/** Energitröskel som skickas till Kotlin-modulen (RMS 0–1). */
const DEFAULT_THRESHOLD = 0.025;
const THRESHOLD_MIN     = 0.005;
const THRESHOLD_MAX     = 0.15;

/** Minsta RF-konfidens för att visa resultatet. */
const DEFAULT_CONF      = 0.75;
const CONF_MIN          = 0.30;
const CONF_MAX          = 0.95;

// ── Label-konfiguration ────────────────────────────────────────────────────────

const LABEL_CONFIG: Record<string, { name: string; color: string; bg: string }> = {
  racket_bounce: { name: 'RACKET', color: '#2ecc71', bg: '#0a2018' },
  table_bounce:  { name: 'BORD',   color: '#4a9eff', bg: '#0a1528' },
  floor_bounce:  { name: 'GOLV',   color: '#e67e22', bg: '#281500' },
};

// ── Typer ──────────────────────────────────────────────────────────────────────

interface Props { setup: PlayerSetup; onDone: () => void; }

interface DetectedBounce { label: string; confidence: number; time: string; }

interface DebugInfo { label: string; conf: number; reason: string; probs: Record<string, number>; }

// ── Komponent ──────────────────────────────────────────────────────────────────

export function LiveClassificationScreen({ setup, onDone }: Props) {
  const insets = useSafeAreaInsets();
  const [isListening,  setIsListening]  = useState(false);
  const [lastBounce,   setLastBounce]   = useState<DetectedBounce | null>(null);
  const [history,      setHistory]      = useState<DetectedBounce[]>([]);
  const [confidence,   setConfidence]   = useState(DEFAULT_CONF);
  const [threshold,    setThreshold]    = useState(DEFAULT_THRESHOLD);
  const [debug,        setDebug]        = useState<DebugInfo | null>(null);
  const [eventCount,   setEventCount]   = useState(0);
  const [hitCount,     setHitCount]     = useState(0);

  const confRef      = useRef(DEFAULT_CONF);
  confRef.current    = confidence;

  // ── Slider helpers ────────────────────────────────────────────────────────────

  function makeSlider(
    value: number,
    setValue: (v: number) => void,
    min: number,
    max: number,
    step: number,
  ) {
    const trackRef  = useRef<View>(null);
    const trackX    = useRef(0);
    const trackW    = useRef(1);

    function applyX(pageX: number) {
      const ratio  = Math.max(0, Math.min(1, (pageX - trackX.current) / trackW.current));
      const steps  = Math.round(ratio / step) * step;
      setValue(Math.max(min, Math.min(max, parseFloat((min + steps * (max - min)).toFixed(4)))));
    }

    const pan = useRef(PanResponder.create({
      onStartShouldSetPanResponder: () => true,
      onMoveShouldSetPanResponder:  () => true,
      onPanResponderGrant: e => applyX(e.nativeEvent.pageX),
      onPanResponderMove:  e => applyX(e.nativeEvent.pageX),
    })).current;

    function onLayout() {
      trackRef.current?.measureInWindow((x, _y, w) => {
        trackX.current = x;
        trackW.current = w;
      });
    }

    const fillPct = ((value - min) / (max - min)) * 100;
    return { trackRef, pan, onLayout, fillPct };
  }

  const confSlider = makeSlider(confidence, setConfidence, CONF_MIN, CONF_MAX, 0.05);
  const thrSlider  = makeSlider(threshold,  v => {
    setThreshold(v);
    if (isListening) AudioStream.setThreshold(v);
  }, THRESHOLD_MIN, THRESHOLD_MAX, 0.005);

  // ── Start / stopp ─────────────────────────────────────────────────────────────

  const toggle = useCallback(() => {
    if (isListening) {
      AudioStream.stopStreaming();
      setIsListening(false);
    } else {
      AudioStream.startStreaming(threshold);
      setIsListening(true);
    }
  }, [isListening, threshold]);

  // ── Event-prenumeration ───────────────────────────────────────────────────────

  useEffect(() => {
    if (!isListening) return;

    const sub = AudioStreamEmitter.addListener('onBounceDetected', (audioB64: string) => {
      setEventCount(n => n + 1);
      try {
        const pcm    = decodeBase64PCM(audioB64);
        const feats  = extractFeatures(pcm);
        const result = rfPredict(feats);

        const probs = result.probabilities;
        if (result.label === 'noise') {
          setDebug({ label: result.label, conf: result.confidence, reason: 'brus', probs });
          return;
        }
        if (result.confidence < confRef.current) {
          setDebug({ label: result.label, conf: result.confidence, reason: 'låg konfidens', probs });
          return;
        }

        setDebug({ label: result.label, conf: result.confidence, reason: 'visas', probs });
        setHitCount(n => n + 1);
        const bounce: DetectedBounce = {
          label:      result.label,
          confidence: result.confidence,
          time: new Date().toLocaleTimeString('sv-SE', {
            hour: '2-digit', minute: '2-digit', second: '2-digit',
          }),
        };
        setLastBounce(bounce);
        setHistory(prev => [bounce, ...prev.slice(0, 29)]);
      } catch { /* ignorera decode/feature-fel */ }
    });

    return () => sub.remove();
  }, [isListening]);

  useEffect(() => () => { AudioStream.stopStreaming(); }, []);

  // ── Render ────────────────────────────────────────────────────────────────────

  const cfg   = lastBounce ? (LABEL_CONFIG[lastBounce.label] ?? null) : null;
  const color = cfg?.color ?? '#333';
  const bg    = cfg?.bg    ?? '#111';
  const name  = cfg?.name  ?? '—';

  const THUMB = 22;

  return (
    <View style={[s.root, { paddingTop: insets.top }]}>
      <StatusBar barStyle="light-content" backgroundColor="#0d0d0d" />

      {/* Header */}
      <View style={s.header}>
        <TouchableOpacity onPress={onDone} style={s.backBtn}>
          <Text style={s.backTxt}>← Tillbaka</Text>
        </TouchableOpacity>
        <View>
          <Text style={s.headerTitle}>Studs-detektor</Text>
          <Text style={s.headerSub}>{setup.name} · on-device</Text>
        </View>
        {isListening && (
          <View style={s.evtBadge}>
            <Text style={s.evtTxt}>{eventCount}</Text>
          </View>
        )}
      </View>

      {/* Studs-display */}
      <View style={[s.predBox, { backgroundColor: bg, borderColor: isListening ? color : '#1a1a1a' }]}>
        {lastBounce ? (
          <>
            <Text style={[s.predName, { color }]}>{name}</Text>
            <Text style={[s.predConf, { color }]}>
              {Math.round(lastBounce.confidence * 100)}%
            </Text>
            <Text style={s.predTime}>{lastBounce.time}</Text>
          </>
        ) : (
          <Text style={s.waitTxt}>
            {isListening ? 'Väntar på studs...' : 'Tryck AKTIVERA för att börja'}
          </Text>
        )}
      </View>

      {/* Debug */}
      {debug && (
        <Text style={s.debugTxt}>
          R:{Math.round((debug.probs.racket_bounce ?? 0) * 100)}%{' '}
          G:{Math.round((debug.probs.floor_bounce ?? 0) * 100)}%{' '}
          B:{Math.round((debug.probs.noise ?? 0) * 100)}%{' '}
          · {debug.reason}
        </Text>
      )}

      {/* Konfidens-slider */}
      <View style={s.sliderSection}>
        <View style={s.sliderHeader}>
          <Text style={s.sliderLabel}>KONFIDENS-TRÖSKEL</Text>
          <Text style={s.sliderValue}>{Math.round(confidence * 100)}%</Text>
        </View>
        <View
          ref={confSlider.trackRef}
          onLayout={confSlider.onLayout}
          style={[s.sliderTrack, { marginHorizontal: THUMB / 2 }]}
          {...confSlider.pan.panHandlers}
        >
          <View style={[s.sliderFill, { width: `${confSlider.fillPct}%` }]} />
          <View style={[s.sliderThumb, { left: `${confSlider.fillPct}%` as any, width: THUMB, height: THUMB, borderRadius: THUMB / 2, top: -(THUMB / 2 - 3) }]} />
        </View>
        <View style={s.sliderHints}>
          <Text style={s.sliderHint}>30%</Text>
          <Text style={s.sliderHint}>95%</Text>
        </View>
      </View>

      {/* Energi-tröskel-slider */}
      <View style={s.sliderSection}>
        <View style={s.sliderHeader}>
          <Text style={s.sliderLabel}>KÄNSLIGHET (adaptiv onset)</Text>
          <Text style={s.sliderValue}>{threshold.toFixed(3)}</Text>
        </View>
        <View
          ref={thrSlider.trackRef}
          onLayout={thrSlider.onLayout}
          style={[s.sliderTrack, { marginHorizontal: THUMB / 2 }]}
          {...thrSlider.pan.panHandlers}
        >
          <View style={[s.sliderFill, { width: `${thrSlider.fillPct}%` }]} />
          <View style={[s.sliderThumb, { left: `${thrSlider.fillPct}%` as any, width: THUMB, height: THUMB, borderRadius: THUMB / 2, top: -(THUMB / 2 - 3) }]} />
        </View>
        <View style={s.sliderHints}>
          <Text style={s.sliderHint}>känslig (0.005)</Text>
          <Text style={s.sliderHint}>strikt (0.15)</Text>
        </View>
      </View>

      {/* Räknare */}
      <View style={s.counterRow}>
        <View style={s.counterBox}>
          <Text style={s.counterNum}>{hitCount}</Text>
          <Text style={s.counterLabel}>TRÄFFAR</Text>
        </View>
        <TouchableOpacity
          style={s.resetBtn}
          onPress={() => { setHitCount(0); setHistory([]); setLastBounce(null); }}
          activeOpacity={0.7}
        >
          <Text style={s.resetTxt}>NOLLSTÄLL</Text>
        </TouchableOpacity>
      </View>

      {/* Toggle */}
      <TouchableOpacity
        style={[s.toggleBtn, isListening ? s.toggleOn : s.toggleOff]}
        onPress={toggle}
        activeOpacity={0.7}
      >
        <Text style={[s.toggleTxt, isListening ? s.toggleTxtOn : s.toggleTxtOff]}>
          {isListening ? '■  STÄNG AV' : '●  AKTIVERA'}
        </Text>
      </TouchableOpacity>

      {/* Historik */}
      {history.length > 0 && (
        <ScrollView style={s.histList} contentContainerStyle={s.histContent}>
          <Text style={s.histHeader}>DETEKTERADE STUDSAR</Text>
          {history.map((h, i) => {
            const hc = LABEL_CONFIG[h.label];
            return (
              <View key={i} style={s.histRow}>
                <View style={[s.histDot, { backgroundColor: hc?.color ?? '#888' }]} />
                <Text style={[s.histLabel, { color: hc?.color ?? '#888' }]}>
                  {hc?.name ?? h.label}
                </Text>
                <Text style={s.histConf}>{Math.round(h.confidence * 100)}%</Text>
                <Text style={s.histTime}>{h.time}</Text>
              </View>
            );
          })}
        </ScrollView>
      )}
    </View>
  );
}

// ── Stilar ─────────────────────────────────────────────────────────────────────

const s = StyleSheet.create({
  root:         { flex: 1, backgroundColor: '#0d0d0d' },

  header:       { flexDirection: 'row', alignItems: 'center', padding: 16, paddingTop: 12, gap: 12 },
  backBtn:      { paddingVertical: 8, paddingRight: 12 },
  backTxt:      { color: '#4a9eff', fontSize: 14 },
  headerTitle:  { color: '#fff', fontSize: 18, fontWeight: '700' },
  headerSub:    { color: '#555', fontSize: 12, flex: 1 },
  evtBadge:     { backgroundColor: '#1a1a1a', borderRadius: 12, paddingHorizontal: 10, paddingVertical: 4 },
  evtTxt:       { color: '#555', fontSize: 12, fontFamily: 'monospace' },

  predBox: {
    marginHorizontal: 16, borderRadius: 20, borderWidth: 2,
    paddingVertical: 44, alignItems: 'center', justifyContent: 'center', minHeight: 180,
  },
  predName:  { fontSize: 60, fontWeight: '900', letterSpacing: 4 },
  predConf:  { fontSize: 30, fontWeight: '300', marginTop: 4 },
  predTime:  { fontSize: 11, color: '#555', marginTop: 6, fontFamily: 'monospace' },
  waitTxt:   { color: '#333', fontSize: 15, textAlign: 'center', paddingHorizontal: 24 },

  debugTxt:  { marginHorizontal: 20, marginTop: 6, fontSize: 10, color: '#444', fontFamily: 'monospace' },

  sliderSection: { marginHorizontal: 16, marginTop: 14 },
  sliderHeader:  { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 10 },
  sliderLabel:   { color: '#444', fontSize: 10, letterSpacing: 1.5 },
  sliderValue:   { color: '#fff', fontSize: 13, fontWeight: '700' },
  sliderTrack:   { height: 6, backgroundColor: '#1a1a1a', borderRadius: 3, justifyContent: 'center' },
  sliderFill:    { position: 'absolute', left: 0, height: 6, backgroundColor: '#4a9eff', borderRadius: 3 },
  sliderThumb:   { position: 'absolute', backgroundColor: '#fff', marginLeft: -11 },
  sliderHints:   { flexDirection: 'row', justifyContent: 'space-between', marginTop: 5 },
  sliderHint:    { color: '#2a2a2a', fontSize: 10 },

  counterRow:   { flexDirection: 'row', marginHorizontal: 16, marginTop: 14, alignItems: 'center', gap: 12 },
  counterBox:   { flex: 1, backgroundColor: '#111', borderRadius: 14, paddingVertical: 12, alignItems: 'center' },
  counterNum:   { color: '#fff', fontSize: 48, fontWeight: '900', fontFamily: 'monospace' },
  counterLabel: { color: '#444', fontSize: 10, letterSpacing: 2, marginTop: 2 },
  resetBtn:     { backgroundColor: '#1a1a1a', borderRadius: 14, paddingVertical: 20, paddingHorizontal: 20 },
  resetTxt:     { color: '#555', fontSize: 11, fontWeight: '700', letterSpacing: 1 },

  toggleBtn:    { marginHorizontal: 16, marginTop: 14, borderRadius: 14, paddingVertical: 18, alignItems: 'center' },
  toggleOff:    { backgroundColor: '#0d2d1a' },
  toggleOn:     { backgroundColor: '#2d0a0a' },
  toggleTxt:    { fontWeight: '700', fontSize: 16, letterSpacing: 2 },
  toggleTxtOff: { color: '#2ecc71' },
  toggleTxtOn:  { color: '#e74c3c' },

  histList:    { flex: 1, marginTop: 12 },
  histContent: { paddingHorizontal: 16, paddingBottom: 16 },
  histHeader:  { color: '#333', fontSize: 10, letterSpacing: 2, marginBottom: 8 },
  histRow:     { flexDirection: 'row', alignItems: 'center', paddingVertical: 7, borderBottomWidth: 1, borderBottomColor: '#111', gap: 10 },
  histDot:     { width: 8, height: 8, borderRadius: 4 },
  histLabel:   { flex: 1, fontWeight: '700', fontSize: 14, letterSpacing: 1 },
  histConf:    { color: '#555', fontSize: 12, width: 36, textAlign: 'right' },
  histTime:    { color: '#444', fontSize: 11, fontFamily: 'monospace', width: 70, textAlign: 'right' },
});
