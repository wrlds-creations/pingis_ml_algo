/**
 * LiveClassificationScreen.tsx
 *
 * Audio-only baseline for binary racket contact detection.
 * Kotlin emits 1s clips around adaptive onsets, JS runs the contact RF model
 * and counts qualified racket_contact events.
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  View, Text, TouchableOpacity, ScrollView, Pressable,
  StyleSheet, StatusBar, PanResponder,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { decodeBase64PCM } from './NativeAudioCapture';
import { AudioStream, AudioStreamEmitter } from './NativeAudioStream';
import { detectAudioContact } from './audioContactEngine';
import type { AudioDetectionEvent, PlayerSetup } from './types';

const DEFAULT_THRESHOLD = 0.020;
const THRESHOLD_MIN = 0.005;
const THRESHOLD_MAX = 0.15;

const DEFAULT_CONF = 0.65;
const CONF_MIN = 0.30;
const CONF_MAX = 0.95;
const DEFAULT_MERGE_WINDOW_MS = 260;
const CONTACT_GROUP_WINDOW_MS = 650;

const LABEL_CONFIG: Record<string, { name: string; color: string; bg: string }> = {
  racket_contact: { name: 'CONTACT', color: '#2ecc71', bg: '#0a2018' },
  not_racket_contact: { name: 'IGNORE', color: '#666666', bg: '#111111' },
};

interface Props { setup: PlayerSetup; onDone: () => void; }

interface DetectedContact {
  label: string;
  confidence: number;
  time: string;
}

interface DebugInfo {
  label: string;
  conf: number;
  reason: string;
  probs: Record<string, number>;
  surfaceLabel?: string;
  surfaceConf?: number;
  groupId?: number;
  groupStatus?: AudioDetectionEvent['group_status'];
}

interface RecentEvent {
  ts: string;
  label: string;
  confidence: number;
  surfaceLabel?: string;
  surfaceConfidence?: number;
  counted: boolean;
  reason: string;
  groupId?: number;
  groupStatus?: AudioDetectionEvent['group_status'];
}

interface ContactGroupState {
  id: number;
  startedAtMs: number;
}

function surfaceLabelName(label?: string) {
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

function reasonLabel(reason: string) {
  if (reason === 'dedup') {
    return 'merge_window';
  }
  return reason;
}

export function LiveClassificationScreen({ setup, onDone }: Props) {
  const insets = useSafeAreaInsets();
  const [isListening, setIsListening] = useState(false);
  const [lastContact, setLastContact] = useState<DetectedContact | null>(null);
  const [history, setHistory] = useState<DetectedContact[]>([]);
  const [recentEvents, setRecentEvents] = useState<RecentEvent[]>([]);
  const [confidence, setConfidence] = useState(DEFAULT_CONF);
  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD);
  const [mergeWindowMs, setMergeWindowMs] = useState(DEFAULT_MERGE_WINDOW_MS);
  const [debug, setDebug] = useState<DebugInfo | null>(null);
  const [eventCount, setEventCount] = useState(0);
  const [hitCount, setHitCount] = useState(0);
  const [showIgnored, setShowIgnored] = useState(false);

  const confRef = useRef(DEFAULT_CONF);
  confRef.current = confidence;
  const mergeWindowRef = useRef(DEFAULT_MERGE_WINDOW_MS);
  mergeWindowRef.current = mergeWindowMs;
  const lastQualifiedTsRef = useRef<number | undefined>(undefined);
  const contactGroupRef = useRef<ContactGroupState | null>(null);
  const groupCounterRef = useRef(0);

  function makeSlider(
    value: number,
    setValue: (v: number) => void,
    min: number,
    max: number,
    step: number,
  ) {
    const trackRef = useRef<View>(null);
    const trackX = useRef(0);
    const trackW = useRef(1);

    function applyX(pageX: number) {
      const ratio = Math.max(0, Math.min(1, (pageX - trackX.current) / trackW.current));
      const steps = Math.round(ratio / step) * step;
      setValue(Math.max(min, Math.min(max, parseFloat((min + steps * (max - min)).toFixed(4)))));
    }

    const pan = useRef(PanResponder.create({
      onStartShouldSetPanResponder: () => true,
      onMoveShouldSetPanResponder: (_event, gestureState) =>
        Math.abs(gestureState.dx) > 2 && Math.abs(gestureState.dx) >= Math.abs(gestureState.dy),
      onPanResponderTerminationRequest: () => false,
      onPanResponderGrant: e => applyX(e.nativeEvent.pageX),
      onPanResponderMove: e => applyX(e.nativeEvent.pageX),
      onPanResponderRelease: e => applyX(e.nativeEvent.pageX),
    })).current;

    function onLayout() {
      trackRef.current?.measureInWindow((x, _y, w) => {
        trackX.current = x;
        trackW.current = w;
      });
    }

    const fillPct = ((value - min) / (max - min)) * 100;
    return { trackRef, pan, onLayout, fillPct, applyX };
  }

  const confSlider = makeSlider(confidence, setConfidence, CONF_MIN, CONF_MAX, 0.05);
  const thrSlider = makeSlider(threshold, v => {
    setThreshold(v);
    if (isListening) AudioStream.setThreshold(v);
  }, THRESHOLD_MIN, THRESHOLD_MAX, 0.005);
  const mergeSlider = makeSlider(mergeWindowMs, setMergeWindowMs, 80, 360, 0.0714285714);

  const toggle = useCallback(() => {
    if (isListening) {
      AudioStream.stopStreaming();
      setIsListening(false);
      return;
    }

    lastQualifiedTsRef.current = undefined;
    contactGroupRef.current = null;
    groupCounterRef.current = 0;
    AudioStream.startStreaming(threshold);
    setIsListening(true);
  }, [isListening, threshold]);

  useEffect(() => {
    if (!isListening) return;

    const sub = AudioStreamEmitter.addListener('onBounceDetected', (audioB64: string) => {
      setEventCount(n => n + 1);
      try {
        const detectedAtMs = Date.now();
        const pcm = decodeBase64PCM(audioB64);
        const result = detectAudioContact({
          detectedAtMs,
          pcm,
          confidenceThreshold: confRef.current,
          dedupMs: mergeWindowRef.current,
          lastQualifiedTsMs: lastQualifiedTsRef.current,
        });

        const activeGroup = contactGroupRef.current;
        const inActiveGroup = !!activeGroup && detectedAtMs - activeGroup.startedAtMs <= CONTACT_GROUP_WINDOW_MS;
        if (result.qualified && inActiveGroup) {
          result.qualified = false;
          result.ignored_reason = 'group_duplicate';
          result.group_id = activeGroup.id;
          result.group_status = 'ignored_duplicate';
        } else if (result.qualified) {
          groupCounterRef.current += 1;
          contactGroupRef.current = { id: groupCounterRef.current, startedAtMs: detectedAtMs };
          result.group_id = groupCounterRef.current;
          result.group_status = 'best_candidate';
        } else if (inActiveGroup) {
          result.group_id = activeGroup.id;
          result.group_status = 'ignored_duplicate';
        } else {
          result.group_status = 'standalone';
        }

        const ts = new Date().toLocaleTimeString('sv-SE', {
          hour: '2-digit', minute: '2-digit', second: '2-digit',
        });

        const appendRecent = (counted: boolean, reason: string) => {
          setRecentEvents(prev => [{
            ts,
            label: result.label,
            confidence: result.confidence,
            surfaceLabel: result.surface_label,
            surfaceConfidence: result.surface_confidence,
            counted,
            reason,
            groupId: result.group_id,
            groupStatus: result.group_status,
          }, ...prev.slice(0, 29)]);
        };

        if (!result.qualified) {
          setDebug({
            label: result.label,
            conf: result.confidence,
            reason: reasonLabel(result.ignored_reason ?? 'ignored'),
            probs: result.probabilities,
            surfaceLabel: result.surface_label,
            surfaceConf: result.surface_confidence,
            groupId: result.group_id,
            groupStatus: result.group_status,
          });
          appendRecent(false, reasonLabel(result.ignored_reason ?? 'ignored'));
          return;
        }

        lastQualifiedTsRef.current = detectedAtMs;
        setDebug({
          label: result.label,
          conf: result.confidence,
          reason: 'counted',
          probs: result.probabilities,
          surfaceLabel: result.surface_label,
          surfaceConf: result.surface_confidence,
          groupId: result.group_id,
          groupStatus: result.group_status,
        });
        appendRecent(true, 'counted');
        setHitCount(n => n + 1);
        const contact: DetectedContact = {
          label: result.label,
          confidence: result.confidence,
          time: ts,
        };
        setLastContact(contact);
        setHistory(prev => [contact, ...prev.slice(0, 29)]);
      } catch {
        setDebug({
          label: 'not_racket_contact',
          conf: 0,
          reason: 'decode_error',
          probs: {},
          surfaceLabel: '-',
          surfaceConf: 0,
        });
      }
    });

    return () => sub.remove();
  }, [isListening]);

  useEffect(() => () => { AudioStream.stopStreaming(); }, []);

  const cfg = lastContact ? (LABEL_CONFIG[lastContact.label] ?? null) : null;
  const color = cfg?.color ?? '#333';
  const bg = cfg?.bg ?? '#111';
  const name = cfg?.name ?? '-';
  const THUMB = 22;

  return (
    <ScrollView
      style={[styles.root, { paddingTop: insets.top }]}
      contentContainerStyle={[styles.scrollContent, { paddingBottom: insets.bottom + 96 }]}
      showsVerticalScrollIndicator
    >
      <StatusBar barStyle="light-content" backgroundColor="#0d0d0d" />

      <View style={styles.header}>
        <TouchableOpacity onPress={onDone} style={styles.backBtn}>
          <Text style={styles.backTxt}>Back</Text>
        </TouchableOpacity>
        <View>
          <Text style={styles.headerTitle}>Studsdetektor</Text>
          <Text style={styles.headerSub}>{setup.name} · binary contact baseline</Text>
        </View>
        {isListening && (
          <View style={styles.evtBadge}>
            <Text style={styles.evtTxt}>{eventCount}</Text>
          </View>
        )}
      </View>

      <View style={[styles.predBox, { backgroundColor: bg, borderColor: isListening ? color : '#1a1a1a' }]}>
        {lastContact ? (
          <>
            <Text style={[styles.predName, { color }]}>{name}</Text>
            <Text style={[styles.predConf, { color }]}>
              {Math.round(lastContact.confidence * 100)}%
            </Text>
            <Text style={styles.predTime}>{lastContact.time}</Text>
          </>
        ) : (
          <Text style={styles.waitTxt}>
            {isListening ? 'Waiting for racket contact...' : 'Tryck AKTIVERA for att borja'}
          </Text>
        )}
      </View>

      {debug && (
        <Text style={styles.debugTxt}>
          C:{Math.round((debug.probs.racket_contact ?? 0) * 100)}%{' '}
          X:{Math.round((debug.probs.not_racket_contact ?? 0) * 100)}%{' '}
          · surf {surfaceLabelName(debug.surfaceLabel)} {Math.round((debug.surfaceConf ?? 0) * 100)}%
          · {debug.reason}
        </Text>
      )}

      <View style={styles.sliderSection}>
        <View style={styles.sliderHeader}>
          <Text style={styles.sliderLabel}>CONFIDENCE</Text>
          <Text style={styles.sliderValue}>{Math.round(confidence * 100)}%</Text>
        </View>
        <View
          ref={confSlider.trackRef}
          onLayout={confSlider.onLayout}
          style={[styles.sliderTouchArea, { marginHorizontal: THUMB / 2 }]}
          {...confSlider.pan.panHandlers}
        >
          <Pressable style={styles.sliderTouchFill} hitSlop={8} onPress={e => confSlider.applyX(e.nativeEvent.pageX)}>
            <View style={styles.sliderTrack}>
              <View style={[styles.sliderFill, { width: `${confSlider.fillPct}%` }]} />
              <View style={[styles.sliderThumb, { left: `${confSlider.fillPct}%` as any, width: THUMB, height: THUMB, borderRadius: THUMB / 2, top: -(THUMB / 2 - 3) }]} />
            </View>
          </Pressable>
        </View>
        <View style={styles.sliderHints}>
          <Text style={styles.sliderHint}>30%</Text>
          <Text style={styles.sliderHint}>95%</Text>
        </View>
      </View>

      <View style={styles.sliderSection}>
        <View style={styles.sliderHeader}>
          <Text style={styles.sliderLabel}>ONSET THRESHOLD</Text>
          <Text style={styles.sliderValue}>{threshold.toFixed(3)}</Text>
        </View>
        <View
          ref={thrSlider.trackRef}
          onLayout={thrSlider.onLayout}
          style={[styles.sliderTouchArea, { marginHorizontal: THUMB / 2 }]}
          {...thrSlider.pan.panHandlers}
        >
          <Pressable style={styles.sliderTouchFill} hitSlop={8} onPress={e => thrSlider.applyX(e.nativeEvent.pageX)}>
            <View style={styles.sliderTrack}>
              <View style={[styles.sliderFill, { width: `${thrSlider.fillPct}%` }]} />
              <View style={[styles.sliderThumb, { left: `${thrSlider.fillPct}%` as any, width: THUMB, height: THUMB, borderRadius: THUMB / 2, top: -(THUMB / 2 - 3) }]} />
            </View>
          </Pressable>
        </View>
        <View style={styles.sliderHints}>
          <Text style={styles.sliderHint}>0.005</Text>
          <Text style={styles.sliderHint}>0.150</Text>
        </View>
      </View>

      <View style={styles.sliderSection}>
        <View style={styles.sliderHeader}>
          <Text style={styles.sliderLabel}>MERGE WINDOW</Text>
          <Text style={styles.sliderValue}>{Math.round(mergeWindowMs)} ms</Text>
        </View>
        <View
          ref={mergeSlider.trackRef}
          onLayout={mergeSlider.onLayout}
          style={[styles.sliderTouchArea, { marginHorizontal: THUMB / 2 }]}
          {...mergeSlider.pan.panHandlers}
        >
          <Pressable style={styles.sliderTouchFill} hitSlop={8} onPress={e => mergeSlider.applyX(e.nativeEvent.pageX)}>
            <View style={styles.sliderTrack}>
              <View style={[styles.sliderFill, { width: `${mergeSlider.fillPct}%` }]} />
              <View style={[styles.sliderThumb, { left: `${mergeSlider.fillPct}%` as any, width: THUMB, height: THUMB, borderRadius: THUMB / 2, top: -(THUMB / 2 - 3) }]} />
            </View>
          </Pressable>
        </View>
        <View style={styles.sliderHints}>
          <Text style={styles.sliderHint}>80</Text>
          <Text style={styles.sliderHint}>360</Text>
        </View>
      </View>

      <View style={styles.counterRow}>
        <View style={styles.counterBox}>
          <Text style={styles.counterNum}>{hitCount}</Text>
          <Text style={styles.counterLabel}>CONTACTS</Text>
        </View>
        <TouchableOpacity
          style={styles.resetBtn}
          onPress={() => {
            lastQualifiedTsRef.current = undefined;
            contactGroupRef.current = null;
            groupCounterRef.current = 0;
            setHitCount(0);
            setHistory([]);
            setRecentEvents([]);
            setLastContact(null);
          }}
          activeOpacity={0.7}
        >
          <Text style={styles.resetTxt}>RESET</Text>
        </TouchableOpacity>
      </View>

      <TouchableOpacity
        style={[styles.toggleBtn, isListening ? styles.toggleOn : styles.toggleOff]}
        onPress={toggle}
        activeOpacity={0.7}
      >
        <Text style={[styles.toggleTxt, isListening ? styles.toggleTxtOn : styles.toggleTxtOff]}>
          {isListening ? 'STOP' : 'AKTIVERA'}
        </Text>
      </TouchableOpacity>

      <View style={styles.histContent}>
        <View style={styles.debugHeader}>
          <Text style={styles.histHeader}>AUDIO DEBUG</Text>
          <TouchableOpacity onPress={() => setShowIgnored(prev => !prev)}>
            <Text style={styles.debugLink}>{showIgnored ? 'Hide ignored' : 'Show ignored'}</Text>
          </TouchableOpacity>
        </View>
        <Text style={styles.debugHelp}>
          If floor bounce still counts, send me one row that shows `counted` and one row that shows `surface`.
        </Text>
        {(showIgnored ? recentEvents : recentEvents.filter(item => item.counted)).length === 0 ? (
          <Text style={styles.emptyTxt}>No audio events in current view.</Text>
        ) : (
          (showIgnored ? recentEvents : recentEvents.filter(item => item.counted)).map((item, index) => (
            <Text key={`${item.ts}-${index}`} style={styles.debugRow}>
              {item.ts} · {item.counted ? 'counted' : item.reason}
              {' | '}grp {item.groupId ?? '-'} {item.groupStatus ?? 'standalone'}
              {' | '}bin {LABEL_CONFIG[item.label]?.name ?? item.label} {Math.round(item.confidence * 100)}%
              {' | '}surf {surfaceLabelName(item.surfaceLabel)} {Math.round((item.surfaceConfidence ?? 0) * 100)}%
            </Text>
          ))
        )}
      </View>

      {history.length > 0 && (
        <View style={styles.histContent}>
          <Text style={styles.histHeader}>COUNTED CONTACTS · confidence</Text>
          {history.map((item, index) => (
            <View key={index} style={styles.histRow}>
              <View style={[styles.histDot, { backgroundColor: LABEL_CONFIG[item.label]?.color ?? '#888' }]} />
              <Text style={[styles.histLabel, { color: LABEL_CONFIG[item.label]?.color ?? '#888' }]}>
                {LABEL_CONFIG[item.label]?.name ?? item.label}
              </Text>
              <Text style={styles.histConf}>{Math.round(item.confidence * 100)}% conf</Text>
              <Text style={styles.histTime}>{item.time}</Text>
            </View>
          ))}
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#0d0d0d' },
  scrollContent: { paddingBottom: 24 },
  header: { flexDirection: 'row', alignItems: 'center', padding: 16, paddingTop: 12, gap: 12 },
  backBtn: { paddingVertical: 8, paddingRight: 12 },
  backTxt: { color: '#4a9eff', fontSize: 14 },
  headerTitle: { color: '#fff', fontSize: 18, fontWeight: '700' },
  headerSub: { color: '#555', fontSize: 12, flex: 1 },
  evtBadge: { backgroundColor: '#1a1a1a', borderRadius: 12, paddingHorizontal: 10, paddingVertical: 4 },
  evtTxt: { color: '#555', fontSize: 12, fontFamily: 'monospace' },
  predBox: {
    marginHorizontal: 16, borderRadius: 20, borderWidth: 2,
    paddingVertical: 44, alignItems: 'center', justifyContent: 'center', minHeight: 180,
  },
  predName: { fontSize: 54, fontWeight: '900', letterSpacing: 4 },
  predConf: { fontSize: 30, fontWeight: '300', marginTop: 4 },
  predTime: { fontSize: 11, color: '#555', marginTop: 6, fontFamily: 'monospace' },
  waitTxt: { color: '#333', fontSize: 15, textAlign: 'center', paddingHorizontal: 24 },
  debugTxt: { marginHorizontal: 20, marginTop: 6, fontSize: 10, color: '#444', fontFamily: 'monospace' },
  sliderSection: { marginHorizontal: 16, marginTop: 14 },
  sliderHeader: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 10 },
  sliderLabel: { color: '#444', fontSize: 10, letterSpacing: 1.5 },
  sliderValue: { color: '#fff', fontSize: 13, fontWeight: '700' },
  sliderTouchArea: { height: 34, justifyContent: 'center' },
  sliderTouchFill: { justifyContent: 'center' },
  sliderTrack: { height: 6, backgroundColor: '#1a1a1a', borderRadius: 3, justifyContent: 'center' },
  sliderFill: { position: 'absolute', left: 0, height: 6, backgroundColor: '#4a9eff', borderRadius: 3 },
  sliderThumb: { position: 'absolute', backgroundColor: '#fff', marginLeft: -11 },
  sliderHints: { flexDirection: 'row', justifyContent: 'space-between', marginTop: 5 },
  sliderHint: { color: '#2a2a2a', fontSize: 10 },
  counterRow: { flexDirection: 'row', marginHorizontal: 16, marginTop: 14, alignItems: 'center', gap: 12 },
  counterBox: { flex: 1, backgroundColor: '#111', borderRadius: 14, paddingVertical: 12, alignItems: 'center' },
  counterNum: { color: '#fff', fontSize: 48, fontWeight: '900', fontFamily: 'monospace' },
  counterLabel: { color: '#444', fontSize: 10, letterSpacing: 2, marginTop: 2 },
  resetBtn: { backgroundColor: '#1a1a1a', borderRadius: 14, paddingVertical: 20, paddingHorizontal: 20 },
  resetTxt: { color: '#555', fontSize: 11, fontWeight: '700', letterSpacing: 1 },
  toggleBtn: { marginHorizontal: 16, marginTop: 14, borderRadius: 14, paddingVertical: 18, alignItems: 'center' },
  toggleOff: { backgroundColor: '#0d2d1a' },
  toggleOn: { backgroundColor: '#2d0a0a' },
  toggleTxt: { fontWeight: '700', fontSize: 16, letterSpacing: 2 },
  toggleTxtOff: { color: '#2ecc71' },
  toggleTxtOn: { color: '#e74c3c' },
  histContent: { paddingHorizontal: 16, paddingBottom: 16, marginTop: 12 },
  debugHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  histHeader: { color: '#333', fontSize: 10, letterSpacing: 2, marginBottom: 8 },
  debugLink: { color: '#4a9eff', fontSize: 12 },
  debugHelp: { color: '#666', fontSize: 11, marginBottom: 8 },
  emptyTxt: { color: '#555', fontSize: 12 },
  debugRow: { color: '#a8a8a8', fontSize: 11, marginBottom: 8, fontFamily: 'monospace' },
  histRow: { flexDirection: 'row', alignItems: 'center', paddingVertical: 7, borderBottomWidth: 1, borderBottomColor: '#111', gap: 10 },
  histDot: { width: 8, height: 8, borderRadius: 4 },
  histLabel: { flex: 1, fontWeight: '700', fontSize: 14, letterSpacing: 1 },
  histConf: { color: '#9fd3ff', fontSize: 12, fontWeight: '700', width: 72, textAlign: 'right' },
  histTime: { color: '#444', fontSize: 11, fontFamily: 'monospace', width: 70, textAlign: 'right' },
});
