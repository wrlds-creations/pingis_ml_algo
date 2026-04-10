/**
 * AudioCollectionScreen.tsx
 *
 * Session-baserad ljudinsamling: välj studs-typ → tryck Starta → spela in fritt
 * i 10–60 sekunder → tryck Stoppa. En lång .wav-fil sparas per session.
 * Python-preprocessing kör sedan onset-detection för att hitta enskilda studsar.
 *
 * Spelar in som råa PCM WAV-filer (22 050 Hz, mono, 16-bit) via AudioCapture-modulen
 * — exakt samma pipeline som live-inferensen använder.
 */

import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
  Alert,
  StatusBar,
  PermissionsAndroid,
  Platform,
} from 'react-native';
import RNFS from 'react-native-fs';
import { AudioCapture } from './NativeAudioCapture';
import type { AudioEvent, AudioLabel, AudioSessionFile, PlayerSetup } from './types';

// ── Konstanter ────────────────────────────────────────────────────────────────

const APP_VERSION = '1.2';
const SESSION_DIR = `${RNFS.ExternalStorageDirectoryPath}/Download/pingis_sessions`;

const LABEL_CONFIG: { label: AudioLabel; title: string; sub: string; color: string; bg: string }[] = [
  { label: 'racket_bounce', title: 'RACKET',  sub: 'boll träffar racketgummi',            color: '#2ecc71', bg: '#0d2d1a' },
  { label: 'table_bounce',  title: 'BORD',    sub: 'boll studsar på bordet',              color: '#4a9eff', bg: '#0d1f33' },
  { label: 'floor_bounce',  title: 'GOLV',    sub: 'boll missar bordet, studsar på golv', color: '#e67e22', bg: '#2d1a00' },
  { label: 'noise',         title: 'BRUS',    sub: 'skrik, prat, applåder',               color: '#e74c3c', bg: '#2d0d0d' },
];

// ── Hjälp: ledigt sessions-undermappsnummer ───────────────────────────────────

async function nextSessionDir(date: string): Promise<string> {
  let n = 1;
  let dir: string;
  do {
    dir = `${SESSION_DIR}/audio_session_${date}_${String(n).padStart(3, '0')}`;
    n++;
  } while (await RNFS.exists(dir));
  return dir;
}

function formatDuration(ms: number): string {
  const s = Math.floor(ms / 1000);
  return `${s}s`;
}

// ── Props ─────────────────────────────────────────────────────────────────────

interface Props {
  setup: PlayerSetup;
  onDone: () => void;
}

// ── Komponent ─────────────────────────────────────────────────────────────────

export function AudioCollectionScreen({ setup, onDone }: Props) {
  const [selectedLabel, setSelectedLabel]   = useState<AudioLabel | null>(null);
  const [isRecording, setIsRecording]       = useState(false);
  const [elapsedMs, setElapsedMs]           = useState(0);
  const [events, setEvents]                 = useState<AudioEvent[]>([]);
  const [permissionGranted, setPermission]  = useState(false);
  const [feedback, setFeedback]             = useState<string | null>(null);

  const sessionDirRef   = useRef<string | null>(null);
  const startTimeRef    = useRef<number>(0);
  const timerRef        = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Tillstånd + sessionsmapp ──────────────────────────────────────────────

  useEffect(() => {
    (async () => {
      if (Platform.OS === 'android') {
        const result = await PermissionsAndroid.request(
          PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
          {
            title:          'Mikrofonåtkomst',
            message:        'Appen behöver mikrofonen för att spela in pingisbollsstudsar.',
            buttonPositive: 'OK',
          },
        );
        if (result !== PermissionsAndroid.RESULTS.GRANTED) {
          Alert.alert('Tillstånd saknas', 'Mikrofontillstånd behövs för att spela in.');
          return;
        }
      }
      setPermission(true);
      await RNFS.mkdir(SESSION_DIR);
      const date = new Date().toISOString().slice(0, 10);
      sessionDirRef.current = await nextSessionDir(date);
    })();

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  // ── Starta inspelning ─────────────────────────────────────────────────────

  const startRecording = useCallback(async () => {
    if (!permissionGranted || !selectedLabel || isRecording) return;

    try {
      if (sessionDirRef.current && !(await RNFS.exists(sessionDirRef.current))) {
        await RNFS.mkdir(sessionDirRef.current);
      }

      const idx      = events.filter(e => e.label === selectedLabel).length;
      const filename = `${selectedLabel}_${String(idx).padStart(3, '0')}.wav`;
      const filePath = `${sessionDirRef.current}/${filename}`;

      await AudioCapture.startSession(filePath);

      startTimeRef.current = Date.now();
      setElapsedMs(0);
      setIsRecording(true);
      setFeedback(null);

      timerRef.current = setInterval(() => {
        setElapsedMs(Date.now() - startTimeRef.current);
      }, 200);
    } catch (e: any) {
      setFeedback(`Fel vid start: ${e.message}`);
    }
  }, [permissionGranted, selectedLabel, isRecording, events]);

  // ── Stoppa inspelning ─────────────────────────────────────────────────────

  const stopRecording = useCallback(async () => {
    if (!isRecording || !selectedLabel) return;

    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    try {
      const duration_ms = await AudioCapture.stopSession() as number;
      const idx         = events.filter(e => e.label === selectedLabel).length;
      const filename    = `${selectedLabel}_${String(idx).padStart(3, '0')}.wav`;

      const event: AudioEvent = {
        label:       selectedLabel,
        recorded_at: new Date(startTimeRef.current).toISOString(),
        wav_filename: filename,
        duration_ms,
      };

      setEvents(prev => [...prev, event]);
      setFeedback(`✓ ${filename} — ${formatDuration(duration_ms)}`);
    } catch (e: any) {
      setFeedback(`Fel vid stopp: ${e.message}`);
    } finally {
      setIsRecording(false);
      setElapsedMs(0);
    }
  }, [isRecording, selectedLabel, events]);

  // ── Spara session ─────────────────────────────────────────────────────────

  const saveSession = useCallback(async () => {
    if (events.length === 0) {
      Alert.alert('Ingen data', 'Spela in minst en session först.');
      return;
    }
    if (!sessionDirRef.current) return;

    try {
      const folder   = sessionDirRef.current;
      const jsonName = `${folder.split('/').pop()}.json`;
      const jsonPath = `${SESSION_DIR}/${jsonName}`;

      const sessionData: AudioSessionFile = {
        session_meta: {
          recorder_name:    setup.name,
          session_date:     new Date().toISOString(),
          app_version:      APP_VERSION,
          clip_duration_ms: 0,   // 0 = session-läge, Python klipper via onset-detection
        },
        events,
      };

      await RNFS.writeFile(jsonPath, JSON.stringify(sessionData, null, 2), 'utf8');
      try { await RNFS.scanFile(jsonPath); } catch (_) {}
      for (const ev of events) {
        try { await RNFS.scanFile(`${folder}/${ev.wav_filename}`); } catch (_) {}
      }

      const counts = events.reduce<Partial<Record<AudioLabel, number>>>(
        (acc, e) => ({ ...acc, [e.label]: (acc[e.label] ?? 0) + 1 }),
        {},
      );
      const totalSec = Math.round(events.reduce((s, e) => s + e.duration_ms, 0) / 1000);

      Alert.alert(
        '✓ Session sparad',
        `${events.length} inspelningar  ·  ${totalSec}s totalt\n` +
        `racket: ${counts.racket_bounce ?? 0}  bord: ${counts.table_bounce ?? 0}  ` +
        `golv: ${counts.floor_bounce ?? 0}  brus: ${counts.noise ?? 0}\n\n` +
        `Fil: Download/pingis_sessions/${jsonName}`,
        [
          { text: 'Ny session', onPress: onDone },
          {
            text: 'Fortsätt',
            onPress: async () => {
              const date = new Date().toISOString().slice(0, 10);
              sessionDirRef.current = await nextSessionDir(date);
              setEvents([]);
              setFeedback(null);
            },
          },
        ],
      );
    } catch (e: any) {
      Alert.alert('Fel', `Kunde inte spara: ${e.message}`);
    }
  }, [events, setup, onDone]);

  // ── Render ────────────────────────────────────────────────────────────────

  const counts = events.reduce<Partial<Record<AudioLabel, number>>>(
    (acc, e) => ({ ...acc, [e.label]: (acc[e.label] ?? 0) + 1 }),
    {},
  );

  const canRecord = permissionGranted && selectedLabel !== null && !isRecording;

  return (
    <ScrollView style={s.root} contentContainerStyle={s.content}>
      <StatusBar barStyle="light-content" backgroundColor="#0d0d0d" />

      {/* Header */}
      <View style={s.header}>
        <View>
          <Text style={s.playerName}>{setup.name}</Text>
          <Text style={s.playerMeta}>Ljud-insamling  ·  Session-läge</Text>
        </View>
        {isRecording && (
          <View style={s.recBadge}>
            <Text style={s.recBadgeTxt}>● REC  {formatDuration(elapsedMs)}</Text>
          </View>
        )}
      </View>

      {!permissionGranted && (
        <View style={s.warnBox}>
          <Text style={s.warnTxt}>⚠ Mikrofontillstånd saknas — bevilja i enhetsinställningarna.</Text>
        </View>
      )}

      {feedback && <Text style={s.feedbackTxt}>{feedback}</Text>}

      {/* Steg 1: Välj typ */}
      <Text style={s.sectionLabel}>1. VÄLJ TYP</Text>
      <View style={s.labelGrid}>
        {LABEL_CONFIG.map(({ label, title, sub, color, bg }) => (
          <TouchableOpacity
            key={label}
            style={[s.labelBtn, { backgroundColor: bg }, selectedLabel === label && { borderColor: color, borderWidth: 2 }]}
            onPress={() => !isRecording && setSelectedLabel(label)}
            disabled={isRecording}
            activeOpacity={0.7}
          >
            <Text style={[s.labelBtnTitle, { color }]}>{title}</Text>
            <Text style={s.labelBtnSub}>{sub}</Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* Steg 2: Spela in */}
      <Text style={s.sectionLabel}>2. SPELA IN</Text>

      {!isRecording ? (
        <TouchableOpacity
          style={[s.recordBtn, !canRecord && s.recordBtnOff]}
          onPress={startRecording}
          disabled={!canRecord}
          activeOpacity={0.7}
        >
          <Text style={[s.recordBtnTxt, !canRecord && s.recordBtnTxtOff]}>● STARTA INSPELNING</Text>
          <Text style={s.recordBtnSub}>
            {selectedLabel
              ? `Studsa bollen — tryck Stoppa när du är klar`
              : `Välj typ ovan först`}
          </Text>
        </TouchableOpacity>
      ) : (
        <TouchableOpacity
          style={[s.recordBtn, s.stopBtn]}
          onPress={stopRecording}
          activeOpacity={0.7}
        >
          <Text style={s.stopBtnTxt}>■ STOPPA  {formatDuration(elapsedMs)}</Text>
          <Text style={s.recordBtnSub}>tryck när du är klar</Text>
        </TouchableOpacity>
      )}

      {/* Statistik */}
      <View style={s.statsBox}>
        <Text style={s.statsTitle}>Session — {events.length} inspelningar</Text>
        <View style={s.statsRow}>
          <Stat label="RACKET" value={counts.racket_bounce ?? 0} color="#2ecc71" />
          <Stat label="BORD"   value={counts.table_bounce  ?? 0} color="#4a9eff" />
          <Stat label="GOLV"   value={counts.floor_bounce  ?? 0} color="#e67e22" />
          <Stat label="BRUS"   value={counts.noise         ?? 0} color="#e74c3c" />
        </View>
        {events.length > 0 && (
          <View style={s.eventList}>
            {events.slice(-5).reverse().map((ev, i) => {
              const cfg = LABEL_CONFIG.find(c => c.label === ev.label)!;
              return (
                <Text key={i} style={[s.eventRow, { color: cfg.color }]}>
                  {cfg.title}  ·  {formatDuration(ev.duration_ms)}  ·  {ev.wav_filename}
                </Text>
              );
            })}
          </View>
        )}
        {events.length === 0 && (
          <Text style={s.statsHint}>Mål: minst 3–5 inspelningar per typ (10–30 s vardera)</Text>
        )}
      </View>

      {/* Spara / Rensa */}
      <View style={s.row}>
        <TouchableOpacity style={[s.secBtn, s.saveBtn]} onPress={saveSession}>
          <Text style={s.saveBtnTxt}>Spara session</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[s.secBtn, s.clearBtn]}
          onPress={() =>
            Alert.alert('Rensa?', 'Listan rensas (inspelade filer behålls).', [
              { text: 'Avbryt' },
              { text: 'Rensa', style: 'destructive', onPress: () => setEvents([]) },
            ])
          }
        >
          <Text style={s.clearBtnTxt}>Rensa</Text>
        </TouchableOpacity>
      </View>

      {/* Info */}
      <View style={s.fileBox}>
        <Text style={s.fileTit}>Hur fungerar det?</Text>
        <Text style={s.fileTxt}>
          Välj typ → Starta → studsa bollen naturligt i 10–60 s → Stoppa.{'\n'}
          Sparas som WAV (rå PCM) — samma format som live-klassificeringen.{'\n'}
          Python hittar automatiskt varje studs i inspelningen (onset-detection).{'\n\n'}
          Filer sparas i: Downloads → pingis_sessions/
        </Text>
      </View>
    </ScrollView>
  );
}

function Stat({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <View style={s.statItem}>
      <Text style={[s.statValue, { color }]}>{value}</Text>
      <Text style={s.statLabel}>{label}</Text>
    </View>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────

const s = StyleSheet.create({
  root:    { flex: 1, backgroundColor: '#0d0d0d' },
  content: { padding: 20, paddingBottom: 50 },

  header:      { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 },
  playerName:  { color: '#fff', fontSize: 20, fontWeight: '700' },
  playerMeta:  { color: '#777', fontSize: 12, marginTop: 2 },
  recBadge:    { backgroundColor: '#2d0d0d', borderRadius: 8, paddingHorizontal: 12, paddingVertical: 6 },
  recBadgeTxt: { color: '#e74c3c', fontSize: 13, fontWeight: '700' },

  warnBox:     { backgroundColor: '#2d1a00', borderRadius: 8, padding: 12, marginBottom: 12 },
  warnTxt:     { color: '#e67e22', fontSize: 13 },
  feedbackTxt: { color: '#4a9eff', fontSize: 13, textAlign: 'center', marginBottom: 10 },

  sectionLabel: { color: '#666', fontSize: 10, letterSpacing: 2, marginTop: 18, marginBottom: 8 },
  row:          { flexDirection: 'row', gap: 12 },

  labelGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 10 },
  labelBtn: {
    width: '47%',
    borderRadius: 12,
    padding: 16,
    borderWidth: 1,
    borderColor: '#222',
    alignItems: 'center',
  },
  labelBtnTitle: { fontSize: 18, fontWeight: '800', letterSpacing: 2, marginBottom: 4 },
  labelBtnSub:   { color: '#666', fontSize: 11, textAlign: 'center' },

  recordBtn: {
    borderRadius: 14,
    padding: 28,
    marginTop: 4,
    alignItems: 'center',
    backgroundColor: '#0d2d0d',
  },
  recordBtnOff:    { backgroundColor: '#111' },
  recordBtnTxt:    { color: '#2ecc71', fontSize: 20, fontWeight: '800', letterSpacing: 2 },
  recordBtnTxtOff: { color: '#2a2a2a' },
  recordBtnSub:    { color: '#666', fontSize: 12, marginTop: 6 },

  stopBtn:    { backgroundColor: '#2d0d0d' },
  stopBtnTxt: { color: '#e74c3c', fontSize: 20, fontWeight: '800', letterSpacing: 2 },

  statsBox:   { backgroundColor: '#111', borderRadius: 12, padding: 16, marginTop: 16 },
  statsTitle: { color: '#aaa', fontWeight: '600', marginBottom: 12 },
  statsRow:   { flexDirection: 'row', justifyContent: 'space-around' },
  statsHint:  { color: '#666', fontSize: 12, fontStyle: 'italic', marginTop: 10, textAlign: 'center' },
  statItem:   { alignItems: 'center' },
  statValue:  { fontSize: 28, fontWeight: '800' },
  statLabel:  { color: '#777', fontSize: 11, marginTop: 2 },

  eventList: { marginTop: 12, borderTopWidth: 1, borderTopColor: '#222', paddingTop: 10 },
  eventRow:  { fontSize: 11, marginBottom: 3, fontFamily: 'monospace' },

  secBtn:      { flex: 1, padding: 14, borderRadius: 8, alignItems: 'center', marginTop: 12 },
  saveBtn:     { backgroundColor: '#0d2d0d' },
  clearBtn:    { backgroundColor: '#2d0d0d' },
  saveBtnTxt:  { color: '#2ecc71', fontWeight: '700' },
  clearBtnTxt: { color: '#e74c3c', fontWeight: '700' },

  fileBox:  { marginTop: 20, borderWidth: 1, borderColor: '#222', borderRadius: 10, padding: 14 },
  fileTit:  { color: '#888', fontWeight: '600', marginBottom: 6 },
  fileTxt:  { color: '#666', fontSize: 12, lineHeight: 18 },
});
