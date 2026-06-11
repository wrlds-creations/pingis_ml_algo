/**
 * BounceSideLiveScreen.tsx
 *
 * "Studs FH/BH LIVE": kameran igång, ljuddetektorn räknar studsar i
 * realtid och kameran avgör vilken racketsida bollen studsar på.
 *
 * Kedjan per studs:
 *   ljud (adaptiv gate + Fable-modellen, bollträff-villkor)
 *   -> captureCrop(): senaste kameraframe -> MediaPipe-pose ->
 *      handleds-ankrad racket-crop 64x64
 *   -> sidomodellen (grid-färgfeatures, tränad på Loves sessioner)
 *   -> FH-/BH-räknare uppdateras.
 *
 * Rekommenderad uppställning (Loves design): mobilen lutad mot något på
 * bordet, snett underifrån, så kameran ser racketens undersida vid träff.
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  PermissionsAndroid, Platform, StyleSheet, StatusBar, Text, TouchableOpacity, View,
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
import { BounceSideCameraView, BounceSideLive } from './NativeBounceSideLive';
import { FableCounter } from './fableEngine';
import { bounceSideFeatures, predictBounceSide, BOUNCE_SIDE_MODEL_VERSION } from './bounceSideInference';
import type { PlayerSetup } from './types';

const ONSET_THRESHOLD = 0.005; // -> native onset-ratio 1.5
const RETRIGGER_MS = 120;
const ABS_MIN_RMS = 0.0015;
/** Under denna sidokonfidens räknas studsen som Osäker i stället för
 *  att gissas till fel sida. */
const SIDE_MIN_CONFIDENCE = 0.6;

interface LiveDebugEvent {
  onset_time_ms: number;
  side: string;
  confidence: number;
  probabilities: Record<string, number>;
  roi_source: string;
  frame_delay_ms: number;
  audio_label: string;
  audio_confidence: number;
  rgb_b64: string;
}

interface Props { setup: PlayerSetup; onDone: () => void; }

function parseNativeEvent(event: NativeAudioBounceEvent): {
  audioB64?: string;
  nativeDebug?: NativeAudioOnsetDebug;
} {
  if (typeof event === 'string') return { audioB64: event };
  return { audioB64: event.audio_b64 ?? undefined, nativeDebug: event.native_debug };
}

export function BounceSideLiveScreen({ setup, onDone }: Props) {
  const insets = useSafeAreaInsets();
  const [isRunning, setIsRunning] = useState(false);
  const [fhCount, setFhCount] = useState(0);
  const [bhCount, setBhCount] = useState(0);
  const [uncertainCount, setUncertainCount] = useState(0);
  const [lastSide, setLastSide] = useState<{ side: 'forehand' | 'backhand' | 'uncertain'; confidence: number; roi: string } | null>(null);
  const [forehandColor, setForehandColor] = useState<'red' | 'black'>('red');
  const [statusText, setStatusText] = useState('Luta mobilen mot något på bordet så kameran ser racketen snett underifrån.');

  const counterRef = useRef(new FableCounter());
  const forehandColorRef = useRef<'red' | 'black'>('red');
  forehandColorRef.current = forehandColor;
  const busyRef = useRef(false);
  const debugEventsRef = useRef<LiveDebugEvent[]>([]);

  const stopAll = useCallback(() => {
    AudioStream.stopStreaming();
    void BounceSideLive.stopCamera();
    setIsRunning(false);
    // Debug-dump: varje räknad studs med crop + beslut, så felanalys kan
    // göras på fakta i stället för teorier. Hämtas via adb från Download.
    const events = debugEventsRef.current;
    if (events.length > 0) {
      const path = `${RNFS.DownloadDirectoryPath}/pingis_live_sidedebug_${Date.now()}.json`;
      RNFS.writeFile(path, JSON.stringify({ model: BOUNCE_SIDE_MODEL_VERSION, events }), 'utf8')
        .then(() => RNFS.scanFile(path))
        .catch(() => {});
      debugEventsRef.current = [];
    }
  }, []);

  const toggle = useCallback(() => {
    if (isRunning) {
      stopAll();
      setStatusText('Stoppad.');
      return;
    }
    void (async () => {
      try {
        if (Platform.OS === 'android') {
          const granted = await PermissionsAndroid.requestMultiple([
            PermissionsAndroid.PERMISSIONS.CAMERA,
            PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
          ]);
          if (granted['android.permission.CAMERA'] !== 'granted'
            || granted['android.permission.RECORD_AUDIO'] !== 'granted') {
            setStatusText('Kamera- och mikrofontillstånd krävs.');
            return;
          }
        }
        counterRef.current.reset();
        debugEventsRef.current = [];
        setFhCount(0);
        setBhCount(0);
        setUncertainCount(0);
        setLastSide(null);
        await BounceSideLive.startCamera(true);
        await AudioStream.startStreaming(ONSET_THRESHOLD);
        await AudioStream.setRetriggerMs(RETRIGGER_MS);
        await AudioStream.setGateConfig('bandpass', false, ABS_MIN_RMS);
        setIsRunning(true);
        setStatusText('Lyssnar och tittar. Studsa bollen på racketen!');
      } catch (error) {
        setStatusText(`Kunde inte starta: ${String((error as Error)?.message ?? error)}`);
      }
    })();
  }, [isRunning, stopAll]);

  useEffect(() => {
    if (!isRunning) return;

    const sub = AudioStreamEmitter.addListener('onBounceDetected', (event: NativeAudioBounceEvent) => {
      const { audioB64, nativeDebug } = parseNativeEvent(event);
      if (!audioB64 || busyRef.current) return;
      busyRef.current = true;
      void (async () => {
        try {
          const onsetTimeMs = nativeDebug?.onset_time_ms ?? Date.now();
          const frameRms = nativeDebug?.rms ?? 0;
          const pcm = decodeBase64PCM(audioB64);
          const result = counterRef.current.process(pcm, onsetTimeMs, frameRms, Date.now());
          if (!result.counted) return;

          // Bilden från TRÄFFÖGONBLICKET (ringbufferten i nativemodulen),
          // inte från "nu" - JS-kedjan hinner ta 200-300 ms.
          const crop = await BounceSideLive.captureCrop(onsetTimeMs);
          const binary = atob(crop.rgb_b64);
          const rgb = new Uint8Array(binary.length);
          for (let i = 0; i < binary.length; i += 1) rgb[i] = binary.charCodeAt(i);
          const prediction = predictBounceSide(bounceSideFeatures(rgb, crop.roi_source));
          const mapped = forehandColorRef.current === 'black'
            ? (prediction.label === 'forehand' ? 'backhand' as const : 'forehand' as const)
            : prediction.label;
          const side = prediction.confidence >= SIDE_MIN_CONFIDENCE ? mapped : 'uncertain' as const;
          if (side === 'forehand') setFhCount(n => n + 1);
          else if (side === 'backhand') setBhCount(n => n + 1);
          else setUncertainCount(n => n + 1);
          setLastSide({ side, confidence: prediction.confidence, roi: crop.roi_source });
          if (debugEventsRef.current.length < 300) {
            debugEventsRef.current.push({
              onset_time_ms: onsetTimeMs,
              side,
              confidence: prediction.confidence,
              probabilities: prediction.probabilities,
              roi_source: crop.roi_source,
              frame_delay_ms: crop.frame_delay_ms,
              audio_label: result.prediction?.label ?? 'unknown',
              audio_confidence: result.prediction?.confidence ?? 0,
              rgb_b64: crop.rgb_b64,
            });
          }
        } catch {
          // ingen kameraframe ännu / capture-fel: räkna inte sida för denna studs
        } finally {
          busyRef.current = false;
        }
      })();
    });

    return () => sub.remove();
  }, [isRunning]);

  useEffect(() => () => { if (isRunning) stopAll(); }, [isRunning, stopAll]);

  return (
    <View style={[styles.root, { paddingTop: insets.top }]}>
      <StatusBar barStyle="light-content" backgroundColor="#000" />
      <View style={styles.header}>
        <TouchableOpacity onPress={() => { stopAll(); onDone(); }}>
          <Text style={styles.back}>‹ Tillbaka</Text>
        </TouchableOpacity>
        <Text style={styles.title}>Studs FH/BH LIVE</Text>
        <Text style={styles.subtitle}>{BOUNCE_SIDE_MODEL_VERSION}</Text>
      </View>

      <View style={styles.cameraWrap}>
        <BounceSideCameraView style={styles.camera} />
        {lastSide ? (
          <View style={[
            styles.sideBadge,
            lastSide.side === 'forehand' ? styles.badgeFh : lastSide.side === 'backhand' ? styles.badgeBh : styles.badgeUncertain,
          ]}>
            <Text style={styles.sideBadgeTxt}>
              {lastSide.side === 'forehand' ? 'FOREHAND' : lastSide.side === 'backhand' ? 'BACKHAND' : 'OSÄKER'} {(lastSide.confidence * 100).toFixed(0)}%
            </Text>
          </View>
        ) : null}
      </View>

      <View style={styles.countRow}>
        <View style={styles.countBox}>
          <Text style={[styles.countValue, { color: '#35c7ff' }]}>{fhCount}</Text>
          <Text style={styles.countLabel}>FOREHAND</Text>
        </View>
        <View style={styles.countBox}>
          <Text style={[styles.countValue, { color: '#f1c40f' }]}>{bhCount}</Text>
          <Text style={styles.countLabel}>BACKHAND</Text>
        </View>
        <View style={styles.countBox}>
          <Text style={[styles.countValue, { color: '#777' }]}>{uncertainCount}</Text>
          <Text style={styles.countLabel}>OSÄKER</Text>
        </View>
      </View>

      <View style={styles.colorRow}>
        <Text style={styles.colorLabel}>Forehandsidans färg:</Text>
        {(['red', 'black'] as const).map(color => {
          const active = forehandColor === color;
          return (
            <TouchableOpacity
              key={color}
              style={[styles.colorBtn, active && styles.colorBtnActive]}
              onPress={() => setForehandColor(color)}
            >
              <Text style={[styles.colorTxt, active && styles.colorTxtActive]}>
                {color === 'red' ? 'Röd' : 'Svart'}
              </Text>
            </TouchableOpacity>
          );
        })}
      </View>

      <TouchableOpacity style={[styles.toggle, isRunning ? styles.toggleStop : styles.toggleStart]} onPress={toggle}>
        <Text style={styles.toggleText}>{isRunning ? 'STOPPA' : 'STARTA'}</Text>
      </TouchableOpacity>

      <Text style={styles.statusText}>{statusText}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#000' },
  header: { paddingHorizontal: 16, paddingBottom: 6 },
  back: { color: '#4a9eff', fontSize: 16, paddingVertical: 6 },
  title: { color: '#fff', fontSize: 24, fontWeight: '700' },
  subtitle: { color: '#555', fontSize: 11 },
  cameraWrap: { flex: 1, margin: 12, borderRadius: 14, overflow: 'hidden', backgroundColor: '#111' },
  camera: { flex: 1 },
  sideBadge: { position: 'absolute', top: 12, alignSelf: 'center', paddingHorizontal: 18, paddingVertical: 8, borderRadius: 20 },
  badgeFh: { backgroundColor: 'rgba(53,199,255,0.85)' },
  badgeBh: { backgroundColor: 'rgba(241,196,15,0.85)' },
  badgeUncertain: { backgroundColor: 'rgba(150,150,150,0.85)' },
  sideBadgeTxt: { color: '#000', fontWeight: '800', fontSize: 16 },
  countRow: { flexDirection: 'row', gap: 12, paddingHorizontal: 12 },
  countBox: { flex: 1, alignItems: 'center', backgroundColor: '#101010', borderRadius: 12, paddingVertical: 10 },
  countValue: { fontSize: 44, fontWeight: '800' },
  countLabel: { color: '#888', fontSize: 12, letterSpacing: 2 },
  colorRow: { flexDirection: 'row', alignItems: 'center', gap: 8, paddingHorizontal: 16, paddingTop: 10 },
  colorLabel: { color: '#888', fontSize: 13 },
  colorBtn: { paddingHorizontal: 14, paddingVertical: 6, borderRadius: 8, borderWidth: 1, borderColor: '#333', backgroundColor: '#111' },
  colorBtnActive: { borderColor: '#2ecc71', backgroundColor: '#12351f' },
  colorTxt: { color: '#888', fontSize: 13 },
  colorTxtActive: { color: '#fff', fontWeight: '700' },
  toggle: { marginHorizontal: 24, marginVertical: 10, paddingVertical: 14, borderRadius: 10, alignItems: 'center' },
  toggleStart: { backgroundColor: '#1d6f42' },
  toggleStop: { backgroundColor: '#8e2b2b' },
  toggleText: { color: '#fff', fontSize: 18, fontWeight: '800' },
  statusText: { color: '#666', fontSize: 12, textAlign: 'center', paddingHorizontal: 16, paddingBottom: 12 },
});
