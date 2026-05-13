import { NativeEventEmitter, NativeModules } from 'react-native';

interface AudioCaptureInterface {
  /** Spelar in `durationMs` millisekunder PCM (22 050 Hz, mono, 16-bit).
   *  Returnerar base64-kodade Int16 little-endian bytes. */
  capture(durationMs: number): Promise<string>;
  /** Startar en lang WAV-sessionsinspelning till angiven sokvag. */
  startSession(outputPath: string, targetDurationMs?: number): Promise<string>;
  /** Stoppar pagaende WAV-session. Returnerar faktisk inspelningstid (ms). */
  stopSession(): Promise<number>;
  /** Importerar en ljudfil via Androids filvaljare och skriver om den till WAV. */
  importAudioFile(outputPath: string): Promise<ImportedAudioFile>;
  addListener(eventName: string): void;
  removeListeners(count: number): void;
}

export interface ImportedAudioFile {
  outputPath: string;
  displayName?: string;
  sourceUri?: string;
  durationMs: number;
  sampleRate: number;
  channels: number;
  writtenSamples: number;
}

export interface AudioCaptureStoppedEvent {
  outputPath: string;
  durationMs: number;
  writtenSamples: number;
}

export const AUDIO_CAPTURE_STOPPED_EVENT = 'onAudioSessionStopped';
export const AudioCapture: AudioCaptureInterface = NativeModules.AudioCapture;
export const AudioCaptureEmitter = new NativeEventEmitter(NativeModules.AudioCapture);

/** Avkodar base64-strang -> Float32Array normerad till [-1, 1]. */
export function decodeBase64PCM(b64: string): Float32Array {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const int16 = new Int16Array(bytes.buffer);
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768.0;
  return float32;
}
