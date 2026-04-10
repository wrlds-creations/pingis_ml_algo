import { NativeModules } from 'react-native';

interface AudioCaptureInterface {
  /** Spelar in `durationMs` millisekunder PCM (22 050 Hz, mono, 16-bit).
   *  Returnerar base64-kodade Int16 little-endian bytes. */
  capture(durationMs: number): Promise<string>;
  /** Startar en lång WAV-sessionsinspelning till angiven sökväg. */
  startSession(outputPath: string): Promise<string>;
  /** Stoppar pågående WAV-session. Returnerar faktisk inspelningstid (ms). */
  stopSession(): Promise<number>;
}

export const AudioCapture: AudioCaptureInterface = NativeModules.AudioCapture;

/** Avkodar base64-sträng → Float32Array normerad till [-1, 1]. */
export function decodeBase64PCM(b64: string): Float32Array {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const int16 = new Int16Array(bytes.buffer);
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768.0;
  return float32;
}
