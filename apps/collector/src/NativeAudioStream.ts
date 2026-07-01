import { NativeModules, NativeEventEmitter } from 'react-native';

interface AudioStreamInterface {
  /** Startar kontinuerlig streaming. threshold = RMS-tröskel (0.0–1.0). */
  startStreaming(threshold: number): Promise<string>;
  stopStreaming(): Promise<string>;
  setThreshold(threshold: number): Promise<string>;
  setRetriggerMs(ms: number): Promise<string>;
  /** Optional debug WAV path for the next streaming session. Pass null/empty to disable. */
  setDebugRecordingPath(path: string | null): Promise<string>;
  /**
   * Fable-läget: gate-RMS-läge ('broadband' | 'bandpass' 1.5–7 kHz),
   * hård spektralgate på/av, absolut RMS-golv. Återställs till
   * broadband/true/0.003 vid varje startStreaming — anropa EFTER start.
   */
  setGateConfig(mode: 'broadband' | 'bandpass', spectralGate: boolean, absMinRms: number): Promise<string>;
  /**
   * T0076 test gate. Disabled by default on every startStreaming, so old screens
   * keep the adaptive RMS gate unless they explicitly enable this after start.
   */
  setPeakGateConfig(
    enabled: boolean,
    smoothMs: number,
    minGapMs: number,
    backgroundWindowMs: number,
    backgroundExcludeMs: number,
    absMin: number,
    ratioMin: number,
    zMin: number,
  ): Promise<string>;
  addListener(eventName: string): void;
  removeListeners(count: number): void;
}

export const AudioStream: AudioStreamInterface = NativeModules.AudioStream;

export interface NativeAudioOnsetDebug {
  gate_id?: string;
  onset_time_ms?: number;
  onset_pos?: number;
  rms?: number;
  background_rms?: number;
  adaptive_threshold?: number;
  onset_ratio?: number;
  retrigger_ms?: number;
  spectral_passed?: boolean;
  ball_ratio?: number;
  flatness?: number;
  peak_value?: number;
  peak_ratio?: number;
  peak_z?: number;
  peak_local_mad?: number;
  peak_smooth_ms?: number;
  peak_min_gap_ms?: number;
  peak_background_window_ms?: number;
  peak_background_exclude_ms?: number;
  peak_abs_min?: number;
  peak_ratio_min?: number;
  peak_z_min?: number;
  native_reject_reason?: string | null;
}

export interface NativeAudioBouncePayload {
  audio_b64?: string | null;
  native_debug?: NativeAudioOnsetDebug;
}

export type NativeAudioBounceEvent = string | NativeAudioBouncePayload;

/** Emitter for "onBounceDetected"; payload can be a legacy base64 string or a debug object. */
export const AudioStreamEmitter = new NativeEventEmitter(NativeModules.AudioStream);
