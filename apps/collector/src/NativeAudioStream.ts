import { NativeModules, NativeEventEmitter } from 'react-native';

interface AudioStreamInterface {
  /** Startar kontinuerlig streaming. threshold = RMS-tröskel (0.0–1.0). */
  startStreaming(threshold: number): Promise<string>;
  stopStreaming(): Promise<string>;
  setThreshold(threshold: number): Promise<string>;
  setRetriggerMs(ms: number): Promise<string>;
  addListener(eventName: string): void;
  removeListeners(count: number): void;
}

export const AudioStream: AudioStreamInterface = NativeModules.AudioStream;

/** Emitter för händelsen "onBounceDetected" → payload: base64-sträng med 22050 Int16 LE samples */
export const AudioStreamEmitter = new NativeEventEmitter(NativeModules.AudioStream);
