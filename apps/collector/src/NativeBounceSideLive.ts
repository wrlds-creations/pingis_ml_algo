import { NativeEventEmitter, NativeModules, requireNativeComponent } from 'react-native';
import type { ViewProps } from 'react-native';

export interface BounceSideLiveCrop {
  /** base64-coded 64*64*3 RGB bytes, row-major. */
  rgb_b64: string;
  roi_source: 'wrist_anchor' | 'center_fallback';
  /** Difference between selected camera frame and target impact time, in ms. */
  frame_delay_ms: number;
}

export interface BounceSideRacketTrack {
  tracked: boolean;
  label: 'racket-red' | 'racket-black' | 'racket' | 'lost';
  color: 'red' | 'black' | 'uncertain';
  confidence: number;
  x: number;
  y: number;
  width: number;
  height: number;
  timestamp_ms: number;
  age_ms: number;
  frame_delay_ms: number;
  source: 'color_blob' | 'hold' | 'lost';
  red_score: number;
  dark_score: number;
  area_ratio: number;
  fill_ratio: number;
}

interface BounceSideLiveInterface {
  startCamera(useFrontCamera: boolean): Promise<string>;
  stopCamera(): Promise<string>;
  addListener(eventName: string): void;
  removeListeners(count: number): void;
  getRacketTrack(targetTimeMs: number): Promise<BounceSideRacketTrack>;
  /** Racket crop from the frame nearest targetTimeMs. */
  captureCrop(targetTimeMs: number): Promise<BounceSideLiveCrop>;
}

const nativeModule = NativeModules.BounceSideLive as BounceSideLiveInterface | undefined;

export const BounceSideLive: BounceSideLiveInterface = nativeModule ?? {
  startCamera: async () => {
    throw new Error('BounceSideLive native module is only available on Android.');
  },
  stopCamera: async () => {
    throw new Error('BounceSideLive native module is only available on Android.');
  },
  addListener: () => {},
  removeListeners: () => {},
  getRacketTrack: async (_targetTimeMs: number) => {
    throw new Error('BounceSideLive native module is only available on Android.');
  },
  captureCrop: async (_targetTimeMs: number) => {
    throw new Error('BounceSideLive native module is only available on Android.');
  },
};

export const BounceSideLiveEmitter = nativeModule
  ? new NativeEventEmitter(nativeModule as unknown as BounceSideLiveInterface)
  : {
      addListener: () => ({ remove: () => undefined }),
    };

export const BounceSideCameraView = requireNativeComponent<ViewProps>('BounceSideCameraView');
