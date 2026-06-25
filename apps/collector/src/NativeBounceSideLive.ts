import { NativeModules, requireNativeComponent } from 'react-native';
import type { ViewProps } from 'react-native';

export interface BounceSideLiveCrop {
  /** base64-kodade 64*64*3 RGB-bytes (rad-major). */
  rgb_b64: string;
  roi_source: 'wrist_anchor' | 'center_fallback';
  /** Hur långt från träffögonblicket den valda framen låg (ms, +=efter). */
  frame_delay_ms: number;
}

interface BounceSideLiveInterface {
  startCamera(useFrontCamera: boolean): Promise<string>;
  stopCamera(): Promise<string>;
  /** Racket-crop ur framen närmast `targetTimeMs` (träffögonblicket,
   *  samma klocka som ljudets onset_time_ms). 0 = senaste framen. */
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
  captureCrop: async (_targetTimeMs: number) => {
    throw new Error('BounceSideLive native module is only available on Android.');
  },
};

/** Kameraförhandsvisningen (CameraX PreviewView). */
export const BounceSideCameraView = requireNativeComponent<ViewProps>('BounceSideCameraView');
