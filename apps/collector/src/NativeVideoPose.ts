import { NativeModules } from 'react-native';
import type { VideoPoseExtractionResult } from './types';

export interface BounceSideCrop {
  timestamp_ms: number;
  frame_ms: number;
  /** base64-kodade 64*64*3 RGB-bytes (rad-major). */
  rgb_b64: string;
  roi_source: 'wrist_anchor' | 'center_fallback';
}

interface VideoPoseInterface {
  extractPose(videoPath: string, sampleFps?: number): Promise<VideoPoseExtractionResult>;
  /**
   * Pose enbart i givna tidsfönster: flat array [start0, end0, start1, end1, ...]
   * i ms. Videon avkodas sekventiellt men ML Kit körs bara i fönstren —
   * stor hastighetsvinst när analysen är ljudankrad.
   */
  extractPoseInWindows(
    videoPath: string,
    sampleFps: number,
    windowsMs: number[],
  ): Promise<VideoPoseExtractionResult>;
  /** Handleds-ankrade racket-crops (64x64 RGB) vid givna tidsstämplar,
   *  för FH-/BH-sidoklassificeringen. */
  extractBounceSideCrops(videoPath: string, timestampsMs: number[]): Promise<BounceSideCrop[]>;
}

const nativeModule = NativeModules.VideoPose as VideoPoseInterface | undefined;

export const VideoPose: VideoPoseInterface = nativeModule ?? {
  extractPose: async () => {
    throw new Error('VideoPose native module is only available on Android.');
  },
  extractPoseInWindows: async () => {
    throw new Error('VideoPose native module is only available on Android.');
  },
  extractBounceSideCrops: async () => {
    throw new Error('VideoPose native module is only available on Android.');
  },
};
