import { NativeModules } from 'react-native';
import type { VideoPoseExtractionResult } from './types';

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
}

const nativeModule = NativeModules.VideoPose as VideoPoseInterface | undefined;

export const VideoPose: VideoPoseInterface = nativeModule ?? {
  extractPose: async () => {
    throw new Error('VideoPose native module is only available on Android.');
  },
  extractPoseInWindows: async () => {
    throw new Error('VideoPose native module is only available on Android.');
  },
};
