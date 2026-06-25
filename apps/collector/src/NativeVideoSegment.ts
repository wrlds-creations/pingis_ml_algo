import { NativeModules } from 'react-native';

export interface VideoSegmentInfo {
  video_filename: string;
  video_path: string;
  take_index: number;
  start_ms: number;
  end_ms: number;
  duration_ms: number;
}

export interface ImportedVideoInfo {
  outputPath: string;
  displayName?: string;
  sourceUri: string;
  durationMs: number;
  rotation: number;
  sizeBytes: number;
}

interface VideoSegmentInterface {
  importVideoFile(outputPath: string): Promise<ImportedVideoInfo>;
  splitVideo(
    videoPath: string,
    outputDir: string,
    filenamePrefix: string,
    segmentDurationMs: number,
    startIndex?: number,
  ): Promise<VideoSegmentInfo[]>;
}

const nativeModule = NativeModules.VideoSegment as VideoSegmentInterface | undefined;

export const VideoSegment: VideoSegmentInterface = nativeModule ?? {
  importVideoFile: async () => {
    throw new Error('VideoSegment native module is only available on Android.');
  },
  splitVideo: async () => {
    throw new Error('VideoSegment native module is only available on Android.');
  },
};
