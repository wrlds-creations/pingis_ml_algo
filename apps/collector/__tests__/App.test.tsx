/**
 * @format
 */

import React from 'react';
import ReactTestRenderer from 'react-test-renderer';
import App from '../App';

jest.mock('react-native-ble-plx', () => ({
  BleManager: jest.fn(() => ({
    startDeviceScan: jest.fn(),
    stopDeviceScan: jest.fn(),
  })),
}));

jest.mock('react-native-fs', () => ({
  ExternalStorageDirectoryPath: '/tmp',
  CachesDirectoryPath: '/tmp',
  mkdir: jest.fn(() => Promise.resolve()),
  exists: jest.fn(() => Promise.resolve(false)),
  readDir: jest.fn(() => Promise.resolve([])),
  readFile: jest.fn(() => Promise.resolve('')),
  writeFile: jest.fn(() => Promise.resolve()),
  scanFile: jest.fn(() => Promise.resolve()),
  unlink: jest.fn(() => Promise.resolve()),
}));

jest.mock('react-native-audio-recorder-player', () => {
  return jest.fn().mockImplementation(() => ({
    startPlayer: jest.fn(() => Promise.resolve('started')),
    stopPlayer: jest.fn(() => Promise.resolve('stopped')),
    addPlayBackListener: jest.fn(),
    removePlayBackListener: jest.fn(),
    setSubscriptionDuration: jest.fn(() => Promise.resolve()),
  }));
});

jest.mock('react-native-vision-camera', () => ({
  Camera: 'VisionCameraView',
  useCameraPermission: () => ({
    hasPermission: true,
    requestPermission: jest.fn(() => Promise.resolve(true)),
  }),
  useCameraDevice: () => ({ id: 'mock-back-camera' }),
  useVideoOutput: () => ({
    createRecorder: jest.fn(() => Promise.resolve({
      isRecording: false,
      startRecording: jest.fn(() => Promise.resolve()),
      stopRecording: jest.fn(() => Promise.resolve()),
      cancelRecording: jest.fn(() => Promise.resolve()),
    })),
  }),
}));

jest.mock('react-native-video', () => 'Video');

jest.mock('../src/NativeAudioStream', () => ({
  AudioStream: {
    startStreaming: jest.fn(() => Promise.resolve('started')),
    stopStreaming: jest.fn(() => Promise.resolve('stopped')),
    setThreshold: jest.fn(() => Promise.resolve('ok')),
  },
  AudioStreamEmitter: {
    addListener: jest.fn(() => ({ remove: jest.fn() })),
  },
}));

jest.mock('../src/NativeAudioCapture', () => ({
  AUDIO_CAPTURE_STOPPED_EVENT: 'onAudioSessionStopped',
  AudioCapture: {
    startSession: jest.fn(() => Promise.resolve('started')),
    stopSession: jest.fn(() => Promise.resolve(1000)),
  },
  AudioCaptureEmitter: {
    addListener: jest.fn(() => ({ remove: jest.fn() })),
  },
  decodeBase64PCM: jest.fn(() => new Float32Array(22050)),
}));

test('renders correctly', async () => {
  await ReactTestRenderer.act(() => {
    ReactTestRenderer.create(<App />);
  });
});
