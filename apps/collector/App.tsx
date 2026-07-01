import React, { useState } from 'react';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import type { Device } from 'react-native-ble-plx';
import type { CalibrationData, CalibrationMode, PlayerSetup } from './src/types';
import { SetupScreen } from './src/SetupScreen';
import { CalibrationScreen } from './src/CalibrationScreen';
import { DataCollectionScreen } from './src/DataCollectionScreen';
import { AudioCollectionScreen } from './src/AudioCollectionScreen';
import { LiveClassificationScreen } from './src/LiveClassificationScreen';
import { FableLiveScreen } from './src/FableLiveScreen';
import { FableTrainingRecorderScreen } from './src/FableTrainingRecorderScreen';
import { BounceAudioTestScreen } from './src/BounceAudioTestScreen';
import { BounceSideLiveScreen } from './src/BounceSideLiveScreen';
import { BounceTestScreen } from './src/BounceTestScreen';
import { VideoOnlyStrokeCollectionScreen } from './src/VideoOnlyStrokeCollectionScreen';

type Screen =
  | 'setup'
  | 'calibration'
  | 'collection'
  | 'audio_video_pose_collection'
  | 'video_only_stroke_collection'
  | 'video_bounce_side_collection'
  | 'fable_training_recorder'
  | 'free_recording'
  | 'live_classification'
  | 'fable_live'
  | 'bounce_audio_test'
  | 'bounce_side_live'
  | 'bounce_free'
  | 'bounce_alternating';

type CalibrationTarget = 'collection' | 'free_recording' | 'bounce_free' | 'bounce_alternating';

interface AppState {
  screen: Screen;
  setup?: PlayerSetup;
  calibration?: CalibrationData;
  bleDevice?: Device;
  calibrationTarget?: CalibrationTarget;
}

function calibrationModeForTarget(target: CalibrationTarget): CalibrationMode {
  return target === 'free_recording' ||
    target === 'bounce_free' ||
    target === 'bounce_alternating'
    ? 'bounce_sides'
    : 'table_only';
}

export default function App() {
  const [state, setState] = useState<AppState>({ screen: 'setup' });

  if (state.screen === 'setup') {
    return (
      <SafeAreaProvider>
        <SetupScreen
          onAudioVideoPoseMode={setup => setState({ screen: 'audio_video_pose_collection', setup })}
          onVideoOnlyStrokeMode={setup => setState({ screen: 'video_only_stroke_collection', setup })}
          onVideoBounceSideMode={setup => setState({ screen: 'video_bounce_side_collection', setup })}
          onFableTrainingRecorderMode={setup => setState({ screen: 'fable_training_recorder', setup })}
          onLiveMode={setup => setState({ screen: 'live_classification', setup })}
          onFableLiveMode={setup => setState({ screen: 'fable_live', setup })}
          onBounceAudioTestMode={setup => setState({ screen: 'bounce_audio_test', setup })}
          onBounceSideLiveMode={setup => setState({ screen: 'bounce_side_live', setup })}
          onBounceFreeMode={setup => setState({ screen: 'calibration', setup, calibrationTarget: 'bounce_free' })}
          onBounceAlternatingMode={setup =>
            setState({ screen: 'calibration', setup, calibrationTarget: 'bounce_alternating' })
          }
        />
      </SafeAreaProvider>
    );
  }

  if (state.screen === 'fable_live' && state.setup) {
    return (
      <SafeAreaProvider>
        <FableLiveScreen setup={state.setup} onDone={() => setState({ screen: 'setup' })} />
      </SafeAreaProvider>
    );
  }

  if (state.screen === 'bounce_audio_test' && state.setup) {
    return (
      <SafeAreaProvider>
        <BounceAudioTestScreen setup={state.setup} onDone={() => setState({ screen: 'setup' })} />
      </SafeAreaProvider>
    );
  }

  if (state.screen === 'bounce_side_live' && state.setup) {
    return (
      <SafeAreaProvider>
        <BounceSideLiveScreen setup={state.setup} onDone={() => setState({ screen: 'setup' })} />
      </SafeAreaProvider>
    );
  }

  if (state.screen === 'fable_training_recorder' && state.setup) {
    return (
      <SafeAreaProvider>
        <FableTrainingRecorderScreen setup={state.setup} onDone={() => setState({ screen: 'setup' })} />
      </SafeAreaProvider>
    );
  }

  if (state.screen === 'free_recording' && state.setup) {
    return (
      <SafeAreaProvider>
        <AudioCollectionScreen
          setup={state.setup}
          calibration={state.calibration}
          device={state.bleDevice}
          mode="free_recording"
          onDone={() => setState({ screen: 'setup' })}
        />
      </SafeAreaProvider>
    );
  }

  if (state.screen === 'live_classification' && state.setup) {
    return (
      <SafeAreaProvider>
        <LiveClassificationScreen setup={state.setup} onDone={() => setState({ screen: 'setup' })} />
      </SafeAreaProvider>
    );
  }

  if (state.screen === 'audio_video_pose_collection' && state.setup) {
    return (
      <SafeAreaProvider>
        <AudioCollectionScreen
          setup={state.setup}
          mode="audio_video_pose"
          onDone={() => setState({ screen: 'setup' })}
        />
      </SafeAreaProvider>
    );
  }

  if (state.screen === 'video_only_stroke_collection' && state.setup) {
    return (
      <SafeAreaProvider>
        <VideoOnlyStrokeCollectionScreen
          setup={state.setup}
          onDone={() => setState({ screen: 'setup' })}
        />
      </SafeAreaProvider>
    );
  }

  if (state.screen === 'video_bounce_side_collection' && state.setup) {
    return (
      <SafeAreaProvider>
        <VideoOnlyStrokeCollectionScreen
          setup={state.setup}
          mode="bounce_side"
          onDone={() => setState({ screen: 'setup' })}
        />
      </SafeAreaProvider>
    );
  }

  if (state.screen === 'calibration' && state.setup && state.calibrationTarget) {
    return (
      <SafeAreaProvider>
        <CalibrationScreen
          setup={state.setup}
          mode={calibrationModeForTarget(state.calibrationTarget)}
          onCalibrated={(calibration, device) =>
            setState(prev => ({
              ...prev,
              screen: prev.calibrationTarget ?? 'setup',
              calibration,
              bleDevice: device,
            }))
          }
          onBack={() => setState({ screen: 'setup' })}
        />
      </SafeAreaProvider>
    );
  }

  if (state.screen === 'collection' && state.setup && state.calibration && state.bleDevice) {
    return (
      <SafeAreaProvider>
        <DataCollectionScreen
          setup={state.setup}
          calibration={state.calibration}
          device={state.bleDevice}
          onDone={() => setState({ screen: 'setup' })}
        />
      </SafeAreaProvider>
    );
  }

  if (state.screen === 'bounce_free' && state.setup && state.calibration && state.bleDevice) {
    return (
      <SafeAreaProvider>
        <BounceTestScreen
          setup={state.setup}
          calibration={state.calibration}
          device={state.bleDevice}
          mode="bounce_free"
          onDone={() => setState({ screen: 'setup' })}
        />
      </SafeAreaProvider>
    );
  }

  if (state.screen === 'bounce_alternating' && state.setup && state.calibration && state.bleDevice) {
    return (
      <SafeAreaProvider>
        <BounceTestScreen
          setup={state.setup}
          calibration={state.calibration}
          device={state.bleDevice}
          mode="bounce_alternating"
          onDone={() => setState({ screen: 'setup' })}
        />
      </SafeAreaProvider>
    );
  }

  return (
    <SafeAreaProvider>
      <SetupScreen
        onAudioVideoPoseMode={setup => setState({ screen: 'audio_video_pose_collection', setup })}
        onVideoOnlyStrokeMode={setup => setState({ screen: 'video_only_stroke_collection', setup })}
        onVideoBounceSideMode={setup => setState({ screen: 'video_bounce_side_collection', setup })}
        onFableTrainingRecorderMode={setup => setState({ screen: 'fable_training_recorder', setup })}
        onLiveMode={setup => setState({ screen: 'live_classification', setup })}
        onFableLiveMode={setup => setState({ screen: 'fable_live', setup })}
        onBounceAudioTestMode={setup => setState({ screen: 'bounce_audio_test', setup })}
        onBounceSideLiveMode={setup => setState({ screen: 'bounce_side_live', setup })}
        onBounceFreeMode={setup => setState({ screen: 'calibration', setup, calibrationTarget: 'bounce_free' })}
        onBounceAlternatingMode={setup =>
          setState({ screen: 'calibration', setup, calibrationTarget: 'bounce_alternating' })
        }
      />
    </SafeAreaProvider>
  );
}
