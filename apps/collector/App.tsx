import React, { useState } from 'react';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import type { Device } from 'react-native-ble-plx';
import type { CalibrationData, CalibrationMode, PlayerSetup } from './src/types';
import { SetupScreen } from './src/SetupScreen';
import { CalibrationScreen } from './src/CalibrationScreen';
import { DataCollectionScreen } from './src/DataCollectionScreen';
import { AudioCollectionScreen } from './src/AudioCollectionScreen';
import { LiveClassificationScreen } from './src/LiveClassificationScreen';
import { BounceTestScreen } from './src/BounceTestScreen';

type Screen =
  | 'setup'
  | 'calibration'
  | 'collection'
  | 'audio_collection'
  | 'bounce_audio_imu_collection'
  | 'live_classification'
  | 'bounce_free'
  | 'bounce_alternating';

type CalibrationTarget = 'collection' | 'bounce_audio_imu_collection' | 'bounce_free' | 'bounce_alternating';

interface AppState {
  screen: Screen;
  setup?: PlayerSetup;
  calibration?: CalibrationData;
  bleDevice?: Device;
  calibrationTarget?: CalibrationTarget;
}

function calibrationModeForTarget(target: CalibrationTarget): CalibrationMode {
  return target === 'bounce_free' || target === 'bounce_alternating'
    ? 'bounce_sides'
    : 'table_only';
}

export default function App() {
  const [state, setState] = useState<AppState>({ screen: 'setup' });

  if (state.screen === 'setup') {
    return (
      <SafeAreaProvider>
        <SetupScreen
          onCollectionMode={setup => setState({ screen: 'calibration', setup, calibrationTarget: 'collection' })}
          onAudioMode={setup => setState({ screen: 'audio_collection', setup })}
          onBounceAudioImuMode={setup =>
            setState({ screen: 'calibration', setup, calibrationTarget: 'bounce_audio_imu_collection' })
          }
          onLiveMode={setup => setState({ screen: 'live_classification', setup })}
          onBounceFreeMode={setup => setState({ screen: 'calibration', setup, calibrationTarget: 'bounce_free' })}
          onBounceAlternatingMode={setup =>
            setState({ screen: 'calibration', setup, calibrationTarget: 'bounce_alternating' })
          }
        />
      </SafeAreaProvider>
    );
  }

  if (state.screen === 'audio_collection' && state.setup) {
    return (
      <SafeAreaProvider>
        <AudioCollectionScreen setup={state.setup} onDone={() => setState({ screen: 'setup' })} />
      </SafeAreaProvider>
    );
  }

  if (
    state.screen === 'bounce_audio_imu_collection' &&
    state.setup &&
    state.calibration &&
    state.bleDevice
  ) {
    return (
      <SafeAreaProvider>
        <AudioCollectionScreen
          setup={state.setup}
          calibration={state.calibration}
          device={state.bleDevice}
          mode="audio_imu"
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
        onCollectionMode={setup => setState({ screen: 'calibration', setup, calibrationTarget: 'collection' })}
        onAudioMode={setup => setState({ screen: 'audio_collection', setup })}
        onBounceAudioImuMode={setup =>
          setState({ screen: 'calibration', setup, calibrationTarget: 'bounce_audio_imu_collection' })
        }
        onLiveMode={setup => setState({ screen: 'live_classification', setup })}
        onBounceFreeMode={setup => setState({ screen: 'calibration', setup, calibrationTarget: 'bounce_free' })}
        onBounceAlternatingMode={setup =>
          setState({ screen: 'calibration', setup, calibrationTarget: 'bounce_alternating' })
        }
      />
    </SafeAreaProvider>
  );
}
