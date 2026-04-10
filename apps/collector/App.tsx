import React, { useState } from 'react';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import type { Device } from 'react-native-ble-plx';
import type { PlayerSetup, CalibrationData } from './src/types';
import { SetupScreen } from './src/SetupScreen';
import { CalibrationScreen } from './src/CalibrationScreen';
import { DataCollectionScreen } from './src/DataCollectionScreen';
import { AudioCollectionScreen } from './src/AudioCollectionScreen';
import { LiveClassificationScreen } from './src/LiveClassificationScreen';

type Screen = 'setup' | 'calibration' | 'collection' | 'audio_collection' | 'live_classification';

interface AppState {
  screen: Screen;
  setup?: PlayerSetup;
  calibration?: CalibrationData;
  bleDevice?: Device;
}

export default function App() {
  const [state, setState] = useState<AppState>({ screen: 'setup' });

  if (state.screen === 'setup') {
    return (
      <SafeAreaProvider>
        <SetupScreen
          onDone={setup => setState({ screen: 'calibration', setup })}
          onAudioMode={setup => setState({ screen: 'audio_collection', setup })}
          onLiveMode={setup => setState({ screen: 'live_classification', setup })}
        />
      </SafeAreaProvider>
    );
  }

  if (state.screen === 'audio_collection' && state.setup) {
    return (
      <SafeAreaProvider>
        <AudioCollectionScreen
          setup={state.setup}
          onDone={() => setState({ screen: 'setup' })}
        />
      </SafeAreaProvider>
    );
  }

  if (state.screen === 'live_classification' && state.setup) {
    return (
      <SafeAreaProvider>
        <LiveClassificationScreen
          setup={state.setup}
          onDone={() => setState({ screen: 'setup' })}
        />
      </SafeAreaProvider>
    );
  }

  if (state.screen === 'calibration' && state.setup) {
    return (
      <SafeAreaProvider>
        <CalibrationScreen
          setup={state.setup}
          onCalibrated={(calibration, device) =>
            setState(prev => ({ ...prev, screen: 'collection', calibration, bleDevice: device }))
          }
          onBack={() => setState({ screen: 'setup' })}
        />
      </SafeAreaProvider>
    );
  }

  if (
    state.screen === 'collection' &&
    state.setup &&
    state.calibration &&
    state.bleDevice
  ) {
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

  // Fallback — bör inte hända
  return (
    <SafeAreaProvider>
      <SetupScreen
        onDone={setup => setState({ screen: 'calibration', setup })}
        onAudioMode={setup => setState({ screen: 'audio_collection', setup })}
      />
    </SafeAreaProvider>
  );
}
