import { NativeModules, Platform } from 'react-native';

interface ReviewOrientationInterface {
  lockLandscape(): Promise<void>;
  lockPortrait(): Promise<void>;
  unlock(): Promise<void>;
}

const nativeModule = NativeModules.ReviewOrientation as ReviewOrientationInterface | undefined;

function noop(): Promise<void> {
  return Promise.resolve();
}

export const ReviewOrientation = {
  lockLandscape(): Promise<void> {
    if (Platform.OS !== 'android' || !nativeModule?.lockLandscape) {
      return noop();
    }
    return nativeModule.lockLandscape();
  },
  lockPortrait(): Promise<void> {
    if (Platform.OS !== 'android' || !nativeModule?.lockPortrait) {
      return noop();
    }
    return nativeModule.lockPortrait();
  },
  unlock(): Promise<void> {
    if (Platform.OS !== 'android' || !nativeModule?.unlock) {
      return noop();
    }
    return nativeModule.unlock();
  },
};
