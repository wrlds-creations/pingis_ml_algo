import type { ImuSample, StrokeMotionMetrics } from './types';

const FEATURE_CHANNELS = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z'] as const;
const WINDOW_SAMPLES = 40;
const IMPACT_HALF_WINDOW = 4;

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function std(values: number[]): number {
  if (values.length === 0) return 0;
  const avg = mean(values);
  const variance = mean(values.map(value => (value - avg) ** 2));
  return Math.sqrt(variance);
}

function rms(values: number[]): number {
  if (values.length === 0) return 0;
  return Math.sqrt(mean(values.map(value => value * value)));
}

function ptp(values: number[]): number {
  if (values.length === 0) return 0;
  return Math.max(...values) - Math.min(...values);
}

export function buildStrokeWindow(samples: ImuSample[]): ImuSample[] | null {
  if (samples.length < WINDOW_SAMPLES) return null;
  return samples.slice(samples.length - WINDOW_SAMPLES);
}

export function extractStrokeFeatures(samples: ImuSample[]): Record<string, number> {
  const features: Record<string, number> = {};

  for (const channel of FEATURE_CHANNELS) {
    const values = samples.map(sample => sample[channel]);
    features[`${channel}_mean`] = mean(values);
    features[`${channel}_std`] = std(values);
    features[`${channel}_min`] = Math.min(...values);
    features[`${channel}_max`] = Math.max(...values);
    features[`${channel}_ptp`] = ptp(values);
    features[`${channel}_rms`] = rms(values);
  }

  const accelMagnitude = samples.map(sample =>
    Math.sqrt(sample.accel_x ** 2 + sample.accel_y ** 2 + sample.accel_z ** 2),
  );
  const gyroMagnitude = samples.map(sample =>
    Math.sqrt(sample.gyro_x ** 2 + sample.gyro_y ** 2 + sample.gyro_z ** 2),
  );

  features.accel_mag_mean = mean(accelMagnitude);
  features.accel_mag_peak = Math.max(...accelMagnitude);
  features.accel_mag_rms = rms(accelMagnitude);
  features.gyro_mag_std = std(gyroMagnitude);
  features.gyro_mag_peak = Math.max(...gyroMagnitude);

  const center = Math.floor(samples.length / 2);
  const impactStart = Math.max(0, center - IMPACT_HALF_WINDOW);
  const impactEnd = Math.min(samples.length, center + IMPACT_HALF_WINDOW);
  const impactRegion = accelMagnitude.slice(impactStart, impactEnd);
  features.accel_impact_peak = impactRegion.length > 0 ? Math.max(...impactRegion) : 0;
  features.accel_impact_std = std(impactRegion);

  return features;
}

export function extractStrokeMotionMetrics(samples: ImuSample[]): StrokeMotionMetrics {
  const accelMagnitude = samples.map(sample =>
    Math.sqrt(sample.accel_x ** 2 + sample.accel_y ** 2 + sample.accel_z ** 2),
  );
  const gyroMagnitude = samples.map(sample =>
    Math.sqrt(sample.gyro_x ** 2 + sample.gyro_y ** 2 + sample.gyro_z ** 2),
  );

  return {
    gyro_peak: Math.max(...gyroMagnitude),
    gyro_std: std(gyroMagnitude),
    accel_peak: Math.max(...accelMagnitude),
    accel_ptp: ptp(accelMagnitude),
  };
}
