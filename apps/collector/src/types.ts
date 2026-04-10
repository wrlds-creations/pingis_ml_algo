// Delade typer för hela Pingis Collector-appen

export interface ImuSample {
  accel_x: number;
  accel_y: number;
  accel_z: number;
  gyro_x: number;
  gyro_y: number;
  gyro_z: number;
  mag_x: number;
  mag_y: number;
  mag_z: number;
  ts_ms: number;
}

export interface PlayerSetup {
  name: string;
  handedness: 'right' | 'left';
}

export interface CalibrationData {
  gravity: { x: number; y: number; z: number };
  gyro_bias: { x: number; y: number; z: number };
}

export interface LabeledEvent {
  label: 'hit' | 'swing_miss' | 'idle';
  stroke_type: 'forehand' | 'backhand' | 'unknown';
  recorded_at: string;
  samples: ImuSample[];
}

export interface SessionFile {
  session_meta: {
    player_name: string;
    handedness: 'right' | 'left';
    calibration_accel: { x: number; y: number; z: number };
    calibration_gyro_bias: { x: number; y: number; z: number };
    session_date: string;
    app_version: string;
  };
  events: LabeledEvent[];
}

// ---- Audio bounce detection ----

export type AudioLabel = 'racket_bounce' | 'table_bounce' | 'floor_bounce' | 'noise';

export interface AudioEvent {
  label: AudioLabel;
  recorded_at: string;   // ISO timestamp för trycket
  wav_filename: string;  // relativ filnamn t.ex. "racket_bounce_000.m4a"
  duration_ms: number;   // faktisk inspelningstid
}

export interface AudioSessionFile {
  session_meta: {
    recorder_name: string;
    session_date: string;
    app_version: string;
    clip_duration_ms: number;
  };
  events: AudioEvent[];
}
