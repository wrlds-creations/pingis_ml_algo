import type { PlayerSetup, VideoPoseFrame } from './types';

export const VIDEO_STROKE_FEATURE_SPEC = 'video_stroke_features_v2' as const;
export const VIDEO_STROKE_WINDOW_PRE_MS = 700;
export const VIDEO_STROKE_WINDOW_POST_MS = 500;
export const VIDEO_STROKE_MIN_FRAMES = 4;
export const VIDEO_STROKE_MIN_AVG_VISIBILITY = 0.35;

export const VIDEO_STROKE_FEATURE_NAMES = [
  'frame_count',
  'avg_visibility',
  'wrist_x_mean',
  'wrist_x_std',
  'wrist_x_min',
  'wrist_x_max',
  'wrist_x_delta',
  'wrist_x_ptp',
  'wrist_y_mean',
  'wrist_y_std',
  'wrist_y_min',
  'wrist_y_max',
  'wrist_y_delta',
  'wrist_y_ptp',
  'elbow_x_mean',
  'elbow_x_std',
  'elbow_x_delta',
  'elbow_x_ptp',
  'elbow_y_mean',
  'elbow_y_std',
  'elbow_y_delta',
  'elbow_y_ptp',
  'wrist_speed_mean',
  'wrist_speed_max',
  'elbow_angle_mean',
  'elbow_angle_min',
  'elbow_angle_max',
  'elbow_angle_delta',
  'wrist_above_shoulder_ratio',
  'wrist_cross_body_ratio',
] as const;

type VideoStrokeFeatureName = typeof VIDEO_STROKE_FEATURE_NAMES[number];

interface NormalizedPoseSample {
  timestamp_ms: number;
  visibility: number;
  wrist_x: number;
  wrist_y: number;
  elbow_x: number;
  elbow_y: number;
  shoulder_x: number;
  shoulder_y: number;
  elbow_angle: number;
}

export interface VideoStrokeFeatureResult {
  /** v1-features (rörelse-gaten läser dessa) + v2-features (modellen läser
   *  dessa via feature_names i video_stroke_model.json). */
  features: Record<string, number>;
  frame_count: number;
  avg_visibility: number;
}

const LEFT_SHOULDER = 11;
const RIGHT_SHOULDER = 12;
const LEFT_ELBOW = 13;
const RIGHT_ELBOW = 14;
const LEFT_WRIST = 15;
const RIGHT_WRIST = 16;

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function std(values: number[]): number {
  if (values.length < 2) return 0;
  const average = mean(values);
  const variance = values.reduce((sum, value) => sum + (value - average) ** 2, 0) / values.length;
  return Math.sqrt(variance);
}

function min(values: number[]): number {
  return values.length === 0 ? 0 : Math.min(...values);
}

function max(values: number[]): number {
  return values.length === 0 ? 0 : Math.max(...values);
}

function delta(values: number[]): number {
  return values.length < 2 ? 0 : values[values.length - 1] - values[0];
}

function pointToPoint(values: number[]): number {
  return max(values) - min(values);
}

function ratio(values: number[], predicate: (value: number) => boolean): number {
  if (values.length === 0) return 0;
  return values.filter(predicate).length / values.length;
}

function angleDegrees(
  shoulderX: number,
  shoulderY: number,
  elbowX: number,
  elbowY: number,
  wristX: number,
  wristY: number,
): number {
  const upperX = shoulderX - elbowX;
  const upperY = shoulderY - elbowY;
  const lowerX = wristX - elbowX;
  const lowerY = wristY - elbowY;
  const upperLength = Math.hypot(upperX, upperY);
  const lowerLength = Math.hypot(lowerX, lowerY);
  if (upperLength <= 0 || lowerLength <= 0) return 0;
  const cosine = Math.max(-1, Math.min(1, (upperX * lowerX + upperY * lowerY) / (upperLength * lowerLength)));
  return Math.acos(cosine) * (180 / Math.PI);
}

function normalizedSamples(
  frames: VideoPoseFrame[],
  markerMs: number,
  handedness: PlayerSetup['handedness'],
): NormalizedPoseSample[] {
  const startMs = markerMs - VIDEO_STROKE_WINDOW_PRE_MS;
  const endMs = markerMs + VIDEO_STROKE_WINDOW_POST_MS;
  const handShoulderType = handedness === 'right' ? RIGHT_SHOULDER : LEFT_SHOULDER;
  const handElbowType = handedness === 'right' ? RIGHT_ELBOW : LEFT_ELBOW;
  const handWristType = handedness === 'right' ? RIGHT_WRIST : LEFT_WRIST;

  const samples: NormalizedPoseSample[] = [];
  for (const frame of frames) {
    if (frame.timestamp_ms < startMs || frame.timestamp_ms > endMs || !frame.pose_detected) continue;
    const landmarks = new Map(frame.landmarks.map(landmark => [landmark.type, landmark]));
    const leftShoulder = landmarks.get(LEFT_SHOULDER);
    const rightShoulder = landmarks.get(RIGHT_SHOULDER);
    const handShoulder = landmarks.get(handShoulderType);
    const handElbow = landmarks.get(handElbowType);
    const handWrist = landmarks.get(handWristType);
    if (!leftShoulder || !rightShoulder || !handShoulder || !handElbow || !handWrist) continue;

    const shoulderWidth = Math.hypot(rightShoulder.x - leftShoulder.x, rightShoulder.y - leftShoulder.y);
    if (shoulderWidth < 0.04) continue;

    const centerX = (leftShoulder.x + rightShoulder.x) / 2;
    const centerY = (leftShoulder.y + rightShoulder.y) / 2;
    const visibility = mean([
      leftShoulder.visibility,
      rightShoulder.visibility,
      handShoulder.visibility,
      handElbow.visibility,
      handWrist.visibility,
    ]);
    if (visibility < 0.2) continue;

    const shoulderX = (handShoulder.x - centerX) / shoulderWidth;
    const shoulderY = (handShoulder.y - centerY) / shoulderWidth;
    const elbowX = (handElbow.x - centerX) / shoulderWidth;
    const elbowY = (handElbow.y - centerY) / shoulderWidth;
    const wristX = (handWrist.x - centerX) / shoulderWidth;
    const wristY = (handWrist.y - centerY) / shoulderWidth;

    samples.push({
      timestamp_ms: frame.timestamp_ms,
      visibility,
      wrist_x: wristX,
      wrist_y: wristY,
      elbow_x: elbowX,
      elbow_y: elbowY,
      shoulder_x: shoulderX,
      shoulder_y: shoulderY,
      elbow_angle: angleDegrees(shoulderX, shoulderY, elbowX, elbowY, wristX, wristY),
    });
  }

  return samples.sort((left, right) => left.timestamp_ms - right.timestamp_ms);
}

function speeds(samples: NormalizedPoseSample[]): number[] {
  const values: number[] = [];
  for (let sampleIndex = 1; sampleIndex < samples.length; sampleIndex += 1) {
    const previous = samples[sampleIndex - 1];
    const current = samples[sampleIndex];
    const deltaMs = current.timestamp_ms - previous.timestamp_ms;
    if (deltaMs <= 0) continue;
    values.push((Math.hypot(current.wrist_x - previous.wrist_x, current.wrist_y - previous.wrist_y) / deltaMs) * 1000);
  }
  return values;
}

// ── v2-features: tidsupplösta, kroppsram-normaliserade, spegel-invarianta ──
// Exakt port av extract_v2_features i train_video_stroke_v2.py (utan
// z-featurerna: z-semantiken skiljer mellan MediaPipe och ML Kit, övriga
// features är enhetsinvarianta via axelbredd-normaliseringen).

const V2_TIME_BINS = 4;
const V2_MIN_FRAMES = 5;

function buildV2Features(
  frames: VideoPoseFrame[],
  markerMs: number,
  handedness: PlayerSetup['handedness'],
): Record<string, number> | null {
  const startMs = markerMs - VIDEO_STROKE_WINDOW_PRE_MS;
  const endMs = markerMs + VIDEO_STROKE_WINDOW_POST_MS;
  const mirror = handedness === 'left' ? -1 : 1;
  const wristType = handedness === 'left' ? LEFT_WRIST : RIGHT_WRIST;
  const elbowType = handedness === 'left' ? LEFT_ELBOW : RIGHT_ELBOW;
  const shoulderType = handedness === 'left' ? LEFT_SHOULDER : RIGHT_SHOULDER;

  interface V2Row { t: number; nx: number; ny: number; angle: number; vis: number; }
  const rows: V2Row[] = [];
  for (const frame of frames) {
    if (frame.timestamp_ms < startMs || frame.timestamp_ms > endMs || !frame.pose_detected) continue;
    const landmarks = new Map(frame.landmarks.map(landmark => [landmark.type, landmark]));
    const ls = landmarks.get(LEFT_SHOULDER);
    const rs = landmarks.get(RIGHT_SHOULDER);
    const w = landmarks.get(wristType);
    const e = landmarks.get(elbowType);
    const s = landmarks.get(shoulderType);
    if (!ls || !rs || !w || !e || !s) continue;
    const cx = (ls.x + rs.x) / 2;
    const cy = (ls.y + rs.y) / 2;
    const width = Math.abs(rs.x - ls.x) + 1e-6;
    rows.push({
      t: frame.timestamp_ms - markerMs,
      nx: (mirror * (w.x - cx)) / width,
      ny: (w.y - cy) / width,
      angle: angleDegrees(s.x, s.y, e.x, e.y, w.x, w.y) || 180.0,
      vis: w.visibility ?? 0,
    });
  }
  if (rows.length < V2_MIN_FRAMES) return null;
  rows.sort((a, b) => a.t - b.t);

  const t = rows.map(r => r.t);
  const nx = rows.map(r => r.nx);
  const ny = rows.map(r => r.ny);
  const ang = rows.map(r => r.angle);
  const vis = rows.map(r => r.vis);

  const vx: number[] = [];
  const vy: number[] = [];
  const vt: number[] = [];
  for (let i = 1; i < rows.length; i++) {
    let dt = (t[i] - t[i - 1]) / 1000;
    if (dt <= 0) dt = 1e-3;
    vx.push((nx[i] - nx[i - 1]) / dt);
    vy.push((ny[i] - ny[i - 1]) / dt);
    vt.push(t[i]);
  }

  const feats: Record<string, number> = {};
  for (let b = 0; b < V2_TIME_BINS; b++) {
    const lo = -VIDEO_STROKE_WINDOW_PRE_MS + (b * 1200) / V2_TIME_BINS;
    const hi = -VIDEO_STROKE_WINDOW_PRE_MS + ((b + 1) * 1200) / V2_TIME_BINS;
    const idx = t.map((v, i) => [v, i] as const).filter(([v]) => v >= lo && v < hi).map(([, i]) => i);
    const vIdx = vt.map((v, i) => [v, i] as const).filter(([v]) => v >= lo && v < hi).map(([, i]) => i);
    const nxBin = idx.map(i => nx[i]);
    const nyBin = idx.map(i => ny[i]);
    feats[`bin${b}_nx_mean`] = mean(nxBin);
    feats[`bin${b}_nx_std`] = std(nxBin);
    feats[`bin${b}_ny_mean`] = mean(nyBin);
    feats[`bin${b}_ny_std`] = std(nyBin);
    feats[`bin${b}_vx_mean`] = mean(vIdx.map(i => vx[i]));
    feats[`bin${b}_vy_mean`] = mean(vIdx.map(i => vy[i]));
    feats[`bin${b}_angle_mean`] = idx.length ? mean(idx.map(i => ang[i])) : 180.0;
  }

  let impactIdx = 0;
  for (let i = 1; i < t.length; i++) {
    if (Math.abs(t[i]) < Math.abs(t[impactIdx])) impactIdx = i;
  }
  feats.impact_nx = nx[impactIdx];
  feats.impact_ny = ny[impactIdx];
  let vi = 0;
  for (let i = 1; i < vt.length; i++) {
    if (Math.abs(vt[i]) < Math.abs(vt[vi])) vi = i;
  }
  feats.impact_vx = vx.length ? vx[vi] : 0;
  feats.impact_vy = vy.length ? vy[vi] : 0;

  let nxMinIdx = 0;
  let nxMaxIdx = 0;
  for (let i = 1; i < nx.length; i++) {
    if (nx[i] < nx[nxMinIdx]) nxMinIdx = i;
    if (nx[i] > nx[nxMaxIdx]) nxMaxIdx = i;
  }
  feats.nx_min = nx[nxMinIdx];
  feats.nx_max = nx[nxMaxIdx];
  feats.nx_argmin_ms = t[nxMinIdx];
  feats.nx_argmax_ms = t[nxMaxIdx];
  feats.ny_min = min(ny);
  feats.ny_max = max(ny);

  let pathLen = 0;
  for (let i = 1; i < nx.length; i++) {
    pathLen += Math.hypot(nx[i] - nx[i - 1], ny[i] - ny[i - 1]);
  }
  feats.path_len = pathLen;
  let curvature = 0;
  if (vx.length > 1) {
    for (let i = 1; i < vx.length; i++) {
      curvature += Math.abs(vx[i] - vx[i - 1]) + Math.abs(vy[i] - vy[i - 1]);
    }
  }
  feats.curvature = curvature;
  feats.cross_body_ratio = ratio(nx, value => value < 0);
  feats.angle_min = min(ang);
  feats.angle_max = max(ang);
  feats.angle_delta = ang[ang.length - 1] - ang[0];
  feats.vis_mean = mean(vis);
  feats.n_frames = rows.length;
  return feats;
}

export function buildVideoStrokeFeatures(
  frames: VideoPoseFrame[],
  markerMs: number,
  handedness: PlayerSetup['handedness'],
): VideoStrokeFeatureResult | null {
  const samples = normalizedSamples(frames, markerMs, handedness);
  if (samples.length < VIDEO_STROKE_MIN_FRAMES) return null;

  const wristX = samples.map(sample => sample.wrist_x);
  const wristY = samples.map(sample => sample.wrist_y);
  const elbowX = samples.map(sample => sample.elbow_x);
  const elbowY = samples.map(sample => sample.elbow_y);
  const elbowAngles = samples.map(sample => sample.elbow_angle);
  const wristSpeeds = speeds(samples);
  const visibility = samples.map(sample => sample.visibility);
  const avgVisibility = mean(visibility);
  if (avgVisibility < VIDEO_STROKE_MIN_AVG_VISIBILITY) return null;

  const handednessSign = handedness === 'right' ? 1 : -1;
  const features: Record<string, number> = {
    frame_count: samples.length,
    avg_visibility: avgVisibility,
    wrist_x_mean: mean(wristX),
    wrist_x_std: std(wristX),
    wrist_x_min: min(wristX),
    wrist_x_max: max(wristX),
    wrist_x_delta: delta(wristX),
    wrist_x_ptp: pointToPoint(wristX),
    wrist_y_mean: mean(wristY),
    wrist_y_std: std(wristY),
    wrist_y_min: min(wristY),
    wrist_y_max: max(wristY),
    wrist_y_delta: delta(wristY),
    wrist_y_ptp: pointToPoint(wristY),
    elbow_x_mean: mean(elbowX),
    elbow_x_std: std(elbowX),
    elbow_x_delta: delta(elbowX),
    elbow_x_ptp: pointToPoint(elbowX),
    elbow_y_mean: mean(elbowY),
    elbow_y_std: std(elbowY),
    elbow_y_delta: delta(elbowY),
    elbow_y_ptp: pointToPoint(elbowY),
    wrist_speed_mean: mean(wristSpeeds),
    wrist_speed_max: max(wristSpeeds),
    elbow_angle_mean: mean(elbowAngles),
    elbow_angle_min: min(elbowAngles),
    elbow_angle_max: max(elbowAngles),
    elbow_angle_delta: delta(elbowAngles),
    wrist_above_shoulder_ratio: ratio(
      samples.map(sample => sample.wrist_y - sample.shoulder_y),
      value => value < 0,
    ),
    wrist_cross_body_ratio: ratio(wristX, value => value * handednessSign < 0),
  };

  // v2-features för modellen (v1 ovan behålls för rörelse-gaten).
  const v2 = buildV2Features(frames, markerMs, handedness);
  if (v2) Object.assign(features, v2);

  return {
    features,
    frame_count: samples.length,
    avg_visibility: avgVisibility,
  };
}
