/**
 * bounceSideInference.ts
 *
 * FH-/BH-sida vid racketstuds från en 64x64-RGB-crop av racketregionen
 * (handleds-ankrad ROI, extraherad i VideoPoseModule.extractBounceSideCrops).
 *
 * TS-port av roi_features + trädmodellen i
 * skills/pingis-stroke-detection/scripts/classify_bounce_side.py
 * (grid-färgfeatures: 4x4-celler med röd-/mörk-andel + V-medel, viktat
 * hue-histogram, globala andelar). Tränad på Loves blandade underifrån-
 * session: 0.96 markör-accuracy på orörd holdout, backhand 13/13.
 * Paritet verifieras av check_bounce_side_ts_parity.js.
 */

import modelJson from './models/bounce_side_model.json';

type Node = number[];

interface BounceSideModel {
  metadata?: { model_version?: string };
  labels: string[];
  feature_names: string[];
  trees: Node[][];
}

const MODEL = modelJson as unknown as BounceSideModel;
const N_CLASSES = MODEL.labels.length;
const SIZE = 64;
const GRID = 4;
const CELL = SIZE / GRID;

export const BOUNCE_SIDE_MODEL_VERSION: string =
  MODEL.metadata?.model_version ?? 'bounce_side_unknown';

export interface BounceSidePrediction {
  label: 'forehand' | 'backhand';
  confidence: number;
  probabilities: Record<string, number>;
}

export type RacketForehandColor = 'red' | 'black';
export type VisibleRacketColor = 'red' | 'black' | 'uncertain';

export interface VisibleColorDecision {
  color: VisibleRacketColor;
  confidence: number;
  redTotal: number;
  darkTotal: number;
  reason: string;
}

export interface ResolvedBounceSide {
  side: 'forehand' | 'backhand' | 'uncertain';
  confidence: number;
  decisionSource: 'visible_color' | 'model' | 'uncertain';
  visibleColor: VisibleRacketColor;
  colorConfidence: number;
  redTotal: number;
  darkTotal: number;
  rawLabel: 'forehand' | 'backhand';
  rawConfidence: number;
  probabilities: Record<string, number>;
}

const RED_CLEAR_TOTAL = 0.22;
const RED_STRONG_TOTAL = 0.35;
const DARK_CLEAR_TOTAL = 0.28;
const DARK_STRONG_TOTAL = 0.50;
const COLOR_DOMINANCE_MARGIN = 0.15;

/** OpenCV-kompatibel RGB -> HSV (8-bit: H 0..179, S/V 0..255).
 *  Viktigt: OpenCV heltalsavrundar och WRAPPAR H vid 180 (hue 359°
 *  hamnar på ~0, inte 179) - annars byter nästan-röda pixlar
 *  hue-histogram-bin mellan Python och appen. */
function rgbToHsv(r: number, g: number, b: number): [number, number, number] {
  const v = Math.max(r, g, b);
  const mn = Math.min(r, g, b);
  const delta = v - mn;
  const s = v === 0 ? 0 : Math.round((255 * delta) / v);
  let h = 0;
  if (delta > 0) {
    if (v === r) h = (60 * (g - b)) / delta;
    else if (v === g) h = 120 + (60 * (b - r)) / delta;
    else h = 240 + (60 * (r - g)) / delta;
    if (h < 0) h += 360;
  }
  let h8 = Math.round(h / 2);
  if (h8 >= 180) h8 -= 180;
  return [h8, s, v];
}

/**
 * 65 grid-färgfeatures från en 64x64-RGB-crop (rad-major, 3 byte/pixel).
 * Måste räkna exakt som Python-referensens roi_features.
 */
export function bounceSideFeatures(rgb: Uint8Array, roiSource: string): Record<string, number> {
  const red = new Float64Array(SIZE * SIZE);
  const dark = new Float64Array(SIZE * SIZE);
  const vNorm = new Float64Array(SIZE * SIZE);
  const hueHist = new Float64Array(12);
  let redTotal = 0;
  let darkTotal = 0;
  let vSum = 0;

  for (let i = 0; i < SIZE * SIZE; i += 1) {
    const r = rgb[i * 3];
    const g = rgb[i * 3 + 1];
    const b = rgb[i * 3 + 2];
    const [h, s, v] = rgbToHsv(r, g, b);
    const isRed = ((h <= 10 || h >= 170) && s >= 80 && v >= 50) ? 1 : 0;
    const isDark = (v < 70 && s < 120) ? 1 : 0;
    red[i] = isRed;
    dark[i] = isDark;
    vNorm[i] = v / 255;
    redTotal += isRed;
    darkTotal += isDark;
    vSum += v;
    let bin = Math.floor(h / 15);
    if (bin > 11) bin = 11;
    if (bin < 0) bin = 0;
    hueHist[bin] += s / 255;
  }

  const feats: Record<string, number> = {};
  for (let gy = 0; gy < GRID; gy += 1) {
    for (let gx = 0; gx < GRID; gx += 1) {
      let cellRed = 0;
      let cellDark = 0;
      let cellV = 0;
      for (let y = gy * CELL; y < (gy + 1) * CELL; y += 1) {
        for (let x = gx * CELL; x < (gx + 1) * CELL; x += 1) {
          const i = y * SIZE + x;
          cellRed += red[i];
          cellDark += dark[i];
          cellV += vNorm[i];
        }
      }
      const n = CELL * CELL;
      feats[`g${gy}${gx}_red`] = cellRed / n;
      feats[`g${gy}${gx}_dark`] = cellDark / n;
      feats[`g${gy}${gx}_v`] = cellV / n;
    }
  }
  let histSum = 0;
  for (let i = 0; i < 12; i += 1) histSum += hueHist[i];
  for (let i = 0; i < 12; i += 1) feats[`hue_${i}`] = hueHist[i] / (histSum + 1e-9);
  feats.red_total = redTotal / (SIZE * SIZE);
  feats.dark_total = darkTotal / (SIZE * SIZE);
  feats.red_minus_dark = feats.red_total - feats.dark_total;
  feats.v_mean = vSum / (SIZE * SIZE) / 255;
  feats.is_fallback = roiSource === 'center_fallback' ? 1 : 0;
  return feats;
}

/**
 * Synlig racketsida från samma färgfeatures som modellen använder.
 *
 * 2026-06-17 live-debug visade att `bounce_side_v2_2026_06_11_underangle`
 * ibland gav hög backhand-probability trots tydligt rött gummi i cropen.
 * Den här guarden håller modellens spatiala signal när färgen är otydlig,
 * men låter stark röd/svart evidens vinna över en uppenbart fel sidolabel.
 */
export function visibleRacketColorFromFeatures(features: Record<string, number>): VisibleColorDecision {
  const redTotal = Number(features.red_total ?? 0);
  const darkTotal = Number(features.dark_total ?? 0);

  const blackStrong = darkTotal >= DARK_STRONG_TOTAL && redTotal < RED_STRONG_TOTAL;
  const redStrong = redTotal >= RED_STRONG_TOTAL && darkTotal < DARK_STRONG_TOTAL;
  const redClear = redTotal >= RED_CLEAR_TOTAL
    && redTotal + COLOR_DOMINANCE_MARGIN >= darkTotal
    && darkTotal < DARK_STRONG_TOTAL;
  const blackClear = darkTotal >= DARK_CLEAR_TOTAL
    && darkTotal >= redTotal + COLOR_DOMINANCE_MARGIN;

  if (blackStrong || blackClear) {
    return {
      color: 'black',
      confidence: Math.min(0.99, Math.max(0.6, 0.55 + darkTotal - Math.max(0, redTotal - darkTotal) * 0.25)),
      redTotal,
      darkTotal,
      reason: blackStrong ? 'black_strong' : 'black_clear',
    };
  }
  if (redStrong || redClear) {
    return {
      color: 'red',
      confidence: Math.min(0.99, Math.max(0.6, 0.55 + redTotal - Math.max(0, darkTotal - redTotal) * 0.25)),
      redTotal,
      darkTotal,
      reason: redStrong ? 'red_strong' : 'red_clear',
    };
  }
  return {
    color: 'uncertain',
    confidence: Math.max(0, Math.min(0.59, 0.5 + Math.abs(redTotal - darkTotal) * 0.2)),
    redTotal,
    darkTotal,
    reason: 'ambiguous_color',
  };
}

function mapVisibleColorToSide(color: VisibleRacketColor, forehandColor: RacketForehandColor): 'forehand' | 'backhand' | 'uncertain' {
  if (color === 'uncertain') return 'uncertain';
  if (forehandColor === 'red') return color === 'red' ? 'forehand' : 'backhand';
  return color === 'black' ? 'forehand' : 'backhand';
}

function mapModelLabelToSide(label: 'forehand' | 'backhand', forehandColor: RacketForehandColor): 'forehand' | 'backhand' {
  if (forehandColor === 'red') return label;
  return label === 'forehand' ? 'backhand' : 'forehand';
}

export function resolveBounceSide(
  features: Record<string, number>,
  prediction: BounceSidePrediction,
  forehandColor: RacketForehandColor,
  minModelConfidence = 0.6,
): ResolvedBounceSide {
  const color = visibleRacketColorFromFeatures(features);
  const colorSide = mapVisibleColorToSide(color.color, forehandColor);
  if (colorSide !== 'uncertain') {
    return {
      side: colorSide,
      confidence: color.confidence,
      decisionSource: 'visible_color',
      visibleColor: color.color,
      colorConfidence: color.confidence,
      redTotal: color.redTotal,
      darkTotal: color.darkTotal,
      rawLabel: prediction.label,
      rawConfidence: prediction.confidence,
      probabilities: prediction.probabilities,
    };
  }

  if (prediction.confidence < minModelConfidence) {
    return {
      side: 'uncertain',
      confidence: prediction.confidence,
      decisionSource: 'uncertain',
      visibleColor: color.color,
      colorConfidence: color.confidence,
      redTotal: color.redTotal,
      darkTotal: color.darkTotal,
      rawLabel: prediction.label,
      rawConfidence: prediction.confidence,
      probabilities: prediction.probabilities,
    };
  }

  return {
    side: mapModelLabelToSide(prediction.label, forehandColor),
    confidence: prediction.confidence,
    decisionSource: 'model',
    visibleColor: color.color,
    colorConfidence: color.confidence,
    redTotal: color.redTotal,
    darkTotal: color.darkTotal,
    rawLabel: prediction.label,
    rawConfidence: prediction.confidence,
    probabilities: prediction.probabilities,
  };
}

function isLeaf(node: Node): boolean {
  return node.length === N_CLASSES;
}

export function predictBounceSide(features: Record<string, number>): BounceSidePrediction {
  const names = MODEL.feature_names;
  const x = new Float64Array(names.length);
  for (let i = 0; i < names.length; i += 1) x[i] = features[names[i]] ?? 0;

  const probSum = new Float64Array(N_CLASSES);
  for (const tree of MODEL.trees) {
    let node = tree[0];
    while (!isLeaf(node)) {
      node = tree[x[node[0]] <= node[1] ? node[2] : node[3]];
    }
    for (let c = 0; c < N_CLASSES; c += 1) probSum[c] += node[c];
  }

  const probabilities: Record<string, number> = {};
  let bestIdx = 0;
  let bestProb = 0;
  for (let c = 0; c < N_CLASSES; c += 1) {
    const p = probSum[c] / MODEL.trees.length;
    probabilities[MODEL.labels[c]] = p;
    if (p > bestProb) { bestProb = p; bestIdx = c; }
  }
  return {
    label: MODEL.labels[bestIdx] as 'forehand' | 'backhand',
    confidence: bestProb,
    probabilities,
  };
}
