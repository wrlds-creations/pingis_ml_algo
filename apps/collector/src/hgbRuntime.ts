/**
 * hgbRuntime.ts
 *
 * Inferens för HistGradientBoosting-modellen i Fable-läget.
 * Modellformat (export_fable_hgb_model_json.py):
 *   labels[4], feature_names[83], scaler_mean/std, baseline[4],
 *   trees: 1600 träd, iterations-major (it0/klass0, it0/klass1, ...).
 *   Intern nod: [feature_idx, threshold, left_idx, right_idx] (längd 4)
 *   Löv:        [value]                                      (längd 1)
 * Beslut: scaled[feature_idx] <= threshold -> left.
 * raw[k] = baseline[k] + summa lövvärden för klass k; prob = softmax(raw).
 */

import modelJson from './models/fable_audio_model.json';

type HgbNode = number[];

interface FableModel {
  metadata?: {
    model_version?: string;
    engine_defaults?: Record<string, number | string | boolean>;
  };
  labels: string[];
  feature_names: string[];
  scaler_mean: number[];
  scaler_std: number[];
  baseline: number[];
  trees: HgbNode[][];
}

const MODEL = modelJson as unknown as FableModel;
const N_CLASSES = MODEL.labels.length;

export const FABLE_MODEL_VERSION: string =
  MODEL.metadata?.model_version ?? 'fable_audio_hgb_unknown';

export const FABLE_ENGINE_DEFAULTS = MODEL.metadata?.engine_defaults ?? {};

export interface FablePrediction {
  label: string;
  confidence: number;
  probabilities: Record<string, number>;
}

export function fablePredict(features: Record<string, number>): FablePrediction {
  const names = MODEL.feature_names;
  const scaled = new Float64Array(names.length);
  for (let i = 0; i < names.length; i++) {
    const raw = features[names[i]] ?? 0;
    const std = MODEL.scaler_std[i] === 0 ? 1 : MODEL.scaler_std[i];
    const value = Number.isFinite(raw) ? raw : 0;
    scaled[i] = (value - MODEL.scaler_mean[i]) / std;
  }
  return fablePredictFromScaled(scaled);
}

/** Inferens direkt på en redan z-skalad featurevektor (även parity-harness). */
export function fablePredictFromScaled(scaled: Float64Array): FablePrediction {
  const rawScores = new Float64Array(N_CLASSES);
  for (let c = 0; c < N_CLASSES; c++) rawScores[c] = MODEL.baseline[c];

  const trees = MODEL.trees;
  for (let t = 0; t < trees.length; t++) {
    const tree = trees[t];
    let node = tree[0];
    while (node.length !== 1) {
      node = tree[scaled[node[0]] <= node[1] ? node[2] : node[3]];
    }
    rawScores[t % N_CLASSES] += node[0];
  }

  // Softmax
  let maxRaw = rawScores[0];
  for (let c = 1; c < N_CLASSES; c++) if (rawScores[c] > maxRaw) maxRaw = rawScores[c];
  let sum = 0;
  const exps = new Float64Array(N_CLASSES);
  for (let c = 0; c < N_CLASSES; c++) {
    exps[c] = Math.exp(rawScores[c] - maxRaw);
    sum += exps[c];
  }

  const probabilities: Record<string, number> = {};
  let bestIdx = 0;
  let bestProb = 0;
  for (let c = 0; c < N_CLASSES; c++) {
    const prob = exps[c] / sum;
    probabilities[MODEL.labels[c]] = prob;
    if (prob > bestProb) { bestProb = prob; bestIdx = c; }
  }

  return { label: MODEL.labels[bestIdx], confidence: bestProb, probabilities };
}
