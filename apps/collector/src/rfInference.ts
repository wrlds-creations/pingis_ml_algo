/**
 * rfInference.ts
 *
 * Kör RandomForest-inferens från JSON-exporterad modell.
 * Modellformat: { labels, scaler_mean, scaler_std, trees }
 * Varje träd är en flat lista av noder:
 *   Intern nod: [feature_idx, threshold, left_child, right_child]  (length 4)
 *   Löv-nod:    [p0, p1, p2, p3, ...]  (length == n_classes)
 */

import modelJson from './models/audio_model.json';

type InternalNode = [number, number, number, number];
type LeafNode     = number[];
type Node         = InternalNode | LeafNode;

const MODEL = modelJson as {
  labels:       string[];
  scaler_mean:  number[];
  scaler_std:   number[];
  trees:        Node[][];
};

const N_CLASSES = MODEL.labels.length;

// ── Hjälp: är noden ett löv? ───────────────────────────────────────────────────
// Interna noder: [feature_idx, threshold, left_child, right_child]
//   → left_child och right_child är alltid ≥ 1 (node-index i trädet)
// Löv-noder: [p0, p1, ..., pN] med alla pi ∈ [0,1] och summa ≈ 1

function isLeaf(node: Node): node is LeafNode {
  if (node.length !== N_CLASSES) return node.length !== 4;
  // N_CLASSES == 4: interna noder har child-indices ≥ 1 (heltal),
  // löv har sannolikheter i [0,1] som summerar till ≈1
  let sum = 0;
  for (let i = 0; i < node.length; i++) {
    if (node[i] < 0 || node[i] > 1) return false;
    sum += node[i];
  }
  return Math.abs(sum - 1.0) < 0.01;
}

// ── Traversera ett träd ───────────────────────────────────────────────────────

function traverseTree(tree: Node[], scaledFeatures: Float64Array): number[] {
  let idx = 0;
  while (!isLeaf(tree[idx])) {
    const node = tree[idx] as InternalNode;
    idx = scaledFeatures[node[0]] <= node[1] ? node[2] : node[3];
  }
  return tree[idx] as LeafNode;
}

// ── Huvud-funktion ─────────────────────────────────────────────────────────────

export interface Prediction {
  label:         string;
  confidence:    number;
  probabilities: Record<string, number>;
}

export function rfPredict(features: Record<string, number>): Prediction {
  // Bygg feature-vektor i rätt ordning (samma som träning)
  const keys = Object.keys(features);
  const raw  = new Float64Array(keys.length);
  for (let i = 0; i < keys.length; i++) raw[i] = features[keys[i]];

  // Standardisera (z-score) med sparad scaler
  const scaled = new Float64Array(raw.length);
  for (let i = 0; i < raw.length; i++) {
    scaled[i] = (raw[i] - MODEL.scaler_mean[i]) / MODEL.scaler_std[i];
  }

  // Samla röster från alla 200 träd
  const probSum = new Float64Array(N_CLASSES);
  for (const tree of MODEL.trees) {
    const proba = traverseTree(tree, scaled);
    for (let c = 0; c < N_CLASSES; c++) probSum[c] += proba[c];
  }

  // Normalisera till sannolikheter
  const nTrees = MODEL.trees.length;
  let maxProb = 0, maxIdx = 0;
  const probabilities: Record<string, number> = {};
  for (let c = 0; c < N_CLASSES; c++) {
    const p = probSum[c] / nTrees;
    probabilities[MODEL.labels[c]] = Math.round(p * 1000) / 1000;
    if (p > maxProb) { maxProb = p; maxIdx = c; }
  }

  return {
    label:         MODEL.labels[maxIdx],
    confidence:    Math.round(maxProb * 1000) / 1000,
    probabilities,
  };
}
