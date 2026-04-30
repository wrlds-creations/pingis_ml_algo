export type RfNode = number[];

export interface RfJsonModel {
  labels: string[];
  feature_names: string[];
  scaler_mean: number[];
  scaler_std: number[];
  trees: RfNode[][];
}

export interface RfPrediction {
  label: string;
  confidence: number;
  probabilities: Record<string, number>;
}

function isLeaf(node: RfNode, nClasses: number): boolean {
  if (node.length !== nClasses) return node.length !== 4;
  let sum = 0;
  for (let i = 0; i < node.length; i++) {
    if (node[i] < 0 || node[i] > 1) return false;
    sum += node[i];
  }
  return Math.abs(sum - 1.0) < 0.01;
}

function traverseTree(tree: RfNode[], scaledFeatures: Float64Array, nClasses: number): number[] {
  let idx = 0;
  while (!isLeaf(tree[idx], nClasses)) {
    const node = tree[idx];
    idx = scaledFeatures[node[0]] <= node[1] ? node[2] : node[3];
  }
  return tree[idx];
}

export function predictWithRfModel(
  model: RfJsonModel,
  features: Record<string, number>,
): RfPrediction {
  const names = model.feature_names;
  const raw = new Float64Array(names.length);
  for (let i = 0; i < names.length; i++) raw[i] = features[names[i]] ?? 0;

  const scaled = new Float64Array(raw.length);
  for (let i = 0; i < raw.length; i++) {
    const std = model.scaler_std[i] === 0 ? 1 : model.scaler_std[i];
    scaled[i] = (raw[i] - model.scaler_mean[i]) / std;
  }

  const nClasses = model.labels.length;
  const probSum = new Float64Array(nClasses);
  for (const tree of model.trees) {
    const proba = traverseTree(tree, scaled, nClasses);
    for (let c = 0; c < nClasses; c++) probSum[c] += proba[c];
  }

  const nTrees = model.trees.length;
  let maxProb = 0;
  let maxIdx = 0;
  const probabilities: Record<string, number> = {};
  for (let c = 0; c < nClasses; c++) {
    const probability = probSum[c] / nTrees;
    probabilities[model.labels[c]] = Math.round(probability * 1000) / 1000;
    if (probability > maxProb) {
      maxProb = probability;
      maxIdx = c;
    }
  }

  return {
    label: model.labels[maxIdx],
    confidence: Math.round(maxProb * 1000) / 1000,
    probabilities,
  };
}
