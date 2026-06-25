import modelJson from './models/video_stroke_model.json';
import { VIDEO_STROKE_FEATURE_NAMES, VIDEO_STROKE_FEATURE_SPEC } from './videoStrokeFeatures';
import type { VideoStrokePredictionLabel } from './types';

type InternalNode = [number, number, number, number];
type LeafNode = number[];
type Node = InternalNode | LeafNode;

interface VideoStrokeModel {
  trained: boolean;
  model_version: string;
  feature_spec: typeof VIDEO_STROKE_FEATURE_SPEC;
  labels: string[];
  feature_names: string[];
  scaler_mean: number[];
  scaler_std: number[];
  trees: Node[][];
}

export interface VideoStrokePrediction {
  label: VideoStrokePredictionLabel;
  raw_label?: string;
  confidence: number;
  probabilities: Record<string, number>;
  model_version: string;
  status: 'ok' | 'uncertain' | 'model_missing';
}

const MODEL = modelJson as VideoStrokeModel;
const CONFIDENCE_MIN = 0.58;

export function hasTrainedVideoStrokeModel(): boolean {
  return !!MODEL.trained && MODEL.trees.length > 0 && MODEL.feature_names.length > 0;
}

export function videoStrokeModelVersion(): string {
  return MODEL.model_version ?? 'video_stroke_untrained';
}

function isLeaf(node: Node, classCount: number): node is LeafNode {
  if (node.length !== classCount) return node.length !== 4;
  let probabilitySum = 0;
  for (const value of node) {
    if (value < 0 || value > 1) return false;
    probabilitySum += value;
  }
  return Math.abs(probabilitySum - 1.0) < 0.01;
}

function traverseTree(tree: Node[], scaledFeatures: Float64Array, classCount: number): number[] {
  let nodeIndex = 0;
  while (!isLeaf(tree[nodeIndex], classCount)) {
    const node = tree[nodeIndex] as InternalNode;
    nodeIndex = scaledFeatures[node[0]] <= node[1] ? node[2] : node[3];
  }
  return tree[nodeIndex] as LeafNode;
}

export function predictVideoStroke(features: Record<string, number>): VideoStrokePrediction {
  if (!hasTrainedVideoStrokeModel()) {
    return {
      label: 'uncertain',
      confidence: 0,
      probabilities: {},
      model_version: videoStrokeModelVersion(),
      status: 'model_missing',
    };
  }

  const featureNames = MODEL.feature_names.length > 0
    ? MODEL.feature_names
    : [...VIDEO_STROKE_FEATURE_NAMES];
  const classCount = MODEL.labels.length;
  const rawFeatures = new Float64Array(featureNames.length);
  for (let featureIndex = 0; featureIndex < featureNames.length; featureIndex += 1) {
    rawFeatures[featureIndex] = features[featureNames[featureIndex]] ?? 0;
  }

  const scaledFeatures = new Float64Array(rawFeatures.length);
  for (let featureIndex = 0; featureIndex < rawFeatures.length; featureIndex += 1) {
    const scale = MODEL.scaler_std[featureIndex] || 1;
    scaledFeatures[featureIndex] = (rawFeatures[featureIndex] - MODEL.scaler_mean[featureIndex]) / scale;
  }

  const probabilitySums = new Float64Array(classCount);
  for (const tree of MODEL.trees) {
    const probabilities = traverseTree(tree, scaledFeatures, classCount);
    for (let classIndex = 0; classIndex < classCount; classIndex += 1) {
      probabilitySums[classIndex] += probabilities[classIndex] ?? 0;
    }
  }

  let bestIndex = 0;
  let bestProbability = 0;
  const probabilities: Record<string, number> = {};
  for (let classIndex = 0; classIndex < classCount; classIndex += 1) {
    const probability = probabilitySums[classIndex] / Math.max(1, MODEL.trees.length);
    probabilities[MODEL.labels[classIndex]] = Math.round(probability * 1000) / 1000;
    if (probability > bestProbability) {
      bestProbability = probability;
      bestIndex = classIndex;
    }
  }

  const rawLabel = MODEL.labels[bestIndex];
  const confidence = Math.round(bestProbability * 1000) / 1000;
  const label = confidence >= CONFIDENCE_MIN && (rawLabel === 'forehand' || rawLabel === 'backhand')
    ? rawLabel
    : 'uncertain';

  return {
    label,
    raw_label: rawLabel,
    confidence,
    probabilities,
    model_version: videoStrokeModelVersion(),
    status: label === 'uncertain' ? 'uncertain' : 'ok',
  };
}
