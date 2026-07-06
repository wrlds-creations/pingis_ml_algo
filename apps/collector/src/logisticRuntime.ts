export interface LogisticBinaryJsonModel {
  bias: number;
  feature_names: string[];
  labels: [string, string];
  metadata?: Record<string, unknown>;
  model_type: 'logistic_binary_v1';
  positive_label: string;
  scaler_mean: number[];
  scaler_std: number[];
  threshold: number;
  weights: number[];
}

export interface LogisticBinaryPrediction {
  confidence: number;
  label: string;
  probability: number;
  threshold: number;
}

function sigmoid(value: number): number {
  const clipped = Math.max(-40, Math.min(40, value));
  return 1 / (1 + Math.exp(-clipped));
}

export function predictWithLogisticBinaryModel(
  model: LogisticBinaryJsonModel,
  features: Record<string, number>,
  thresholdOverride?: number,
): LogisticBinaryPrediction {
  let score = model.bias;
  for (let i = 0; i < model.feature_names.length; i += 1) {
    const name = model.feature_names[i];
    const value = features[name] ?? 0;
    const raw = Number.isFinite(value) ? value : 0;
    const std = model.scaler_std[i] === 0 ? 1 : model.scaler_std[i];
    const scaled = (raw - model.scaler_mean[i]) / std;
    score += scaled * model.weights[i];
  }

  const probability = sigmoid(score);
  const threshold = typeof thresholdOverride === 'number' && Number.isFinite(thresholdOverride)
    ? Math.max(0, Math.min(1, thresholdOverride))
    : model.threshold;
  const positive = probability >= threshold;
  return {
    confidence: positive ? probability : 1 - probability,
    label: positive
      ? model.positive_label
      : model.labels.find(label => label !== model.positive_label) ?? model.labels[0],
    probability,
    threshold,
  };
}
