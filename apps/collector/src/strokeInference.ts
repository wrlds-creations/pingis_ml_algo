import hitModelJson from './models/stroke_hit_model.json';
import strokeTypeModelJson from './models/stroke_type_model.json';
import { predictWithRfModel } from './rfRuntime';
import type { BounceSide, StrokeLabel } from './types';

const HIT_MODEL = hitModelJson as any;
const STROKE_TYPE_MODEL = strokeTypeModelJson as any;

export interface StrokePrediction {
  hit_label: StrokeLabel;
  hit_confidence: number;
  hit_probabilities: Record<string, number>;
  stroke_side: BounceSide;
  stroke_confidence: number;
  stroke_probabilities: Record<string, number>;
}

export function predictStroke(features: Record<string, number>): StrokePrediction {
  const hit = predictWithRfModel(HIT_MODEL, features);
  const stroke = predictWithRfModel(STROKE_TYPE_MODEL, features);

  const side = stroke.label === 'forehand' || stroke.label === 'backhand'
    ? stroke.label
    : 'uncertain';

  return {
    hit_label: hit.label === 'hit' ? 'hit' : 'swing_miss',
    hit_confidence: hit.confidence,
    hit_probabilities: hit.probabilities,
    stroke_side: side,
    stroke_confidence: stroke.confidence,
    stroke_probabilities: stroke.probabilities,
  };
}
