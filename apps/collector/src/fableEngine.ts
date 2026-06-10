/**
 * fableEngine.ts
 *
 * Räknemotor för Fable-läget. Speglar replay_nr_live.py:s beslutslogik
 * (den valda "känsliga" profilen, se noise_robust/RESULTS.md):
 *
 *   features -> HistGB-modell -> argmax == racket_bounce && conf >= 0.5
 *   -> merge-fönster 120 ms (mot senast räknade)
 *   -> grupp-fönster 80 ms (mot gruppstart)
 *   -> eko-gate: inom (merge, 300] ms efter senast räknade studs förkastas
 *      detektionen om dess onset-frame-RMS <= 0.6 x den räknades RMS
 *      (racket-skrammel/efterklang, inte en ny studs).
 *
 * Tidsstämplar: native onset_time_ms (inte JS-klockan) så att JS-latens
 * inte påverkar fönsterlogiken.
 */

import { extractFableFeatures } from './nrFeatures';
import { fablePredict, FablePrediction } from './hgbRuntime';

export interface FableEngineConfig {
  confidence: number;
  mergeMs: number;
  groupMs: number;
  echoMs: number;
  echoRatio: number;
}

export const FABLE_DEFAULT_CONFIG: FableEngineConfig = {
  confidence: 0.5,
  mergeMs: 120,
  groupMs: 80,
  echoMs: 300,
  echoRatio: 0.6,
};

export type FableRejectReason =
  | 'not_racket'
  | 'low_confidence'
  | 'merge_window'
  | 'group_window'
  | 'echo_window';

export interface FableDetectionResult {
  counted: boolean;
  rejectReason?: FableRejectReason;
  prediction: FablePrediction;
  featureMs: number;
  predictMs: number;
}

interface CountedEvent {
  tsMs: number;
  frameRms: number;
}

export class FableCounter {
  private config: FableEngineConfig;
  private lastCounted: CountedEvent | null = null;
  private groupStartMs: number | null = null;

  constructor(config: Partial<FableEngineConfig> = {}) {
    this.config = { ...FABLE_DEFAULT_CONFIG, ...config };
  }

  reset(): void {
    this.lastCounted = null;
    this.groupStartMs = null;
  }

  setConfidence(value: number): void {
    this.config.confidence = value;
  }

  /**
   * Klassificera ett 300 ms-klipp och avgör om det räknas som racketstuds.
   * `onsetTimeMs` = native onset-tid, `frameRms` = native onset-frame-RMS.
   */
  process(pcm: Float32Array, onsetTimeMs: number, frameRms: number): FableDetectionResult {
    const t0 = Date.now();
    const features = extractFableFeatures(pcm);
    const t1 = Date.now();
    const prediction = fablePredict(features);
    const t2 = Date.now();

    const result: FableDetectionResult = {
      counted: false,
      prediction,
      featureMs: t1 - t0,
      predictMs: t2 - t1,
    };

    if (prediction.label !== 'racket_bounce') {
      result.rejectReason = 'not_racket';
      return result;
    }
    if (prediction.confidence < this.config.confidence) {
      result.rejectReason = 'low_confidence';
      return result;
    }
    if (this.lastCounted !== null) {
      const sinceCounted = onsetTimeMs - this.lastCounted.tsMs;
      if (sinceCounted <= this.config.mergeMs) {
        result.rejectReason = 'merge_window';
        return result;
      }
      if (this.groupStartMs !== null && onsetTimeMs - this.groupStartMs <= this.config.groupMs) {
        result.rejectReason = 'group_window';
        return result;
      }
      if (
        this.config.echoMs > this.config.mergeMs &&
        sinceCounted <= this.config.echoMs &&
        frameRms <= this.config.echoRatio * this.lastCounted.frameRms
      ) {
        result.rejectReason = 'echo_window';
        return result;
      }
    }

    this.lastCounted = { tsMs: onsetTimeMs, frameRms };
    this.groupStartMs = onsetTimeMs;
    result.counted = true;
    return result;
  }
}
