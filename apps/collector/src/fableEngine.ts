/**
 * fableEngine.ts
 *
 * Räknemotor för Fable-läget. Speglar replay_nr_live.py:s beslutslogik
 * (den valda "känsliga" profilen, se noise_robust/RESULTS.md) med tre
 * device-iterationer från Loves livetest 2026-06-10:
 *
 *   1. Fönsterkontroller (merge/grupp/eko) körs FÖRE feature-extraktionen —
 *      de beror bara på onset-tid + frame-RMS, och features kostade 365 ms
 *      per kandidat på Motorola. Utfallet är ekvivalent: en fönster-avvisad
 *      kandidat uppdaterar aldrig räknartillståndet oavsett klass.
 *   2. Inaktualitetsspärr: kandidater vars onset är äldre än `staleMs` när
 *      de når JS-tråden (kö-eftersläpning) släpps utan klassificering så
 *      att räknaren kommer ikapp i stället för att visa siffror sekunder
 *      för sent.
 *   3. Adaptiv konfidens: i tyst miljö räcker 0.65 (äkta studsar ligger
 *      ~1.0), men när bakgrunden är ljudlig (rå förstudsnivå
 *      nr_bg_rms_db >= -42 dB: musik/prat/sorl) krävs 0.9 — Kent-FP:arna
 *      i livetestet låg på 0.51-0.88 medan äkta studsar i studsrytm låg
 *      >= 0.9 även med musik på.
 *
 * Tidsstämplar: native onset_time_ms (inte JS-klockan) så att JS-latens
 * inte påverkar fönsterlogiken.
 */

import { extractFableFeatures } from './nrFeatures';
import { fablePredict, FablePrediction } from './hgbRuntime';

export interface FableEngineConfig {
  quietConfidence: number;
  loudConfidence: number;
  loudBgDb: number;
  mergeMs: number;
  /** Allt inom detta fönster efter en räknad studs är SAMMA studs.
   *  Fysik: två äkta studsar < 250 ms isär kräver studshöjd < 8 cm.
   *  Loves 2026-06-11-sessioner visade dubbelräkningar med 121-140 ms
   *  gap där den ANDRA träffen är 1.2-17x starkare (förljud + smäll);
   *  ankaret uppdateras därför till den starkaste träffen i fönstret. */
  sameBounceMs: number;
  groupMs: number;
  echoMs: number;
  echoRatio: number;
  staleMs: number;
}

export const FABLE_DEFAULT_CONFIG: FableEngineConfig = {
  quietConfidence: 0.65,
  loudConfidence: 0.9,
  loudBgDb: -42,
  mergeMs: 120,
  sameBounceMs: 250,
  groupMs: 80,
  echoMs: 300,
  echoRatio: 0.6,
  staleMs: 1500,
};

export type FableRejectReason =
  | 'not_racket'
  | 'low_confidence'
  | 'low_confidence_loud_bg'
  | 'same_bounce'
  | 'merge_window'
  | 'group_window'
  | 'echo_window'
  | 'stale_backlog';

export interface FableDetectionResult {
  counted: boolean;
  rejectReason?: FableRejectReason;
  prediction?: FablePrediction;
  bgMode?: 'quiet' | 'loud';
  bgRmsDb?: number;
  featureMs: number;
  predictMs: number;
  /** Studshöjd i meter ur fritt fall mellan två på varandra följande
   *  räknade studsar: h = g * (dt/2)^2 / 2. Sätts bara när gapet är
   *  fysikaliskt rimligt för upp/ner-studsande (250-1500 ms). */
  bounceHeightM?: number;
  bounceGapMs?: number;
}

const GRAVITY = 9.82;
const HEIGHT_MIN_GAP_MS = 250;   // < 250 ms = eko/dubbelträff, inte en hel flygbana
const HEIGHT_MAX_GAP_MS = 1500;  // > 1.5 s = bollen var inte i kontinuerligt studs

/** h = g * (dt/2)^2 / 2 — bollen stiger halva tiden, faller halva. */
export function bounceHeightMeters(gapMs: number): number {
  const tUp = gapMs / 2000; // sekunder upp
  return (GRAVITY * tUp * tUp) / 2;
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

  /**
   * Klassificera ett 300 ms-klipp och avgör om det räknas som racketstuds.
   * `onsetTimeMs` = native onset-tid, `frameRms` = native onset-frame-RMS,
   * `nowMs` = JS-klocka vid bearbetning (för inaktualitetsspärren).
   */
  process(pcm: Float32Array, onsetTimeMs: number, frameRms: number, nowMs?: number): FableDetectionResult {
    const result: FableDetectionResult = { counted: false, featureMs: 0, predictMs: 0 };

    // Inaktualitetsspärr före allt annat: gammal kö-kandidat hjälper ingen.
    if (nowMs !== undefined && nowMs - onsetTimeMs > this.config.staleMs) {
      result.rejectReason = 'stale_backlog';
      return result;
    }

    // Fönsterkontroller FÖRE features: beror inte på klassificeringen och
    // sparar hela extraktionskostnaden för täta kandidater.
    if (this.lastCounted !== null) {
      const sinceCounted = onsetTimeMs - this.lastCounted.tsMs;
      const rmsRatio = frameRms / Math.max(this.lastCounted.frameRms, 1e-9);
      if (sinceCounted <= this.config.sameBounceMs) {
        // Inom 250 ms avgör styrkeförhållandet vad det är:
        //  - klart STARKARE (>=1.1x): förljud+smäll = samma studs; flytta
        //    ankaret till smällen (Loves dubbelräkningar 2026-06-11 hade
        //    alla detta mönster: 121-140 ms gap, andra träffen 1.2-17x).
        //  - klart SVAGARE (<=0.6x): eko/skrammel, släng.
        //  - LIKVÄRDIG styrka och gap >= 150 ms: äkta snabb studs (8 cm
        //    studs = 255 ms, 3 cm = 156 ms - snabbt drillande är verkligt)
        //    -> släpp igenom till klassificering.
        //  - likvärdig men gap < 150 ms (< 2.8 cm flygbana): samma studs.
        if (rmsRatio >= 1.1) {
          this.lastCounted = { tsMs: onsetTimeMs, frameRms };
          this.groupStartMs = onsetTimeMs;
          result.rejectReason = 'same_bounce';
          return result;
        }
        if (rmsRatio <= this.config.echoRatio) {
          result.rejectReason = 'echo_window';
          return result;
        }
        if (sinceCounted < 150) {
          result.rejectReason = 'same_bounce';
          return result;
        }
        // likvärdig styrka, 150-250 ms: behandla som äkta snabb studs.
      } else {
        if (this.groupStartMs !== null && onsetTimeMs - this.groupStartMs <= this.config.groupMs) {
          result.rejectReason = 'group_window';
          return result;
        }
        if (sinceCounted <= this.config.echoMs && rmsRatio <= this.config.echoRatio) {
          result.rejectReason = 'echo_window';
          return result;
        }
      }
    }

    const t0 = Date.now();
    const features = extractFableFeatures(pcm);
    const t1 = Date.now();
    const prediction = fablePredict(features);
    const t2 = Date.now();
    result.featureMs = t1 - t0;
    result.predictMs = t2 - t1;
    result.prediction = prediction;

    const bgRmsDb = features.nr_bg_rms_db ?? -100;
    const loud = bgRmsDb >= this.config.loudBgDb;
    result.bgMode = loud ? 'loud' : 'quiet';
    result.bgRmsDb = bgRmsDb;
    const confidenceThreshold = loud ? this.config.loudConfidence : this.config.quietConfidence;

    if (prediction.label !== 'racket_bounce') {
      result.rejectReason = 'not_racket';
      return result;
    }
    if (prediction.confidence < confidenceThreshold) {
      result.rejectReason = loud ? 'low_confidence_loud_bg' : 'low_confidence';
      return result;
    }

    if (this.lastCounted !== null) {
      const gapMs = onsetTimeMs - this.lastCounted.tsMs;
      if (gapMs >= HEIGHT_MIN_GAP_MS && gapMs <= HEIGHT_MAX_GAP_MS) {
        result.bounceGapMs = gapMs;
        result.bounceHeightM = bounceHeightMeters(gapMs);
      }
    }
    this.lastCounted = { tsMs: onsetTimeMs, frameRms };
    this.groupStartMs = onsetTimeMs;
    result.counted = true;
    return result;
  }
}
