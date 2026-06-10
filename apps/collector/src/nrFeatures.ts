/**
 * nrFeatures.ts
 *
 * TS-port av de 21 brusrobusta `nr_`-featurerna från
 * skills/pingis-audio-classification/scripts/noise_robust/nr_features.py
 * (extract_robust_features). Beräknas på det RÅA 300 ms-klippet
 * (6 615 samples @ 22 050 Hz), INTE på den 1 s-paddade buffern.
 *
 * Paritetskrav: samma STFT-geometri (n_fft=512, hop=128, periodisk Hann,
 * center=False), samma bandgränser, samma sosfiltfilt-bandpass (Butterworth
 * ordning 4, 1.5–7 kHz) med scipy-identisk udda kantförlängning och
 * initialtillstånd, samma PCEN-rekursion som librosa.pcen med default-
 * parametrar. Verifieras av parity-harnessen i
 * skills/pingis-audio-classification/scripts/noise_robust/check_fable_ts_parity.js
 */

import { extractFeatures } from './audioFeatures';

const SR = 22_050;
const CLIP_SAMPLES = 6_615;          // 100 ms pre + 200 ms post onset
const N_FFT = 512;
const HOP = 128;
const N_BINS = N_FFT / 2 + 1;        // 257
const N_FRAMES = 1 + Math.floor((CLIP_SAMPLES - N_FFT) / HOP); // 48
const BACKGROUND_END_SAMPLE = 1_764; // första 80 ms
const EPS = 1e-10;
const ENV_SMOOTH = 110;              // ~5 ms glidande medel
const N_MELS_PCEN = 40;

// Periodisk Hann (librosa stft default: fftbins=True => /N, inte /(N-1)).
const HANN512 = new Float64Array(N_FFT);
for (let i = 0; i < N_FFT; i++) {
  HANN512[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N_FFT);
}

const FREQS = new Float64Array(N_BINS);
for (let k = 0; k < N_BINS; k++) FREQS[k] = (k * SR) / N_FFT;

// Bandmaskar: low/mid/high [lo, hi), vhigh [6000, 11025] inkl. Nyquist.
const BANDS: [string, number, number, boolean][] = [
  ['low', 200, 800, false],
  ['mid', 800, 2500, false],
  ['high', 2500, 6000, false],
  ['vhigh', 6000, 11_025, true],
];

// Butterworth ordning 4 bandpass 1500–7000 Hz @ 22050 (scipy butter, sos).
// Rad: [b0, b1, b2, a1, a2] (a0 = 1).
const BP_SOS: number[][] = [
  [0.09331299315653971, 0.18662598631307942, 0.09331299315653971, 0.19676635870899223, 0.09988498828008321],
  [1.0, -2.0, 1.0, -1.1925622397360882, 0.39614858444118883],
  [1.0, 2.0, 1.0, 0.6137123805223851, 0.5737584318188238],
  [1.0, -2.0, 1.0, -1.6102285405185548, 0.7781414737694868],
];

// ── FFT (radix-2, lokala buffrar för 512) ────────────────────────────────────

const _re512 = new Float64Array(N_FFT);
const _im512 = new Float64Array(N_FFT);

function _fft512(): void {
  const n = N_FFT;
  let j = 0;
  for (let i = 1; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = _re512[i]; _re512[i] = _re512[j]; _re512[j] = t;
      t = _im512[i]; _im512[i] = _im512[j]; _im512[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const ang = (-2 * Math.PI) / len;
    const wr0 = Math.cos(ang);
    const wi0 = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let wr = 1.0, wi = 0.0;
      for (let k = 0; k < half; k++) {
        const ur = _re512[i + k], ui = _im512[i + k];
        const vr = _re512[i + k + half] * wr - _im512[i + k + half] * wi;
        const vi = _re512[i + k + half] * wi + _im512[i + k + half] * wr;
        _re512[i + k] = ur + vr;
        _im512[i + k] = ui + vi;
        _re512[i + k + half] = ur - vr;
        _im512[i + k + half] = ui - vi;
        const nwr = wr * wr0 - wi * wi0;
        wi = wr * wi0 + wi * wr0;
        wr = nwr;
      }
    }
  }
}

// ── sosfilt / sosfiltfilt (scipy-paritet) ────────────────────────────────────

/** lfilter_zi för en biquad [b0,b1,b2], [1,a1,a2] (scipy closed form). */
function biquadZi(b0: number, b1: number, b2: number, a1: number, a2: number): [number, number] {
  // IminusA = [[1+a1, -1], [a2, 1]], B = [b1 - a1*b0, b2 - a2*b0]
  const B0 = b1 - a1 * b0;
  const B1 = b2 - a2 * b0;
  const det = (1 + a1) * 1 - (-1) * a2; // = 1 + a1 + a2
  const zi0 = (B0 * 1 - (-1) * B1) / det;
  const zi1 = ((1 + a1) * B1 - a2 * B0) / det;
  return [zi0, zi1];
}

/** sosfilt_zi: per-sektion zi skalad med kumulativ DC-gain (scipy). */
function sosfiltZi(): number[][] {
  const zi: number[][] = [];
  let scale = 1.0;
  for (const [b0, b1, b2, a1, a2] of BP_SOS) {
    const [z0, z1] = biquadZi(b0, b1, b2, a1, a2);
    zi.push([z0 * scale, z1 * scale]);
    scale *= (b0 + b1 + b2) / (1 + a1 + a2);
  }
  return zi;
}

/** sosfilt med givna initialtillstånd (direct form II transposed). Muterar zi. */
function sosfiltInPlace(x: Float64Array, zi: number[][]): Float64Array {
  const y = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) {
    let v = x[i];
    for (let s = 0; s < BP_SOS.length; s++) {
      const [b0, b1, b2, a1, a2] = BP_SOS[s];
      const st = zi[s];
      const out = b0 * v + st[0];
      st[0] = b1 * v - a1 * out + st[1];
      st[1] = b2 * v - a2 * out;
      v = out;
    }
    y[i] = v;
  }
  return y;
}

/**
 * scipy.signal.sosfiltfilt med default padtype='odd':
 * padlen = 3 * (2*n_sections + 1) = 27 för denna SOS.
 */
export function sosFiltFilt(x: Float64Array): Float64Array {
  const padlen = 3 * (2 * BP_SOS.length + 1); // 27
  const n = x.length;
  const ext = new Float64Array(n + 2 * padlen);
  // Udda förlängning: 2*x[0] - x[padlen..1], sedan x, sedan 2*x[n-1] - x[n-2..n-1-padlen]
  for (let i = 0; i < padlen; i++) ext[i] = 2 * x[0] - x[padlen - i];
  ext.set(x, padlen);
  for (let i = 0; i < padlen; i++) ext[padlen + n + i] = 2 * x[n - 1] - x[n - 2 - i];

  const ziTemplate = sosfiltZi();
  // Framåt
  const ziF = ziTemplate.map(s => [s[0] * ext[0], s[1] * ext[0]]);
  const fwd = sosfiltInPlace(ext, ziF);
  // Bakåt
  const rev = new Float64Array(fwd.length);
  for (let i = 0; i < fwd.length; i++) rev[i] = fwd[fwd.length - 1 - i];
  const ziB = ziTemplate.map(s => [s[0] * rev[0], s[1] * rev[0]]);
  const bwd = sosfiltInPlace(rev, ziB);
  // Vänd tillbaka + trimma
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = bwd[bwd.length - 1 - padlen - i];
  return out;
}

// ── Hjälpfunktioner ──────────────────────────────────────────────────────────

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 === 1 ? sorted[mid] : 0.5 * (sorted[mid - 1] + sorted[mid]);
}

function spectralFlatness(power: Float64Array | number[]): number {
  let logSum = 0;
  let arith = 0;
  const n = power.length;
  for (let i = 0; i < n; i++) {
    const p = power[i] + EPS;
    logSum += Math.log(p);
    arith += p;
  }
  const geo = Math.exp(logSum / n);
  return geo / (arith / n + EPS);
}

// Slaney mel-filterbank för PCEN (n_mels=40, n_fft=512), samma formel som
// audioFeatures.ts men med lokala parametrar.
const MEL40_FB = (() => {
  const fSp = 200.0 / 3.0;
  const logHz = 1000.0;
  const logMel = logHz / fSp;
  const logStep = Math.log(6.4) / 27.0;
  const hzToMel = (f: number) => (f < logHz ? f / fSp : logMel + Math.log(f / logHz) / logStep);
  const melToHz = (m: number) => (m < logMel ? fSp * m : logHz * Math.exp(logStep * (m - logMel)));
  const melMin = hzToMel(0);
  const melMax = hzToMel(SR / 2);
  const pts = new Float64Array(N_MELS_PCEN + 2);
  for (let i = 0; i <= N_MELS_PCEN + 1; i++) {
    pts[i] = melToHz(melMin + ((melMax - melMin) * i) / (N_MELS_PCEN + 1));
  }
  const fb = new Float64Array(N_MELS_PCEN * N_BINS);
  for (let m = 0; m < N_MELS_PCEN; m++) {
    const lo = pts[m], ctr = pts[m + 1], hi = pts[m + 2];
    const bw = hi - lo;
    for (let k = 0; k < N_BINS; k++) {
      const f = FREQS[k];
      let v = 0;
      if (f > lo && f < ctr) v = (f - lo) / (ctr - lo);
      else if (f >= ctr && f < hi) v = (hi - f) / (hi - ctr);
      fb[m * N_BINS + k] = (v * 2.0) / bw;
    }
  }
  return fb;
})();

// Sparse nollskilda intervall per mel-band (samma optimering som i
// audioFeatures.ts — identisk summa, mycket färre multiplikationer).
const MEL40_RANGES = (() => {
  const ranges = new Int32Array(N_MELS_PCEN * 2);
  for (let m = 0; m < N_MELS_PCEN; m++) {
    let start = -1;
    let end = -1;
    for (let k = 0; k < N_BINS; k++) {
      if (MEL40_FB[m * N_BINS + k] !== 0) {
        if (start < 0) start = k;
        end = k + 1;
      }
    }
    ranges[m * 2] = start < 0 ? 0 : start;
    ranges[m * 2 + 1] = end < 0 ? 0 : end;
  }
  return ranges;
})();

// PCEN-konstanter (librosa defaults: gain=0.98, bias=2, power=0.5,
// time_constant=0.4, eps=1e-6, S skalad med 2^31).
const PCEN_GAIN = 0.98;
const PCEN_BIAS = 2.0;
const PCEN_POWER = 0.5;
const PCEN_EPS = 1e-6;
const PCEN_SCALE = 2 ** 31;
const PCEN_B = (() => {
  const tFrames = (0.4 * SR) / HOP;
  return (Math.sqrt(1 + 4 * tFrames * tFrames) - 1) / (2 * tFrames * tFrames);
})();

// ── Huvudfunktion: 21 nr_-features på rått 300 ms-klipp ──────────────────────

export function extractNrFeatures(clipIn: Float32Array): Record<string, number> {
  // fix_length till exakt 6615 (zeros på slutet om kortare)
  const clip = new Float32Array(CLIP_SAMPLES);
  clip.set(clipIn.subarray(0, Math.min(clipIn.length, CLIP_SAMPLES)));

  const feats: Record<string, number> = {};

  // ── STFT: power-spektrum per frame ─────────────────────────────────────────
  const S = new Float64Array(N_FRAMES * N_BINS); // frame-major
  const mel = new Float64Array(N_FRAMES * N_MELS_PCEN);
  for (let t = 0; t < N_FRAMES; t++) {
    const start = t * HOP;
    _re512.fill(0); _im512.fill(0);
    for (let i = 0; i < N_FFT; i++) _re512[i] = clip[start + i] * HANN512[i];
    _fft512();
    const off = t * N_BINS;
    for (let k = 0; k < N_BINS; k++) {
      S[off + k] = _re512[k] * _re512[k] + _im512[k] * _im512[k];
    }
    // Mel (power) för PCEN — sparse intervall, exakt samma summa.
    for (let m = 0; m < N_MELS_PCEN; m++) {
      let s = 0;
      const fOff = m * N_BINS;
      const kStart = MEL40_RANGES[m * 2];
      const kEnd = MEL40_RANGES[m * 2 + 1];
      for (let k = kStart; k < kEnd; k++) s += MEL40_FB[fOff + k] * S[off + k];
      mel[t * N_MELS_PCEN + m] = s;
    }
  }

  // Frame-grupper
  const bgFrames: number[] = [];
  const impactFrames: number[] = [];
  for (let t = 0; t < N_FRAMES; t++) {
    const start = t * HOP;
    if (start + N_FFT <= BACKGROUND_END_SAMPLE) bgFrames.push(t);
    if (start >= BACKGROUND_END_SAMPLE) impactFrames.push(t);
  }

  // Bandenergier + totalenergi (bins >= 200 Hz) per frame
  const bandEnergy: number[][] = BANDS.map(() => new Array(N_FRAMES).fill(0));
  const totalEnergy = new Float64Array(N_FRAMES);
  for (let t = 0; t < N_FRAMES; t++) {
    const off = t * N_BINS;
    for (let k = 0; k < N_BINS; k++) {
      const f = FREQS[k];
      const p = S[off + k];
      if (f >= 200) totalEnergy[t] += p;
      for (let b = 0; b < BANDS.length; b++) {
        const [, lo, hi, inclusive] = BANDS[b];
        if (f >= lo && (inclusive ? f <= hi : f < hi)) bandEnergy[b][t] += p;
      }
    }
  }

  const bgBand = BANDS.map((_, b) => median(bgFrames.map(t => bandEnergy[b][t])));
  const bgTotal = median(bgFrames.map(t => totalEnergy[t]));

  // Peak-frame: argmax totalenergi bland impact-frames (fallback: alla)
  let p = 0;
  {
    const cands = impactFrames.length ? impactFrames : Array.from({ length: N_FRAMES }, (_, i) => i);
    let best = -Infinity;
    for (const t of cands) {
      if (totalEnergy[t] > best) { best = totalEnergy[t]; p = t; }
    }
  }

  const deltas: number[] = [];
  for (let b = 0; b < BANDS.length; b++) {
    const d = Math.log10(bandEnergy[b][p] + EPS) - Math.log10(bgBand[b] + EPS);
    feats[`nr_band_delta_${BANDS[b][0]}`] = d;
    deltas.push(d);
  }
  let dMax = deltas[0], dArg = 0;
  for (let b = 1; b < deltas.length; b++) {
    if (deltas[b] > dMax) { dMax = deltas[b]; dArg = b; }
  }
  feats.nr_band_delta_max = dMax;
  feats.nr_band_delta_argmax = dArg;

  feats.nr_snr_db_est = 10 * Math.log10((totalEnergy[p] + EPS) / (bgTotal + EPS));

  let bgSq = 0;
  for (let i = 0; i < BACKGROUND_END_SAMPLE; i++) bgSq += clip[i] * clip[i];
  feats.nr_bg_rms_db = 20 * Math.log10(Math.sqrt(bgSq / BACKGROUND_END_SAMPLE) + EPS);

  // Spektral flux i 90..150 ms (frames vars start ligger i intervallet)
  const fluxLo = Math.floor(0.09 * SR);
  const fluxHi = Math.floor(0.15 * SR);
  let maxFlux = 0;
  for (let t = 1; t < N_FRAMES; t++) {
    const start = t * HOP;
    if (start < fluxLo || start > fluxHi) continue;
    let flux = 0;
    const off = t * N_BINS;
    const offPrev = (t - 1) * N_BINS;
    for (let k = 0; k < N_BINS; k++) {
      const d = Math.sqrt(S[off + k]) - Math.sqrt(S[offPrev + k]);
      if (d > 0) flux += d;
    }
    if (flux > maxFlux) maxFlux = flux;
  }
  feats.nr_flux_onset = Math.log10(EPS + maxFlux);

  // Flatness: medelbakgrundsspektrum + peak-frame (alla 257 bins inkl. DC)
  if (bgFrames.length) {
    const meanBg = new Float64Array(N_BINS);
    for (const t of bgFrames) {
      const off = t * N_BINS;
      for (let k = 0; k < N_BINS; k++) meanBg[k] += S[off + k];
    }
    for (let k = 0; k < N_BINS; k++) meanBg[k] /= bgFrames.length;
    feats.nr_bg_flatness = spectralFlatness(meanBg);
  } else {
    feats.nr_bg_flatness = 1.0;
  }
  {
    const off = p * N_BINS;
    const peakSpec = new Float64Array(N_BINS);
    for (let k = 0; k < N_BINS; k++) peakSpec[k] = S[off + k];
    feats.nr_impact_flatness = spectralFlatness(peakSpec);
  }

  // ── Bandpassad envelope (sosfiltfilt, scipy-paritet) ───────────────────────
  const clip64 = new Float64Array(CLIP_SAMPLES);
  for (let i = 0; i < CLIP_SAMPLES; i++) clip64[i] = clip[i];
  const bp = sosFiltFilt(clip64);

  const absBp = new Float64Array(CLIP_SAMPLES);
  let bpPeak = 0, bpSq = 0;
  for (let i = 0; i < CLIP_SAMPLES; i++) {
    const a = Math.abs(bp[i]);
    absBp[i] = a;
    if (a > bpPeak) bpPeak = a;
    bpSq += bp[i] * bp[i];
  }
  const bpRms = Math.sqrt(bpSq / CLIP_SAMPLES);

  // np.convolve(absBp, ones(110)/110, 'same'): fönster [i-55, i+54], /110.
  const env = new Float64Array(CLIP_SAMPLES);
  {
    const prefix = new Float64Array(CLIP_SAMPLES + 1);
    for (let i = 0; i < CLIP_SAMPLES; i++) prefix[i + 1] = prefix[i] + absBp[i];
    const half = ENV_SMOOTH >> 1; // 55
    for (let i = 0; i < CLIP_SAMPLES; i++) {
      const lo = Math.max(0, i - half);
      const hi = Math.min(CLIP_SAMPLES, i + half); // exklusivt: i+54 inklusive
      env[i] = (prefix[hi] - prefix[lo]) / ENV_SMOOTH;
    }
  }

  let envPeakIdx = 0, envPeak = 0;
  for (let i = 0; i < CLIP_SAMPLES; i++) {
    if (env[i] > envPeak) { envPeak = env[i]; envPeakIdx = i; }
  }

  // Attack: 10% -> 90% bakåt från peak (max 50 ms lookback)
  let attackMs = 0;
  if (envPeak > 0) {
    const lookback = Math.floor(0.05 * SR);
    const segStart = Math.max(0, envPeakIdx - lookback);
    const segLen = envPeakIdx + 1 - segStart;
    let idx90: number | null = null;
    let idx10: number | null = null;
    for (let j = segLen - 1; j >= 0; j--) {
      const v = env[segStart + j];
      if (idx90 === null && v <= 0.9 * envPeak) idx90 = j;
      if (v <= 0.1 * envPeak) { idx10 = j; break; }
    }
    if (idx90 !== null && idx10 !== null && idx90 >= idx10) {
      attackMs = ((idx90 - idx10) / SR) * 1000;
    }
  }
  feats.nr_bp_attack_ms = attackMs;

  // Decay till 50 % (cap 150 ms)
  const decayCap = Math.floor(0.15 * SR);
  const decayEnd = Math.min(CLIP_SAMPLES, envPeakIdx + decayCap);
  let decaySamples = decayEnd - envPeakIdx;
  if (envPeak > 0) {
    for (let i = envPeakIdx; i < decayEnd; i++) {
      if (env[i] < 0.5 * envPeak) { decaySamples = i - envPeakIdx; break; }
    }
  }
  feats.nr_bp_decay50_ms = (decaySamples / SR) * 1000;

  feats.nr_bp_crest = bpPeak / (bpRms + EPS);
  let clipPeak = 0;
  for (let i = 0; i < CLIP_SAMPLES; i++) {
    const a = Math.abs(clip[i]);
    if (a > clipPeak) clipPeak = a;
  }
  feats.nr_bp_peak_ratio = bpPeak / (clipPeak + EPS);

  const idx50 = Math.min(envPeakIdx + Math.floor(0.05 * SR), CLIP_SAMPLES - 1);
  const idx100 = Math.min(envPeakIdx + Math.floor(0.1 * SR), CLIP_SAMPLES - 1);
  feats.nr_post_decay_db_50ms = 20 * Math.log10((env[idx50] + EPS) / (envPeak + EPS));
  feats.nr_post_decay_db_100ms = 20 * Math.log10((env[idx100] + EPS) / (envPeak + EPS));

  // ── PCEN (librosa-paritet) ──────────────────────────────────────────────────
  // S_pcen = mel * 2^31; M[t] = b*S[t] + (1-b)*M[t-1], M init via zi = (1-b)
  // (librosa: lfilter([b],[1,b-1], S, zi=lfilter_zi=:(1-b)) => y0 = b*S0 + (1-b)*1)
  {
    const tSeries = new Float64Array(N_FRAMES);
    const M = new Float64Array(N_MELS_PCEN);
    for (let m = 0; m < N_MELS_PCEN; m++) M[m] = 1.0; // zi/(1-b)-ekvivalent: state = (1-b)*M_prev, M_prev init 1.0
    for (let t = 0; t < N_FRAMES; t++) {
      let sum = 0;
      for (let m = 0; m < N_MELS_PCEN; m++) {
        const sVal = mel[t * N_MELS_PCEN + m] * PCEN_SCALE;
        const smooth = PCEN_B * sVal + (1 - PCEN_B) * M[m];
        M[m] = smooth;
        const gainTerm = Math.exp(-PCEN_GAIN * (Math.log(PCEN_EPS) + Math.log1p(smooth / PCEN_EPS)));
        const pOut = Math.pow(sVal * gainTerm + PCEN_BIAS, PCEN_POWER) - Math.pow(PCEN_BIAS, PCEN_POWER);
        sum += pOut;
      }
      tSeries[t] = sum / N_MELS_PCEN;
    }
    let tMax = -Infinity, tSum = 0;
    for (let t = 0; t < N_FRAMES; t++) {
      if (tSeries[t] > tMax) tMax = tSeries[t];
      tSum += tSeries[t];
    }
    const tMean = tSum / N_FRAMES;
    let tVar = 0;
    for (let t = 0; t < N_FRAMES; t++) tVar += (tSeries[t] - tMean) ** 2;
    feats.nr_pcen_max = tMax;
    feats.nr_pcen_mean = tMean;
    feats.nr_pcen_std = Math.sqrt(tVar / N_FRAMES);
  }

  feats.nr_bp_peak_db = 20 * Math.log10(envPeak + EPS);

  return feats;
}

/**
 * Alla 83 features för Fable-modellen: base62 (på 1 s END-zero-paddad buffer,
 * samma som övriga modeller) + 21 nr_ på det råa 300 ms-klippet.
 */
export function extractFableFeatures(clip: Float32Array): Record<string, number> {
  const padded = new Float32Array(SR);
  padded.set(clip.subarray(0, Math.min(clip.length, SR)));
  const base = extractFeatures(padded);
  const robust = extractNrFeatures(clip);
  return { ...base, ...robust };
}
