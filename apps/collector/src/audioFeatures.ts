/**
 * audioFeatures.ts
 *
 * Extraherar 61+ features som matchar librosa's standardinställningar
 * (n_fft=2048, hop_length=512, n_mels=128, n_mfcc=13, center=True, Slaney mel)
 * plus transient-envelope, sub-band energi, spectral contrast/flatness,
 * och onset-region MFCC.
 *
 * Input: Float32Array med 22 050 Hz mono PCM (≥ 22 050 samples).
 * Output: Record<string, number> med feature nyckel-värde-par.
 */

// ── Konstanter ─────────────────────────────────────────────────────────────────

const SR          = 22_050;
const N_FFT       = 2_048;
const HOP         = 512;
const N_MELS      = 128;
const N_MFCC      = 13;
const N_BINS      = N_FFT / 2 + 1;   // 1 025
const PAD         = N_FFT >> 1;       // 1 024  (zero pad on each side)
const N_FRAMES    = 1 + Math.floor(SR / HOP);  // 44
const ROLL_PCT    = 0.85;
const LOG_EPS     = 1e-10;

// ── Hann-fönster (pre-computed) ────────────────────────────────────────────────

const HANN = new Float64Array(N_FFT);
for (let i = 0; i < N_FFT; i++) {
  HANN[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (N_FFT - 1));
}

// ── FFT bin-frekvenser ─────────────────────────────────────────────────────────

const FFT_FREQS = new Float64Array(N_BINS);
for (let k = 0; k < N_BINS; k++) FFT_FREQS[k] = (k * SR) / N_FFT;

// ── Mel-skala (librosa Slaney, htk=False) ─────────────────────────────────────

const MEL_F_SP       = 200.0 / 3.0;
const MEL_LOG_HZ     = 1_000.0;
const MEL_LOG_MEL    = MEL_LOG_HZ / MEL_F_SP;          // 15
const MEL_LOGSTEP    = Math.log(6.4) / 27.0;

function hzToMel(f: number): number {
  return f < MEL_LOG_HZ
    ? f / MEL_F_SP
    : MEL_LOG_MEL + Math.log(f / MEL_LOG_HZ) / MEL_LOGSTEP;
}

function melToHz(m: number): number {
  return m < MEL_LOG_MEL
    ? MEL_F_SP * m
    : MEL_LOG_HZ * Math.exp(MEL_LOGSTEP * (m - MEL_LOG_MEL));
}

// ── Mel-filterbank (pre-computed, row-major N_MELS × N_BINS) ──────────────────

const MEL_FB = (() => {
  const melMin = hzToMel(0);
  const melMax = hzToMel(SR / 2);
  const pts = new Float64Array(N_MELS + 2);
  for (let i = 0; i <= N_MELS + 1; i++) {
    pts[i] = melToHz(melMin + ((melMax - melMin) * i) / (N_MELS + 1));
  }

  const fb = new Float64Array(N_MELS * N_BINS);
  for (let m = 0; m < N_MELS; m++) {
    const lo = pts[m], ctr = pts[m + 1], hi = pts[m + 2];
    const bw = hi - lo; // Slaney normalization
    for (let k = 0; k < N_BINS; k++) {
      const f = FFT_FREQS[k];
      let v = 0;
      if (f > lo && f < ctr)         v = (f - lo) / (ctr - lo);
      else if (f >= ctr && f < hi)   v = (hi - f) / (hi - ctr);
      fb[m * N_BINS + k] = (v * 2.0) / bw;
    }
  }
  return fb;
})();

// ── FFT (iterativ Cooley-Tukey radix-2, in-place) ─────────────────────────────
// Delade buffrar – inte thread-safe men JS är single-threaded.

const _re = new Float64Array(N_FFT);
const _im = new Float64Array(N_FFT);

function _fft(): void {
  const n = N_FFT;
  // Bit-reversal
  let j = 0;
  for (let i = 1; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = _re[i]; _re[i] = _re[j]; _re[j] = t;
      t = _im[i]; _im[i] = _im[j]; _im[j] = t;
    }
  }
  // Butterfly
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const ang  = (-2 * Math.PI) / len;
    const wr0  = Math.cos(ang);
    const wi0  = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let wr = 1.0, wi = 0.0;
      for (let k = 0; k < half; k++) {
        const ur = _re[i + k],          ui = _im[i + k];
        const vr = _re[i+k+half]*wr - _im[i+k+half]*wi;
        const vi = _re[i+k+half]*wi + _im[i+k+half]*wr;
        _re[i + k]       = ur + vr;
        _im[i + k]       = ui + vi;
        _re[i + k + half] = ur - vr;
        _im[i + k + half] = ui - vi;
        const nwr = wr * wr0 - wi * wi0;
        wi = wr * wi0 + wi * wr0;
        wr = nwr;
      }
    }
  }
}

// ── Ortho DCT-II (matchar scipy.fftpack.dct type=2, norm='ortho') ─────────────

function _dctOrtho(x: Float64Array): number[] {
  const N   = x.length;
  const out: number[] = new Array(N_MFCC);
  const k0  = Math.sqrt(1.0 / N);
  const kk  = Math.sqrt(2.0 / N);
  for (let k = 0; k < N_MFCC; k++) {
    let s = 0;
    const f = (Math.PI * k) / (2 * N);
    for (let n = 0; n < N; n++) s += x[n] * Math.cos(f * (2 * n + 1));
    out[k] = s * (k === 0 ? k0 : kk);
  }
  return out;
}

// ── Hitta energitopp (onset-hjälp) ────────────────────────────────────────────

/**
 * Returnerar sampel-index för den 50 ms-frame med högst RMS-energi.
 * Används för att centrera klassificeringsfönstret runt studsen.
 */
export function findPeakFrame(pcm: Float32Array): number {
  const frameSize = Math.round(SR * 0.05); // 50 ms
  let maxEnergy = 0;
  let maxStart  = 0;
  for (let start = 0; start + frameSize <= pcm.length; start += frameSize) {
    let e = 0;
    for (let j = 0; j < frameSize; j++) e += pcm[start + j] ** 2;
    if (e > maxEnergy) { maxEnergy = e; maxStart = start; }
  }
  return maxStart + Math.round(frameSize / 2); // mitten av peak-framen
}

// ── Spectral contrast: oktavband-gränser (librosa default, n_bands=6) ────────
// fmin=200, 6 oktavband → 7 gränser: 200, 400, 800, 1600, 3200, 6400, nyquist
const SC_N_BANDS = 7; // 6 bands + 1 valley-band (index 6)
const SC_EDGES: number[] = [];
{
  const fmin = 200;
  for (let i = 0; i <= 6; i++) SC_EDGES.push(fmin * Math.pow(2, i));
  SC_EDGES[SC_EDGES.length - 1] = Math.min(SC_EDGES[SC_EDGES.length - 1], SR / 2);
}

// ── Sub-band energi: band-gränser ────────────────────────────────────────────
const SUB_BANDS: [string, number, number][] = [
  ['low',   200,  800],
  ['mid',   800,  2500],
  ['high',  2500, 6000],
  ['vhigh', 6000, SR / 2],
];

// ── Transient features (tidsdomän) ────────────────────────────────────────────

function _extractTransient(y: Float32Array): Record<string, number> {
  const feats: Record<string, number> = {};
  let peakIdx = 0;
  let peakVal = 0;
  for (let i = 0; i < y.length; i++) {
    const a = Math.abs(y[i]);
    if (a > peakVal) { peakVal = a; peakIdx = i; }
  }

  if (peakVal < 1e-6) {
    return {
      attack_time_ms: 0, decay_time_ms: 0, attack_slope: 0,
      crest_factor: 0, temporal_centroid: 0.5, energy_decay_rate: 0,
    };
  }

  // Attack time: 10% → 90% av peak (sök bakåt 50ms)
  const thresh10 = peakVal * 0.1;
  const thresh90 = peakVal * 0.9;
  const searchStart = Math.max(0, peakIdx - Math.round(0.05 * SR));
  let idx10 = peakIdx, idx90 = peakIdx;
  for (let i = searchStart; i <= peakIdx; i++) {
    if (Math.abs(y[i]) >= thresh10 && idx10 === peakIdx) idx10 = i;
    if (Math.abs(y[i]) >= thresh90) { idx90 = i; break; }
  }
  const attackSamples = Math.max(1, idx90 - idx10);
  feats.attack_time_ms = attackSamples / SR * 1000;

  // Decay time: peak → 50% (sök framåt 100ms)
  const thresh50 = peakVal * 0.5;
  const searchEnd = Math.min(y.length, peakIdx + Math.round(0.1 * SR));
  let decaySamples = searchEnd - peakIdx;
  for (let i = peakIdx; i < searchEnd; i++) {
    if (Math.abs(y[i]) < thresh50) { decaySamples = i - peakIdx; break; }
  }
  feats.decay_time_ms = decaySamples / SR * 1000;

  // Attack slope
  feats.attack_slope = peakVal / Math.max(feats.attack_time_ms, 0.01);

  // Crest factor: peak / RMS (100ms runt onset)
  const rStart = Math.max(0, peakIdx - Math.round(0.05 * SR));
  const rEnd = Math.min(y.length, peakIdx + Math.round(0.05 * SR));
  let sqSum = 0;
  for (let i = rStart; i < rEnd; i++) sqSum += y[i] * y[i];
  const regionRms = Math.sqrt(sqSum / Math.max(1, rEnd - rStart)) + 1e-9;
  feats.crest_factor = peakVal / regionRms;

  // Temporal centroid (200ms runt onset)
  const tcStart = Math.max(0, peakIdx - Math.round(0.1 * SR));
  const tcEnd = Math.min(y.length, peakIdx + Math.round(0.1 * SR));
  let eSum = 0, eWeighted = 0;
  const tcLen = tcEnd - tcStart;
  for (let i = 0; i < tcLen; i++) {
    const e = y[tcStart + i] * y[tcStart + i];
    eSum += e;
    eWeighted += i * e;
  }
  feats.temporal_centroid = eSum > 0 ? eWeighted / eSum / Math.max(1, tcLen) : 0.5;

  // Energy decay rate: linjär regression av log-RMS i 5ms-frames efter onset
  const frameSize = Math.max(1, Math.round(0.005 * SR));
  const nFrames = Math.min(20, Math.floor((y.length - peakIdx) / frameSize));
  if (nFrames >= 3) {
    const logRms: number[] = [];
    for (let i = 0; i < nFrames; i++) {
      const s = peakIdx + i * frameSize;
      let sq2 = 0;
      for (let j = 0; j < frameSize; j++) sq2 += y[s + j] * y[s + j];
      logRms.push(Math.log10(Math.sqrt(sq2 / frameSize) + 1e-9));
    }
    // Enkel linjär regression: slope
    let sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (let i = 0; i < nFrames; i++) {
      sx += i; sy += logRms[i]; sxx += i * i; sxy += i * logRms[i];
    }
    feats.energy_decay_rate = (nFrames * sxy - sx * sy) / (nFrames * sxx - sx * sx);
  } else {
    feats.energy_decay_rate = 0;
  }

  return feats;
}

// ── Sub-band energi (FFT på onset-frame) ──────────────────────────────────────

function _extractSubband(y: Float32Array): Record<string, number> {
  const feats: Record<string, number> = {};

  // Hitta onset (50ms med mest energi)
  const frameSize = Math.round(0.05 * SR);
  let peakIdx = 0, maxE = 0;
  for (let i = 0; i < y.length; i++) {
    const a = Math.abs(y[i]);
    if (a > maxE) { maxE = a; peakIdx = i; }
  }
  const onsetStart = Math.max(0, peakIdx - Math.floor(frameSize / 2));
  const onsetEnd = Math.min(y.length, onsetStart + frameSize);

  // FFT på onset-frame
  const nFft = 2048;
  _re.fill(0); _im.fill(0);
  for (let i = onsetStart; i < Math.min(onsetEnd, onsetStart + nFft); i++) {
    const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * (i - onsetStart)) / (Math.min(onsetEnd - onsetStart, nFft) - 1));
    _re[i - onsetStart] = y[i] * w;
  }
  _fft();

  // Power-spektrum
  const nBins = nFft / 2 + 1;
  const power = new Float64Array(nBins);
  for (let k = 0; k < nBins; k++) {
    power[k] = _re[k] * _re[k] + _im[k] * _im[k];
  }

  const energies: Record<string, number> = {};
  for (const [name, lo, hi] of SUB_BANDS) {
    let e = 1e-12;
    for (let k = 0; k < nBins; k++) {
      const f = (k * SR) / nFft;
      if (f >= lo && f < hi) e += power[k];
    }
    energies[name] = e;
    feats[`band_energy_${name}`] = Math.log10(e);
  }

  feats.ratio_mid_low = Math.log10(energies.mid / energies.low);
  feats.ratio_high_mid = Math.log10(energies.high / energies.mid);
  feats.ratio_low_high = Math.log10(energies.low / energies.high);

  const bandNames = Object.keys(energies);
  let maxIdx = 0, maxVal = 0;
  for (let i = 0; i < bandNames.length; i++) {
    if (energies[bandNames[i]] > maxVal) { maxVal = energies[bandNames[i]]; maxIdx = i; }
  }
  feats.band_peak_idx = maxIdx;

  return feats;
}

// ── Huvud-funktion ─────────────────────────────────────────────────────────────

export function extractFeatures(pcm: Float32Array): Record<string, number> {
  // 1. Fix till exakt 1 sekund = 22 050 samples
  const y = new Float32Array(SR);
  y.set(pcm.subarray(0, Math.min(pcm.length, SR)));

  // 2. Zero-padding (librosa stft default: center=True, pad_mode='constant')
  const padded = new Float32Array(y.length + 2 * PAD);
  padded.set(y, PAD);

  // 3. Per-frame ackumulatorer
  const mfccSum = new Float64Array(N_MFCC);
  const mfccSq  = new Float64Array(N_MFCC);

  let centSum = 0, centSq = 0;
  let rollSum = 0, rollSq = 0;
  let zcrSum  = 0, zcrSq  = 0;
  let rmsSum  = 0, rmsSq  = 0;

  let onsetMax = 0;
  const prevLogMel = new Float64Array(N_MELS);
  const curLogMel  = new Float64Array(N_MELS);
  let firstFrame = true;

  // Nya ackumulatorer
  const contrastSum = new Float64Array(SC_N_BANDS);  // spectral contrast
  let flatSum = 0, flatSq = 0;                       // spectral flatness
  let peakRmsFrame = 0;                               // index av frame med mest RMS
  let peakRmsVal = 0;
  const peakMfcc = new Float64Array(4);               // onset-region MFCC (4 koeff)
  const rmsPerFrame = new Float64Array(N_FRAMES);     // för att hitta peak frame

  // Pre-allocated per-frame buffers
  const melFrame = new Float64Array(N_MELS);
  const magFrame = new Float64Array(N_BINS);

  // 4. STFT-loop
  for (let t = 0; t < N_FRAMES; t++) {
    const start = t * HOP;

    // Ladda frame + Hann-fönster till FFT-buffrar
    _re.fill(0); _im.fill(0);
    for (let i = 0; i < N_FFT; i++) {
      _re[i] = padded[start + i] * HANN[i];
    }
    _fft();

    // Power- och magnitude-spektrum
    let magTotal = 0;
    for (let k = 0; k < N_BINS; k++) {
      const p  = _re[k] * _re[k] + _im[k] * _im[k];
      magFrame[k] = Math.sqrt(p);
      magTotal += magFrame[k];
    }

    // ── Mel-spektrogram (power) + log ────────────────────────────────────────
    for (let m = 0; m < N_MELS; m++) {
      let s = 0;
      const off = m * N_BINS;
      for (let k = 0; k < N_BINS; k++) s += MEL_FB[off + k] * (magFrame[k] * magFrame[k]);
      melFrame[m] = s;
      curLogMel[m] = 10 * Math.log10(Math.max(melFrame[m], LOG_EPS));
    }

    // ── MFCC ─────────────────────────────────────────────────────────────────
    const coeffs = _dctOrtho(curLogMel);
    for (let k = 0; k < N_MFCC; k++) {
      mfccSum[k] += coeffs[k];
      mfccSq[k]  += coeffs[k] * coeffs[k];
    }

    // ── Onset strength (positiv diff av log-mel) ──────────────────────────────
    if (!firstFrame) {
      let onsetVal = 0;
      for (let m = 0; m < N_MELS; m++) {
        const d = curLogMel[m] - prevLogMel[m];
        if (d > 0) onsetVal += d;
      }
      onsetVal /= N_MELS;
      if (onsetVal > onsetMax) onsetMax = onsetVal;
    }
    prevLogMel.set(curLogMel);
    firstFrame = false;

    // ── Spectral centroid (magnitude-viktad) ──────────────────────────────────
    let centNum = 0;
    for (let k = 0; k < N_BINS; k++) centNum += FFT_FREQS[k] * magFrame[k];
    const cent = magTotal > 0 ? centNum / magTotal : 0;
    centSum += cent; centSq += cent * cent;

    // ── Spectral rolloff ──────────────────────────────────────────────────────
    const thr = ROLL_PCT * magTotal;
    let cumsum = 0;
    let rollFreq = FFT_FREQS[N_BINS - 1];
    for (let k = 0; k < N_BINS; k++) {
      cumsum += magFrame[k];
      if (cumsum >= thr) { rollFreq = FFT_FREQS[k]; break; }
    }
    rollSum += rollFreq; rollSq += rollFreq * rollFreq;

    // ── ZCR (time-domain, matchar librosa: sign_changes / (2*(N-1))) ─────────
    let zc = 0;
    for (let i = 1; i < N_FFT; i++) {
      if (padded[start + i - 1] * padded[start + i] < 0) zc++;
    }
    const zcr = zc / (2 * (N_FFT - 1));
    zcrSum += zcr; zcrSq += zcr * zcr;

    // ── RMS ───────────────────────────────────────────────────────────────────
    let sq = 0;
    for (let i = 0; i < N_FFT; i++) { const s = padded[start + i]; sq += s * s; }
    const rmsV = Math.sqrt(sq / N_FFT);
    rmsSum += rmsV; rmsSq += rmsV * rmsV;
    rmsPerFrame[t] = rmsV;

    // Spara MFCC för peak-RMS-frame (onset-region MFCC)
    if (rmsV > peakRmsVal) {
      peakRmsVal = rmsV;
      peakRmsFrame = t;
      for (let k = 0; k < 4; k++) peakMfcc[k] = coeffs[k];
    }

    // ── Spectral contrast (per oktavband) ────────────────────────────────────
    // librosa: för varje band, sortera magnitude-bins, kontrast = mean(topp) - mean(botten)
    for (let b = 0; b < SC_N_BANDS; b++) {
      const loHz = b < SC_EDGES.length ? SC_EDGES[b] : 0;
      const hiHz = b + 1 < SC_EDGES.length ? SC_EDGES[b + 1] : SR / 2;
      const bandMags: number[] = [];
      for (let k = 0; k < N_BINS; k++) {
        if (FFT_FREQS[k] >= loHz && FFT_FREQS[k] < hiHz) {
          bandMags.push(magFrame[k] * magFrame[k]); // power
        }
      }
      if (bandMags.length > 0) {
        bandMags.sort((a, b2) => a - b2);
        const nPeek = Math.max(1, Math.floor(bandMags.length * 0.2));
        let topSum = 0, botSum = 0;
        for (let i = 0; i < nPeek; i++) {
          botSum += bandMags[i];
          topSum += bandMags[bandMags.length - 1 - i];
        }
        const topMean = topSum / nPeek + LOG_EPS;
        const botMean = botSum / nPeek + LOG_EPS;
        contrastSum[b] += 10 * Math.log10(topMean) - 10 * Math.log10(botMean);
      }
    }

    // ── Spectral flatness ────────────────────────────────────────────────────
    // geometric_mean(S) / arithmetic_mean(S) via exp(mean(log(S))) / mean(S)
    let logSum2 = 0, arithSum = 0;
    let validBins = 0;
    for (let k = 1; k < N_BINS; k++) {  // skip DC
      const p = magFrame[k] * magFrame[k] + LOG_EPS;
      logSum2 += Math.log(p);
      arithSum += p;
      validBins++;
    }
    if (validBins > 0 && arithSum > 0) {
      const geoMean = Math.exp(logSum2 / validBins);
      const ariMean = arithSum / validBins;
      const flat = geoMean / ariMean;
      flatSum += flat;
      flatSq += flat * flat;
    }
  }

  // 5. Beräkna mean och std (ddof=0, matchar numpy.std)
  const feats: Record<string, number> = {};

  for (let k = 0; k < N_MFCC; k++) {
    const mean = mfccSum[k] / N_FRAMES;
    feats[`mfcc_${k}_mean`] = mean;
    feats[`mfcc_${k}_std`]  = Math.sqrt(Math.max(0, mfccSq[k] / N_FRAMES - mean * mean));
  }

  const _ms = (sum: number, sq: number) => {
    const m = sum / N_FRAMES;
    return { m, s: Math.sqrt(Math.max(0, sq / N_FRAMES - m * m)) };
  };

  const c = _ms(centSum, centSq);
  feats['spectral_centroid_mean'] = c.m;
  feats['spectral_centroid_std']  = c.s;

  const r = _ms(rollSum, rollSq);
  feats['spectral_rolloff_mean'] = r.m;
  feats['spectral_rolloff_std']  = r.s;

  const z = _ms(zcrSum, zcrSq);
  feats['zcr_mean'] = z.m;
  feats['zcr_std']  = z.s;

  const ms = _ms(rmsSum, rmsSq);
  feats['rms_mean'] = ms.m;
  feats['rms_std']  = ms.s;

  feats['onset_strength_max'] = onsetMax;

  // ── Nya features: transient-envelope (6) ───────────────────────────────────
  const transient = _extractTransient(y);
  Object.assign(feats, transient);

  // ── Nya features: sub-band energi (8) ──────────────────────────────────────
  const subband = _extractSubband(y);
  Object.assign(feats, subband);

  // ── Nya features: spectral contrast (7) ────────────────────────────────────
  for (let b = 0; b < SC_N_BANDS; b++) {
    feats[`spectral_contrast_band_${b}`] = contrastSum[b] / N_FRAMES;
  }

  // ── Nya features: spectral flatness (2) ────────────────────────────────────
  const fl = _ms(flatSum, flatSq);
  feats['spectral_flatness_mean'] = fl.m;
  feats['spectral_flatness_std']  = fl.s;

  // ── Nya features: onset-region MFCC (4) ────────────────────────────────────
  for (let k = 0; k < 4; k++) {
    feats[`onset_mfcc_${k}`] = peakMfcc[k];
  }

  return feats;
}
