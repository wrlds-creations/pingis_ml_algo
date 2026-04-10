/**
 * audioFeatures.ts
 *
 * Extraherar exakt samma 35 features som librosa's standardinställningar
 * (n_fft=2048, hop_length=512, n_mels=128, n_mfcc=13, center=True, Slaney mel).
 *
 * Input: Float32Array med 22 050 Hz mono PCM (≥ 22 050 samples).
 * Output: Record<string, number> med 35 nyckel-värde-par.
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

// ── Huvud-funktion ─────────────────────────────────────────────────────────────

export function extractFeatures(pcm: Float32Array): Record<string, number> {
  // 1. Fix till exakt 1 sekund = 22 050 samples
  const y = new Float32Array(SR);
  y.set(pcm.subarray(0, Math.min(pcm.length, SR)));

  // 2. Zero-padding (librosa stft default: center=True, pad_mode='constant')
  //    padded = [zeros(PAD), y, zeros(PAD)]
  const padded = new Float32Array(y.length + 2 * PAD);
  // Float32Array initialises to 0, so left/right pads are already zero
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

  return feats;
}
