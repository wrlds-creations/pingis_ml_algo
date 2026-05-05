import RNFS from 'react-native-fs';
import { Buffer } from 'buffer';
import { detectAudioContact } from './audioContactEngine';
import type {
  AudioDetectionEvent,
  AudioContactKind,
  AudioNotRacketKind,
  AudioReviewBounceSide,
  AudioReviewClassLabel,
  AudioReviewEventType,
  AudioReviewLabel,
  AudioReviewMarker,
  AudioScenarioId,
} from './types';

export const TARGET_AUDIO_SR = 22050;
export const REVIEW_PRE_MS = 300;
export const REVIEW_POST_MS = 700;
export const REVIEW_MARKER_ZOOM_PRE_MS = 80;
export const REVIEW_MARKER_ZOOM_POST_MS = 80;

const REVIEW_REQUIRED_SCENARIOS = new Set<AudioScenarioId>([
  'racket_bounce_fh',
  'racket_bounce_bh',
  'racket_bounce_mixed',
  'table_bounce',
  'floor_bounce',
  'catch_after_sound',
  'speech_music_noise',
  'free_recording',
  'racket_quiet',
  'racket_counting',
  'racket_music_low',
  'racket_music_mid',
  'speech_only',
  'desk_keyboard_only',
  'music_low_only',
  'music_mid_only',
  'table_quiet',
  'floor_quiet',
]);

const FRAME_MS = 12;
const MIN_CANDIDATE_GAP_MS = 180;
const MIN_ENVELOPE_THRESHOLD = 0.01;
const LOCAL_FRAME_MS = 4;
const LOCAL_HOP_MS = 1;
const AUTO_REFINE_SEARCH_PRE_MS = 120;
const AUTO_REFINE_SEARCH_POST_MS = 40;
const MANUAL_SNAP_RADIUS_MS = 140;
const REVIEW_CONTACT_CONFIDENCE = 0.65;
const PLAYING_REVIEW_CONTACT_FLOOR = 0.45;
const REVIEW_SURFACE_CONFIDENCE = 0.55;
const REVIEW_SURFACE_VETO_CONFIDENCE = 0.75;
const REVIEW_MARKER_DEDUP_MS = 450;

export interface DecodedWavFile {
  sampleRate: number;
  samples: Float32Array;
  durationMs: number;
}

export interface MarkerZoomWaveformWindow {
  bins: number[];
  start_ms: number;
  end_ms: number;
  focus_ms: number;
  peak_ms: number;
}

export interface AudioSyncPoint {
  timestamp_ms: number;
  score: number;
  confidence: number;
  window_start_ms: number;
  window_end_ms: number;
}

export function requiresAudioReview(scenarioId: AudioScenarioId): boolean {
  return REVIEW_REQUIRED_SCENARIOS.has(scenarioId);
}

export function suggestedReviewLabelForScenario(scenarioId: AudioScenarioId): AudioReviewLabel {
  return scenarioId.startsWith('racket_') || scenarioId === 'free_recording'
    ? 'racket_contact'
    : 'not_racket_contact';
}

function contactKindForScenario(scenarioId: AudioScenarioId): AudioContactKind | undefined {
  return scenarioId.startsWith('racket_') || scenarioId === 'free_recording' ? 'racket_bounce' : undefined;
}

function notRacketKindForScenario(scenarioId: AudioScenarioId): AudioNotRacketKind | undefined {
  if (scenarioId === 'table_bounce' || scenarioId === 'table_quiet') return 'table_bounce';
  if (scenarioId === 'floor_bounce' || scenarioId === 'floor_quiet') return 'floor_bounce';
  if (scenarioId === 'catch_after_sound') return 'catch_after_sound';
  if (
    scenarioId === 'speech_music_noise' ||
    scenarioId === 'speech_only' ||
    scenarioId === 'music_low_only' ||
    scenarioId === 'music_mid_only'
  ) {
    return 'voice_music_noise';
  }
  if (scenarioId === 'desk_keyboard_only') return 'other_impact';
  return undefined;
}

function bounceSideForScenario(scenarioId: AudioScenarioId): AudioReviewBounceSide {
  if (scenarioId === 'racket_bounce_fh') return 'forehand';
  if (scenarioId === 'racket_bounce_bh') return 'backhand';
  return 'unknown';
}

function eventTypeForNotRacketKind(kind?: AudioNotRacketKind): AudioReviewEventType {
  return kind === 'table_bounce' || kind === 'floor_bounce' ? 'bounce' : 'noise';
}

function classLabelForRacketSide(side: AudioReviewBounceSide): AudioReviewClassLabel {
  if (side === 'forehand') return 'forehand';
  if (side === 'backhand') return 'backhand';
  return 'racket_bounce';
}

function metadataForScenario(
  scenarioId: AudioScenarioId,
  finalLabel: AudioReviewLabel,
  decision?: AudioDetectionEvent,
) {
  if (finalLabel === 'racket_contact') {
    const bounceSide = bounceSideForScenario(scenarioId);
    return {
      contact_kind: contactKindForScenario(scenarioId),
      not_racket_kind: undefined,
      bounce_side: bounceSide,
      event_type: 'racket_hit' as AudioReviewEventType,
      class_label: classLabelForRacketSide(bounceSide),
    };
  }
  if (finalLabel === 'not_racket_contact') {
    const detectedKind = (decision?.surface_label === 'table_bounce' || decision?.surface_label === 'floor_bounce')
      ? decision.surface_label
      : undefined;
    const notRacketKind = notRacketKindForScenario(scenarioId) ?? detectedKind;
    return {
      contact_kind: undefined,
      not_racket_kind: notRacketKind,
      bounce_side: undefined,
      event_type: eventTypeForNotRacketKind(notRacketKind),
      class_label: notRacketKind ?? 'other_impact',
    };
  }
  return {
    contact_kind: undefined,
    not_racket_kind: undefined,
    bounce_side: undefined,
    event_type: 'ignore' as AudioReviewEventType,
    class_label: 'ignore' as AudioReviewClassLabel,
  };
}

export function normalizeReviewMarkerForLabel(
  marker: AudioReviewMarker,
  scenarioId: AudioScenarioId,
  finalLabel: AudioReviewLabel,
  reviewStatus = marker.review_status ?? (marker.source === 'manual' ? 'edited' : 'pending'),
): AudioReviewMarker {
  return {
    ...marker,
    final_label: finalLabel,
    review_status: reviewStatus,
    ...metadataForScenario(scenarioId, finalLabel),
  };
}

export function buildSuggestedReviewMarkers(
  samples: Float32Array,
  sampleRate: number,
  scenarioId: AudioScenarioId,
): AudioReviewMarker[] {
  const markers: AudioReviewMarker[] = [];
  let lastAcceptedMs = -Infinity;
  const contactConfidenceThreshold = scenarioId === 'free_recording'
    ? PLAYING_REVIEW_CONTACT_FLOOR
    : REVIEW_CONTACT_CONFIDENCE;

  for (const candidate of detectReviewCandidates(samples, sampleRate)) {
    if (candidate.refined_timestamp_ms - lastAcceptedMs < REVIEW_MARKER_DEDUP_MS) continue;

    const decision = detectAudioContact({
      detectedAtMs: candidate.refined_timestamp_ms,
      pcm: slicePreviewWindow(samples, sampleRate, candidate.refined_timestamp_ms),
      confidenceThreshold: contactConfidenceThreshold,
      dedupMs: REVIEW_MARKER_DEDUP_MS,
      lastQualifiedTsMs: Number.isFinite(lastAcceptedMs) ? lastAcceptedMs : undefined,
    });
    const suggestedLabel = reviewLabelForDecision(scenarioId, decision, contactConfidenceThreshold);
    if (!suggestedLabel) continue;

    markers.push({
      id: `auto_${markers.length}_${candidate.refined_timestamp_ms}`,
      timestamp_ms: candidate.refined_timestamp_ms,
      source: 'auto',
      suggested_label: suggestedLabel,
      final_label: suggestedLabel,
      review_status: 'pending',
      contact_confidence: decision.confidence,
      surface_label: decision.surface_label,
      surface_confidence: decision.surface_confidence,
      ...metadataForScenario(scenarioId, suggestedLabel, decision),
    });
    lastAcceptedMs = candidate.refined_timestamp_ms;
  }

  return markers;
}

export function detectAudioSyncPoint(
  samples: Float32Array,
  sampleRate: number,
  searchStartMs = 250,
  searchEndMs = 3000,
): AudioSyncPoint | null {
  if (samples.length === 0 || sampleRate <= 0) return null;

  const totalDurationMs = (samples.length / sampleRate) * 1000;
  const startMs = Math.max(0, Math.min(searchStartMs, totalDurationMs));
  const endMs = Math.max(startMs, Math.min(searchEndMs, totalDurationMs));
  const envelope = buildLocalEnvelope(samples, sampleRate, startMs, endMs);
  if (envelope.values.length < 3) return null;

  const sorted = [...envelope.values].sort((a, b) => a - b);
  const median = Math.max(0.0001, percentile(sorted, 0.5));
  const p85 = percentile(sorted, 0.85);
  const p97 = percentile(sorted, 0.97);
  const minimumPeak = Math.max(MIN_ENVELOPE_THRESHOLD * 1.2, median * 2.6, p85 * 1.2, p97 * 0.55);
  const candidatePeaks = detectLocalEnvelopePeaks(envelope.values);
  const peakIndices = candidatePeaks.length > 0
    ? candidatePeaks
    : envelope.values.map((_, index) => index);

  let bestPeak = peakIndices[0];
  for (const index of peakIndices) {
    if (envelope.values[index] > envelope.values[bestPeak]) {
      bestPeak = index;
    }
  }

  const peakValue = envelope.values[bestPeak];
  if (peakValue < minimumPeak) return null;

  const prominence = peakValue / median;
  const approxTimestampMs = startMs + bestPeak * envelope.hop_ms;
  const timestampMs = refineAttackTimestamp(samples, sampleRate, approxTimestampMs);
  const confidence = Math.max(
    0,
    Math.min(1, ((prominence - 2.2) / 7) + ((peakValue - minimumPeak) / Math.max(minimumPeak, 0.0001)) * 0.25),
  );

  return {
    timestamp_ms: Math.round(timestampMs),
    score: Number(peakValue.toFixed(5)),
    confidence: Number(confidence.toFixed(3)),
    window_start_ms: Math.round(startMs),
    window_end_ms: Math.round(endMs),
  };
}

function reviewLabelForDecision(
  scenarioId: AudioScenarioId,
  decision: AudioDetectionEvent,
  contactConfidenceThreshold = REVIEW_CONTACT_CONFIDENCE,
): AudioReviewLabel | null {
  const surfaceLabel = decision.surface_label;
  const surfaceConfidence = decision.surface_confidence ?? 0;

  if (scenarioId.startsWith('racket_')) {
    const surfaceVeto = (
      (surfaceLabel === 'table_bounce' || surfaceLabel === 'floor_bounce') &&
      surfaceConfidence >= REVIEW_SURFACE_VETO_CONFIDENCE
    );
    if (
      decision.label === 'racket_contact' &&
      decision.confidence >= contactConfidenceThreshold &&
      !surfaceVeto
    ) {
      return 'racket_contact';
    }
    return null;
  }

  if (scenarioId === 'free_recording') {
    const surfaceVeto = (
      (surfaceLabel === 'table_bounce' || surfaceLabel === 'floor_bounce') &&
      surfaceConfidence >= REVIEW_SURFACE_CONFIDENCE
    );
    if (surfaceVeto) return 'not_racket_contact';
    return decision.label === 'racket_contact' && decision.confidence >= contactConfidenceThreshold
      ? 'racket_contact'
      : null;
  }

  if (scenarioId === 'table_bounce' || scenarioId === 'table_quiet') {
    return surfaceLabel === 'table_bounce' && surfaceConfidence >= REVIEW_SURFACE_CONFIDENCE
      ? 'not_racket_contact'
      : null;
  }

  if (scenarioId === 'floor_bounce' || scenarioId === 'floor_quiet') {
    return surfaceLabel === 'floor_bounce' && surfaceConfidence >= REVIEW_SURFACE_CONFIDENCE
      ? 'not_racket_contact'
      : null;
  }

  if (scenarioId === 'catch_after_sound' || scenarioId === 'desk_keyboard_only') {
    return decision.label === 'not_racket_contact' || decision.confidence < REVIEW_CONTACT_CONFIDENCE
      ? 'not_racket_contact'
      : null;
  }

  if (
    scenarioId === 'speech_music_noise' ||
    scenarioId === 'speech_only' ||
    scenarioId === 'music_low_only' ||
    scenarioId === 'music_mid_only'
  ) {
    return (
      decision.label === 'not_racket_contact' &&
      (surfaceLabel === 'noise' || surfaceConfidence < REVIEW_SURFACE_CONFIDENCE)
    )
      ? 'not_racket_contact'
      : null;
  }

  return null;
}

export async function decodeWavFile(filePath: string): Promise<DecodedWavFile> {
  const base64 = await RNFS.readFile(filePath, 'base64');
  const bytes = Uint8Array.from(Buffer.from(base64, 'base64'));
  if (bytes.length < 44) {
    throw new Error(`WAV too short: ${filePath}`);
  }

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const riff = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
  const wave = String.fromCharCode(bytes[8], bytes[9], bytes[10], bytes[11]);
  if (riff !== 'RIFF' || wave !== 'WAVE') {
    throw new Error(`Unsupported WAV container: ${filePath}`);
  }

  let offset = 12;
  let sampleRate = TARGET_AUDIO_SR;
  let bitsPerSample = 16;
  let channels = 1;
  let dataOffset = -1;
  let dataSize = 0;

  while (offset + 8 <= bytes.length) {
    const chunkId = String.fromCharCode(
      bytes[offset],
      bytes[offset + 1],
      bytes[offset + 2],
      bytes[offset + 3],
    );
    const chunkSize = view.getUint32(offset + 4, true);
    const chunkDataOffset = offset + 8;

    if (chunkId === 'fmt ' && chunkSize >= 16) {
      channels = view.getUint16(chunkDataOffset + 2, true);
      sampleRate = view.getUint32(chunkDataOffset + 4, true);
      bitsPerSample = view.getUint16(chunkDataOffset + 14, true);
    } else if (chunkId === 'data') {
      dataOffset = chunkDataOffset;
      dataSize = chunkSize;
      break;
    }

    offset += 8 + chunkSize + (chunkSize % 2);
  }

  if (dataOffset < 0) {
    throw new Error(`WAV data chunk missing: ${filePath}`);
  }
  if (bitsPerSample !== 16 || channels !== 1) {
    throw new Error(`Only PCM16 mono is supported for review: ${filePath}`);
  }

  const sampleCount = Math.floor(dataSize / 2);
  const samples = new Float32Array(sampleCount);
  for (let i = 0; i < sampleCount; i++) {
    const value = view.getInt16(dataOffset + i * 2, true);
    samples[i] = value / 32768;
  }

  return {
    sampleRate,
    samples,
    durationMs: Math.round((sampleCount / sampleRate) * 1000),
  };
}

export function buildWaveformBins(samples: Float32Array, binCount: number): number[] {
  if (samples.length === 0 || binCount <= 0) return [];
  const bins: number[] = [];
  const samplesPerBin = Math.max(1, Math.floor(samples.length / binCount));

  for (let start = 0; start < samples.length; start += samplesPerBin) {
    let peak = 0;
    const end = Math.min(samples.length, start + samplesPerBin);
    for (let i = start; i < end; i++) {
      const amplitude = Math.abs(samples[i]);
      if (amplitude > peak) peak = amplitude;
    }
    bins.push(Number(Math.min(1, peak).toFixed(4)));
  }

  return bins;
}

export function buildMarkerZoomWaveformWindow(
  samples: Float32Array,
  sampleRate: number,
  focusMs: number,
  binCount: number,
  preMs = REVIEW_MARKER_ZOOM_PRE_MS,
  postMs = REVIEW_MARKER_ZOOM_POST_MS,
): MarkerZoomWaveformWindow {
  if (samples.length === 0) {
    return { bins: [], start_ms: 0, end_ms: 0, focus_ms: 0, peak_ms: 0 };
  }

  const totalDurationMs = (samples.length / sampleRate) * 1000;
  const safeFocusMs = clampMs(focusMs, totalDurationMs);
  const startMs = Math.max(0, safeFocusMs - preMs);
  const endMs = Math.min(totalDurationMs, safeFocusMs + postMs);
  const startSample = Math.max(0, Math.floor((startMs / 1000) * sampleRate));
  const endSample = Math.min(samples.length, Math.ceil((endMs / 1000) * sampleRate));
  const peakMs = findLocalPeakTimestamp(samples, sampleRate, startMs, endMs);

  return {
    bins: buildWaveformBins(samples.slice(startSample, endSample), binCount),
    start_ms: Math.round(startMs),
    end_ms: Math.round(endMs),
    focus_ms: Math.round(safeFocusMs),
    peak_ms: Math.round(peakMs),
  };
}

export async function writePreviewClip(
  samples: Float32Array,
  sampleRate: number,
  timestampMs: number,
  playbackRate = 1,
): Promise<string> {
  const preview = slicePreviewWindow(samples, sampleRate, timestampMs);
  return writePlaybackClipFromSamples(
    preview,
    sampleRate,
    `pingis_preview_${Date.now()}_${Math.round(timestampMs)}`,
    playbackRate,
  );
}

export async function writeTakePlaybackClip(
  samples: Float32Array,
  sampleRate: number,
  startMs: number,
  playbackRate = 1,
): Promise<string> {
  const startSample = Math.max(0, Math.floor((startMs / 1000) * sampleRate));
  return writePlaybackClipFromSamples(
    samples.slice(startSample),
    sampleRate,
    `pingis_take_${Date.now()}_${Math.round(startMs)}`,
    playbackRate,
  );
}

async function writePlaybackClipFromSamples(
  sourceSamples: Float32Array,
  sampleRate: number,
  basename: string,
  playbackRate: number,
): Promise<string> {
  const wav = encodeMonoPcm16Wav(stretchSamplesForPlayback(sourceSamples, playbackRate), sampleRate);
  const safeRate = String(playbackRate).replace('.', '_');
  const filePath = `${RNFS.CachesDirectoryPath}/${basename}_${safeRate}x.wav`;
  await RNFS.writeFile(filePath, Buffer.from(wav).toString('base64'), 'base64');
  return filePath;
}

export function createManualMarker(
  timestampMs: number,
  scenarioId: AudioScenarioId,
): AudioReviewMarker {
  const finalLabel = suggestedReviewLabelForScenario(scenarioId);
  return {
    id: `manual_${Date.now()}_${Math.round(timestampMs)}`,
    timestamp_ms: Math.max(0, Math.round(timestampMs)),
    source: 'manual',
    suggested_label: finalLabel,
    final_label: finalLabel,
    review_status: 'edited',
    ...metadataForScenario(scenarioId, finalLabel),
  };
}

export function snapMarkerToAttack(
  samples: Float32Array,
  sampleRate: number,
  approxTimestampMs: number,
): number {
  const totalDurationMs = (samples.length / sampleRate) * 1000;
  const localStartMs = Math.max(0, approxTimestampMs - MANUAL_SNAP_RADIUS_MS);
  const localEndMs = Math.min(totalDurationMs, approxTimestampMs + MANUAL_SNAP_RADIUS_MS);
  const localEnvelope = buildLocalEnvelope(samples, sampleRate, localStartMs, localEndMs);
  if (localEnvelope.values.length < 3) {
    return clampMs(approxTimestampMs, totalDurationMs);
  }

  const peaks = detectLocalEnvelopePeaks(localEnvelope.values);
  if (peaks.length === 0) {
    return clampMs(approxTimestampMs, totalDurationMs);
  }

  const approxFrame = Math.round((approxTimestampMs - localStartMs) / localEnvelope.hop_ms);
  let bestPeak = peaks[0];
  let bestScore = Number.POSITIVE_INFINITY;
  for (const peak of peaks) {
    const distance = Math.abs(peak - approxFrame);
    const score = distance - localEnvelope.values[peak] * 18;
    if (score < bestScore) {
      bestScore = score;
      bestPeak = peak;
    }
  }

  const peakMs = localStartMs + bestPeak * localEnvelope.hop_ms;
  return refineAttackTimestamp(samples, sampleRate, peakMs);
}

function detectReviewCandidates(
  samples: Float32Array,
  sampleRate: number,
): Array<{ timestamp_ms: number; refined_timestamp_ms: number; score: number }> {
  if (samples.length === 0) return [];

  const frameSize = Math.max(32, Math.round((sampleRate * FRAME_MS) / 1000));
  const hopSize = Math.max(16, Math.round(frameSize / 2));
  const envelope: number[] = [];

  for (let start = 0; start + frameSize <= samples.length; start += hopSize) {
    let energy = 0;
    for (let i = start; i < start + frameSize; i++) energy += samples[i] * samples[i];
    envelope.push(Math.sqrt(energy / frameSize));
  }

  if (envelope.length < 3) return [];

  const sorted = [...envelope].sort((a, b) => a - b);
  const mean = envelope.reduce((sum, value) => sum + value, 0) / envelope.length;
  const p90 = percentile(sorted, 0.9);
  const p98 = percentile(sorted, 0.98);
  const threshold = Math.max(MIN_ENVELOPE_THRESHOLD, mean * 2.2, p90 * 0.6, p98 * 0.35);

  const localPeaks: Array<{ frame: number; score: number }> = [];
  for (let i = 1; i < envelope.length - 1; i++) {
    const value = envelope[i];
    if (value < threshold) continue;
    if (value >= envelope[i - 1] && value >= envelope[i + 1]) {
      localPeaks.push({ frame: i, score: value });
    }
  }

  localPeaks.sort((a, b) => b.score - a.score);
  const minGapFrames = Math.max(1, Math.round(MIN_CANDIDATE_GAP_MS / ((hopSize / sampleRate) * 1000)));
  const accepted: Array<{ frame: number; score: number }> = [];

  for (const peak of localPeaks) {
    const tooClose = accepted.some(existing => Math.abs(existing.frame - peak.frame) <= minGapFrames);
    if (!tooClose) accepted.push(peak);
  }

  accepted.sort((a, b) => a.frame - b.frame);
  return accepted.map(peak => ({
    timestamp_ms: Math.round(((peak.frame * hopSize) / sampleRate) * 1000),
    refined_timestamp_ms: refineAttackTimestamp(
      samples,
      sampleRate,
      Math.round(((peak.frame * hopSize) / sampleRate) * 1000),
    ),
    score: peak.score,
  }));
}

function percentile(sortedValues: number[], q: number): number {
  if (sortedValues.length === 0) return 0;
  const index = Math.max(0, Math.min(sortedValues.length - 1, Math.floor(q * (sortedValues.length - 1))));
  return sortedValues[index];
}

function slicePreviewWindow(
  samples: Float32Array,
  sampleRate: number,
  timestampMs: number,
): Float32Array {
  const centerSample = Math.round((timestampMs / 1000) * sampleRate);
  const preSamples = Math.round((REVIEW_PRE_MS / 1000) * sampleRate);
  const totalSamples = Math.round(((REVIEW_PRE_MS + REVIEW_POST_MS) / 1000) * sampleRate);
  const start = Math.max(0, centerSample - preSamples);
  const end = Math.min(samples.length, start + totalSamples);
  const slice = samples.slice(start, end);
  if (slice.length === totalSamples) return slice;

  const padded = new Float32Array(totalSamples);
  padded.set(slice);
  return padded;
}

function refineAttackTimestamp(
  samples: Float32Array,
  sampleRate: number,
  approxTimestampMs: number,
): number {
  const totalDurationMs = (samples.length / sampleRate) * 1000;
  const localStartMs = Math.max(0, approxTimestampMs - AUTO_REFINE_SEARCH_PRE_MS);
  const localEndMs = Math.min(totalDurationMs, approxTimestampMs + AUTO_REFINE_SEARCH_POST_MS);
  const localEnvelope = buildLocalEnvelope(samples, sampleRate, localStartMs, localEndMs);
  if (localEnvelope.values.length < 3) {
    return clampMs(approxTimestampMs, totalDurationMs);
  }

  const peaks = detectLocalEnvelopePeaks(localEnvelope.values);
  if (peaks.length === 0) {
    return clampMs(approxTimestampMs, totalDurationMs);
  }

  const peakIndex = peaks.reduce((best, current) =>
    localEnvelope.values[current] > localEnvelope.values[best] ? current : best,
  );

  const sortedLocal = [...localEnvelope.values].sort((a, b) => a - b);
  const noiseFloor = percentile(sortedLocal, 0.2);
  const peakValue = localEnvelope.values[peakIndex];
  const attackThreshold = Math.max(
    MIN_ENVELOPE_THRESHOLD * 0.4,
    noiseFloor * 1.7,
    peakValue * 0.18,
  );

  let onsetIndex = peakIndex;
  while (onsetIndex > 0 && localEnvelope.values[onsetIndex - 1] >= attackThreshold) {
    onsetIndex -= 1;
  }

  const onsetMs = localStartMs + onsetIndex * localEnvelope.hop_ms;
  return clampMs(onsetMs, totalDurationMs);
}

function findLocalPeakTimestamp(
  samples: Float32Array,
  sampleRate: number,
  startMs: number,
  endMs: number,
): number {
  const localEnvelope = buildLocalEnvelope(samples, sampleRate, startMs, endMs);
  if (localEnvelope.values.length === 0) {
    return clampMs(startMs, (samples.length / sampleRate) * 1000);
  }

  let peakIndex = 0;
  for (let i = 1; i < localEnvelope.values.length; i++) {
    if (localEnvelope.values[i] > localEnvelope.values[peakIndex]) {
      peakIndex = i;
    }
  }

  return startMs + peakIndex * localEnvelope.hop_ms;
}

function buildLocalEnvelope(
  samples: Float32Array,
  sampleRate: number,
  startMs: number,
  endMs: number,
): { values: number[]; hop_ms: number } {
  const frameSize = Math.max(24, Math.round((sampleRate * LOCAL_FRAME_MS) / 1000));
  const hopSize = Math.max(12, Math.round((sampleRate * LOCAL_HOP_MS) / 1000));
  const startSample = Math.max(0, Math.floor((startMs / 1000) * sampleRate));
  const endSample = Math.min(samples.length, Math.ceil((endMs / 1000) * sampleRate));
  const values: number[] = [];

  for (let start = startSample; start + frameSize <= endSample; start += hopSize) {
    let energy = 0;
    for (let i = start; i < start + frameSize; i++) {
      energy += samples[i] * samples[i];
    }
    values.push(Math.sqrt(energy / frameSize));
  }

  return {
    values,
    hop_ms: (hopSize / sampleRate) * 1000,
  };
}

function detectLocalEnvelopePeaks(values: number[]): number[] {
  if (values.length < 3) return [];
  const sorted = [...values].sort((a, b) => a - b);
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const p85 = percentile(sorted, 0.85);
  const p97 = percentile(sorted, 0.97);
  const threshold = Math.max(MIN_ENVELOPE_THRESHOLD * 0.5, mean * 1.25, p85 * 0.75, p97 * 0.4);
  const peaks: number[] = [];

  for (let i = 1; i < values.length - 1; i++) {
    const current = values[i];
    if (current < threshold) continue;
    if (current >= values[i - 1] && current >= values[i + 1]) {
      peaks.push(i);
    }
  }

  return peaks;
}

function clampMs(timestampMs: number, totalDurationMs: number): number {
  return Math.max(0, Math.min(Math.round(totalDurationMs), Math.round(timestampMs)));
}

function stretchSamplesForPlayback(samples: Float32Array, playbackRate: number): Float32Array {
  if (samples.length === 0) return samples;
  const safeRate = Math.max(0.1, playbackRate);
  if (Math.abs(safeRate - 1) < 0.001) return samples;

  const stretchedLength = Math.max(1, Math.round(samples.length / safeRate));
  const stretched = new Float32Array(stretchedLength);

  for (let i = 0; i < stretchedLength; i++) {
    const sourceIndex = i * safeRate;
    const leftIndex = Math.floor(sourceIndex);
    const rightIndex = Math.min(samples.length - 1, leftIndex + 1);
    const mix = sourceIndex - leftIndex;
    const left = samples[Math.min(leftIndex, samples.length - 1)];
    const right = samples[rightIndex];
    stretched[i] = left + (right - left) * mix;
  }

  return stretched;
}

function encodeMonoPcm16Wav(samples: Float32Array, sampleRate: number): Uint8Array {
  const dataBytes = samples.length * 2;
  const buffer = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buffer);

  writeAscii(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataBytes, true);
  writeAscii(view, 8, 'WAVE');
  writeAscii(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, 'data');
  view.setUint32(40, dataBytes, true);

  for (let i = 0; i < samples.length; i++) {
    const clipped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, Math.round(clipped * 32767), true);
  }

  return new Uint8Array(buffer);
}

function writeAscii(view: DataView, offset: number, value: string) {
  for (let i = 0; i < value.length; i++) {
    view.setUint8(offset + i, value.charCodeAt(i));
  }
}
