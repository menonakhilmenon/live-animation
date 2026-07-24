import { FFT } from './fft';

/**
 * Offline whole-file audio analysis. Because the user accepts near-realtime
 * latency, we analyze the entire buffer up front and drive animation from a
 * precise, frame-accurate timeline instead of causal per-frame guesses:
 * per-frame loudness, spectral brightness, a spectral-flux onset envelope
 * (the timing backbone of co-speech gesture), and a coarse speech / silence
 * classification.
 */

export type ContentMode = 'speech' | 'silence';

export interface Timeline {
  duration: number;
  hopTime: number;
  mode: ContentMode;
  /** Mean 300–3000 Hz energy fraction over active frames (classifier input). */
  speechRatio: number;
  frames: {
    rms: Float32Array;
    centroid: Float32Array;
    onset: Float32Array;
  };
}

const FRAME = 2048;
const HOP = 512;

export function analyzeBuffer(samples: Float32Array, sampleRate: number): Timeline {
  const hopTime = HOP / sampleRate;
  const nFrames = Math.max(0, Math.floor((samples.length - FRAME) / HOP) + 1);
  const fft = new FFT(FRAME);
  const window = new Float32Array(FRAME);
  for (let i = 0; i < FRAME; i++) window[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / FRAME);

  const nBins = FRAME / 2;
  const nyquist = sampleRate / 2;
  const binHz = nyquist / nBins;
  const bin = (hz: number) => Math.min(nBins - 1, Math.max(0, Math.round(hz / binHz)));
  const B = { bassHi: bin(250), trebleHi: bin(8000), speechLo: bin(300), speechHi: bin(3000) };

  const rms = new Float32Array(nFrames);
  const centroid = new Float32Array(nFrames);
  const speechBand = new Float32Array(nFrames);
  const onset = new Float32Array(nFrames);
  const bassFlux = new Float32Array(nFrames);

  const re = new Float32Array(FRAME);
  const im = new Float32Array(FRAME);
  let prevLogMag: Float32Array | null = null;
  const logMag = new Float32Array(nBins);

  for (let f = 0; f < nFrames; f++) {
    const off = f * HOP;
    let sq = 0;
    for (let i = 0; i < FRAME; i++) {
      const s = samples[off + i];
      sq += s * s;
      re[i] = s * window[i];
      im[i] = 0;
    }
    rms[f] = Math.sqrt(sq / FRAME);
    fft.transform(re, im);

    let speechE = 0, totalE = 0, wSum = 0;
    let fluxAll = 0, fluxBass = 0;
    for (let k = 1; k < nBins; k++) {
      const mag = Math.hypot(re[k], im[k]);
      const lm = Math.log1p(10 * mag);
      if (prevLogMag) {
        const d = lm - prevLogMag[k];
        if (d > 0 && k <= B.trebleHi) {
          fluxAll += d;
          if (k <= B.bassHi) fluxBass += d;
        }
      }
      logMag[k] = lm;
      totalE += mag;
      wSum += k * mag;
      if (k >= B.speechLo && k <= B.speechHi) speechE += mag;
    }
    speechBand[f] = totalE > 1e-6 ? speechE / totalE : 0;
    centroid[f] = totalE > 1e-6 ? Math.min(1, (wSum / totalE / nBins) * 4) : 0;
    onset[f] = fluxAll;
    bassFlux[f] = fluxBass;
    prevLogMag = prevLogMag ?? new Float32Array(nBins);
    prevLogMag.set(logMag);
  }

  // Onset envelope conditioning. Bass flux and full-band flux are
  // normalized SEPARATELY before combining: a kick lives in ~3 FFT bins
  // while a hi-hat spreads over hundreds, so raw flux sums let broadband
  // noise dominate. After per-envelope normalization, low-frequency
  // percussive onsets compete on equal footing with broadband ones.
  const bassEnv = bassFlux.slice();
  normalizeByPercentile(bassEnv, 0.95);
  const fullEnv = onset.slice();
  normalizeByPercentile(fullEnv, 0.95);
  const combined = new Float32Array(nFrames);
  for (let f = 0; f < nFrames; f++) combined[f] = bassEnv[f] + 0.5 * fullEnv[f];

  const meanWin = Math.round(0.4 / hopTime);
  const conditioned = subtractMovingMean(combined, meanWin);
  normalizeByPercentile(conditioned, 0.95);

  const { mode, speechRatio } = classifyContent(rms, speechBand);

  return {
    duration: samples.length / sampleRate,
    hopTime,
    mode,
    speechRatio,
    frames: { rms, centroid, onset: conditioned },
  };
}

function movingMean(x: Float32Array, win: number): Float32Array {
  const out = new Float32Array(x.length);
  let sum = 0;
  const half = Math.max(1, Math.floor(win / 2));
  for (let i = 0; i < Math.min(x.length, half); i++) sum += x[i];
  let count = Math.min(x.length, half);
  for (let i = 0; i < x.length; i++) {
    if (i + half < x.length) {
      sum += x[i + half];
      count++;
    }
    if (i - half - 1 >= 0) {
      sum -= x[i - half - 1];
      count--;
    }
    out[i] = sum / Math.max(1, count);
  }
  return out;
}

function subtractMovingMean(x: Float32Array, win: number): Float32Array {
  const mean = movingMean(x, win);
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = Math.max(0, x[i] - mean[i]);
  return out;
}

function normalizeByPercentile(x: Float32Array, p: number, cap = 1.5): void {
  const sorted = Array.from(x).sort((a, b) => a - b);
  const ref = sorted[Math.floor(sorted.length * p)] || 1;
  if (ref <= 0) return;
  for (let i = 0; i < x.length; i++) x[i] = Math.min(cap, x[i] / ref);
}

function classifyContent(
  rms: Float32Array,
  speechBand: Float32Array,
): { mode: ContentMode; speechRatio: number } {
  const active = Array.from(rms).filter((v) => v > 0.01);
  if (active.length < rms.length * 0.1) return { mode: 'silence', speechRatio: 0 };

  // Mean speech-band dominance over active frames only.
  let speechSum = 0;
  let n = 0;
  for (let i = 0; i < rms.length; i++) {
    if (rms[i] > 0.01) {
      speechSum += speechBand[i];
      n++;
    }
  }
  const speechRatio = n ? speechSum / n : 0;
  return { mode: 'speech', speechRatio };
}
