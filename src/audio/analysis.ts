import { FFT } from './fft';

/**
 * Offline whole-file audio analysis. Because the user accepts near-realtime
 * latency, we analyze the entire buffer up front and drive animation from a
 * precise, *future-aware* timeline instead of causal per-frame guesses:
 * exact beat grid (spectral-flux onsets + tempo autocorrelation + DP beat
 * selection), downbeats, loudness sections with drop marking, and a coarse
 * music / speech / silence classification.
 */

export type SectionLevel = 'quiet' | 'mid' | 'loud';
export type ContentMode = 'music' | 'speech' | 'silence';

export interface Section {
  start: number;
  end: number;
  level: SectionLevel;
  /** True when this section begins with a sharp loudness jump (a drop). */
  drop: boolean;
}

export interface Timeline {
  duration: number;
  hopTime: number;
  bpm: number;
  /** 0–1 normalized autocorrelation peak — how strongly periodic the onsets are. */
  tempoStrength: number;
  /** Beat times in seconds (empty when no reliable tempo). */
  beats: number[];
  /** beats[i] with i % 4 === downbeatOffset are bar starts. */
  downbeatOffset: number;
  sections: Section[];
  mode: ContentMode;
  /** Mean 300–3000 Hz energy fraction over active frames (classifier input). */
  speechRatio: number;
  frames: {
    rms: Float32Array;
    bass: Float32Array;
    mid: Float32Array;
    treble: Float32Array;
    centroid: Float32Array;
    /** Slow (~1.5 s) normalized loudness envelope. */
    loudness: Float32Array;
    onset: Float32Array;
  };
}

const FRAME = 2048;
const HOP = 512;
const TIGHTNESS = 100; // DP beat tracking: penalty weight on tempo deviation

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
  const B = { bassLo: bin(20), bassHi: bin(250), midHi: bin(2000), trebleHi: bin(8000), speechLo: bin(300), speechHi: bin(3000) };

  const rms = new Float32Array(nFrames);
  const bass = new Float32Array(nFrames);
  const mid = new Float32Array(nFrames);
  const treble = new Float32Array(nFrames);
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

    let bassE = 0, midE = 0, trebleE = 0, speechE = 0, totalE = 0, wSum = 0;
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
      if (k >= B.bassLo && k <= B.bassHi) bassE += mag;
      else if (k <= B.midHi) midE += mag;
      else if (k <= B.trebleHi) trebleE += mag;
      if (k >= B.speechLo && k <= B.speechHi) speechE += mag;
    }
    const norm = 1 / (nBins * 0.02); // rough magnitude normalization
    bass[f] = Math.min(1, (bassE / (B.bassHi - B.bassLo + 1)) * norm * 8);
    mid[f] = Math.min(1, (midE / (B.midHi - B.bassHi)) * norm * 8);
    treble[f] = Math.min(1, (trebleE / (B.trebleHi - B.midHi)) * norm * 8);
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
  // noise dominate and can pull the beat grid onto off-beats. After
  // per-envelope normalization, low-frequency percussive onsets (which
  // define the beat) compete on equal footing.
  const bassEnv = bassFlux.slice();
  normalizeByPercentile(bassEnv, 0.95);
  const fullEnv = onset.slice();
  normalizeByPercentile(fullEnv, 0.95);
  const combined = new Float32Array(nFrames);
  for (let f = 0; f < nFrames; f++) combined[f] = bassEnv[f] + 0.5 * fullEnv[f];

  // Uncapped copy for downbeat scoring — the 1.5 cap used for beat tracking
  // flattens accent differences (all loud-section kicks saturate), which is
  // exactly the information downbeat detection needs.
  const accentEnv = bassFlux.slice();
  normalizeByPercentile(accentEnv, 0.95, Infinity);
  const accentFull = onset.slice();
  normalizeByPercentile(accentFull, 0.95, Infinity);
  for (let f = 0; f < nFrames; f++) accentEnv[f] += 0.5 * accentFull[f];
  const meanWin = Math.round(0.4 / hopTime);
  const conditioned = subtractMovingMean(combined, meanWin);
  normalizeByPercentile(conditioned, 0.95);

  const { bpm, periodFrames, tempoStrength } = estimateTempo(conditioned, hopTime);
  // Beat processing happens in frame time; the window-center offset is
  // applied once at the end so grid fitting and onset lookups stay aligned.
  let beats =
    tempoStrength > 0.1 && periodFrames > 0
      ? trackBeats(conditioned, periodFrames).map((f) => f * hopTime)
      : [];
  if (beats.length) beats = regularizeBeats(beats, conditioned, periodFrames * hopTime, hopTime);
  beats = beats.map((t) => t + FRAME / (2 * sampleRate));

  // Downbeats: which beat-phase carries the most bass onset energy.
  let downbeatOffset = 0;
  if (beats.length >= 8) {
    let best = -Infinity;
    for (let o = 0; o < 4; o++) {
      let sum = 0;
      for (let i = o; i < beats.length; i += 4) {
        const f = Math.min(nFrames - 1, Math.round((beats[i] * sampleRate - FRAME / 2) / HOP));
        // ±1 frame: sharp (crash-marked) onsets can peak a frame early.
        for (let k = Math.max(0, f - 1); k <= Math.min(nFrames - 1, f + 1); k++) sum += accentEnv[k];
      }
      if (sum > best) {
        best = sum;
        downbeatOffset = o;
      }
    }
  }

  // Loudness envelope + sections.
  const loudness = movingMean(rms, Math.round(1.5 / hopTime));
  normalizeByPercentile(loudness, 0.95);
  const sections = segmentSections(loudness, hopTime);

  const { mode, speechRatio } = classifyContent(rms, speechBand, tempoStrength);

  return {
    duration: samples.length / sampleRate,
    hopTime,
    bpm,
    tempoStrength,
    beats,
    downbeatOffset,
    sections,
    mode,
    speechRatio,
    frames: { rms, bass, mid, treble, centroid, loudness, onset: conditioned },
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

function estimateTempo(onset: Float32Array, hopTime: number): {
  bpm: number;
  periodFrames: number;
  tempoStrength: number;
} {
  const minLag = Math.round(60 / 180 / hopTime);
  const maxLag = Math.round(60 / 60 / hopTime);
  if (onset.length < maxLag * 3) return { bpm: 0, periodFrames: 0, tempoStrength: 0 };

  let acorr0 = 1e-9;
  for (let i = 0; i < onset.length; i++) acorr0 += onset[i] * onset[i];

  let bestLag = 0;
  let bestVal = 0;
  for (let lag = minLag; lag <= maxLag; lag++) {
    let sum = 0;
    for (let i = lag; i < onset.length; i++) sum += onset[i] * onset[i - lag];
    // Log-normal prior centered on 120 BPM, one octave standard deviation.
    const period = lag * hopTime;
    const prior = Math.exp(-0.5 * Math.pow(Math.log2(period / 0.5), 2));
    const val = (sum / acorr0) * prior;
    if (val > bestVal) {
      bestVal = val;
      bestLag = lag;
    }
  }
  return { bpm: bestLag ? 60 / (bestLag * hopTime) : 0, periodFrames: bestLag, tempoStrength: bestVal };
}

/**
 * Ellis-style dynamic-programming beat selection: pick frame indices that
 * maximize onset strength while spacing beats near the estimated period.
 */
function trackBeats(onset: Float32Array, period: number): number[] {
  const n = onset.length;
  const score = new Float32Array(n).fill(-Infinity);
  const back = new Int32Array(n).fill(-1);
  const lo = Math.round(period * 0.5);
  const hi = Math.round(period * 2);

  for (let i = 0; i < n; i++) {
    score[i] = onset[i];
    const jMin = Math.max(0, i - hi);
    const jMax = i - lo;
    for (let j = jMin; j <= jMax; j++) {
      if (score[j] === -Infinity) continue;
      const dev = Math.log((i - j) / period);
      const s = score[j] - TIGHTNESS * dev * dev * 0.01 + onset[i];
      if (s > score[i]) {
        score[i] = s;
        back[i] = j;
      }
    }
  }

  // Backtrack from the best-scoring frame near the end.
  let end = n - 1;
  for (let i = Math.max(0, n - hi); i < n; i++) if (score[i] > score[end]) end = i;
  const beats: number[] = [];
  for (let i = end; i >= 0; i = back[i]) {
    beats.push(i);
    if (back[i] === -1) break;
  }
  return beats.reverse();
}

/**
 * Regularize the DP output into an arithmetic beat grid. The DP is good at
 * finding the period but its head/tail can be ragged, which breaks
 * index-mod-4 downbeat phase. So: refine the period from the DP output's
 * central spacings, exhaustively search the grid phase that maximizes onset
 * energy at grid points, then emit anchor + n*period with each output beat
 * snapped to the local onset peak (±10% period) for sub-hop accuracy.
 */
function regularizeBeats(
  beats: number[],
  onset: Float32Array,
  nominalPeriod: number,
  hopTime: number,
): number[] {
  const duration = onset.length * hopTime;
  const strength = (t: number) => {
    const f = Math.round(t / hopTime);
    return f >= 0 && f < onset.length ? onset[f] : 0;
  };

  // Median of plausible consecutive spacings refines the period.
  const diffs = beats
    .slice(1)
    .map((b, i) => b - beats[i])
    .filter((d) => d > nominalPeriod * 0.7 && d < nominalPeriod * 1.3)
    .sort((a, b) => a - b);
  const period = diffs.length ? diffs[Math.floor(diffs.length / 2)] : nominalPeriod;

  // Phase search at hop resolution.
  let bestPhase = 0;
  let bestScore = -Infinity;
  for (let phase = 0; phase < period; phase += hopTime) {
    let score = 0;
    for (let t = phase; t < duration; t += period) score += strength(t);
    if (score > bestScore) {
      bestScore = score;
      bestPhase = phase;
    }
  }

  // Emit the grid, snapping each beat to the strongest onset nearby.
  const snapWin = Math.max(1, Math.round((period * 0.1) / hopTime));
  const out: number[] = [];
  for (let t = bestPhase; t < duration; t += period) {
    const f0 = Math.round(t / hopTime);
    let bestF = Math.max(0, Math.min(onset.length - 1, f0));
    for (let f = Math.max(0, f0 - snapWin); f <= Math.min(onset.length - 1, f0 + snapWin); f++) {
      if (onset[f] > onset[bestF]) bestF = f;
    }
    out.push(bestF * hopTime);
  }
  return out;
}

function segmentSections(loudness: Float32Array, hopTime: number): Section[] {
  const level = (v: number): SectionLevel => (v >= 0.6 ? 'loud' : v <= 0.32 ? 'quiet' : 'mid');
  const minLen = 1.5 / hopTime;
  const sections: Section[] = [];
  let start = 0;
  let cur = level(loudness[0] ?? 0);
  for (let i = 1; i <= loudness.length; i++) {
    const l = i === loudness.length ? null : level(loudness[i]);
    if (l !== cur) {
      if (i - start >= minLen || sections.length === 0) {
        sections.push({ start: start * hopTime, end: i * hopTime, level: cur, drop: false });
        start = i;
        cur = l ?? cur;
      } else if (l !== null) {
        // Too short — absorb into the previous section.
        start = sections.length ? i : start;
        cur = l;
        if (sections.length) sections[sections.length - 1].end = i * hopTime;
      }
    }
  }
  if (sections.length === 0) sections.push({ start: 0, end: loudness.length * hopTime, level: cur, drop: false });
  else sections[sections.length - 1].end = loudness.length * hopTime;

  // A "drop": loud section entered from a non-loud one with a fast rise.
  for (let s = 1; s < sections.length; s++) {
    if (sections[s].level === 'loud' && sections[s - 1].level !== 'loud') {
      const i0 = Math.max(0, Math.round(sections[s].start / hopTime) - Math.round(0.75 / hopTime));
      const i1 = Math.min(loudness.length - 1, Math.round(sections[s].start / hopTime) + Math.round(0.75 / hopTime));
      const rise = loudness[i1] - loudness[i0];
      if (rise > 0.25) sections[s].drop = true;
    }
  }
  return sections;
}

function classifyContent(
  rms: Float32Array,
  speechBand: Float32Array,
  tempoStrength: number,
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

  // Strongly periodic onsets → music, regardless of band balance (songs
  // with vocals have high speech-band energy too). Weak periodicity plus
  // speech-band dominance → speech. Speech syllables autocorrelate weakly
  // (quasi-regular), so the margin between the two is what discriminates.
  if (tempoStrength > 0.45) return { mode: 'music', speechRatio };
  if (speechRatio > 0.45 && tempoStrength < 0.35) return { mode: 'speech', speechRatio };
  return { mode: 'music', speechRatio };
}
