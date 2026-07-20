/**
 * Audio features extracted per frame. All values are normalized to roughly
 * [0, 1] so animation layers can consume them without knowing about FFTs.
 */
export interface AudioFeatures {
  /** Overall loudness (RMS of the time-domain signal), smoothed. */
  rms: number;
  /** Energy in ~20–250 Hz. Drives bounce / weight. */
  bass: number;
  /** Energy in ~250–2000 Hz. Vocals and most melodic content. */
  mid: number;
  /** Energy in ~2–8 kHz. Drives small, sharp motion. */
  treble: number;
  /**
   * Spectral centroid mapped to [0, 1] — a cheap "brightness" measure.
   * Dark/bassy audio ≈ 0, bright/hissy audio ≈ 1.
   */
  brightness: number;
  /** True on the frame a beat is detected (bass onset). */
  beat: boolean;
  /**
   * Decaying beat envelope: jumps toward 1 on a beat and falls off over
   * ~300 ms. Use this (not `beat`) for anything continuous.
   */
  beatPulse: number;
  /** How strong the last detected beat was relative to the recent average. */
  beatStrength: number;
  /** Seconds since the last beat (large when idle). */
  timeSinceBeat: number;
  /** Estimated tempo in BPM (0 until the tracker locks on). */
  bpm: number;
  /** 0–1 confidence in the tempo estimate. */
  tempoConfidence: number;
  /**
   * Continuous bar phase (4 beats per bar), integer part = bar index,
   * downbeat at fractional part 0. Only meaningful with an offline timeline;
   * falls back to beatPhase/4 for live input.
   */
  barPhase: number;
  /** 0–1 loudness of the current song section (quiet verse → loud chorus). */
  section: number;
  /** Seconds until the next detected drop (Infinity when none ahead). */
  nextDropIn: number;
  /** Content mode: 'music' | 'speech' | 'silence' | 'live' (no timeline). */
  mode: string;
  /**
   * Continuous beat phase from the phase-locked tempo oscillator; the
   * fractional part is the position within the current beat (0 = on the
   * beat). Lets animation anticipate beats instead of reacting to them.
   */
  beatPhase: number;
}

export function emptyFeatures(): AudioFeatures {
  return {
    rms: 0,
    bass: 0,
    mid: 0,
    treble: 0,
    brightness: 0,
    beat: false,
    beatPulse: 0,
    beatStrength: 0,
    timeSinceBeat: 1e9,
    bpm: 0,
    tempoConfidence: 0,
    beatPhase: 0,
    barPhase: 0,
    section: 0.5,
    nextDropIn: Infinity,
    mode: 'live',
  };
}

/** Exponential moving average with a time constant in seconds. */
export function ema(current: number, target: number, tau: number, dt: number): number {
  const a = 1 - Math.exp(-dt / Math.max(tau, 1e-4));
  return current + (target - current) * a;
}
