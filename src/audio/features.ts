/**
 * Audio features extracted per frame. All values are normalized to roughly
 * [0, 1] so animation layers can consume them without knowing about FFTs.
 */
export interface AudioFeatures {
  /** Overall loudness (RMS of the time-domain signal), smoothed. */
  rms: number;
  /**
   * Spectral centroid mapped to [0, 1] — a cheap "brightness" measure.
   * Dark/bassy audio ≈ 0, bright/hissy audio ≈ 1.
   */
  brightness: number;
  /** Content mode: 'speech' | 'silence' | 'live' (no timeline). */
  mode: string;
  /**
   * Instantaneous onset strength (~0–1.5): spectral-flux novelty at the
   * current playback frame. Speech gestures key on this. 0 for live input.
   */
  onset: number;
}

export function emptyFeatures(): AudioFeatures {
  return {
    rms: 0,
    brightness: 0,
    mode: 'live',
    onset: 0,
  };
}

/** Exponential moving average with a time constant in seconds. */
export function ema(current: number, target: number, tau: number, dt: number): number {
  const a = 1 - Math.exp(-dt / Math.max(tau, 1e-4));
  return current + (target - current) * a;
}
