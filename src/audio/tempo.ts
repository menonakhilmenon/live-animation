import { ema } from './features';

/** Track tempo in the human-danceable range. */
const MIN_PERIOD = 60 / 180; // 180 BPM
const MAX_PERIOD = 60 / 60; //  60 BPM
const HISTORY = 16;
/** How strongly a detected beat pulls the oscillator phase back on-beat. */
const PLL_GAIN = 0.35;

/**
 * Estimates tempo from detected beat onsets and runs a phase-locked
 * oscillator so animation can *anticipate* beats instead of reacting to
 * them. Inter-onset intervals are folded into the 60–180 BPM octave (a
 * missed beat then still votes for the same tempo), the period is the
 * running median, and confidence is the fraction of recent intervals that
 * agree with it.
 */
export class TempoTracker {
  /** Estimated tempo. 0 until enough beats have been seen. */
  bpm = 0;
  /** 0–1: how consistently recent onsets fit the estimated period. */
  confidence = 0;
  /**
   * Continuous beat phase; fractional part is position within the current
   * beat (0 = on the beat). Advances every frame once tempo is known.
   */
  phase = 0;

  private onsets: number[] = [];
  private period = 0;
  private timeSec = 0;

  /** Call every frame; `beat` is the detector's onset flag for this frame. */
  update(dt: number, beat: boolean): void {
    this.timeSec += dt;

    if (beat) this.onBeat();

    if (this.period > 0) {
      this.phase += dt / this.period;
      // Decay confidence if beats stop arriving (song ended, breakdown).
      const sinceLast = this.timeSec - (this.onsets[this.onsets.length - 1] ?? -1e9);
      if (sinceLast > this.period * 4) {
        this.confidence = ema(this.confidence, 0, 1.5, dt);
      }
    }
  }

  private onBeat(): void {
    this.onsets.push(this.timeSec);
    if (this.onsets.length > HISTORY) this.onsets.shift();
    if (this.onsets.length < 5) return;

    // Fold intervals into [MIN_PERIOD, MAX_PERIOD] so half/double-time
    // detections vote for the same underlying tempo.
    const folded: number[] = [];
    for (let i = 1; i < this.onsets.length; i++) {
      let iv = this.onsets[i] - this.onsets[i - 1];
      if (iv > MAX_PERIOD * 2) continue; // gap — not adjacent beats
      while (iv > MAX_PERIOD) iv /= 2;
      while (iv < MIN_PERIOD) iv *= 2;
      folded.push(iv);
    }
    if (folded.length < 4) return;

    const sorted = [...folded].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    const agreeing = folded.filter((iv) => Math.abs(iv - median) / median < 0.1);
    this.confidence = agreeing.length / folded.length;

    if (this.confidence < 0.5) return;

    // Refine the period from the agreeing intervals only.
    const refined = agreeing.reduce((a, b) => a + b, 0) / agreeing.length;
    this.period = this.period === 0 ? refined : this.period * 0.7 + refined * 0.3;
    this.bpm = 60 / this.period;

    // Phase-locked loop: nudge the oscillator so phase ≈ 0 lands on beats.
    const err = this.phase - Math.round(this.phase); // [-0.5, 0.5)
    this.phase -= err * PLL_GAIN;
  }
}
