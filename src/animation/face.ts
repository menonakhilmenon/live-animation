import { AudioFeatures } from '../audio/features';
import { FaceDriver } from '../rig/humanoid';
import { ema } from '../audio/features';

/**
 * Drives facial expressions from audio features:
 *  - lip sync in speech mode: mouth opening from level, viseme chosen by
 *    spectral brightness (dark → ou, mid → aa, bright → ih)
 *  - subtle "singing along" mouth in loud music sections
 *  - blinking on a natural randomized schedule
 *  - mood: happier as musical energy rises, relaxed in silence
 */
/** One timed lip-sync event from the TTS: [t0, t1, viseme, weight]. */
export type VisemeEvent = [number, number, 'aa' | 'ih' | 'ou' | 'sil', number];

export class FaceAnimator {
  /** External emotional bias (e.g. from a generated clip's emotion). */
  moodBias = 0;

  /** Phoneme-timed lip sync (from TTS); null falls back to the audio
   * brightness heuristic. `clock` returns playback time in seconds. */
  private track: { events: VisemeEvent[]; clock: () => number } | null = null;
  private trackIdx = 0;

  private aa = 0;
  private ih = 0;
  private ou = 0;
  private mood = 0;
  private blinkTimer = 1 + Math.random() * 1.5; // first blink lands early
  private blinkPhase = -1; // <0: idle, otherwise progress through a blink
  private blinkCount = 0;

  constructor(private face: FaceDriver) {}

  setVisemeTrack(events: VisemeEvent[] | null, clock?: () => number): void {
    this.track = events && events.length && clock ? { events, clock } : null;
    this.trackIdx = 0;
  }

  get hasVisemeTrack(): boolean {
    return this.track !== null;
  }

  /** Active viseme event at time t, or null (events are time-sorted). */
  private eventAt(t: number): VisemeEvent | null {
    const ev = this.track!.events;
    if (this.trackIdx > 0 && t < ev[this.trackIdx][0]) this.trackIdx = 0; // seek back
    while (this.trackIdx < ev.length - 1 && ev[this.trackIdx][1] < t) this.trackIdx++;
    const e = ev[this.trackIdx];
    return e[0] <= t && t <= e[1] ? e : null;
  }

  update(f: AudioFeatures, dt: number): void {
    // --- Mouth ---
    let open = 0;
    if (f.mode === 'speech') {
      open = Math.min(1, Math.pow(f.rms * 4.5, 0.85));
    } else if (f.mode === 'music') {
      // Quiet "singing along" only in energetic sections.
      open = Math.min(0.35, f.mid * 0.5 * f.section);
    }
    let ihShare: number, ouShare: number, aaShare: number;
    const ev = this.track ? this.eventAt(this.track.clock()) : null;
    if (ev) {
      // Phoneme timing chooses the viseme; loudness still scales opening
      // so the mouth follows the actual energy of the voice.
      aaShare = ev[2] === 'aa' ? 1 : 0;
      ihShare = ev[2] === 'ih' ? 1 : 0;
      ouShare = ev[2] === 'ou' ? 1 : 0;
      open = Math.min(1, (0.35 + Math.min(1, f.rms * 4.5) * 0.75) * ev[3]);
    } else if (this.track) {
      open = 0; // between phonemes / silence
      aaShare = ihShare = ouShare = 0;
    } else {
      const bright = f.brightness;
      ihShare = Math.min(1, Math.max(0, (bright - 0.28) / 0.2));
      ouShare = Math.min(1, Math.max(0, (0.22 - bright) / 0.12));
      aaShare = Math.max(0, 1 - ihShare - ouShare);
    }
    // Fast attack, slower release — mouths close slower than they open.
    const tau = (target: number, cur: number) => (target > cur ? 0.04 : 0.09);
    this.aa = ema(this.aa, open * aaShare, tau(open * aaShare, this.aa), dt);
    this.ih = ema(this.ih, open * ihShare, tau(open * ihShare, this.ih), dt);
    this.ou = ema(this.ou, open * ouShare, tau(open * ouShare, this.ou), dt);
    this.face.setMouth(this.aa, this.ih, this.ou);

    // --- Blink ---
    let blink = 0;
    if (this.blinkPhase >= 0) {
      this.blinkPhase += dt / 0.15; // 150 ms blink
      blink = this.blinkPhase < 0.5 ? this.blinkPhase * 2 : Math.max(0, 2 - this.blinkPhase * 2);
      if (this.blinkPhase >= 1) this.blinkPhase = -1;
    } else {
      this.blinkTimer -= dt;
      if (this.blinkTimer <= 0) {
        this.blinkPhase = 0;
        this.blinkCount++;
        this.blinkTimer = 2 + Math.random() * 3;
      }
    }
    this.face.setBlink(blink);

    // --- Mood ---
    const base =
      f.mode === 'music' ? 0.15 + f.section * 0.5 : f.mode === 'speech' ? 0.1 : -0.3;
    const target = Math.max(-1, Math.min(1, base + this.moodBias));
    this.mood = ema(this.mood, target, 1.2, dt);
    this.face.setMood(this.mood);
  }

  debug(): Record<string, number> {
    return { aa: this.aa, ih: this.ih, ou: this.ou, mood: this.mood, blinkCount: this.blinkCount };
  }
}
