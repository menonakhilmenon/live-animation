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
export class FaceAnimator {
  /** External emotional bias (e.g. from a generated clip's emotion). */
  moodBias = 0;

  private aa = 0;
  private ih = 0;
  private ou = 0;
  private mood = 0;
  private blinkTimer = 1 + Math.random() * 1.5; // first blink lands early
  private blinkPhase = -1; // <0: idle, otherwise progress through a blink
  private blinkCount = 0;

  constructor(private face: FaceDriver) {}

  update(f: AudioFeatures, dt: number): void {
    // --- Mouth ---
    let open = 0;
    if (f.mode === 'speech') {
      open = Math.min(1, Math.pow(f.rms * 4.5, 0.85));
    } else if (f.mode === 'music') {
      // Quiet "singing along" only in energetic sections.
      open = Math.min(0.35, f.mid * 0.5 * f.section);
    }
    const bright = f.brightness;
    const ihShare = Math.min(1, Math.max(0, (bright - 0.28) / 0.2));
    const ouShare = Math.min(1, Math.max(0, (0.22 - bright) / 0.12));
    const aaShare = Math.max(0, 1 - ihShare - ouShare);
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
