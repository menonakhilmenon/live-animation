import { AudioFeatures } from '../audio/features';
import { HumanoidRig, resetToRest } from '../rig/humanoid';
import { Spring } from './spring';

/**
 * Layered procedural animator. Each frame it resets the rig to its rest pose
 * and applies motion layers on top, each scaled by live audio features:
 *
 *  - breathing   — always on; grows subtler as energy rises
 *  - groove      — hips bounce + knee flex from bass and the beat pulse
 *  - sway        — weight shift and torso lean from overall energy
 *  - head        — nod on beats, micro-bob with level, tilt with brightness
 *  - arms        — swing with energy, raise with sustained intensity
 */
export class Animator {
  private t = 0;
  /** Phase accumulator that speeds up with energy so motion follows intensity. */
  private groovePhase = 0;

  private bounce = new Spring(0, 60);
  private energy = new Spring(0, 8);
  private headNod = new Spring(0, 55);
  private armRaise = new Spring(0, 10);
  private sideStep = new Spring(0, 12);
  /** Which side the weight is on: alternates on beats. */
  private weightSide = 1;

  constructor(private rig: HumanoidRig) {}

  update(f: AudioFeatures, dt: number): void {
    dt = Math.min(dt, 1 / 20); // avoid spring blow-ups on tab-switch stalls
    this.t += dt;

    // Sustained intensity (slow spring) gates most layers so the character
    // relaxes to idle in silence and commits to the groove on loud sections.
    const energy = this.energy.update(Math.min(1, f.rms * 3 + f.beatPulse * 0.3), dt);
    this.groovePhase += dt * (0.8 + energy * 3.2);

    if (f.beat) this.weightSide = -this.weightSide;

    const bounce = this.bounce.update(f.beatPulse * 0.09 + f.bass * 0.03, dt);
    const nod = this.headNod.update(f.beatPulse, dt);
    const raise = this.armRaise.update(Math.max(0, energy - 0.45) * 1.8, dt);
    const side = this.sideStep.update(this.weightSide * Math.min(1, energy * 1.4), dt);

    const r = this.rig;
    resetToRest(r);
    const j = r.joints;
    // Positional offsets below are authored in meters; ps converts them to
    // the rig's local units (see HumanoidRig.positionScale).
    const ps = r.positionScale;

    // --- Breathing (idle layer) ---
    const breath = Math.sin(this.t * 1.9) * (0.02 - energy * 0.012);
    j.chest.rotation.x += breath;
    j.chest.position.y += breath * 0.15 * ps;

    // --- Groove: bounce hips, flex knees to keep feet planted ---
    j.hips.position.y -= bounce * ps;
    const knee = bounce * 3.2;
    for (const s of ['left', 'right'] as const) {
      j[`${s}UpperLeg`].rotation.x -= knee * 0.9;
      j[`${s}LowerLeg`].rotation.x += knee * 1.8;
      j[`${s}Foot`].rotation.x -= knee * 0.9;
    }

    // --- Weight shift / sway ---
    j.hips.position.x += side * 0.05 * ps;
    j.hips.rotation.z -= side * 0.06;
    j.spine.rotation.z += side * 0.08;
    j.chest.rotation.z += side * 0.05;
    // Counter-rotate torso around Y for a loose, dancing feel.
    j.spine.rotation.y += Math.sin(this.groovePhase) * 0.1 * energy;

    // --- Head ---
    j.head.rotation.x += nod * 0.28 + Math.sin(this.t * 2.3) * 0.02;
    j.head.rotation.z -= side * 0.1;
    // Brightness tilts the head up slightly on bright/airy audio.
    j.neck.rotation.x -= f.brightness * 0.12;

    // --- Arms: pendulum swing scaled by energy, raised by sustained energy ---
    for (const [s, sign] of [['left', 1], ['right', -1]] as const) {
      const swing = Math.sin(this.groovePhase + (sign === 1 ? 0 : Math.PI));
      const upper = j[`${s}UpperArm`];
      const lower = j[`${s}LowerArm`];

      // Rest: arms hang down. Abduct outward with raise, swing forward/back
      // with the groove, bend elbows more as energy rises.
      upper.rotation.z += sign * (0.08 + raise * 1.5);
      upper.rotation.x += swing * (0.12 + energy * 0.45) - raise * 0.4;
      lower.rotation.x -= 0.15 + energy * 0.9 + Math.max(0, -swing) * 0.35 * energy;
      lower.rotation.x += nod * 0.25; // elbows pump a little on the beat

      j[`${s}Hand`].rotation.x -= energy * 0.3;
    }

    // Shoulders shrug slightly with treble (hi-hats, snares).
    j.leftShoulder.position.y += f.treble * 0.02 * ps;
    j.rightShoulder.position.y += f.treble * 0.02 * ps;
  }
}
