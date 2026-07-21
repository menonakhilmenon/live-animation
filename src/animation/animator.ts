import * as THREE from 'three';
import { AudioFeatures } from '../audio/features';
import { HumanoidRig, resetToRest } from '../rig/humanoid';
import { ClipPlayer, MotionClip } from './clip';
import { FaceAnimator } from './face';
import { GestureSchedule, SchedulePlayer } from './schedule';
import { pinEffector, setWorldQuaternion } from './ik';
import { Spring } from './spring';

function smoothstep(lo: number, hi: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - lo) / (hi - lo)));
  return t * t * (3 - 2 * t);
}

/** Wrap an angle difference into [-PI, PI). */
function wrapAngle(a: number): number {
  return ((a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
}

/**
 * Style parameters blended continuously from sustained energy:
 * calm (idle sway) → groove (dancing) → hype (headbang territory).
 */
const STYLES = {
  calm: { bounce: 0.45, nod: 0.1, sway: 1.15, arms: 0.35, elbows: 0.4 },
  groove: { bounce: 1.0, nod: 0.28, sway: 1.0, arms: 1.0, elbows: 1.0 },
  hype: { bounce: 1.35, nod: 0.55, sway: 0.8, arms: 1.45, elbows: 1.25 },
};
type StyleParams = (typeof STYLES)['calm'];

/**
 * Layered procedural animator. Each frame it resets the rig to its rest pose
 * and applies motion layers on top, each scaled by live audio features:
 *
 *  - breathing   — always on; grows subtler as energy rises
 *  - groove      — hips bounce + knee flex; phase-locked to the estimated
 *                  tempo when confidence is high (anticipates beats), else
 *                  reactive to the raw beat pulse
 *  - sway        — weight shift and torso lean, alternating per beat
 *  - head        — nod on beats, micro-bob with level, tilt with brightness
 *  - arms        — swing with energy, raise with sustained intensity
 *
 * Style (amplitude balance across layers) crossfades with sustained energy.
 */
export class Animator {
  private t = 0;
  /** Arm-swing phase; one full swing spans two beats when tempo-locked. */
  private groovePhase = 0;
  private prevBeatPhase = 0;

  private bounce = new Spring(0, 60);
  private energy = new Spring(0, 8);
  private headNod = new Spring(0, 55);
  private armRaise = new Spring(0, 10);
  private sideStep = new Spring(0, 12);
  private lock = new Spring(0, 6);
  /** Which side the weight is on: alternates on beats. */
  private weightSide = 1;

  /** World-space rest pose of each foot — the ground-contact IK targets. */
  private footAnchors: { pos: THREE.Vector3; quat: THREE.Quaternion }[] | null = null;

  private tmpQuat = new THREE.Quaternion();

  /** Co-speech gesture state: one beat-gesture envelope per hand. */
  private gesture = { left: new Spring(0, 35), right: new Spring(0, 35) };
  private gestureTarget = { left: 0, right: 0 };
  private gestureCooldown = 0;
  private gestureSide: 'left' | 'right' = 'right';

  /** Arm move rotation: switches on bar boundaries when locked. */
  private moveIndex = 0;
  private movePrev = 0;
  /** Slightly underdamped so move changes carry follow-through overshoot. */
  private moveBlend = new Spring(1, 28, 0.75);
  private lastPhrase = -1;

  /** Finger curl per hand (smoothed toward per-behavior targets). */
  private curl = { left: new Spring(0.25, 25), right: new Spring(0.25, 25) };
  private curlTarget = { left: 0.25, right: 0.25 };

  /** Drop anticipation state. */
  private windup = new Spring(0, 20);
  private prevDropIn = Infinity;
  private hitTimer = 0;

  /** Current arm move (exposed for tests/debug). */
  get currentMove(): number {
    return this.moveIndex;
  }

  readonly faceAnimator: FaceAnimator | null;

  /** Generated-motion playback; overrides procedural layers while active. */
  readonly clipPlayer = new ClipPlayer();

  /** Prebaked-library playback driven by a gesture schedule (preferred —
   * base loops + additive accents; raw poses never reach the skeleton). */
  readonly schedulePlayer = new SchedulePlayer();

  constructor(private rig: HumanoidRig) {
    this.faceAnimator = rig.face ? new FaceAnimator(rig.face) : null;
  }

  playClip(clip: MotionClip): void {
    this.clipPlayer.play(clip);
  }

  playSchedule(schedule: GestureSchedule, clock: () => number): boolean {
    const ok = this.schedulePlayer.play(schedule, clock);
    if (ok) this.clipPlayer.stop();
    return ok;
  }

  update(f: AudioFeatures, dt: number): void {
    dt = Math.min(dt, 1 / 20); // avoid spring blow-ups on tab-switch stalls
    this.t += dt;

    // Scheduled prebaked playback: base library pose + additive accents,
    // then a light procedural pass (breathing, onset nods) for life.
    if (this.schedulePlayer.active) {
      resetToRest(this.rig);
      this.ensureFootAnchors();
      if (this.faceAnimator) this.faceAnimator.moodBias = this.schedulePlayer.mood;
      const w = this.schedulePlayer.apply(this.rig, dt, f);
      const j = this.rig.joints;
      // Gaze leveling: source recordings often look below the camera —
      // pull head/neck world orientation partway toward the forward-facing
      // T-pose so the character keeps addressing the audience while the
      // gesture-correlated head motion survives at reduced amplitude.
      for (const [joint, k] of [['head', 0.5], ['neck', 0.3]] as const) {
        j[joint].getWorldQuaternion(this.tmpQuat);
        this.tmpQuat.slerp(this.rig.tposeWorld[joint], k);
        setWorldQuaternion(j[joint], this.tmpQuat);
      }
      const breath = Math.sin(this.t * 1.9) * 0.015;
      j.chest.rotation.x += breath;
      const nod = this.headNod.update(Math.min(1, f.onset * 0.6), dt);
      j.head.rotation.x += nod * 0.08 * w;
      j.neck.rotation.y += Math.sin(this.t * 0.23) * 0.03;
      this.finishFrame(f, dt, true);
      return;
    }

    // An active motion clip is the pose source; procedural layers stand
    // down but face/lip-sync, fingers, and (per-clip) foot IK still run.
    if (this.clipPlayer.active) {
      resetToRest(this.rig);
      this.ensureFootAnchors();
      if (this.faceAnimator) this.faceAnimator.moodBias = this.clipPlayer.mood;
      const pin = this.clipPlayer.apply(this.rig, dt);
      this.finishFrame(f, dt, pin);
      return;
    }
    if (this.faceAnimator) this.faceAnimator.moodBias = 0;

    // Sustained intensity (slow spring) gates most layers so the character
    // relaxes to idle in silence and commits to the groove on loud sections.
    // Song-section loudness scales the target: subdued verses, big choruses.
    const energy = this.energy.update(
      Math.min(1, (f.rms * 3 + f.beatPulse * 0.3) * (0.45 + 0.75 * f.section)),
      dt,
    );

    // Drop anticipation: crouch in during the last ~1.2 s before a known
    // drop, then release into a hit pose right as it lands.
    const windup = this.windup.update(
      f.nextDropIn < 1.2 ? smoothstep(0, 1.05, 1.2 - f.nextDropIn) : 0,
      dt,
    );
    if (this.prevDropIn < 0.2 && !(f.nextDropIn < 0.2)) {
      this.hitTimer = 1.2; // the drop just landed
      this.movePrev = this.moveIndex;
      this.moveIndex = 2; // raised groove hit
      this.moveBlend.value = 0;
      this.moveBlend.velocity = 0;
    }
    this.prevDropIn = f.nextDropIn;
    this.hitTimer = Math.max(0, this.hitTimer - dt);
    const hit = smoothstep(0, 0.25, this.hitTimer);

    // Style crossfade weights (calm + groove + hype = 1).
    const wHype = smoothstep(0.55, 0.8, energy);
    const wCalm = (1 - wHype) * (1 - smoothstep(0.2, 0.45, energy));
    const wGroove = 1 - wCalm - wHype;
    const S: StyleParams = {
      bounce: wCalm * STYLES.calm.bounce + wGroove * STYLES.groove.bounce + wHype * STYLES.hype.bounce,
      nod: wCalm * STYLES.calm.nod + wGroove * STYLES.groove.nod + wHype * STYLES.hype.nod,
      sway: wCalm * STYLES.calm.sway + wGroove * STYLES.groove.sway + wHype * STYLES.hype.sway,
      arms: wCalm * STYLES.calm.arms + wGroove * STYLES.groove.arms + wHype * STYLES.hype.arms,
      elbows: wCalm * STYLES.calm.elbows + wGroove * STYLES.groove.elbows + wHype * STYLES.hype.elbows,
    };

    // How much to trust the tempo tracker (springed so handoffs are smooth).
    const lock = this.lock.update(f.bpm > 0 && f.tempoConfidence > 0.6 ? 1 : 0, dt);

    // Arm-swing phase: free-runs with energy when unlocked; when locked it
    // follows the beat phase (one swing per two beats) via a soft correction.
    this.groovePhase += dt * ((1 - lock) * (0.8 + energy * 3.2) + lock * (f.bpm / 60) * Math.PI);
    if (lock > 0.01) {
      this.groovePhase -= wrapAngle(this.groovePhase - Math.PI * f.beatPhase) * 3 * lock * dt;
    }

    // Weight flips on beat boundaries when locked (phase wraps), else on
    // raw detector beats.
    const phaseWrapped = Math.floor(f.beatPhase) !== Math.floor(this.prevBeatPhase);
    this.prevBeatPhase = f.beatPhase;
    if (lock > 0.5 ? phaseWrapped : f.beat) this.weightSide = -this.weightSide;

    // Anticipatory pulse: peaks exactly on the beat (phase 0), rising just
    // before it — the dancer "knows" the beat is coming. Crossfaded with the
    // reactive envelope so low-confidence audio still moves.
    const frac = f.beatPhase - Math.floor(f.beatPhase);
    const phasedPulse = Math.pow(Math.max(0, Math.cos(2 * Math.PI * frac)), 3);
    const pulse = lock * phasedPulse * Math.min(1, energy * 2) + (1 - lock) * f.beatPulse;

    // Rotate to the next arm move each bar — with an offline timeline the
    // bar phase is downbeat-aligned, so changes land on real musical
    // boundaries. Calm idling stays on the pendulum swing.
    const phrase = Math.floor(f.barPhase);
    if (lock > 0.5 && phrase !== this.lastPhrase && this.hitTimer <= 0) {
      if (this.lastPhrase >= 0 && energy > 0.35) {
        this.movePrev = this.moveIndex;
        this.moveIndex = (this.moveIndex + 1) % 4;
        this.moveBlend.value = 0;
        this.moveBlend.velocity = 0;
      }
      this.lastPhrase = phrase;
    }
    // Underdamped: blend passes 1 and settles back — arm moves arrive with
    // a little momentum instead of easing in sterilely.
    const blend = Math.max(0, Math.min(1.15, this.moveBlend.update(1, dt)));

    const bounce = this.bounce.update(
      (pulse * 0.09 + f.bass * 0.03) * S.bounce * (1 + hit * 0.5) + windup * 0.055,
      dt,
    );
    const nod = this.headNod.update(pulse, dt);
    const raise = this.armRaise.update(Math.max(0, energy - 0.45) * 1.8, dt);
    const side = this.sideStep.update(this.weightSide * Math.min(1, energy * 1.4) * S.sway, dt);

    const r = this.rig;
    resetToRest(r);
    const j = r.joints;

    this.ensureFootAnchors();
    // Positional offsets below are authored in meters; ps converts them to
    // the rig's local units (see HumanoidRig.positionScale).
    const ps = r.positionScale;

    // --- Breathing (always on) ---
    const breath = Math.sin(this.t * 1.9) * (0.02 - energy * 0.012);
    j.chest.rotation.x += breath;
    j.chest.position.y += breath * 0.15 * ps;

    // --- Behavior dispatch ---
    // speech → conversational gestures; near-silence → idle life;
    // otherwise the dance path below.
    const behavior =
      f.mode === 'speech' ? 'speech' : f.rms < 0.02 && energy < 0.12 ? 'idle' : 'dance';
    if (behavior === 'speech') {
      this.speechLayers(f, dt);
      this.finishFrame(f, dt);
      return;
    }
    if (behavior === 'idle') {
      this.idleLayers();
      this.finishFrame(f, dt);
      return;
    }

    // --- Groove: bounce hips, flex knees to keep feet planted ---
    j.hips.position.y -= bounce * ps;
    // Pre-bend the knees in their natural direction; foot IK below does the
    // exact ground pinning, this seed just keeps CCD from picking a weird bend.
    const knee = bounce * 3.2;
    for (const s of ['left', 'right'] as const) {
      j[`${s}UpperLeg`].rotation.x -= knee * 0.9;
      j[`${s}LowerLeg`].rotation.x += knee * 1.8;
    }

    // --- Weight shift / sway ---
    j.hips.position.x += side * 0.05 * ps;
    j.hips.rotation.z -= side * 0.06;
    j.spine.rotation.z += side * 0.08;
    j.chest.rotation.z += side * 0.05;
    // Counter-rotate torso around Y for a loose, dancing feel.
    const torsoYaw = Math.sin(this.groovePhase) * 0.1 * energy;
    j.spine.rotation.y += torsoYaw;
    // Hip arcs: subtle forward/back drift at double time plus a hint of hip
    // yaw turns the vertical bounce + lateral sway into a figure-8 weight
    // path instead of a piston.
    j.hips.position.z += Math.sin(this.groovePhase * 2) * 0.008 * ps * energy;
    j.hips.rotation.y += Math.sin(this.groovePhase) * 0.05 * energy;
    // Gaze stabilization: the head counter-rotates most of the torso yaw so
    // the face stays on the audience — dancers stabilize their gaze.
    j.neck.rotation.y -= torsoYaw * 0.55;
    j.head.rotation.y -= torsoYaw * 0.3;

    // --- Head ---
    // Wind-up: head tucks slightly, then snaps up on the hit.
    j.head.rotation.x += windup * 0.22 - hit * 0.15;
    j.head.rotation.x += nod * S.nod + Math.sin(this.t * 2.3) * 0.02;
    j.head.rotation.z -= side * 0.1;
    // Brightness tilts the head up slightly on bright/airy audio.
    j.neck.rotation.x -= f.brightness * 0.12;

    // --- Arms: a small move repertoire, crossfaded on phrase boundaries ---
    // Each move returns [swing, abduct, flex] in canonical semantics
    // (positive = hand forward / arm out / elbow bend); the rig's probed
    // axes translate those into whatever local frames the skeleton uses.
    const armPose = (move: number, osc: number): [number, number, number] => {
      switch (move) {
        case 1: // beat pump: elbows bent, forearms punch with the pulse
          return [0.35 + pulse * 0.25, 0.3, 1.35 + pulse * 0.6];
        case 2: // raised groove: hands up, elbows bent, swaying with the phase
          return [0.25 + osc * 0.2, 2.0 + osc * 0.15, 0.8 + Math.max(0, osc) * 0.35];
        case 3: // side sway: arms hang, swinging laterally with the hips
          return [osc * 0.15, 0.18 + Math.sin(this.groovePhase) * 0.3, 0.25 + energy * 0.4];
        default: // pendulum swing (also the calm/unlocked fallback)
          return [
            osc * (0.12 + energy * 0.45) * S.arms + raise * 0.4,
            0.08 + raise * 1.5,
            0.15 + energy * 0.9 + Math.max(0, -osc) * 0.35 * energy,
          ];
      }
    };

    for (const s of ['left', 'right'] as const) {
      const osc = Math.sin(this.groovePhase + (s === 'left' ? 0 : Math.PI));
      const cur = armPose(this.moveIndex, osc);
      const prev = armPose(this.movePrev, osc);
      const swingA = prev[0] + (cur[0] - prev[0]) * blend - windup * 0.1;
      const abductA = prev[1] + (cur[1] - prev[1]) * blend - windup * 0.25;
      const flexA = (prev[2] + (cur[2] - prev[2]) * blend + nod * 0.25) * S.elbows + windup * 0.55;

      const axes = r.armAxes[s];
      j[`${s}UpperArm`].rotateOnAxis(axes.swing, swingA);
      j[`${s}UpperArm`].rotateOnAxis(axes.abduct, abductA);
      j[`${s}LowerArm`].rotateOnAxis(axes.flex, flexA);
      j[`${s}Hand`].rotateOnAxis(axes.flex, energy * 0.3);
      this.curlTarget[s] = 0.3 + pulse * 0.2 + energy * 0.1;
    }

    // Shoulders shrug slightly with treble (hi-hats, snares).
    j.leftShoulder.position.y += f.treble * 0.02 * ps;
    j.rightShoulder.position.y += f.treble * 0.02 * ps;

    this.finishFrame(f, dt);
  }

  /** Capture the feet's world rest transforms as ground-contact IK targets. */
  private ensureFootAnchors(): void {
    if (this.footAnchors) return;
    const j = this.rig.joints;
    this.rig.root.updateWorldMatrix(true, true);
    this.footAnchors = (['left', 'right'] as const).map((s) => {
      const pos = new THREE.Vector3();
      const quat = new THREE.Quaternion();
      j[`${s}Foot`].getWorldPosition(pos);
      j[`${s}Foot`].getWorldQuaternion(quat);
      return { pos, quat };
    });
  }

  /** Foot IK + fingers + face + model runtime update — all behaviors. */
  private finishFrame(f: AudioFeatures, dt: number, pinFeet = true): void {
    const j = this.rig.joints;
    if (pinFeet) {
      for (const [i, s] of (['left', 'right'] as const).entries()) {
        const anchor = this.footAnchors![i];
        pinEffector([j[`${s}LowerLeg`], j[`${s}UpperLeg`]], j[`${s}Foot`], anchor.pos);
        setWorldQuaternion(j[`${s}Foot`], anchor.quat);
      }
    }
    // Procedural curl stands down when an active clip animates the fingers.
    const clipOwnsFingers =
      (this.clipPlayer.active && this.clipPlayer.hasFingers) || this.schedulePlayer.active;
    if (this.rig.fingers && !clipOwnsFingers) {
      for (const s of ['left', 'right'] as const) {
        const c = this.curl[s].update(this.curlTarget[s], dt);
        for (const fj of this.rig.fingers[s]) {
          fj.node.quaternion.copy(fj.rest);
          fj.node.rotateOnAxis(fj.curlAxis, c);
        }
      }
    }
    this.faceAnimator?.update(f, dt);
    this.rig.tick?.(dt);
  }

  /**
   * Conversational behavior: still stance, head nods on stressed syllables,
   * alternating-hand beat gestures keyed on onset strength — the timing
   * backbone of co-speech gesture, hands accenting the prosodic stresses.
   */
  private speechLayers(f: AudioFeatures, dt: number): void {
    const j = this.rig.joints;
    const r = this.rig;

    // Gesture triggering: onset above threshold, per-hand envelope with a
    // short cooldown; strong onsets recruit both hands.
    this.gestureCooldown -= dt;
    this.gestureTarget.left *= Math.exp(-dt / 0.3);
    this.gestureTarget.right *= Math.exp(-dt / 0.3);
    if (f.onset > 0.55 && this.gestureCooldown <= 0) {
      this.gestureSide = this.gestureSide === 'left' ? 'right' : 'left';
      this.gestureTarget[this.gestureSide] = Math.min(1, f.onset * 0.8);
      if (f.onset > 1.1) this.gestureTarget[this.gestureSide === 'left' ? 'right' : 'left'] = 0.6;
      this.gestureCooldown = 0.32;
    }

    // Head: nod with stress, slow attentive wander.
    const nod = this.headNod.update(Math.min(1, f.onset * 0.7), dt);
    j.head.rotation.x += nod * 0.12 + Math.sin(this.t * 1.1) * 0.015;
    j.head.rotation.y += Math.sin(this.t * 0.31) * 0.06;
    j.neck.rotation.y += Math.sin(this.t * 0.17 + 1.2) * 0.04;

    // Slow, small weight shift — standing, not dancing.
    const side = Math.sin(this.t * 0.27) * 0.35;
    j.hips.rotation.z -= side * 0.02;
    j.spine.rotation.z += side * 0.03;
    j.hips.position.x += side * 0.012 * r.positionScale;

    // Arms: relaxed base pose plus per-hand gesture envelopes.
    for (const s of ['left', 'right'] as const) {
      const g = this.gesture[s].update(this.gestureTarget[s], dt);
      const axes = r.armAxes[s];
      j[`${s}UpperArm`].rotateOnAxis(axes.swing, 0.06 + g * 0.35);
      j[`${s}UpperArm`].rotateOnAxis(axes.abduct, 0.06 + g * 0.22);
      j[`${s}LowerArm`].rotateOnAxis(axes.flex, 0.4 + g * 0.85);
      j[`${s}Hand`].rotateOnAxis(axes.flex, -g * 0.5); // hand opens on the beat
      // Chest leans a touch toward the gesturing hand; fingers open with it.
      j.chest.rotation.z += (s === 'left' ? -1 : 1) * g * 0.02;
      this.curlTarget[s] = Math.max(0.05, 0.28 - g * 0.25);
    }
  }

  /**
   * Idle life in silence: micro weight shifts and gaze wander built from
   * incommensurate sines, so the pattern never visibly repeats.
   */
  private idleLayers(): void {
    const j = this.rig.joints;
    const r = this.rig;
    const t = this.t;

    const side = (Math.sin(t * 0.11) + 0.6 * Math.sin(t * 0.047 + 1.7)) * 0.5;
    j.hips.position.x += side * 0.02 * r.positionScale;
    j.hips.rotation.z -= side * 0.03;
    j.spine.rotation.z += side * 0.04;

    const yaw = 0.3 * (Math.sin(t * 0.13) + 0.5 * Math.sin(t * 0.041 + 2.1));
    const pitch = 0.06 * Math.sin(t * 0.09 + 0.5);
    j.neck.rotation.y += yaw * 0.5;
    j.head.rotation.y += yaw * 0.4;
    j.head.rotation.x += pitch;

    for (const s of ['left', 'right'] as const) {
      const axes = r.armAxes[s];
      j[`${s}LowerArm`].rotateOnAxis(axes.flex, 0.18 + Math.sin(t * 0.19 + (s === 'left' ? 0 : 2)) * 0.03);
      this.curlTarget[s] = 0.2;
    }
  }
}
