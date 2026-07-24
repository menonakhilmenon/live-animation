import * as THREE from 'three';
import { AudioFeatures } from '../audio/features';
import { HumanoidRig, resetToRest } from '../rig/humanoid';
import { ClipPlayer, MotionClip } from './clip';
import { FaceAnimator } from './face';
import { GestureSchedule, SchedulePlayer } from './schedule';
import { pinEffector, setWorldQuaternion } from './ik';
import { Spring } from './spring';

/**
 * Layered procedural animator. Each frame it resets the rig to its rest pose
 * and applies motion layers on top:
 *
 *  - breathing   — always on
 *  - speech      — conversational gestures + head nods on stressed onsets
 *  - idle        — micro weight shifts and gaze wander in silence
 *
 * Scheduled prebaked playback and direct motion-clip playback each override
 * the procedural layers while active (face/lip-sync and foot IK still run).
 */
export class Animator {
  private t = 0;

  private headNod = new Spring(0, 55);

  /** World-space rest pose of each foot — the ground-contact IK targets. */
  private footAnchors: { pos: THREE.Vector3; quat: THREE.Quaternion }[] | null = null;

  private tmpQuat = new THREE.Quaternion();

  /** Co-speech gesture state: one beat-gesture envelope per hand. */
  private gesture = { left: new Spring(0, 35), right: new Spring(0, 35) };
  private gestureTarget = { left: 0, right: 0 };
  private gestureCooldown = 0;
  private gestureSide: 'left' | 'right' = 'right';

  /** Finger curl per hand (smoothed toward per-behavior targets). */
  private curl = { left: new Spring(0.25, 25), right: new Spring(0.25, 25) };
  private curlTarget = { left: 0.25, right: 0.25 };

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

    const r = this.rig;
    resetToRest(r);
    const j = r.joints;

    this.ensureFootAnchors();
    // Positional offsets below are authored in meters; ps converts them to
    // the rig's local units (see HumanoidRig.positionScale).
    const ps = r.positionScale;

    // --- Breathing (always on) ---
    const breath = Math.sin(this.t * 1.9) * 0.02;
    j.chest.rotation.x += breath;
    j.chest.position.y += breath * 0.15 * ps;

    // --- Behavior dispatch ---
    // speech → conversational gestures; anything else → idle life.
    if (f.mode === 'speech') {
      this.speechLayers(f, dt);
      this.finishFrame(f, dt);
      return;
    }
    this.idleLayers();
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
