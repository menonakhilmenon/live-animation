import * as THREE from 'three';
import { HumanoidRig, JointName } from '../rig/humanoid';

/**
 * A generated (or authored) motion sequence in the canonical clip format:
 * for each frame and joint, a WORLD-space rotation delta from the canonical
 * T-pose (Y-up, character facing +Z, arms along ±X, meters).
 *
 * This representation is deliberately rig-agnostic: it says "the upper arm
 * is rotated 90° forward from where it points in a T-pose, measured in world
 * space", never "set local rotation.x". Playback converts it into whatever
 * local bone frames a skeleton uses via the rig's captured `tposeWorld`
 * orientations — the same probe-not-assume approach as the rest of the rig
 * layer. ML models emitting SMPL/BVH-style poses are converted to this
 * format on the generation side.
 */
export interface MotionClip {
  fps: number;
  /** Joints present in `rotations` frames, in the same per-frame order. */
  joints: JointName[];
  /** [frame][jointIndex] = [x, y, z, w] world-space delta from T-pose. */
  rotations: number[][][];
  /** Optional [frame] = [x, y, z] hips offset from rest, meters, world axes. */
  hipsPosition?: number[][];
  /** Keep ground-contact foot IK active during playback (default true —
   * right for upper-body gesture clips; full-body locomotion sets false). */
  pinFeet?: boolean;
  loop?: boolean;
  /** Emotional bias for the face while this clip plays (-1 sad … +1 happy). */
  mood?: number;
  /**
   * Optional articulated-hand tracks: VRM finger bone names (parent bones
   * before child bones within each digit) with per-frame world deltas from
   * T-pose, same convention as the body. Played on rigs exposing
   * `fingerRetarget`; other rigs keep their procedural curl.
   */
  fingers?: { joints: string[]; rotations: number[][][] };
}

/** Parent-first application order (ancestors before descendants). */
const APPLY_ORDER: JointName[] = [
  'hips', 'spine', 'chest', 'neck', 'head',
  'leftShoulder', 'leftUpperArm', 'leftLowerArm', 'leftHand',
  'rightShoulder', 'rightUpperArm', 'rightLowerArm', 'rightHand',
  'leftUpperLeg', 'leftLowerLeg', 'leftFoot',
  'rightUpperLeg', 'rightLowerLeg', 'rightFoot',
];

const tmpQa = new THREE.Quaternion();
const tmpQb = new THREE.Quaternion();
const tmpParent = new THREE.Quaternion();
const tmpVec = new THREE.Vector3();

/**
 * Plays MotionClips on a HumanoidRig. Call `apply` after resetToRest; the
 * procedural animator treats an active clip as the pose source and keeps
 * running face/lip-sync, fingers, and (optionally) foot IK on top.
 */
export class ClipPlayer {
  private clip: MotionClip | null = null;
  private time = 0;
  /** Index into APPLY_ORDER-sorted joints: [applyIdx, clipJointIdx][]. */
  private order: [JointName, number][] = [];

  play(clip: MotionClip): void {
    this.clip = clip;
    this.time = 0;
    this.order = APPLY_ORDER.filter((j) => clip.joints.includes(j)).map((j) => [
      j,
      clip.joints.indexOf(j),
    ]);
  }

  stop(): void {
    this.clip = null;
  }

  get active(): boolean {
    return this.clip !== null;
  }

  get duration(): number {
    return this.clip ? this.clip.rotations.length / this.clip.fps : 0;
  }

  get currentTime(): number {
    return this.time;
  }

  get mood(): number {
    return this.clip?.mood ?? 0;
  }

  /** Whether the active clip carries its own finger animation. */
  get hasFingers(): boolean {
    return !!this.clip?.fingers;
  }

  /**
   * Advance time and pose the rig (which must already be in rest pose).
   * Returns whether feet should stay IK-pinned this frame.
   */
  apply(rig: HumanoidRig, dt: number): boolean {
    const clip = this.clip;
    if (!clip) return true;
    this.time += dt;
    const frames = clip.rotations.length;
    let ft = this.time * clip.fps;
    if (clip.loop) {
      ft %= frames;
    } else if (ft >= frames - 1) {
      ft = frames - 1;
      if (this.time > frames / clip.fps + 0.05) this.clip = null; // finished
    }
    const f0 = Math.min(Math.floor(ft), frames - 1);
    const f1 = Math.min(f0 + 1, frames - 1);
    const mix = ft - f0;

    // Ease in from rest and back out near the end (non-looping clips) so
    // playback never pops between procedural pose and generated pose.
    const FADE = 0.4;
    let w = Math.min(1, this.time / FADE);
    if (!clip.loop) {
      const remain = frames / clip.fps - this.time;
      w = Math.min(w, Math.max(0, remain / FADE));
    }
    w = w * w * (3 - 2 * w);

    // Hips translation: world-space meter offsets → hips-parent local units.
    if (clip.hipsPosition) {
      const p0 = clip.hipsPosition[f0];
      const p1 = clip.hipsPosition[f1 < clip.hipsPosition.length ? f1 : f0];
      tmpVec.set(
        p0[0] + (p1[0] - p0[0]) * mix,
        p0[1] + (p1[1] - p0[1]) * mix,
        p0[2] + (p1[2] - p0[2]) * mix,
      );
      const hips = rig.joints.hips;
      if (hips.parent) {
        hips.parent.getWorldQuaternion(tmpParent);
        tmpVec.applyQuaternion(tmpParent.invert());
      }
      hips.position.addScaledVector(tmpVec, rig.positionScale * w);
    }

    // Rotations, parents before children so each getWorldQuaternion below
    // sees the already-posed ancestors.
    for (const [joint, idx] of this.order) {
      const r0 = clip.rotations[f0][idx];
      const r1 = clip.rotations[f1][idx];
      tmpQa.set(r0[0], r0[1], r0[2], r0[3]);
      tmpQb.set(r1[0], r1[1], r1[2], r1[3]);
      tmpQa.slerp(tmpQb, mix);

      // desiredWorld = delta * tposeWorld;  local = parentWorld⁻¹ * desiredWorld
      const node = rig.joints[joint];
      tmpQa.multiply(rig.tposeWorld[joint]);
      if (node.parent) {
        node.parent.getWorldQuaternion(tmpParent);
        tmpQa.premultiply(tmpParent.invert());
      }
      // Fade weight blends from the rest pose the caller just applied.
      if (w < 1) node.quaternion.slerp(tmpQa, w);
      else node.quaternion.copy(tmpQa);
      node.updateMatrixWorld(true);
    }

    // Articulated fingers, when both the clip and the rig support them.
    if (clip.fingers && rig.fingerRetarget) {
      for (const [i, name] of clip.fingers.joints.entries()) {
        const target = rig.fingerRetarget[name];
        if (!target) continue;
        const r0 = clip.fingers.rotations[f0][i];
        const r1 = clip.fingers.rotations[f1][i];
        tmpQa.set(r0[0], r0[1], r0[2], r0[3]);
        tmpQb.set(r1[0], r1[1], r1[2], r1[3]);
        tmpQa.slerp(tmpQb, mix);
        tmpQa.multiply(target.tposeWorld);
        if (target.node.parent) {
          target.node.parent.getWorldQuaternion(tmpParent);
          tmpQa.premultiply(tmpParent.invert());
        }
        if (w < 1) target.node.quaternion.slerp(tmpQa, w);
        else target.node.quaternion.copy(tmpQa);
        target.node.updateMatrixWorld(false);
      }
    }

    return clip.pinFeet !== false;
  }
}
