import * as THREE from 'three';
import { HumanoidRig, JointName } from '../rig/humanoid';
import { MotionClip } from './clip';

/**
 * A gesture schedule: the generative side DECIDES (which prebaked clip
 * plays when, which accents punctuate it) and this player EXECUTES with
 * guaranteed-clean motion — base loops crossfade into each other and
 * accents are applied additively on top, so there is no frame where raw
 * model output reaches the skeleton directly.
 */
export interface GestureSchedule {
  /** Base-layer segments, time-sorted, non-overlapping. Clip names resolve
   * via the AnimationLibrary; clips loop for the segment duration. */
  base: { name: string; t0: number; t1: number }[];
  /** One-shot ADDITIVE accents (e.g. a nod): the clip's world-delta
   * relative to its own first frame rides on top of the base pose. */
  accents: { name: string; t: number; scale?: number }[];
  mood?: number;
}

/** Crossfade length between base segments and accent fade, seconds. */
const FADE = 0.45;

/** Accents are HEAD gestures: even when the accent clip animates the whole
 * body (Xbot's agree/headShake are full-body nod-bows), only the head
 * chain receives it, tapering down the spine — otherwise a nod bows the
 * torso over the base pose and reads as a glitch. */
const ACCENT_WEIGHT: Partial<Record<JointName, number>> = {
  head: 1,
  neck: 0.9,
  chest: 0.25,
  spine: 0.1,
};
/** Longest slice of an accent clip to play, seconds. */
const ACCENT_MAX_S = 1.6;

const qA = new THREE.Quaternion();
const qB = new THREE.Quaternion();
const qRef = new THREE.Quaternion();
const qParent = new THREE.Quaternion();
const vTmp = new THREE.Vector3();

function smoothstep(x: number): number {
  const t = Math.min(1, Math.max(0, x));
  return t * t * (3 - 2 * t);
}

/** Sample a clip's stored world-delta for one joint index at local time. */
function sampleDelta(clip: MotionClip, jointIdx: number, time: number, out: THREE.Quaternion): void {
  const frames = clip.rotations.length;
  const ft = ((time * clip.fps) % frames + frames) % frames;
  const f0 = Math.floor(ft) % frames;
  const f1 = (f0 + 1) % frames;
  const r0 = clip.rotations[f0][jointIdx];
  const r1 = clip.rotations[f1][jointIdx];
  out.set(r0[0], r0[1], r0[2], r0[3]);
  qB.set(r1[0], r1[1], r1[2], r1[3]);
  out.slerp(qB, ft - f0);
}

function sampleHips(clip: MotionClip, time: number, out: THREE.Vector3): void {
  if (!clip.hipsPosition) {
    out.set(0, 0, 0);
    return;
  }
  const frames = clip.hipsPosition.length;
  const ft = ((time * clip.fps) % frames + frames) % frames;
  const f0 = Math.floor(ft) % frames;
  const f1 = (f0 + 1) % frames;
  const p0 = clip.hipsPosition[f0];
  const p1 = clip.hipsPosition[f1];
  const k = ft - f0;
  out.set(
    p0[0] + (p1[0] - p0[0]) * k,
    p0[1] + (p1[1] - p0[1]) * k,
    p0[2] + (p1[2] - p0[2]) * k,
  );
}

export class SchedulePlayer {
  private schedule: GestureSchedule | null = null;
  private clock: (() => number) | null = null;
  private library: Record<string, MotionClip> = {};
  /** Global blend-in weight so scheduled playback eases in from rest. */
  private startedAt = -1;

  setLibrary(clips: Record<string, MotionClip>): void {
    this.library = { ...this.library, ...clips };
  }

  get libraryNames(): string[] {
    return Object.keys(this.library);
  }

  /** Returns false when no base clip is available (caller may fall back). */
  play(schedule: GestureSchedule, clock: () => number): boolean {
    // Drop schedule entries whose clips we don't have — a schedule from a
    // newer server must degrade gracefully.
    this.schedule = {
      ...schedule,
      base: schedule.base.filter((s) => this.library[s.name]),
      accents: schedule.accents.filter((a) => this.library[a.name]),
    };
    this.clock = clock;
    this.startedAt = -1;
    if (!this.schedule.base.length) {
      this.schedule = null;
      return false;
    }
    return true;
  }

  stop(): void {
    this.schedule = null;
    this.clock = null;
  }

  get active(): boolean {
    if (!this.schedule || !this.clock) return false;
    const t = this.clock();
    const lastBase = this.schedule.base[this.schedule.base.length - 1];
    return lastBase ? t <= lastBase.t1 + FADE : false;
  }

  get mood(): number {
    return this.schedule?.mood ?? 0;
  }

  /** Pose the rig (already reset to rest). Returns overall blend weight. */
  apply(rig: HumanoidRig, dtWall: number): number {
    const sched = this.schedule!;
    const t = this.clock!();
    if (this.startedAt < 0) this.startedAt = t;
    const wIn = smoothstep((t - this.startedAt) / FADE);
    const lastBase = sched.base[sched.base.length - 1];
    const wOut = lastBase ? smoothstep((lastBase.t1 + FADE - t) / FADE) : 1;
    const w = Math.min(wIn, wOut);
    if (w <= 0) return 0;

    // --- Base layer: active segment, crossfaded from the previous one ---
    let cur = -1;
    for (let i = 0; i < sched.base.length; i++) {
      if (t >= sched.base[i].t0 - 1e-6 && (i === sched.base.length - 1 || t < sched.base[i + 1].t0)) {
        cur = i;
      }
    }
    if (cur < 0) return 0;
    const seg = sched.base[cur];
    const clip = this.library[seg.name];
    const prevSeg = cur > 0 ? sched.base[cur - 1] : null;
    const xfade = prevSeg ? smoothstep((t - seg.t0) / FADE) : 1;
    const prevClip = prevSeg ? this.library[prevSeg.name] : null;

    const applyJointSet = (
      names: string[],
      resolve: (name: string) => { node: THREE.Object3D; tpose: THREE.Quaternion } | null,
      idxOf: (clipOfSeg: MotionClip, name: string) => number,
      clipOf: (c: MotionClip) => MotionClip | null,
    ) => {
      for (const name of names) {
        const target = resolve(name);
        if (!target) continue;
        const c = clipOf(clip);
        if (!c) continue;
        const i = idxOf(c, name);
        if (i < 0) continue;
        sampleDelta(c, i, t - seg.t0, qA);
        if (prevClip && xfade < 1) {
          const pc = clipOf(prevClip);
          const pi = pc ? idxOf(pc, name) : -1;
          if (pc && pi >= 0) {
            sampleDelta(pc, pi, t - prevSeg!.t0, qB);
            qB.slerp(qA, xfade);
            qA.copy(qB);
          }
        }
        qA.multiply(target.tpose);
        if (target.node.parent) {
          target.node.parent.getWorldQuaternion(qParent);
          qA.premultiply(qParent.invert());
        }
        if (w < 1) target.node.quaternion.slerp(qA, w);
        else target.node.quaternion.copy(qA);
        target.node.updateMatrixWorld(true);
      }
    };

    applyJointSet(
      clip.joints,
      (n) => ({ node: rig.joints[n as JointName], tpose: rig.tposeWorld[n as JointName] }),
      (c, n) => c.joints.indexOf(n as JointName),
      (c) => c,
    );
    if (rig.fingerRetarget) {
      applyJointSet(
        clip.fingers?.joints ?? [],
        (n) => {
          const f = rig.fingerRetarget![n];
          return f ? { node: f.node, tpose: f.tposeWorld } : null;
        },
        (c, n) => (c.fingers ? c.fingers.joints.indexOf(n) : -1),
        (c) => (c.fingers ? ({ ...c, joints: c.fingers.joints as JointName[], rotations: c.fingers.rotations } as MotionClip) : null),
      );
    }

    // Base hips bob (world-space meters, weight-scaled).
    sampleHips(clip, t - seg.t0, vTmp);
    if (prevClip && xfade < 1) {
      const prev = new THREE.Vector3();
      sampleHips(prevClip, t - prevSeg!.t0, prev);
      vTmp.lerp(prev, 1 - xfade);
    }
    const hips = rig.joints.hips;
    if (hips.parent) {
      hips.parent.getWorldQuaternion(qParent);
      vTmp.applyQuaternion(qParent.invert());
    }
    hips.position.addScaledVector(vTmp, rig.positionScale * w);

    // --- Accents: additive head-gesture one-shots on top of the base ---
    for (const accent of sched.accents) {
      const aclip = this.library[accent.name];
      const dur = Math.min(aclip.rotations.length / aclip.fps, ACCENT_MAX_S);
      const at = t - accent.t;
      if (at < 0 || at > dur) continue;
      const env =
        smoothstep(at / 0.25) * smoothstep((dur - at) / 0.35) *
        Math.min(0.7, accent.scale ?? 0.7) * w;
      if (env <= 0) continue;
      for (const [i, name] of aclip.joints.entries()) {
        const jointWeight = ACCENT_WEIGHT[name as JointName];
        if (!jointWeight) continue;
        const node = rig.joints[name as JointName];
        if (!node) continue;
        sampleDelta(aclip, i, Math.min(at, dur - 1e-3), qA);
        const r0 = aclip.rotations[0][i];
        qRef.set(r0[0], r0[1], r0[2], r0[3]).invert();
        qA.multiply(qRef); // delta relative to the accent's first frame
        qParent.identity().slerp(qA, env * jointWeight); // scaled additive
        node.getWorldQuaternion(qB);
        qB.premultiply(qParent); // additive in world space
        if (node.parent) {
          node.parent.getWorldQuaternion(qParent);
          qB.premultiply(qParent.invert());
        }
        node.quaternion.copy(qB);
        node.updateMatrixWorld(true);
      }
    }
    void dtWall;
    return w;
  }
}
