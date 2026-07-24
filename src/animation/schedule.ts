import * as THREE from 'three';
import { AudioFeatures } from '../audio/features';
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
  /**
   * Continuous ADDITIVE overlay layers, composited on top of the base for
   * the whole utterance. Each layer's per-frame rotation relative to its
   * own first frame is added (world-space) to the base pose, scaled by
   * `weight` and speech energy. This is the base+additive composition that
   * unlocks game conversational libraries: FFXV/FFXVI store talk motion as
   * additive-over-base layers, and this plays them on any base stance. */
  additive?: { name: string; weight?: number; loop?: boolean }[];
  mood?: number;
  /**
   * Bias the additive layer's per-arm strength by the base clip's own
   * handedness, so the additive rides the arm the base is already gesturing
   * with and lets the other rest. Set for game-faithful playback: real game
   * dialogue gestures with one hand (FFXVI's Clive: left ~25°/s, right ~6),
   * and a symmetric co-speech additive would otherwise wake the resting arm.
   */
  matchHandedness?: boolean;
  /**
   * Gesture-amplitude multiplier for the expressive layers (additive +
   * accents). 1 = as authored; >1 pushes toward an exaggerated cinematic
   * "game-feel" (bigger sweeps, snappier nods) by scaling each gesture's
   * rotation angle. Posture (the base) is left alone so it never distorts.
   */
  exaggeration?: number;
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

/**
 * Speech-energy gesture gain: how strongly each joint's motion is scaled
 * by voice activity. Arms follow the voice (people gesture on stressed
 * syllables and rest between phrases); torso/head keep more of their
 * baseline sway so the character never freezes.
 */
const GAIN_WEIGHT: Partial<Record<JointName, number>> = {
  leftShoulder: 1, leftUpperArm: 1, leftLowerArm: 1, leftHand: 1,
  rightShoulder: 1, rightUpperArm: 1, rightLowerArm: 1, rightHand: 1,
  spine: 0.45, chest: 0.45, neck: 0.3, head: 0.3, hips: 0.45,
};
const GAIN_MIN = 0.25;

/** Arm joints per side, for handedness-biased additive (see matchHandedness). */
const LEFT_ARM = new Set<string>(['leftShoulder', 'leftUpperArm', 'leftLowerArm', 'leftHand']);
const RIGHT_ARM = new Set<string>(['rightShoulder', 'rightUpperArm', 'rightLowerArm', 'rightHand']);

const qA = new THREE.Quaternion();
const qB = new THREE.Quaternion();
const qRef = new THREE.Quaternion();
const qParent = new THREE.Quaternion();
const vTmp = new THREE.Vector3();

function smoothstep(x: number): number {
  const t = Math.min(1, Math.max(0, x));
  return t * t * (3 - 2 * t);
}

/**
 * Scale a unit quaternion's rotation ANGLE about its own axis by `f`, in
 * place, keeping the shortest arc. Used to exaggerate gesture amplitude
 * (game-feel). The resulting angle is capped just under 180° so a large
 * gesture can't wrap or flip.
 */
function scaleAngle(q: THREE.Quaternion, f: number): void {
  const w = Math.min(1, Math.max(-1, q.w));
  const vlen = Math.hypot(q.x, q.y, q.z);
  const half = Math.acos(Math.abs(w)); // half-angle, shortest arc
  if (half < 1e-5 || vlen < 1e-9) return; // ~identity: nothing to scale
  const sign = w < 0 ? -1 : 1; // fold to w>=0 so axis matches |w|
  const nh = Math.min(half * f, Math.PI * 0.49); // cap → angle < ~176°
  const s = Math.sin(nh) / vlen;
  q.set(sign * q.x * s, sign * q.y * s, sign * q.z * s, Math.cos(nh));
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
  /** Smoothed speech-activity gain (fast attack, slow release). */
  private gain = GAIN_MIN;
  /** Per-base-clip {left,right} arm additive multipliers (see matchHandedness). */
  private handedness = new Map<string, { left: number; right: number }>();

  setLibrary(clips: Record<string, MotionClip>): void {
    this.library = { ...this.library, ...clips };
  }

  /**
   * How much of the additive each arm should receive, from the base clip's
   * own per-arm motion. The busier arm gets 1.0; the quieter arm is squashed
   * (ratio², floored) so a symmetric co-speech overlay doesn't wake a hand
   * the base is deliberately resting. Cached per clip name.
   */
  private armHandedness(name: string, clip: MotionClip): { left: number; right: number } {
    const cached = this.handedness.get(name);
    if (cached) return cached;
    const speed = (joints: string[]): number => {
      let sum = 0;
      let n = 0;
      for (const jn of joints) {
        const idx = clip.joints.indexOf(jn as JointName);
        if (idx < 0) continue;
        for (let f = 1; f < clip.rotations.length; f++) {
          const a = clip.rotations[f - 1][idx];
          const b = clip.rotations[f][idx];
          const d = Math.min(1, Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]));
          sum += 2 * Math.acos(d);
          n++;
        }
      }
      return n ? sum / n : 0;
    };
    const l = speed(['leftHand', 'leftLowerArm']);
    const r = speed(['rightHand', 'rightLowerArm']);
    const m = Math.max(l, r, 1e-6);
    const shape = (x: number): number => Math.max(0.12, (x / m) ** 2);
    const res = { left: shape(l), right: shape(r) };
    this.handedness.set(name, res);
    return res;
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
      additive: (schedule.additive ?? []).filter((a) => this.library[a.name]),
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
  apply(rig: HumanoidRig, dtWall: number, features?: AudioFeatures): number {
    const sched = this.schedule!;
    const t = this.clock!();

    // Voice-activity gain: gestures bloom on speech energy and settle to a
    // calm stance in the gaps — a short greeting no longer inherits a full
    // monologue's arm choreography.
    const target = features
      ? Math.min(1, GAIN_MIN + features.rms * 4.2 + features.onset * 0.25)
      : 1;
    const tau = target > this.gain ? 0.09 : 0.5;
    this.gain += (target - this.gain) * Math.min(1, dtWall / tau);
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
      gainOf: (name: string) => number,
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
        // Voice-activity gain: pull toward the clip's calm frame 0.
        const gw = gainOf(name);
        if (gw > 0 && this.gain < 1) {
          const g = 1 - gw * (1 - this.gain);
          const r0 = c.rotations[0][i];
          qRef.set(r0[0], r0[1], r0[2], r0[3]);
          qRef.slerp(qA, g);
          qA.copy(qRef);
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
      (n) => GAIN_WEIGHT[n as JointName] ?? 0,
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
        () => 1, // fingers ride with the hands
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
    hips.position.addScaledVector(
      vTmp,
      rig.positionScale * w * (1 - (GAIN_WEIGHT.hips ?? 0.45) * (1 - this.gain)),
    );

    // --- Additive overlays: continuous expressive layers on the base ---
    // Each overlay's rotation relative to its own frame 0 is added in world
    // space to whatever the base already posed, scaled by weight × energy
    // gain. This is the base+additive composition: a subtle full-body talk
    // layer (ours or a game's additive clip) rides any base stance.
    // Handedness bias from the CURRENT base segment's clip (game-faithful).
    const hand = sched.matchHandedness ? this.armHandedness(seg.name, clip) : null;
    const exaggeration = sched.exaggeration ?? 1;
    for (const layer of sched.additive ?? []) {
      const lclip = this.library[layer.name];
      if (!lclip) continue;
      const ldur = lclip.rotations.length / lclip.fps;
      const lt = layer.loop === false ? Math.min(t, ldur - 1e-3) : t % ldur;
      const baseWeight = (layer.weight ?? 1) * w;
      for (const [i, name] of lclip.joints.entries()) {
        const gw = GAIN_WEIGHT[name as JointName] ?? 0.4;
        let env = baseWeight * (1 - gw * (1 - this.gain));
        if (hand) {
          // Ride the arm the base gestures with; let the resting arm rest.
          if (LEFT_ARM.has(name)) env *= hand.left;
          else if (RIGHT_ARM.has(name)) env *= hand.right;
        }
        if (env <= 0.01) continue;
        const node = rig.joints[name as JointName];
        if (!node) continue;
        sampleDelta(lclip, i, lt, qA);
        const r0 = lclip.rotations[0][i];
        qRef.set(r0[0], r0[1], r0[2], r0[3]).invert();
        qA.multiply(qRef); // delta from the layer's first frame
        if (exaggeration !== 1) scaleAngle(qA, exaggeration);
        qParent.identity().slerp(qA, env);
        node.getWorldQuaternion(qB);
        qB.premultiply(qParent); // additive, world space
        if (node.parent) {
          node.parent.getWorldQuaternion(qParent);
          qB.premultiply(qParent.invert());
        }
        node.quaternion.copy(qB);
        node.updateMatrixWorld(true);
      }
    }

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
        if (exaggeration !== 1) scaleAngle(qA, exaggeration);
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
