import * as THREE from 'three';
import { HumanoidRig, JointName } from '../rig/humanoid';
import { MotionClip } from './clip';

/**
 * Prebaked-animation library. Two sources:
 *  - JSON clips in the canonical MotionClip format (baked offline by
 *    ml/bake_library.py — smoothed, loopable EMAGE generations)
 *  - THREE.AnimationClips embedded in a loaded GLB (e.g. Xbot's
 *    idle/agree/headShake), converted here by sampling the source skeleton
 *    and reading world-space deltas against its captured T-pose — the same
 *    canonical convention, so any rig can play them.
 */

const SAMPLE_FPS = 30;

/**
 * Bone-name map for skeletons extracted from FFXV (Luminous engine, e.g.
 * via Noesis): rest pose is a meter-scale T-pose facing +Z with left=+X —
 * the canonical frame — so world deltas convert with no basis change.
 */
export const FFXV_BONES: Record<string, JointName> = {
  C_Hip: 'hips', C_Spine1: 'spine', C_Spine3: 'chest', C_Neck1: 'neck', C_Head: 'head',
  L_Shoulder: 'leftShoulder', L_UpperArm: 'leftUpperArm', L_Elbow: 'leftLowerArm', L_Hand: 'leftHand',
  R_Shoulder: 'rightShoulder', R_UpperArm: 'rightUpperArm', R_Elbow: 'rightLowerArm', R_Hand: 'rightHand',
  L_UpperLeg: 'leftUpperLeg', L_Knee: 'leftLowerLeg', L_Foot: 'leftFoot',
  R_UpperLeg: 'rightUpperLeg', R_Knee: 'rightLowerLeg', R_Foot: 'rightFoot',
};

/**
 * Convert animations from ANY named skeleton (e.g. a Noesis-exported FBX)
 * into canonical MotionClips. The skeleton's CURRENT pose is taken as its
 * T-pose reference — verify it rests in a T-pose before calling.
 */
export function convertNamedSkeletonAnimations(
  root: THREE.Object3D,
  animations: THREE.AnimationClip[],
  nameMap: Record<string, JointName>,
): Record<string, MotionClip> {
  const joints = {} as Partial<Record<JointName, THREE.Object3D>>;
  root.traverse((node) => {
    const mapped = nameMap[node.name];
    if (mapped && !joints[mapped]) joints[mapped] = node;
  });
  root.updateMatrixWorld(true);
  const jointNames = Object.keys(joints) as JointName[];
  const tpose = {} as Record<JointName, THREE.Quaternion>;
  for (const j of jointNames) tpose[j] = joints[j]!.getWorldQuaternion(new THREE.Quaternion());
  const hips0 = joints.hips ? joints.hips.getWorldPosition(new THREE.Vector3()) : new THREE.Vector3();

  const mixer = new THREE.AnimationMixer(root);
  const qTmp = new THREE.Quaternion();
  const vTmp = new THREE.Vector3();
  const out: Record<string, MotionClip> = {};

  for (const anim of animations) {
    const action = mixer.clipAction(anim);
    action.play();
    const frames = Math.max(2, Math.round(anim.duration * SAMPLE_FPS));
    const rotations: number[][][] = [];
    const hipsPosition: number[][] = [];
    for (let f = 0; f < frames; f++) {
      mixer.setTime(f / SAMPLE_FPS);
      root.updateMatrixWorld(true);
      rotations.push(
        jointNames.map((j) => {
          joints[j]!.getWorldQuaternion(qTmp);
          qTmp.multiply(tpose[j].clone().invert());
          return [+qTmp.x.toFixed(5), +qTmp.y.toFixed(5), +qTmp.z.toFixed(5), +qTmp.w.toFixed(5)];
        }),
      );
      joints.hips!.getWorldPosition(vTmp);
      hipsPosition.push([
        +(vTmp.x - hips0.x).toFixed(5),
        +(vTmp.y - hips0.y).toFixed(5),
        +(vTmp.z - hips0.z).toFixed(5),
      ]);
    }
    action.stop();
    mixer.uncacheClip(anim);
    out[anim.name] = { fps: SAMPLE_FPS, joints: jointNames, rotations, hipsPosition };
  }
  return out;
}

export async function loadClipJSON(url: string): Promise<MotionClip> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`loadClipJSON(${url}): ${res.status}`);
  return (await res.json()) as MotionClip;
}

/**
 * Convert embedded GLTF animations into canonical MotionClips using the
 * (freshly loaded) source rig. Restores the rig's pose afterwards.
 */
export function convertGLTFAnimations(
  rig: HumanoidRig,
  animations: THREE.AnimationClip[],
): Record<string, MotionClip> {
  const out: Record<string, MotionClip> = {};
  if (!animations.length) return out;

  // Snapshot EVERY descendant's local transform — the mixer animates bones
  // outside our joint map too (Spine1, fingers, toes...).
  const snapshot: { node: THREE.Object3D; pos: THREE.Vector3; quat: THREE.Quaternion }[] = [];
  rig.root.traverse((node) => {
    snapshot.push({ node, pos: node.position.clone(), quat: node.quaternion.clone() });
  });

  const mixer = new THREE.AnimationMixer(rig.root);
  const jointNames = Object.keys(rig.joints) as JointName[];
  const fingerNames = Object.keys(rig.fingerRetarget ?? {});
  const qTmp = new THREE.Quaternion();
  const vTmp = new THREE.Vector3();

  for (const anim of animations) {
    const action = mixer.clipAction(anim);
    action.play();
    const frames = Math.max(2, Math.round(anim.duration * SAMPLE_FPS));
    const rotations: number[][][] = [];
    const fingerRots: number[][][] = [];
    const hipsPosition: number[][] = [];
    let hips0: THREE.Vector3 | null = null;

    for (let f = 0; f < frames; f++) {
      mixer.setTime(f / SAMPLE_FPS);
      rig.root.updateMatrixWorld(true);
      rotations.push(
        jointNames.map((j) => {
          rig.joints[j].getWorldQuaternion(qTmp);
          qTmp.multiply(rig.tposeWorld[j].clone().invert());
          return [+qTmp.x.toFixed(5), +qTmp.y.toFixed(5), +qTmp.z.toFixed(5), +qTmp.w.toFixed(5)];
        }),
      );
      if (fingerNames.length) {
        fingerRots.push(
          fingerNames.map((n) => {
            const fr = rig.fingerRetarget![n];
            fr.node.getWorldQuaternion(qTmp);
            qTmp.multiply(fr.tposeWorld.clone().invert());
            return [+qTmp.x.toFixed(5), +qTmp.y.toFixed(5), +qTmp.z.toFixed(5), +qTmp.w.toFixed(5)];
          }),
        );
      }
      rig.joints.hips.getWorldPosition(vTmp);
      if (!hips0) hips0 = vTmp.clone();
      hipsPosition.push([
        +(vTmp.x - hips0.x).toFixed(5),
        +(vTmp.y - hips0.y).toFixed(5),
        +(vTmp.z - hips0.z).toFixed(5),
      ]);
    }
    action.stop();
    mixer.uncacheClip(anim);

    const clip: MotionClip = { fps: SAMPLE_FPS, joints: jointNames, rotations, hipsPosition };
    if (fingerRots.length) clip.fingers = { joints: fingerNames, rotations: fingerRots };
    out[anim.name] = clip;
  }

  for (const s of snapshot) {
    s.node.position.copy(s.pos);
    s.node.quaternion.copy(s.quat);
  }
  rig.root.updateMatrixWorld(true);
  return out;
}
