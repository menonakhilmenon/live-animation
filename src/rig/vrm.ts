import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { VRM, VRMHumanBoneName, VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { collectFingers, finalizeRig } from './loader';
import { FaceDriver, HumanoidRig, JointName } from './humanoid';

/** three-vrm normalized humanoid bone name for each of our joints. */
const VRM_BONES: Record<JointName, VRMHumanBoneName> = {
  hips: 'hips',
  spine: 'spine',
  chest: 'chest',
  neck: 'neck',
  head: 'head',
  leftShoulder: 'leftShoulder',
  leftUpperArm: 'leftUpperArm',
  leftLowerArm: 'leftLowerArm',
  leftHand: 'leftHand',
  rightShoulder: 'rightShoulder',
  rightUpperArm: 'rightUpperArm',
  rightLowerArm: 'rightLowerArm',
  rightHand: 'rightHand',
  leftUpperLeg: 'leftUpperLeg',
  leftLowerLeg: 'leftLowerLeg',
  leftFoot: 'leftFoot',
  rightUpperLeg: 'rightUpperLeg',
  rightLowerLeg: 'rightLowerLeg',
  rightFoot: 'rightFoot',
};

class VRMFace implements FaceDriver {
  private values: Record<string, number> = {};

  constructor(private vrm: VRM) {}

  private set(name: string, v: number): void {
    this.values[name] = v;
    this.vrm.expressionManager?.setValue(name, v);
  }

  setMouth(aa: number, ih: number, ou: number): void {
    this.set('aa', aa);
    this.set('ih', ih);
    this.set('ou', ou);
  }

  setBlink(v: number): void {
    this.set('blink', v);
  }

  setMood(v: number): void {
    this.set('happy', Math.max(0, v));
    this.set('relaxed', Math.max(0, -v));
  }

  debug(): Record<string, number> {
    return { ...this.values };
  }
}

/**
 * Load a VRM avatar and adapt it to HumanoidRig. Animation drives the
 * NORMALIZED humanoid bones (identity rest frames, meters, T-pose), and
 * `tick` runs `vrm.update()` each frame to propagate bone rotations and
 * expression weights onto the actual model.
 */
export async function loadVRMRig(url: string): Promise<HumanoidRig> {
  const loader = new GLTFLoader();
  loader.register((parser) => new VRMLoaderPlugin(parser));
  const gltf = await loader.loadAsync(url);
  const vrm: VRM = gltf.userData.vrm;
  if (!vrm) throw new Error(`loadVRMRig(${url}): not a VRM file`);

  // VRM 0.x models face -Z; rotate so they match the scene convention (+Z).
  VRMUtils.rotateVRM0(vrm);
  VRMUtils.removeUnnecessaryVertices(gltf.scene);
  VRMUtils.combineSkeletons(gltf.scene);

  vrm.scene.traverse((obj) => {
    if ((obj as THREE.Mesh).isMesh) obj.castShadow = true;
  });

  const root = new THREE.Group();
  root.name = 'vrm-humanoid';
  root.add(vrm.scene);

  // chest / neck / shoulders are optional in the VRM spec — fall back to a
  // neighboring joint so the animator's writes still land somewhere sane.
  const FALLBACK: Partial<Record<JointName, JointName>> = {
    chest: 'spine',
    neck: 'head',
    leftShoulder: 'leftUpperArm',
    rightShoulder: 'rightUpperArm',
  };
  const joints = {} as Record<JointName, THREE.Object3D>;
  for (const key of Object.keys(VRM_BONES) as JointName[]) {
    const node = vrm.humanoid.getNormalizedBoneNode(VRM_BONES[key]);
    if (node) joints[key] = node;
  }
  const missing: string[] = [];
  for (const key of Object.keys(VRM_BONES) as JointName[]) {
    if (!joints[key]) {
      const fb = FALLBACK[key];
      if (fb && joints[fb]) joints[key] = joints[fb];
      else missing.push(key);
    }
  }
  if (missing.length) throw new Error(`loadVRMRig(${url}): missing bones: ${missing.join(', ')}`);

  // Capture finger T-pose world orientations BEFORE finalizeRig lowers the
  // arms — clip retargeting needs the canonical T-pose reference frame.
  root.updateMatrixWorld(true);
  const fingerRetarget: NonNullable<HumanoidRig['fingerRetarget']> = {};
  for (const side of ['left', 'right'] as const) {
    for (const digit of ['Thumb', 'Index', 'Middle', 'Ring', 'Little']) {
      for (const seg of digit === 'Thumb'
        ? ['Metacarpal', 'Proximal', 'Distal']
        : ['Proximal', 'Intermediate', 'Distal']) {
        const name = `${side}${digit}${seg}`;
        const node = vrm.humanoid.getNormalizedBoneNode(name as VRMHumanBoneName);
        if (node) {
          fingerRetarget[name] = {
            node,
            tposeWorld: node.getWorldQuaternion(new THREE.Quaternion()),
          };
        }
      }
    }
  }

  const rig = finalizeRig(root, joints);
  if (Object.keys(fingerRetarget).length) rig.fingerRetarget = fingerRetarget;
  rig.face = new VRMFace(vrm);
  rig.tick = (dt) => vrm.update(dt);

  // Finger bones via the normalized humanoid (all spec finger names).
  const fingerNodes = { left: [] as THREE.Object3D[], right: [] as THREE.Object3D[] };
  for (const side of ['left', 'right'] as const) {
    for (const digit of ['Thumb', 'Index', 'Middle', 'Ring', 'Little']) {
      for (const seg of digit === 'Thumb'
        ? ['Metacarpal', 'Proximal', 'Distal']
        : ['Proximal', 'Intermediate', 'Distal']) {
        const node = vrm.humanoid.getNormalizedBoneNode(
          `${side}${digit}${seg}` as VRMHumanBoneName,
        );
        if (node) fingerNodes[side].push(node);
      }
    }
  }
  if (fingerNodes.left.length || fingerNodes.right.length) {
    rig.fingers = collectFingers(
      root,
      joints,
      { left: rig.armAxes.left.flex, right: rig.armAxes.right.flex },
      fingerNodes,
    );
  }
  return rig;
}
