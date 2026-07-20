import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { probeArmAxes, probeAxis } from './axes';
import { HumanoidRig, JointName, RestPose } from './humanoid';

/**
 * Accepted (normalized) bone names per joint. Covers Mixamo naming; add
 * aliases here to support other conventions (VRM, UE, etc.).
 * Names are normalized before matching: lowercased, punctuation stripped,
 * "mixamorig" prefix dropped — so `mixamorig:LeftForeArm` → `leftforearm`.
 */
const BONE_ALIASES: Record<JointName, string[]> = {
  hips: ['hips'],
  spine: ['spine'],
  chest: ['spine2', 'chest', 'upperchest'],
  neck: ['neck'],
  head: ['head'],
  leftShoulder: ['leftshoulder'],
  leftUpperArm: ['leftarm', 'leftupperarm'],
  leftLowerArm: ['leftforearm', 'leftlowerarm'],
  leftHand: ['lefthand'],
  rightShoulder: ['rightshoulder'],
  rightUpperArm: ['rightarm', 'rightupperarm'],
  rightLowerArm: ['rightforearm', 'rightlowerarm'],
  rightHand: ['righthand'],
  leftUpperLeg: ['leftupleg', 'leftupperleg'],
  leftLowerLeg: ['leftleg', 'leftlowerleg'],
  leftFoot: ['leftfoot'],
  rightUpperLeg: ['rightupleg', 'rightupperleg'],
  rightLowerLeg: ['rightleg', 'rightlowerleg'],
  rightFoot: ['rightfoot'],
};

function normalizeBoneName(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]/g, '')
    .replace(/^mixamorig/, '');
}

/**
 * Loads a GLB character with a Mixamo-style skeleton and adapts it to the
 * HumanoidRig interface the animator consumes.
 *
 * Steps beyond bone matching:
 *  - arms are lowered from the authored T-pose to a relaxed hang *before*
 *    the rest pose is captured, since animation layers assume arms-down rest
 *  - `positionScale` converts the animator's meter-space positional offsets
 *    into the skeleton's local units (Mixamo rigs are centimeters under a
 *    0.01-scaled armature)
 */
export async function loadGLBRig(url: string): Promise<HumanoidRig> {
  const gltf = await new GLTFLoader().loadAsync(url);
  const root = new THREE.Group();
  root.name = 'glb-humanoid';
  root.add(gltf.scene);

  gltf.scene.traverse((obj) => {
    if ((obj as THREE.Mesh).isMesh) {
      obj.castShadow = true;
      obj.receiveShadow = false;
    }
  });

  // Match bones to joints by normalized name (exact match, first wins).
  const joints = {} as Record<JointName, THREE.Object3D>;
  const wanted = Object.entries(BONE_ALIASES) as [JointName, string[]][];
  gltf.scene.traverse((obj) => {
    const n = normalizeBoneName(obj.name);
    for (const [joint, aliases] of wanted) {
      if (!joints[joint] && aliases.includes(n)) joints[joint] = obj;
    }
  });

  const missing = wanted.filter(([j]) => !joints[j]).map(([j]) => j);
  if (missing.length) {
    throw new Error(`loadGLBRig(${url}): unmapped joints: ${missing.join(', ')}`);
  }

  return finalizeRig(root, joints);
}

/**
 * Shared skeleton adaptation for loaded models (GLB, VRM): lower the
 * T-pose arms so "rest" means relaxed, compute the meters→local-units
 * position scale, probe per-rig arm axes, and capture the rest pose.
 */
export function finalizeRig(root: THREE.Group, joints: Record<JointName, THREE.Object3D>): HumanoidRig {
  root.updateMatrixWorld(true);
  calibrateArmsDown(root, joints);

  // Positional offsets from the animator are in meters (world space); convert
  // to hips-local units via the inverse world scale of the hips' parent.
  const parentScale = new THREE.Vector3(1, 1, 1);
  joints.hips.parent?.getWorldScale(parentScale);
  const positionScale = 1 / Math.max(parentScale.y, 1e-6);

  // Probe arm axes in the calibrated (arms-down) pose; the probe restores
  // any rotations it applies, so the rest capture below is unaffected.
  const armAxes = probeArmAxes({ root, joints });

  const rest = {} as Record<JointName, RestPose>;
  for (const key of Object.keys(joints) as JointName[]) {
    rest[key] = {
      position: joints[key].position.clone(),
      quaternion: joints[key].quaternion.clone(),
    };
  }

  return { root, joints, rest, positionScale, armAxes };
}

function calibrateArmsDown(root: THREE.Group, joints: Record<JointName, THREE.Object3D>): void {
  const down = new THREE.Vector3(0, -1, 0);
  for (const s of ['left', 'right'] as const) {
    const upper = joints[`${s}UpperArm`];
    const lower = joints[`${s}LowerArm`];
    const hand = joints[`${s}Hand`];
    // Swing the arm ~65° down toward the body around whichever local axis
    // actually lowers the hand — bone frames differ per format (Mixamo Y-
    // along-bone, VRM0 normalized bones under a 180° parent rotation, ...),
    // so the axis is probed, not assumed.
    upper.rotateOnAxis(probeAxis(root, upper, hand, down), 1.15);
    root.updateMatrixWorld(true);
    // A slight forward forearm bend keeps the pose from looking stiff
    // (probed against +Z since "down" is degenerate for a hanging arm).
    lower.rotateOnAxis(probeAxis(root, lower, hand, new THREE.Vector3(0, 0, 1)), 0.15);
    root.updateMatrixWorld(true);
  }
}
