import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { probeArmAxes, probeAxis } from './axes';
import { FingerJoint, HumanoidRig, JointName, RestPose } from './humanoid';

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

  // Finger bones (Mixamo naming: LeftHandIndex1..3 etc.). World T-pose
  // orientations are captured NOW, before finalizeRig's arms-down
  // calibration, so generated clips can retarget articulated fingers.
  const DIGIT_VRM: Record<string, string> = {
    thumb: 'Thumb', index: 'Index', middle: 'Middle', ring: 'Ring', pinky: 'Little',
  };
  const fingerNodes = { left: [] as THREE.Object3D[], right: [] as THREE.Object3D[] };
  const fingerRetarget: NonNullable<HumanoidRig['fingerRetarget']> = {};
  root.updateMatrixWorld(true);
  gltf.scene.traverse((obj) => {
    const m = normalizeBoneName(obj.name).match(/^(left|right)hand(thumb|index|middle|ring|pinky)(\d)$/);
    if (!m) return;
    fingerNodes[m[1] as 'left' | 'right'].push(obj);
    const digit = DIGIT_VRM[m[2]];
    const segs = digit === 'Thumb'
      ? ['Metacarpal', 'Proximal', 'Distal']
      : ['Proximal', 'Intermediate', 'Distal'];
    const seg = segs[Number(m[3]) - 1];
    if (seg) {
      fingerRetarget[`${m[1]}${digit}${seg}`] = {
        node: obj,
        tposeWorld: obj.getWorldQuaternion(new THREE.Quaternion()),
      };
    }
  });

  const rig = finalizeRig(root, joints);
  if (Object.keys(fingerRetarget).length) rig.fingerRetarget = fingerRetarget;
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

/**
 * Shared skeleton adaptation for loaded models (GLB, VRM): lower the
 * T-pose arms so "rest" means relaxed, compute the meters→local-units
 * position scale, probe per-rig arm axes, and capture the rest pose.
 */
export function finalizeRig(root: THREE.Group, joints: Record<JointName, THREE.Object3D>): HumanoidRig {
  root.updateMatrixWorld(true);

  // Loaded models arrive in T-pose (Mixamo authored, VRM normalized bones);
  // capture each joint's world orientation NOW, before the arms-down
  // calibration, as the canonical reference frame for motion-clip retargeting.
  const tposeWorld = {} as Record<JointName, THREE.Quaternion>;
  for (const key of Object.keys(joints) as JointName[]) {
    tposeWorld[key] = joints[key].getWorldQuaternion(new THREE.Quaternion());
  }

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

  return { root, joints, rest, positionScale, armAxes, tposeWorld };
}

/**
 * Package finger bones for animation: capture rest rotations and resolve
 * the curl direction. The curl axis is the hand's flex axis (finger frames
 * are consistent with the hand within a rig); its sign is probed by test-
 * curling all fingers both ways and keeping the direction that brings the
 * fingertips toward the body — with calibrated hanging arms the palms face
 * the thighs, so curling inward reduces distance to the hips.
 */
export function collectFingers(
  root: THREE.Object3D,
  joints: Record<JointName, THREE.Object3D>,
  flexAxis: { left: THREE.Vector3; right: THREE.Vector3 },
  nodesPerSide: { left: THREE.Object3D[]; right: THREE.Object3D[] },
): { left: FingerJoint[]; right: FingerJoint[] } {
  const hipsPos = new THREE.Vector3();
  const tipPos = new THREE.Vector3();
  const result = { left: [] as FingerJoint[], right: [] as FingerJoint[] };

  for (const side of ['left', 'right'] as const) {
    const nodes = nodesPerSide[side];
    if (!nodes.length) continue;
    const rests = nodes.map((n) => n.quaternion.clone());

    const totalTipDistance = (sign: number): number => {
      for (const [i, n] of nodes.entries()) {
        n.quaternion.copy(rests[i]);
        n.rotateOnAxis(flexAxis[side], sign * 0.5);
      }
      root.updateMatrixWorld(true);
      joints.hips.getWorldPosition(hipsPos);
      let sum = 0;
      for (const n of nodes) {
        const tip = n.children[0] ?? n;
        tip.getWorldPosition(tipPos);
        sum += tipPos.distanceTo(hipsPos);
      }
      return sum;
    };

    const sign = totalTipDistance(1) <= totalTipDistance(-1) ? 1 : -1;
    for (const [i, n] of nodes.entries()) {
      n.quaternion.copy(rests[i]);
      result[side].push({
        node: n,
        rest: rests[i],
        curlAxis: flexAxis[side].clone().multiplyScalar(sign),
      });
    }
    root.updateMatrixWorld(true);
  }
  return result;
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
