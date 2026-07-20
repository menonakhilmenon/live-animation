import * as THREE from 'three';
import { HumanoidRig, JointName } from './humanoid';

/**
 * Local rotation axes for the arm joints, resolved so that a POSITIVE angle
 * always means the same world-space motion regardless of the skeleton's
 * local bone frames:
 *   swing  — hand moves toward character-forward (+Z)
 *   abduct — hand moves outward, away from the body
 *   flex   — elbow bends, hand moving forward/up
 */
export interface ArmAxes {
  swing: THREE.Vector3;
  abduct: THREE.Vector3;
  flex: THREE.Vector3;
}

const CANDIDATES = [
  new THREE.Vector3(1, 0, 0),
  new THREE.Vector3(-1, 0, 0),
  new THREE.Vector3(0, 1, 0),
  new THREE.Vector3(0, -1, 0),
  new THREE.Vector3(0, 0, 1),
  new THREE.Vector3(0, 0, -1),
];

const _before = new THREE.Vector3();
const _after = new THREE.Vector3();

/**
 * Empirically determine arm rotation axes for a rig in its rest pose (arms
 * hanging, character facing +Z): each candidate local axis gets a small test
 * rotation and the one moving the hand furthest in the desired world
 * direction wins. This sidesteps every per-skeleton bone-frame convention
 * (Mixamo, VRM, hand-built) with one cheap probe at load time.
 *
 * Must be called before the rest pose is used for animation, but restores
 * all rotations it touches.
 */
export function probeArmAxes(rig: Pick<HumanoidRig, 'root' | 'joints'>): {
  left: ArmAxes;
  right: ArmAxes;
} {
  const result = {} as { left: ArmAxes; right: ArmAxes };
  for (const side of ['left', 'right'] as const) {
    const upper = rig.joints[`${side}UpperArm` as JointName];
    const lower = rig.joints[`${side}LowerArm` as JointName];
    const hand = rig.joints[`${side}Hand` as JointName];
    const forward = new THREE.Vector3(0, 0, 1);
    const outward = new THREE.Vector3(side === 'left' ? 1 : -1, 0, 0);
    result[side] = {
      swing: bestAxis(rig.root, upper, hand, forward),
      abduct: bestAxis(rig.root, upper, hand, outward),
      flex: bestAxis(rig.root, lower, hand, forward),
    };
  }
  return result;
}

function bestAxis(
  root: THREE.Object3D,
  joint: THREE.Object3D,
  effector: THREE.Object3D,
  desiredWorldDir: THREE.Vector3,
): THREE.Vector3 {
  const q0 = joint.quaternion.clone();
  root.updateMatrixWorld(true);
  effector.getWorldPosition(_before);

  let best = CANDIDATES[0];
  let bestScore = -Infinity;
  for (const axis of CANDIDATES) {
    joint.quaternion.copy(q0);
    joint.rotateOnAxis(axis, 0.3);
    root.updateMatrixWorld(true);
    effector.getWorldPosition(_after);
    const score = _after.sub(_before).dot(desiredWorldDir);
    if (score > bestScore) {
      bestScore = score;
      best = axis;
    }
  }
  joint.quaternion.copy(q0);
  root.updateMatrixWorld(true);
  return best.clone();
}
