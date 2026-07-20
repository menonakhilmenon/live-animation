import * as THREE from 'three';

const _jointPos = new THREE.Vector3();
const _effector = new THREE.Vector3();
const _v1 = new THREE.Vector3();
const _v2 = new THREE.Vector3();
const _qWorld = new THREE.Quaternion();
const _qParent = new THREE.Quaternion();
const _qParentInv = new THREE.Quaternion();
const _qDelta = new THREE.Quaternion();

/**
 * Pin an end effector (ankle) to a world-space target by adjusting a
 * two-joint chain (hip, knee) with a few CCD iterations.
 *
 * CCD is used instead of an analytic two-bone solver because it needs no
 * knowledge of the rig's local bone axes — it works identically on the
 * procedural capsule rig and Mixamo skeletons. The animation layers bend
 * the knee in its natural direction *before* this runs, so CCD starts near
 * a good pose and converges in a handful of iterations without inventing
 * sideways knee bends.
 */
export function pinEffector(
  joints: THREE.Object3D[],
  effector: THREE.Object3D,
  targetWorld: THREE.Vector3,
  iterations = 12,
): void {
  for (let iter = 0; iter < iterations; iter++) {
    // Innermost joint first (knee), then hip.
    for (const joint of joints) {
      joint.updateWorldMatrix(true, true);
      joint.getWorldPosition(_jointPos);
      effector.getWorldPosition(_effector);

      _v1.subVectors(_effector, _jointPos);
      _v2.subVectors(targetWorld, _jointPos);
      if (_v1.lengthSq() < 1e-10 || _v2.lengthSq() < 1e-10) continue;
      _v1.normalize();
      _v2.normalize();
      if (_v1.dot(_v2) > 0.999999) continue;

      // World-space rotation carrying the effector direction onto the target
      // direction, converted into the joint's local frame:
      // local' = inv(parentWorld) * delta * parentWorld * local
      _qDelta.setFromUnitVectors(_v1, _v2);
      joint.parent!.getWorldQuaternion(_qParent);
      _qParentInv.copy(_qParent).invert();
      _qWorld.copy(_qParentInv).multiply(_qDelta).multiply(_qParent);
      joint.quaternion.premultiply(_qWorld);
    }
  }
}

/** Set an object's world orientation (used to keep pinned feet flat). */
export function setWorldQuaternion(obj: THREE.Object3D, worldQuat: THREE.Quaternion): void {
  obj.parent!.updateWorldMatrix(true, false);
  obj.parent!.getWorldQuaternion(_qParent);
  obj.quaternion.copy(_qParent.invert()).multiply(worldQuat);
}
