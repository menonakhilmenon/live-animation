import * as THREE from 'three';

/**
 * Joint names follow the VRM/Mixamo humanoid convention so a loaded GLB rig
 * can be mapped onto the same interface later.
 */
export type JointName =
  | 'hips'
  | 'spine'
  | 'chest'
  | 'neck'
  | 'head'
  | 'leftShoulder'
  | 'leftUpperArm'
  | 'leftLowerArm'
  | 'leftHand'
  | 'rightShoulder'
  | 'rightUpperArm'
  | 'rightLowerArm'
  | 'rightHand'
  | 'leftUpperLeg'
  | 'leftLowerLeg'
  | 'leftFoot'
  | 'rightUpperLeg'
  | 'rightLowerLeg'
  | 'rightFoot';

export interface RestPose {
  position: THREE.Vector3;
  quaternion: THREE.Quaternion;
}

/**
 * A humanoid rig: a root object plus named joint transforms and their rest
 * pose. The animator works purely against this interface, so any skeleton
 * (procedural, GLB, VRM) can drive the same animation code.
 */
export interface HumanoidRig {
  root: THREE.Object3D;
  joints: Record<JointName, THREE.Object3D>;
  rest: Record<JointName, RestPose>;
  /**
   * Multiplier converting meter-space positional offsets into the rig's
   * local units (1 for the procedural rig; ~100 for centimeter-unit
   * Mixamo skeletons under a 0.01-scaled armature).
   */
  positionScale: number;
}

const BONE_MAT = new THREE.MeshStandardMaterial({ color: 0x9fb4cc, roughness: 0.55 });
const JOINT_MAT = new THREE.MeshStandardMaterial({ color: 0x5c86b0, roughness: 0.4 });

/** A joint pivot with a capsule "bone" mesh hanging along -Y toward its child. */
function segment(name: string, length: number, radius: number): THREE.Object3D {
  const pivot = new THREE.Object3D();
  pivot.name = name;
  const capsule = new THREE.Mesh(
    new THREE.CapsuleGeometry(radius, Math.max(length - radius * 2, 0.01), 4, 10),
    BONE_MAT,
  );
  capsule.position.y = -length / 2;
  capsule.castShadow = true;
  pivot.add(capsule);
  const ball = new THREE.Mesh(new THREE.SphereGeometry(radius * 1.15, 12, 10), JOINT_MAT);
  ball.castShadow = true;
  pivot.add(ball);
  return pivot;
}

/**
 * Builds a simple articulated humanoid (~1.7 units tall) out of capsules.
 * Every joint's +Y axis points up the limb at rest; limbs hang along -Y.
 */
export function createHumanoid(): HumanoidRig {
  const root = new THREE.Group();
  root.name = 'humanoid';

  const hips = new THREE.Object3D();
  hips.name = 'hips';
  hips.position.y = 0.95;
  root.add(hips);

  const pelvis = new THREE.Mesh(
    new THREE.SphereGeometry(0.14, 14, 12),
    JOINT_MAT,
  );
  pelvis.scale.set(1.25, 0.8, 0.9);
  pelvis.castShadow = true;
  hips.add(pelvis);

  // --- Torso (spine chain grows upward, so segments point +Y) ---
  const spine = new THREE.Object3D();
  spine.name = 'spine';
  spine.position.y = 0.1;
  hips.add(spine);

  const chest = new THREE.Object3D();
  chest.name = 'chest';
  chest.position.y = 0.22;
  spine.add(chest);

  const torso = new THREE.Mesh(
    new THREE.CapsuleGeometry(0.13, 0.22, 4, 12),
    BONE_MAT,
  );
  torso.position.y = 0.12;
  torso.scale.set(1.25, 1, 0.8);
  torso.castShadow = true;
  chest.add(torso);

  const neck = new THREE.Object3D();
  neck.name = 'neck';
  neck.position.y = 0.32;
  chest.add(neck);

  const head = new THREE.Object3D();
  head.name = 'head';
  head.position.y = 0.08;
  neck.add(head);

  const skull = new THREE.Mesh(new THREE.SphereGeometry(0.11, 16, 14), BONE_MAT);
  skull.position.y = 0.1;
  skull.castShadow = true;
  head.add(skull);

  // --- Arms ---
  const makeArm = (side: 1 | -1, prefix: 'left' | 'right') => {
    const shoulder = new THREE.Object3D();
    shoulder.name = `${prefix}Shoulder`;
    shoulder.position.set(0.17 * side, 0.26, 0);
    chest.add(shoulder);

    const upperArm = segment(`${prefix}UpperArm`, 0.28, 0.05);
    shoulder.add(upperArm);

    const lowerArm = segment(`${prefix}LowerArm`, 0.26, 0.042);
    lowerArm.position.y = -0.28;
    upperArm.add(lowerArm);

    const hand = new THREE.Object3D();
    hand.name = `${prefix}Hand`;
    hand.position.y = -0.26;
    lowerArm.add(hand);
    const palm = new THREE.Mesh(new THREE.SphereGeometry(0.05, 10, 8), BONE_MAT);
    palm.scale.set(0.8, 1.2, 0.5);
    palm.position.y = -0.04;
    palm.castShadow = true;
    hand.add(palm);

    return { shoulder, upperArm, lowerArm, hand };
  };

  // --- Legs ---
  const makeLeg = (side: 1 | -1, prefix: 'left' | 'right') => {
    const upperLeg = segment(`${prefix}UpperLeg`, 0.42, 0.065);
    upperLeg.position.set(0.1 * side, -0.02, 0);
    hips.add(upperLeg);

    const lowerLeg = segment(`${prefix}LowerLeg`, 0.42, 0.055);
    lowerLeg.position.y = -0.42;
    upperLeg.add(lowerLeg);

    const foot = new THREE.Object3D();
    foot.name = `${prefix}Foot`;
    foot.position.y = -0.42;
    lowerLeg.add(foot);
    const shoe = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.06, 0.2), JOINT_MAT);
    shoe.position.set(0, -0.05, 0.05);
    shoe.castShadow = true;
    foot.add(shoe);

    return { upperLeg, lowerLeg, foot };
  };

  const lArm = makeArm(1, 'left');
  const rArm = makeArm(-1, 'right');
  const lLeg = makeLeg(1, 'left');
  const rLeg = makeLeg(-1, 'right');

  const joints: Record<JointName, THREE.Object3D> = {
    hips,
    spine,
    chest,
    neck,
    head,
    leftShoulder: lArm.shoulder,
    leftUpperArm: lArm.upperArm,
    leftLowerArm: lArm.lowerArm,
    leftHand: lArm.hand,
    rightShoulder: rArm.shoulder,
    rightUpperArm: rArm.upperArm,
    rightLowerArm: rArm.lowerArm,
    rightHand: rArm.hand,
    leftUpperLeg: lLeg.upperLeg,
    leftLowerLeg: lLeg.lowerLeg,
    leftFoot: lLeg.foot,
    rightUpperLeg: rLeg.upperLeg,
    rightLowerLeg: rLeg.lowerLeg,
    rightFoot: rLeg.foot,
  };

  const rest = {} as Record<JointName, RestPose>;
  for (const key of Object.keys(joints) as JointName[]) {
    rest[key] = {
      position: joints[key].position.clone(),
      quaternion: joints[key].quaternion.clone(),
    };
  }

  return { root, joints, rest, positionScale: 1 };
}

/** Reset every joint to its captured rest pose (call before applying layers). */
export function resetToRest(rig: HumanoidRig): void {
  for (const key of Object.keys(rig.joints) as JointName[]) {
    rig.joints[key].position.copy(rig.rest[key].position);
    rig.joints[key].quaternion.copy(rig.rest[key].quaternion);
  }
}
