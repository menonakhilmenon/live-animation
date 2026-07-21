import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { AudioEngine } from './audio/engine';
import { Animator } from './animation/animator';
import { MotionClip } from './animation/clip';
import { createHumanoid } from './rig/humanoid';
import { loadGLBRig } from './rig/loader';
import { loadVRMRig } from './rig/vrm';
import { setupUI, updateMeters } from './ui';

const container = document.getElementById('app')!;

// --- Renderer / scene / camera ---
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0e13);
scene.fog = new THREE.Fog(0x0b0e13, 6, 14);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 50);
camera.position.set(0, 1.5, 3.2);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 1, 0);
controls.enableDamping = true;
controls.maxDistance = 8;
controls.minDistance = 1.2;

// --- Lights ---
scene.add(new THREE.HemisphereLight(0x8fb0d8, 0x1a1f28, 0.7));
const key = new THREE.DirectionalLight(0xffffff, 1.6);
key.position.set(2.5, 4, 2);
key.castShadow = true;
key.shadow.mapSize.set(1024, 1024);
key.shadow.camera.left = -3;
key.shadow.camera.right = 3;
key.shadow.camera.top = 3;
key.shadow.camera.bottom = -1;
scene.add(key);
const rim = new THREE.DirectionalLight(0x6f9dff, 0.5);
rim.position.set(-2, 2, -2.5);
scene.add(rim);

// --- Ground ---
const ground = new THREE.Mesh(
  new THREE.CircleGeometry(6, 48),
  new THREE.MeshStandardMaterial({ color: 0x151b24, roughness: 0.9 }),
);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);
const grid = new THREE.GridHelper(12, 24, 0x2a3442, 0x1c2530);
scene.add(grid);

// --- Rig + systems ---
const audio = new AudioEngine();
const capsuleRig = createHumanoid();
scene.add(capsuleRig.root);

interface RigEntry {
  name: string;
  rig: ReturnType<typeof createHumanoid>;
}
const rigs: RigEntry[] = [{ name: 'capsule', rig: capsuleRig }];
let rigIdx = 0;
let activeRig = capsuleRig;
let animator = new Animator(activeRig);
setupUI(audio, (clip) => animator.playClip(clip));

const charBtn = document.getElementById('btn-character') as HTMLButtonElement;

const setActiveRig = (idx: number) => {
  rigIdx = idx;
  activeRig = rigs[idx].rig;
  animator = new Animator(activeRig);
  for (const [i, entry] of rigs.entries()) entry.rig.root.visible = i === idx;
  charBtn.textContent = `Character: ${rigs[idx].name}`;
  charBtn.hidden = rigs.length < 2;
  if (import.meta.env.DEV) {
    (window as unknown as Record<string, unknown>).__app = {
      rig: activeRig,
      audio,
      animator,
      rigCount: rigs.length,
      playClip: (clip: MotionClip) => animator.playClip(clip),
    };
  }
};
setActiveRig(0);

charBtn.addEventListener('click', () => setActiveRig((rigIdx + 1) % rigs.length));

// Load optional character models (see README: npm run fetch:model). Each
// falls back silently when its file is absent. The VRM avatar is preferred
// when available — it's the only rig with facial expressions.
loadGLBRig('/models/Xbot.glb')
  .then((rig) => {
    scene.add(rig.root);
    rigs.push({ name: 'model', rig });
    setActiveRig(rigs[rigIdx].name === 'capsule' ? rigs.length - 1 : rigIdx);
  })
  .catch((err) => console.warn('No GLB model loaded:', err.message));

loadVRMRig('/models/avatar.vrm')
  .then((rig) => {
    scene.add(rig.root);
    rigs.push({ name: 'avatar', rig });
    setActiveRig(rigs.length - 1);
  })
  .catch((err) => console.warn('No VRM avatar loaded:', err.message));

// --- Loop ---
const clock = new THREE.Clock();
renderer.setAnimationLoop(() => {
  const dt = clock.getDelta();
  const features = audio.update(dt);
  animator.update(features, dt);
  updateMeters(audio);
  controls.update();
  renderer.render(scene, camera);
});

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
