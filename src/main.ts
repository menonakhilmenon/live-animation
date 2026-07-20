import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { AudioEngine } from './audio/engine';
import { Animator } from './animation/animator';
import { createHumanoid } from './rig/humanoid';
import { loadGLBRig } from './rig/loader';
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
const capsuleRig = createHumanoid();
scene.add(capsuleRig.root);

const audio = new AudioEngine();
let activeRig = capsuleRig;
let animator = new Animator(activeRig);
setupUI(audio);

const setActiveRig = (rig: typeof capsuleRig) => {
  activeRig = rig;
  animator = new Animator(rig);
  if (import.meta.env.DEV) {
    (window as unknown as Record<string, unknown>).__app = { rig, audio };
  }
};

// Load the real character model if present (see README: npm run fetch:model).
// Falls back silently to the capsule rig when the GLB is missing.
loadGLBRig('/models/Xbot.glb')
  .then((glbRig) => {
    scene.add(glbRig.root);
    capsuleRig.root.visible = false;
    setActiveRig(glbRig);

    const btn = document.getElementById('btn-character') as HTMLButtonElement;
    btn.hidden = false;
    btn.textContent = 'Character: model';
    btn.addEventListener('click', () => {
      const useCapsule = activeRig === glbRig;
      capsuleRig.root.visible = useCapsule;
      glbRig.root.visible = !useCapsule;
      setActiveRig(useCapsule ? capsuleRig : glbRig);
      btn.textContent = useCapsule ? 'Character: capsule' : 'Character: model';
    });
  })
  .catch((err) => console.warn('No character model loaded, using capsule rig:', err.message));

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

// Dev-only hook so automated tests can sample the rig and audio features.
if (import.meta.env.DEV) {
  (window as unknown as Record<string, unknown>).__app = { rig: activeRig, audio };
}

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
