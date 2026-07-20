// Downloads the demo character (Xbot — Mixamo-rigged, from the three.js
// examples) into public/models/. The model is not committed to the repo;
// the app falls back to the procedural capsule rig when it's absent.
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';

const URL_GLB =
  'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/models/gltf/Xbot.glb';
const outDir = path.join(import.meta.dirname, '..', 'public', 'models');
const out = path.join(outDir, 'Xbot.glb');

await mkdir(outDir, { recursive: true });
const res = await fetch(URL_GLB);
if (!res.ok) throw new Error(`Download failed: ${res.status} ${res.statusText}`);
await writeFile(out, Buffer.from(await res.arrayBuffer()));
console.log('Saved', out);
