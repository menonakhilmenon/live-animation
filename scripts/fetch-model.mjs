// Downloads the demo character (Xbot — Mixamo-rigged, from the three.js
// examples) into public/models/. The model is not committed to the repo;
// the app falls back to the procedural capsule rig when it's absent.
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';

const MODELS = [
  {
    // Mixamo-rigged demo character from the three.js examples.
    url: 'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/models/gltf/Xbot.glb',
    file: 'Xbot.glb',
  },
  {
    // VRoid sample avatar (see https://github.com/madjin/vrm-samples —
    // VRoid Studio sample models, freely usable per their conditions).
    // Has VRM expressions: lip-sync visemes, blink, happy/relaxed.
    url: 'https://raw.githubusercontent.com/madjin/vrm-samples/master/vroid/fem_vroid.vrm',
    file: 'avatar.vrm',
  },
];

const outDir = path.join(import.meta.dirname, '..', 'public', 'models');
await mkdir(outDir, { recursive: true });
for (const { url, file } of MODELS) {
  const out = path.join(outDir, file);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Download failed for ${file}: ${res.status} ${res.statusText}`);
  await writeFile(out, Buffer.from(await res.arrayBuffer()));
  console.log('Saved', out);
}
