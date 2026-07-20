# live-animation

Real-time procedural humanoid animation driven by audio. Feed it an audio
file or your microphone and a humanoid rig dances to it — no pre-baked
animation clips, everything is generated live from the audio signal.

## Quick start

```sh
npm install
npm run fetch:model   # optional: downloads the demo character (Xbot, ~3 MB)
npm run dev           # opens a Vite dev server, usually http://localhost:5173
```

Then either **Load audio file** (plays back with a scrub bar) or **Use
microphone** (analysis only, no playback). Drag to orbit the camera.

`npm run build` type-checks and produces a static bundle in `dist/`.

## How it works

```
audio file / mic
      │
      ▼
AudioEngine (src/audio/engine.ts)
  Web Audio AnalyserNode → per-frame features:
  RMS level, bass/mid/treble energy, spectral brightness,
  beat detection (adaptive bass-onset threshold) + beat pulse envelope,
  tempo estimation (src/audio/tempo.ts): BPM from folded inter-onset
  intervals + a phase-locked oscillator exposing continuous beat phase
      │
      ▼  AudioFeatures (src/audio/features.ts)
Animator (src/animation/animator.ts)
  Layered procedural motion, smoothed by critically-damped springs:
  breathing · hip bounce + knee flex · weight shift/sway ·
  head nod · arm swing/raise · shoulder shrug
  When tempo confidence is high the groove phase-locks to the beat and
  anticipates it (pulse peaks ON the beat, not after detection); style
  amplitudes crossfade with sustained energy: calm sway → groove → hype
  Foot IK (src/animation/ik.ts): rig-agnostic CCD pins the feet to their
  ground rest positions while hips bounce and sway
  Choreography: four arm moves (pendulum, beat pump, raised groove, side
  sway) rotate on 4-beat phrase boundaries with crossfades; arm rotations
  use per-rig axes probed at load time (src/rig/axes.ts), so moves are
  authored once in world-space semantics and work on any skeleton
      │
      ▼
HumanoidRig (src/rig/humanoid.ts, src/rig/loader.ts)
  Named joint map (VRM/Mixamo-style names) + rest pose + positionScale.
  Two implementations: a self-contained capsule humanoid built in code,
  and a GLB loader that maps Mixamo-named bones onto the same interface
  (lowering T-pose arms before capturing the rest pose). The demo model
  (Xbot from the three.js examples) is fetched by `npm run fetch:model`
  and not committed; without it the app falls back to the capsule rig.
  Use the "Character" button to switch rigs at runtime.
      │
      ▼
Three.js scene (src/main.ts) — rendered every frame
```

Design notes:

- **Features, not FFT bins, cross the boundary.** Animation layers only see
  normalized `AudioFeatures`, so the analysis side can change freely.
- **Rest pose + additive layers.** Each frame the rig is reset to its rest
  pose and motion layers are added on top, scaled by live features. Layers
  compose without fighting each other.
- **Springs everywhere.** Beats and energy jumps are discontinuous; every
  layer follows its target through a critically-damped spring
  (`src/animation/spring.ts`) so motion stays organic.
- **Swappable rig.** `HumanoidRig` is just `{ root, joints, rest }` with
  VRM/Mixamo-style joint names. To use a real character model, load a GLB,
  map its bones into that record, and the animator works unchanged.

## Roadmap ideas

- Lip sync from mid-band energy when a face mesh is available
- Stepping: move the foot anchors when the weight shift exceeds a threshold
