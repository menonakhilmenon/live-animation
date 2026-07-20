# live-animation

Lifelike procedural humanoid behavior driven by audio. Feed it an audio
file or your microphone and a character reacts like a person would — dancing
to music, gesturing and lip-syncing to speech, idling in silence. No
pre-baked animation clips; everything is generated from the audio signal.

## Quick start

```sh
npm install
npm run fetch:model   # optional: downloads the demo characters
                      # (Xbot ~3 MB, VRoid VRM avatar ~12 MB)
npm run dev           # opens a Vite dev server, usually http://localhost:5173
```

Then either **Load audio file** (analyzed in full on load, plays with a
scrub bar) or **Use microphone** (live causal analysis, no playback). Drag
to orbit the camera; the **Character** button cycles capsule / Xbot / VRM.

`npm run build` type-checks and produces a static bundle in `dist/`.
`npm run test:e2e` runs the headless end-to-end suite (see
`.claude/skills/run-app/SKILL.md` for the harness).

## How it works

```
audio file ──► offline analysis (src/audio/analysis.ts, fft.ts)
      │          whole-file STFT → spectral-flux onsets (bass and full-band
      │          envelopes normalized separately) → tempo autocorrelation →
      │          DP beat selection → arithmetic grid fit → downbeats from
      │          accent scoring → loudness sections + drop marking →
      │          music / speech / silence classification
      │                            │
mic ──► causal tracking            ▼  Timeline
  (AnalyserNode features,   AudioEngine (src/audio/engine.ts)
   adaptive beat detector,  merges timeline (exact beat/bar phase, section
   PLL tempo tracker)       loudness, onset novelty, seconds-until-drop)
                            into per-frame AudioFeatures
                                           │
                                           ▼
                     Animator (src/animation/animator.ts)
   behavior dispatch: music → dance · speech → gestures · silence → idle
   DANCE   layered, bar-aligned: breathing · hip bounce + knee flex ·
           weight shift with figure-8 arcs · beat-anticipating pulse ·
           4 arm moves rotating on downbeats (underdamped crossfades) ·
           gaze stabilization · wind-up crouch before a known drop, hit
           pose as it lands · style scales with section loudness
   SPEECH  still stance · alternating-hand beat gestures on stressed
           onsets · head nods · attentive gaze wander
   IDLE    micro weight shifts + gaze wander from incommensurate sines
   All modes: rig-agnostic CCD foot IK (ik.ts) pins feet to the ground;
   finger curl; face (face.ts): lip-sync visemes by brightness, blink
   scheduler, mood follows energy — via VRM expressions when available
                                           │
                                           ▼
                     HumanoidRig (src/rig/)
   Named joint map + rest pose + positionScale + probed axes + fingers.
   Implementations: procedural capsule (humanoid.ts), Mixamo GLB
   (loader.ts), VRM avatar with expressions (vrm.ts). All skeleton
   differences are resolved empirically at load: arms-down calibration
   and every animation axis are probed by test-rotating and measuring
   world-space hand motion — no per-format bone-frame assumptions.
                                           │
                                           ▼
                     Three.js scene (src/main.ts)
```

Design notes:

- **Features, not FFT bins, cross the boundary.** Animation only sees
  normalized `AudioFeatures`; the analysis side can change freely.
- **Offline-first for accuracy.** A little latency is acceptable, so files
  are fully analyzed up front; animation runs off an exact, future-aware
  timeline (it knows the next beat and the next drop before they happen).
  The mic path stays causal.
- **Rest pose + additive layers.** Each frame the rig resets to rest and
  layers add on top — layers compose without fighting each other.
- **Springs everywhere.** Discontinuous features drive targets through
  damped springs (`spring.ts`); slightly underdamped ones add
  follow-through.
- **Probe, don't assume.** Every rig quirk that broke something (Mixamo
  bone frames, VRM0's 180° rotation, finger curl direction) is resolved by
  empirical probing at load time, keeping the animator fully rig-agnostic.

## Test assets

`e2e/make-wav.cjs` authors ground-truth audio: a 16 s structured 120 BPM
"song" (accented downbeats, quiet verse, abrupt drop at 6 s) and 8 s of
formant-shaped synthetic speech. The e2e suite asserts beat-grid and
downbeat error in milliseconds, drop/section detection, speech
classification, mode-appropriate body behavior, lip sync, and blinking.

## Roadmap ideas

- Stepping: move the foot anchors when the weight shift exceeds a threshold
- Analysis in a Web Worker for very long files
- Per-section content modes (speech intro → music) instead of per-track
- Beat-gesture vocabulary growth (iconic/deictic gestures, shrugs)
