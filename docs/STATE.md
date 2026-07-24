# State of the project

Last updated: 2026-07-21. Branch: `worktree-audio-humanoid-setup` (all work
on `main`). This is the orientation doc — read it first, then the deeper
docs it links.

## What this is

An audio-driven humanoid animation system that grew into a **generative
animation suite**. Two ways in:

- **Reactive** (browser-only): load an audio file or mic → a character
  dances to music, gestures and lip-syncs to speech, idles in silence.
  Everything derived from the audio signal, no pre-baked clips.
- **Generative** (needs the Python sidecar): type text, pick an emotion →
  local TTS speaks it and a model generates matching full-body gesture
  animation, played back on the avatar in sync with lip-sync and mood.

The generative path is the focus of recent work. Its design principle:
**the model decides *what* plays and *when*; curated/authored motion is
what actually reaches the skeleton** (prebaked base + additive layers).
Raw model output never drives bones directly — that eliminated the motion
artifacts early attempts had.

## Architecture at a glance

```
text + emotion ─► Kokoro TTS (CPU) ─► wav + word/phoneme timing
     │                                        │
     └──────────► EMAGE gesture model ────────┤  (sidecar, ml/server.py)
                  (emotion-conditioned)        │
                                               ▼
              build_schedule(): DECIDE base clip + accents + additive layer
                                               │  GestureSchedule (JSON)
                                               ▼
   browser: SchedulePlayer composes  base loop  (crossfaded segments)
                                    + additive   (continuous overlay, energy-gated)
                                    + accents     (one-shot head nods/shakes)
                                    + procedural  (breathing, gaze, foot IK, lip-sync)
                                               ▼
                        canonical MotionClip → any rig (capsule / Mixamo / VRM)
```

Three layers, exactly as intended:

1. **Base** — a looping stance clip (our BEAT2-baked emotion loops, or a
   game-authored idle). Carries posture and idle life.
2. **Additive** — continuous expressive overlays (delta-from-own-frame-0,
   composited in world space, gated by speech energy) + one-shot accents.
   Carries the gestures.
3. **Procedural** — breathing, gaze leveling, foot IK, finger curl,
   lip-sync — always on top.

Everything crosses layer boundaries as the **canonical MotionClip**:
per-frame *world-space rotation deltas from a canonical T-pose*. This is
what makes any source (our model, or FFXV/FFXVI/BG3 mocap) play on any rig
(capsule, Mixamo GLB, VRM) with no per-format bone assumptions.

## Where things live

| Path | What |
|---|---|
| `src/audio/` | Offline analysis (beat grid, sections, speech/music/silence) + causal mic path |
| `src/rig/` | `HumanoidRig` abstraction; capsule / Mixamo (`loader.ts`) / VRM (`vrm.ts`); probed bone axes |
| `src/animation/clip.ts` | Canonical `MotionClip` format + `ClipPlayer` (single-clip playback, retargeting) |
| `src/animation/schedule.ts` | `SchedulePlayer` — base + additive + accents composition (the core new engine) |
| `src/animation/library.ts` | Clip conversion: GLTF/GLB, and `convertNamedSkeletonAnimations` + per-game bone maps |
| `src/animation/animator.ts` | Behavior dispatch (dance/speech/idle), schedule vs clip vs procedural |
| `ml/server.py` | FastAPI sidecar: TTS + EMAGE + `build_schedule` (the "decide" step) |
| `ml/generate.py` | audio → EMAGE → MotionClip; seed-pose + VQ sampling live here |
| `ml/train_emotion.py` | Emotion fine-tune of EMAGE on BEAT2 |
| `ml/train_scheduler.py` | Learned accent (nod/shake) predictor |
| `ml/bake_library.py` | Bakes the mocap base loops (`public/anims/talk_*.json`) |
| `public/anims/*.json` | Committed base/overlay clips (small; the runtime library) |
| `e2e/*.cjs` | Headless browser tests (clip, gesture, speak, drive) |
| `docs/` | This file + the deep docs below |

Deep docs: **[generative-animation.md](generative-animation.md)** (model
research, training, the ML architecture decisions) ·
**[game-animation-extraction.md](game-animation-extraction.md)** (how FFXV /
FFXVI / BG3 assets were extracted) · **[ml/README.md](../ml/README.md)**
(Python setup + ROCm notes) · **[.claude/skills/run-app/SKILL.md](../.claude/skills/run-app/SKILL.md)**
(run + test recipe).

## What works, measured

- **Generation fidelity**: after seed-pose conditioning + tuned VQ nucleus
  sampling (upper T=0.8, hands T=0.7), generated motion matches held-out
  BEAT2 mocap — arm posture 76° vs 72°, wrist velocity ~40 vs 35°/s (argmax
  was 6.6, a 5× regression to the mean). Verified in `ml/compare_gt.py`.
- **Emotion conditioning** (v2 model): each emotion diverges 0.54–1.41 rad
  from neutral on the same audio, with sensible signatures (sadness most
  divergent yet least mobile; anger most animated). `ml/eval_emotion.py`.
- **Base+additive**: no raw model pose reaches the skeleton; e2e asserts a
  motion-continuity guard (no >0.45 m/125 ms hand jumps).
- **Multi-game references**: FFXV (Luminous), FFXVI (Havok), BG3 (Granny)
  idle stances all extracted, rest-pose-corrected (arms 71–73°), and
  selectable as base layers. Cross-engine finding: all three store
  *conversational* animation as additive-over-base — independent validation
  of this architecture. Verified against the raw game clips: base_ff16
  reproduces FFXVI `talk_relax` within ~1 deg/s per joint.
- **Motion-style knob** — a single continuous slider (0..1) spanning
  game-faithful (calm) ↔ expressive ↔ game-feel (cinematic). It blends three
  things at once: additive weight (0.12→0.7→0.9 on a game base), handedness
  match (1→0, so the calm end rests one hand like the game and the loud end
  is symmetric), and a gesture-angle exaggeration (1→2× at the player, base
  posture untouched, capped <176° so nothing wraps). Measured composed hand
  motion on the FFXVI base sweeps smoothly and monotonically: 24/10 (0.0) →
  88/126 (0.35) → 151/195 (0.6) → 263/305 (1.0) deg/s. The sidecar still
  accepts the legacy `motion_style` preset and `game_faithful` bool.
  Each emotion has a default slider position (sad 0.12, calm 0.20, neutral
  0.35, happy 0.45, angry 0.55, excited 0.62) served from `/health`; the UI
  snaps the slider there on emotion change (the user then trims), and it's
  also the server-side fallback when a client sends no explicit style.
- **Game-faithful motion**: our co-speech mocap moves the hands
  ~4–6× more than real game dialogue (BEAT2 talk ~100–146 deg/s vs FFXVI
  `talk_relax` ~6–25). The `game_faithful` flag (UI checkbox → sidecar)
  mutes the BEAT2 additive to a whisper (0.7→0.12) and softens nods, so a
  game base's own motion carries the performance — composed hand motion
  drops ~2–3× toward the source game's calm style. It also biases the
  additive by the base clip's own handedness (measured per clip), so the
  resting hand stays resting. This reproduces each game's own gesture
  style, validated against the raw clips:
  - **FFXVI** (`base_ff16`, Clive): left-dominant. Composed L 26.3 / R 8.9
    deg/s vs raw `talk_relax` L 24.8 / R 6.1 — one-handed.
  - **BG3** (`base_bg3`): right-dominant. Composed lower-arm L 7.7 / R 17.5
    (R 2.3× L) vs raw BG3 idle L 5.1 / R 12.0 (R 2.4× L).
  - **FFXV** (`base_ffxv`): two-handed. Composed L 35.8 / R 42.4 —
    ~symmetric, matching the FFXV idle, which uses both arms.
- **All four e2e suites pass; `npm run build` clean.** (2026-07-21)

## How to run

```sh
npm install
npm run fetch:model          # demo characters (Xbot GLB, VRoid VRM)
npm run dev                  # reactive app at http://localhost:5173

# Generative "Speak" panel also needs the sidecar (see ml/README.md for setup):
npm run sidecar              # ml/.venv/bin/python ml/server.py, port 8600
```

Tests: `.claude/skills/run-app/SKILL.md` has the headless-browser harness.
Gotcha that bites: launch the test browser with `--disable-audio-output`
or its audio clock freezes and every audio assertion fails at once.

## Reproducibility & heavy artifacts (important)

Everything needed to build and run the **reactive** app is committed. The
**generative** app depends on artifacts that are gitignored or external:

| Artifact | Where | Status | If lost |
|---|---|---|---|
| Emotion model `emage_emotion` (~531 MB) | `ml/checkpoints/` (gitignored) | **Local + backed up** | Backup archive (verified, checksummed) at `/var/mnt/hdd/backups/live-animation/emage_emotion_v2_*.tar.gz` on a separate physical disk (`/dev/sda2`). Restore: `tar xzf <archive> -C ml/checkpoints/`. Or retrain via `ml/train_emotion.py` (hours). Sidecar auto-falls back to base EMAGE if absent. |
| Baked base/overlay clips | `public/anims/*.json` | **Committed** | Rebuild via `ml/bake_library.py` (mocap) or the game pipeline. |
| Learned accent model | `ml/checkpoints/scheduler/` | **Committed** (small) | — |
| Seed pose | `ml/seed_pose.npy` | **Committed** | — |
| Python env / EMAGE / Kokoro weights | `ml/.venv`, `ml/vendor`, HF cache | **Local**, reproducible | `ml/README.md` setup. |
| Game extraction scratch | `~/ffxv_work`, `/tmp/lslib-src`, `/tmp/bg3out*` (~2.6 GB) | **External, not reproducible without re-extracting** | The extraction pipelines are documented; game assets never enter the repo (licensing). The *derived base clips* are committed, so the runtime doesn't need them. |

Bottom line: a fresh clone runs the reactive app immediately, and the
generative app after `ml/README.md` setup + (optionally) retraining the
emotion model. The committed base clips mean the *look* survives even
without the checkpoint.

## Known limitations / honest notes

- **One big branch.** 30 commits of layered scope on a single worktree
  branch, all pushed to `main`. Coherent but large.
- **Scheduler is mostly rules.** `build_schedule` places accents by
  punctuation (text) or a weak learned prosody model (audio-only); the
  learned nod/shake predictor has low precision (~0.2) — honest limit of
  audio→gesture prediction, so it only drives the transcript-less path.
- **FFXV base is the least calm** of the game idles (the extracted packs
  are combat-oriented; ~23°/s vs FFXVI's 2.6). Rest-pose correction fixes
  arm posture but not busyness.
- **ROCm quirks** (documented in `ml/README.md`): fused SDPA backward has
  no gfx1201 kernel (training forces math SDPA); Kokoro's LSTM runs on CPU;
  the sidecar auto-falls back to CPU when VRAM is held by other services.

## Natural next steps (not started)

- Consolidate onto a clean feature branch / open a PR for review.
- Back up or shrink the emotion checkpoint (it's the one heavy
  irreplaceable-fast artifact).
- Strengthen the scheduler (a real learned gesture-timing model vs rules).
- Learn the per-emotion style defaults from the reference data instead of
  hand-set values.
