# Generative animation: research and architecture

Goal: a dynamic animation suite — given speech **text**, **emotion**, and/or
**audio**, generate lifelike full-body gesture + face animation sequences for
the humanoid rigs, with a locally trained/fine-tuned model (image-gen-style
generative modeling, applied to motion).

Research date: 2026-07-21 (three parallel web surveys; links inline).

## How the field works (it *is* image-gen logic)

Modern motion generation uses exactly the two architecture families from
image generation:

1. **Diffusion / flow matching over pose sequences** — denoise a
   `[frames × joints × rotation]` tensor conditioned on audio/text/style
   (MDM, DiffuseStyleGesture, MotionLCM 1-step distillation,
   GestureLSM flow matching).
2. **VQ-VAE + token generation** — compress motion into discrete codebook
   tokens, then generate token sequences with a (masked or autoregressive)
   transformer, like DALL-E/MaskGIT (T2M-GPT, MoMask, **EMAGE**).

Conditioning (audio features, text embeddings, emotion/style labels) enters
via cross-attention/AdaIN exactly as in image models; classifier-free
guidance and LoRA style adapters carry over too (LoRA-MDM trains a style
LoRA in ~4k steps on a consumer GPU).

## Candidate models (co-speech gesture — audio-driven, full body)

| System | License / weights | Notes |
|---|---|---|
| **EMAGE** ([PantoMatrix](https://github.com/PantoMatrix/PantoMatrix), CVPR'24) | **Apache-2.0**, [ungated HF weights](https://huggingface.co/H-Liu1997/emage_audio) ~613 MB | audio → full-body SMPL-X + FLAME face, 30 fps; masked-token (fast, no diffusion loop); plain PyTorch → ROCm OK; train/eval code included; **no emotion input** (that's our gap to fill) |
| [GestureLSM](https://github.com/andypinxinliu/GestureLSM) (ICCV'25) | MIT, HF weights | audio+text, flow matching, 1-step sampling (real-time-capable); MFA alignment setup cost |
| [DiffuseStyleGesture](https://github.com/YoungSeng/DiffuseStyleGesture) | MIT code | explicit style/emotion one-hots (ZEGGS styles) but slow DDPM; BVH out |
| [ZeroEGGS](https://github.com/ubisoft/ubisoft-laforge-ZeroEGGS) | CC BY-NC-ND | example-clip style transfer, CPU real-time, unmaintained |
| [AMUSE](https://amuse.is.tue.mpg.de) | MPI research, gated | only true emotion-disentangled model; A100-class, Blender in loop — impractical here |

## Candidate models (text/emotion → motion)

| System | License | Notes |
|---|---|---|
| **[MoMask](https://github.com/EricGuo5513/momask-codes)** (CVPR'24) | MIT | 45M params, 0.18 s/clip, best FID of classic set, explicit duration control, built-in joints→BVH IK; CPU feasible |
| **[MDM](https://github.com/GuyTevet/motion-diffusion-model)+[DiP](https://github.com/GuyTevet/motion-diffusion-model/blob/main/DiP.md)** + [LoRA-MDM](https://github.com/haimsaw/LoRA-MDM) | MIT | 18M base; DiP = 10-step autoregressive streaming; LoRA-MDM = cheap per-style (incl. emotional styles) fine-tuning — the proven local emotion path |
| [MotionLCM-V2](https://github.com/Dai-Wenxun/MotionLCM) | non-commercial | ~30 ms/clip (1-step); license taints it |
| [SMooDi](https://github.com/neu-vi/SMooDi) | MIT | text + style-reference-motion transfer; heavy guidance loop |

Emotion as a first-class input **barely exists** in released models — the
field does style transfer or prompt adverbs. Training explicit emotion
conditioning (BEAT2's 8 emotion labels; 100STYLE/Bandai emotional styles)
is where a custom model adds real value.

## Datasets

- **[BEAT2](https://huggingface.co/datasets/H-Liu1997/BEAT2)** — 60 h,
  25 speakers, SMPL-X+FLAME 30 fps, **8 emotion labels + per-frame semantic
  relevancy**, plain `git lfs` download, no registration. Our training set.
- [100STYLE](https://ianxmason.github.io/100style/) / [Bandai-Namco](https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset)
  / [Kinematic actors emotions](https://physionet.org/content/kinematic-actors-emotions/2.1.0/)
  — stylized/emotional locomotion & acting clips (BVH), for emotion-style work.
- HumanML3D must be rebuilt from AMASS (MPI registration, non-commercial) —
  avoid depending on it.

## Local hardware (verified)

- AMD Radeon RX 9070 XT (gfx1201/RDNA4), 62 GB RAM, 16 cores.
- PyTorch **2.9.1+rocm6.4** in `ml/.venv` (uv, Python 3.12): GPU visible,
  **7.0 TFLOPS fp32** measured, transformer layer fwd+bwd OK.
- Verdict: pretrained inference trivially fine; training/fine-tuning
  10–100M-param motion models realistic (originals trained on 2080Ti/3090).
- Flash-attention custom kernels are the one ROCm risk — stick to plain
  SDPA models (all recommended models qualify). Avoid Mamba-kernel models
  (Light-T2M, MambaTalk).

## Chosen architecture

```
            text + emotion                      audio file/mic
                 │                                   │
     [Kokoro TTS: wav + word/phoneme timestamps]     │
                 │                                   │
                 ▼                                   ▼
   Python sidecar (ml/):  gesture/motion model (EMAGE now; our
   emotion-conditioned fine-tune next) → SMPL-X pose sequence
                 │
                 ▼
   convert: accumulate local→world rotations along the SMPL chain
   (SMPL rest frames are world-aligned ⇒ world rotation IS the
   T-pose delta) → MotionClip JSON {fps, joints, world-delta quats,
   hips translation}
                 │
                 ▼
   web app: ClipPlayer (src/animation/clip.ts) retargets deltas onto
   any rig via probed/captured tposeWorld; face/lip-sync + foot IK
   layers stay procedural
```

Decisions and reasons:

- **Python sidecar over in-browser ONNX** — zero prior art for motion
  models in ORT-web; ROCm PyTorch is the robust path; ONNX Runtime's ROCm
  EP is deprecated (MIGraphX). Revisit ONNX only for a distilled 1-step
  model later.
- **MotionClip world-delta format over BVH/.vrma** — one converter on the
  Python side, one player on the web side, works on *all three* rigs (not
  just VRM), and avoids three.js `SkeletonUtils.retarget` pitfalls.
  Verified by `e2e/clip-test.cjs`: identical world-space T-pose and wave
  geometry on capsule/GLB/VRM.
- **EMAGE first** (only zero-friction pretrained full-body co-speech
  model: Apache-2.0, ungated, BEAT2-native), then **train emotion
  conditioning on BEAT2** — either an emotion-embedding fine-tune of
  EMAGE's transformer or a compact VQ+masked-transformer of our own.
- **SMPL-X → our joints**: pelvis→hips, spine1/2/3→spine/chest/(chest),
  neck→neck, head→head, collars→shoulders, shoulders→upperArms,
  elbows→lowerArms, wrists→hands, hips→upperLegs, knees→lowerLegs,
  ankles→feet (verified against smplx `joint_names.py`).
- Known SMPL pitfalls to handle in the converter: AMASS-style Z-up
  `global_orient` (bake −90° X if present), pelvis offset `trans + J0(β)`,
  hips-height ratio scaling for root motion.

## Roadmap

1. ~~ClipPlayer + canonical clip format + cross-rig e2e~~ (done, fa9a77a)
2. EMAGE inference locally: speech WAV → SMPL-X → MotionClip → plays in
   browser on VRM (e2e-verified).
3. TTS leg: Kokoro (Apache-2.0, 82M, CPU real-time, native word timestamps)
   → text+emotion input produces audio + lip-sync-aligned gestures.
4. Training: BEAT2 subset → emotion-conditioned gesture model on the 9070 XT
   (fine-tune EMAGE or train compact VQ+transformer; LoRA-style emotion
   adapters as the cheap fallback).
5. Facial channel: EMAGE's FLAME face output → VRM expressions mapping.
