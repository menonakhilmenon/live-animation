# Generating humanoid animation from natural language — research synthesis

**Compiled 2026-08-12.** Five parallel literature/tooling surveys, verified against live sources.
Target context: solo developer, Linux (Fedora), AMD RX 9070 XT (gfx1201, 16 GB, ROCm),
accuracy and realism prioritized over latency (~1 min generation budget acceptable).

---

## 1. The short answer

The intuitive framing — "text-to-image, but for motion" — is the wrong mental model, and the
reason is not architectural but informational.

Text-to-image works because the web supplied billions of image–caption pairs. **Motion has no
such corpus and cannot get one.** At 30 fps with the standard 4× temporal downsampling, motion
yields ~7.5 tokens/second, so:

| Corpus | Hours | ≈ Tokens |
|---|---|---|
| HumanML3D (the benchmark everyone reports on) | 28.6 | ~0.76 M |
| MotionMillion (largest fully released) | 2,000 | ~54 M |
| HY-Motion internal (Tencent) | 3,000 | ~81 M |
| AnyMo OmniHuMo (largest claimed) | 5,000+ | ~135 M |
| *Llama-3 text pretraining, for scale* | — | *1.5 × 10¹³* |

**The largest motion corpus on Earth is under 10⁸ tokens — five orders of magnitude below
language pretraining.** People are training 7B-parameter models on ~5×10⁷ tokens, a
params:tokens ratio near 100:1 where Chinchilla wants ~1:20.

Four independent scaling studies converge on the same consequence:

- **ScaMo** (CVPR'25) fits a real scaling law (`L = −1.062·log₁₀(C) + 13.839`) and finds
  vocabulary must scale *faster* than the model (`N_v ∝ N_nv^1.467`, R²=0.95).
- **MotionMillion**: FID 31.3 (1B) → 10.8 (3B) → **10.3 (7B)**, R-Precision flat across all three.
  3B→7B buys essentially nothing for 2.3× the parameters.
- **Being-M0**: 7B→13B lifts R@1 by ~1 point; 0.5M→1.2M data lifts it by ~1 point.
- **HY-Motion** (Tencent, 38 authors, 3,000 h) states outright that instruction-following improves
  with size but **motion quality plateaus beyond 0.46B** — and they shipped a 1B model, not a 7B one.
  That is a revealed preference from the best-resourced team in the field.

So: **do not plan to train a motion LLM.** The leverage is elsewhere.

The architecture the evidence actually supports — independently converged on by Sony (SIGGRAPH
2025, for anime characters), PKU (SIGGRAPH Asia 2025, dyadic conversation), and MPI-INF
(CVPR 2026) — is:

> **LLM as *director*, never as *animator*.**
> Free-form reasoning → constrained emission of a small, bounded, semantically-named control
> space (beats + durations + gesture/skill IDs + root trajectory + style parameters) → a trained
> motion model or retrieval layer densifies it into poses → a **deterministic external verifier**
> (joint limits, continuity, foot contact, penetration, IK reachability) rejects and repairs →
> procedural layers own blink, saccade, breathing, weight shift.

---

## 2. What LLMs demonstrably win at, and where they fail

This is the best-measured part of the whole survey, and the numbers are unambiguous.

### 2.1 The direct measurement

**"How Much Do LLMs Know about Human Motion? A Case Study in 3D Avatar Control"** —
[arXiv:2505.21531](https://arxiv.org/abs/2505.21531). LLM plans steps → specifies body-part
positions → linear interpolation → human scoring (κ=0.74 / 0.531), 20 instructions, 5 models.

| Capability | Result |
|---|---|
| **High-level planning** (HPS/5.0) | Claude 3.5 Sonnet & GPT-4o **4.57–4.68**, Llama-3.1-70B 4.07 |
| **Body Part Position Accuracy** | Claude 3.5 Sonnet **73.52%**, GPT-4o 70.87%, Llama-3.1-70B 52.60% |
| **Whole Body Score** | Claude **3.29/5.0** vs. oracle annotations at 4.57/5.0 |

Documented failure modes, in the authors' framing:

1. **Joint rotations fail in both direction and magnitude** — not just imprecise, wrong sign.
2. **Error scales with degrees of freedom** — upper arms/legs systematically worse than
   constrained joints.
3. **Errors compound across body parts** — one example scored 78.12% BPPA yet crossed the arms
   and failed to toss the ball. Per-joint accuracy does not compose into a correct pose.
4. **Multi-step degradation** — spatial relations drift; implicit auxiliary adjustments are missed.
5. **Root translation** direction roughly right, magnitudes unreasonable.
6. **Timing quantizes to whole and half seconds.** This is the mechanical root of "uncanny
   timing" — easing, anticipation, overlap, follow-through and weight shift all live *below*
   that resolution. Measured, not folklore.

### 2.2 Corroborating evidence

- **Re²MoGen** ([2604.17807](https://arxiv.org/abs/2604.17807)): raw LLM-emitted keyframes measured
  at **floating 16.85 mm, penetration 32.84 mm**. After full RL repair in Isaac Gym: 2.46 mm and
  21.70 mm. Even after physics repair, penetration only improves ~34%.
- **Precision is capped by tokenization.** The one system that had GPT-4 emit raw quaternions
  ([2310.17838](https://arxiv.org/abs/2310.17838), NeurIPS'23 workshop) had to **truncate every
  float to a single significant figure** to fit context — ~0.1 precision on unit quaternion
  components, enough to visibly break a pose.
- **The retreat is documented.** Normoyle, Sedoc & Durupinar tried LLM control of body motion and
  gave up in print: *"generating motion from language is not as straightforward as tuning AU
  intensities. So, in this work, instead of having GPT-3.5 control the content of motion, we ask
  it to infer the **style** of a given motion via text"* — falling back to 4 Laban Effort floats.
  Their LLM viseme generation was also abandoned for an off-the-shelf lip-sync plugin.
- **"Executes" ≠ "correct."** PRISM measures an **Execution-Spatial Gap of ~41%**; P3D-Bench finds
  executable validity >99% and semantic alignment ~0.93 but **geometric alignment only ~0.35**.
- **The easiest possible case still needs ~16 prompts.** Apple's **Keyframer** study
  ([2402.06071](https://arxiv.org/abs/2402.06071)) had GPT-4 generate **CSS animations for SVGs** —
  2D, no rig, no physics, no contacts. 13 participants averaged **15.8 prompts each, 48.8% of them
  repairs**. Dominant semantic failure was *grouping* ("animate the sparkles" moved three sparkles
  as one rigid body) — exactly the bone-vs-bone-chain error class a humanoid rig would surface.
  Scale that to 60 bones with contacts.

### 2.3 Where LLMs genuinely win

1. **Decomposition and long-horizon structure.** The one unambiguous win. HPS 4.57–4.68/5, and
   LaMoGen's ablation quantifies the downstream benefit: semantic alignment **0.583 with GPT-4.1
   guidance vs 0.523 with the same decoder and no LLM**.
2. **Anything symbolic, discrete, or nameable.** Clip selection, camera/staging tool calls,
   blend-shape and FACS parameters, LMA style descriptors, root trajectory *as a closed-form
   math function* (CoMA's trick). Kuaishou's Cutscene Agent hits **100% tool-selection accuracy**
   with Claude Opus 4.6 because the action space is named and finite.
3. **Reward and constraint authoring for offline skill training.** GROVE (CVPR'25 Oral) is real —
   naturalness 6.79 vs 5.94 for AnySkill, 7 min/skill vs 59 — but it's an *authoring* tool, not a
   runtime path; each candidate costs a full RL run.
4. **Editing, conversation, and intent repair.** Multi-turn refinement is a natural LLM strength
   and an awkward fit for one-shot diffusion.
5. **Open vocabulary as a reranker.** TEXEDO's **0.873 → 0.984 success at N=32** is the cleanest
   evidence, and it's the cheapest pattern to reproduce.

### 2.4 Structured-output engineering — one cost that survives

If the LLM emits JSON, budget for **diversity collapse**. A 44-model study
([2607.18476](https://arxiv.org/abs/2607.18476)) measures: mean answer surprisal **1.80 → 1.58 bits**
(p=.0002), modal-answer share **41% → 64%**, distinct answers 52 → 36. Format-specific: JSON −0.22,
XML −0.19, YAML/CSV ≈0, a bare bracket wrapper **+0.13**. **Two-pass does not fix it.** For this
project that means a JSON keyframe/beat schema will homogenize toward a modal generic walk/wave
even when every sample is schema-valid.

The accuracy question is more nuanced and now largely resolved: the original "Let Me Speak Freely?"
result (Claude-3-Haiku GSM8K 86.5% → 23.4% under JSON mode) was substantially a prompt artifact —
under JSON mode **100% of GPT-3.5 responses placed the answer before the reasoning**, destroying
chain-of-thought. CRANE (ICML'25) supplies the theory (a fully-constrained constant-depth LLM
collapses to TC⁰ — no room for intermediate reasoning tokens), and "The Format Tax" (2026) the
resolution: **the format-requesting instruction causes most of the loss, before any decoder mask
applies.** Three independent lines therefore agree on the same fix: **reason free-form, then emit
constrained.** Never constrain the reasoning itself.

Engine choice: **llguidance** (~50 µs/token, powers OpenAI Structured Outputs, vLLM ≥0.8.2,
llama.cpp) or **XGrammar** — but note JSONSchemaBench found XGrammar has **38 under-constrained
failures**, silently permitting invalid output on some schema features. If the beat schema uses
`minItems`/`maxItems`/numeric bounds, verify enforcement.

---

## 3. Recommended architecture

```
natural language
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ DIRECTOR  (frozen general LLM — no fine-tuning)             │
│  free-form reasoning, THEN constrained emission of:         │
│    · beats: [{intent, duration_s, style, gesture_tag}]      │
│    · root trajectory as a closed-form function              │
│    · emotion / LMA effort parameters                        │
│    · gaze targets, emphasis markers                         │
│  NEVER: joint angles, quaternions, per-frame keyframes      │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ PERFORMER  (per-beat, parallel)                             │
│  · body:    trained text-to-motion model (§4)               │
│  · gesture: retrieval from a curated semantic library       │
│  · face:    ARKit-52 native model (§6.1)                    │
│  · hands:   retrieval / refinement, NOT generation (§6.2)   │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ CRITIC  ← the highest-leverage unbuilt component (§5)       │
│  N candidates → deterministic verifier → rank → repair      │
│  joint limits · C¹ continuity · foot skate · penetration    │
│  · IK reachability · physics rollout feasibility            │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ COMPOSITOR                                                  │
│  transition repair · physics cleanup (Morph) · procedural   │
│  layers: breathing, sway, blink, saccade, micro-saccade     │
│  → VRM humanoid bones + expression weights → .vrma          │
└─────────────────────────────────────────────────────────────┘
```

Two structural notes:

- **Every planner system in the literature ships a dedicated transition-repair module**
  (Story-to-Motion's progressive mask transformer for "unnatural pose and foot sliding",
  Motion-Agent's "Smoothening Transition"). The LLM cannot produce the blend. Budget for it.
- **Separate the hand latent from the body latent.** This recurs in everything that works —
  EMAGE (4 separate VQ-VAEs), HumanTOMATO (H²VQ two codebooks), FUSION, DigitCode. A shared
  codebook loses the hands, because body motion dominates the reconstruction loss.

---

## 4. Body motion: model selection

### 4.1 Read this before any leaderboard

**HumanML3D FID is saturated and the benchmark is partly broken.** Top methods cluster at
0.030–0.076, below the noise floor of a 28-hour dataset. Worse, **three mutually incomparable
evaluation protocols are now in circulation:**

| Protocol | Evaluator | Used by |
|---|---|---|
| **A. Classic** (Guo et al., 263-dim) | standard | MDM, MoMask, MMM, T2M-GPT, MoGenTS, SALAD, ScaleMoGen… |
| **B. MARDM "essential features"** (67-dim) | retrained, no Diversity metric | MARDM, ACMDM, MoRAE |
| **C. 272-dim TMR** | TMR retrained; FID on a different scale entirely | MotionStreamer, LLaMo, UMO |

**MoMask self-reports FID 0.045 under A; scores 0.116 under B; scores 12.232 under C.** Same
weights, same model. MotionStreamer's headline 11.790 is *not* 250× worse than SALAD's 0.076.

Further: several 2026 papers report **R-Precision above ground-truth motion** (real = 0.797;
SALAD 0.857, ScaleMoGen 0.856, MoGeFlow 0.873). That is a protocol artifact, not a model that
beats reality.

Two more criticisms worth internalizing:

- **MARDM** (CVPR'25) showed HumanML3D's 263-dim vector has 7 feature groups but only the first 4
  are needed to reconstruct motion. The redundant dims help VQ training and *hurt* diffusion — and
  the standard evaluator over-weights exactly those dims, systematically penalizing diffusion.
- **Physical plausibility is invisible to FID.** Models score well while sliding, floating and
  self-penetrating. Adopt **PP-Motion** ([2508.08179](https://arxiv.org/abs/2508.08179)) or
  **T2MBench**'s seven physical metrics early; optimizing skate/float/penetrate alone overfits to
  metrics that don't track perception.

**Practical tip:** [MoLingo](https://github.com/hynann/MoLingo) (Apache-2.0) is the only repo that
ships **all three evaluators**. Start there for any honest comparison.

### 4.2 The picks

**Tier 1 — start here**

| Pick | Why | License |
|---|---|---|
| **HY-Motion 1.0-Lite (0.46B)** [repo](https://github.com/Tencent-Hunyuan/HY-Motion-1.0) | The only model producing animation a person would accept, and **the only one exporting FBX/GLB directly**. 3,000 h training → generalizes well beyond HumanML3D vocabulary. Abandoned FID for human eval: instruction-following **3.24** vs competitors' 2.17–2.31, quality **3.43** vs 2.79–3.11, SSAE **78.6%** vs 42.7–58.0%. | Tencent Community — commercial OK, **excludes EU/UK/South Korea**, 1M MAU cap |
| **MoLingo** (CVPR'26) | Best-engineered research codebase found. Apache-2.0, weights for 263-d *and* 272-d, ships all three evaluators, demo on a single 3090. | **Apache-2.0** |
| **MoMask** (CVPR'24) | Still the right baseline economically: ~50M params, trains on one consumer GPU, ~40–70 ms inference, HF demo runs on **CPU**. Every discrete-token model since is a MoMask fork. | **MIT** |

**HY-Motion VRAM, correctly understood:** the quoted 24–26 GB is dominated by the **Qwen3-8B text
encoder**, not the DiT. The [ComfyUI node](https://github.com/jtydhr88/ComfyUI-HY-Motion1) shows
the 1B DiT needs ~8 GB and Lite ~4 GB; Qwen3-8B drops to ~3–5 GB at 4-bit or zero with CPU offload.
**It fits in 16 GB.** A community Kaggle notebook runs it on a free-tier P100.

Its stated limits: no multi-person, no non-humanoid, no environment/camera/object conditioning,
**no complex emotion**, no seamless-loop or in-place modes, English prompts under 60 words, <5 s
clips at 16 GB.

**Tier 2 — specific needs**

- **Spatial control:** **MaskControl** (ICCV'25 Oral) — FID **0.061**, trajectory error **0.00**,
  average error **0.98 cm** vs MDM's 59.59 cm. Fast variant 4.94 s, Accurate 71.72 s (your budget
  affords Accurate). ⚠️ CC-BY-NC-ND-4.0 *plus* inherited licenses from five upstream projects.
- **Multi-sentence composition:** **FlowMDM** (CVPR'24) — transition metrics AUJ **0.51** vs
  DoubleTake's 2.10 on HumanML3D, i.e. transitions stop reading as cuts. CC BY-NC-SA.
- **Training-free clause fixing:** **MultiAct** (SIGGRAPH 2026,
  [2605.30925](https://arxiv.org/abs/2605.30925)) — inference-time only, no retraining, amplifies
  cross-attention for under-represented prompt components. Directly targets the "walk to the door,
  then turn and **wave**" dropped-clause failure. Very high leverage if released.
- **Fine-tuning on own data:** **MotionMillion** — the only genuinely open large model **and**
  dataset pair (Apache-2.0, 3B + 7B weights, 2M sequences). Given the saturation evidence,
  fine-tune the 3B.

**Explicitly avoid:** MDM/MotionDiffuse as a starting point (1000 steps, FID 0.5+, every technique
superseded — read MDM for the x₀-prediction insight, then move on). MMM/BAMM if this ever ships
(CC-BY-NC-**ND** blocks even distributing modifications). MotionLCM commercially (non-commercial
*and* forces open-sourcing your changes).

### 4.3 Architectural bet, if building rather than using

**Flow matching over a structured latent, not DDPM over the 263-dim vector.** The evidence is
converging and unambiguous: FlowCoMotion measured a **15× speedup from swapping diffusion for flow
at identical representation** (0.52 s vs 7.66 s); ACMDM's author notes flow *"converges so much
faster"*; HY-Motion, ViMoGen, MotionHiFlow, MoGeFlow, MoLingo and MoRAE all chose flow.

Pair it with **SMPL-native coordinates (272-d), not the 263-dim redundant vector** (§7.2).

Second bet: **don't train from scratch.** **UMO** ([2603.15975](https://arxiv.org/abs/2603.15975),
MIT/Brown/Meta/MPI-IS) freezes a pretrained HY-Motion and unlocks text-to-motion, inpainting,
instruction editing, geometric constraints and two-person reaction through in-context learning with
a **0.207M-parameter adapter** — and the unified model *beats the per-task experts*. That is the
strongest evidence a genuine foundation-model prior exists in motion, and it is precisely the shape
of work one person with one GPU can execute.

---

## 5. The highest-leverage unbuilt component: a motion verifier

**No motion linter exists.** arXiv full-text search returns **0 results** for `"motion linter"` and
`"animation linter"`; **0** for LLM + rigging/skeleton/armature validation; **0** for
VLM-as-verifier-for-animation. No published system validates generated animation against joint
limits + bone-hierarchy validity + temporal continuity + IK solvability as a unified check.

This matters because the self-repair literature is clear that **an external oracle beats LLM
self-critique**. "Is Self-Repair a Silver Bullet?" (ICLR 2024) finds repair is bottlenecked by the
model's ability to critique itself; a stronger feedback model helps far more, and human feedback
beats all self-feedback. Meanwhile BlenderGym measured the best VLM verifier at only **66% agreement
with human judgment vs 79% inter-human** — and its blend-shape task, the closest thing to animation
in any benchmark, is where the human/model gap is **widest, at ~10×**.

The pieces exist unassembled:

- **Joint limits** — Herda, Urtasun & Fua (CVIU 2005), hierarchical implicit-surface swing/twist
  limits on coupled joints. Classical, pre-LLM, still correct.
- **IK solvability as an accept/reject oracle** — ModuLoop ([2606.03047](https://arxiv.org/abs/2606.03047))
  is the only LLM-loop system doing this, and it's robotics.
- **Physical metrics** — T2MBench ([2602.13751](https://arxiv.org/abs/2602.13751)) defines the seven
  a linter would implement: Jitter Degree, Ground Penetration, Foot Floating (5 mm tolerance), Foot
  Sliding, Skate, PFC, Body Penetration.
- **Symbolic pre-execution** — SGA ([2607.18116](https://arxiv.org/abs/2607.18116)) partially executes
  LLM-generated Manim code to extract a scene graph and detect spatial conflicts **without
  rendering**. The closest existing thing to a graphics linter; +16.1% relative.

**This is where the ~1 minute latency budget converts directly into quality**, and it is the one
component where the field has an actual hole rather than a crowded field.

### 5.1 Test-time scaling: the cheapest large win

**TEXEDO** ([2606.22998](https://arxiv.org/abs/2606.22998), code + HF weights **released**) is the
reference pattern: sample N candidates from a *frozen* generator, then rank with two verifiers — a
**dynamic feasibility** verifier trained on whole-body tracking rollouts, and a **semantic
alignment** verifier in a learned text-motion embedding space. **Success 0.873 → 0.984 at N=32**,
Empjpe 39.09 vs 44.34. Validated in sim and on a real Unitree G1.

Its framing is exactly right for this project: generators trained on retargeted human data produce
motion that is *"semantically plausible but difficult or impossible to execute,"* because they know
nothing about balance, contact dynamics, actuation limits, or controller failure modes.

Arithmetic for this box: MaskControl-Fast is 4.94 s, so **N=8 with reranking costs ~40 s** — well
inside budget. This improves *both* physics and text alignment with **no better generator**.

Related: **ICMPG** ([2606.26981](https://arxiv.org/abs/2606.26981)) — LLM planner proposes, physics
simulation + semantic alignment selects, closed-loop refinement entirely at inference time, no
retraining. HumanML3D FID 0.116, R@1 0.491; BABEL open-vocab success 82.9%. ~10 s/sequence.

---

## 6. Physics, expressivity, and the parts that are separate problems

### 6.0 Physics cleanup — largely solved off the shelf

**Morph** ([2411.14951](https://arxiv.org/abs/2411.14951), ICCV 2025) is the direct answer to
"physically clean up kinematic motion." Code **and weights**, **Apache-2.0**, at
[WeChatCV/Morph](https://github.com/WeChatCV/Morph). Plugs into MDM / MotionDiffuse / T2M-GPT /
MoMask without retraining them.

| | Penetrate ↓ | Float ↓ | Skate ↓ | FID ↓ | R-Top3 ↑ |
|---|---|---|---|---|---|
| Morph-MDM | **0.000** | 2.258 | 0.016 | 0.482 | 0.757 |
| Morph-T2M-GPT | **0.000** | 2.700 | 0.039 | 0.105 | 0.784 |
| Morph-MoMask | **0.000** | 2.141 | **0.010** | **0.041** | **0.816** |

Penetration goes to *exactly* zero in every case, and FID and R-precision **improve**. It's
motion-free — trained entirely on synthetic noisy motion from pretrained generators, and ablations
show synthetic-only beats real data here because of domain alignment. Cost: Isaac Gym install, and
the repo is thin on applying refinement to your own clips.

Alternatives: **CLoSD** (ICLR'25 Spotlight, code + auto-downloading checkpoints, ~4 GB VRAM
inference, penetration 0.022 mm, skating 2×10⁻³) — but its task repertoire is navigate/strike/sit/
get-up chained by a state machine, not open vocabulary. **ProtoMotions3** (NVIDIA, **Apache-2.0**)
ships a text-prompt → physics-policy pipeline via Kimodo, but no text-conditioned pretrained policy
is listed and all trackers are IsaacLab-only.

### 6.1 Face — genuinely solved, if you pick the right camp

**The field splits into two incompatible output camps, and this decides everything:**

| Camp | Emits | Who | Usable on a VRM? |
|---|---|---|---|
| **Academic** | FLAME params or per-vertex offsets on fixed topology | VOCA, FaceFormer, CodeTalker, EMOTE, ProbTalk3D, ~90% of arXiv | **No** — needs a lossy solve |
| **Production** | **ARKit 52 blendshape weights** | NVIDIA Audio2Face-3D, Azure TTS, SAiD | **Yes**, near-directly |

**There is no maintained FLAME→ARKit converter.** GitHub search returns zero; FLAME-Universe (the
canonical index) lists none; PantoMatrix has **three open issues** asking for exactly this,
discussing ad-hoc transformation matrices. The conversion is lossy by construction — FLAME's ~100-d
*PCA* basis (global, signed, unbounded) → ARKit's 52 *semantic, non-negative, [0,1]-bounded,
locally-supported* shapes. No exact linear map exists. And FLAME is fit to scanned humans while a
VRM is a stylized character with non-anatomical proportions: doubly ill-posed.

**So: use ARKit-native sources only.**

- **SAiD** ([2401.08655](https://arxiv.org/abs/2401.08655)) — **Apache-2.0**, weights on HF, emits
  blendshape coefficients as CSV against a bundled ARKit reference. The practical pick.
- **Azure TTS visemes** — zero-ML option. `FacialExpression` mode emits **55 floats @ 60 FPS where
  channels 1–52 are exactly the ARKit names**, plus headRoll/leftEyeRoll/rightEyeRoll. Free tier
  0.5M chars/month. A table lookup, not a solve. ⚠️ No offline path (embedded NTTS explicitly does
  not support it).
- **NVIDIA Audio2Face-3D** — open-sourced Sept 2025, SDK/plugins **MIT**, training framework
  **Apache-2.0**, weights under NVIDIA Open Model License permitting commercial use. Best quality
  available. ⚠️ **SDK requires TensorRT ≥10.13 + CUDA ≥12.8, "no non-NVIDIA GPU option"** — dead on
  ROCm. **Worth 30 minutes:** check whether the Apache-2.0 training framework's Python inference
  path (`.npy` output) is plain PyTorch and therefore ROCm-viable. High payoff.

**VRM reality:** VRM 1.0 defines exactly **19 preset expressions** — 5 vowels (`aa/ih/ou/ee/oh`),
6 emotions, 3 blink, 4 gaze. **No consonant shapes at all** — no lip closure for /m/, /b/, /p/, no
labiodental for /f/, /v/. That impoverishment is precisely why the community "Perfect Sync"
convention exists: ARKit-52 bolted into `expressions.custom`, giving you `mouthClose`, `mouthFunnel`,
`mouthPucker`, `mouthRoll*`, `mouthPress*` — the shapes that make consonants readable.

⚠️ **Architectural fork, not a gradient:** VSeeFace's docs state Perfect Sync *"bypasses traditional
VRM expression blendshapes entirely."* It is ARKit-52 **or** VRM presets, not gracefully both. Decide
early. Note VRoid Studio does **not** export ARKit blendshapes by default.

**Steal this VRM design detail regardless of engine:** the spec classifies blink and gaze as
*procedural* and provides **`overrideBlink`** with three modes (`none` / `block` / `blend`) so
emotional expressions can suppress procedural ones. Motivating example from the spec: *"Close eyes
`sad` and `blink` are applied ⇒ eyes close twice and eyelids pierce cheeks."* That is the exact bug
every hand-rolled procedural blink system ships with.

**For a stylized/anime character specifically:** Sony's SIGGRAPH 2025 poster
([2506.16159](https://arxiv.org/abs/2506.16159)) is the most directly copyable architecture found —
and it is deliberately *not* end-to-end neural. Gesture: phrase → SentenceBERT → retrieve from a
500-instance library → blend/time-stretch. Face: blendshape values mined from **manga (Manga109)**
via SD Tagger + anime face landmarks + a multimodal LLM → **>10,000 data points automatically**,
~50% containing anime-specific exaggerations (sweat, blush, stylized eyes) → ~130 emotion
categories → LLM infers emotion from dialogue → nearest by cosine similarity. ⚠️ No code released.

### 6.2 Hands — the weak link, and the honest answer is retrieval

**No open-weights text-to-motion model produces believable, semantically-correct finger motion.**
The failure is structural, with three compounding causes:

1. **HumanML3D is 22 joints, body-only.** MDM, MotionDiffuse, MLD, T2M-GPT, MoMask, MotionLCM and
   OmniControl all inherit it and *physically cannot* emit a finger pose.
2. **Where hands exist, they are mostly pseudo-labels from monocular video.** Motion-X's hands come
   from HaMeR + SMPL-X fitting; only EgoBody and GRAB carry genuine hand ground truth, and AMASS has
   *"roughly static hand motions."* The authors justify it: *"text-driven motion generation ... has
   a higher tolerance of motion capture error."* **That tolerance argument is exactly why generated
   hands look wrong.**
3. **Even hand-aware models regress to the mean** — finger motion is high-frequency, low-variance,
   and unpenalized by every standard metric.

The 2025 survey states it plainly: *"generating motions where only fingers move while the rest of
the body remains stationary remains an unsolved problem."*

**Where good finger motion actually is:**

- **Semantic Gesticulator's SeG library** — **MIT**, **208 semantic gesture types, 544 mocap BVH
  files** with real finger motion, semantics-indexed. Ships RVQ-VAE + GPT-2 generators + a Qwen-2.5-7B
  retriever. **By license and content, the best open finger resource found.** If nothing is ever
  trained, this alone is a usable phrase→gesture asset base.
- **HandX** (CVPR 2026, [2603.28766](https://arxiv.org/abs/2603.28766)) — genuine step change for
  bimanual: real mocap, contact/flexion-aware LLM captions, hand-specific metrics, code + data
  released. ⚠️ Hands-only, no body; MANO non-commercial.
- **Text2HOI** (CVPR'24) — **MIT**, checkpoints available, MANO output. The only permissively-licensed
  weights-available finger generator found. Hand-object grasp scope.
- **GRIP** / **MOCHI** — hand *refiners* that take bad or absent hands and add plausible ones from
  spatial cues. Conceptually the right tool.

**Verdict: generate the body from text, drive hands from a separate specialized source.** Do not
expect one model to do both. BEAT2 is instructive here — it was cut **76h → 60h by deleting 5
speakers for low finger quality**. Even purpose-built finger mocap has a ~20% junk rate.

### 6.3 Gesture — good motion, unreliable meaning

The grounding result for this entire area, and it should temper expectations:

**"Towards Reliable Human Evaluations in Gesture Generation"**
([2511.01233](https://arxiv.org/abs/2511.01233), CVPR Findings 2026), from the GENEA collaboration:

| System | Motion-realism Elo | Speech-gesture alignment |
|---|---|---|
| **Mocap (ground truth)** | **1133** | **~74%** |
| ConvoFusion | 1102 | ~50% |
| RAG-Gesture | 1088 | ~50% |
| HoloGest | 1084 | ~60% |
| Semantic Gesticulator | 1070 | ~57% |
| AMUSE | 824 | ~50% |
| DiffuseStyleGesture | 701 | ~60% |

Two conclusions: BEAT2 is **saturated for realism** (four models project 41–46% win rates against
mocap), and for the weaker models *"the motion they generate is only as valid for their input speech
as for randomly chosen speech segments"* — **50% is chance**. Several published models with strong
self-reported alignment are, under a correct protocol, **not actually synchronized to the speech**.

**You will get plausible beat gestures. You will not get a model that reliably points when the
character says "over there."** That is precisely why the whole 2026 literature — RAG-Gesture,
Semantic Gesticulator, SemConFlow, DuoGesture, MIBURI — converged on **retrieval + LLM direction**
rather than end-to-end generation.

Best options: **SynTalker** (**MIT**, weights, Colab, takes **audio + a free-text control prompt** —
natural fit for LLM direction, but no face); **MIBURI** (CVPR 2026, the only runnable online
body+hands+face system, SMPL-X 25 fps, code + 1.7 GB checkpoints — ⚠️ **CC BY-NC 4.0**);
**Gelina** (ICASSP 2026 oral, text → **speech + gesture jointly**, code + checkpoints — ⚠️ no
license file found).

The 4th GENEA Challenge results (on Meta's 4,000-hour Seamless Interaction dataset, first with a
dedicated dyadic-alignment task) land at ECCV, **8–9 Sept 2026**. Worth a follow-up in ~4 weeks.

### 6.4 Idle — newly solved, and commercially clean

**StayStill** ([2605.13693](https://arxiv.org/abs/2605.13693), SCA 2026 / CGF) is three months old
and is the best thing to happen to this problem.

- ~6 hours, **50 subjects, ~650,000 frames, 1,634 clips**, 30 fps
- Three categories: general idle, phone-use idle, and **18 named idle actions** (looking around,
  checking watch/phone, scratching head/arm/leg/back, touching face, stretching, rubbing eyes,
  yawning, **balance shifts**)
- **Data CC BY 4.0 — commercially usable. Code MIT.** BVH in two skeletons (FreeMoCap and LaFAN1)
- Ships pretrained in-betweening baseline checkpoints

⚠️ Its stated limits: markerless capture yields *"foot sliding, self-penetration and jittery
motion"*, and **fingers were deleted** because detection quality was too low.

**The gap this leaves:** the only idle dataset has no fingers, and every hand dataset has no idle.
**Nothing covers idle hands.** That layer must be built — likely procedural noise on finger curl
plus occasional retrieved fidget poses.

**Concrete procedural parameters** (shipped defaults, tuned for MetaHumans, from the Runtime
MetaHuman Lip Sync plugin — good starting points regardless of engine):

| Auto-blink | Value | | Micro-saccade | Value |
|---|---|---|---|---|
| Min interval | 2.0 s | | Yaw amplitude | 1.5° |
| Max interval | 4.0 s | | Pitch amplitude | 1.0° |
| Min closed hold | 0.04 s | | Min interval | 0.6 s |
| Max closed hold | 0.08 s | | Max interval | 2.5 s |
| Close interp speed | 35.0 | | | |
| Open interp speed | **18.0** (deliberately slower) | | | |

2–4 s ⇒ ~15–30 blinks/min, consistent with the literature's 15–20/min. But **blink rate is
strongly task-dependent and a fixed timer is a tell**: ~10/min while reading, ~12/min baseline,
~20/min under cognitive load. Modulate by character state.

For gaze, copy **Pejsa, Andrist, Gleicher & Mutlu (ACM TiiS 2015)** — it implements Eyes Alive's
statistical idle saccades *inside* a neurophysiologically-grounded gaze-shift model with the
**eye→head→torso cascade** that actually sells aliveness (head amplitude decreases as eyes start
contralateral; upper body follows with substantial latency; VOR handled explicitly).

For sway/settle use **spring-damper math, not a sine wave** — Holden's
[Perfect Tracking with Springs](https://theorangeduck.com/page/perfect-tracking-springs).

Evidence it's worth doing at all: *Scientific Reports* 2021 shows **natural postural oscillations
measurably enhance the empathic response** to a virtual character's facial expression.

---

## 7. The substrate: data, representation, rig, runtime

### 7.1 Licensing — the thing that will actually bite

**The universal blocker is upstream data, not model code.**

**AMASS license**, verbatim: *"Any other use, in particular any use for commercial purposes, is
prohibited"* and it *"prohibits the use of the Dataset to train methods/algorithms/neural
networks/etc. for commercial use of any kind."* **SMPL model license** is equally flat.

**Net: every HumanML3D-trained checkpoint in this survey is non-commercial in practice, regardless
of its stated code license.**

Three tiers:

| Tier | Datasets | Usable for |
|---|---|---|
| **MPI academic** | AMASS, SMPL/-H/-X model files, BABEL, GRAB, HumanML3D-as-data | non-commercial only |
| **CC-NC** | Motion-X(++), MotionMillion, SnapMoGen, BEAT2, Seamless Interaction, LAFAN1 (also ND) | non-commercial, share-alike |
| **Actually shippable** | **CMU mocap**, **100STYLE** (CC BY 4.0), **StayStill** (CC BY 4.0), IDEA400, Mixamo (in-product), **SeG library** (MIT) | commercial |

**Two escape hatches if commercial ever matters:**
1. AMASS commercial rights are **purchasable** via Meshcapade (`ps-license@tue.mpg.de`).
2. **CMU is the sleeper.** Its terms: *"You may include this data in commercially-sold products,
   but you may not resell this data directly."* CMU is a large slice of AMASS — so fitting SMPL to
   raw CMU yourself yields a commercially-shippable corpus without AMASS's terms. Real work
   (MoSh++-equivalent marker fitting), but it's the only clean DIY path.

### 7.2 Representation: use 272-dim, not 263-dim

**HumanML3D 263-dim** = root Y-angular-velocity (1) + root XZ linear velocity (2) + root height (1)
+ root-relative joint positions (63) + 6D joint rotations (126) + local velocities (66) + foot
contacts (4), at 20 fps.

Its problem: it is root- and frame-relative (velocities integrate → drift), massively redundant
(positions *and* rotations *and* velocities for the same joints, which a model can make mutually
inconsistent), and — the killer — **nearly every downstream repo discards the 6D rotation block and
re-derives rotations by IK from recovered joint positions.** MoMask's own README calls its solution
*"naive foot ik"* that *"sometimes works well, but sometimes will fail."* **That IK step is where
foot sliding, knee popping and rig jitter come from.**

**272-dim** (MotionStreamer) = root XZ linear velocity (2) + root angular velocity as full **6D**
(6) + local joint positions (66) + local joint velocities (66) + local 6D joint rotations (132), at
**30 fps**, K=22 SMPL joints, canonicalized to face +Z. The paper: it *"removes the post-processing
step and we could directly use it for animating a SMPL character model."* MIT tooling, including a
working `representation_272_to_bvh.py`.

**MotionMillion and MotionStreamer both standardized on 272. Train on 272; keep a 263 export path
only for benchmark numbers.** Skip absolute-coordinate representations (ACMDM) unless doing editing
research — they put IK right back in.

### 7.3 Rig and format: SMPL-X → VRM 1.0 → `.vrma`

**The semantic-layer problem decides this.** glTF, FBX and BVH all encode *a* skeleton. **None
encodes "this node is the left upper arm."** That mapping lives in a vendor layer — Unity Avatar
(not portable), Godot `SkeletonProfileHumanoid` (not portable), or **`VRMC_vrm.humanoid.humanBones`
(inside the glTF file)**.

**VRM/VRMA is the only interchange path where humanoid semantics survive the file boundary.**

And the fit is almost exact. VRMA's normative constraint: *"the animation data for the Humanoid bone
must not include scales ... must not include translations for bones other than the Hips bone."*
**Rotations only + hips translation — precisely SMPL/SMPL-X's parameterization**
(`global_orient` + `body_pose` + `transl`). The mapping is near-mechanical. VRMA also carries
**expression weights and eye gaze**, so one file format holds everything this project generates.

Better still, **VRM's normalized humanoid bones / ControlRig** resolve motion onto any avatar's rest
pose *at load time* — which makes the entire rest-pose/bone-roll bug class someone else's problem.
Only **15 required bones**: hips, spine, head, 6 leg, 6 arm.

**Retargeting pitfalls this sidesteps** (and which will cost days if you don't):

1. **Rest-pose mismatch is the #1 artifact source.** SMPL's zero pose ≠ Mixamo T-pose ≠ VRM T-pose
   ≠ your character's A-pose. Verify empirically by rendering `pose=0` — do not assume.
2. **Bone roll is not in the file.** glTF joints are just nodes; Blender *guesses* tips on import.
   90°-different rolls produce twisted forearms no name-mapping fixes.
3. **Hip height / scale → foot sliding.** Normalize or scale before transferring.
4. **Up-axis and units:** SMPL is Z-up, HumanML3D-processed is Y-up, glTF is Y-up/+Z-forward,
   **VRM 0.x is −Z-forward** (0.x → 1.0 is a 180° yaw about Y; the changelog is explicit:
   *"glTF: z- forward => z+ forward"*), BVH declares nothing at all.
5. **glTF animation is quaternion-only** — you cannot express >180° continuous twist between two
   keys. **Always sample densely on export.**
6. **glTF spec rule people violate constantly:** *"the transform of the skinned mesh node MUST be
   ignored."* Double-transform bugs trace here.

**Tools:** VRM Add-on for Blender (MIT/GPL dual, Blender 2.93→5.2, `.vrma` import *and* export — the
best-maintained thing in this space), Meshcapade `SMPL_blender_addon` (SMPL-X → armature),
**BVH-Motion-Retargeter** (MIT, Blender 4.2+, **VRM humanoid as a first-class target**) — that last
one closes the BVH→VRM gap for DiffuseStyleGesture, Semantic Gesticulator and StayStill.

⚠️ Rokoko's Blender retargeter is **broken on Blender 5.0** (issue #131). ⚠️ `pip install PyMO`
installs a MongoDB library, not the mocap toolkit — use **`bvhio`**.

**A caveat on the retarget step:** PhysDrift argues the standard human → SMPL-X → target pipeline
*"compresses motion diversity and weakens prosody-motion synchronization."* DiffuseStyleGesture's
own README independently notes manual SMPL-X conversion yields *"relatively lower motion realism."*
Budget for quality loss here — a stylized VRM's proportions differ from SMPL-X's more than a
realistic avatar's do.

**And benchmarks don't transfer to your character.** *Reality Check*
([2605.06063](https://arxiv.org/abs/2605.06063), May 2026) tested seven avatar types and found
*"avatar and face presentation systematically shift perceptual judgments."* Evaluate on your own
VRM or you're measuring nothing.

### 7.4 Preview — three tiers

1. **three.js (95% of iterations, sub-second).** `three` r185 + `@pixiv/three-vrm` 3.5.5 +
   `@pixiv/three-vrm-animation` 3.5.5. Use **`WebGLRenderer`, not WebGPU** — the manual still calls
   WebGPU *"experimental"*, and WebGPU on Linux/Mesa/AMD is the least-tested combination; you don't
   want renderer bugs contaminating your read on motion quality. Static page + WebSocket pushing
   fresh `.vrma`, `AnimationMixer` for scrub/loop/A-B. **Zero ROCm in this path** — that's the point.
2. **Godot 4 + godot-vrm (rig-truth check, seconds).** Confirms motion survives
   `SkeletonProfileHumanoid` + Rest Fixer `Overwrite Axis` onto a real game rig with a different
   rest pose. This catches "fine in three.js, foot-slides on the real rig."
3. **Blender 5.x headless (beauty renders, minutes).** `blender -b -P`, **EEVEE via EGL** — headless
   GPU rendering uses EGL and *"only works on Linux."* You're on the one platform where this works.
   ⚠️ Blender 5.0 renamed the engine id `BLENDER_EEVEE_NEXT` → `BLENDER_EEVEE`; any 4.x script
   silently breaks. ⚠️ Keep `bpy` **out of the PyTorch venv** — wheels are interpreter-pinned
   (cp313-only for 5.1+, cp311-only for ≤5.0 and both LTS lines).

Skip Unity: Linux support is Ubuntu-only (22.04/24.04), and this box is Fedora.

### 7.5 ROCm / gfx1201 — status and hazards

**The card is supported; the distro isn't; and there is one silent-corruption bug that must be
guarded against.**

gfx1201 got official support in **ROCm 6.4.1** (May 2025); current is 7.14 (Jul 2026). But AMD's
system requirements list consumer Radeon support only for **Ubuntu 24.04/22.04 and RHEL 9/10** —
**Fedora is not listed**, and there's an open issue where `amdgpu-install` reports "No AMD devices
detected" on Fedora Atomic with a 9070 XT.

**Distro-independent install path:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ "rocm[libraries,device-gfx1201]"
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/rocm7.2
export PYTORCH_HIP_ALLOC_CONF=roundup_power2_divisions:16   # NOT expandable_segments
# do NOT set HSA_OVERRIDE_GFX_VERSION / TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL / TORCH_ROCM_FA_PREFER_CK
```
TheRock rates gfx1201 on Linux **Build Passing / Sanity Tested / Release Ready**. rocm7.2 is the
newest stable PyTorch channel (rocm7.3+ 403s).

**Attention has improved since my earlier notes.** gfx1201 is in AOTriton's `AOTRITON_TARGET_ARCH`
as **non-experimental** (0.10b release notes: *"Official support of gfx950/gfx1201"*), and it is
**not** in `isArchExperimentallySupported()` — so no env var is needed on a 9070 XT. Standalone
**FlashAttention-2 forward *and* backward now work on gfx1201** (Dao-AILab PR #2613, *"Tested on
gfx1100/gfx1201/gfx942 ... all passed"*). Use plain `F.scaled_dot_product_attention`; head-dim
limit is 512. ⚠️ **Verify empirically on this card before relying on it** — gfx1201 is compiled
into shipped wheels but is **not in upstream CI test hardware**, which explains most bugs below.

**Open breakages, ranked by how much they'll cost:**

1. 🚨 **Silent fp32 GEMM corruption** — [ROCm #6595](https://github.com/ROCm/ROCm/issues/6595).
   `torch.mm`/`addmm`/`bmm`/`F.linear` return **silently wrong results** on gfx1201 when M > 2^19
   (524,288) rows. **No error raised.** fp16/bf16 fine. **Keep any flattened batch×seq dimension
   ≤ 262,144, and add a one-time numerical guard** against a CPU reference. Given accuracy-over-
   latency priorities this is the one that should change behavior.
2. **GPU page faults during LoRA/QLoRA fine-tuning** — ROCm #6600, OOB read in a hipBLASLt gfx1201
   kernel. `TORCH_BLAS_PREFER_HIPBLASLT=0` fixes it at ~2.5× step-time cost.
3. **`expandable_segments:True` hangs** uninterruptibly — the standard fragmentation remedy is
   unavailable. Use `roundup_power2_divisions:16`.
4. **Stay on ROCm 7.2, not 7.14** — three open regressions (false OOM, `apply_rope1` invalid
   argument, SDPA visual artifacts) with no compensating gfx1201 feature.
5. **Kernel cache dead on gfx1201** after ROCm 7.2.4 + firmware 20260519 — **385 s vs 5.5 s cached,
   a 60× regression.** Pin rocm-core 7.2.3 + firmware 20260410.

**Do motion repos run on ROCm? Yes — the blocker is packaging, not kernels.** Full git trees for
MDM, MoMask and T2M-GPT contain **zero `.cu`/`.cuh` files**; MotionGPT imports `CUDAExtension` but
declares no `ext_modules` (inert). The real cost is dependency-pin rot: MDM pins Python 3.7 +
`cudatoolkit=11.0`, MoMask pins `torch==1.12.0` + cu113 + `chumpy`, T2M-GPT pins
`pytorch=1.8.1=py3.8_cuda10.1`. None of those torch pins have a ROCm wheel. **That's a Python
packaging job, not a kernel port.** (`chumpy` is pure Python but breaks on `np.bool` and
`inspect.getargspec` — pin `numpy<1.24` or shim before import.)

⚠️ **PyTorch3D appears in HY-Motion's and ReMoMask's dependency sets** and is a genuine
hipify-and-fix job. Not needed for the MDM/MoMask/T2M-GPT family.

**16 GB reality:** at fp32 master + Adam ≈ 16 bytes/param, a 50M model needs 0.8 GB and a 500M model
8 GB of optimizer state. An MDM-shaped model (~20–30M params) trains comfortably at batch 64–128 —
throughput-bound, not memory-bound. Budget ~14 GB usable; prefer gradient checkpointing over
shrinking batch, since RDNA4 has compute headroom but modest bandwidth (640 GB/s).

---

## 8. Prioritized build order

**Phase 0 — de-risk the substrate (days)**
1. Verify the ROCm stack on this card: install per §7.5, run the **#6595 numerical guard**, benchmark
   SDPA (AOTriton flash vs math) to confirm whether the old "math SDPA only" workaround is still needed.
2. Stand up the three.js + `@pixiv/three-vrm-animation` preview harness. Everything downstream is
   judged through it, and it has no ROCm in the path.
3. Get **HY-Motion-1.0-Lite** running with a 4-bit Qwen3-8B text encoder. It is the quality ceiling
   and the baseline to beat; its FBX/GLB export also validates the whole rig pipeline end-to-end.

**Phase 1 — the Director/Performer split (weeks)**
4. Define the beat schema. Emit **free-form reasoning first, constrained JSON second** (§2.4).
   Watch for diversity collapse; consider a bracket wrapper over strict JSON.
5. Per-beat generation via HY-Motion or a 272-dim MoLingo/MotionStreamer, composed with **FlowMDM**
   or **MultiAct** for multi-sentence coherence.
6. **Morph** as the physics post-process (Apache-2.0, weights, drop-in). Penetration → 0.

**Phase 2 — the Critic, where the latency budget pays off (weeks)**
7. Build the motion verifier that doesn't exist: joint limits (Herda et al.), C¹ continuity, the
   seven T2MBench physical metrics, IK reachability. Deterministic, no LLM.
8. **Test-time scaling**: N=8 candidates → rank by verifier + text alignment → repair. ~40 s in
   budget, and TEXEDO's evidence says it improves both physics *and* semantics with no better
   generator. **This is the single most differentiated thing to build.**

**Phase 3 — expressivity as separate tracks (weeks)**
9. Face: ARKit-52 native only (SAiD or Azure), Perfect Sync onto the VRM. Never FLAME.
10. Idle/ambient: StayStill (CC BY 4.0) + BVH-Motion-Retargeter, plus procedural breathing, sway,
    blink (with `overrideBlink` semantics), and the Pejsa eye→head→torso gaze cascade.
11. Hands: retrieval from the SeG library (MIT, 544 BVH with real fingers), not generation.
12. Gesture: SynTalker (MIT, takes audio + free-text control prompt — a natural LLM-direction fit).

**Track, don't build on yet:** SCRIPT (best 2026 text+physics numbers — FID 0.164 vs 0.728 — but
code unreleased), PRISM / CMDM (streaming long-horizon, code unconfirmed), OmniMotion-X (would be
the unified answer; repo is a placeholder), GENEA 2026 challenge results (ECCV, 8–9 Sept 2026).

---

## 8b. Keyframes as the intermediate representation

*Added 2026-08-12 after a follow-up survey, in response to the design decision to make sparse
keyframes + timings the central IR rather than generating dense motion directly.*

### 8b.1 The argument for it got stronger

Beyond the debuggability and data-split arguments, there is now direct empirical support:
**keyframe conditioning is what buys out-of-distribution generalization.** MoMADiff (ACM MM 2025)
trained *only* on HumanML3D and tested zero-shot:

| Zero-shot R-Precision Top-1 | IDEA400 (12,042 seqs) | Kungfu (1,032 seqs) |
|---|---|---|
| MDM (text only) | 0.411 | 0.285 |
| **MoMADiff (text + keyframes)** | **0.644** | **0.701** |

Given the 10⁸-token data ceiling in §1, this matters more than it looks: the keyframes carry the
information the training corpus doesn't have. A keyframe proposer that can reach poses HumanML3D
never saw gets a densifier that will follow it there.

### 8b.2 ⭐ The counterintuitive core finding: do not enforce keyframes exactly

**Goel, Tevet, Liu & Fatahalian, "Generating Detailed Character Motion from Blocking Poses"**
(SIGGRAPH Asia 2025, [arXiv:2509.16064](https://arxiv.org/abs/2509.16064), DOI 10.1145/3757377.3763874).
Their thesis: *"Running a standard motion-inbetweening model on blocking poses can result in
unrealistic motion, because the blocking poses themselves are unrealistic."*

Blocking poses have three defects — temporally sparse, **imprecisely timed**, and **coarsely
posed** (only a few joints reflect intent). Their fix refines the *condition*, not the output:
every N denoising steps, take an extra step with an **unconditioned** model as a plausibility
prior, and blend it into the blocking poses under a **per-pose, per-joint tolerance vector**
(default 0.85; >0.95 for joints the animator insists on; low for unposed joints), searching
**±10 frames** around each key for the closest matching plausible pose.

| Method | FootSkate ↓ | Jitter ↓ | FID ↓ | Keyframe Err ↓ |
|---|---|---|---|---|
| **Tolerant (c=0.50)** | **5.14** | **0.223** | **0.038** | **0.106** |
| Tolerant (c=0.85) | 5.96 | 0.241 | 0.058 | 0.105 |
| No tolerance (c=1.0) | 8.19 | 0.282 | 0.077 | 0.145 |
| **CondMDI (exact adherence)** | **29.51** | **2.293** | **0.305** | **0.664** |

**Read the last row against the first.** A model built to match keyframes exactly is 5× worse on
foot-skate and 6× worse on *keyframe error itself* — because it faithfully reproduces bad poses and
then has to fake the transitions between them. **Exact adherence is actively harmful when keyframes
are proposals rather than ground truth.**

> 🚨 **UNIT ERROR — the table above is missing its column multipliers.** Source Table 1 of
> [2509.16064](https://arxiv.org/abs/2509.16064) reads **FootSkate ×10⁻³** and **Jitter ×10⁻²** and
> **KE ×10⁻²**. So CondMDI's `29.51` is **0.0295** — *lower* than CondMDI's own published
> 0.0806–0.0936. Different metric definitions on different clip lengths (60-frame splices here vs
> 196-frame there). **No number in this table compares to anything outside it.**
>
> ⚠️ **And the CondMDI row is a stress test, not a like-for-like benchmark.** Every method was fed
> deliberately degraded input: ≤10 keys per 60-frame clip, only "important" joints retained
> (root/shoulders/elbows/hips/knees) with **the rest set to neutral**, plus **±5-frame timing
> noise**. The authors are explicit that CondMDI *"expects the posing and timing of input keyframes
> to appear in the output exactly as they appear in the input"* and *"struggles when faced with
> coarsely specified input."* The row shows **exact-adherence models break on proposal-quality
> keys** — a design property, and precisely the property that disqualifies exact adherence here. It
> does **not** show CondMDI is a weak in-betweener, and this document does not claim that.
>
> ✅ **The claim survives without that row — quote this instead.** The clean internal ablation is
> `R-NoTolerance` vs `Ours (c=0.50)`: identical architecture, identical protocol, tolerance the only
> variable. **FootSkate 8.19 → 5.14 (−37%), FID 0.077 → 0.038 (−51%), Jitter −21%, and KE
> 0.145 → 0.106 (−27%).** Tolerance improves keyframe error *while* improving quality. That is the
> load-bearing result.
>
> CondMDI reference figures for comparison ([2405.11126](https://arxiv.org/abs/2405.11126) Tables 2/3
> — FID/R-Prec/Div/FootSkate/KeyframeErr): Random K=1 `0.1551/0.6787/9.5807/0.0936/0.3739`, K=5
> `0.1731/0.6823/9.3053/0.0850/0.1789`, K=20 `0.2253/0.6821/9.1151/0.0806/0.0754`, root joint
> `0.2474/…/0.0854/0.0525`, VR joints `0.2969/…/0.0794/0.0422`.
> ⚠️ **CondMDI is arXiv 2405.11126.** The ID `2405.10874`, which circulated in earlier drafts of this
> document, is a GNSS-visual-inertial navigation paper.
>
> **General rule for this document:** third-party reproductions of CondMDI in this literature are
> frequently misconfigured or run out-of-distribution — e.g. NINB ([2605.12778](https://arxiv.org/abs/2605.12778))
> reports CondMDI FIDs ~10× its own under a different evaluator (its *keyframe-error* column matches
> CondMDI's within noise, proving the model ran fine and only the FID scale differs), and AnchorRoute
> ([2605.14716](https://arxiv.org/abs/2605.14716)) reports 0.507 m control error against CondMDI's own
> 0.0422. Always name the paper that produced a number and check the protocol before comparing across
> papers; flag any third-party row deviating >2× from a self-reported one. Relatedly, "Less is More"
> ([2503.13859](https://arxiv.org/abs/2503.13859)) does **not** beat CondMDI at in-betweening — its own
> Table 7 has it worse on FID (0.551 vs 0.153), keyframe error (0.218 vs 0.081) and skating
> (0.082 vs 0.067); its only CondMDI win is plain text-to-motion, a different task. And NINB's
> advertised repo (`Coondinator/NINB`) is empty — created the day of the arXiv post, one zero-byte
> README, never pushed. **Not** code-available.

Since keys here will come from an LLM, retrieval, or a small learned model, **this project is in
the blocking-pose regime by construction, not the clean-mocap regime.** Tolerance is not an
optimization to add later; it is the correct default from day one.

⚠️ Code listed as "Coming Soon", not released as of 2026-08-12. But the mechanism is simple enough
to reimplement, and CondMDI already ships an unconditional checkpoint — both halves are available.

Their stated failure mode is also the one to design around: when blocking is *semantically*
insufficient (*"only two T-poses on the ground"* will not yield a jump), the fix they name is
**text conditioning alongside the keys** — i.e. exactly this architecture.

### 8b.3 Timing is a solved-enough problem, and it directly fixes the LLM's worst weakness

**Goel, Zhang, Liu & Fatahalian, "Generative Motion Infilling From Imprecisely Timed Keyframes"**
(Eurographics 2025, [arXiv:2503.01016](https://arxiv.org/abs/2503.01016), code at
[purvigoel/motion-retiming](https://github.com/purvigoel/motion-retiming)). A dual-headed transformer
diffusion model that jointly predicts an **explicit global time-warping function** (retiming the
mistimed keys) and **local pose residuals** — jointly, because timing and spatial detail are
coupled. Built on MDM, trained on an AMASS subset, checkpoints on Drive. ⚠️ No LICENSE file.

The framing is precisely the failure mode of an LLM-proposed schedule: existing in-betweeners
*"expect the timing of input keyframes to be perfect"*; when they aren't, characters *"move between
poses unrealistically fast or fail to reach intended positions."* **Given that LLM timing quantizes
to whole and half seconds (§2.1), a densifier that is allowed to slide keys by ±5–10 frames is not
optional.**

Supporting components:
- **Duration prediction** — Guo et al.'s `text-to-motion` length estimator: BiGRU over GloVe + POS
  tags predicting length in **discretized 4-frame bins as classification**. Simple, cheap, directly
  reusable as a "how long is this beat" module. MoMask inherits it.
- **Variable transition lengths** — Harvey et al. (SIGGRAPH 2020) time-to-arrival embedding
  modifier is still canonical. LAFAN1's benchmark protocol is 5/15/30-frame transitions.
- **Phase** — Local Motion Phases (SIGGRAPH 2020) and DeepPhase (SIGGRAPH 2022), both in
  [AI4Animation](https://github.com/sebastianstarke/AI4Animation); Dai et al.
  ([2503.08180](https://arxiv.org/abs/2503.08180)) carry part-wise phase into in-betweening. The
  strongest existing non-musical "when does the beat land" signal.
- ⚠️ **Explicit animation principles (anticipation, overlap, follow-through, ease) are an open
  gap.** The only academic hit is a 2006 procedural anticipation method; everything modern learns
  them implicitly from mocap. Goel's retiming is the closest thing to modeling a timing principle
  explicitly.

### 8b.4 Densifier options

| Model | Sparse keys | **Partial poses** | Text | Speed | Code / license |
|---|---|---|---|---|---|
| **CondMDI** (SIGGRAPH'24) | ✅ | ✅ per-joint masks | ✅ | ⚠️ 1000 steps, **54.4 s** | [setarehc/diffusion-motion-inbetweening](https://github.com/setarehc/diffusion-motion-inbetweening) — **MIT**, 3 checkpoints, **training code** |
| **ARDY** (SIGGRAPH'26, NVIDIA) | ✅ | ✅ + waypoints, EE rot, foot contacts, **out-of-horizon goals** | ✅ online | **33 ms** (4-step) | [nv-tlabs/ardy](https://github.com/nv-tlabs/ardy) — Apache-2.0 code, NVIDIA Open Model weights; ⚠️ **inference only, no training code**; text encoder Llama-3-8B ~14 GB |
| **MaskControl** (ICCV'25 Oral) | ✅ any-joint-any-frame | ✅ | ✅ | 0.46 s / 68.65 s w/ opt | ⚠️ **CC-BY-NC-ND** — no commercial, **no derivatives** |
| **MoMADiff** (ACM MM'25) | ✅ ~1 key/s | ❌ **whole frames only** | ✅ | 3.28 ms/frame | [zzysteve/MoMADiff](https://github.com/zzysteve/MoMADiff) — MIT, HF weights |
| Two-Stage Transformers (Qin, TOG'22) | ✅ | — | ❌ | fast | The practical default for skeleton-space LAFAN1 work |

**CondMDI's conditioning mechanism is ~20 lines**: overwrite the noisy input at observed frames and
concatenate the observation mask as an extra channel, training with randomly sampled frame- and
joint-masks so one model covers dense, sparse and partial. ARDY independently arrived at the same
masked-constraint formulation, which is good evidence the design is right.

⚠️ CondMDI's own ablation is a warning about how to enforce constraints: imputation-only gives
FID 0.360 / KE 0.515; adding reconstruction *guidance* gives KE 0.0034 but **FID 1.707**; explicit
conditioning gives FID 0.173. **Guidance buys constraint satisfaction and destroys realism.**

### 8b.5 Sparsity budget: ~1 key/second

CondMDI Table 2 (random keyframes, 196-frame clips) shows the tradeoff cleanly:

| K (keys per 196 frames) | FID ↓ | Keyframe Err ↓ |
|---|---|---|
| 1 | **0.1551** | 0.3739 |
| 5 | 0.1731 | 0.1789 |
| 20 | 0.2253 | **0.0754** |

Note the inversion: **FID gets *worse* as keys are added; keyframe error gets better.** Quality
does not collapse when keys are sparse — *adherence* does. At K=1 the model produces something that
looks fine and **silently ignores the specification**, which for a keyframes-as-IR system is the
worst possible failure mode.

Converging evidence for ~1 key/s: MoMADiff evaluates at exactly 1 key/s; ARDY's generation window is
2 s and DiP degrades from 2.48 cm to 17.64 cm error once goals leave the horizon; CondMDI at K=5
(~1 key/1.6 s) holds FID 0.17; "Less Is More" keeps 20% of frames. **Treat gaps >2 s as requiring
text conditioning to disambiguate** — which is exactly the fix the blocking-poses authors name.

### 8b.6 Manufacturing supervision — keyframe extraction

This is how the data problem gets sidestepped: paired (sparse keys → dense motion) supervision is
free from any mocap.

- **Visvalingam–Whyatt is the current pick.** "Less Is More" ([2503.13859](https://arxiv.org/abs/2503.13859))
  represents each frame as a **64-d vector of root-relative joint positions + temporal index**, then
  iteratively removes frames by smallest **effective area** (triangle formed with neighbors), at
  **60–80% reduction**. VW is the area-based sibling of RDP and better suited to motion because it
  produces a *rank ordering* by significance — one pass yields every sparsity level.
- **Pose saliency** — Liu, Chen & Lin, *The Visual Computer* 39:4943–4953 (2023),
  [doi:10.1007/s00371-022-02639-3](https://link.springer.com/article/10.1007/s00371-022-02639-3).
  Limb rotation angles + inter-joint distances → multiscale limb saliency → peaks → reconstruction-
  error refinement. Its related-work section maps the whole prior-art space (curve simplification,
  PCA, GA, PSO, sparse representation, affinity propagation).
- **Learned selection** — **AutoKeyframe** (SIGGRAPH 2025, [Cr7st/AutoKeyframe](https://github.com/Cr7st/AutoKeyframe),
  **MIT**) ships **pre-extracted LaFAN1 keyframes** you can download directly, selected by an RL
  selector. ⚠️ Its own in-betweening component is not released; the authors recommend Two-Stage
  Transformers instead. Also SIDQL ([2407.00925](https://arxiv.org/abs/2407.00925)), Deep-Q keyframe
  selection, <0.09 reconstruction error at five keyframes on CMU.
- **Tooling equivalent:** Blender Graph Editor → **Decimate Keyframes** (Ratio and Error Margin
  modes). ⚠️ Which algorithm backs Error Margin is unconfirmed — do not assume RDP.

**Open gap worth filling cheaply:** nobody evaluates extractor choice against a *generative*
densifier — existing work scores reconstruction after linear/spherical interpolation only. A
bake-off (VW vs RDP vs saliency-peaks vs uniform, scored by FID/footskate/KE after the actual
densifier) is roughly a day of compute and would directly tune the data pipeline.

### 8b.7 Recommended starting point

**Start from CondMDI** (MIT, weights, **training code**, natively does sparse + partial + text) and
make three modifications, in order:

1. **Fix the sampler first.** 1000 steps / 54 s per sample makes iteration miserable. DiP (10 steps,
   MIT, in the MDM repo) and ARDY (4 steps, 33 ms) both prove few-step motion diffusion works —
   ARDY's ablation shows 1–2 steps break down but **4 steps is near-parity with 100**. Distill or
   retrain to ~10 steps before anything else.
2. **Build the extraction pipeline on Visvalingam–Whyatt**, sampling the reduction ratio randomly
   during training so one model spans all sparsity levels. CondMDI already trains with random
   frame/joint masks, so this is a change to the mask sampler, not the model. Use AutoKeyframe's
   pre-extracted LaFAN1 keys as a free second opinion.
3. **Do not train for exact adherence.** Adopt the blocking-pose tolerance formulation from day one
   — per-joint tolerance ~0.85, blended against the unconditional checkpoint every N steps, with a
   ±5–10 frame timing search window.
   - ✅ **Direction confirmed by measurement, magnitude corrected — see §8d.9.** Relaxing exact
     adherence helps even when the keyframes are *perfect*: on CondMDI at θ=0, hard imputation vs
     native conditioning costs 3× the jerk and +65% foot skate. But the measured knee sits at
     **c ≈ 0.2–0.5**, not 0.85, and the benefit is a uniformly better baseline — **not** absorption
     of keyframe error, which does not happen at any tolerance. Note the two `c`s are different
     mechanisms (2509.16064's per-joint blocking-pose tolerance vs CondMDI's imputation blend
     weight), so treat the numeric value as indicative and re-measure on whatever stack you build.

Why not the alternatives: ARDY is faster and better but ships **inference-only**, so it can never be
adapted to a custom keyframe format, and its default text encoder alone wants ~14 GB. MaskControl is
**CC-BY-NC-ND** — no commercial use *and* no derivatives. MoMADiff is MIT but **whole-frame keys
only**, which kills partial poses. SFControl has no checkpoints yet.

CondMDI's backbone is a 1D-conv UNet (512 ch), not a giant transformer — plain PyTorch, comfortable
in 16 GB, ROCm-friendly.

---

## 8c. Image-domain pose priors — investigated and rejected

*Added 2026-08-12. Recorded as a negative result so it doesn't get re-proposed.*

**The idea:** text → generated image of a posed character → lift to 3D via HMR → use as a keyframe.
Motivation: images have billions of text-paired examples, motion has ~10⁸ tokens, so borrow the
image domain's semantic coverage. **Verdict: not sound as stated.** Three reasons.

### 8c.1 It was done at scale and did not win

**Make-An-Animation** (Azadi et al., Meta, ICCV 2023, [2305.09662](https://arxiv.org/abs/2305.09662))
is this idea, verbatim: **35M pose–text pairs mined from large-scale image-text datasets**, poses
extracted with Detectron2 + PyMAF-X into 3D SMPL pseudo-poses, then finetuned on AMASS/HumanML3D.

| HumanML3D | R-Precision ↑ | FID ↓ | Diversity ↑ |
|---|---|---|---|
| Real | 0.797 | 0.002 | 9.50 |
| MDM | 0.611 | **0.544** | 9.559 |
| T2M | **0.740** | 1.067 | 9.188 |
| **MAA** | 0.676 | 0.774 | **8.23** |

Worse than T2M on R-precision, worse than MDM on FID, lowest diversity of the three. The SOTA claim
rests on human eval only, presented graphically with no numeric table. **No code, no weights, no
limitations section, and no follow-up work building on the pseudo-pose-from-images mechanism** —
2025+ citers that use vision priors (Motion-2-to-3, AnimaX) use *video*, and mostly 2D or
multi-view, not lifted single images.

Companion paper, same author, static poses: [2304.07410](https://arxiv.org/abs/2304.07410) — a
text-to-3D-pose diffusion model trained on in-the-wild poses mined from image datasets. Also no code.

### 8c.2 The errors compound and are structurally unobservable

Two uncontrolled stages in series, with no ground truth anywhere in the loop:

| Stage | Error | Source |
|---|---|---|
| Prompt → pose actually realized in image | **~35–60% of conditioning not faithfully realized** (COCO AP 40–64, vs 75–80 for a pose estimator on real photos) | Stable-Pose (NeurIPS'24) Table; TrioPose 2026 best-in-class AP 64.33 |
| Anatomical validity of generated humans | **~50% of T2I human images contain distortions**; Anatomical Error Rate FLUX **0.417**, SDXL **0.873** | Distortion-5K ([2503.00811](https://arxiv.org/abs/2503.00811)), AbHuman, AGHI-QA, ASAP/HAF-Bench |
| Image → 3D articulation | **38–43 mm PA-MPJPE, 15–16° MPJAE** on photoreal in-the-wild | SAM 3D Body, NLF |
| ...on hard/rare poses | **86 mm PA-MPJPE — 4–5× worse** (17.2 → 86.4) | SAM 3D Body Table 6 |
| ...on stylized imagery | **unmeasured.** 2D AP collapses 75.6 → 6–10 without finetuning | Human-Art (CVPR'23) |
| + global orientation | **1.5–1.75× on top of PA-MPJPE**, a ratio stable across every model | §3.4 of the source survey |
| Absolute translation | **129 mm (GT camera) / 274 mm (predicted)**; metres if the camera assumption is wrong | Multi-HMR 2 |
| **Silent 2D-fit camera error** | **146–300 mm MPJPE while looking perfectly 2D-aligned** | TokenHMR controlled experiment |

That last row is the trap. **TokenHMR** ([2404.16752](https://arxiv.org/abs/2404.16752)) showed that
projecting *ground-truth* 3D bodies through HMR2.0's assumed camera scores **PCK0.5 = 0.66 — worse
than HMR2.0's own predictions (0.78)**. The GT looks worse in 2D than the prediction does, proving
the camera model is the error source. Running SMPLify to maximize 2D alignment gives **146 mm MPJPE
at 100 iterations and >300 mm at 200**, both with excellent 2D alignment. **A generated image gives
you no camera intrinsics**, so this is exactly the regime you'd be in.

And because SMPL/SMPL-X/MHR regressors are *constrained* — they can only emit a kinematically valid
human — a three-armed or six-fingered generated figure yields a **confident, clean, plausible, wrong
mesh with no error signal**. TokenHMR makes this worse by construction: it *"restricts the estimated
poses to the space of valid poses."*

**Gen-B** ([2411.08663](https://arxiv.org/abs/2411.08663), MPI) explains why the field went the other
way: *"the more realistic the generated images, the more they deviate from the ground truth, making
them inappropriate for training and evaluation... We empirically verify that this misalignment
causes the accuracy of HPS networks to decline when trained with generated images."* Realism and
pose-fidelity are in **direct tension**. Everyone therefore keeps the 3D as the *control signal*
(known GT) rather than trying to recover it — PoseDreamer, BEDLAM 2.0, Gen-B all run the pipeline
in that direction.

### 8c.3 ⭐ The decisive argument: it is most accurate where it adds nothing

Everything this would be built for — rare, expressive, extreme, compositional poses that mocap
lacks — is where **every stage degrades simultaneously**:

- T2I anatomical error rate is highest on self-interaction and full-body prompts
- ControlNet fidelity is worst on Human-Art (artistic/unusual) vs LAION-Human
- 2D detection collapses first, and it gates everything downstream
- HMR articulation degrades 4–5× on `pose_3d:very_hard`
- the constrained pose prior regresses hardest toward the mean precisely when the true pose is
  farthest from it

On common standing/walking poses the pipeline works fine — **and those are exactly the poses already
available in AMASS, MotionMillion, and every mocap library on earth.**

### 8c.4 One premise correction, in both directions

The source survey argued the data-scarcity premise is "two orders of magnitude out of date."
**That over-corrects** — §1's claim was that the largest corpus is under 10⁸ tokens, and the
survey's own table (MotionMillion 2,000 h ≈ 5.4×10⁷; OmniHuMo 5,000 h ≈ 1.35×10⁸) confirms it.
The image/motion asymmetry vs ~10¹³ text tokens is real.

**But the actionable gap is smaller than it looked**, and that's the part that matters:
**MotionMillion released 3B and 7B weights *and* the 2M-sequence dataset** (§4.2). If the motivation
is "I need broader semantic coverage," downloading 2,000 hours dominates building a two-stage lossy
pipeline to manufacture it.

> 🚨 **License correction (was wrong in the first draft of this section).** MotionMillion is **not**
> Apache-2.0. Only the *code* is; the **dataset and weights on HuggingFace
> (`InternRobotics/MotionMillion`) are CC BY-NC-SA 4.0**, gated behind a contact form, and the corpus
> bundles AMASS, BABEL, AIST and HumanML3D — so it is **AMASS-tainted and non-commercial**. It is
> usable for research and prototyping, not for anything shipped commercially. The same trap applies
> broadly: MoMask, MDM, CondMDI, OmniControl and MotionGPT all have MIT/Apache *code* whose released
> *weights* inherit AMASS's non-commercial terms. See §8c.6 for the one genuinely clean stack, and
> §8d for the route that dissolves this problem.

### 8c.5 What survives — the instinct was right, the extraction operator was wrong

> 🚨 **RETRACTED 2026-08-12, same day.** The multi-view recommendation below did not survive
> verification. **MAS has no text conditioning whatsoever** — it trains a separate *unconditional*
> 2D motion diffusion model per domain, and the released weights are NBA basketball (~60K), horse
> jumping (~2K) and rhythmic gymnastics (~500), all scraped video. There is no mechanism to ask it
> for "a person crouching to pick up a box," and no engineering fixes that without retraining on
> captioned data that does not exist. Three further reasons: (a) the 2D prior is architecturally a
> **sequence** model — c-MAS confirms it *"does not explicitly address processing single static
> frames"* — so it cannot emit sparse keys without generating dense motion first, inverting the
> pipeline; (b) therefore sparse output is **not** cheaper (c-MAS runs at ~2.4 FPS with **1,000
> optimization iterations per denoising step**, V=7 views); (c) absolute accuracy is mediocre —
> c-MAS reaches **113.4 mm MPJPE** on Yoga90, which is genuinely impressive for an *unsupervised*
> method beating supervised lifting, but is still 2D→3D lifting **from a real image**, not
> generation from text.
>
> **The replacement for this stage is retrieval, not multi-view generation — see §8d.8.**
> One thing here is still worth stealing: CoT-Pose's *offline data synthesis* path
> (FLUX-LoRA → SMPLest-X) exploits the 2D image prior exactly as intended. ⚠️ But its yield was
> **239 usable samples after manual filtering.** Viable to expand a pose bank offline; not viable as
> an online proposer.
>
> MAS ([roykapon/MAS](https://github.com/roykapon/MAS)) is **MIT with released weights and no AMASS
> in its lineage** — the cleanest license in this entire document, and unusable here for want of text
> conditioning. c-MAS has **no license file**. AnimaX animates *arbitrary articulated meshes*, not
> humans specifically.

Using the image/video domain as the semantic engine is sound. **Monocular lifting is the wrong way
to extract from it**, because it is an ill-posed inverse problem with no error signal. Two
approaches make it well-posed instead:

- **Multi-view generation + triangulation.** **MAS** ([2310.14729](https://arxiv.org/abs/2310.14729),
  Tevet et al.) denoises several 2D views of the same 3D motion simultaneously, reconciling them into
  3D at each diffusion step. **AnimaX** ([2506.19851](https://arxiv.org/abs/2506.19851)) represents
  3D motion as multi-view multi-frame 2D pose maps, triangulates to 3D joints, then IK to mesh;
  trained on 160K rigged sequences, SOTA on VBench. **c-MAS**
  ([2605.15583](https://arxiv.org/abs/2605.15583), code released) extends it to single-view lifting
  with anatomical constraints, beating supervised SOTA cross-domain on extreme (yoga) poses.
- **2D-local / 3D-global disentanglement.** **Motion-2-to-3**
  ([2412.13111](https://arxiv.org/abs/2412.13111), ZJU3DV) trains a 2D local motion generator on a
  large text-2D corpus from video, then finetunes into a multi-view generator — explicitly
  separating local articulation (what 2D data carries) from global trajectory (what it doesn't).
  The stable 1.5–1.75× PA-MPJPE→MPJPE ratio across every HMR model is the empirical justification:
  **articulation transfers from 2D; orientation and translation do not.**
- **Video-RAG at inference.** **VimoRAG** ([2508.12081](https://arxiv.org/abs/2508.12081)) retrieves
  from large in-the-wild video databases for 2D motion signals to fix motion-LLM OOD failures —
  same motivation, no lifting step at all.

### 8c.6 Valuable side-findings that apply regardless

- **⭐ A commercially-clean stack exists, and it sidesteps SMPL entirely.** **SAM 3D Body**
  ([2602.15989](https://arxiv.org/abs/2602.15989), Meta 2026) is current single-image SOTA
  (EMDB 38.2 PA-MPJPE / 61.7 MPJPE), ships **code and weights under the SAM License (commercial
  use permitted)**, and its **MHR body model is Apache-2.0**. **Anny**
  ([2511.03589](https://arxiv.org/abs/2511.03589)) is an Apache-2.0 scan-free all-age body model.
  Together with a self-captioned commercially-clean mocap corpus (§8d.2 — *not* MotionMillion, whose
  weights and data are CC BY-NC-SA) that is a genuine commercial path — worth knowing before
  committing to SMPL, since the MPI license *"prohibits the use of the Software to train
  methods/algorithms/neural networks/etc. for commercial use."*
- **⚠️ The license trap is real and specific:** NLF and 4D-Humans have **MIT code but
  non-commercial or unlicensed weights**. A permissive repo license does not launder AMASS-derived
  weights — and **every** general-purpose pose prior (VPoser, GAN-S, Pose-NDF, DPoser, PoseScript)
  is AMASS-trained.
- **⭐ Geometry-only plausibility filters carry no data license at all**, which makes them the right
  basis for the §5 verifier. **HumanScore** ([2604.20157](https://arxiv.org/abs/2604.20157),
  Stanford 2026) gives six directly reusable metrics: extra-limb detection, bone-length consistency,
  joint ROM vs biomechanics tables, self-collision (BVH triangle-triangle, mild/severe), kinematic
  extremes, motion smoothness (jerk). Best video generators score **91.1 vs 94.3 for real footage**.
- 🚨 **DISPUTED — do not cite until re-verified.** An earlier draft claimed **OpenT2M**
  ([2603.18623](https://arxiv.org/abs/2603.18623)) filters video-extracted motion through an
  AMASS-trained RL tracking policy and **discarded over 37%**. A later verification pass reports
  2603.18623 is *"OpenT2M: No-frill Motion Generation with Open-source, Large-scale, High-quality
  Data"* (Cao et al., Mar 2026) — a **2,800-hour dataset plus tokenizer, not a physics policy**. The
  two readings are incompatible. **The 37% figure is unsupported until someone opens the paper.** The
  underlying idea (use a tracking policy as a plausibility oracle) remains sound on PHC's evidence
  below; only this citation is in question.
- **DPoser** ([2312.05541](https://arxiv.org/abs/2312.05541), code MIT) is the current best
  general-purpose pose prior — FID 0.07 / Precision 0.72 / Recall 0.80 vs VPoser 0.66/0.29/0.42.
  ⚠️ Note **Pose-NDF's headline diversity (APD 18.75) comes from generating off-manifold poses** —
  Precision 0.02, Recall 0.00. Don't read its APD as a quality win.
- **Canonicalization recipe (HumanML3D, MIT code):** floor-align by subtracting the lowest joint's
  Y; zero root XZ while preserving height; compute the forward vector from the cross product of the
  hip-line and shoulder-line and rotate so frame 0 faces +Z. **Heading (yaw) is removed from the
  body frame — not by zeroing `global_orient`.** For shape: HumanML3D discards betas entirely via a
  leg-length-ratio rescale onto one template, so the whole text-to-motion literature operates on a
  single fixed body. **If you only need joint rotations, set `betas = 0`** — which also sidesteps
  the documented "regression to the mean" in HMR shape estimation (waist error 7.6–8.8 cm).
- **Text→static-3D-pose exists but is weak.** PoseScript ([2210.11795](https://arxiv.org/abs/2210.11795))
  tops out at **mRecall R/G 37.5%**, is **CC BY-NC-SA**, and requires low-level joint prose
  (*"left arm raised 45 degrees"*) rather than intent (*"reaching for a book on a high shelf"*).
  **CoT-Pose** ([2508.07540](https://arxiv.org/abs/2508.07540), ICCVW 2025) attacks exactly that
  abstraction gap and is worth watching.

---

## 8d. Multi-stage decomposition with manufactured intermediate supervision

*Added 2026-08-12. The current design direction: rather than solving text→motion in one shot, stage
it, and manufacture the training data for the intermediate steps.*

### 8d.1 ⭐ The taxonomy that dissolves the "staged vs end-to-end" debate

Three unrelated things are all called "decomposition," and conflating them is the single largest
source of confusion in this literature:

| | Boundary | Differentiable? | Verdict |
|---|---|---|---|
| **D1 — representation staging** (VQ/RVQ tokenizer → generator) | learned codes | effectively yes | **universal in 2026, mandatory** |
| **D2 — intra-model hierarchy** (coarse→fine temporal scales, part-factored latents + fusion) | latents in one graph | yes | **works; modest (~1.8× FID)** |
| **D3 — symbolic inference-time pipeline** (LLM script → keyframes → interp → critic) | human-readable symbols | **no** | **loses on quality, wins on control** |

**Every 2026 leaderboard leader is D1+D2. None is D3.** Papers claiming "staged pipelines beat
monolithic in 2026" nearly always mean D1 staged *training* — tokenizer, then generator — which says
nothing about a symbolic pipeline. Read every such claim against this table first.

**No D3 system beats a strong end-to-end baseline on both FID and R-Precision.** The signature is
always alignment up, realism flat or much worse:

- **Text2BFM** ([2605.29906](https://arxiv.org/abs/2605.29906), 2026), the purest "decouple semantic
  planning from motion execution" system: **R@3 0.876** (best in the literature, vs MoMask 0.807)
  and **FID 1.172 vs MoMask's 0.045 — 26× worse.**
- **FG-MDM** ([2312.02772](https://arxiv.org/abs/2312.02772)): FID 0.663 vs T2M-GPT 0.116. The
  authors state it plainly: *"under within-dataset settings, our model does not exceed those SOTA
  models."*
- **LaMoGen** ([2603.11605](https://arxiv.org/abs/2603.11605)): a wash against ReMoDiffuse, and the
  authors attribute the FID lag to Laban abstraction *"collaps[ing] low-level variation"* — the
  bottleneck, named by its own designers.
- PlanMoGPT's own abstract: *"LLM-based methods lag far behind non-LLM methods."* MotionGPT-2 with a
  fine-tuned LLaMA-3.1-8B scores FID 0.191 — worse than T2M-GPT (2023, GPT-2-scale) and **4.2×
  worse than MoMask, which uses no LLM at all.**

**Why: the intermediate representation is an information bottleneck that discards exactly what makes
motion look real.** A beat script and sparse keys encode *what* and *when*; they do not encode
momentum, weight transfer, contact state, anticipation, or sub-beat phase. Four independent
measurements of the same bottleneck:

| Boundary | Measured cost |
|---|---|
| Vanilla VQ tokenizer | recon FID **0.070** / MPJPE 58.0 mm — a floor under T2M-GPT's 0.141 generation FID, i.e. **half its total error is the boundary** |
| RVQ tokenizer (MoMask) | recon FID **0.019** / 29.5 mm → generation FID 0.141 → **0.045** |
| Pose-keyframe injection by guidance | GMD FID 0.212 → **0.874** (4.1×) |
| Multi-joint keyframe injection | OmniControl 0.218 → 0.624; TLControl's reproduction **2.614** |
| Part-split decoder | CoMA recon FID 0.027 → **0.046** |
| Independent part generation, no fusion | LGTM 0.218 → **7.384**; ParCo 0.109 → **3.652** |
| Segment seams, best method | FlowMDM subseq FID 0.29 → transition FID **1.38** (4.8×) |
| Multi-action semantic drift | Text2BFM order accuracy **0.671** (3 actions) → **0.509** (4 actions) |

**The largest FID improvement in the field's history came from *weakening* a stage boundary**
(MoMask's RVQ), not from a better generator. That is the argument against D3 in one sentence.

⚠️ Note what the LGTM/ParCo ablations actually measure. FID 0.218 → 7.384 is not "hierarchy helps";
it is "decomposition is catastrophic without a repair stage, and the repair stage does all the
work." Self-inflicted wound plus bandage.

### 8d.2 ⭐⭐ The licensing unlock — the reason to build a captioner regardless

This is the most valuable single finding in the entire research effort, and it is a side effect.

Every commercially-clean mocap corpus — **CMU** (license explicitly permits inclusion in
commercially-sold products), **100STYLE** (CC BY 4.0), **StayStill** (CC BY 4.0), **IDEA400** — has
**no text annotations.** That absence is *precisely why* nobody uses them for text-to-motion, and
why the entire field is built on AMASS and is therefore non-commercial (see §8c.6 and §8d.6).

**A rule-based captioner you write yourself is the mechanism that converts commercially-shippable
geometry into commercially-shippable (text, motion) pairs, with no AMASS anywhere in the lineage.**
The rules are algorithmic and reimplementable: PoseScript's **77 posecodes + 10 super-posecodes** are
a *specification*, not data. SMD's 26-joint biomechanical rule set is CC BY 4.0.

This turns licensing from a blocking constraint into a solvable engineering task. Nothing else in
this research does that.

### 8d.3 Backtranslation: works, but only as pretraining

PoseScript ([2210.11795](https://arxiv.org/abs/2210.11795), Table II) ran exactly the relevant
experiment. It has stood unchallenged for four years:

| Stage-2 training data | mRecall on **human-written** test captions |
|---|---|
| Auto-captions only (100k poses × 3) | **5.9** ±0.4 |
| Human captions only (~4.4k) | 23.0 ±0.6 |
| **Auto-pretrain → human-finetune** | **40.9** ±0.1 |

*(the same auto-only model scores 72.8 on its own auto-caption test set)*

Two conclusions, and both matter:

1. **Zero-shot transfer from templated captions to free-form language does not happen.** 5.9 is
   *worse* than training on 4.4k human captions alone, and 12× worse than the model's in-domain
   score. The template distribution is a different language. The paper: *"the performance degrades
   on human captions, as many words from the richer human vocabulary are unseen during training."*
2. **As pretraining it is the largest win in the text-to-pose literature.** Generation FID
   **0.29 → 0.04**; mRecall **5.2% → 19.5%**. Larger than any architecture change published (adding
   a transformer encoder: +2.4 mRecall; mirroring: +2.0). And it scales monotonically with
   manufactured volume (10k → 20k → 100k), so budget converts to accuracy.

**Therefore: ~5k human captions is a non-negotiable line item.** PoseScript used 6,283 (54.2 tokens
average, 1,866-word vocabulary). Collect them on *your* prompt distribution, not AMT-generic.

Cost calibration: PoseScript generated **300k captions in under 10 minutes**. HumanML3D's 44,970
human descriptions over 14,616 motions is the comparison point.

**Do not iterate a captioner on its own output.** Model collapse occurs when synthetic data
*replaces* real; *"accumulating the successive generations of synthetic data alongside the original
real data avoids model collapse"* with error bounded independent of iteration count
([2404.01413](https://arxiv.org/abs/2404.01413)). Accumulate, never recurse.

### 8d.4 ⭐ Architectural rule: rules perceive, the LLM speaks

The strongest motion captioner is **not** a learned model. **SMD**
([2604.21668](https://arxiv.org/abs/2604.21668), CC BY 4.0) computes **26 biomechanical joint angles**
in ISB anatomical frames, segments each time-series into *increases / decreases / holds / repeats N
cycles*, emits hierarchical structured text, and LoRA-finetunes an LLM to turn it into prose. It
beats every learned motion encoder by ~9 BLEU@4:

| Method | BLEU@4 | CIDEr | BertScore |
|---|---|---|---|
| TM2T | 7.0 | 35.1 | 32.2 |
| MotionGPT | 12.5 | 39.4 | 32.4 |
| LaMP | 13.0 | 39.7 | 32.7 |
| MotionGPT3 | 19.4 | 40.6 | 35.2 |
| **SMD** (rules → LLM) | **22.7** | **53.2** | **45.6** |

For calibration, COCO image captioning passed BLEU@4 ≈ 40 years ago. **BLEU@4 of 13 is not a
pseudo-labeler — never use a learned motion captioner as your annotator.**

🚨 **Do not render mocap and caption it with a VLM.** Three independent measurements:

- **UniPose** ([2411.16781](https://arxiv.org/abs/2411.16781)): on pose-captioning from images,
  **GPT-4V scores BLEU-4 7.1** vs 18.2 for a small specialist — 2.5× worse.
- **MotionBench** ([2501.02955](https://arxiv.org/abs/2501.02955), CVPR 2025): SOTA VLMs are **below
  60%** on fine-grained motion QA (chance = 0.25). *Action-order* sits at 0.30–0.39 and repetition
  counting at **0.25–0.33, i.e. chance**. Text-only GPT-4o scores 0.33 — most apparent competence is
  language prior, not perception. Verbatim: *"significantly below the threshold for practical
  applications."*
- **ActPLD** ([2509.23517](https://arxiv.org/abs/2509.23517)): MLLMs show *"consistently low
  performance"* on point-light joint displays — the stimulus closest to a rendered skeleton, which
  humans read trivially.

Expected failure modes if you try it anyway: temporal order and repetition count near chance;
view-dependent left/right and toward/away ambiguity; subtle joint angles invisible at render
resolution; and GPT-4o documented inventing hand-held objects and occluders on SMPL renders. **No
paper found does this in production.** The field caption *source video* (MotionMillion via GPT-4o,
OmniHuMo via Qwen3-VL-32B) or apply *rules to geometry*. That convergence is the answer.

### 8d.5 ⭐ Solving the abstraction gap in the director, not the pose model

The gap: a geometry-trained pose model expects *"left arm raised 45°"*; a director emits *"reaching
for a book on a high shelf."* **CoT-Pose** ([2508.07540](https://arxiv.org/abs/2508.07540)) is the
only paper attacking it directly — it trains on **239 synthetic samples**, and its own Table 3 shows
that *removing* the reasoning loss gives **better** numbers (MPJPE 115.14 vs 124.91). The bridge is
not yet an engineering result.

**But the pose model does not have to bridge it.** Translating intent into geometric prose is a pure
language task requiring no training data — an LLM already does it. Put the bridge in the director,
and the pose model only ever sees the geometry-prose distribution it was trained on. The scarce stage
stops being *semantics → pose* and becomes *geometry-prose → pose*, which is exactly what 300k
manufactured captions teach.

⚠️ **Absolute accuracy is poor regardless.** Best text→static-pose is **UniPose at 308.6 mm MPJPE**
(PoseScript baseline 318.0). That is a coarse initializer, not a shippable keyframe. Whether it
suffices depends entirely on downstream error tolerance (§8d.7).

### 8d.6 Which stages survive, with the evidence

| Stage | Data situation | Verdict |
|---|---|---|
| 1. Text → beat script | LLM emits it; free | **keep — as a control surface and training-time signal, not a generation bottleneck** |
| 2. Beat → sparse keyframe poses | the only genuinely starved stage | **keep, but change how keys enter the model** |
| 3. Keys + timings → dense motion | **self-supervised by construction** — mask any clip, learn to restore | **keep — near-free at ≤0.5 s spacing** |
| 4. Plausibility critic | **license-free** — geometry + physics | **⭐ keep — best-supported stage in the pipeline** |
| 5. Retarget to VRM | deterministic | keep |

**⭐ Stage 2 — the keyframe tax is a backbone choice, not a law:**

| How keys enter | Behavior as keys densify |
|---|---|
| Guidance bolted onto an unconditional model (GMD, OmniControl) | **collapses** — 0.212 → 0.874; 0.218 → 0.624 → 2.614 |
| Natively trained on keyframe conditioning (CondMDI) | **graceful** — unconstrained 0.2538 → K=1 **0.1551, better than unconstrained** |
| Masked-generative / latent optimization (MaskControl, TLControl) | **improves with density** — 0.077 at 1 frame → **0.054** at 196 |

**Never bolt keyframe guidance onto an unconditional model.** A natively-conditioned or
masked-generative backbone pays little or nothing, and denser conditioning can *help* by removing
ambiguity. This single choice is worth more than the rest of the pipeline design.

**Stage 3 — the sparsity budget: ~1 key/second stands. There is no cliff.**

An intermediate draft of this section halved the budget to 0.5 s on the strength of SILK's
in-betweening curve. **That was an over-correction, and the reconciliation is the useful part.**

SILK ([2506.09075](https://arxiv.org/abs/2506.09075)) on LaFAN1, **L2P** by gap:

| Method | 5f (0.17 s) | 15f (0.5 s) | 30f (1 s) | 45f (1.5 s) |
|---|---|---|---|---|
| SLERP | 0.37 | 1.38 | 2.49 | 3.45 |
| Harvey RMIB (2020) | 0.23 | 0.65 | 1.28 | 2.24 |
| **SILK (2025)** | **0.13** | **0.38** | **0.83** | **1.59** |

versus [2504.09413](https://arxiv.org/abs/2504.09413), uniformly-spaced keys over 200-frame
sequences, measuring **K-FID**:

| Key interval | ≈ @30 fps | K-FID ↓ | K-Error ↓ |
|---|---|---|---|
| 25 f | 0.83 s | 1.884 | 8.1 mm |
| 50 f | 1.67 s | 2.406 | 11.1 mm |
| 100 f | 3.33 s | 3.433 | 15.0 mm |
| 200 f | 6.67 s | 5.292 | 19.4 mm |

**Both are correct about their own metric, and only one of them is your metric.** L2P measures
divergence from *one specific ground-truth motion* — a reconstruction task, where error must grow
with gap length because the ambiguity grows. **You are not reconstructing a known motion; you are
generating a plausible one, where any manifold-valid answer is correct.** K-FID is the relevant
measure, it degrades smoothly and roughly logarithmically, and **there is no cliff anywhere out to
6.7 seconds**. At ~1 key/s you sit at K-FID ≈ 2.0 / K-Error ≈ 9 mm — the comfortable end, with ~3×
headroom to the sparsest point anyone has measured.

**⭐ And density is not free — CondMDI's FID gets *worse* as you add keys** (Table 2 above): K=1
`0.1551` → K=20 `0.2253`, **45% worse**, while keyframe error improves 5× (0.3739 → 0.0754). Foot
skate and R-precision stay flat. **Sparser keys mean more freedom and more natural motion.** There is
no quality collapse at high sparsity; there is a quality *penalty* at low sparsity, because the model
is forced to hit poses it would not have chosen. Corroborated by SFControl
([2503.15557](https://arxiv.org/abs/2503.15557), ICCV'25), which runs at **1 key per 1.5 s with only
half of six keyjoints — 0.454% of the total joint signal** — and still reaches FID 0.224.

⚠️ Note what none of this measures: *"is it the motion the text asked for."* CondMDI's R-precision is
flat at ~0.68 across all densities. Sparsity governs plausibility; **semantic fidelity is the
proposer's job** (§8d.8).

**⭐ Stage 3 — drop text from the densifier entirely.** CondMDI's unconditional synthesis (no keys,
no text) scores FID 0.2538 / R-prec **0.6450**. Add 5 keyframes and *no text*: FID 0.1731 / R-prec
**0.6823**. **A text-alignment metric improves from keyframes alone — keyframes carry the
semantics.** NINB ([2605.12778](https://arxiv.org/abs/2605.12778)) is a 2026 in-betweener that uses
**no text at any stage** and positions text-freedom as an advantage; 2504.09413 and SFControl are
text-optional; CondMDI ships a text-free *and* an unconditional checkpoint (10% text dropout, 10%
keyframe dropout, so one checkpoint covers all four conditioning modes).

This is a real architectural simplification: it confines *all* text conditioning to the proposer,
removes CLIP/T5 from the densifier, and eliminates its dependence on **captioned** mocap — which is
far scarcer than raw mocap and is the whole reason self-supervision matters here.
⚠️ No paper runs the clean "same model, same keys, text on vs off" ablation across a sparsity sweep.
Expect text to matter more as keys get sparser (at K=1 it is doing most of the work); **that curve
does not exist in the literature.**

**⭐⭐ Stage 3 — train on a realistic mask distribution. Highest-leverage deviation from standard
practice.** Every in-betweener from 2023–2026 trains on uniform random masking, verified from method
sections and repo configs. CondMDI, verbatim: *"the number of keyframes k is first sampled within the
length of the motion sequence, and then these k keyframes are randomly picked out of all the
frames."* **Uniform random masking teaches the model to expect keys that are already perfectly on the
ground-truth manifold at exactly the right time — which a proposer's keys will not be.**

The one exception, and the one to copy: Goel et al. CGF 2025
([2503.01016](https://arxiv.org/abs/2503.01016)) mines **extrema poses** (apex of a jump, etc.),
shifts each by random **Δ ∈ [−5, 5)** frames, and deletes a surrounding window. Its measured payoff
under only **±5 frames** of timing noise, HumanML3D 60-frame clips:

| | **LT (time-warp head)** | NoWarp | **CondMDI** | Interp(5) |
|---|---|---|---|---|
| Keypose Error (10⁻²) | **0.019** | 0.025 | **0.120** | 1.68 |
| L2-Pos global (10⁻¹) | **0.03** | 0.04 | **0.43** | 1.35 |
| Jitter (10⁻²) | **0.22** | 0.26 | **0.75** | 0.43 |

**CondMDI's keypose error is 6.3× worse and its global position error 14× worse** under ±5 frames of
jitter. The fix is an explicit **time-warping head plus spatial residuals**. Secondary support: IKMo
([2505.21146](https://arxiv.org/abs/2505.21146)) reports FID **0.594 without** noise-augmented
training vs **0.238 with**, under perturbed keys — a 2.5× blowup that augmentation flattens.
*(Unverified at source.)*

**⭐ Stage 4 — the best-supported component in the whole design:**

| Critic | Target metric | FID side-effect |
|---|---|---|
| **PhysDiff** ([2212.02500](https://arxiv.org/abs/2212.02500)), HumanML3D | phys-err **31.572 → 4.111 mm**; penetrate 11.291 → 0.998; float 18.876 → 2.601; skate 1.406 → 0.512 | 0.544 → **0.433 (better)** |
| PhysDiff, UESTC | phys-err **28.371 → 1.463 mm** | 12.81 → 13.27 (**worse**) |
| **MotionCritic** ([2407.02272](https://arxiv.org/abs/2407.02272), ICLR'25) | critic score −1.64 → **+2.78**; RL fine-tune = **0.23% of pretraining cost** | 0.13 → 0.18 (**worse**) |
| **AToM** ([2411.18654](https://arxiv.org/abs/2411.18654), CVPR'25) | event-level alignment up | 0.655 → **0.613 (better)** |

MotionCritic's discriminative power on 52,563 human preference pairs: **85.07%** (MDM subset) /
**81.43%** (FLAME subset), vs MoBERT 49.40/52.40, person-ground-contact heuristic 71.78/69.82,
physical-foot-contact 64.79/66.00. Public code and weights.

> 🚨 **A working critic makes FID worse.** Two of the four rows above improve their target metric
> dramatically while FID regresses. **If FID is your acceptance criterion you will delete a working
> stage.** Track physics metrics, a learned perceptual critic, and FID as three independent axes.

**Two things to drop:**

- **Hard segment stitching.** FlowMDM ([2402.15509](https://arxiv.org/abs/2402.15509)) beats
  stitching by **3.4–4.9× on jerk** (AUJ 0.13 vs TEACH 0.44, DoubleTake 0.64 on BABEL). The field's
  whole trajectory since is boundary removal: stitching → joint denoising → MotionStreamer's causal
  latents. **Generate long and refine locally; do not generate short and glue.** ⚠️ FID cannot detect
  a bad seam — FlowMDM notes the *"lack of correlation between FID and AUJ."*
- **The belief that semantic beat boundaries must be accurate.** SegMo
  ([2512.21237](https://arxiv.org/abs/2512.21237)) found **uniform, semantically meaningless
  segmentation beat both change-point detection and clustering.** Consistency mattered; semantic
  precision did not. SegMo's actual win (FID 0.045 → 0.042, R@1 0.521 → 0.553) came from using LLM
  segmentation as a **training-time contrastive alignment loss**, not an inference-time handoff.

### 8d.7 Before building any of it — the cheaper fix, and the honest caveats

**⭐ Fix text conditioning first.** **CASIM** ([2502.02063](https://arxiv.org/abs/2502.02063)) solves
the compositional-semantics problem that motivates a beat script *with no staging at all*, by
replacing pooled `[CLS]` conditioning with token-level cross-attention:

| Backbone | R@1 | FID |
|---|---|---|
| T2M-GPT → CASIM | 0.491 → **0.539** | 0.116 → **0.105** |
| MoMask → CASIM | 0.510 → **0.532** | 0.064 → **0.057** |
| MDM → CASIM | 0.455 → **0.502** | 0.489 → **0.165** |

**Every staged system trades one metric for the other; CASIM improves both, on every backbone.** It
is the only result in this entire survey that does not trade. Likewise **MultiAct**
([2605.30925](https://arxiv.org/abs/2605.30925)), a *training-free* inference-time attention
reweighting on monolithic MDM, beats STMC — the flagship staged timeline system — **83.69 vs 20.78
on human-judged alignment (4×)**. Note also that all R@1 values there are 0.03–0.11: **composite
prompts are unsolved by everyone, staged or not.**

⚠️ **Compose parts semantically or not at all.** SINC ([2304.10417](https://arxiv.org/abs/2304.10417)):
random part-composition gives **literally zero** gain (0.618 vs 0.618 baseline); GPT-guided part
assignment gives 0.647 (+3.9%). GPT-3 part-labelling accuracy: free-form 56%, with a body-part list
78%, **list + few-shot 87%** (velocity baseline 39%).

⚠️ **STMC is the most mis-cited paper in this space.** Its timeline is *user-specified or taken from
ground truth*; GPT-3 only optionally tags body parts. **There is no LLM planning stage.** Similarly,
MotionGPT/-2, MotionChain, AvatarGPT and PlanMoGPT are unified single models, not planner/generator
splits; Motion-Agent emits a sub-instruction list with no timing and no body-part fields.

**Evidence caveats, stated plainly:**

- **There is no controlled staged-vs-end-to-end study at matched data and compute for text-to-motion.**
  Everything above is indirect. This is the largest gap in the evidence base and it cuts both ways.
- **Papers With Code shut down July 2025.** There is no canonical maintained HumanML3D leaderboard;
  every ranking is self-reported.
- **MARDM** ([2409.19686](https://arxiv.org/abs/2409.19686)) retrained the Guo et al. feature
  extractor that every FID/R-Precision since 2022 depends on. **Pre- and post-2025 numbers are not
  comparable.**
- **R-Precision is exploitable** — several methods score *above real motion* (real = 0.797; Text2BFM
  reports 0.876). Any architecture argument resting on R-Precision alone is unsafe.
- **Treat any claimed HumanML3D FID improvement under ~0.02–0.03 as noise.**
- Nobody publishes **requested-vs-realized duration error.** Text2BFM's order accuracy (0.671 at
  N=3, 0.509 at N=4) is the closest proxy. **If timing fidelity is this pipeline's value
  proposition, that metric must be built here.**
- No controlled auto-caption-vs-human-caption ablation exists for *motion* — only for static pose
  (PoseScript). MotionMillion, OmniHuMo and CompMo, the three largest auto-captioned corpora, report
  **zero** caption-quality measurements.

**Licensing, updated:** MoMask (MIT), MDM (MIT), CondMDI (MIT), OmniControl (MIT), T2M-GPT and ParCo
(Apache-2.0) form a clean-*code* stack — but every released checkpoint inherits AMASS's
non-commercial terms. **FlowMDM and STMC — the two systems most relevant to composition and timeline
control — are themselves non-commercial licensed.** CoMo is CC BY-NC-**ND** (no derivatives at all).
§8d.2 is the only route around this.

### 8d.8 ⭐ The proposer: retrieve, do not generate

**Text→pose generation is not competitive.** SOTA on the standard benchmark:

| Text→pose generation | MPJPE ↓ | PA-MPJPE ↓ | FID ↓ |
|---|---|---|---|
| PoseScript | 318.0 mm | 161.3 | 0.075 |
| **UniPose** ([2411.16781](https://arxiv.org/abs/2411.16781), CVPR'25) | **308.6 mm** | 171.1 | **0.038** |

**308 mm of error per joint — roughly the length of a femur.** For calibration, PHC tracks *full
motion* at 48 mm. The field has been optimizing retrieval-flavoured proxies (RT2P) and FID while
absolute joint error stayed enormous. Note also that **ChatPose scores 17.6 RT2P@5 against the 2022
PoseScript baseline's 40.4** — LLM-based pose generation is not automatically better.

**Retrieval, same task:**

| Method | mRecall | Gallery |
|---|---|---|
| PoseScript ([2210.11795](https://arxiv.org/abs/2210.11795)) | 45.3 | 1,234 |
| **PoseEmbroider** ([2409.06535](https://arxiv.org/abs/2409.06535), ECCV'24) | **78.7** (T→P) | ~10k |
| **CLEP** (CVPR 2026) | 75.69 | **2M** |

Retrieval also guarantees something generation cannot: **every retrieved pose is a real human pose**,
so there is nothing off-manifold for the critic to catch. And it is cheap — PoseEmbroider reports
*"below 4 minutes for a 10k-sized set"* on a V100; at 10M poses this is a sub-second FAISS ANN
lookup. Supporting evidence that text genuinely informs 3D pose: adding a text cue to image-based
SMPL regression improves PA-MPJPE **49 → 44 mm**.

⚠️ **The unproven assumption is gallery scale.** TMR ([2305.00976](https://arxiv.org/abs/2305.00976))
shows *motion*-level R@1 collapsing **67.2% → 5.7% purely by growing the gallery from 32 to 4,380**.
Pose space is lower-dimensional and far more densely covered, so CLEP's 2M result is plausible — but
**that number came from a search snippet; the CVPR PDF returns 403 and there is no arXiv mirror.
Single-source, unverified.** Also note AMASS flattened is only **~4–9M frames**; exceeding 10M
requires MotionMillion (~216M frames) or Motion-X (15.6M), both non-commercial.

**⭐ Retrieval-augmented generation wins where it matters most — out of domain.** VimoRAG
([2508.12081](https://arxiv.org/abs/2508.12081), NeurIPS 2025) gives the same backbone a 3.8× FID
improvement in-domain (MotionGPT 0.501 → 0.131), which merely *ties* the best non-RAG models. Zero-shot
on IDEA400 it is not close:

| IDEA400 zero-shot | FID ↓ |
|---|---|
| **VimoRAG** | **2.388** |
| T2M-GPT | 5.359 |
| MotionGPT | 6.202 |
| ReMoDiffuse | 9.639 |

**A 4× margin over the next best.** For a system fielding open-ended user text, that is the relevant
number. Retrieval bank: HcVD, 425,988 videos. ⚠️ Code *"will be open soon"* — **not released**.

**Critic:** **DPoser-X** ([2508.00599](https://arxiv.org/abs/2508.00599), ICCV 2025 **oral**) — a
diffusion whole-body pose prior supporting completion, denoising and IK. **MIT, weights released**
([moonbow721/DPoser](https://github.com/moonbow721/DPoser)). Snapping a proposed key onto the pose
manifold before densification is cheap insurance and the natural home for the §5 verifier.

**⭐ Cost of the retrieval embedding — this is now the main trainable component, and it is cheap.**
PoseScript's `PoseText` is a complete published recipe: VPoser-style pose encoder (156-d axis-angle →
512-d, ~1–2M params) + **frozen DistilBERT** → linear 768→512 → 4-layer transformer (d=512, 4 heads,
FF 1024, ~8.4M). Symmetric batch-based contrastive loss with learned temperature.

| | |
|---|---|
| **Trainable params** | **10–12M** (4× smaller than MoMask; frozen DistilBERT adds 66M non-trainable) |
| **Training** | **~13–27 GPU-hours in bf16**, ~1–2 days fp32 |
| **VRAM** | **<8 GB** at batch 512 |
| Schedule | pretrain 1000 ep / batch 512 / lr 2e-4 on 100K poses × 3 captions ≈ 586K iters; finetune 200–300 ep / batch 32 on the 6,283 human captions (**under an hour**) |

**⭐ The trick that makes it cheap: DistilBERT is frozen, so cache its per-token hidden states once**
(300K captions × 54 tokens × 768 × 2 B ≈ **24.9 GB** as an `np.memmap`; one-time pass ~5 min in
bf16). The text branch then collapses to the 4-layer topping. That is **1/10th to 1/50th the cost of
training MoMask** — roughly one MDM eval run.
⚠️ Contrastive loss wants large batches because the batch *is* the negative pool. Naive gradient
accumulation shrinks that pool and costs recall; prefer batch 256–512 in bf16, or use gathered
negatives / cross-batch memory if you must accumulate.

**⚠️⚠️ Correction to the index size implied above — make it ~20× smaller.** PoseScript's own corpus
construction argues against a 5–10M index: they *"excluded the first and last 25 frames of each
sequence"*, *"sampled only one pose from every 25 frames"*, then ran **farthest-point sampling** to
land on 100K. AMASS at 30 fps is ~4.9M frames, so **"flatten all of AMASS" is raw data at ~50×
redundancy** — consecutive frames are near-duplicates and a nearest-neighbour query returns forty
views of the same instant.

| Index | 10M × 512-d | Latency | Note |
|---|---|---|---|
| `IndexFlatIP` fp32 | 20.5 GB RAM | ~50 ms (BLAS, 8–16 cores) | exact |
| `IVF_SQ8` | 5.1 GB | ~1 ms | near-exact |
| `IVFPQ` m=64×8bit | **640 MB** | 0.5–2 ms | recall@1 ~0.80–0.95 w/ re-rank |
| **500K flat fp32 (recommended)** | **1.0 GB** | **2–5 ms** | **exact, no approximation at all** |

**Subsample 1-in-25 then FPS/k-means to 100K–500K distinct poses.** A 500K exact flat index beats an
approximate 10M index on every axis, and it is exactly the corpus construction the source paper
validated. **FAISS is MIT** — one of the few permissive components in this stack.

**⚠️ Honest ceiling: mRecall 45.3% / R@1 22.3% / R@5 50.1% / R@10 62.9%** on *human-written* captions
(base 40.9 → +transformer 43.3 → +mirroring 45.3). The intended pose is in the top-10 about 63% of
the time — workable as a **proposer feeding a refiner**, not as a one-shot selector.
🚨 **Do not quote PoseScript's ~72.8% mRecall on *automatic* captions as if it applied to real user
text**, and note §8d.8's 78.7 figure is PoseEmbroider on BEDLAM-Script — a different benchmark.
TMR++'s cross-dataset result warns this degrades further out-of-domain.

**⭐ Retrieval buys physical plausibility for free.** Every candidate is real mocap, so self-collision,
joint ROM, bone-length consistency and balance **pass by construction**. That collapses proposer
evaluation to semantic alignment alone — the one thing physics heuristics cannot measure anyway
(§8d.12) — and repurposes the physics metrics as a **regression guard on the densifier** instead.

🚨 **But it sharpens the licensing problem rather than softening it.** A retrieval system does not
merely *train* on AMASS — it **redistributes AMASS poses verbatim at inference time**. That is a
distribution question and strictly harder than the generative case. For any commercial path the index
must be built from **100STYLE (CC BY 4.0, commercial use explicitly permitted** — 4M frames /
1,125 min / 100 styles / 60 fps / 28-bone BVH, **locomotion only)** or your own capture.
⚠️ Also correcting §8d.2: **MotionMillion's repo requires SMPL+H and DMPL downloads**, so even its
Apache-2.0 *code* is not a clean commercial path.
⚠️ And retrieval cannot produce poses outside the corpus — novel or stylized actions fail with **no
graceful degradation.**

### 8d.9 ✅ RUN — the answer is: it propagates

**No paper anywhere injects angular noise into keyframe joint rotations and measures downstream
degradation.** The literature has tested timing error (±5 frames, 2503.01016), spatial incompleteness
(2509.16064), and positional perturbation (IKMo). **Per-joint rotational error of 10–20°: nothing.**

That was the load-bearing assumption of this entire architecture — *if a proposed keyframe is 10–20°
off per joint, does the densifier absorb it or propagate it?* **It is now measured.** Code, method
and full tables: `experiments/keyframe-noise/` (see `RESULTS.md`). CondMDI `condmdi_randomframes`
@750k, 32 motions × 5 angles × 5 tolerances, paired contrasts, geometric metrics.

**🚨 The densifier does not absorb keyframe error. It propagates it, at every tolerance.**
Output-to-clean joint error tracks the injected keyframe displacement to within 4% (θ=30° injects
236 mm; the best result over all tolerances is 226 mm), and whole-motion divergence from the
unperturbed reference tracks it roughly one-for-one. **Any pose-accuracy budget for the proposer is
the pose-accuracy budget for the finished animation.** There is no downstream cleanup.

**🚨 The prior expectation in this section was wrong, and this corrects it.** "Tolerant conditioning
at c≈0.5–0.85 is *designed* to absorb exactly this" — it is not, and it does not. Tolerance shifts
the whole quality curve down (jerk drops ~67% at every θ going from hard imputation to native
conditioning) but leaves the **degradation slope unchanged**: θ=0→30 costs +158% jerk at c=1.0 and
+157% at c=0.0. Tolerance buys a better baseline, not robustness. §8b.2's `R-NoTolerance` ablation
is real but means something narrower than it appeared to.

**⭐ Two distinct failure modes, separable by metric.** A control holding θ fixed while varying only
whether consecutive keyframes err in the *same* direction:

| θ=30°, native conditioning | independent error | coherent (systematic) error |
|---|---:|---:|
| foot skate | 0.230 | **0.119** (= the θ=0 baseline) |
| jerk (m/s³) | 457 | 362 |

**A 30° systematic per-joint bias produces no foot-skate increase at all.** Foot skate measures
keyframes *disagreeing with each other*; jerk measures the pose being *wrong*. Track both — they say
which problem you have. (This also refines §8d.12: foot skate is a poor discriminator of pose error,
but a strong one for inconsistency.) Under hard imputation the distinction vanishes — the override
generates the jerk itself and the two modes become indistinguishable, so **c=1 destroys the
diagnostic as well as the motion.**

**Design consequences:**
- **Do not use hard imputation.** At θ=0, with *perfect* keyframes, c=1.0 vs c=0.0 triples jerk
  (518→169) and raises foot skate 65%. The cost of relaxing it is 62.5 mm of keyframe error — the
  model declines to hit keyframes it is not forced to hit. Knee of the trade-off is **c ≈ 0.2–0.5**.
- **10° per joint is not a safe budget** (83 mm displacement, +38% jerk). ~5° is.
- **Prefer retrieving a contiguous clip and retargeting it once** over retrieving each keyframe
  independently — systematic error is markedly cheaper than independent error. This strengthens the
  §8d.8 retrieval proposer against a generative one on a second axis: retrieved poses are both
  on-manifold *and* consistently wrong rather than independently wrong.

**⚠️ What this does not establish.** References are model samples, not AMASS mocap (so this is the
easy case, and real-mocap degradation should be at least this bad); absolute values are not
comparable to the paper because we generate a fixed 196 frames per caption where CondMDI's eval
respects true sequence length; no FID or R-precision; one keyframe density (2.04 keys/s); root
trajectory never perturbed. Within-sweep contrasts are sound — every cell shares prompts, lengths
and diffusion noise.

**⚙️ Two implementation traps, both of which silently produce a null result.** (1) This checkpoint
predicts x₀ (`ModelMeanType.START_X`), so the live imputation override is in `p_mean_variance`, not
the `impute()` closure in `p_sample` — that closure is on the EPSILON path and never executes.
Patching only the closure yields a sweep where every tolerance returns byte-identical output.
(2) A HumanML3D 263-vector encodes each pose four times (rotations, ric positions, local velocities,
foot contacts). Perturbing `[67:193]` alone produces a self-contradictory keyframe and measures
robustness-to-garbage instead of robustness-to-angular-error; the rotations must be pushed through
forward kinematics and every derived channel rebuilt.

**✅ PyTorch3D is NOT a CondMDI dependency — verified from the repo, not the README.**
`requirements.txt` contains no pytorch3d and the README's install block has no such line.
`utils/rotation_conversions.py` is already a **vendored pure-torch copy** (its
`# Check PYTORCH3D_LICENCE before use` header is attribution boilerplate and the sole reason
PyTorch3D is named anywhere); it imports only `functools`, `typing`, `torch`, `torch.nn.functional`
and defines all 23 rotation converters. `model/rotation2xyz.py` and `model/smpl.py` import no
pytorch3d. **`eval/eval_humanml.py` — the exact file this experiment runs — imports nothing external
at all** beyond the repo's own modules; no pytorch3d, no smplx, no chumpy on the evaluation path.

**The real blockers are smaller and different:**
- 🚨 **Strip `torch==1.13.1` and the four `nvidia-*-cu11` wheels.** Install ROCm torch first, then
  `pip install -r requirements.txt --no-deps`. Same trap as `human_body_prior`'s `torch>=2.5,<2.6`.
- `numpy==1.21.5` is the tightest pin; relaxing it means fixing a few `np.float`/`np.int` aliases.
- **`chumpy` and `human-body-prior` are only needed under `visualize/`** for SMPL mesh work. This
  experiment never touches it — skip both. Stick-figure matplotlib rendering already ships in-repo.
  - ❌ **Half wrong, as run.** `human-body-prior` is indeed unnecessary, but **`smplx` is imported on
    the model-construction path**, not just under `visualize/`: `sample/conditional_synthesis.py` →
    `utils/model_util.py` → `model/mdm.py` → `model/rotation2xyz.py` → `model/smpl.py` →
    `from smplx import SMPLLayer`. `Rotation2xyz.__init__` then eagerly builds `SMPL()`, which
    unpickles `SMPL_NEUTRAL.pkl` — **a chumpy pickle**, and chumpy does not import on Python ≥3.11 /
    numpy ≥1.24. Sampling dies at model construction, before any `visualize/` code is reachable.
    **Fix: make SMPL construction lazy** (`experiments/keyframe-noise/condmdi-tolerance.patch`) —
    nothing on the `hml_vec` path needs a body model, so deferring it removes the dependency
    entirely rather than fighting chumpy. `smplx` itself installs fine and is pure Python.
- `spacy==3.3.1` + `en_core_web_sm` and OpenAI CLIP **are** required (CLIP is MDM/CondMDI's text
  encoder).
- The evaluator is the classic Guo all-263-dims one — **Protocol 1** of §8d.13. Correct for a
  within-protocol sweep where only relative movement matters; **do not compare the absolute FIDs to
  any paper on MARDM's essential-dims evaluator.**

**⚠️ The actual cost driver is inference time, not setup.** CondMDI runs **54.4 s per sample** (1000-step
DDPM, on a 2070). The standard protocol wants the full 4,384-text test set × 20 repetitions; times 4
perturbation levels × N tolerance values that is **weeks of GPU time, not hours.** Cut to a fixed
1,000-sample subset × 3 repetitions and **report the reduced protocol explicitly**, or use
DDIM/respacing to cut steps.

> ✏️ **As-run correction: that estimate was ~16× pessimistic, because 54.4 s/sample is unbatched.**
> On the 9070 XT, a batch of 16 × 196 frames × 1000 DDPM steps takes **75 s total — 4.7 s/sample**,
> at ~2.5 GiB. The full 5θ × 5c sweep at n=32 was **~65 min**, not weeks, with no DDIM respacing and
> no protocol reduction beyond sample count. Batch the perturbation levels, not the denoising steps.
> The remaining reason to subsample is the *evaluator* (FID/R-precision over 4,384 texts), not the
> sampler.

**⚠️ MoMask is not a substitute** — no keyframe-conditioning mechanism and no tolerance parameter, so
the experiment does not transfer. Stay on CondMDI.

**⭐ Track more than FID.** Perturbing joints by 5–30° will move plausibility metrics (NRDF distance,
joint-ROM violation) and keyframe error **long before it moves FID** — those are the sensitive
readouts here. CondMDI's published keyframe errors (K=1 0.3739, K=5 0.1789, K=20 0.0754) give a
calibrated axis.

### 8d.12 🚨 Training-free plausibility metrics barely beat chance

**This corrects §8c.6's enthusiasm for geometry-only scorers.** MotionCritic's Table 1 measures each
metric's pairwise accuracy at predicting human preference — **chance = 50%**:

| Metric | MDM set | FLAME set |
|---|---|---|
| **MotionCritic** (learned) | **85.07%** | **81.43%** |
| Person–Ground Contact | 71.78% | 69.82% |
| Jerk | 65.48% | 65.84% |
| PFC (EDGE) | 64.79% | 66.00% |
| Acceleration | 64.26% | 66.67% |
| Joint AE | 62.73% | 58.37% |
| Pose-NDF | 55.13% | 53.07% |
| Floating | 55.1% | — |
| **Foot–floor penetration** | **53.61%** | 55.56% |
| **Foot skating** | **52.46%** | — |
| **MoBERT** (learned!) | **49.40%** | 52.40% |

Corroborated independently by 4DHumanQA (48 in-lab subjects, SROCC vs MOS): **MPJPE 0.436**, global
translation 0.389, LDJ smoothness 0.267, Hausdorff 0.177, **foot contact 0.144**.

And MoBERT's own correlation study ([2309.10248](https://arxiv.org/abs/2309.10248), 1,400 AMT-rated
pairs) is blunt about the standard metrics at the **sample** level: **R-Precision 0.036 / 0.042,
MM-Dist 0.025 / 0.014, Root AVE −0.013 / 0.007.** Verbatim: *"none of the metrics currently used for
this task show even a moderate correlation with human judgments on a sample level."* Model-level
correlation is fine (R-Precision 0.816) — **these metrics rank systems, they do not rank samples.**

**→ Treat physics metrics as artifact detectors and regression guards, never as quality scores.**
Only person–ground contact and a *learned* critic carry usable signal.

⚠️ **Foot skating has six mutually incompatible definitions; cross-paper numbers are meaningless.**
The dominant one (GMD → OmniControl → most 2024–26 work) is `calculate_skating_ratio` in
[guided-motion-diffusion](https://github.com/korrawe/guided-motion-diffusion) (MIT): foot skids
>2.5 cm while foot height <5 cm. **Undocumented gotchas:** hardcoded fps=20, y-up, joints **10/11 are
toes not ankles**, an **undocumented 5-frame uniform-filter smoothing**, and two stale source
comments (`thresh_vel = 0.50 # 20 cm/s` is actually 50 cm/s) that will give you a different metric if
you reimplement from the comments.
🚨 **GT skating on HumanML3D is 0.0733** — so OmniControl's 0.0547 **scores better than real human
motion.** OmniControl's "Real = 0.000" row is a table artifact. **Target your GT's value, never zero.**

**What to build instead (~1 day, permissive, CPU-only, and nobody has released it):**
1. **Capsule self-collision** — ~20 capsules on SMPL limbs, pairwise distance, skip adjacent pairs;
   `trimesh`/`numpy`. (⚠️ `torch-mesh-isect` **requires CUDA and will not build on ROCm**;
   `selfcontact`/TUCH is non-commercial.)
2. **Empirical joint limits** from the 0.1/99.9 percentiles of *your own* corpus — training-free,
   honest, dataset-matched, no license.
3. **Ground contact + penetration** against your GT's 99th percentile.
4. **⭐ Static balance** — signed distance from the mass-weighted CoM's horizontal projection to the
   convex hull of contact vertices; negative means the pose falls over. **No released
   support-polygon/balance checker for human poses exists at all.**

**Report every one against your GT dataset's distribution, never against zero.**
⚠️ **Bone-length consistency is identically zero if you emit SMPL rotations and run FK** — it only
means anything if you emit joint *positions*.

**Physics simulation** is the only grounded test, and there is exactly one ROCm-viable route:
**[SMPLSim](https://github.com/ZhengyiLuo/SMPLSim) is BSD-3-Clause and runs on MuJoCo (CPU)**. PHC
needs IsaacGym + CUDA. ZMP is effectively unused for evaluating generated human motion, and SMPL
ships no segment masses.

### 8d.13 The three incompatible HumanML3D protocols, verified

**Axis 1 — all-263-dims vs essential-dims evaluator.** HumanML3D's 263-d vector has 7 feature groups
but **only the first 4 render the motion**; joint rotations, local velocities and foot-contact flags
are redundant. Guo et al.'s evaluator scores all 263. Per MARDM
([2411.16575](https://arxiv.org/abs/2411.16575), CVPR 2025): **noise on the *redundant* dims alone
moves FID 0.116 → 21.032 (≈180×)**, versus 2.021 for the same noise on essential dims. Re-scored:
T2M-GPT **0.115 → 0.335** (R@1 0.497 → 0.470), MDM 0.481 → 0.518. *"The gap between VQ and diffusion
methods closes substantially."* **This is a live fork** — ACMDM reports R@3 **0.713** on the new
evaluator while MoMask reports **0.807** on the classic one. Different units.

**Axis 2 — evaluator drift within the "same" protocol.** MLD's FID is **0.473** in Light-T2M's table
and **0.673** in MLD's own paper — 42%, unexplained. Style-SALAD's re-eval of LoRA-MDM gives
SRA 42.05 / FID 0.607 where LoRA-MDM's own paper gives 27.1 / 0.32.

**Axis 3 — entirely incomparable feature spaces.** FID lives in the evaluator's latent space, so
SMooDi's ~1.6, PersonaBooth's ~3 and MotionMillion-Eval's 10–31 are simply not on MoMask's 0.045
scale.

**Under-documented sub-axes:** R-Precision is computed within retrieval batches of **32** (1 correct +
31 mismatched), not full-set retrieval — MLD never states the pool size. Results are **mean ± 95% CI
over 20 runs**; some papers run once.

🚨 **Saturation is complete. GT scores FID 0.002 / R@1 0.511 / MM-Dist 2.974 — and MoMask reports
R@1 0.521 and MM-Dist 2.958, better than real motion on both.** Top-of-leaderboard separation
(Light-T2M 0.040 vs MoMask 0.045 ± 0.002) is inside the confidence intervals.

**⭐ Cheapest evaluation win: judge keyframes, not clips.** Single-pose 2AFC takes ~4–5 s vs ~12 s for
a clip, so the same $50 buys **~1,500–2,000 pose judgments instead of ~500** — 3× the power, and it
sidesteps every temporal-metric incompatibility above.

**Human-eval reality check:** **not one generation paper reports a confidence interval, a
significance test, or verbatim question text.** MDM used 31 raters and reports **42.3% preference
against ground truth**; MoMask 42 AMT Masters, 42% vs GT; FlowMDM, OmniControl and MotionLCM ran no
user study at all. The one fully-specified copyable protocol is MoBERT's (25 gold questions, 5
embedded per HIT, reject if >2 of 5 deviate by >1 point, forced full-video watch, $1.25/HIT).
**Power:** for 80% power at α=0.05 against 0.50 you need N≈47 at p=0.70, 85 at 0.65, **194 at 0.60**.
A $50 design — 40 prompt pairs, 30 raters × 16 comparisons = 480 judgments — gives effective N≈190,
**±7 pp**. Include a ground-truth-mocap anchor so the number is comparable to MDM's 42.3%.

### 8d.14 Training economics: compute is not the bottleneck

| Model | Params | Authors' hardware | Wall-clock | 16 GB verdict |
|---|---|---|---|---|
| **Light-T2M** | **4.48M** | 2× 3090 Ti | — | ✅ 1–2 days |
| **MoMask** | 44.85M | not reported | not reported | ✅ **best pick**, est. 1.5–3 days |
| **MDM** | 17.88M | **1× RTX 2080 Ti** | **~3 days** | ✅ ~4–5 days (⚠️ eval alone ~20 h) |
| MotionLCM | MLD's 25M | 1× RTX 4090 | — | ✅ distillation only, <4 GB |
| FlowMDM | — | 1× RTX 3090 | 2 d / 4 d | ⚠️ ~6–9 days / 2–3 weeks |
| MLD | ~26M | **8× V100** (~96 GPU-h) | 12 h | ⚠️ **download it, don't train it** |
| T2M-GPT | 247.80M | 1× V100-32G | **92 h** | ⚠️ ~10–16 days; dominated by MoMask |
| CondMDI | — | **1× A100**, 1M iters | — | ❌ ~3–6 weeks to *train* (fine to *use*) |
| MotionGPT | 60–770M | **8× V100** | — | ❌ thousands of GPU-h |
| MotionMillion 3B/7B | 1–7B | **5 nodes × 8×80 GB** | 10⁴–10⁵ GPU-h | ❌ **even inference fails** (7B bf16 ≈14 GB + T5-XL ≈6 GB > 16 GB) |

**⭐ Scale is not the axis that matters: Light-T2M at 4.48M scores FID 0.040, beating MoMask's 0.045
at 44.85M, and T2M-GPT at 248M is 3× worse than MoMask at 45M.** The whole frontier is 4.5M–45M
params. Calibration at ~7 fp32 TFLOPS: MDM ≈ 0.64 s/step at 30% MFU → 500K steps ≈ 3.7 days, matching
the authors' 3 days on a 2080 Ti. **bf16 autocast on RDNA4 WMMA is the single highest-leverage
optimization** and is numerically safe at these sizes. Note that at these params on ≤200 tokens you
are often **kernel-launch/dataloader bound, not FLOPS bound** — and ROCm's launch overhead is worse
than CUDA's.

**If you ever do finetune:** nobody has published a text-to-motion finetune below **~11 minutes of
mocap per concept**, and **zero** papers report metrics below 100 clips. LoRA-MDM
([2503.19557](https://arxiv.org/abs/2503.19557), **code MIT, weights released**) is the closest
match — rank-5 LoRA on attention (~2% of those params), 4,000 steps, lr 1e-5, single L40S. Its
ablations are the useful part: rank trades style for text monotonically (r=3 → SRA 62.1/R 0.814;
r=10 → 79.8/0.775), and **the prior-preservation batch source is decisive — self-generated motions
give R-precision 0.631 vs 0.789 for real HumanML3D data. Never self-distill the prior batch.**
⚠️ Most "few-shot" claims in this literature mean one-shot *at inference* after training on the full
style corpus (SMooDi, Style-SALAD, PersonaBooth). **No paper compares finetuning against
from-scratch at small data** — state that as a gap, not a preference.

---

### 8d.10 Two tasks that are routinely conflated

Most published "keyframe control" comparisons are invalid because two different tasks share the name:

- **Task 1 — full-pose in-betweening.** Condition = complete poses at sparse frames. *This is the
  project's task.* CondMDI, NINB, Goel et al., 2504.09413, AnchorRoute.
- **Task 2 — sparse joint-xyz spatial control.** Condition = 3D positions of a few joints over time.
  OmniControl, MotionLCM, TLControl, MaskControl, GMD.

**None of the nine major Task-2 papers compares against CondMDI at all** — not an oversight; they
measure something else. Do not read MaskControl's FID against CondMDI's, and treat the §8d.6
"how keys enter" table as a claim about *mechanism* (guidance vs native vs masked-generative), not a
cross-comparable leaderboard.

**Task 1 current best — AnchorRoute** ([2605.14716](https://arxiv.org/abs/2605.14716), May 2026):
frozen transition-masked-diffusion generator plus a "RouteSolver" refinement head, **only 1.2M
trainable params** (vs 12–30M for alternatives), CC BY 4.0. Its Table III exposes the
generator/solver trade-off directly — RS500 drives control error to 0.006 m at the cost of FID
(0.070 → 0.153). **That knob is the tolerance dial, exposed as a hyperparameter.**
⚠️ Its reported CondMDI baseline (control error **0.507 m**, vs CondMDI's own 0.0422 for sparse VR
joints) is a >10× degradation that implies CondMDI was run out-of-distribution — spatially-sparse
keyjoints instead of the full-pose keyframes it is built to impute. **Read AnchorRoute's margin over
CondMDI with that caveat.**

⚠️ **NINB's advertised repo is empty** — `Coondinator/NINB`, created the day of the arXiv post, one
zero-byte README, never pushed. **Not code-available**, whatever the paper says. Verify code
availability by API, never by the paper's claim.

### 8d.11 Physics: skip it, at least initially

Real HumanML3D motion has foot-skate ratio **0.000**; CondMDI produces **0.0806–0.0936**; AnchorRoute
and SFControl reach **0.060–0.061**. **The 2026 kinematic SOTA already cuts foot skate ~35% below
CondMDI with no physics at all**, via a 1.2M-parameter solver head. CondMDI's own authors agree the
cheap fix suffices: *"minor footskate and motion jitter... could likely be addressed with an
appropriate footskate or smoothness loss."*

The heavyweight option, **PHC** ([2305.06456](https://arxiv.org/abs/2305.06456)):

| Split | Success ↑ | MPJPE-global ↓ | MPJPE-local ↓ |
|---|---|---|---|
| AMASS-Train (11,411 seqs) | **98.3%** | 37.7 mm | 27.5 mm |
| AMASS-Test (140 seqs) | **95.7%** | 48.2 mm | 31.1 mm |

Runs >30 FPS at inference, but costs *"around a week to train all primitives and composer"* on a
single **A100** with Isaac Gym — not reachable on this project's hardware. **And the trade is bad on
its face: a physics pass replaces your motion with a simulator's re-enactment, injecting 30–50 mm of
tracking error everywhere to remove <0.1 of foot-skate ratio.** Failures concentrate in *"highly
dynamic motions such as high jumping and back flipping"* — exactly the motions worth having.

**Order of operations:** (1) start from a densifier already at foot-skate ~0.060; (2) add a
lightweight solver/IK contact pass in the AnchorRoute mould; (3) only then consider physics — and
only via **ProtoMotions** ([NVlabs/ProtoMotions](https://github.com/NVlabs/ProtoMotions), **Apache-2.0**,
the sole permissive option), never PHC's **CC BY-NC-SA** weights. ⚠️ ProtoMotions' Apache-2.0 header
covers the framework; the SMPL/SMPL-H notices in `legal/` carry their own non-commercial terms.

---

## 9. Verification caveats

All five surveys exhausted their web-search budgets partway and finished on direct fetches. Known
uncertainties carried forward:

- **Metric tables were mis-parsed on first pass in several papers** (MoLingo, AnyMo, Motion-Agent,
  DiMo). Hand-verify anything load-bearing.
- **Repos with no LICENSE file are all-rights-reserved**, not permissive: SALAD, MoGenTS
  (conflicting signals — GitHub UI says MIT, direct fetch found no file), ScaMo, M³GPT, MOGO,
  UniTalker, DiffPoseTalk, VOCA, Gelina.
- **Never released despite claims:** PhysDiff (NVlabs 404), Motion Mamba (404, independently
  confirmed by Light-T2M's authors), HoloGest (404 despite the paper claiming release).
- **Unverified numbers:** MotionPCM's FID contradicts its own abstract; MotionFlux's 0.005 s
  contradicts its own speedup claim. Both have no code.
- **LaMoGen is CVPR 2026**, not NeurIPS 2025 as search results claim (verified against the CVPR
  virtual site).
- **`MotionEdit` (2512.10284) is *image* editing**, not 3D motion — easy to confuse by name.
- **Token-count arithmetic in §1** is derived (hours × 30 fps ÷ 4), not quoted from papers.
- **ROCm compatibility for every model here is untested** — inferred from architecture, not verified.
- Several 2026 arXiv IDs were fetched and confirmed real, but the 2026 sweep is a sample, not
  exhaustive; NeurIPS/ICLR 2026 proceedings were never reached.

**Best ongoing index:** [github.com/Zilize/awesome-text-to-motion](https://github.com/Zilize/awesome-text-to-motion)
(comprehensive, though its year/venue attributions contain errors).
