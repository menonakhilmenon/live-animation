# Keyframe angular-noise tolerance sweep (research doc §8d.9)

**The question.** The multi-stage plan in `docs/research-text-to-animation.md` §8d assumes a
keyframe densifier absorbs error in the keyframes an upstream stage proposes. No published paper
measures that. The literature has tested keyframe *timing* error (±5 frames, 2503.01016), spatial
*incompleteness* (2509.16064) and *positional* perturbation (IKMo). Per-joint **rotational** error
of 5–30° — the error a retrieval index or an LLM director would actually make — has no published
answer. This measures it.

**The answer, in one line.** See `RESULTS.md`.

## What runs

| file | role |
|---|---|
| `preflight.py` | gfx1201 numerical sanity: fp32 GEMM parity, ROCm #6595, #6299, SDPA fwd+bwd, hipBLASLt agreement |
| `patch_condmdi.sh` | makes the 2023-era CondMDI checkout run on numpy 2 / torch 2.13 / ROCm, then applies `condmdi-tolerance.patch` |
| `condmdi-tolerance.patch` | the two semantic changes: a `keyframe_tolerance` knob on imputation, and lazy SMPL construction |
| `kfnoise.py` | angular perturbation with consistent feature rebuild, plus the geometric metrics |
| `test_kfnoise.py` | self-tests for the above — run this before trusting any sweep output |
| `run_sweep.py` | the θ × c sweep driver |
| `analyze.py` | turns a sweep CSV into the tables in `RESULTS.md` |

## Method

1. **References.** Generate N motions from HumanML3D test-split captions with the keyframe
   conditioning empty. HumanML3D's `new_joint_vecs/` requires an AMASS registration we have not
   done, so model samples stand in for mocap. Stated as a limitation, not hidden: the references
   are on the model's own manifold, so absolute degradation is optimistic. The *shape* of the
   curve across θ and the tolerance trade-off are what this measures.
2. **Perturbation.** Rotate every joint by exactly θ° about a random axis, independently at each
   keyframe, slerped in between. Then run forward kinematics and rebuild the ric, velocity and
   contact channels from the resulting positions.

   This last step is the whole point. A HumanML3D 263-vector is redundant — rotations (67:193),
   root-relative positions (4:67), local velocities (193:259) and foot contacts (259:263) all
   describe the same pose. Perturbing rotations alone produces a self-contradictory keyframe, and
   conditioning on one measures robustness-to-garbage rather than robustness-to-angular-error.
3. **Regeneration.** Condition on those keyframes at tolerance c, reusing the *same* diffusion
   noise as the reference, so every cell differs from the reference in exactly one place.
4. **Metrics.** Keyframe error against both the perturbed and the clean keyframes, foot-skate
   ratio, jerk, ground penetration, and divergence from the reference.

### What tolerance means here

CondMDI feeds keyframes to the model through two independent channels:

- **native conditioning** — `obs_x0` / `obs_mask` go into the network itself. Soft: the model may
  disagree with a keyframe. This is what the paper's headline numbers use.
- **imputation** — the predicted `x0` is hard-overwritten at the observed entries on every
  denoising step. A law, not a target.

`keyframe_tolerance` c blends the second: `c = 1` is the upstream hard override, `c = 0` disables
imputation and leaves only native conditioning. Note that this checkpoint predicts `x0`
(`ModelMeanType.START_X`), so the live override is in `p_mean_variance`, **not** the `impute()`
closure in `p_sample` — that closure is on the EPSILON path and never executes here. Patching only
the closure produces a sweep in which every tolerance gives byte-identical results.

## Validation

The pipeline is checked two ways, and both matter:

- `test_kfnoise.py` verifies the ric encode/decode inverse round-trips to 1.2e-7 m, that the
  injected rotation error is exactly θ at keyframes, and that the root is untouched.
- Order-of-magnitude agreement with the paper. The checkpoint's own eval log
  (`save/condmdi_randomframes/eval_humanml_cond_*.log`) reports foot skate 0.0938 and keyframe
  error 0.119 m. Our θ=0, c=0 cell — the closest analogue, since the paper's headline numbers use
  native conditioning with no imputation — gives skate 0.137 and keyframe error 0.068 m.

  **Do not read this as a reproduction.** A 4-motion probe landed on 0.0936 and I briefly took that
  for an exact match; at n=16 it moves to 0.137, so the agreement was small-sample luck. Two known
  differences explain the gap: we generate a fixed 196 frames for every caption, where the paper's
  eval respects each sequence's true length (padding a short caption out to 9.8 s invites idling and
  foot shuffle), and our references are model samples rather than mocap, which lowers keyframe error
  for the same reason it inflates skate. **Absolute values here are not comparable to the paper.
  Within-sweep contrasts are, because every cell shares prompts, lengths and diffusion noise.**

## Reproducing

```bash
bash experiments/keyframe-noise/patch_condmdi.sh
.venv-rocm/bin/python experiments/keyframe-noise/preflight.py
CUDA_VISIBLE_DEVICES= .venv-rocm/bin/python experiments/keyframe-noise/test_kfnoise.py
PYTORCH_HIP_ALLOC_CONF=roundup_power2_divisions:16 \
  .venv-rocm/bin/python experiments/keyframe-noise/run_sweep.py \
    --num_samples 16 --thetas 0,5,10,20,30 --tols 1.0,0.5,0.2,0.05,0.0 --tag full
.venv-rocm/bin/python experiments/keyframe-noise/analyze.py \
  experiments/keyframe-noise/results/full-*.csv
```

Needs `third_party/CondMDI` checked out with `save/condmdi_randomframes/model000750000.pt`,
`dataset/HumanML3D/{texts,test.txt,Mean_abs_3d.npy,Std_abs_3d.npy}`, and `dataset/000021.npy`.
No AMASS, no SMPL body model, no chumpy.

## Environment notes

- ROCm #6595 **reproduces on this card**: fp32 `mm` above M = 2^19 returns silently wrong results
  (`max|diff| = 37.11` at M = 525312, exact at M = 262144). Flattened batch×seq here is
  16 × 196 = 3136, far below the threshold, but any batching change must respect it.
- `TORCH_ROCM_FA_PREFER_CK=1` must stay unset — CK SDPA supports only gfx942/gfx950. AOTriton
  ships fwd+bwd kernels for gfx120x and is selected by default.
- Sampling is ~27 s per cell at batch 4 and 1000 DDPM steps, ~7 GiB peak. Do not run this while
  the resident ollama server holds the card.
