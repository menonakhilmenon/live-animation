# Keyframe angular noise: results

**Model** CondMDI `condmdi_randomframes` @ 750k steps (natively keyframe-conditioned, MIT).
**Protocol** 32 motions (2 seeds × 16 HumanML3D test captions), 196 frames, 20 keyframes
(2.04 keys/s), 1000-step DDPM, text CFG 2.5. Every cell shares prompts and diffusion noise with
every other, so all contrasts are paired. Geometric metrics only — no FID, no R-precision
(HumanML3D's `new_joint_vecs/` needs an AMASS registration we have not done).

**θ** = per-joint rotation error injected at each keyframe, in degrees, each keyframe wrong in its
own random direction. **c** = imputation tolerance: `c=1` is CondMDI's hard override of the
predicted x₀ at observed entries on every denoising step; `c=0` disables imputation and leaves only
the model's native keyframe conditioning (which is what the paper's published numbers use).

---

## 1. The model does not correct keyframe error. It propagates it.

This is the finding that matters, and it is unambiguous.

| θ (deg) | keyframe displacement injected | best output-to-clean error over all c | recovery |
|---:|---:|---:|---:|
| 5 | 41.3 mm | 41.3 mm | 0% |
| 10 | 82.7 mm | 82.7 mm | 0% |
| 20 | 162.9 mm | 158.1 mm | −3% |
| 30 | 236.0 mm | 226.0 mm | −4% |

Divergence of the whole generated motion from the unperturbed reference tracks the injected
keyframe displacement almost one-for-one, and is **independent of tolerance**:

| | θ=0 | θ=5 | θ=10 | θ=20 | θ=30 |
|---|---:|---:|---:|---:|---:|
| injected at keyframes | 0 | 41 | 83 | 163 | 236 mm |
| whole-motion divergence, c=1.0 | 62 | 79 | 105 | 166 | 225 mm |
| whole-motion divergence, c=0.0 | 68 | 80 | 108 | 167 | 226 mm |

The densifier has a strong prior over *how a body moves* and essentially none over *which pose is
correct*. Given a wrong keyframe it produces a smooth, plausible motion that goes to the wrong
place. **It will not clean up an upstream stage's pose error — at any tolerance.**

## 2. Angular noise shows up as jerk, and jerk is the sensitive readout

| jerk (m/s³) | θ=0 | θ=5 | θ=10 | θ=20 | θ=30 |
|---|---:|---:|---:|---:|---:|
| c = 1.00 | 518 | 605 | 715 | 1002 | 1333 |
| c = 0.50 | 332 | 361 | 433 | 606 | 826 |
| c = 0.20 | 218 | 233 | 274 | 396 | 539 |
| c = 0.05 | 177 | 186 | 218 | 318 | 445 |
| c = 0.00 | 169 | 180 | 212 | 310 | 435 |

Monotonic in both axes, with no threshold or cliff. Paired |t| = 21.3 for the θ=0→30 effect at
c=1.0 and 33.7 at c=0.0 (n=32).

Foot skate moves the same way but is a **much weaker instrument**: paired |t| = 2.4 for the same
contrast, non-monotonic in θ (it peaks around θ=10–20 and falls at θ=30), and its per-motion standard
deviation is comparable to its mean. Ground penetration barely moves at all (0.2 → 1.0 mm).
**Report jerk; treat foot skate as corroborating at best.** This is consistent with §8d.12 — geometric
plausibility metrics are weak discriminators, and foot skate specifically was the weakest of them.

## 3. Tolerance lowers the whole curve, but does not flatten it

Reading down each column above: dropping from hard imputation to native conditioning cuts jerk by
**about two thirds at every θ** (67% at θ=30, 67% at θ=0).

Reading across each row: the *relative* degradation from θ=0 to θ=30 is **+158% at c=1.0 and +157%
at c=0.0** — identical.

So tolerance is not a robustness mechanism. It buys a uniformly better baseline; it does not make
the model less sensitive to keyframe error. The sensitivity slope is a property of the model, not of
how hard you clamp the keyframes.

## 4. Hard imputation costs quality even when the keyframes are perfect

At θ=0 the keyframes are exactly right, and hard imputation still makes the motion worse:

| θ=0 | c = 1.00 (hard) | c = 0.00 (native) | paired |t| |
|---|---:|---:|---:|
| jerk (m/s³) | 518 | 169 | 5.4 |
| foot skate | 0.168 | 0.102 | 6.0 |
| keyframe error | 0.0 mm | 62.5 mm | — |

Overwriting x₀ at observed entries on every denoising step forces discontinuities the model then has
to absorb, tripling jerk. The price of relaxing it is 62.5 mm of keyframe error — the model quietly
declines to hit keyframes it is not forced to hit, even correct ones.

The trade-off across the full tolerance range at θ=30:

| c | keyframe error to the given keyframes | jerk |
|---:|---:|---:|
| 1.00 | 0.0 mm | 1333 |
| 0.50 | 83.9 mm | 826 |
| 0.20 | 133.6 mm | 539 |
| 0.05 | 158.3 mm | 445 |
| 0.00 | 164.5 mm | 435 |

The knee is around **c ≈ 0.2–0.5**: c=0.5 buys back 38% of the jerk for 84 mm of keyframe error,
while going further to c=0.05 costs another 74 mm for only 9 more points of jerk reduction.

## 5. What this means for the multi-stage plan (§8d)

- **A keyframe proposer's pose error passes straight through to the output.** Any accuracy budget
  for stage 2 is the accuracy budget for the final animation; there is no downstream cleanup. This
  raises the bar on the retrieval proposer of §8d.8 considerably, and it is an argument *for*
  retrieval over generation — retrieved poses are on-manifold and exactly correct as poses, whereas
  a generator's 308.6 mm MPJPE would land intact in the output.
- **10° per joint is not a safe budget.** It costs 83 mm of joint displacement and +38% jerk at
  c=1.0. If the proposer cannot do better than ~5°, the densifier will not rescue it.
- **Do not use hard imputation.** Native keyframe conditioning at c=0, or a soft c≈0.2–0.5, is
  strictly better on every plausibility metric. The paper's own headline configuration already does
  this; the hard-override path is the trap.
- **Tolerance is not a noise-robustness knob.** It was tempting to read §8b.2's `R-NoTolerance` vs
  `Ours (c=0.50)` ablation as evidence that tolerant conditioning absorbs keyframe error. It does
  not. It improves the baseline uniformly, which is worth having, but the degradation slope is
  unchanged.

## 6. Limitations, stated plainly

- **The references are model samples, not mocap.** Without AMASS there is no ground-truth HumanML3D
  motion to draw keyframes from, so the reference motions are generated by the same model with empty
  keyframe conditioning. Every keyframe is therefore on the model's own manifold, which makes this
  the *easy* case; degradation against real mocap keyframes should be at least this bad.
- **Absolute values are not comparable to the paper.** We generate a fixed 196 frames for every
  caption where CondMDI's eval respects each sequence's true length. Our θ=0, c=0 cell gives foot
  skate 0.102 against the checkpoint's published 0.0938, and keyframe error 62.5 mm against its
  0.119 m. Within-sweep contrasts are valid because every cell is matched; cross-paper comparisons
  are not.
- **No FID, no R-precision.** The t2m evaluator download is quota-blocked and would need AMASS
  besides. Per §8d.9's own note, keyframe error and plausibility metrics move long before FID does,
  so the sensitive readouts are the ones present — but distributional realism is unmeasured.
- **One keyframe density.** Everything here is at 2.04 keys/s. §8b's sparsity findings suggest the
  picture changes with density, and that axis is untested.
- **Foot skate is underpowered** at n=32 for the θ contrast (|t| = 2.4); the jerk results are not.
