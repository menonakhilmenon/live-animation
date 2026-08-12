# Keyframe density: how many keyframes should the proposer emit?

**Model** CondMDI `condmdi_randomframes` @ 750k (MIT). **Protocol** 16 HumanML3D test captions,
196 frames, seed 0, 1000-step DDPM, text CFG 2.5, `independent` error mode, all 22 joints observed.
Every cell shares prompts, diffusion noise and reference motions, so all contrasts are paired.
`tl` is the gap between keyframes in frames: tl=5/10/20/40 gives 40/20/10/5 keyframes over the clip,
i.e. **4.08 / 2.04 / 1.02 / 0.51 keys per second**. The committed sweep was tl=10 throughout, and
tl=10 here reproduces it (θ=0, c=1.0 jerk 562.8 vs 563 before).

The previous experiment held density fixed and found that keyframe pose error passes straight
through to the output. This one asks the follow-up a proposer actually needs answered: given that
its keyframes will be wrong by some amount, **how many should it emit?**

---

## 1. Denser keyframes are only worth it if they are accurate

Divergence of the generated motion from the unperturbed reference, at hard imputation (c=1.0):

| | θ=0 | θ=10 | θ=30 |
|---|---:|---:|---:|
| 4.08 keys/s | 42.5 mm | 90.3 | 229.2 |
| 2.04 keys/s | 66.0 | 107.6 | 235.7 |
| 1.02 keys/s | 104.8 | 137.4 | 237.8 |
| 0.51 keys/s | 191.8 | 196.6 | 371.2 |

Read the columns. **What 8× more keyframes buys you, as a function of how wrong they are** (1.02 →
4.08 keys/s):

| θ | fidelity gained | jerk paid |
|---:|---:|---:|
| 0 | −59% (104.8 → 42.5 mm) | +55% (442 → 685) |
| 10 | −34% (137.4 → 90.3 mm) | +86% (546 → 1015) |
| 30 | **−4%** (237.8 → 229.2 mm) | **+191%** (857 → 2497) |

At 30° per joint, going from 1 key/s to 4 keys/s buys **4% of fidelity for triple the jerk**. The
extra keyframes are all telling the model the same wrong thing, so they add no information about
where the motion should go — only more discontinuities for it to absorb. Denser specification is a
*multiplier on the proposer's accuracy*, in both directions.

## 2. Sparse conditioning is far more robust to error

The θ=0 → 30 penalty, paired within each density:

| | jerk, c=1.0 | jerk, c=0.0 | divergence, c=1.0 |
|---|---:|---:|---:|
| 4.08 keys/s | +265% (\|t\|=18.0) | +328% (\|t\|=26.9) | +439% (\|t\|=24.9) |
| 2.04 keys/s | +144% (\|t\|=14.9) | +136% (\|t\|=21.6) | +257% (\|t\|=20.0) |
| 1.02 keys/s | +94% (\|t\|=14.1) | +85% (\|t\|=13.1) | +127% (\|t\|=11.8) |
| 0.51 keys/s | +78% (\|t\|=8.3) | +64% (\|t\|=8.7) | +94% (\|t\|=6.8) |

Monotone, large, and highly significant at every density. This is the opposite of the §8d.9 tolerance
result: tolerance lowered the whole curve without changing the slope, whereas **density changes the
slope by more than 3×**. Density, not tolerance, is the robustness knob.

The mechanism is visible in absolute jerk at θ=30, c=1.0: 2497 → 1374 → 857 → 660 as keyframes get
sparser. Each keyframe under hard imputation is a hard constraint the model must reconcile with its
neighbours; when they disagree, more of them means more reconciling.

## 3. Foot skate confirms the mechanism

| foot skate, c=1.0 | θ=0 | θ=10 | θ=30 |
|---|---:|---:|---:|
| 4.08 keys/s | 0.225 | **0.728** | 0.289 |
| 2.04 keys/s | 0.205 | **0.405** | 0.259 |
| 1.02 keys/s | 0.160 | 0.206 | 0.147 |
| 0.51 keys/s | 0.141 | 0.157 | 0.148 |

§8d.9 §5 established that foot skate measures keyframes *disagreeing with each other*, not pose error.
That is exactly what this table shows: the blow-up at θ=10 is worst where keyframes are densest
(0.728 at 4.08 keys/s), and it vanishes entirely below ~1 key/s. Note the non-monotonicity in θ
survives here too — skate peaks at θ=10 and falls at θ=30 — so it remains a poor instrument for
grading pose accuracy and a good one for detecting inconsistency.

## 4. What this means for the proposer

- **Match density to accuracy.** Emitting more keyframes than your accuracy justifies actively hurts:
  it buys almost no fidelity and costs a lot of plausibility. There is a defensible default here —
  around **1 key/s** the error penalty has already dropped to a third of its 4 keys/s value while
  divergence at θ=0 is still ~10 cm.
- **The earlier "10° is not a safe budget" conclusion was measured at 2 keys/s and is density-
  dependent.** At 1.02 keys/s a 10° error costs +23% jerk and 33 mm of extra divergence, against
  +35% and 42 mm at 2.04 keys/s. A proposer that cannot hit 5° should emit fewer keyframes rather
  than try to compensate with more.
- **This is another argument for retrieval over generation, and for contiguous clips.** A retrieved
  clip gives many accurate keyframes, where density pays off. A generator with 300 mm MPJPE should
  be asked for as few keyframes as the motion allows.
- **Do not read the θ=0 row as free.** Sparse conditioning costs real fidelity when the keyframes are
  right: 42.5 → 191.8 mm from 4.08 to 0.51 keys/s. The claim is not "sparse is better", it is
  "the value of density collapses with error, while its cost does not".

## 5. Limitations

- Same limitations as §8d.9: model-sampled references rather than mocap, no FID/R-precision, fixed
  196-frame generation, root motion never perturbed.
- **n=16, one seed** (the committed θ×c sweep was n=32 across two seeds). The density effects are
  large and the paired |t| values run 5.5–27, so the direction is not in doubt, but the exact
  percentages will move.
- θ ∈ {0, 10, 30} only — the 5° and 20° points that pinned down linearity in §8d.9 were dropped to
  keep this to one GPU session.
- **Keyframes are uniformly spaced.** A real proposer would place them at motion-salient moments,
  which should favour sparse specification further; that is untested here.
