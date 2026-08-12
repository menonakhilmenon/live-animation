"""§8d.9 -- keyframe angular-noise tolerance sweep on CondMDI.

Does a keyframe-conditioned diffusion model absorb angular error in its
keyframes, and does softening the imputation override ("tolerance") buy back
plausibility at the cost of keyframe fidelity?

Protocol
--------
1. Generate N reference motions from text with the keyframe conditioning empty.
   These are the "ground truth" the rest of the sweep is measured against.
   (HumanML3D's new_joint_vecs needs AMASS, which we have not registered for;
   using model samples as references is the deliberate substitution. See the
   caveat at the bottom of this docstring.)
2. Round-trip each reference through forward kinematics and a feature rebuild,
   with an angular perturbation of theta degrees applied to every joint at every
   keyframe. theta=0 is the control and goes through the identical round trip.
3. Re-generate conditioned on those keyframes, at tolerance c, from the *same*
   diffusion noise as the reference. Every condition therefore differs from the
   reference in exactly one place: the keyframes.
4. Measure geometric quality and keyframe fidelity.

Caveat, stated plainly: the references are model samples, not mocap, so the
model is being asked to reconstruct motion from its own manifold. Absolute
degradation is therefore optimistic. The *shape* of the degradation curve across
theta, and the tolerance trade-off, are what this measures.

Usage:
    PYTHONPATH=third_party/CondMDI .venv-rocm/bin/python \
        experiments/keyframe-noise/run_sweep.py [--quick]
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
REPO = os.path.abspath(REPO)
CONDMDI = os.path.join(REPO, "third_party", "CondMDI")
sys.path.insert(0, CONDMDI)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(CONDMDI)  # the repo hardcodes ./dataset/... relative paths

from utils.parser_util import cond_synt_args           # noqa: E402
from utils.model_util import create_model_and_diffusion, load_saved_model  # noqa: E402
from utils import dist_util                            # noqa: E402
from utils.fixseed import fixseed                      # noqa: E402
from utils.editing_util import get_keyframes_mask, joint_to_full_mask  # noqa: E402
from model.cfg_sampler import ClassifierFreeSampleModel  # noqa: E402

import kfnoise                                         # noqa: E402

MAX_FRAMES = 196
DATA_ROOT = os.path.join(CONDMDI, "dataset", "HumanML3D")
MODEL_PATH = "./save/condmdi_randomframes/model000750000.pt"
OUT_DIR = os.path.join(REPO, "experiments", "keyframe-noise", "results")

THETAS = [0, 5, 10, 20, 30]
TOLERANCES = [1.0, 0.85, 0.5, 0.0]
TRANS_LENGTH = 10  # keyframe every 10 frames = every 0.5 s at 20 fps


# ------------------------------------------------------------------ data shim
class _T2M:
    """Just enough of Text2MotionDatasetV2 for sampling: the normaliser."""

    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def inv_transform(self, d):
        return d * self.std + self.mean

    def transform(self, d):
        return (d - self.mean) / self.std


class _Dataset:
    def __init__(self, t2m):
        self.t2m_dataset = t2m


class _Data:
    def __init__(self, dataset):
        self.dataset = dataset


def load_texts(n, seed=0):
    """First caption of n sequences drawn from the HumanML3D test split."""
    with open(os.path.join(DATA_ROOT, "test.txt")) as f:
        names = [line.strip() for line in f if line.strip()]
    rng = np.random.RandomState(seed)
    picked, texts, used = [], [], set()
    for idx in rng.permutation(len(names)):
        name = names[idx]
        path = os.path.join(DATA_ROOT, "texts", name + ".txt")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            caption = f.readline().split("#")[0].strip()
        if not caption or caption in used:
            continue
        used.add(caption)
        picked.append(name)
        texts.append(caption)
        if len(texts) == n:
            break
    return picked, texts


# ---------------------------------------------------------------- generation
def build_kwargs(args, texts, obs_x0, obs_mask, device, tolerance=None):
    n = len(texts)
    mk = {
        "y": {
            "mask": torch.ones((n, 1, 1, MAX_FRAMES), dtype=torch.bool, device=device),
            "lengths": torch.full((n,), MAX_FRAMES, dtype=torch.long, device=device),
            "text": list(texts),
            "diffusion_steps": args.diffusion_steps,
        },
        "obs_x0": obs_x0,
        "obs_mask": obs_mask,
    }
    if tolerance is not None:
        # zero_keyframe_loss is False for this checkpoint, so imputation is the
        # optional hard-constraint path layered on top of native conditioning.
        mk["y"].update(
            imputate=1,
            stop_imputation_at=0,
            replacement_distribution="conditional",
            inpainted_motion=obs_x0,
            inpainting_mask=obs_mask,
            reconstruction_guidance=False,
            keyframe_tolerance=tolerance,
        )
    if args.guidance_param != 1:
        mk["y"]["text_scale"] = torch.ones(n, device=device) * args.guidance_param
    return mk


def sample(model, diffusion, args, mk, noise, n):
    with torch.no_grad():
        return diffusion.p_sample_loop(
            model,
            (n, model.njoints, model.nfeats, MAX_FRAMES),
            clip_denoised=False,
            model_kwargs=mk,
            skip_timesteps=0,
            init_image=None,
            progress=False,
            dump_steps=None,
            noise=noise,
            const_noise=False,
        )


# ------------------------------------------------------------------- metrics
def evaluate(sample_norm, inv, skel, kf_idx, clean_pos, pert_pos, ref_pos,
             pert_joints=None, obs_joints=None):
    """sample_norm: [n, 263, 1, T] normalised. Returns a list of per-sample dicts.

    Metrics are also reported split by whether a joint received the error. If
    `hit` degrades and `spared` does not, error stays local and an upstream stage
    only needs to be accurate about the joints it actually specifies.
    """
    m = inv(sample_norm.cpu().permute(0, 2, 3, 1)).float()[:, 0]  # [n, T, 263]
    hit = kfnoise.moved_joints(pert_joints or range(kfnoise.NJOINTS))
    spared = [j for j in range(kfnoise.NJOINTS) if j not in hit]
    # Whole-body keyframe error is not interpretable when only part of the body
    # is specified: the free joints are *supposed* to go their own way. Split it.
    obs = sorted(obs_joints or range(kfnoise.NJOINTS))
    free = [j for j in range(kfnoise.NJOINTS) if j not in obs]
    rows = []
    for i in range(m.shape[0]):
        pos = kfnoise.joint_positions(m[i])
        r = dict(
            kf_err_to_perturbed=kfnoise.keyframe_error(pos, pert_pos[i], kf_idx),
            kf_err_to_clean=kfnoise.keyframe_error(pos, clean_pos[i], kf_idx),
            skate_ratio=kfnoise.skating_ratio(pos),
            jerk=kfnoise.jerk(pos),
            ground_pen=kfnoise.ground_penetration(pos),
            div_from_ref=kfnoise.divergence(pos, ref_pos[i]),
            kf_err_obs=kfnoise.keyframe_error(pos, pert_pos[i], kf_idx, joints=obs),
            kf_err_obs_clean=kfnoise.keyframe_error(pos, clean_pos[i], kf_idx,
                                                    joints=obs),
            jerk_upper=kfnoise.jerk(pos, joints=kfnoise.UPPER),
            jerk_lower=kfnoise.jerk(pos, joints=kfnoise.LOWER),
            div_upper=kfnoise.divergence(pos, ref_pos[i], joints=kfnoise.UPPER),
            div_lower=kfnoise.divergence(pos, ref_pos[i], joints=kfnoise.LOWER),
        )
        r["div_hit"] = kfnoise.divergence(pos, ref_pos[i], joints=hit)
        r["div_spared"] = (kfnoise.divergence(pos, ref_pos[i], joints=spared)
                           if spared else float("nan"))
        r["jerk_spared"] = (kfnoise.jerk(pos, joints=spared)
                            if spared else float("nan"))
        r["div_free"] = (kfnoise.divergence(pos, ref_pos[i], joints=free)
                         if free else float("nan"))
        rows.append(r)
    return rows, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--guidance", type=float, default=2.5)
    ap.add_argument("--quick", action="store_true",
                    help="2 samples, theta in {0,30}, c in {1.0,0.5} -- a smoke test")
    ap.add_argument("--thetas", type=str, default=None,
                    help="comma-separated perturbation angles in degrees")
    ap.add_argument("--tols", type=str, default=None,
                    help="comma-separated imputation tolerances in [0,1]")
    ap.add_argument("--tag", type=str, default="sweep")
    ap.add_argument("--trans_length", type=int, default=TRANS_LENGTH,
                    help="frames between keyframes; 10 = 2 keys/s at 20 fps")
    ap.add_argument("--model", default="condmdi_randomframes",
                    help="checkpoint dir under third_party/CondMDI/save/. Use "
                         "condmdi_randomframejoints for per-joint keyframe masks -- "
                         "randomframes was trained on whole-frame keys only, so "
                         "partial joint masks are out of distribution for it.")
    ap.add_argument("--obs_joints", default="all", choices=sorted(kfnoise.JOINT_GROUPS),
                    help="which joints the model observes at keyframes")
    ap.add_argument("--perturb_joints", default="all",
                    choices=sorted(kfnoise.JOINT_GROUPS),
                    help="which joints receive the angular error")
    ap.add_argument("--perturb_mode", choices=["independent", "coherent"],
                    default="independent",
                    help="independent: each keyframe wrong in its own direction. "
                         "coherent: one fixed per-joint bias across the sequence.")
    opt = ap.parse_args()

    thetas, tols = THETAS, TOLERANCES
    if opt.quick:
        opt.num_samples, thetas, tols = 2, [0, 30], [1.0, 0.5]
    if opt.thetas:
        thetas = [int(x) for x in opt.thetas.split(",")]
    if opt.tols:
        tols = [float(x) for x in opt.tols.split(",")]

    os.makedirs(OUT_DIR, exist_ok=True)
    fixseed(opt.seed)

    ckpt = MODEL_PATH if opt.model == "condmdi_randomframes" else \
        f"./save/{opt.model}/" + sorted(
            f for f in os.listdir(f"./save/{opt.model}") if f.startswith("model")
        )[-1]
    sys.argv = ["run_sweep", "--model_path", ckpt]
    args = cond_synt_args()
    args.guidance_param = opt.guidance
    args.keyframe_guidance_param = 1.0
    args.num_samples = args.batch_size = opt.num_samples
    args.edit_mode = "benchmark_sparse"
    args.transition_length = opt.trans_length
    args.editable_features = "pos_rot_vel"

    dist_util.setup_dist(args.device)
    device = dist_util.dev()
    print(f"device: {device}")

    mean = np.load(os.path.join(DATA_ROOT, "Mean_abs_3d.npy"))
    std = np.load(os.path.join(DATA_ROOT, "Std_abs_3d.npy"))
    t2m = _T2M(torch.from_numpy(mean).float(), torch.from_numpy(std).float())
    data = _Data(_Dataset(t2m))

    names, texts = load_texts(opt.num_samples, seed=opt.seed)
    print("prompts:")
    for nm, tx in zip(names, texts):
        print(f"  {nm}: {tx}")

    print("creating model...")
    model, diffusion = create_model_and_diffusion(args, data)
    load_saved_model(model, args.model_path)
    model = ClassifierFreeSampleModel(model) if args.guidance_param != 1 else model
    model.to(device).eval()

    n = opt.num_samples
    njoints, nfeats = model.njoints, model.nfeats
    skel = kfnoise.build_skeleton(device="cpu")

    # One fixed noise tensor shared by every condition, so differences between
    # runs come from the keyframes and nothing else.
    g = torch.Generator(device="cpu").manual_seed(opt.seed)
    noise = torch.randn(n, njoints, nfeats, MAX_FRAMES, generator=g).to(device)

    lengths = torch.full((n,), MAX_FRAMES, dtype=torch.long)
    empty_mask, _ = get_keyframes_mask(
        data=torch.zeros(n, njoints, nfeats, MAX_FRAMES),
        lengths=lengths, edit_mode="uncond", get_joint_mask=True,
    )
    kf_idx = list(range(0, MAX_FRAMES, opt.trans_length))
    # The root (joint 0) is always observed. Global joint positions are measured
    # in world space, so leaving the trajectory free makes every observed joint
    # look wrong by however far the pelvis wandered -- confounding "did the
    # unspecified joints degrade" with "did the body walk somewhere else". A
    # director that specifies a body part but not where the body is standing is
    # not the configuration we care about anyway.
    obs_joints = sorted(set(kfnoise.JOINT_GROUPS[opt.obs_joints]) | {0})
    pert_joints = kfnoise.JOINT_GROUPS[opt.perturb_joints]
    unseen = sorted(set(kfnoise.moved_joints(pert_joints)) - set(obs_joints))
    if unseen:
        print(f"  warning: joints {unseen} are perturbed but never observed, so "
              f"the model cannot see that error at all")

    # get_keyframes_mask has no mode for "these frames AND these joints", so
    # build the joint mask directly and expand it through the repo's own
    # joint -> 263-feature matrix.
    obs_joint_mask = torch.zeros((n, kfnoise.NJOINTS, nfeats, MAX_FRAMES), dtype=torch.bool)
    for j in obs_joints:
        obs_joint_mask[:, j, :, kf_idx] = True
    kf_mask = joint_to_full_mask(obs_joint_mask, mode="pos_rot_vel")

    empty_mask, kf_mask = empty_mask.to(device), kf_mask.to(device)
    print(f"{len(kf_idx)} keyframes over {MAX_FRAMES} frames "
          f"({len(kf_idx) / (MAX_FRAMES / 20.0):.2f} keys/s); "
          f"observing {opt.obs_joints} ({len(obs_joints)} joints), "
          f"perturbing {opt.perturb_joints} ({len(pert_joints)} joints)")

    # ------------------------------------------------------- 1. references
    print("\n[1/3] generating reference motions (no keyframe conditioning)...")
    t0 = time.time()
    zeros = torch.zeros(n, njoints, nfeats, MAX_FRAMES, device=device)
    mk = build_kwargs(args, texts, zeros, empty_mask, device, tolerance=None)
    ref_norm = sample(model, diffusion, args, mk, noise, n)
    print(f"      done in {time.time() - t0:.1f}s")

    ref_unnorm = t2m.inv_transform(ref_norm.cpu().permute(0, 2, 3, 1)).float()[:, 0]

    # ------------------------------------------- 2. perturbed keyframe sets
    print("\n[2/3] building perturbed keyframes...")
    pert_gen = torch.Generator(device="cpu").manual_seed(opt.seed + 1)
    keyframe_sets, clean_pos, pert_pos_by_theta = {}, [], {}

    for theta in thetas:
        motions, positions = [], []
        for i in range(n):
            pm = kfnoise.perturb_motion(
                ref_unnorm[i], kf_idx, theta, skel, pert_gen, abs_3d=True,
                mode=opt.perturb_mode, joints=pert_joints,
            )
            motions.append(pm)
            positions.append(kfnoise.joint_positions(pm))
        stacked = torch.stack(motions)                      # [n, T, 263]
        norm = t2m.transform(stacked)[:, None].permute(0, 3, 1, 2)  # [n,263,1,T]
        keyframe_sets[theta] = norm.to(device).float()
        pert_pos_by_theta[theta] = positions
        if theta == 0:
            clean_pos = positions
            # how far the FK round trip alone moved the pose: the model's own
            # rotation/position inconsistency, and the floor of every later number
            drift = np.mean([
                kfnoise.divergence(positions[i], kfnoise.joint_positions(ref_unnorm[i]))
                for i in range(n)
            ])
            print(f"      FK round-trip drift at theta=0: {drift * 1000:.1f} mm")
        else:
            err = np.mean([
                kfnoise.keyframe_error(positions[i], clean_pos[i], kf_idx)
                for i in range(n)
            ])
            print(f"      theta={theta:2d} deg -> keyframe displacement "
                  f"{err * 1000:6.1f} mm")

    ref_pos = [kfnoise.joint_positions(ref_unnorm[i]) for i in range(n)]

    # -------------------------------------------------------- 3. the sweep
    print("\n[3/3] sweeping theta x tolerance...")
    rows = []
    total = len(thetas) * len(tols)
    done = 0
    for theta in thetas:
        obs_x0 = keyframe_sets[theta]
        for c in tols:
            t0 = time.time()
            mk = build_kwargs(args, texts, obs_x0, kf_mask, device, tolerance=c)
            out = sample(model, diffusion, args, mk, noise, n)
            per_sample, _ = evaluate(
                out, t2m.inv_transform, skel, kf_idx,
                clean_pos, pert_pos_by_theta[theta], ref_pos,
                pert_joints=pert_joints, obs_joints=obs_joints,
            )
            for i, r in enumerate(per_sample):
                rows.append(dict(seed=opt.seed, mode=opt.perturb_mode,
                                 trans_length=opt.trans_length,
                                 obs_joints=opt.obs_joints,
                                 perturb_joints=opt.perturb_joints,
                                 theta=theta, tolerance=c,
                                 sample=i, text=texts[i], **r))
            done += 1
            agg = {k: np.mean([r[k] for r in per_sample])
                   for k in per_sample[0] if k != "text"}
            print(f"  [{done:2d}/{total}] theta={theta:2d} c={c:.2f} "
                  f"({time.time() - t0:.0f}s)  "
                  f"kf_err_obs={agg['kf_err_obs'] * 1000:7.2f}mm  "
                  f"kf_err_all={agg['kf_err_to_perturbed'] * 1000:7.2f}mm  "
                  f"skate={agg['skate_ratio']:.4f}  "
                  f"jerk={agg['jerk']:7.1f}  "
                  f"div={agg['div_from_ref'] * 1000:6.1f}mm  "
                  f"div_hit={agg['div_hit'] * 1000:6.1f}mm  "
                  f"div_spared={agg['div_spared'] * 1000:6.1f}mm")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    csv_path = os.path.join(OUT_DIR, f"{opt.tag}-{stamp}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT_DIR, f"{opt.tag}-{stamp}.json"), "w") as f:
        json.dump(dict(model=opt.model, checkpoint=ckpt, seed=opt.seed,
                       guidance=opt.guidance,
                       perturb_mode=opt.perturb_mode,
                       obs_joints=opt.obs_joints,
                       perturb_joints=opt.perturb_joints,
                       num_samples=n, thetas=thetas, tolerances=tols,
                       trans_length=opt.trans_length, texts=texts, names=names),
                  f, indent=2)
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
