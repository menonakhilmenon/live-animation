"""Self-tests for kfnoise. Runs on CPU in seconds; no model, no GPU.

If the ric encode/decode inverse or the perturbation magnitude is wrong, every
number in the sweep is wrong in a way that still looks plausible. So check them
directly rather than inferring correctness from the sweep output.
"""

import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CONDMDI = os.path.join(REPO, "third_party", "CondMDI")
sys.path.insert(0, CONDMDI)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(CONDMDI)

import kfnoise  # noqa: E402
from data_loaders.humanml.common.quaternion import cont6d_to_matrix  # noqa: E402

T = 60
KF = list(range(0, T, 10))
FAIL = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def random_motion(seed=0):
    """A synthetic but structurally valid 263-vector: real rotations, real root."""
    g = torch.Generator().manual_seed(seed)
    m = torch.zeros(T, 263)
    # smooth root trajectory
    t = torch.linspace(0, 2 * np.pi, T)
    m[:, 0] = 0.5 * torch.sin(t)                      # absolute yaw (abs_3d)
    m[:, 1] = torch.linspace(0, 1.5, T)               # absolute x
    m[:, 2] = 0.3 * torch.cos(t)                      # absolute z
    m[:, 3] = 0.9 + 0.02 * torch.sin(3 * t)           # root height
    # random-but-small joint rotations, smoothed over time
    q = torch.randn(T, 21, 4, generator=g) * 0.15
    q[..., 0] += 1.0
    q = q / q.norm(dim=-1, keepdim=True)
    R = kfnoise._quat_to_matrix(q)
    m[:, kfnoise.ROT] = kfnoise.matrix_to_cont6d(R).reshape(T, -1)
    return m


def geodesic_deg(A, B):
    """Angle of A^T B, in degrees."""
    rel = torch.matmul(A.transpose(-1, -2), B)
    tr = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.rad2deg(torch.acos(((tr - 1) / 2).clamp(-1.0, 1.0)))


skel = kfnoise.build_skeleton()
m = random_motion()

# --- 1. ric encode/decode are exact inverses -------------------------------
pos_fk = kfnoise.forward_kinematics(m, skel)
m2 = kfnoise.positions_to_features(pos_fk, m)
pos_back = kfnoise.joint_positions(m2)
d = (pos_fk - pos_back).abs().max().item()
check("ric encode -> decode round-trips to FK positions", d < 1e-4,
      f"max|diff| = {d:.3e} m")

# --- 2. rot6d encode/decode are exact inverses ------------------------------
R = cont6d_to_matrix(m[:, kfnoise.ROT].reshape(T, 21, 6))
r6 = kfnoise.matrix_to_cont6d(R)
R2 = cont6d_to_matrix(r6)
d = (R - R2).abs().max().item()
check("matrix <-> cont6d round-trip", d < 1e-5, f"max|diff| = {d:.3e}")

# --- 3. theta=0 perturbation is a pure round trip ---------------------------
g = torch.Generator().manual_seed(0)
p0 = kfnoise.perturb_motion(m, KF, 0, skel, g)
d = (p0 - m2).abs().max().item()
check("theta=0 perturbation equals the plain round trip", d < 1e-4,
      f"max|diff| = {d:.3e}")

# --- 4. perturbation magnitude is exactly theta at keyframes ---------------
for theta in (5, 10, 20, 30):
    g = torch.Generator().manual_seed(1)
    p = kfnoise.perturb_motion(m, KF, theta, skel, g)
    Ra = cont6d_to_matrix(m2[:, kfnoise.ROT].reshape(T, 21, 6))
    Rb = cont6d_to_matrix(p[:, kfnoise.ROT].reshape(T, 21, 6))
    ang = geodesic_deg(Ra, Rb)[KF]
    err = (ang - theta).abs().max().item()
    check(f"per-joint rotation error is {theta} deg at keyframes", err < 0.05,
          f"mean={ang.mean():.3f} max_dev={err:.3e}")

# --- 4b. coherent mode: same magnitude, but the same error at every keyframe -
for theta in (10, 30):
    g = torch.Generator().manual_seed(7)
    p = kfnoise.perturb_motion(m, KF, theta, skel, g, mode="coherent")
    Ra = cont6d_to_matrix(m2[:, kfnoise.ROT].reshape(T, 21, 6))
    Rb = cont6d_to_matrix(p[:, kfnoise.ROT].reshape(T, 21, 6))
    ang = geodesic_deg(Ra, Rb)[KF]
    # magnitude still exactly theta, and identical across keyframes per joint
    spread = (ang - ang.mean(dim=0, keepdim=True)).abs().max().item()
    check(f"coherent theta={theta}: exact magnitude, no keyframe-to-keyframe drift",
          (ang - theta).abs().max().item() < 0.05 and spread < 0.05,
          f"mean={ang.mean():.3f}, across-keyframe spread={spread:.3e}")

# --- 5. perturbation moves joints monotonically, and root stays put --------
prev = 0.0
for theta in (0, 5, 10, 20, 30):
    g = torch.Generator().manual_seed(2)
    p = kfnoise.perturb_motion(m, KF, theta, skel, g)
    pos = kfnoise.joint_positions(p)
    disp = (pos[KF] - kfnoise.joint_positions(p0)[KF]).norm(dim=-1).mean().item()
    root = (pos[:, 0] - kfnoise.joint_positions(p0)[:, 0]).abs().max().item()
    check(f"theta={theta:2d}: displacement grows, root unmoved",
          disp >= prev - 1e-9 and root < 1e-5,
          f"joint disp {disp * 1000:6.1f} mm, root drift {root:.2e} m")
    prev = disp

# --- 6. derived channels are consistent, not stale -------------------------
g = torch.Generator().manual_seed(3)
p30 = kfnoise.perturb_motion(m, KF, 30, skel, g)
ric_changed = (p30[:, kfnoise.RIC] - m2[:, kfnoise.RIC]).abs().max().item()
vel_changed = (p30[:, kfnoise.VEL] - m2[:, kfnoise.VEL]).abs().max().item()
check("ric and velocity channels were rebuilt, not carried over",
      ric_changed > 1e-3 and vel_changed > 1e-4,
      f"ric delta {ric_changed:.3e}, vel delta {vel_changed:.3e}")

# --- 7. metrics are finite and sane ---------------------------------------
pos = kfnoise.joint_positions(m2)
vals = dict(skate=kfnoise.skating_ratio(pos), jerk=kfnoise.jerk(pos),
            pen=kfnoise.ground_penetration(pos),
            kf=kfnoise.keyframe_error(pos, pos, KF))
check("metrics finite; self keyframe error is zero",
      all(np.isfinite(v) for v in vals.values()) and vals["kf"] < 1e-6,
      ", ".join(f"{k}={v:.4g}" for k, v in vals.items()))

# --- 8. joint-restricted perturbation stays inside its subtree ------------
for group in ("arms", "legs", "upper", "lower"):
    js = kfnoise.JOINT_GROUPS[group]
    moved = kfnoise.moved_joints(js)
    still = [j for j in range(kfnoise.NJOINTS) if j not in moved]
    g = torch.Generator().manual_seed(11)
    p = kfnoise.perturb_motion(m, KF, 30, skel, g, joints=js)
    d = (kfnoise.joint_positions(p) - kfnoise.joint_positions(p0)).norm(dim=-1)
    check(f"joints={group}: {len(still)} unselected joints exactly unmoved",
          d[:, still].max().item() < 1e-5 and d[KF][:, moved].mean().item() > 1e-3,
          f"spared max {d[:, still].max().item():.2e} m, "
          f"hit mean {d[KF][:, moved].mean().item() * 1000:.1f} mm")

# arms and legs are disjoint subtrees, so their damage must not overlap
check("arms and legs move disjoint joint sets",
      not (set(kfnoise.moved_joints(kfnoise.ARMS))
           & set(kfnoise.moved_joints(kfnoise.LEGS))),
      f"arms->{kfnoise.moved_joints(kfnoise.ARMS)}, "
      f"legs->{kfnoise.moved_joints(kfnoise.LEGS)}")

# --- 9. per-region metrics respond only to their own region ---------------
g = torch.Generator().manual_seed(12)
p = kfnoise.perturb_motion(m, KF, 30, skel, g, joints=kfnoise.ARMS)
pos_a, pos_c = kfnoise.joint_positions(p), kfnoise.joint_positions(p0)
check("arms-only perturbation: upper divergence >> lower divergence",
      kfnoise.divergence(pos_a, pos_c, joints=kfnoise.UPPER) > 1e-3
      and kfnoise.divergence(pos_a, pos_c, joints=kfnoise.LOWER) < 1e-5,
      f"upper {kfnoise.divergence(pos_a, pos_c, joints=kfnoise.UPPER) * 1000:.1f} mm, "
      f"lower {kfnoise.divergence(pos_a, pos_c, joints=kfnoise.LOWER) * 1000:.3f} mm")

print()
if FAIL:
    print(f"{len(FAIL)} FAILURE(S): " + ", ".join(FAIL))
else:
    print("all self-tests passed")
sys.exit(1 if FAIL else 0)
