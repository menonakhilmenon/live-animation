"""Angular keyframe perturbation and geometric metrics for the §8d.9 experiment.

The question: if an upstream stage (retrieval, an LLM director, a pose lifter)
proposes keyframes that are off by 5-30 degrees per joint, does the diffusion
model absorb the error or propagate it?

Perturbing a HumanML3D 263-vector is not a matter of adding noise to a slice.
The vector is redundant: joint rotations (67:193), root-relative joint positions
(4:67), local velocities (193:259) and foot contacts (259:263) all describe the
same pose. Perturbing rotations alone yields a *self-contradictory* keyframe, and
conditioning on one measures robustness-to-garbage, not robustness-to-angular-
error. So we perturb the rotations, run forward kinematics, and rebuild every
derived channel from the resulting positions.

Feature layout (HumanML3D, 22 joints, abs_3d variant):
    [0]       root Y-rotation   (absolute angle under abs_3d, velocity otherwise)
    [1:3]     root XZ           (absolute under abs_3d)
    [3]       root height
    [4:67]    ric   -- joints 1..21, root-relative, yaw-canonicalised
    [67:193]  rot6d -- joints 1..21, local rotation relative to parent
    [193:259] local velocities -- all 22 joints, yaw-canonicalised
    [259:263] foot contacts     -- joints 7, 10, 8, 11
"""

import numpy as np
import torch

from data_loaders.humanml.common.quaternion import (
    cont6d_to_matrix,
    qinv,
    qrot,
    quaternion_to_cont6d,
)
from data_loaders.humanml.common.skeleton import Skeleton
from data_loaders.humanml.scripts.motion_process import (
    recover_from_ric,
    recover_root_rot_pos,
)
from data_loaders.humanml.utils.paramUtil import (
    t2m_kinematic_chain,
    t2m_raw_offsets,
)

NJOINTS = 22
RIC = slice(4, 4 + (NJOINTS - 1) * 3)          # 4:67
ROT = slice(67, 67 + (NJOINTS - 1) * 6)        # 67:193
VEL = slice(193, 193 + NJOINTS * 3)            # 193:259
CNT = slice(259, 263)

FID_L, FID_R = [7, 10], [8, 11]
FEET_THRE = 0.002  # HumanML3D's own contact threshold, from process_file


def build_skeleton(example_npy="./dataset/000021.npy", device="cpu"):
    """Target skeleton with HumanML3D's canonical bone lengths."""
    raw_offsets = torch.from_numpy(t2m_raw_offsets)
    skel = Skeleton(raw_offsets, t2m_kinematic_chain, device)
    # 000021.npy is SMPL-24; the first 22 joints are the HumanML3D body, and
    # get_offsets_joints only walks as far as the raw-offset table (22).
    raw = np.load(example_npy)
    example = torch.from_numpy(raw.reshape(len(raw), -1, 3))
    skel.set_offset(skel.get_offsets_joints(example[0].float()))
    return skel


def matrix_to_cont6d(mat):
    """Inverse of cont6d_to_matrix: the first two *columns*, flattened."""
    return torch.cat([mat[..., :, 0], mat[..., :, 1]], dim=-1)


def forward_kinematics(motion, skel, abs_3d=True):
    """263-vector -> global joint positions, driven purely by the rotation channels.

    motion: [T, 263] unnormalised.  Returns [T, 22, 3].
    """
    r_rot_quat, r_pos = recover_root_rot_pos(motion, abs_3d=abs_3d)
    cont6d = torch.cat(
        [quaternion_to_cont6d(r_rot_quat), motion[..., ROT]], dim=-1
    ).view(-1, NJOINTS, 6)
    return skel.forward_kinematics_cont6d(cont6d.float(), r_pos.float())


def positions_to_features(positions, motion, abs_3d=True):
    """Rebuild ric / velocity / contact channels from global joint positions.

    The root channels [0:4] and the rotation channels [67:193] are carried over
    from `motion` untouched -- the caller owns those. Everything else is derived,
    so the returned vector is internally consistent by construction.

    positions: [T, 22, 3] global.  motion: [T, 263].  Returns [T, 263].
    """
    T = positions.shape[0]
    r_rot_quat, r_pos = recover_root_rot_pos(motion, abs_3d=abs_3d)

    # global -> yaw-canonical root-relative frame. recover_from_ric applies
    # qinv(r_rot_quat) going the other way, so the forward direction is bare.
    local = positions.clone()
    local[..., 0] -= r_pos[:, None, 0]
    local[..., 2] -= r_pos[:, None, 2]
    local = qrot(r_rot_quat[:, None].expand(T, NJOINTS, 4).contiguous(), local)
    ric = local[:, 1:].reshape(T, -1)

    # local velocities: frame-to-frame displacement, rotated by the *previous*
    # frame's yaw (matching process_file), then padded to length T.
    disp = positions[1:] - positions[:-1]
    vel = qrot(r_rot_quat[:-1, None].expand(T - 1, NJOINTS, 4).contiguous(), disp)
    vel = vel.reshape(T - 1, -1)
    vel = torch.cat([vel, vel[-1:]], dim=0)

    # foot contacts: squared 3D displacement below threshold
    def contact(fids):
        d = (positions[1:, fids] - positions[:-1, fids]).pow(2).sum(-1)
        return (d < FEET_THRE).float()

    cnt = torch.cat([contact(FID_L), contact(FID_R)], dim=-1)
    cnt = torch.cat([cnt, cnt[-1:]], dim=0)

    out = motion.clone()
    out[..., RIC] = ric
    out[..., VEL] = vel
    out[..., CNT] = cnt
    return out


def _random_axes(shape, generator, device):
    v = torch.randn(*shape, 3, generator=generator, device=device)
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _axis_angle_to_quat(axis, angle):
    half = angle / 2.0
    return torch.cat(
        [torch.cos(half) * torch.ones_like(axis[..., :1]), torch.sin(half) * axis],
        dim=-1,
    )


def _quat_to_matrix(q):
    """Local re-implementation: quaternion_to_matrix in the repo is (w,x,y,z)."""
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def _slerp(q0, q1, w):
    """Shortest-arc slerp. q*: [..., 4], w: [...] broadcastable."""
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(max=1.0 - 1e-7)
    theta = torch.acos(dot)
    sin_t = torch.sin(theta).clamp_min(1e-8)
    w = w[..., None]
    a = torch.sin((1 - w) * theta) / sin_t
    b = torch.sin(w * theta) / sin_t
    out = a * q0 + b * q1
    # near-parallel: fall back to lerp
    near = (theta < 1e-4).expand_as(out)
    out = torch.where(near, q0 + w * (q1 - q0), out)
    return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def perturbation_field(T, kf_idx, theta_deg, generator, device, mode="independent"):
    """Per-frame, per-joint rotation error of magnitude theta at every keyframe.

    Two error models, because they mean different things for an upstream stage:

    "independent" -- each keyframe is wrong in its own random direction. This is
        a retrieval index or an LLM director proposing each keyframe separately.
        Consecutive keyframes disagree, so the densifier must interpolate between
        mutually inconsistent poses.
    "coherent" -- one fixed error per joint, held across the whole sequence. This
        is a systematic bias: a miscalibrated retarget, a consistently wrong
        elbow. Every keyframe is equally wrong but they agree with each other.

    Contrasting the two separates "the model cannot handle wrong poses" from
    "the model cannot handle inconsistent poses".

    Either way the magnitude is exactly theta at every keyframe. Between
    keyframes the independent field is slerped, because the velocity and contact
    channels are finite differences and per-frame noise would make them
    meaningless.

    Returns [T, 21, 3, 3] rotation matrices.
    """
    theta = torch.full((1,), np.deg2rad(theta_deg), device=device)
    n_kf = len(kf_idx)
    if mode == "coherent":
        axes = _random_axes((1, NJOINTS - 1), generator, device).expand(n_kf, -1, -1)
    elif mode == "independent":
        axes = _random_axes((n_kf, NJOINTS - 1), generator, device)
    else:
        raise ValueError(f"unknown perturbation mode: {mode}")
    q_kf = _axis_angle_to_quat(axes, theta)  # [n_kf, 21, 4]

    frames = torch.arange(T, device=device).float()
    kf = torch.as_tensor(kf_idx, device=device).float()
    hi = torch.searchsorted(kf, frames.contiguous()).clamp(1, n_kf - 1)
    lo = hi - 1
    span = (kf[hi] - kf[lo]).clamp_min(1e-6)
    w = ((frames - kf[lo]) / span).clamp(0, 1)  # [T]

    q = _slerp(q_kf[lo], q_kf[hi], w[:, None].expand(T, NJOINTS - 1))
    return _quat_to_matrix(q)


def perturb_motion(motion, kf_idx, theta_deg, skel, generator, abs_3d=True,
                   mode="independent"):
    """Rotate each joint by theta degrees about a random axis, then rebuild.

    motion: [T, 263] unnormalised.  Returns [T, 263], unnormalised.

    theta_deg = 0 is *not* a no-op: it still round-trips through FK and the
    feature rebuild. That is deliberate -- it makes theta=0 the correct control,
    isolating angular error from the model's own rot/ric inconsistency.
    """
    T = motion.shape[0]
    rot = cont6d_to_matrix(motion[..., ROT].reshape(T, NJOINTS - 1, 6))
    if theta_deg > 0:
        P = perturbation_field(T, kf_idx, theta_deg, generator, motion.device,
                               mode=mode)
        # post-multiply: rotate the joint's own frame, so descendants swing with it
        rot = torch.matmul(rot, P)
    out = motion.clone()
    out[..., ROT] = matrix_to_cont6d(rot).reshape(T, -1)
    positions = forward_kinematics(out, skel, abs_3d=abs_3d)
    return positions_to_features(positions, out, abs_3d=abs_3d)


# --------------------------------------------------------------------- metrics

def joint_positions(motion, abs_3d=True):
    """263-vector -> [T, 22, 3] via the position channels (what the eval uses)."""
    return recover_from_ric(motion.float(), NJOINTS, abs_3d=abs_3d)


def keyframe_error(pred_pos, target_pos, kf_idx):
    """Mean per-joint L2 distance at keyframes, in metres."""
    d = (pred_pos[kf_idx] - target_pos[kf_idx]).norm(dim=-1)
    return d.mean().item()


def jerk(pos, fps=20.0):
    """Mean magnitude of the third time derivative of joint position, m/s^3."""
    if pos.shape[0] < 4:
        return float("nan")
    d3 = pos[3:] - 3 * pos[2:-1] + 3 * pos[1:-2] - pos[:-3]
    return (d3.norm(dim=-1) * fps**3).mean().item()


def skating_ratio(pos):
    """Fraction of frames with a planted foot sliding. pos: [T, 22, 3]."""
    from data_loaders.humanml.utils.metrics import calculate_skating_ratio

    m = pos.permute(1, 2, 0)[None]  # [1, 22, 3, T]
    ratio, _ = calculate_skating_ratio(m)
    return float(ratio[0])


def ground_penetration(pos):
    """Mean depth below the floor plane of the lowest joint, in metres."""
    below = (-pos[..., 1]).clamp_min(0.0)
    return below.max(dim=-1).values.mean().item()


def divergence(a_pos, b_pos):
    """Mean per-joint L2 distance between two motions, in metres."""
    return (a_pos - b_pos).norm(dim=-1).mean().item()
