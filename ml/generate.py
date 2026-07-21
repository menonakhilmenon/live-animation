"""Generate a co-speech gesture MotionClip from a speech WAV via EMAGE.

Usage:
    ml/.venv/bin/python ml/generate.py --audio speech.wav --out clip.json

Pipeline: EMAGE (PantoMatrix, Apache-2.0 pretrained weights from
H-Liu1997/emage_audio) decodes audio into SMPL-X axis-angle poses at 30 fps;
we accumulate the per-joint local rotations along the SMPL-X kinematic chain
into WORLD-space rotations. SMPL-X rest joint frames are world-aligned
(identity), so the accumulated world rotation IS the world-space delta from
T-pose — exactly the canonical MotionClip format the web app's ClipPlayer
retargets onto any rig (see src/animation/clip.ts).
"""

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.join(os.path.dirname(__file__), "vendor", "PantoMatrix")
sys.path.insert(0, REPO)

# SMPL-X body joint order (indices into the 55-joint pose vector) and the
# kinematic parents for the first 22 body joints.
SMPLX_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
# SMPL-X index -> our JointName (None = keep for chain accumulation only).
SMPLX_TO_JOINT = {
    0: "hips", 1: "leftUpperLeg", 2: "rightUpperLeg", 3: "spine",
    4: "leftLowerLeg", 5: "rightLowerLeg", 6: None, 7: "leftFoot",
    8: "rightFoot", 9: "chest", 10: None, 11: None, 12: "neck",
    13: "leftShoulder", 14: "rightShoulder", 15: "head",
    16: "leftUpperArm", 17: "rightUpperArm", 18: "leftLowerArm",
    19: "rightLowerArm", 20: "leftHand", 21: "rightHand",
}


def sample_tokens(logits, temperature: float = 0.0, top_p: float = 0.9, generator=None):
    """Pick VQ tokens from classifier logits. temperature<=0 -> argmax
    (EMAGE default, which regresses to the mean: measured 5x slower hands
    than mocap); otherwise nucleus sampling restores motion dynamics."""
    import torch
    import torch.nn.functional as F

    if temperature <= 0:
        return torch.max(F.log_softmax(logits, dim=2), dim=2)[1]
    probs = F.softmax(logits / temperature, dim=-1)
    sp, si = probs.sort(-1, descending=True)
    cum = sp.cumsum(-1)
    sp[cum - sp > top_p] = 0
    sp = sp / sp.sum(-1, keepdim=True)
    flat = sp.reshape(-1, sp.shape[-1])
    choice = torch.multinomial(flat, 1, generator=generator)
    return si.reshape(-1, si.shape[-1]).gather(1, choice).reshape(logits.shape[:-1])


SEED_POSE_PATH = os.path.join(os.path.dirname(__file__), "seed_pose.npy")


def build_seed(device, frames: int = 4):
    """Seed-pose conditioning: (masked_motion, mask) holding a natural
    bent-elbow talking pose (extracted from BEAT2 mocap) in the first
    `frames` frames. EMAGE's autoregressive windows carry the seed forward,
    anchoring posture — unseeded inference rests arms ~30 deg lower than
    real speakers. Returns (None, None) when no seed file exists."""
    if not os.path.exists(SEED_POSE_PATH):
        return None, None
    import torch

    from emage_utils.rotation_conversions import axis_angle_to_rotation_6d

    pose = torch.from_numpy(np.load(SEED_POSE_PATH)).float().reshape(1, 1, 55, 3)
    six = axis_angle_to_rotation_6d(pose).reshape(1, 1, -1)  # (1,1,330)
    row = torch.cat([six, torch.zeros(1, 1, 3), torch.ones(1, 1, 4)], dim=-1)
    masked_motion = row.repeat(1, frames, 1).to(device)
    mask = torch.zeros_like(masked_motion)  # 0 = provided, not to generate
    return masked_motion, mask


def axis_angle_to_quat(aa: np.ndarray) -> np.ndarray:
    """(..., 3) axis-angle -> (..., 4) quaternion [x, y, z, w]."""
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)
    small = angle < 1e-8
    axis = np.where(small, np.array([1.0, 0.0, 0.0]), aa / np.where(small, 1.0, angle))
    half = angle / 2
    return np.concatenate([axis * np.sin(half), np.cos(half)], axis=-1)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of [x,y,z,w] quaternion arrays."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


# SMPL-X finger joints: indices 25-39 (left), 40-54 (right); chain order
# index/middle/pinky/ring/thumb × three segments, each segment's parent is
# the previous one and segment 1 hangs off the wrist (20 left / 21 right).
FINGER_DIGITS = [("index", "Index"), ("middle", "Middle"), ("pinky", "Little"),
                 ("ring", "Ring"), ("thumb", "Thumb")]


def finger_layout():
    """[(smplxIdx, parentSmplxIdx, vrmName)] in parent-first chain order."""
    out = []
    for side, base, wrist in (("left", 25, 20), ("right", 40, 21)):
        for d, (_, vrm_digit) in enumerate(FINGER_DIGITS):
            segs = (["Metacarpal", "Proximal", "Distal"] if vrm_digit == "Thumb"
                    else ["Proximal", "Intermediate", "Distal"])
            for s in range(3):
                idx = base + d * 3 + s
                parent = wrist if s == 0 else idx - 1
                out.append((idx, parent, f"{side}{vrm_digit}{segs[s]}"))
    return out


def poses_to_clip(poses: np.ndarray, trans: np.ndarray, fps: int, pin_feet: bool) -> dict:
    """(t, 55*3) SMPL-X axis-angle + (t, 3) translation -> MotionClip dict."""
    t = poses.shape[0]
    aa = poses.reshape(t, -1, 3)
    nj = aa.shape[1]
    local = axis_angle_to_quat(aa)  # (t, nj, 4)

    world = np.zeros_like(local)
    for j in range(22):
        p = SMPLX_PARENTS[j]
        world[:, j] = local[:, j] if p < 0 else quat_mul(world[:, p], local[:, j])

    joints = [name for i, name in sorted(SMPLX_TO_JOINT.items()) if name]
    idxs = [i for i, name in sorted(SMPLX_TO_JOINT.items()) if name]
    rotations = world[:, idxs]  # (t, len(joints), 4)

    fingers = None
    if nj >= 55:
        layout = finger_layout()
        for idx, parent, _ in layout:  # parent-first: parents already done
            world[:, idx] = quat_mul(world[:, parent], local[:, idx])
        fingers = {
            "joints": [name for _, _, name in layout],
            "rotations": np.round(world[:, [i for i, _, _ in layout]], 5).tolist(),
        }

    # Root translation: EMAGE integrates a predicted velocity, which wanders
    # over long clips (~0.8 m over 8 s observed). High-pass X/Z against a
    # slow EMA so short-term weight shifts survive but drift is removed;
    # Y (weight bobs) is kept as the offset from the first frame.
    hips = (trans - trans[0:1]).copy()
    ema = hips[0, [0, 2]].copy()
    alpha = 1.0 / (fps * 2.0)  # ~2 s time constant
    for i in range(t):
        ema += alpha * (hips[i, [0, 2]] - ema)
        hips[i, [0, 2]] -= ema
    clip = {
        "fps": fps,
        "joints": joints,
        "rotations": np.round(rotations, 5).tolist(),
        "hipsPosition": np.round(hips, 5).tolist(),
        "pinFeet": pin_feet,
    }
    if fingers:
        clip["fingers"] = fingers
    return clip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--npz-out", help="also save the raw SMPL-X npz here")
    ap.add_argument("--free-feet", action="store_true", help="disable foot IK pinning")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="VQ sampling temperature for the upper body (0 = argmax)")
    ap.add_argument("--hands-temperature", type=float, default=0.7,
                    help="VQ sampling temperature for the hands")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import librosa
    import torch
    import torch.nn.functional as F
    from models.emage_audio import EmageAudioModel, EmageVAEConv, EmageVQModel, EmageVQVAEConv

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    hub = "H-Liu1997/emage_audio"
    print(f"loading EMAGE from {hub} on {device} ...", flush=True)
    motion_vq = EmageVQModel(
        face_model=EmageVQVAEConv.from_pretrained(hub, subfolder="emage_vq/face").to(device),
        upper_model=EmageVQVAEConv.from_pretrained(hub, subfolder="emage_vq/upper").to(device),
        lower_model=EmageVQVAEConv.from_pretrained(hub, subfolder="emage_vq/lower").to(device),
        hands_model=EmageVQVAEConv.from_pretrained(hub, subfolder="emage_vq/hands").to(device),
        global_model=EmageVAEConv.from_pretrained(hub, subfolder="emage_vq/global").to(device),
    ).to(device).eval()
    model = EmageAudioModel.from_pretrained(hub).to(device).eval()

    audio, _ = librosa.load(args.audio, sr=model.cfg.audio_sr)
    audio_t = torch.from_numpy(audio).to(device).unsqueeze(0)
    speaker = torch.zeros(1, 1).long().to(device)
    trans_seed = torch.zeros(1, 1, 3).to(device)

    print(f"generating gestures for {len(audio) / model.cfg.audio_sr:.1f}s of audio ...", flush=True)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    seed_motion, seed_mask = build_seed(device)
    with torch.no_grad():
        lat = model.inference(audio_t, speaker, motion_vq, masked_motion=seed_motion, mask=seed_mask)
        cfg = model.cfg
        pick = lambda cls_key, rec_key, c, l, temp=0.0: (  # noqa: E731
            sample_tokens(lat[cls_key].cpu(), temp, generator=gen).to(device) if c > 0 else None,
            lat[rec_key] if l > 0 and c == 0 else None,
        )
        face_index, face_latent = pick("cls_face", "rec_face", cfg.cf, cfg.lf)
        upper_index, upper_latent = pick("cls_upper", "rec_upper", cfg.cu, cfg.lu, args.temperature)
        hands_index, hands_latent = pick("cls_hands", "rec_hands", cfg.ch, cfg.lh, args.hands_temperature)
        lower_index, lower_latent = pick("cls_lower", "rec_lower", cfg.cl, cfg.ll)
        pred = motion_vq.decode(
            face_latent=face_latent, upper_latent=upper_latent,
            lower_latent=lower_latent, hands_latent=hands_latent,
            face_index=face_index, upper_index=upper_index,
            lower_index=lower_index, hands_index=hands_index,
            get_global_motion=True, ref_trans=trans_seed[:, 0],
        )

    t = pred["motion_axis_angle"].shape[1]
    poses = pred["motion_axis_angle"].cpu().numpy().reshape(t, -1)
    trans = pred["trans"].cpu().numpy().reshape(t, -1)
    expressions = pred["expression"].cpu().numpy().reshape(t, -1)
    print(f"decoded {t} frames @ {cfg.pose_fps} fps", flush=True)

    if args.npz_out:
        np.savez(args.npz_out, poses=poses, trans=trans, expressions=expressions)

    clip = poses_to_clip(poses, trans, cfg.pose_fps, pin_feet=not args.free_feet)
    with open(args.out, "w") as f:
        json.dump(clip, f)
    print(f"wrote {args.out}: {len(clip['joints'])} joints, {t} frames", flush=True)


if __name__ == "__main__":
    main()
