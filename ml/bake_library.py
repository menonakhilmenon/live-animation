"""Bake the prebaked-animation library from the emotion-conditioned model.

    ml/.venv/bin/python ml/bake_library.py [--out public/anims]

For each emotion style we synthesize a representative spoken line (Kokoro),
generate gestures with the fine-tuned EMAGE, then CLEAN the motion into a
prebaked loop:

  - quaternion smoothing (slerp-based moving average) removes the frame
    jitter raw VQ decoding can show
  - the final second is crossfaded into the first so the clip loops
  - root translation is dropped (base loops stay planted; the web side
    re-adds weight shift procedurally)

Output: canonical MotionClip JSON in public/anims/, committed as assets —
the runtime never plays raw model output, only these curated loops plus
Xbot's embedded clips (idle/agree/headShake), per the base+additive
architecture.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vendor", "PantoMatrix"))

from generate import poses_to_clip  # noqa: E402

# (line, BEAT emotion id, conditioning intensity). Intensity < 1 lerps the
# emotion embedding toward neutral: BEAT2's emotion sessions are acted and
# full-strength styles read as theatrical on casual utterances. The idle
# loop is NOT baked — EMAGE degenerates on silence; the scheduler uses
# Xbot's authored 'idle' clip instead.
LINES = {
    "talk_neutral": ("So I was thinking about this earlier, and there are a couple of things worth mentioning before we get into the details.", 0, 1.0),
    "talk_happiness": ("Oh that's really nice to hear, I'm glad it worked out, and honestly it makes me happy just thinking about it.", 1, 0.6),
    "talk_anger": ("Look, I've said this before and I'll say it again, this is not how it was supposed to go and you know it.", 2, 0.6),
    "talk_sadness": ("Yeah, I heard about it this morning. It's been on my mind all day, and I still don't really know what to say.", 3, 0.6),
}


def slerp(q0, q1, t):
    d = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(d < 0, -q1, q1)
    return normalize(q0 + (q1 - q0) * t)  # nlerp — fine for tiny angles


def normalize(q):
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def smooth_quats(q, passes=2):
    """(t, j, 4) light temporal smoothing via neighbor averaging."""
    for _ in range(passes):
        prev = np.roll(q, 1, axis=0)
        nxt = np.roll(q, -1, axis=0)
        prev[0] = q[0]
        nxt[-1] = q[-1]
        q = normalize(q + 0.5 * (align(prev, q) + align(nxt, q)))
    return q


def align(a, ref):
    d = np.sum(a * ref, axis=-1, keepdims=True)
    return np.where(d < 0, -a, a)


def remove_root_yaw(aa):
    """Zero the Y (turning) component of the pelvis rotation per frame —
    the source speaker wanders/turns, but a talk LOOP must stay facing
    forward or it replays whole-body twisting against the foot IK."""
    out = aa.copy()
    root = out[:, 0]
    # Twist-swing decomposition about Y via quaternions.
    q = axis_angle_to_quat_1(root)
    twist = q.copy()
    twist[:, 0] = 0.0
    twist[:, 2] = 0.0
    norm = np.linalg.norm(twist, axis=-1, keepdims=True)
    ok = norm[:, 0] > 1e-8
    twist[ok] /= norm[ok]
    twist[~ok] = np.array([0, 0, 0, 1.0])
    # swing = q * conj(twist)
    conj = twist * np.array([-1, -1, -1, 1.0])
    swing = quat_mul_np(q, conj)
    out[:, 0] = quat_to_axis_angle(swing)
    return out


def axis_angle_to_quat_1(aa):
    from generate import axis_angle_to_quat

    return axis_angle_to_quat(aa)


def quat_mul_np(a, b):
    from generate import quat_mul

    return quat_mul(a, b)


def quat_to_axis_angle(q):
    q = normalize(q)
    w = np.clip(q[..., 3:4], -1, 1)
    ang = 2 * np.arccos(np.abs(w))
    sign = np.where(w < 0, -1.0, 1.0)
    s = np.sqrt(np.clip(1 - w * w, 1e-12, None))
    return q[..., :3] * sign * (ang / s)


def rotate_to_calmest_start(q):
    """Cycle the loop so frame 0 is the frame closest to the sequence mean
    (short utterances then begin from a near-neutral pose)."""
    mean = normalize(q.mean(axis=0, keepdims=True))
    d = np.sum(align(q, mean) * mean, axis=-1).clip(-1, 1)
    dist = (2 * np.arccos(np.abs(d))).sum(axis=-1)
    start = int(np.argmin(dist))
    return np.roll(q, -start, axis=0)


def loopify(q, fps, blend_s=1.0):
    """Crossfade the tail into the head so the clip loops seamlessly."""
    n = q.shape[0]
    b = min(int(blend_s * fps), n // 3)
    out = q.copy()
    for i in range(b):
        t = (i + 1) / (b + 1)
        w = t * t * (3 - 2 * t)
        out[n - b + i] = slerp(q[n - b + i], q[i], w)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "public", "anims"))
    ap.add_argument("--seconds", type=float, default=7.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import torch
    import torch.nn.functional as F
    from kokoro import KPipeline
    from models.emage_audio import EmageAudioModel, EmageVAEConv, EmageVQModel, EmageVQVAEConv

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        free, _ = torch.cuda.mem_get_info()
        if free < 1.2 * 2**30:
            device = "cpu"
    hub = "H-Liu1997/emage_audio"
    ckpt = os.path.join(os.path.dirname(__file__), "checkpoints", "emage_emotion")
    motion_vq = EmageVQModel(
        face_model=EmageVQVAEConv.from_pretrained(hub, subfolder="emage_vq/face").to(device),
        upper_model=EmageVQVAEConv.from_pretrained(hub, subfolder="emage_vq/upper").to(device),
        lower_model=EmageVQVAEConv.from_pretrained(hub, subfolder="emage_vq/lower").to(device),
        hands_model=EmageVQVAEConv.from_pretrained(hub, subfolder="emage_vq/hands").to(device),
        global_model=EmageVAEConv.from_pretrained(hub, subfolder="emage_vq/global").to(device),
    ).to(device).eval()
    model = EmageAudioModel.from_pretrained(ckpt if os.path.isdir(ckpt) else hub).to(device).eval()
    tts = KPipeline(lang_code="a", device="cpu", repo_id="hexgrad/Kokoro-82M")

    # Scratch embedding row for partial-intensity conditioning.
    import torch.nn as nn

    if model.cfg.speaker_dims >= 8:
        for attr in ("speaker_embedding_body", "speaker_embedding_face"):
            old = getattr(model, attr)
            new = nn.Embedding(old.num_embeddings + 1, old.embedding_dim).to(device)
            new.weight.data[: old.num_embeddings] = old.weight.data
            setattr(model, attr, new)
        scratch = model.speaker_embedding_body.num_embeddings - 1

    for name, (line, emo_id, intensity) in LINES.items():
        import librosa

        chunks = [r.audio.numpy() for r in tts(line, voice="af_heart")]
        audio = np.concatenate(chunks)
        audio = librosa.resample(audio, orig_sr=24000, target_sr=model.cfg.audio_sr)

        audio_t = torch.from_numpy(audio.astype(np.float32)).to(device).unsqueeze(0)
        if model.cfg.speaker_dims >= 8:
            with torch.no_grad():
                for attr in ("speaker_embedding_body", "speaker_embedding_face"):
                    w = getattr(model, attr).weight
                    w.data[scratch] = (1 - intensity) * w.data[0] + intensity * w.data[emo_id]
            sid = scratch
        else:
            sid = 0
        speaker = torch.full((1, 1), sid).long().to(device)
        with torch.no_grad():
            lat = model.inference(audio_t, speaker, motion_vq, masked_motion=None, mask=None)
            cfg = model.cfg
            pick = lambda ck, rk, c, l: (  # noqa: E731
                torch.max(F.log_softmax(lat[ck], dim=2), dim=2)[1] if c > 0 else None,
                lat[rk] if l > 0 and c == 0 else None,
            )
            fi, fl = pick("cls_face", "rec_face", cfg.cf, cfg.lf)
            ui, ul = pick("cls_upper", "rec_upper", cfg.cu, cfg.lu)
            hi, hl = pick("cls_hands", "rec_hands", cfg.ch, cfg.lh)
            li, ll = pick("cls_lower", "rec_lower", cfg.cl, cfg.ll)
            pred = motion_vq.decode(
                face_latent=fl, upper_latent=ul, lower_latent=ll, hands_latent=hl,
                face_index=fi, upper_index=ui, lower_index=li, hands_index=hi,
                get_global_motion=True, ref_trans=torch.zeros(1, 3).to(device),
            )
        t = pred["motion_axis_angle"].shape[1]
        poses = pred["motion_axis_angle"].cpu().numpy().reshape(t, 55, 3)

        # Trim to the target length from the middle (skip the seeded start).
        want = int(args.seconds * cfg.pose_fps)
        if t > want + 30:
            start = (t - want) // 2
            poses = poses[start : start + want]
        t = poses.shape[0]

        # Smooth + loopify in quaternion space, then back to axis-angle via
        # the clip converter (which consumes axis-angle directly).
        from generate import axis_angle_to_quat  # noqa: E402

        poses = remove_root_yaw(poses)
        q = axis_angle_to_quat(poses)  # local quats (t, 55, 4)
        q = smooth_quats(q, passes=2)
        # Seal the loop seam FIRST, then rotate the phase — rolling before
        # sealing drags the raw end->start discontinuity into the middle of
        # the loop (observed as a one-frame 0.33 m hand teleport).
        q = loopify(q, cfg.pose_fps)
        q = rotate_to_calmest_start(q)
        # quats -> axis-angle
        q = normalize(q)
        w = np.clip(q[..., 3:4], -1, 1)
        ang = 2 * np.arccos(np.abs(w))
        sign = np.where(w < 0, -1.0, 1.0)
        s = np.sqrt(np.clip(1 - w * w, 1e-12, None))
        aa = q[..., :3] * sign * (ang / s)

        clip = poses_to_clip(aa.reshape(t, -1), np.zeros((t, 3), dtype=np.float32), cfg.pose_fps, pin_feet=True)
        del clip["hipsPosition"]  # base loops stay planted
        path = os.path.join(args.out, f"{name}.json")
        with open(path, "w") as f:
            json.dump(clip, f)
        print(f"baked {name}: {t} frames -> {path} ({os.path.getsize(path) // 1024} KB)", flush=True)


if __name__ == "__main__":
    main()
