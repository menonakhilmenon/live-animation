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

LINES = {
    "idle_calm": ("", 0),  # near-silence -> subtle idle sway
    "talk_neutral": ("Let me walk you through what happened here, step by step, so it all makes sense.", 0),
    "talk_happiness": ("This is wonderful news, honestly one of the best things I have heard all year!", 1),
    "talk_anger": ("This is completely unacceptable, and I need you to understand exactly why right now.", 2),
    "talk_sadness": ("I really wish things had turned out differently for everyone involved in this.", 3),
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

    for name, (line, emo_id) in LINES.items():
        if line:
            import librosa

            chunks = [r.audio.numpy() for r in tts(line, voice="af_heart")]
            audio = np.concatenate(chunks)
            audio = librosa.resample(audio, orig_sr=24000, target_sr=model.cfg.audio_sr)
        else:
            audio = np.random.default_rng(7).normal(0, 1e-4, model.cfg.audio_sr * int(args.seconds)).astype(np.float32)

        audio_t = torch.from_numpy(audio.astype(np.float32)).to(device).unsqueeze(0)
        sid = emo_id if model.cfg.speaker_dims >= 8 else 0
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

        q = axis_angle_to_quat(poses)  # local quats (t, 55, 4)
        q = smooth_quats(q, passes=2)
        q = loopify(q, cfg.pose_fps)
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
