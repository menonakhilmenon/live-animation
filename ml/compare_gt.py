"""Ground truth vs generation: same audio, same converter, same avatar.

    ml/.venv/bin/python ml/compare_gt.py [--stem 2_scott_0_9_9] [--seconds 10]

Loads a held-out BEAT2 recording, converts the REAL mocap through the
exact same SMPL-X -> MotionClip pipeline the generations use, generates
EMAGE motion for the same audio window, and prints matched motion
statistics. Writes gt.json / gen.json (+ the audio window) so the web app
can play both on the avatar for visual comparison.

Interpretation: if GT looks natural on the avatar, the conversion and
retargeting are faithful and any visual gap is the model's; if GT also
looks wrong, the pipeline itself distorts motion.
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor", "PantoMatrix"))

from generate import poses_to_clip  # noqa: E402
from train_emotion import DATA, emotion_of_stem  # noqa: E402

UPPER = {
    "spine": [3], "chest": [6, 9], "head": [12, 15],
    "arms": [16, 17, 18, 19], "wrists": [20, 21],
    "fingers": list(range(25, 55)),
}


def motion_stats(poses):
    """(t, 165) axis-angle -> readable per-group posture/dynamics stats."""
    aa = poses.reshape(len(poses), -1, 3)
    ang = np.linalg.norm(aa, axis=-1)  # (t, 55) local angle magnitude
    vel = np.abs(np.diff(ang, axis=0)) * 30  # rad/s
    out = {}
    for name, idxs in UPPER.items():
        out[name] = (
            float(np.degrees(ang[:, idxs].mean())),
            float(np.degrees(ang[:, idxs].max())),
            float(np.degrees(vel[:, idxs].mean())),
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="2_scott_0_9_9")  # held-out neutral speech
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "..", "e2e", ".artifacts", "compare"))
    ap.add_argument("--temperature", type=float, default=0.0, help="VQ sampling temperature (0 = argmax)")
    ap.add_argument("--hands-temperature", type=float, default=None,
                    help="hands head temperature (default: same as --temperature)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-seed-pose", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import librosa
    import soundfile as sf
    import torch
    import torch.nn.functional as F
    from models.emage_audio import EmageAudioModel, EmageVAEConv, EmageVQModel, EmageVQVAEConv

    npz = np.load(os.path.join(DATA, "smplxflame_30", args.stem + ".npz"), allow_pickle=True)
    audio, _ = librosa.load(os.path.join(DATA, "wave16k", args.stem + ".wav"), sr=16000)
    poses = npz["poses"].astype(np.float32)
    trans = npz["trans"].astype(np.float32)
    n = len(poses)
    want = int(args.seconds * 30)
    start = max(0, (n - want) // 2)
    poses_gt = poses[start : start + want]
    trans_gt = trans[start : start + want]
    a0 = start * 16000 // 30
    audio_win = audio[a0 : a0 + int(args.seconds * 16000)]
    emo = emotion_of_stem(args.stem)
    print(f"recording {args.stem} (emotion id {emo}), window {args.seconds}s from frame {start}")

    device = "cpu" if not torch.cuda.is_available() else "cuda"
    if device == "cuda":
        free, _ = torch.cuda.mem_get_info()
        if free < 1.2 * 2**30:
            device = "cpu"
    hub = "H-Liu1997/emage_audio"
    ckpt = os.path.join(ROOT, "checkpoints", "emage_emotion")
    motion_vq = EmageVQModel(
        face_model=EmageVQVAEConv.from_pretrained(hub, subfolder="emage_vq/face").to(device),
        upper_model=EmageVQVAEConv.from_pretrained(hub, subfolder="emage_vq/upper").to(device),
        lower_model=EmageVQVAEConv.from_pretrained(hub, subfolder="emage_vq/lower").to(device),
        hands_model=EmageVQVAEConv.from_pretrained(hub, subfolder="emage_vq/hands").to(device),
        global_model=EmageVAEConv.from_pretrained(hub, subfolder="emage_vq/global").to(device),
    ).to(device).eval()
    model = EmageAudioModel.from_pretrained(ckpt if os.path.isdir(ckpt) else hub).to(device).eval()

    audio_t = torch.from_numpy(audio_win).to(device).unsqueeze(0)
    sid = emo if model.cfg.speaker_dims >= 8 else 0
    speaker = torch.full((1, 1), sid).long().to(device)
    from generate import build_seed, sample_tokens

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    seed_motion, seed_mask = (None, None) if args.no_seed_pose else build_seed(device)
    with torch.no_grad():
        lat = model.inference(audio_t, speaker, motion_vq, masked_motion=seed_motion, mask=seed_mask)
        cfg = model.cfg
        hands_t = args.hands_temperature if args.hands_temperature is not None else args.temperature
        pick = lambda ck, rk, c, l, temp: (  # noqa: E731
            sample_tokens(lat[ck].cpu(), temp, generator=gen).to(device) if c > 0 else None,
            lat[rk] if l > 0 and c == 0 else None,
        )
        # Production policy (ml/server.py): face and lower body stay argmax.
        fi, fl = pick("cls_face", "rec_face", cfg.cf, cfg.lf, 0.0)
        ui, ul = pick("cls_upper", "rec_upper", cfg.cu, cfg.lu, args.temperature)
        hi, hl = pick("cls_hands", "rec_hands", cfg.ch, cfg.lh, hands_t)
        li, ll = pick("cls_lower", "rec_lower", cfg.cl, cfg.ll, 0.0)
        pred = motion_vq.decode(
            face_latent=fl, upper_latent=ul, lower_latent=ll, hands_latent=hl,
            face_index=fi, upper_index=ui, lower_index=li, hands_index=hi,
            get_global_motion=True, ref_trans=torch.zeros(1, 3).to(device),
        )
    t = pred["motion_axis_angle"].shape[1]
    poses_gen = pred["motion_axis_angle"].cpu().numpy().reshape(t, -1)
    trans_gen = pred["trans"].cpu().numpy().reshape(t, -1)

    print(f"\n{'group':<8}{'GT mean/max deg (vel deg/s)':<34}{'GEN mean/max deg (vel deg/s)'}")
    gt_s, gen_s = motion_stats(poses_gt), motion_stats(poses_gen)
    for g in UPPER:
        a, b = gt_s[g], gen_s[g]
        print(f"{g:<8}{a[0]:6.1f} /{a[1]:6.1f}  ({a[2]:6.1f})        {b[0]:6.1f} /{b[1]:6.1f}  ({b[2]:6.1f})")

    for name, p, tr in (("gt", poses_gt, trans_gt), ("gen", poses_gen, trans_gen)):
        clip = poses_to_clip(p, tr, 30, pin_feet=True)
        with open(os.path.join(args.out, f"{name}.json"), "w") as f:
            json.dump(clip, f)
    sf.write(os.path.join(args.out, "audio.wav"), audio_win, 16000)
    print(f"\nwrote gt.json / gen.json / audio.wav -> {args.out}")


if __name__ == "__main__":
    main()
