"""Quantify what the emotion fine-tune learned.

    ml/.venv/bin/python ml/eval_emotion.py --audio e2e/.artifacts/speech.wav

Generates gestures for ONE audio input under every BEAT emotion id with the
fine-tuned model (ml/checkpoints/emage_emotion) and reports, per emotion:

  - upper-body motion energy (mean |axis-angle| over arm/torso joints)
  - hand-position travel per second (world-space, via chain accumulation)
  - divergence from the neutral generation (mean geodesic angle between
    per-joint rotations)

If the embeddings learned nothing, every row is identical (divergence ~0).
Distinct rows = the conditioning channel steers generation.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vendor", "PantoMatrix"))

from generate import axis_angle_to_quat, quat_mul  # noqa: E402
from train_emotion import EMOTIONS  # noqa: E402

UPPER = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
HANDS = [20, 21]
SMPLX_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]


def world_quats(aa22):
    local = axis_angle_to_quat(aa22)
    world = np.zeros_like(local)
    for j in range(22):
        p = SMPLX_PARENTS[j]
        world[:, j] = local[:, j] if p < 0 else quat_mul(world[:, p], local[:, j])
    return world


def geodesic(q1, q2):
    dot = np.abs(np.sum(q1 * q2, axis=-1).clip(-1, 1))
    return 2 * np.arccos(dot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--ckpt", default=os.path.join(os.path.dirname(__file__), "checkpoints", "emage_emotion"))
    ap.add_argument("--out", help="write per-emotion clips to this dir")
    args = ap.parse_args()

    import librosa
    import torch
    import torch.nn.functional as F
    from models.emage_audio import EmageAudioModel, EmageVAEConv, EmageVQModel, EmageVQVAEConv

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hub = "H-Liu1997/emage_audio"
    motion_vq = EmageVQModel(
        face_model=EmageVQVAEConv.from_pretrained(hub, subfolder="emage_vq/face").to(device),
        upper_model=EmageVQVAEConv.from_pretrained(hub, subfolder="emage_vq/upper").to(device),
        lower_model=EmageVQVAEConv.from_pretrained(hub, subfolder="emage_vq/lower").to(device),
        hands_model=EmageVQVAEConv.from_pretrained(hub, subfolder="emage_vq/hands").to(device),
        global_model=EmageVAEConv.from_pretrained(hub, subfolder="emage_vq/global").to(device),
    ).to(device).eval()
    model = EmageAudioModel.from_pretrained(args.ckpt).to(device).eval()
    print(f"checkpoint: {args.ckpt} (speaker_dims={model.cfg.speaker_dims})")

    audio, _ = librosa.load(args.audio, sr=model.cfg.audio_sr)
    audio_t = torch.from_numpy(audio).to(device).unsqueeze(0)

    results = {}
    for emo_id, emo in enumerate(EMOTIONS):
        speaker = torch.full((1, 1), emo_id).long().to(device)
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
        poses = pred["motion_axis_angle"].cpu().numpy().reshape(t, -1, 3)[:, :22]
        results[emo] = poses

    neutral_w = world_quats(results["neutral"])
    print(f"\n{'emotion':<12}{'upper energy':<14}{'hand travel':<13}{'vs neutral'}")
    for emo, poses in results.items():
        energy = float(np.linalg.norm(poses[:, UPPER], axis=-1).mean())
        w = world_quats(poses)
        # Hand "travel": frame-to-frame world-rotation change of the wrists —
        # a proxy that needs no body model.
        travel = float(geodesic(w[1:, HANDS], w[:-1, HANDS]).mean() * 30)
        div = float(geodesic(w[:, UPPER], neutral_w[:, UPPER]).mean())
        print(f"{emo:<12}{energy:<14.4f}{travel:<13.4f}{div:.4f}")

    if args.out:
        from generate import poses_to_clip

        os.makedirs(args.out, exist_ok=True)
        for emo, poses in results.items():
            t = poses.shape[0]
            clip = poses_to_clip(
                np.concatenate([poses.reshape(t, -1), np.zeros((t, 99), dtype=poses.dtype)], axis=1),
                np.zeros((t, 3), dtype=np.float32), 30, pin_feet=True,
            )
            with open(os.path.join(args.out, f"{emo}.json"), "w") as f:
                json.dump(clip, f)
        print(f"\nwrote per-emotion clips to {args.out}")


if __name__ == "__main__":
    main()
