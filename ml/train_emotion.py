"""Fine-tune EMAGE into an EMOTION-conditioned co-speech gesture model.

    ml/.venv/bin/python ml/train_emotion.py [--steps 4000] [--bs 16]

Data: BEAT2 speaker 2 (scott) — 130 recordings spanning neutral conversation
and the 8 BEAT emotion self-talk sessions (ml/data/BEAT2, ungated HF
download; see ml/prepare note in docs/generative-animation.md).

Conditioning trick: the pretrained checkpoint has speaker_dims=1 and the
original training always feeds speaker_id=0, so the speaker-embedding input
is a free conditioning channel. We resize both speaker embeddings to
len(EMOTIONS) rows (all initialized from the pretrained row -> identical
behavior at step 0) and feed the recording's emotion id during fine-tuning.
At inference, the emotion id steers gesture style.

Loss/masking replicate PantoMatrix's train_emage_audio.py (rec + cls over
three forward passes: seed-frames mask, random mask + audio, random mask
without audio). VQ codebooks stay frozen. Foot contact is approximated
from root translation velocity (the exact version needs gated SMPL-X body
files; talking-sequence feet are near-static so this is benign).
"""

import argparse
import functools
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT, "vendor", "PantoMatrix"))

DATA = os.path.join(ROOT, "data", "BEAT2", "beat_english_v2.0.0")
OUT = os.path.join(ROOT, "checkpoints", "emage_emotion")

# BEAT filename -> emotion, from the official BEAT README + dataloader
# (scripts/BEAT_2022/readme_beat.md L36-46 and EMAGE_2024/dataloaders/
# beat_sep.py L454-478 in PantoMatrix@6ca70b9): filenames are
# {speakerID}_{name}_{recType}_{seqStart}_{seqEnd}; recType 0 = English
# self-talk (emotion follows seqStart per the table below), recType 1 =
# English conversation (always neutral).
EMOTIONS = ["neutral", "happiness", "anger", "sadness", "contempt", "surprise", "fear", "disgust"]
EMOTION_BLOCKS = {
    (1, 64): 0, (65, 72): 1, (73, 80): 2, (81, 86): 3,
    (87, 94): 4, (95, 102): 5, (103, 110): 6, (111, 118): 7,
}


def emotion_of_stem(stem: str) -> int:
    parts = stem.split("_")
    rectype, seq = int(parts[2]), int(parts[3])
    if rectype % 2 == 1:
        return 0  # conversation sessions are all neutral
    for (lo, hi), emo in EMOTION_BLOCKS.items():
        if lo <= seq <= hi:
            return emo
    return 0


@functools.lru_cache(maxsize=160)
def load_recording(stem: str):
    import librosa

    npz = np.load(os.path.join(DATA, "smplxflame_30", stem + ".npz"), allow_pickle=True)
    audio, _ = librosa.load(os.path.join(DATA, "wave16k", stem + ".wav"), sr=16000)
    poses = npz["poses"].astype(np.float32)
    trans = npz["trans"].astype(np.float32)
    expr = npz["expressions"].astype(np.float32)
    # Foot-contact proxy: root nearly still => feet planted.
    v = np.zeros(len(trans), dtype=np.float32)
    v[1:] = np.linalg.norm(np.diff(trans[:, [0, 2]], axis=0), axis=1)
    contact = (v < 0.003).astype(np.float32)[:, None].repeat(4, axis=1)
    return poses, trans, expr, contact, audio


def build_windows(length: int, stride: int):
    """(stem, emotion, start) windows from the downloaded recordings.

    Split: hold out the last recording of each emotion (the official CSV
    marks most of scott as test — he's the benchmark speaker — which would
    starve an emotion fine-tune of data).
    """
    stems = sorted(
        f[:-4] for f in os.listdir(os.path.join(DATA, "smplxflame_30")) if f.endswith(".npz")
    )
    by_emo = {}
    for stem in stems:
        by_emo.setdefault(emotion_of_stem(stem), []).append(stem)
    val_stems = {stems[-1] for stems in by_emo.values() if len(stems) > 1}

    train, val = [], []
    for stem in stems:
        emo = emotion_of_stem(stem)
        poses, *_ = load_recording(stem)
        n = poses.shape[0]
        wins = [(stem, emo, s) for s in range(0, n - length, stride)]
        (val if stem in val_stems else train).extend(wins)
    return train, val


class Windows:
    def __init__(self, wins, length):
        self.wins = wins
        self.length = length

    def __len__(self):
        return len(self.wins)

    def __getitem__(self, i):
        import torch

        stem, emo, s = self.wins[i]
        poses, trans, expr, contact, audio = load_recording(stem)
        e = s + self.length
        # Fixed audio window length — per-window rounding would make batch
        # items differ by a sample and break collation.
        na = self.length * 16000 // 30
        sa = s * 16000 // 30
        a = audio[sa : sa + na]
        if len(a) < na:
            a = np.pad(a, (0, na - len(a)))
        return dict(
            motion=torch.from_numpy(poses[s:e]),
            audio=torch.from_numpy(a),
            expressions=torch.from_numpy(expr[s:e]),
            trans=torch.from_numpy(trans[s:e]),
            foot_contact=torch.from_numpy(contact[s:e]),
            emotion=torch.tensor([emo]),
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None,
                    help="default: 1e-2 for --embeddings-only, else 5e-5")
    ap.add_argument("--length", type=int, default=64)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--resume", default=None, help="checkpoint dir to resume from")
    ap.add_argument(
        "--embeddings-only", action="store_true",
        help="freeze the pretrained net; train just the emotion embeddings "
        "(textual-inversion-style — tiny VRAM, base quality preserved)",
    )
    ap.add_argument(
        "--unfreeze", default=None,
        help="comma list of extra trainable groups: heads (VQ cls + out "
        "projections, ~2.4M), latent (motion2latent adapters, ~3.5M). "
        "Implies embeddings are trainable too.",
    )
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    import emage_utils.rotation_conversions as rc
    from models.emage_audio import EmageAudioModel, EmageVAEConv, EmageVQModel, EmageVQVAEConv

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    if device == "cuda":
        # The fused SDPA backward (AOTriton) ships no gfx1201 kernel in the
        # rocm6.4 wheels — 'no kernel image' at .backward(). Math SDPA works.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    hub = "H-Liu1997/emage_audio"
    motion_vq = EmageVQModel(
        face_model=EmageVQVAEConv.from_pretrained(hub, subfolder="emage_vq/face").to(device),
        upper_model=EmageVQVAEConv.from_pretrained(hub, subfolder="emage_vq/upper").to(device),
        lower_model=EmageVQVAEConv.from_pretrained(hub, subfolder="emage_vq/lower").to(device),
        hands_model=EmageVQVAEConv.from_pretrained(hub, subfolder="emage_vq/hands").to(device),
        global_model=EmageVAEConv.from_pretrained(hub, subfolder="emage_vq/global").to(device),
    ).to(device).eval()
    for p in motion_vq.parameters():
        p.requires_grad = False

    model = EmageAudioModel.from_pretrained(args.resume or hub).to(device)
    if model.cfg.speaker_dims < len(EMOTIONS):
        # Widen the (single-row) speaker embedding into an emotion embedding.
        for name in ("speaker_embedding_body", "speaker_embedding_face"):
            old = getattr(model, name)
            new = nn.Embedding(len(EMOTIONS), old.embedding_dim).to(device)
            new.weight.data[:] = old.weight.data[0:1]
            setattr(model, name, new)
        model.cfg.speaker_dims = len(EMOTIONS)
        model.config.speaker_dims = len(EMOTIONS)
    if args.embeddings_only or args.unfreeze:
        groups = set((args.unfreeze or "").split(",")) - {""}
        def trainable_name(n: str) -> bool:
            if "speaker_embedding" in n:
                return True
            if "heads" in groups and ("_cls" in n or "cls_" in n or "out_proj" in n):
                return True
            if "latent" in groups and "motion2latent" in n:
                return True
            return False
        for name, p in model.named_parameters():
            p.requires_grad = trainable_name(name)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"trainable params: {trainable/1e6:.2f}M (groups: embeddings+{groups})", flush=True)
    model.train()

    train_wins, val_wins = build_windows(args.length, args.stride)
    emo_counts = np.bincount([w[1] for w in train_wins], minlength=len(EMOTIONS))
    print(f"train windows: {len(train_wins)}  val: {len(val_wins)}", flush=True)
    print("per-emotion train windows:", dict(zip(EMOTIONS, emo_counts.tolist())), flush=True)
    # Emotion classes are imbalanced (64 neutral conversation recordings vs
    # 4 per emotion) — oversample so each batch sees emotions uniformly.
    weights = torch.tensor([1.0 / max(1, emo_counts[w[1]]) for w in train_wins])
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(train_wins))
    train_loader = DataLoader(
        Windows(train_wins, args.length), batch_size=args.bs, sampler=sampler,
        drop_last=True, num_workers=4,
    )
    val_loader = DataLoader(Windows(val_wins, args.length), batch_size=args.bs, num_workers=2)

    lr = args.lr if args.lr is not None else (1e-2 if args.embeddings_only else 5e-5)
    emb_params = [p for n, p in model.named_parameters() if p.requires_grad and "speaker_embedding" in n]
    other_params = [p for n, p in model.named_parameters() if p.requires_grad and "speaker_embedding" not in n]
    # Pretrained (or warm-resumed) non-embedding layers get a gentler LR
    # than the conditioning embeddings.
    opt = torch.optim.Adam(
        [
            {"params": emb_params, "lr": lr if args.lr is not None else 1e-3},
            {"params": other_params, "lr": lr if args.lr is not None else 1e-4},
        ],
        betas=(0.9, 0.999),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    cls_fn = nn.NLLLoss()

    def losses(batch, it, backward=False):
        motion = batch["motion"].to(device)
        audio = batch["audio"].to(device)
        expr = batch["expressions"].to(device)
        trans = batch["trans"].to(device)
        contact = batch["foot_contact"].to(device)
        emo = batch["emotion"].to(device)
        bs, t, jc = motion.shape
        motion6d = rc.axis_angle_to_rotation_6d(motion.reshape(bs, t, jc // 3, 3)).reshape(bs, t, -1)
        with torch.no_grad():
            idx_gt = motion_vq.map2index(motion6d, expr, tar_contact=contact, tar_trans=trans)
            lat_gt = motion_vq.map2latent(motion6d, expr, tar_contact=contact, tar_trans=trans)
        masked = torch.cat([motion6d, trans, contact], dim=-1)
        cfg = model.cfg

        def fwd(mask, use_audio):
            pred = model(audio, emo, masked_motion=masked, mask=mask, use_audio=use_audio)
            rec = sum(
                w * F.mse_loss(pred[f"rec_{k}"], lat_gt[k])
                for k, w in (("upper", cfg.lu), ("lower", cfg.ll), ("hands", cfg.lh), ("face", cfg.lf))
            )
            cls = sum(
                w * cls_fn(F.log_softmax(pred[f"cls_{k}"], dim=2).permute(0, 2, 1), idx_gt[k])
                for k, w in (("upper", cfg.cu), ("lower", cfg.cl), ("hands", cfg.ch), ("face", cfg.cf))
                if w > 0
            )
            return rec + cls

        seed_mask = torch.ones_like(masked)
        seed_mask[:, : cfg.seed_frames] = 0
        ratio = min(1.0, (it / args.steps) * 0.95 + 0.05)
        rand_mask = (torch.rand(bs, t, masked.shape[-1], device=device) < ratio).float()
        # Backward each pass immediately (grads accumulate) so only ONE
        # forward graph is live at a time — the VRAM budget is tiny because
        # the user's other GPU services stay running.
        total = 0.0
        for mask, use_audio in ((seed_mask, True), (rand_mask, True), (rand_mask, False)):
            loss = fwd(mask, use_audio)
            if backward:
                loss.backward()
                total += loss.item()
            else:
                total += loss.item() if not loss.requires_grad else loss.detach().item()
        return total

    os.makedirs(OUT, exist_ok=True)
    it, t0, ema_loss, best_val = 0, time.time(), None, float("inf")
    while it < args.steps:
        for batch in train_loader:
            if it >= args.steps:
                break
            opt.zero_grad()
            loss_val = losses(batch, it, backward=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.99)
            opt.step()
            sched.step()
            ema_loss = loss_val if ema_loss is None else 0.98 * ema_loss + 0.02 * loss_val
            it += 1
            if it % 25 == 0:
                print(
                    f"iter {it}/{args.steps} loss {loss_val:.4f} ema {ema_loss:.4f} "
                    f"lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / it:.2f}s/it",
                    flush=True,
                )
            if it % args.save_every == 0 or it == args.steps:
                model.save_pretrained(OUT)
                with torch.no_grad():
                    model.eval()
                    # Fixed-difficulty eval: the training mask ratio ramps
                    # with `it`, which would make val numbers incomparable
                    # across checkpoints — pin it to mid-curriculum.
                    vl = [losses(b, args.steps // 2) for _, b in zip(range(24), val_loader)]
                    model.train()
                val = float(np.mean(vl))
                if val < best_val:
                    best_val = val
                    model.save_pretrained(OUT + "_best")
                print(f"iter {it} saved -> {OUT}  val_loss {val:.4f} (best {best_val:.4f})", flush=True)
    print("TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()
