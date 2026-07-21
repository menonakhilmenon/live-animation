"""Train the learned accent scheduler: audio -> when to nod / head-shake.

    ml/.venv/bin/python ml/train_scheduler.py [--epochs 12]

The prebaked+additive architecture leaves one true decision to a model:
WHICH accent plays WHEN. Ground truth comes from BEAT2 motion itself —
nod events (1-4 Hz head-pitch oscillation bursts) and shake events (yaw
sign alternations) are detected in the mocap, and a small MLP learns to
predict them from the speech audio alone (prosody). The sidecar then runs
this model on TTS output to place accents, replacing the punctuation
heuristic (which needs a transcript and knows nothing about prosody).

Trains on CPU in minutes (~100k params). Artifacts ->
ml/checkpoints/scheduler/{accent_model.pt, meta.json}.
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data", "BEAT2", "beat_english_v2.0.0")
OUT = os.path.join(ROOT, "checkpoints", "scheduler")

FPS = 30
CTX = 8  # feature context: +-8 frames (~0.53 s)
NOD_AMP = 0.10  # rad, min peak-to-peak pitch swing
SHAKE_AMP = 0.08


def head_signals(poses):
    """(t, 165) axis-angle -> (pitch, yaw) of neck+head, radians."""
    aa = poses.reshape(len(poses), -1, 3)
    ang = aa[:, 12] + aa[:, 15]  # neck + head, world-aligned local frames
    return ang[:, 0], ang[:, 1]


def bandpass(x, lo=1.0, hi=4.5):
    """Cheap FFT bandpass for event detection."""
    n = len(x)
    f = np.fft.rfftfreq(n, 1 / FPS)
    X = np.fft.rfft(x - x.mean())
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, n)


def detect_events(sig, amp):
    """Oscillation bursts: windows where the bandpassed signal swings more
    than `amp` peak-to-peak with at least one direction change. Returns
    event start frames (deduped, min 0.8 s apart)."""
    b = bandpass(sig)
    events = []
    w = int(0.6 * FPS)
    i = 0
    while i < len(b) - w:
        seg = b[i : i + w]
        if seg.max() - seg.min() > amp:
            dv = np.diff(np.sign(np.diff(seg)))
            if np.abs(dv).sum() >= 2:
                events.append(i)
                i += int(0.8 * FPS)
                continue
        i += 3
    return events


def audio_features(audio, sr, n_frames):
    """Per-motion-frame prosody features (n_frames, 7): energy, onset,
    brightness, voicing, dEnergy, F0 (log, voiced-only), dF0. Pitch contour
    is the strongest nod cue (phrase-final falls)."""
    import librosa

    hop = sr // FPS
    rms = librosa.feature.rms(y=audio, frame_length=hop * 2, hop_length=hop)[0]
    onset = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop)
    cent = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop)[0]
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=hop * 2, hop_length=hop)[0]
    f0 = librosa.yin(audio, fmin=60, fmax=400, sr=sr, hop_length=hop)
    logf0 = np.log2(np.clip(f0, 60, 400) / 60.0)
    # Unvoiced yin output pins to fmax; zero it out there.
    logf0 = np.where(f0 > 390, 0.0, logf0)
    feats = []
    for arr in (rms, onset, cent / 4000.0, zcr, None, logf0, None):
        if arr is None:
            continue
        feats.append(np.resize(arr, n_frames).astype(np.float32))
    drms = np.gradient(feats[0]).astype(np.float32)
    df0 = np.gradient(feats[4]).astype(np.float32)
    return np.stack(feats[:4] + [drms, feats[4], df0], axis=1)  # (n, 7)


def stack_context(feats):
    """(n, f) -> (n, f*(2*CTX+1)) by stacking neighboring frames."""
    n, f = feats.shape
    out = np.zeros((n, f * (2 * CTX + 1)), dtype=np.float32)
    for k, off in enumerate(range(-CTX, CTX + 1)):
        idx = np.clip(np.arange(n) + off, 0, n - 1)
        out[:, k * f : (k + 1) * f] = feats[idx]
    return out


def build_dataset():
    import librosa

    from train_emotion import build_windows, load_recording  # split logic reuse

    # Recording-level split identical to train_emotion's (val stems held out).
    stems = sorted(
        f[:-4] for f in os.listdir(os.path.join(DATA, "smplxflame_30")) if f.endswith(".npz")
    )
    X_tr, y_tr, X_va, y_va = [], [], [], []
    from train_emotion import emotion_of_stem

    by_emo = {}
    for s in stems:
        by_emo.setdefault((s.split("_")[1], emotion_of_stem(s)), []).append(s)
    val_stems = {v[-1] for v in by_emo.values() if len(v) > 1}

    for stem in stems:
        poses, trans, expr, contact, audio = load_recording(stem)
        n = len(poses)
        pitch, yaw = head_signals(poses)
        labels = np.zeros((n, 2), dtype=np.float32)
        for f0 in detect_events(pitch, NOD_AMP):
            labels[max(0, f0 - 3) : f0 + 4, 0] = 1
        for f0 in detect_events(yaw, SHAKE_AMP):
            labels[max(0, f0 - 3) : f0 + 4, 1] = 1
        feats = stack_context(audio_features(audio, 16000, n))
        if stem in val_stems:
            X_va.append(feats)
            y_va.append(labels)
        else:
            X_tr.append(feats)
            y_tr.append(labels)
    return (
        np.concatenate(X_tr), np.concatenate(y_tr),
        np.concatenate(X_va), np.concatenate(y_va),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=4096)
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    print("building dataset (event detection + prosody features)...", flush=True)
    X_tr, y_tr, X_va, y_va = build_dataset()
    mean, std = X_tr.mean(0), X_tr.std(0) + 1e-6
    X_tr = (X_tr - mean) / std
    X_va = (X_va - mean) / std
    pos_rate = y_tr.mean(0)
    print(f"train {len(X_tr)} frames, val {len(X_va)}; positive rate nod {pos_rate[0]:.4f} shake {pos_rate[1]:.4f}", flush=True)

    model = nn.Sequential(
        nn.Linear(X_tr.shape[1], 96), nn.ReLU(),
        nn.Linear(96, 48), nn.ReLU(),
        nn.Linear(48, 2),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pos_weight = torch.tensor([1.0 / max(1e-4, pos_rate[0]), 1.0 / max(1e-4, pos_rate[1])]) * 0.5
    lossfn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Xt = torch.from_numpy(X_tr)
    yt = torch.from_numpy(y_tr)
    Xv = torch.from_numpy(X_va)
    yv = torch.from_numpy(y_va)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(Xt), args.bs):
            idx = perm[i : i + args.bs]
            opt.zero_grad()
            loss = lossfn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xv))
            # F1 at 0.5 per head
            stats = []
            for h in range(2):
                pred = (pv[:, h] > 0.5).float()
                tp = (pred * yv[:, h]).sum().item()
                prec = tp / max(1, pred.sum().item())
                rec = tp / max(1, yv[:, h].sum().item())
                f1 = 2 * prec * rec / max(1e-6, prec + rec)
                stats.append((prec, rec, f1))
        thresholds = [0.5, 0.5]
        print(
            f"epoch {ep + 1}/{args.epochs} loss {tot / len(Xt):.4f} "
            f"nod P/R/F1 {stats[0][0]:.2f}/{stats[0][1]:.2f}/{stats[0][2]:.2f} "
            f"shake {stats[1][0]:.2f}/{stats[1][1]:.2f}/{stats[1][2]:.2f}",
            flush=True,
        )

    # Operating thresholds: highest-recall point with precision >= 0.3
    # (precision-first — a missed nod is invisible, a wrong one is not).
    with torch.no_grad():
        pv = torch.sigmoid(model(Xv))
    for h in range(2):
        best = None
        for th in np.arange(0.3, 0.96, 0.05):
            pred = (pv[:, h] > th).float()
            tp = (pred * yv[:, h]).sum().item()
            prec = tp / max(1, pred.sum().item())
            rec = tp / max(1, yv[:, h].sum().item())
            if prec >= 0.3 and (best is None or rec > best[1]):
                best = (th, rec, prec)
        thresholds[h] = round(float(best[0]), 2) if best else 0.85
        print(f"head {h}: threshold {thresholds[h]}" + (f" (P={best[2]:.2f} R={best[1]:.2f})" if best else " (precision 0.3 never reached)"), flush=True)

    os.makedirs(OUT, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUT, "accent_model.pt"))
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(
            {
                "mean": mean.tolist(), "std": std.tolist(),
                "ctx": CTX, "fps": FPS, "features": 7,
                "hidden": [96, 48],
                "thresholds": thresholds,
                "val_f1": {"nod": stats[0][2], "shake": stats[1][2]},
            },
            f,
        )
    print(f"saved -> {OUT}", flush=True)
    print("SCHED TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()
