"""Animation sidecar: text+emotion (or raw audio) -> speech WAV + MotionClip.

    ml/.venv/bin/python ml/server.py          # http://localhost:8600

POST /animate
    {"text": "...", "emotion": "happy", "voice": "af_heart"}
  or
    {"audioB64": "<base64 wav/mp3/...>", "emotion": "neutral"}
returns
    {"clip": MotionClip, "audioB64": "<base64 wav>", "words": [[w, t0, t1]...],
     "emotion": "...", "mood": -1..1}

Models: Kokoro-82M TTS on CPU (ROCm MIOpen breaks its LSTM; CPU is
real-time anyway) with per-word timestamps; EMAGE co-speech gesture
generation on the GPU. Emotion is applied as a post-process today
(gesture amplitude/tempo styling + face mood) — the trained
emotion-conditioned model replaces this hook later.
"""

import base64
import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vendor", "PantoMatrix"))

from generate import SMPLX_PARENTS, SMPLX_TO_JOINT, axis_angle_to_quat, poses_to_clip  # noqa: E402

# emotion -> (gesture amplitude, tts speed, face mood -1..1, BEAT emotion id)
# The BEAT id feeds the fine-tuned emotion embedding when a trained
# checkpoint exists (ml/train_emotion.py); BEAT order: 0 neutral,
# 1 happiness, 2 anger, 3 sadness, 4 contempt, 5 surprise, 6 fear, 7 disgust.
EMOTIONS = {
    "neutral": (1.0, 1.0, 0.0, 0),
    "happy": (1.15, 1.05, 0.7, 1),
    "excited": (1.3, 1.12, 1.0, 5),
    "calm": (0.8, 0.92, 0.15, 0),
    "sad": (0.6, 0.85, -0.6, 3),
    "angry": (1.2, 1.08, -0.9, 2),
}

EMOTION_CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "emage_emotion")

_models = {}


def get_models():
    if _models:
        return _models
    import torch
    from kokoro import KPipeline
    from models.emage_audio import EmageAudioModel, EmageVAEConv, EmageVQModel, EmageVQVAEConv

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # Other services (ollama, user's TTS) may own nearly all VRAM;
        # EMAGE needs ~1 GiB. Fall back to CPU (~30 s per generation).
        try:
            free, _ = torch.cuda.mem_get_info()
            if free < 1.2 * 2**30:
                print(f"only {free / 2**30:.1f} GiB VRAM free — using CPU", flush=True)
                device = "cpu"
        except Exception:
            device = "cpu"
    hub = "H-Liu1997/emage_audio"
    print(f"loading EMAGE ({device}) + Kokoro (cpu) ...", flush=True)
    _models["motion_vq"] = (
        EmageVQModel(
            face_model=EmageVQVAEConv.from_pretrained(hub, subfolder="emage_vq/face").to(device),
            upper_model=EmageVQVAEConv.from_pretrained(hub, subfolder="emage_vq/upper").to(device),
            lower_model=EmageVQVAEConv.from_pretrained(hub, subfolder="emage_vq/lower").to(device),
            hands_model=EmageVQVAEConv.from_pretrained(hub, subfolder="emage_vq/hands").to(device),
            global_model=EmageVAEConv.from_pretrained(hub, subfolder="emage_vq/global").to(device),
        )
        .to(device)
        .eval()
    )
    if os.path.isdir(EMOTION_CKPT):
        print(f"using emotion-conditioned fine-tune: {EMOTION_CKPT}", flush=True)
        model = EmageAudioModel.from_pretrained(EMOTION_CKPT).to(device).eval()
        # Widen each emotion embedding by one SCRATCH row used to hold a
        # per-request neutral↔emotion interpolation (intensity control).
        import torch.nn as nn

        for name in ("speaker_embedding_body", "speaker_embedding_face"):
            old = getattr(model, name)
            new = nn.Embedding(old.num_embeddings + 1, old.embedding_dim).to(device)
            new.weight.data[: old.num_embeddings] = old.weight.data
            new.weight.data[-1] = old.weight.data[0]
            setattr(model, name, new)
        _models["scratch_row"] = model.speaker_embedding_body.num_embeddings - 1
        _models["emage"] = model
        _models["emotion_conditioned"] = True
    else:
        _models["emage"] = EmageAudioModel.from_pretrained(hub).to(device).eval()
        _models["emotion_conditioned"] = False
    _models["tts"] = KPipeline(lang_code="a", device="cpu", repo_id="hexgrad/Kokoro-82M")
    _models["device"] = device
    print("models ready", flush=True)
    return _models


# Misaki/IPA phoneme characters -> viseme class. 'aa' wide-open, 'ih'
# spread, 'ou' rounded; bilabials close the mouth; other consonants get a
# small neutral opening (weight handles that).
VISEME_OF = {}
for ch in "aæɑʌAIW":
    VISEME_OF[ch] = "aa"
for ch in "iɪeɛjY":
    VISEME_OF[ch] = "ih"
for ch in "oɔuʊOQw":
    VISEME_OF[ch] = "ou"
for ch in "pbm":
    VISEME_OF[ch] = "sil"


def viseme_track(tokens):
    """[(text, phonemes, t0, t1)] -> [[t0, t1, viseme, weight], ...]"""
    events = []
    for text, phonemes, t0, t1 in tokens:
        ph = [c for c in (phonemes or "") if c.isalpha() or c in VISEME_OF]
        if not ph or t1 <= t0:
            continue
        step = (t1 - t0) / len(ph)
        for i, c in enumerate(ph):
            v = VISEME_OF.get(c)
            w = 1.0
            if v is None:
                v, w = "ih", 0.3  # generic consonant: slightly open
            elif v == "sil":
                w = 0.0
            events.append([round(t0 + i * step, 3), round(t0 + (i + 1) * step, 3), v, w])
    # Merge consecutive identical visemes to keep the payload small.
    merged = []
    for e in events:
        if merged and merged[-1][2] == e[2] and merged[-1][3] == e[3] and abs(merged[-1][1] - e[0]) < 0.02:
            merged[-1][1] = e[1]
        else:
            merged.append(e)
    return merged


def tts(text: str, voice: str, speed: float):
    """text -> (mono float32 24 kHz, [[word, t0, t1], ...], viseme track)"""
    m = get_models()
    chunks, words, tokens, offset = [], [], [], 0.0
    for res in m["tts"](text, voice=voice, speed=speed):
        if res.tokens:
            for t in res.tokens:
                if t.start_ts is not None:
                    words.append([t.text, round(offset + t.start_ts, 3), round(offset + t.end_ts, 3)])
                    tokens.append(
                        (t.text, getattr(t, "phonemes", ""), offset + t.start_ts, offset + t.end_ts)
                    )
        audio = res.audio.numpy()
        chunks.append(audio)
        offset += len(audio) / 24000
    return np.concatenate(chunks), words, viseme_track(tokens)


def gestures(
    audio: np.ndarray, sr: int, amplitude: float, emotion_id: int = 0, intensity: float = 1.0
) -> dict:
    import librosa
    import torch
    import torch.nn.functional as F

    m = get_models()
    model, motion_vq, device = m["emage"], m["motion_vq"], m["device"]
    if sr != model.cfg.audio_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=model.cfg.audio_sr)
    audio_t = torch.from_numpy(audio.astype(np.float32)).to(device).unsqueeze(0)
    # With the fine-tuned model the "speaker" slot IS the emotion id
    # (see ml/train_emotion.py); the base model only has row 0. Fractional
    # intensity interpolates neutral→emotion into the scratch row.
    sid = emotion_id if m["emotion_conditioned"] else 0
    if m["emotion_conditioned"] and 0.0 <= intensity < 1.0 and emotion_id != 0:
        scratch = m["scratch_row"]
        with torch.no_grad():
            for name in ("speaker_embedding_body", "speaker_embedding_face"):
                w = getattr(model, name).weight
                w.data[scratch] = (1 - intensity) * w.data[0] + intensity * w.data[emotion_id]
        sid = scratch
    speaker = torch.full((1, 1), sid).long().to(device)
    from generate import build_seed, sample_tokens

    # Deterministic per input; T=0.9 on upper/hands restores real-motion
    # dynamics (argmax measures 5x slower hands than mocap); legs and face
    # keep argmax for stability.
    gen = torch.Generator(device="cpu").manual_seed(int(np.abs(audio[:800]).sum() * 1e6) % 2**31)
    seed_motion, seed_mask = build_seed(device)
    with torch.no_grad():
        lat = model.inference(audio_t, speaker, motion_vq, masked_motion=seed_motion, mask=seed_mask)
        cfg = model.cfg
        pick = lambda ck, rk, c, l, temp=0.0: (  # noqa: E731
            sample_tokens(lat[ck].cpu(), temp, generator=gen).to(device) if c > 0 else None,
            lat[rk] if l > 0 and c == 0 else None,
        )
        fi, fl = pick("cls_face", "rec_face", cfg.cf, cfg.lf)
        # Measured against held-out mocap: upper 0.8 puts wrist velocity at
        # 39.8 deg/s (GT 35.3; 0.9 ran 47.2) with posture unchanged; hands
        # (fingers) 0.7 calms digit motion.
        ui, ul = pick("cls_upper", "rec_upper", cfg.cu, cfg.lu, 0.8)
        hi, hl = pick("cls_hands", "rec_hands", cfg.ch, cfg.lh, 0.7)
        li, ll = pick("cls_lower", "rec_lower", cfg.cl, cfg.ll)
        pred = motion_vq.decode(
            face_latent=fl, upper_latent=ul, lower_latent=ll, hands_latent=hl,
            face_index=fi, upper_index=ui, lower_index=li, hands_index=hi,
            get_global_motion=True, ref_trans=torch.zeros(1, 3).to(device),
        )
    t = pred["motion_axis_angle"].shape[1]
    poses = pred["motion_axis_angle"].cpu().numpy().reshape(t, -1)
    trans = pred["trans"].cpu().numpy().reshape(t, -1)

    if amplitude != 1.0:
        # Emotion styling: scale UPPER-body joint angles (axis-angle norm is
        # the angle, so plain multiplication scales rotation magnitude);
        # legs/hips stay untouched to preserve the stance.
        upper = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        aa = poses.reshape(t, -1, 3)
        aa[:, upper] *= amplitude
        poses = aa.reshape(t, -1)
    return poses_to_clip(poses, trans, cfg.pose_fps, pin_feet=True)


# UI emotion -> baked base-loop name (ml/bake_library.py).
BASE_LOOP = {
    "neutral": "talk_neutral", "calm": "talk_neutral",
    "happy": "talk_happiness", "excited": "talk_happiness",
    "angry": "talk_anger", "sad": "talk_sadness",
}
NEGATIVE_WORDS = {"no", "not", "never", "nothing", "wrong", "unacceptable", "worst", "bad"}

# Base-layer style: which clip provides the standing stance the gestures ride
# on. "mocap" uses our BEAT2-baked emotion loops; "bg3" uses a game-authored
# standing idle (Larian, extracted) with the emotion gestures as an additive
# overlay — game reference as the base layer.
GAME_BASES = {"bg3": "base_bg3", "ff16": "base_ff16", "ffxv": "base_ffxv"}
# Continuous motion-style axis, 0..1: 0 = game-faithful (calm, handedness
# matched), STYLE_EXPRESSIVE = our natural co-speech level, 1 = full
# cinematic game-feel. MAX_EXAGG is the gesture-angle multiplier at style=1
# (applied in the player to additive + accent deltas; posture/base untouched).
STYLE_EXPRESSIVE = 0.35
MAX_EXAGG = 2.0
# Named presets → a style value, for the legacy motion_style string.
STYLE_PRESETS = {"faithful": 0.0, "expressive": STYLE_EXPRESSIVE, "exaggerated": 1.0}

# Per-emotion default position on the style axis: where the slider starts for
# each emotion (the user then trims). Subdued emotions sit calmer, high-arousal
# ones livelier — matches the measured emotion signatures (sadness least
# mobile, anger/excitement most animated). Also the server-side fallback when
# a client sends no explicit style.
EMOTION_STYLE = {
    "neutral": STYLE_EXPRESSIVE,  # 0.35
    "calm": 0.20,
    "sad": 0.12,
    "happy": 0.45,
    "excited": 0.62,
    "angry": 0.55,
}


def style_blend(style):
    """Split a style value into (faithful, exaggerated) blend factors in
    0..1. Only one is non-zero: below STYLE_EXPRESSIVE we blend toward
    faithful, above it toward game-feel."""
    e0 = STYLE_EXPRESSIVE
    fa = max(0.0, min(1.0, (e0 - style) / e0)) if e0 > 0 else 0.0
    ex = max(0.0, min(1.0, (style - e0) / (1 - e0))) if style > e0 else 0.0
    return fa, ex


def _lerp(a, b, t):
    return a + (b - a) * t
ACCENT_CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "scheduler")


def get_accent_model():
    """Learned accent scheduler (ml/train_scheduler.py), lazily (re)loaded
    so a server started before training finishes picks the weights up."""
    if "accent" in _models:
        return _models["accent"]
    path = os.path.join(ACCENT_CKPT, "accent_model.pt")
    meta_path = os.path.join(ACCENT_CKPT, "meta.json")
    if not (os.path.exists(path) and os.path.exists(meta_path)):
        return None
    import json as _json

    import torch
    import torch.nn as nn

    meta = _json.load(open(meta_path))
    dims = meta["features"] * (2 * meta["ctx"] + 1)
    model = nn.Sequential(
        nn.Linear(dims, meta["hidden"][0]), nn.ReLU(),
        nn.Linear(meta["hidden"][0], meta["hidden"][1]), nn.ReLU(),
        nn.Linear(meta["hidden"][1], 2),
    )
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    _models["accent"] = (model, np.array(meta["mean"], dtype=np.float32),
                         np.array(meta["std"], dtype=np.float32), meta)
    print(f"learned accent scheduler loaded (val F1 {meta.get('val_f1')})", flush=True)
    return _models["accent"]


def learned_accents(audio: np.ndarray, sr: int, intensity: float):
    """Run the accent model over audio -> [{name, t, scale}] or None."""
    loaded = get_accent_model()
    if loaded is None:
        return None
    import librosa
    import torch

    from train_scheduler import audio_features, stack_context

    model, mean, std, meta = loaded
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    n = int(len(audio) / 16000 * meta["fps"])
    if n < meta["fps"]:
        return []
    X = (stack_context(audio_features(audio, 16000, n)) - mean) / std
    with torch.no_grad():
        p = torch.sigmoid(model(torch.from_numpy(X))).numpy()  # (n, 2)
    # The model's calibration is conservative (precision-first thresholds
    # rarely fire) but its RANKING carries signal — select the top peaks at
    # a natural accent rate (~1 per 4 s) with NMS, probability floor 0.25.
    fps = meta["fps"]
    want = max(1, int(n / fps / 4.0))
    peak = p.max(axis=1)
    order = np.argsort(-peak)
    chosen = []
    for f in order:
        if peak[f] < 0.25 or len(chosen) >= want:
            break
        t = f / fps
        if any(abs(t - c) < 1.2 for c in chosen):
            continue
        chosen.append(t)
    accents = []
    for t in sorted(chosen):
        f = int(t * fps)
        name = "agree" if p[f, 0] >= p[f, 1] else "headShake"
        accents.append({"name": name, "t": round(t, 2),
                        "scale": round((0.35 + 0.65 * float(peak[f])) * (0.5 + 0.5 * intensity), 2)})
    return accents


def build_schedule(words, duration, emotion, intensity, audio=None, sr=16000,
                   base_style="mocap", style=STYLE_EXPRESSIVE):
    """Decide which prebaked clip plays when: talk loops over speech spans,
    idle over long gaps, additive accents. Accents come from the LEARNED
    scheduler (audio prosody -> nod/shake, trained on BEAT2 head motion)
    when its checkpoint exists, else from punctuation/negation rules.

    base_style selects the base layer: 'mocap' = our emotion loops; a game
    key ('bg3') = a game-authored standing idle as the base, with the
    emotion gesture layered as a stronger additive overlay on top.

    style (0..1, continuous) scales the expressive layer between two extremes:
    - 0 = game-faithful (only meaningful with a game base): mute the BEAT2
      additive to a whisper so the game's OWN baked motion carries the
      performance. Our co-speech mocap moves the hands ~4-6x more than real
      game dialogue (BEAT2 talk ~100-146 deg/s vs FFXVI talk_relax ~6-25);
      at full weight it overrides the game's calm style. Faithful keeps a
      hint of additive and softens nods, so it reads as the source game.
    - STYLE_EXPRESSIVE = our natural co-speech gesture level.
    - 1 = cinematic 'game-feel': heavier additive weights plus a
      schedule-level angle-scale (in the player) for bigger sweeps.
    The blend is continuous; weights and accent softness interpolate."""
    game_base = GAME_BASES.get(base_style)
    base_name = game_base or BASE_LOOP.get(emotion, "talk_neutral")
    fa, ex = style_blend(style)
    e0 = STYLE_EXPRESSIVE
    # Nod/shake accents soften toward the faithful end (games gesture
    # sparingly); the player's angle-scale grows nods toward game-feel.
    accent_scale = _lerp(1.0, 0.5, fa)
    emo_loop = BASE_LOOP.get(emotion, "talk_neutral")

    # Additive layers riding the base. On a game base, the emotion's own talk
    # loop rides additively (delta-from-its-neutral = talking arms) so the
    # game idle carries stance while our motion carries the gesture; plus a
    # lighter FFXV flourish. On a mocap base, just the FFXV flourish.
    if game_base:
        # faithful 0.12/0.06 → expressive 0.7/0.25 → game-feel 0.9/0.4.
        if style <= e0:
            k = style / e0 if e0 > 0 else 1.0
            w, fl = _lerp(0.12, 0.7, k), _lerp(0.06, 0.25, k)
        else:
            w, fl = _lerp(0.7, 0.9, ex), _lerp(0.25, 0.4, ex)
        additive = [
            {"name": emo_loop, "weight": round(w * intensity, 2), "loop": True},
            {"name": "talk_overlay_ffxv", "weight": round(fl * intensity, 2), "loop": True},
        ]
    else:
        # Mocap base has no game to be faithful to; below expressive it just
        # calms a little, above it grows toward game-feel.
        fl = _lerp(0.2, 0.35, style / e0) if (e0 > 0 and style <= e0) else _lerp(0.35, 0.5, ex)
        additive = [{"name": "talk_overlay_ffxv", "weight": round(fl * intensity, 2), "loop": True}]

    segs, accents = [], []
    if not words:
        # Raw audio, no transcript: the learned prosody model is the only
        # accent signal available.
        segs.append({"name": base_name, "t0": 0.0, "t1": round(duration, 2)})
        if audio is not None:
            accents = learned_accents(audio, sr, intensity) or []
            if accent_scale != 1.0:
                accents = [{**a, "scale": round(a["scale"] * accent_scale, 2)} for a in accents]
        return {"base": segs, "accents": accents, "additive": additive}

    # Speech spans: words separated by <0.9 s belong to one span.
    spans = []
    cur = [words[0][1], words[0][2]]
    for _, t0, t1 in words[1:]:
        if t0 - cur[1] > 0.9:
            spans.append(cur)
            cur = [t0, t1]
        else:
            cur[1] = t1
    spans.append(cur)

    cursor = 0.0
    for s0, s1 in spans:
        if s0 - cursor > 1.2:
            # Gap filler: the game base IS an idle, so reuse it; otherwise
            # Xbot's authored idle (EMAGE-on-silence baked garbage).
            segs.append({"name": game_base or "idle", "t0": round(cursor, 2), "t1": round(s0 - 0.15, 2)})
            cursor = s0 - 0.15
        segs.append({"name": base_name, "t0": round(cursor, 2), "t1": round(s1 + 0.25, 2)})
        cursor = s1 + 0.25
    segs[-1]["t1"] = round(max(segs[-1]["t1"], duration), 2)

    # Text inputs know their sentence structure — punctuation places nods
    # more reliably than the prosody model (val precision ~0.2, an honest
    # limit of audio->gesture prediction). The learned model serves the
    # transcript-less path below.
    flip = True
    for w, t0, t1 in words:
        token = w.strip().lower()
        if w.strip() in {".", "!", "?"} or token.endswith((".", "!", "?")):
            if flip:
                accents.append({"name": "agree", "t": round(max(0.0, t0 - 0.15), 2),
                                "scale": round((0.5 + 0.5 * intensity) * accent_scale, 2)})
            flip = not flip
        elif token.strip('.,!?') in NEGATIVE_WORDS and emotion in ("angry", "sad"):
            accents.append({"name": "headShake", "t": round(max(0.0, t0 - 0.1), 2),
                            "scale": round((0.4 + 0.6 * intensity) * accent_scale, 2)})
    # Continuous additive layer: a game-derived expressive talk overlay
    # (arms/torso, delta-from-base) rides the base loop, energy-gated. This
    # is the base+additive composition — the base carries emotion-specific
    # stance, the overlay adds hand-talk life on top. Weight scales with
    # emotion intensity.
    return {"base": segs, "accents": accents, "additive": additive}


def wav_b64(audio: np.ndarray, sr: int) -> str:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode()


def create_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    app = FastAPI(title="live-animation sidecar")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    class AnimateRequest(BaseModel):
        text: str | None = None
        audioB64: str | None = None
        emotion: str = "neutral"
        voice: str = "af_heart"
        intensity: float = 1.0  # 0 = neutral gestures, 1 = full emotion
        raw: bool = False  # also return the raw EMAGE clip (debug/fallback)
        base_style: str = "mocap"  # 'mocap' or a game key ('bg3')
        # Continuous style axis 0..1: 0 faithful, ~0.35 expressive, 1 game-feel.
        # motion_style (named preset) and game_faithful kept for back-compat.
        style: float | None = None
        motion_style: str | None = None
        game_faithful: bool = False

    @app.get("/health")
    def health():
        return {"ok": True, "emotions": list(EMOTIONS), "base_styles": ["mocap", *GAME_BASES],
                "emotion_styles": EMOTION_STYLE}

    @app.post("/animate")
    def animate(req: AnimateRequest):
        if req.emotion not in EMOTIONS:
            raise HTTPException(400, f"unknown emotion {req.emotion!r}; one of {list(EMOTIONS)}")
        amplitude, speed, mood, emotion_id = EMOTIONS[req.emotion]
        visemes = []
        if req.text:
            audio, words, visemes = tts(req.text, req.voice, speed)
            sr = 24000
        elif req.audioB64:
            import librosa

            with tempfile.NamedTemporaryFile(suffix=".audio") as f:
                f.write(base64.b64decode(req.audioB64))
                f.flush()
                audio, sr = librosa.load(f.name, sr=None, mono=True)
            words = []
        else:
            raise HTTPException(400, "need text or audioB64")

        s = max(0.0, min(1.0, req.intensity))
        duration = len(audio) / sr
        # Resolve the continuous style value (0..1). Precedence: explicit
        # `style` float > legacy `motion_style` preset > game_faithful bool.
        if req.style is not None:
            v = req.style
        elif req.motion_style is not None:
            v = STYLE_PRESETS.get(req.motion_style, STYLE_EXPRESSIVE)
        elif req.game_faithful:
            v = 0.0
        else:
            v = EMOTION_STYLE.get(req.emotion, STYLE_EXPRESSIVE)
        v = max(0.0, min(1.0, v))
        fa, ex = style_blend(v)
        schedule = build_schedule(words, duration, req.emotion, s, audio=audio, sr=sr,
                                  base_style=req.base_style, style=v)
        schedule["mood"] = mood * s
        # Toward the faithful end, the additive follows the base clip's own
        # handedness (continuous 0..1) so the resting hand stays resting.
        schedule["handedness"] = round(fa, 3) if req.base_style in GAME_BASES else 0.0
        # Toward game-feel, the player scales gesture angles up (1..MAX_EXAGG).
        schedule["exaggeration"] = round(1.0 + ex * (MAX_EXAGG - 1.0), 3)
        out = {
            "schedule": schedule,
            "audioB64": wav_b64(audio, sr),
            "words": words,
            "visemes": visemes,
            "emotion": req.emotion,
            "mood": mood * s,
            "intensity": s,
        }
        if req.raw:
            clip = gestures(audio, sr, 1.0 + (amplitude - 1.0) * s, emotion_id, s)
            clip["mood"] = mood * s
            out["clip"] = clip
        return out

    return app


if __name__ == "__main__":
    import uvicorn

    get_models()  # fail fast + warm start
    uvicorn.run(create_app(), host="127.0.0.1", port=8600, log_level="warning")
