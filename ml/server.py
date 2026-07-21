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
        _models["emage"] = EmageAudioModel.from_pretrained(EMOTION_CKPT).to(device).eval()
        _models["emotion_conditioned"] = True
    else:
        _models["emage"] = EmageAudioModel.from_pretrained(hub).to(device).eval()
        _models["emotion_conditioned"] = False
    _models["tts"] = KPipeline(lang_code="a", device="cpu", repo_id="hexgrad/Kokoro-82M")
    _models["device"] = device
    print("models ready", flush=True)
    return _models


def tts(text: str, voice: str, speed: float):
    """text -> (mono float32 24 kHz, [[word, t0, t1], ...])"""
    m = get_models()
    chunks, words, offset = [], [], 0.0
    for res in m["tts"](text, voice=voice, speed=speed):
        if res.tokens:
            for t in res.tokens:
                if t.start_ts is not None:
                    words.append([t.text, round(offset + t.start_ts, 3), round(offset + t.end_ts, 3)])
        audio = res.audio.numpy()
        chunks.append(audio)
        offset += len(audio) / 24000
    return np.concatenate(chunks), words


def gestures(audio: np.ndarray, sr: int, amplitude: float, emotion_id: int = 0) -> dict:
    import librosa
    import torch
    import torch.nn.functional as F

    m = get_models()
    model, motion_vq, device = m["emage"], m["motion_vq"], m["device"]
    if sr != model.cfg.audio_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=model.cfg.audio_sr)
    audio_t = torch.from_numpy(audio.astype(np.float32)).to(device).unsqueeze(0)
    # With the fine-tuned model the "speaker" slot IS the emotion id
    # (see ml/train_emotion.py); the base model only has row 0.
    sid = emotion_id if m["emotion_conditioned"] else 0
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

    @app.get("/health")
    def health():
        return {"ok": True, "emotions": list(EMOTIONS)}

    @app.post("/animate")
    def animate(req: AnimateRequest):
        if req.emotion not in EMOTIONS:
            raise HTTPException(400, f"unknown emotion {req.emotion!r}; one of {list(EMOTIONS)}")
        amplitude, speed, mood, emotion_id = EMOTIONS[req.emotion]
        if req.text:
            audio, words = tts(req.text, req.voice, speed)
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

        clip = gestures(audio, sr, amplitude, emotion_id)
        clip["mood"] = mood
        return {
            "clip": clip,
            "audioB64": wav_b64(audio, sr),
            "words": words,
            "emotion": req.emotion,
            "mood": mood,
        }

    return app


if __name__ == "__main__":
    import uvicorn

    get_models()  # fail fast + warm start
    uvicorn.run(create_app(), host="127.0.0.1", port=8600, log_level="warning")
