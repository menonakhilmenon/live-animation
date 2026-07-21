# ml/ — generative animation sidecar and training

Python side of the generative animation suite (see
`docs/generative-animation.md` for the research and architecture).

## Setup

```sh
uv venv --python 3.12 ml/.venv
uv pip install --python ml/.venv/bin/python \
    --index-url https://download.pytorch.org/whl/rocm6.4 torch
uv pip install --python ml/.venv/bin/python -r ml/requirements.txt \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
git clone --depth 1 https://github.com/PantoMatrix/PantoMatrix ml/vendor/PantoMatrix
```

Model weights (EMAGE ~640 MB Apache-2.0, Kokoro ~330 MB Apache-2.0)
download automatically from HuggingFace on first use.

## Pieces

- **`server.py`** — FastAPI sidecar on `:8600` behind the app's Speak
  panel. `POST /animate {text|audioB64, emotion}` → Kokoro TTS (CPU,
  per-word timestamps) + EMAGE co-speech gestures (GPU) → MotionClip JSON
  + WAV. Uses the emotion fine-tune below when present.
- **`generate.py`** — CLI: speech WAV → MotionClip JSON
  (`--audio in.wav --out clip.json`). Houses the SMPL-X → canonical
  world-delta conversion shared by everything.
- **`train_emotion.py`** — fine-tunes EMAGE into an emotion-conditioned
  model on BEAT2 (textual-inversion style: the unused speaker-embedding
  input widened to 8 BEAT emotions, everything else frozen with
  `--embeddings-only`). Data: `ml/data/BEAT2` (ungated HF download of
  speaker 2). Checkpoint → `ml/checkpoints/emage_emotion`.
- **`eval_emotion.py`** — same audio under all 8 emotion ids; reports
  per-emotion motion energy / hand travel / divergence from neutral.

## ROCm notes (RX 9070 XT, gfx1201)

- PyTorch 2.9.1+rocm6.4 wheels work; ~7 TFLOPS fp32 measured.
- Fused SDPA **backward** has no gfx1201 kernel ("no kernel image") —
  training forces math SDPA. Inference is unaffected.
- Kokoro's LSTM hits a MIOpen bug on GPU — it runs on CPU (real-time).
- Other services (ollama, etc.) may hold most VRAM; training is built to
  fit a ~1 GiB budget (embeddings-only + per-pass backward + small batch).
