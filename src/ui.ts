import { MotionClip } from './animation/clip';
import { VisemeEvent } from './animation/face';
import { GestureSchedule } from './animation/schedule';
import { AudioEngine } from './audio/engine';

/** Where the Python animation sidecar listens (ml/server.py). */
const SIDECAR = 'http://127.0.0.1:8600';

export interface UIHooks {
  playClip: (clip: MotionClip) => void;
  /** Preferred playback: prebaked library clips per the sidecar's schedule.
   * Returns false when the library can't serve it (caller falls back). */
  playSchedule: (schedule: GestureSchedule, clock: () => number) => boolean;
  /** End scheduled playback (procedural behavior resumes). */
  stopSchedule: () => void;
  /** Install (or clear) a phoneme-timed lip-sync track. */
  setVisemes: (events: VisemeEvent[] | null, clock?: () => number) => void;
}

/** Wires the control panel: audio file loading, mic toggle, feature meters. */
export function setupUI(engine: AudioEngine, hooks: UIHooks): void {
  const fileBtn = document.getElementById('btn-file') as HTMLButtonElement;
  const micBtn = document.getElementById('btn-mic') as HTMLButtonElement;
  const fileInput = document.getElementById('file-input') as HTMLInputElement;
  const audioEl = document.getElementById('audio-el') as HTMLAudioElement;

  fileBtn.addEventListener('click', () => fileInput.click());

  const analysisLabel = document.getElementById('analysis-label')!;

  fileInput.addEventListener('change', async () => {
    const file = fileInput.files?.[0];
    if (!file) return;
    hooks.setVisemes(null);
    audioEl.src = URL.createObjectURL(file);
    audioEl.hidden = false;
    await engine.useMediaElement(audioEl);
    // Offline whole-file analysis runs alongside playback start; the
    // animator upgrades from causal tracking to the exact timeline the
    // moment it lands (typically well under a second).
    analysisLabel.textContent = 'analyzing…';
    engine
      .analyzeFile(file, audioEl)
      .then((tl) => {
        analysisLabel.textContent = tl.mode;
      })
      .catch((err) => {
        console.warn('offline analysis failed:', err);
        analysisLabel.textContent = 'analysis failed (live tracking)';
      });
    await audioEl.play();
    fileBtn.classList.add('active');
    micBtn.classList.remove('active');
  });

  // --- Speak: text + emotion → sidecar-generated speech and gestures ---
  const speakBtn = document.getElementById('btn-speak') as HTMLButtonElement;
  const speakText = document.getElementById('speak-text') as HTMLTextAreaElement;
  const speakEmotion = document.getElementById('speak-emotion') as HTMLSelectElement;
  const speakIntensity = document.getElementById('speak-intensity') as HTMLInputElement;
  const speakStyle = document.getElementById('speak-style') as HTMLInputElement;
  const speakStyleLabel = document.getElementById('speak-style-label');
  const speakStatus = document.getElementById('speak-status')!;

  // Live readout for the continuous style slider (calm ↔ game-feel).
  const describeStyle = (v: number): string => {
    if (v <= 0.1) return 'game-faithful (calm)';
    if (v < 0.3) return 'toward faithful';
    if (v <= 0.4) return 'expressive';
    const factor = (1 + ((v - 0.35) / 0.65)).toFixed(2);
    return `game-feel ×${factor}`;
  };
  const syncStyleLabel = () => {
    if (speakStyleLabel) speakStyleLabel.textContent = describeStyle(Number(speakStyle.value));
  };
  speakStyle.addEventListener('input', syncStyleLabel);
  syncStyleLabel();

  // Per-emotion default style position: the slider snaps here when the
  // emotion changes (the user then trims). Sourced from the sidecar, with a
  // local fallback so it works before /health resolves.
  const FALLBACK_EMOTION_STYLE: Record<string, number> = {
    neutral: 0.35, calm: 0.2, sad: 0.12, happy: 0.45, excited: 0.62, angry: 0.55,
  };
  let emotionStyle: Record<string, number> = { ...FALLBACK_EMOTION_STYLE };
  const applyEmotionStyle = () => {
    const d = emotionStyle[speakEmotion.value];
    if (d !== undefined) {
      speakStyle.value = String(d);
      syncStyleLabel();
    }
  };
  speakEmotion.addEventListener('change', applyEmotionStyle);
  applyEmotionStyle();
  fetch(`${SIDECAR}/health`)
    .then((r) => r.json())
    .then((h) => {
      if (h?.emotion_styles) {
        emotionStyle = { ...FALLBACK_EMOTION_STYLE, ...h.emotion_styles };
        applyEmotionStyle();
      }
    })
    .catch(() => {});

  speakBtn.addEventListener('click', async () => {
    const text = speakText.value.trim();
    if (!text) return;
    speakBtn.disabled = true;
    speakStatus.textContent = 'generating…';
    try {
      const res = await fetch(`${SIDECAR}/animate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          emotion: speakEmotion.value,
          intensity: Number(speakIntensity.value),
          base_style: (document.getElementById('speak-base') as HTMLSelectElement).value,
          style: Number(speakStyle.value),
        }),
      });
      if (!res.ok) throw new Error(`sidecar ${res.status}: ${await res.text()}`);
      const { clip, schedule, audioB64, visemes } = (await res.json()) as {
        clip?: MotionClip;
        schedule?: GestureSchedule;
        audioB64: string;
        visemes?: VisemeEvent[];
      };

      const bytes = Uint8Array.from(atob(audioB64), (c) => c.charCodeAt(0));
      const file = new File([bytes], 'speech.wav', { type: 'audio/wav' });
      audioEl.src = URL.createObjectURL(file);
      audioEl.hidden = false;
      await engine.useMediaElement(audioEl);
      engine.analyzeFile(file, audioEl).catch(() => {});
      // Start the gesture clip the moment audio actually starts so lip
      // sync (audio-driven) and body motion (clip-driven) stay aligned.
      hooks.setVisemes(visemes ?? null, () => audioEl.currentTime);
      audioEl.addEventListener(
        'playing',
        () => {
          // Prebaked schedule is the artifact-free path; raw model clip is
          // the fallback for older servers / missing library entries.
          const scheduled =
            !!schedule?.base?.length && hooks.playSchedule(schedule, () => audioEl.currentTime);
          if (!scheduled && clip) hooks.playClip(clip);
          // The audio clock freezes at the end, which would hold the last
          // scheduled pose forever — release back to procedural behavior.
          if (scheduled) {
            audioEl.addEventListener('ended', () => hooks.stopSchedule(), { once: true });
          }
        },
        { once: true },
      );
      await audioEl.play();
      speakStatus.textContent = `speaking (${speakEmotion.value})`;
    } catch (err) {
      console.warn('speak failed:', err);
      speakStatus.textContent = 'sidecar offline? run: ml/.venv/bin/python ml/server.py';
    } finally {
      speakBtn.disabled = false;
    }
  });

  micBtn.addEventListener('click', async () => {
    try {
      hooks.setVisemes(null);
      audioEl.pause();
      await engine.useMicrophone();
      micBtn.classList.add('active');
      fileBtn.classList.remove('active');
    } catch (err) {
      console.error('Microphone access failed:', err);
      micBtn.textContent = 'Mic unavailable';
    }
  });
}

/** Update the small level meters in the panel. Call once per frame. */
export function updateMeters(engine: AudioEngine): void {
  const f = engine.features;
  setMeter('m-rms', Math.min(1, f.rms * 2.5));
}

function setMeter(id: string, value: number): void {
  const bar = document.querySelector<HTMLDivElement>(`#${id} > div`);
  if (bar) bar.style.width = `${(value * 100).toFixed(1)}%`;
}
