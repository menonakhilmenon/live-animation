import { MotionClip } from './animation/clip';
import { VisemeEvent } from './animation/face';
import { AudioEngine } from './audio/engine';

/** Where the Python animation sidecar listens (ml/server.py). */
const SIDECAR = 'http://127.0.0.1:8600';

export interface UIHooks {
  playClip: (clip: MotionClip) => void;
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
        analysisLabel.textContent = `${tl.mode} · ${tl.bpm ? tl.bpm.toFixed(0) + ' BPM · ' : ''}${tl.sections.length} section${tl.sections.length === 1 ? '' : 's'}`;
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
  const speakStatus = document.getElementById('speak-status')!;

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
        }),
      });
      if (!res.ok) throw new Error(`sidecar ${res.status}: ${await res.text()}`);
      const { clip, audioB64, visemes } = (await res.json()) as {
        clip: MotionClip;
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
      audioEl.addEventListener('playing', () => hooks.playClip(clip), { once: true });
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
  setMeter('m-bass', f.bass);
  setMeter('m-beat', f.beatPulse);
  const bpmLabel = document.getElementById('bpm-label');
  if (bpmLabel) {
    bpmLabel.textContent =
      f.bpm > 0
        ? `tempo: ${f.bpm.toFixed(0)} BPM${f.tempoConfidence > 0.6 ? ' (locked)' : ''}`
        : 'tempo: —';
  }
}

function setMeter(id: string, value: number): void {
  const bar = document.querySelector<HTMLDivElement>(`#${id} > div`);
  if (bar) bar.style.width = `${(value * 100).toFixed(1)}%`;
}
