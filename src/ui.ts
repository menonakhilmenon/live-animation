import { AudioEngine } from './audio/engine';

/** Wires the control panel: audio file loading, mic toggle, feature meters. */
export function setupUI(engine: AudioEngine): void {
  const fileBtn = document.getElementById('btn-file') as HTMLButtonElement;
  const micBtn = document.getElementById('btn-mic') as HTMLButtonElement;
  const fileInput = document.getElementById('file-input') as HTMLInputElement;
  const audioEl = document.getElementById('audio-el') as HTMLAudioElement;

  let mediaElementConnected = false;

  fileBtn.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', async () => {
    const file = fileInput.files?.[0];
    if (!file) return;
    audioEl.src = URL.createObjectURL(file);
    audioEl.hidden = false;
    // createMediaElementSource can only be called once per element.
    if (!mediaElementConnected) {
      await engine.useMediaElement(audioEl);
      mediaElementConnected = true;
    }
    await audioEl.play();
    fileBtn.classList.add('active');
    micBtn.classList.remove('active');
  });

  micBtn.addEventListener('click', async () => {
    try {
      audioEl.pause();
      await engine.useMicrophone();
      mediaElementConnected = false; // source node was replaced
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
}

function setMeter(id: string, value: number): void {
  const bar = document.querySelector<HTMLDivElement>(`#${id} > div`);
  if (bar) bar.style.width = `${(value * 100).toFixed(1)}%`;
}
