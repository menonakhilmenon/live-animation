import { AudioFeatures, emptyFeatures, ema } from './features';
import { TempoTracker } from './tempo';

const BASS_RANGE: [number, number] = [20, 250];
const MID_RANGE: [number, number] = [250, 2000];
const TREBLE_RANGE: [number, number] = [2000, 8000];

/** Minimum gap between detected beats (seconds) — caps detection at 240 BPM. */
const MIN_BEAT_INTERVAL = 0.25;
/** Bass energy must exceed its recent average by this factor to count as a beat. */
const BEAT_THRESHOLD = 1.35;
/** Ignore beats when overall level is near silence. */
const BEAT_MIN_LEVEL = 0.015;

/**
 * Wraps the Web Audio API: routes an audio file or the microphone through an
 * AnalyserNode and extracts per-frame features for the animation system.
 */
export class AudioEngine {
  readonly features: AudioFeatures = emptyFeatures();

  private ctx: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private sourceNode: AudioNode | null = null;
  private micStream: MediaStream | null = null;

  private freqData: Uint8Array<ArrayBuffer> = new Uint8Array(0);
  private timeData: Uint8Array<ArrayBuffer> = new Uint8Array(0);

  private tempo = new TempoTracker();

  /** Rolling history of instantaneous bass energy, for the beat threshold. */
  private bassHistory: number[] = [];
  private lastBeatTime = -1e9;
  private timeSec = 0;

  /** Attach an <audio> element (file playback). Replaces any current source. */
  async useMediaElement(el: HTMLMediaElement): Promise<void> {
    const ctx = this.ensureContext();
    this.disconnectSource();
    const src = ctx.createMediaElementSource(el);
    src.connect(this.analyser!);
    this.analyser!.connect(ctx.destination);
    this.sourceNode = src;
    await ctx.resume();
  }

  /** Use the microphone. Analysis only — mic audio is not played back. */
  async useMicrophone(): Promise<void> {
    const ctx = this.ensureContext();
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false },
    });
    this.disconnectSource();
    const src = ctx.createMediaStreamSource(stream);
    src.connect(this.analyser!);
    this.sourceNode = src;
    this.micStream = stream;
    await ctx.resume();
  }

  /** Call once per render frame with the frame delta time in seconds. */
  update(dt: number): AudioFeatures {
    this.timeSec += dt;
    const f = this.features;
    if (!this.analyser || !this.ctx) return f;

    this.analyser.getByteFrequencyData(this.freqData);
    this.analyser.getByteTimeDomainData(this.timeData);

    // RMS from the time-domain waveform.
    let sumSq = 0;
    for (let i = 0; i < this.timeData.length; i++) {
      const v = (this.timeData[i] - 128) / 128;
      sumSq += v * v;
    }
    const rmsNow = Math.sqrt(sumSq / this.timeData.length);

    const bassNow = this.bandEnergy(BASS_RANGE);
    const midNow = this.bandEnergy(MID_RANGE);
    const trebleNow = this.bandEnergy(TREBLE_RANGE);

    // Smooth features. Fast attack, slower release keeps motion punchy but
    // stops it from flickering.
    f.rms = ema(f.rms, rmsNow, rmsNow > f.rms ? 0.03 : 0.15, dt);
    f.bass = ema(f.bass, bassNow, bassNow > f.bass ? 0.03 : 0.12, dt);
    f.mid = ema(f.mid, midNow, 0.1, dt);
    f.treble = ema(f.treble, trebleNow, 0.1, dt);
    f.brightness = ema(f.brightness, this.spectralCentroid(), 0.25, dt);

    this.detectBeat(bassNow, rmsNow, dt);
    f.timeSinceBeat = this.timeSec - this.lastBeatTime;

    this.tempo.update(dt, f.beat);
    f.bpm = this.tempo.bpm;
    f.tempoConfidence = this.tempo.confidence;
    f.beatPhase = this.tempo.phase;
    return f;
  }

  private detectBeat(bassNow: number, rmsNow: number, dt: number): void {
    const f = this.features;
    f.beat = false;
    // Decay the pulse envelope (~300 ms fall).
    f.beatPulse = Math.max(0, f.beatPulse - dt / 0.3);

    this.bassHistory.push(bassNow);
    if (this.bassHistory.length > 43) this.bassHistory.shift(); // ≈0.7 s at 60 fps

    if (this.bassHistory.length < 10) return;
    const avg = this.bassHistory.reduce((a, b) => a + b, 0) / this.bassHistory.length;
    const sinceLast = this.timeSec - this.lastBeatTime;

    if (
      rmsNow > BEAT_MIN_LEVEL &&
      sinceLast > MIN_BEAT_INTERVAL &&
      bassNow > avg * BEAT_THRESHOLD &&
      bassNow > 0.05
    ) {
      f.beat = true;
      f.beatStrength = Math.min(1, (bassNow / Math.max(avg, 1e-4) - 1) / 1.5);
      f.beatPulse = Math.max(f.beatPulse, 0.6 + 0.4 * f.beatStrength);
      this.lastBeatTime = this.timeSec;
    }
  }

  /** Mean normalized magnitude over a frequency range in Hz. */
  private bandEnergy([lo, hi]: [number, number]): number {
    const { freqData } = this;
    const nyquist = this.ctx!.sampleRate / 2;
    const loBin = Math.max(0, Math.floor((lo / nyquist) * freqData.length));
    const hiBin = Math.min(freqData.length - 1, Math.ceil((hi / nyquist) * freqData.length));
    if (hiBin <= loBin) return 0;
    let sum = 0;
    for (let i = loBin; i <= hiBin; i++) sum += freqData[i];
    return sum / ((hiBin - loBin + 1) * 255);
  }

  private spectralCentroid(): number {
    const { freqData } = this;
    let weighted = 0;
    let total = 0;
    for (let i = 0; i < freqData.length; i++) {
      weighted += i * freqData[i];
      total += freqData[i];
    }
    if (total === 0) return 0;
    // Normalize: centroid bin / bin count, then stretch the useful low range.
    return Math.min(1, (weighted / total / freqData.length) * 4);
  }

  private ensureContext(): AudioContext {
    if (!this.ctx) {
      this.ctx = new AudioContext();
      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = 2048;
      this.analyser.smoothingTimeConstant = 0.5;
      this.freqData = new Uint8Array(this.analyser.frequencyBinCount);
      this.timeData = new Uint8Array(this.analyser.fftSize);
    }
    return this.ctx;
  }

  private disconnectSource(): void {
    this.sourceNode?.disconnect();
    this.sourceNode = null;
    this.micStream?.getTracks().forEach((t) => t.stop());
    this.micStream = null;
  }
}
