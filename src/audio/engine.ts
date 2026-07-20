import { Timeline, analyzeBuffer } from './analysis';
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

/** Index of the largest element <= t in an ascending array, or -1. */
function binarySearchLE(arr: number[], t: number): number {
  let lo = 0;
  let hi = arr.length - 1;
  let ans = -1;
  while (lo <= hi) {
    const m = (lo + hi) >> 1;
    if (arr[m] <= t) {
      ans = m;
      lo = m + 1;
    } else hi = m - 1;
  }
  return ans;
}

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

  /** Offline whole-file analysis (null for mic / before analysis finishes). */
  timeline: Timeline | null = null;
  private clockEl: HTMLMediaElement | null = null;
  private lastTimelineBeat = -1;

  /** Rolling history of instantaneous bass energy, for the beat threshold. */
  private bassHistory: number[] = [];
  private lastBeatTime = -1e9;
  private timeSec = 0;

  private mediaSrc: MediaElementAudioSourceNode | null = null;
  private mediaEl: HTMLMediaElement | null = null;

  /**
   * Attach an <audio> element (file playback). Replaces any current source.
   * The MediaElementSource is created once per element and reconnected on
   * later calls — createMediaElementSource throws if called twice for the
   * same element (e.g. file → mic → file again).
   */
  async useMediaElement(el: HTMLMediaElement): Promise<void> {
    const ctx = this.ensureContext();
    this.disconnectSource();
    if (!this.mediaSrc || this.mediaEl !== el) {
      this.mediaSrc = ctx.createMediaElementSource(el);
      this.mediaEl = el;
    }
    this.mediaSrc.connect(this.analyser!);
    this.analyser!.connect(ctx.destination);
    this.sourceNode = this.mediaSrc;
    await ctx.resume();
  }

  /**
   * Decode and analyze a full audio file offline, then drive features from
   * the resulting timeline synced to the media element's playback clock.
   */
  async analyzeFile(file: File, el: HTMLMediaElement): Promise<Timeline> {
    const ctx = this.ensureContext();
    const buf = await ctx.decodeAudioData(await file.arrayBuffer());
    // Mix to mono for analysis.
    const mono = new Float32Array(buf.length);
    for (let c = 0; c < buf.numberOfChannels; c++) {
      const ch = buf.getChannelData(c);
      for (let i = 0; i < buf.length; i++) mono[i] += ch[i] / buf.numberOfChannels;
    }
    const tl = analyzeBuffer(mono, buf.sampleRate);
    this.timeline = tl;
    this.clockEl = el;
    this.lastTimelineBeat = -1;
    return tl;
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
    this.timeline = null; // mic is live-only; fall back to causal tracking
    this.clockEl = null;
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
    f.barPhase = this.tempo.phase / 4;
    f.section = ema(f.section, Math.min(1, f.rms * 3), 2.0, dt);
    f.nextDropIn = Infinity;
    f.mode = 'live';

    // Offline timeline overrides the causal estimates with exact values.
    if (this.timeline && this.clockEl && !this.clockEl.paused) {
      this.applyTimeline(this.timeline, this.clockEl.currentTime, dt);
    }
    return f;
  }

  private applyTimeline(tl: Timeline, t: number, dt: number): void {
    const f = this.features;
    f.mode = tl.mode;

    // Frame-sampled features (deterministic, from the offline STFT).
    const idx = Math.max(0, Math.min(tl.frames.rms.length - 1, Math.round(t / tl.hopTime)));
    f.brightness = tl.frames.centroid[idx];
    f.bass = ema(f.bass, tl.frames.bass[idx], 0.05, dt);
    f.mid = ema(f.mid, tl.frames.mid[idx], 0.08, dt);
    f.treble = ema(f.treble, tl.frames.treble[idx], 0.08, dt);
    f.section = tl.frames.loudness[idx];

    // Exact beat grid → beat events, continuous beat/bar phase.
    if (tl.beats.length >= 2) {
      let i = binarySearchLE(tl.beats, t);
      const period = 60 / tl.bpm;
      let frac: number;
      if (i < 0) {
        frac = Math.max(0, 1 - (tl.beats[0] - t) / period);
        i = -1;
      } else {
        const next = i + 1 < tl.beats.length ? tl.beats[i + 1] : tl.beats[i] + period;
        frac = Math.min(1, (t - tl.beats[i]) / Math.max(1e-3, next - tl.beats[i]));
      }
      f.beatPhase = i + frac;
      f.barPhase = (i - tl.downbeatOffset + frac) / 4;
      f.bpm = tl.bpm;
      f.tempoConfidence = Math.min(1, tl.tempoStrength * 4);
      if (i !== this.lastTimelineBeat && i >= 0) {
        this.lastTimelineBeat = i;
        f.beat = true;
        f.beatStrength = Math.max(f.beatStrength, 0.7);
        f.beatPulse = Math.max(f.beatPulse, 0.9);
      }
    }

    // Future awareness: time until the next drop section begins.
    f.nextDropIn = Infinity;
    for (const s of tl.sections) {
      if (s.drop && s.start > t - 0.25) {
        f.nextDropIn = Math.max(0, s.start - t);
        break;
      }
    }
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
