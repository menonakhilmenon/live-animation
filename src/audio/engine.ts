import { Timeline, analyzeBuffer } from './analysis';
import { AudioFeatures, emptyFeatures, ema } from './features';

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

  /** Offline whole-file analysis (null for mic / before analysis finishes). */
  timeline: Timeline | null = null;
  private clockEl: HTMLMediaElement | null = null;
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

    // Smooth features. Fast attack, slower release keeps motion punchy but
    // stops it from flickering.
    f.rms = ema(f.rms, rmsNow, rmsNow > f.rms ? 0.03 : 0.15, dt);
    f.brightness = ema(f.brightness, this.spectralCentroid(), 0.25, dt);

    f.mode = 'live';
    f.onset = 0; // live has no timeline; speech onsets come from the timeline

    // Offline timeline overrides the causal estimates with exact values.
    if (this.timeline && this.clockEl && !this.clockEl.paused) {
      this.applyTimeline(this.timeline, this.clockEl.currentTime);
    }
    return f;
  }

  private applyTimeline(tl: Timeline, t: number): void {
    const f = this.features;
    f.mode = tl.mode;

    // Frame-sampled features (deterministic, from the offline STFT).
    const idx = Math.max(0, Math.min(tl.frames.rms.length - 1, Math.round(t / tl.hopTime)));
    f.brightness = tl.frames.centroid[idx];
    f.onset = tl.frames.onset[idx];
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
