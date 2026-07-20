// Generates test audio into e2e/.artifacts/:
//
// test.wav — a 16 s structured "song" at exactly 120 BPM:
//   kicks every 0.5 s; accented downbeat (louder kick + crash) every 2 s;
//   quiet first 6 s (verse) then full-level 6–16 s, entered abruptly (drop).
//   Ground truth for beat-grid, downbeat, section, and drop assertions.
//
// speech.wav — 8 s of speech-like audio: harmonic voiced bursts at jittered
//   syllable rate with pauses; no periodic beat. Ground truth for the
//   music/speech classifier.
const fs = require('fs');
const path = require('path');
const outDir = path.join(__dirname, '.artifacts');
fs.mkdirSync(outDir, { recursive: true });
const sr = 44100;

function writeWav(name, samples) {
  const pcm = Buffer.alloc(samples.length * 2);
  for (let i = 0; i < samples.length; i++) {
    pcm.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(samples[i] * 32767 * 0.8))), i * 2);
  }
  const header = Buffer.alloc(44);
  header.write('RIFF', 0);
  header.writeUInt32LE(36 + pcm.length, 4);
  header.write('WAVE', 8);
  header.write('fmt ', 12);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(1, 22);
  header.writeUInt32LE(sr, 24);
  header.writeUInt32LE(sr * 2, 28);
  header.writeUInt16LE(2, 32);
  header.writeUInt16LE(16, 34);
  header.write('data', 36);
  header.writeUInt32LE(pcm.length, 40);
  fs.writeFileSync(path.join(outDir, name), Buffer.concat([header, pcm]));
  console.log('WROTE', path.join(outDir, name));
}

// --- test.wav: structured 120 BPM song ---
{
  const dur = 16;
  const n = sr * dur;
  const samples = new Float32Array(n);
  const beat = 0.5; // 120 BPM
  for (let i = 0; i < n; i++) {
    const t = i / sr;
    const sectionAmp = t < 6 ? 0.32 : 1.0; // verse → drop at exactly 6 s
    const beatT = t % beat;
    const downbeat = Math.floor(t / beat) % 4 === 0;
    // Kick: 60 Hz burst, accented on downbeats.
    let s = Math.sin(2 * Math.PI * 60 * beatT) * Math.exp(-beatT * 18) * (downbeat ? 1.2 : 0.85);
    // Crash noise on downbeats only — a clear bar marker.
    if (downbeat) s += (Math.random() * 2 - 1) * Math.exp(-beatT * 25) * 0.3;
    // Hi-hat on off-beats.
    const offT = (t + beat / 2) % beat;
    s += (Math.random() * 2 - 1) * Math.exp(-offT * 60) * 0.15;
    // Pad.
    s += Math.sin(2 * Math.PI * 220 * t) * 0.05 + Math.sin(2 * Math.PI * 440 * t) * 0.04;
    samples[i] = s * sectionAmp;
  }
  writeWav('test.wav', samples);
}

// --- speech.wav: syllabic voiced bursts, aperiodic ---
{
  const dur = 8;
  const n = sr * dur;
  const samples = new Float32Array(n);
  // Deterministic LCG so the file is reproducible.
  let seed = 42;
  const rand = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  let t = 0.2;
  const syllables = [];
  while (t < dur - 0.3) {
    const len = 0.09 + rand() * 0.18; // syllable length
    syllables.push({ start: t, len, f0: 140 + rand() * 110 });
    t += len + 0.03 + rand() * 0.22; // jittered gap
    if (rand() < 0.18) t += 0.35 + rand() * 0.3; // occasional pause
  }
  for (const syl of syllables) {
    const i0 = Math.floor(syl.start * sr);
    const i1 = Math.min(n, Math.floor((syl.start + syl.len) * sr));
    // Formant-shaped harmonics: real speech energy concentrates around the
    // vowel formants (~600 Hz, ~1800 Hz), not the fundamental.
    const gauss = (f, mu, sig) => Math.exp(-0.5 * ((f - mu) / sig) ** 2);
    const amps = [];
    for (let h = 1; h <= 12; h++) {
      const fh = syl.f0 * h;
      amps.push((1 / h) * (0.15 + 2.2 * gauss(fh, 600, 220) + 1.6 * gauss(fh, 1800, 320)));
    }
    for (let i = i0; i < i1; i++) {
      const tt = (i - i0) / sr;
      const env = Math.sin((Math.PI * (i - i0)) / (i1 - i0)); // smooth burst
      let v = 0;
      for (let h = 1; h <= 12; h++) v += Math.sin(2 * Math.PI * syl.f0 * h * tt) * amps[h - 1];
      // Slight breathy noise.
      v += (Math.random() * 2 - 1) * 0.1;
      samples[i] += v * env * 0.5;
    }
  }
  writeWav('speech.wav', samples);
}
