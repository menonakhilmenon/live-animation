// Generates test.wav: 8 s, 44.1 kHz mono — 120 BPM kick pattern (60 Hz
// decaying sine bursts) plus hi-hat noise ticks on the off-beats.
const fs = require('fs');
const path = require('path');
const outDir = path.join(__dirname, '.artifacts');
fs.mkdirSync(outDir, { recursive: true });
const sr = 44100;
const dur = 8;
const n = sr * dur;
const samples = new Float32Array(n);
const bps = 2; // 120 BPM
for (let i = 0; i < n; i++) {
  const t = i / sr;
  const beatT = t % (1 / bps);
  // Kick: 60 Hz sine, sharp exponential decay
  samples[i] += Math.sin(2 * Math.PI * 60 * beatT) * Math.exp(-beatT * 18) * 0.9;
  // Hi-hat on off-beat: white noise burst
  const offT = (t + 0.25) % (1 / bps);
  samples[i] += (Math.random() * 2 - 1) * Math.exp(-offT * 60) * 0.15;
  // Quiet mid-range pad so mid band isn't empty
  samples[i] += Math.sin(2 * Math.PI * 440 * t) * 0.05;
}
const pcm = Buffer.alloc(n * 2);
for (let i = 0; i < n; i++) {
  pcm.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(samples[i] * 32767 * 0.8))), i * 2);
}
const header = Buffer.alloc(44);
header.write('RIFF', 0);
header.writeUInt32LE(36 + pcm.length, 4);
header.write('WAVE', 8);
header.write('fmt ', 12);
header.writeUInt32LE(16, 16);
header.writeUInt16LE(1, 20); // PCM
header.writeUInt16LE(1, 22); // mono
header.writeUInt32LE(sr, 24);
header.writeUInt32LE(sr * 2, 28);
header.writeUInt16LE(2, 32);
header.writeUInt16LE(16, 34);
header.write('data', 36);
header.writeUInt32LE(pcm.length, 40);
fs.writeFileSync(path.join(outDir, 'test.wav'), Buffer.concat([header, pcm]));
console.log('WROTE', path.join(outDir, 'test.wav'), 44 + pcm.length, 'bytes');
