/**
 * Minimal in-place radix-2 FFT — enough for offline STFT analysis without
 * pulling in a dependency. Input length must be a power of two.
 */
export class FFT {
  private cosTable: Float32Array;
  private sinTable: Float32Array;

  constructor(readonly size: number) {
    if ((size & (size - 1)) !== 0) throw new Error('FFT size must be a power of two');
    this.cosTable = new Float32Array(size / 2);
    this.sinTable = new Float32Array(size / 2);
    for (let i = 0; i < size / 2; i++) {
      this.cosTable[i] = Math.cos((2 * Math.PI * i) / size);
      this.sinTable[i] = Math.sin((2 * Math.PI * i) / size);
    }
  }

  /** In-place transform of interleaved re/im arrays (length = size). */
  transform(re: Float32Array, im: Float32Array): void {
    const n = this.size;
    // Bit-reversal permutation.
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        [re[i], re[j]] = [re[j], re[i]];
        [im[i], im[j]] = [im[j], im[i]];
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const half = len >> 1;
      const step = n / len;
      for (let i = 0; i < n; i += len) {
        for (let k = 0; k < half; k++) {
          const idx = k * step;
          const c = this.cosTable[idx];
          const s = this.sinTable[idx];
          const reB = re[i + k + half] * c + im[i + k + half] * s;
          const imB = im[i + k + half] * c - re[i + k + half] * s;
          re[i + k + half] = re[i + k] - reB;
          im[i + k + half] = im[i + k] - imB;
          re[i + k] += reB;
          im[i + k] += imB;
        }
      }
    }
  }
}
