/**
 * Critically-damped spring for smooth target-following. Turns discontinuous
 * audio features (beats, energy jumps) into natural-looking motion.
 */
export class Spring {
  value: number;
  velocity = 0;

  constructor(
    initial = 0,
    /** Higher = snappier. ~10 is loose, ~40 is tight. */
    public stiffness = 20,
    /** Damping ratio: 1 = critically damped, <1 overshoots (follow-through). */
    public zeta = 1,
  ) {
    this.value = initial;
  }

  update(target: number, dt: number): number {
    const d = 2 * this.zeta * Math.sqrt(this.stiffness);
    const accel = (target - this.value) * this.stiffness - this.velocity * d;
    this.velocity += accel * dt;
    this.value += this.velocity * dt;
    return this.value;
  }
}
