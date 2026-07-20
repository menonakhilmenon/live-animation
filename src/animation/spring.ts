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
  ) {
    this.value = initial;
  }

  update(target: number, dt: number): number {
    // Critical damping: damping = 2 * sqrt(stiffness)
    const d = 2 * Math.sqrt(this.stiffness);
    const accel = (target - this.value) * this.stiffness - this.velocity * d;
    this.velocity += accel * dt;
    this.value += this.velocity * dt;
    return this.value;
  }
}
