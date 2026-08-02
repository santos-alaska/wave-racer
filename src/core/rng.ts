/**
 * Deterministic RNG. Every procedural decision in the game (cloud placement,
 * AI mistake timing, sparkle seeds) draws from a seeded stream so the
 * screenshot harness can reproduce an exact frame.
 *
 * mulberry32 — small, fast, good enough distribution for visuals.
 */
export class Rng {
  private s: number;
  constructor(seed = 0x1a2b3c4d) {
    this.s = seed >>> 0;
  }
  /** [0, 1) */
  next(): number {
    this.s = (this.s + 0x6d2b79f5) >>> 0;
    let t = this.s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  /** [a, b) */
  range(a: number, b: number) {
    return a + this.next() * (b - a);
  }
  /** Integer in [a, b] */
  int(a: number, b: number) {
    return Math.floor(this.range(a, b + 1));
  }
  /** Symmetric, [-m, m] */
  sym(m = 1) {
    return this.range(-m, m);
  }
  pick<T>(arr: readonly T[]): T {
    return arr[Math.floor(this.next() * arr.length) % arr.length];
  }
  bool(p = 0.5) {
    return this.next() < p;
  }
}

/** Shared global stream, seeded from ?seed= so harness runs are reproducible. */
const seedParam = new URLSearchParams(location.search).get('seed');
export const rng = new Rng(seedParam ? Number(seedParam) >>> 0 : 0x51ed7e11);
