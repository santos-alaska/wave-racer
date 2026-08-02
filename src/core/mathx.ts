/** Small maths helpers used across subsystems. Frame-rate independent by design. */

export const clamp = (v: number, a: number, b: number) => (v < a ? a : v > b ? b : v);
export const clamp01 = (v: number) => clamp(v, 0, 1);
export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
export const invLerp = (a: number, b: number, v: number) => (b === a ? 0 : (v - a) / (b - a));
export const smoothstep = (e0: number, e1: number, x: number) => {
  const t = clamp01((x - e0) / (e1 - e0 || 1e-6));
  return t * t * (3 - 2 * t);
};
export const TAU = Math.PI * 2;

/** Shortest signed angular difference, result in (-π, π]. */
export function angleDelta(from: number, to: number) {
  let d = (to - from) % TAU;
  if (d > Math.PI) d -= TAU;
  if (d < -Math.PI) d += TAU;
  return d;
}

/**
 * Frame-rate independent exponential smoothing.
 * `rate` is roughly "how many e-foldings per second" — higher snaps faster.
 * Use this instead of a raw `lerp(a, b, 0.1)`, which changes behaviour with dt.
 */
export function damp(current: number, target: number, rate: number, dt: number) {
  return lerp(target, current, Math.exp(-rate * dt));
}

/** Same, for angles — takes the short way round. */
export function dampAngle(current: number, target: number, rate: number, dt: number) {
  return current + angleDelta(current, target) * (1 - Math.exp(-rate * dt));
}

/**
 * Critically damped spring. Returns the new value and writes the new velocity
 * back into `vel[0]`. Stable at large dt, unlike a naive spring integrator.
 */
export function springDamp(
  current: number,
  target: number,
  vel: { v: number },
  stiffness: number,
  dt: number,
) {
  const omega = stiffness;
  const x = current - target;
  const exp = Math.exp(-omega * dt);
  const newVel = (vel.v - omega * x) * exp;
  const newPos = target + (x + (vel.v + omega * x) * dt) * exp;
  vel.v = newVel + omega * (newPos - target) * 0; // keep API simple; vel is already correct
  vel.v = newVel;
  return newPos;
}

/** Map a value through a set of hard bands — the numeric twin of cel shading. */
export function quantize(v: number, bands: number) {
  return Math.floor(clamp01(v) * bands) / (bands - 1 || 1);
}

/** Seconds → "1:23.456" for lap times. */
export function formatTime(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) return "--'--.---";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 1000);
  return `${m}:${s.toString().padStart(2, '0')}.${ms.toString().padStart(3, '0')}`;
}

/** Ordinal suffix for finishing positions. */
export function ordinal(n: number): string {
  const s = ['th', 'st', 'nd', 'rd'];
  const v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
}
