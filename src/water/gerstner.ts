/**
 * Gerstner wave field — THE single source of truth for the ocean surface.
 *
 * Both sides of the game read this file:
 *   • the GPU displaces the ocean mesh with GERSTNER_GLSL, fed by uniforms
 *     produced by `buildWaveUniforms()`
 *   • the CPU samples the exact same field via `sampleOcean()` for buoyancy,
 *     the racing-line ribbon, gates, foam decals and camera collision
 *
 * If these two ever diverge the boats visibly float above or sink through the
 * water, so the wave table is defined ONCE here and shipped to the shader as
 * uniforms rather than being duplicated in GLSL source.
 *
 * ── Gerstner formulation ────────────────────────────────────────────────────
 * For a wave with unit direction D, wavenumber k = 2π/λ, amplitude A,
 * steepness Q and angular frequency ω, a point on the flat grid at p = (x, z)
 * is displaced to:
 *
 *   θ = k·dot(D, p) − ω·t + φ
 *   P.x = p.x + Q·A·D.x·cos θ
 *   P.z = p.z + Q·A·D.y·cos θ
 *   P.y =        A·sin θ
 *
 * The horizontal term is what makes crests sharpen and troughs broaden — it is
 * the whole reason we use Gerstner instead of stacked sines. Q is normalised
 * per-wave (see `normaliseSteepness`) so the sum never loops the surface back
 * on itself, which would produce the classic "wave turning inside out" artefact.
 */

import { Vector2, Vector3 } from 'three';

export const GRAVITY = 9.81;

export interface GerstnerWave {
  /** Unit direction in the XZ plane. */
  dir: Vector2;
  /** Crest-to-crest distance, metres. */
  wavelength: number;
  /** Half peak-to-trough height, metres. */
  amplitude: number;
  /** 0 = pure sine, 1 = maximally pinched crest. Normalised before use. */
  steepness: number;
  /** Multiplier on the deep-water dispersion speed. 1 = physical. */
  speed: number;
  /** Static phase offset, keeps the waves from all cresting at the origin. */
  phase: number;
}

const dir = (deg: number) => {
  const r = (deg * Math.PI) / 180;
  return new Vector2(Math.cos(r), Math.sin(r));
};

/**
 * The wave table. Six waves in three tiers:
 *   0–1  long swell   — the big rolling motion the boats ride and launch off
 *   2–3  mid chop     — gives the surface its readable, animated silhouette
 *   4–5  fine detail  — high-frequency ripple, mostly a normal-map contributor
 *
 * ── Decorrelation, and why it is not cosmetic ───────────────────────────────
 * A captured aerial frame proved the first version of this table was legible as
 * a *pattern*: the whole ocean read as parallel dashes at one dominant angle
 * and a fixed pitch, like a fingerprint or camo fabric. Two causes, both here:
 *
 *   1. **Near-harmonic wavelengths.** The old ratios ran 1.51, 2.10, 1.54,
 *      2.02, 1.62. The two ratios sitting on ~2.0 mean those trains share
 *      crest positions every other crest, and a sum of octave-related waves
 *      reinforces into a visibly regular beat rather than a sea. Ratios are now
 *      all irrational-ish and clustered near φ (1.6–1.86), so no two trains
 *      come back into phase over any distance the player can see.
 *   2. **Insufficient direction spread.** The old set spanned 14°…122° with two
 *      pairs inside 30° of each other, which produced a dominant axis. The set
 *      below spans 219° with a minimum separation of 37°, so no single angle
 *      survives at altitude.
 *
 * Total amplitude is deliberately held at ~2.98 m (was 2.987) so the boat
 * handling and buoyancy tuning built against the old table still holds.
 */
export const WAVES: GerstnerWave[] = [
  { dir: dir(8), wavelength: 71.3, amplitude: 1.30, steepness: 0.92, speed: 1.0, phase: 0.0 },
  { dir: dir(63), wavelength: 44.1, amplitude: 0.82, steepness: 0.85, speed: 0.97, phase: 1.7 },
  { dir: dir(-34), wavelength: 23.7, amplitude: 0.44, steepness: 0.78, speed: 1.04, phase: 3.1 },
  { dir: dir(101), wavelength: 13.9, amplitude: 0.255, steepness: 0.7, speed: 1.11, phase: 0.6 },
  { dir: dir(-71), wavelength: 7.63, amplitude: 0.108, steepness: 0.6, speed: 1.19, phase: 4.4 },
  { dir: dir(148), wavelength: 4.31, amplitude: 0.056, steepness: 0.5, speed: 1.27, phase: 2.2 },
];

/** Precomputed per-wave scalars, rebuilt whenever the table or sea state changes. */
interface CompiledWave {
  dx: number;
  dz: number;
  k: number;
  a: number;
  qa: number; // Q·A, the horizontal displacement magnitude
  w: number; // angular frequency
  phase: number;
}

let compiled: CompiledWave[] = [];
/** Global sea-state multiplier — 1 = the tuned default. */
let seaState = 1.0;

/**
 * Normalise steepness so that Σ(Qᵢ·Aᵢ·kᵢ) ≤ 1. Above 1 the parametric surface
 * folds over itself and you get inverted crests. We budget to 0.92 to leave a
 * little headroom for the sea-state multiplier.
 */
function compile() {
  const raw = WAVES.map((w) => {
    const k = (2 * Math.PI) / w.wavelength;
    const amp = w.amplitude * seaState;
    return { w, k, amp, qak: w.steepness * amp * k };
  });
  const total = raw.reduce((s, r) => s + r.qak, 0);
  const scale = total > 0.92 ? 0.92 / total : 1;

  compiled = raw.map(({ w, k, amp }) => ({
    dx: w.dir.x,
    dz: w.dir.y,
    k,
    a: amp,
    qa: w.steepness * scale * amp,
    // Deep-water dispersion: ω = sqrt(g·k). Long waves genuinely travel faster
    // than short ones, which is what makes the surface read as water and not
    // as a scrolling texture.
    w: Math.sqrt(GRAVITY * k) * w.speed,
    phase: w.phase,
  }));
}
compile();

/** Adjust overall wave energy (used by the results/menu camera to calm the sea). */
export function setSeaState(v: number) {
  seaState = v;
  compile();
  buildWaveUniforms(); // keep the GPU-side arrays in sync
}
export function getSeaState() {
  return seaState;
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU side
// ─────────────────────────────────────────────────────────────────────────────

export const WAVE_COUNT = WAVES.length;

/**
 * Uniform payload. Packed as two vec4 arrays so it costs 12 uniform slots
 * total and can be uploaded with a single `needsUpdate`.
 *   A = (dir.x, dir.z, k, amplitude)
 *   B = (omega, Q·A, phase, unused)
 */
export const waveUniformA: Vector3[] = [];
export const waveUniformArrays = {
  uWaveA: new Float32Array(WAVE_COUNT * 4),
  uWaveB: new Float32Array(WAVE_COUNT * 4),
};

export function buildWaveUniforms() {
  const { uWaveA, uWaveB } = waveUniformArrays;
  for (let i = 0; i < WAVE_COUNT; i++) {
    const w = compiled[i];
    uWaveA[i * 4 + 0] = w.dx;
    uWaveA[i * 4 + 1] = w.dz;
    uWaveA[i * 4 + 2] = w.k;
    uWaveA[i * 4 + 3] = w.a;
    uWaveB[i * 4 + 0] = w.w;
    uWaveB[i * 4 + 1] = w.qa;
    uWaveB[i * 4 + 2] = w.phase;
    uWaveB[i * 4 + 3] = 0;
  }
  return waveUniformArrays;
}
buildWaveUniforms();

/**
 * GLSL implementation, injected into the ocean vertex shader and anything else
 * that must sit on the water (racing line, gates, wake ribbons, buoys).
 *
 * Exposes:
 *   vec3  gerstnerDisplace(vec2 p, float t)            → world offset
 *   void  gerstnerSurface(vec2 p, float t,
 *                         out vec3 pos, out vec3 nrm,
 *                         out float jacobian)          → full surface sample
 *
 * `jacobian` is the determinant of the horizontal deformation matrix. Where it
 * drops below ~1 the surface is compressing — that is physically where real
 * water piles up and breaks, so foam keys off it rather than off height alone.
 * Keying foam off height gives you foam on every crest at once, which reads as
 * stripes; keying off the Jacobian gives you foam only where waves collide.
 */
export const GERSTNER_GLSL = /* glsl */ `
#define WAVE_COUNT ${WAVE_COUNT}
uniform vec4 uWaveA[WAVE_COUNT]; // dirX, dirZ, k, amplitude
uniform vec4 uWaveB[WAVE_COUNT]; // omega, Q*A, phase, _
uniform float uTime;

// Horizontal + vertical offset from the flat grid position.
vec3 gerstnerDisplace(vec2 p, float t) {
  vec3 acc = vec3(0.0);
  for (int i = 0; i < WAVE_COUNT; i++) {
    vec4 A = uWaveA[i];
    vec4 B = uWaveB[i];
    vec2 d = A.xy;
    float theta = A.z * dot(d, p) - B.x * t + B.z;
    float s = sin(theta), c = cos(theta);
    acc.xz += B.y * d * c;
    acc.y  += A.w * s;
  }
  return acc;
}

// Full surface sample: displaced position, analytic normal, and the horizontal
// Jacobian used as a foam/whitecap mask.
void gerstnerSurface(vec2 p, float t, out vec3 pos, out vec3 nrm, out float jacobian) {
  vec3 acc = vec3(0.0);
  // Partial derivatives of the displacement w.r.t. the flat grid axes.
  float jxx = 0.0, jxz = 0.0, jzz = 0.0;
  vec3 dPdx = vec3(0.0), dPdz = vec3(0.0);

  for (int i = 0; i < WAVE_COUNT; i++) {
    vec4 A = uWaveA[i];
    vec4 B = uWaveB[i];
    vec2 d = A.xy;
    float k = A.z, amp = A.w, qa = B.y;
    float theta = k * dot(d, p) - B.x * t + B.z;
    float s = sin(theta), c = cos(theta);

    acc.xz += qa * d * c;
    acc.y  += amp * s;

    float qak = qa * k * s;
    jxx -= qak * d.x * d.x;
    jxz -= qak * d.x * d.y;
    jzz -= qak * d.y * d.y;

    float akc = amp * k * c;
    dPdx.y += akc * d.x;
    dPdz.y += akc * d.y;
  }

  pos = vec3(p.x + acc.x, acc.y, p.y + acc.z);

  dPdx.x = 1.0 + jxx; dPdx.z = jxz;
  dPdz.x = jxz;       dPdz.z = 1.0 + jzz;
  nrm = normalize(cross(dPdz, dPdx));

  // det of [[1+jxx, jxz], [jxz, 1+jzz]] — < 1 means the surface is pinching.
  jacobian = (1.0 + jxx) * (1.0 + jzz) - jxz * jxz;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// CPU side
// ─────────────────────────────────────────────────────────────────────────────

export interface OceanSample {
  /** World-space point on the surface. */
  position: Vector3;
  /** Unit surface normal. */
  normal: Vector3;
  /** Surface height at the queried (x, z). */
  height: number;
  /** Horizontal Jacobian; < 1 means compression → whitecap. */
  jacobian: number;
}

const _fwdPos = new Vector3();
const _fwdNrm = new Vector3();
const _dPdx = new Vector3();
const _dPdz = new Vector3();

/**
 * Forward evaluation: where does grid point `p` end up, and what's the normal
 * there? This is the exact CPU mirror of `gerstnerSurface` above.
 */
function surfaceAt(px: number, pz: number, t: number, out: OceanSample): OceanSample {
  let ax = 0,
    ay = 0,
    az = 0;
  let jxx = 0,
    jxz = 0,
    jzz = 0;
  let dxy = 0,
    dzy = 0;

  for (let i = 0; i < compiled.length; i++) {
    const w = compiled[i];
    const theta = w.k * (w.dx * px + w.dz * pz) - w.w * t + w.phase;
    const s = Math.sin(theta),
      c = Math.cos(theta);

    ax += w.qa * w.dx * c;
    az += w.qa * w.dz * c;
    ay += w.a * s;

    const qak = w.qa * w.k * s;
    jxx -= qak * w.dx * w.dx;
    jxz -= qak * w.dx * w.dz;
    jzz -= qak * w.dz * w.dz;

    const akc = w.a * w.k * c;
    dxy += akc * w.dx;
    dzy += akc * w.dz;
  }

  out.position.set(px + ax, ay, pz + az);
  _dPdx.set(1 + jxx, dxy, jxz);
  _dPdz.set(jxz, dzy, 1 + jzz);
  out.normal.copy(_dPdz).cross(_dPdx).normalize();
  out.height = ay;
  out.jacobian = (1 + jxx) * (1 + jzz) - jxz * jxz;
  return out;
}

/**
 * Inverse sample: "what is the water doing at world position (x, z)?"
 *
 * Because Gerstner waves displace horizontally, the grid point that *lands* on
 * (x, z) is not (x, z) itself. We recover it with fixed-point iteration —
 * three passes converges to well under a centimetre at our steepness budget,
 * which is far below what the eye can catch on a bobbing hull.
 *
 * Pass `out` to sample in a hot loop without allocating (buoyancy calls this
 * 6× per boat per frame).
 */
const _scratch: OceanSample = {
  position: new Vector3(),
  normal: new Vector3(0, 1, 0),
  height: 0,
  jacobian: 1,
};

export function sampleOcean(
  x: number,
  z: number,
  t: number,
  out: OceanSample = { position: new Vector3(), normal: new Vector3(0, 1, 0), height: 0, jacobian: 1 },
): OceanSample {
  let gx = x,
    gz = z;
  for (let i = 0; i < 3; i++) {
    surfaceAt(gx, gz, t, _scratch);
    gx += x - _scratch.position.x;
    gz += z - _scratch.position.z;
  }
  surfaceAt(gx, gz, t, out);
  // Snap the horizontal components to the query point: the caller asked about
  // (x, z) and the residual is sub-centimetre, so reporting the exact query
  // position keeps downstream maths clean.
  out.position.x = x;
  out.position.z = z;
  return out;
}

/** Convenience: height only, when the normal isn't needed. */
export function sampleHeight(x: number, z: number, t: number): number {
  return sampleOcean(x, z, t, _heightScratch).height;
}
const _heightScratch: OceanSample = {
  position: new Vector3(),
  normal: new Vector3(0, 1, 0),
  height: 0,
  jacobian: 1,
};

/** Largest possible surface height — used for culling and camera clamping. */
export function maxWaveHeight(): number {
  return compiled.reduce((s, w) => s + w.a, 0);
}
