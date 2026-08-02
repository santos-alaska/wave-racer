/**
 * Wake ribbons — the persistent trail every boat leaves in the water.
 *
 * ── Why a ribbon and not a decal ────────────────────────────────────────────
 * A projected decal has to be re-projected onto a moving displaced surface
 * every frame and always ends up either floating or clipping. A ribbon of
 * geometry whose vertices are *evaluated on the wave field in the vertex
 * shader* cannot float: it rides the exact same `gerstnerSurface` the ocean
 * mesh and the buoyancy solver use, so it stays welded to the water no matter
 * how the swell moves under it.
 *
 * ── Shape of the system ─────────────────────────────────────────────────────
 * One `Mesh`, one draw call, for all four boats. Each boat owns a slice of a
 * shared vertex buffer: a ring buffer of `POINTS` trail samples, two vertices
 * per sample (port and starboard). Every frame:
 *
 *   1. if the boat has moved far enough, push a new sample at the stern
 *   2. age every sample
 *   3. rewrite the boat's vertex slice oldest → newest
 *
 * The lateral offset is computed on the CPU (we know the direction between
 * consecutive samples there) so the shader only has to do the wave lookup.
 *
 * ── Age drives three things ─────────────────────────────────────────────────
 *   • **spread** — the ribbon widens behind the boat, the way a real Kelvin
 *     wake diverges. Width is baked into the CPU-side vertex positions.
 *   • **alpha**, quantised to hard steps. A smooth fade is the single fastest
 *     way to make foam read as a semi-transparent PNG rather than as ink.
 *   • **dissipation** — the alpha step is compared against world-space noise, so
 *     as the ribbon ages holes open in it and it breaks into patches instead of
 *     ghosting out uniformly. This is what makes the trail read as foam
 *     *dispersing* rather than as a fading ribbon.
 *
 * A gap opens automatically when a boat is airborne: those samples are marked
 * invalid and written with zero width, which collapses the quad to a degenerate
 * sliver.
 */

import {
  BufferAttribute,
  BufferGeometry,
  DoubleSide,
  DynamicDrawUsage,
  Group,
  LinearMipmapLinearFilter,
  Mesh,
  NormalBlending,
  Quaternion,
  ShaderMaterial,
  Texture,
  Vector3,
} from 'three';
import { GERSTNER_GLSL, waveUniformArrays } from './gerstner';
import { CONFIG } from '../core/config';
import { PAL } from '../core/palette';
import { SHARED } from '../render/celMaterial';
import { makeNoiseTexture } from '../render/textures';
import { SprayField } from './spray';
import type { GameContext, Racer } from '../core/types';

// ─────────────────────────────────────────────────────────────────────────────
// Hull wetness — how much water FX a hull is entitled to, this frame
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Three points down the keel line, boat-local. Not the buoyancy probes and not
 * the physics contact table — those belong to the boat subsystem. This is the
 * water's own answer to "is there a waterline to draw?", measured against the
 * one shared wave field, which is the only way the collar can be guaranteed to
 * agree with the surface it is drawn on.
 */
const KEEL_LINE: readonly Vector3[] = [
  new Vector3(0, -0.1, 1.9), // forefoot
  new Vector3(0, -0.32, 0.0), // keel amidships — the deepest part of the shell
  new Vector3(0, -0.4, -1.9), // run aft to the transom
];

const _keelQ = new Quaternion();
const _keelP = new Vector3();

/**
 * 1 while any part of the keel is in the water, ramping to 0 a third of a metre
 * clear of it.
 *
 * ── Why this is not `state.airborne` ────────────────────────────────────────
 * It used to be. Every hull water effect — collar, wake, spray — was gated on
 * that one boolean, so any frame in which the flag was set arrived with the boat
 * pasted onto the surface: no collar, no bow wave, no wake head, no spray, a
 * razor-sharp polygon edge at the waterline. The r3 review found exactly that in
 * hero.png and ocean_low.png (both `airborne = true` at 28 m/s with the hull
 * visibly in contact) and reported "no foam ring around any hull, in any of the
 * fifteen shots" as the set's largest single defect.
 *
 * A boolean owned by another subsystem is the wrong input for a *continuous*
 * visual quantity in any case. Wetness is graded, so a hull with five
 * centimetres of keel in the water still gets a collar, a hull leaving a crest
 * has its collar fade over ~0.1 s instead of snapping off, and no amount of
 * retuning the airborne latch can silently delete the water FX again.
 */
function hullWetness(ctx: GameContext, racer: Racer): number {
  _keelQ.setFromEuler(racer.root.rotation);
  let clearance = Infinity;
  for (let i = 0; i < KEEL_LINE.length; i++) {
    _keelP.copy(KEEL_LINE[i]).applyQuaternion(_keelQ).add(racer.root.position);
    const c = _keelP.y - ctx.ocean.height(_keelP.x, _keelP.z, ctx.time);
    if (c < clearance) clearance = c;
  }
  return 1 - Math.min(1, Math.max(0, clearance / 0.34));
}

/** Trail samples per boat. 96 × ~1.15 m ≈ 110 m of visible wake. */
const POINTS = 96;
/**
 * Metres of travel between samples. 1.15 m puts ~4.8 s of trail on screen at
 * race speed, which is what makes the age-driven dissipation legible — at the
 * 0.85 m of the first pass the whole ribbon was under 3.5 s old and the tail
 * never got far enough into its life to visibly break up.
 */
const EMIT_STEP = 1.15;
/**
 * Seconds a sample lives before it is fully dissipated.
 *
 * 4.6 s was too long to *read* as dissipation: with coverage still at a third of
 * its birth value at the end of life the tail simply stopped, and the r13 capture
 * showed the ribbon ending in a straight cut edge mid-water. At 2.8 s, coverage
 * is visibly breaking into patches for the last second of the trail, which is the
 * behaviour the brief names.
 */
const LIFE = 2.8;
/** Metres per second the ribbon half-width grows — the Kelvin divergence. */
const SPREAD = 1.9;
/**
 * Vertices across the ribbon.
 *
 * Two is not enough, and the reason is geometric: with only the two edges as
 * vertices, Y is linearly interpolated across a strip that can be ten metres
 * wide, so the middle of the ribbon cuts a straight chord under a curved wave
 * and sinks below the ocean mesh, where it is depth-rejected. Five vertices
 * across follow the swell closely enough that the whole ribbon stays on top of
 * the water, and give the fragment shader a real cross-section to shade.
 */
const CROSS = 5;

interface Trail {
  /** Chronological ring buffer. `head` is the index of the newest sample. */
  x: Float32Array;
  z: Float32Array;
  /** Unit lateral direction (perpendicular to travel) at this sample. */
  nx: Float32Array;
  nz: Float32Array;
  age: Float32Array;
  /** Half-width at emission, metres. */
  w0: Float32Array;
  /** 0…1 — how much churn this sample was born with (speed, drift, landing). */
  power: Float32Array;
  /** 0 = the boat was airborne or stopped; the segment collapses. */
  valid: Float32Array;
  head: number;
  count: number;
  lastX: number;
  lastZ: number;
  /** Distance travelled since the last emitted sample. */
  carry: number;
}

function makeTrail(): Trail {
  return {
    x: new Float32Array(POINTS),
    z: new Float32Array(POINTS),
    nx: new Float32Array(POINTS),
    nz: new Float32Array(POINTS),
    age: new Float32Array(POINTS),
    w0: new Float32Array(POINTS),
    power: new Float32Array(POINTS),
    valid: new Float32Array(POINTS),
    head: -1,
    count: 0,
    lastX: 0,
    lastZ: 0,
    carry: 0,
  };
}

export class WakeRibbons {
  readonly mesh: Mesh;
  private material: ShaderMaterial;
  private trails: Trail[] = [];

  private aPos: BufferAttribute;
  private aSide: BufferAttribute;
  private aAge: BufferAttribute;
  private aPower: BufferAttribute;
  /** Distance along the ribbon from the stern, metres — drives the churn head. */
  private aRun: BufferAttribute;

  constructor(racerCount: number, noise: Texture) {
    for (let i = 0; i < racerCount; i++) this.trails.push(makeTrail());

    const verts = racerCount * POINTS * CROSS;
    const positions = new Float32Array(verts * 3);
    const sides = new Float32Array(verts);
    const ages = new Float32Array(verts);
    const powers = new Float32Array(verts);
    const runs = new Float32Array(verts);

    // Index buffer is static: two triangles per (segment × cross strip), per boat.
    const quads = racerCount * (POINTS - 1) * (CROSS - 1);
    const indices = new Uint16Array(quads * 6);
    let n = 0;
    for (let r = 0; r < racerCount; r++) {
      const base = r * POINTS * CROSS;
      for (let i = 0; i < POINTS - 1; i++) {
        for (let c = 0; c < CROSS - 1; c++) {
          const a = base + i * CROSS + c;
          const b = a + 1;
          const d = a + CROSS;
          const e = d + 1;
          indices[n++] = a;
          indices[n++] = b;
          indices[n++] = d;
          indices[n++] = b;
          indices[n++] = e;
          indices[n++] = d;
        }
      }
    }

    const geo = new BufferGeometry();
    this.aPos = new BufferAttribute(positions, 3);
    this.aSide = new BufferAttribute(sides, 1);
    this.aAge = new BufferAttribute(ages, 1);
    this.aPower = new BufferAttribute(powers, 1);
    this.aRun = new BufferAttribute(runs, 1);
    this.aPos.setUsage(DynamicDrawUsage);
    this.aAge.setUsage(DynamicDrawUsage);
    this.aPower.setUsage(DynamicDrawUsage);
    this.aRun.setUsage(DynamicDrawUsage);
    geo.setAttribute('position', this.aPos);
    geo.setAttribute('aSide', this.aSide);
    geo.setAttribute('aAge', this.aAge);
    geo.setAttribute('aPower', this.aPower);
    geo.setAttribute('aRun', this.aRun);
    geo.setIndex(new BufferAttribute(indices, 1));
    geo.boundingSphere = null;

    // Static per-vertex side, written once: −1 … +1 across the ribbon.
    for (let i = 0; i < verts; i++) sides[i] = ((i % CROSS) / (CROSS - 1)) * 2 - 1;

    this.material = new ShaderMaterial({
      name: 'wakeRibbon',
      transparent: true,
      depthWrite: false,
      blending: NormalBlending,
      side: DoubleSide,
      uniforms: {
        uWaveA: { value: waveUniformArrays.uWaveA },
        uWaveB: { value: waveUniformArrays.uWaveB },
        uTime: SHARED.uTime,
        uCameraPos: SHARED.uCameraPos,
        uSunDir: SHARED.uSunDir,
        uNoise: { value: noise },
        uFoam: { value: PAL.foam.clone() },
        uFoamShade: { value: PAL.foamShade.clone() },
        uCrest: { value: PAL.waterCrest.clone() },
        /** The ribbon draws its own contour; nothing else can ink it. */
        uInk: { value: PAL.inkSoft.clone() },
        /**
         * Metres above the surface the ribbon sits. The ocean mesh draws the
         * *band-limited* field and the ribbon the raw one, so they differ by a
         * few centimetres near steep crests; the lift has to clear that as well
         * as depth precision.
         */
        uLift: { value: 0.2 },
        uLife: { value: LIFE },
        /** Noise tile in metres — large, so the wake never reads as tiled. */
        uTile: { value: 5.6 },
        /** Cutout, so this is 1. Kept as a uniform for the critic loop only. */
        uOpacity: { value: 1.0 },
        /**
         * Width of the ribbon's own ink contour, in device pixels. The wake never
         * enters the G-buffer (it would scribble Sobel lines across the swell and
         * the ocean reads that buffer for its contact mask), so if the foam is to
         * have a drawn edge it has to draw it itself.
         */
        uInkPx: { value: 2.6 },
      },
      vertexShader: /* glsl */ `
        ${GERSTNER_GLSL}

        attribute float aSide;
        attribute float aAge;
        attribute float aPower;
        attribute float aRun;

        uniform float uLift;

        varying float vSide;
        varying float vAge;
        varying float vPower;
        varying float vRun;
        varying vec3 vWorldPos;

        void main() {
          // The CPU baked the spread into the XZ position; all the shader does is
          // put that point on the shared wave field, which is the only way the
          // ribbon can stay welded to a displaced surface.
          vec3 p; vec3 nrm; float j;
          gerstnerSurface(position.xz, uTime, p, nrm, j);
          p.y += uLift;

          vSide = aSide;
          vAge = aAge;
          vPower = aPower;
          vRun = aRun;
          vWorldPos = p;

          gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0);
        }
      `,
      fragmentShader: /* glsl */ `
        precision highp float;

        uniform sampler2D uNoise;
        uniform vec3 uFoam, uFoamShade, uCrest, uInk;
        uniform float uTime, uLife, uTile, uOpacity, uInkPx;

        varying float vSide;
        varying float vAge;
        varying float vPower;
        varying float vRun;
        varying vec3 vWorldPos;

        void main() {
          float life = clamp(1.0 - vAge / uLife, 0.0, 1.0);
          float v = abs(vSide);

          // ── Dissipation noise, with an 8 px floor on feature size ─────────
          // The r3 review measured the wake's boundary as "a salt-and-pepper
          // stipple of isolated 2–4 px squares". A hard threshold against a noise
          // field stops being an edge and becomes a dither the moment the field's
          // features approach a pixel, so every octave has to be gated on whether
          // it is drawable — and the gate has to use the octave's REAL feature
          // size, which is where the previous version went wrong.
          //
          // The noise map packs r = fbm (feature ≈ tile/7), g = one octave
          // (tile/14) and b = one octave (tile/112). The old code gated the b
          // channel of a fetch taken at 3.7× the base rate against an estimate of
          // tile/26 — a feature size twelve times too generous. Its true size is
          // uTile/(3.7·112) = 1.4 cm, which is sub-pixel at *any* camera distance
          // in this game, so a sub-pixel field was reaching a step() every frame.
          // That is the stipple, and once the ribbon gained a signed-distance
          // contour it became salt-and-pepper *ink* as well, because the gradient
          // of a sub-pixel field is random per pixel.
          //
          // The b channel is therefore gone, not gated: it is never drawable. The
          // two survivors are gated on 8–16 px using their actual scales.
          float fp = max(max(fwidth(vWorldPos.x), fwidth(vWorldPos.z)), 1e-5);
          float wMid = smoothstep(fp * 8.0, fp * 16.0, uTile / 14.0);
          float wFine = smoothstep(fp * 8.0, fp * 16.0, uTile / 51.8);
          vec2 uv = vWorldPos.xz / uTile;
          vec4 nA = texture2D(uNoise, uv + vec2(uTime * 0.01, uTime * -0.006));
          vec4 nB = texture2D(uNoise, uv * 3.7 - vec2(uTime * 0.03, 0.0));
          float wSum = 0.60 + 0.26 * wMid + 0.14 * wFine;
          float grain = (nA.r * 0.60 + nA.g * 0.26 * wMid + nB.g * 0.14 * wFine) / wSum;
          grain = clamp(0.5 + (grain - 0.5) * 1.35, 0.0, 1.0);

          // ── Across-ribbon profile ─────────────────────────────────────────
          //   • two divergent foam rails — a *ridge*, not a ramp from the middle
          //   • solid churn immediately behind the transom
          //   • a sparse turbulent field between the rails
          //
          // The rails sit at |v| ≈ 0.72 rather than 0.86, and coverage is taken to
          // zero before |v| = 1. That is the fix for the single worst thing in the
          // r3 wake: at 0.86 with a ±0.2 width the rail ran *past* the strip's own
          // edge, so its outer boundary was not the foam's silhouette at all — it
          // was the geometry's, and the geometry is a straight strip. Hence "the
          // right band's left edge is a perfectly straight line for 400 px". A
          // foam shape may never be clipped by its own carrier quad.
          float railPos = 0.72 + (grain - 0.5) * 0.15;
          float railW = 0.11 + grain * 0.06;
          float rail = 1.0 - smoothstep(0.0, railW, abs(v - railPos));
          float centre = 1.0 - smoothstep(0.0, 0.38, v);
          // 9 m of solid churn behind the transom was a white slab from a low
          // camera. 5 m is a churn *head*, which is what it is for.
          float head = 1.0 - smoothstep(1.2, 5.0, vRun);
          float fill = (1.0 - smoothstep(0.42, 0.95, v)) * 0.24;
          float shape = max(max(rail * 0.95, centre * head), fill * (0.3 + 0.7 * life));
          // Hard taper before the carrier's edge. Nothing reaches |v| = 1.
          shape *= 1.0 - smoothstep(0.86, 0.99, v);

          // ── Dissipation ──────────────────────────────────────────────────
          // Coverage must reach *zero* before the sample expires, and it has to
          // get there faster than linearly or the tail reads as a painted band
          // that simply stops. life² spends most of the trail's length visibly
          // breaking up, which is what "spreads and dissipates" means.
          float fadeOut = smoothstep(0.0, 0.30, life);
          float aged = life * life;
          float coverage = clamp(
            shape * fadeOut * (0.16 + 0.84 * aged) * (0.42 + 0.72 * vPower), 0.0, 0.78);

          // ── Hard cutout, no soft alpha ───────────────────────────────────
          // The whole foam layer is an alpha *cutout*: the mask is 0 or 1 and the
          // fragment is opaque. Alpha-blending an already-quantised foam tone over
          // already-quantised water is what generated the intermediate "mush"
          // tones and made the foam layer look like a different render from the
          // water. Fading is expressed as *less coverage* and as a *lower tone*,
          // never as transparency.
          //
          // sd is the signed distance to that cutout in coverage units, and it
          // is what turns a threshold into a drawn shape: dividing by its own
          // screen gradient converts it to PIXELS, so the ribbon can put a
          // constant-width ink contour on its outer silhouette and quantise a
          // shadow tone just inside it. That is the difference between a shape
          // with an edge and a noise-thresholded quad.
          float sd = coverage - (1.0 - grain);
          if (sd < 0.0) discard;

          // The contour is measured against the COARSE silhouette, not the full
          // grain. Inking the full grain looked right in principle and was wrong in
          // practice: a foam island the size of the finest live octave is narrower
          // than the ink width, so it comes out entirely ink — a scatter of dark
          // dots through the white, which is the r3 stipple in a new colour. The
          // coarse channel's features are ~0.8 m (30–60 px at chase range), so its
          // boundary is a line long enough to draw, and the finer octaves still
          // chew holes in the fill inside it.
          float sdLo = coverage - (1.0 - clamp(0.5 + (nA.r - 0.5) * 1.35, 0.0, 1.0));
          float loPx = sdLo / max(length(vec2(dFdx(sdLo), dFdy(sdLo))), 1e-5);

          // ── Three committed values plus ink ──────────────────────────────
          // Outermost 2.6 px: the contour. Then a pale-cyan shadow rim. Then the
          // body, whose tone drops down the ladder as the sample ages. Only fresh,
          // powerful churn on the rails reaches uFoam, so the wake never blooms.
          float hot = step(0.55, aged) * step(0.4, max(rail, centre * head))
                    * step(0.35, vPower);
          vec3 col = mix(uCrest, uFoamShade, step(0.26, aged));
          col = mix(col, uFoam, hot);
          // Shadow side: a band of pale cyan just inside the silhouette, so the
          // foam mass has a lit face and a turned-away face instead of being one
          // flat value across its whole width.
          float onEdge = step(0.0, sdLo);
          col = mix(col, uCrest,
                    onEdge * step(loPx, uInkPx * 3.4) * (1.0 - step(loPx, uInkPx)));
          col = mix(col, uInk, onEdge * step(loPx, uInkPx));

          gl_FragColor = vec4(col, uOpacity);
        }
      `,
    });

    this.mesh = new Mesh(geo, this.material);
    this.mesh.name = 'wakeRibbons';
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 4;
    // FX never enter the G-buffer: they would scribble Sobel lines across the
    // water and the ocean reads that buffer to build its contact foam.
    this.mesh.userData.skipPrepass = true;
  }

  get tunables() {
    return this.material.uniforms;
  }

  /** Push one sample onto a boat's ring buffer. */
  private emit(
    t: Trail,
    x: number,
    z: number,
    nx: number,
    nz: number,
    w0: number,
    power: number,
    valid: number,
  ) {
    t.head = (t.head + 1) % POINTS;
    const i = t.head;
    t.x[i] = x;
    t.z[i] = z;
    t.nx[i] = nx;
    t.nz[i] = nz;
    t.age[i] = 0;
    t.w0[i] = w0;
    t.power[i] = power;
    t.valid[i] = valid;
    if (t.count < POINTS) t.count++;
  }

  update(ctx: GameContext, racers: Racer[], wetness: Float32Array) {
    const dt = ctx.dt;
    const pos = this.aPos.array as Float32Array;
    const ages = this.aAge.array as Float32Array;
    const powers = this.aPower.array as Float32Array;
    const runs = this.aRun.array as Float32Array;
    const beam = CONFIG.boat.beam;
    const halfLen = CONFIG.boat.length * 0.5;

    for (let r = 0; r < racers.length && r < this.trails.length; r++) {
      const racer = racers[r];
      const s = racer.state;
      const t = this.trails[r];

      for (let i = 0; i < POINTS; i++) t.age[i] += dt;

      // Stern position: heading 0 faces +Z, so forward is (sin h, cos h).
      const fx = Math.sin(s.heading);
      const fz = Math.cos(s.heading);
      const sx = racer.root.position.x - fx * halfLen;
      const sz = racer.root.position.z - fz * halfLen;

      const speed = Math.abs(s.forwardSpeed);
      const dx = sx - t.lastX;
      const dz = sz - t.lastZ;
      t.carry += Math.hypot(dx, dz);
      t.lastX = sx;
      t.lastZ = sz;

      // While the boat is in the air it lays no wake — and, importantly, it must
      // not *overwrite* the wake it already laid. The first pass emitted invalid
      // samples through the whole flight, which walked the ring buffer forward
      // and erased the trail; the foam_wake capture came back with a boat in
      // mid-air and no wake at all behind it.
      //
      // The contact test is the water's own graded wetness, not `state.airborne`.
      // See `hullWetness` — in r3 the flag was set on five of fifteen shots with
      // the hull visibly planing, and every one of those frames lost its wake head.
      const laying = speed > 1.6 && wetness[r] > 0.3;
      // Lateral is the travel perpendicular; from the hull axis it is (fz, -fx).
      const w0 = beam * 0.42 + speed * 0.028 + (s.drifting ? 0.55 : 0.0);
      const power = Math.min(
        1,
        speed / CONFIG.boat.topSpeed +
          Math.abs(s.lateralSpeed) * 0.12 +
          (s.boostTime > 0 ? 0.35 : 0),
      );
      if (laying && (t.head < 0 || t.carry >= EMIT_STEP)) {
        t.carry = 0;
        this.emit(t, sx, sz, fz, -fx, w0, power, 1);
      }

      // ── Rewrite this boat's vertex slice, oldest → newest ─────────────────
      const base = r * POINTS * CROSS;
      let run = 0;
      let px = 0;
      let pz = 0;
      for (let k = 0; k < POINTS; k++) {
        // k = 0 is the OLDEST sample so the ribbon runs tail → stern.
        const src = (t.head + 1 + k) % POINTS;
        const age = t.age[src];
        const valid = t.valid[src] > 0.5 && t.count > 1 && age < LIFE;
        // Kelvin-style divergence: the ribbon widens as it ages.
        const halfW = valid ? t.w0[src] + age * SPREAD : 0;
        const cx = t.x[src];
        const cz = t.z[src];
        const nx = t.nx[src];
        const nz = t.nz[src];

        if (k > 0) run += Math.hypot(cx - px, cz - pz);
        px = cx;
        pz = cz;

        const row = base + k * CROSS;
        for (let c = 0; c < CROSS; c++) {
          const side = (c / (CROSS - 1)) * 2 - 1;
          const i0 = (row + c) * 3;
          pos[i0 + 0] = cx + nx * halfW * side;
          pos[i0 + 1] = 0;
          pos[i0 + 2] = cz + nz * halfW * side;
          ages[row + c] = valid ? age : LIFE * 2;
          powers[row + c] = t.power[src];
          runs[row + c] = run;
        }
      }
      // ── Live head: weld the newest row to the transom, this frame ─────────
      // Samples are only pushed every EMIT_STEP metres, so the newest one sits up
      // to 1.15 m astern of the boat. The r13 pack shot caught exactly that as a
      // gap between the green boat's transom and the start of its wake. Rewriting
      // the newest row from the live stern transform every frame closes the gap
      // without spending a ring-buffer slot per frame, and it costs nothing: the
      // row is one segment long and lies on the path the boat has just travelled.
      if (laying && t.count > 1) {
        const row = base + (POINTS - 1) * CROSS;
        for (let c = 0; c < CROSS; c++) {
          const side = (c / (CROSS - 1)) * 2 - 1;
          const i0 = (row + c) * 3;
          pos[i0 + 0] = sx + fz * w0 * side;
          pos[i0 + 1] = 0;
          pos[i0 + 2] = sz - fx * w0 * side;
          ages[row + c] = 0;
          powers[row + c] = power;
        }
      }

      // `run` accumulates from the tail; the shader wants distance from the
      // stern, so flip it in a second pass over the same slice.
      for (let k = 0; k < POINTS * CROSS; k++) runs[base + k] = run - runs[base + k];
    }

    this.aPos.needsUpdate = true;
    this.aAge.needsUpdate = true;
    this.aPower.needsUpdate = true;
    this.aRun.needsUpdate = true;
  }

  dispose() {
    this.mesh.geometry.dispose();
    this.material.dispose();
  }
}

/**
 * Foam collar around every hull.
 *
 * ── Why this exists as geometry ─────────────────────────────────────────────
 * The ocean shader also builds a contact mask from the G-buffer depth
 * difference, and that mask is correct — but it is geometrically incapable of
 * being a *collar*. A depth-difference mask can only mark water that is in
 * front of recorded geometry, i.e. the sliver of surface that overlaps the
 * submerged part of the hull in screen space. From a chase camera that sliver is
 * two or three pixels wide. Verified in shots/water_r3 and r4: the boat sat on
 * the water with a clean outline and no churn at the waterline at all.
 *
 * So the weld is a real patch of geometry: a small grid per boat, riding the
 * shared wave field in the vertex shader like everything else that touches the
 * water, carrying an elliptical annulus mask in local hull space. One mesh, one
 * draw call, four boats.
 */
/**
 * Collar lattice density.
 *
 * 11 × 15 over a 5.9 × 10.6 m patch is a 0.6 × 0.76 m cell, and each vertex is
 * placed on the raw wave field, so the patch is a piecewise-linear chord across a
 * curved surface. On a crest the chord passes *below* the ocean mesh by more than
 * the 13 cm lift, and those cells are depth-rejected — which is what the r6
 * capture showed as a collar present aft and absent at the bow, with straight
 * diagonal boundaries between the two. 15 × 21 quarters the sagitta.
 */
const COLLAR_U = 15;
const COLLAR_V = 21;

export class HullCollars {
  readonly mesh: Mesh;
  private material: ShaderMaterial;
  private aPos: BufferAttribute;
  private aPower: BufferAttribute;
  /** Cached local-space lattice — read every frame, so never look it up by name. */
  private localUv: Float32Array;

  constructor(racerCount: number, noise: Texture) {
    const per = COLLAR_U * COLLAR_V;
    const verts = racerCount * per;
    const positions = new Float32Array(verts * 3);
    const uvs = new Float32Array(verts * 2);
    const powers = new Float32Array(verts);

    for (let r = 0; r < racerCount; r++) {
      for (let j = 0; j < COLLAR_V; j++) {
        for (let i = 0; i < COLLAR_U; i++) {
          const k = r * per + j * COLLAR_U + i;
          uvs[k * 2 + 0] = (i / (COLLAR_U - 1)) * 2 - 1;
          uvs[k * 2 + 1] = (j / (COLLAR_V - 1)) * 2 - 1;
        }
      }
    }

    const quads = racerCount * (COLLAR_U - 1) * (COLLAR_V - 1);
    const indices = new Uint16Array(quads * 6);
    let n = 0;
    for (let r = 0; r < racerCount; r++) {
      const base = r * per;
      for (let j = 0; j < COLLAR_V - 1; j++) {
        for (let i = 0; i < COLLAR_U - 1; i++) {
          const a = base + j * COLLAR_U + i;
          const b = a + 1;
          const c = a + COLLAR_U;
          const d = c + 1;
          indices[n++] = a;
          indices[n++] = c;
          indices[n++] = b;
          indices[n++] = b;
          indices[n++] = c;
          indices[n++] = d;
        }
      }
    }

    const geo = new BufferGeometry();
    this.localUv = uvs;
    this.aPos = new BufferAttribute(positions, 3);
    this.aPower = new BufferAttribute(powers, 1);
    this.aPos.setUsage(DynamicDrawUsage);
    this.aPower.setUsage(DynamicDrawUsage);
    geo.setAttribute('position', this.aPos);
    geo.setAttribute('aUv', new BufferAttribute(uvs, 2));
    geo.setAttribute('aPower', this.aPower);
    geo.setIndex(new BufferAttribute(indices, 1));
    geo.boundingSphere = null;

    this.material = new ShaderMaterial({
      name: 'hullCollar',
      transparent: true,
      depthWrite: false,
      blending: NormalBlending,
      side: DoubleSide,
      uniforms: {
        uWaveA: { value: waveUniformArrays.uWaveA },
        uWaveB: { value: waveUniformArrays.uWaveB },
        uTime: SHARED.uTime,
        uNoise: { value: noise },
        uFoam: { value: PAL.foam.clone() },
        uFoamShade: { value: PAL.foamShade.clone() },
        uCrest: { value: PAL.waterCrest.clone() },
        uInk: { value: PAL.inkSoft.clone() },
        /**
         * 13 cm was not enough to clear the difference between the raw field this
         * patch rides and the band-limited field the ocean mesh draws, plus the
         * sagitta of the patch's own tessellation. See COLLAR_V.
         */
        uLift: { value: 0.2 },
        uTile: { value: 3.3 },
        /**
         * Inner and outer radius of the annulus, in units where 1.0 is the hull's
         * own outline. 0.80 → 1.42 is a collar that starts hidden under the shell
         * and spills ~40 cm proud of it at the beam.
         */
        uInner: { value: 0.8 },
        uOuter: { value: 1.42 },
        /** Cutout, so this is 1. Kept as a uniform for the critic loop only. */
        uOpacity: { value: 1.0 },
        /** Width of the collar's own ink contour, in device pixels. */
        uInkPx: { value: 2.6 },
        /**
         * Bow wave. `uVSpread` is how far the arms splay per unit of hull length
         * at rest; speed opens them further. Written per frame from the boat's
         * speed by `HullCollars.update` via aPower, so a stationary boat has a
         * collar and no V, and a boat at 29 m/s has a wide one.
         */
        uVSpread: { value: 0.45 },
        uVWidth: { value: 0.13 },
      },
      vertexShader: /* glsl */ `
        ${GERSTNER_GLSL}
        attribute vec2 aUv;
        attribute float aPower;
        uniform float uLift;
        varying vec2 vUv;
        varying float vPower;
        varying vec3 vWorldPos;
        void main() {
          vec3 p; vec3 nrm; float j;
          gerstnerSurface(position.xz, uTime, p, nrm, j);
          p.y += uLift;
          vUv = aUv;
          vPower = aPower;
          vWorldPos = p;
          gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0);
        }
      `,
      fragmentShader: /* glsl */ `
        precision highp float;
        uniform sampler2D uNoise;
        uniform vec3 uFoam, uFoamShade, uCrest, uInk;
        uniform float uTime, uTile, uInner, uOuter, uOpacity, uInkPx;
        uniform float uVSpread, uVWidth;
        varying vec2 vUv;
        varying float vPower;
        varying vec3 vWorldPos;

        void main() {
          vec2 uv = vWorldPos.xz / uTile;
          vec4 nA = texture2D(uNoise, uv + vec2(uTime * 0.05, uTime * -0.03));
          vec4 nB = texture2D(uNoise, uv * 2.9 - vec2(uTime * 0.11, 0.0));
          // Feature-size floor, 8–16 px, same rule as the ribbon and the ocean,
          // and gated on each octave's REAL scale. The old third term was nB.b at
          // 2.9× the base rate — a feature size of uTile/325 ≈ 1 cm, i.e. sub-pixel
          // everywhere, and thresholding that is what put a salt-and-pepper fringe
          // right where the collar's silhouette needed to be a drawn shape.
          float fp = max(max(fwidth(vWorldPos.x), fwidth(vWorldPos.z)), 1e-5);
          float wMid = smoothstep(fp * 8.0, fp * 16.0, uTile / 14.0);
          float wFine = smoothstep(fp * 8.0, fp * 16.0, uTile / 40.6);
          float grain = (nA.r * 0.52 + nA.g * 0.28 * wMid + nB.g * 0.20 * wFine)
                      / (0.52 + 0.28 * wMid + 0.20 * wFine);
          grain = clamp(0.5 + (grain - 0.5) * 1.3, 0.0, 1.0);

          // Elliptical distance in hull-footprint units: 1.0 is the hull's own
          // outline, so the annulus starts just inside it and spills outward.
          //
          // The radius is warped by the noise *before* the annulus is built. The
          // r7 capture showed why: threshold a clean ellipse and, however lacy the
          // alpha afterwards, the outer boundary is still a visible circular arc
          // sitting on the water. Warping the metric means there is no arc to see.
          // The divisors put e = 1 exactly on the hull's own outline (0.32 × the
          // patch half-width is 0.94 m against a half-beam of 0.95 m; 0.435 × the
          // patch half-length is 2.30 m against a half-length of 2.30 m). They used
          // to be 0.42/0.56, i.e. e = 1 sat 30% *outside* the hull, so the whole
          // annulus was pushed off the waterline and the r5 capture showed the
          // collar as a floe of foam near the boat rather than a ring welded to it.
          float e = length(vec2(vUv.x / 0.32, vUv.y / 0.435));
          // Warp the metric before building the annulus, so there is no clean arc
          // anywhere. Held to ±0.18 — about a quarter of the annulus width — since
          // beyond that the ring stops being a ring.
          e *= 1.0 + (grain - 0.5) * 0.22 + (nB.g - 0.5) * 0.14;

          // The collar must be a *thick* band, not a hairline: the r3 review found
          // the waterline was "a razor-sharp geometric polygon edge with the water
          // band colour changing on the very next pixel", and a two-pixel ring
          // would not fix that. It starts inside the hull's own footprint (the hull
          // covers that part) and spills well outside it.
          float annulus = smoothstep(uInner, uInner + 0.14, e)
                        * (1.0 - smoothstep(uOuter * 0.72, uOuter, e));
          // Bow push and stern churn: the water piles up ahead of the hull and
          // boils behind it, so the collar is not a uniform ring.
          // ...but it must be *continuous* all the way round first. At a 0.55 base
          // the amidships collar — the part a chase camera actually sees — sat at
          // the minimum and the noise ate most of it.
          float bow = smoothstep(0.1, 0.6, vUv.y) * 0.32;
          float stern = smoothstep(-0.1, -0.8, vUv.y) * 0.55;
          float shape = annulus * (0.78 + bow + stern);

          // ── Bow wave: two divergent arms off the forefoot ──────────────────
          // The r3 review asked for "a bow-wave V that scales with speed" — the
          // cue that reads as hull speed and hull weight. Its arms leave the bow
          // and splay aft, so the apex is welded to the stem and the legs are
          // where the displaced water actually runs. The splay angle opens with
          // vPower, i.e. with speed, which is what makes it a speed cue rather
          // than decoration.
          // apex 0.46 puts it on the stem (0.46 × 5.29 m = 2.43 m forward).
          float apex = 0.46;
          float aft = clamp(apex - vUv.y, 0.0, 2.0);
          float spread = uVSpread * (0.55 + 0.75 * vPower);
          float arm = abs(abs(vUv.x) - (0.16 + spread * aft));
          float vwid = uVWidth * (1.0 + 0.9 * aft);
          float vee = (1.0 - smoothstep(0.0, vwid, arm))
                    * step(vUv.y, apex)
                    // ...and it fades out well inside the patch, so the arms end
                    // in foam breaking up rather than at the carrier's edge.
                    * (1.0 - smoothstep(0.55, 1.4, aft))
                    * (1.0 - smoothstep(0.72, 0.96, abs(vUv.x)))
                    * smoothstep(0.12, 0.45, vPower);
          shape = max(shape, vee * 1.15);

          // ── Hard cutout, then a drawn edge ───────────────────────────────
          // The mask is 0 or 1 and the fragment is opaque; fading is expressed as
          // less coverage and a lower tone, never as transparency. Capped below 1
          // so the noise always has range left to bite holes with — four grid-start
          // collars merging at full coverage is the "spilled paint puddle" the
          // countdown capture showed.
          float coverage = clamp(shape * (0.45 + 0.8 * vPower), 0.0, 0.86);
          float sd = coverage - (1.0 - grain);
          if (sd < 0.0) discard;
          // Signed distance converted to PIXELS. This is what makes the collar a
          // hard two-tone *shape with a contour* rather than an alpha falloff:
          // outermost 2.6 px is ink, the next band is the pale-cyan shadow side,
          // and the interior is white churn.
          //
          // Measured against the COARSE channel for the same reason as the ribbon:
          // a foam island narrower than the ink width would otherwise come out
          // entirely ink, i.e. as dark speckle inside the collar.
          float sdLo = coverage - (1.0 - clamp(0.5 + (nA.r - 0.5) * 1.3, 0.0, 1.0));
          float loPx = sdLo / max(length(vec2(dFdx(sdLo), dFdy(sdLo))), 1e-5);
          float onEdge = step(0.0, sdLo);

          vec3 col = mix(uFoamShade, uFoam, step(0.42, coverage));
          col = mix(col, uCrest,
                    onEdge * step(loPx, uInkPx * 3.2) * (1.0 - step(loPx, uInkPx)));
          col = mix(col, uInk, onEdge * step(loPx, uInkPx));
          gl_FragColor = vec4(col, uOpacity);
        }
      `,
    });

    this.mesh = new Mesh(geo, this.material);
    this.mesh.name = 'hullCollars';
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 3;
    this.mesh.userData.skipPrepass = true;
  }

  get tunables() {
    return this.material.uniforms;
  }

  update(_ctx: GameContext, racers: Racer[], wetness: Float32Array) {
    const pos = this.aPos.array as Float32Array;
    const powers = this.aPower.array as Float32Array;
    const per = COLLAR_U * COLLAR_V;
    // The patch is deliberately larger than the hull so the collar has room to
    // spill outward instead of being clipped at the hull's own silhouette.
    const halfW = CONFIG.boat.beam * 1.55;
    const halfL = CONFIG.boat.length * 1.15;

    for (let r = 0; r < racers.length; r++) {
      const s = racers[r].state;
      const p = racers[r].root.position;
      const fx = Math.sin(s.heading);
      const fz = Math.cos(s.heading);
      const rx = fz;
      const rz = -fx;
      const speed = Math.abs(s.forwardSpeed);
      // A hull with no waterline has no collar, so the collar is scaled by the
      // water's own graded wetness rather than by `state.airborne`: it fades over
      // the ~10 cm the keel takes to leave the surface instead of snapping off the
      // instant another subsystem's boolean latches. That boolean is what removed
      // the collar from hero.png and ocean_low.png in r3.
      //
      // A hull sitting still still displaces water, so the floor is well above
      // zero: the results screen has speed 0 and the boats must not go back to
      // being pasted onto a plane.
      const power =
        Math.min(1, 0.38 + speed / CONFIG.boat.topSpeed + Math.abs(s.lateralSpeed) * 0.1) *
        wetness[r];

      const base = r * per;
      for (let k = 0; k < per; k++) {
        const u = this.localUv[(base + k) * 2];
        const v = this.localUv[(base + k) * 2 + 1];
        const i0 = (base + k) * 3;
        pos[i0 + 0] = p.x + rx * u * halfW + fx * v * halfL;
        pos[i0 + 1] = 0;
        pos[i0 + 2] = p.z + rz * u * halfW + fz * v * halfL;
        powers[base + k] = power;
      }
    }

    this.aPos.needsUpdate = true;
    this.aPower.needsUpdate = true;
  }

  dispose() {
    this.mesh.geometry.dispose();
    this.material.dispose();
  }
}

/**
 * Everything the water throws into the air or leaves behind it, behind one
 * update call: the wake ribbons, the hull collars and the spray field.
 *
 * `Ocean` owns an instance of this so the FX arrive with the water rather than
 * needing a separate registration in `main.ts`. It is still a plain object with
 * `update(ctx)`, so promoting it to its own `Subsystem` at order 70 is a
 * two-line change if the integrator prefers that.
 */
export class WaterFX {
  readonly group = new Group();
  readonly wake: WakeRibbons;
  readonly collars: HullCollars;
  readonly spray: SprayField;

  /**
   * Per-racer hull wetness, recomputed once per frame and shared by all three
   * effects. Preallocated — the frame loop allocates nothing.
   */
  private wetness = new Float32Array(4);

  constructor(racerCount = CONFIG.race.racerCount) {
    // A private cache key so switching this map to trilinear cannot affect any
    // other subsystem's use of makeNoiseTexture.
    const noise = makeNoiseTexture(512, 5, 5);
    noise.minFilter = LinearMipmapLinearFilter;
    noise.generateMipmaps = true;
    noise.anisotropy = 8;
    noise.needsUpdate = true;

    this.wake = new WakeRibbons(racerCount, noise);
    this.collars = new HullCollars(racerCount, noise);
    this.spray = new SprayField(noise);
    this.group.name = 'waterFx';
    this.group.add(this.collars.mesh);
    this.group.add(this.wake.mesh);
    this.group.add(this.spray.points);
  }

  update(ctx: GameContext) {
    const n = Math.min(ctx.racers.length, this.wetness.length);
    for (let i = 0; i < n; i++) this.wetness[i] = hullWetness(ctx, ctx.racers[i]);
    this.wake.update(ctx, ctx.racers, this.wetness);
    this.collars.update(ctx, ctx.racers, this.wetness);
    this.spray.update(ctx, ctx.racers, this.wetness);
  }

  dispose() {
    this.wake.dispose();
    this.collars.dispose();
    this.spray.dispose();
  }
}
