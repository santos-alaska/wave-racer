/**
 * The infinite ocean mesh.
 *
 * ── Why a radial grid ──────────────────────────────────────────────────────
 * The brief rules out three specific failures: visible seams, visible tiling
 * repetition, and LOD popping. Each candidate approach fails at least one:
 *
 *   • tiled patches with per-tile LOD  → seams at the stitch, popping on swap
 *   • a single uniform grid            → either far too dense or far too coarse
 *   • a screen-space projected grid    → no seams, but degenerate at grazing
 *                                        angles and awkward for CPU queries
 *
 * A single **radial disc** centred on the camera has none of them. It is one
 * mesh, so there is no seam anywhere. Vertex density falls off exponentially
 * with radius, so detail is where the camera is without any discrete LOD level
 * to pop between. And because the wave field is evaluated in *absolute world
 * space*, sliding the disc under the camera produces no repetition — the mesh
 * moves, the water does not.
 *
 * The disc is re-centred every frame and snapped to a fixed grid. Snapping
 * matters: without it, vertices creep continuously relative to the wave field
 * and high-frequency crests visibly shimmer as the sampling points slide.
 *
 * ── Two fixes the first captures forced ─────────────────────────────────────
 * 1. **Rings are rotated by a hashed offset.** With every ring's vertices at
 *    the same angles, the grid's spokes line up into faint radial rays that
 *    the eye picks up as rotational structure centred on the player. A
 *    per-ring angular offset destroys that alignment; the quads skew by at
 *    most half a cell, which is invisible.
 * 2. **The grid publishes its own spacing to the shader.** The ocean material
 *    band-limits the wave field to the local vertex spacing, so it needs to
 *    know `spacing(r) ≈ 2πr / angularSteps` — measured here rather than
 *    hard-coded there, so retuning the mesh cannot silently desync the filter.
 */

import { BufferAttribute, BufferGeometry, Mesh, Texture, Vector3 } from 'three';
import { CONFIG } from '../core/config';
import { createOceanMaterial } from './oceanMaterial';
import { WaterFX } from './foam';
import { maxWaveHeight, sampleHeight, sampleOcean } from './gerstner';
import type { OceanSample } from './gerstner';
import type { GameContext, OceanSampler, Subsystem } from '../core/types';

/**
 * Scratch for the per-frame camera direction. Module scope, because the frame
 * loop must not allocate.
 */
const _camFwd = new Vector3();

/** Deterministic per-ring hash, so the mesh is identical on every run. */
function ringHash(i: number): number {
  let x = Math.imul(i ^ 0x9e3779b9, 0x85ebca6b) >>> 0;
  x ^= x >>> 13;
  x = Math.imul(x, 0xc2b2ae35) >>> 0;
  return ((x ^ (x >>> 16)) >>> 0) / 4294967296;
}

/**
 * Build the radial lattice.
 *
 * Radius distribution is exponential: r(i) = R·(e^(k·i/N) − 1)/(e^k − 1).
 * k trades near-field density against mid-field density. k = 5.6 (the first
 * attempt) put half the rings inside 150 m, which starved the 50–400 m band
 * where the aliasing stripes were worst; k = 5.0 pushes that out to ~240 m and
 * the mid-field reads much better with the same vertex count.
 *
 * The centre is a fan of triangles to a single origin vertex, avoiding the
 * pinched degenerate quads a naive polar grid produces at r → 0.
 */
function buildRadialGrid(radialSteps: number, angularSteps: number, radius: number, k: number) {
  const vertCount = 1 + radialSteps * angularSteps;
  const positions = new Float32Array(vertCount * 3);
  const uvs = new Float32Array(vertCount * 2);

  positions[0] = 0;
  positions[1] = 0;
  positions[2] = 0;

  const denom = Math.exp(k) - 1;
  const angStep = (Math.PI * 2) / angularSteps;

  let p = 3;
  let t = 2;
  for (let i = 0; i < radialSteps; i++) {
    const fi = (i + 1) / radialSteps;
    const r = (radius * (Math.exp(k * fi) - 1)) / denom;
    // Break the spoke alignment that otherwise reads as radial rays.
    const off = (ringHash(i) - 0.5) * angStep;
    for (let a = 0; a < angularSteps; a++) {
      const theta = a * angStep + off;
      positions[p++] = Math.cos(theta) * r;
      positions[p++] = 0;
      positions[p++] = Math.sin(theta) * r;
      uvs[t++] = fi;
      uvs[t++] = a / angularSteps;
    }
  }

  const triCount = angularSteps + (radialSteps - 1) * angularSteps * 2;
  const indices = new Uint32Array(triCount * 3);
  let n = 0;

  // Centre fan.
  for (let a = 0; a < angularSteps; a++) {
    indices[n++] = 0;
    indices[n++] = 1 + ((a + 1) % angularSteps);
    indices[n++] = 1 + a;
  }
  // Rings.
  for (let i = 0; i < radialSteps - 1; i++) {
    const base = 1 + i * angularSteps;
    const next = base + angularSteps;
    for (let a = 0; a < angularSteps; a++) {
      const a1 = (a + 1) % angularSteps;
      indices[n++] = base + a;
      indices[n++] = next + a1;
      indices[n++] = next + a;
      indices[n++] = base + a;
      indices[n++] = base + a1;
      indices[n++] = next + a1;
    }
  }

  const geo = new BufferGeometry();
  geo.setAttribute('position', new BufferAttribute(positions, 3));
  geo.setAttribute('uv', new BufferAttribute(uvs, 2));
  geo.setIndex(new BufferAttribute(indices, 1));
  // The mesh follows the camera, so a bounding volume test is meaningless.
  geo.boundingSphere = null;
  return geo;
}

export class Ocean implements Subsystem, OceanSampler {
  readonly name = 'ocean';
  readonly order = 20;

  readonly mesh: Mesh;
  private handles = createOceanMaterial();

  /**
   * Wake ribbons + spray. Parented to the ocean mesh so that adding the ocean
   * to the scene brings the water FX with it — the FX have no meaningful
   * existence without the surface they live on, and this keeps the whole water
   * subsystem to a single `scene.add()` and a single `update()`.
   *
   * The mesh is re-centred on the camera every frame, so the FX group carries
   * the inverse offset to put its children back in absolute world space.
   */
  readonly fx = new WaterFX();

  constructor(radialSteps = 224, angularSteps = 224, radius = CONFIG.ocean.extent, k = 5.0) {
    const geo = buildRadialGrid(radialSteps, angularSteps, radius, k);
    this.mesh = new Mesh(geo, this.handles.material);
    this.mesh.name = 'ocean';
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 0;
    // The water is excluded from the G-buffer for two reasons: we do not want
    // Sobel lines scribbled across the swell, and — more importantly — the
    // water *reads* the G-buffer to build its foam ring, so it must not be in
    // it. See oceanMaterial.ts.
    this.mesh.userData.skipPrepass = true;

    // Publish the mesh's own spacing law to the shader's band-limiting filter.
    //   angular spacing = 2πr / angularSteps
    //   radial  spacing = R·k/(N·(e^k−1)) · e^(k·fi)  — the same order, and at
    //   the innermost ring it is the larger of the two, so it sets the floor.
    const u = this.handles.material.uniforms;
    u.uFilterSlope.value = (Math.PI * 2) / angularSteps;
    u.uFilterBase.value = (radius * k) / (radialSteps * (Math.exp(k) - 1));

    this.mesh.add(this.fx.group);
  }

  /** Called by the renderer once the G-buffer for this frame exists. */
  setSceneDepth(tex: Texture | null) {
    this.handles.setSceneDepth(tex);
  }

  get material() {
    return this.handles.material;
  }

  /** Exposed so the harness/critic loop can twiddle a threshold without a rebuild. */
  get tunables() {
    return this.handles.uniforms;
  }

  update(ctx: GameContext) {
    // Re-centre on the camera, snapped so vertices do not creep against the
    // wave field. The snap interval must be larger than the finest wave
    // wavelength / 2 to be effective, and small enough not to be visible.
    const snap = CONFIG.ocean.snap;
    const cx = Math.round(ctx.camera.position.x / snap) * snap;
    const cz = Math.round(ctx.camera.position.z / snap) * snap;
    this.mesh.position.set(cx, 0, cz);
    // Undo the parent's offset so wake ribbons and spray stay in absolute world
    // space; both of them address the wave field by world XZ.
    this.fx.group.position.set(-cx, 0, -cz);

    // ── Publish the view elevation, once per frame ────────────────────────────
    // The shader needs "how much is the camera looking along the water rather
    // than down at it" to scale its one view-dependent shading term. Measuring
    // that per fragment (`1 - abs(V.y)`) makes it a radial function of the point
    // under the camera, and quantising a radial function draws concentric rings —
    // which is what put fingerprint whorls in the aerial capture. One scalar for
    // the whole frame cannot have azimuthal or radial structure, by construction.
    ctx.camera.getWorldDirection(_camFwd);
    const graze = 1 - Math.min(1, Math.abs(_camFwd.y));
    this.handles.material.uniforms.uGraze.value = graze * graze;

    // FX run after the boats have moved. Ocean is order 20 and boat physics is
    // order 30, so what we read here is last frame's hull position — one frame
    // of latency on a trail that is metres long and seconds old is not
    // observable, and it keeps the whole water subsystem on one registration.
    this.fx.update(ctx);
  }

  // ── OceanSampler ──────────────────────────────────────────────────────────
  sample(x: number, z: number, t: number, out?: OceanSample) {
    return sampleOcean(x, z, t, out);
  }
  height(x: number, z: number, t: number) {
    return sampleHeight(x, z, t);
  }
  maxHeight() {
    return maxWaveHeight();
  }
}
