/**
 * Procedural racing-boat hull.
 *
 * Everything here is authored as raw triangles in a tiny `Surface` builder and
 * merged into **three** BufferGeometries — one per cel material — so a boat
 * costs 3 shaded draw calls plus 3 inverted-hull outlines rather than one call
 * per greeble.
 *
 * ── Why hand-rolled triangles instead of CSG or lathes ─────────────────────
 * The brief is cel art, not CAD. A cel silhouette wants *few, confident*
 * planes with hard creases: every crease is a place the Sobel pass can put an
 * interior line and every large flat plane is a place the banded specular can
 * land as a readable shape. Smooth lathed forms give you neither — they read as
 * soft plastic under a quantised ramp. So the hull is a loft over eight
 * hand-tuned stations with a **hard chine** (a deliberate crease running the
 * whole length) and the superstructure is a set of tapered boxes.
 *
 * ── Silhouette contract ────────────────────────────────────────────────────
 * The player mostly sees rivals at 40–80 m, where only the outline survives.
 * Read from the side, the boat must break down into five distinct masses:
 *
 *      spoiler ─┐        ┌── raked windscreen
 *               ▼        ▼
 *          ╭──▒▒▒╮   ╭─╮
 *      ────┤ cowl│═══╡ ╞═══════════◣  ← pointed, rising bow
 *          ╰─────╯   ╰─╯            ◤
 *             │  saddle
 *             fin
 *
 * Those five (bow spike, windscreen, saddle, engine hump, spoiler) are what
 * make it read as a *racing* boat and not a dinghy at distance.
 *
 * ── Normals, and the stipple this file used to produce ─────────────────────
 * Geometry is non-indexed with per-face normals, i.e. genuinely faceted. That
 * matters twice: the main pass gets flat shading without needing the
 * FLAT_SHADING define, and the *prepass* also gets flat normals, so the
 * screen-space edge pass inks every crease. `computeSmoothNormals` (called by
 * `applyCel`) then welds by position for the outline hull, so the outline stays
 * closed across those same creases.
 *
 * Per-*triangle* normals were not enough, and the failure was visible: the
 * foredeck in shots/boat_b0/outline_check.png carried a fine 45° dashed
 * stipple laid out in nested arcs, and the same pattern crawled over the
 * cowling and the gunwale insides. It reads as moiré, and it is neither dither
 * nor hatching — it is the *triangulation itself* showing through the cel ramp.
 * A crowned deck is 7 columns × 12 stations = 168 slivers, each with its own
 * constant normal; a ramp stop sweeping across that grid quantises each sliver
 * independently, so the band boundary becomes a dashed contour along the
 * triangulation diagonals. The non-planar quads make it worse: the two
 * triangles of one quad differ by 15–40°, which is a normal break the Sobel
 * pass can ink on its own.
 *
 * The fix is `beginFacet()` / `endFacet()`. Triangles written inside a facet
 * share ONE area-weighted normal, so a whole deck panel or hull strake is a
 * single flat tone no matter how many triangles it is built from, and its
 * *boundaries* are the only creases left to ink. That is what cel art wants
 * anyway: few, confident planes with drawn edges between them.
 */

import { BufferAttribute, BufferGeometry, Color, Group, Mesh, Object3D, Vector3 } from 'three';
import { PAL, RACER_COLORS } from '../core/palette';
import { applyCel, createCelMaterial, type CelMaterialSet } from '../render/celMaterial';

// ─────────────────────────────────────────────────────────────────────────────
// Triangle soup builder
// ─────────────────────────────────────────────────────────────────────────────

type V3 = readonly [number, number, number];
/** 2D cross-section point, (x, y). */
type P2 = readonly [number, number];
/** Corner rect for a tapered box: [x0, x1, y0, y1]. */
type Rect = readonly [number, number, number, number];

const mirrorX = (v: V3): V3 => [-v[0], v[1], v[2]];

/**
 * Facet ids are handed out from one module-level counter rather than per
 * Surface, so `append()` can merge two builders without two facets colliding
 * on the same id and being welded into one tone.
 */
let facetCounter = 0;

class Surface {
  private p: number[] = [];
  /**
   * Per-triangle facet id, parallel to `p` in units of triangles. 0 means "no
   * facet" — that triangle keeps its own geometric normal.
   */
  private f: number[] = [];
  private facet = 0;

  /**
   * Open a shading facet. Every triangle written until `endFacet()` is shaded
   * with one shared, area-weighted normal — one flat cel tone across the whole
   * form, with creases only at its edges.
   */
  beginFacet() {
    this.facet = ++facetCounter;
    return this;
  }

  endFacet() {
    this.facet = 0;
  }

  tri(a: V3, b: V3, c: V3) {
    this.p.push(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
    this.f.push(this.facet);
  }

  /** Convex quad a→b→c→d. Normal is cross(b-a, c-a). */
  quad(a: V3, b: V3, c: V3, d: V3) {
    this.tri(a, b, c);
    this.tri(a, c, d);
  }

  /**
   * Same quad mirrored to port when `sign < 0`. Mirroring flips winding, so the
   * vertex order is reversed to keep the normal pointing outboard.
   */
  quadMirrored(sign: number, a: V3, b: V3, c: V3, d: V3) {
    if (sign > 0) this.quad(a, b, c, d);
    else this.quad(mirrorX(a), mirrorX(d), mirrorX(c), mirrorX(b));
  }

  /** Loft two equal-length open rings. `flip` reverses the quad winding. */
  loft(ringA: V3[], ringB: V3[], flip = false, from = 0, to = -1) {
    const end = to < 0 ? ringA.length - 1 : to;
    for (let j = from; j < end; j++) {
      if (flip) this.quad(ringA[j], ringB[j], ringB[j + 1], ringA[j + 1]);
      else this.quad(ringA[j], ringA[j + 1], ringB[j + 1], ringB[j]);
    }
  }

  /**
   * Cap a ring with a triangle fan about `centre`, winding chosen so the face
   * normal agrees with `want`. Deciding the winding from the geometry rather
   * than by hand is why none of the caps in this file can end up inside-out.
   */
  cap(ring: V3[], centre: V3, want: V3, closed = true) {
    const n = ring.length;
    const probe = triNormal(centre, ring[0], ring[1]);
    const flip = probe[0] * want[0] + probe[1] * want[1] + probe[2] * want[2] < 0;
    const last = closed ? n : n - 1;
    for (let j = 0; j < last; j++) {
      const a = ring[j];
      const b = ring[(j + 1) % n];
      if (flip) this.tri(centre, b, a);
      else this.tri(centre, a, b);
    }
  }

  /** Axis-aligned box with outward normals. */
  box(x0: number, x1: number, y0: number, y1: number, z0: number, z1: number) {
    this.taperBox(z0, z1, [x0, x1, y0, y1], [x0, x1, y0, y1]);
  }

  /**
   * Box tapered along Z between two corner rects. This is the workhorse for
   * the greebles — a frustum has six flat faces and twelve hard creases, which
   * is exactly what the cel pipeline wants to chew on.
   */
  taperBox(
    z0: number,
    z1: number,
    rect0: Rect,
    rect1: Rect,
    capBack = true,
    capFront = true,
    /**
     * Omit the +Y face. Used by the windscreen, whose outer surface is split
     * between two materials: laying a second coplanar plate on top of a
     * complete box would z-fight, and pushing it proud opens a lit seam all the
     * way round it.
     */
    skipTop = false,
  ) {
    // Normalise the inputs. Mirrored greebles are written as `sign * 0.8` pairs
    // and half of them come out with x0 > x1; an unsorted rect flips the winding
    // and the box renders inside-out, which is invisible in the code and very
    // visible as a hole in the outline hull.
    if (z0 > z1) {
      const t = z0; z0 = z1; z1 = t;
      const rt = rect0; rect0 = rect1; rect1 = rt;
    }
    const r0 = sortRect(rect0);
    const r1 = sortRect(rect1);
    const A0: V3 = [r0[0], r0[2], z0], B0: V3 = [r0[1], r0[2], z0];
    const C0: V3 = [r0[1], r0[3], z0], D0: V3 = [r0[0], r0[3], z0];
    const A1: V3 = [r1[0], r1[2], z1], B1: V3 = [r1[1], r1[2], z1];
    const C1: V3 = [r1[1], r1[3], z1], D1: V3 = [r1[0], r1[3], z1];
    this.quad(B0, C0, C1, B1); // +X
    this.quad(A1, D1, D0, A0); // -X
    if (!skipTop) this.quad(D0, D1, C1, C0); // +Y
    this.quad(A1, A0, B0, B1); // -Y
    if (capFront) this.quad(A1, B1, C1, D1); // +Z
    if (capBack) this.quad(B0, A0, D0, C0); // -Z
  }

  /** Extrude a closed CCW (x, y) profile along Z. */
  prism(profile: P2[], z0: number, z1: number, capBack = true, capFront = true) {
    const n = profile.length;
    for (let j = 0; j < n; j++) {
      const p = profile[j];
      const q = profile[(j + 1) % n];
      this.quad([p[0], p[1], z0], [q[0], q[1], z0], [q[0], q[1], z1], [p[0], p[1], z1]);
    }
    const cx = profile.reduce((s, p) => s + p[0], 0) / n;
    const cy = profile.reduce((s, p) => s + p[1], 0) / n;
    const ring0 = profile.map((p): V3 => [p[0], p[1], z0]);
    const ring1 = profile.map((p): V3 => [p[0], p[1], z1]);
    if (capBack) this.cap(ring0, [cx, cy, z0], [0, 0, -1]);
    if (capFront) this.cap(ring1, [cx, cy, z1], [0, 0, 1]);
  }

  append(other: Surface) {
    for (let i = 0; i < other.p.length; i++) this.p.push(other.p[i]);
    for (let i = 0; i < other.f.length; i++) this.f.push(other.f[i]);
  }

  get triangleCount() {
    return this.p.length / 9;
  }

  /**
   * Bake to a BufferGeometry with flat per-face normals. UVs are a cheap planar
   * projection — nothing samples them today, but three declares the attribute
   * unconditionally so shipping real values avoids a disabled-attrib warning.
   */
  geometry(name: string): BufferGeometry {
    const count = this.p.length / 3;
    const tris = count / 3;
    const pos = new Float32Array(this.p);
    const nrm = new Float32Array(count * 3);
    const uv = new Float32Array(count * 2);

    // Pass 1: geometric normal per triangle, scaled by twice its area, so a
    // facet's shared normal is area-weighted and a sliver cannot outvote the
    // panel it belongs to.
    const fx = new Float64Array(tris);
    const fy = new Float64Array(tris);
    const fz = new Float64Array(tris);
    for (let t = 0; t < tris; t++) {
      const o = t * 9;
      const ux = pos[o + 3] - pos[o], uy = pos[o + 4] - pos[o + 1], uz = pos[o + 5] - pos[o + 2];
      const vx = pos[o + 6] - pos[o], vy = pos[o + 7] - pos[o + 1], vz = pos[o + 8] - pos[o + 2];
      fx[t] = uy * vz - uz * vy;
      fy[t] = uz * vx - ux * vz;
      fz[t] = ux * vy - uy * vx;
    }

    // Pass 2: sum each facet's weighted normals.
    const acc = new Map<number, [number, number, number]>();
    for (let t = 0; t < tris; t++) {
      const id = this.f[t] ?? 0;
      if (id === 0) continue;
      const a = acc.get(id);
      if (a) {
        a[0] += fx[t];
        a[1] += fy[t];
        a[2] += fz[t];
      } else acc.set(id, [fx[t], fy[t], fz[t]]);
    }

    // Pass 3: write. Facet members take the shared normal; loners keep theirs.
    for (let t = 0; t < tris; t++) {
      const id = this.f[t] ?? 0;
      let nx = fx[t], ny = fy[t], nz = fz[t];
      if (id !== 0) {
        const a = acc.get(id)!;
        // A facet folded so far that its members cancel out has no meaningful
        // shared normal; fall back to the triangle's own rather than emitting a
        // zero normal, which would shade black.
        if (Math.hypot(a[0], a[1], a[2]) > 1e-9) {
          nx = a[0];
          ny = a[1];
          nz = a[2];
        }
      }
      const len = Math.hypot(nx, ny, nz) || 1;
      nx /= len;
      ny /= len;
      nz /= len;
      const o = t * 9;
      for (let k = 0; k < 3; k++) {
        nrm[o + k * 3 + 0] = nx;
        nrm[o + k * 3 + 1] = ny;
        nrm[o + k * 3 + 2] = nz;
      }
    }
    for (let i = 0; i < count; i++) {
      uv[i * 2 + 0] = pos[i * 3 + 0] * 0.5 + 0.5;
      uv[i * 2 + 1] = pos[i * 3 + 2] * 0.2 + 0.5;
    }
    const g = new BufferGeometry();
    g.name = name;
    g.setAttribute('position', new BufferAttribute(pos, 3));
    g.setAttribute('normal', new BufferAttribute(nrm, 3));
    g.setAttribute('uv', new BufferAttribute(uv, 2));
    g.computeBoundingSphere();
    return g;
  }
}

function sortRect(r: Rect): Rect {
  return [Math.min(r[0], r[1]), Math.max(r[0], r[1]), Math.min(r[2], r[3]), Math.max(r[2], r[3])];
}

function triNormal(a: V3, b: V3, c: V3): V3 {
  const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
  const vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
  const nx = uy * vz - uz * vy;
  const ny = uz * vx - ux * vz;
  const nz = ux * vy - uy * vx;
  const len = Math.hypot(nx, ny, nz) || 1;
  return [nx / len, ny / len, nz / len];
}

// ─────────────────────────────────────────────────────────────────────────────
// Hull stations
// ─────────────────────────────────────────────────────────────────────────────

interface Station {
  /** Longitudinal position, metres. +Z is the bow. */
  z: number;
  /** Half-beam at the chine. */
  w: number;
  /** Keel (centreline bottom) height. Negative = below the origin plane. */
  keel: number;
  /** Chine height — the hard crease where the bottom meets the side. */
  chine: number;
  /** Sheer height — the top edge of the side, where the deck starts. */
  sheer: number;
}

/**
 * Eight control stations, stern → bow. Read the `keel` column top-to-bottom and
 * you can see the rocker: flat and deep amidships, lifting sharply forward so
 * the boat has a raked forefoot rather than a barge nose. The `sheer` column
 * rises the same way, which is what gives the profile its wedge.
 */
const CONTROL: Station[] = [
  { z: -2.32, w: 0.88, keel: -0.22, chine: -0.06, sheer: 0.20 },
  { z: -1.60, w: 0.94, keel: -0.29, chine: -0.10, sheer: 0.19 },
  { z: -0.70, w: 0.95, keel: -0.33, chine: -0.12, sheer: 0.18 },
  { z: 0.20, w: 0.90, keel: -0.32, chine: -0.11, sheer: 0.22 },
  { z: 0.90, w: 0.78, keel: -0.26, chine: -0.04, sheer: 0.34 },
  { z: 1.55, w: 0.58, keel: -0.12, chine: 0.06, sheer: 0.46 },
  { z: 2.05, w: 0.30, keel: 0.10, chine: 0.22, sheer: 0.58 },
  { z: 2.34, w: 0.05, keel: 0.34, chine: 0.40, sheer: 0.64 },
];

/** Stations the loft is actually evaluated at — denser where the eye looks. */
const STATION_Z = [
  -2.32, -2.05, -1.70, -1.32, -1.06, -0.60, -0.10, 0.35, 0.90, 1.35, 1.80, 2.10, 2.34,
];

function stationAt(z: number): Station {
  let i = 0;
  while (i < CONTROL.length - 2 && CONTROL[i + 1].z < z) i++;
  const a = CONTROL[i];
  const b = CONTROL[i + 1];
  const t = Math.max(0, Math.min(1, (z - a.z) / (b.z - a.z)));
  // Smoothstep between control stations: linear interpolation puts a visible
  // kink at every control point, which reads as a dent in the hull.
  const s = t * t * (3 - 2 * t);
  const mix = (u: number, v: number) => u + (v - u) * s;
  return {
    z,
    w: mix(a.w, b.w),
    keel: mix(a.keel, b.keel),
    chine: mix(a.chine, b.chine),
    sheer: mix(a.sheer, b.sheer),
  };
}

const DECK_CROWN = 0.055;
const RAIL_H = 0.07;
const RAIL_W = 0.055;
/**
 * Fraction of the sheer half-width the deck columns sit at. The innermost pair
 * is deliberately narrow: the first build used ±0.30 and the racing stripe
 * swallowed half the foredeck, which read as a bare metal panel rather than a
 * stripe.
 */
const DECK_COLS = [-1, -0.62, -0.32, -0.11, 0.11, 0.32, 0.62, 1];
/** Columns inside this fraction get the bright racing stripe. */
const STRIPE_HALF = 0.115;

/**
 * Longitudinal shading zones, as inclusive ranges of *segment* index into
 * `STATION_Z`. Each (strip, zone) pair becomes one flat cel facet. The
 * boundaries are placed where the hull's form genuinely changes — the aft body,
 * the midbody, the flare into the bow — so a shared normal is never a lie about
 * the shape it covers.
 *
 * Segment i spans STATION_Z[i] → STATION_Z[i+1]:
 *   0:-2.32  1:-2.05  2:-1.70  3:-1.32  4:-1.06  5:-0.60
 *   6:-0.10  7: 0.35  8: 0.90  9: 1.35 10: 1.80 11: 2.10 (→2.34)
 */
const FLANK_ZONES: readonly (readonly [number, number])[] = [[0, 5], [6, 8], [9, 11]];
/** Deck zones. `[4, 6]` is exactly the cockpit, which is where the footwell is cut. */
const DECK_ZONES: readonly (readonly [number, number])[] = [[0, 3], [4, 6], [7, 8], [9, 11]];

// ── Cockpit footwell ────────────────────────────────────────────────────────
/**
 * The footwell is a genuine recess, not a tonal plate. The previous build laid
 * flat trim rectangles on the deck 8 mm proud of it, and the review frames read
 * the whole deck — saddle, bars, plates — as "a jumble of undifferentiated
 * grey-blue boxes". Boxes on a plane cannot read as a cockpit; a hole in the
 * plane can, because it has walls, and walls catch a different band of the ramp
 * from the deck they are cut into.
 *
 * Span is quoted in deck-column fractions so the well follows the hull's taper
 * instead of going square amidships and hanging over the sheer forward.
 */
const WELL_COLS: readonly [number, number] = [0.32, 0.62];
const WELL_Z0 = -1.06;
const WELL_Z1 = 0.35;
/** Depth of the well floor below the local deck, metres. */
const WELL_DROP = 0.19;

function isWellColumn(f0: number, f1: number): boolean {
  const a = Math.min(Math.abs(f0), Math.abs(f1));
  const b = Math.max(Math.abs(f0), Math.abs(f1));
  return (
    Math.sign(f0) === Math.sign(f1) &&
    Math.abs(a - WELL_COLS[0]) < 1e-4 &&
    Math.abs(b - WELL_COLS[1]) < 1e-4
  );
}

const sheerHalf = (st: Station) => st.w * 0.985;

/**
 * Hull cross-section, port sheer → keel → starboard sheer. Eleven points:
 * the extra pair either side of the chine is the **spray rail**, a hard step
 * that both throws water outward in reality and gives us a crisp horizontal
 * crease to hang the waterline stripe off.
 */
function hullRing(st: Station): V3[] {
  const w = st.w;
  const sw = sheerHalf(st);
  const keelMid = st.keel + (st.chine - st.keel) * 0.34;
  const band = st.chine + (st.sheer - st.chine) * 0.32;
  const z = st.z;
  return [
    [-sw, st.sheer, z],
    [-w * 0.995, band, z],
    [-w, st.chine + 0.022, z],
    [-w * 0.82, st.chine, z],
    [-w * 0.42, keelMid, z],
    [0, st.keel, z],
    [w * 0.42, keelMid, z],
    [w * 0.82, st.chine, z],
    [w, st.chine + 0.022, z],
    [w * 0.995, band, z],
    [sw, st.sheer, z],
  ];
}

function deckPoint(st: Station, f: number): V3 {
  const sw = sheerHalf(st);
  return [f * sw, st.sheer + DECK_CROWN * (1 - f * f), st.z];
}

// ─────────────────────────────────────────────────────────────────────────────
// Superstructure tables
// ─────────────────────────────────────────────────────────────────────────────

/** Engine hump: the biggest silhouette mass and the boat's identity from behind. */
const COWL: { z: number; hw: number; top: number }[] = [
  { z: -1.04, hw: 0.32, top: 0.66 },
  { z: -1.30, hw: 0.46, top: 0.75 },
  { z: -1.62, hw: 0.47, top: 0.71 },
  { z: -1.92, hw: 0.42, top: 0.56 },
  { z: -2.16, hw: 0.34, top: 0.40 },
];

/** Saddle the rider straddles. Narrow waist so the legs read as legs. */
const SADDLE: { z: number; hw: number; top: number }[] = [
  { z: -1.06, hw: 0.30, top: 0.60 },
  { z: -0.80, hw: 0.335, top: 0.575 },
  { z: -0.42, hw: 0.32, top: 0.545 },
  { z: -0.05, hw: 0.25, top: 0.495 },
];

const HUMP_BASE = 0.22;

/**
 * Rounded-trapezoid cross-section used by both the cowling and the saddle.
 *
 * Nine points, not seven: the extra pair at ±0.30·hw exists purely so the
 * cowling's crown stripe can be a *stripe*. With the crown as one wide segment
 * the bright band covered 74 % of the hump and read as a white block bolted to
 * the engine.
 */
function humpRing(hw: number, top: number, z: number): V3[] {
  const h = top - HUMP_BASE;
  return [
    [-hw, HUMP_BASE, z],
    [-hw * 1.05, HUMP_BASE + h * 0.55, z],
    [-hw * 0.74, top, z],
    [-hw * 0.3, top + 0.016, z],
    [0, top + 0.022, z],
    [hw * 0.3, top + 0.016, z],
    [hw * 0.74, top, z],
    [hw * 1.05, HUMP_BASE + h * 0.55, z],
    [hw, HUMP_BASE, z],
  ];
}
/** Ring segments that carry the crown stripe. Indexes into `humpRing`. */
const CROWN_SEGMENTS = [3, 4];

// ─────────────────────────────────────────────────────────────────────────────
// Seat contract for the rider subsystem
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Where a rider's **hips** go, in boat-local space. The rider subsystem should
 * parent its rig root here (or read `boatMesh.seat`, an `Object3D` named
 * `"seat"` living under the boat's mesh group) rather than hard-coding an
 * offset, so re-proportioning the hull cannot silently sink the rider into the
 * deck.
 *
 * Derived, not guessed: the saddle's top surface runs 0.60 → 0.50 m over
 * z = -1.06 → -0.05, so hips at y = 0.615 puts the seat pad just under them
 * with the pelvis clear of the coaming.
 */
export const SEAT_LOCAL = new Vector3(0, 0.615, -0.46);

/** Where the rider's hands want to be — the handlebar grip centre, boat-local. */
export const GRIP_LOCAL = new Vector3(0, 0.775, -0.01);
/** Half-distance between the two grips. */
export const GRIP_HALF_WIDTH = 0.31;

export interface BoatMesh {
  group: Group;
  /** Named `"seat"`; parent a rider rig here. */
  seat: Object3D;
  /** Named `"grip"`; hands reach for this. */
  grip: Object3D;
  materials: CelMaterialSet[];
  triangles: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Cel material set
// ─────────────────────────────────────────────────────────────────────────────

/**
 * `createCelMaterial` multiplies the ramp texture into the base colour, so the
 * ramp must be a *luminance* ladder or the hull hue gets squared and a
 * vermilion boat turns black in shadow (which is exactly what the first capture
 * of this scene showed). We build the ladder from palette hues — cool for the
 * shadow steps, warm for the lit steps — normalised so its brightest channel is
 * 1.0, then scaled. That keeps the shadow side reading as *sea bounce on paint*
 * rather than as dimmer paint.
 */
function normalisedHue(c: Color): Color {
  const m = Math.max(c.r, c.g, c.b) || 1;
  return c.clone().multiplyScalar(1 / m);
}
const NEUTRAL = normalisedHue(PAL.hudPaper);
/**
 * The hues are pulled most of the way back to neutral. Full-strength
 * `normalisedHue(PAL.skyMid)` is (0.07, 0.40, 1.00) and multiplying paint by
 * that does not read as "in shadow", it reads as "someone turned the red
 * channel off": the first capture had a black stripe down the shaded side of
 * the hull and a *yellow* racing stripe, because the warm hue was equally
 * extreme. A multiplicative ramp needs mostly-neutral steps with a hue *lean*.
 */
const COOL = normalisedHue(PAL.skyMid).lerp(NEUTRAL, 0.62);
const WARM = normalisedHue(PAL.sun).lerp(NEUTRAL, 0.74);

function paintRamp(): Color[] {
  return [
    COOL.clone().multiplyScalar(0.44),
    COOL.clone().multiplyScalar(0.68),
    WARM.clone().multiplyScalar(0.92),
    WARM.clone().multiplyScalar(1.0),
  ];
}
/** Terminator pushed late and hard — a wide lit band, a narrow decisive shadow. */
const PAINT_STOPS = [0.0, 0.4, 0.56, 0.82];

function makeMaterials(id: number) {
  const hullColor = RACER_COLORS[id].hull;
  const hull = createCelMaterial({
    name: `hull${id}`,
    color: hullColor,
    rampColors: paintRamp(),
    rampStops: PAINT_STOPS,
    // A glint, not a wash. The default 0.86 turned the whole boat white; 0.945
    // at strength 0.3 still washed every up-facing plane — the foredeck came out
    // pale pink and read as a bare panel rather than as paint. The spec band on
    // a cel surface has to be small enough to be a *shape*.
    specSize: 0.978,
    specSize2: 0.994,
    specStrength: 0.14,
    rimColor: PAL.skyHorizon,
    rimPower: 3.4,
    rimStrength: 0.5,
    outlineWidthPx: 5.0,
  });
  const trim = createCelMaterial({
    name: `trim${id}`,
    color: PAL.inkSoft,
    // Graphite parts need a brighter ladder or they collapse into the outline.
    rampColors: [
      COOL.clone().multiplyScalar(1.6),
      COOL.clone().multiplyScalar(2.8),
      WARM.clone().multiplyScalar(4.2),
      WARM.clone().multiplyScalar(5.2),
    ],
    rampStops: PAINT_STOPS,
    specSize: 0.965,
    specSize2: 0.992,
    specStrength: 0.34,
    rimColor: PAL.skyHorizon,
    rimPower: 2.6,
    rimStrength: 0.9,
    outlineWidthPx: 3.4,
  });
  const bright = createCelMaterial({
    name: `bright${id}`,
    color: PAL.hudPaper,
    rampColors: [
      COOL.clone().multiplyScalar(0.56),
      COOL.clone().multiplyScalar(0.76),
      WARM.clone().multiplyScalar(0.94),
      WARM.clone().multiplyScalar(1.0),
    ],
    rampStops: PAINT_STOPS,
    specSize: 0.975,
    specSize2: 0.993,
    specStrength: 0.1,
    rimColor: PAL.waterShallow,
    rimPower: 3.0,
    rimStrength: 0.4,
    outlineWidthPx: 3.4,
  });
  /**
   * Windscreen glazing. Its own material for two reasons that the previous
   * build's single pale slab proves: `hudPaper` is the *foam* white, so a screen
   * painted in it is the same value as the spray around the hull and reads as a
   * blank card; and glass wants a two-band ladder, not four, because the whole
   * point of a cel window is that it is two flat values and a streak.
   *
   * Two `rampColors` with the stops pushed apart give exactly one hard step
   * across the pane. The tint is `skyHorizon` — glass in this palette is a piece
   * of sky, which is also why it never competes with the hull red.
   */
  const glass = createCelMaterial({
    name: `glass${id}`,
    color: PAL.skyHorizon,
    rampColors: [COOL.clone().multiplyScalar(0.62), WARM.clone().multiplyScalar(1.06)],
    rampStops: [0.0, 0.5],
    specSize: 0.955,
    specSize2: 0.988,
    specStrength: 0.4,
    rimColor: PAL.foam,
    rimPower: 2.2,
    rimStrength: 0.85,
    outlineWidthPx: 3.4,
  });
  return { hull, trim, bright, glass };
}

// ─────────────────────────────────────────────────────────────────────────────
// Assembly
// ─────────────────────────────────────────────────────────────────────────────

export function createBoatMesh(id: number): BoatMesh {
  const HULL = new Surface();
  const TRIM = new Surface();
  const BRIGHT = new Surface();
  const GLASS = new Surface();

  const stations = STATION_Z.map(stationAt);
  const rings = stations.map(hullRing);
  const nStations = stations.length;

  // ── Hull shell ────────────────────────────────────────────────────────────
  // Ten longitudinal strips per segment. Strips 1 and 8 are the low side band
  // just above the spray rail; they go to the bright material and become a
  // waterline stripe that runs the full length of the boat. That single stripe
  // does more for reading the hull's sheer line than any amount of shading.
  //
  // Each strip is shaded as three long facets (aft body, midbody, bow) rather
  // than as twelve independent segments. See the facet note at the top of the
  // file: twelve normals per strake put a dashed 45° contour down the flank the
  // moment a ramp stop crossed it.
  for (let j = 0; j < 10; j++) {
    const target = j === 1 || j === 8 ? BRIGHT : HULL;
    for (const [z0, z1] of FLANK_ZONES) {
      target.beginFacet();
      for (let i = z0; i <= z1; i++) {
        const A = rings[i];
        const B = rings[i + 1];
        target.quad(A[j], A[j + 1], B[j + 1], B[j]);
      }
      target.endFacet();
    }
  }

  // Transom: a closed polygon running sheer → keel → sheer and then back across
  // the top along the *crowned deck edge*.
  //
  // It used to close with a straight chord between the two sheer points, which
  // is 5.5 cm below the deck's own aft edge at the centreline (DECK_CROWN). That
  // left a 1.74 m × 5.5 cm lens-shaped hole at the stern, and what showed
  // through it was the *inside* of the hull shell: the stray 1-px bright
  // orange-red line running across the stern in shots/boat_b0/hero.png, visible
  // at 8× in the skeg crop. Following the deck edge closes it.
  {
    const st = stations[0];
    const r = rings[0];
    const top: V3[] = [];
    for (let k = DECK_COLS.length - 2; k >= 1; k--) {
      const d = deckPoint(st, DECK_COLS[k]);
      top.push([d[0], d[1], st.z]);
    }
    const loop = [...r, ...top];
    HULL.beginFacet();
    HULL.cap(loop, [0, (st.keel + st.sheer) * 0.5, st.z], [0, 0, -1], true);
    HULL.endFacet();
  }
  // Bow: the last station is nearly a point, so this is a sliver.
  {
    const st = stations[nStations - 1];
    const r = rings[nStations - 1];
    HULL.beginFacet();
    HULL.cap(r, [0, (st.keel + st.sheer) * 0.5, st.z], [0, 0, 1], true);
    HULL.endFacet();
  }

  // ── Deck ──────────────────────────────────────────────────────────────────
  // Seven columns, four longitudinal zones — 28 flat deck panels rather than 168
  // triangle slivers. The two central columns forward of the cockpit are bright:
  // that is the racing stripe, and it is the single strongest cue that tells you
  // which way a boat 60 m away is pointing.
  //
  // The two outboard-but-not-outermost columns are *omitted* over the cockpit
  // zone; that span is the footwell, built below.
  const STRIPE_FROM_Z = 0.3;
  for (let j = 0; j < DECK_COLS.length - 1; j++) {
    const f0 = DECK_COLS[j];
    const f1 = DECK_COLS[j + 1];
    const central = Math.abs(f0) <= STRIPE_HALF && Math.abs(f1) <= STRIPE_HALF;
    const wellCol = isWellColumn(f0, f1);
    for (const [z0, z1] of DECK_ZONES) {
      const stripe = stations[z0].z >= STRIPE_FROM_Z;
      const target = stripe && central ? BRIGHT : HULL;
      target.beginFacet();
      for (let i = z0; i <= z1; i++) {
        const sa = stations[i];
        const sb = stations[i + 1];
        if (wellCol && sa.z >= WELL_Z0 - 1e-4 && sb.z <= WELL_Z1 + 1e-4) continue;
        target.quad(deckPoint(sa, f0), deckPoint(sb, f0), deckPoint(sb, f1), deckPoint(sa, f1));
      }
      target.endFacet();
    }
  }

  // ── Gunwale rail ──────────────────────────────────────────────────────────
  // A 7 cm bead following the sheer from transom to bow, in the dark trim. It
  // reads as a drawn line at any distance and it thickens the silhouette edge,
  // which is what stops the boat looking like folded paper.
  for (const sign of [-1, 1]) {
    // One facet per (side, zone, face). The rail's inner face is the worst
    // offender in the file for the stipple — it is defined against `deckPoint`,
    // so every one of its quads is non-planar, and the review frames caught the
    // hatch on "the gunwale insides" specifically.
    for (const [z0, z1] of FLANK_ZONES) {
      for (let face = 0; face < 3; face++) {
        TRIM.beginFacet();
        for (let i = z0; i <= z1; i++) {
          const sa = stations[i];
          const sb = stations[i + 1];
          const swa = sheerHalf(sa);
          const swb = sheerHalf(sb);
          const ta = sa.sheer + RAIL_H;
          const tb = sb.sheer + RAIL_H;
          const wa = Math.min(RAIL_W, swa * 0.5);
          const wb = Math.min(RAIL_W, swb * 0.5);
          if (face === 0) {
            TRIM.quadMirrored(sign, [swa, sa.sheer, sa.z], [swa, ta, sa.z], [swb, tb, sb.z], [swb, sb.sheer, sb.z]);
          } else if (face === 1) {
            TRIM.quadMirrored(sign, [swa, ta, sa.z], [swa - wa, ta, sa.z], [swb - wb, tb, sb.z], [swb, tb, sb.z]);
          } else {
            TRIM.quadMirrored(
              sign,
              [swa - wa, ta, sa.z],
              [swa - wa, deckPoint(sa, (swa - wa) / swa)[1], sa.z],
              [swb - wb, deckPoint(sb, (swb - wb) / swb)[1], sb.z],
              [swb - wb, tb, sb.z],
            );
          }
        }
        TRIM.endFacet();
      }
    }
    // Close the rail's ends so the inverted hull has no aperture.
    for (const idx of [0, nStations - 1]) {
      const st = stations[idx];
      const sw = sheerHalf(st);
      const w = Math.min(RAIL_W, sw * 0.5);
      const t = st.sheer + RAIL_H;
      const inner = deckPoint(st, (sw - w) / sw)[1];
      const zFace = idx === 0 ? st.z - 0.001 : st.z + 0.001;
      TRIM.quadMirrored(
        sign,
        [sw, st.sheer, zFace],
        [sw, t, zFace],
        [sw - w, t, zFace],
        [sw - w, inner, zFace],
      );
    }
  }

  // ── Engine cowling ────────────────────────────────────────────────────────
  {
    const cowlRings = COWL.map((c) => humpRing(c.hw, c.top, c.z));
    // One facet per longitudinal panel: the cowling is eight flat planes with
    // eight hard creases, which is what a cel image can actually draw.
    for (let j = 0; j < cowlRings[0].length - 1; j++) {
      const target = CROWN_SEGMENTS.includes(j) ? BRIGHT : HULL;
      target.beginFacet();
      for (let i = 0; i < cowlRings.length - 1; i++) {
        const A = cowlRings[i];
        const B = cowlRings[i + 1];
        // Segments 2 and 3 are the crown. Running the racing stripe over them
        // gives the boat a readable graphic from directly behind, which is the
        // angle the player spends most of the race looking at a rival from.
        //
        // COWL is listed forward → aft (z decreasing). `humpRing` runs
        // port-bottom over the crown to starboard-bottom, and with the ring pair
        // going aft the default winding is already outboard. Getting this
        // backwards is not a subtle bug: the shell vanishes behind backface
        // culling and the *outline* hull shows through instead, so the cowling
        // renders as a solid ink blob — which is exactly how it looked in the
        // first capture of this file.
        target.quad(A[j], A[j + 1], B[j + 1], B[j]);
      }
      target.endFacet();
    }
    const front = COWL[0];
    const back = COWL[COWL.length - 1];
    HULL.beginFacet();
    HULL.cap(cowlRings[0], [0, (HUMP_BASE + front.top) * 0.5, front.z], [0, 0, 1], true);
    HULL.endFacet();
    HULL.beginFacet();
    HULL.cap(
      cowlRings[cowlRings.length - 1],
      [0, (HUMP_BASE + back.top) * 0.5, back.z],
      [0, 0, -1],
      true,
    );
    HULL.endFacet();
    // Floor, so the shell is closed where it meets the deck.
    HULL.quad(
      [-front.hw, HUMP_BASE, front.z],
      [-back.hw, HUMP_BASE, back.z],
      [back.hw, HUMP_BASE, back.z],
      [front.hw, HUMP_BASE, front.z],
    );

    // Air intake: a forward-facing scoop on the crown, plus a lip standing a
    // couple of centimetres proud of its forward face.
    //
    // `box` is (x0, x1, y0, y1, z0, z1) — it is NOT `taperBox`'s z-first order.
    // The lip was originally written as `box(-1.235, -1.19, -0.15, 0.15, 0.6,
    // 0.82)`, i.e. with the intended z pair in the x slots, which put a
    // 0.045 × 0.30 × 0.22 m slab at x = -1.21: an orphan grey blade floating
    // 1.2 m off the port beam, at the waterline, complete with its own ink
    // outline. It appeared in five of fifteen review frames, once in mid-air.
    TRIM.taperBox(-1.66, -1.24, [-0.2, 0.2, 0.7, 0.795], [-0.155, 0.155, 0.66, 0.83]);
    TRIM.box(-0.175, 0.175, 0.645, 0.85, -1.29, -1.235);
  }

  // ── Saddle ────────────────────────────────────────────────────────────────
  {
    const saddleRings = SADDLE.map((c) => humpRing(c.hw, c.top, c.z));
    for (let j = 0; j < saddleRings[0].length - 1; j++) {
      TRIM.beginFacet();
      for (let i = 0; i < saddleRings.length - 1; i++) {
        const A = saddleRings[i];
        const B = saddleRings[i + 1];
        TRIM.quad(A[j], B[j], B[j + 1], A[j + 1]);
      }
      TRIM.endFacet();
    }
    const back = SADDLE[0];
    const front = SADDLE[SADDLE.length - 1];
    TRIM.beginFacet();
    TRIM.cap(saddleRings[0], [0, (HUMP_BASE + back.top) * 0.5, back.z], [0, 0, -1], true);
    TRIM.endFacet();
    TRIM.beginFacet();
    TRIM.cap(
      saddleRings[saddleRings.length - 1],
      [0, (HUMP_BASE + front.top) * 0.5, front.z],
      [0, 0, 1],
      true,
    );
    TRIM.endFacet();
    TRIM.quad(
      [-back.hw, HUMP_BASE, back.z],
      [back.hw, HUMP_BASE, back.z],
      [front.hw, HUMP_BASE, front.z],
      [-front.hw, HUMP_BASE, front.z],
    );
    // Seat pad — bright, so the rider reads as sitting *on* something.
    BRIGHT.taperBox(-1.0, -0.2, [-0.245, 0.245, 0.575, 0.605], [-0.205, 0.205, 0.52, 0.55]);
  }

  // ── Cockpit footwell ──────────────────────────────────────────────────────
  // A genuine recess: floor, two long walls, two end walls. It replaces two flat
  // trim plates lifted 8 mm off the deck, which read as neither a footwell nor
  // anything else — and whose 8 mm lip caught the sun as a 1-px bright line
  // along the deck in the review frames.
  //
  // Each wall is one facet, so from any angle the cockpit resolves into four
  // large flat tones stepping down from the deck. That step *is* the drawing.
  {
    const wellStations: Station[] = [];
    for (const st of stations) if (st.z >= WELL_Z0 - 1e-4 && st.z <= WELL_Z1 + 1e-4) wellStations.push(st);
    const [fIn, fOut] = WELL_COLS;
    const floorOf = (st: Station, f: number): V3 => {
      const d = deckPoint(st, f);
      return [d[0], d[1] - WELL_DROP, d[2]];
    };
    for (const sign of [-1, 1]) {
      // Floor.
      TRIM.beginFacet();
      for (let i = 0; i < wellStations.length - 1; i++) {
        const sa = wellStations[i];
        const sb = wellStations[i + 1];
        TRIM.quadMirrored(sign, floorOf(sa, fIn), floorOf(sb, fIn), floorOf(sb, fOut), floorOf(sa, fOut));
      }
      TRIM.endFacet();
      // Outboard wall, rising to the deck. Wound so it faces inboard.
      TRIM.beginFacet();
      for (let i = 0; i < wellStations.length - 1; i++) {
        const sa = wellStations[i];
        const sb = wellStations[i + 1];
        TRIM.quadMirrored(
          sign,
          floorOf(sa, fOut),
          floorOf(sb, fOut),
          deckPoint(sb, fOut),
          deckPoint(sa, fOut),
        );
      }
      TRIM.endFacet();
      // Inboard wall — buried in the saddle's flank along most of its length, so
      // it never shows daylight under the hump.
      TRIM.beginFacet();
      for (let i = 0; i < wellStations.length - 1; i++) {
        const sa = wellStations[i];
        const sb = wellStations[i + 1];
        TRIM.quadMirrored(
          sign,
          deckPoint(sa, fIn),
          deckPoint(sb, fIn),
          floorOf(sb, fIn),
          floorOf(sa, fIn),
        );
      }
      TRIM.endFacet();
      // End walls, aft then forward.
      const aft = wellStations[0];
      const fwd = wellStations[wellStations.length - 1];
      TRIM.quadMirrored(sign, floorOf(aft, fIn), deckPoint(aft, fIn), deckPoint(aft, fOut), floorOf(aft, fOut));
      TRIM.quadMirrored(sign, floorOf(fwd, fOut), deckPoint(fwd, fOut), deckPoint(fwd, fIn), floorOf(fwd, fIn));
      // Non-slip pad on the floor — the one bright value inside the well, which
      // is what makes the recess read as depth rather than as a dark decal.
      const padIn = fIn + (fOut - fIn) * 0.18;
      const padOut = fIn + (fOut - fIn) * 0.82;
      const lift = 0.012;
      const pad = (st: Station, f: number): V3 => {
        const p = floorOf(st, f);
        return [p[0], p[1] + lift, p[2]];
      };
      BRIGHT.beginFacet();
      for (let i = 0; i < wellStations.length - 1; i++) {
        const sa = wellStations[i];
        const sb = wellStations[i + 1];
        BRIGHT.quadMirrored(sign, pad(sa, padIn), pad(sb, padIn), pad(sb, padOut), pad(sa, padOut));
      }
      BRIGHT.endFacet();
    }
  }

  // ── Seat back ─────────────────────────────────────────────────────────────
  // A backrest bridging the saddle's aft end and the cowling's forward face. It
  // costs 12 triangles and it is what turns "a hump with a pad on it" into a
  // seat when the boat is viewed from astern — which is the angle the player
  // looks at their own boat from for the entire race.
  TRIM.taperBox(-1.14, -1.0, [-0.28, 0.28, 0.5, 0.88], [-0.25, 0.25, 0.5, 0.84]);
  BRIGHT.taperBox(-1.0, -0.965, [-0.215, 0.215, 0.6, 0.83], [-0.205, 0.205, 0.6, 0.8]);

  // ── Handlebar column and bar ──────────────────────────────────────────────
  // The column leans back off the foredeck to meet the bar, which sits where
  // GRIP_LOCAL says the rider's hands are. A vertical post at the wrong z looks
  // like a bollard and gives the rider nothing to hold.
  TRIM.taperBox(-0.02, 0.34, [-0.055, 0.055, 0.7, 0.79], [-0.085, 0.085, 0.22, 0.3]);
  TRIM.box(-0.35, 0.35, 0.757, 0.793, -0.042, 0.006);
  for (const sign of [-1, 1]) {
    BRIGHT.box(sign * 0.235, sign * 0.352, 0.748, 0.802, -0.058, 0.022);
  }

  // ── Console and windscreen ────────────────────────────────────────────────
  // The screen used to be a single pale slab 0.66 m wide and 0.66 m along its
  // slope, laid on the red foredeck at a 48° rake: measured against the review
  // frames it was the largest untextured area on the boat, it read as a blank
  // card, and from the bow camera it covered the rider's chest.
  //
  // It is now a quarter of that area, stood on a dark console instead of lying
  // on the deck, framed on all four edges in the trim tone, and glazed in its
  // own pale-cyan material with a hard highlight wedge across it. The frame is
  // what gives it an ink contour on every edge — the inverted hull can only ink
  // a silhouette, so an unframed plate seen face-on has no line at all.
  {
    // Console body: a wedge rising out of the foredeck, tallest at its aft face
    // where the screen stands. Its aft face is 0.44 m tall, which is what gives
    // the deck a hard horizon line between "cockpit" and "foredeck".
    TRIM.taperBox(0.30, 0.70, [-0.40, 0.40, 0.18, 0.62], [-0.32, 0.32, 0.24, 0.40]);
    // Instrument bezel let into the console's top — a second value so the
    // console is not one flat block.
    BRIGHT.taperBox(0.37, 0.62, [-0.21, 0.21, 0.60, 0.618], [-0.175, 0.175, 0.44, 0.458]);

    // Screen. Base sits on the console's aft-top edge, leaning back over the
    // bars: 0.24 m of run for 0.27 m of rise, i.e. 48° from horizontal — the
    // same rake as the slab it replaces, at a third of the area.
    const bz = 0.34, by = 0.585; // base, on the console
    const tz = 0.10, ty = 0.855; // top, over the handlebar
    const bw = 0.285, tw = 0.225;
    const T = 0.034; // glass thickness, measured vertically

    // Body: everything but the outer face, which is split between two materials
    // below.
    GLASS.taperBox(tz, bz, [-tw, tw, ty, ty + T], [-bw, bw, by, by + T], true, true, true);

    // The outer face is a genuine plane (its two end edges are parallel lines
    // along X), so any polygon drawn on it is planar and shades as one flat
    // value. Split it on a diagonal: the port side is the glare, the rest is
    // glass. Two flat fills and one hard diagonal — no gradient, and the
    // material change gives the Sobel pass an interior line for free.
    const P = (u: number, v: number): V3 => {
      const w = bw + (tw - bw) * u;
      return [v * w, by + T + (ty - by) * u, bz + (tz - bz) * u];
    };
    const vBase = 0.34;
    const vTop = -0.62;
    BRIGHT.beginFacet();
    BRIGHT.quad(P(0, -1), P(0, vBase), P(1, vTop), P(1, -1));
    BRIGHT.endFacet();
    GLASS.beginFacet();
    GLASS.quad(P(0, vBase), P(0, 1), P(1, 1), P(1, vTop));
    GLASS.endFacet();

    // Frame. Two raked stanchions outboard of the glass plus a cap rail along
    // the top edge: three dark bars that close the screen into a loop, so it has
    // an ink contour on every edge instead of on the two the silhouette happens
    // to give it.
    for (const sign of [-1, 1]) {
      TRIM.taperBox(
        tz,
        bz,
        [sign * tw, sign * (tw + 0.036), ty - 0.008, ty + T + 0.014],
        [sign * bw, sign * (bw + 0.036), by - 0.008, by + T + 0.014],
      );
    }
    TRIM.taperBox(
      tz - 0.042,
      tz + 0.02,
      [-tw - 0.036, tw + 0.036, ty - 0.01, ty + T + 0.016],
      [-tw - 0.036, tw + 0.036, ty - 0.01, ty + T + 0.016],
    );
  }

  // ── Wing and tail fin ─────────────────────────────────────────────────────
  // Two attempts got binned here. A wing as wide as the beam read as a swim
  // platform; adding pale end plates turned it into a grey table with white
  // legs. What actually works is a narrow dark wing plus a *vertical* fin in the
  // racer's own colour: the fin adds 25 cm of height to the silhouette without
  // adding any width, and it is the single element that survives at 100 m.
  //
  // The whole assembly is a chain — deck → pylon → wing → fin — and every link
  // has to physically interpenetrate the next one. It did not: the pylons
  // stopped 11 cm above the deck and the fin overhung the wing's leading edge
  // by 14 cm with nothing but air below it (the cowling crown is only at
  // y ≈ 0.51 that far aft). The review frames called it, correctly, a grey/red
  // T-frame floating above the transom.
  // Wing: z -2.44 → -2.14. Every number below is quoted against this span.
  TRIM.taperBox(-2.44, -2.14, [-0.5, 0.5, 0.805, 0.85], [-0.45, 0.45, 0.835, 0.88]);
  // Fin: lives entirely inside the wing's footprint, and its underside is below
  // the wing's upper surface at both ends, so it is buried in the plate rather
  // than balanced on it.
  HULL.taperBox(-2.42, -2.16, [-0.05, 0.05, 0.82, 1.14], [-0.042, 0.042, 0.84, 0.99]);
  for (const sign of [-1, 1]) {
    // Pylons. Bottom at y = 0.16 is ~9 cm *inside* the deck (whose crown is at
    // y ≈ 0.25 this far aft) and the top is inside the wing's thickness, so the
    // silhouette closes with no aperture for the outline hull to leak through.
    const sx0 = sign * 0.2;
    const sx1 = sign * 0.28;
    TRIM.taperBox(-2.3, -2.16, [sx0, sx1, 0.16, 0.845], [sx0, sx1, 0.18, 0.862]);
  }

  // ── Nose spike ────────────────────────────────────────────────────────────
  // Must straddle the bow's sheer/keel band (0.34 → 0.64 at z = 2.34) or it
  // sits *inside* the hull and contributes nothing to the silhouette.
  TRIM.taperBox(2.2, 2.54, [-0.085, 0.085, 0.3, 0.66], [-0.02, 0.02, 0.47, 0.51]);

  // ── Transom ───────────────────────────────────────────────────────────────
  // The stern wall used to be, from astern: a thin strip of red, a full-beam
  // bright bar, another thin strip of red. The bright bar was the boarding step,
  // 1.24 m wide and standing 10 cm proud of the transom, and it swallowed the
  // whole stern face. That is why the review frames reported "the transom is
  // absent — you look straight through into the interior floor".
  //
  // It is now a composed stern: a dark plate over the lower two-thirds, a
  // narrower bright step let into it, and a bright accent band under the deck
  // line. Four values plus ink, in a stack, in the one place the player looks at
  // for the whole race.
  {
    const st = stations[0];
    const sw = sheerHalf(st);
    // Dark plate covering the lower transom. Proud by 2 cm so its edge silhouette
    // separates from the red shell.
    TRIM.taperBox(-2.345, -2.28, [-sw * 0.9, sw * 0.9, -0.2, 0.055], [-sw * 0.94, sw * 0.94, -0.21, 0.06]);
    // Boarding step: narrower than the transom, so red shell reads either side.
    BRIGHT.box(-0.4, 0.4, -0.055, 0.06, -2.4, -2.3);
    // Accent band immediately under the deck edge, full beam. This is the line
    // that says "stern" at 60 m.
    BRIGHT.taperBox(-2.35, -2.3, [-sw * 0.86, sw * 0.86, 0.12, 0.175], [-sw * 0.9, sw * 0.9, 0.115, 0.18]);
  }

  // ── Drive leg ─────────────────────────────────────────────────────────────
  // Previously a 9.6 cm blade and a flat plate: the review frames called it "a
  // plain grey cross". It is now a real outboard leg — strut, gearcase pod, prop
  // boss — so the one part of the boat that touches water has enough mass to
  // carry an outline and enough value change to read as a mechanism.
  //
  // The cel outline is an inverted hull pushed a constant number of *screen*
  // pixels along the smoothed normal. On a 9.6 cm blade seen edge-on there is
  // barely any silhouette for it to sit outside of, which is why the line came
  // and went along its length. 16 cm fixes that without changing the profile the
  // buoyancy contact table quotes (skeg tip stays at y = -0.60, z = -2.24).
  TRIM.taperBox(-2.36, -1.66, [-0.08, 0.08, -0.6, -0.18], [-0.055, 0.055, -0.3, -0.2]);
  // Gearcase pod: a horizontal torpedo at the foot of the strut.
  TRIM.taperBox(-2.5, -2.02, [-0.075, 0.075, -0.52, -0.35], [-0.105, 0.105, -0.56, -0.3]);
  // Prop boss, in the racer's own hull colour — a small warm accent at the very
  // stern that tells you where the thrust comes from.
  HULL.taperBox(-2.58, -2.48, [-0.035, 0.035, -0.47, -0.4], [-0.075, 0.075, -0.51, -0.36]);
  // Cavitation plate, straddling the leg above the pod. Bright, so the leg has a
  // horizontal value break across it instead of being one grey mass.
  BRIGHT.box(-0.26, 0.26, -0.295, -0.25, -2.44, -2.08);
  for (const sign of [-1, 1]) {
    TRIM.box(sign * 0.2, sign * 0.34, 0.115, 0.235, -2.44, -2.3);
  }

  // ── Bake ──────────────────────────────────────────────────────────────────
  const mats = makeMaterials(id);
  const group = new Group();
  group.name = `boatMesh${id}`;

  const parts: [Surface, CelMaterialSet, string][] = [
    [HULL, mats.hull, 'hullShell'],
    [TRIM, mats.trim, 'hullTrim'],
    [BRIGHT, mats.bright, 'hullBright'],
    [GLASS, mats.glass, 'hullGlass'],
  ];
  let triangles = 0;
  for (const [surf, set, name] of parts) {
    const mesh = new Mesh(surf.geometry(`${name}${id}`));
    mesh.name = `${name}${id}`;
    applyCel(mesh, set);
    group.add(mesh);
    triangles += surf.triangleCount;
  }

  const seat = new Object3D();
  seat.name = 'seat';
  seat.position.copy(SEAT_LOCAL);
  group.add(seat);

  const grip = new Object3D();
  grip.name = 'grip';
  grip.position.copy(GRIP_LOCAL);
  group.add(grip);

  return {
    group,
    seat,
    grip,
    materials: [mats.hull, mats.trim, mats.bright, mats.glass],
    triangles,
  };
}
