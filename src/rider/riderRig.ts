/**
 * INK TIDE — rider rig: skeleton + procedural cel geometry.
 *
 * ── Why a hand-rolled skin instead of THREE.SkinnedMesh ─────────────────────
 * The cel pipeline owns the vertex stage: `createCelMaterial` builds the main,
 * prepass and outline materials from *one* set of vertex chunks, and the
 * inverted-hull outline pushes along `aSmoothNormal` inside that same stage.
 * Three's skinning lives in `MeshStandardMaterial`-style shader chunks we do
 * not have (and must not add — no PBR). So the skinning is a `chunks.vertexBody`
 * snippet: two bone matrices per vertex, a scalar blend weight, applied to
 * `transformed`, `objectNormal` *and* `smoothNormal`. Because all three
 * materials share the chunk, the outline hull and the G-buffer deform with the
 * pose for free — which is the whole reason the chunk hook exists.
 *
 * ── Why one merged geometry per rider ───────────────────────────────────────
 * The perf contract says a rider is ~12 parts and must not cost 12 draw calls.
 * Every part of a rider is emitted into one of *two* merged BufferGeometries —
 * `soft` (suit, skin, gloves, scarf) and `hard` (helmet, visor, pads, boots) —
 * split only because those two families want different specular/rim treatment.
 * Part colour is carried per-vertex in a custom `aTint` attribute and multiplied
 * into `baseColor` in `chunks.fragmentBody`, so one material paints a dozen
 * palette tones. That is 2 shaded meshes + 2 outline hulls = 4 draw calls per
 * rider.
 *
 * (`aTint` rather than the material's `vertexColors` option on purpose: the
 * prepass material is built without `vertexColors`, so the `USE_VERTEX_COLORS`
 * define would reference an undeclared `color` attribute there and fail to
 * compile. A chunk-declared attribute is shared correctly by all three.)
 */

import {
  BufferAttribute,
  BufferGeometry,
  Color,
  Group,
  Matrix3,
  Matrix4,
  Mesh,
  Quaternion,
  Sphere,
  Vector3,
} from 'three';
import { PAL, RACER_COLORS } from '../core/palette';
import { applyCel, createCelMaterial, paletteTone, type CelMaterialSet } from '../render/celMaterial';
// The hull publishes its seat and handlebar geometry for exactly this purpose —
// see the "Seat contract for the rider subsystem" block in boatMesh.ts. Reading
// it beats hard-coding an offset that a hull re-proportion would silently break.
import { GRIP_HALF_WIDTH, GRIP_LOCAL, SEAT_LOCAL } from '../boat/boatMesh';

// ─────────────────────────────────────────────────────────────────────────────
// Bones
// ─────────────────────────────────────────────────────────────────────────────

export const B = {
  hips: 0,
  spine: 1,
  chest: 2,
  neck: 3,
  head: 4,
  clavL: 5,
  upArmL: 6,
  loArmL: 7,
  handL: 8,
  clavR: 9,
  upArmR: 10,
  loArmR: 11,
  handR: 12,
  thighL: 13,
  shinL: 14,
  footL: 15,
  thighR: 16,
  shinR: 17,
  footR: 18,
  scarfA: 19,
  scarfB: 20,
  scarfC: 21,
} as const;

export const BONE_COUNT = 22;

/** Per-racer physique. Same rig, different animal. */
export interface RiderBuild {
  /** Vertical scale on the whole skeleton. */
  height: number;
  /** Limb + torso radius multiplier. */
  girth: number;
  armLen: number;
  legLen: number;
  headSize: number;
  /** Baseline forward pitch of the spine, radians. Posture, not animation. */
  hunch: number;
  /** Shoulders raised toward the ears — reads as tension. */
  shrug: number;
  shoulderPad: 'none' | 'left' | 'both';
  crest: 'none' | 'fin' | 'mohawk';
  /** 0 = no scarf. Otherwise a length multiplier. */
  scarf: number;
  /** Animation rate multiplier — a twitchy rider vs a smooth one. */
  tempo: number;
  /** Phase offset so the four riders never bob in lockstep. */
  phase: number;
  /**
   * Constant posture asymmetry, -1…1. Signed, and *not* animated: it drops one
   * shoulder, cocks the head and cants the spine by a couple of degrees for the
   * whole race. Phase offsets alone are not enough — four riders running the
   * same cycle out of phase still read as one puppet at pack distance, because
   * every frame they all pass through the same neutral shape. A standing bias
   * means they never share a silhouette at all.
   */
  bias: number;
}

export const RIDER_BUILDS: RiderBuild[] = [
  // 0 — player. Compact, neutral, textbook racing crouch.
  {
    height: 1.12, girth: 1.02, armLen: 1.0, legLen: 1.0, headSize: 1.08,
    hunch: 0.0, shrug: 0.0, shoulderPad: 'left', crest: 'fin',
    scarf: 1.0, tempo: 1.0, phase: 0.0, bias: 0.25,
  },
  // 1 — KAIRA. Tall, long-limbed, upright and loose.
  {
    height: 1.2, girth: 0.92, armLen: 1.1, legLen: 1.06, headSize: 1.02,
    hunch: -0.12, shrug: -0.06, shoulderPad: 'none', crest: 'mohawk',
    scarf: 1.35, tempo: 0.88, phase: 1.9, bias: -1.0,
  },
  // 2 — NOX. Heavy, hunched over the bars, shoulders up.
  {
    height: 1.07, girth: 1.22, armLen: 0.94, legLen: 0.94, headSize: 1.14,
    hunch: 0.26, shrug: 0.14, shoulderPad: 'both', crest: 'none',
    scarf: 0.0, tempo: 1.12, phase: 3.6, bias: 0.7,
  },
  // 3 — PIP. Small, springy, very fast timing.
  {
    height: 1.01, girth: 0.96, armLen: 0.96, legLen: 0.9, headSize: 1.24,
    hunch: 0.07, shrug: 0.04, shoulderPad: 'left', crest: 'fin',
    scarf: 1.15, tempo: 1.3, phase: 5.1, bias: -0.55,
  },
];

interface BoneDef {
  parent: number;
  head: Vector3;
  tip: Vector3;
  /** Reference axis used to fix the bind twist; 'z' for bones that run roughly
   *  vertically, 'y' for bones that run roughly horizontally. Picking the wrong
   *  one leaves the basis near-degenerate and boxy parts twist randomly. */
  hint: 'y' | 'z';
}

const V = (x: number, y: number, z: number) => new Vector3(x, y, z);

/**
 * Bind skeleton, authored as joint *positions* in rider space, which is the
 * boat's `"seat"` node: **origin at the hips**, +Y up, +Z forward. That is the
 * contract `boatMesh.ts` publishes (`SEAT_LOCAL` is documented as the hips
 * position), so parenting the rig root straight onto the seat with no offset is
 * correct and survives the hull being re-proportioned.
 *
 * The pose is a jet-racer crouch: knees folded up beside the saddle hump, torso
 * pitched forward, hands down and out on the bars.
 *
 * Note what `height` does and does not scale. The seat and the footwell are
 * fixed by the hull, so the *leg* chain keeps its vertical reach for every
 * rider — a short rider is short above the waist, not floating above the deck.
 */
function boneDefs(b: RiderBuild): BoneDef[] {
  const H = b.height;
  const A = b.armLen;
  const L = b.legLen;
  const g = b.girth;
  const hy = (y: number) => y * H;

  // Torso chain.
  const hips = V(0, 0, 0);
  const spine = V(0, hy(0.16), 0.03);
  const chest = V(0, hy(0.36), 0.09);
  const neck = V(0, hy(0.52), 0.13);
  const headBase = V(0, hy(0.6), 0.14);
  const headTip = V(0, hy(0.6) + 0.22 * b.headSize, 0.155);

  // Arm chain, left side; mirrored below. Reach scales with `armLen` about the
  // shoulder; the IK then puts the wrist on the bar whatever the reach is.
  const shoulder = V(-0.185 * g, hy(0.455), 0.125);
  const clav = V(-0.05, hy(0.485), 0.115);
  const elbow = V(shoulder.x - 0.15 * A * g, shoulder.y - 0.14 * A * H, shoulder.z + 0.13 * A);
  const wrist = V(elbow.x + 0.025 * A * g, elbow.y - 0.16 * A * H, elbow.z + 0.195 * A);
  const handTip = V(wrist.x, wrist.y - 0.022, wrist.z + 0.095);

  // Leg chain, left side. Ankle height is deliberately independent of `height`.
  const hipJ = V(-0.115 * g, -0.02, 0.01);
  const knee = V(-0.215 * g, -0.185, 0.2 + 0.07 * L);
  const ankle = V(-0.235 * g, -0.415, 0.08);
  const toe = V(-0.24 * g, -0.465, 0.2);

  const mir = (v: Vector3) => V(-v.x, v.y, v.z);

  // Scarf: authored already arced, each segment dropping further than the last,
  // so even the rest pose has cloth curvature instead of a straight spar.
  const s = b.scarf > 0 ? b.scarf : 1;
  const sA = V(0, hy(0.5), 0.05);
  const sB = V(0, hy(0.46) - 0.015 * s, 0.05 - 0.115 * s);
  const sC = V(0.012, hy(0.4) - 0.04 * s, 0.05 - 0.215 * s);
  const sT = V(0.03, hy(0.31) - 0.08 * s, 0.05 - 0.295 * s);

  const defs: BoneDef[] = [];
  defs[B.hips] = { parent: -1, head: hips, tip: spine, hint: 'z' };
  defs[B.spine] = { parent: B.hips, head: spine, tip: chest, hint: 'z' };
  defs[B.chest] = { parent: B.spine, head: chest, tip: neck, hint: 'z' };
  defs[B.neck] = { parent: B.chest, head: neck, tip: headBase, hint: 'z' };
  defs[B.head] = { parent: B.neck, head: headBase, tip: headTip, hint: 'z' };

  defs[B.clavL] = { parent: B.chest, head: clav, tip: shoulder, hint: 'z' };
  defs[B.upArmL] = { parent: B.clavL, head: shoulder, tip: elbow, hint: 'z' };
  defs[B.loArmL] = { parent: B.upArmL, head: elbow, tip: wrist, hint: 'y' };
  defs[B.handL] = { parent: B.loArmL, head: wrist, tip: handTip, hint: 'y' };

  defs[B.clavR] = { parent: B.chest, head: mir(clav), tip: mir(shoulder), hint: 'z' };
  defs[B.upArmR] = { parent: B.clavR, head: mir(shoulder), tip: mir(elbow), hint: 'z' };
  defs[B.loArmR] = { parent: B.upArmR, head: mir(elbow), tip: mir(wrist), hint: 'y' };
  defs[B.handR] = { parent: B.loArmR, head: mir(wrist), tip: mir(handTip), hint: 'y' };

  defs[B.thighL] = { parent: B.hips, head: hipJ, tip: knee, hint: 'z' };
  defs[B.shinL] = { parent: B.thighL, head: knee, tip: ankle, hint: 'z' };
  defs[B.footL] = { parent: B.shinL, head: ankle, tip: toe, hint: 'y' };

  defs[B.thighR] = { parent: B.hips, head: mir(hipJ), tip: mir(knee), hint: 'z' };
  defs[B.shinR] = { parent: B.thighR, head: mir(knee), tip: mir(ankle), hint: 'z' };
  defs[B.footR] = { parent: B.shinR, head: mir(ankle), tip: mir(toe), hint: 'y' };

  defs[B.scarfA] = { parent: B.chest, head: sA, tip: sB, hint: 'y' };
  defs[B.scarfB] = { parent: B.scarfA, head: sB, tip: sC, hint: 'y' };
  defs[B.scarfC] = { parent: B.scarfB, head: sC, tip: sT, hint: 'y' };
  return defs;
}

export interface Bone {
  name: string;
  parent: number;
  /** Length head→tip in the bind pose, metres. */
  len: number;
  bindWorld: Matrix4;
  invBind: Matrix4;
  bindLocalPos: Vector3;
  bindLocalQuat: Quaternion;
  /** Animation rotation, *relative to bind*. Written by the animator. */
  anim: Quaternion;
  /**
   * Animation translation, added to the bind offset. Used on the root bone for
   * crouch / heave / weight shift. Deliberately *not* applied to the rider's
   * Object3D: the handlebar IK targets live in rider space, so if the whole
   * rider translated the bars would follow the body instead of the boat.
   */
  animPos: Vector3;
  world: Matrix4;
  worldQuat: Quaternion;
}

const _q = new Quaternion();
const _m = new Matrix4();
const _m2 = new Matrix4();
const _p = new Vector3();
const ONE = new Vector3(1, 1, 1);

/**
 * A bind-posed bone hierarchy plus the flat `mat4[]` the shader reads.
 * Bones are stored parents-first so one forward pass resolves the whole tree.
 */
export class RiderSkeleton {
  readonly bones: Bone[] = [];
  /** Column-major skinning matrices, BONE_COUNT × 16. Uploaded as a uniform. */
  readonly skin = new Float32Array(BONE_COUNT * 16);
  /** Bind-pose joint positions in rider space, for IK targets and debugging. */
  readonly bindHead: Vector3[] = [];

  constructor(build: RiderBuild) {
    const defs = boneDefs(build);
    const names = Object.keys(B) as (keyof typeof B)[];

    for (let i = 0; i < BONE_COUNT; i++) {
      const d = defs[i];
      const dir = new Vector3().subVectors(d.tip, d.head);
      const len = dir.length() || 1e-4;
      dir.multiplyScalar(1 / len);

      // Right-handed basis with +Y along the bone. The hint axis pins the twist.
      const hint = d.hint === 'z' ? new Vector3(0, 0, 1) : new Vector3(0, 1, 0);
      const x = new Vector3().crossVectors(dir, hint);
      if (x.lengthSq() < 1e-6) x.crossVectors(dir, new Vector3(1, 0, 0));
      x.normalize();
      const z = new Vector3().crossVectors(x, dir);

      const bindWorld = new Matrix4().makeBasis(x, dir, z).setPosition(d.head);

      this.bones[i] = {
        name: names[i],
        parent: d.parent,
        len,
        bindWorld,
        invBind: bindWorld.clone().invert(),
        bindLocalPos: new Vector3(),
        bindLocalQuat: new Quaternion(),
        anim: new Quaternion(),
        animPos: new Vector3(),
        world: new Matrix4(),
        worldQuat: new Quaternion(),
      };
      this.bindHead[i] = d.head.clone();
    }

    // Bind pose expressed in each parent's frame.
    for (const bone of this.bones) {
      if (bone.parent < 0) _m.copy(bone.bindWorld);
      else _m.multiplyMatrices(this.bones[bone.parent].invBind, bone.bindWorld);
      _m.decompose(bone.bindLocalPos, bone.bindLocalQuat, new Vector3());
    }

    this.update();
  }

  /** Resolve every bone's world matrix and repack the skinning array. */
  update() {
    for (let i = 0; i < this.bones.length; i++) this.refresh(i);
  }

  /** Resolve one bone (and repack it). Parents must already be resolved. */
  refresh(i: number) {
    const b = this.bones[i];
    _q.copy(b.bindLocalQuat).multiply(b.anim);
    _p.addVectors(b.bindLocalPos, b.animPos);
    _m.compose(_p, _q, ONE);
    if (b.parent < 0) b.world.copy(_m);
    else b.world.multiplyMatrices(this.bones[b.parent].world, _m);
    b.worldQuat.setFromRotationMatrix(b.world);
    _m2.multiplyMatrices(b.world, b.invBind);
    _m2.toArray(this.skin, i * 16);
  }

  worldPos(i: number, out: Vector3): Vector3 {
    const e = this.bones[i].world.elements;
    return out.set(e[12], e[13], e[14]);
  }

  resetAnim() {
    for (const b of this.bones) b.anim.identity();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Geometry construction
// ─────────────────────────────────────────────────────────────────────────────

/** A 2D lathe profile point: radius, height along the bone. */
type Profile = [number, number][];

/**
 * Emits primitives into flat arrays, transforming each vertex by the current
 * bone's bind matrix so the merged geometry ends up in a single bind-pose space
 * that the skinning chunk can then deform. Build-time only — allocates freely.
 */
class Builder {
  pos: number[] = [];
  nrm: number[] = [];
  tint: number[] = [];
  bone: number[] = [];
  wt: number[] = [];
  uv: number[] = [];
  idx: number[] = [];

  private m = new Matrix4();
  private nm = new Matrix3();
  private boneA = 0;
  private boneB = 0;
  /** Blend weight toward boneA as a function of the *pre-transform* local y. */
  private weightFn: ((y: number) => number) | null = null;
  private _v = new Vector3();
  private _n = new Vector3();

  constructor(private skel: RiderSkeleton) {}

  /** Author the next primitives in `boneA`'s space, optionally offset by `local`. */
  at(boneA: number, local?: Matrix4, boneB = boneA, weightFn: ((y: number) => number) | null = null) {
    this.m.copy(this.skel.bones[boneA].bindWorld);
    if (local) this.m.multiply(local);
    this.nm.setFromMatrix4(this.m);
    this.boneA = boneA;
    this.boneB = boneB;
    this.weightFn = weightFn;
    return this;
  }

  private push(x: number, y: number, z: number, nx: number, ny: number, nz: number, u: number, v: number, c: Color) {
    this._v.set(x, y, z).applyMatrix4(this.m);
    this._n.set(nx, ny, nz).applyMatrix3(this.nm).normalize();
    this.pos.push(this._v.x, this._v.y, this._v.z);
    this.nrm.push(this._n.x, this._n.y, this._n.z);
    this.uv.push(u, v);
    this.tint.push(c.r, c.g, c.b);
    this.bone.push(this.boneA, this.boneB);
    this.wt.push(this.weightFn ? this.weightFn(y) : 1);
  }

  /**
   * Revolve a profile around +Y.
   *
   * Each quad is emitted with its **own four vertices** and a single tint taken
   * from the quad centre. That is not wasteful bookkeeping, it is the whole
   * point: on a shared vertex grid a colour change interpolates across a full
   * quad, so a racing stripe on a 14-sided limb arrives as a soft gradient —
   * exactly the airbrushed mush a cel look must not have. Independent corners
   * put the colour break on a polygon edge, hard.
   *
   * Normals stay *per-corner* (computed from the profile tangent), so the
   * surface still shades smoothly — flat tint, smooth form. And because
   * `computeSmoothNormals` welds by position afterwards, the outline hull is
   * unaffected by the duplication.
   */
  lathe(
    profile: Profile,
    seg: number,
    color: Color | ((u: number, v: number) => Color),
    sx = 1,
    sz = 1,
    /** Partial revolve, in turns. Used for shell-hugging patches like a visor. */
    t0 = 0,
    t1 = 1,
  ) {
    const rings = profile.length;
    // Per-ring: radius, height, profile normal (r, y component), v coordinate.
    const R: number[] = [];
    const Y: number[] = [];
    const NR: number[] = [];
    const NY: number[] = [];
    for (let i = 0; i < rings; i++) {
      R[i] = profile[i][0];
      Y[i] = profile[i][1];
      const p0 = profile[Math.max(0, i - 1)];
      const p1 = profile[Math.min(rings - 1, i + 1)];
      let tr = p1[0] - p0[0];
      let ty = p1[1] - p0[1];
      const tl = Math.hypot(tr, ty) || 1;
      tr /= tl;
      ty /= tl;
      NR[i] = ty;
      NY[i] = -tr;
    }
    const CA: number[] = [];
    const SA: number[] = [];
    for (let j = 0; j <= seg; j++) {
      const a = (t0 + (j / seg) * (t1 - t0)) * Math.PI * 2;
      CA[j] = Math.cos(a);
      SA[j] = Math.sin(a);
    }

    const vAt = (i: number) => i / (rings - 1 || 1);
    for (let i = 0; i < rings - 1; i++) {
      for (let j = 0; j < seg; j++) {
        const c =
          typeof color === 'function' ? color((j + 0.5) / seg, vAt(i) + 0.5 / (rings - 1 || 1)) : color;
        const base = this.pos.length / 3;
        // a = (i,j)  b = (i,j+1)  c2 = (i+1,j+1)  d = (i+1,j)
        const corner = (ri: number, cj: number) =>
          this.push(
            R[ri] * CA[cj] * sx,
            Y[ri],
            R[ri] * SA[cj] * sz,
            // Inverse-scale the normal so flattened parts still shade correctly.
            (NR[ri] * CA[cj]) / sx,
            NY[ri],
            (NR[ri] * SA[cj]) / sz,
            cj / seg,
            vAt(ri),
            c,
          );
        corner(i, j);
        corner(i, j + 1);
        corner(i + 1, j + 1);
        corner(i + 1, j);
        // Wound outward: at theta = 0 this gives a +X-facing normal.
        this.idx.push(base, base + 3, base + 1, base + 3, base + 2, base + 1);
      }
    }
  }

  /**
   * Axis-aligned box in the current local space. Each face gets its own four
   * vertices with a hard normal — `computeSmoothNormals` later merges them into
   * one averaged normal for the outline hull, so the box keeps crisp shading
   * *and* a watertight ink silhouette.
   */
  box(w: number, h: number, d: number, cx: number, cy: number, cz: number, color: Color) {
    const hx = w * 0.5;
    const hy = h * 0.5;
    const hz = d * 0.5;
    const normals = [
      [+1, 0, 0],
      [-1, 0, 0],
      [0, +1, 0],
      [0, -1, 0],
      [0, 0, +1],
      [0, 0, -1],
    ];
    const corners = [
      [-1, -1],
      [+1, -1],
      [+1, +1],
      [-1, +1],
    ];
    for (const n of normals) {
      const base = this.pos.length / 3;
      const nv = new Vector3(n[0], n[1], n[2]);
      // bt = nv × t, so t × bt = nv: the quad below winds outward by construction.
      const up = Math.abs(nv.y) > 0.9 ? new Vector3(0, 0, 1) : new Vector3(0, 1, 0);
      const t = new Vector3().crossVectors(up, nv).normalize();
      const bt = new Vector3().crossVectors(nv, t);
      const cen = new Vector3(cx + nv.x * hx, cy + nv.y * hy, cz + nv.z * hz);
      const sT = Math.abs(t.x) * hx + Math.abs(t.y) * hy + Math.abs(t.z) * hz;
      const sB = Math.abs(bt.x) * hx + Math.abs(bt.y) * hy + Math.abs(bt.z) * hz;
      for (let k = 0; k < 4; k++) {
        const [su, sv] = corners[k];
        this.push(
          cen.x + t.x * sT * su + bt.x * sB * sv,
          cen.y + t.y * sT * su + bt.y * sB * sv,
          cen.z + t.z * sT * su + bt.z * sB * sv,
          nv.x,
          nv.y,
          nv.z,
          (su + 1) * 0.5,
          (sv + 1) * 0.5,
          color,
        );
      }
      this.idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
    }
  }

  toGeometry(): BufferGeometry {
    const g = new BufferGeometry();
    g.setAttribute('position', new BufferAttribute(new Float32Array(this.pos), 3));
    g.setAttribute('normal', new BufferAttribute(new Float32Array(this.nrm), 3));
    g.setAttribute('uv', new BufferAttribute(new Float32Array(this.uv), 2));
    g.setAttribute('aTint', new BufferAttribute(new Float32Array(this.tint), 3));
    g.setAttribute('aBone', new BufferAttribute(new Float32Array(this.bone), 2));
    g.setAttribute('aBoneW', new BufferAttribute(new Float32Array(this.wt), 1));
    g.setIndex(this.idx);
    // The skinning happens on the GPU, so the bind-pose bounds are wrong once
    // the rider moves. A generous hand-set sphere beats per-frame recomputation
    // and stops limbs popping at the edge of frame.
    g.boundingSphere = new Sphere(new Vector3(0, 0.7, -0.1), 1.5);
    return g;
  }

  get triCount() {
    return this.idx.length / 3;
  }
}

/** Rounded limb profile: cap, taper, cap. */
function limbProfile(
  rBot: number,
  rTop: number,
  len: number,
  capBot: number,
  capTop: number,
  capSegs = 3,
  bulge = 0,
): Profile {
  const p: Profile = [];
  if (capBot > 0) {
    const h = rBot * capBot;
    for (let i = 0; i <= capSegs; i++) {
      const t = (i / capSegs) * (Math.PI / 2);
      p.push([rBot * Math.sin(t), -h * Math.cos(t)]);
    }
  } else {
    p.push([rBot, 0]);
  }
  if (bulge !== 0) p.push([((rBot + rTop) * 0.5) * (1 + bulge), len * 0.5]);
  p.push([rTop, len]);
  if (capTop > 0) {
    const h = rTop * capTop;
    for (let i = capSegs - 1; i >= 0; i--) {
      const t = (i / capSegs) * (Math.PI / 2);
      p.push([rTop * Math.sin(t), len + h * Math.cos(t)]);
    }
  }
  return p;
}

/**
 * Cloth folds, as geometry rather than paint.
 *
 * `limbProfile` gives a smooth taper, and a smooth taper is exactly what makes a
 * limb read as a moulded plastic tube. Real race leather bunches: it gathers
 * behind the elbow, above the knee, at the cuff. This inserts a **real
 * concavity** at each normalised position in `creases` — radius dips to 82% over
 * a ±4% window — which does three things a painted band cannot:
 *
 *   1. the groove wall turns away from the key light, so the cel ramp puts a
 *      hard band on it (a fold *shape*, not a gradient),
 *   2. the depth/normal discontinuity makes the Sobel interior pass ink a line
 *      along it, which is what an animator would draw,
 *   3. it survives the two-bone skin blend, so folds move with the joint.
 *
 * The same trick `knuckleProfile` uses for finger gaps, generalised.
 */
function creasedLimb(
  rBot: number,
  rTop: number,
  len: number,
  capBot: number,
  capTop: number,
  capSegs: number,
  creases: number[],
  bulge = 0,
): Profile {
  const p: Profile = [];
  const rAt = (t: number) => rBot + (rTop - rBot) * t + bulge * (rBot + rTop) * 0.5 * Math.sin(t * Math.PI);
  if (capBot > 0) {
    const h = rBot * capBot;
    for (let i = 0; i <= capSegs; i++) {
      const t = (i / capSegs) * (Math.PI / 2);
      p.push([rBot * Math.sin(t), -h * Math.cos(t)]);
    }
  } else {
    p.push([rBot, 0]);
  }
  for (const t of creases) {
    const r = rAt(t);
    p.push([r * 1.07, len * (t - 0.055)]);
    p.push([r * 0.82, len * t]);
    p.push([r * 1.05, len * (t + 0.055)]);
  }
  p.push([rTop, len]);
  if (capTop > 0) {
    const h = rTop * capTop;
    for (let i = capSegs - 1; i >= 0; i--) {
      const t = (i / capSegs) * (Math.PI / 2);
      p.push([rTop * Math.sin(t), len + h * Math.cos(t)]);
    }
  }
  return p;
}

/**
 * Four-knuckle finger roll: a lathe whose radius scallops in and out `n` times
 * along its length, so revolving it produces a run of four fused sausages.
 *
 * This is how the hands get fingers without four separate limbs. The roll is
 * laid *across* the handlebar (its axis parallel to the bar), so the scallops
 * fall where the finger gaps belong, and — the point — each groove is a real
 * concavity, so the Sobel interior pass inks a line between every finger. A
 * flat-tinted stripe would have read as paint; this reads as drawing.
 */
function knuckleProfile(r: number, len: number, n = 4): Profile {
  const p: Profile = [];
  const step = len / n;
  p.push([r * 0.42, 0]);
  for (let i = 0; i < n; i++) {
    // Fingers get slightly shorter toward the little finger, so the roll tapers
    // instead of reading as a machined cylinder.
    const k = 1 - i * 0.075;
    p.push([r * 0.9 * k, i * step + step * 0.14]);
    p.push([r * k, i * step + step * 0.5]);
    p.push([r * 0.88 * k, (i + 1) * step - step * 0.1]);
  }
  p.push([r * 0.4 * (1 - (n - 1) * 0.075), len]);
  return p;
}

/** Sphere/ellipsoid centred at `cy`, poles on Y. */
function sphereProfile(r: number, cy: number, rings = 8, yScale = 1): Profile {
  return sphereBand(r, cy, 0, 1, rings, yScale);
}

/**
 * A latitude band of a sphere, `v0`…`v1` measured from the south pole.
 * Combined with a partial revolve this is how the visor is built: a patch that
 * lies exactly on the helmet shell. A box pressed into a sphere pokes its
 * corners out through the surface — which is precisely what the first pass did.
 */
function sphereBand(r: number, cy: number, v0: number, v1: number, rings = 6, yScale = 1): Profile {
  const p: Profile = [];
  for (let i = 0; i <= rings; i++) {
    const t = (v0 + (i / rings) * (v1 - v0)) * Math.PI;
    p.push([r * Math.sin(t), cy - r * yScale * Math.cos(t)]);
  }
  return p;
}

// ─────────────────────────────────────────────────────────────────────────────
// The rider mesh
// ─────────────────────────────────────────────────────────────────────────────

export interface RiderMesh {
  root: Group;
  skel: RiderSkeleton;
  soft: Mesh;
  hard: Mesh;
  /** Handlebar grip targets in rider space, one per hand. */
  gripL: Vector3;
  gripR: Vector3;
  /**
   * Wrist-to-palm correction, rider space.
   *
   * The IK solves the **wrist**, but what has to touch the bar is the *palm*,
   * which sits `PALM_ALONG_HAND` further down the hand bone. Solving the wrist
   * straight onto the grip is what put the fists a hand's length past the bar in
   * every earlier capture — the arms looked as if they were reaching *through*
   * it, and because the reach was then within a centimetre of full extension
   * the elbows straightened into sticks. Adding this vector to the grip pulls
   * the wrist back along the hand's own axis, so the fingers close on the bar
   * and the elbow gets a real bend to draw.
   */
  palmOffset: Vector3;
  triangles: number;
  sets: CelMaterialSet[];
}

/**
 * Distance from the wrist joint to the centre of the finger roll, in metres —
 * i.e. how far down the hand bone the grip actually happens. Shared by the
 * geometry that places the roll and by the IK offset that aims for it, so the
 * two cannot drift apart.
 */
export const PALM_ALONG_HAND = 0.058;

/**
 * Shading ramp for the riders.
 *
 * Every part colour arrives per-vertex, so the material's own colour is white
 * and the ramp has to be a neutral *multiplier*: cool and dark in shadow,
 * neutral at the terminator, warm-white in the hot band. Built from palette
 * tones only (same trick `celMaterial`'s `defaultRamp` uses internally) — the
 * shadow leans on `waterDeep` so a rider in shadow sits in the sea's colour
 * family instead of going muddy grey.
 *
 * ── Why the ladder is this wide ─────────────────────────────────────────────
 * The previous version ran 0.88 / 0.95 / 1.00 / 1.10 — a 22% total spread. A
 * critic measurement of a 3× rider crop came back with *six near-tones between
 * L85 and L171 and no readable ramp*: the bands were mathematically present and
 * visually absent, because a 6-luminance step is below the threshold at which a
 * flat fill reads as a separate shape. The ladder is now 0.30 / 0.62 / 1.00 /
 * 1.36, i.e. a 4.5× spread, which puts roughly 55 luminance points between
 * consecutive bands on a mid-value fabric. `createCelMaterial` normalises the
 * >1 top band into the 8-bit ramp texture and restores the scale in the shader
 * (`uRampScale`), so the hot band is a real overbright and not a clamp.
 *
 * The stops are pulled tighter than the default: a 1.3 m character seen at
 * 5 m needs its terminator inside the silhouette, not smeared across it.
 */
function riderRamp(): Color[] {
  const w = PAL.cloudLit;
  return [
    w.clone().lerp(PAL.waterDeep, 0.5).multiplyScalar(0.6),
    w.clone().lerp(PAL.waterShadow, 0.34).multiplyScalar(0.88),
    w.clone(),
    w.clone().lerp(PAL.sun, 0.45).multiplyScalar(1.36),
  ];
}
const RIDER_RAMP_STOPS = [0.0, 0.34, 0.5, 0.82];

function skinChunks(bones: { value: Float32Array }) {
  return {
    uniforms: { uBones: bones as unknown as { value: Float32Array } },
    vertexHead: /* glsl */ `
      attribute vec3 aTint;
      attribute vec2 aBone;
      attribute float aBoneW;
      uniform mat4 uBones[${BONE_COUNT}];
      varying vec3 vTint;
    `,
    // Two-bone linear blend. `smoothNormal` is deformed alongside `objectNormal`
    // because the inverted-hull outline pushes along it — skip that and the ink
    // line stays in the bind pose while the model animates out from under it.
    vertexBody: /* glsl */ `
      {
        mat4 mA = uBones[int(aBone.x)];
        mat4 mB = uBones[int(aBone.y)];
        vec3 pA = (mA * vec4(transformed, 1.0)).xyz;
        vec3 pB = (mB * vec4(transformed, 1.0)).xyz;
        transformed = mix(pB, pA, aBoneW);
        mat3 rA = mat3(mA);
        mat3 rB = mat3(mB);
        objectNormal = normalize(mix(rB * objectNormal, rA * objectNormal, aBoneW));
        smoothNormal = normalize(mix(rB * smoothNormal, rA * smoothNormal, aBoneW));
        vTint = aTint;
      }
    `,
    fragmentHead: /* glsl */ `varying vec3 vTint;`,
    fragmentBody: /* glsl */ `baseColor *= vTint;`,
  };
}

/**
 * Build one rider: skeleton, two merged geometries, two cel material sets.
 *
 * Part layout (soft / hard):
 *   soft — pelvis, torso (2-bone blend), neck, face, upper+lower arms, gloves,
 *          thighs, shins, scarf
 *   hard — helmet, visor wedge, crest, shoulder pad(s), boots
 */
export function createRiderMesh(racerId: number, build: RiderBuild): RiderMesh {
  const colors = RACER_COLORS[racerId];
  const skel = new RiderSkeleton(build);
  const g = build.girth;

  // ── Value plan ────────────────────────────────────────────────────────────
  // Five values and one accent, because a cel figure is read as *shapes of
  // value* long before anyone sees its hue:
  //   pale   back panel, race-number plate, boot soles   (L≈245)
  //   lit    fold highlights, sleeve caps                (L≈180)
  //   mid    the suit itself                             (L≈125, most of body)
  //   shade  folds, chest front, pad interiors           (L≈75)
  //   dark   belt, gorget, boots, palms                  (L≈35)
  //   accent racer hull colour: helmet, yoke, gloves, stripes, scarf
  //
  // ── Two bugs fixed here, both measured ────────────────────────────────────
  // 1. **Every tint went out uncorrected.** `celMaterial` documents that
  //    `palette.ts` applies a second sRGB→linear decode and exports
  //    `paletteTone()` to undo it; `createCelMaterial` runs `opts.color` through
  //    it, which is why the *hulls* are vermilion and the riders were not. The
  //    rider's colours do not arrive as `opts.color` — they arrive per-vertex in
  //    `aTint` and are multiplied into `baseColor` in the same place, so they
  //    need exactly the same correction and were not getting it. Every tone below
  //    now goes through `T()`, and every mix is done *after* correction (mixing
  //    in the doubly-decoded space is a different, wrong interpolation).
  // 2. **The fabric was mixed out of hue.** It was
  //    `suit → skyHorizon(0.38) → hull(0.34)`: navy, plus pale cyan, plus
  //    vermilion. Those are three points spread right around the wheel, so the
  //    mix landed on grey. Predicted output for racer 0 was (165,111,140) and the
  //    critic measured exactly (165,111,140) in a 3× crop — "an unsaturated
  //    grey-mauve leotard", the only surface in frame off the high-saturation
  //    palette. The palette already ships the right tone for this: `suitMid0..3`
  //    were committed as "mid-value race-suit fabric, ~40% relative luminance,
  //    hue-matched to the suit tones" and were simply never used. They are the
  //    fabric now, undiluted, so the suit is a saturated hue and each racer's
  //    fabric hue is already distinct without borrowing the hull colour.
  const T = paletteTone;
  const SUIT_MID = [PAL.suitMid0, PAL.suitMid1, PAL.suitMid2, PAL.suitMid3];
  const hull = T(colors.hull);
  const suit = T(colors.suit);
  /** The committed mid-value fabric: saturated, hue-matched, ~L125 rendered. */
  const fabric = T(SUIT_MID[racerId % SUIT_MID.length]);
  /** Same hue, dropped toward the committed suit tone. Folds and shadow planes. */
  const fabricDark = fabric.clone().lerp(suit, 0.52);
  /** Same hue, lifted. Fold highlights and the sleeve cap. */
  const fabricLit = fabric.clone().lerp(T(PAL.foam), 0.4);
  const light = T(PAL.foamShade);
  const litePanel = T(PAL.foam);
  const skin = T(PAL.skin);
  const skinDark = T(PAL.skinShade);
  /** Underside of the scarf, and any cloth face turned away from the sun. */
  const accentShade = hull.clone().lerp(suit, 0.55);
  // Belt, gorget, boots and palms keep the *committed* suit tone, undiluted.
  // They are the one place its near-black value is an asset: dark extremities
  // and a dark waist terminate the figure and give it weight, the way ink does
  // in a cel drawing.
  const dark = suit;
  /** Visor glass. Dark cyan, not ink: ink is the outline's colour, and a visor
   *  painted in it merges with its own rim line instead of reading as glass. */
  const visorGlass = T(PAL.waterDeep).lerp(T(PAL.waterShadow), 0.45);
  /** The one hard highlight band raked across the visor. */
  const visorLit = T(PAL.waterCrest);

  /** True on the outward-facing columns of a limb, for either side. */
  const outerStripe = (side: number, u: number) =>
    side < 0 ? u > 0.42 && u < 0.58 : u < 0.08 || u > 0.92;

  const soft = new Builder(skel);
  const hard = new Builder(skel);
  const local = new Matrix4();
  const localPos = new Vector3();
  const localQuat = new Quaternion();

  const AX = new Vector3(1, 0, 0);
  const AY = new Vector3(0, 1, 0);
  const AZ = new Vector3(0, 0, 1);
  const spare = new Quaternion();
  /** Offset transform inside a bone's space, for parts that are not lathes. */
  const setLocal = (px: number, py: number, pz: number, rx = 0, ry = 0, rz = 0) => {
    localQuat.identity();
    if (rx) localQuat.multiply(spare.setFromAxisAngle(AX, rx));
    if (ry) localQuat.multiply(spare.setFromAxisAngle(AY, ry));
    if (rz) localQuat.multiply(spare.setFromAxisAngle(AZ, rz));
    local.compose(localPos.set(px, py, pz), localQuat, ONE);
    return local;
  };

  // ── Pelvis ────────────────────────────────────────────────────────────────
  soft.at(B.hips);
  soft.lathe(limbProfile(0.13 * g, 0.115 * g, skel.bones[B.hips].len, 0.9, 0.2, 3), 12, fabric, 1.12, 0.9);

  // ── Torso: blended across spine → chest so the waist bends, not creases ───
  const torsoLen = skel.bones[B.spine].len + skel.bones[B.chest].len;
  soft.at(B.spine, undefined, B.chest, (y) => 1 - Math.min(1, Math.max(0, (y / torsoLen - 0.15) / 0.7)));
  soft.lathe(
    [
      [0.112 * g, -0.02],
      [0.13 * g, torsoLen * 0.2],
      [0.144 * g, torsoLen * 0.52],
      [0.148 * g, torsoLen * 0.8],
      [0.126 * g, torsoLen * 1.02],
      [0.095 * g, torsoLen * 1.12],
    ],
    14,
    (u, v) => {
      // Four shapes, no patchwork. An earlier pass painted a belt band, a spine
      // stripe and a chest panel here; with 5 rings and 14 columns that lands as
      // a mosaic of small blocks — cel art wants few, big shapes.
      // u = 0 faces +X, 0.25 is the chest, 0.75 the back.
      // The waist was `dark` for one capture round. `dark` is the committed suit
      // tone, which is also the tone the boat's saddle is painted, so the rider's
      // whole lower torso merged into the seat behind it and the figure lost its
      // legs. The belt keeps the dark line; the fabric keeps its hue.
      if (v < 0.3) return fabricDark;
      if (u > 0.62 && u < 0.9) return light; // pale back panel, carries the number
      if (u > 0.12 && u < 0.4) return fabricDark; // chest front, turned from the sun
      return fabric; // flanks
    },
    // Was 1.2 wide. At that beam the torso was wider than the shoulders, so from
    // directly behind it *ate both arms* — a capture of KAIRA showed a grey sack
    // with one arm and no left arm at all. The chest now sits inside the
    // shoulder line, which is what lets the limbs draw their own silhouettes.
    1.06,
    0.86,
  );

  // Shoulder yoke — a wide cap in the racer's own colour over the top of the
  // torso. Two jobs: the torso stops being one tapering tube, and the strongest
  // identity colour lands on the highest, most-lit, least-occluded surface.
  soft.at(B.chest);
  soft.lathe(
    [
      [0.15 * g, skel.bones[B.chest].len * 0.4],
      [0.176 * g, skel.bones[B.chest].len * 0.6],
      [0.17 * g, skel.bones[B.chest].len * 0.86],
      [0.12 * g, skel.bones[B.chest].len * 1.0],
    ],
    14,
    hull,
    1.32,
    0.8,
  );

  // ── Belt + buckle ─────────────────────────────────────────────────────────
  // A hard dark band across the narrowest part of the torso. It is the single
  // cheapest "this is a costume, not a leotard" mark available: it cuts the
  // body into two shapes with different jobs, and the buckle puts one small
  // saturated accent on the figure's centre line.
  soft.at(B.spine, setLocal(0, torsoLen * 0.16, 0));
  soft.lathe(
    [
      [0.128 * g, -0.012],
      [0.142 * g, 0.0],
      [0.143 * g, 0.03],
      [0.13 * g, 0.042],
    ],
    14,
    // u ≈ 0.25 is dead ahead, so a narrow window there is a front buckle.
    (u) => (u > 0.19 && u < 0.31 ? hull : dark),
    1.08,
    0.9,
  );

  // Chest harness — a strap raked across the ribs, canted so it is not a second
  // belt. Scaled to the torso's own ellipse so it hugs rather than hoops, and
  // the rake is kept to ~10°: at the 24° first attempt the ring's own tilt
  // exceeded the clearance and the strap lifted clean off the body on one side.
  soft.at(B.spine, setLocal(0, torsoLen * 0.58, 0, 0, 0, 0.18));
  soft.lathe(
    [
      [0.15 * g, -0.011],
      [0.157 * g, 0.0],
      [0.157 * g, 0.022],
      [0.148 * g, 0.033],
    ],
    14,
    (u) => (u > 0.6 && u < 0.9 ? fabricLit : dark),
    1.06,
    0.88,
  );

  // Race-number plate on the pale back panel, with `racerId + 1` tally bars.
  // A logo is one of the critic's named misses, and a *counted* logo also makes
  // the four riders non-interchangeable from behind, which phase offsets alone
  // never achieve.
  //
  // Built as a partial-revolve shell on the torso's own ellipse, not as a box:
  // a flat plate pressed against a curved back is buried at its centre and
  // floating ~8 mm clear at its edges, which is exactly the "poking corners"
  // failure the visor already taught this file.
  //
  // The tally bars are painted into the plate's own tint function rather than
  // stacked on as separate arcs. As separate geometry they had to sit ~2 mm
  // proud of the plate, which is under a device pixel at closeup range: the bars
  // dithered against the panel and simply did not appear. Inside the tint they
  // are polygon-edge hard by construction, cost nothing, and cannot z-fight.
  // `lathe` hands the tint `u` as the fraction *across the revolved arc*, so the
  // 9 columns of the plate double as a 9-cell grid to print into.
  const plateSpan = 0.2;
  const plateT0 = 0.75 - plateSpan * 0.5;
  const tally = (racerId % 4) + 1;
  /** Column indices, out of 9, that carry a tally bar. Centred, one gap apart. */
  const bars = new Set<number>();
  for (let i = 0; i < tally; i++) bars.add(4 - (tally - 1) + i * 2);
  soft.at(B.spine);
  soft.lathe(
    [
      [0.14 * g, torsoLen * 0.6],
      [0.15 * g, torsoLen * 0.64],
      [0.15 * g, torsoLen * 0.79],
      [0.14 * g, torsoLen * 0.83],
    ],
    9,
    (u, v) => (v > 0.2 && v < 0.8 && bars.has(Math.floor(u * 9)) ? dark : litePanel),
    1.06,
    0.86,
    plateT0,
    plateT0 + plateSpan,
  );

  // ── Neck + face ───────────────────────────────────────────────────────────
  soft.at(B.neck);
  soft.lathe(limbProfile(0.052 * g, 0.047 * g, skel.bones[B.neck].len * 1.15, 0.2, 0, 2), 8, skinDark);
  // Gorget. The critic's crop showed "no neck" — correctly, because the helmet
  // sat straight on the shoulder yoke with nothing between them. A dark collar
  // ring wider than the neck puts a hard shadow shape under the jaw, which is
  // what makes a helmeted head read as *carried by* a neck rather than balanced
  // on a torso. Every rider gets one; the scarf collar (below) stacks on top.
  soft.at(B.neck, setLocal(0, 0.008, 0));
  soft.lathe(limbProfile(0.074 * g, 0.062 * g, 0.052, 0.35, 0.3, 2), 10, dark, 1.06, 1.06);
  soft.at(B.head);
  soft.lathe(sphereProfile(0.088 * build.headSize, 0.085, 9, 1.12), 12, (_u, v) => (v < 0.4 ? skinDark : skin), 1, 1.05);

  // ── Arms ──────────────────────────────────────────────────────────────────
  for (const side of [-1, 1] as const) {
    const up = side < 0 ? B.upArmL : B.upArmR;
    const lo = side < 0 ? B.loArmL : B.loArmR;
    const hand = side < 0 ? B.handL : B.handR;

    // Deltoid — the shoulder cap that hides the arm/torso join. It used to be
    // painted `light` at 0.068, which at this scale is a pale sphere wider than
    // the arm it caps: the captures came back with a white marshmallow stuck on
    // each shoulder. Smaller, and in the racer's own colour, it reads as a
    // shoulder seam and puts the identity hue on the highest lit surface.
    soft.at(up);
    soft.lathe(sphereProfile(0.059 * g, 0.006, 8, 0.92), 10, fabricLit);
    // Sleeve: mid-value, with an accent stripe down the *outer* face. u = 0.5
    // faces -X and u = 0/1 faces +X, so the stripe has to flip with the side —
    // painted at a fixed u it runs down the inside of one arm.
    //
    // Two creases where the fabric gathers above the elbow. See `creasedLimb`:
    // these are real grooves, so the ramp bands them and the Sobel inks them.
    soft.lathe(
      creasedLimb(0.062 * g, 0.05 * g, skel.bones[up].len, 0, 0.4, 3, [0.58, 0.82]),
      10,
      (u) => (outerStripe(side, u) ? hull : fabric),
    );
    soft.at(lo);
    soft.lathe(sphereProfile(0.051 * g, 0.0, 6, 1), 8, fabricDark);
    soft.lathe(
      creasedLimb(0.05 * g, 0.042 * g, skel.bones[lo].len, 0, 0.3, 2, [0.3]),
      10,
      (u) => (outerStripe(side, u) ? accentShade : fabricDark),
    );
    // One arm band, on one arm only. Constant asymmetry beats any amount of
    // animated variety for making four riders read as four people: `bias` is
    // already the build's standing left/right lean, so hanging the band off its
    // sign means each rider wears it on the side their posture already favours.
    if ((build.bias >= 0 ? -1 : 1) === side) {
      soft.at(up, setLocal(0, skel.bones[up].len * 0.34, 0));
      soft.lathe(limbProfile(0.062 * g, 0.06 * g, 0.026, 0.25, 0.25, 2), 10, hull, 1.0, 1.0);
    }

    // ── Hand ──────────────────────────────────────────────────────────────
    // Not a fist-shaped lump. A rider who is not visibly *gripping* reads as a
    // prop being carried by the boat, so the hand is built as four parts that
    // each do one silhouette job:
    //
    //   cuff    a hard accent ring where the sleeve ends — the limb terminates
    //           in a drawn line instead of fading into a dark blob
    //   back    a flat slab, wide across the bar and thin vertically
    //   roll    the four fingers, laid ALONG the bar so they wrap it
    //   thumb   crossing inboard over the bar, the read that says "grip"
    //
    // In this bone's space +Y runs wrist→fingertips and local X is the world
    // side axis (see the `hint` note on `BoneDef`). The bar also runs along the
    // world side axis, so the finger roll is the bone's own lathe rotated a
    // quarter turn about Z, which maps its +Y axis onto ∓X.
    //
    // ── Why the glove is now a saturated accent ─────────────────────────────
    // Every part of it was previously `dark` or `fabricDark` — near-black cloth
    // against a near-black ink line — and the critic read the result as "arms
    // terminate in blunt cylinders with no hands and no gloves". A hand is ~14
    // device px across at pack distance; at that size it only exists if it is a
    // different *hue* from the sleeve, not just a different shape. So the glove
    // is the racer's own colour with a pale cuff above it and a dark palm below:
    // three values inside 14 px, and the brightest thing on the whole arm sits
    // exactly where the eye is meant to go — the grip.
    soft.at(hand);
    soft.lathe(limbProfile(0.052 * g, 0.049 * g, 0.02, 0.4, 0, 2), 10, litePanel, 1.0, 1.06);
    // Wrist strap, dark, so the pale cuff has a hard lower boundary.
    soft.at(hand, setLocal(0, 0.019, 0));
    soft.lathe(limbProfile(0.05 * g, 0.048 * g, 0.014, 0.2, 0.2, 2), 10, dark, 1.0, 1.06);
    // Back of the hand.
    soft.at(hand);
    soft.lathe(
      [
        [0.028 * g, 0.03],
        [0.04 * g, 0.04],
        [0.046 * g, 0.06],
        [0.039 * g, 0.076],
      ],
      10,
      // u ≈ 0.25 is the back of the hand (local +Z), u ≈ 0.75 the palm.
      (u) => (u > 0.55 && u < 0.95 ? dark : hull),
      1.32,
      0.74,
    );
    // Fingers. Centred on PALM_ALONG_HAND, which is exactly where the IK aims
    // the palm — so the roll lands on the bar rather than beside it.
    //
    // `rz = -side · π/2` sends the lathe's +Y onto bone-space `side · X̂`, so the
    // roll grows from its origin in that direction. It therefore has to *start*
    // half a length the other way to end up centred on the wrist — starting at
    // `+side · half` (the first attempt) pushed the whole fist a full roll
    // length outboard, which is why the captured hand sat beside its own arm.
    const rollLen = 0.108 * g;
    soft.at(
      hand,
      setLocal(-side * rollLen * 0.5, PALM_ALONG_HAND, 0.008 * g, 0, 0, -side * Math.PI * 0.5),
    );
    // Fingers in the racer's colour, knuckles shaded. A fist painted in the
    // committed suit colour is the same value as its own ink outline: the crit
    // called the celebration hand "a stump" because at 40 px the whole hand was
    // one black silhouette, and called it a missing hand entirely at pack size.
    // The knuckle grooves are real concavities, so the Sobel pass inks a line
    // between every finger and the hand draws as a hand rather than a lozenge.
    soft.lathe(
      knuckleProfile(0.04 * g, rollLen),
      8,
      (u) => (u > 0.55 && u < 0.95 ? accentShade : hull),
      1.0,
      1.14,
    );
    // Thumb, laid inboard across the top of the bar and angled forward. `side`
    // twice over: once to pick the inboard direction in bone space, once to
    // send the lathe axis the same way.
    soft.at(
      hand,
      setLocal(side * 0.012 * g, PALM_ALONG_HAND - 0.02, -0.016 * g, 0, -0.55, -side * Math.PI * 0.5),
    );
    soft.lathe(limbProfile(0.023 * g, 0.018 * g, 0.058 * g, 0.9, 0.9, 2), 8, hull, 1.0, 1.1);
  }

  // ── Legs ──────────────────────────────────────────────────────────────────
  for (const side of [-1, 1] as const) {
    const th = side < 0 ? B.thighL : B.thighR;
    const sh = side < 0 ? B.shinL : B.shinR;
    const ft = side < 0 ? B.footL : B.footR;
    soft.at(th);
    soft.lathe(
      creasedLimb(0.082 * g, 0.062 * g, skel.bones[th].len, 0.3, 0.2, 3, [0.42, 0.72], 0.04),
      10,
      (u) => (outerStripe(side, u) ? hull : fabric),
    );
    soft.at(sh);
    soft.lathe(sphereProfile(0.062 * g, 0, 7, 0.9), 10, fabricDark);
    soft.lathe(
      creasedLimb(0.056 * g, 0.044 * g, skel.bones[sh].len, 0, 0.2, 2, [0.34]),
      10,
      (u) => (outerStripe(side, u) ? accentShade : fabric),
    );
    // Boot — hard family: it wants the tight, glossy highlight. Built as a
    // tapered lathe along the foot bone rather than a box: a brick on the end of
    // a leg is the single most obvious "programmer art" tell.
    //
    // The whole boot used to be one `dark` tone, i.e. one silhouette the same
    // value as its own outline — "no boots" in the crit, and correctly so. In
    // this bone's space +Y runs ankle→toe, so the *sole* is a u band rather than
    // a v band: local +Z is roughly up, which puts down at u ≈ 0.75.
    hard.at(ft);
    hard.lathe(
      limbProfile(0.056 * g, 0.04 * g, skel.bones[ft].len * 1.5, 0.9, 0.7, 3),
      10,
      (u) => (u > 0.6 && u < 0.9 ? light : dark),
      1.0,
      0.95,
    );
    // Ankle cuff, half a size up, so the boot has a top edge to ink.
    hard.at(ft);
    hard.lathe(limbProfile(0.066 * g, 0.058 * g, 0.045, 0.3, 0.2, 2), 10, dark, 1.0, 1.0);
    // Instep strap in the racer's colour: a boot needs one bright mark or it is
    // just the dark end of a leg.
    hard.at(ft, setLocal(0, skel.bones[ft].len * 0.62, 0));
    hard.lathe(limbProfile(0.05 * g, 0.047 * g, 0.024, 0.2, 0.2, 2), 10, hull, 1.0, 0.96);
  }

  // ── Helmet ────────────────────────────────────────────────────────────────
  //
  // ── The circle problem ──────────────────────────────────────────────────────
  // The visor, brow and chin bands below are all *front* features, and the shots
  // that matter are chase-camera: from behind, the previous helmet was a
  // perfectly circular disc with a specular dot on it, which is what the critic
  // measured ("a perfect featureless sphere ... no visor, no jaw, no chin"). A
  // character's head has to break its own circle from **every** azimuth, so the
  // shell now carries three rear/side features as well: ear pods on ±X, a nape
  // skirt that drops behind the jaw line, and a fore-aft aero fin. From directly
  // behind the silhouette is now a shell with two side lobes, a tail and a
  // dorsal blade.
  const hr = 0.118 * build.headSize;
  const hcy = 0.088;
  hard.at(B.head);
  // Egg, not ball: 1.13 in Z makes the shell longer front-to-back than it is
  // wide, so even a pure profile view has a direction to it.
  hard.lathe(sphereProfile(hr, hcy, 11, 1.05), 14, (_u, v) => (v < 0.2 ? accentShade : hull), 1.0, 1.13);
  // Visor: a dark-cyan glass band lying *on* the helmet shell — same sphere, 3%
  // larger, revolved only across the front 165°. That guarantees it hugs the
  // helmet at every corner, and because it is a shell the inverted hull inks
  // only its rim, which is exactly the drawn line a visor wants.
  // +Z is a quarter turn round from +X, so the front arc is centred on 0.25.
  hard.lathe(sphereBand(hr * 1.03, hcy, 0.36, 0.63, 3, 1.05), 12, visorGlass, 1.0, 1.13, 0.02, 0.48);
  // The one hard highlight band. Deliberately **not** symmetric: it runs across
  // the upper-left of the glass only, because a symmetric highlight reads as two
  // eyes and an off-centre one reads as a reflection of the sky.
  hard.lathe(sphereBand(hr * 1.05, hcy, 0.535, 0.6, 1, 1.05), 12, visorLit, 1.0, 1.13, 0.06, 0.245);
  // Brow: a second, thinner band above the visor in the racer colour.
  hard.lathe(sphereBand(hr * 1.06, hcy, 0.63, 0.71, 2, 1.05), 12, hull, 1.0, 1.13, 0.01, 0.49);
  // Chin guard below the visor, closing the face opening. In the *racer's* dark
  // tone, not the suit's: at `dark` it was the same value as the glass above it,
  // so the visor and the chin bar merged into one navy hood covering two thirds
  // of the head (shots/rider_r2/ocean_low.png, bow camera). A dark red chin bar
  // separates from dark cyan glass and the helmet reads as a full-face lid.
  hard.lathe(sphereBand(hr * 1.04, hcy, 0.26, 0.36, 2, 1.05), 12, accentShade, 1.0, 1.13, 0.05, 0.45);
  // Jaw. A forward-and-down wedge under the chin guard — the feature that turns
  // the profile silhouette from a circle into a face shape.
  hard.at(B.head, setLocal(0, hcy - hr * 0.52, hr * 0.74, -0.35));
  hard.lathe(limbProfile(hr * 0.46, hr * 0.3, hr * 0.34, 0.7, 0.6, 2), 10, accentShade, 1.5, 0.72);
  // Ear pods. Break the circle on both flanks from *any* azimuth, and give the
  // helmet the one thing a sphere cannot have: a left and a right.
  for (const side of [-1, 1] as const) {
    hard.at(B.head, setLocal(side * hr * 0.88, hcy - hr * 0.1, hr * 0.02, 0, 0, side * Math.PI * 0.5));
    hard.lathe(limbProfile(hr * 0.34, hr * 0.26, hr * 0.3, 0.5, 0.55, 2), 8, dark, 1.0, 1.25);
  }
  // Nape skirt — a partial-revolve shell across the back, dropped below the
  // shell's own equator so it hangs over the gorget. This is the rear
  // silhouette break: from the chase camera the head now has a tail.
  hard.at(B.head);
  hard.lathe(sphereBand(hr * 1.04, hcy - hr * 0.16, 0.2, 0.46, 3, 1.14), 12, accentShade, 1.0, 1.13, 0.56, 0.94);

  // Crown detail: a fore-aft aero blade, half-buried in the shell.
  //
  // It used to be `hard.box(0.015, hr*0.46, hr*1.15, …, litePanel)` — a
  // near-white axis-aligned cuboid on a red sphere, which the critic reported
  // literally as "a stray grey box intersecting the helmet crown". Two faults:
  // a box has no taper so it cannot read as a fin from any angle, and painting
  // it the palest tone in the set made it the loudest shape on the head. It is
  // now a flattened lathe — `sx` squashes the revolve onto the YZ plane, so the
  // profile's radius becomes the blade's fore-aft chord and its height becomes
  // the blade's rise — tapered to a point at both ends, in the *dark* tone with
  // the racer's colour on its top edge, so it reads as a fin with a lit spine.
  // The blade has to *clear* the crown or it does not exist: the first version
  // put its top ring exactly at `crown`, i.e. flush with the shell, and the
  // rear-quarter capture showed a perfectly smooth dome. Its origin is now
  // 0.34·hr below the crown with 0.66·hr of rise above that, so 0.32·hr — about
  // 11 device px at closeup scale — stands clear of the shell.
  const crown = hcy + hr * 1.05;
  const bladeTint = (_u: number, v: number) => (v > 0.7 ? hull : dark);
  if (build.crest === 'fin') {
    hard.at(B.head, setLocal(0, crown - hr * 0.34, -hr * 0.1));
    hard.lathe(
      [
        [0.0, -hr * 0.44],
        [hr * 0.66, -hr * 0.16],
        [hr * 0.8, hr * 0.24],
        [hr * 0.46, hr * 0.54],
        [0.0, hr * 0.66],
      ],
      7,
      bladeTint,
      0.26,
      1.0,
    );
  } else if (build.crest === 'mohawk') {
    for (let i = 0; i < 3; i++) {
      hard.at(B.head, setLocal(0, crown - hr * 0.3 - i * hr * 0.14, hr * (0.3 - i * 0.34)));
      hard.lathe(
        [
          [0.0, -hr * 0.4],
          [hr * 0.24, -hr * 0.14],
          [hr * 0.3, hr * (0.3 - i * 0.05)],
          [0.0, hr * (0.5 - i * 0.08)],
        ],
        6,
        bladeTint,
        0.24,
        1.0,
      );
    }
  } else {
    // No crest: the shell still needs a rear feature, so it gets a low ridge
    // sitting over the nape — squat and swept, so NOX still reads as helmeted
    // rather than finned.
    hard.at(B.head, setLocal(0, crown - hr * 0.34, -hr * 0.36));
    hard.lathe(
      [
        [0.0, -hr * 0.36],
        [hr * 0.44, -hr * 0.1],
        [hr * 0.5, hr * 0.2],
        [0.0, hr * 0.36],
      ],
      6,
      bladeTint,
      0.3,
      1.0,
    );
  }

  // ── Shoulder pad(s) — the silhouette break that makes a rider read ────────
  const padSides: number[] = build.shoulderPad === 'both' ? [-1, 1] : build.shoulderPad === 'left' ? [-1] : [];
  for (const side of padSides) {
    const up = side < 0 ? B.upArmL : B.upArmR;
    // Profile points must run in *increasing* y. The lathe derives its normals
    // from the profile tangent, so a descending profile silently produces
    // inward-facing normals — the pad renders as a black hole where you can see
    // the inside of the shell. (It did exactly that for two capture rounds.)
    hard.at(up, setLocal(0, 0.01, 0));
    hard.lathe(
      [
        [0.028 * g, -0.076],
        [0.078 * g, -0.064],
        [0.097 * g, -0.026],
        [0.088 * g, 0.018],
        [0.042 * g, 0.05],
      ],
      10,
      // No pale tone on the pad at all any more. `v < 0.3` is the pad's inner
      // cap, which from a bow camera is presented dead face-on: painted
      // `litePanel` it came back as a white disc the size of the helmet — the
      // "marshmallow" the crit has now flagged twice. The pad is two values of
      // the racer's own colour, which keeps it a silhouette break instead of a
      // value event, and the pale highlight job moves to the deltoid underneath.
      //
      // v = 0 is the *shoulder* end: `up` runs shoulder→elbow, so +Y goes down
      // the arm.
      (_u, v) => (v < 0.34 ? accentShade : hull),
      1.12,
      0.95,
    );
    // Hard lower lip in the dark tone so the pad has a drawn boundary against
    // the sleeve rather than fading into it.
    hard.at(up, setLocal(0, 0.028, 0));
    hard.lathe(limbProfile(0.09 * g, 0.062 * g, 0.016, 0, 0, 2), 10, dark, 1.12, 0.95);
  }

  // ── Scarf ─────────────────────────────────────────────────────────────────
  if (build.scarf > 0) {
    // Collar wrap.
    soft.at(B.neck, setLocal(0, 0.02, 0));
    soft.lathe(limbProfile(0.082 * g, 0.075 * g, 0.045, 0.5, 0.4, 2), 10, hull, 1.05, 1.1);
    // A ribbon, not a sausage: wide across the rider's back (local X, which is
    // the world side axis for these bones) and thinner vertically, so from the
    // chase camera — where it is actually seen — it presents its broad face.
    //
    // The vertical squash was 0.14, i.e. a 2 cm-thick, 27 cm-wide sheet of
    // paper. Captured from behind at deck height that is *edge on*: it came out
    // as a scatter of orange hairlines on KAIRA and as a flat red shard with a
    // blunt point on the player. Cloth has to have a cross-section. 0.42 gives a
    // ribbon ~5 cm deep — still clearly a ribbon in profile, but it now has a
    // silhouette to ink from every angle, and a lit top face against a shaded
    // underside instead of one flat tint.
    const seg = [B.scarfA, B.scarfB, B.scarfC];
    for (let i = 0; i < seg.length; i++) {
      const b = seg[i];
      const w = (0.112 - i * 0.026) * g;
      soft.at(b);
      soft.lathe(
        limbProfile(w, w * 0.7, skel.bones[b].len * 1.06, 0, i === 2 ? 0.9 : 0, 2),
        8,
        // u = 0.25 faces local +Z, which is up for these bones.
        (u) => (u > 0.02 && u < 0.5 ? hull : accentShade),
        1,
        0.42,
      );
    }
  }

  // ── Materials ─────────────────────────────────────────────────────────────
  const bonesUniform = { value: skel.skin };
  const ramp = riderRamp();

  const softSet = createCelMaterial({
    color: PAL.cloudLit,
    rampColors: ramp,
    rampStops: RIDER_RAMP_STOPS,
    rimColor: PAL.skyHorizon,
    rimPower: 2.6,
    rimStrength: 0.55,
    // Blinn thresholds have to be *tight*. `step(0.93, dot(N,H))` sounds small
    // but on a sphere it is a 21-degree cap — a pale disc the size of the whole
    // helmet, which is what turned the racer's red shell into a grey blob in the
    // first captures. These sizes give a glint, not a headlamp.
    specColor: PAL.foam,
    specSize: 0.978,
    specSize2: 0.995,
    specStrength: 0.13,
    // Cloth is not glossy. The shared gloss matcap was landing thresholded marks
    // on the suit and the shoulder pads — the flat hexagonal grey patch visible
    // on the player's pad in shots/r3/ocean_low.png is one of its disc marks, and
    // on fabric it reads as a hole rather than as a reflection. `null` drops the
    // USE_MATCAP define entirely, so this is one less texture fetch too.
    matcap: null,
    outlineWidthPx: 2.0,
    // Raised from 1.15. The rider is one merged mesh, so the Sobel pass has no
    // object-id discontinuity to work with where an arm crosses the chest — only
    // depth and normal — and 1.15 was leaving that crossing un-inked at pack
    // distance. Interior ink on limb-over-torso is an explicit art-direction ask.
    edgeBias: 1.5,
    name: `riderSoft${racerId}`,
    chunks: skinChunks(bonesUniform),
  });

  const hardSet = createCelMaterial({
    color: PAL.cloudLit,
    rampColors: ramp,
    rampStops: RIDER_RAMP_STOPS,
    rimColor: PAL.skyHorizon,
    rimPower: 3.6,
    // Kept modest deliberately: a strong rim on an ink-dark visor lifts it to
    // pale blue and the helmet stops reading as a helmet. The banded spec does
    // the "hard shell" job instead.
    rimStrength: 0.42,
    specColor: PAL.foam,
    specSize: 0.966,
    specSize2: 0.991,
    specStrength: 0.5,
    // Halved from the 0.34 default: the helmet keeps a hint of drawn reflection,
    // but at full strength the disc's sky strip washed a second pale cap across
    // the shell right next to the banded spec, giving the head two highlights.
    matcapStrength: 0.16,
    outlineWidthPx: 2.2,
    edgeBias: 1.5,
    name: `riderHard${racerId}`,
    chunks: skinChunks(bonesUniform),
  });

  const root = new Group();
  root.name = `rider${racerId}`;

  const softMesh = new Mesh(soft.toGeometry());
  softMesh.name = `rider${racerId}:soft`;
  applyCel(softMesh, softSet);
  root.add(softMesh);

  const hardMesh = new Mesh(hard.toGeometry());
  hardMesh.name = `rider${racerId}:hard`;
  applyCel(hardMesh, hardSet);
  root.add(hardMesh);

  // Handlebar grips, straight from the hull's published contract and expressed
  // in seat space, which is rider space. The animator IKs the wrists here, so a
  // rider with longer arms simply carries a deeper elbow bend.
  const gripL = new Vector3(-GRIP_HALF_WIDTH, 0, 0).add(GRIP_LOCAL).sub(SEAT_LOCAL);
  const gripR = new Vector3(GRIP_HALF_WIDTH, 0, 0).add(GRIP_LOCAL).sub(SEAT_LOCAL);

  // Column 1 of the hand bone's bind matrix is its +Y axis in rider space — the
  // wrist→fingertip direction. Walking *back* along it by PALM_ALONG_HAND is
  // the vector that turns a grip point into a wrist target. Taken from the bind
  // matrix rather than hard-coded so a re-proportioned hand stays in register.
  const he = skel.bones[B.handL].bindWorld.elements;
  const palmOffset = new Vector3(he[4], he[5], he[6]).normalize().multiplyScalar(-PALM_ALONG_HAND);

  return {
    root,
    skel,
    soft: softMesh,
    hard: hardMesh,
    gripL,
    gripR,
    palmOffset,
    triangles: soft.triCount + hard.triCount,
    sets: [softSet, hardSet],
  };
}
