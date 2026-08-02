/**
 * INK TIDE — procedural rider animation.
 *
 * No clips, no keyframes, no external data: every frame the pose is *derived*
 * from the boat's state. The structure is deliberately layered rather than a
 * pile of conditionals:
 *
 *   1. SIGNALS   — the raw BoatState is turned into the handful of smoothed,
 *                  frame-rate-independent quantities an animator would actually
 *                  key against: yaw rate, longitudinal g, hull vertical
 *                  acceleration, throttle, slip.
 *   2. STATES    — five whole-body poses (ride, brace, tuck, land, celebrate).
 *                  Each writes an *absolute* pose into a buffer; the machine
 *                  blends them by weight and renormalises, so transitions are
 *                  cross-fades and a pose can be added without touching the
 *                  others.
 *   3. MODIFIERS — additive layers on top of the blend: balance (counter-roll),
 *                  lean into the turn, weight shift, sea-heave, head lead,
 *                  scarf whip.
 *   4. IK        — the hands are then *solved* onto the handlebars. This is the
 *                  layer that sells it: whatever the torso does, the grip stays
 *                  put, because the bars belong to the boat and not to the
 *                  rider.
 *
 * Every smoothing call is `damp()`/`dampAngle()` against dt. Nothing here
 * allocates: all vector/quaternion scratch is module scope.
 */

import { Matrix4, Quaternion, Vector3 } from 'three';
import { angleDelta, clamp, clamp01, damp, smoothstep } from '../core/mathx';
import type { GameContext, Racer } from '../core/types';
import { B, BONE_COUNT, type RiderBuild, type RiderMesh, type RiderSkeleton } from './riderRig';

// ─────────────────────────────────────────────────────────────────────────────
// Pose buffers
// ─────────────────────────────────────────────────────────────────────────────

/** BONE_COUNT euler triples (XYZ, bone-local, relative to bind) + a hips offset. */
const POSE_LEN = BONE_COUNT * 3 + 3;
const HIP_OFF = BONE_COUNT * 3;

type Pose = Float32Array;
const newPose = (): Pose => new Float32Array(POSE_LEN);

function setE(p: Pose, bone: number, x: number, y: number, z: number) {
  const i = bone * 3;
  p[i] = x;
  p[i + 1] = y;
  p[i + 2] = z;
}

// ─────────────────────────────────────────────────────────────────────────────
// Signals
// ─────────────────────────────────────────────────────────────────────────────

interface Signals {
  t: number;
  /** Turning right = +1, left = -1. Derived from the measured yaw rate so it
   *  is immune to whatever sign convention the steer input happens to use. */
  turn: number;
  /** Longitudinal g: + accelerating, - braking. Normalised-ish. */
  lonG: number;
  /** Hull vertical acceleration, m/s². The sea, felt through the seat. */
  heaveA: number;
  /** Body heave offset in metres — the lagged reaction to `heaveA`. */
  heave: number;
  speedFrac: number;
  throttle: number;
  brake: number;
  slip: number;
  roll: number;
  pitch: number;
  /** State machine weights. */
  wAir: number;
  wLand: number;
  wCeleb: number;
  wBrace: number;
  wDrift: number;
  /** Seconds since the celebration started. */
  celebT: number;
  /** Countdown rev-up, 0…1. */
  rev: number;
  build: RiderBuild;
}

// ─────────────────────────────────────────────────────────────────────────────
// States — each writes one absolute pose
// ─────────────────────────────────────────────────────────────────────────────

/**
 * RIDE. The base. A racing crouch that opens up as speed falls off and packs
 * down as it climbs — the single most important read for "this thing is fast".
 */
function poseRide(p: Pose, s: Signals) {
  const sp = s.speedFrac;
  const b = s.build;
  p.fill(0);
  setE(p, B.hips, -0.02 + sp * 0.1, 0, 0);
  setE(p, B.spine, 0.16 + b.hunch + sp * 0.3, 0, 0);
  setE(p, B.chest, 0.1 + b.hunch * 0.5 + sp * 0.16, 0, 0);
  setE(p, B.neck, -0.14 - sp * 0.16, 0, 0);
  // Head lifts as the torso packs down, so the eyeline stays on the horizon.
  setE(p, B.head, -0.16 - sp * 0.24, 0, 0);
  // Shoulders: shrug is posture, the speed term is wind pressure.
  setE(p, B.clavL, 0, 0, -b.shrug - sp * 0.06);
  setE(p, B.clavR, 0, 0, b.shrug + sp * 0.06);
  // Legs: the bind pose is already folded, this is the fine trim.
  const knee = sp * 0.08;
  setE(p, B.thighL, -knee, 0, 0);
  setE(p, B.thighR, -knee, 0, 0);
  setE(p, B.shinL, knee * 1.4, 0, 0);
  setE(p, B.shinR, knee * 1.4, 0, 0);
  // Scarf lift. The bind tail already droops ~20° below horizontal, and these
  // three add up, so the numbers are small on purpose — at full speed the tip
  // ends just above the horizontal, streaming, not saluting.
  setE(p, B.scarfA, 0.04 + sp * 0.14, 0, 0);
  setE(p, B.scarfB, 0.03 + sp * 0.1, 0, 0);
  setE(p, B.scarfC, 0.02 + sp * 0.08, 0, 0);
  p[HIP_OFF + 1] = -sp * 0.02;
}

/**
 * BRACE. Hard on the brakes: weight goes back, arms straighten, chest opens.
 */
function poseBrace(p: Pose, s: Signals) {
  poseRide(p, s);
  setE(p, B.hips, -0.24, 0, 0);
  setE(p, B.spine, -0.12 + s.build.hunch * 0.4, 0, 0);
  setE(p, B.chest, -0.08, 0, 0);
  setE(p, B.neck, 0.02, 0, 0);
  setE(p, B.head, -0.02, 0, 0);
  setE(p, B.thighL, 0.16, 0, 0);
  setE(p, B.thighR, 0.16, 0, 0);
  setE(p, B.shinL, -0.1, 0, 0);
  setE(p, B.shinR, -0.1, 0, 0);
  p[HIP_OFF + 1] = 0.015;
  p[HIP_OFF + 2] = -0.05;
}

/**
 * TUCK. Airborne. Knees come up under the chest, spine curls, head stays level
 * because the rider is watching the landing. Elbows tuck in (via the grip
 * target, not here).
 */
function poseTuck(p: Pose, s: Signals) {
  poseRide(p, s);
  // These numbers are roughly double the first pass. At the old amplitudes a
  // capture at airTime = 0.3 s was indistinguishable from the ride pose — the
  // crit read it, correctly, as "zero tuck". A tuck has to be a *silhouette*
  // change: the head drops below the shoulder line, the back domes over the
  // bars, and the knees come up in front of the hips.
  setE(p, B.hips, 0.26, 0, 0);
  setE(p, B.spine, 0.48 + s.build.hunch, 0, 0);
  setE(p, B.chest, 0.28, 0, 0);
  // Neck and head lift hard *against* the fold: the total forward pitch of the
  // torso is over 60°, and if the head went with it the rider would be looking
  // at the deck. It has to fold and still watch the landing.
  setE(p, B.neck, -0.36, 0, 0);
  setE(p, B.head, -0.46, 0, 0);
  setE(p, B.clavL, 0, 0, -0.28);
  setE(p, B.clavR, 0, 0, 0.28);
  // Knees up: fold the thigh forward and let the shin trail.
  setE(p, B.thighL, -0.58, 0.06, 0);
  setE(p, B.thighR, -0.58, -0.06, 0);
  setE(p, B.shinL, 0.7, 0, 0);
  setE(p, B.shinR, 0.7, 0, 0);
  // Scarf streams straight out behind and lifts on the way up.
  setE(p, B.scarfA, 0.34, 0, 0);
  setE(p, B.scarfB, 0.26, 0, 0);
  setE(p, B.scarfC, 0.2, 0, 0);
  // Hips forward and slightly down: the rider comes off the seat and folds over
  // the bars, rather than being lifted clear of the boat.
  p[HIP_OFF + 1] = -0.018;
  p[HIP_OFF + 2] = 0.035;
}

/**
 * LAND. The absorb. Knees eat the impact, hips drop, chest comes up and the
 * shoulders rise — a compression, not a slouch.
 */
function poseLand(p: Pose, s: Signals) {
  poseRide(p, s);
  // The absorb is the biggest single deformation in the whole rig, because it is
  // the only one the player *feels* — it has to arrive as a visible squash on
  // the frame of impact. 11 cm of hip drop on a 1.1 m torso is about a tenth of
  // the figure's height; anything less and the shot reads as "no crouch".
  setE(p, B.hips, 0.16, 0, 0);
  setE(p, B.spine, 0.46 + s.build.hunch, 0, 0);
  setE(p, B.chest, 0.06, 0, 0);
  setE(p, B.neck, -0.34, 0, 0);
  setE(p, B.head, -0.46, 0, 0);
  // Shoulders up around the ears: a compression, not a slouch.
  setE(p, B.clavL, 0, 0, -0.4);
  setE(p, B.clavR, 0, 0, 0.4);
  setE(p, B.thighL, -0.46, 0, 0);
  setE(p, B.thighR, -0.46, 0, 0);
  setE(p, B.shinL, 1.0, 0, 0);
  setE(p, B.shinR, 1.0, 0, 0);
  p[HIP_OFF + 1] = -0.115;
  p[HIP_OFF + 2] = 0.012;
}

/**
 * CELEBRATE. Torso twists, head goes back over the right shoulder to look at
 * the field, hips ride high. The raised arm itself is the IK target moving —
 * see `gripTarget()`.
 */
function poseCeleb(p: Pose, s: Signals) {
  poseRide(p, s);
  const t = s.celebT * s.build.tempo;
  const pump = Math.sin(t * 3.1) * 0.5 + 0.5;
  const sway = Math.sin(t * 1.5);
  setE(p, B.hips, -0.12, 0.14 * sway, 0.07 * sway);
  setE(p, B.spine, -0.2 - pump * 0.08, 0.34, -0.1 * sway);
  setE(p, B.chest, -0.14, 0.3, -0.06 * sway);
  setE(p, B.neck, 0.06, 0.34, 0.05);
  setE(p, B.head, 0.1 - pump * 0.12, 0.42, 0.08 * sway);
  // Right shoulder drives up with the arm (+Z raises the right clavicle).
  setE(p, B.clavR, 0, 0, 0.3 + pump * 0.25);
  setE(p, B.clavL, 0, 0, -0.1);
  setE(p, B.thighL, 0.1, 0, 0);
  setE(p, B.thighR, 0.12, 0, 0);
  setE(p, B.shinL, -0.16, 0, 0);
  setE(p, B.shinR, -0.2, 0, 0);
  setE(p, B.scarfA, 0.06 + pump * 0.12, 0, -0.2 * sway);
  setE(p, B.scarfB, 0.05 + pump * 0.14, 0, -0.3 * sway);
  setE(p, B.scarfC, 0.04 + pump * 0.16, 0, -0.4 * sway);
  p[HIP_OFF + 1] = 0.03 + pump * 0.03;
}

// ─────────────────────────────────────────────────────────────────────────────
// IK
// ─────────────────────────────────────────────────────────────────────────────

const _S = new Vector3();
const _T = new Vector3();
const _v = new Vector3();
const _dir = new Vector3();
const _n = new Vector3();
const _pole = new Vector3();
const _upper = new Vector3();
const _elbow = new Vector3();
const _fore = new Vector3();
const _x = new Vector3();
const _z = new Vector3();
const _bx = new Vector3();
const _qa = new Quaternion();
const _qb = new Quaternion();
const _qc = new Quaternion();
const _mb = new Matrix4();

/**
 * Orientation with +Y along `dirY`, twisted to sit as close as possible to the
 * bone's bind X axis. Choosing the twist this way (rather than from the bend
 * plane) is what keeps a sleeve stripe on the outside of the arm and a fist the
 * right way up — an IK chain that only constrains direction will happily roll
 * the limb 180°.
 */
function aimQuat(dirY: Vector3, refX: Vector3, out: Quaternion) {
  _x.copy(refX).addScaledVector(dirY, -refX.dot(dirY));
  if (_x.lengthSq() < 1e-8) {
    _x.set(1, 0, 0).addScaledVector(dirY, -dirY.x);
    if (_x.lengthSq() < 1e-8) _x.set(0, 0, 1).addScaledVector(dirY, -dirY.z);
  }
  _x.normalize();
  _z.crossVectors(_x, dirY);
  _mb.makeBasis(_x, dirY, _z);
  return out.setFromRotationMatrix(_mb);
}

/** Column 0 of a bone's bind matrix — its rest X axis in rider space. */
function bindX(skel: RiderSkeleton, bone: number, out: Vector3) {
  const e = skel.bones[bone].bindWorld.elements;
  return out.set(e[0], e[1], e[2]);
}

/**
 * Two-bone analytic IK (law of cosines). `poleX/Y/Z` biases which way the
 * elbow breaks; `weight` cross-fades against whatever the pose machine already
 * put on these bones, so IK can be faded out (nothing to grip, mid-air flail)
 * without a pop.
 */
function solveTwoBoneIK(
  skel: RiderSkeleton,
  upIdx: number,
  loIdx: number,
  endIdx: number,
  target: Vector3,
  poleX: number,
  poleY: number,
  poleZ: number,
  weight: number,
) {
  if (weight <= 0.001) return;
  const up = skel.bones[upIdx];
  const lo = skel.bones[loIdx];
  const L1 = up.len;
  const L2 = lo.len;

  skel.worldPos(upIdx, _S);
  _T.copy(target);
  _v.subVectors(_T, _S);
  let d = _v.length();
  const dMin = Math.abs(L1 - L2) * 1.02 + 1e-4;
  const dMax = (L1 + L2) * 0.995;
  d = clamp(d, dMin, dMax);
  if (_v.lengthSq() < 1e-8) _v.set(0, -1, 0);
  _dir.copy(_v).normalize();

  // Interior angle at the shoulder between (shoulder→target) and the upper arm.
  const cosA = clamp((L1 * L1 + d * d - L2 * L2) / (2 * L1 * d), -1, 1);
  const a = Math.acos(cosA);

  _pole.set(poleX, poleY, poleZ).normalize();
  _n.crossVectors(_dir, _pole);
  if (_n.lengthSq() < 1e-8) _n.set(0, 1, 0).cross(_dir);
  _n.normalize();

  // Rotating `dir` about (dir × pole) by +a tilts it toward the pole.
  _upper.copy(_dir).applyQuaternion(_qa.setFromAxisAngle(_n, a));
  _elbow.copy(_S).addScaledVector(_upper, L1);
  _fore.subVectors(_T, _elbow);
  if (_fore.lengthSq() < 1e-8) _fore.copy(_upper);
  _fore.normalize();

  // Upper arm.
  bindX(skel, upIdx, _bx);
  aimQuat(_upper, _bx, _qb);
  _qc.copy(skel.bones[up.parent].worldQuat).multiply(up.bindLocalQuat).invert().multiply(_qb);
  up.anim.slerp(_qc, weight);
  skel.refresh(upIdx);

  // Forearm.
  bindX(skel, loIdx, _bx);
  aimQuat(_fore, _bx, _qb);
  _qc.copy(skel.bones[lo.parent].worldQuat).multiply(lo.bindLocalQuat).invert().multiply(_qb);
  lo.anim.slerp(_qc, weight);
  skel.refresh(loIdx);
  skel.refresh(endIdx);
}

// ─────────────────────────────────────────────────────────────────────────────
// Animator
// ─────────────────────────────────────────────────────────────────────────────

const _euler = new Quaternion();
const _eq = new Quaternion();
const AX = new Vector3(1, 0, 0);
const AY = new Vector3(0, 1, 0);
const AZ = new Vector3(0, 0, 1);
const _grip = new Vector3();
const _hipOff = new Vector3();
const _target = new Vector3();

/** How the scarf's total stream angle is split across its three bones… */
const SCARF_SHARE = [0.55, 0.28, 0.17];
/** …plus a constant curl on the trailing two. Cloth never straightens out
 *  completely, and a dead-straight tail is the thing that reads as a plank. */
const SCARF_CURL = [0.0, -0.2, -0.3];

/** Scratch pose buffers, shared by every rider — one at a time, in one thread. */
const P_RIDE = newPose();
const P_BRACE = newPose();
const P_TUCK = newPose();
const P_LAND = newPose();
const P_CELEB = newPose();
const P_OUT = newPose();

export class RiderAnimator {
  private sig: Signals;
  private prevHeading = 0;
  private prevVy = 0;
  private prevFwd = 0;
  private started = false;

  constructor(
    private mesh: RiderMesh,
    private build: RiderBuild,
  ) {
    this.sig = {
      t: 0,
      turn: 0,
      lonG: 0,
      heaveA: 0,
      heave: 0,
      speedFrac: 0,
      throttle: 0,
      brake: 0,
      slip: 0,
      roll: 0,
      pitch: 0,
      wAir: 0,
      wLand: 0,
      wCeleb: 0,
      wBrace: 0,
      wDrift: 0,
      celebT: 0,
      rev: 0,
      build,
    };
  }

  // ── 1. Signals ────────────────────────────────────────────────────────────

  private readSignals(racer: Racer, ctx: GameContext) {
    const s = this.sig;
    const st = racer.state;
    const dt = Math.max(ctx.dt, 1e-4);
    const b = this.build;
    s.t = ctx.time * b.tempo + b.phase;

    if (!this.started) {
      this.prevHeading = st.heading;
      this.prevVy = st.velocity.y;
      this.prevFwd = st.forwardSpeed;
      this.started = true;
    }

    // Yaw rate → turn direction. Reading the measured heading change rather
        // than the steer axis keeps the rider correct whatever the boat's input
    // convention is, and it also picks up AI drivers for free.
    const yawRate = angleDelta(this.prevHeading, st.heading) / dt;
    this.prevHeading = st.heading;
    s.turn = damp(s.turn, clamp(yawRate / 1.5, -1, 1), 9, ctx.dt);

    // Longitudinal g and hull vertical acceleration.
    const lon = (st.forwardSpeed - this.prevFwd) / dt;
    this.prevFwd = st.forwardSpeed;
    s.lonG = damp(s.lonG, clamp(lon / 14, -1.4, 1.4), 7, ctx.dt);

    const vy = st.velocity.y;
    const hv = (vy - this.prevVy) / dt;
    this.prevVy = vy;
    s.heaveA = damp(s.heaveA, clamp(hv, -60, 60), 16, ctx.dt);
    // The body is a mass on a spring: when the deck accelerates up, the rider
    // compresses. Reacting to the *hull's* acceleration (not a free sine) is
    // what makes the idle bob read as the sea rather than as an animation.
    s.heave = damp(s.heave, clamp(-s.heaveA * 0.0042, -0.05, 0.038), 11, ctx.dt);

    s.speedFrac = st.speedFrac;
    s.throttle = damp(s.throttle, racer.controls.throttle, 12, ctx.dt);
    s.brake = damp(s.brake, racer.controls.brake, 12, ctx.dt);
    s.slip = damp(s.slip, clamp(st.lateralSpeed / 9, -1.4, 1.4), 6, ctx.dt);
    s.roll = st.roll;
    s.pitch = st.pitch;

    // ── State weights ───────────────────────────────────────────────────────
    // The tuck has to be *there* by the time the boat is a fifth of a second off
    // the water, which is when a jump reads. At the old rate of 5.5 the weight
    // was still under 0.8 at airTime = 0.3 and the pose was being averaged away
    // against the ride pose — the `air` shot showed an upright rider mid-flight.
    s.wAir = damp(s.wAir, st.airborne ? 1 : 0, st.airborne ? 13 : 9, ctx.dt);
    // Landing is an impulse: spike the weight, then let it bleed out. The bleed
    // rate sets how long the absorb reads for.
    //
    // The divisor was 9, which meant a routine 2 m/s re-entry produced a weight
    // of 0.22 — a 20% crouch, i.e. none. Any landing worth an impact sound gets
    // most of the absorb; only the divisor's tail distinguishes a hard one.
    if (st.landingImpact > 0.8) {
      s.wLand = Math.max(s.wLand, 0.55 + clamp01(st.landingImpact / 6) * 0.45);
    }
    s.wLand = damp(s.wLand, 0, 2.4, ctx.dt);
    s.wBrace = damp(s.wBrace, clamp01(racer.controls.brake * 0.8 + clamp01(-s.lonG) * 0.7), 6, ctx.dt);
    s.wDrift = damp(s.wDrift, st.drifting ? 1 : 0, 6, ctx.dt);

    const celebrating = racer.finished || ctx.race.phase === 'results';
    s.wCeleb = damp(s.wCeleb, celebrating ? 1 : 0, 2.6, ctx.dt);
    s.celebT = celebrating ? s.celebT + ctx.dt : 0;

    s.rev = damp(s.rev, ctx.race.phase === 'countdown' ? 1 : 0, 5, ctx.dt);
  }

  // ── 2. State machine ──────────────────────────────────────────────────────

  private blendPose() {
    const s = this.sig;
    poseRide(P_RIDE, s);
    const wAir = s.wAir;
    const wLand = s.wLand;
    const wCeleb = s.wCeleb;
    const wBrace = s.wBrace * (1 - wAir) * (1 - wCeleb);
    // Ride yields to whatever else is asking for the body, so a full-weight
    // celebration is a *pure* celebration and not a 50/50 mush.
    const wRide = Math.max(0, 1 - (wAir + wLand * 0.9 + wCeleb + wBrace * 0.7));

    let total = wRide + wAir + wLand + wCeleb + wBrace;
    if (total < 1e-4) total = 1;
    const inv = 1 / total;

    P_OUT.fill(0);
    const acc = (p: Pose, w: number) => {
      if (w <= 1e-4) return;
      const k = w * inv;
      for (let i = 0; i < POSE_LEN; i++) P_OUT[i] += p[i] * k;
    };
    acc(P_RIDE, wRide);
    if (wBrace > 1e-4) {
      poseBrace(P_BRACE, s);
      acc(P_BRACE, wBrace);
    }
    if (wAir > 1e-4) {
      poseTuck(P_TUCK, s);
      acc(P_TUCK, wAir);
    }
    if (wLand > 1e-4) {
      poseLand(P_LAND, s);
      acc(P_LAND, wLand);
    }
    if (wCeleb > 1e-4) {
      poseCeleb(P_CELEB, s);
      acc(P_CELEB, wCeleb);
    }
  }

  // ── 3. Additive modifiers ─────────────────────────────────────────────────

  private modifiers() {
    const s = this.sig;
    const b = this.build;
    const grounded = 1 - s.wAir;
    const free = 1 - s.wCeleb;

    // BALANCE. The hull rolls; the rider does not go with it. Counter-rotating
    // through the hips and spine (and holding the head near level) is the whole
    // difference between a passenger and an athlete.
    const counter = -s.roll * 0.5 * free;
    P_OUT[B.hips * 3 + 2] += counter * 0.55;
    P_OUT[B.spine * 3 + 2] += counter * 0.3;
    P_OUT[B.chest * 3 + 2] += counter * 0.2;
    P_OUT[B.head * 3 + 2] += -s.roll * 0.55 * free;
    // Pitch: the rider stays vertical over a nose-up hull.
    P_OUT[B.spine * 3] += -s.pitch * 0.34 * free;
    P_OUT[B.head * 3] += -s.pitch * 0.3 * free;

    // LEAN INTO THE TURN. Lean is toward the inside of the corner: rotating
    // about +Z lifts +X, so a right-hand turn wants a negative Z.
    const lean = (s.turn * 0.9 + s.slip * 0.22) * grounded * free;
    P_OUT[B.hips * 3 + 2] += -lean * 0.16;
    P_OUT[B.spine * 3 + 2] += -lean * 0.2;
    P_OUT[B.chest * 3 + 2] += -lean * 0.16;
    P_OUT[B.spine * 3 + 1] += s.turn * 0.1 * grounded * free;
    // Drift adds a shoulder-first commitment on top.
    P_OUT[B.chest * 3 + 1] += s.turn * 0.16 * s.wDrift * free;

    // HEAD LEADS. Look where you are going, half a beat before the boat does.
    P_OUT[B.head * 3 + 1] += (s.turn * 0.5 + s.slip * 0.1) * free;
    P_OUT[B.neck * 3 + 1] += s.turn * 0.2 * free;

    // WEIGHT SHIFT. Longitudinal g pushes the torso back under power and folds
    // it forward under braking; the hips slide in the seat.
    P_OUT[B.spine * 3] += -s.lonG * 0.14 * free;
    P_OUT[B.chest * 3] += -s.lonG * 0.08 * free;
    P_OUT[B.head * 3] += s.lonG * 0.06 * free;
    P_OUT[HIP_OFF + 2] += -s.lonG * 0.018 * free;

    // SEA. The heave offset is a real spring reaction to hull acceleration; the
    // knees take part of it so the body absorbs rather than teleports.
    P_OUT[HIP_OFF + 1] += s.heave;
    const absorb = clamp(-s.heave * 6, -0.3, 0.3);
    P_OUT[B.thighL * 3] += -absorb * 0.5;
    P_OUT[B.thighR * 3] += -absorb * 0.5;
    P_OUT[B.shinL * 3] += absorb;
    P_OUT[B.shinR * 3] += absorb;
    P_OUT[B.spine * 3] += absorb * 0.25;

    // IDLE + BIAS. Four racers driving the same racing line at the same speed
    // get near-identical signals, so every pose-machine output above is nearly
    // the same for all of them — which is exactly what the crit saw: "all four
    // racers hold the identical upright pose". Two layers fix that, and both are
    // needed:
    //
    //   BIAS   a *constant* asymmetry per racer. Phase offsets alone are not
    //          enough, because four riders on the same cycle still pass through
    //          the same neutral shape every period. A standing cant means they
    //          never share a silhouette at all.
    //   IDLE   three slow, mutually-prime oscillators read at `s.t`, which
    //          already carries this rider's `phase` and `tempo`. Small in
    //          amplitude — this is breathing and balance, not animation — but
    //          it is the phase difference, not the size, that the eye reads at
    //          pack distance.
    const bias = b.bias * free;
    P_OUT[B.chest * 3 + 2] += bias * 0.055;
    P_OUT[B.spine * 3 + 1] += bias * 0.05;
    P_OUT[B.clavL * 3 + 2] += -bias * 0.09;
    P_OUT[B.clavR * 3 + 2] += -bias * 0.05;
    P_OUT[B.head * 3 + 1] += bias * 0.13;
    P_OUT[B.head * 3 + 2] += -bias * 0.07;

    const idle = free * (1 - s.wLand * 0.6);
    const i1 = Math.sin(s.t * 0.83);
    const i2 = Math.sin(s.t * 1.27 + 1.1);
    const i3 = Math.sin(s.t * 0.61 + 2.3);
    P_OUT[B.spine * 3] += i1 * 0.05 * idle;
    P_OUT[B.chest * 3 + 1] += i2 * 0.07 * idle;
    P_OUT[B.chest * 3 + 2] += i3 * 0.055 * idle;
    P_OUT[B.neck * 3 + 1] += i2 * 0.06 * idle;
    P_OUT[B.head * 3 + 1] += (i2 * 0.15 + i3 * 0.1) * idle;
    P_OUT[B.head * 3] += i1 * 0.08 * idle;
    P_OUT[B.clavL * 3 + 2] += -i3 * 0.06 * idle;
    P_OUT[B.clavR * 3 + 2] += i1 * 0.06 * idle;
    P_OUT[HIP_OFF] += i3 * 0.013 * idle;
    P_OUT[HIP_OFF + 1] += i1 * 0.011 * idle;

    // Engine buzz — tiny, high frequency, only at speed. Reads as vibration
    // through the bars rather than as animation.
    const buzz = Math.sin(s.t * 41) * 0.0022 * s.speedFrac * grounded;
    P_OUT[HIP_OFF + 1] += buzz;
    P_OUT[B.chest * 3 + 2] += buzz * 1.5;

    // Countdown: blipping the throttle. Shoulders and wrists twitch.
    if (s.rev > 0.01) {
      const blip = Math.max(0, Math.sin(s.t * 7.5)) * s.rev;
      P_OUT[B.spine * 3] += blip * 0.06;
      P_OUT[B.clavR * 3 + 2] += -blip * 0.08;
    }

    // SCARF WHIP. For these bones local X is the world side axis, local Y runs
    // along the ribbon and local Z is up — so X is lift, Y is *twist* and Z is
    // the lateral swing. Getting that mapping wrong is how a scarf ends up
    // pointing at the sky. Each segment lags the one before it, so the tail
    // cracks rather than swinging as one rigid plank.
    const flow = 0.3 + s.speedFrac * 0.9;
    // A scarf answers to the airflow, not to the shoulders it is tied to. The
    // torso pitches forward ~30° at speed, which rotates a backward-hanging
    // tail *upward* — so the accumulated torso pitch is subtracted out and the
    // tail is aimed in something much closer to world space: hanging when
    // stopped, streaming just above the horizontal at full chat.
    const torsoPitch = P_OUT[B.hips * 3] + P_OUT[B.spine * 3] + P_OUT[B.chest * 3];
    //
    // The speed term was 0.86, which is not enough to beat the torso pitch it is
    // being corrected against: at 84 km/h the chase capture showed the tail
    // hanging dead-straight down the rider's back, which is what a scarf does at
    // a standstill. 1.4 puts the tail at about the horizontal by full chat and
    // still lets it hang when the boat is stopped.
    const stream = (-0.3 + 1.4 * s.speedFrac - torsoPitch) * (1 - s.wCeleb * 0.5);
    for (let i = 0; i < 3; i++) {
      const bone = B.scarfA + i;
      const lag = 1 + i * 0.7;
      const ph = s.t * (3.1 + i * 0.7) - i * 0.9;
      P_OUT[bone * 3] += stream * SCARF_SHARE[i] + SCARF_CURL[i];
      P_OUT[bone * 3] += Math.sin(ph) * 0.06 * flow * lag;
      // Twist was ±0.3 · flow · lag, which on the tail segment is over 40° — and
      // rolling a ribbon 40° is precisely how you turn its broad face away and
      // present the edge. Cut to a suggestion of flutter.
      P_OUT[bone * 3 + 1] += Math.sin(ph * 0.77 + 0.6) * 0.14 * flow * lag;
      // A standing yaw bias fans the tail off to one side, so it presents some
      // of its face from every camera angle instead of vanishing edge-on. Held
      // small: at -0.1 · lag the tail swung out level with the shoulder and read
      // as a shard sticking out of the rider rather than as cloth trailing.
      P_OUT[bone * 3 + 2] +=
        -0.045 * lag + Math.cos(ph * 0.9) * 0.05 * flow * lag - s.slip * 0.08 * lag;
      // The heave snaps the scarf up on a hard landing.
      P_OUT[bone * 3] += clamp(-s.heaveA * 0.0022, -0.2, 0.3) * lag * 0.4;
    }
  }

  // ── 4. Grip targets + IK ──────────────────────────────────────────────────

  /**
   * Where a hand should be this frame. The bars belong to the boat, so these
   * are in rider space and barely move — except that they *yaw with the steer*
   * (one hand forward, one back), get pushed with the throttle, pulled on the
   * brake, and for the raised celebration arm leave the bars altogether.
   */
  private gripTarget(side: -1 | 1, out: Vector3) {
    const s = this.sig;
    const m = this.mesh;
    out.copy(side < 0 ? m.gripL : m.gripR);

    // Bar yaw: rotate the grip around the bar centre. `turn` is right-positive
    // and the near hand comes back, so the far hand leads.
    const yaw = s.turn * 0.3 * (1 - s.wCeleb);
    const cz = out.z - 0.08;
    const dx = out.x;
    const dz = out.z - cz;
    out.x = dx * Math.cos(yaw) + dz * Math.sin(yaw);
    out.z = cz - dx * Math.sin(yaw) + dz * Math.cos(yaw);
    out.y += -Math.abs(yaw) * 0.02 + yaw * side * 0.03;

    // Throttle / brake work at the grip.
    out.z += s.throttle * 0.022 - s.brake * 0.05;
    out.y += -s.throttle * 0.012 + s.brake * 0.022;
    // Countdown blip: short, sharp wrist pumps on the throttle side.
    if (s.rev > 0.01 && side > 0) {
      const blip = Math.max(0, Math.sin(s.t * 7.5)) * s.rev;
      out.z += blip * 0.02;
      out.y -= blip * 0.012;
    }
    // Airborne: elbows come in and down as the body tucks.
    out.z -= s.wAir * 0.05;
    out.y -= s.wAir * 0.02;
    // Landing: arms brace against the bars.
    out.z -= s.wLand * 0.02;

    // Wrist, not palm. Everything above is a *bar* position; the IK end effector
    // is the wrist joint, which sits PALM_ALONG_HAND behind the grip along the
    // hand's own axis. Without this the fists floated a hand's length past the
    // bar and the arms locked out straight at full extension.
    out.add(m.palmOffset);

    // Celebration: right arm punches the air and pumps; left stays on the bar.
    // Rider space is hips-relative, so "above the head" is ~0.9 m, not 1.6 —
    // an out-of-reach target just pins the arm straight at max extension.
    if (s.wCeleb > 0.001 && side > 0) {
      const t = s.celebT * this.build.tempo;
      const pump = Math.sin(t * 3.1) * 0.5 + 0.5;
      _grip.set(
        0.3 * this.build.girth,
        (0.82 + pump * 0.1) * this.build.height,
        -0.04 - pump * 0.05,
      );
      out.lerp(_grip, smoothstep(0, 1, s.wCeleb));
    }
    return out;
  }

  // ── Drive ─────────────────────────────────────────────────────────────────

  update(racer: Racer, ctx: GameContext) {
    this.readSignals(racer, ctx);
    this.blendPose();
    this.modifiers();

    const skel = this.mesh.skel;

    // Euler triples → bone quaternions. XYZ order, applied about the bone's own
    // axes; the angles are small enough that order is a styling choice, not a
    // correctness one.
    for (let i = 0; i < BONE_COUNT; i++) {
      const x = P_OUT[i * 3];
      const y = P_OUT[i * 3 + 1];
      const z = P_OUT[i * 3 + 2];
      const q = skel.bones[i].anim;
      q.identity();
      if (x) q.multiply(_euler.setFromAxisAngle(AX, x));
      if (y) q.multiply(_eq.setFromAxisAngle(AY, y));
      if (z) q.multiply(_eq.setFromAxisAngle(AZ, z));
    }

    _hipOff.set(P_OUT[HIP_OFF], P_OUT[HIP_OFF + 1], P_OUT[HIP_OFF + 2]);
    skel.bones[B.hips].animPos.copy(_hipOff);

    skel.update();

    // Hands last, so they land on the bars no matter what the torso just did.
    const s = this.sig;
    // Mid-air with the bars still in reach the grip holds; a big tuck lets the
    // solve relax slightly so the arms do not look bolted on.
    const ikW = 1 - s.wAir * 0.15;
    solveTwoBoneIK(skel, B.upArmL, B.loArmL, B.handL, this.gripTarget(-1, _target), -0.5, -0.55, -0.68, ikW);
    solveTwoBoneIK(skel, B.upArmR, B.loArmR, B.handR, this.gripTarget(1, _target), 0.5, -0.55, -0.68, ikW);
  }
}

