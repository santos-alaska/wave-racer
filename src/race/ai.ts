/**
 * AI DRIVERS.
 *
 * Three rules govern this file:
 *
 *   1. **The AI drives the same struct the player does.** Everything below ends
 *      in `r.controls.{steer, throttle, brake, drift}`. There is no AI-only
 *      force, no path snapping, no cornering cheat. If an AI boat does something
 *      the player's boat could not, that is a bug.
 *   2. **Speed comes from the track, not from a magic number.** `Track` publishes
 *      a back-propagated speed profile derived from the boat's *own* steering
 *      model, so "how fast can this corner be taken" is one lookup and every
 *      driver brakes at a physically correct point. Personality then scales it.
 *   3. **Mistakes are deliberate and legible.** Random throttle noise is not a
 *      mistake, it is mush. A mistake here is a wide entry, an overshoot, or a
 *      lift — timed to land on a corner, where the player can see it happen and
 *      take advantage.
 *
 * ── The three temperaments ─────────────────────────────────────────────────
 *  KAIRA — aggressive. Lives 3.6 m inside the line, brakes 0.24 s later than
 *          anyone else, dives for the inside of a boat ahead instead of going
 *          around it, leans on rivals rather than yielding, and overcooks a
 *          corner about once a lap.
 *  NOX   — clean. Sits on the racing line, brakes early and precisely, gives way
 *          in traffic, and is boringly consistent. The benchmark.
 *  PIP   — erratic. Runs 3.9 m wide of the line and wanders another 4 m either
 *          side of that, breathes the throttle, and runs wide into corners two
 *          or three times a lap.
 *
 * Those three lane biases are load-bearing, not flavour. Four drivers reading
 * one `lineOffset` table drive one groove, and a field on one groove is a train
 * with interpenetrating hulls — which is exactly what the first build shipped.
 *
 * ── Collision avoidance ────────────────────────────────────────────────────
 * Two terms, neither of them a temperament: a hard lateral separation inside
 * 3.4 m abeam (larger than any hull half-width, applied undamped straight to the
 * steering aim point), and a longitudinal speed cap behind a boat within 11 m.
 *
 * ── Rubber-banding ─────────────────────────────────────────────────────────
 * Capped at CONFIG.ai.rubberBand (11 %), asymmetric (a trailing boat gets the
 * full 11 %, a leading boat only 6 % of a penalty), and pushed through a 2.2 s
 * damper so it can never read as a boat suddenly finding power. It scales the
 * *pace fraction of the track's speed profile*, which means a rubber-banded AI
 * still brakes for the hairpin; it just carries a little more everywhere.
 */

import { Vector3 } from 'three';
import { CONFIG } from '../core/config';
import { angleDelta, clamp, clamp01, damp, smoothstep } from '../core/mathx';
import { Rng } from '../core/rng';
import type { GameContext, Racer, Subsystem } from '../core/types';
import type { CornerPreview, Track } from './track';

// ─────────────────────────────────────────────────────────────────────────────
// Scratch
// ─────────────────────────────────────────────────────────────────────────────

const _aim = new Vector3();
const _preview: CornerPreview = {
  distance: Infinity,
  direction: 0,
  severity: 0,
  radius: Infinity,
  speed: CONFIG.boat.topSpeed,
};

const enum Mistake {
  None = 0,
  /** Runs wide on entry — the classic "he's left the door open". */
  WideEntry = 1,
  /** Carries far too much speed in, then has to stand on the brakes. */
  Overshoot = 2,
  /** A brief lift; small, cheap, keeps the gaps breathing. */
  Lift = 3,
}

interface Profile {
  /** Fraction of the track's speed profile the driver uses. */
  pace: number;
  /**
   * Metres this driver habitually sits off the racing line, + = track right.
   *
   * Without this every AI reads the same `lineOffset` table and drives the same
   * groove, so the field arrives at a corner nose-to-tail with the hulls
   * overlapping — measured in shots/race_fix0/land.png, four boats single file
   * inside two boat lengths, and in pack.png where the red and yellow hulls are
   * touching. A persistent bias per temperament is what makes them arrive three
   * abreast instead, which is also the only way the pack shot can demonstrate
   * the per-racer colour separation it exists to prove.
   *
   * Faded out with corner severity: on a hairpin the line is the line.
   */
  laneBias: number;
  /**
   * Seconds of lookahead used to read the speed profile. Small = brakes late.
   * This is the single most character-defining number in the file.
   */
  reaction: number;
  /** 0…1 — how faithfully they track the racing line. */
  discipline: number;
  /** Lateral wander, metres, and its frequency in Hz. */
  wanderAmp: number;
  wanderHz: number;
  /** Proportional and derivative gains on heading error. */
  steerP: number;
  steerD: number;
  /** How hard they move aside for another boat. Low = they lean on you. */
  avoid: number;
  /** How readily they dive for an inside gap. */
  aggression: number;
  /** Mistakes per lap, roughly. */
  mistakesPerLap: number;
  /** Which mistakes they make, weighted. */
  mistakeKinds: readonly Mistake[];
  /** Corner severity above which they powerslide for a boost. */
  driftAt: number;
}

const PROFILES: Record<string, Profile> = {
  aggressive: {
    pace: 1.005,
    laneBias: -3.6, // lives on the inside, ready to close a door
    reaction: 0.10,
    discipline: 0.82,
    wanderAmp: 1.1,
    wanderHz: 0.21,
    steerP: 2.4,
    steerD: 0.30,
    avoid: 0.35,
    aggression: 1.0,
    mistakesPerLap: 1.1,
    mistakeKinds: [Mistake.Overshoot, Mistake.Overshoot, Mistake.WideEntry],
    driftAt: 0.34,
  },
  clean: {
    pace: 0.99,
    laneBias: 0.5, // the benchmark: on the line, a shade to the right of it
    reaction: 0.34,
    discipline: 1.0,
    wanderAmp: 0.35,
    wanderHz: 0.13,
    steerP: 2.1,
    steerD: 0.38,
    avoid: 1.0,
    aggression: 0.25,
    mistakesPerLap: 0.25,
    mistakeKinds: [Mistake.Lift],
    driftAt: 0.5,
  },
  erratic: {
    pace: 1.0,
    laneBias: 3.9, // sits out wide and drifts about out there
    reaction: 0.26,
    discipline: 0.55,
    wanderAmp: 4.2,
    wanderHz: 0.33,
    steerP: 2.6,
    steerD: 0.22,
    avoid: 0.7,
    aggression: 0.55,
    mistakesPerLap: 2.4,
    mistakeKinds: [Mistake.WideEntry, Mistake.Lift, Mistake.WideEntry, Mistake.Overshoot],
    driftAt: 0.28,
  },
};

interface Brain {
  profile: Profile;
  rng: Rng;
  phase: number;
  /** Smoothed lane offset actually being driven to, metres (+ = track right). */
  lane: number;
  /** Smoothed steer output. */
  steer: number;
  /** Heading last frame, for a measured yaw rate (the D term). */
  lastHeading: number;
  yawRate: number;
  /** Damped rubber-band multiplier, −rubberBand…+rubberBand. */
  rubber: number;
  /** Current mistake. */
  mistake: Mistake;
  mistakeTime: number;
  mistakeDur: number;
  mistakeLane: number;
  mistakeCooldown: number;
  /** Powerslide hysteresis. */
  driftHold: number;
  /** Seconds with no forward speed, and the reverse-out timer. */
  stuck: number;
  recover: number;
}

export class AiDrivers implements Subsystem {
  readonly name = 'ai';
  readonly order = 40;

  private brains = new Map<number, Brain>();

  constructor(
    private racers: Racer[],
    private trk: Track,
  ) {
    let seed = 0xa11ce;
    for (const r of racers) {
      if (r.isPlayer) continue;
      const profile = PROFILES[r.personality ?? 'clean'] ?? PROFILES.clean;
      seed = (seed * 1664525 + 1013904223) >>> 0;
      const rng = new Rng(seed);
      this.brains.set(r.id, {
        profile,
        rng,
        phase: rng.range(0, Math.PI * 2),
        lane: 0,
        steer: 0,
        lastHeading: r.state.heading,
        yawRate: 0,
        rubber: 0,
        mistake: Mistake.None,
        mistakeTime: 0,
        mistakeDur: 0,
        mistakeLane: 0,
        // Stagger the first opportunity so all three do not err on lap 1 turn 1.
        mistakeCooldown: rng.range(6, 20),
        driftHold: 0,
        stuck: 0,
        recover: 0,
      });
    }
  }

  update(ctx: GameContext) {
    for (const r of this.racers) {
      if (r.isPlayer) continue;
      this.drive(ctx, r);
    }
  }

  // ───────────────────────────────────────────────────────────────────────────
  private drive(ctx: GameContext, r: Racer) {
    const b = this.brains.get(r.id)!;
    const c = r.controls;
    const dt = ctx.dt;
    const cfg = CONFIG.boat;
    const L = this.trk.length;

    // Measured yaw rate, for the derivative term. `BoatState` does not publish
    // one, and differentiating heading here is exact enough and costs nothing.
    const raw = angleDelta(b.lastHeading, r.state.heading) / Math.max(dt, 1e-4);
    b.lastHeading = r.state.heading;
    b.yawRate = damp(b.yawRate, raw, 22, dt);

    const proj = this.trk.project(r.root.position);
    const u = proj.u;
    const fwd = r.state.forwardSpeed;
    const speed = Math.hypot(r.state.velocity.x, r.state.velocity.z);

    // ── Recovery ────────────────────────────────────────────────────────────
    // Beached against a wave face, spun around, or shunted 60 m off line. Back
    // off, swing the nose round, go again. Uses nothing but brake and steer.
    const offCourse = Math.abs(proj.lateral) > 55;
    b.stuck = fwd < 2.2 && !r.state.airborne && ctx.race.phase === 'racing' ? b.stuck + dt : 0;
    if (b.stuck > 2.4 && b.recover <= 0) b.recover = 1.5;
    if (b.recover > 0) {
      b.recover -= dt;
      const wantHeading = this.aimHeading(r, u, 26, 0);
      const err = angleDelta(r.state.heading, wantHeading);
      c.throttle = 0;
      c.brake = 1;
      // Reversing: the stern leads, so the steering sign flips.
      c.steer = clamp(err * 1.4, -1, 1);
      c.drift = false;
      return;
    }

    if (r.finished) {
      // Keep them moving so the results camera has something to look at, but
      // clearly off the pace.
      const vt = this.trk.speedLimit(u + 30 / L) * 0.5;
      this.applySpeed(c, fwd, vt, 0);
      c.steer = this.steerTo(r, b, u, 0, dt);
      c.drift = false;
      return;
    }

    // ── Rubber band ─────────────────────────────────────────────────────────
    // Gap to the *player*, in metres of centreline. Positive = the AI is behind.
    const gap = (ctx.player.progress - r.progress) * L;
    const raw01 = clamp(gap / 200, -1, 1);
    let want = raw01 * CONFIG.ai.rubberBand;
    if (want < 0) want *= 0.55; // leaders are penalised only half as much
    if (ctx.race.raceTime < 5) want = 0; // never on the run to turn one
    // 2.2 s time constant. Any faster and a trailing boat visibly surges.
    b.rubber = damp(b.rubber, want, 0.45, dt);

    // ── Corner read ─────────────────────────────────────────────────────────
    this.trk.cornerPreview(u, _preview);
    const severity = _preview.severity;

    // ── Mistakes ────────────────────────────────────────────────────────────
    this.updateMistake(b, dt, severity, _preview.distance, _preview.direction, ctx.race.raceTime);

    // ── Lane target ─────────────────────────────────────────────────────────
    const lookDist = CONFIG.ai.lookaheadBase + Math.abs(fwd) * CONFIG.ai.lookaheadPerSpeed;
    const uAhead = u + lookDist / L;
    let lane = this.trk.lineOffset(uAhead) * b.profile.discipline;
    // The driver's habitual lane, faded out where the corner dictates the line.
    lane += b.profile.laneBias * (1 - severity * 0.75);
    // Wander: a slow lateral drift, not per-frame noise. Noise looks like a
    // broken controller; drift looks like a driver who is not quite settled.
    lane +=
      Math.sin(ctx.time * b.profile.wanderHz * Math.PI * 2 + b.phase) *
      b.profile.wanderAmp *
      (1 - severity * 0.6);
    if (b.mistake === Mistake.WideEntry) lane += b.mistakeLane;
    // Keep them inside a believable corridor, and honour the gate they must pass.
    lane = clamp(lane, -13, 13);
    b.lane = damp(b.lane, lane, 2.6, dt);

    // Traffic is added AFTER the damper, deliberately. A 2.6 s⁻¹ lag on the
    // racing line is character; the same lag on the term that keeps two hulls
    // out of each other is a bug — at 29 m/s it is most of a boat length of
    // interpenetration before the avoidance even reaches the steering.
    const laneNow = clamp(b.lane + this.trafficOffset(ctx, r, b, uAhead), -16, 16);

    // ── Speed target ────────────────────────────────────────────────────────
    // Read the profile `reaction` seconds ahead. A late braker looks 0.10 s
    // ahead and arrives hot; the clean driver looks 0.42 s ahead and is already
    // slowing. Same code, entirely different character.
    let react = b.profile.reaction;
    if (b.mistake === Mistake.Overshoot) react = 0.02;
    const probe = u + (Math.abs(fwd) * react + 5) / L;
    let vTarget = this.trk.speedLimit(probe) * b.profile.pace * (1 + b.rubber);
    if (b.mistake === Mistake.Overshoot) vTarget *= 1.3;
    if (b.mistake === Mistake.Lift) vTarget *= 0.62;
    // Off course: crawl back rather than blast across the sea at 29 m/s.
    if (offCourse) vTarget = Math.min(vTarget, 15);
    // Don't drive into the back of the boat in front. This is the longitudinal
    // half of collision avoidance and it is what breaks up the nose-to-tail
    // train: a driver stuck behind someone either has to move out (the lateral
    // term above) or lift.
    vTarget = Math.min(vTarget, this.followCap(ctx, r));
    vTarget = clamp(vTarget, 5, cfg.boostTopSpeed);

    // ── Outputs ─────────────────────────────────────────────────────────────
    this.applySpeed(c, fwd, vTarget, severity);
    c.steer = this.steerTo(r, b, u, laneNow, dt);

    // ── Powerslide ──────────────────────────────────────────────────────────
    // The AI earns boost the same way the player does: hold the slide through a
    // tight corner, release on exit. Hysteresis on the hold so it does not
    // flicker on a crest.
    const canDrift =
      severity > b.profile.driftAt &&
      Math.abs(fwd) > 14 &&
      Math.abs(c.steer) > 0.42 &&
      _preview.distance < 26;
    b.driftHold = canDrift ? 0.45 : Math.max(0, b.driftHold - dt);
    c.drift = b.driftHold > 0;
  }

  // ── Steering ──────────────────────────────────────────────────────────────

  /** Heading that points at the racing line `dist` metres ahead, offset laterally. */
  private aimHeading(r: Racer, u: number, dist: number, lateral: number): number {
    this.trk.linePoint(u + dist / this.trk.length, lateral, _aim);
    return Math.atan2(_aim.x - r.root.position.x, _aim.z - r.root.position.z);
  }

  /**
   * PD controller on heading error, expressed in the boat's own units.
   *
   * The physics integrates `desiredYawVel = −steer · turnRate · rudder`, so the
   * controller works out the yaw rate it wants, divides by the authority the
   * hull actually has at this speed, and negates. That makes the loop gain
   * independent of speed — a fixed `steer = −err × 1.9` (the placeholder)
   * understeers at 29 m/s and weaves at 8 m/s.
   */
  private steerTo(r: Racer, b: Brain, u: number, lateral: number, dt: number): number {
    const cfg = CONFIG.boat;
    const fwd = Math.abs(r.state.forwardSpeed);
    const lookDist = CONFIG.ai.lookaheadBase + fwd * CONFIG.ai.lookaheadPerSpeed;
    const want = this.aimHeading(r, u, lookDist, lateral);
    const err = angleDelta(r.state.heading, want);

    const turnRate =
      cfg.turnRateLow + (cfg.turnRateHigh - cfg.turnRateLow) * clamp01(fwd / cfg.topSpeed);
    const rudder = 0.12 + 0.88 * smoothstep(0, 5.5, fwd);
    const authority = Math.max(0.25, turnRate * rudder);

    const desiredYaw = err * b.profile.steerP;
    let steer = -(desiredYaw - b.yawRate * b.profile.steerD) / authority;
    steer = clamp(steer, -1, 1);
    // Smooth, at roughly the rate the player's own input ramp uses, so the AI is
    // not steering with a resolution the player cannot match.
    b.steer = damp(b.steer, steer, 14, dt);
    return clamp(b.steer, -1, 1);
  }

  /**
   * Throttle and brake from a speed target.
   *
   * The feed-forward term is the throttle that exactly balances the boat's drag
   * at the target speed — solved from the same two drag coefficients the physics
   * uses — so the controller sits on its target instead of hunting around it.
   * The proportional term only has to cover the difference.
   */
  private applySpeed(
    c: Racer['controls'],
    fwd: number,
    vTarget: number,
    severity: number,
  ) {
    const cfg = CONFIG.boat;
    const dragAccel = cfg.dragLinear * vTarget + cfg.dragQuadratic * vTarget * vTarget;
    const hold = clamp01(dragAccel / (cfg.thrust / cfg.mass));
    const err = vTarget - fwd;
    c.throttle = clamp01(hold + err * 0.3);
    // Brake only when genuinely too fast, and harder in a corner where the extra
    // deceleration is worth the lost speed.
    c.brake = err < -1.6 ? clamp01((-err - 1.6) / (5.5 - severity * 2.5)) : 0;
    if (c.brake > 0.05) c.throttle = 0;
  }

  // ── Traffic ───────────────────────────────────────────────────────────────

  /**
   * Lateral offset to add for other boats.
   *
   * Three behaviours. The first two are temperament, the third is not optional:
   *   • *avoid* — see a hull ahead in your lane, move to the side it is not on.
   *   • *dive*  — see a hull ahead on a corner approach and take the inside line
   *     past it, accepting contact.
   *   • *separate* — a hard, personality-independent push whenever another hull
   *     is inside SEPARATION metres abeam. The hull is 4.6 × 1.9 m and the
   *     physics approximates it with two spheres of radius 1.0, so two centres
   *     3.4 m apart cannot touch whatever the yaw angle. The push saturates to
   *     its full value at that distance rather than ramping from it, because a
   *     term that is still ramping when the hulls meet has already failed —
   *     which is what shots/race_fix0/pack.png shows, red and yellow overlapping
   *     with no visible reaction.
   */
  private trafficOffset(ctx: GameContext, r: Racer, b: Brain, uAhead: number): number {
    const fx = Math.sin(r.state.heading);
    const fz = Math.cos(r.state.heading);
    const rx = -fz;
    const rz = fx;
    const R = CONFIG.ai.avoidRadius;
    /** Centre separation, metres, below which hulls can touch. */
    const SEPARATION = 3.4;
    let push = 0;

    for (const o of ctx.racers) {
      if (o === r) continue;
      const dx = o.root.position.x - r.root.position.x;
      const dz = o.root.position.z - r.root.position.z;
      const along = dx * fx + dz * fz;
      const side = dx * rx + dz * rz;

      // ── Hard separation, both directions along the hull ───────────────────
      // 4.6 m of hull plus a margin: anything inside this box is a contact
      // waiting to happen, in front, behind or alongside.
      if (Math.abs(along) < 5.4 && Math.abs(side) < SEPARATION) {
        const bite = 1 - smoothstep(0, SEPARATION, Math.abs(side));
        // Ties break on racer id, not on the rng: pulling a random number from a
        // per-frame path would make the mistake schedule frame-rate dependent.
        const away = side > 0 ? -1 : side < 0 ? 1 : r.id % 2 === 0 ? 1 : -1;
        push += away * (2.5 + 7.5 * bite);
      }

      if (along < -R || along > 30) continue;

      if (along > 3.0) {
        // Someone ahead. Closeness weight in both axes.
        const wAlong = 1 - smoothstep(6, 30, along);
        const wSide = 1 - smoothstep(1.5, R + 1.5, Math.abs(side));
        const urgency = wAlong * wSide;
        if (urgency <= 0.001) continue;
        // The inside of the coming corner. `signedCurvature > 0` = left turn, so
        // the inside is track-left, i.e. a negative offset.
        const k = this.trk.signedCurvature(uAhead);
        const insideSign = k > 1e-4 ? -1 : k < -1e-4 ? 1 : side > 0 ? -1 : 1;
        const wantDive = b.profile.aggression > 0.5 && Math.abs(k) > 1 / 90;
        const dir = wantDive ? insideSign : side > 0 ? -1 : 1;
        const strength = wantDive ? 5.0 * b.profile.aggression : 6.5 * b.profile.avoid;
        push += dir * urgency * strength;
      } else if (along > -3.0) {
        // Alongside. Keep separating even outside the hard box, so a pair does
        // not grind along each other for a whole straight.
        const overlap = 1 - smoothstep(SEPARATION, 5.6, Math.abs(side));
        if (overlap > 0.001) push += (side > 0 ? -1 : 1) * overlap * 3.0;
      }
    }
    return clamp(push, -11, 11);
  }

  /**
   * Speed cap imposed by the boat directly ahead.
   *
   * Only bites when someone is genuinely in the way — ahead, inside 2.6 m abeam
   * — and it caps at their speed rather than braking hard, so the follower sits
   * in the tow and looks for a way past instead of stamping on the brakes. Once
   * the gap is under a hull length the cap goes 1.8 m/s *below* theirs, because a
   * cap exactly equal to theirs holds a gap it cannot open, and the measured
   * closest approach of a whole 3-lap race was a 2.2 m nose-to-tail overlap. A
   * floor of 9 m/s stops a slow boat ahead from dragging the whole field to a
   * crawl.
   */
  private followCap(ctx: GameContext, r: Racer): number {
    const fx = Math.sin(r.state.heading);
    const fz = Math.cos(r.state.heading);
    const rx = -fz;
    const rz = fx;
    let cap = Infinity;
    for (const o of ctx.racers) {
      if (o === r) continue;
      const dx = o.root.position.x - r.root.position.x;
      const dz = o.root.position.z - r.root.position.z;
      const along = dx * fx + dz * fz;
      if (along < 0.6 || along > 12.0) continue;
      const side = dx * rx + dz * rz;
      if (Math.abs(side) > 2.6) continue;
      // Closer = closer to matching their speed, then dropping below it.
      const tight = 1 - smoothstep(4.6, 12.0, along);
      const theirs = Math.max(9, Math.abs(o.state.forwardSpeed));
      cap = Math.min(cap, theirs - tight * 1.8 + (1 - tight) * 8);
    }
    return cap;
  }

  // ── Mistakes ──────────────────────────────────────────────────────────────

  /**
   * Mistakes only start on the approach to a real corner (severity > 0.3, inside
   * 95 m), because a mistake the player cannot see is indistinguishable from the
   * AI simply being slow. Each one has a cooldown drawn from the personality's
   * mistakes-per-lap so the *rate* is designed rather than emergent.
   */
  private updateMistake(
    b: Brain,
    dt: number,
    severity: number,
    distance: number,
    cornerDir: number,
    raceTime: number,
  ) {
    if (b.mistakeTime > 0) {
      b.mistakeTime -= dt;
      if (b.mistakeTime <= 0) {
        b.mistake = Mistake.None;
        // Roughly `mistakesPerLap` per 60 s lap, jittered.
        const base = 60 / Math.max(0.15, b.profile.mistakesPerLap);
        b.mistakeCooldown = base * b.rng.range(0.55, 1.45);
      }
      return;
    }
    b.mistakeCooldown -= dt;
    if (b.mistakeCooldown > 0) return;
    if (raceTime < 6) return; // never on the opening lap's first corner
    if (severity < 0.3 || distance > 95) return;

    b.mistake = b.rng.pick(b.profile.mistakeKinds);
    switch (b.mistake) {
      case Mistake.WideEntry: {
        b.mistakeDur = b.rng.range(1.8, 3.0);
        // Run wide to the OUTSIDE of the corner. `cornerDir` is +1 for a left
        // turn, whose inside is track-left, so the outside is track-right —
        // which is what makes it read as "he's missed the apex" rather than as
        // a random swerve.
        const out = cornerDir !== 0 ? cornerDir : b.rng.bool() ? 1 : -1;
        b.mistakeLane = b.rng.range(6.5, 11) * out;
        break;
      }
      case Mistake.Overshoot:
        b.mistakeDur = b.rng.range(1.1, 1.7);
        b.mistakeLane = 0;
        break;
      default:
        b.mistakeDur = b.rng.range(0.8, 1.5);
        b.mistakeLane = 0;
        break;
    }
    b.mistakeTime = b.mistakeDur;
  }

  /** Harness introspection: what each AI is doing right now. */
  debug() {
    const out: Record<string, unknown> = {};
    for (const [id, b] of this.brains) {
      out[`r${id}`] = {
        lane: +b.lane.toFixed(2),
        rubber: +b.rubber.toFixed(3),
        mistake: b.mistake,
        mistakeTime: +b.mistakeTime.toFixed(2),
        cooldown: +b.mistakeCooldown.toFixed(1),
        drift: b.driftHold > 0,
        stuck: +b.stuck.toFixed(1),
      };
    }
    return out;
  }
}
