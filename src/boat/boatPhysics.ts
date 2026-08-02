/**
 * Buoyancy, handling, drift/boost and boat-vs-boat collision.
 *
 * ── The shape of the model ─────────────────────────────────────────────────
 * Longitudinal and lateral motion are solved in the *hull frame* (surge and
 * sway), then written back into the world-space `velocity` the camera and FX
 * read. Yaw is a real second-order channel with its own angular velocity, so
 * the boat has rotational inertia — flicking the stick does not teleport the
 * nose. Pitch and roll are second-order too, driven by buoyancy torque about
 * the hull's own axes, which is what produces the slam: the bow buries in a
 * trough, the restoring torque overshoots, the hull kicks its nose up, and the
 * whole thing rings down over about half a second.
 *
 * ── Why probe-based buoyancy and not a height-follow ───────────────────────
 * A single `y = sampleHeight(x, z)` glues the hull to the surface: no pitch, no
 * roll, no slam, no airtime, and the boat visibly slides on glass. Instead six
 * probes on a flat reference plane 26 cm under the origin each report their own
 * submersion depth, and those depths are summed into one vertical force and two
 * torques. The probe layout is deliberately balanced (Σx = 0 and Σz = 0), so
 * flat water produces exactly zero torque and the boat sits level without any
 * corrective fudge.
 *
 * ── Where the airtime comes from, and where it stops ──────────────────────
 * Nothing here launches the boat. The heave channel has a natural frequency of
 * √buoyancy ≈ 8.4 rad/s; a boat at 29 m/s crossing the 41 m swell encounters it
 * at 4.4 rad/s, well inside that band, so the hull genuinely tracks the wave
 * face and genuinely leaves it when the crest's downward acceleration exceeds g.
 * No jump is scripted.
 *
 * What *was* wrong is that nothing bounded it. A spring storing up to 0.85 m of
 * draft against 70 m/s² per metre is a catapult, and measuring 60 s of autopilot
 * racing showed the consequence: the hull was clear of the water 28 % of the
 * time, more than half a metre clear 16 % of the time, and peaked 6.3 m above the
 * local surface. Two of fifteen review frames had the boat hanging over open
 * water with no splash and no contact — and no foam ring can appear around a
 * hull that is a metre in the air, which is why the whole set read as "nothing
 * connects a hull to the water".
 *
 * `maxLaunchSpeed` bounds it. The same 60 s now measures 84 % of the time in
 * contact, 8 % more than half a metre clear, a 3.3 m peak, and 23 separate
 * flights of 0.07–1.07 s. The airtime is still there; the moon jump is not.
 */

import { Quaternion, Vector3 } from 'three';
import { CONFIG } from '../core/config';
import { clamp, clamp01, damp, smoothstep } from '../core/mathx';
import type { BoatState, GameContext, Racer, Subsystem } from '../core/types';
import type { OceanSample } from '../water/gerstner';

const GRAVITY = 9.81;

// ─────────────────────────────────────────────────────────────────────────────
// Hull probes
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Six probes on a flat plane 26 cm below the boat origin — roughly the chine
 * line amidships. They are NOT placed on the modelled keel: a curved probe
 * plane makes flat water produce a standing pitch torque, and then every
 * handling number has to be tuned against a permanently nose-down boat.
 *
 * Σx = 0 and Σz = 0 by construction. Read the table and you can see it.
 */
const PROBE_Y = -0.26;
const PROBES: readonly Vector3[] = [
  new Vector3(0.0, PROBE_Y, 2.0), // bow
  new Vector3(-0.7, PROBE_Y, 0.95),
  new Vector3(0.7, PROBE_Y, 0.95),
  new Vector3(-0.82, PROBE_Y, -0.95),
  new Vector3(0.82, PROBE_Y, -0.95),
  new Vector3(0.0, PROBE_Y, -2.0), // transom
];

/**
 * **Contact points** — the genuinely lowest geometry on the hull, boat-local.
 *
 * These are NOT the buoyancy probes and they exist for one reason: deciding
 * whether the boat has left the water. The probe plane sits at y = -0.26 for the
 * torque reasons above, but the skeg's tip is at y = -0.60 and the keel is at
 * -0.33, so "every probe is 45 cm above the surface" was still true with a third
 * of a metre of skeg dragging through it. That is what put an AIR badge on a
 * planing boat in three of the review frames.
 *
 * Sampled with `ocean.height()` — the same field the water shader displaces with
 * `GERSTNER_GLSL` — so the flag and the rendered surface cannot disagree.
 */
const CONTACTS: readonly Vector3[] = [
  new Vector3(0.0, 0.06, 2.06), // forefoot: matters only when the bow is down
  new Vector3(0.0, -0.33, 0.1), // keel, amidships — deepest point of the shell
  new Vector3(-0.86, -0.05, -1.4), // chine, port — the low point when heeled over
  new Vector3(0.86, -0.05, -1.4), // chine, starboard
  new Vector3(0.0, -0.6, -2.24), // skeg and its cavitation plate: the true low point
];

/** Σz² / n and Σx² / n — the restoring-torque coefficients of the layout above. */
const PITCH_INERTIA = PROBES.reduce((s, p) => s + p.z * p.z, 0) / PROBES.length;
const ROLL_INERTIA = PROBES.reduce((s, p) => s + p.x * p.x, 0) / PROBES.length;

/** Collision proxy: two spheres per hull, which is a capsule in all but name. */
const COLLIDER_Z = 0.95;

// ─────────────────────────────────────────────────────────────────────────────
// Scratch — module scope, never allocated in the frame loop
// ─────────────────────────────────────────────────────────────────────────────

const _probe = new Vector3();
const _quat = new Quaternion();
const _sample: OceanSample = {
  position: new Vector3(),
  normal: new Vector3(0, 1, 0),
  height: 0,
  jacobian: 1,
};
const _ca = new Vector3();
const _cb = new Vector3();

/** Per-racer integrator state that does not belong in the public `BoatState`. */
interface Internal {
  /** Spooled engine output, 0…1. Lags the control input in both directions. */
  engine: number;
  /** Yaw rate, rad/s. */
  yawVel: number;
  pitchVel: number;
  rollVel: number;
  /** Last frame's mean surface height under the hull, for water vertical speed. */
  prevSurfaceY: number;
  /**
   * Raw, per-frame "no part of the hull is touching water". The *physics*
   * branches key off this; `state.airborne` is the deglitched version of it that
   * the HUD, rider, foam and audio see. Keeping them separate is what lets the
   * badge have a minimum duration without the boat keeping its water drag and
   * lateral grip for the first quarter-second of a jump.
   */
  inAir: boolean;
  /** Seconds since `inAir` last went false — the coyote timer. */
  airGap: number;
  /**
   * Ceiling on `velocity.y`, frozen at the frame the hull left the water. A
   * flight's launch speed is decided once; see the launch clamp for why a
   * continuously re-evaluated ceiling is not the same thing.
   */
  launchCeil: number;
  /** Low-passed drift-charge gate, so a twitchy slip angle can't ratchet tiers. */
  slip: number;
  /** Seconds since the last collision, throttles the impact audio. */
  hitCooldown: number;
  /**
   * Steady-state attitude offsets, kept out of `state.pitch` / `state.roll` on
   * purpose: the values other subsystems read stay the *wave-driven* attitude,
   * so a rider leans against the sea rather than against static planing trim.
   */
  trimPitch: number;
  bankRoll: number;
  initialised: boolean;
}

function makeInternal(): Internal {
  return {
    engine: 0,
    yawVel: 0,
    pitchVel: 0,
    rollVel: 0,
    prevSurfaceY: 0,
    inAir: false,
    airGap: 0,
    launchCeil: Infinity,
    slip: 0,
    hitCooldown: 0,
    trimPitch: 0,
    bankRoll: 0,
    initialised: false,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
export class BoatPhysics implements Subsystem {
  readonly name = 'boatPhysics';
  readonly order = 30;

  private internals = new Map<number, Internal>();

  constructor(private racers: Racer[]) {
    for (const r of racers) this.internals.set(r.id, makeInternal());
  }

  update(ctx: GameContext) {
    for (const r of this.racers) this.step(ctx, r);
    this.resolveCollisions(ctx);
    // Attitude is written after collisions so a hit that kicks yaw shows up on
    // the same frame rather than one frame late.
    for (const r of this.racers) {
      const g = this.internals.get(r.id)!;
      const s = r.state;
      const rot = r.root.rotation;
      // Yaw → pitch → roll. Every other subsystem that reads state.pitch /
      // state.roll assumes this order.
      rot.order = 'YXZ';
      rot.y = s.heading;
      rot.x = s.pitch + g.trimPitch;
      rot.z = s.roll + g.bankRoll;
    }
  }

  /** Debug/telemetry hook — the harness reads this when tuning feel. */
  debugState(id: number) {
    const g = this.internals.get(id);
    if (!g) return null;
    return { engine: g.engine, yawVel: g.yawVel, pitchVel: g.pitchVel, rollVel: g.rollVel };
  }

  // ───────────────────────────────────────────────────────────────────────────
  private step(ctx: GameContext, racer: Racer) {
    const dt = ctx.dt;
    if (dt <= 0) return;
    const s = racer.state;
    const c = racer.controls;
    const cfg = CONFIG.boat;
    const g = this.internals.get(racer.id)!;
    const pos = racer.root.position;

    // `main.ts` resets a racer by zeroing its velocity and boost. Detect that
    // and clear our own integrator state, otherwise a restart inherits the yaw
    // and pitch rates from the moment of the reset.
    if (!g.initialised || (s.velocity.lengthSq() === 0 && s.forwardSpeed === 0 && s.boostTime === 0)) {
      g.engine = 0;
      g.yawVel = 0;
      g.pitchVel = 0;
      g.rollVel = 0;
      g.slip = 0;
      s.pitch = 0;
      s.roll = 0;
      g.prevSurfaceY = ctx.ocean.height(pos.x, pos.z, ctx.time);
      // Seat the hull at its equilibrium draft rather than wherever the grid put
      // it. Dropping in from the grid height counts as an airborne frame and
      // fires a landing impact — so the race used to open with a camera shake
      // and a splash before the lights went green.
      pos.y = g.prevSurfaceY - PROBE_Y - cfg.restDraft;
      s.airborne = false;
      s.airTime = 0;
      g.inAir = false;
      g.airGap = 0;
      g.launchCeil = Infinity;
      g.initialised = true;
    }
    g.hitCooldown = Math.max(0, g.hitCooldown - dt);

    // ── Hull frame ──────────────────────────────────────────────────────────
    // `right` matches the sign convention the AI and the input layer expect:
    // positive steer *decreases* heading, i.e. turns toward screen-right.
    const fx = Math.sin(s.heading);
    const fz = Math.cos(s.heading);
    const rx = fz;
    const rz = -fx;

    let surge = s.velocity.x * fx + s.velocity.z * fz;
    let sway = s.velocity.x * rx + s.velocity.z * rz;

    // ── Buoyancy probes ─────────────────────────────────────────────────────
    // Probes are rotated by the *current* attitude, so a rolled hull genuinely
    // presents its lee side deeper and the roll couples into the next frame.
    _quat.setFromEuler(racer.root.rotation);
    let sumSurface = 0;
    let sumDepth = 0;
    let pitchTorque = 0;
    let rollTorque = 0;
    let submerged = 0;

    for (let i = 0; i < PROBES.length; i++) {
      const p = PROBES[i];
      _probe.copy(p).applyQuaternion(_quat).add(pos);
      const surf = ctx.ocean.sample(_probe.x, _probe.z, ctx.time, _sample);
      sumSurface += surf.height;
      const clearance = _probe.y - surf.height;
      if (clearance < 0) {
        const depth = Math.min(-clearance, cfg.maxDraft);
        submerged++;
        sumDepth += depth;
        pitchTorque += depth * p.z;
        rollTorque += depth * p.x;
      }
    }

    const n = PROBES.length;
    const surfaceY = sumSurface / n;
    const meanDepth = sumDepth / n;
    const submergedFrac = submerged / n;
    // Vertical speed of the water itself. Damping the hull against *this* rather
    // than against the world is what lets the boat ride a swell without the
    // damper fighting the wave on the way up.
    const waterVy = clamp((surfaceY - g.prevSurfaceY) / dt, -18, 18);
    g.prevSurfaceY = surfaceY;

    // ── Contact state ───────────────────────────────────────────────────────
    // Measured on the hull's lowest geometry, not on the probe plane, and then
    // deglitched before it is published.
    //
    // In this sea the hull genuinely breaks contact for two or three frames on
    // most crests. Publishing that raw gives a flag that flickers several times
    // a second — an AIR badge that strobes, a rider that twitches into a tuck,
    // and a wake ribbon with a hole punched in it every crest. So there are two
    // states: `g.inAir` (raw, this frame, drives the physics) and
    // `s.airborne` (latches after `airMinDuration`, and only lets go when
    // something actually touches water or the clearance has been gone for
    // `airCoyote`).
    let hullClearance = Infinity;
    for (let i = 0; i < CONTACTS.length; i++) {
      _probe.copy(CONTACTS[i]).applyQuaternion(_quat).add(pos);
      const c = _probe.y - ctx.ocean.height(_probe.x, _probe.z, ctx.time);
      if (c < hullClearance) hullClearance = c;
    }

    const wasInAir = g.inAir;
    const inAir = submerged === 0 && hullClearance > cfg.airborneClearance;
    g.inAir = inAir;

    if (inAir) {
      g.airGap = 0;
      s.airTime += dt;
      if (!s.airborne && s.airTime >= cfg.airMinDuration) {
        // A duration gate alone is not enough. A 0.30 s hop crosses a 0.25 s
        // gate and shows the badge for 50 ms — the very flicker the gate exists
        // to remove. So also require the flight to have time *left* in it:
        // ballistic time to splash from the current clearance and vertical
        // speed, one sqrt, no allocation.
        const h = Math.max(0, hullClearance - cfg.airborneClearance);
        const vy = s.velocity.y;
        const toSplash = (vy + Math.sqrt(vy * vy + 2 * GRAVITY * h)) / GRAVITY;
        if (toSplash >= cfg.airMinDuration) s.airborne = true;
      }
    } else {
      g.airGap += dt;
      if (submerged > 0 || g.airGap >= cfg.airCoyote) {
        // A landing. Crisp — no grace period once water is involved.
        s.airborne = false;
        s.airTime = 0;
      } else {
        // Skimming a crest. Still the same flight.
        s.airTime += dt;
      }
    }

    // ── Longitudinal ────────────────────────────────────────────────────────
    // Engine spool: slower up than down, so stabbing the throttle out of a
    // corner has a beat of lag and the boat feels like it has mass.
    const wantEngine = clamp01(c.throttle);
    g.engine = damp(
      g.engine,
      wantEngine,
      wantEngine > g.engine ? cfg.throttleUpRate : cfg.throttleDownRate,
      dt,
    );
    s.appliedThrottle = g.engine;

    const boosting = s.boostTime > 0;
    // A propeller in air does nothing; a propeller in a trough does everything.
    const bite = inAir ? cfg.airThrust : 0.45 + 0.55 * clamp01(submergedFrac * 1.6);
    let accel = g.engine * (cfg.thrust / cfg.mass) * bite;
    if (boosting) accel += (cfg.boostForce / cfg.mass) * (inAir ? cfg.airThrust : 1);
    accel -= clamp01(c.brake) * (cfg.reverseThrust / cfg.mass) * (inAir ? 0 : 1);

    // Drag: linear + quadratic, tuned so full throttle settles exactly on
    // `topSpeed` and full throttle plus boost settles on `boostTopSpeed`.
    // In the air only the linear term survives, so a good jump is genuinely
    // faster than the water line — which is the whole reward for finding one.
    const dragScale = inAir ? 0.14 : 0.55 + 0.45 * submergedFrac;
    accel -= (cfg.dragLinear * surge + cfg.dragQuadratic * surge * Math.abs(surge)) * dragScale;
    // Plough drag: burying the bow costs you speed. Also the only thing that
    // stops the boat submarining through a wave face at boost speed.
    if (meanDepth > cfg.restDraft) accel -= (meanDepth - cfg.restDraft) * cfg.ploughDrag * clamp01(surge / 6);

    surge = clamp(surge + accel * dt, -9, cfg.boostTopSpeed * 1.15);

    // ── Yaw ─────────────────────────────────────────────────────────────────
    const speedT = clamp01(Math.abs(surge) / cfg.topSpeed);
    // Turn authority falls off with speed: 2.35 rad/s at a crawl, 1.15 flat out.
    let turnRate = cfg.turnRateLow + (cfg.turnRateHigh - cfg.turnRateLow) * speedT;
    // Below walking pace there is no water flowing over the rudder, so the boat
    // can only pivot slowly. Without this the boat spins on the spot at t = 0.
    const rudder = 0.12 + 0.88 * smoothstep(0, 5.5, Math.abs(surge));

    // Powerslide gate. Needs a held drift button, real speed, real steering
    // input and water under the hull — you cannot initiate a slide in mid-air.
    //
    // The speed floor has hysteresis. Without it, a slide that costs the boat
    // speed drops below the threshold, the drift cancels, the charge resets to
    // zero, and the drift re-arms a frame later: the player holds a perfect
    // powerslide for three seconds and never sees a tier. That is not a
    // hypothetical — it is what the telemetry showed before this line existed.
    const speedFloor = s.drifting ? cfg.driftMinSpeed * 0.55 : cfg.driftMinSpeed;
    // A slide also survives a short hop. In this sea the hull is airborne a
    // seventh of the time, so cancelling on every crest would make drifting
    // unusable exactly where the player most wants it.
    const airOk = !inAir || (s.drifting && s.airTime < 0.6);
    const wantDrift =
      c.drift && Math.abs(surge) > speedFloor && Math.abs(c.steer) > 0.2 && airOk;
    if (wantDrift) turnRate *= cfg.driftYawGain;

    const authority = inAir ? cfg.airControl : 1;
    const desiredYawVel = -c.steer * turnRate * rudder * authority;
    g.yawVel = damp(g.yawVel, desiredYawVel, cfg.yawResponse, dt);
    s.heading += g.yawVel * dt;

    // ── Lateral grip ────────────────────────────────────────────────────────
    // Grip is an exponential decay rate on sway, not a force. That makes the
    // difference between gripping and drifting a single, very legible number.
    s.drifting = wantDrift;
    const grip = inAir ? cfg.airGrip : s.drifting ? cfg.driftGrip : cfg.lateralGrip;
    // Sway is *not* generated here. Yawing the hull without rotating the world
    // velocity already means next frame's decomposition finds a lateral
    // component of −surge·Δheading; adding an explicit term on top cancels it
    // almost exactly, which is what made the first build's powerslide produce a
    // measured 0.5 m/s of slip and never charge a single boost tier.
    //
    // So the only thing to do is decide how fast that slip decays. Because the
    // steady-state slip angle is atan(yawRate / grip), grip is a direct handle
    // on the visual: 12 → about 7° of crab in a hard corner, 3.4 → past 30° in
    // a powerslide.
    //
    // Killing sway outright also destroys the speed it represents, and that bill
    // is enormous: a 7 m/s slide at grip 3.8 bleeds 15 m/s² — more than the
    // engine makes — so a three-second powerslide measured a 55 % speed loss and
    // spat the boat out slower than it went in. Real water redirects most of
    // that momentum along the hull instead of deleting it, so a fraction of the
    // speed grip takes is handed back as surge. The remainder is the deliberate
    // cost of the slide: about 20 % at full drift, almost nothing when gripping.
    const speedBefore = Math.hypot(surge, sway);
    sway *= Math.exp(-grip * dt);
    const recovered =
      (speedBefore - Math.hypot(surge, sway)) *
      (s.drifting ? cfg.driftRecovery : cfg.gripRecovery);
    surge += surge < 0 ? -recovered : recovered;
    s.lateralSpeed = sway;

    // ── Drift charge → boost ────────────────────────────────────────────────
    const slipping = Math.abs(sway) > cfg.driftSlipThreshold;
    g.slip = damp(g.slip, slipping && s.drifting ? 1 : 0, 9, dt);
    if (s.drifting) {
      // Charge rate scales with how hard the boat is actually sliding, so a
      // half-hearted drift is a slow charge. Holding the button while gripping
      // charges nothing at all.
      s.driftCharge += dt * g.slip * (0.55 + 0.45 * clamp01(Math.abs(sway) / 7));
      let tier = 0;
      for (let i = 0; i < cfg.driftTiers.length; i++) if (s.driftCharge >= cfg.driftTiers[i]) tier = i + 1;
      s.driftTier = tier;
      s.boostMeter = clamp01(s.driftCharge / cfg.driftTiers[cfg.driftTiers.length - 1]);
    } else if (c.drift && Math.abs(c.steer) > 0.2 && inAir) {
      // Drift interrupted by a long flight rather than by the player. Freeze the
      // charge instead of firing or dumping it: losing a tier-3 charge because a
      // crest threw you into the air mid-corner reads as the game cheating.
      s.boostMeter = clamp01(s.driftCharge / cfg.driftTiers[cfg.driftTiers.length - 1]);
    } else {
      if (s.driftTier > 0) {
        s.boostTime = cfg.boostDuration[s.driftTier - 1];
        // Releasing a drift also snaps some of the surviving sway into surge —
        // the payoff feels like being fired out of the corner, not like a
        // stat bonus.
        surge += Math.abs(sway) * cfg.driftExitKick;
        ctx.audio.boost();
        if (racer.isPlayer) ctx.cameraRig.addShake(0.12 + 0.06 * s.driftTier);
      }
      s.driftCharge = 0;
      s.driftTier = 0;
      s.boostMeter = boosting
        ? clamp01(s.boostTime / cfg.boostDuration[cfg.boostDuration.length - 1])
        : damp(s.boostMeter, 0, 6, dt);
    }
    s.boostTime = Math.max(0, s.boostTime - dt);

    // ── Write the horizontal velocity back ──────────────────────────────────
    s.velocity.x = fx * surge + rx * sway;
    s.velocity.z = fz * surge + rz * sway;
    s.forwardSpeed = surge;

    // ── Heave ───────────────────────────────────────────────────────────────
    // Buoyant acceleration is proportional to mean submersion; `buoyancy` is
    // literally "m/s² of lift per metre of draft", so the rest draft the boat
    // settles at is g / buoyancy and nothing needs to be hand-placed.
    const lift = Math.min(meanDepth * cfg.buoyancy, cfg.maxBuoyantAccel);
    s.velocity.y += (lift - GRAVITY) * dt;
    if (submergedFrac > 0) {
      const rel = s.velocity.y - waterVy;
      // Weighted by submersion, but with a floor: one probe in the water is
      // still a hull in the water, and at the raw 1/6 the spring gave back
      // almost everything it stored on the way down.
      const w = Math.max(submergedFrac, cfg.buoyancyDampFloor);
      s.velocity.y -= rel * cfg.buoyancyDamping * w * dt;
    }
    // Planing lift: at speed the hull climbs onto its own bow wave and rides
    // visibly higher. It is a small number that does a lot of the "fast" read.
    if (!inAir) s.velocity.y += cfg.planingLift * surge * surge * submergedFrac * dt;

    // ── Launch clamp ────────────────────────────────────────────────────────
    // A ceiling on how hard the sea may throw the hull. See `maxLaunchSpeed` in
    // config for the measurements that made this necessary: without it the
    // buoyancy spring behaved as a catapult and the boat spent more than a
    // quarter of the race in the air, peaking 6.3 m up, which is what put an AIR
    // badge on the money shot and left every frame with a hull that touches
    // nothing.
    //
    // Two clamps, and the split matters:
    //
    //  • **In contact** the ceiling is relative to `waterVy`, so the hull can
    //    still climb a wave face at the ~10 m/s the encounter rate demands at
    //    29 m/s. Clamping absolute vertical speed here would glue the boat to the
    //    mean water plane and delete the ride entirely. This is the clamp that
    //    does the work: it stops the spring winding up energy it cannot use.
    //
    //  • **Airborne** the ceiling is frozen at the value it had on the frame the
    //    hull separated. A single continuous relative clamp looked equivalent and
    //    was not: a crest rising under a flying hull pushes `waterVy` to +9 m/s,
    //    which lifted the ceiling to 13.6 and re-opened the catapult. Flight is
    //    ballistic; its launch speed is decided once.
    if (!wasInAir && inAir) g.launchCeil = waterVy + cfg.maxLaunchSpeed;
    const ceil = inAir ? g.launchCeil : waterVy + cfg.maxLaunchSpeed;
    if (submergedFrac > 0 || inAir) s.velocity.y = Math.min(s.velocity.y, ceil);

    // ── Integrate position ──────────────────────────────────────────────────
    pos.addScaledVector(s.velocity, dt);

    // Hard floor: never let the hull tunnel through the wave field.
    const floor = surfaceY - cfg.maxDraft;
    if (pos.y < floor) {
      pos.y = floor;
      if (s.velocity.y < waterVy) s.velocity.y = waterVy;
    }
    // And never let it stay in orbit.
    if (pos.y > surfaceY + 14) {
      pos.y = surfaceY + 14;
      if (s.velocity.y > 0) s.velocity.y = 0;
    }

    // ── Landing ─────────────────────────────────────────────────────────────
    // Fired on the *raw* transition, not the published one: the pitch recoil and
    // the splash belong to the moment the water is actually hit.
    s.landingImpact = 0;
    if (wasInAir && !inAir) {
      const impact = Math.max(0, waterVy - s.velocity.y);
      s.landingImpact = impact;
      // The bow is usually down on re-entry, so the water kicks the nose back
      // up. Driving pitchVel rather than pitch means the recoil rings out over
      // several frames instead of snapping.
      g.pitchVel -= impact * cfg.slamPitchKick;
      surge *= 1 - clamp01(impact / 34) * 0.22;
      s.forwardSpeed = surge;
      s.velocity.x = fx * surge + rx * sway;
      s.velocity.z = fz * surge + rz * sway;
      if (impact > 1.6) {
        // Progressive, not linear. The hull lands ~15 times a minute at racing
        // pace; a linear curve gave every little re-entry a 0.3 shake and the
        // camera never stopped wobbling. Raised to the 1.5 power, taps are
        // almost free and a real slam is unmistakable.
        const strength = clamp01(impact / 9);
        const punch = Math.pow(strength, 1.5);
        if (racer.isPlayer) ctx.cameraRig.addShake(punch * 0.85);
        ctx.audio.impact(strength);
        ctx.audio.splash(clamp01(0.3 + strength * 0.7));
      }
    }

    // ── Pitch and roll ──────────────────────────────────────────────────────
    // Second-order about the hull axes. The restoring coefficient is the probe
    // layout's own inertia, so tuning `pitchStiffness` reads directly as a
    // natural frequency: ω = √(pitchStiffness · PITCH_INERTIA).
    if (inAir) {
      // Ballistic: the hull follows its velocity vector, nose dropping as it
      // falls. Reduced authority, so a launch reads as a launch.
      const horiz = Math.hypot(s.velocity.x, s.velocity.z);
      const ballistic = clamp(Math.atan2(-s.velocity.y, Math.max(horiz, 3)) * 0.55, -0.4, 0.34);
      g.pitchVel = damp(g.pitchVel, (ballistic - s.pitch) * 3.4, 5, dt);
      g.rollVel = damp(g.rollVel, -s.roll * 2.4, 4, dt);
    } else {
      const pitchAccel =
        -cfg.pitchStiffness * (pitchTorque / n) - cfg.pitchDamping * g.pitchVel;
      g.pitchVel += pitchAccel * dt;
      const rollAccel = cfg.rollStiffness * (rollTorque / n) - cfg.rollDamping * g.rollVel;
      g.rollVel += rollAccel * dt;
    }
    s.pitch = clamp(s.pitch + g.pitchVel * dt, -0.62, 0.56);
    s.roll = clamp(s.roll + g.rollVel * dt, -0.66, 0.66);

    // Trim and bank are steady-state offsets layered on top of the dynamics,
    // not part of the spring — otherwise they get eaten by the damping.
    const bank = clamp(-cfg.bankGain * g.yawVel * surge, -0.36, 0.36);
    const slipLean = clamp(sway * cfg.slipLeanGain, -0.2, 0.2);
    g.trimPitch = damp(g.trimPitch, -cfg.planeTrim * speedT * speedT, 3.5, dt);
    g.bankRoll = damp(g.bankRoll, bank + slipLean, 6.5, dt);

    s.speedFrac = clamp01(Math.hypot(s.velocity.x, s.velocity.z) / cfg.boostTopSpeed);
  }

  // ───────────────────────────────────────────────────────────────────────────
  /**
   * Boat-vs-boat. Four racers means six pairs and four sphere-pairs each, so
   * twenty-four distance tests per frame — a spatial structure here would be
   * pure ceremony. Resolution is a positional split plus an elastic-ish impulse
   * along the contact normal, and a yaw kick proportional to the moment arm so
   * a stern-quarter hit spins you the way it should.
   */
  private resolveCollisions(ctx: GameContext) {
    const cfg = CONFIG.boat;
    const r = cfg.collisionRadius;
    const racers = this.racers;

    for (let i = 0; i < racers.length; i++) {
      for (let j = i + 1; j < racers.length; j++) {
        const A = racers[i];
        const B = racers[j];
        // Cheap reject on hull centres before touching the sphere pairs.
        const cdx = B.root.position.x - A.root.position.x;
        const cdz = B.root.position.z - A.root.position.z;
        const reach = COLLIDER_Z + r;
        if (cdx * cdx + cdz * cdz > 4 * reach * reach) continue;

        for (let a = 0; a < 2; a++) {
          for (let b = 0; b < 2; b++) {
            colliderCentre(A, a === 0 ? COLLIDER_Z : -COLLIDER_Z, _ca);
            colliderCentre(B, b === 0 ? COLLIDER_Z : -COLLIDER_Z, _cb);
            let dx = _cb.x - _ca.x;
            let dz = _cb.z - _ca.z;
            let d2 = dx * dx + dz * dz;
            const min = r * 2;
            if (d2 >= min * min) continue;

            let d = Math.sqrt(d2);
            if (d < 1e-4) {
              // Exactly coincident: pick a deterministic escape direction so the
              // pair cannot get stuck jittering.
              dx = 1;
              dz = 0;
              d = 1;
            }
            const nx = dx / d;
            const nz = dz / d;
            const overlap = min - d;

            A.root.position.x -= nx * overlap * 0.5;
            A.root.position.z -= nz * overlap * 0.5;
            B.root.position.x += nx * overlap * 0.5;
            B.root.position.z += nz * overlap * 0.5;

            const rvx = B.state.velocity.x - A.state.velocity.x;
            const rvz = B.state.velocity.z - A.state.velocity.z;
            const closing = rvx * nx + rvz * nz;
            if (closing < 0) {
              const jimp = -(1 + cfg.collisionRestitution) * closing * 0.5;
              A.state.velocity.x -= nx * jimp;
              A.state.velocity.z -= nz * jimp;
              B.state.velocity.x += nx * jimp;
              B.state.velocity.z += nz * jimp;

              // Yaw kick from the moment arm of the contact point.
              const gA = this.internals.get(A.id)!;
              const gB = this.internals.get(B.id)!;
              const armA = a === 0 ? 1 : -1;
              const armB = b === 0 ? 1 : -1;
              const spin = clamp(-closing * cfg.collisionSpin, 0, 2.2);
              const sideA = Math.sign(nx * Math.cos(A.state.heading) - nz * Math.sin(A.state.heading));
              const sideB = Math.sign(nx * Math.cos(B.state.heading) - nz * Math.sin(B.state.heading));
              gA.yawVel += spin * armA * sideA * -1;
              gB.yawVel += spin * armB * sideB;

              for (const racer of [A, B]) {
                const g = this.internals.get(racer.id)!;
                if (g.hitCooldown > 0) continue;
                g.hitCooldown = 0.18;
                const strength = clamp01(-closing / 16);
                ctx.audio.impact(strength * 0.7);
                if (racer.isPlayer) ctx.cameraRig.addShake(0.14 + strength * 0.4);
              }
            }
          }
        }
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────

function colliderCentre(racer: Racer, localZ: number, out: Vector3) {
  const h = racer.state.heading;
  out.set(
    racer.root.position.x + Math.sin(h) * localZ,
    racer.root.position.y,
    racer.root.position.z + Math.cos(h) * localZ,
  );
}

export type { BoatState };
