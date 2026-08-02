/**
 * Chase camera.
 *
 * Spring-damped position and look-at, FOV that opens with speed, a dutch-tilt
 * roll that leans into the slide, and an impulse-driven shake for slams. Also
 * hosts the fixed presets the screenshot harness drives, so every captured frame
 * comes from a real in-game camera rather than a bespoke debug view.
 *
 * Three ideas do most of the work:
 *
 *  1. **Yaw follows the direction of travel, not the hull.** During a powerslide
 *     the rig biases hard toward the *velocity* heading, which leaves the hull
 *     visibly rotated inside the frame. Snapping to the boat's heading hides the
 *     slide, and hiding the slide is what makes a drift feel like nothing.
 *  2. **The boat sits low and the horizon sits high-ish.** A low subject leaves
 *     room for the wake to read, and pinning the look-at slightly above the hull
 *     keeps the water plane sweeping through frame, which is where the sense of
 *     speed actually comes from.
 *  2b. **Vertically the rig is sprung, not bolted.** See `FOLLOW_Y_RATE`: the
 *     camera lags the hull's height, so airtime, a slam and the swell all read as
 *     the boat moving inside the frame. A rigid vertical follow makes every one of
 *     them invisible.
 *  3. **Everything is damped, nothing is snapped.** Position, yaw, look-at, FOV
 *     and roll all use `damp()`, so behaviour is identical at 30 and 144 fps.
 *
 * Presets are a contract with the other subsystems' capture runs: `chase`, `far`,
 * `bow`, `wake`, `rider`, `orbit_near`, `far_boat`, `broadcast`, `aerial`, `sky`,
 * `auto`. Do not change their framing without telling the owners of water, boat
 * and rider — their before/after comparisons depend on it.
 */

import { PerspectiveCamera, Vector3 } from 'three';
import { CONFIG } from '../core/config';
import { clamp, clamp01, damp, dampAngle } from '../core/mathx';
import type { CameraRig, GameContext } from '../core/types';

export type CameraMode = 'chase' | 'orbit' | 'cinematic' | 'far' | 'bow';

/** Fixed rigs the harness can request by name. */
export type CameraPreset =
  | 'chase' | 'far' | 'bow' | 'wake' | 'rider'
  | 'orbit_near' | 'far_boat' | 'broadcast' | 'aerial' | 'sky' | 'auto';

/**
 * Framing trim on top of `CONFIG.camera`. Kept here rather than in config
 * because it is a composition choice owned by this rig; the config numbers are
 * the shared "how far back is a chase camera" default.
 */
const CHASE_DIST_TRIM = 0.58;
/**
 * Height trim. Dropped from 0.62 (2.6 m over the hull, ~3.5 m over the water)
 * to 0.44 (1.8 m / ~2.7 m) after a skyline measurement: shots/r3/hero.png — the
 * only shot in the set framed by this rig — had a skyline deviation range of
 * 48 px / stdev 13.1, against 90–190 px for every other framing of the same
 * shader (bow 90, hud 131, ocean_wide 188). A camera looking *down* at swell
 * foreshortens it into painted markings on a plane; the lower the rig, the more
 * the crests stand up against the sky and break the horizon line.
 */
const CHASE_HEIGHT_TRIM = 0.44;

/**
 * Vertical follow rates, 1/s. The rig tracks the hull's height through a damp,
 * **not** rigidly, and this is load-bearing rather than a smoothing nicety:
 * with the camera pinned to `boat.y` the hull cannot move vertically in frame, so
 * a 0.3 s launch off a crest looked exactly like level cruising — no gap under
 * the keel, no separation from the water (shots/pres_base/air.png). Airborne the
 * rate is dropped almost to zero, so the boat visibly climbs out of frame centre
 * and the horizon stays put.
 */
const FOLLOW_Y_RATE = 4.2;
const FOLLOW_Y_RATE_AIR = 1.0;

const _target = new Vector3();
const _desired = new Vector3();
const _look = new Vector3();
const _tmp = new Vector3();
const _orbitPos = new Vector3();
const _orbitLook = new Vector3();

export class ChaseCamera implements CameraRig {
  readonly camera: PerspectiveCamera;

  private mode: CameraMode = 'chase';
  private preset: CameraPreset | null = null;
  private pos = new Vector3(0, 6, -14);
  private vel = new Vector3();
  private lookAt = new Vector3();
  private yaw = 0;
  private shake = 0;
  /**
   * Fixed, not `Math.random()`. The shake displaces the camera, so a random seed
   * makes any frame captured during a slam irreproducible — and the harness's
   * whole value is that the same shot name yields the same pixels.
   */
  private shakeSeed = 37.13;
  private orbitAngle = 0;
  private baseFov = CONFIG.render.fov;
  /** Current dutch tilt, radians. Damped, applied after lookAt(). */
  private roll = 0;
  /** Lateral rail offset, metres. Slides the rig to the outside of a drift. */
  private railOffset = 0;
  /** Extra FOV punched in on the frame a boost fires, decayed out. */
  private fovPunch = 0;
  private lastBoostTime = 0;
  /** Damped height the chase rig hangs off. See FOLLOW_Y_RATE. */
  private followY = 0;
  /**
   * 0…1 while the hull is out of the water. The rig ducks and pulls back on it,
   * which is what actually puts sky and horizon *under the keel* — a chase camera
   * that keeps looking down at a launched boat shows the same silhouette it shows
   * on flat water.
   */
  private airLift = 0;
  /**
   * Water re-entry recoil, 1 → 0. Drives a camera dip and a tilt spike, both of
   * which are visible in a *still* frame — a positional shake is not, which is
   * why a landing capture could contain a slam and show nothing.
   */
  private landKick = 0;
  /**
   * Set to `ctx.frame` by `applyCinematicOrbit`. `main.ts` calls the orbit and
   * then `update()` in the same frame; without this latch `update()` would run
   * the chase rig straight over the orbit it was just handed, which is exactly
   * what used to happen — the countdown "orbit" was really a chase cam fighting
   * an orbit and losing.
   */
  private cinematicFrame = -1;
  private snapNext = true;

  constructor(aspect: number) {
    this.camera = new PerspectiveCamera(
      CONFIG.render.fov,
      aspect,
      CONFIG.render.near,
      CONFIG.render.far,
    );
    this.camera.position.copy(this.pos);
  }

  setMode(mode: CameraMode) {
    this.mode = mode;
    this.preset = null;
  }

  /** Harness entry point. `auto` means "whatever the race phase wants". */
  setPreset(preset: CameraPreset) {
    this.preset = preset === 'auto' ? null : preset;
    if (preset === 'auto') this.mode = 'chase';
  }

  addShake(amount: number) {
    this.shake = clamp01(this.shake + amount);
  }

  snapToTarget() {
    this.vel.set(0, 0, 0);
    this.snapNext = true;
    this.roll = 0;
    this.railOffset = 0;
    this.fovPunch = 0;
    this.shake = 0;
    this.landKick = 0;
  }

  update(ctx: GameContext) {
    const { dt } = ctx;
    const boat = ctx.player;
    const s = boat.state;

    // Camera targets the hull a little above its origin so the boat sits low
    // in frame and the horizon stays visible — a low horizon reads as speed.
    _target.copy(boat.root.position);
    _target.y += 0.9;

    // Landing recoil. Latched off `landingImpact`, which is set for exactly one
    // frame, so it has to be captured here and decayed rather than read live.
    // Normalised against 3.5 m/s rather than the physics slam threshold: an
    // everyday re-entry has to produce a *visible* reaction, or a landing capture
    // contains a landing and shows nothing.
    // Normalised against 2.0 m/s, not 3.5: the harness hunts `landingImpact > 1.2`
    // and captures two frames later, which at the old normalisation left the rig
    // reacting at 0.3 of full strength — a 1.6° FOV change and a 24 cm dip, i.e. a
    // frame captioned "water re-entry, camera shake" that contained no visible
    // reaction at all (shots/pres2_r3/land.png). A slam also fires the shake
    // impulse; nothing else in the game calls `addShake`, so if the camera does not
    // arm it on re-entry, the shake channel is dead code.
    if (s.landingImpact > 0.4) {
      const k = clamp01(s.landingImpact / 2.0);
      if (k > this.landKick) this.addShake(k * 0.55);
      this.landKick = Math.max(this.landKick, k);
    }
    this.landKick = damp(this.landKick, 0, 3.6, dt);
    this.airLift = damp(this.airLift, s.airborne ? 1 : 0, s.airborne ? 6.5 : 3.4, dt);

    this.orbitAngle += dt * 0.32;

    if (this.preset) {
      this.applyPreset(ctx, _target);
    } else if (this.cinematicFrame === ctx.frame) {
      this.camera.position.copy(_orbitPos);
      this.lookAt.copy(_orbitLook);
      this.roll = damp(this.roll, 0, 4, dt);
    } else {
      this.applyChase(ctx, _target);
    }

    // ── FOV kick ────────────────────────────────────────────────────────────
    // Opens with speed, and opens further during a boost. This is most of what
    // makes an arcade racer feel fast; the actual velocity change is secondary.
    // The punch on the frame a boost fires is short and violent on purpose: a
    // ramp is invisible, a snap-and-settle is felt.
    if (s.boostTime > this.lastBoostTime + 0.01) this.fovPunch = 5.5;
    this.lastBoostTime = s.boostTime;
    this.fovPunch = damp(this.fovPunch, 0, 6.5, dt);

    const boosting = s.boostTime > 0 ? 1 : 0;
    const targetFov =
      this.baseFov +
      CONFIG.render.fovSpeedKick * s.speedFrac +
      boosting * 6.5 +
      this.fovPunch +
      // A landing throws the frame open for a beat. Unlike a shake this survives
      // a screenshot.
      this.landKick * 9;
    this.camera.fov = damp(this.camera.fov, targetFov, 5.5, dt);
    this.camera.updateProjectionMatrix();

    // ── Shake ───────────────────────────────────────────────────────────────
    if (this.shake > 0.0005) {
      const t = ctx.time * 34 + this.shakeSeed;
      const amp = this.shake * this.shake * 0.42; // squared → punchy, not woolly
      this.camera.position.x += Math.sin(t * 1.7) * amp;
      this.camera.position.y += Math.sin(t * 2.3 + 1.1) * amp;
      this.camera.position.z += Math.sin(t * 1.9 + 2.7) * amp;
      this.shake = damp(this.shake, 0, CONFIG.camera.shakeDecay, dt);
    }

    // Never let the camera dip below the water surface — a frame of underwater
    // camera is far more jarring than a slightly high camera. Sampled at the
    // final position, after the shake, so a slam cannot punch through.
    const surface = ctx.ocean.height(this.camera.position.x, this.camera.position.z, ctx.time);
    const minY = surface + 0.85;
    if (this.camera.position.y < minY) this.camera.position.y = minY;

    this.camera.lookAt(this.lookAt);
    // Roll last: lookAt() rebuilds the quaternion from scratch, so a dutch tilt
    // has to be applied on top of it rather than baked into the up vector.
    if (Math.abs(this.roll) > 1e-4) this.camera.rotateZ(this.roll);
    this.snapNext = false;
  }

  // ── Modes ─────────────────────────────────────────────────────────────────

  private applyChase(ctx: GameContext, target: Vector3) {
    const { dt } = ctx;
    const s = ctx.player.state;
    const cfg = CONFIG.camera;

    // Trail the direction of travel. During a drift we bias hard toward the
    // *velocity* heading so the hull visibly rotates within the frame; the rig
    // also gets looser (lower stiffness) so it arrives late and you see the
    // rotation happen rather than finding the boat already sideways.
    const vel = s.velocity;
    const speed = Math.hypot(vel.x, vel.z);
    const velHeading = Math.atan2(vel.x, vel.z);
    const driftAmt = s.drifting ? clamp01(Math.abs(s.lateralSpeed) / 5) : 0;
    // Almost fully aligned to the direction of travel in a committed drift: the
    // crab angle the physics produces is only 10–25 deg, so anything less than
    // ~0.9 blend and the rotation is not legible in frame at all.
    const blend = speed > 3 ? (s.drifting ? 0.55 + driftAmt * 0.42 : 0.24) : 0;
    let desiredYaw = s.heading;
    if (blend > 0) {
      const d = Math.atan2(Math.sin(velHeading - desiredYaw), Math.cos(velHeading - desiredYaw));
      desiredYaw += d * blend;
    }
    const rotRate = cfg.rotStiffness * (s.drifting ? 0.62 : 1) * (s.airborne ? 0.7 : 1);
    this.yaw = this.snapNext ? desiredYaw : dampAngle(this.yaw, desiredYaw, rotRate, dt);

    // Pull back and up as speed rises.
    //
    // The two trim factors below are a framing decision, not a physics one: at
    // the config's raw 11.2 m / 4.15 m the hull was ~8 % of frame height and sat
    // as a red dot in the middle of the sea (shots/pres_r0/hero.png). Tighter and
    // lower puts the boat at a readable size and lifts the horizon, which is
    // where the sense of speed comes from.
    //
    // They were tightened again after measuring the hull at 194 of 2880 px across
    // (6.7 % of frame width) in shots/presentation_fix/hero.png. The lever here is
    // distance, deliberately *not* FOV: the field of view is shared by every
    // harness preset, and narrowing it would silently reframe the water, boat and
    // rider subsystems' capture sets as well as flattening the speed cue.
    const dist =
      cfg.distance * CHASE_DIST_TRIM * (1 + s.speedFrac * 0.22) + this.airLift * 1.5;
    const height =
      cfg.height * CHASE_HEIGHT_TRIM * (1 + s.speedFrac * 0.1) - this.airLift * 1.5;

    // Damped vertical follow, and the landing dip on top of it. Everything the
    // rig hangs off vertically comes from `followY`, so the hull's own height is
    // free to read as height in frame.
    const yRate = s.airborne ? FOLLOW_Y_RATE_AIR : FOLLOW_Y_RATE;
    this.followY = this.snapNext ? target.y : damp(this.followY, target.y, yRate, dt);
    const baseY = this.followY - this.landKick * 1.7;

    // Rail offset: slide the rig toward the outside of the slide so the drift is
    // seen across the hull's flank instead of down its centreline.
    const railTarget = clamp(-s.lateralSpeed * 0.42, -3.4, 3.4) * (s.drifting ? 1 : 0.3);
    this.railOffset = damp(this.railOffset, railTarget, 3.2, dt);
    const rightX = Math.cos(this.yaw);
    const rightZ = -Math.sin(this.yaw);

    _desired.set(
      target.x - Math.sin(this.yaw) * dist + rightX * this.railOffset,
      baseY + height,
      target.z - Math.cos(this.yaw) * dist + rightZ * this.railOffset,
    );

    // Critically-damped follow. A hard snap on reset stops the rig winding in
    // from the previous position for the first second of a scripted capture.
    if (this.snapNext) this.camera.position.copy(_desired);
    else this.smoothFollow(_desired, cfg.posStiffness, dt);

    // Look ahead of the boat, further at speed.
    _look.set(
      target.x + Math.sin(this.yaw) * cfg.lookAhead * (0.5 + s.speedFrac),
      // Airborne, the look point drops as well as the rig, so the hull climbs
      // toward the top of frame instead of the whole composition translating.
      baseY + 0.4 - s.speedFrac * 0.5 - this.airLift * 0.8,
      target.z + Math.cos(this.yaw) * cfg.lookAhead * (0.5 + s.speedFrac),
    );
    if (this.snapNext) this.lookAt.copy(_look);
    else this.lookAt.lerp(_look, 1 - Math.exp(-8 * dt));

    // Dutch tilt. Driven by slip and by how hard the rig itself is rotating —
    // both of those are "the world is being thrown sideways", which is what the
    // tilt is meant to say. Capped low; past ~3.5° it reads as a bug.
    const yawErr = Math.atan2(Math.sin(desiredYaw - this.yaw), Math.cos(desiredYaw - this.yaw));
    let rollTarget =
      clamp(s.lateralSpeed * 0.014 + yawErr * 0.6, -0.075, 0.075) * (0.35 + s.speedFrac * 0.65);
    // Re-entry tilt spike, signed by which way the hull was already leaning so it
    // reads as the slam knocking the rig rather than as a random jolt.
    rollTarget += this.landKick * 0.075 * (s.roll >= 0 ? 1 : -1);
    // Ceiling on the total. Steady-state drift roll is capped at 4.3° — past that
    // a *held* tilt reads as a bug — but a slam is allowed to knock the horizon
    // to ~8° for the fifth of a second the kick lasts, because that spike is the
    // only part of a landing that survives a screenshot.
    const cap = 0.075 + this.landKick * 0.065;
    this.roll = damp(this.roll, clamp(rollTarget, -cap, cap), 4.6, dt);
  }

  private smoothFollow(desired: Vector3, stiffness: number, dt: number) {
    // Exponential approach on position with a velocity term, which gives the
    // slight overshoot-free "weight" a raw lerp lacks.
    const f = 1 - Math.exp(-stiffness * dt);
    _tmp.copy(desired).sub(this.camera.position).multiplyScalar(f);
    this.camera.position.add(_tmp);
  }

  private applyPreset(ctx: GameContext, target: Vector3) {
    const p = ctx.player;
    const h = p.state.heading;
    const sin = Math.sin(h),
      cos = Math.cos(h);
    const cam = this.camera;
    this.roll = damp(this.roll, 0, 6, ctx.dt);

    switch (this.preset) {
      case 'chase':
        this.applyChase(ctx, target);
        return;

      case 'far':
        cam.position.set(target.x - sin * 46, target.y + 24, target.z - cos * 46);
        this.lookAt.set(target.x + sin * 40, target.y - 2, target.z + cos * 40);
        return;

      case 'bow':
        // Low, just off the bow, near the waterline — shows the wave silhouette.
        cam.position.set(target.x + sin * 5.4 + cos * 2.2, target.y + 0.55, target.z + cos * 5.4 - sin * 2.2);
        this.lookAt.copy(target).setY(target.y + 0.2);
        return;

      case 'wake':
        // Behind and low, looking down the wake ribbon.
        cam.position.set(target.x - sin * 13, target.y + 2.4, target.z - cos * 13);
        this.lookAt.copy(target).setY(target.y - 0.35);
        return;

      case 'rider':
        // Three-quarter close-up on the rider.
        cam.position.set(target.x - sin * 3.6 - cos * 3.1, target.y + 1.75, target.z - cos * 3.6 + sin * 3.1);
        this.lookAt.set(target.x, target.y + 1.05, target.z);
        return;

      case 'orbit_near': {
        const a = this.orbitAngle;
        cam.position.set(target.x + Math.sin(a) * 7.5, target.y + 2.6, target.z + Math.cos(a) * 7.5);
        this.lookAt.copy(target).setY(target.y + 0.7);
        return;
      }

      case 'far_boat':
        // Deliberately the same subject as orbit_near, much further away, so
        // the two captures can be compared for constant outline width.
        cam.position.set(target.x + 74, target.y + 17, target.z + 74);
        this.lookAt.copy(target);
        return;

      case 'broadcast':
        cam.position.set(target.x - sin * 30 + cos * 20, target.y + 13, target.z - cos * 30 - sin * 20);
        this.lookAt.copy(target);
        return;

      case 'aerial':
        cam.position.set(target.x, target.y + 210, target.z - 40);
        this.lookAt.copy(target);
        return;

      case 'sky':
        cam.position.set(target.x - sin * 12, target.y + 3.5, target.z - cos * 12);
        this.lookAt.set(target.x - sin * 60, target.y + 42, target.z - cos * 60);
        return;

      default:
        this.applyChase(ctx, target);
    }
  }

  /**
   * Cinematic orbit used during the countdown and on the results screen.
   *
   * Called by `main.ts` immediately before `update()`. It only *stages* the
   * result — `update()` consumes it via the `cinematicFrame` latch — so the
   * clamp, the shake and the FOV logic still run over the top of it and there is
   * exactly one place that writes the camera transform.
   *
   * The framing is not a constant-radius turntable. During the countdown it
   * dollies in and drops toward the waterline as the lights run down, which puts
   * the grid low and wide at "3" and tight on the player's shoulder by "1". On
   * the results board it pulls out and rises, letting the board breathe.
   */
  applyCinematicOrbit(ctx: GameContext, radius = 13, height = 4.2, speed = 0.28) {
    const target = ctx.player.root.position;
    const phase = ctx.race.phase;

    let r = radius;
    let hgt = height;
    let spd = speed;
    /** Raising the look point drops the subject in frame. */
    let lookLift = 0.95;
    /**
     * Shifting the look point along camera-right pushes the subject to the
     * *left* of frame. The results board lives on the right, so this is what
     * keeps the celebrating boat out from behind it.
     */
    let lookBias = 0;
    /** Vertical bob amplitude, metres. Small on a low hero framing. */
    let bob = 1.2;
    if (phase === 'countdown') {
      // 0 → 1 across the countdown.
      const k = clamp01(1 + ctx.race.raceTime / CONFIG.race.countdownSeconds);
      r = radius * (1.55 - 0.62 * k);
      hgt = height * (1.28 - 0.62 * k);
      spd = speed * (1.5 - 0.7 * k);
    } else if (phase === 'results' || phase === 'finished') {
      // Results: a three-quarter, slightly-low hero framing.
      //
      // This used to sit at 4.4 m over a 9.6 m radius, i.e. a 25° downward look —
      // near enough plan view that the winning boat read as a deck plan and the
      // rider's raised-fist celebration was seen from above (shots/pres2_base/
      // results.png). 2.3 m over 8.7 m is a 15° look, which is a hero shot: you
      // see the flank, the bow rise and the rider's silhouette against the sea.
      // `lookBias` is pushed further right along camera-right, which drives the
      // subject further *left* of frame and out from under the board panel.
      r = radius * 0.78;
      hgt = height * 0.36;
      spd = speed * 0.6;
      // Just below the camera's own height, so the rig looks very slightly *down*
      // and the horizon lands near mid-frame. Above it the rig looks up, which
      // filled the top 40 % of the frame with near-zenith sky.
      lookLift = 0.8;
      lookBias = 6.2;
      bob = 0.75;
    }

    this.orbitAngle += ctx.dt * spd;
    const a = this.orbitAngle;
    _orbitPos.set(
      target.x + Math.sin(a) * r,
      target.y + hgt + Math.sin(a * 0.7) * bob,
      target.z + Math.cos(a) * r,
    );
    // Look slightly *past* the boat so it sits off-centre in frame — a subject
    // dead in the middle of a cinematic is the look of a debug orbit.
    // Camera right for a lookAt from `_orbitPos` with world up is
    // (cos a, 0, −sin a), since the view axis is (sin a, ·, cos a).
    _orbitLook.set(
      target.x - Math.sin(a) * 1.1 + Math.cos(a) * lookBias,
      target.y + lookLift,
      target.z - Math.cos(a) * 1.1 - Math.sin(a) * lookBias,
    );
    this.cinematicFrame = ctx.frame;
  }

  resize(aspect: number) {
    this.camera.aspect = aspect;
    this.camera.updateProjectionMatrix();
  }
}
