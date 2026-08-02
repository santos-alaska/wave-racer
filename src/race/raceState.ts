/**
 * Race flow: countdown → racing → finished → results.
 *
 * ── Progress, and why it is measured in arc length ──────────────────────────
 * Position in the field is ranked by a *monotonic* progress scalar — laps
 * completed plus the fraction of the lap covered — rather than by distance to
 * the next gate. A distance-based rank flickers whenever two boats straddle a
 * gate, and a raw spline parameter jumps by a full lap at the start/finish seam.
 *
 * Progress is integrated: every frame we take the signed change in lap fraction
 * since last frame, unwrap it across the seam, and accumulate it in metres. That
 * makes `lapDistance` a genuine odometer along the centreline, negative before
 * the first crossing of the line (the grid sits 14–26 m behind it), and it is
 * the single quantity that drives gates, laps, standings and the HUD gap.
 *
 * ── Why checkpoints are arc-length milestones, not proximity spheres ────────
 * The first version fired a checkpoint when the hull came within
 * `halfWidth × 1.6` of the gate centre. That is 27 m on a 17 m gate, so a boat
 * running a wide line through a corner could miss a gate entirely and jam its
 * `nextCheckpoint` forever — and a boat rocking near a gate could double-fire
 * it. Milestones on the odometer cannot miss, cannot double-fire, and cannot be
 * cut, because the odometer only advances along the centreline. Gates are then
 * purely a visual statement of the route, free to be as narrow as the corner
 * needs.
 */

import { Vector3 } from 'three';
import { CONFIG } from '../core/config';
import { clamp } from '../core/mathx';
import type { GameContext, Racer, RaceAPI, RacePhase, Subsystem, TrackPoint } from '../core/types';
import type { Track } from './track';

const _tp: TrackPoint = {
  position: new Vector3(),
  tangent: new Vector3(0, 0, 1),
  curvature: 0,
  u: 0,
};

/** Seconds after the leader finishes before stragglers are called in. */
const FINISH_GRACE = 22;

interface RacerProgress {
  /** Last known lap fraction, for the seam unwrap. */
  lastU: number;
  /** Signed odometer along the centreline this lap, metres. Negative on the grid. */
  lapDistance: number;
  lapStartTime: number;
  /** How long the racer has been pointing backwards; the wrong-way debounce. */
  wrongTimer: number;
  /** Seconds with no forward progress — surfaced for the harness. */
  stuckTimer: number;
  /** Peak odometer reached, so a spin cannot un-fire gates it already passed. */
  furthest: number;
}

export class RaceState implements RaceAPI, Subsystem {
  readonly name = 'raceState';
  readonly order = 50;

  phase: RacePhase = 'countdown';
  raceTime = -CONFIG.race.countdownSeconds;
  countdownNumber = 3;

  readonly racers: Racer[];
  readonly player: Racer;

  private progressData = new Map<number, RacerProgress>();
  private finishOrder: Racer[] = [];
  private resultsTimer = 0;
  private lastCountdownBeep = 99;
  private leaderFinishedAt = Infinity;
  /** Harness telemetry: gates fired, in order, per racer. */
  readonly gateLog = new Map<number, number[]>();

  constructor(
    racers: Racer[],
    private track: Track,
    private onRestart: () => void,
  ) {
    this.racers = racers;
    this.player = racers.find((r) => r.isPlayer)!;
    this.resetProgress();
  }

  private resetProgress() {
    this.progressData.clear();
    this.gateLog.clear();
    for (const r of this.racers) {
      const proj = this.track.project(r.root.position);
      // Signed distance from the start/finish line, in (−L/2, +L/2]. On the grid
      // this is negative, which is what keeps the first crossing of the line from
      // counting as a completed lap.
      const signed = (((proj.u + 0.5) % 1) - 0.5) * this.track.length;
      this.progressData.set(r.id, {
        lastU: proj.u,
        lapDistance: signed,
        lapStartTime: 0,
        wrongTimer: 0,
        stuckTimer: 0,
        furthest: signed,
      });
      // Gate 0 *is* the start/finish line, and crossing it completes a lap
      // rather than being a checkpoint to collect. So the first gate to chase is
      // gate 1.
      r.nextCheckpoint = 1;
      r.progress = signed / this.track.length;
      r.wrongWay = false;
      this.gateLog.set(r.id, []);
    }
    this.leaderFinishedAt = Infinity;
    this.assignPlaces();
  }

  restart() {
    this.phase = 'countdown';
    this.raceTime = -CONFIG.race.countdownSeconds;
    this.countdownNumber = 3;
    this.finishOrder = [];
    this.resultsTimer = 0;
    this.lastCountdownBeep = 99;
    this.onRestart();
    this.resetProgress();
  }

  standings(): Racer[] {
    return [...this.racers].sort((a, b) => {
      if (a.finished && b.finished) return a.finishTime - b.finishTime;
      if (a.finished) return -1;
      if (b.finished) return 1;
      return b.progress - a.progress;
    });
  }

  /** Metres of centreline `r` has covered this lap. Negative before the line. */
  lapDistanceOf(r: Racer): number {
    return this.progressData.get(r.id)?.lapDistance ?? 0;
  }

  /** Seconds `r` has spent making no forward progress. Harness diagnostic. */
  stuckTimeOf(r: Racer): number {
    return this.progressData.get(r.id)?.stuckTimer ?? 0;
  }

  update(ctx: GameContext) {
    const { dt } = ctx;

    switch (this.phase) {
      case 'countdown': {
        this.raceTime += dt;
        const n = Math.max(0, Math.ceil(-this.raceTime));
        this.countdownNumber = n;
        if (n < this.lastCountdownBeep) {
          this.lastCountdownBeep = n;
          ctx.audio.horn(n === 0 ? 1.0 : 0.55 + (3 - n) * 0.08);
        }
        // Throttle is locked out until the lights change. This runs *after* the
        // AI (order 40 < 50), so it also stops the AI jumping the start.
        for (const r of this.racers) {
          r.controls.throttle = 0;
          r.controls.brake = 0;
          r.controls.steer = 0;
          r.controls.drift = false;
        }
        if (this.raceTime >= 0) {
          this.phase = 'racing';
          this.countdownNumber = -1;
          for (const r of this.racers) this.progressData.get(r.id)!.lapStartTime = 0;
        }
        break;
      }

      case 'racing': {
        this.raceTime += dt;
        for (const r of this.racers) this.trackProgress(ctx, r);
        this.assignPlaces();

        // Once the leader is home, stragglers get a grace period and are then
        // classified where they stand. Without this a single AI stuck against
        // the swell holds the results screen hostage forever.
        if (this.finishOrder.length > 0 && !isFinite(this.leaderFinishedAt)) {
          this.leaderFinishedAt = this.raceTime;
        }
        const graceOver = this.raceTime - this.leaderFinishedAt > FINISH_GRACE;
        if (this.racers.every((r) => r.finished) || (graceOver && this.player.finished)) {
          for (const r of this.standings()) {
            if (r.finished) continue;
            r.finished = true;
            r.finishTime = this.raceTime;
            this.finishOrder.push(r);
            r.place = this.finishOrder.length;
          }
          this.phase = 'finished';
          this.resultsTimer = 0;
        }
        break;
      }

      case 'finished': {
        this.raceTime += dt;
        this.resultsTimer += dt;
        if (this.resultsTimer > 2.2) this.phase = 'results';
        break;
      }

      case 'results': {
        this.resultsTimer += dt;
        if (ctx.input.restartPressed || ctx.input.startPressed) this.restart();
        break;
      }
    }
  }

  // ── Progress ──────────────────────────────────────────────────────────────

  private trackProgress(ctx: GameContext, r: Racer) {
    if (r.finished) return;
    const d = this.progressData.get(r.id)!;
    const proj = this.track.project(r.root.position);

    // Signed step in lap fraction, unwrapped across the start/finish seam.
    let du = proj.u - d.lastU;
    if (du > 0.5) du -= 1;
    if (du < -0.5) du += 1;
    d.lastU = proj.u;
    const step = du * this.track.length;
    d.lapDistance += step;
    if (d.lapDistance > d.furthest) d.furthest = d.lapDistance;

    r.progress = r.lap + d.lapDistance / this.track.length;

    // Stall detector. Purely diagnostic — recovery is the driver's job.
    d.stuckTimer = step / Math.max(ctx.dt, 1e-4) < 1.5 ? d.stuckTimer + ctx.dt : 0;

    this.updateWrongWay(ctx, r, d, proj.u);
    this.advanceGates(ctx, r, d);
  }

  /**
   * Wrong-way is a *debounced* test, not an instantaneous one. A stationary boat
   * rocking on a 1.3 m swell has a velocity vector that swings through a wide
   * arc several times a second, so the old `dot < −0.35` check flickered on at
   * idle. Requiring 4.5 m/s of real speed *and* 0.55 s of sustained misalignment
   * makes it impossible to trigger from wave motion, and it still lights within
   * half a second of a genuine U-turn.
   */
  private updateWrongWay(ctx: GameContext, r: Racer, d: RacerProgress, u: number) {
    const v = r.state.velocity;
    const speed = Math.hypot(v.x, v.z);
    let align = 1;
    if (speed > 0.4) {
      this.track.sample(u, _tp);
      align = (v.x * _tp.tangent.x + v.z * _tp.tangent.z) / speed;
    }
    const bad = speed > 4.5 && align < CONFIG.race.wrongWayDot;
    d.wrongTimer = clamp(d.wrongTimer + (bad ? ctx.dt : -ctx.dt * 2.5), 0, 1.4);
    r.wrongWay = d.wrongTimer > 0.55;
  }

  /**
   * Gates as odometer milestones. `nextCheckpoint` walks 1 → 11 → 0, where 0
   * means "the finish line", whose milestone is a full lap of distance.
   */
  private advanceGates(ctx: GameContext, r: Racer, d: RacerProgress) {
    const cps = this.track.checkpoints;
    // Forward: a fast boat can clear two milestones in a frame at 60 fps only if
    // gates were 0.5 m apart, but the loop costs nothing and removes the class of
    // bug entirely.
    for (let guard = 0; guard <= cps.length; guard++) {
      const idx = r.nextCheckpoint;
      if (idx === 0) {
        if (d.lapDistance < this.track.length) break;
        this.completeLap(ctx, r, d);
        if (r.finished) return;
        continue;
      }
      const milestone = cps[idx].s;
      if (d.lapDistance < milestone) break;
      r.nextCheckpoint = (idx + 1) % cps.length;
      this.gateLog.get(r.id)!.push(idx);
      if (r.isPlayer) ctx.audio.checkpoint();
    }
    // Backward: a spin or a wrong-way excursion rewinds the odometer, so rewind
    // the gate pointer with it. Otherwise the racer keeps their credit for a
    // gate they are now behind, and the next lap's gate order is nonsense.
    for (let guard = 0; guard <= cps.length; guard++) {
      const idx = r.nextCheckpoint;
      const prev = (idx - 1 + cps.length) % cps.length;
      if (prev === 0) break; // never rewind past the finish line
      const milestone = cps[prev].s;
      if (d.lapDistance >= milestone - 10) break;
      r.nextCheckpoint = prev;
      const log = this.gateLog.get(r.id)!;
      if (log[log.length - 1] === prev) log.pop();
    }
  }

  private completeLap(ctx: GameContext, r: Racer, d: RacerProgress) {
    const lapTime = this.raceTime - d.lapStartTime;
    d.lapStartTime = this.raceTime;
    d.lapDistance -= this.track.length;
    d.furthest = d.lapDistance;
    r.nextCheckpoint = 1;
    r.lapTimes.push(lapTime);
    if (lapTime < r.bestLap) r.bestLap = lapTime;
    r.lap++;

    if (r.lap >= CONFIG.race.laps) {
      r.finished = true;
      r.finishTime = this.raceTime;
      this.finishOrder.push(r);
      r.place = this.finishOrder.length;
      if (r.isPlayer) ctx.audio.horn(0.85);
    }
  }

  private assignPlaces() {
    const sorted = this.standings();
    for (let i = 0; i < sorted.length; i++) {
      if (!sorted[i].finished) sorted[i].place = i + 1;
    }
  }
}
