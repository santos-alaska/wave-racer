/**
 * Shared contracts.
 *
 * Every subsystem depends on this file and nothing else depends on a
 * subsystem's internals. If you are implementing a subsystem, implement the
 * interface here — do not reach into another subsystem's module.
 */

import type { Camera, Object3D, PerspectiveCamera, Scene, Vector3, WebGLRenderer } from 'three';
import type { InputState } from './input';
import type { OceanSample } from '../water/gerstner';

// ─────────────────────────────────────────────────────────────────────────────
// Ocean
// ─────────────────────────────────────────────────────────────────────────────

/** Read-only view of the water, handed to physics, track and FX. */
export interface OceanSampler {
  /** Full surface sample at a world XZ. `out` is optional, for hot loops. */
  sample(x: number, z: number, t: number, out?: OceanSample): OceanSample;
  /** Height only — cheaper when the normal isn't needed. */
  height(x: number, z: number, t: number): number;
  /** Upper bound on surface height, for culling. */
  maxHeight(): number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Racers
// ─────────────────────────────────────────────────────────────────────────────

export type RacerId = 0 | 1 | 2 | 3;

/** Everything the rest of the game needs to know about one boat. */
export interface BoatState {
  /** World position of the hull's centre of mass. */
  position: Vector3;
  /** Linear velocity, world space, m/s. */
  velocity: Vector3;
  /** Heading in radians, 0 = +Z. Yaw only — pitch/roll live on the Object3D. */
  heading: number;
  /** Signed forward speed along the hull axis, m/s. */
  forwardSpeed: number;
  /** Lateral slip speed, m/s. Drives drift FX and tyre-squeal-equivalent. */
  lateralSpeed: number;
  /** 0…1 normalised against boostTopSpeed, for HUD and FOV kick. */
  speedFrac: number;
  /** Hull pitch / roll in radians, produced by the buoyancy solver. */
  pitch: number;
  roll: number;
  /** True while the hull has left the water. */
  airborne: boolean;
  /** Seconds spent in the current airborne stretch. */
  airTime: number;
  /** Set for one frame on water re-entry; magnitude = impact speed. */
  landingImpact: number;
  /** Powerslide state. */
  drifting: boolean;
  driftCharge: number;
  /** 0 = none, 1..3 = tier reached. */
  driftTier: number;
  /** Seconds of boost remaining. */
  boostTime: number;
  /** 0…1 boost meter for the HUD. */
  boostMeter: number;
  /** Throttle actually applied this frame, post-assist. */
  appliedThrottle: number;
}

/** Per-racer AI temperament. Tuned so each opponent is recognisable. */
export type AiPersonality = 'aggressive' | 'clean' | 'erratic';

export interface Racer {
  readonly id: RacerId;
  readonly isPlayer: boolean;
  readonly name: string;
  /** Root transform; the hull mesh, outline hull and rider are parented here. */
  readonly root: Object3D;
  readonly state: BoatState;
  /** Controls for this frame — written by the player input or the AI driver. */
  readonly controls: InputState;
  readonly personality: AiPersonality | null;

  // Race progress, owned by the race subsystem.
  lap: number;
  nextCheckpoint: number;
  /** Monotonic distance along the spline, including laps. Sorts the field. */
  progress: number;
  place: number;
  finished: boolean;
  finishTime: number;
  lapTimes: number[];
  bestLap: number;
  wrongWay: boolean;
}

// ─────────────────────────────────────────────────────────────────────────────
// Track
// ─────────────────────────────────────────────────────────────────────────────

export interface TrackPoint {
  position: Vector3;
  /** Unit tangent along the direction of travel. */
  tangent: Vector3;
  /** Curvature magnitude — used for the corner-preview indicator and AI braking. */
  curvature: number;
  /** Normalised distance along the lap, 0…1. */
  u: number;
}

export interface Checkpoint {
  index: number;
  position: Vector3;
  /** Unit forward through the gate. */
  forward: Vector3;
  /** Half-width of the gate opening. */
  halfWidth: number;
}

export interface TrackAPI {
  /** Total lap length in metres. */
  readonly length: number;
  readonly checkpoints: Checkpoint[];
  /** Sample the centreline at normalised distance u (wraps). */
  sample(u: number, out?: TrackPoint): TrackPoint;
  /** Sample by arc length in metres (wraps). */
  sampleDistance(d: number, out?: TrackPoint): TrackPoint;
  /** Nearest point on the centreline to a world position; returns u. */
  project(position: Vector3): { u: number; distance: number; lateral: number };
  /** Grid-start transforms for the four racers. */
  startGrid(index: number): { position: Vector3; heading: number };
}

// ─────────────────────────────────────────────────────────────────────────────
// Race flow
// ─────────────────────────────────────────────────────────────────────────────

export type RacePhase = 'boot' | 'countdown' | 'racing' | 'finished' | 'results';

export interface RaceAPI {
  phase: RacePhase;
  /** Seconds since the lights went green. Negative during the countdown. */
  raceTime: number;
  /** 3, 2, 1, 0 = GO. -1 when not counting down. */
  countdownNumber: number;
  readonly racers: Racer[];
  readonly player: Racer;
  /** Field sorted by current position. */
  standings(): Racer[];
  restart(): void;
}

// ─────────────────────────────────────────────────────────────────────────────
// Presentation
// ─────────────────────────────────────────────────────────────────────────────

export interface AudioAPI {
  /** Must be called from a user gesture before anything is audible. */
  unlock(): Promise<void>;
  update(ctx: GameContext): void;
  /** One-shots. */
  impact(strength: number): void;
  splash(strength: number): void;
  horn(pitch: number): void;
  boost(): void;
  checkpoint(): void;
  setMuted(m: boolean): void;
}

export interface HudAPI {
  resize(width: number, height: number, dpr: number): void;
  render(ctx: GameContext): void;
}

export interface CameraRig {
  readonly camera: PerspectiveCamera;
  update(ctx: GameContext): void;
  /** Additive screenshake impulse, 0…1. */
  addShake(amount: number): void;
  /** Used by the harness and the countdown/results cinematics. */
  setMode(mode: 'chase' | 'orbit' | 'cinematic' | 'far' | 'bow'): void;
  snapToTarget(): void;
}

// ─────────────────────────────────────────────────────────────────────────────
// The one object passed to every per-frame update
// ─────────────────────────────────────────────────────────────────────────────

export interface GameContext {
  renderer: WebGLRenderer;
  scene: Scene;
  camera: PerspectiveCamera;
  /** Seconds since boot — the value fed to every wave/animation shader. */
  time: number;
  /** Clamped frame delta, seconds. */
  dt: number;
  /** Unclamped, for FPS reporting. */
  rawDt: number;
  frame: number;

  ocean: OceanSampler;
  track: TrackAPI;
  race: RaceAPI;
  racers: Racer[];
  player: Racer;
  input: InputState;
  audio: AudioAPI;
  cameraRig: CameraRig;

  /** Screen size in CSS pixels, and the current device pixel ratio. */
  width: number;
  height: number;
  pixelRatio: number;

  /** Rolling perf numbers, published by the perf controller. */
  perf: { fps: number; frameMs: number; gpuScale: number; drawCalls: number; triangles: number };
}

/**
 * Anything with a per-frame update. Subsystems register one of these with the
 * main loop rather than being called by name, so ordering stays explicit and
 * in one place.
 */
export interface Subsystem {
  readonly name: string;
  /** Lower runs first. Physics < track < ai < fx < camera < hud. */
  readonly order: number;
  update(ctx: GameContext): void;
  dispose?(): void;
}
