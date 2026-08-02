/**
 * Boat subsystem entry point.
 *
 * Two exports matter to the rest of the game and their signatures are fixed by
 * `main.ts`:
 *
 *   createRacer(id, position, heading) → Racer
 *   new BoatPhysics(racers)           → Subsystem @ order 30
 *
 * Geometry lives in `boatMesh.ts`, the solver in `boatPhysics.ts`. This file is
 * just the wiring plus the `Racer` record itself.
 *
 * ── For the rider subsystem ────────────────────────────────────────────────
 * Do not hard-code a seat offset. Either read `SEAT_LOCAL` / `GRIP_LOCAL` from
 * `boatMesh.ts`, or call `getSeat(racer)` below, which returns the `Object3D`
 * named `"seat"` that already lives inside the boat's mesh group at the hips
 * position. Parenting the rig to that Object3D means re-proportioning the hull
 * moves the rider with it instead of sinking them into the deck.
 */

import { Group, Object3D, Vector3 } from 'three';
import { createInputState } from '../core/input';
import type { BoatState, Racer, RacerId } from '../core/types';
import { createBoatMesh, GRIP_HALF_WIDTH, GRIP_LOCAL, SEAT_LOCAL, type BoatMesh } from './boatMesh';
import { BoatPhysics } from './boatPhysics';

const RACER_NAMES = ['YOU', 'KAIRA', 'NOX', 'PIP'];
const PERSONALITIES = [null, 'aggressive', 'clean', 'erratic'] as const;

/** Mesh handles, by racer id. Lets other subsystems find the seat and the hull. */
const meshes = new Map<number, BoatMesh>();

export function createRacer(id: RacerId, position: Vector3, heading: number): Racer {
  const root = new Group();
  root.name = `racer${id}`;
  root.rotation.order = 'YXZ';
  root.position.copy(position);
  root.rotation.y = heading;

  const mesh = createBoatMesh(id);
  meshes.set(id, mesh);
  root.add(mesh.group);

  const state: BoatState = {
    // Aliased on purpose: the solver integrates `root.position` directly, so
    // every reader of `state.position` sees the same Vector3 with no copy step.
    position: root.position,
    velocity: new Vector3(),
    heading,
    forwardSpeed: 0,
    lateralSpeed: 0,
    speedFrac: 0,
    pitch: 0,
    roll: 0,
    airborne: false,
    airTime: 0,
    landingImpact: 0,
    drifting: false,
    driftCharge: 0,
    driftTier: 0,
    boostTime: 0,
    boostMeter: 0,
    appliedThrottle: 0,
  };

  return {
    id,
    isPlayer: id === 0,
    name: RACER_NAMES[id],
    root,
    state,
    controls: createInputState(),
    personality: PERSONALITIES[id] as Racer['personality'],
    lap: 0,
    nextCheckpoint: 0,
    progress: 0,
    place: id + 1,
    finished: false,
    finishTime: 0,
    lapTimes: [],
    bestLap: Infinity,
    wrongWay: false,
  };
}

/** The rider attachment point for a racer: an `Object3D` at the hips. */
export function getSeat(racer: Racer): Object3D | null {
  return meshes.get(racer.id)?.seat ?? null;
}

/** The handlebar grip centre for a racer. Hands reach here. */
export function getGrip(racer: Racer): Object3D | null {
  return meshes.get(racer.id)?.grip ?? null;
}

export function getBoatMesh(id: number): BoatMesh | undefined {
  return meshes.get(id);
}

export { BoatPhysics, SEAT_LOCAL, GRIP_LOCAL, GRIP_HALF_WIDTH };
