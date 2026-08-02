/**
 * INK TIDE — riders.
 *
 * The brief is blunt about this: a stiff rider sinks the whole look. So the
 * rider is not a prop bolted to the deck — it is a two-bone-blended skin over a
 * 22-bone rig whose every joint is derived from BoatState each frame, with the
 * hands solved onto the handlebars by IK so the grip never slides.
 *
 * Split across three files:
 *   riderRig.ts   skeleton, procedural cel geometry, the skinning shader chunk
 *   riderAnim.ts  signals → weighted pose state machine → modifiers → IK
 *   rider.ts      this: the Subsystem, attachment to the boat, per-racer variety
 *
 * Cost: 2 shaded meshes + 2 inverted-hull outlines per rider = 4 draw calls,
 * ~2.4 k triangles. See ARCHITECTURE.md's rule that a ~12-part rider must not
 * cost 12 draw calls.
 */

import type { GameContext, Racer, Subsystem } from '../core/types';
import { setOutlineWidth } from '../render/celMaterial';
import { getSeat, SEAT_LOCAL } from '../boat/boat';
import { RIDER_BUILDS, createRiderMesh, type RiderBuild, type RiderMesh } from './riderRig';
import { RiderAnimator } from './riderAnim';

/**
 * The rig's own space has the hips at the origin, which is exactly what the
 * hull's `"seat"` node means, so attachment is a plain `seat.add(riderRoot)`
 * with no offset. If a hull ever fails to publish a seat we fall back to
 * `SEAT_LOCAL` on the boat root — same number, one less indirection.
 */
const SEAT_FALLBACK = SEAT_LOCAL.clone();

interface RiderInstance {
  mesh: RiderMesh;
  anim: RiderAnimator;
  build: RiderBuild;
  /** True when the hull published a seat node and we are parented to it. */
  seated: boolean;
}

export class Riders implements Subsystem {
  readonly name = 'riders';
  readonly order = 60;

  private riders = new Map<number, RiderInstance>();
  /** Total triangles across all riders — reported, not used. */
  readonly triangles: number;

  constructor(racers: Racer[]) {
    let tris = 0;
    for (const r of racers) {
      const build = RIDER_BUILDS[r.id % RIDER_BUILDS.length];
      const mesh = createRiderMesh(r.id, build);
      const seat = getSeat(r);
      if (seat) {
        seat.add(mesh.root);
      } else {
        mesh.root.position.copy(SEAT_FALLBACK);
        r.root.add(mesh.root);
      }
      // Player riders are seen closest, so they carry the heaviest line; the
      // pack reads better with slightly finer ink so four of them at distance
      // do not turn into four black blobs.
      setOutlineWidth(mesh.root, r.isPlayer ? 2.8 : 2.3);
      this.riders.set(r.id, {
        mesh,
        anim: new RiderAnimator(mesh, build),
        build,
        seated: !!seat,
      });
      tris += mesh.triangles;
    }
    this.triangles = tris;
  }

  update(ctx: GameContext) {
    for (const r of ctx.racers) {
      const inst = this.riders.get(r.id);
      if (!inst) continue;
      inst.anim.update(r, ctx);
    }
  }

  dispose() {
    for (const inst of this.riders.values()) {
      inst.mesh.soft.geometry.dispose();
      inst.mesh.hard.geometry.dispose();
      for (const set of inst.mesh.sets) {
        set.main.dispose();
        set.prepass.dispose();
        set.outline?.dispose();
      }
      inst.mesh.root.removeFromParent();
    }
    this.riders.clear();
  }
}
