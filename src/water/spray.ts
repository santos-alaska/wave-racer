/**
 * Spray — one `Points` system for every emitter in the game.
 *
 * The performance contract is explicit that particles get one system per FX
 * type, never one per emitter, so all four boats' bow spray, drift sheets and
 * landing bursts come out of the single ring buffer below. Adding a fifth
 * emitter (a gate being clipped, a collision) costs an `emit()` call, not a
 * draw call.
 *
 * ── Why this reads as anime spray and not as a particle system ───────────────
 *   • the silhouette is a lumpy three-lobe polar blob generated in the fragment
 *     shader, hard cut — no soft round blobs, ever, and no shared stamp to
 *     recognise (the first pass used the lacy foam texture and at droplet size
 *     it read unmistakably as cartoon clouds)
 *   • alpha is quantised to three steps, so a droplet pops out of existence at a
 *     defined moment instead of ghosting away
 *   • size is in *world metres*: the quad is billboarded and scaled in world
 *     space, so a droplet 3 m away and one 40 m away are the same physical size
 *     rather than the same number of pixels
 *   • the lobe phases are driven by a per-droplet seed, so a burst is fifty
 *     different shapes rather than fifty copies of one image
 *
 * Motion is a plain ballistic integration with drag on the CPU. There is no
 * allocation in the loop: all state lives in preallocated Float32Arrays and dead
 * slots are recycled by a cursor.
 */

import {
  BufferAttribute,
  DynamicDrawUsage,
  InstancedBufferAttribute,
  InstancedBufferGeometry,
  Mesh,
  NormalBlending,
  ShaderMaterial,
  Texture,
} from 'three';
import { CONFIG } from '../core/config';
import { PAL } from '../core/palette';
import { SHARED } from '../render/celMaterial';

import { rng } from '../core/rng';
import type { GameContext, Racer } from '../core/types';

const CAPACITY = 1500;
/** Gravity for droplets — exaggerated, because arcade water is snappy. */
const GRAVITY = 17.5;
/** Air drag as an exponential rate, per second. */
const DRAG = 1.35;
/**
 * Global droplet scale.
 *
 * The emitters ask for 0.2–0.95 m droplets, and at chase-camera range a 0.5 m
 * world-sized quad is ~75 device pixels across. Seventy-five pixels of soft
 * additive white is a cotton ball, which is exactly what the r13 capture found
 * behind the boats. Now that a droplet is an opaque hard-edged shape it has to
 * be droplet-sized to read as spray, so every emitted size passes through this
 * and the emission *rates* are raised to compensate: many small hard chips, not
 * a few big soft ones.
 */
const SIZE_SCALE = 0.40;

export class SprayField {
  /** One draw call for every droplet in the game. */
  readonly points: Mesh;
  private material: ShaderMaterial;

  private pos: Float32Array;
  private vel: Float32Array;
  private age: Float32Array;
  private life: Float32Array;
  private size: Float32Array;
  private seed: Float32Array;
  private cursor = 0;
  private alive = 0;

  private aPos: InstancedBufferAttribute;
  private aAge: InstancedBufferAttribute;
  private aLife: InstancedBufferAttribute;
  private aSize: InstancedBufferAttribute;
  private aSeed: InstancedBufferAttribute;

  /** Per-racer emission accumulators, so rate is dt-independent. */
  private bowCarry = [0, 0, 0, 0];
  private driftCarry = [0, 0, 0, 0];
  private sternCarry = [0, 0, 0, 0];

  constructor(_noise: Texture) {
    this.pos = new Float32Array(CAPACITY * 3);
    this.vel = new Float32Array(CAPACITY * 3);
    this.age = new Float32Array(CAPACITY);
    this.life = new Float32Array(CAPACITY);
    this.size = new Float32Array(CAPACITY);
    this.seed = new Float32Array(CAPACITY);
    // Everything starts dead: age past life means size 0 in the shader.
    for (let i = 0; i < CAPACITY; i++) {
      this.life[i] = 1;
      this.age[i] = 2;
      this.seed[i] = rng.next();
    }

    // ── Geometry: billboarded instanced quads, not a Points cloud ───────────
    // The first implementation was a `Points` system sized with `gl_PointSize`,
    // and across three captured rounds (r7, r8, r9) not one droplet ever
    // appeared on screen while the emitters were provably firing — the wake
    // ribbons, driven from the same update call, were visible throughout. Point
    // size is the one part of the pipeline the ANGLE/Metal backend does not
    // reliably honour, and there is no way to work around it from shader code.
    //
    // Instanced quads billboarded in the vertex shader cost the same single draw
    // call, are immune to that, and are strictly more capable: the quad can be
    // rotated per droplet, which is where the per-instance shape variation comes
    // from. The performance contract allows either form; this is the one that
    // actually renders.
    const geo = new InstancedBufferGeometry();
    geo.setAttribute(
      'position',
      new BufferAttribute(
        new Float32Array([-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0]),
        3,
      ),
    );
    geo.setIndex(new BufferAttribute(new Uint16Array([0, 1, 2, 0, 2, 3]), 1));
    this.aPos = new InstancedBufferAttribute(this.pos, 3);
    this.aAge = new InstancedBufferAttribute(this.age, 1);
    this.aLife = new InstancedBufferAttribute(this.life, 1);
    this.aSize = new InstancedBufferAttribute(this.size, 1);
    this.aSeed = new InstancedBufferAttribute(this.seed, 1);
    this.aPos.setUsage(DynamicDrawUsage);
    this.aAge.setUsage(DynamicDrawUsage);
    this.aLife.setUsage(DynamicDrawUsage);
    this.aSize.setUsage(DynamicDrawUsage);
    this.aSeed.setUsage(DynamicDrawUsage);
    geo.setAttribute('iPos', this.aPos);
    geo.setAttribute('iAge', this.aAge);
    geo.setAttribute('iLife', this.aLife);
    geo.setAttribute('iSize', this.aSize);
    geo.setAttribute('iSeed', this.aSeed);
    geo.instanceCount = CAPACITY;
    geo.boundingSphere = null;

    this.material = new ShaderMaterial({
      name: 'spray',
      transparent: true,
      depthWrite: false,
      /**
       * NOT additive.
       *
       * Additive was the magenta streak. A white sprite added over bright cyan
       * water saturates the red channel first — measured in the r13 hero shot as
       * R pinned at 255 while G sat at 152-211 — so every dense burst turned hot
       * pink, including where it landed on top of a hull. There is no exposure
       * setting that fixes that; the layer has to composite rather than sum.
       * With a 0/1 cutout mask the sprite simply replaces, so no channel can
       * clip and every droplet lands on a committed palette value.
       */
      blending: NormalBlending,
      uniforms: {
        uFoam: { value: PAL.foam.clone() },
        uFoamShade: { value: PAL.foamShade.clone() },
        uCrest: { value: PAL.waterCrest.clone() },
        uStrength: { value: 1.0 },
      },
      vertexShader: /* glsl */ `
        attribute vec3 iPos;
        attribute float iAge;
        attribute float iLife;
        attribute float iSize;
        attribute float iSeed;

        varying vec2 vQuad;
        varying float vT;
        varying float vSeed;

        void main() {
          vT = iAge / max(iLife, 1e-3);
          vSeed = iSeed;
          vQuad = position.xy;

          // Droplets grow as they break up, then shrink as they fall apart.
          float grow = mix(0.55, 1.35, smoothstep(0.0, 0.45, vT))
                     * (1.0 - smoothstep(0.72, 1.0, vT));
          // Dead instances collapse the quad to a point, which rasterises nothing.
          float s = iSize * grow * step(vT, 1.0);

          // Rotate AND stretch the quad, not the texture lookup: the silhouette is
          // generated from the *unrotated, unstretched* local coordinate, so the
          // geometry transform spins and elongates the shape. Combined with the
          // four authored silhouettes in the fragment stage that gives every
          // droplet in a burst its own outline, which is what stops a burst from
          // reading as one sprite stamped repeatedly — the r13 capture found the
          // same 4-lobed clover at (1285,1073) and again three times in hud.png.
          float hAsp = fract(iSeed * 17.13);
          vec2 asp = vec2(0.68 + 0.74 * hAsp, 0.68 + 0.74 * (1.0 - hAsp));
          vec2 qp = position.xy * asp;
          float a = iSeed * 6.2831853;
          float c = cos(a), sn = sin(a);
          vec2 q = vec2(qp.x * c - qp.y * sn, qp.x * sn + qp.y * c) * s;

          // Camera basis straight out of the view matrix — a screen-facing quad
          // without needing the camera's world matrix as a uniform.
          vec3 camRight = vec3(viewMatrix[0][0], viewMatrix[1][0], viewMatrix[2][0]);
          vec3 camUp    = vec3(viewMatrix[0][1], viewMatrix[1][1], viewMatrix[2][1]);
          vec3 world = iPos + camRight * q.x + camUp * q.y;

          gl_Position = projectionMatrix * viewMatrix * vec4(world, 1.0);
        }
      `,
      fragmentShader: /* glsl */ `
        precision highp float;

        uniform vec3 uFoam, uFoamShade, uCrest;
        uniform float uStrength;

        varying vec2 vQuad;
        varying float vT;
        varying float vSeed;

        void main() {
          // ── Four authored silhouettes, one per droplet ────────────────────
          // The previous single three-lobe polar radius had lobe amplitudes of
          // 0.085 and 0.05 against a base radius of 0.40 — a 12% wobble on a
          // circle. That is why every droplet in the r13 capture was the same
          // 4-lobed clover: the shapes were not actually different. These four
          // are different *outlines*, not different phases of one outline, and
          // the vertex stage rotates and stretches each instance on top.
          float r = length(vQuad);
          float ang = atan(vQuad.y, vQuad.x);
          float shp = floor(fract(vSeed * 5.71) * 4.0);
          float radius;

          if (shp < 1.0) {
            // Shard: a teardrop with one sharp point — a droplet torn off a crest.
            radius = 0.15 + 0.30 * pow(max(cos(ang * 0.5), 0.0), 2.2);
          } else if (shp < 2.0) {
            // Splat: five sharp spikes off a small core.
            radius = 0.17 + 0.27 * pow(abs(cos(ang * 2.5 + vSeed * 9.0)), 3.0);
          } else if (shp < 3.0) {
            // Chip: a hard-edged flake. A straight-sided polygon among the
            // curved shapes is what keeps the set from reading as one family.
            radius = 0.42 / (abs(cos(ang)) + 1.45 * abs(sin(ang)));
          } else {
            // Comma: a blob with a bite out of it, drawn as a crescent.
            radius = 0.34 + 0.11 * sin(ang * 3.0 + vSeed * 13.0);
            // The bite has to reach *past* the silhouette edge, otherwise it
            // punches a hole in the middle and the droplet reads as a bubble
            // ring rather than as a crescent of torn water.
            if (length(vQuad - vec2(0.24, 0.13)) < 0.24) discard;
          }

          // Droplets tear apart rather than fading: the radius shrinks with age
          // and a bite is taken out of one side.
          radius *= 1.0 - vT * 0.28;
          radius -= (0.5 + 0.5 * sin(ang * 2.0 + vSeed * 19.0)) * vT * 0.13;
          if (r > radius) discard;

          // ── Hard cutout: alpha is 1, never a ramp ────────────────────────
          // Soft alpha is what made the spray read as photographic cotton wool
          // and as a different render from the water. Ageing is expressed by the
          // shrinking silhouette and by dropping down the tone ladder — never by
          // transparency, and never additively (see the blending note above).
          float fade = 1.0 - vT;
          float core = step(r, radius * 0.5);
          vec3 col = uCrest;
          col = mix(col, uFoamShade, step(0.26, fade));
          col = mix(col, uFoam, step(0.42, fade) * core);
          gl_FragColor = vec4(col, uStrength);
        }
      `,
    });

    this.points = new Mesh(geo, this.material);
    this.points.name = 'spray';
    this.points.frustumCulled = false;
    this.points.renderOrder = 6;
    this.points.userData.skipPrepass = true;
  }

  get tunables() {
    return this.material.uniforms;
  }

  /** Spawn one droplet. Recycles the oldest slot when full — never allocates. */
  emit(
    x: number,
    y: number,
    z: number,
    vx: number,
    vy: number,
    vz: number,
    size: number,
    life: number,
  ) {
    const i = this.cursor;
    this.cursor = (this.cursor + 1) % CAPACITY;
    const p = i * 3;
    this.pos[p] = x;
    this.pos[p + 1] = y;
    this.pos[p + 2] = z;
    this.vel[p] = vx;
    this.vel[p + 1] = vy;
    this.vel[p + 2] = vz;
    this.age[i] = 0;
    this.life[i] = life;
    this.size[i] = size * SIZE_SCALE;
    this.seed[i] = rng.next();
  }

  /**
   * Emit `count` droplets in a cone. One call per burst keeps the per-frame
   * emitter logic short and makes every FX site read the same way.
   */
  burst(
    x: number,
    y: number,
    z: number,
    dx: number,
    dy: number,
    dz: number,
    count: number,
    speed: number,
    size: number,
    spread = 0.55,
  ) {
    for (let i = 0; i < count; i++) {
      const sp = speed * rng.range(0.55, 1.25);
      this.emit(
        x + rng.sym(0.28),
        y + rng.range(-0.05, 0.3),
        z + rng.sym(0.28),
        (dx + rng.sym(spread)) * sp,
        (dy + rng.range(0, spread * 0.9)) * sp,
        (dz + rng.sym(spread)) * sp,
        size * rng.range(0.6, 1.5),
        rng.range(0.5, 1.15),
      );
    }
  }

  update(ctx: GameContext, racers: Racer[], wetness: Float32Array) {
    const dt = ctx.dt;

    // ── Emitters ────────────────────────────────────────────────────────────
    for (let r = 0; r < racers.length; r++) {
      const racer = racers[r];
      const s = racer.state;
      const px = racer.root.position.x;
      const py = racer.root.position.y;
      const pz = racer.root.position.z;
      const fx = Math.sin(s.heading);
      const fz = Math.cos(s.heading);
      // Right-hand lateral for a +Z forward, Y-up frame.
      const rx = fz;
      const rz = -fx;
      const speed = Math.abs(s.forwardSpeed);
      // ── Emit OUTSIDE the hull, always ──────────────────────────────────────
      // The r3 review found droplets drawn on top of the red deck and on top of
      // the yellow AI hull in rider_closeup.png and air.png, and read them as a
      // depth-test failure. They are not — the particle pass depth-tests against
      // the opaque scene — but the emitters were spawning *inside the hull's own
      // footprint*: bow spray at beam·0.42 = 0.80 m and transom wash spread over
      // ±beam·0.45 = 0.86 m, against a half-beam of 0.95 m. A droplet born inside
      // the hull volume is genuinely in front of the deck from half the possible
      // camera angles, so it reads as composited however the depth buffer is
      // configured. Everything now leaves from clear of the shell:
      //   • bow spray at 0.86 of the beam, i.e. just outboard of the chine
      //   • transom wash from behind the transom, not under it
      //   • drift sheet a full beam out on the sliding side
      const halfBeam = CONFIG.boat.beam * 0.5;
      const bowX = px + fx * CONFIG.boat.length * 0.46;
      const bowZ = pz + fz * CONFIG.boat.length * 0.46;

      // Landing impact: a fat ring of spray thrown outward and up.
      if (s.landingImpact > 1.4) {
        const k = Math.min(1, s.landingImpact / 13);
        const y = ctx.ocean.height(px, pz, ctx.time);
        const count = Math.round(34 + k * 84);
        for (let i = 0; i < count; i++) {
          const a = (i / count) * Math.PI * 2 + rng.sym(0.4);
          const out = 3.5 + k * 9 * rng.range(0.6, 1.3);
          this.emit(
            px + Math.cos(a) * 0.8,
            y + 0.15,
            pz + Math.sin(a) * 0.8,
            Math.cos(a) * out + s.velocity.x * 0.2,
            (5.5 + k * 9) * rng.range(0.6, 1.3),
            Math.sin(a) * out + s.velocity.z * 0.2,
            rng.range(0.5, 1.35),
            rng.range(0.8, 1.7),
          );
        }
      }

      // Water cannot be thrown by a hull that is not touching water. Gated on the
      // water subsystem's own graded wetness rather than on `state.airborne`, for
      // the reasons in `hullWetness` — in r3 the flag was set on a third of the
      // shots while the boat was visibly planing, and every one of those lost its
      // spray as well as its collar.
      const wet = wetness[r];
      if (wet < 0.25) {
        this.bowCarry[r] = 0;
        this.driftCarry[r] = 0;
        this.sternCarry[r] = 0;
        continue;
      }

      // Transom wash — the rooster tail straight out of the back.
      //
      // This is the emitter the first pass was missing, and the reason no spray
      // was visible in any chase-camera capture: bow spray is thrown at the front
      // of the hull and travels ~1.5 m before drag kills it, so from behind the
      // boat the hull occludes all of it. The transom throw is dead centre of the
      // chase framing.
      if (speed > 4.5) {
        const rate = (speed - 4.5) * 6.0;
        this.sternCarry[r] += rate * dt;
        while (this.sternCarry[r] >= 1) {
          this.sternCarry[r] -= 1;
          const ex = px - fx * CONFIG.boat.length * 0.62 + rx * rng.sym(halfBeam * 0.8);
          const ez = pz - fz * CONFIG.boat.length * 0.62 + rz * rng.sym(halfBeam * 0.8);
          const y = ctx.ocean.height(ex, ez, ctx.time);
          this.emit(
            ex,
            y + 0.12,
            ez,
            -fx * speed * 0.16 + rx * rng.sym(3.4),
            rng.range(2.4, 6.0) + speed * 0.05,
            -fz * speed * 0.16 + rz * rng.sym(3.4),
            rng.range(0.2, 0.52),
            rng.range(0.45, 0.95),
          );
        }
      } else {
        this.sternCarry[r] = 0;
      }

      // Bow spray: rate rises with speed, thrown forward-out and up.
      if (speed > 6.5) {
        const rate = (speed - 6.5) * 6.4;
        this.bowCarry[r] += rate * dt;
        while (this.bowCarry[r] >= 1) {
          this.bowCarry[r] -= 1;
          const side = rng.bool() ? 1 : -1;
          const y = ctx.ocean.height(bowX, bowZ, ctx.time);
          const out = 1.4 + speed * 0.09;
          this.emit(
            bowX + rx * side * halfBeam * 1.22,
            y + 0.1,
            bowZ + rz * side * halfBeam * 1.22,
            fx * speed * 0.22 + rx * side * out + rng.sym(0.7),
            rng.range(2.4, 5.2) + speed * 0.045,
            fz * speed * 0.22 + rz * side * out + rng.sym(0.7),
            rng.range(0.24, 0.6),
            rng.range(0.45, 0.9),
          );
        }
      } else {
        this.bowCarry[r] = 0;
      }

      // Drift sheet: thrown off the outside of the slide, nearly horizontal.
      const slip = Math.abs(s.lateralSpeed);
      if (slip > 2.6) {
        const dir = s.lateralSpeed > 0 ? -1 : 1;
        this.driftCarry[r] += (slip - 2.6) * 6.4 * dt;
        while (this.driftCarry[r] >= 1) {
          this.driftCarry[r] -= 1;
          const along = rng.sym(CONFIG.boat.length * 0.4);
          const ex = px + fx * along + rx * dir * halfBeam * 1.18;
          const ez = pz + fz * along + rz * dir * halfBeam * 1.18;
          const y = ctx.ocean.height(ex, ez, ctx.time);
          const out = 2.2 + slip * 0.5;
          this.emit(
            ex,
            y + 0.08,
            ez,
            rx * dir * out + s.velocity.x * 0.25 + rng.sym(0.9),
            rng.range(1.4, 3.6),
            rz * dir * out + s.velocity.z * 0.25 + rng.sym(0.9),
            rng.range(0.3, 0.75),
            rng.range(0.45, 0.95),
          );
        }
      } else {
        this.driftCarry[r] = 0;
      }
    }

    // ── Integrate ───────────────────────────────────────────────────────────
    // Exponential drag, evaluated once per frame rather than per particle, so
    // the motion is frame-rate independent without a per-particle exp().
    const damping = Math.exp(-DRAG * dt);
    const pos = this.pos;
    const vel = this.vel;
    let alive = 0;
    for (let i = 0; i < CAPACITY; i++) {
      const a = this.age[i];
      if (a >= this.life[i]) continue;
      this.age[i] = a + dt;
      const p = i * 3;
      vel[p] *= damping;
      vel[p + 1] = (vel[p + 1] - GRAVITY * dt) * damping;
      vel[p + 2] *= damping;
      pos[p] += vel[p] * dt;
      pos[p + 1] += vel[p + 1] * dt;
      pos[p + 2] += vel[p + 2] * dt;
      // Cheap kill: droplets that have fallen back below the rest plane have
      // rejoined the sea. Sampling the true surface per droplet is not worth
      // 1500 Gerstner inversions a frame for something that lasts 0.6 s.
      if (pos[p + 1] < -0.35) this.age[i] = this.life[i];
      else alive++;
    }
    this.alive = alive;

    this.aPos.needsUpdate = true;
    this.aAge.needsUpdate = true;
    this.aLife.needsUpdate = true;
    this.aSize.needsUpdate = true;
    this.aSeed.needsUpdate = true;
  }

  get liveCount() {
    return this.alive;
  }

  dispose() {
    this.points.geometry.dispose();
    this.material.dispose();
  }
}
