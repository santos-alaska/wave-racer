/**
 * Minimap.
 *
 * Split out of the HUD because it has two very different halves: an expensive
 * static half (frame, inked course ribbon, gate ticks, start line) that is baked
 * into the HUD's chrome layer once per resize, and a cheap live half (four racer
 * pips, the next-gate pulse) redrawn every frame.
 *
 * The map is north-up rather than rotating with the player. A rotating map is
 * better for navigating a road course you cannot see; on a closed circuit made
 * of open water, a fixed map lets you learn the shape of the lap, which is what
 * makes the "where am I in the field" read instant.
 */

import { HEX } from '../core/palette';
import { clamp01 } from '../core/mathx';
import {
  FONT_STACK,
  PLATE_SKEW,
  cutPath,
  diamondPath,
  hatch,
  inkText,
  inked,
  plate,
  rgba,
} from './inkDraw';
import type { GameContext, TrackAPI } from '../core/types';

export class Minimap {
  /** Screen rect, CSS px. */
  x = 0;
  y = 0;
  size = 0;

  private minX = 0;
  private minZ = 0;
  private scale = 1;
  private padX = 0;
  private padY = 0;
  private path: Path2D | null = null;
  private gates: { x: number; y: number; nx: number; ny: number }[] = [];
  private start = { x: 0, y: 0, nx: 0, ny: 0 };

  constructor(private track: TrackAPI) {}

  /** Recompute the projection. Called from the HUD's resize path only. */
  layout(x: number, y: number, size: number) {
    this.x = x;
    this.y = y;
    this.size = size;

    const N = 260;
    const pts = new Float32Array(N * 2);
    let minX = Infinity,
      maxX = -Infinity,
      minZ = Infinity,
      maxZ = -Infinity;
    for (let i = 0; i < N; i++) {
      const p = this.track.sample(i / N).position;
      pts[i * 2] = p.x;
      pts[i * 2 + 1] = p.z;
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.z < minZ) minZ = p.z;
      if (p.z > maxZ) maxZ = p.z;
    }
    const inset = size * 0.13;
    const usable = size - inset * 2;
    const spanX = maxX - minX || 1;
    const spanZ = maxZ - minZ || 1;
    this.scale = usable / Math.max(spanX, spanZ);
    this.minX = minX;
    this.minZ = minZ;
    // Centre the course inside the frame rather than pinning it to a corner.
    this.padX = inset + (usable - spanX * this.scale) * 0.5;
    this.padY = inset + (usable - spanZ * this.scale) * 0.5;

    const path = new Path2D();
    for (let i = 0; i < N; i++) {
      const px = this.mx(pts[i * 2]);
      const py = this.my(pts[i * 2 + 1]);
      if (i === 0) path.moveTo(px, py);
      else path.lineTo(px, py);
    }
    path.closePath();
    this.path = path;

    // Gate ticks: a short bar perpendicular to the gate forward.
    this.gates = this.track.checkpoints.map((cp) => ({
      x: this.mx(cp.position.x),
      y: this.my(cp.position.z),
      nx: cp.forward.z,
      ny: -cp.forward.x,
    }));
    const s0 = this.track.sample(0);
    this.start = {
      x: this.mx(s0.position.x),
      y: this.my(s0.position.z),
      nx: s0.tangent.z,
      ny: -s0.tangent.x,
    };
  }

  private mx(worldX: number) {
    return (worldX - this.minX) * this.scale + this.padX;
  }
  private my(worldZ: number) {
    return (worldZ - this.minZ) * this.scale + this.padY;
  }

  // ── Static half ────────────────────────────────────────────────────────────

  drawChrome(g: CanvasRenderingContext2D, s: number) {
    const size = this.size;
    g.save();
    g.translate(this.x, this.y);

    // Chamfer matched to the plate skew ratio so the map window belongs to the
    // same corner language as every slanted plate in the overlay.
    const cut = size * PLATE_SKEW;
    const frame = cutPath(0, 0, size, size, cut, 0b1010); // TL + BR chamfer
    // Opaque sea, then the drawn swell hatch on top of it. The old frame filled
    // at 0.68 and washed the sea in at 0.55, so the ocean shader and the racing
    // line showed through the map — a map you can see the world through is not a
    // map, and it was the same transparency bug as the standings rows.
    plate(g, frame, s, { fill: rgba(HEX.waterDeep, 1) });
    g.save();
    g.clip(frame);
    hatch(g, frame, 0, 0, size, size, rgba(HEX.waterMid, 0.5), 11 * s, 1.4 * s);
    g.restore();

    // Inner rule — a double line is the cheapest way to make a panel look made.
    const inner = cutPath(5 * s, 5 * s, size - 10 * s, size - 10 * s, cut - 4 * s, 0b1010);
    inked(g, inner, null, rgba(HEX.hudDim, 0.55), 1.4 * s);

    // Course ribbon: ink under, race line over.
    if (this.path) {
      g.lineJoin = 'round';
      g.lineCap = 'round';
      g.strokeStyle = rgba(HEX.ink, 0.95);
      g.lineWidth = 8 * s;
      g.stroke(this.path);
      g.strokeStyle = rgba(HEX.raceLine, 0.95);
      g.lineWidth = 3.4 * s;
      g.stroke(this.path);
    }

    // Gate ticks.
    for (const gt of this.gates) {
      const l = 6.5 * s;
      g.strokeStyle = rgba(HEX.gate, 0.85);
      g.lineWidth = 2.4 * s;
      g.beginPath();
      g.moveTo(gt.x - gt.nx * l, gt.y - gt.ny * l);
      g.lineTo(gt.x + gt.nx * l, gt.y + gt.ny * l);
      g.stroke();
    }

    // Start / finish: a short chequered bar.
    {
      const st = this.start;
      const l = 9 * s;
      for (let i = -2; i < 2; i++) {
        g.fillStyle = i % 2 === 0 ? rgba(HEX.hudPaper, 0.95) : rgba(HEX.ink, 0.95);
        const t0 = (i / 2) * l;
        const t1 = ((i + 1) / 2) * l;
        g.beginPath();
        g.moveTo(st.x + st.nx * t0, st.y + st.ny * t0);
        g.lineTo(st.x + st.nx * t1, st.y + st.ny * t1);
        g.lineTo(st.x + st.nx * t1 - st.ny * 3.4 * s, st.y + st.ny * t1 + st.nx * 3.4 * s);
        g.lineTo(st.x + st.nx * t0 - st.ny * 3.4 * s, st.y + st.ny * t0 + st.nx * 3.4 * s);
        g.closePath();
        g.fill();
      }
    }

    // Corner tab label.
    inkText(g, 'COURSE', 9 * s, size - 7 * s, {
      font: `700 ${Math.round(9.5 * s)}px ${FONT_STACK}`,
      fill: rgba(HEX.hudDim, 0.95),
      tracking: 2.4 * s,
    });

    g.restore();
  }

  // ── Live half ──────────────────────────────────────────────────────────────

  drawLive(g: CanvasRenderingContext2D, ctx: GameContext, s: number, pulse: number) {
    g.save();
    g.translate(this.x, this.y);
    const cut = this.size * 0.16;
    g.clip(cutPath(0, 0, this.size, this.size, cut, 0b1010));

    // Next gate for the player, pulsing so the objective is never ambiguous.
    const nextIdx = ctx.player.nextCheckpoint % Math.max(1, this.gates.length);
    const gt = this.gates[nextIdx];
    if (gt) {
      const r = (7 + Math.sin(pulse * 6.2) * 2.2) * s;
      g.strokeStyle = rgba(HEX.gateFar, 0.9);
      g.lineWidth = 2.2 * s;
      g.beginPath();
      g.arc(gt.x, gt.y, r, 0, Math.PI * 2);
      g.stroke();
    }

    // Racers. Draw the player last so it always sits on top of the pack.
    const order = [...ctx.racers].sort((a, b) => (a.isPlayer ? 1 : 0) - (b.isPlayer ? 1 : 0));
    for (const r of order) {
      const px = this.mx(r.root.position.x);
      const py = this.my(r.root.position.z);
      const hex = [HEX.hull0, HEX.hull1, HEX.hull2, HEX.hull3][r.id];

      if (r.isPlayer) {
        // Heading wedge: world heading 0 faces +Z, which is +y on the map.
        const a = Math.atan2(Math.cos(r.state.heading), Math.sin(r.state.heading));
        g.save();
        g.translate(px, py);
        g.rotate(a);
        const L = 12 * s;
        const wedge = new Path2D();
        wedge.moveTo(L, 0);
        wedge.lineTo(-L * 0.55, L * 0.62);
        wedge.lineTo(-L * 0.2, 0);
        wedge.lineTo(-L * 0.55, -L * 0.62);
        wedge.closePath();
        inked(g, wedge, rgba(hex, 1), rgba(HEX.ink, 1), 2.2 * s);
        g.restore();
        // Locator ring — makes the player findable in a tight pack.
        g.strokeStyle = rgba(HEX.hudPaper, 0.75);
        g.lineWidth = 1.6 * s;
        g.beginPath();
        g.arc(px, py, 14.5 * s, 0, Math.PI * 2);
        g.stroke();
      } else {
        const d = diamondPath(px, py, 5.2 * s, 6.4 * s);
        inked(g, d, rgba(hex, 1), rgba(HEX.ink, 1), 2 * s);
      }
    }

    g.restore();

    // Lap-progress rail welded under the frame — a second read on "how far
    // round am I" that does not need the pip to be legible in a tight pack.
    const u = clamp01(ctx.player.progress % 1);
    const bx = this.x;
    const by = this.y + this.size + 4 * s;
    const bw = this.size;
    g.save();
    const rail = cutPath(bx, by, bw, 6 * s, 3 * s, 0b1010);
    inked(g, rail, rgba(HEX.hudInk, 1), rgba(HEX.hudPaper, 0.7), 1.6 * s);
    g.clip(rail);
    g.fillStyle = rgba(HEX.raceLine, 0.95);
    g.fillRect(bx, by, bw * u, 6 * s);
    g.restore();
  }

  /** Bottom of the map cluster, including the progress rail. */
  get bottom() {
    return this.y + this.size + 10;
  }
}
