/**
 * Full-screen presentation states: the countdown, the GO flash, the lap banner
 * and the results board.
 *
 * These are drawn on the same canvas as the HUD but kept in their own file
 * because they are *compositions* rather than instruments — they own the whole
 * frame for a second or two and their job is impact, not information density.
 *
 * All animation is driven off wall-clock deltas that the HUD hands in, so a
 * dropped frame shortens no beat. Nothing here reads `Date.now()`.
 */

import { CONFIG } from '../core/config';
import { clamp01, formatTime, ordinal } from '../core/mathx';
import { HEX } from '../core/palette';
import {
  FONT_STACK,
  PLATE_SKEW,
  PLATE_W,
  cutPath,
  diamondPath,
  hatch,
  inkText,
  inked,
  plate,
  rgba,
  segText,
  slantPath,
} from './inkDraw';
import type { GameContext, RacePhase } from '../core/types';

const easeOutCubic = (t: number) => 1 - Math.pow(1 - clamp01(t), 3);
const easeOutBack = (t: number) => {
  const c = 2.4;
  const u = clamp01(t) - 1;
  return 1 + (c + 1) * u * u * u + c * u * u;
};

export class Screens {
  /** Seconds since the current race phase began. */
  private phaseTime = 0;
  private lastPhase: RacePhase | 'none' = 'none';
  /** Latch for the GO flash, which outlives the countdown phase. */
  private goTimer = -1;
  private lastCountNumber = 99;
  /** 1 → 0 pop applied to the countdown numeral when it changes. */
  private countPop = 0;
  private lapBanner = -1;
  private lastLap = -1;

  update(ctx: GameContext, dt: number) {
    const phase = ctx.race.phase;
    if (phase !== this.lastPhase) {
      this.phaseTime = 0;
      this.lastPhase = phase;
      if (phase === 'racing') this.goTimer = 0;
    } else {
      this.phaseTime += dt;
    }
    if (this.goTimer >= 0) {
      this.goTimer += dt;
      if (this.goTimer > 1.15) this.goTimer = -1;
    }
    if (this.countPop > 0) this.countPop = Math.max(0, this.countPop - dt * 3.2);

    const lap = ctx.player.lap;
    if (this.lastLap < 0) this.lastLap = lap;
    else if (lap !== this.lastLap) {
      this.lastLap = lap;
      if (lap > 0 && lap < CONFIG.race.laps) this.lapBanner = 0;
    }
    if (this.lapBanner >= 0) {
      this.lapBanner += dt;
      if (this.lapBanner > 2.1) this.lapBanner = -1;
    }
  }

  // ── Countdown ──────────────────────────────────────────────────────────────

  /**
   * `raceTime` is negative through the countdown, so the seconds remaining and
   * the fraction through the current second both come straight out of it. We do
   * not use `race.countdownNumber` because it counts the full configured length
   * (4…1) and never reaches zero — the phase flips to racing first — so GO has
   * to be latched on the transition instead.
   */
  countdown(g: CanvasRenderingContext2D, ctx: GameContext, w: number, h: number, s: number) {
    const remain = -ctx.race.raceTime;
    if (remain <= 0) return;
    const n = Math.ceil(remain);
    const frac = clamp01(n - remain); // 0 at the top of the beat, 1 at its end
    if (n !== this.lastCountNumber) {
      this.lastCountNumber = n;
      this.countPop = 1;
    }

    // Sit the numeral high, clear of the horizon the cinematic orbit produces
    // (~y 0.27–0.31 h) and clear of the grid, which the orbit puts low-centre.
    const cx = w * 0.5;
    const cy = h * 0.2;

    // NO full-frame wash.
    //
    // This used to lay a radial ink gradient over the entire render at 0.6 alpha
    // in the centre. Measured against a racing frame it dropped sky-region mean
    // saturation from 0.552 to 0.454 and turned the two committed cloud tones
    // (white and skyHorizon, sat 0.34) into flat greys — (140,147,157) sat 0.11
    // and (242,251,255) sat 0.05 — i.e. the first frame a player sees was a
    // desaturated copy of a different, duller game. A soft radial alpha ramp is
    // also, optically, photographic haze, which is the one thing the art
    // direction forbids. The badge below is its own opaque scrim; the render is
    // now left exactly as the racing phase renders it.

    if (n > 3) {
      // The lead-in beat of a 4 s countdown: a marshalling tab, no numeral.
      this.tab(g, cx, cy, 'STAND BY', s, rgba(HEX.hudPaper, 0.92), frac);
      return;
    }

    const pop = this.countPop;
    const scale = 0.82 + easeOutBack(1 - pop) * 0.18;

    // Impact star behind the numeral.
    //
    // Two rings, not one filled blob. The previous version was a single solid
    // 13-point star at R = 0.3 h with an inner radius of 0.58 R, which at capture
    // scale was a 670 × 640 px dead-ink mass sitting exactly on the horizon line —
    // it deleted the one line that establishes the scene from the first frame of
    // the game, and swallowed a mid-distance gate with it.
    //
    // Now: a compact opaque core the numeral sits on (312 px wide at capture
    // scale, 15 % of frame width instead of 23 %), plus long thin manga impact
    // spikes radiating out of it. The spikes are hard-edged and fully opaque, but
    // they cover only about half the annulus, so sky, horizon and gates read
    // between them. Total extent is ~60 % of the old badge.
    g.save();
    g.translate(cx, cy);
    g.rotate((n * 0.7 + frac * 0.12) % Math.PI);
    const burst = (0.62 + easeOutCubic(1 - pop) * 0.42) * (1 - frac * 0.18);
    const R = h * 0.185 * burst;
    const spikes = 13;
    // Spike ring: outer tip R, base on the core radius, so each spike is a thin
    // wedge with an equally wide gap beside it.
    const rays = new Path2D();
    for (let i = 0; i < spikes * 2; i++) {
      const a = (i / (spikes * 2)) * Math.PI * 2;
      const rr = i % 2 === 0 ? R : R * 0.5;
      const px = Math.cos(a) * rr;
      const py = Math.sin(a) * rr * 0.82;
      if (i === 0) rays.moveTo(px, py);
      else rays.lineTo(px, py);
    }
    rays.closePath();
    inked(g, rays, rgba(HEX.ink, 1), rgba(HEX.waterCrest, 0.75), 2.4 * s);
    // Opaque core the numeral is legible against. Faceted, not a circle — a disc
    // reads as a loading spinner.
    const core = new Path2D();
    for (let i = 0; i < 12; i++) {
      const a = (i / 12) * Math.PI * 2 + 0.13;
      const rr = R * (i % 2 === 0 ? 0.56 : 0.5);
      const px = Math.cos(a) * rr;
      const py = Math.sin(a) * rr * 0.82;
      if (i === 0) core.moveTo(px, py);
      else core.lineTo(px, py);
    }
    core.closePath();
    inked(g, core, rgba(HEX.ink, 1), rgba(HEX.waterCrest, 0.9), 3.2 * s);
    g.restore();

    // Shock ring, expanding out of the beat. Kept inside the badge's own footprint
    // so it cannot draw a second soft line across the sea.
    const ring = easeOutCubic(frac * 1.6);
    if (ring < 1) {
      g.save();
      g.strokeStyle = rgba(HEX.waterCrest, 0.55 * (1 - ring));
      g.lineWidth = 6 * s * (1 - ring);
      g.beginPath();
      g.ellipse(cx, cy, h * 0.12 + ring * h * 0.24, (h * 0.12 + ring * h * 0.24) * 0.72, 0, 0, Math.PI * 2);
      g.stroke();
      g.restore();
    }

    // Banner ribbons behind the numeral. Short and boldly edged so they read as
    // graphic furniture; the full-width bars they replaced read as grey slabs
    // floating in the sea.
    g.save();
    g.translate(cx, cy);
    for (const [dy, hh, al, edge] of [
      [-h * 0.082, 13 * s, 1, HEX.boost],
      [h * 0.09, 8 * s, 1, HEX.waterCrest],
    ] as const) {
      const bw2 = w * 0.22;
      const bar = slantPath(-bw2 * 0.5, dy, bw2, hh, 18 * s);
      inked(g, bar, rgba(HEX.ink, al), rgba(edge, 0.95), PLATE_W * s);
    }
    g.restore();

    // The numeral.
    g.save();
    g.translate(cx, cy);
    g.scale(scale, scale);
    const px = Math.round(h * 0.235);
    inkText(g, String(n), 0, px * 0.37, {
      font: `900 ${px}px ${FONT_STACK}`,
      fill: rgba(HEX.hudPaper, 1),
      ink: rgba(HEX.ink, 1),
      inkWidth: 13 * s,
      align: 'center',
      skew: 0.16,
      ghost: rgba(HEX.boost, 0.95),
      ghostDx: 9 * s,
      ghostDy: 10 * s,
    });
    g.restore();

    // Beat pips sit *above* the numeral, like a gantry of start lights. Below it
    // they landed on the player's bow (shots/pres_r2/countdown.png).
    const pipY = cy - h * 0.155;
    for (let i = 0; i < 3; i++) {
      const lit = 3 - i <= n;
      const d = diamondPath(cx + (i - 1) * 30 * s, pipY, 11 * s, 14 * s);
      inked(
        g,
        d,
        lit ? rgba(HEX.warn, 1) : rgba(HEX.hudInk, 0.85),
        rgba(HEX.hudPaper, lit ? 0.95 : 0.45),
        2.6 * s,
      );
    }
  }

  /** GO! — latched on the countdown→racing transition, lives ~1 s. */
  go(g: CanvasRenderingContext2D, w: number, h: number, s: number) {
    if (this.goTimer < 0) return;
    const t = this.goTimer / 1.15;
    const cx = w * 0.5;
    const cy = h * 0.34;
    const grow = easeOutCubic(Math.min(1, t * 2.6));
    const fade = 1 - clamp01((t - 0.55) / 0.45);

    // Radial speed streaks — the read is "the frame itself accelerated".
    g.save();
    g.globalAlpha = fade;
    g.translate(cx, cy);
    g.strokeStyle = rgba(HEX.hudPaper, 0.5);
    for (let i = 0; i < 22; i++) {
      const a = (i / 22) * Math.PI * 2 + 0.13;
      const r0 = h * (0.16 + grow * 0.34);
      const r1 = r0 + h * 0.1 * (0.4 + ((i * 37) % 11) / 11);
      g.lineWidth = (1.2 + ((i * 13) % 5) * 0.7) * s;
      g.beginPath();
      g.moveTo(Math.cos(a) * r0, Math.sin(a) * r0 * 0.8);
      g.lineTo(Math.cos(a) * r1, Math.sin(a) * r1 * 0.8);
      g.stroke();
    }
    g.restore();

    g.save();
    g.globalAlpha = fade;
    g.translate(cx, cy);
    g.scale(0.7 + grow * 0.42, 0.7 + grow * 0.42);
    const px = Math.round(h * 0.3);
    // Ink slab behind the word so it reads even over bright foam.
    const slab = slantPath(-w * 0.24, -px * 0.62, w * 0.48, px * 0.98, 34 * s);
    inked(g, slab, rgba(HEX.ink, 1), rgba(HEX.raceLine, 0.9), PLATE_W * s);
    inkText(g, 'GO!', 0, px * 0.3, {
      font: `900 ${px}px ${FONT_STACK}`,
      fill: rgba(HEX.raceLine, 1),
      ink: rgba(HEX.ink, 1),
      inkWidth: 13 * s,
      align: 'center',
      skew: 0.2,
      tracking: 2 * s,
      ghost: rgba(HEX.boostHot, 0.9),
      ghostDx: 9 * s,
      ghostDy: 9 * s,
    });
    g.restore();
  }

  /** Small tab used for STAND BY and the lap banner. */
  private tab(
    g: CanvasRenderingContext2D,
    cx: number,
    cy: number,
    label: string,
    s: number,
    color: string,
    pulse = 0,
  ) {
    g.save();
    g.font = `800 ${Math.round(26 * s)}px ${FONT_STACK}`;
    const tw = g.measureText(label).width + 12 * s * label.length * 0.12 + 56 * s;
    const hgt = 48 * s;
    const p = slantPath(cx - tw / 2, cy - hgt / 2, tw, hgt, 16 * s);
    inked(g, p, rgba(HEX.ink, 1), rgba(HEX.hudPaper, 0.55 + 0.35 * Math.sin(pulse * 8)), PLATE_W * s);
    g.restore();
    inkText(g, label, cx, cy + 9 * s, {
      font: `800 ${Math.round(26 * s)}px ${FONT_STACK}`,
      fill: color,
      align: 'center',
      skew: 0.14,
      tracking: 5 * s,
    });
  }

  /** "LAP 2" slide-in, shown for a beat after crossing the line. */
  lapFlash(g: CanvasRenderingContext2D, ctx: GameContext, w: number, h: number, s: number) {
    if (this.lapBanner < 0) return;
    const t = this.lapBanner;
    const inT = easeOutCubic(Math.min(1, t / 0.3));
    const out = 1 - clamp01((t - 1.5) / 0.6);
    const y = h * 0.26;
    const x = w * 0.5 + (1 - inT) * w * 0.28;
    g.save();
    g.globalAlpha = out;
    this.tab(g, x, y, `LAP ${Math.min(ctx.player.lap + 1, CONFIG.race.laps)}`, s, rgba(HEX.boostHot, 1), t);
    g.restore();
  }

  // ── Results ────────────────────────────────────────────────────────────────

  /**
   * Results board.
   *
   * Anchored to the **right** of frame rather than centred. The cinematic orbit
   * puts the winning boat and its rider low-left, and a centred board sat right
   * on top of them (shots/pres_r3/results.png — the hull is a barely-visible
   * smudge behind row 4). A right-hand board plus a left-biased camera look point
   * gives the celebration half the frame and the data the other half, which is
   * how a real race broadcast lays this out.
   *
   * Rows are right-aligned with a staggered left edge, so the times form one
   * clean column while the plates still step down the screen.
   */
  results(g: CanvasRenderingContext2D, ctx: GameContext, w: number, h: number, s: number) {
    const T = this.phaseTime;
    const board = ctx.race.standings();
    const wash = clamp01(T / 0.45);

    const rowW = Math.min(w * 0.54, 660 * s);
    const rowH = 62 * s;
    const gap = 11 * s;
    const boardRight = w - 54 * s;
    const x0 = boardRight - rowW;
    const boardCx = x0 + rowW * 0.5;
    const y0 = h * 0.31;
    const rowsBottom = y0 + board.length * (rowH + gap);

    // ── Backdrop ─────────────────────────────────────────────────────────────
    //
    // ONE hard-edged opaque panel on the board side. Nothing washes the render.
    //
    // What was here before: a full-frame horizontal ink gradient from 0.5 to 0.9
    // alpha. That is a multiplicative grey wash over the whole image — the water
    // came out a flat grey-navy, the hull vermilion came out brick, the clouds
    // came out mid-grey, and the frame the player stares at longest looked like a
    // different, duller game than the one they just raced. A soft alpha ramp is
    // also photographic, which the art direction rules out outright.
    //
    // The replacement is a graphic: a flat `ink` parallelogram with a hard slanted
    // leading edge and a crest keyline down it, leaning with the same PLATE_SKEW
    // as every other plate in the game. Left of that edge the render is untouched
    // at full saturation, which is where the celebration is. Right of it nothing
    // from the world can composite through — which is also what stops a gate mast
    // running diagonally behind the VICTORY / 1ST lockup.
    const panelLean = 104 * s;
    const panelX = x0 - 46 * s;
    const panel = slantPath(panelX, -4 * s, w - panelX + 8 * s, h + 8 * s, panelLean);
    g.save();
    g.globalAlpha = wash;
    inked(g, panel, rgba(HEX.ink, 1), null, 0);
    // Diagonal hatch, now genuinely confined to the panel backing. Over an already
    // opaque fill this is authored shading, not transparency.
    hatch(g, panel, panelX, 0, w - panelX + panelLean, h, rgba(HEX.inkSoft, 0.9), 16 * s, 2 * s);
    // The leading edge, drawn as a line rather than implied by an alpha ramp.
    g.strokeStyle = rgba(HEX.waterCrest, 0.9);
    g.lineWidth = 3.4 * s;
    g.beginPath();
    g.moveTo(panelX + panelLean, -4 * s);
    g.lineTo(panelX, h + 4 * s);
    g.stroke();
    // A second, darker rule inboard of it, so the edge is a two-tone brushed
    // border like every plate in the game rather than one hairline. Deliberately
    // *not* magenta: a thin magenta line beside a thin cyan one is exactly the
    // chromatic-aberration read the racing-line ribbon was pulled up for.
    g.strokeStyle = rgba(HEX.inkSoft, 1);
    g.lineWidth = 5 * s;
    g.beginPath();
    g.moveTo(panelX + panelLean + 8 * s, -4 * s);
    g.lineTo(panelX + 8 * s, h + 4 * s);
    g.stroke();
    g.restore();

    const player = ctx.player;
    const headY = h * 0.145;

    // ── Headline: the player's placement, which is the only number that matters
    const hIn = easeOutBack(Math.min(1, T / 0.5));
    g.save();
    g.globalAlpha = clamp01(T / 0.25);
    g.translate(boardCx, headY);
    g.scale(0.82 + hIn * 0.18, 0.82 + hIn * 0.18);
    const won = player.place === 1;
    inkText(g, won ? 'VICTORY' : 'RACE COMPLETE', 0, -34 * s, {
      font: `800 ${Math.round(21 * s)}px ${FONT_STACK}`,
      fill: rgba(won ? HEX.boostHot : HEX.hudDim, 1),
      align: 'center',
      skew: 0.14,
      tracking: 9 * s,
    });
    const place = ordinal(player.place).toUpperCase();
    inkText(g, place, 0, 42 * s, {
      font: `900 ${Math.round(80 * s)}px ${FONT_STACK}`,
      fill: rgba(HEX.hudPaper, 1),
      ink: rgba(HEX.ink, 1),
      inkWidth: 10 * s,
      align: 'center',
      skew: 0.17,
      ghost: rgba(won ? HEX.boostHot : HEX.boost, 0.9),
      ghostDx: 8 * s,
      ghostDy: 9 * s,
    });
    // Chevron wings either side of the placement.
    for (const dir of [-1, 1]) {
      g.save();
      g.strokeStyle = rgba(HEX.hudPaper, 0.7);
      g.lineWidth = 4 * s;
      for (let i = 0; i < 3; i++) {
        const x = dir * (122 + i * 20) * s;
        g.beginPath();
        g.moveTo(x, 6 * s);
        g.lineTo(x + dir * 13 * s, 24 * s);
        g.lineTo(x, 42 * s);
        g.stroke();
      }
      g.restore();
    }
    g.restore();

    // ── Standings rows ───────────────────────────────────────────────────────
    for (let i = 0; i < board.length; i++) {
      const r = board[i];
      const t = clamp01((T - 0.28 - i * 0.11) / 0.44);
      if (t <= 0) continue;
      const e = easeOutCubic(t);
      // Left edge steps in as we go down the order; the right edge — and so the
      // whole time column — stays put.
      const rw = rowW - i * 18 * s;
      const rx = x0 + i * 18 * s + (1 - e) * 150 * s;
      const ry = y0 + i * (rowH + gap);
      const hex = [HEX.hull0, HEX.hull1, HEX.hull2, HEX.hull3][r.id];
      const isPlayer = r.isPlayer;
      const by = ry + rowH * 0.5;

      g.save();
      g.globalAlpha = e;

      // Same plate treatment as every panel in the game: opaque fill, hard ink
      // keyline, one outline weight. Which row is yours is said by the chip, the
      // rail and the name brightness, not by a heavier frame.
      const row = slantPath(rx, ry, rw, rowH, rowH * PLATE_SKEW);
      plate(g, row, s, {
        fill: rgba(isPlayer ? HEX.hudInk : HEX.inkSoft, 1),
        edge: rgba(HEX.hudPaper, isPlayer ? 0.95 : 0.72),
      });

      // Colour chip welded to the left edge; the player's row also gets a
      // full-length hull-colour rail so it reads as *your* row, not merely as
      // the highlighted one.
      const chip = slantPath(rx + 4 * s, ry + 4 * s, 12 * s, rowH - 8 * s, 20 * s);
      inked(g, chip, rgba(hex, 1), rgba(HEX.ink, 0.9), 2 * s);
      if (isPlayer) {
        const rail = slantPath(rx + 2 * s, ry + rowH - 7 * s, rw - 4 * s, 4 * s, 1.4 * s);
        inked(g, rail, rgba(hex, 0.9), null, 0);
      }

      // Place badge. Typeset, not segmented: a seven-segment "1" inside a
      // diamond reads as a stray tally mark.
      const bx = rx + 56 * s;
      const badge = diamondPath(bx, by, 23 * s, 26 * s);
      inked(g, badge, rgba(i === 0 ? HEX.boostHot : HEX.hudInk, 1), rgba(HEX.hudPaper, 0.85), 2.6 * s);
      inkText(g, String(i + 1), bx + 1 * s, by + 12 * s, {
        font: `900 ${Math.round(33 * s)}px ${FONT_STACK}`,
        fill: rgba(i === 0 ? HEX.ink : HEX.hudPaper, 1),
        align: 'center',
        skew: 0.16,
      });

      inkText(g, r.name.toUpperCase(), rx + 96 * s, by + 11 * s, {
        font: `800 ${Math.round(30 * s)}px ${FONT_STACK}`,
        fill: rgba(isPlayer ? HEX.hudPaper : HEX.foamShade, 1),
        align: 'left',
        skew: 0.13,
        tracking: 2.2 * s,
      });

      // Gap to the winner, filling the space between name and time.
      const gapX = rx + rw - 190 * s;
      if (i > 0) {
        const lead = board[0];
        const d = r.finished && lead.finished ? r.finishTime - lead.finishTime : NaN;
        inkText(g, isFinite(d) ? `+${d.toFixed(1)}` : '—', gapX, by + 8 * s, {
          font: `800 ${Math.round(21 * s)}px ${FONT_STACK}`,
          fill: rgba(HEX.hudDim, 1),
          align: 'right',
          skew: 0.12,
        });
      } else {
        inkText(g, 'WINNER', gapX, by + 6 * s, {
          font: `800 ${Math.round(13 * s)}px ${FONT_STACK}`,
          fill: rgba(HEX.boostHot, 1),
          align: 'right',
          tracking: 3 * s,
        });
      }

      // Finish time, right-aligned, in the same seven-segment face as the HUD.
      const tx = rx + rw - 24 * s;
      if (r.finished) {
        // Lit segments only. The unlit field behind these turned "3:41.300" and
        // "L1 1:11.200" into ambiguous glyphs at capture scale.
        segText(g, formatTime(r.finishTime), tx, ry + 11 * s, 25 * s, {
          lit: rgba(HEX.hudPaper, 1),
          ink: rgba(HEX.ink, 1),
          inkWidth: 1.4 * s,
          align: 'right',
          skew: 0.09,
        });
      } else {
        inkText(g, 'DNF', tx, ry + 33 * s, {
          font: `800 ${Math.round(24 * s)}px ${FONT_STACK}`,
          fill: rgba(HEX.warn, 1),
          align: 'right',
          skew: 0.12,
        });
      }
      const best = isFinite(r.bestLap) ? formatTime(r.bestLap) : '--:--.---';
      inkText(g, `BEST ${best}`, tx, ry + rowH - 9 * s, {
        font: `700 ${Math.round(13 * s)}px ${FONT_STACK}`,
        fill: rgba(HEX.hudDim, 1),
        align: 'right',
        tracking: 1.4 * s,
      });

      g.restore();
    }

    // ── Fastest lap plaque, player splits, restart prompt ────────────────────
    const fl = board.reduce((a, b) => (b.bestLap < a.bestLap ? b : a), board[0]);
    const footY = rowsBottom + 30 * s;
    const fIn = clamp01((T - 0.85) / 0.4);
    if (fIn > 0) {
      g.save();
      g.globalAlpha = fIn;
      const pw = Math.min(rowW * 0.72, 424 * s);
      const px = boardCx - pw * 0.5;
      const plaque = cutPath(px, footY - 24 * s, pw, 46 * s, 14 * s, 0b0101);
      plate(g, plaque, s, { edge: rgba(HEX.raceLine, 0.9) });
      inkText(g, 'FASTEST LAP', px + 18 * s, footY + 4 * s, {
        font: `800 ${Math.round(13 * s)}px ${FONT_STACK}`,
        fill: rgba(HEX.raceLine, 1),
        align: 'left',
        tracking: 2.6 * s,
      });
      inkText(g, fl.name.toUpperCase(), px + 160 * s, footY + 4 * s, {
        font: `800 ${Math.round(14 * s)}px ${FONT_STACK}`,
        fill: rgba(HEX.foamShade, 1),
        align: 'left',
        skew: 0.1,
        tracking: 1.6 * s,
      });
      const lapStr = isFinite(fl.bestLap) ? formatTime(fl.bestLap) : '0:00.000';
      segText(g, lapStr, px + pw - 24 * s, footY - 14 * s, 25 * s, {
        lit: rgba(HEX.hudPaper, 1),
        ink: rgba(HEX.ink, 1),
        inkWidth: 1.4 * s,
        align: 'right',
        skew: 0.09,
      });
      g.restore();

      // Player's lap-by-lap splits. This is the only place in the game the whole
      // race is legible at once, so the splits belong here rather than being
      // thrown away with the HUD.
      const laps = ctx.player.lapTimes;
      if (laps.length) {
        const cellW = Math.min(150 * s, (rowW - (laps.length - 1) * 10 * s) / laps.length);
        const totalW = laps.length * cellW + (laps.length - 1) * 10 * s;
        const sx = boardCx - totalW * 0.5;
        const sy = footY + 40 * s;
        const bestVal = Math.min(...laps);
        g.save();
        g.globalAlpha = clamp01((T - 1.0) / 0.4);
        for (let i = 0; i < laps.length; i++) {
          const x = sx + i * (cellW + 10 * s);
          const isBest = laps[i] === bestVal;
          const cell = slantPath(x, sy, cellW, 38 * s, 38 * s * PLATE_SKEW);
          plate(g, cell, s, {
            edge: isBest ? rgba(HEX.raceLine, 0.9) : rgba(HEX.hudPaper, 0.72),
          });
          inkText(g, `L${i + 1}`, x + 16 * s, sy + 26 * s, {
            font: `800 ${Math.round(14 * s)}px ${FONT_STACK}`,
            fill: rgba(isBest ? HEX.raceLine : HEX.hudDim, 1),
            align: 'left',
            skew: 0.12,
          });
          segText(g, formatTime(laps[i]), x + cellW - 12 * s, sy + 10 * s, 18 * s, {
            lit: rgba(isBest ? HEX.raceLine : HEX.foamShade, 1),
            ink: rgba(HEX.ink, 1),
            inkWidth: 1.1 * s,
            align: 'right',
            skew: 0.09,
          });
        }
        g.restore();
      }
    }

    const pIn = clamp01((T - 1.15) / 0.4);
    if (pIn > 0) {
      const blink = 0.55 + 0.45 * Math.sin(T * 5.2);
      g.save();
      g.globalAlpha = pIn;
      const py = h - 46 * s;
      for (const dir of [-1, 1]) {
        g.strokeStyle = rgba(HEX.boostHot, 0.9 * blink);
        g.lineWidth = 3.4 * s;
        const bx2 = boardCx + dir * 168 * s;
        g.beginPath();
        g.moveTo(bx2 - dir * 10 * s, py - 12 * s);
        g.lineTo(bx2 + dir * 6 * s, py - 3 * s);
        g.lineTo(bx2 - dir * 10 * s, py + 6 * s);
        g.stroke();
      }
      inkText(g, 'PRESS  R  TO RACE AGAIN', boardCx, py + 2 * s, {
        font: `800 ${Math.round(17 * s)}px ${FONT_STACK}`,
        fill: rgba(HEX.hudPaper, 0.55 + 0.45 * blink),
        align: 'center',
        skew: 0.1,
        tracking: 4 * s,
      });
      g.restore();
    }
  }
  /** Cross-the-line flash used while the race is `finished` but pre-results. */
  finishFlash(g: CanvasRenderingContext2D, ctx: GameContext, w: number, h: number, s: number) {
    const T = this.phaseTime;
    const fade = 1 - clamp01((T - 1.2) / 0.9);
    g.save();
    g.globalAlpha = fade;
    const cx = w * 0.5;
    const cy = h * 0.4;
    const slab = slantPath(cx - w * 0.26, cy - 44 * s, w * 0.52, 88 * s, 30 * s);
    inked(g, slab, rgba(HEX.ink, 1), rgba(HEX.boostHot, 0.9), PLATE_W * s);
    inkText(g, 'FINISH', cx, cy + 22 * s, {
      font: `900 ${Math.round(62 * s)}px ${FONT_STACK}`,
      fill: rgba(HEX.hudPaper, 1),
      ink: rgba(HEX.ink, 1),
      inkWidth: 9 * s,
      align: 'center',
      skew: 0.18,
      tracking: 6 * s,
      ghost: rgba(HEX.boost, 0.85),
      ghostDx: 7 * s,
      ghostDy: 8 * s,
    });
    inkText(g, ordinal(ctx.player.place).toUpperCase(), cx, cy + 74 * s, {
      font: `800 ${Math.round(24 * s)}px ${FONT_STACK}`,
      fill: rgba(HEX.boostHot, 1),
      align: 'center',
      skew: 0.12,
      tracking: 6 * s,
    });
    g.restore();
  }

  /** Reset the numeral latch when the harness rewinds the race. */
  resetLatches() {
    this.lastCountNumber = 99;
    this.goTimer = -1;
    this.lapBanner = -1;
    this.lastLap = -1;
  }
}
