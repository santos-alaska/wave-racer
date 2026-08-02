/**
 * INK TIDE — the HUD.
 *
 * Canvas 2D, hand-composed, no DOM, no fonts beyond the system stack (headline
 * *numerals* are drawn as geometry — see `inkDraw.segText`). Everything is built
 * from the primitives in `inkDraw.ts` so the whole overlay reads as one piece of
 * ink art rather than four widgets.
 *
 * Structure of a frame:
 *
 *   1. Blit the **chrome layer** — the static half of the overlay (gauge dial,
 *      tick ring, minimap frame and course ribbon). It is baked once per resize
 *      into its own DPR-sized canvas. Re-stroking a 260-point spline and 30 tick
 *      marks every frame is pure waste, and baking also keeps the hairlines
 *      identical frame to frame instead of shimmering.
 *   2. Draw the **live layer** — needle, digits, meters, pips, standings.
 *   3. Draw the **screens** (countdown / GO / lap flash / results) on top.
 *
 * The in-race instruments fade out entirely for the countdown and the results
 * board, so those moments get the whole frame. `hudAlpha` is damped, never
 * snapped, so the fade reads as a deliberate wipe.
 *
 * Layout is anchored to the four corners and scaled by a single `s` factor
 * derived from the viewport height, so the composition holds from a laptop
 * window to a retina 1440p capture without a media query.
 */

import { CONFIG } from '../core/config';
import { HEX } from '../core/palette';
import { clamp, clamp01, damp, formatTime, ordinal } from '../core/mathx';
import {
  FONT_STACK,
  PLATE_SKEW,
  PLATE_W,
  chevronPath,
  cutPath,
  hatch,
  inkText,
  inked,
  plate,
  rgba,
  segText,
  segWidth,
  slantPath,
} from './inkDraw';
import { Minimap } from './minimap';
import { Screens } from './screens';
import type { GameContext, HudAPI, Racer, TrackAPI } from '../core/types';

/** Full-scale gauge reading. boostTopSpeed is 39 m/s ≈ 140 km/h. */
const GAUGE_MAX_KMH = 150;
const REDLINE_KMH = 120;
/** Dial sweep, canvas angles (0 = +x, positive clockwise because y is down). */
const DIAL_A0 = (130 * Math.PI) / 180;
const DIAL_SWEEP = (280 * Math.PI) / 180;
/** Even angular division of the scale — one mark per 10 km/h, long every third. */
const TICKS = GAUGE_MAX_KMH / 10;
/**
 * Power band radius (inset from `gr`) and width. The band, the redline and the
 * needle tip all live on this radius so the three cannot appear to disagree: the
 * old band sat 24 px inside the tick ring, far enough that reading it against the
 * scale was guesswork.
 */
const BAND_R = 20;
const BAND_W = 9;

interface Layout {
  s: number;
  /** Speedometer dial. */
  gx: number;
  gy: number;
  gr: number;
  /** Digital readout plate, left of the dial. */
  rx: number;
  ry: number;
  rw: number;
  rh: number;
  /** Top-left info slab — one plate, placement + lap + clock + splits. */
  ix: number;
  iy: number;
  iw: number;
  /** Height of the placement/lap band, and of the whole slab. */
  ih1: number;
  ih: number;
  /** Boost meter origin. */
  bx: number;
  by: number;
  bw: number;
  bh: number;
}

export class Hud implements HudAPI {
  private ctx2d: CanvasRenderingContext2D;
  private w = 1;
  private h = 1;
  private dpr = 1;

  /** Baked static overlay. */
  private chrome: HTMLCanvasElement;
  private chromeCtx: CanvasRenderingContext2D;

  private minimap: Minimap;
  private screens = new Screens();
  private L: Layout = {
    s: 1, gx: 0, gy: 0, gr: 1, rx: 0, ry: 0, rw: 1, rh: 1,
    ix: 0, iy: 0, iw: 1, ih1: 1, ih: 1, bx: 0, by: 0, bw: 1, bh: 1,
  };

  // ── Animation state ────────────────────────────────────────────────────────
  /** Needle and readout lag the physics slightly; a gauge with no inertia looks fake. */
  private needleKmh = 0;
  private shownKmh = 0;
  private hudAlpha = 0;
  /** Position-change animation: 1 → 0, sign in `placeDir`. */
  private placeFlash = 0;
  private placeDir = 0;
  private lastPlace = 0;
  /** Per-tier charge flash. */
  private tierFlash = [0, 0, 0];
  private lastTier = 0;
  private boostPulse = 0;
  private lastBoostTime = 0;
  private pulse = 0;

  constructor(
    private canvas: HTMLCanvasElement,
    private track: TrackAPI,
  ) {
    const c = canvas.getContext('2d');
    if (!c) throw new Error('HUD: 2D context unavailable');
    this.ctx2d = c;
    this.minimap = new Minimap(track);

    this.chrome = document.createElement('canvas');
    const cc = this.chrome.getContext('2d');
    if (!cc) throw new Error('HUD: chrome 2D context unavailable');
    this.chromeCtx = cc;
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Layout
  // ───────────────────────────────────────────────────────────────────────────

  resize(width: number, height: number, dpr: number) {
    this.w = width;
    this.h = height;
    this.dpr = dpr;

    for (const cv of [this.canvas, this.chrome]) {
      cv.width = Math.max(1, Math.floor(width * dpr));
      cv.height = Math.max(1, Math.floor(height * dpr));
    }
    this.canvas.style.width = width + 'px';
    this.canvas.style.height = height + 'px';
    this.ctx2d.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.chromeCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // One scale factor for the whole overlay. Tied to height because that is
    // what constrains a bottom-anchored gauge.
    const s = clamp(height / 810, 0.62, 1.55);
    // Minimap + standings stack, sized so the *bottom of the last standings row*
    // clears the highest horizon the chase camera produces. Measured across the
    // capture set, the horizon runs y 640–800 device px (0.40–0.49 h); the stack
    // used to end at y 780 device, so it permanently occupied right-hand racing
    // space and in shots/pres2_base/ocean_low.png the 2nd-place AI — the most
    // tactically important object in frame — was a quarter hidden behind it.
    // 22 + 160 + 12 + 4 × 28 = 306 CSS px = 612 device px, i.e. above the band.
    const map = Math.round(160 * s);
    // Speedometer tucked further into the corner (and slightly smaller) for the
    // same reason: in pack.png the violet AI was behind it. This moves the dial's
    // top-left corner 26 CSS / 52 device px right and 40 CSS / 80 device px down.
    const gr = 76 * s;
    const gx = width - 100 * s;
    const gy = height - 108 * s;
    const rw = 186 * s;

    this.L = {
      s,
      gr,
      gx,
      gy,
      rw,
      rh: 84 * s,
      rx: gx - gr - 32 * s - rw,
      ry: gy - 42 * s,
      ix: 28 * s,
      iy: 26 * s,
      iw: 336 * s,
      ih1: 92 * s,
      ih: 144 * s,
      bx: 30 * s,
      by: height - 78 * s,
      bw: 78 * s,
      bh: 31 * s,
    };

    this.minimap.layout(width - 30 * s - map, 22 * s, map);
    this.computeChromeRects();
    this.bakeChrome();
  }

  /** Draw everything that does not change between frames. */
  private bakeChrome() {
    const g = this.chromeCtx;
    const { s, gx, gy, gr } = this.L;
    g.clearRect(0, 0, this.w, this.h);

    this.minimap.drawChrome(g, s);

    // ── Speedometer dial ────────────────────────────────────────────────────
    g.save();
    g.translate(gx, gy);

    // Dial face: a *filled wedge* — the sweep arc closed back through the centre
    // — not an annulus. As an annulus the hub area was open scene, and a whole
    // yellow AI boat composited through the middle of the gauge
    // (shots/pres_base/rider_closeup.png). The open bottom of the sweep is still
    // a real cut in the shape rather than a clipped circle.
    const hatchRect: [number, number, number, number] = [
      -gr - 12 * s, -gr - 12 * s, (gr + 12 * s) * 2, (gr + 12 * s) * 2,
    ];
    const face = new Path2D();
    face.moveTo(0, 0);
    face.arc(0, 0, gr + 9 * s, DIAL_A0, DIAL_A0 + DIAL_SWEEP);
    face.closePath();
    plate(g, face, s, { hatchRect });

    // Hub disc over the wedge, so the *mouth* of the sweep reads as dial rather
    // than as sea: with the wedge alone, the 80° cut at the bottom left a bright
    // triangle of open water inside the instrument — the same bug as a
    // transparent plate. Its edge doubles as the dial's inner rule.
    const disc = new Path2D();
    disc.arc(0, 0, gr - 40 * s, 0, Math.PI * 2);
    inked(g, disc, rgba(HEX.hudInk, 1), rgba(HEX.hudDim, 0.55), 1.6 * s);
    hatch(g, disc, hatchRect[0], hatchRect[1], hatchRect[2], hatchRect[3],
      rgba(HEX.inkSoft, 0.85), 10 * s, 1.3 * s);

    // Redline band, hard-ended, sitting on the same radius as the power band so
    // the two read on one scale.
    g.lineCap = 'butt';
    const rlFrom = DIAL_A0 + DIAL_SWEEP * (REDLINE_KMH / GAUGE_MAX_KMH);
    g.strokeStyle = rgba(HEX.ink, 1);
    g.lineWidth = BAND_W * s + 4 * s;
    g.beginPath();
    g.arc(0, 0, gr - BAND_R * s, rlFrom, DIAL_A0 + DIAL_SWEEP);
    g.stroke();
    g.strokeStyle = rgba(HEX.warn, 1);
    g.lineWidth = BAND_W * s;
    g.beginPath();
    g.arc(0, 0, gr - BAND_R * s, rlFrom, DIAL_A0 + DIAL_SWEEP);
    g.stroke();

    // Ticks, on an even angular division of the scale: 16 marks at 10 km/h, every
    // third one long. No numerals — the digital readout beside the dial carries
    // the exact figure and the redline band marks the top of the scale.
    for (let i = 0; i <= TICKS; i++) {
      const a = DIAL_A0 + DIAL_SWEEP * (i / TICKS);
      const major = i % 3 === 0;
      const r0 = gr + 2 * s;
      const r1 = gr - (major ? 13 : 7) * s;
      g.strokeStyle = major ? rgba(HEX.hudPaper, 0.95) : rgba(HEX.hudDim, 0.9);
      g.lineWidth = (major ? 3.4 : 1.8) * s;
      g.beginPath();
      g.moveTo(Math.cos(a) * r0, Math.sin(a) * r0);
      g.lineTo(Math.cos(a) * r1, Math.sin(a) * r1);
      g.stroke();
    }
    g.restore();

    // ── Digital readout plate ───────────────────────────────────────────────
    // SPEED and KM/H share the top line. Stacking the unit under the digits put
    // it inside their descender band and the numerals struck straight through it
    // (visible in shots/pres_r3/hero.png).
    const { rx, ry, rw, rh } = this.L;

    // Spine welding the readout to the dial, so the two are one instrument
    // rather than two floating plates.
    const spine = slantPath(rx + rw - 6 * s, ry + rh * 0.42, gx - gr - rx - rw + 20 * s, 9 * s, 3 * s);
    inked(g, spine, rgba(HEX.hudInk, 1), rgba(HEX.hudPaper, 0.55), 2 * s);

    const readout = slantPath(rx, ry, rw, rh, rh * PLATE_SKEW);
    plate(g, readout, s, { hatchRect: [rx, ry, rw, rh] });
    inkText(g, 'SPEED', rx + 18 * s, ry + 21 * s, {
      font: `800 ${Math.round(11 * s)}px ${FONT_STACK}`,
      fill: rgba(HEX.hudDim, 1),
      align: 'left',
      tracking: 3.4 * s,
    });
    inkText(g, 'KM/H', rx + rw - 14 * s, ry + 21 * s, {
      font: `800 ${Math.round(11.5 * s)}px ${FONT_STACK}`,
      fill: rgba(HEX.foamShade, 0.9),
      align: 'right',
      tracking: 3.2 * s,
    });
    // A rule under the caption line, then the digits below it.
    g.strokeStyle = rgba(HEX.hudDim, 0.45);
    g.lineWidth = 1.6 * s;
    g.beginPath();
    g.moveTo(rx + 14 * s, ry + 27 * s);
    g.lineTo(rx + rw - 10 * s, ry + 27 * s);
    g.stroke();
    // No unlit "888" field is baked here. Even at 9 % alpha the leading ghost
    // digit read as a real one — "88" became "888" at 4× magnification in
    // shots/pres_base/hero.png. The lit digits are the whole display.

    // ── Boost meter: backing plate, caption, empty cells ────────────────────
    const { bx, by, bw, bh } = this.L;
    // The meter used to float as three unbacked chevrons in the corner while
    // every other cluster sat on a plate; giving it the same slab pulls the
    // bottom-left of the composition back into the same design language.
    const bpw = 3 * (bw + 7 * s) + 20 * s;
    const bph = bh + 44 * s;
    const bPlate = slantPath(bx - 14 * s, by - 34 * s, bpw, bph, bph * PLATE_SKEW);
    plate(g, bPlate, s, { hatchRect: [bx - 14 * s, by - 34 * s, bpw, bph] });
    inkText(g, 'DRIFT', bx + 2 * s, by - 12 * s, {
      font: `800 ${Math.round(11.5 * s)}px ${FONT_STACK}`,
      fill: rgba(HEX.foamShade, 0.9),
      align: 'left',
      tracking: 3.2 * s,
    });
    for (let i = 0; i < 3; i++) {
      const x = bx + i * (bw + 7 * s);
      const cell = chevronPath(x, by, bw, bh, 11 * s);
      // Empty cell: opaque body, authored hatch, and the tier index in *ink* on
      // the paper — dark-on-light is legible at any size, where the old
      // hudDim-on-hudDim numeral was a 20 %-alpha smudge.
      inked(g, cell, rgba(HEX.inkSoft, 1), rgba(HEX.hudPaper, 0.92), PLATE_W * s);
      hatch(g, cell, x, by, bw, bh, rgba(HEX.hudDim, 0.5), 8 * s, 1.3 * s);
      segText(g, String(i + 1), x + 19 * s, by + bh * 0.5 - 6 * s, 12 * s, {
        lit: rgba(HEX.hudPaper, 0.85),
        align: 'center',
        skew: 0.1,
      });
    }
  }

  /**
   * Sub-rects of the chrome layer, one per screen cluster, in CSS px:
   * [x, y, w, h]. Recomputed on resize. Non-overlapping by construction.
   */
  private CHROME_MAP: [number, number, number, number] = [0, 0, 1, 1];
  private CHROME_GAUGE: [number, number, number, number] = [0, 0, 1, 1];
  private CHROME_BOOST: [number, number, number, number] = [0, 0, 1, 1];

  private computeChromeRects() {
    const { s, gx, gy, gr, rx, bx, by } = this.L;
    const mapLeft = Math.max(0, this.minimap.x - 12 * s);
    this.CHROME_MAP = [mapLeft, 0, this.w - mapLeft, this.minimap.y + this.minimap.size + 18 * s];
    const gTop = gy - gr - 16 * s;
    const gLeft = Math.max(0, rx - 12 * s);
    this.CHROME_GAUGE = [gLeft, gTop, this.w - gLeft, this.h - gTop];
    const bTop = by - 44 * s;
    // Left edge must clear the drift plate's *bottom*-left corner — the plate is a
    // parallelogram, so its lowest corner sits a full skew to the left of its top
    // one, and the ink keyline hangs 3 px outside that again. At `bx - 12` the
    // blit sliced ~3 px off the plate's own outline.
    this.CHROME_BOOST = [Math.max(0, bx - 24 * s), bTop, 400 * s, this.h - bTop];
  }

  private blitChrome(r: [number, number, number, number]) {
    const d = this.dpr;
    this.ctx2d.drawImage(
      this.chrome,
      r[0] * d, r[1] * d, r[2] * d, r[3] * d,
      r[0], r[1], r[2], r[3],
    );
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Frame
  // ───────────────────────────────────────────────────────────────────────────

  render(ctx: GameContext) {
    const g = this.ctx2d;
    const { s } = this.L;
    const dt = ctx.dt;
    const p = ctx.player;
    const st = p.state;
    const phase = ctx.race.phase;

    g.clearRect(0, 0, this.w, this.h);
    this.advance(ctx, dt);
    this.screens.update(ctx, dt);

    // ── In-race instruments ─────────────────────────────────────────────────
    if (this.hudAlpha > 0.004) {
      g.save();
      g.globalAlpha = this.hudAlpha;
      // Instruments slide in from the edges as they fade in — a static fade
      // reads as an opacity tween, a slide reads as hardware powering up.
      //
      // The baked chrome has to move *with* its own cluster, so it is blitted in
      // three sub-rects rather than as one full-screen image. Blitting it whole
      // and offsetting only the live layer leaves the minimap frame and its pips
      // travelling in different directions for the length of the fade.
      const off = (1 - this.hudAlpha) * 34 * s;

      g.save();
      g.translate(off, 0);
      this.blitChrome(this.CHROME_MAP);
      this.minimap.drawLive(g, ctx, s, this.pulse);
      this.drawStandings(ctx, s);
      g.restore();

      g.save();
      g.translate(0, off);
      this.blitChrome(this.CHROME_GAUGE);
      this.blitChrome(this.CHROME_BOOST);
      this.drawGauge(ctx, s);
      this.drawBoost(ctx, s);
      g.restore();

      g.save();
      g.translate(-off, 0);
      this.drawInfoSlab(ctx, s);
      g.restore();

      if (st.boostTime > 0) this.drawBoostFrame(s);
      if (p.wrongWay && phase === 'racing') this.drawWrongWay(s);
      g.restore();
    }

    // ── Screens ─────────────────────────────────────────────────────────────
    if (phase === 'countdown') this.screens.countdown(g, ctx, this.w, this.h, s);
    this.screens.go(g, this.w, this.h, s);
    this.screens.lapFlash(g, ctx, this.w, this.h, s);
    if (phase === 'finished') this.screens.finishFlash(g, ctx, this.w, this.h, s);
    if (phase === 'results') this.screens.results(g, ctx, this.w, this.h, s);

    if (CONFIG.debug.enabled) this.drawDebug(ctx, s);
  }

  /** All time-varying state advanced in one place, frame-rate independent. */
  private advance(ctx: GameContext, dt: number) {
    const st = ctx.player.state;
    const phase = ctx.race.phase;
    this.pulse += dt;

    const kmh = Math.abs(st.forwardSpeed) * 3.6;
    this.needleKmh = damp(this.needleKmh, kmh, 9, dt);
    this.shownKmh = damp(this.shownKmh, kmh, 16, dt);

    const target = phase === 'racing' || phase === 'finished' ? 1 : 0;
    this.hudAlpha = damp(this.hudAlpha, target, 5.5, dt);
    if (Math.abs(this.hudAlpha - target) < 0.01) this.hudAlpha = target;

    // Position change.
    const place = ctx.player.place;
    if (this.lastPlace === 0) this.lastPlace = place;
    else if (place !== this.lastPlace) {
      this.placeDir = place < this.lastPlace ? 1 : -1;
      this.placeFlash = 1;
      this.lastPlace = place;
    }
    this.placeFlash = Math.max(0, this.placeFlash - dt * 0.85);

    // Drift tier ups.
    if (st.driftTier > this.lastTier) this.tierFlash[st.driftTier - 1] = 1;
    this.lastTier = st.driftTier;
    for (let i = 0; i < 3; i++) this.tierFlash[i] = Math.max(0, this.tierFlash[i] - dt * 2.6);

    // Boost fired: rising edge on boostTime.
    if (st.boostTime > this.lastBoostTime + 0.01) this.boostPulse = 1;
    this.lastBoostTime = st.boostTime;
    this.boostPulse = Math.max(0, this.boostPulse - dt * 2.2);

    if (ctx.race.raceTime < -CONFIG.race.countdownSeconds + 0.05) this.screens.resetLatches();
  }

  // ── Speedometer ────────────────────────────────────────────────────────────

  private drawGauge(ctx: GameContext, s: number) {
    const g = this.ctx2d;
    const { gx, gy, gr } = this.L;
    const st = ctx.player.state;
    const boosting = st.boostTime > 0;
    const frac = clamp01(this.needleKmh / GAUGE_MAX_KMH);

    g.save();
    g.translate(gx, gy);

    // Power band — the filled arc from zero to the needle. Same `frac`, same
    // angular range and same radius as the needle and the redline, so the two
    // indicators cannot tell different stories. Butt caps: a rounded cap is a
    // photographic treatment and it also overshoots the value it is reporting.
    const a = DIAL_A0 + DIAL_SWEEP * frac;
    if (frac > 0.002) {
      g.lineCap = 'butt';
      g.strokeStyle = rgba(HEX.ink, 1);
      g.lineWidth = (BAND_W + 4) * s;
      g.beginPath();
      g.arc(0, 0, gr - BAND_R * s, DIAL_A0, a);
      g.stroke();
      g.strokeStyle = boosting ? rgba(HEX.boostHot, 1) : rgba(HEX.waterCrest, 1);
      g.lineWidth = BAND_W * s;
      g.beginPath();
      g.arc(0, 0, gr - BAND_R * s, DIAL_A0, a);
      g.stroke();
    }

    // Needle: tapered blade plus a stub counterweight. Long enough to cross the
    // band and reach the tick roots, so needle, band and scale meet at one point.
    g.save();
    g.rotate(a);
    const len = gr - 6 * s;
    // Slim blade. The needle is drawn over the power band, so every pixel of
    // blade width hides a slice of the band's end and reads as the two indicators
    // disagreeing — at the old width the visible cyan stopped ~7 degrees short of
    // the needle even though both are driven by the same `frac`.
    const nd = new Path2D();
    nd.moveTo(len, 0);
    nd.lineTo(len - 14 * s, -3.4 * s);
    nd.lineTo(6 * s, -6.4 * s);
    nd.lineTo(6 * s, 6.4 * s);
    nd.lineTo(len - 14 * s, 3.4 * s);
    nd.closePath();
    inked(g, nd, rgba(HEX.hull0, 1), rgba(HEX.ink, 1), 2 * s);
    const tail = new Path2D();
    tail.moveTo(-6 * s, -6 * s);
    tail.lineTo(-26 * s, -3.4 * s);
    tail.lineTo(-26 * s, 3.4 * s);
    tail.lineTo(-6 * s, 6 * s);
    tail.closePath();
    inked(g, tail, rgba(HEX.hudPaper, 0.9), rgba(HEX.ink, 1), 2 * s);
    g.restore();

    // Hub.
    const hub = new Path2D();
    hub.arc(0, 0, 11 * s, 0, Math.PI * 2);
    inked(g, hub, rgba(HEX.hudInk, 1), rgba(HEX.hudPaper, 0.9), PLATE_W * s);
    const dot = new Path2D();
    dot.arc(0, 0, 3.4 * s, 0, Math.PI * 2);
    inked(g, dot, rgba(HEX.hull0, 1), null, 0);

    g.restore();

    // Digital readout: lit digits only, right-aligned, no unlit field behind.
    const val = Math.round(clamp(this.shownKmh, 0, 999));
    const { rx, ry, rw } = this.L;
    segText(g, String(val), rx + rw - 16 * s, ry + 33 * s, 44 * s, {
      lit: boosting ? rgba(HEX.boostHot, 1) : rgba(HEX.hudPaper, 1),
      ink: rgba(HEX.ink, 1),
      inkWidth: 2.6 * s,
      align: 'right',
      skew: 0.1,
    });

    // Airborne / reverse annunciator in the dial's open mouth.
    let tag: string | null = null;
    let tagCol: number = HEX.waterCrest;
    if (st.airborne) {
      tag = 'AIR';
      tagCol = HEX.waterCrest;
    } else if (st.forwardSpeed < -0.6) {
      tag = 'REV';
      tagCol = HEX.warn;
    }
    if (tag) {
      const tw = 62 * s;
      const tp = cutPath(gx - tw * 0.5, gy + 30 * s, tw, 23 * s, 7 * s, 0b0101);
      plate(g, tp, s, { edge: rgba(tagCol, 0.95) });
      inkText(g, tag, gx, gy + 47 * s, {
        font: `800 ${Math.round(13 * s)}px ${FONT_STACK}`,
        fill: rgba(tagCol, 1),
        align: 'center',
        tracking: 2.4 * s,
      });
    }
  }

  // ── Boost / drift meter ────────────────────────────────────────────────────

  private drawBoost(ctx: GameContext, s: number) {
    const g = this.ctx2d;
    const { bx, by, bw, bh } = this.L;
    const st = ctx.player.state;
    const boosting = st.boostTime > 0;
    // Tier thresholds normalised the same way boatPhysics normalises boostMeter.
    const tiers = CONFIG.boat.driftTiers;
    const full = tiers[tiers.length - 1];
    const edges = [0, tiers[0] / full, tiers[1] / full, 1];
    const meter = clamp01(st.boostMeter);

    for (let i = 0; i < 3; i++) {
      const x = bx + i * (bw + 7 * s);
      const cell = chevronPath(x, by, bw, bh, 11 * s);
      // Fill fraction of this cell.
      let f = boosting ? 1 : clamp01((meter - edges[i]) / (edges[i + 1] - edges[i]));
      if (boosting) f = clamp01(st.boostTime / 0.35 - (2 - i) * 0.25);
      if (f <= 0.001 && this.tierFlash[i] <= 0) continue;

      g.save();
      g.clip(cell);
      // Cell colour ramps crest-cyan → hot yellow across the three tiers, so the
      // meter reads as one element heating up rather than three separate lamps.
      // It used to start at `boost` magenta, a colour that appears nowhere else
      // in the racing frame; magenta is now reserved for the moment a boost
      // actually fires (the BOOST callout and the frame rails).
      // Two committed tones, not a gradient of mixes: crest cyan while the meter
      // is charging, hot yellow on the last cell, which is the "ready" read.
      g.fillStyle = rgba(boosting || i === 2 ? HEX.boostHot : HEX.waterCrest, 1);
      g.fillRect(x - 2 * s, by - 2 * s, (bw + 4 * s) * f, bh + 4 * s);
      // Leading edge highlight so the fill has a drawn front, not a soft ramp.
      if (f < 0.999) {
        g.fillStyle = rgba(HEX.foam, 0.85);
        g.fillRect(x - 2 * s + (bw + 4 * s) * f - 3 * s, by - 2 * s, 3 * s, bh + 4 * s);
      }
      g.restore();

      // Re-ink the cell over the fill.
      inked(g, cell, null, rgba(HEX.hudPaper, 0.92), PLATE_W * s);

      // Tier-up flash: a white wash plus an expanding outline.
      const tf = this.tierFlash[i];
      if (tf > 0) {
        g.save();
        g.globalAlpha = tf;
        inked(g, cell, rgba(HEX.foam, 0.75 * tf), null, 0);
        g.restore();
        const k = 1 - tf;
        const grow = 10 * s * k;
        const ring = chevronPath(x - grow, by - grow, bw + grow * 2, bh + grow * 2, 11 * s);
        inked(g, ring, null, rgba(HEX.boostHot, tf * 0.9), 2.6 * s);
      }
    }

    // "BOOST" callout while a boost is burning, punched by the fire pulse.
    if (boosting || this.boostPulse > 0) {
      const k = Math.max(boosting ? 1 : 0, this.boostPulse);
      const cx = bx + 3 * (bw + 7 * s) + 16 * s;
      g.save();
      g.globalAlpha = k;
      inkText(g, 'BOOST', cx, by + bh - 6 * s, {
        font: `900 ${Math.round(27 * s)}px ${FONT_STACK}`,
        fill: rgba(HEX.boostHot, 1),
        ink: rgba(HEX.ink, 1),
        inkWidth: 5 * s,
        align: 'left',
        skew: 0.2,
        tracking: 1.5 * s,
        ghost: rgba(HEX.boost, 0.9),
        ghostDx: 4 * s,
        ghostDy: 4 * s,
      });
      g.restore();
    }

    // Slip indicator — a graphic, drawn *inside* the drift plate's caption row
    // rather than as a fifth floating widget. It replaces a "DRIFT 8 deg"
    // numeric telemetry read-out: degrees of slip is a dev overlay, and it
    // arrived on its own magenta pill with its own border weight, which is a
    // good part of what made the HUD read as four unrelated design systems.
    //
    // Reads as: a centre notch, and a wedge that slides toward the side the
    // stern is sliding to, growing and heating up as the slide deepens.
    const slip = clamp(st.lateralSpeed / 7, -1, 1);
    if (st.drifting || Math.abs(slip) > 0.1) {
      const trackW = 104 * s;
      const cx = bx + 3 * (bw + 7 * s) - 4 * s - trackW;
      const cy = by - 17 * s;
      const mag = Math.abs(slip);
      g.strokeStyle = rgba(HEX.hudDim, 0.8);
      g.lineWidth = 2 * s;
      g.beginPath();
      g.moveTo(cx, cy);
      g.lineTo(cx + trackW, cy);
      g.stroke();
      g.strokeStyle = rgba(HEX.hudPaper, 0.9);
      g.lineWidth = 2.2 * s;
      g.beginPath();
      g.moveTo(cx + trackW * 0.5, cy - 6 * s);
      g.lineTo(cx + trackW * 0.5, cy + 6 * s);
      g.stroke();
      const dir = slip >= 0 ? 1 : -1;
      const px = cx + trackW * (0.5 + slip * 0.44);
      const wl = (8 + mag * 10) * s;
      const wedge = new Path2D();
      wedge.moveTo(px + dir * wl, cy);
      wedge.lineTo(px - dir * wl * 0.4, cy - (5 + mag * 4) * s);
      wedge.lineTo(px - dir * wl * 0.4, cy + (5 + mag * 4) * s);
      wedge.closePath();
      inked(
        g,
        wedge,
        st.drifting ? rgba(mag > 0.6 ? HEX.boostHot : HEX.waterCrest, 1) : rgba(HEX.hudDim, 1),
        rgba(HEX.ink, 1),
        2 * s,
      );
    }
  }

  /**
   * Frame-edge boost cue: two rails plus hard manga speed lines up the sides.
   *
   * The speed lines replace a pair of 130 × 405 px triangles filled at 0.22 alpha.
   * A large low-alpha wedge over sky is optically a haze gradient no matter how
   * hard its own edges are — in shots/pres2_r3/foam_wake.png the two of them read
   * as pale washes across the top corners, which is the one photographic treatment
   * this game is not allowed. Twelve opaque tapered bars cover a fraction of the
   * area, say the same thing louder, and are unambiguously drawn.
   */
  private drawBoostFrame(s: number) {
    const g = this.ctx2d;
    // Floor the pulse well above zero: at the old 0.55 ± 0.45 the trough alpha was
    // 0.06 and the whole cue vanished for half of every 0.29 s beat, so whether a
    // boost frame showed anything at all was a coin toss per screenshot.
    const k = 0.78 + 0.22 * Math.sin(this.pulse * 22);
    g.save();
    g.fillStyle = rgba(HEX.boost, 0.9 * k);
    g.fillRect(0, 0, this.w, 3.5 * s);
    g.fillRect(0, this.h - 3.5 * s, this.w, 3.5 * s);
    for (const side of [0, 1]) {
      const X = side * this.w;
      const d = side ? -1 : 1;
      for (let i = 0; i < 6; i++) {
        // Deterministic pseudo-random lengths and thicknesses — no allocation, and
        // the same every frame at a given pulse so nothing crawls.
        const n = ((i * 7 + side * 3) % 5) / 4;
        const y = this.h * (0.1 + i * 0.14) + side * this.h * 0.06;
        const len = (70 * s + n * 150 * s);
        const th = (5 + ((i + side) % 3) * 3.2) * s;
        const bar = new Path2D();
        bar.moveTo(X, y);
        bar.lineTo(X + d * len, y + th * 0.45);
        bar.lineTo(X + d * len, y + th * 0.9);
        bar.lineTo(X, y + th * 1.5);
        bar.closePath();
        g.fillStyle = rgba(i % 2 === 0 ? HEX.boostHot : HEX.boost, (0.72 + 0.28 * n) * k);
        g.fill(bar);
      }
    }
    g.restore();
  }

  // ── Top-left: position, lap, clock, splits ─────────────────────────────────

  private drawInfoSlab(ctx: GameContext, s: number) {
    const g = this.ctx2d;
    const { ix, iy, iw, ih1, ih } = this.L;
    const p = ctx.player;

    // ONE plate for the whole cluster.
    //
    // The split times used to live on a second plate hung *below* this one,
    // outside its frame, with its own thinner outline and its own colour — the
    // clearest single symptom of the HUD being four design systems. LAST/BEST are
    // part of "how is my race going", so they are inside the same frame, divided
    // by a rule instead of by a border.
    const slab = slantPath(ix, iy, iw, ih, ih * PLATE_SKEW);
    plate(g, slab, s, { hatchRect: [ix, iy, iw, ih] });

    const lead = p.place === 1;
    // Divider between the placement and the lap block. Leaning with the plate.
    const dx = ix + 124 * s;
    g.strokeStyle = rgba(HEX.hudPaper, 0.4);
    g.lineWidth = 2 * s;
    g.beginPath();
    g.moveTo(dx + 20 * s, iy + 8 * s);
    g.lineTo(dx - 2 * s, iy + ih1 - 6 * s);
    g.stroke();
    // Rule above the split row, following the same lean.
    const y2 = iy + ih1;
    g.beginPath();
    g.moveTo(ix + (ih - ih1) * PLATE_SKEW + 8 * s, y2);
    g.lineTo(ix + iw - 8 * s, y2);
    g.stroke();

    // ── Placement. The suffix is set small and raised, like a race programme.
    const flash = this.placeFlash;
    const pop = 1 + flash * flash * 0.16;
    const num = String(p.place);
    const suf = ordinal(p.place).slice(String(p.place).length).toUpperCase();
    g.save();
    g.translate(ix + 66 * s, iy + ih1 * 0.5);
    g.scale(pop, pop);
    const flashCol =
      flash > 0
        ? this.placeDir > 0
          ? rgba(HEX.raceLine, 1)
          : rgba(HEX.warn, 1)
        : lead
          ? rgba(HEX.boostHot, 1)
          : rgba(HEX.hudPaper, 1);
    // No offset colour ghost here. Magenta is the boost colour; on the placement
    // numeral it was a fifth accent in a frame that already had four.
    inkText(g, num, -6 * s, 25 * s, {
      font: `900 ${Math.round(70 * s)}px ${FONT_STACK}`,
      fill: flashCol,
      ink: rgba(HEX.ink, 1),
      inkWidth: 6.5 * s,
      align: 'center',
      skew: 0.17,
    });
    inkText(g, suf, 24 * s, -8 * s, {
      font: `800 ${Math.round(22 * s)}px ${FONT_STACK}`,
      fill: rgba(HEX.hudDim, 1),
      align: 'left',
      skew: 0.17,
    });
    g.restore();

    // Position-change arrow, riding out of the badge.
    if (flash > 0 && this.placeDir !== 0) {
      const up = this.placeDir > 0;
      const rise = (1 - flash) * 26 * s;
      g.save();
      g.globalAlpha = flash;
      g.translate(ix + 108 * s, iy + ih1 * 0.5 - (up ? rise : -rise));
      const arr = new Path2D();
      const d = up ? -1 : 1;
      arr.moveTo(0, 11 * s * d);
      arr.lineTo(9 * s, -3 * s * d);
      arr.lineTo(3.4 * s, -3 * s * d);
      arr.lineTo(3.4 * s, -12 * s * d);
      arr.lineTo(-3.4 * s, -12 * s * d);
      arr.lineTo(-3.4 * s, -3 * s * d);
      arr.lineTo(-9 * s, -3 * s * d);
      arr.closePath();
      inked(g, arr, up ? rgba(HEX.raceLine, 1) : rgba(HEX.warn, 1), rgba(HEX.ink, 1), 2 * s);
      g.restore();
    }

    // ── Lap counter.
    const lx = ix + 150 * s;
    inkText(g, 'LAP', lx, iy + 26 * s, {
      font: `800 ${Math.round(12 * s)}px ${FONT_STACK}`,
      fill: rgba(HEX.hudDim, 1),
      align: 'left',
      tracking: 3.4 * s,
    });
    const lapNum = String(Math.min(p.lap + 1, CONFIG.race.laps));
    segText(g, `${lapNum}/${CONFIG.race.laps}`, lx, iy + 32 * s, 30 * s, {
      lit: rgba(HEX.hudPaper, 1),
      ink: rgba(HEX.ink, 1),
      inkWidth: 1.2 * s,
      align: 'left',
      skew: 0.1,
    });

    // ── Race clock. Lit segments only — the unlit ghosts behind this readout
    // made 1s and 7s ambiguous at capture scale.
    const t = Math.max(0, ctx.race.raceTime);
    segText(g, formatTime(t), lx, iy + 66 * s, 19 * s, {
      lit: rgba(HEX.waterCrest, 1),
      align: 'left',
      skew: 0.1,
    });

    // ── Split row: last lap and best lap, inside the frame.
    const sy = y2 + 12 * s;
    const last = p.lapTimes.length ? p.lapTimes[p.lapTimes.length - 1] : NaN;
    const bestIsNew =
      p.lapTimes.length > 0 && isFinite(p.bestLap) && Math.abs(p.bestLap - last) < 1e-6;
    const cols: [string, number, string][] = [
      ['LAST', last, rgba(HEX.hudPaper, 1)],
      ['BEST', p.bestLap, bestIsNew ? rgba(HEX.raceLine, 1) : rgba(HEX.foamShade, 1)],
    ];
    cols.forEach(([label, value, col], i) => {
      const cx = ix + 20 * s + i * (iw - 44 * s) * 0.5;
      inkText(g, label, cx, sy + 22 * s, {
        font: `800 ${Math.round(11 * s)}px ${FONT_STACK}`,
        fill: rgba(HEX.hudDim, 1),
        align: 'left',
        tracking: 2.2 * s,
      });
      const has = isFinite(value) && value > 0;
      const tx = cx + 40 * s;
      if (has) {
        // Full-brightness digits. These were previously drawn at ~10 % alpha as
        // an "unlit field", i.e. unreadable in every capture.
        segText(g, formatTime(value), tx, sy + 6 * s, 17 * s, {
          lit: col,
          ink: rgba(HEX.ink, 1),
          inkWidth: 1.2 * s,
          align: 'left',
          skew: 0.1,
        });
      } else {
        // No lap yet: draw dashes at a legible tone. A ghosted 0:00.000 reads as
        // a broken display; a dash reads as "no data", which is the truth.
        g.strokeStyle = rgba(HEX.hudDim, 0.9);
        g.lineWidth = 3 * s;
        const dw = segWidth('0:00.000', 17 * s);
        for (let k = 0; k < 3; k++) {
          const x0 = tx + 4 * s + k * dw * 0.33;
          g.beginPath();
          g.moveTo(x0, sy + 14 * s);
          g.lineTo(x0 + dw * 0.2, sy + 14 * s);
          g.stroke();
        }
      }
    });
  }

  // ── Standings ladder ───────────────────────────────────────────────────────

  private drawStandings(ctx: GameContext, s: number) {
    const g = this.ctx2d;
    const board = ctx.race.standings();
    // Compact rows. See `resize()`: the whole stack has to finish above the
    // horizon band, so the ladder is four 24 px rows rather than four 30 px ones.
    const rowH = 24 * s;
    const gap = 4 * s;
    const w = this.minimap.size;
    const x = this.minimap.x;
    const y0 = this.minimap.y + this.minimap.size + 14 * s;
    const leader = board[0];

    for (let i = 0; i < board.length; i++) {
      const r = board[i];
      const y = y0 + i * (rowH + gap);
      const hex = [HEX.hull0, HEX.hull1, HEX.hull2, HEX.hull3][r.id];
      const isPlayer = r.isPlayer;
      // Rows stagger right as they go down: the ladder reads as a ranked stack.
      const rx = x + i * 6 * s;
      const plateW = w - i * 6 * s;
      const row = slantPath(rx, y, plateW, rowH, rowH * PLATE_SKEW);
      // One weight, one outline colour, opaque fill — for every row. The old
      // ladder drew non-player rows at 0.66 alpha with a 1.8 px dim outline, and
      // a pink buoy panel plus a white mast rendered straight through
      // "2 KAIRA" and "3 NOX" (shots/pres_base/pack.png). Whose row it is is
      // said by the hull-colour chip and the name's brightness, not by the frame.
      plate(g, row, s, {
        fill: rgba(isPlayer ? HEX.hudInk : HEX.inkSoft, 1),
        edge: rgba(HEX.hudPaper, isPlayer ? 0.95 : 0.7),
      });

      // Colour chip.
      const chip = slantPath(rx + 4 * s, y + 3.5 * s, 7 * s, rowH - 7 * s, (rowH - 7 * s) * PLATE_SKEW);
      inked(g, chip, rgba(hex, 1), rgba(HEX.ink, 1), 1.6 * s);

      // Place number in the type face, not the segment face: a seven-segment
      // "1" is a bare vertical bar and at ladder size it reads as a tally mark
      // rather than as a position.
      inkText(g, String(i + 1), rx + 23 * s, y + rowH - 7 * s, {
        font: `900 ${Math.round(17 * s)}px ${FONT_STACK}`,
        fill: rgba(i === 0 ? HEX.boostHot : HEX.hudPaper, 0.98),
        align: 'center',
        skew: 0.16,
      });

      inkText(g, r.name.toUpperCase(), rx + 37 * s, y + rowH - 7 * s, {
        font: `800 ${Math.round(12 * s)}px ${FONT_STACK}`,
        fill: rgba(isPlayer ? HEX.hudPaper : HEX.foamShade, 0.95),
        align: 'left',
        skew: 0.1,
        tracking: 1.1 * s,
      });

      // Gap to the leader, estimated from spline progress and current pace.
      const gapTxt = this.gapText(ctx, r, leader);
      inkText(g, gapTxt, rx + plateW - 9 * s, y + rowH - 7 * s, {
        font: `800 ${Math.round(11.5 * s)}px ${FONT_STACK}`,
        fill:
          gapTxt === 'LEAD'
            ? rgba(HEX.boostHot, 1)
            : isPlayer
              ? rgba(HEX.warn, 1)
              : rgba(HEX.hudDim, 1),
        align: 'right',
        skew: 0.1,
      });
    }
  }

  /**
   * Split to the leader in seconds. Distance-behind divided by pace is a rough
   * model, but it is the same one broadcast timing uses and it is stable — a
   * per-gate delta flickers every time two boats straddle a checkpoint.
   */
  private gapText(ctx: GameContext, r: Racer, leader: Racer) {
    if (r === leader) return 'LEAD';
    const dLaps = leader.progress - r.progress;
    const metres = dLaps * ctx.track.length;
    const pace = Math.max(9, Math.abs(r.state.forwardSpeed));
    const secs = metres / pace;
    return '+' + (secs < 100 ? secs.toFixed(1) : Math.round(secs).toString());
  }

  // ── Warnings ───────────────────────────────────────────────────────────────

  private drawWrongWay(s: number) {
    const g = this.ctx2d;
    const blink = 0.45 + 0.55 * Math.abs(Math.sin(this.pulse * 6));
    const cx = this.w * 0.5;
    const cy = this.h * 0.155;
    g.save();
    g.globalAlpha = blink;
    const slab = slantPath(cx - 190 * s, cy - 28 * s, 380 * s, 56 * s, 20 * s);
    inked(g, slab, rgba(HEX.ink, 0.82), rgba(HEX.warn, 0.95), 3 * s);
    // Hazard ticks along the bottom edge.
    g.save();
    g.clip(slab);
    g.strokeStyle = rgba(HEX.warn, 0.5);
    g.lineWidth = 6 * s;
    for (let i = -2; i < 30; i++) {
      const bxx = cx - 190 * s + i * 14 * s;
      g.beginPath();
      g.moveTo(bxx, cy + 28 * s);
      g.lineTo(bxx + 10 * s, cy + 18 * s);
      g.stroke();
    }
    g.restore();
    inkText(g, 'WRONG WAY', cx, cy + 12 * s, {
      font: `900 ${Math.round(34 * s)}px ${FONT_STACK}`,
      fill: rgba(HEX.warn, 1),
      ink: rgba(HEX.ink, 1),
      inkWidth: 6 * s,
      align: 'center',
      skew: 0.16,
      tracking: 4 * s,
    });
    // Reverse chevrons either side.
    for (const dir of [-1, 1]) {
      g.strokeStyle = rgba(HEX.warn, 0.9);
      g.lineWidth = 4 * s;
      for (let i = 0; i < 2; i++) {
        const bxx = cx + dir * (150 + i * 16) * s;
        g.beginPath();
        g.moveTo(bxx + dir * 9 * s, cy - 12 * s);
        g.lineTo(bxx - dir * 6 * s, cy);
        g.lineTo(bxx + dir * 9 * s, cy + 12 * s);
        g.stroke();
      }
    }
    g.restore();
  }

  private drawDebug(ctx: GameContext, s: number) {
    const g = this.ctx2d;
    g.save();
    g.font = `600 ${Math.round(12 * s)}px ui-monospace, monospace`;
    g.textAlign = 'left';
    g.fillStyle = rgba(HEX.raceLine, 0.9);
    const lines = [
      `fps ${ctx.perf.fps.toFixed(0)}  ${ctx.perf.frameMs.toFixed(1)}ms`,
      `dpr ${ctx.pixelRatio.toFixed(2)}  scale ${ctx.perf.gpuScale.toFixed(2)}`,
      `draws ${ctx.perf.drawCalls}  tris ${(ctx.perf.triangles / 1000).toFixed(0)}k`,
    ];
    lines.forEach((l, i) => g.fillText(l, this.w * 0.5 - 90 * s, this.h - 54 * s + i * 15 * s));
    g.restore();
  }
}
