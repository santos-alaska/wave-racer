/**
 * Renderer setup + adaptive resolution.
 *
 * The pixel-ratio controller is the whole reason this file exists. Retina on a
 * MacBook means we are potentially shading 4× the pixels of a 1× buffer, and
 * the ocean is a full-screen, high-instruction-count surface. Rather than pick
 * a fixed compromise resolution, we measure the frame cost and scale the
 * backbuffer between `minPixelRatio` and `maxPixelRatio` to hold the budget.
 *
 * Scaling is deliberately asymmetric and hysteretic: drop resolution quickly
 * when we blow the budget (the player feels a stutter immediately), climb back
 * slowly and only from a stable median (so we don't oscillate visibly, which
 * is more distracting than simply running a little soft).
 */

import { PCFSoftShadowMap, SRGBColorSpace, WebGLRenderer } from 'three';
import { CONFIG } from '../core/config';

export interface RendererBundle {
  renderer: WebGLRenderer;
  canvas: HTMLCanvasElement;
}

export function createRenderer(canvas: HTMLCanvasElement): RendererBundle {
  const renderer = new WebGLRenderer({
    canvas,
    antialias: false, // we get our edges from the ink passes; MSAA would fight them
    alpha: false,
    stencil: false,
    depth: true,
    powerPreference: 'high-performance',
    // A cel image has large flat areas; banding in an 8-bit buffer is visible
    // in the sky gradient, so we ask for high precision where available.
    precision: 'highp',
  });

  renderer.outputColorSpace = SRGBColorSpace;
  renderer.setClearColor(0x061024, 1);
  renderer.shadowMap.enabled = false; // stylised contact shadows instead
  renderer.shadowMap.type = PCFSoftShadowMap;
  renderer.autoClear = false; // the composer owns clearing
  renderer.info.autoReset = false;

  return { renderer, canvas };
}

/** Rolling frame-time statistics with an adaptive backbuffer scale. */
export class AdaptiveResolution {
  /** Current multiplier applied on top of the device pixel ratio, 0…1. */
  scale = 1;
  fps = 60;
  frameMs = 16.6;

  private samples: number[] = [];
  private cooldown = 0;
  private climbTimer = 0;
  private readonly maxDpr: number;
  private readonly minScale: number;

  constructor(private renderer: WebGLRenderer) {
    this.maxDpr = Math.min(window.devicePixelRatio || 1, CONFIG.render.maxPixelRatio);
    this.minScale = CONFIG.render.minPixelRatio / this.maxDpr;
  }

  /** Effective device pixel ratio for the current scale. */
  get pixelRatio() {
    return this.maxDpr * this.scale;
  }

  update(frameMs: number, dt: number) {
    // Ignore absurd deltas — tab switches, breakpoints, the first frame.
    if (frameMs > 0 && frameMs < 500) this.samples.push(frameMs);
    if (this.samples.length > 45) this.samples.shift();
    if (this.samples.length < 12) return false;

    // Median, not mean: a single GC spike shouldn't drop our resolution.
    const sorted = [...this.samples].sort((a, b) => a - b);
    const median = sorted[sorted.length >> 1];
    this.frameMs = median;
    this.fps = 1000 / Math.max(median, 0.001);

    this.cooldown -= dt;
    if (this.cooldown > 0) return false;

    const budget = CONFIG.render.frameBudgetMs;
    let changed = false;

    if (median > budget * 1.14 && this.scale > this.minScale) {
      // Over budget — back off proportionally to how far over we are, so a
      // badly overloaded frame recovers in one step rather than five.
      const overshoot = median / budget;
      this.scale = Math.max(this.minScale, this.scale / Math.sqrt(overshoot));
      this.cooldown = 0.5;
      this.climbTimer = 0;
      changed = true;
    } else if (median < budget * 0.74 && this.scale < 1) {
      // Comfortably under budget for a sustained period — creep back up.
      this.climbTimer += dt;
      if (this.climbTimer > 1.4) {
        this.scale = Math.min(1, this.scale * 1.07);
        this.cooldown = 0.7;
        this.climbTimer = 0;
        changed = true;
      }
    } else {
      this.climbTimer = 0;
    }

    return changed;
  }

  /** Reported for the debug overlay. */
  drawStats() {
    const info = this.renderer.info;
    return {
      drawCalls: info.render.calls,
      triangles: info.render.triangles,
    };
  }
}
