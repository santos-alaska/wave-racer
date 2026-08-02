/**
 * The cel-shading core. Every solid surface in the game is drawn with a
 * CelMaterial, and every one of them also produces:
 *
 *   • a **prepass** material — GLSL3, writes view normal + linear depth +
 *     object id into an MRT G-buffer for the screen-space edge pass
 *   • an **outline** material — inverted hull, back faces, pushed along
 *     smoothed normals by a *constant number of screen pixels*
 *
 * All three are generated from the same vertex chunks, which is the only way
 * to keep them in register when a surface displaces in the vertex shader (the
 * ocean, the wake ribbons, and every rider limb do exactly that). A hand-
 * written outline shader that forgets the displacement produces the classic
 * "outline floating off the model" bug.
 *
 * ── Why not MeshToonMaterial ───────────────────────────────────────────────
 * MeshToonMaterial gives you a gradient-map diffuse and nothing else: no
 * banded specular, no fresnel term you can shape, no hook for the G-buffer,
 * and its lighting goes through three's physical pipeline. We want full
 * control of the terminator, so we own the whole shader.
 */

import {
  AdditiveBlending,
  BackSide,
  BufferAttribute,
  BufferGeometry,
  Color,
  DoubleSide,
  FrontSide,
  GLSL3,
  type IUniform,
  Material,
  Mesh,
  type Object3D,
  RawShaderMaterial,
  ShaderMaterial,
  type Texture,
  Vector2,
  Vector3,
  Vector4,
} from 'three';
import { PAL, SUN_DIR } from '../core/palette';
import { makeGlossMatcap, makeMatcapTexture, makeRampTexture } from './textures';

// ─────────────────────────────────────────────────────────────────────────────
// Palette gamma guard
//
// `src/core/palette.ts` builds every tone as `new Color(hex).convertSRGBToLinear()`.
// three r152+ has ColorManagement on by default, so `new Color(hex)` has *already*
// decoded sRGB into the linear working space — the explicit convert applies a
// second decode and every mid tone collapses. Measured: PAL.hull0 comes out
// (1, 0.0070, 0.0036) where the intended vermilion is (1, 0.0782, 0.0467), an
// 11× error on the green channel. That is why the first capture of this scene had
// pure-primary hulls with pitch-black shadow sides and riders that read as
// silhouettes.
//
// palette.ts is foundation-owned so the fix has been reported, not applied.
// Meanwhile every tone this file *authors* is corrected back through `paletteTone`,
// which probes a mid-value palette entry at module load and collapses to the
// identity the moment palette.ts is fixed. Nothing here reinterprets a colour a
// caller passed in as a ramp *multiplier* — those are ratios, not tones.
// ─────────────────────────────────────────────────────────────────────────────

/** True while palette.ts is applying a second sRGB→linear decode. */
export const PALETTE_DOUBLE_DECODED = PAL.skyMid.g < 0.15;

/** Undo palette.ts's extra decode, if it is present. Identity once it is fixed. */
export function paletteTone(c: Color): Color {
  const out = c.clone();
  if (PALETTE_DOUBLE_DECODED) out.convertLinearToSRGB();
  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared uniforms — one object per name, referenced by every material, so the
// main loop updates them once and the whole scene sees it.
// ─────────────────────────────────────────────────────────────────────────────

export const SHARED = {
  uTime: { value: 0 } as IUniform<number>,
  uSunDir: { value: new Vector3(SUN_DIR.x, SUN_DIR.y, SUN_DIR.z).normalize() } as IUniform<Vector3>,
  uCameraPos: { value: new Vector3() } as IUniform<Vector3>,
  /** (width, height) in device pixels — outline width is measured against this. */
  uResolution: { value: new Vector2(1, 1) } as IUniform<Vector2>,
  /** tan(fovY / 2), needed to convert pixels to world units at a given depth. */
  uTanHalfFov: { value: 0.5 } as IUniform<number>,
  uNear: { value: 0.1 } as IUniform<number>,
  uFar: { value: 4000 } as IUniform<number>,
};

/** Chunk injection points, so subsystems can extend the shader without forking it. */
export interface CelChunks {
  uniforms?: Record<string, IUniform>;
  defines?: Record<string, string | number | boolean>;
  /** Declarations available to the vertex stage. */
  vertexHead?: string;
  /**
   * Runs after `transformed` (vec3, object space) and `objectNormal` are set.
   * Mutate them to displace. Shared verbatim by main / prepass / outline.
   */
  vertexBody?: string;
  fragmentHead?: string;
  /**
   * Runs with `baseColor` (vec3), `ndl` (float), `celShade` (vec3) in scope,
   * just before the final composite. Mutate `baseColor` / `celShade`.
   */
  fragmentBody?: string;
}

export interface CelMaterialOptions {
  color?: Color;
  /** Ramp colours from shadow → light. 3 or 4 entries. */
  rampColors?: Color[];
  /** Where each band starts, in NdotL space. Tuned by eye; see textures.ts. */
  rampStops?: number[];
  rimColor?: Color;
  /** Higher = tighter rim. 2–5 is the useful range. */
  rimPower?: number;
  rimStrength?: number;
  /** Banded specular. `specSize` is a hard threshold on the blinn term. */
  specColor?: Color;
  specSize?: number;
  specStrength?: number;
  /** Second, smaller spec band — gives the highlight a stepped shoulder. */
  specSize2?: number;
  matcap?: Texture | null;
  matcapStrength?: number;
  /**
   * Ambient bounce weight per band, 0…1, shadow → light. Added (not
   * multiplied) so the shadow bands shift *hue* toward the ambient instead of
   * just going darker. Derived from `rampColors` when omitted.
   */
  rampAmbient?: number[];
  /** The bounce tone added in the shadow bands. Defaults to sea-and-sky. */
  ambientColor?: Color;
  /** Master scale on the ambient bounce. */
  ambientStrength?: number;
  /**
   * Inverted-hull outline. Width is in *device pixels* and is constant at every
   * distance — and clamped to `outlineMaxPx`, a pipeline-wide ceiling, so no
   * material can put a heavier line in the frame than any other.
   */
  outline?: boolean;
  outlineWidthPx?: number;
  outlineColor?: Color;
  /** Ceiling on ink weight, device px. Shared by the whole image; leave it. */
  outlineMaxPx?: number;
  /**
   * Projected radius (device px) under which the ink line fades out entirely.
   * A 15 px rider cannot carry a 3 px line on each side — it becomes a
   * silhouette. The line never *thins*; it fades. Set 0 to keep it at all sizes.
   */
  outlineFadePx?: number;
  /**
   * 0 = this surface never contributes to the stylised flare. Written to the
   * G-buffer, so the threshold pass can keep hard white foam and spray out of
   * the bloom. 1 = normal.
   */
  flareMask?: number;
  /** Written to the G-buffer; a discontinuity here forces an interior line. */
  objectId?: number;
  /** Multiplier on the Sobel response for this surface. 0 = never inked. */
  edgeBias?: number;
  side?: typeof FrontSide | typeof BackSide | typeof DoubleSide;
  transparent?: boolean;
  opacity?: number;
  vertexColors?: boolean;
  /** Flat-shade the fragment normal — good for faceted, low-poly forms. */
  flatShading?: boolean;
  /**
   * Drawn glass banding, 0…1. Hard parallel streaks in *object space*, so they
   * are painted onto the model rather than sliding across it.
   *
   * A large near-white plate has nowhere for the diffuse ramp to land: NdotL is
   * constant across it, so all four bands collapse to one and the surface reads
   * as a blank card. Measured on the player's windscreen in
   * shots/cel_r2/ocean_low.png: a 120x70 device-px parallelogram of one tone.
   * A cel windscreen is drawn as two or three hard diagonal reflection bands, so
   * that is what this adds.
   *
   * Left at the default it is *auto-gated* to exactly that case — a near-white,
   * near-neutral surface whose object normal is raked (leaning up and fore/aft).
   * See GLASS_BANDS in the fragment source for why the gate is where it is, and
   * pass 0 to opt a material out.
   */
  glassBands?: number;
  chunks?: CelChunks;
  name?: string;
}

/**
 * Band thresholds in half-lambert space, tuned against real frames.
 *
 * Note where they are *not*: evenly spaced. 0.42 puts the terminator well past
 * the geometric one, so band 2 (the base tone) owns roughly half the sphere and
 * the two shadow bands are squeezed into a narrow, decisive wedge on the dark
 * side. That asymmetry is the whole difference between "3-tone cel painting"
 * and "quantised Lambert": the earlier 0.36/0.52/0.74 spacing gave four bands of
 * similar width and every curved surface read as a stepped gradient.
 *
 * Band 3 starts at 0.86 so the hot band is a narrow rim on the sun side — a
 * *shape*, not a wash.
 */
const DEFAULT_RAMP_STOPS = [0.0, 0.30, 0.42, 0.86];

/** Normalise a tone to its hue ratio, brightest channel = 1. */
function hueRatio(c: Color): Color {
  const m = Math.max(c.r, c.g, c.b) || 1;
  return c.clone().multiplyScalar(1 / m);
}

/**
 * Pipeline-wide minimum separation between adjacent ramp bands.
 *
 * The ramp is a *linear* multiplier and the image is viewed in sRGB, so a step
 * that looks reasonable as a number can be invisible as a value. Measured on
 * frames rather than argued: the yellow AI's gunwale stepped (234,213,155) L213
 * → (241,219,160) L219 and the gate crossbar (255,251,136) L241 →
 * (255,251,166) L243, i.e. a nominally 4-band ladder delivering three readable
 * tones. Both ladders had their top two entries at ×0.94 and ×1.0.
 *
 * 0.78 in linear multiplier space is ≈ 0.895 in sRGB value space, which on a
 * white material is a 27 L step — small but decisively visible. Bands closer
 * than that are pushed apart from the top down (the lit band is the anchor; it
 * is the one the eye reads the material's colour from).
 */
const RAMP_MIN_RATIO = 0.80;

/**
 * Enforce RAMP_MIN_RATIO on a shadow→light ladder, in place of the caller's
 * spacing, preserving the caller's hues. Bands are treated as values (brightest
 * channel) and only ever pushed *down*, never up, so no material gets brighter
 * than it asked for.
 */
function separateRamp(colors: Color[]): Color[] {
  if (colors.length < 2) return colors;
  const out = colors.map((c) => c.clone());
  const val = (c: Color) => Math.max(c.r, c.g, c.b);
  for (let i = out.length - 2; i >= 0; i--) {
    const above = val(out[i + 1]);
    const here = val(out[i]);
    const cap = above * RAMP_MIN_RATIO;
    if (here > cap && here > 1e-5) out[i].multiplyScalar(cap / here);
  }
  return out;
}

// The ramp is a *multiplier*, so its steps must be near-neutral with a hue
// *lean* — a full-strength blue multiplier does not read as "in shadow", it
// reads as "the red channel is switched off".
const NEUTRAL = hueRatio(paletteTone(PAL.hudPaper));
const COOL_LEAN = hueRatio(paletteTone(PAL.skyMid)).lerp(NEUTRAL, 0.6);
const WARM_LEAN = hueRatio(paletteTone(PAL.sun)).lerp(NEUTRAL, 0.72);

/**
 * The default 4-step ladder. Values, not hues, carry a cel image, so the steps
 * are chosen as a value plan first: 0.30 / 0.55 / 0.94 / 1.0. The gap between
 * band 1 and band 2 is the biggest jump in the ladder, which is what puts a
 * decisive terminator on the form; band 3 is only a whisker above band 2 so the
 * hot rim reads as a highlight rather than as a fifth tone.
 */
function defaultRamp(): Color[] {
  return [
    COOL_LEAN.clone().multiplyScalar(0.30),
    COOL_LEAN.clone().multiplyScalar(0.55),
    WARM_LEAN.clone().multiplyScalar(0.94),
    WARM_LEAN.clone().multiplyScalar(1.0),
  ];
}

/**
 * Derive per-band ambient bounce from the ladder itself: the darker the band,
 * the more sea-bounce is added into it. Squared so the mid band picks up only a
 * hint and the deep shadow picks up most of it.
 */
function ambientFromRamp(ramp: Color[], lean: number): number[] {
  const val = ramp.map((c) => Math.max(c.r, c.g, c.b));
  const top = Math.max(...val, 1e-4);
  return val.map((v) => {
    const d = Math.max(0, 1 - v / top);
    return Math.min(1, d * d * 1.9 * lean);
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Shader source
// ─────────────────────────────────────────────────────────────────────────────

const VERT_COMMON = /* glsl */ `
  attribute vec3 aSmoothNormal;

  // Always present in the uniform block; unused ones are optimised out by the
  // compiler, so declaring them unconditionally keeps the three variants
  // (main / prepass / outline) sharing one vertex source.
  uniform float uTime;
  uniform vec2 uResolution;
  uniform float uTanHalfFov;
  uniform float uNear;
  uniform float uFar;
  uniform float uOutlineWidthPx;
  uniform float uOutlineMaxPx;
  uniform float uOutlineRadius;
  uniform float uOutlineFadePx;
  uniform vec3 uCameraPos;

  varying vec3 vWorldNormal;
  varying vec3 vViewNormal;
  varying vec3 vWorldPos;
  varying vec3 vViewPos;
  varying vec2 vUv;
  varying vec3 vColor4;
  /** Outline coverage, 0 on objects too small on screen to carry an ink line. */
  varying float vOutlineFade;
  /**
   * Object-space position and the object-space glass gate, packed into one
   * varying to stay well inside the varying budget (chunk-heavy materials —
   * the rider carries skinning plus a tint — add their own on top of these).
   * .xyz is the post-displacement object position, .w is 1 on a raked plate.
   */
  varying vec4 vObjPosGlass;

  CHUNK_VERTEX_HEAD

  void main() {
    vec3 transformed = position;
    vec3 objectNormal = normal;
    vec3 smoothNormal = aSmoothNormal;
    vOutlineFade = 1.0;
    vUv = uv;
    #ifdef USE_VERTEX_COLORS
      vColor4 = color;
    #else
      vColor4 = vec3(1.0);
    #endif

    CHUNK_VERTEX_BODY

    vec4 worldPos = modelMatrix * vec4(transformed, 1.0);
    vec4 mvPosition = viewMatrix * worldPos;

    vWorldPos = worldPos.xyz;
    vViewPos = mvPosition.xyz;
    vWorldNormal = normalize(mat3(modelMatrix) * objectNormal);
    vViewNormal = normalize(normalMatrix * objectNormal);

    // The glass gate, in object space so it selects the same faces however the
    // hull is pitched or heading. See GLASS_BANDS.
    {
      vec3 no = normalize(objectNormal);
      float raked = step(0.34, no.y) * (1.0 - step(0.87, no.y)) * step(0.34, abs(no.z));
      vObjPosGlass = vec4(transformed, raked);
    }

    CHUNK_VERTEX_OUTLINE

    gl_Position = projectionMatrix * mvPosition;
  }
`;

/**
 * The outline push.
 *
 *   unitsPerPixel = 2 · depth · tan(fovY/2) / screenHeightInPixels
 *
 * Multiply by the desired pixel width and the line is the same thickness on a
 * boat 3 m away and a gate 300 m away. Scaling the push by a constant instead
 * — the common shortcut — gives fat lines up close and lines that vanish in
 * the distance, which the brief explicitly rules out.
 *
 * ── Why the push is XY-only ────────────────────────────────────────────────
 * The obvious version offsets along the full view-space normal,
 * `mvPosition.xyz += vn * w`. That moves every vertex whose smooth normal leans
 * toward the camera *toward the camera*, so the inverted hull's back shell
 * surfaces through the front faces wherever the two are within depth precision
 * of each other. On the boat's foredeck — a large, near-planar surface seen at a
 * grazing angle — it produced dense curved moiré across the whole panel
 * (shots/cel_r0/rider_closeup.png, unmistakable at 3×).
 *
 * Expanding in screen space instead is both cheaper to reason about and exactly
 * right: the outline only needs to grow *sideways* on screen. Vertices whose
 * normal points at or away from the camera have `vn.xy ≈ 0` and do not move at
 * all, so the shell stays buried behind the surface it belongs to, while
 * silhouette vertices (`|vn.xy| ≈ 1`) get the full width. The line is then
 * exactly `w` pixels wide by construction rather than approximately.
 *
 * A constant *relative* depth bias pushes the shell away from the eye so
 * coincident geometry (thin plates, the transom lip) always loses the depth
 * test rather than dithering against it.
 *
 * ── Why the width is clamped, and why it never tapers ──────────────────────
 * `uOutlineMaxPx` is a *pipeline-wide* ceiling, not a per-material preference.
 * Callers used to pick their own width (the hull asked for 5 px, the rider for
 * 2.3) and the result measured 13-17 device px of ink across the hull in
 * shots/r2/outline_check.png against 1-4 px on the same boat in outline_far.png.
 * One ink weight for the whole image is the requirement, so every material is
 * clamped to the same ceiling here and the earlier size-proportional taper is
 * gone entirely: at any distance where the line is drawn at all it is exactly
 * the same number of device pixels.
 *
 * The failure the taper was papering over is real, but thinning is the wrong
 * answer to it — a 1 px broken line reads as dirt. Instead the line *fades out*
 * as a whole (`vOutlineFade`, alpha) once the object's projected radius drops
 * under `uOutlineFadePx`, so a boat 300 m away, or a rider 15 px tall, loses its
 * ink rather than being swallowed by it.
 *
 * ── Why the magnitude has a floor ──────────────────────────────────────────
 * The push used to be weighted by `smoothstep(0.02, 0.30, |vn.xy|)`, i.e. the
 * shell only grew where the smoothed normal already lay in the screen plane. On
 * coarse geometry — this hull is a few dozen faces — the true silhouette usually
 * falls *inside* a face, and the vertices of that face can be 20-40° off tangent,
 * so they got a fraction of the width and the ring collapsed. That is exactly
 * the gap pattern measured in shots/r2/outline_check.png: strong ink along the
 * top gunwale (where faces do turn through the silhouette) and none at all along
 * the hull bottom, the bow leading edge, or the underside of the windshield
 * plate. The weight is now a floor-and-ramp: never less than 68% of the width,
 * so no edge of a closed shell can lose its line.
 *
 * The normal used is `aSmoothNormal`: an area-weighted normal merged across
 * split vertices. Using the shading normal instead tears the hull open at
 * every hard edge, which is where inverted-hull outlines usually fall apart.
 */
const OUTLINE_PUSH = /* glsl */ `
  {
    vec3 vn = normalize(normalMatrix * smoothNormal);
    float depth = max(-mvPosition.z, uNear);
    float unitsPerPixel = (2.0 * depth * uTanHalfFov) / uResolution.y;

    // One ink weight for the whole image, in device pixels.
    float w = min(uOutlineWidthPx, uOutlineMaxPx);

    vec2 dir = vn.xy;
    float len = length(dir);
    dir = len > 1.0e-4 ? dir / len : vec2(1.0, 0.0);
    // Floor-and-ramp, not a gate: see the note above. 0.68 is enough to close
    // the ring on a 30° face without letting a face pointing straight at the
    // camera drag its shell out sideways into view.
    float mag = mix(0.68, 1.0, smoothstep(0.0, 0.26, len));
    mvPosition.xy += dir * (w * mag * unitsPerPixel);
    // Away from the eye (view space looks down -Z), so a coincident shell
    // always loses the depth test instead of dithering against it. Scaled with
    // the push, because a shell displaced w pixels sideways across a grazing
    // surface is also displaced in depth and will otherwise surface through it.
    mvPosition.z -= unitsPerPixel * (2.0 + w);

    // Apparent size of this object, in pixels of radius → line alpha.
    float projPx = uOutlineRadius / max(unitsPerPixel, 1e-6);
    vOutlineFade = (uOutlineFadePx > 0.0 && uOutlineRadius > 0.0)
      ? smoothstep(uOutlineFadePx * 0.45, uOutlineFadePx, projPx)
      : 1.0;
  }
`;

const FRAG_MAIN = /* glsl */ `
  precision highp float;

  uniform vec3 uColor;
  uniform sampler2D uRamp;
  uniform float uRampScale;
  uniform vec3 uRimColor;
  uniform float uRimPower;
  uniform float uRimStrength;
  uniform vec3 uSpecColor;
  uniform float uSpecSize;
  uniform float uSpecSize2;
  uniform float uSpecStrength;
  uniform sampler2D uMatcap;
  uniform float uMatcapStrength;
  uniform vec3 uAmbientColor;
  uniform float uAmbientStrength;
  uniform float uOpacity;
  uniform vec3 uSunDir;
  uniform vec3 uCameraPos;
  uniform float uTime;
  uniform float uGlassBands;
  uniform vec3 uGlassShade;

  varying vec3 vWorldNormal;
  varying vec3 vViewNormal;
  varying vec3 vWorldPos;
  varying vec3 vViewPos;
  varying vec2 vUv;
  varying vec3 vColor4;
  varying vec4 vObjPosGlass;

  CHUNK_FRAGMENT_HEAD

  void main() {
    vec3 N = normalize(vWorldNormal);
    #ifdef FLAT_SHADING
      N = normalize(cross(dFdx(vWorldPos), dFdy(vWorldPos)));
    #endif
    vec3 V = normalize(uCameraPos - vWorldPos);
    vec3 L = normalize(uSunDir);

    vec3 baseColor = uColor * vColor4;

    // ── Quantised diffuse ───────────────────────────────────────────────────
    // Half-lambert widens the usable range so the ramp's bands land where we
    // placed them rather than crushing everything past the terminator into
    // band 0. The ramp texture itself does the stepping (NearestFilter).
    float ndl = dot(N, L) * 0.5 + 0.5;
    vec4 rampSample = texture2D(uRamp, vec2(clamp(ndl, 0.01, 0.99), 0.5));
    vec3 celShade = rampSample.rgb * uRampScale;
    float celAmbient = rampSample.a;

    CHUNK_FRAGMENT_BODY

    vec3 lit = baseColor * celShade;

    // ── Ambient bounce in the shadow bands ──────────────────────────────────
    // The ramp alone can only *scale* the surface colour, so a shadow could only
    // ever be a darker version of the base hue — a vermilion hull measured
    // (58,2,2) in the first capture, a black-red with the green and blue
    // channels effectively switched off. Adding a cool sea-and-sky bounce into
    // the dark bands is what makes the terminator a *hue* transition, and it
    // lifts the shadow enough that the ink line reads as a separate mark
    // instead of merging into it.
    // 0.62 hue retention, not 0.45: at 0.45 the ambient overwhelmed small dark
    // objects and the shadow side of the racer's red helmet came out flat
    // grey-blue, so the helmet read as two different materials
    // (shots/cel_r4/rider_closeup.png at 3×).
    lit += uAmbientColor * (celAmbient * uAmbientStrength) * mix(vec3(1.0), baseColor, 0.62);

    // ── Everything below this line is a *mask*, never a gradient ────────────
    // Each remaining term is thresholded to a hard shape and then *substituted*
    // into the surface tone with mix() rather than added to it. Adding was the
    // defect: three additive terms (spec + rim + matcap), each smoothly varying
    // in amplitude, summed into a continuously varying surface, and the ramp's
    // banding underneath was buried. Measured on the player's deck in
    // shots/r2/outline_check.png: 4666 distinct colours in a 626×62 patch with
    // no step anywhere across it — quantised Lambert plus three airbrushes.
    // Substitution keeps the flat fill flat: a pixel is either the band tone or
    // the highlight tone, and the boundary between them is one pixel wide.

    // ── Drawn glass bands ───────────────────────────────────────────────────
    // GLASS_BANDS. Three hard parallel streaks along a fixed object-space axis:
    // a wide cool "sky" band, a gap, and a narrow bright band. Object space, not
    // view space, because this is a *painted* reflection — an animator draws two
    // streaks on a windscreen and leaves them there. A view-dependent version
    // slides across the pane as the boat turns and immediately reads as a
    // simulated environment probe, which is the look the brief rules out.
    //
    // Why it is gated the way it is. The failure this fixes is specific: a large
    // near-white plate has a constant NdotL, so the four-band ramp has nowhere
    // to land and the plate is one flat fill (the windscreen in
    // shots/cel_r2/ocean_low.png). The gate therefore selects exactly that case
    // and nothing else:
    //   • .w  — object normal raked (0.34 < n.y < 0.87, |n.z| > 0.34). Selects
    //           the windscreen's pane and rejects the deck (n.y ≈ 1), the
    //           waterline stripe and the grips (n.y ≈ 0), and every axis-aligned
    //           box face on the boat.
    //   • near-white, near-neutral base — the "paper" surfaces only. The hull's
    //           vermilion, the graphite trim, the gates and the rider's tinted
    //           suit all fail it, so none of them pick up streaks.
    // Materials can force it on or off with glassBands.
    {
      float bright = step(0.80, min(min(baseColor.r, baseColor.g), baseColor.b));
      float g = dot(vObjPosGlass.xyz, normalize(vec3(0.62, 0.55, -0.56))) * 3.1;
      float f = fract(g);
      // One wide band and one narrow band per cycle: two marks read as intent.
      float wide   = step(0.06, f) * (1.0 - step(0.40, f));
      float narrow = step(0.56, f) * (1.0 - step(0.66, f));
      float gate = bright * vObjPosGlass.w * uGlassBands;
      // Substituted into the shaded tone, like every other mask in this shader,
      // so the result is still two flat fills with a one-pixel boundary.
      lit = mix(lit, lit * uGlassShade, wide * gate);
      lit = mix(lit, mix(lit, uSpecColor, 0.85), narrow * gate);
    }

    // ── Banded specular ─────────────────────────────────────────────────────
    // Two hard thresholds on the Blinn term, never a pow() falloff, gated by
    // the diffuse band: an unshadowed step() on N·H puts highlights on faces
    // the key light never reaches, the single most PBR-looking mistake here.
    vec3 H = normalize(L + V);
    float spec = dot(N, H);
    float specLight = step(0.5, ndl);
    float specMask = specLight * (step(uSpecSize, spec) * 0.42 + step(uSpecSize2, spec) * 0.58);
    lit = mix(lit, uSpecColor, clamp(specMask * uSpecStrength * 2.0, 0.0, 1.0));

    // ── Fresnel rim ─────────────────────────────────────────────────────────
    // Two steps on the fresnel term, so the rim is a drawn edge and not an
    // airbrush, and gated — not weighted — everywhere else: on the lit side of
    // the form only, and off up-facing planes, where a rim floods a whole deck
    // with a pale wash instead of drawing a line. Every factor here is a step()
    // for the same reason: a smoothly varying stroke width reads as a shaded
    // shell, not as a line an animator drew.
    float fres = 1.0 - max(dot(N, V), 0.0);
    float rim = pow(fres, uRimPower);
    float rimSide = step(-0.18, dot(N, L)) * 0.42 + step(0.28, dot(N, L)) * 0.58;
    float rimFlank = 1.0 - step(0.74, N.y) * 0.85;
    // Thresholds are low on purpose. A rim confined to the last 2% of the form
    // is completely hidden underneath the inverted-hull ink line, which sits in
    // exactly that band — the first captures had a mathematically correct rim
    // that could not be seen anywhere. These two steps put the light-line
    // *inboard* of the ink, which is where an animator draws it.
    float rimMask = (step(0.26, rim) * 0.40 + step(0.55, rim) * 0.60) * rimSide * rimFlank;
    lit = mix(lit, uRimColor, clamp(rimMask * uRimStrength, 0.0, 1.0));

    // ── Faked reflection ────────────────────────────────────────────────────
    // A drawn matcap, sampled by the view-space normal. Deliberately not a
    // cubemap: an accurate reflection is the fastest way to make a surface read
    // as physically based. The disc is a *drawing* (see makeGlossMatcap), so its
    // marks are admitted through two hard thresholds — a flat plane that samples
    // an unmarked part of the disc picks up exactly nothing, which is what stops
    // the foredeck reading as a pale bare panel.
    #ifdef USE_MATCAP
      vec3 vn = normalize(vViewNormal);
      vec2 mUv = vn.xy * 0.5 + 0.5;
      vec3 mc = texture2D(uMatcap, mUv).rgb;
      float mcl = max(max(mc.r, mc.g), mc.b);
      float mcMask = (step(0.10, mcl) * 0.45 + step(0.30, mcl) * 0.55) * step(0.56, ndl);
      lit = mix(lit, mix(uSpecColor, baseColor, 0.30), clamp(mcMask * uMatcapStrength, 0.0, 1.0));
    #endif

    gl_FragColor = vec4(lit, uOpacity);
  }
`;

/**
 * Prepass fragment. GLSL3 with two colour attachments:
 *   layout 0 → view-space normal (rgb, [0,1] encoded) + edge bias (a)
 *   layout 1 → linear view depth (r), object id (g), flare mask (b), 1 (a)
 *
 * The Sobel pass reads both. Depth alone misses edges between coplanar
 * surfaces; normals alone miss edges where two parallel surfaces overlap at
 * different depths; the object id catches the remaining case where two
 * separate objects meet at a similar depth *and* a similar normal — which is
 * exactly a rider's arm crossing their chest.
 */
const PREPASS_FRAG = /* glsl */ `
  precision highp float;

  uniform float uObjectId;
  uniform float uEdgeBias;
  uniform float uFlareMask;
  uniform float uFar;

  in vec3 vWorldNormal;
  in vec3 vViewNormal;
  in vec3 vWorldPos;
  in vec3 vViewPos;
  in vec2 vUv;
  in vec3 vColor4;

  layout(location = 0) out vec4 gNormal;
  layout(location = 1) out vec4 gDepthId;

  void main() {
    vec3 vn = normalize(vViewNormal);
    gNormal = vec4(vn * 0.5 + 0.5, uEdgeBias);
    gDepthId = vec4(clamp(-vViewPos.z / uFar, 0.0, 1.0), uObjectId, uFlareMask, 1.0);
  }
`;

// ─────────────────────────────────────────────────────────────────────────────
// Material construction
// ─────────────────────────────────────────────────────────────────────────────

/** A CelMaterial plus the two companion materials generated alongside it. */
export interface CelMaterialSet {
  main: ShaderMaterial;
  prepass: ShaderMaterial;
  outline: ShaderMaterial | null;
  /** Uniforms shared by all three; write here to affect every pass. */
  uniforms: Record<string, IUniform>;
}

function applyChunks(src: string, chunks: CelChunks | undefined, outline: boolean): string {
  return src
    .replace('CHUNK_VERTEX_HEAD', chunks?.vertexHead ?? '')
    .replace('CHUNK_VERTEX_BODY', chunks?.vertexBody ?? '')
    .replace('CHUNK_VERTEX_OUTLINE', outline ? OUTLINE_PUSH : '')
    .replace('CHUNK_FRAGMENT_HEAD', chunks?.fragmentHead ?? '')
    .replace('CHUNK_FRAGMENT_BODY', chunks?.fragmentBody ?? '');
}

/** GLSL3 needs `in`/`out` instead of `attribute`/`varying`. */
function toGLSL3Vertex(src: string): string {
  return src
    .replace(/\battribute\b/g, 'in')
    .replace(/\bvarying\b/g, 'out')
    .replace(/\btexture2D\b/g, 'texture');
}

let objectIdCounter = 1;
/** Object ids are packed into an 8-bit channel, so they wrap at 255. */
export function nextObjectId(): number {
  objectIdCounter = (objectIdCounter % 250) + 1;
  return objectIdCounter / 255;
}

/**
 * The shared default gloss disc. One texture for the whole scene: it is a
 * *stylisation*, not a probe, so there is nothing per-object about it.
 */
let defaultMatcap: Texture | null = null;
function getDefaultMatcap(): Texture {
  if (!defaultMatcap) {
    defaultMatcap = makeGlossMatcap(
      paletteTone(PAL.skyHorizon).multiplyScalar(0.34),
      paletteTone(PAL.waterMid).multiplyScalar(0.30),
      paletteTone(PAL.foam).multiplyScalar(0.60),
    );
  }
  return defaultMatcap;
}

/** Cool bounce added into the shadow bands: sea below, sky above, averaged. */
const DEFAULT_AMBIENT = paletteTone(PAL.waterMid)
  .lerp(paletteTone(PAL.skyHorizon), 0.52)
  .multiplyScalar(0.72);

export function createCelMaterial(opts: CelMaterialOptions = {}): CelMaterialSet {
  // A palette tone, so it is corrected; ramp colours are ratios and are not.
  const color = paletteTone(opts.color ?? PAL.hull0);
  const rampColors = separateRamp(opts.rampColors ?? defaultRamp());
  const rampStops = opts.rampStops ?? DEFAULT_RAMP_STOPS;
  const rampAmbient = opts.rampAmbient ?? ambientFromRamp(rampColors, 1.0);
  // The ramp lives in an 8-bit texture, so a ladder whose steps exceed 1.0 (the
  // graphite trim on the boats runs up to ×5.2) has to be normalised and the
  // scale restored in the shader. Without this the values used to *wrap* in the
  // Uint8Array — ×1.6 came back as 46/255 — and clamping them instead simply
  // flattened the top of the ladder into one tone.
  const rampScale = Math.min(
    8,
    Math.max(1, ...rampColors.map((c) => Math.max(c.r, c.g, c.b))),
  );
  const rampNorm =
    rampScale > 1 ? rampColors.map((c) => c.clone().multiplyScalar(1 / rampScale)) : rampColors;
  const ramp = makeRampTexture(rampNorm, rampStops, 64, rampAmbient);

  const uniforms: Record<string, IUniform> = {
    uColor: { value: color },
    uRamp: { value: ramp },
    uRampScale: { value: rampScale },
    uRimColor: { value: paletteTone(opts.rimColor ?? PAL.skyHorizon) },
    uRimPower: { value: opts.rimPower ?? 3.0 },
    uRimStrength: { value: opts.rimStrength ?? 0.6 },
    uSpecColor: { value: paletteTone(opts.specColor ?? PAL.foam) },
    uSpecSize: { value: opts.specSize ?? 0.94 },
    uSpecSize2: { value: opts.specSize2 ?? 0.985 },
    uSpecStrength: { value: opts.specStrength ?? 0.3 },
    uMatcap: { value: opts.matcap === null ? null : (opts.matcap ?? getDefaultMatcap()) },
    // The disc is now marks-on-black and its contribution is thresholded, so
    // this is the opacity of a *mark* rather than the weight of a wash. 0.5 was
    // painting the whole foredeck pale blue off the disc's sky strip.
    uMatcapStrength: { value: opts.matcapStrength ?? 0.34 },
    uAmbientColor: { value: paletteTone(opts.ambientColor ?? PAL.waterMid).lerp(paletteTone(PAL.skyHorizon), 0.52) },
    uAmbientStrength: { value: opts.ambientStrength ?? 0.62 },
    uOpacity: { value: opts.opacity ?? 1.0 },
    uGlassBands: { value: opts.glassBands ?? 1.0 },
    // The cool tone the wide reflection band multiplies toward. 0.58 is a ~35 L
    // step down from paper white, which is the separation the rest of the image's
    // ladders now carry (see RAMP_MIN_RATIO).
    uGlassShade: { value: hueRatio(paletteTone(PAL.skyMid)).lerp(NEUTRAL, 0.45).multiplyScalar(0.58) },
    uObjectId: { value: opts.objectId ?? nextObjectId() },
    uEdgeBias: { value: opts.edgeBias ?? 1.0 },
    uFlareMask: { value: opts.flareMask ?? 1.0 },
    uOutlineWidthPx: { value: opts.outlineWidthPx ?? 2.6 },
    // 2.8 device px = 1.4 CSS px at retina. Measured against frames rather than
    // chosen: at 3.6 the line closes up the gap between the gunwale and the
    // rub-rail on the player's hull at close range, at 2.2 it disappears into
    // the water's own ink at mid distance.
    uOutlineMaxPx: { value: opts.outlineMaxPx ?? 2.8 },
    uOutlineColor: { value: paletteTone(opts.outlineColor ?? PAL.ink) },
    // Filled in per mesh by applyCel from the geometry's bounding sphere.
    // 0 means "unknown", which disables the fade rather than guessing.
    uOutlineRadius: { value: 0.0 },
    // Projected radius, device px. Below ~18 px of radius (36 px across) a
    // 2.8 px line on each side is a fifth of the form, so it goes.
    uOutlineFadePx: { value: opts.outlineFadePx ?? 19 },
    // Shared references — assigning the same IUniform object keeps every
    // material in the scene in sync from a single write per frame.
    uTime: SHARED.uTime,
    uSunDir: SHARED.uSunDir,
    uCameraPos: SHARED.uCameraPos,
    uResolution: SHARED.uResolution,
    uTanHalfFov: SHARED.uTanHalfFov,
    uNear: SHARED.uNear,
    uFar: SHARED.uFar,
    ...(opts.chunks?.uniforms ?? {}),
  };

  const defines: Record<string, string | number | boolean> = { ...(opts.chunks?.defines ?? {}) };
  if (uniforms.uMatcap.value) defines.USE_MATCAP = '';
  if (opts.vertexColors) defines.USE_VERTEX_COLORS = '';
  if (opts.flatShading) defines.FLAT_SHADING = '';

  // `vertexColors` has to be set on *all three* materials, not just `main`:
  // three only emits `attribute vec3 color;` when the flag is on, so the prepass
  // and outline shaders would reference an undeclared attribute and fail to
  // compile. This is the trap two subsystems reported hitting.
  const vertexColors = !!opts.vertexColors;

  const main = new ShaderMaterial({
    name: opts.name ?? 'cel',
    uniforms,
    defines,
    vertexShader: applyChunks(VERT_COMMON, opts.chunks, false),
    fragmentShader: applyChunks(FRAG_MAIN, opts.chunks, false),
    side: opts.side ?? FrontSide,
    transparent: opts.transparent ?? false,
    opacity: opts.opacity ?? 1,
    vertexColors,
  });

  const prepass = new ShaderMaterial({
    name: (opts.name ?? 'cel') + ':prepass',
    uniforms,
    defines,
    glslVersion: GLSL3,
    vertexShader: toGLSL3Vertex(applyChunks(VERT_COMMON, opts.chunks, false)),
    fragmentShader: PREPASS_FRAG,
    side: opts.side ?? FrontSide,
    vertexColors,
  });

  let outline: ShaderMaterial | null = null;
  if (opts.outline !== false) {
    outline = new ShaderMaterial({
      name: (opts.name ?? 'cel') + ':outline',
      uniforms,
      defines,
      vertexColors,
      vertexShader: applyChunks(VERT_COMMON, opts.chunks, true),
      fragmentShader: /* glsl */ `
        precision highp float;
        uniform vec3 uOutlineColor;
        uniform vec3 uSunDir;
        uniform vec3 uAmbientColor;
        varying vec3 vWorldNormal;
        varying vec3 vWorldPos;
        varying vec3 vViewPos;
        varying float vOutlineFade;
        void main() {
          // Ink is not flat black and it is not one value all the way round.
          // A brush line thins and lightens where light rakes across the form
          // and thickens into the shadow, so the ink is lifted toward the
          // ambient on the sun side. Two hard steps, not a gradient — this is
          // still a drawn line, and a smoothly varying stroke reads as a
          // rendered shell rather than as ink.
          float lightSide = dot(normalize(vWorldNormal), normalize(uSunDir)) * 0.5 + 0.5;
          float lift = step(0.56, lightSide) * 0.55 + step(0.80, lightSide) * 0.45;
          vec3 ink = uOutlineColor + uAmbientColor * (lift * 0.022);
          // Alpha, not width, is how a line leaves at distance. See OUTLINE_PUSH.
          if (vOutlineFade < 0.004) discard;
          gl_FragColor = vec4(ink, vOutlineFade);
        }
      `,
      side: BackSide,
      // Transparent so the fade is a fade and not a dashed, sub-pixel line.
      // The shell still depth-tests against the scene, so it is only ever
      // visible in the ring outside the silhouette it belongs to.
      transparent: true,
      depthWrite: true,
      // Slope-scaled depth bias, on top of the constant push in OUTLINE_PUSH.
      //
      // The constant push is a fixed number of units-per-pixel, which is the
      // right amount for a surface facing the camera and far too little for one
      // seen at a grazing angle: shifting the shell w pixels sideways across a
      // near-edge-on panel moves it many pixels' worth in *depth*, so it
      // surfaces back through the panel and draws a dense diagonal hatch across
      // it — the moiré across the hull's aft flank in shots/cel_r2/ocean_low.png
      // at 3x. polygonOffset's factor term scales with the polygon's own depth
      // slope, which is exactly the quantity the constant push cannot see.
      polygonOffset: true,
      polygonOffsetFactor: 4,
      polygonOffsetUnits: 8,
    });
  }

  return { main, prepass, outline, uniforms };
}

// ─────────────────────────────────────────────────────────────────────────────
// Geometry helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Build the `aSmoothNormal` attribute an inverted-hull outline needs.
 *
 * Vertices that share a position but have different shading normals (any hard
 * edge, any UV seam) get one merged, area-weighted normal. Without this the
 * outline hull splits apart at every crease and you see the model's shell
 * through the gaps — the single most common inverted-hull artefact.
 *
 * Call this on every geometry that will be outlined. It is idempotent.
 */
export function computeSmoothNormals(geometry: BufferGeometry): BufferGeometry {
  if (geometry.getAttribute('aSmoothNormal')) return geometry;
  if (!geometry.getAttribute('normal')) geometry.computeVertexNormals();

  const pos = geometry.getAttribute('position');
  const nrm = geometry.getAttribute('normal');
  const count = pos.count;
  const smooth = new Float32Array(count * 3);

  // Bucket by quantised position — 0.1 mm grid, tight enough not to weld
  // genuinely separate surfaces, loose enough to catch float drift.
  const map = new Map<string, number[]>();
  const q = (v: number) => Math.round(v * 10000);
  for (let i = 0; i < count; i++) {
    const key = `${q(pos.getX(i))},${q(pos.getY(i))},${q(pos.getZ(i))}`;
    const list = map.get(key);
    if (list) list.push(i);
    else map.set(key, [i]);
  }

  for (const indices of map.values()) {
    let nx = 0,
      ny = 0,
      nz = 0;
    for (const i of indices) {
      nx += nrm.getX(i);
      ny += nrm.getY(i);
      nz += nrm.getZ(i);
    }
    const len = Math.hypot(nx, ny, nz) || 1;
    nx /= len;
    ny /= len;
    nz /= len;
    for (const i of indices) {
      smooth[i * 3 + 0] = nx;
      smooth[i * 3 + 1] = ny;
      smooth[i * 3 + 2] = nz;
    }
  }

  geometry.setAttribute('aSmoothNormal', new BufferAttribute(smooth, 3));
  return geometry;
}

/**
 * Attach a cel material set to a mesh: assigns the main material, registers
 * the prepass material for the G-buffer pass, and parents an inverted-hull
 * outline child if one was built.
 *
 * The outline is a *child* rather than a second scene-level mesh so it
 * inherits every transform and animation automatically — including skinned
 * and vertex-displaced motion, since it runs the same vertex chunks.
 */
export function applyCel(mesh: Mesh, set: CelMaterialSet, renderOrder = 0): Mesh {
  computeSmoothNormals(mesh.geometry);

  // Feed the outline's size taper. When several meshes share one material set
  // (a rider's soft and hard shells, a boat's hull and trim) the largest wins,
  // so a small part never thins out while the form it belongs to is still big
  // on screen.
  if (!mesh.geometry.boundingSphere) mesh.geometry.computeBoundingSphere();
  const r = mesh.geometry.boundingSphere?.radius ?? 1;
  const u = set.uniforms.uOutlineRadius;
  if (u) u.value = Math.max(u.value as number, r);

  mesh.material = set.main;
  mesh.renderOrder = renderOrder;
  mesh.userData.prepassMaterial = set.prepass;
  mesh.userData.celSet = set;

  if (set.outline) {
    const hull = new Mesh(mesh.geometry, set.outline);
    hull.name = mesh.name + ':outline';
    // Outlines render first so the shaded surface z-tests cleanly over them.
    hull.renderOrder = renderOrder - 1;
    hull.frustumCulled = mesh.frustumCulled;
    // The outline is not part of the G-buffer — the hull trick and the Sobel
    // pass would otherwise double up and produce a doubled, muddy line.
    hull.userData.skipPrepass = true;
    hull.userData.isOutline = true;
    mesh.add(hull);
  }
  return mesh;
}

/** Walk a subtree and set outline width on every cel material found. */
export function setOutlineWidth(root: Object3D, px: number) {
  root.traverse((o) => {
    const set = (o as Mesh).userData?.celSet as CelMaterialSet | undefined;
    if (set) set.uniforms.uOutlineWidthPx.value = px;
  });
}
