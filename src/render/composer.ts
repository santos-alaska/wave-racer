/**
 * Post stack — the second of the game's two ink systems.
 *
 * Pipeline per frame:
 *
 *   1. G-buffer pass (MRT, 2 attachments)
 *        → view-space normal + edge bias
 *        → linear view depth + object id
 *   2. Beauty pass into an HDR-ish target
 *   3. Stylised flare: hard threshold → two-tap separable blur at ¼ res
 *   4. Composite: Sobel edges over beauty, plus flare, plus a paper vignette
 *
 * ── Why two line systems ───────────────────────────────────────────────────
 * The inverted hull draws exterior silhouettes beautifully and cheaply, but it
 * physically cannot draw a line *inside* a silhouette — the crease where a
 * rider's arm meets their torso, the panel line down a hull, the lip of a
 * cockpit. Those need a screen-space pass.
 *
 * The two must not overlap or every silhouette gets inked twice and reads as a
 * thick, muddy, slightly doubled edge. We keep them apart by excluding outline
 * meshes from the G-buffer entirely (`userData.skipPrepass`), and by biasing
 * the Sobel response *down* where the depth gradient is very large — a large
 * depth gap is exactly a silhouette, which the hull already owns.
 */

import {
  HalfFloatType,
  Mesh,
  NearestFilter,
  NoBlending,
  OrthographicCamera,
  type PerspectiveCamera,
  PlaneGeometry,
  RGBAFormat,
  Scene,
  ShaderMaterial,
  UnsignedByteType,
  Vector2,
  WebGLRenderer,
  WebGLRenderTarget,
} from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { PAL } from '../core/palette';
import { paletteTone } from './celMaterial';

const FULLSCREEN_VERT = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

/**
 * Sobel over the G-buffer.
 *
 * Three independent edge signals, because no single one is sufficient:
 *   • **depth**  — catches silhouettes and depth discontinuities, but misses
 *                  creases on a continuous surface
 *   • **normal** — catches creases, but misses two parallel surfaces at
 *                  different depths (a wing over a fuselage)
 *   • **id**     — catches two *different objects* meeting at similar depth
 *                  and similar normal, which is exactly an arm across a chest
 *
 * Depth is compared *relatively* (Δd / d) so a line is the same weight at 3 m
 * and 300 m. An absolute threshold gives you dense scribble in the foreground
 * and nothing at all in the distance.
 */
const EdgeShader = {
  uniforms: {
    tDiffuse: { value: null as any },
    tNormal: { value: null as any },
    tDepthId: { value: null as any },
    tFlare: { value: null as any },
    uResolution: { value: new Vector2() },
    uInkColor: { value: paletteTone(PAL.ink) },
    /**
     * The *light* ink. See the HALO block in the shader: where a silhouette
     * falls on a background darker than the ink, a dark line cannot be seen and
     * the contour is drawn in this tone instead.
     */
    uHaloColor: { value: paletteTone(PAL.waterCrest) },
    /**
     * Linear luminance below which a background counts as "darker than ink".
     * The ocean's five bands measure 0.033 (its second-darkest) and 0.136 (its
     * mid band) in linear luminance, so 0.062 splits them: the two bands that
     * swallow the ink get a light contour, the three that do not are untouched.
     */
    uHaloThreshold: { value: 0.062 },
    /** Light-contour width, device px. Matched to the ink it replaces. */
    uHaloWidthPx: { value: 3.2 },
    /**
     * Radius, device px, inside which the inverted-hull outline is assumed to
     * have already inked this pixel's silhouette, so the Sobel stays off it.
     * Must cover the widest shell in the scene (uOutlineMaxPx = 2.8) plus the
     * Sobel's own reach, or the two lines abut into one doubled band.
     */
    uHullOwnedPx: { value: 4.0 },
    /** Line thickness in pixels. */
    uThickness: { value: 1.15 },
    uDepthThreshold: { value: 0.0060 },
    /**
     * 1 − dot(n, n'), so a 30° crease is 0.134, a 36° crease is 0.19 and a 60°
     * crease is 0.50. The original 0.42 needed a *55° break* before it would ink
     * anything, which is why the hull's chine — an authored, per-face-normal
     * crease — never drew a line in any capture.
     *
     * 0.235 is the window between two things that are both present in this
     * geometry: the boat is authored non-indexed, so every non-planar quad has a
     * ~15-40° break along its triangulation diagonal, and at 0.13 those diagonals
     * inked as a dotted hatch across the deck and hull flanks (visible at 4x in
     * shots/cel_fix4 and still faintly at 0.185 in shots/cel_fix6). The chine is a
     * 50-70° break — 0.41 to 0.66 — so it clears 0.235 with room to spare, which
     * shots/cel_fix6/countdown.png confirms: the flank crease is inked.
     */
    uNormalThreshold: { value: 0.235 },
    uEdgeStrength: { value: 0.95 },
    uFlareStrength: { value: 0.24 },
    /** Dead. See `finish()` — a post-band multiplier cannot be part of this look. */
    uVignette: { value: 0.0 },
    /**
     * Off. This was ±1 LSB of interleaved-gradient noise applied to *every*
     * pixel of the final image, to hide 8-bit contouring in the sky gradient.
     * What it actually did was make the whole frame per-pixel unique: 92.3% of
     * sampled water pixels in shots/r2/course.png differed from all four of
     * their neighbours, and a 626×62 patch of the player's deck held 4666
     * distinct colours. A cel image is flat fills; anything that perturbs every
     * pixel is the opposite of the brief, and it would boil at 60 fps. The sky's
     * contouring is now solved where it belongs — by banding the sky on purpose.
     */
    uDither: { value: 0.0 },
  },
  vertexShader: FULLSCREEN_VERT,
  fragmentShader: /* glsl */ `
    precision highp float;

    uniform sampler2D tDiffuse;
    uniform sampler2D tNormal;
    uniform sampler2D tDepthId;
    uniform sampler2D tFlare;
    uniform vec2 uResolution;
    uniform vec3 uInkColor;
    uniform vec3 uHaloColor;
    uniform float uHaloThreshold;
    uniform float uHaloWidthPx;
    uniform float uHullOwnedPx;
    uniform float uThickness;
    uniform float uDepthThreshold;
    uniform float uNormalThreshold;
    uniform float uEdgeStrength;
    uniform float uFlareStrength;
    uniform float uVignette;
    uniform float uDither;

    varying vec2 vUv;

    float ditherNoise(vec2 c) {
      vec2 p = floor(mod(c, 8.0));
      return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
    }

    /**
     * Applied last, in the composite, so nothing downstream re-quantises.
     *
     * ── Why there is no vignette any more ──────────────────────────────────
     * There were two stepped rings here, on the theory that hard steps keep each
     * fill flat *within* a ring. They do, but every band in the image still came
     * out in three near-identical variants — one per ring — and that is exactly
     * what the ramp contract forbids: the final pixel must land on a band.
     * Measured on shots/r3/hero.png: (2,35,147)/(2,34,145)/(2,33,141) and
     * (9,100,206)/(9,98,202)/(8,95,197) for a water field with five bands, and
     * 624 unique colours in the water region. A ~0.97 multiplier applied after
     * the band lookup is a multiplier applied after the band lookup no matter
     * how few discrete values it takes.
     *
     * uVignette is kept as a uniform so the number is discoverable and dead,
     * rather than being silently reintroduced by someone re-deriving it.
     */
    vec3 finish(vec3 col, vec2 uv) {
      float lsb = 0.0034 * pow(max(max(col.r, col.g), col.b), 0.55) * uDither;
      return col + (ditherNoise(gl_FragCoord.xy) - 0.5) * lsb;
    }

    /**
     * Highest and lowest G-buffer coverage flag within radPx of this pixel,
     * as (max, min).
     *
     * tDepthId.a is 1 wherever anything wrote the G-buffer and 0 where nothing
     * did — the sky, the ocean, foam and spray, and every outline shell. So
     * max == 1 means "geometry is nearby" and min == 0 means "background is
     * nearby", and a pixel with both is on a silhouette.
     *
     * Two rings of eight, at radPx and 55% of it. Eight directions is enough on a
     * contour — a silhouette is a *line*, so one of eight evenly spaced taps
     * lands within 22.5 degrees of its perpendicular — and the inner ring stops a
     * thin object slipping between the outer taps.
     */
    vec2 gbufRing(vec2 uv, float radPx) {
      vec2 r = radPx / uResolution;
      float hi = 0.0, lo = 1.0;
      const float K = 0.7071;
      vec2 dirs[8];
      dirs[0] = vec2(1.0, 0.0);      dirs[1] = vec2(-1.0, 0.0);
      dirs[2] = vec2(0.0, 1.0);      dirs[3] = vec2(0.0, -1.0);
      dirs[4] = vec2(K, K);          dirs[5] = vec2(-K, K);
      dirs[6] = vec2(K, -K);         dirs[7] = vec2(-K, -K);
      for (int i = 0; i < 8; i++) {
        float a = texture2D(tDepthId, uv + dirs[i] * r).a;
        float b = texture2D(tDepthId, uv + dirs[i] * r * 0.55).a;
        hi = max(hi, max(a, b));
        lo = min(lo, min(a, b));
      }
      return vec2(step(0.5, hi), step(0.5, lo));
    }

    /**
     * The flare, quantised into two hard shells.
     *
     * The streak buffer is a blurred field, so adding it directly paints a soft
     * radial glow — which is exactly what shots/r2/sky.png shows around the sun
     * (a soft white ellipse in a soft yellow halo) and what put a soft white
     * blob on the rider's helmet in shots/r2/rider_closeup.png. Thresholding its
     * luminance turns the same buffer into a *shape* with a crisp boundary: the
     * blur decides the outline of the shape, the steps decide that it has one.
     */
    vec3 flareAt(vec2 uv) {
      vec3 f = texture2D(tFlare, uv).rgb;
      float l = dot(f, vec3(0.2126, 0.7152, 0.0722));
      float m = step(0.030, l) * 0.42 + step(0.115, l) * 0.58;
      // Normalised to its own hue so the two shells are flat fills, not a ramp.
      vec3 hue = f / max(max(max(f.r, f.g), f.b), 1e-4);
      return hue * m * uFlareStrength;
    }

    void main() {
      vec2 texel = uThickness / uResolution;
      vec3 scene = texture2D(tDiffuse, vUv).rgb;

      vec4 nc = texture2D(tNormal, vUv);
      vec4 dc = texture2D(tDepthId, vUv);
      float edgeBias = nc.a;
      float centreDepth = dc.r;
      float centreId = dc.g;

      // ── Nothing wrote the G-buffer here: sky, ocean, foam, outline shells ──
      //
      // HALO. This is where the value-adaptive contour is drawn, and it is the
      // answer to a measured defect: the ink (10,26,46) is L23, the ocean's
      // second-darkest band (1,22,101) is L25 and its darkest (1,7,40) is L9, so
      // roughly a quarter of every hull's silhouette was a dark line on an
      // equally dark or darker field — no readable contour at all
      // (shots/cel_r2/hud.png: an ink-detector keyed to PAL.ink returns 49,043
      // water pixels).
      //
      // Thickening the line cannot fix a value collision, and the palette's water
      // bands are not this subsystem's to lift. What an animator does instead is
      // switch the contour: dark ink over light ground, *light* ink over dark
      // ground. So on the outside of a silhouette — which is where the
      // inverted-hull shell has laid its ink, since the shell is only ever
      // visible outside the surface it belongs to — the local background tone
      // decides which of the two inks is drawn. The width is one constant number
      // of device pixels, exactly as with the dark line.
      //
      // The background tone is sampled *further out* than the contour, because
      // the pixel under the contour has already been painted with ink and no
      // longer carries the ground's value.
      if (dc.a < 0.5 || edgeBias <= 0.001) {
        vec3 outc = scene;
        if (dc.a < 0.5) {
          vec2 ring = gbufRing(vUv, uHaloWidthPx);
          vec2 far = (uHaloWidthPx + 4.0) / uResolution;
          vec3 g0 = texture2D(tDiffuse, vUv + vec2(far.x, 0.0)).rgb;
          vec3 g1 = texture2D(tDiffuse, vUv - vec2(far.x, 0.0)).rgb;
          vec3 g2 = texture2D(tDiffuse, vUv + vec2(0.0, far.y)).rgb;
          vec3 g3 = texture2D(tDiffuse, vUv - vec2(0.0, far.y)).rgb;
          // Darkest of the four, so a hull sitting half on a bright crest and
          // half in a trough still gets the light line along the dark half.
          vec3 ground = min(min(g0, g1), min(g2, g3));
          float groundLum = dot(ground, vec3(0.2126, 0.7152, 0.0722));
          float dark = 1.0 - step(uHaloThreshold, groundLum);
          outc = mix(outc, uHaloColor, ring.x * dark);
        }
        outc += flareAt(vUv);
        gl_FragColor = vec4(finish(outc, vUv), 1.0);
        return;
      }

      // HULL-OWNED SILHOUETTES. The inverted hull draws every exterior contour,
      // so the screen-space pass must not draw one too — two lines abutting is
      // the measured 6-7 device-px two-tone band on the course gates, double the
      // 3 px the boats carry. The old test for this was the *magnitude* of the
      // depth gradient, which is only a proxy and got it wrong in both
      // directions: it let a distant gate's silhouette through, and it suppressed
      // the rider's arm-over-torso boundary, which is not an exterior silhouette
      // at all and which the shell physically cannot draw (the arm rests on the
      // chest, so the shell's back face is at the contact depth and loses the
      // depth test). Ask the question directly instead: is background within
      // uHullOwnedPx? If so the shell owns this contour; if not, the Sobel does.
      float hullOwned = 1.0 - gbufRing(vUv, uHullOwnedPx).y;

      vec3 centreNormal = nc.rgb * 2.0 - 1.0;

      // 3×3 Sobel kernels, applied to all three signals in one sweep.
      const vec3 kx = vec3(-1.0, 0.0, 1.0);
      float gxDepth = 0.0, gyDepth = 0.0;
      float normalSum = 0.0;
      float idDiff = 0.0;

      for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
          vec2 off = vec2(float(i), float(j)) * texel;
          vec4 sd = texture2D(tDepthId, vUv + off);
          vec4 sn = texture2D(tNormal, vUv + off);

          // Sobel weights: [-1 0 1; -2 0 2; -1 0 1]
          float wx = kx[i + 1] * (j == 0 ? 2.0 : 1.0);
          float wy = kx[j + 1] * (i == 0 ? 2.0 : 1.0);
          gxDepth += sd.r * wx;
          gyDepth += sd.r * wy;

          // Mean, not max. A max over the 3x3 fires on a *single* aliased
          // sample, which is what a grazing surface produces in quantity — one
          // facet a pixel wide, one wildly different normal, one dot of ink. A
          // real crease runs through the kernel and shows up in three or four of
          // the eight neighbours, so the mean separates the two: a line scores
          // ~0.83 of the true break, an isolated alias ~0.27.
          vec3 sNormal = sn.rgb * 2.0 - 1.0;
          normalSum += 1.0 - dot(centreNormal, sNormal);
          idDiff = max(idDiff, abs(sd.g - centreId) > 0.002 ? 1.0 : 0.0);
        }
      }

      // Relative depth gradient → distance-invariant line weight.
      float depthGrad = length(vec2(gxDepth, gyDepth)) / max(centreDepth, 1e-4);
      float depthEdge = smoothstep(uDepthThreshold, uDepthThreshold * 3.4, depthGrad);
      float normalDiff = (normalSum / 8.0) * 2.2;
      float normalEdge = 0.0;

      // Grazing surfaces are the one place a normal-difference Sobel lies. A
      // panel seen almost edge-on packs many facets into a few pixels, so the
      // G-buffer normal is undersampled and every tessellation seam reads as a
      // crease — dense diagonal hatch across the boat's foredeck in
      // shots/cel_r2/outline_check.png.
      //
      // Fading the signal out there (what this did before) throws the baby out:
      // the hull's chine is a 50-70° authored break on a *flank*, i.e. exactly a
      // grazing surface, and multiplying its response by ~0 is why that crease
      // has never been inked in any capture despite the boat being authored
      // non-indexed with per-face normals to provide it. Raise the *threshold*
      // instead — a grazing surface must break harder to earn a line, so real
      // creases survive and tessellation seams (a few degrees) do not.
      float facing = abs(centreNormal.z);
      float grazeThr = uNormalThreshold * mix(2.4, 1.0, smoothstep(0.10, 0.45, facing));
      normalEdge = smoothstep(grazeThr, grazeThr * 1.75, normalDiff);

      // Object-over-object silhouettes — a boat crossing a gate, a rider's helmet
      // against the deck — are not adjacent to background, so hullOwned cannot
      // see them, but the shell does draw them (there is a real depth gap for it
      // to win). Those are the only case the depth-gradient proxy is still needed
      // for, so the thresholds are now set where a genuine metres-deep gap lives
      // and nothing else: the arm-on-chest case measures ~0.05 relative gradient
      // and stays fully inked, a hull in front of a gate 40 m behind it measures
      // an order of magnitude more and does not.
      float silhouette = max(hullOwned, smoothstep(uDepthThreshold * 30.0, uDepthThreshold * 80.0, depthGrad));
      depthEdge *= (1.0 - silhouette * 0.96);
      normalEdge *= (1.0 - silhouette * 0.96);

      // Interior lines are the whole reason this pass exists, so the normal
      // signal leads and the depth signal only fills in where two parallel
      // surfaces overlap (a wing over a cowl).
      float edge = clamp(max(max(depthEdge * 0.85, normalEdge), idDiff * (1.0 - silhouette * 0.96) * 0.8), 0.0, 1.0);
      edge *= edgeBias * uEdgeStrength;

      // A line is drawn or it is not there — two weights, hard thresholds.
      //
      // The smoothstep tail was producing ink at 1-2% opacity all over any
      // tessellated surface whose facets differ by a couple of degrees, and 1-2%
      // ink in a 1 px pattern is exactly the horizontal scanline striping the
      // critic measured across the hull and deck (values alternating de4836 /
      // e24936 down the deck in shots/cel_fix3/outline_check.png). It is
      // sub-pixel high-frequency detail on a moving object, so it shimmers at
      // speed. Clipping the tail to zero removes it at the source, and the two
      // surviving weights keep the line from reading as one uniform machine
      // stroke.
      edge = step(0.34, edge) * (0.60 + 0.40 * step(0.62, edge));

      // One ink value for the whole image.
      //
      // This was uInkColor + scene * 0.20, so the screen-space pass drew a
      // *different* ink from the inverted hull's: measured on the course gates in
      // shots/r3/hud.png as two abutting families, (10,26,46) from the shell and
      // (17,34,54)/(16,34,53) from here. Two ink constants side by side is what
      // makes a doubled edge read as muddy rather than merely thick, and the
      // brief asks for one ink weight and one ink value everywhere.
      vec3 col = mix(scene, uInkColor, edge);
      col += flareAt(vUv);

      gl_FragColor = vec4(finish(col, vUv), 1.0);
    }
  `,
};

/**
 * Hard threshold, plus a G-buffer opt-out.
 *
 * `uThreshold` is high on purpose. At 0.82 the palette's foam (0xf2fbff) sits
 * just over the line, so every whitecap and every spray droplet bloomed and the
 * water's hard ink edges turned to airbrush — the water subsystem was forced to
 * keep its foam *below* white to work around it. At 0.94 only the sun, the sun's
 * own drawn flare and a genuine specular glint qualify, which is the correct set.
 *
 * Materials opt out entirely via `flareMask: 0` on `createCelMaterial`, which
 * lands in the G-buffer's blue channel.
 *
 * ── Nothing outside the G-buffer flares ────────────────────────────────────
 * Geometry that never writes the G-buffer used to keep the luminance test only,
 * which meant the two brightest things in the game — the sky's sun and every
 * foam/spray particle — were the two things guaranteed to bloom. That is where
 * shots/r2/sky.png's photographic sun came from: a drawn hard-edged disc, an
 * anime flare made entirely of step()s, and then a quarter-res blur smeared a
 * soft white ellipse, a soft yellow halo and four soft tapered spikes on top of
 * it. The sun's flare is *drawn*, in sky.ts, and does not want a lens model
 * over it; foam wants to be able to reach white without turning to airbrush,
 * which the water subsystem asked for explicitly.
 *
 * So the rule is inverted: no G-buffer, no flare. What is left flaring is a
 * genuine specular glint on a hull or a helmet — the only case where a bloom is
 * doing artistic work — and any surface can still opt out with `flareMask: 0`.
 */
const ThresholdShader = {
  uniforms: {
    tDiffuse: { value: null as any },
    tDepthId: { value: null as any },
    uThreshold: { value: 0.985 },
    uIntensity: { value: 1.0 },
  },
  vertexShader: FULLSCREEN_VERT,
  fragmentShader: /* glsl */ `
    uniform sampler2D tDiffuse;
    uniform sampler2D tDepthId;
    uniform float uThreshold;
    uniform float uIntensity;
    varying vec2 vUv;
    void main() {
      vec3 c = texture2D(tDiffuse, vUv).rgb;
      float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
      // step(), not smoothstep(): the flare should have a defined shape.
      float m = step(uThreshold, lum);
      vec4 g = texture2D(tDepthId, vUv);
      // g.a < 0.5 → nothing wrote the G-buffer here (sky, particles, foam,
      // outline shells): never flares. See the note above.
      float allow = g.a < 0.5 ? 0.0 : step(0.5, g.b);
      gl_FragColor = vec4(c * m * allow * uIntensity, 1.0);
    }
  `,
};

/**
 * The flare kernel — a **cross star**, not a bloom.
 *
 * An isotropic gaussian is what makes a highlight look photographic: it is a
 * lens model, and the frame it produced (shots/cel_r0/sky.png) was a soft orange
 * radial haze with blurred spokes, exactly what the brief rules out. Anime flares
 * are anisotropic — long straight streaks along a few fixed axes with a small
 * hard core — so this samples along four axes only, with taps spaced
 * geometrically so the streaks reach far without needing many fetches.
 *
 * Running four directions in one pass rather than as a separable pair is what
 * keeps them *straight*: two separable passes would smear the star into a square
 * blob.
 */
const StreakShader = {
  uniforms: {
    tDiffuse: { value: null as any },
    uResolution: { value: new Vector2() },
    uRadius: { value: 1.0 },
    uCore: { value: 0.18 },
    uStreak: { value: 0.42 },
  },
  vertexShader: FULLSCREEN_VERT,
  fragmentShader: /* glsl */ `
    uniform sampler2D tDiffuse;
    uniform vec2 uResolution;
    uniform float uRadius;
    uniform float uCore;
    uniform float uStreak;
    varying vec2 vUv;

    void main() {
      vec2 px = uRadius / uResolution;
      vec3 centre = texture2D(tDiffuse, vUv).rgb;

      // Four axes: horizontal, vertical and both diagonals.
      const int AXES = 4;
      vec2 dirs[4];
      dirs[0] = vec2(1.0, 0.0);
      dirs[1] = vec2(0.0, 1.0);
      dirs[2] = vec2(0.7071, 0.7071);
      dirs[3] = vec2(0.7071, -0.7071);

      vec3 streak = vec3(0.0);
      float wsum = 0.0;
      for (int a = 0; a < AXES; a++) {
        // Geometric tap spacing: 1,2,4,8,16,28 px — a long reach for 12 fetches.
        float d = 1.0;
        for (int i = 0; i < 6; i++) {
          float w = 1.0 / (1.0 + d * 0.85);
          vec2 o = dirs[a] * px * d;
          streak += (texture2D(tDiffuse, vUv + o).rgb + texture2D(tDiffuse, vUv - o).rgb) * w;
          wsum += 2.0 * w;
          d *= 1.85;
        }
      }
      streak /= max(wsum, 1e-4);

      // Small isotropic core so the very centre of a glint is solid, and the
      // long streaks on top of it. The core is deliberately tiny: it is a
      // *highlight*, and anything wider starts reading as a lens.
      // The per-axis average is divided by the total weight of *all* axes, so
      // multiplying back by the axis count restores one axis' worth of energy.
      // Leaving the ×4 in (as the first pass did) made every near-white pixel a
      // blown streak across the hull — shots/cel_r1/outline_check.png.
      gl_FragColor = vec4(centre * uCore + streak * uStreak * float(AXES), 1.0);
    }
  `,
};

export class InkComposer {
  readonly composer: EffectComposer;
  private gbuffer: WebGLRenderTarget;
  private flareA: WebGLRenderTarget;
  private flareB: WebGLRenderTarget;
  private thresholdMat: ShaderMaterial;
  private streakMat: ShaderMaterial;
  private edgePass: ShaderPass;
  private renderPass: RenderPass;
  private fsScene = new Scene();
  private fsCamera = new OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private fsQuad: Mesh;
  private width = 1;
  private height = 1;

  constructor(
    private renderer: WebGLRenderer,
    private scene: Scene,
    private camera: PerspectiveCamera,
  ) {
    // MRT G-buffer. HalfFloat because linear depth in 8 bits gives visibly
    // stepped edge weights on distant geometry.
    this.gbuffer = new WebGLRenderTarget(1, 1, {
      count: 2,
      type: HalfFloatType,
      format: RGBAFormat,
      minFilter: NearestFilter,
      magFilter: NearestFilter,
      depthBuffer: true,
      stencilBuffer: false,
    });
    this.gbuffer.textures[0].name = 'gNormal';
    this.gbuffer.textures[1].name = 'gDepthId';

    const flareOpts = { type: UnsignedByteType, format: RGBAFormat, depthBuffer: false };
    this.flareA = new WebGLRenderTarget(1, 1, flareOpts);
    this.flareB = new WebGLRenderTarget(1, 1, flareOpts);

    this.composer = new EffectComposer(renderer, new WebGLRenderTarget(1, 1, { type: HalfFloatType }));
    this.composer.renderToScreen = true;

    this.renderPass = new RenderPass(scene, camera);
    this.composer.addPass(this.renderPass);

    this.edgePass = new ShaderPass(EdgeShader);
    this.edgePass.material.blending = NoBlending;
    this.composer.addPass(this.edgePass);

    this.composer.addPass(new OutputPass());

    // Standalone full-screen quad used for the flare chain, which runs outside
    // the composer so it can work at quarter resolution.
    this.thresholdMat = new ShaderMaterial({ ...ThresholdShader, blending: NoBlending });
    this.streakMat = new ShaderMaterial({ ...StreakShader, blending: NoBlending });
    this.fsQuad = new Mesh(new PlaneGeometry(2, 2), this.thresholdMat);
    this.fsQuad.frustumCulled = false;
    this.fsScene.add(this.fsQuad);
  }

  setSize(width: number, height: number, pixelRatio: number) {
    this.width = Math.max(1, Math.floor(width * pixelRatio));
    this.height = Math.max(1, Math.floor(height * pixelRatio));
    this.composer.setSize(width, height);
    this.composer.setPixelRatio(pixelRatio);
    this.gbuffer.setSize(this.width, this.height);
    this.flareA.setSize(this.width >> 2, this.height >> 2);
    this.flareB.setSize(this.width >> 2, this.height >> 2);

    const u = this.edgePass.uniforms;
    u.uResolution.value.set(this.width, this.height);
    this.streakMat.uniforms.uResolution.value.set(
      Math.max(1, this.width >> 2),
      Math.max(1, this.height >> 2),
    );
  }

  /**
   * Swap every eligible mesh to its prepass material and render the G-buffer.
   *
   * Meshes opt out by setting `userData.skipPrepass` (outlines, particles,
   * the sky dome) — those are hidden for the pass rather than drawn, so they
   * cannot contribute spurious edges.
   */
  private renderGBuffer() {
    const swapped: { mesh: Mesh; material: any; visible: boolean }[] = [];

    this.scene.traverse((o) => {
      const m = o as Mesh;
      if (!(m as any).isMesh && !(m as any).isPoints && !(m as any).isLine) return;
      const prepass = m.userData.prepassMaterial;
      if (m.userData.skipPrepass || !prepass) {
        if (m.visible) {
          swapped.push({ mesh: m, material: null, visible: true });
          m.visible = false;
        }
        return;
      }
      swapped.push({ mesh: m, material: m.material, visible: m.visible });
      m.material = prepass;
    });

    const prevTarget = this.renderer.getRenderTarget();
    this.renderer.setRenderTarget(this.gbuffer);
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.clear(true, true, false);
    this.renderer.render(this.scene, this.camera);
    this.renderer.setRenderTarget(prevTarget);

    for (const s of swapped) {
      if (s.material === null) s.mesh.visible = s.visible;
      else {
        s.mesh.material = s.material;
        s.mesh.visible = s.visible;
      }
    }
  }

  /** Threshold + one cross-star streak pass at ¼ res → the graphic flare. */
  private renderFlare(sourceTexture: any) {
    const prevTarget = this.renderer.getRenderTarget();

    this.fsQuad.material = this.thresholdMat;
    this.thresholdMat.uniforms.tDiffuse.value = sourceTexture;
    this.thresholdMat.uniforms.tDepthId.value = this.gbuffer.textures[1];
    this.renderer.setRenderTarget(this.flareB);
    this.renderer.clear(true, false, false);
    this.renderer.render(this.fsScene, this.fsCamera);

    // Single pass, four axes. Re-blurring a streak just rounds it off.
    this.fsQuad.material = this.streakMat;
    this.streakMat.uniforms.tDiffuse.value = this.flareB.texture;
    this.renderer.setRenderTarget(this.flareA);
    this.renderer.clear(true, false, false);
    this.renderer.render(this.fsScene, this.fsCamera);

    this.renderer.setRenderTarget(prevTarget);
  }

  render() {
    this.renderGBuffer();

    const u = this.edgePass.uniforms;
    u.tNormal.value = this.gbuffer.textures[0];
    u.tDepthId.value = this.gbuffer.textures[1];

    // The composer's first read target holds the beauty pass once RenderPass
    // has run; we need it before the edge pass, so we run the flare from the
    // previous frame's result. One frame of latency on a soft glow is
    // imperceptible and saves an entire full-res resolve.
    u.tFlare.value = this.flareA.texture;

    this.renderer.setClearColor(0x061024, 1);
    this.composer.render();

    this.renderFlare(this.composer.readBuffer.texture);
  }

  /**
   * The linear-depth + object-id attachment. The water samples this to build
   * its depth-difference foam ring, which is why the ocean itself must stay out
   * of the G-buffer.
   */
  get gbufferDepth() {
    return this.gbuffer.textures[1];
  }

  /** View-space normal attachment, exposed for debugging the edge pass. */
  get gbufferNormal() {
    return this.gbuffer.textures[0];
  }

  get edgeUniforms() {
    return this.edgePass.uniforms;
  }
  get flareUniforms() {
    return this.thresholdMat.uniforms;
  }
  /** Streak kernel controls — radius, core weight, streak weight. */
  get streakUniforms() {
    return this.streakMat.uniforms;
  }

  dispose() {
    this.gbuffer.dispose();
    this.flareA.dispose();
    this.flareB.dispose();
    this.composer.dispose();
  }
}
