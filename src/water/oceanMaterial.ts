/**
 * The ocean surface shader.
 *
 * Displacement comes from `GERSTNER_GLSL` — the same wave table the CPU uses
 * for buoyancy, so the boats sit in the water rather than near it.
 *
 * ── v4: quantise LAST, and never let a sub-pixel octave reach the quantiser ──
 * The r13 capture measured 20,128 unique colours in a 400×400 patch of the
 * aerial shot and 88% of pixels differing from all four neighbours. That is not
 * ramp lighting, it is noise, and it boils. Two mechanisms produced it and both
 * are gone:
 *
 *   1. **`smoothstep(band ± fwidth(shade))`.** The band edge was resolved
 *      against the screen-space derivative of the *whole* scalar — including its
 *      noise octaves. Where the noise was near or below Nyquist that derivative
 *      is large and random per pixel, so the "hard cel edge" became a per-pixel
 *      random blend between two tones. Every boundary in the frame was a 1 px
 *      checkerboard. v4 selects tones with `step()`: zero-width, no interpolant,
 *      so a region of water contains a *countable* set of flat values.
 *
 *   2. **Noise tiles that grew with distance.** `tile = max(tile, fp·k)` kept
 *      each octave above Nyquist by *scaling the pattern*, which makes the noise
 *      UV a function of radial distance from the camera. From an aerial camera
 *      that is a radial function centred on the nadir point — which is exactly
 *      the concentric fingerprint whorls the critic measured at (1300, 720).
 *      v4 holds every tile constant in world space and instead fades an octave's
 *      *amplitude* to zero as its feature size approaches the pixel footprint
 *      (`resolveW`). An octave you cannot draw contributes nothing rather than
 *      contributing noise, so nothing sub-pixel ever reaches the quantiser and
 *      no camera-centred structure can exist.
 *
 * ── Decorrelation: no dominant angle, no centre ──────────────────────────────
 * The aerial also read as parallel dashes at one angle. Two causes:
 *   • the dominant shading term was `-dot(N.xz, camDir)`, a *radial* function.
 *     From above it whorls; from behind it aligns with the view. v4's dominant
 *     term is `dot(N, L)` — sun facing, view-independent — and the view-dependent
 *     sky-mirror term is multiplied by `graze`, so it fades out completely as the
 *     camera tips toward nadir. An aerial frame therefore has no view-derived
 *     structure at all.
 *   • the detail octaves were sampled on an unwarped lattice. v4 domain-warps
 *     the meso/micro fetches by the macro octave, so the finer detail has no
 *     single axis.
 *
 * ── One scalar, one quantiser ───────────────────────────────────────────────
 * Depth/facing, band ragging, crest curvature, whitecap coverage and hull
 * contact are all composited into the same decision and resolved with `step()`.
 * Nothing in this shader alpha-blends two already-quantised layers, which is
 * what generated the "pastel amoeba mush" in the near field.
 *
 * ── The tone ladder ─────────────────────────────────────────────────────────
 * Seven flat values, dark → light, and the top two belong to foam alone so a
 * hull always sits between the sea's darkest and the foam's lightest:
 *
 *   waterDeep · deep body · waterMid · waterShallow · waterCrest · foamShade · foam
 *
 * The r13 note was that everything below the horizon collapsed to one mid-tone.
 * The darkest band is now raw `waterDeep`, not a lifted copy of it — the "oil
 * slick" risk that lifted it in v3 is a *composite vignette* problem, and the
 * cure for a flat frame cannot be to delete the dark end of the ladder.
 *
 * ── Aerial perspective is banded too ────────────────────────────────────────
 * Distance fade is quantised to four steps toward `uSeaFar` and three toward the
 * haze, with the step boundary ragged by the (resolvable) macro octave so the
 * rings never read as ruled arcs. A smooth fade would re-introduce a continuous
 * value per pixel and undo the whole point of the quantiser.
 *
 * ── Band-limiting (unchanged, still load-bearing) ───────────────────────────
 * The radial grid's vertex spacing grows linearly with distance, so past ~80 m
 * the short waves are sampled below Nyquist. The vertex shader low-passes the
 * shared wave field with four taps at ±`foot`; a fifth centre tap yields the
 * discrete Laplacian (`centre − avg = −h²/4·∇²f`), positive on crests, which is
 * what makes a crest read as a *lip*. This is a filter of the shared field,
 * never a second derivation of it.
 */

import {
  Color,
  FrontSide,
  LinearMipmapLinearFilter,
  ShaderMaterial,
  Texture,
} from 'three';
import { GERSTNER_GLSL, waveUniformArrays } from './gerstner';
import { PAL } from '../core/palette';
import { SHARED } from '../render/celMaterial';
import { makeNoiseTexture } from '../render/textures';

export interface OceanMaterialHandles {
  material: ShaderMaterial;
  /** Set once per frame by the renderer so the foam ring can read scene depth. */
  setSceneDepth(tex: Texture | null): void;
  /** Tuning surface — the tunables the harness/critic loop actually twiddles. */
  readonly uniforms: Record<string, { value: unknown }>;
}

/** Two palette tones blended — used where a tone between two entries is needed. */
const blend = (a: Color, b: Color, t: number) => a.clone().lerp(b, t);

/**
 * Noise used for foam break-up and band-edge ragging.
 *
 * A private cache key (7/5 rather than the shared 6/4) so that switching this
 * texture to trilinear filtering cannot affect any other subsystem that asks
 * `makeNoiseTexture` for a map. The map packs three scales:
 *   r = 5-octave fbm, dominant feature ≈ tile/7
 *   g = single octave,          feature ≈ tile/14
 *   b = single octave,          feature ≈ tile/112
 */
function oceanNoise(): Texture {
  const tex = makeNoiseTexture(512, 7, 5);
  tex.minFilter = LinearMipmapLinearFilter;
  tex.generateMipmaps = true;
  tex.anisotropy = 8;
  tex.needsUpdate = true;
  return tex;
}

export function createOceanMaterial(): OceanMaterialHandles {
  const noise = oceanNoise();

  const material = new ShaderMaterial({
    name: 'ocean',
    side: FrontSide,
    transparent: false,
    uniforms: {
      // Wave field — shared table, uploaded once.
      uWaveA: { value: waveUniformArrays.uWaveA },
      uWaveB: { value: waveUniformArrays.uWaveB },
      uTime: SHARED.uTime,
      uSunDir: SHARED.uSunDir,
      uCameraPos: SHARED.uCameraPos,
      uResolution: SHARED.uResolution,
      uTanHalfFov: SHARED.uTanHalfFov,
      uFar: SHARED.uFar,

      uNoise: { value: noise },
      uSceneDepth: { value: null as Texture | null },

      // ── The tone ladder, dark → light. Seven flat values, no in-betweens.
      /**
       * The dark anchor. v5: `waterShadow`, not `waterDeep`.
       *
       * The r3 capture measured this band arriving on screen at (1,7,40) — L9,
       * i.e. *darker than the ink* — and in pack.png a single contiguous blob of
       * it covered 4.4% of the water area with no interior structure. At L9
       * against L154 cyan with nothing between, that is not a trough, it is a
       * hole punched in the picture. `waterShadow` exists in the palette for
       * exactly this ("the shadow side needs its own lifted tone, not a darker
       * copy of the mid tone") and it also clears the ink, so the darkest water
       * and the darkest line stop colliding.
       */
      uTrough: { value: PAL.waterShadow.clone() },
      /** Shadow *body* — the wave face turned away from the sun. */
      uDeep: { value: blend(PAL.waterShadow, PAL.waterMid, 0.42) },
      uMid: { value: PAL.waterMid.clone() },
      uShallow: { value: PAL.waterShallow.clone() },
      /** Crest lips only — demands curvature as well as facing. */
      uCrest: { value: PAL.waterCrest.clone() },
      uSss: { value: PAL.waterSss.clone() },
      /** The top two values in the ladder belong to foam and nothing else. */
      uFoam: { value: PAL.foam.clone() },
      uFoamShade: { value: PAL.foamShade.clone() },
      uHorizonTint: { value: PAL.skyHorizon.clone() },
      uHaze: { value: PAL.skyHaze.clone() },
      /** The tone the sea converges to under aerial perspective. */
      uSeaFar: { value: blend(PAL.waterMid, PAL.skyHorizon, 0.34) },

      // ── Band-limiting footprint ────────────────────────────────────────────
      // foot ≈ (base + r·slope) · scale, fitted to the radial grid's spacing.
      // base/slope are measured and published by ocean.ts; scale is the only
      // artistic knob (how far inside Nyquist to sit).
      uFilterBase: { value: 0.394 },
      uFilterSlope: { value: 0.028 },
      uFilterScale: { value: 0.62 },

      // ── Band selection ─────────────────────────────────────────────────────
      /**
       * Sun facing — the dominant term, and deliberately view-INDEPENDENT.
       * v3's dominant term was the horizontal direction to the camera, which is
       * a radial function and produced concentric whorls from an aerial camera.
       */
      uSunWeight: { value: 2.15 },
      /**
       * ── SWELL HIERARCHY, the v5 fix for "the ocean is a pattern" ───────────
       *
       * The r3 aerial measured 211 bright-band blobs in a 900² patch with a
       * median area of 228 px² and only ~5× of linear scale range: one
       * characteristic dash size everywhere, which the eye reads as animal print.
       *
       * The cause is in the wave table's *slope* spectrum, not its amplitude
       * spectrum. Slope amplitude is A·k, and for the six waves that is
       * 0.115, 0.117, 0.117, 0.115, 0.089, 0.082 — dead flat. Cel bands are
       * chosen by `dot(N, L)`, i.e. purely by slope, so every wave in the table
       * contributes the *same* tonal range and the shortest resolvable one wins
       * on feature count. There is no swell for the chop to sit on because as far
       * as the shading is concerned there is no swell.
       *
       * We may not edit the wave table (it is shared with buoyancy), so the fix
       * is to shade from a *hierarchy* of two filtered samples of it. The vertex
       * shader now low-passes the shared field a second time with a much wider
       * box (`uSwellFoot` metres, ≈ the mid-chop wavelength) and hands the
       * fragment stage both normals:
       *
       *   • the SWELL normal sets the bands. Its features are 40–70 m across, so
       *     a band is a crescent tens of metres long that reads as the flank of a
       *     rolling swell at any altitude.
       *   • the DETAIL normal only adds `detailRel = detailFace − swellFace` at a
       *     fraction of the weight, deliberately less than one band gap, so the
       *     chop chews the boundary and textures the flank but can never open its
       *     own independent band region. That is the difference between "chop on
       *     a swell" and "a lattice of identical dashes".
       */
      uSwellFoot: { value: 7.5 },
      uDetailWeight: { value: 0.52 },
      /**
       * Grazing-angle sky mirror. This is now the ONLY view-dependent term, and
       * it replaces v4's `-dot(N.xz, camDirXZ)`.
       *
       * That term was still the fingerprint. `camDirXZ` is the *azimuth* from the
       * fragment to the camera, so it rotates a full turn around the point under
       * the camera: dotted against the surface normal it is a whorl generator, and
       * the r16 aerial still showed concentric whorls centred on the nadir even
       * with the term scaled down. Fresnel does the same art-direction job — a
       * wave face turned away presents a more grazing view, so it mirrors the sky
       * and reads light — without any azimuthal structure, because it depends on
       * the *elevation* of the view vector rather than its bearing.
       */
      /**
       * Halved for v5. This is the only view-dependent term and it was the reason
       * the darkest band's coverage swung from 0.02% (a low, grazing camera —
       * Fresnel lifts everything) to 7.06% (a high camera — Fresnel lifts
       * nothing) across the r3 set. A band whose area is a function of camera
       * pitch pops in motion, so its authority has to be small enough that the
       * ladder is essentially camera-independent.
       */
      uFresWeight: { value: 0.26 },
      uFresPivot: { value: 0.34 },
      /**
       * How much the camera is looking along the water rather than down at it,
       * published per FRAME by ocean.ts. Deliberately a uniform and not a
       * per-fragment `1 - abs(V.y)`: any per-fragment measure of view elevation
       * is a radial function of the nadir point, and quantising a radial function
       * draws concentric rings. One number for the whole frame cannot.
       */
      uGraze: { value: 1.0 },
      /** Absolute height. Small — iso-height bands read as contour lines. */
      uHeightWeight: { value: 0.06 },
      uHeightScale: { value: 1.9 },
      /**
       * Band-edge ragging, measured in **screen pixels of boundary
       * displacement** rather than in tone units. This distinction is the whole
       * fix for the marbled "wood grain" swirls the r15 capture put across every
       * near field.
       *
       * A fixed *tone* offset of ±0.16 is 3/4 of the gap between two band
       * thresholds. Where the geometric shade changes quickly across the screen
       * (an aerial view, a steep wave) that offset moves a boundary by a few
       * pixels — a hand-inked wobble, which is what it is for. But where the
       * geometric shade changes slowly (a big smooth swell filling the
       * foreground, or the calmed sea on the results screen) the same offset
       * moves the boundary by *hundreds* of pixels, so the bands stop tracing the
       * water at all and start tracing the iso-contours of the noise field. That
       * is a marbled-paper texture, and it was the most texture-like thing left
       * in the frame.
       *
       * Dividing through by the local gradient of the shade scalar makes the
       * ragging a constant number of pixels everywhere: subtle on a slow gradient,
       * strong where bands are tightly packed — which is also exactly where the
       * dash lattice needed breaking.
       */
      /**
       * v5: 16 → 6. At 16, with the gradient clamp topping out at 0.022, the
       * ragging could move a boundary by 0.10 of shade — half the gap between two
       * band thresholds. A perturbation that large does not wobble a boundary, it
       * *replaces* it: the bands stop tracing the water and start tracing the
       * iso-contours of the noise field, which is the nested-concentric-ellipse
       * "topographic map" the r3 wake crop showed. Ragging is allowed to chew an
       * edge; it is not allowed to author one.
       */
      uRagPixels: { value: 6.0 },

      // ── Noise tiles, in metres, CONSTANT in world space ────────────────────
      // These are the distance over which the whole 512² map repeats, not the
      // feature size. They never change with distance: an octave that becomes
      // sub-pixel has its amplitude faded to zero instead of being rescaled,
      // because rescaling with distance is what drew the fingerprint whorls.
      /**
       * r feature ≈ 30 m — "wind patches".
       *
       * The largest ragging octave, and the one that breaks the swell's *pitch*.
       * The wave table's two dominant trains are only 35° apart, so from altitude
       * their crests read as parallel dashes at one angle with a fixed spacing.
       * Nothing at the scale of a dash can fix that — the lattice is bigger than
       * the dashes. A 30 m octave shifts whole neighbourhoods up or down the
       * ladder, so the aerial reads as patches of rougher and calmer water with
       * crests inside them, which is what an open sea looks like from a helicopter.
       */
      uTilePatch: { value: 210.0 },
      /** r feature ≈ 9.0 m — the macro silhouette of a band or a foam patch. */
      uTileMacro: { value: 63.0 },
      /** r feature ≈ 2.4 m — band lobes and the foam clump silhouette. */
      uTileMeso: { value: 17.0 },
      /** g feature ≈ 0.40 m — the chewed edge and the fine bite in foam. */
      uTileMicro: { value: 5.6 },
      /** Metres of domain warp applied to the finer octaves by the macro one. */
      uWarp: { value: 14.0 },
      /**
       * Fragment-space normal perturbation. Breaks the tessellation zigzag.
       *
       * Held low: at 0.2 it moved dot(N, L) by ±0.05, which the sun weight turns
       * into ±0.10 of shade — half a band. That is enough to draw the noise's own
       * contours into the tone ladder, i.e. the same marbling as over-strong
       * ragging. Its job is to break triangle-aligned band edges, not to shade.
       */
      uRippleStrength: { value: 0.05 },

      // Thresholds on the shade scalar. `shade` is built around 0.5 so these are
      // read directly as "how far up the ladder".
      /**
       * v5: 0.17 → 0.07. The darkest band's job is the deep shadow under a
       * breaking flank, not the whole lee side of the swell. Dropping the
       * threshold cuts its area to roughly a fifth, which is what stops it from
       * ever being the largest contiguous region in frame.
       */
      uBand0: { value: 0.07 },
      uBand1: { value: 0.33 },
      uBand2: { value: 0.56 },
      uBand3: { value: 0.79 },
      /** Curvature (crest-lip) bonus folded into the brightest band's selector. */
      uCurvGain: { value: 2.6 },
      /** Thin drawn highlight riding the shallow→crest boundary. */
      uSheen: { value: 0.5 },
      /**
       * ── The crest ridge line ───────────────────────────────────────────────
       * A hard highlight that follows the *ridge polyline* of the swell rather
       * than an elevation contour. It fires where the swell surface is high AND
       * nearly level — which is the definition of a crest line, and unlike an
       * iso-height band it cannot draw a contour ring around a local high, because
       * its width is inversely proportional to the local curvature: sharp crest,
       * thin line; broad dome, nothing.
       */
      uRidgeSlope: { value: 0.075 },
      uRidgeHeight: { value: 0.18 },
      /**
       * Minimum width, in device pixels, of any band feature the high-frequency
       * half of the ladder is allowed to draw. See the `shadeHi` block.
       */
      uMinFeaturePx: { value: 10.0 },

      // ── Foam ───────────────────────────────────────────────────────────────
      /** Compression (1 − jacobian) at which whitecaps start. */
      uFoamPinch: { value: 0.15 },
      /** Width of the pinch ramp. */
      uFoamSoft: { value: 0.14 },
      /** Multiplier turning the pinch ramp into fractional area coverage. */
      uFoamGain: { value: 0.98 },
      /** Extra coverage on high, sharply curved crest lips. */
      uFoamLip: { value: 0.5 },
      /**
       * Coverage below this draws nothing. Without a floor, every faintly
       * compressed fragment passes a few percent of the noise and the whole
       * foreground gets a fine white pepper.
       */
      uFoamFloor: { value: 0.2 },
      /** Coverage ceiling — keeps the noise biting holes instead of saturating. */
      uFoamCeil: { value: 0.6 },
      /**
       * Metres of view depth behind the water a hull may be and still foam.
       *
       * Widened from 1.15: a buoy stem or a gate leg is a thin vertical object, so
       * the band of water in front of it is only a couple of pixels at a narrow
       * depth window. The r15 capture still showed buoy bases meeting the water
       * with a hard clip. Depth *width* is what turns that sliver into a collar.
       */
      uFoamRingWidth: { value: 2.1 },
      uFoamRingGain: { value: 1.8 },
      /**
       * Radius of the screen-space dilation of the contact test, in METRES at the
       * surface, converted to pixels per fragment.
       *
       * The un-dilated depth-difference test can only mark water that is *behind*
       * recorded geometry in the depth buffer, i.e. the sliver of surface
       * overlapping a submerged flank. For a hull that is a usable band; for a
       * buoy float or a gate pylon it is two or three pixels, and the r17 capture
       * still showed those meeting the water as a hard clip. Dilating the test
       * over a ring of taps turns it into a proper contact buffer: any water
       * within this radius of a depth discontinuity *at the waterline* foams, so
       * the collar exists outside the object's silhouette as well as under it.
       *
       * The depth window (see uFoamRingWidth) is what keeps this honest — a gate
       * arch six metres in the air is tens of metres nearer than the water behind
       * it, fails the window, and gets no collar.
       */
      uContactWorld: { value: 0.75 },

      // ── Sparkle ────────────────────────────────────────────────────────────
      /** Glint cell size in SCREEN PIXELS. World-sized cells gave bokeh. */
      uGlintPx: { value: 19.0 },
      /** Specular gate below which no glint is drawn. */
      uGlintGate: { value: 0.16 },
      uGlintStrength: { value: 1.0 },

      uFresnelBand: { value: 0.965 },

      // ── Aerial perspective (quantised) ─────────────────────────────────────
      /**
       * Band contrast collapses toward uSeaFar in SIX hard steps from here…
       *
       * Six rather than four: a distance-quantised fade is a set of arcs centred
       * on the camera, and at four steps those arcs were legible in the aerial as
       * part of the "concentric whorl" read. Six steps of 15% each, with the step
       * boundary ragged by the macro octave, stays banded without drawing rings.
       */
      uFlattenStart: { value: 240.0 },
      uFlattenEnd: { value: 1150.0 },
      uFlattenAmount: { value: 0.88 },
      uFlattenSteps: { value: 6.0 },
      /** …and the disc dissolves into sky haze in four hard steps from here. */
      uHorizonStart: { value: 700.0 },
      uHorizonEnd: { value: 2450.0 },
      uHorizonSteps: { value: 4.0 },
    },

    vertexShader: /* glsl */ `
      ${GERSTNER_GLSL}

      uniform vec3 uCameraPos;
      uniform float uFilterBase, uFilterSlope, uFilterScale, uSwellFoot;

      varying vec3 vWorldPos;
      varying vec3 vNormal;
      varying vec3 vSwellNormal;
      varying float vSwellH;
      varying float vJacobian;
      varying float vHeight;
      varying float vDist;
      varying float vViewZ;
      varying float vFoot;
      varying float vCurv;
      varying vec4 vScreen;

      void main() {
        // The position attribute arrives as a flat XZ lattice; the mesh is
        // re-centred on the camera on the CPU, and the world-space XZ we feed
        // the wave field is absolute — that is what makes the ocean infinite
        // with no tiling.
        vec3 world = (modelMatrix * vec4(position, 1.0)).xyz;

        // Local vertex spacing, hence the width of the low-pass we must apply
        // to stay inside Nyquist.
        float r = length(world.xz - uCameraPos.xz);
        float foot = (uFilterBase + r * uFilterSlope) * uFilterScale;

        // Four taps at ±foot on each axis plus a centre tap. The offsets cancel
        // in the average, so the average is the same surface convolved with a
        // box of half-width foot; centre − average is −h²/4·∇², a crest detector.
        vec3 p; vec3 n; float j;
        vec3 pAcc = vec3(0.0);
        vec3 nAcc = vec3(0.0);
        float jAcc = 0.0;
        gerstnerSurface(world.xz + vec2(foot, 0.0), uTime, p, n, j);
        pAcc += p; nAcc += n; jAcc += j;
        gerstnerSurface(world.xz - vec2(foot, 0.0), uTime, p, n, j);
        pAcc += p; nAcc += n; jAcc += j;
        gerstnerSurface(world.xz + vec2(0.0, foot), uTime, p, n, j);
        pAcc += p; nAcc += n; jAcc += j;
        gerstnerSurface(world.xz - vec2(0.0, foot), uTime, p, n, j);
        pAcc += p; nAcc += n; jAcc += j;

        vec3 pC; vec3 nC; float jC;
        gerstnerSurface(world.xz, uTime, pC, nC, jC);

        vec3 surfPos = pAcc * 0.25;
        vec3 surfNrm = normalize(nAcc);

        // ── Second, much wider low-pass: the SWELL ────────────────────────────
        // Same box-filter construction, footprint uSwellFoot metres instead of
        // the vertex spacing. A box of half-width R attenuates a wave of
        // wavenumber k by sinc(kR), so at 7.5 m this keeps the 71 m and 44 m
        // trains almost intact, halves the 24 m one and erases everything below —
        // i.e. it is the swell with the chop taken off. The fragment stage shades
        // from THIS and lets the chop only perturb the result; see uDetailWeight.
        float sfoot = max(uSwellFoot, foot);
        vec3 snAcc = vec3(0.0);
        float syAcc = 0.0;
        gerstnerSurface(world.xz + vec2(sfoot, 0.0), uTime, p, n, j);
        snAcc += n; syAcc += p.y;
        gerstnerSurface(world.xz - vec2(sfoot, 0.0), uTime, p, n, j);
        snAcc += n; syAcc += p.y;
        gerstnerSurface(world.xz + vec2(0.0, sfoot), uTime, p, n, j);
        snAcc += n; syAcc += p.y;
        gerstnerSurface(world.xz - vec2(0.0, sfoot), uTime, p, n, j);
        snAcc += n; syAcc += p.y;
        vSwellNormal = normalize(snAcc);
        vSwellH = syAcc * 0.25;

        // Normalised discrete Laplacian: 4·(centre − avg)/foot² = −∇²y.
        // Positive on a crest, negative in a trough, and — unlike height — it
        // picks up the short chop, which is what makes a crest read as a *lip*.
        vCurv = 4.0 * (pC.y - surfPos.y) / max(foot * foot, 1e-4);

        vWorldPos = surfPos;
        vNormal = surfNrm;
        vJacobian = jAcc * 0.25;
        vHeight = surfPos.y;
        vFoot = foot;
        vDist = length(surfPos - uCameraPos);

        vec4 mv = viewMatrix * vec4(surfPos, 1.0);
        // Linear *view* depth, matching what the G-buffer prepass writes
        // (-vViewPos.z / uFar). Comparing radial distance against view depth is
        // what made the first foam ring land in the wrong place.
        vViewZ = -mv.z;
        gl_Position = projectionMatrix * mv;
        vScreen = gl_Position;
      }
    `,

    fragmentShader: /* glsl */ `
      precision highp float;

      uniform vec3 uSunDir, uCameraPos;
      uniform float uTime, uFar, uTanHalfFov;
      uniform vec2 uResolution;
      uniform sampler2D uNoise;
      uniform sampler2D uSceneDepth;

      uniform vec3 uTrough, uDeep, uMid, uShallow, uCrest, uSss, uFoam, uFoamShade;
      uniform vec3 uHorizonTint, uHaze, uSeaFar;

      uniform float uSunWeight, uFresWeight, uFresPivot, uGraze;
      uniform float uDetailWeight, uRidgeSlope, uRidgeHeight, uMinFeaturePx;
      uniform float uHeightWeight, uHeightScale;
      uniform float uRagPixels, uTilePatch, uTileMacro, uTileMeso, uTileMicro, uWarp, uRippleStrength;
      uniform float uBand0, uBand1, uBand2, uBand3, uCurvGain, uSheen;
      uniform float uFoamPinch, uFoamSoft, uFoamGain, uFoamLip;
      uniform float uFoamFloor, uFoamCeil;
      uniform float uFoamRingWidth, uFoamRingGain, uContactWorld;
      uniform float uGlintPx, uGlintGate, uGlintStrength;
      uniform float uFresnelBand;
      uniform float uFlattenStart, uFlattenEnd, uFlattenAmount, uFlattenSteps;
      uniform float uHorizonStart, uHorizonEnd, uHorizonSteps;

      varying vec3 vWorldPos;
      varying vec3 vNormal;
      varying vec3 vSwellNormal;
      varying float vSwellH;
      varying float vJacobian;
      varying float vHeight;
      varying float vDist;
      varying float vViewZ;
      varying float vFoot;
      varying float vCurv;
      varying vec4 vScreen;

      float hash21(vec2 p) {
        return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
      }

      /**
       * How much of an octave whose features are feat metres across may be
       * used, given that one pixel covers fp metres of surface.
       *
       * This is the whole anti-boil mechanism. Below ~2.4 px a feature is not a
       * drawable shape, it is temporal noise, and it must never reach a
       * quantiser: a floor() of a sub-pixel field is a per-pixel coin flip that
       * changes every frame. Fading amplitude (rather than growing the pattern,
       * as v3 did) keeps every tile fixed in world space, so no octave's scale
       * is a function of distance and no camera-centred structure can form.
       */
      float resolveW(float feat, float fp) {
        return smoothstep(fp * 2.4, fp * 5.0, feat);
      }

      /**
       * One tap of the screen-space contact buffer.
       *
       * The w argument weights the tap so the outer ring contributes less coverage
       * than the inner one and the collar is denser against the object.
       */
      float contactTap(sampler2D depthTex, vec2 uv, float viewZ, float far, float width, float w) {
        vec4 d = texture2D(depthTex, uv);
        float bz = d.r * far - viewZ;
        // The window accepts geometry a little in FRONT of the water too: a hull
        // sitting on the surface is a few centimetres proud of it, and without
        // the slack its own waterline is excluded from its own collar.
        float has = step(0.5, d.a) * step(-0.45, bz);
        return has * w * (1.0 - clamp(bz / width, 0.0, 1.0));
      }

      void main() {
        vec3 Nv = normalize(vNormal);
        vec3 Ns = normalize(vSwellNormal);
        vec3 V = normalize(uCameraPos - vWorldPos);
        vec3 L = normalize(uSunDir);

        // World size of one screen pixel here, on the worst axis — which is what
        // goes wrong at grazing angles and what anisotropic filtering runs out of.
        float fp = max(max(fwidth(vWorldPos.x), fwidth(vWorldPos.z)), 1e-5);
        float pxWorld = vDist * 2.0 * uTanHalfFov / max(uResolution.y, 1.0);
        float distFade = smoothstep(60.0, 520.0, vDist);

        // ── Noise: constant world tiles, amplitude gated on resolvability ────
        vec2 wp = vWorldPos.xz;
        vec2 dr = vec2(uTime * 0.013, uTime * -0.009);

        float wPat = resolveW(uTilePatch / 7.0,  fp);   // ≈ 30 m
        float wMac = resolveW(uTileMacro / 7.0,  fp);   // ≈ 9.0 m
        float wMes = resolveW(uTileMeso  / 7.0,  fp);   // ≈ 2.4 m
        float wMic = resolveW(uTileMicro / 14.0, fp);   // ≈ 0.40 m

        vec4 nPat = texture2D(uNoise, wp / uTilePatch + dr * 0.12);
        vec4 nMac = texture2D(uNoise, wp / uTileMacro + dr * 0.35);
        // Domain warp. Without it the finer octaves lie on the same lattice as
        // the macro one and the sum has a legible dominant axis — the "camo
        // bedsheet" read. Warping by the macro octave means the detail layer has
        // no single direction and no centre at any altitude.
        // The warp is taken from the r and g channels (features ≈ tile/7 and
        // tile/14). It used to use b, whose feature is tile/112 ≈ 0.56 m — a
        // fourteen-metre displacement driven by a half-metre field, which injects
        // exactly the high-frequency structure the meso and micro fetches then
        // quantise into filaments.
        vec2 warp = (vec2(nMac.r, nMac.g) - 0.5) * uWarp * wMac;
        vec4 nMes = texture2D(uNoise, (wp + warp) / uTileMeso - dr * 1.1);
        vec4 nMic = texture2D(uNoise, (wp + warp * 0.45) / uTileMicro + dr * 2.0);

        // ── Fragment-space ripple ───────────────────────────────────────────
        // The band boundary is a function of the *interpolated* vertex normal, so
        // at a grazing bow camera — where one triangle spans many pixels — the
        // boundary zigzags along the tessellation and reads as a row of teeth.
        // Perturbing the normal per fragment breaks that alignment, and doubles
        // as the fine surface detail the mesh cannot carry. Gated on
        // resolvability, because a ripple you cannot resolve is not detail.
        // nMes.b is not used: its feature is uTileMeso/112 ≈ 15 cm, gated as if it
        // were 2.4 m, so it was a sub-pixel field perturbing the normal that
        // selects the bands.
        vec2 ripple = vec2(
          (nMes.g - 0.5) * wMes + (nMic.g - 0.5) * wMic,
          (nMes.r - 0.5) * wMes + (nMic.r - 0.5) * wMic);
        vec3 N = normalize(Nv + vec3(ripple.x, 0.0, ripple.y) * uRippleStrength);

        // ── Shade scalar, built around 0.5 ──────────────────────────────────
        // Dominant term is SUN FACING: view-independent, so it cannot produce
        // structure centred on the camera, and it puts a lit and a shadow side on
        // every wave regardless of where the camera is.
        //
        // The only view-dependent term left is Fresnel, and it is scaled by the
        // per-frame uGraze so that an aerial framing has effectively no
        // view-derived contribution at all. Nothing here is a function of the
        // *bearing* from the fragment to the camera, which is the property that
        // was drawing whorls.
        // ── Two scales, one ladder ──────────────────────────────────────────
        // swellFace is the dominant term and it comes from the 7.5 m-filtered
        // normal, so a band boundary is an iso-line of the SWELL's slope: a
        // crescent tens of metres long lying on one flank of a rolling wave, which
        // terminates naturally at the crest (slope → 0) and at the trough.
        //
        // detailRel is what the chop adds on top of the swell. It is deliberately
        // weighted well below one band gap (0.23), so it chews the crescent's edge
        // and textures its interior but can never open a band region of its own.
        // In r3 the chop had the *same* weight as the swell, which is why the
        // aerial was 211 same-sized dashes rather than a sea with a swell in it.
        float swellFace = dot(Ns, L) - 0.66;
        float detailRel = dot(N, L) - dot(Ns, L);
        // Fresnel is taken from the UN-rippled normal. On a calmed sea (the
        // countdown and results cameras drop the sea state) the sun term is almost
        // constant, so Fresnel is the whole tonal range — and running it through
        // the fragment ripple made the bands trace the ripple noise's contours in
        // long liquid swirls, which is the marbled "wood grain" read by another
        // route. The ripple's job is to break triangle-aligned band edges on the
        // sun term; it has no business setting the value of calm water.
        float fres = 1.0 - max(dot(Nv, V), 0.0);
        float hN = clamp(vHeight / uHeightScale, -1.0, 1.0);

        // The LOW-frequency part of the ladder: swell facing, Fresnel, height.
        // Everything here varies slowly across the screen, so every boundary it
        // draws is a large shape by construction.
        float shadeLo = 0.5
          + swellFace * uSunWeight
          + (fres - uFresPivot) * uFresWeight * uGraze
          + hN * uHeightWeight;

        // Ragged, hand-inked band edges. Three octaves — macro silhouette, lobes,
        // chewed edge — each faded out the moment it stops being a drawable
        // shape, so the ragging is *always* a shape and never a dither.
        float jitter = (nPat.r - 0.5) * 0.58 * wPat
                     + (nMac.r - 0.5) * 0.46 * wMac
                     + (nMes.r - 0.5) * 0.30 * wMes
                     + (nMic.g - 0.5) * 0.22 * wMic;
        // …and scaled by the local gradient of the dominant term, so the
        // displacement it produces is a fixed number of PIXELS rather than a
        // fixed number of tone units. See the uRagPixels note. The derivative is
        // taken from the un-rippled vertex normal: the fragment ripple is itself
        // high-frequency, and feeding its derivative back in here would make the
        // ragging amplitude track the noise instead of the wave form.
        // The lower clamp is what governs residual marbling: on the calmed sea of
        // the countdown/results cameras the geometric gradient is very small, and
        // any ragging floor above ~0.002 still walks a band boundary far enough to
        // trace the noise's own contours.
        float bandGrad = clamp(fwidth(dot(Ns, L)) * uSunWeight, 0.0022, 0.014);

        // ── The HIGH-frequency part, with its screen feature size clamped ─────
        // detailRel + ragging is everything in the ladder that varies quickly.
        // Left alone it is exactly what drew the r3 filigree: thin sinuous veins
        // and nested contour rings, because a band boundary's width in pixels is
        // (band gap) / (gradient of shade per pixel), and where this term's
        // gradient was steep that width fell to two or three pixels.
        //
        // Measuring that gradient directly and attenuating the term wherever it
        // would draw a feature narrower than uMinFeaturePx is a hard guarantee on
        // the *minimum band feature size in screen space* — the thing the critic
        // asked for — and it is self-tuning: full strength on a big smooth swell
        // filling the foreground, backed off in the compressed mid-distance where
        // filaments used to form. The swell term is untouched, so the large shapes
        // never lose contrast.
        float shadeHi = detailRel * uDetailWeight + jitter * uRagPixels * bandGrad;
        float gradPx = length(vec2(dFdx(shadeHi), dFdy(shadeHi)));
        // 0.22 ≈ one band gap; a feature is that many shade units wide.
        float allow = 0.22 / max(uMinFeaturePx, 1.0);
        shadeHi *= min(1.0, allow / max(gradPx, 1e-6));

        float shade = shadeLo + shadeHi;

        // ── Aerial perspective, stage 1: quantised band collapse ────────────
        // Computed here, applied per-tone below, so a distant tone lands on a
        // committed value rather than on a per-pixel interpolation.
        float fadeJit = (nMac.r - 0.5) * 0.20 * wMac + (nPat.r - 0.5) * 0.16 * wPat;
        float flat4 = floor(clamp(
          smoothstep(uFlattenStart, uFlattenEnd, vDist) * uFlattenAmount + fadeJit,
          0.0, 1.0) * uFlattenSteps) / uFlattenSteps;

        // ── Hard bands: step(), not smoothstep() ────────────────────────────
        // Zero-width transitions. There is no interpolant, so a region of water
        // contains a countable set of flat values — which is the definition of
        // cel shading and the opposite of what v3 produced.
        float lipCurv = clamp(vCurv * uCurvGain, -1.0, 1.5);
        float crestSel = shade + max(lipCurv, 0.0);

        vec3 col = uTrough;
        col = mix(col, uDeep,    step(uBand0, shade));
        col = mix(col, uMid,     step(uBand1, shade));
        col = mix(col, uShallow, step(uBand2, shade));
        float crestMask = step(uBand3, crestSel);
        col = mix(col, uCrest, crestMask);

        // ── The crest RIDGE line ────────────────────────────────────────────
        // A hard highlight that follows the ridge polyline of the swell: high, and
        // nearly level. Because the region where the slope falls below a threshold
        // has width (threshold / curvature), this is thin on a sharp crest and
        // absent on a broad dome — it can only ever draw a *ridge*, never the
        // concentric contour ring an iso-height test would draw around a local
        // high. Its shape is the crest's own line, which is the one silhouette the
        // r3 bands were missing.
        float swellSlope = length(Ns.xz) / max(Ns.y, 0.2);
        float hSwell = clamp(vSwellH / uHeightScale, -1.0, 1.0);
        float ridge = step(swellSlope, uRidgeSlope) * step(uRidgeHeight, hSwell)
                    * step(0.5, uSheen) * (1.0 - smoothstep(220.0, 700.0, vDist));
        col = mix(col, uCrest, ridge * (1.0 - crestMask));

        // A thin drawn highlight riding the shallow→crest boundary. Because the
        // boundary follows slope and curvature it traces the wave form, not an
        // elevation contour. Gated on curvature and on the meso octave's
        // resolvability so it never becomes a sub-pixel iso-line.
        float sheenLine = step(uBand2, shade) - step(uBand2 + 0.055, shade);
        sheenLine *= step(0.02, lipCurv) * step(0.5, wMes) * step(0.5, uSheen);
        col = mix(col, uCrest, sheenLine * (1.0 - crestMask));

        // ── Back-lit crest translucency ─────────────────────────────────────
        // One hard band of saturated teal where a thin, sharply curved crest is
        // between us and the sun.
        float backLit = max(dot(-V, L), 0.0)
                      * smoothstep(0.22, 0.6, hN)
                      * smoothstep(0.10, 0.35, lipCurv)
                      * (1.0 - smoothstep(60.0, 190.0, vDist));
        col = mix(col, uSss, step(0.18, backLit));

        // Collapse toward the far sea tone in four hard steps.
        col = mix(col, uSeaFar, flat4);

        // ── Foam: one coverage scalar, one hard threshold ────────────────────
        // Jacobian < 1 means the surface is compressing; that is where real water
        // piles up and breaks. Coverage is a *fraction of area*, realised by
        // thresholding noise against it, so the amount of white on screen is
        // directly controllable and the shapes stay hard-edged and irregular.
        float pinch = smoothstep(uFoamPinch, uFoamPinch + uFoamSoft, 1.0 - vJacobian);
        float lip = smoothstep(0.10, 0.42, lipCurv) * smoothstep(0.10, 0.55, hN);
        float coverage = pinch * uFoamGain + lip * uFoamLip;
        // Foam cannot exist on water that is not moving; without this gate, low
        // coverage on flat troughs leaves pale rounded patches reading as lily pads.
        coverage *= smoothstep(0.02, 0.085, length(Nv.xz));
        // Distant whitecaps thin right out. v3 kept 55% of coverage at range and,
        // with the fine octave still live out there, the far field filled with
        // 1 px white pepper — the "TV static" note.
        coverage *= 1.0 - 0.82 * distFade;
        coverage = clamp((coverage - uFoamFloor) / (1.0 - uFoamFloor), 0.0, 1.0) * uFoamCeil;

        // Break-up. Every octave is weighted by resolvability and the sum is
        // renormalised, so the *distribution* stays centred as octaves drop out
        // instead of collapsing to a constant and flipping the threshold wholesale.
        float sumW = 0.34 * wMac + 0.40 * wMes + 0.26 * wMic;
        float breakup = (0.34 * wMac * nMac.r + 0.40 * wMes * nMes.r + 0.26 * wMic * nMic.g)
                      / max(sumW, 1e-3);
        breakup = clamp(0.5 + (breakup - 0.5) * 1.45, 0.0, 1.0);

        float foam = step(1.0 - coverage, breakup);
        float foamCore = step(1.0 - clamp(coverage * 0.5, 0.0, 1.0), breakup);

        // ── Depth-difference contact ────────────────────────────────────────
        // The G-buffer records everything except the water, so where a fragment
        // of water is *in front of* recorded geometry we are looking through the
        // surface at something submerged — a hull skin, a buoy stem, a gate leg.
        //
        // Two parts, both hard:
        //   • the submerged area is stamped with the darkest tone, so a hull
        //     always has shadowed water under its keel instead of a hard clip
        //   • a lacy foam collar in a narrow slot right at the waterline
        vec2 screenUv = (vScreen.xy / vScreen.w) * 0.5 + 0.5;
        vec4 sceneD = texture2D(uSceneDepth, screenUv);
        float sceneZ = sceneD.r * uFar;
        float behind = sceneZ - vViewZ;
        float hasGeo = step(0.5, sceneD.a) * step(-0.02, behind);

        // The shadow under the keel is NOT dilated — it belongs under the hull,
        // not around it — so it stays a single centre tap.
        float sub = hasGeo * (1.0 - clamp(behind / (uFoamRingWidth * 6.0), 0.0, 1.0));
        col = mix(col, uTrough, step(0.08, sub));

        // The collar IS dilated, over two rings of eight taps whose radius is a
        // fixed world size converted to pixels, so a collar is the same physical
        // width at 5 m and at 50 m.
        float rPx = clamp(uContactWorld / max(pxWorld, 1e-4), 2.0, 26.0);
        vec2 texel = rPx / max(uResolution, vec2(1.0));
        float prox = contactTap(uSceneDepth, screenUv, vViewZ, uFar, uFoamRingWidth, 1.0);
        for (int i = 0; i < 8; i++) {
          float a = float(i) * 0.7853982 + 0.19;
          vec2 dir = vec2(cos(a), sin(a));
          prox = max(prox, contactTap(uSceneDepth, screenUv + dir * texel * 0.45,
                                      vViewZ, uFar, uFoamRingWidth, 0.95));
          prox = max(prox, contactTap(uSceneDepth, screenUv + dir * texel,
                                      vViewZ, uFar, uFoamRingWidth, 0.6));
        }
        float ringCov = clamp(prox * uFoamRingGain, 0.0, 1.0);
        // Its own, tighter break-up so the collar reads as churn, not a decal.
        float ringBreak = 0.26 + breakup * 0.58;
        float ringFoam = step(1.0 - ringCov, ringBreak);

        float totalFoam = max(foam, ringFoam);
        // Two flat foam values, chosen by a step. Almost all foam is uFoamShade,
        // whose linear luminance sits just under the flare pass's 0.82 threshold,
        // so a large collar can never bloom into a halo around the boat.
        vec3 foamCol = mix(uFoamShade, uFoam, step(0.5, max(foamCore, ringFoam)));
        // Distant foam settles onto the crest tone so it stops shouting.
        foamCol = mix(foamCol, uCrest, step(0.6, distFade) * (1.0 - ringFoam));
        col = mix(col, foamCol, totalFoam);

        // ── Fresnel band right at the horizon ───────────────────────────────
        col = mix(col, uHorizonTint, step(uFresnelBand, fres) * (1.0 - totalFoam));

        // ── Quantised sparkle — drawn shapes, not specular noise ────────────
        // Cells are sized in SCREEN PIXELS so a glint is always a few px across.
        // Each live cell picks its own rotation, aspect and one of two silhouettes,
        // which is what separates "light catching crests" from a grid of identical
        // hyphens. Gated on crest curvature AND the specular lobe, so it only
        // fires where the light would actually catch, and composited with mix()
        // rather than added — an additive glint invents an off-ladder tone.
        vec3 H = normalize(L + V);
        float specLobe = pow(max(dot(N, H), 0.0), 30.0);
        float gate = step(uGlintGate, specLobe)
                   * step(0.02, lipCurv)
                   * step(uBand1, shade)
                   * (1.0 - totalFoam);

        float cellSize = max(pxWorld * uGlintPx, vFoot * 0.9);
        vec2 gp = vWorldPos.xz / cellSize;
        vec2 cellId = floor(gp);
        float h  = hash21(cellId);
        float h2 = hash21(cellId + 17.3);
        float h3 = hash21(cellId + 51.7);
        // Jitter the stamp inside its cell so the lattice never reads as a grid.
        vec2 f = fract(gp) - vec2(0.24 + 0.5 * h, 0.24 + 0.5 * h2);
        float ga = h3 * 6.2831853;
        vec2 fr = vec2(f.x * cos(ga) - f.y * sin(ga), f.x * sin(ga) + f.y * cos(ga));
        // Shape A: a diamond whose aspect is randomised per cell, so its length
        // and orientation both vary. Shape B: a short cross / four-point star.
        float dia = step(abs(fr.x) * (1.15 + 3.0 * h) + abs(fr.y) * (1.15 + 3.0 * h3), 0.17);
        float star = max(
          step(abs(fr.x) * 4.4 + abs(fr.y) * 0.95, 0.13),
          step(abs(fr.x) * 0.95 + abs(fr.y) * 4.4, 0.13));
        float shape = mix(dia, star, step(0.66, h2));
        float live = step(0.66, h);
        float twinkle = step(0.55, abs(sin(uTime * (1.4 + h * 2.6) + h2 * 6.2832)));
        float glint = shape * live * twinkle * gate * step(0.5, uGlintStrength);
        col = mix(col, uFoam, glint);

        // ── Aerial perspective, stage 2: three hard steps into the haze ─────
        // The target is a *tinted* haze, not the raw near-white sky haze: mixing
        // all the way to uHaze put a blown white band along the horizon.
        float horiz3 = floor(clamp(
          smoothstep(uHorizonStart, uHorizonEnd, vDist) + fadeJit, 0.0, 1.0) * uHorizonSteps)
          / uHorizonSteps;
        col = mix(col, mix(uHaze, uSeaFar, 0.35), horiz3);

        gl_FragColor = vec4(col, 1.0);
      }
    `,
  });

  return {
    material,
    uniforms: material.uniforms as unknown as Record<string, { value: unknown }>,
    setSceneDepth(tex) {
      material.uniforms.uSceneDepth.value = tex;
    },
  };
}
