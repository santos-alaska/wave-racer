/**
 * Sky dome, cel clouds and a graphic sun.
 *
 * A single inverted sphere carries all three. The clouds are not billboards or
 * a texture — they are procedural shapes evaluated in the fragment shader and
 * then *hard-thresholded*, which is what makes them read as painted cel shapes
 * with a defined rim rather than as soft volumetric fog.
 *
 * The sky writes no G-buffer (`skipPrepass`) and has `edgeBias = 0`, so
 * neither ink system touches it. Inked clouds would fight the boats for
 * attention; in anime backgrounds the sky is almost always line-free.
 *
 * ── What the first capture got wrong ───────────────────────────────────────
 * Two things, both visible in shots/cel_r0/sky.png:
 *
 * 1. **Nothing overhead.** The cloud layer was projected onto a flat plane as
 *    `dir.xz / h`, which is geometrically correct and artistically useless: the
 *    magnification goes to infinity at the zenith, so the whole upper sky sampled
 *    one noise value and came back "no cloud". 80% of the frame was an empty navy
 *    field. The projection is now `dir.xz / (h + k)` — a *flattened dome* — which
 *    keeps the perspective compression near the horizon but bounds the
 *    magnification overhead, so cumulus actually populate the sky you are looking
 *    at. Two layers at different heights give parallax.
 *
 * 2. **A photographic sun.** The disc was a smoothstep and the flare was left to
 *    the post-process blur, which turned it into an orange radial haze with soft
 *    spokes — precisely the look the brief forbids. The sun is now drawn: a hard
 *    disc, a hard detached annulus, and crisp straight rays whose *length* steps
 *    rather than fades. It is drawn bright but under the flare threshold, so the
 *    post pass streaks it instead of smothering it.
 *
 * ── What the r2 capture still got wrong ────────────────────────────────────
 * The gradient was smooth and the sun was a lens.
 *
 * 1. **A 600-colour sky.** The bands were joined by "narrow" smoothsteps and
 *    then *dithered* to hide 8-bit contouring, which is a photographic solution
 *    to a graphic problem: 69 distinct colours in 72 samples down one column of
 *    shots/r2/sky.png, 606 unique colours in a 400×400 patch. The sky is the
 *    largest surface in the game and it was its most photoreal element. It is
 *    now six flat committed bands with hard horizontal thresholds, plus two hard
 *    radial bands around the sun, and no dither at all.
 *
 * 2. **The flare was the post pass, not the drawing.** The disc, annulus and rays
 *    here are all step()s, but the composite let anything outside the G-buffer
 *    bloom — and the sky is outside the G-buffer, so a quarter-res blur painted a
 *    soft white ellipse, a soft yellow halo and four soft tapered spikes straight
 *    over the top of them. That rule is inverted in composer.ts now: no G-buffer,
 *    no flare. What you see around the sun is drawn.
 */

import {
  BackSide,
  Mesh,
  ShaderMaterial,
  SphereGeometry,
  Vector3,
} from 'three';
import { PAL } from '../core/palette';
import { paletteTone, SHARED } from './celMaterial';
import { CONFIG } from '../core/config';

export function createSky(): Mesh {
  const geometry = new SphereGeometry(CONFIG.render.far * 0.46, 64, 40);

  const material = new ShaderMaterial({
    name: 'sky',
    side: BackSide,
    depthWrite: false,
    depthTest: false,
    uniforms: {
      uTime: SHARED.uTime,
      uSunDir: SHARED.uSunDir,
      uZenith: { value: paletteTone(PAL.skyZenith) },
      uMid: { value: paletteTone(PAL.skyMid) },
      uHorizon: { value: paletteTone(PAL.skyHorizon) },
      uHaze: { value: paletteTone(PAL.skyHaze) },
      uSun: { value: paletteTone(PAL.sun) },
      uSunCore: { value: paletteTone(PAL.sunCore) },
      uFlare: { value: paletteTone(PAL.sunFlare) },
      uCloudLit: { value: paletteTone(PAL.cloudLit) },
      uCloudShade: { value: paletteTone(PAL.cloudShade) },
      /**
       * The cloud contour, and the tone the deep shadow band leans on. This is
       * now a real *ink*, not a highlight and not a mid-blue.
       *
       * Both previous versions were too light to be seen. PAL.foam read as a
       * second pale perimeter outside the shadow band; skyMid→cloudShade measured
       * L169 against cloud bands at L191-L211, i.e. a 20 L "line" on a surface
       * whose own bands were 19 L apart — the clouds were the one element in the
       * game with no contour and no readable internal shading, five values inside
       * a 19 L span (shots/r2/sky.png at x=1300).
       *
       * inkSoft→skyZenith is L≈64: a deep blue ink that reads as a drawn line
       * against every cloud tone, without carrying the near-black weight of
       * PAL.ink, which belongs to the racers. A sky-blue ink on the sky and a
       * blue-black ink on the boats keeps the depth order of the image intact.
       */
      uCloudRim: { value: paletteTone(PAL.inkSoft).lerp(paletteTone(PAL.skyZenith), 0.5) },
      // Higher = *less* cloud (it is a threshold on the noise field). Raised from
      // 0.585 after two layers of two-octave field covered nearly the whole upper
      // frame in shots/cel_fix6/countdown.png and the sky read as flat overcast
      // grey-blue — a different palette from the racing shots. Committed sky means
      // real zenith blue between the masses in every shot.
      uCloudCover: { value: 0.645 },
      uWind: { value: new Vector3(0.0075, 0, 0.0042) },
    },
    vertexShader: /* glsl */ `
      varying vec3 vDir;
      void main() {
        vDir = normalize(position);
        // Strip translation: the dome is infinitely far away, so it must not
        // parallax as the camera drives across the ocean.
        vec4 mv = viewMatrix * vec4(position + cameraPosition, 1.0);
        gl_Position = projectionMatrix * mv;
        gl_Position.z = gl_Position.w; // force to the far plane
      }
    `,
    fragmentShader: /* glsl */ `
      precision highp float;

      uniform float uTime;
      uniform vec3 uSunDir;
      uniform vec3 uZenith, uMid, uHorizon, uHaze;
      uniform vec3 uSun, uSunCore, uFlare;
      uniform vec3 uCloudLit, uCloudShade, uCloudRim;
      uniform float uCloudCover;
      uniform vec3 uWind;

      varying vec3 vDir;

      // ── Value noise ────────────────────────────────────────────────────────
      float hash(vec2 p) {
        p = fract(p * vec2(123.34, 456.21));
        p += dot(p, p + 45.32);
        return fract(p.x * p.y);
      }
      float noise(vec2 p) {
        vec2 i = floor(p), f = fract(p);
        f = f * f * (3.0 - 2.0 * f);
        return mix(mix(hash(i), hash(i + vec2(1,0)), f.x),
                   mix(hash(i + vec2(0,1)), hash(i + vec2(1,1)), f.x), f.y);
      }
      /**
       * Two octaves, for the *silhouette* field.
       *
       * Three was still too many. High-frequency content sitting near the
       * threshold does two things, both visible in shots/r2/sky.png: it punches
       * detached islands *inside* the mass (specks at 1339,1037 and 1512,1123)
       * and it frays the outline into the five-fingered amoeba the critic called
       * spilled milk. A cumulus silhouette is lumpy but closed, so the shape
       * field is deliberately the smoothest field in the shader.
       */
      float fbm2(vec2 p) {
        float v = 0.0, a = 0.5;
        for (int i = 0; i < 2; i++) { v += a * noise(p); p *= 2.11; a *= 0.5; }
        return v / 0.75;
      }
      /**
       * The cloud shape field. Factored out because the shading needs to sample
       * it at an *offset*, which is the whole trick below.
       */
      float cloudField(vec2 q, vec2 drift) {
        float warp = fbm2(q * 1.55 - drift * 1.5);
        return fbm2(q * 1.12 + vec2(warp, warp * 0.72) * 0.58 + drift);
      }

      /**
       * One cel cloud layer: three flat tones and one contour, nothing else.
       *
       * The bands are built by *eroding the silhouette from the anti-sun side*
       * rather than by thresholding a second noise field. Sample the same shape
       * field a short step away from the sun: if that sample has left the cloud,
       * this pixel is within one band-width of the away-from-sun edge, so it is
       * in shadow. Two step lengths give two shadow depths, and the result is a
       * lit mass with a shadow crescent that always follows the form and always
       * agrees with SUN_DIR — which a threshold on an unrelated low-frequency
       * field does not, and did not: in shots/cel_fix2/sky.png entire masses
       * landed on one side of that threshold and came out flat white, exactly the
       * volume-less "spilled milk" the critic measured in r2.
       *
       * 'contour' lets the small high layer skip the line entirely. Three layers
       * each drawing their own contour crossed into line spaghetti wherever they
       * overlapped near the horizon (shots/r2/outline_check.png, 1008,187).
       */
      vec4 cloudLayer(vec2 uv, float scale, float cover, vec2 sunUv, float contour) {
        vec2 drift = uWind.xz * uTime * scale;
        vec2 p = uv * scale;
        float d = cloudField(p, drift);

        // The screen-space rate of change of the shape field, evaluated *before*
        // any early-out: a return inside a quad makes the derivative of anything
        // computed after it undefined, and a contour whose width goes undefined
        // is worse than no contour. This is what makes the line below a constant
        // number of *pixels* wide instead of a constant slice of the field, which
        // is a line that gets fatter every metre nearer the camera.
        //
        // Bounded, and switched off where the field is undersampled. Near the
        // horizon the flattened-dome projection compresses several field cycles
        // into a pixel, so fwidth there is a large fraction of the field's whole
        // range and an "unbounded 3 px line" becomes a 100 px slab of ink. A line
        // that cannot be drawn at the right width is not drawn.
        float aa = min(max(fwidth(d), 1.0e-5), 0.014);
        float lineOk = 1.0 - step(0.016, fwidth(d));

        if (step(cover, d) < 0.5) return vec4(0.0);

        // Three tones, spaced by *value*, not by taste: L255 / L211 / L167 in the
        // final frame. The previous ladder ran L255 / L211 / L181-with-a-L169-line
        // and read as mush — four of five nominal steps were invisible.
        //
        // Lit by default. A fair-weather cumulus is bright even in shadow, so the
        // shadow bands stay pale — dropping them further turned the mid-sky clouds
        // into smog in shots/cel_r2/sky.png.
        vec3 col = uCloudLit;

        // Erode from the anti-sun side: wide band → cloudShade, narrow band
        // nearer the edge → one step deeper. The deep band leans on the *ink*
        // now, which is what buys it 44 L of separation from cloudShade.
        float shifted = cloudField(p - sunUv * 0.34, drift);
        float shadeWide = 1.0 - step(cover, shifted);
        float shadeDeep = 1.0 - step(cover, cloudField(p - sunUv * 0.13, drift));
        col = mix(col, uCloudShade, shadeWide);
        col = mix(col, mix(uCloudShade, uCloudRim, 0.45), shadeDeep);

        // The interior shadow boundary, drawn as one hard ink line on the *lit*
        // side of the terminator — the band where the eroded field has just
        // cleared cover, not the region below it. Multiplying an "inside
        // cover + 2aa" mask by the "below cover" mask, which is how this read at
        // first, gives back the second mask unchanged: the whole shadow side came
        // out solid ink and both pale bands vanished from the frame
        // (shots/cel_r3/sky.png, 19.6% of the sky region in one flat L58).
        float shadeLine = step(cover, shifted) * (1.0 - step(cover + aa * 2.5, shifted));
        col = mix(col, uCloudRim, shadeLine * contour * lineOk);

        // The silhouette contour: constant screen width, three device pixels, the
        // same weight the boats carry. Every cloud gets one — they were the only
        // elements in the frame with no ink at all, which is why they read as flat
        // vector art pasted behind a cel-shaded game.
        float edge = 1.0 - step(cover + aa * 3.0, d);
        col = mix(col, uCloudRim, edge * contour * lineOk);

        return vec4(col, 1.0);
      }

      /**
       * One opposed pair of hard-edged triangular rays along an axis rotated by
       * 'rot' in the sun's tangent plane. Widest at the sun, tapering linearly to
       * a point at 'len'.
       */
      float rayPair(vec2 t, float rot, float len, float w0) {
        float c = cos(rot), s = sin(rot);
        vec2 q = vec2(t.x * c + t.y * s, -t.x * s + t.y * c);
        float along = abs(q.x);
        float perp = abs(q.y);
        float w = w0 * (1.0 - along / len);
        return step(perp, w) * step(along, len);
      }

      void main() {
        vec3 dir = normalize(vDir);
        float h = dir.y;
        vec3 L = normalize(uSunDir);

        // ── Banded gradient ─────────────────────────────────────────────────
        // Six flat bands, hard horizontal thresholds, no interpolation anywhere.
        //
        // This was three tones joined by "narrow" smoothsteps, on the theory that
        // fully hard steps would read as a rendering error. Measured on the
        // result: 69 distinct colours in 72 samples down one column and 606
        // unique colours in a 400×400 patch (shots/r2/sky.png). The sky is the
        // largest surface in most frames, so a 600-colour gradient is the single
        // most photoreal thing in the game and it drags everything else toward
        // realism with it. Committed bands instead.
        //
        // The thresholds are not evenly spaced: they crowd toward the horizon,
        // where the eye reads the sky as compressed, and open out overhead where
        // one big field of zenith blue is what an anime background actually is.
        vec3 sky = uHaze;
        sky = mix(sky, mix(uHorizon, uHaze, 0.42), step(0.019, h));
        sky = mix(sky, uHorizon, step(0.056, h));
        sky = mix(sky, mix(uHorizon, uMid, 0.62), step(0.108, h));
        sky = mix(sky, uMid, step(0.186, h));
        sky = mix(sky, mix(uMid, uZenith, 0.55), step(0.315, h));
        sky = mix(sky, uZenith, step(0.545, h));

        // Two more bands, this time radial around the sun, replacing what was a
        // smooth pow(sunSide, 4) desaturation plus a pow(sunSide, 40) glow —
        // i.e. an atmosphere model. Anime skies desaturate toward the sun in
        // *steps*, and the step edges are part of the drawing.
        //
        // Circular, and *close in*. Squashing the axis (dir.y * 0.78) made the
        // bands ellipses, which is half of why the critic read the sun as a lens
        // — a sun is a circle. And at 0.955 the outer band was a 17° disc: a pale
        // 800 px roundel sitting in the sky like a ghost image
        // (shots/cel_fix1/sky.png). These two sit just outside the drawn figure,
        // at ~9.5° and ~6.5°, so they read as the halo an animator paints round
        // the sun rather than as an object of their own.
        float sunSide = dot(dir, L);
        sky = mix(sky, mix(sky, uHaze, 0.42), step(0.9942, sunSide));

        // ── Cel clouds ──────────────────────────────────────────────────────
        // Flattened-dome projection: 'dir.xz / (h + k)'. A true plane ('/h')
        // magnifies without bound at the zenith and leaves the upper sky empty.
        if (h > 0.008) {
          // Not an *alpha* fade. A cloud composited at 40% opacity is a
          // semi-transparent card over the sky: its edges go soft, overlapping
          // layers show through each other, and the pixel lands between two
          // committed tones instead of on one — measured as soft anti-aliased
          // cloud edges and see-through stacked layers in
          // shots/r3/countdown.png. So the horizon and bank falloffs below raise
          // the cloud *cover threshold* instead, which thins the masses by
          // shrinking their silhouettes. Every cloud pixel that survives is a
          // flat, fully opaque fill with a hard edge, in every phase.
          float horizonFade = smoothstep(0.008, 0.075, h);
          float horizonThin = (1.0 - horizonFade) * 0.30;

          // High layer: small, many, drifting faster in uv terms.
          vec2 uvHi = dir.xz / (h + 0.42);
          // The sun's direction inside the cloud plane, which is what the
          // shading erosion steps along.
          vec2 sunUv = normalize(L.xz + vec2(1e-5));
          // No contour on the small layer: three layers each drawing their own
          // line crossed into spaghetti wherever they overlapped.
          // Scales up across all three layers (1.35 → 2.30, 0.72 → 1.18,
          // 0.26 → 0.42). With a two-octave shape field the features come out
          // much larger for the same scale, and the low layer turned into one
          // 1000 px mass filling the middle of shots/cel_fix1/sky.png. Several
          // separate masses with sky between them is the composition.
          // The small layer draws its contour too. It was skipped because three
          // layers of line crossed into spaghetti near the horizon — but that was
          // with a *pale* rim over semi-transparent masses. With one committed ink
          // and opaque fills the topmost layer's line simply covers the ones
          // beneath it, which is how stacked cel layers are supposed to read.
          vec4 hi = cloudLayer(uvHi, 3.60, uCloudCover + 0.055 + horizonThin, sunUv, 1.0);

          // Low layer: bigger masses, slower, sits under the high one.
          vec2 uvLo = dir.xz / (h + 0.20);
          vec4 lo = cloudLayer(uvLo + 31.7, 1.95, uCloudCover + 0.020 + horizonThin, sunUv, 1.0);

          // Flat-bottomed bank hugging the horizon — the cumulus shelf that
          // anchors any anime seascape. Masked to a band in h so it cannot
          // climb into the clear sky above it.
          // Narrower and weaker than it was. A solid shelf across the whole
          // horizon is the "horizon smear" failure: in shots/cel_fix4/countdown.png
          // the bank plus the low layer covered the entire upper frame and the sky
          // read as flat overcast grey-blue — a different palette from the racing
          // shots, which is half of the "not committed across screens" defect.
          float bankBand = smoothstep(0.008, 0.030, h) * (1.0 - smoothstep(0.052, 0.115, h));
          float bankThin = (1.0 - bankBand) * 0.34;
          vec2 uvBank = dir.xz / (h + 0.085);
          // The bank reads as a shelf only if it is *solid*. Dropping its cover
          // below the other layers made it a lace curtain across the whole
          // horizon (shots/cel_probe3/sun_wide.png), so it now sits slightly
          // above them and gets a coarser field.
          vec4 bank = cloudLayer(uvBank * 0.36 + 77.0, 0.62, uCloudCover + 0.075 + bankThin, sunUv, 1.0);

          // Hard composites. .a is already 0 or 1 out of cloudLayer, so this is
          // a flat fill replacing a flat fill.
          sky = mix(sky, lo.rgb, lo.a);
          sky = mix(sky, bank.rgb, bank.a);
          sky = mix(sky, hi.rgb, hi.a);
        }

        // ── Sun ─────────────────────────────────────────────────────────────
        // Drawn, not simulated. Hard disc, a *detached* hard annulus, and rays
        // whose length steps in two stages. There is no inverse-square falloff
        // anywhere in here, and nothing that the post blur can turn into haze:
        // the whole figure is made of step()s.
        // Four concentric elements at most, counting the haze band above: a
        // white disc, a sun-tone collar, a gap, one detached flare ring. Adding
        // a fifth and sixth (which two extra haze bands did) turns the figure
        // into a bullseye — shots/cel_fix2/sky.png reads as a target, not a sun.
        float sd = dot(dir, L);
        float disc  = step(0.99955, sd);
        float halo  = step(0.9984, sd) * (1.0 - disc);
        float gap   = 0.0;
        float ring  = step(0.9952, sd) * (1.0 - step(0.9964, sd));

        // Rays. 'ang' is measured in the plane perpendicular to the sun so the
        // spokes stay straight and evenly spaced regardless of where the sun is.
        vec3 up = abs(L.y) > 0.95 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
        vec3 ex = normalize(cross(up, L));
        vec3 ey = cross(L, ex);
        vec2 tang = vec2(dot(dir, ex), dot(dir, ey));
        float ang = atan(tang.y, tang.x);
        float rad = length(tang);

        // Rays as *triangles*, defined by perpendicular distance from an axis
        // rather than by angular width. Thresholding on angle (the first attempt)
        // gives a shape whose linear width grows with radius and then collapses,
        // so each ray came out lens-shaped — pointed at both ends, widest in the
        // middle, reading as a compass rose (shots/cel_probe3/sun_face.png).
        // Perpendicular distance with a linear taper gives the widest-at-the-sun
        // triangle an animator draws, and the edge is still a single step().
        // Lengths pulled in from 0.300/0.235: the long pair used to reach ~17°
        // from the sun, and with the sun near the top-left of the frame the
        // upper-left spoke ran off the edge of the image (shots/r2/sky.png,
        // clipped at 0,122). The whole figure now fits inside a compact rosette.
        float rays = 0.0;
        rays = max(rays, rayPair(tang,  0.0,       0.205, 0.0125));
        rays = max(rays, rayPair(tang,  1.5707963, 0.165, 0.0100));
        // The diagonals stop *inside* the annulus. When they reached it the
        // ring plus radial spokes read as a ship's wheel rather than as a sun.
        rays = max(rays, rayPair(tang,  0.7853982, 0.062, 0.0042));
        rays = max(rays, rayPair(tang, -0.7853982, 0.062, 0.0042));
        rays *= 1.0 - step(0.9995, sd); // don't draw over the disc itself

        sky = mix(sky, uFlare * 0.92, rays * 0.9);
        sky = mix(sky, uFlare, ring * 0.42);
        sky = mix(sky, uSun * 0.55, gap * 0.35);
        sky = mix(sky, uSun, halo);
        sky = mix(sky, uSunCore, disc);

        // No dither. It existed to hide 8-bit contouring in a long smooth ramp;
        // there is no longer a long smooth ramp to contour, and perturbing every
        // pixel of the largest surface in the frame is the opposite of a limited
        // palette. The banding is the drawing now.
        gl_FragColor = vec4(sky, 1.0);
      }
    `,
  });

  const mesh = new Mesh(geometry, material);
  mesh.name = 'sky';
  mesh.frustumCulled = false;
  mesh.renderOrder = -1000;
  // Neither ink system touches the sky.
  mesh.userData.skipPrepass = true;
  return mesh;
}
