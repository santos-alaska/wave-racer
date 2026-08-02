# INK TIDE

A cel-shaded arcade boat racing game on an infinite procedural ocean.
Vite + TypeScript + Three.js. **Zero external assets** — every mesh, every
texture and every sound is generated in code.

```bash
npm install
npm run dev      # → http://localhost:5173
```

## Controls

| Action | Keyboard | Gamepad |
|---|---|---|
| Throttle | `W` / `↑` | RT or A |
| Brake / reverse | `S` / `↓` | LT |
| Steer | `A` `D` / `← →` | Left stick |
| Powerslide (hold, release for boost) | `Shift` / `Space` | B or RB |
| Race again (on results) | `R` | — |
| Cycle camera | `C` | — |

Hold the powerslide through a corner to charge boost — three tiers, longer
charge means a longer boost. Take the crests on the cross-swell leg at speed
and you will get air; land flat or you lose time.

## What's in here

**Three laps, four boats, one closed circuit** marked on open water by a glowing
racing line that rides the swell. Countdown start, checkpoint gates, wrong-way
detection, split times, results board.

- **Infinite ocean** — 6 summed Gerstner waves (long swell + chop + fine detail)
  displaced in the vertex shader, on a radial grid re-centred on the camera.
  Absolute world-space wave sampling means no tiling repetition, and a single
  mesh means no seams and no LOD popping.
- **Cel-shaded everything** — quantised diffuse through a NearestFilter ramp,
  banded specular, fresnel rim, drawn matcaps instead of environment probes.
- **Two ink systems** — inverted-hull outlines at constant screen-space width
  for exterior silhouettes, plus an MRT G-buffer and a Sobel pass for the
  interior lines a hull trick cannot produce.
- **Real buoyancy** — the hull is sampled against the same wave field at six
  points, so it pitches, rolls and slams into troughs.
- **Procedural riders** — rigged and animated in code: they lean into turns,
  shift weight under acceleration, work the throttle, crouch on landings and
  celebrate at the finish.
- **Synthesised audio** — engine tone tracking RPM, speed-scaled water rush,
  impact thuds and a start horn, all Web Audio.

## Architecture

`ARCHITECTURE.md` is the binding contract: coordinate system, module ownership,
the cel pipeline API, the frame-loop order, and the performance budget. Read it
before changing anything.

Quick orientation:

```
src/
  core/       contracts, palette, config, input, maths, RNG
  water/      gerstner.ts (THE wave field), ocean mesh, water shader, foam
  render/     cel materials, procedural textures, post stack, sky
  boat/       hull geometry, buoyancy, handling
  rider/      rig + procedural animation
  race/       spline circuit, gates, lap logic, AI drivers
  camera/     spring-damped chase rig + harness presets
  ui/         canvas-2D HUD, minimap, screens
  audio/      Web Audio synthesis
harness/      Playwright retina screenshot harness
```

See `KNOWN_GAPS.md` for an honest, measured account of what is not yet at target.

Two rules matter more than the rest:

1. **One wave field.** `src/water/gerstner.ts` is the single source of truth.
   CPU code calls `sampleOcean()`; GPU code includes `GERSTNER_GLSL` and is fed
   the shared `uWaveA`/`uWaveB` uniforms. If these ever diverge, boats visibly
   float above or sink through the water.
2. **One palette.** `src/core/palette.ts`. No colour literals in subsystems.

## The screenshot harness

Every visual claim in this project is verified against a real captured frame,
never against reasoning about what the code should do.

```bash
node harness/capture.mjs                        # full shot list
node harness/capture.mjs --shots=hero,foam_wake # named shots
node harness/capture.mjs --list                 # what each shot proves
node harness/capture.mjs --out=shots/round7 --dpr=2 --width=1600
```

It boots the game headless in Chromium with a real Metal GL backend, drives it
to a precise deterministic moment (seeded RNG, fixed-step simulation), and
captures from named in-game camera rigs. The same shot name always produces the
same frame, which is what makes before/after comparison meaningful when you are
iterating on a shader.

The game exposes `window.__INKTIDE__` under `?harness=1` — `simulate()`,
`setPhase()`, `setControls()`, `setCameraPreset()`, `stats()`. See
`ARCHITECTURE.md` for the full table.

## Debug

`?debug=1` adds a perf overlay (fps, frame time, pixel ratio, draw calls,
triangles). `?seed=1234` reseeds every procedural decision.

## Performance

Measured, not assumed. `harness/perf.mjs` runs the **production build** on the
real rAF clock and samples actual frame intervals — the capture harness's own
numbers are meaningless here because it steps with a fixed dt and never waits on
vsync.

```bash
npm run build && node harness/perf.mjs --seconds=14 --dpr=2
```

Mid-race, four boats, chase camera, 1440×810 at device pixel ratio 2.0,
Apple M5 Pro / Chrome (ANGLE Metal):

| | |
|---|---|
| mean | 8.32 ms (120 fps) |
| p50 | 8.30 ms |
| p95 | 9.20 ms |
| worst | 10.80 ms |
| frames over 16.9 ms | **0 of 1675** |
| draw calls | 40 (budget 220) |
| triangles | 167k (budget 1.6 M) |
| adaptive pixel ratio | settled at full 2.00 |

The adaptive controller never had to reduce resolution. It measures a *median*
frame time (so one GC spike cannot drop the resolution), backs off fast when the
budget is blown, and climbs back slowly — an oscillating resolution is more
distracting than running slightly soft.
