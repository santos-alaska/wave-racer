/**
 * Screenshot harness.
 *
 * Boots the game headless in Chromium at retina scale, drives it to a precise
 * deterministic moment, and captures frames from named camera setups.
 *
 * This exists so that no visual claim in this project is ever made against
 * "what the code should do" — only against a real captured pixel buffer.
 *
 * Usage:
 *   node harness/capture.mjs                        # default shot list
 *   node harness/capture.mjs --shots=hero,foam      # named shots only
 *   node harness/capture.mjs --out=shots/round3     # output directory
 *   node harness/capture.mjs --list                 # print available shots
 *   node harness/capture.mjs --width=1600 --dpr=2
 *
 * Every shot is defined by: a simulated time to advance to, a camera preset,
 * and optional scripted input. Because the sim is seeded and stepped with a
 * fixed dt, the same shot name always produces the same frame — which is what
 * makes before/after comparison meaningful when iterating on a shader.
 */

import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import process from 'node:process';

// ── Args ────────────────────────────────────────────────────────────────────
const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const [k, v] = a.replace(/^--/, '').split('=');
    return [k, v === undefined ? true : v];
  }),
);

const OUT = path.resolve(args.out ?? 'shots/latest');
const WIDTH = Number(args.width ?? 1440);
const HEIGHT = Number(args.height ?? 810);
const DPR = Number(args.dpr ?? 2);
const PORT = Number(args.port ?? 5173);
const URL = args.url ?? `http://localhost:${PORT}/`;
const SEED = args.seed ?? '2247';
const TIMEOUT = Number(args.timeout ?? 90000);

/**
 * Shot list.
 *
 * `t`      seconds of simulation to advance before capturing
 * `until`  JS expression over stats() — advance until true instead of a fixed t.
 *          Use this for transient states (airborne, landing). A fixed timestamp
 *          silently stops proving anything the moment physics is retuned, which
 *          already happened once to the `air` shot in this project.
 * `maxT`   safety bound for `until`
 * `settle` frames to render before capture (default 8). Transient states need
 *          1-2, or the moment is over before the shutter opens.
 * `cam`    camera preset understood by the in-game harness API
 * `drive`  scripted controls applied during the run: {steer, throttle, drift}
 * `phase`  force a race phase, e.g. 'racing' to skip the countdown
 * `note`   what this shot is meant to prove — the critic reads these
 */
const SHOTS = {
  hero: {
    t: 26, cam: 'chase', phase: 'racing', drive: { autopilot: true, throttle: 1 },
    note: 'The money shot. Player boat at speed, wake, water, sky, HUD all together.',
  },
  ocean_wide: {
    t: 14, cam: 'far', phase: 'racing', drive: { autopilot: true, throttle: 1 },
    note: 'Open water to the horizon: checks for tiling repetition, LOD popping, horizon seam.',
  },
  ocean_low: {
    t: 18, cam: 'bow', phase: 'racing', drive: { autopilot: true, throttle: 1 },
    note: 'Camera near the waterline. Checks wave silhouette, crest sharpening, foam threshold.',
  },
  foam_wake: {
    until: 's.drifting && s.driftTier >= 1', maxT: 90, settle: 2,
    cam: 'wake', phase: 'racing', drive: { autopilot: true, throttle: 1, drift: true },
    note: 'Hard drift. Wake ribbon spread/dissipation, foam ring around hull, spray particles.',
  },
  air: {
    until: 's.airborne && s.airTime > 0.28', maxT: 90, settle: 1,
    cam: 'chase', phase: 'racing', drive: { autopilot: true, throttle: 1 },
    note: 'Boat genuinely airborne off a crest. Checks rider tuck, hull pitch, spray on launch.',
  },
  land: {
    until: 's.landingImpact > 1.2', maxT: 90, settle: 2,
    cam: 'chase', phase: 'racing', drive: { autopilot: true, throttle: 1 },
    note: 'Water re-entry. Checks landing crouch, impact spray burst, camera shake.',
  },
  rider_closeup: {
    until: 'Math.abs(s.speed) > 18', maxT: 60, cam: 'rider', phase: 'racing', drive: { autopilot: true, throttle: 1 },
    note: 'Rider leaning into a turn. Checks pose, outline on limbs, cel banding on cloth/skin.',
  },
  outline_check: {
    t: 12, cam: 'orbit_near', phase: 'racing', drive: { autopilot: true, throttle: 0.4 },
    note: 'Slow orbit close to the hull. Checks outline width consistency and silhouette breaks.',
  },
  outline_far: {
    t: 12, cam: 'far_boat', phase: 'racing', drive: { autopilot: true, throttle: 1 },
    note: 'Same boat at distance. Outline must be the same screen width as outline_check.',
  },
  pack: {
    t: 8, cam: 'broadcast', phase: 'racing', drive: { autopilot: true, throttle: 1 },
    note: 'All four racers in frame. Checks per-racer colour separation and AI spacing.',
  },
  course: {
    t: 5, cam: 'aerial', phase: 'racing', drive: { autopilot: true, throttle: 1 },
    note: 'High aerial. Checks racing line ribbon riding the swell, gate placement, circuit shape.',
  },
  countdown: {
    t: 1.6, cam: 'auto', phase: 'countdown', drive: {},
    note: 'Countdown cinematic orbit + HUD countdown treatment.',
  },
  results: {
    t: 2.5, cam: 'auto', phase: 'results', drive: {},
    note: 'Results screen: placements, times, celebration animation, orbit camera.',
  },
  hud: {
    until: 's.drifting && s.boostMeter > 0.35', maxT: 90, settle: 2,
    cam: 'chase', phase: 'racing', drive: { autopilot: true, throttle: 1, drift: true },
    note: 'HUD under load: boost meter charging, speed high, minimap, position, splits.',
  },
  sky: {
    t: 10, cam: 'sky', phase: 'racing', drive: { autopilot: true, throttle: 1 },
    note: 'Camera tilted up. Checks sky gradient banding, cel clouds, sun flare treatment.',
  },
};

if (args.list) {
  for (const [k, v] of Object.entries(SHOTS)) console.log(`${k.padEnd(16)} ${v.note}`);
  process.exit(0);
}

const requested = args.shots
  ? String(args.shots).split(',').map((s) => s.trim()).filter(Boolean)
  : Object.keys(SHOTS);

for (const name of requested) {
  if (!SHOTS[name]) {
    console.error(`Unknown shot "${name}". Run with --list to see available shots.`);
    process.exit(1);
  }
}

// ── Dev server ──────────────────────────────────────────────────────────────
async function isUp(url) {
  try {
    const r = await fetch(url, { signal: AbortSignal.timeout(1200) });
    return r.ok;
  } catch {
    return false;
  }
}

let server = null;
async function ensureServer() {
  if (await isUp(URL)) {
    console.log(`▸ using existing server at ${URL}`);
    return;
  }
  if (!existsSync('node_modules')) {
    console.error('node_modules missing — run `npm install` first.');
    process.exit(1);
  }
  console.log('▸ starting vite…');
  server = spawn('npx', ['vite', '--port', String(PORT), '--strictPort'], {
    stdio: ['ignore', 'pipe', 'pipe'],
    detached: false,
  });
  server.stderr.on('data', (d) => process.stderr.write(`  vite: ${d}`));
  const deadline = Date.now() + 30000;
  while (Date.now() < deadline) {
    if (await isUp(URL)) {
      console.log('▸ vite ready');
      return;
    }
    await new Promise((r) => setTimeout(r, 350));
  }
  throw new Error('vite did not come up within 30s');
}

function stopServer() {
  if (server && !server.killed) server.kill('SIGTERM');
}

// ── Capture ─────────────────────────────────────────────────────────────────
async function main() {
  await ensureServer();
  await mkdir(OUT, { recursive: true });

  const browser = await chromium.launch({
    headless: true,
    args: [
      // Headless Chromium needs a real GL backend or every shader silently
      // falls back to SwiftShader and the captures stop reflecting the game.
      '--use-gl=angle',
      '--use-angle=metal',
      '--enable-unsafe-webgpu',
      '--ignore-gpu-blocklist',
      '--enable-gpu-rasterization',
      '--disable-frame-rate-limit',
      '--hide-scrollbars',
      '--mute-audio',
    ],
  });

  const context = await browser.newContext({
    viewport: { width: WIDTH, height: HEIGHT },
    deviceScaleFactor: DPR,
    reducedMotion: 'no-preference',
  });

  const page = await context.newPage();
  const logs = [];
  page.on('console', (m) => logs.push(`[${m.type()}] ${m.text()}`));
  page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}\n${e.stack ?? ''}`));

  const target = `${URL}?harness=1&seed=${SEED}`;
  console.log(`▸ loading ${target}`);
  await page.goto(target, { waitUntil: 'domcontentloaded', timeout: TIMEOUT });

  // Wait for the game to announce it is fully constructed.
  try {
    await page.waitForFunction(() => window.__INKTIDE__?.ready === true, null, { timeout: TIMEOUT });
  } catch (err) {
    console.error('\n✗ Game never became ready. Console output:\n');
    console.error(logs.join('\n') || '(no console output)');
    await page.screenshot({ path: path.join(OUT, '_failure.png') });
    await browser.close();
    stopServer();
    process.exit(2);
  }

  const renderer = await page.evaluate(() => window.__INKTIDE__.rendererInfo?.() ?? {});
  console.log(`▸ GL: ${renderer.renderer ?? 'unknown'}`);
  if (/swiftshader|llvmpipe|software/i.test(renderer.renderer ?? '')) {
    console.warn('  ⚠ software rasteriser — captures will be slow but still valid');
  }

  const manifest = [];
  for (const name of requested) {
    const shot = SHOTS[name];
    process.stdout.write(`▸ ${name} … `);
    const t0 = Date.now();

    const result = await page.evaluate(
      async ([shotName, shotDef]) => {
        const H = window.__INKTIDE__;
        H.reset();
        if (shotDef.phase) H.setPhase(shotDef.phase);
        H.setControls(shotDef.drive ?? {});

        // Fixed-step simulation: identical every run, independent of how fast
        // the headless machine happens to be.
        let hunted = null;
        if (shotDef.until) {
          hunted = await H.simulateUntil(shotDef.until, shotDef.maxT ?? 90, 1 / 60);
        } else {
          await H.simulate(shotDef.t, 1 / 60);
        }

        H.setCameraPreset(shotDef.cam);
        // Settle frames so spring cameras and particle systems land. Transient
        // states want very few, or the moment passes before capture.
        await H.settle(shotDef.settle ?? 8);
        return { ...H.stats(), hunted };
      },
      [name, shot],
    );

    const file = path.join(OUT, `${name}.png`);
    await page.screenshot({ path: file, animations: 'disabled' });
    manifest.push({ name, file: path.relative(process.cwd(), file), note: shot.note, stats: result });
    let extra = '';
    if (shot.until) {
      extra = result.hunted?.found
        ? `  hunted@t=${result.hunted.t.toFixed(2)}`
        : `  ⚠ STATE NEVER REACHED (${shot.until}) — this shot proves nothing`;
    }
    console.log(
      `${Date.now() - t0}ms  (tris ${result.triangles ?? '?'}, draws ${result.drawCalls ?? '?'})${extra}`,
    );
  }

  await writeFile(
    path.join(OUT, 'manifest.json'),
    JSON.stringify({ width: WIDTH, height: HEIGHT, dpr: DPR, seed: SEED, shots: manifest }, null, 2),
  );

  const errors = logs.filter((l) => l.startsWith('[error]') || l.startsWith('[pageerror]'));
  if (errors.length) {
    console.log(`\n⚠ ${errors.length} console error(s):`);
    console.log(errors.slice(0, 20).join('\n'));
    await writeFile(path.join(OUT, 'console.log'), logs.join('\n'));
  }

  await browser.close();
  stopServer();
  console.log(`\n✓ ${manifest.length} shot(s) → ${OUT}  (${WIDTH * DPR}×${HEIGHT * DPR})`);
}

process.on('SIGINT', () => {
  stopServer();
  process.exit(130);
});

main().catch((err) => {
  console.error(err);
  stopServer();
  process.exit(1);
});
