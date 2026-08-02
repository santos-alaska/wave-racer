/**
 * Real-time performance measurement.
 *
 * The capture harness steps the simulation with a fixed dt, which makes its
 * frame-time numbers meaningless as a performance signal (it happily reports
 * thousands of "fps" because it is not waiting on vsync). This script instead
 * lets the game run on its own rAF clock and samples the real frame interval,
 * which is the only way to substantiate a 60 fps claim.
 *
 *   node harness/perf.mjs --port=5320 --seconds=12 --dpr=2
 *
 * Reports the frame-time distribution, the adaptive resolution scale the
 * renderer settled on, and draw/triangle counts. A p95 above ~16.6 ms means we
 * are not actually holding 60.
 */

import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import process from 'node:process';

const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const [k, v] = a.replace(/^--/, '').split('=');
    return [k, v === undefined ? true : v];
  }),
);

const PORT = Number(args.port ?? 5320);
const URL = `http://localhost:${PORT}/`;
const SECONDS = Number(args.seconds ?? 12);
const DPR = Number(args.dpr ?? 2);
const WIDTH = Number(args.width ?? 1440);
const HEIGHT = Number(args.height ?? 810);

async function isUp(u) {
  try {
    return (await fetch(u, { signal: AbortSignal.timeout(1200) })).ok;
  } catch {
    return false;
  }
}

let server = null;
async function ensureServer() {
  if (await isUp(URL)) return;
  // Measure the production build, not the dev server: dev ships unminified
  // modules and an HMR client, neither of which the player will run.
  server = spawn('npx', ['vite', 'preview', '--port', String(PORT), '--strictPort'], {
    stdio: ['ignore', 'ignore', 'pipe'],
  });
  server.stderr.on('data', (d) => process.stderr.write(`  vite: ${d}`));
  const deadline = Date.now() + 30000;
  while (Date.now() < deadline) {
    if (await isUp(URL)) return;
    await new Promise((r) => setTimeout(r, 300));
  }
  throw new Error('vite preview did not start — run `npm run build` first');
}

const pct = (sorted, p) => sorted[Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length))];

async function main() {
  await ensureServer();
  const browser = await chromium.launch({
    headless: true,
    args: [
      '--use-gl=angle',
      '--use-angle=metal',
      '--ignore-gpu-blocklist',
      '--enable-gpu-rasterization',
      '--mute-audio',
      // Do NOT pass --disable-frame-rate-limit here: we want to measure against
      // a real vsync, the way the player experiences it.
    ],
  });
  const page = await browser.newPage({
    viewport: { width: WIDTH, height: HEIGHT },
    deviceScaleFactor: DPR,
  });

  const errs = [];
  page.on('pageerror', (e) => errs.push(e.message));

  await page.goto(`${URL}?harness=1&seed=2247`, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.__INKTIDE__?.ready === true, null, { timeout: 60000 });

  const info = await page.evaluate(() => window.__INKTIDE__.rendererInfo());
  console.log(`GL: ${info.renderer}`);
  if (/swiftshader|software|llvmpipe/i.test(info.renderer)) {
    console.log('⚠ software rasteriser — these numbers say nothing about real hardware');
  }

  // Put the game into a representative worst case: mid-race, full throttle,
  // all four boats on screen, chase camera.
  await page.evaluate(() => {
    const H = window.__INKTIDE__;
    H.reset();
    H.setPhase('racing');
    H.setControls({ autopilot: true, throttle: 1 });
    H.setCameraPreset('chase');
    H.release(); // hand the clock back to rAF — this is the whole point
  });

  console.log(`Measuring ${SECONDS}s of real-time frames at ${WIDTH}x${HEIGHT} dpr=${DPR}…`);

  const result = await page.evaluate(async (secs) => {
    const samples = [];
    let last = performance.now();
    const started = last;
    await new Promise((resolve) => {
      const tick = (now) => {
        samples.push(now - last);
        last = now;
        if (now - started >= secs * 1000) return resolve(null);
        requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    });
    const s = window.__INKTIDE__.stats();
    return { samples, stats: s };
  }, SECONDS);

  // Drop the first few frames — shader compilation and the first GPU upload are
  // real but they are a load cost, not a steady-state frame cost.
  const samples = result.samples.slice(8).filter((v) => v > 0.1 && v < 500);
  samples.sort((a, b) => a - b);

  const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
  const p50 = pct(samples, 50);
  const p95 = pct(samples, 95);
  const p99 = pct(samples, 99);
  const worst = samples[samples.length - 1];
  const over16 = samples.filter((v) => v > 16.9).length;
  const over33 = samples.filter((v) => v > 33.4).length;

  console.log('');
  console.log(`frames sampled   ${samples.length}`);
  console.log(`mean             ${mean.toFixed(2)} ms  (${(1000 / mean).toFixed(1)} fps)`);
  console.log(`p50              ${p50.toFixed(2)} ms  (${(1000 / p50).toFixed(1)} fps)`);
  console.log(`p95              ${p95.toFixed(2)} ms  (${(1000 / p95).toFixed(1)} fps)`);
  console.log(`p99              ${p99.toFixed(2)} ms`);
  console.log(`worst            ${worst.toFixed(2)} ms`);
  console.log(`> 16.9 ms        ${over16} (${((over16 / samples.length) * 100).toFixed(1)}%)  — missed 60`);
  console.log(`> 33.4 ms        ${over33} (${((over33 / samples.length) * 100).toFixed(1)}%)  — missed 30`);
  console.log('');
  console.log(`draw calls       ${result.stats.drawCalls}`);
  console.log(`triangles        ${result.stats.triangles}`);
  console.log(`pixel ratio      ${result.stats.pixelRatio?.toFixed?.(2)}  (adaptive scaler settled here)`);
  console.log('');
  const verdict = p95 <= 17.2 ? '✓ holds 60 fps' : p95 <= 22 ? '~ mostly 60, some misses' : '✗ does NOT hold 60';
  console.log(`VERDICT: ${verdict}  (p95 ${p95.toFixed(2)} ms)`);

  if (errs.length) console.log(`\n⚠ page errors:\n${[...new Set(errs)].slice(0, 5).join('\n')}`);

  await browser.close();
  if (server) server.kill('SIGTERM');
}

main().catch((e) => {
  console.error(e);
  if (server) server.kill('SIGTERM');
  process.exit(1);
});
