/**
 * Diagnostic probe. Boots the game headless and dumps per-racer numbers at
 * several points in a race, so claims like "the boats are driving the wrong
 * way" can be checked against values instead of screenshots.
 *
 *   node harness/probe.mjs --port=5310 --at=0,2,8,20
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

const PORT = Number(args.port ?? 5310);
const URL = `http://localhost:${PORT}/`;
const AT = String(args.at ?? '0,2,8,20,45').split(',').map(Number);
const SEED = args.seed ?? '2247';

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
  server = spawn('npx', ['vite', '--port', String(PORT), '--strictPort'], {
    stdio: ['ignore', 'ignore', 'pipe'],
  });
  server.stderr.on('data', (d) => process.stderr.write(`  vite: ${d}`));
  const deadline = Date.now() + 30000;
  while (Date.now() < deadline) {
    if (await isUp(URL)) return;
    await new Promise((r) => setTimeout(r, 300));
  }
  throw new Error('vite did not start');
}

const fmt = (n, w = 7, d = 2) => String(typeof n === 'number' ? n.toFixed(d) : n).padStart(w);

async function main() {
  await ensureServer();
  const browser = await chromium.launch({
    headless: true,
    args: ['--use-gl=angle', '--use-angle=metal', '--ignore-gpu-blocklist', '--mute-audio'],
  });
  const page = await browser.newPage({ viewport: { width: 900, height: 520 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text());
  });

  await page.goto(`${URL}?harness=1&seed=${SEED}`, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.__INKTIDE__?.ready === true, null, { timeout: 60000 });

  await page.evaluate(() => {
    const H = window.__INKTIDE__;
    H.reset();
    H.setPhase('racing');
    H.setControls({ throttle: 1 });
  });

  let prev = 0;
  for (const t of AT) {
    const step = t - prev;
    prev = t;
    if (step > 0) await page.evaluate((s) => window.__INKTIDE__.simulate(s, 1 / 60), step);
    const rows = await page.evaluate(() => window.__INKTIDE__.probe());

    console.log(`\n── t = ${t}s ────────────────────────────────────────────────────────`);
    console.log(
      'id name    hdg·tan vel·tan   speed      u  latrl  dist  lap  cp  prog  pl  wrong',
    );
    for (const r of rows) {
      console.log(
        `${r.id}  ${r.name.padEnd(6)} ${fmt(r.headingDotTangent)} ${fmt(r.velDotTangent)} ` +
          `${fmt(r.speed)} ${fmt(r.u, 6, 3)} ${fmt(r.lateral, 6, 1)} ${fmt(r.distToLine, 5, 1)} ` +
          `${String(r.lap).padStart(4)} ${String(r.nextCheckpoint).padStart(3)} ` +
          `${fmt(r.progress, 5, 2)} ${String(r.place).padStart(3)}  ${r.wrongWay ? 'WRONG' : '  ok'}`,
      );
    }
  }

  if (errors.length) {
    console.log(`\n⚠ ${errors.length} console error(s):`);
    console.log([...new Set(errors)].slice(0, 10).join('\n'));
  }

  await browser.close();
  if (server) server.kill('SIGTERM');
}

main().catch((e) => {
  console.error(e);
  if (server) server.kill('SIGTERM');
  process.exit(1);
});
