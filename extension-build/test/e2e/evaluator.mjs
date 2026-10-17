#!/usr/bin/env node
// Differential test of the coin-selection evaluator (vtPortfolioEvaluator) against the exact
// portfolio aggregation xB («Группы бэктестов») on random portfolios: start-time ties, zero-length,
// open and cancelled deals, concurrency limit, dynamic limit, open-MAE penalty. Also times both.
//
//   node extension-build/build.mjs --test
//   node extension-build/test/e2e/evaluator.mjs [--cases=400]
//
// Needs playwright-core (PLAYWRIGHT_CORE=/path/to/playwright-core/index.mjs to point at a copy)
// and a Chromium (CHROMIUM=/path/to/chrome, default /opt/pw-browsers/chromium-*/chrome-linux/chrome).

import { existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { dirname, extname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..', '..');
const testDist = join(root, 'dist', 'test');
const args = Object.fromEntries(
  process.argv
    .slice(2)
    .filter((arg) => arg.startsWith('--'))
    .map((arg) => {
      const [key, value] = arg.slice(2).split('=');
      return [key, value ?? 'true'];
    }),
);

const COINS =
  'ONDO NEO LDO KAS JASMY ICP GALA ENS CVX CRV ADA JTO XTZ DOT LINK UNI AAVE ATOM AVAX NEAR FIL ETC XLM VET ALGO SAND MANA AXS APE OP ARB INJ SUI SEI TIA DYDX GRT SNX COMP ZEC WLD JUP PYTH ORDI FET RUNE KSM EGLD FLOW THETA HBAR QNT IMX STX MINA CHZ ENJ ROSE ZIL ONE KAVA CELO 1INCH YFI'
    .split(' ')
    .slice(0, Number(args.coins ?? 40));

if (!existsSync(join(testDist, 'app.js'))) {
  throw new Error('Run `node extension-build/build.mjs --test` first');
}
writeFileSync(
  join(testDist, 'index.html'),
  '<!doctype html><html><head><meta charset="utf-8"></head><body><div id="root"></div>' +
    '<script src="./market.js"></script><script type="module" src="./app.js"></script></body></html>',
);
writeFileSync(join(testDist, 'market.js'), readFileSync(join(root, 'test', 'sim', 'market.js')));

const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.png': 'image/png' };
const server = createServer((request, response) => {
  const path = join(testDist, decodeURIComponent(new URL(request.url, 'http://x').pathname));
  if (!path.startsWith(testDist) || !existsSync(path)) {
    response.writeHead(404).end();
    return;
  }
  response.writeHead(200, { 'content-type': types[extname(path)] ?? 'application/octet-stream' });
  response.end(readFileSync(path));
});
await new Promise((ready) => server.listen(0, '127.0.0.1', ready));
const port = server.address().port;

const playwrightPath = process.env.PLAYWRIGHT_CORE ?? 'playwright-core';
const { chromium } = await import(playwrightPath.startsWith('/') ? pathToFileURL(playwrightPath).href : playwrightPath);
const chromiumPath =
  process.env.CHROMIUM ??
  readdirSync('/opt/pw-browsers')
    .filter((name) => /^chromium-\d+$/.test(name))
    .map((name) => `/opt/pw-browsers/${name}/chrome-linux/chrome`)[0];
const browser = await chromium.launch({ executablePath: chromiumPath, args: ['--no-sandbox'] });
const page = await browser.newPage();
const errors = [];
page.on('pageerror', (error) => errors.push(String(error)));
await page.goto(`http://127.0.0.1:${port}/index.html`);
await page.waitForFunction(() => window.__VELES_TEST__ && window.VelesSim, null, { timeout: 30000 });

const result = await page.evaluate((cases) => {
  const T = window.__VELES_TEST__;
  let seed = 20261016;
  const rnd = () => {
    seed = (seed * 1103515245 + 12345) % 2147483648;
    return seed / 2147483648;
  };
  const HOUR = 3600e3;
  const t0 = Date.parse('2024-01-01T00:00:00Z');
  const portfolio = (coins, dealsPerCoin) =>
    Array.from({ length: coins }, (_, k) => {
      const deals = [];
      let t = t0 + Math.floor(rnd() * 48) * HOUR;
      const count = Math.floor(rnd() * dealsPerCoin);
      for (let d = 0; d < count; d++) {
        const s = t + Math.floor(rnd() * 72) * HOUR;
        const st = rnd() < 0.05 ? 'C' : d === count - 1 && rnd() < 0.3 ? 'S' : 'F';
        const length = rnd() < 0.05 ? 0 : Math.floor(rnd() * 200) * HOUR;
        const e = st === 'S' ? t0 + 6000 * HOUR : s + length;
        const o = Array.from({ length: Math.floor(rnd() * 9) }, (_, i) => s + (i + 1) * Math.floor(1 + rnd() * 10) * HOUR);
        deals.push({ s, e, n: Math.round((rnd() - 0.3) * 1000) / 100, m: Math.round(rnd() * 500) / 100, st, o });
        t = s + length;
      }
      return {
        symbol: `C${k}/USDT`,
        backtestId: 1000 + k,
        from: new Date(t0).toISOString(),
        to: new Date(t0 + 6000 * HOUR).toISOString(),
        depositAmount: 37.5,
        deals,
      };
    });
  const keys = [
    ['profit', 'adjustedProfit'],
    ['risk', 'risk'],
    ['monthlyPercent', 'monthlyPercent'],
    ['drawdownPercent', 'drawdownPercent'],
  ];
  const mismatches = [];
  let checked = 0;
  for (let c = 0; c < cases; c++) {
    const entries = portfolio(2 + Math.floor(rnd() * 25), 30);
    const settings = {
      totalDeposit: 5000,
      maxConcurrentPositions: 1 + Math.floor(rnd() * 12),
      penalizeActiveMae: rnd() < 0.5,
      dynamicBlockOrder: rnd() < 0.5 ? 1 + Math.floor(rnd() * 6) : null,
    };
    const fast = T.vtPortfolioEvaluator(entries, settings);
    for (let q = 0; q < 4; q++) {
      const subset = q === 0 ? entries : entries.filter(() => rnd() < 0.7);
      if (subset.length === 0) continue;
      checked += 1;
      const a = fast(subset);
      const b = T.xB(subset, settings).metrics;
      const bad = keys.find(([x, y]) => Math.abs(a[x] - b[y]) > 1e-6 * Math.max(1, Math.abs(b[y])));
      if (bad) mismatches.push({ c, q, key: bad[0], fast: a[bad[0]], exact: b[bad[1]], settings });
      // The full replica used for scoring: every metric, deal counts and per-coin statistics.
      const full = T.vtFastXB(subset, settings);
      const exact = T.xB(subset, settings);
      if (!T.vtSameXB(full, exact)) {
        const key = Object.keys(exact.metrics).find((k) => full.metrics[k] !== exact.metrics[k]) ?? 'perSymbol/deals';
        mismatches.push({ c, q, full: true, key, fast: full.metrics[key], exact: exact.metrics[key], settings });
      }
    }
  }
  // Timing on a large portfolio (150 coins × up to 600 deals).
  const big = portfolio(150, 600);
  const settings = { totalDeposit: 5000, maxConcurrentPositions: 60, penalizeActiveMae: true, dynamicBlockOrder: 5 };
  const fast = T.vtPortfolioEvaluator(big, settings);
  const time = (fn) => {
    const start = performance.now();
    for (let i = 0; i < 5; i++) fn();
    return Math.round((performance.now() - start) / 5);
  };
  const subset = big.slice(1);
  return {
    checked,
    mismatches: mismatches.length,
    examples: mismatches.slice(0, 3),
    deals: big.reduce((sum, entry) => sum + entry.deals.length, 0),
    fastMs: time(() => fast(subset)),
    fullFastMs: time(() => T.vtFastXB(subset, settings)),
    exactMs: time(() => T.xB(subset, settings)),
    fullSame: T.vtSameXB(T.vtFastXB(subset, settings), T.xB(subset, settings)),
    same: JSON.stringify(fast(subset)) === JSON.stringify((({ adjustedProfit, risk, monthlyPercent, drawdownPercent }) => ({ profit: adjustedProfit, risk, monthlyPercent, drawdownPercent, coins: subset.length }))(T.xB(subset, settings).metrics)),
  };
}, Number(args.cases ?? 400));
console.log(JSON.stringify(result, null, 2));
await browser.close();
server.close();
process.exit(result.mismatches === 0 && errors.length === 0 ? 0 : 1);
