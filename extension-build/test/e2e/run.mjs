#!/usr/bin/env node
// Simulated end-to-end run of the optimizer engine in headless Chromium: the real app bundle
// (test build) with a fake Veles transport backed by test/sim/market.js.
//
//   node extension-build/build.mjs --test
//   node extension-build/test/e2e/run.mjs [--method=pattern|genetic] [--budget=110] [--coins=40]
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

const summary = await page.evaluate(
  async ({ method, budget, coins, robustness, months, screen, depositChange, folds, stopAfter, mode }) => {
    const T = window.__VELES_TEST__;
    const sim = window.VelesSim;
    const strategy = sim.makeStrategy();
    let form =
      mode === 'single'
        ? { ...T.Bft({}), mode: 'single', symbolsText: coins.slice(0, 6).join(', ') }
        : T.Fft(T.Bft({}), coins);
    form = {
      ...form,
      stressText: '',
      stressEnabled: false,
      robustness: { ...form.robustness, enabled: robustness },
      portfolio: {
        ...form.portfolio,
        ...(months !== null ? { screeningMonths: months } : {}),
        ...(screen !== null ? { screeningCount: screen } : {}),
        ...(depositChange !== null ? { maxDepositChangePercent: depositChange } : {}),
        ...(folds !== null ? { foldConsistency: folds } : {}),
      },
      genetic: {
        ...form.genetic,
        searchMethod: method,
        ...(budget ? { searchBudget: budget } : {}),
      },
    };
    const config = T.Vft(form, {
      botId: strategy.id,
      botName: strategy.name,
      ranges: {},
      botDeposit: strategy.deposit.amount,
    });
    const results = new Map();
    let nextId = 1;
    let engine = null;
    const transport = {
      submit: async (payload) => {
        const id = nextId++;
        if (stopAfter && id === stopAfter && engine) {
          setTimeout(() => engine.stop(), 0);
        }
        results.set(id, sim.backtest(payload));
        return id;
      },
      fetchStatistics: async (id) => ({ id, ...results.get(id).stats }),
      fetchStatus: async () => 'FINISHED',
      fetchLastCycle: async () => null,
      finalize: async (stats) => ({ ...stats, netQuote: stats.profitQuote - stats.commissionQuote }),
      fetchAllCycles: async (id) => results.get(id).cycles,
    };
    const logs = [];
    const callbacks = {
      onLog: (level, message) => logs.push(`[${level}] ${message}`),
      onProgress: () => {},
      onCandidate: () => {},
      onLeaderboard: () => {},
    };
    const deps = {
      fetchStrategy: async () => strategy,
      transport,
      listingDate: async () => null,
      persist: Boolean(stopAfter),
      schedulerOptions: { minIntervalMs: 0, pollIntervalMs: 1, sleep: async () => {}, maxInFlight: 16 },
    };
    engine = new T.Nf(config, callbacks, deps);
    const started = performance.now();
    let leaderboard = await engine.start();
    let backtestsBeforeStop = null;
    if (stopAfter) {
      backtestsBeforeStop = engine.backtestsRun;
      logs.push(`[test] stopped after ${backtestsBeforeStop} backtests in phase ${engine.phase}; resuming`);
      const resumed = T.Nf.fromSavedState(callbacks, deps);
      if (!resumed) {
        throw new Error('no saved state to resume');
      }
      leaderboard = await resumed.resume();
      engine = { backtestsRun: backtestsBeforeStop + resumed.backtestsRun, archive: resumed.archive };
    }
    const describe = (candidate) => {
      if (!candidate) {
        return null;
      }
      const final = candidate.validation?.portfolio;
      return {
        key: candidate.key.slice(0, 6),
        isBase: candidate.isBase,
        searchScore: candidate.fitness.score,
        validatedScore: candidate.validatedScore,
        verdict: candidate.robustness?.verdict ?? null,
        reasons: candidate.robustness?.reasons ?? null,
        final: final
          ? {
              monthly: final.monthlyPercent,
              drawdown: final.drawdownPercent,
              profit: final.adjustedProfit,
              scale: final.depositScale,
              deals: candidate.validation.totalDeals,
            }
          : null,
      };
    };
    return {
      ms: Math.round(performance.now() - started),
      backtests: engine.backtestsRun,
      evaluated: engine.archive.size,
      top: leaderboard.top.slice(0, 6).map(describe),
      base: describe(leaderboard.base),
      sensitivity: (leaderboard.sensitivity ?? []).slice(0, 8).map((row) => `${row.label}: ${row.delta}`),
      history: leaderboard.history,
      logs,
      leaderboard,
      config,
    };
  },
  {
    method: args.method ?? 'pattern',
    budget: args.budget ? Number(args.budget) : null,
    coins: COINS,
    robustness: args.robustness !== 'false',
    months: args.months !== undefined ? Number(args.months) : null,
    screen: args.screen !== undefined ? Number(args.screen) : null,
    depositChange: args.deposit !== undefined ? Number(args.deposit) : null,
    folds: args.folds !== undefined ? args.folds !== 'false' : null,
    stopAfter: args.stopAfter ? Number(args.stopAfter) : null,
    mode: args.mode ?? 'portfolio',
  },
);

await browser.close();
server.close();
if (args.out) {
  writeFileSync(args.out, JSON.stringify(summary, null, 2));
}
const { leaderboard: _leaderboard, config: _config, logs, ...brief } = summary;
if (args.logs !== 'false') {
  console.log(logs.join('\n'));
}
console.log(JSON.stringify(brief, null, 2));
if (errors.length > 0) {
  console.log('page errors:', errors.slice(0, 5));
}
