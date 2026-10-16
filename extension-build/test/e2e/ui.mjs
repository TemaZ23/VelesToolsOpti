#!/usr/bin/env node
// UI walk-through of the optimizer page against the simulated veles.finance (fake-chrome.js):
// opens the page, runs "Найти устойчивые настройки" through the real UI and saves screenshots.
//
//   node extension-build/build.mjs --test
//   node extension-build/test/e2e/ui.mjs --shots=/tmp/shots [--theme=dark]

import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
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
const shots = resolve(args.shots ?? join(testDist, 'shots'));
mkdirSync(shots, { recursive: true });

const css = readdirSync(join(testDist, 'assets')).find((name) => name.endsWith('.css'));
for (const name of readdirSync(join(testDist, 'assets'))) {
  copyFileSync(join(testDist, 'assets', name), join(testDist, name));
}
writeFileSync(
  join(testDist, 'ui.html'),
  `<!doctype html><html lang="ru"><head><meta charset="utf-8"><link rel="stylesheet" href="./assets/${css}"></head>` +
    '<body><div id="root"></div><script src="./fake-chrome.js"></script><script src="./market.js"></script>' +
    '<script type="module" src="./app.js"></script></body></html>',
);
writeFileSync(join(testDist, 'market.js'), readFileSync(join(root, 'test', 'sim', 'market.js')));
writeFileSync(join(testDist, 'fake-chrome.js'), readFileSync(join(here, 'fake-chrome.js')));

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
const base = `http://127.0.0.1:${server.address().port}`;

const playwrightPath = process.env.PLAYWRIGHT_CORE ?? 'playwright-core';
const { chromium } = await import(playwrightPath.startsWith('/') ? pathToFileURL(playwrightPath).href : playwrightPath);
const chromiumPath =
  process.env.CHROMIUM ??
  readdirSync('/opt/pw-browsers')
    .filter((name) => /^chromium-\d+$/.test(name))
    .map((name) => `/opt/pw-browsers/${name}/chrome-linux/chrome`)[0];
const browser = await chromium.launch({ executablePath: chromiumPath, args: ['--no-sandbox'] });
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
const errors = [];
page.on('pageerror', (error) => errors.push(String(error)));
page.on('console', (message) => {
  if (message.type() === 'error') {
    errors.push(`console: ${message.text()}`);
  }
});
if (args.theme) {
  await page.addInitScript((theme) => localStorage.setItem('veles-theme-mode', theme), args.theme);
}
const shot = async (name, options = {}) => {
  await page.waitForTimeout(400);
  await page.screenshot({ path: join(shots, `${name}.png`), ...options });
  console.log('shot', name);
};

await page.goto(`${base}/ui.html#/optimizer`);
const findButton = page.getByRole('button', { name: /Найти устойчивые настройки/ });
await findButton.waitFor({ timeout: 30000 });
await page.waitForFunction(() => document.body.innerText.includes('Монеты стратегии:'), null, { timeout: 30000 });
await page.waitForTimeout(1500);
await shot('01-simple');

await findButton.click();
await page.getByRole('dialog').waitFor({ timeout: 10000 });
await shot('02-start-modal');
if (args.months) {
  await page.getByRole('dialog').getByText(args.months === '0' ? 'весь период' : `${Number(args.months) / 12} год`).click();
}
await page.getByRole('dialog').getByRole('button', { name: /Запустить/ }).click();

const started = Date.now();
await page.waitForFunction(() => document.body.innerText.includes('🔬 Влияние параметров'), null, { timeout: 600000 });
await page.setViewportSize({ width: 1440, height: 2400 });
await shot('03-running');
await page.setViewportSize({ width: 1440, height: 1000 });
await page.waitForFunction(() => document.body.innerText.includes('🧾 Запущено бэктестов'), null, { timeout: 1800000 });
console.log(`run took ${Math.round((Date.now() - started) / 1000)} s`);
await page.waitForTimeout(1500);
await page.setViewportSize({ width: 1440, height: 3400 });
await shot('04-results');
await page.setViewportSize({ width: 1440, height: 1000 });

await page.getByRole('button', { name: 'Подробнее' }).first().click();
const dialog = page.getByRole('dialog');
await dialog.waitFor({ timeout: 10000 });
await shot('05-details');
const foldsTab = dialog.getByRole('tab', { name: /Срезы поиска/ });
if (await foldsTab.count()) {
  await foldsTab.click();
  await shot('06-details-folds');
}
await dialog.getByRole('button', { name: /Создать бота/ }).click();
await page.waitForTimeout(800);
await shot('07-create-bot');
await page.keyboard.press('Escape');
await page.waitForTimeout(300);
await page.keyboard.press('Escape');
await page.waitForTimeout(500);

await page.getByText('Ручная настройка').first().click();
await page.waitForTimeout(800);
await page.setViewportSize({ width: 1440, height: 3000 });
await shot('08-manual');

const logText = await page.evaluate(() => {
  const nodes = [...document.querySelectorAll('div')].filter((node) => node.innerText?.startsWith('Лог'));
  return document.body.innerText;
});
writeFileSync(join(shots, 'page.txt'), logText);
await browser.close();
server.close();
console.log('errors:', errors.length > 0 ? errors.slice(0, 10) : 'none');
