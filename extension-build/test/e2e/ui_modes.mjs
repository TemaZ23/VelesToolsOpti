#!/usr/bin/env node
// UI walk-through of the optimizer mode tabs (2.16) against the simulated veles.finance: runs the
// scalper tab and the coin-selection tab through the real UI, checks that only one run goes at a
// time across tabs, and saves screenshots.
//
//   node extension-build/build.mjs --test
//   node extension-build/test/e2e/ui_modes.mjs --shots=/tmp/shots [--tabs=scalper,coins]

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


const tabs = (args.tabs ?? 'scalper,coins').split(',').filter(Boolean);
const buttons = { scalper: 'Найти скальпер', short: 'Найти шорт-бот', long: 'Собрать LONG-бота', coins: 'Подобрать монеты' };
const done = { scalper: '🧾 Запущено бэктестов', short: '🧾 Запущено бэктестов', long: '🧾 Запущено бэктестов', coins: '🪙 Подобранные монеты' };
const failures = [];
const check = (condition, message) => {
  if (!condition) {
    failures.push(message);
    console.log('FAIL', message);
  }
};
const visibleText = () => page.evaluate(() => document.querySelector('.ant-tabs-tabpane-active')?.innerText ?? '');

await page.goto(`${base}/ui.html#/optimizer`);
await page.getByRole('tab', { name: /Скальпер/ }).waitFor({ timeout: 30000 });
await shot('00-tabs');

let index = 0;
for (const tab of tabs) {
  index += 1;
  const label = { scalper: /Скальпер/, short: /Шорт-бот/, long: /LONG DCA/, coins: /Подбор монет/ }[tab];
  await page.getByRole('tab', { name: label }).click();
  await page.waitForTimeout(500);
  check((await page.evaluate(() => location.hash)).endsWith(`/optimizer/${tab}`), `${tab}: hash follows the tab`);
  const pane = page.locator('.ant-tabs-tabpane-active');
  const start = pane.getByRole('button', { name: buttons[tab] });
  await start.waitFor({ timeout: 30000 });
  await page.waitForFunction(() => /Монет в прогоне: \d+/.test(document.querySelector('.ant-tabs-tabpane-active')?.innerText ?? ''), null, { timeout: 30000 });
  await page.waitForTimeout(1500);
  const before = await visibleText();
  check(!/Монет в прогоне: 0\b/.test(before), `${tab}: coins are filled`);
  if (tab !== 'coins') {
    check(before.includes('🧩 Шаблон:'), `${tab}: template preview shown`);
  }
  await shot(`${index}1-${tab}`);
  await start.click();
  const dialog = page.getByRole('dialog');
  await dialog.waitFor({ timeout: 10000 });
  await shot(`${index}2-${tab}-modal`);
  const modalText = await dialog.innerText();
  check(tab !== 'coins' || !modalText.includes('стресс-тест'), `${tab}: no stress option for coins`);
  await dialog.getByRole('button', { name: /Запустить/ }).click();
  await page.waitForTimeout(1500);
  if (index === 1) {
    // Another tab must refuse to start while this run goes on.
    const other = tabs.length > 1 ? tabs[1] : tab === 'coins' ? 'scalper' : 'coins';
    await page.getByRole('tab', { name: { scalper: /Скальпер/, short: /Шорт-бот/, long: /LONG DCA/, coins: /Подбор монет/ }[other] }).click();
    await page.waitForTimeout(800);
    const otherButton = page.locator('.ant-tabs-tabpane-active').getByRole('button', { name: buttons[other] });
    await otherButton.waitFor({ timeout: 30000 });
    await page.waitForTimeout(1500);
    await otherButton.click();
    // The simulator speeds up timers, so the toast lives only a fraction of a second: poll right away.
    const warned = await page
      .waitForFunction(() => document.body.innerText.includes('Идёт прогон во вкладке'), null, { timeout: 3000, polling: 20 })
      .then(() => true)
      .catch(() => false);
    await page.waitForTimeout(500);
    check(warned, 'second tab refuses to start while a run goes on');
    check((await page.getByRole('dialog').count()) === 0 || !(await page.getByRole('dialog').isVisible()), 'no start modal in the second tab');
    const runningTab = await page.evaluate(() => [...document.querySelectorAll('.ant-tabs-tab')].map((node) => node.innerText).find((text) => text.includes('⏳')) ?? '');
    check(runningTab.length > 0, 'running tab is marked');
    await page.getByRole('tab', { name: label }).click();
    await page.waitForTimeout(500);
  }
  const started = Date.now();
  await page.waitForFunction((text) => (document.querySelector('.ant-tabs-tabpane-active')?.innerText ?? '').includes(text), done[tab], { timeout: 1800000 });
  console.log(`${tab}: run took ${Math.round((Date.now() - started) / 1000)} s`);
  await page.waitForTimeout(2000);
  await page.setViewportSize({ width: 1440, height: 3200 });
  await shot(`${index}3-${tab}-results`);
  await page.setViewportSize({ width: 1440, height: 1000 });
  const after = await visibleText();
  writeFileSync(join(shots, `${tab}.txt`), after);
  if (tab === 'coins') {
    check(after.includes('Создать бота с этими монетами'), 'coins: result card with create button');
    await pane.getByRole('button', { name: 'Создать бота с этими монетами' }).click();
    await page.waitForTimeout(800);
    await shot(`${index}4-${tab}-create-bot`);
    await page.keyboard.press('Escape');
    await page.waitForTimeout(400);
  } else {
    check(after.includes('🏆 Лучший вариант'), `${tab}: best variant card`);
    check(!after.includes('Исходная на всех монетах'), `${tab}: template wording`);
  }
}

// Tabs keep their own results: the improve tab is untouched by the mode runs.
await page.getByRole('tab', { name: /Улучшить стратегию/ }).click();
await page.waitForTimeout(800);
const improveText = await visibleText();
check(improveText.includes('Найти устойчивые настройки'), 'improve tab still shows its own card');
check(!improveText.includes('🪙 Подобранные монеты'), 'improve tab has no coin results');

// Render errors (a crashed tab blanks the page) fail the run; a missing favicon (404) does not.
const fatal = errors.filter((error) => !error.includes('status of 404'));
check(fatal.length === 0, `no page errors (${fatal.slice(0, 3).join(' | ').slice(0, 400)})`);

await browser.close();
server.close();
console.log('errors:', errors.length > 0 ? errors.slice(0, 10) : 'none');
console.log(failures.length === 0 ? 'UI MODES OK' : `UI MODES FAILED: ${failures.length}`);
process.exit(failures.length === 0 ? 0 : 1);
