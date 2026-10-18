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

// Template tab ranges editor: «По умолчанию» + «Сохранить» keeps the mode windows (no own overrides).
if (tabs.includes('scalper')) {
  await page.getByRole('tab', { name: /Скальпер/ }).click();
  await page.waitForTimeout(500);
  const scalperPane = page.locator('.ant-tabs-tabpane-active');
  await scalperPane.getByText('Ручная настройка').first().click();
  await page.waitForTimeout(800);
  await scalperPane.getByRole('button', { name: /Диапазоны поиска/ }).click();
  const rangesDialog = page.getByRole('dialog');
  await rangesDialog.waitFor({ timeout: 10000 });
  await rangesDialog.getByRole('button', { name: 'По умолчанию' }).click();
  await rangesDialog.getByRole('button', { name: 'Сохранить' }).click();
  await page.waitForTimeout(800);
  const stored = await page.evaluate(() => JSON.parse(localStorage.getItem('veles_optimizer_bot_ranges_v3') ?? '{}'));
  const scalperRanges = Object.entries(stored).find(([key]) => key.endsWith('__scalper'));
  check(!!scalperRanges, 'scalper ranges are stored under the tab key');
  check(scalperRanges && Object.keys(scalperRanges[1].overrides).length === 0, `reset + save keeps the mode windows (${scalperRanges ? Object.keys(scalperRanges[1].overrides).length : '—'} own rows)`);
  check(!Object.keys(stored).some((key) => /^\d+$/.test(key)), 'improve-tab ranges untouched');
  await shot('90-scalper-manual');
  await scalperPane.getByText('Устойчивые настр').first().click();
  await page.waitForTimeout(300);
}

// 2.17 memory: run history, export / import of the memory file, period end, full enumeration and
// the certificate controls.
if (args.memory !== 'false') {
  const firstTab = tabs[0];
  const firstLabel = { scalper: /Скальпер/, short: /Шорт-бот/, long: /LONG DCA/, coins: /Подбор монет/ }[firstTab];
  await page.getByRole('tab', { name: firstLabel }).click();
  await page.waitForTimeout(600);
  let pane = page.locator('.ant-tabs-tabpane-active');
  await pane.getByRole('button', { name: 'История' }).click();
  let history = page.getByRole('dialog', { name: 'История запусков' });
  await history.waitFor({ timeout: 10000 });
  await page.waitForTimeout(600);
  const historyText = await history.innerText();
  check(historyText.includes('Показать'), `${firstTab}: the finished run is in the history`);
  check(/бэктестов: \d+, из памяти: \d+/.test(historyText), `${firstTab}: history row has the summary`);
  await shot('91-history');
  await history.getByRole('button', { name: 'Показать' }).first().click();
  // Toasts live a fraction of a second under the simulator's fast timers: poll right away.
  const shown = await page
    .waitForFunction(() => document.body.innerText.includes('Показан запуск от'), null, { timeout: 5000, polling: 20 })
    .then(() => true)
    .catch(() => false);
  await page.waitForTimeout(500);
  check(shown, 'history entry can be shown');
  check(!(await history.isVisible()), 'history closes after showing an entry');

  // Export the memory (a gzip or JSON file with every backtest) and load it back.
  const [download] = await Promise.all([page.waitForEvent('download', { timeout: 20000 }), pane.getByRole('button', { name: 'Сохранить' }).click()]);
  const file = join(shots, download.suggestedFilename());
  await download.saveAs(file);
  const bytes = readFileSync(file);
  const { gunzipSync } = await import('node:zlib');
  const lines = (bytes[0] === 0x1f && bytes[1] === 0x8b ? gunzipSync(bytes) : bytes)
    .toString('utf8')
    .split('\n')
    .filter(Boolean)
    .map((line) => JSON.parse(line));
  const results = lines.filter((line) => line.r).length;
  const runs = lines.filter((line) => line.h).length;
  check(lines[0]?.format === 'veles-optimizer-memory' && results > 0, `memory file has backtests (${results})`);
  check(runs >= 1, `memory file has the run history (${runs})`);
  const [chooser] = await Promise.all([page.waitForEvent('filechooser', { timeout: 10000 }), pane.getByRole('button', { name: 'Загрузить' }).click()]);
  await chooser.setFiles(file);
  const loaded = await page
    .waitForFunction(() => document.body.innerText.includes('Память загружена'), null, { timeout: 20000, polling: 20 })
    .then(() => true)
    .catch(() => false);
  check(loaded, 'memory file loads back');

  // Improve tab, manual settings: full enumeration count, certificate switch, period end control.
  await page.getByRole('tab', { name: /Улучшить стратегию/ }).click();
  await page.waitForTimeout(800);
  pane = page.locator('.ant-tabs-tabpane-active');
  await pane.getByText('Ручная настройка').first().click();
  await page.waitForTimeout(1000);
  await pane.getByText('Перебор', { exact: true }).click();
  await page.waitForTimeout(600);
  let manual = await visibleText();
  check(/Сейчас: .*комбинац/.test(manual), 'full enumeration shows the number of combinations');
  check(manual.includes('больше предела'), 'the default space is refused as too large');
  await shot('92-grid');
  await pane.getByText('Пошаговый', { exact: true }).click();
  await page.waitForTimeout(500);
  await pane.getByText('Сертификат лучшего', { exact: true }).locator('xpath=preceding-sibling::button[1]').click();
  await page.waitForTimeout(500);
  manual = await visibleText();
  check(/\+ до [\d\s ]+ вариантов/.test(manual), 'certificate switch shows its cost');
  check(manual.includes('Конец периода') && manual.includes('конец прошлого месяца'), 'period end control is shown');
  const now = new Date();
  const monthEnd = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), 0)).toISOString().slice(0, 10);
  const dates = await pane.locator('input[type="date"]').evaluateAll((nodes) => nodes.map((node) => node.value));
  check(dates.includes(monthEnd), `rolling period ends at the previous month end (${monthEnd}; ${dates.join(', ')})`);
  await shot('93-certificate');
  await pane.getByText('Устойчивые настр').first().click();
  await page.waitForTimeout(300);
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
