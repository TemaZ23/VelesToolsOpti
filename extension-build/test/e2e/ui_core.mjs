#!/usr/bin/env node
// UI walk-through of the 2.19 optimizer (one flow: bot → coins → best stable profit / МПУ) against
// the simulated veles.finance: coin sources, period and held-out part, what to tune, search depth
// (full enumeration only when the grid is small), a run through the real UI, run history and the
// memory file. Saves screenshots.
//
//   node extension-build/build.mjs --test
//   node extension-build/test/e2e/ui_core.mjs --shots=/tmp/shots

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


const failures = [];
const check = (condition, message) => {
  if (!condition) {
    failures.push(message);
    console.log('FAIL', message);
  }
};
const text = () => page.evaluate(() => document.body.innerText);

await page.goto(`${base}/ui.html#/optimizer/scalper`);
const findButton = page.getByRole('button', { name: /Найти лучшие настройки/ });
await findButton.waitFor({ timeout: 30000 });
await page.waitForFunction(() => /\d+ монет[аы]?: /.test(document.body.innerText), null, { timeout: 30000 });
await page.waitForTimeout(1500);
let page0 = await text();
check(!/Скальпер|Шорт-бот|Подбор монет|LONG DCA с нуля|Ручная настройка/.test(page0), 'no mode tabs or manual settings');
check((await page.evaluate(() => location.hash)) === '#/optimizer', `old tab URLs lead to the optimizer (${await page.evaluate(() => location.hash)})`);
check(page0.includes('Отложить для проверки') && /Поиск: .* Проверка на .*/.test(page0), 'held-out part of the period is shown');
const botCoins = Number((page0.match(/(\d+) монет[аы]?: /) ?? [])[1]);
check(botCoins > 1, `bot coins are filled (${botCoins})`);
await shot('01-form');

// Coin sources: only your list (one coin), then the bot's coins plus your list.
await page.getByText('Только свои', { exact: true }).click();
await page.getByPlaceholder(/BTC ETH SOL/).fill('BTC');
await page.waitForTimeout(500);
page0 = await text();
check(/1 монета: BTC/.test(page0), 'own list with one coin');
await page.getByText('Бота + свои', { exact: true }).click();
await page.waitForTimeout(500);
page0 = await text();
check(new RegExp(`${botCoins + 1} монет`).test(page0), `bot coins plus own list (${botCoins + 1})`);
await shot('02-coins');

// Depth: full enumeration is refused for the default space, allowed for two parameters.
const gridOption = page.locator('.ant-segmented-item').filter({ hasText: 'Перебор всего' });
check((await gridOption.getAttribute('class')).includes('disabled'), 'full enumeration disabled for a large space');
for (const label of ['Отступы ордеров сетки', 'Объёмы ордеров (сумма сохраняется)', 'Пороги фильтров входа', 'Пороги фильтров в ордерах сетки', 'Пороги фильтров тейка и стопа', 'Заменять фильтры похожими индикаторами', 'Пробовать выключать фильтры', 'Стоп-лосс']) {
  const box = page.getByText(label, { exact: true });
  if (await box.count()) {
    await box.first().click({ timeout: 2000 }).catch(() => {}); // a disabled item (no such setting in the bot) stays off
  }
}
await page.waitForTimeout(600);
page0 = await text();
const combos = Number(((page0.match(/комбинаций: ([\d\s ]+)/) ?? [])[1] ?? '').replace(/\D/g, ''));
check(combos > 0 && combos <= 20000, `small space after narrowing (${combos} combinations)`);
check(!(await gridOption.getAttribute('class')).includes('disabled'), 'full enumeration allowed for a small space');
await gridOption.click();
await page.waitForTimeout(400);
check(/все [\d\s ]+ комбинаций/.test(await text()), 'full enumeration shows its size');
await page.locator('.ant-segmented-item').filter({ hasText: 'Максимум' }).click();
await page.waitForTimeout(400);
check((await text()).includes('сертификат'), 'maximum depth includes the certificate');
await page.locator('.ant-segmented-item').filter({ hasText: 'Стандарт' }).click();
await shot('03-depth');

// Run on the bot's coins.
await page.locator('.ant-segmented-item').filter({ hasText: /^Бота \(/ }).first().click();
await page.waitForTimeout(400);
await findButton.click();
const dialog = page.getByRole('dialog');
await dialog.waitFor({ timeout: 10000 });
const modal = await dialog.innerText();
check(modal.includes('без последних 25% периода') && modal.includes('отложенный период'), 'start modal describes the held-out part');
check(!/стресс-тест/i.test(modal), 'no stress-test option');
await shot('04-start');
await dialog.getByRole('button', { name: /Запустить/ }).click();
const started = Date.now();
await page.waitForFunction(() => document.body.innerText.includes('🧾 Запущено бэктестов'), null, { timeout: 1800000 });
console.log(`run took ${Math.round((Date.now() - started) / 1000)} s`);
await page.waitForTimeout(2000);
await page.setViewportSize({ width: 1440, height: 3400 });
await shot('05-results');
await page.setViewportSize({ width: 1440, height: 1000 });
const result = await text();
writeFileSync(join(shots, 'result.txt'), result);
check(result.includes('🔒 Отложенный период'), 'log announces the held-out period');
check(result.includes('Цель — прибыль / МПУ'), 'log states the profit / МПУ goal');
check(result.includes('🏆 Лучший вариант'), 'best variant card');
check(/Отложенный период .* \(поиск его не видел\)/.test(result), 'robustness checks the held-out period');
check(!result.includes('Стресс:'), 'no stress test in the run');

// Run history and the memory file.
await page.getByRole('button', { name: 'История' }).click();
const history = page.getByRole('dialog', { name: 'История запусков' });
await history.waitFor({ timeout: 10000 });
await page.waitForTimeout(600);
check((await history.innerText()).includes('Показать'), 'the finished run is in the history');
await shot('06-history');
await page.keyboard.press('Escape');
await page.waitForTimeout(400);
const [download] = await Promise.all([page.waitForEvent('download', { timeout: 20000 }), page.getByRole('button', { name: 'Сохранить' }).click()]);
const file = join(shots, download.suggestedFilename());
await download.saveAs(file);
const bytes = readFileSync(file);
const { gunzipSync } = await import('node:zlib');
const lines = (bytes[0] === 0x1f && bytes[1] === 0x8b ? gunzipSync(bytes) : bytes)
  .toString('utf8')
  .split('\n')
  .filter(Boolean)
  .map((line) => JSON.parse(line));
check(lines[0]?.format === 'veles-optimizer-memory' && lines.filter((line) => line.r).length > 0, 'memory file has backtests');
const [chooser] = await Promise.all([page.waitForEvent('filechooser', { timeout: 10000 }), page.getByRole('button', { name: 'Загрузить' }).click()]);
await chooser.setFiles(file);
const loaded = await page
  .waitForFunction(() => document.body.innerText.includes('Память загружена'), null, { timeout: 20000, polling: 20 })
  .then(() => true)
  .catch(() => false);
check(loaded, 'memory file loads back');

// Render errors fail the run; a missing favicon (404) does not.
const fatal = errors.filter((error) => !error.includes('status of 404'));
check(fatal.length === 0, `no page errors (${fatal.slice(0, 3).join(' | ').slice(0, 400)})`);

await browser.close();
server.close();
console.log('errors:', errors.length > 0 ? errors.slice(0, 10) : 'none');
console.log(failures.length === 0 ? 'UI CORE OK' : `UI CORE FAILED: ${failures.length}`);
process.exit(failures.length === 0 ? 0 : 1);
