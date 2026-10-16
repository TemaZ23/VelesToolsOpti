// Test page only: pretends to be the extension runtime and veles.finance (via the proxy-request
// messages the UI sends), answering backtests with test/sim/market.js. Also speeds up timers so
// the optimizer's request pacing does not slow the test down.
(() => {
  try {
    localStorage.setItem('__VELES_TG_CHANNEL_SHOWN', 'shown');
  } catch {}
  const realSetTimeout = window.setTimeout.bind(window);
  window.setTimeout = (handler, delay, ...rest) =>
    realSetTimeout(handler, typeof delay === 'number' && delay > 20 ? Math.max(1, delay / 400) : delay, ...rest);

  const COINS =
    'ONDO NEO LDO KAS JASMY ICP GALA ENS CVX CRV ADA JTO XTZ DOT LINK UNI AAVE ATOM AVAX NEAR FIL ETC XLM VET ALGO SAND MANA AXS APE OP ARB INJ SUI SEI TIA DYDX GRT SNX COMP ZEC'.split(' ');
  const BOT_ID = 4242;
  const bots = () => {
    const strategy = window.VelesSim.makeStrategy();
    const chunks = [COINS.slice(0, 14), COINS.slice(14, 28), COINS.slice(28)];
    return chunks.map((coins, index) => ({
      ...strategy,
      id: BOT_ID + index,
      name: `${strategy.name} ${index + 1}`,
      status: 'STOPPED',
      apiKey: 1,
      symbols: coins.map((coin) => `${coin}/USDT`),
    }));
  };
  const backtests = new Map();
  let nextId = 100000;
  const ok = (body) => ({ ok: true, status: 200, statusText: 'OK', body });
  const notFound = (url) => ({ ok: false, status: 404, statusText: 'Not Found', body: null, error: `mock: ${url}` });

  const proxy = ({ url, init }) => {
    const parsed = new URL(url);
    const path = parsed.pathname.replace(/\/+$/, '');
    const method = (init?.method ?? 'GET').toUpperCase();
    let match;
    if (path === '/api/bots' && method === 'PUT') {
      const list = bots();
      return ok({ content: list, totalPages: 1, totalElements: list.length, number: 0, size: 100 });
    }
    if ((match = /^\/api\/bots\/(\d+)$/.exec(path))) {
      return ok(bots().find((bot) => bot.id === Number(match[1])) ?? bots()[0]);
    }
    if (path === '/api/backtests/limits') {
      return ok({ tariff: 'PRO', counter: 365 * 22756, threads: 4 });
    }
    if (path === '/api/backtests/tariffs') {
      return ok([{ key: 'PRO', name: 'Pro', dailyAccrualYears: 40, maxAccumulationYears: 30000 }]);
    }
    if (path === '/api/pairs/availability/dictionary') {
      return ok({ payload: [] });
    }
    if (path === '/api/backtests/v2' && method === 'POST') {
      const id = nextId++;
      backtests.set(id, window.VelesSim.backtest(JSON.parse(init.body)));
      return ok({ id });
    }
    if (path === '/api/backtests/statistics' && method === 'GET') {
      return ok({ content: [], totalPages: 0, totalElements: 0 });
    }
    if ((match = /^\/api\/backtests\/statistics\/(\d+)$/.exec(path))) {
      const result = backtests.get(Number(match[1]));
      return result ? ok({ id: Number(match[1]), ...result.stats }) : notFound(url);
    }
    if ((match = /^\/api\/backtests\/statistics\/(\d+)\/cycles$/.exec(path))) {
      const result = backtests.get(Number(match[1]));
      if (!result) {
        return notFound(url);
      }
      const page = Number(parsed.searchParams.get('page') ?? 0);
      const size = Number(parsed.searchParams.get('size') ?? 200);
      const sorted = [...result.cycles].sort((a, b) => Date.parse(b.date) - Date.parse(a.date));
      return ok({
        content: sorted.slice(page * size, (page + 1) * size),
        totalPages: Math.max(1, Math.ceil(sorted.length / size)),
        totalElements: sorted.length,
      });
    }
    if ((match = /^\/api\/backtests\/(\d+)$/.exec(path))) {
      return backtests.has(Number(match[1])) ? ok({ id: Number(match[1]), status: 'FINISHED' }) : notFound(url);
    }
    return notFound(url);
  };

  const handle = (message) => {
    switch (message?.action) {
      case 'proxy-request':
        return proxy(message.payload);
      case 'active-veles-tab':
        return { tabs: [{ id: 1, url: `https://veles.finance/cabinet/bot/${BOT_ID}` }] };
      case 'ping':
      case 'connection-status':
        return { ok: true, connected: true, origin: 'https://veles.finance', lastChecked: Date.now() };
      default:
        return { ok: true };
    }
  };

  window.chrome = {
    runtime: {
      id: 'veles-tools-test',
      lastError: undefined,
      getManifest: () => ({ version: 'test' }),
      sendMessage: (message, callback) => {
        Promise.resolve()
          .then(() => handle(message))
          .then((response) => callback?.(response));
      },
      onMessage: { addListener: () => {}, removeListener: () => {} },
    },
    tabs: { create: () => {} },
  };
})();
