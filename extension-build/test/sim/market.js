// Deterministic synthetic market and DCA grid backtester (LONG and SHORT, optional stop-loss) — test only.
// Lets the optimizer run end to end offline: the fake transport in test/e2e answers backtest
// requests with deals simulated here from the strategy payload the optimizer submits.
(() => {
  const BAR_MS = 4 * 3600 * 1000;
  const YEAR_MS = 365.25 * 24 * 3600 * 1000;
  const START = Date.parse('2019-12-01T00:00:00Z');
  const END = Date.parse('2026-12-31T00:00:00Z');
  const BARS = Math.floor((END - START) / BAR_MS);

  const hash = (text) => {
    let h = 2166136261;
    for (let i = 0; i < text.length; i++) {
      h ^= text.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return h >>> 0;
  };
  const makeRng = (seed) => {
    let state = seed >>> 0;
    return () => {
      state = (state + 0x6d2b79f5) >>> 0;
      let t = state;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };
  const gauss = (rng) => {
    const u = Math.max(rng(), 1e-12);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
  };

  // Market regimes (annualised drift of the "market factor").
  const REGIMES = [
    ['2019-12-01', 0.3],
    ['2020-02-20', -2.5],
    ['2020-04-01', 1.4],
    ['2021-05-10', -2.0],
    ['2021-07-20', 1.2],
    ['2021-11-15', -1.1],
    ['2022-05-05', -3.0],
    ['2022-06-20', -0.6],
    ['2023-01-01', 0.5],
    ['2023-08-01', -0.3],
    ['2023-10-15', 1.3],
    ['2024-04-01', -0.5],
    ['2024-10-01', 1.0],
    ['2025-02-01', -1.0],
    ['2025-05-01', 0.4],
    ['2025-10-10', -1.5],
    ['2025-12-01', 0.1],
  ].map(([date, drift]) => [Date.parse(`${date}T00:00:00Z`), drift]);
  const regimeDrift = (time) => {
    let drift = REGIMES[0][1];
    for (const [from, value] of REGIMES) {
      if (time >= from) {
        drift = value;
      }
    }
    return drift;
  };

  const seriesCache = new Map();
  const getSeries = (symbol) => {
    const cached = seriesCache.get(symbol);
    if (cached) {
      return cached;
    }
    const rng = makeRng(hash(symbol));
    const vol = 0.55 + rng() * 0.9;
    const beta = 0.6 + rng() * 0.9;
    const idio = (rng() - 0.5) * 0.5;
    const close = new Float64Array(BARS);
    const high = new Float64Array(BARS);
    const low = new Float64Array(BARS);
    const sigma = vol * Math.sqrt(BAR_MS / YEAR_MS);
    let price = 0.5 + rng() * 40;
    let crash = 0;
    for (let i = 0; i < BARS; i++) {
      const time = START + i * BAR_MS;
      if (crash === 0 && rng() < 0.0006) {
        crash = 4 + Math.floor(rng() * 6);
      }
      let ret = ((beta * regimeDrift(time) + idio) * BAR_MS) / YEAR_MS + sigma * gauss(rng);
      if (crash > 0) {
        ret -= 0.03 + rng() * 0.03;
        crash -= 1;
      }
      const open = price;
      price = Math.max(1e-6, price * Math.exp(ret));
      high[i] = Math.max(open, price) * Math.exp(Math.abs(gauss(rng)) * sigma * 0.5);
      low[i] = Math.min(open, price) * Math.exp(-Math.abs(gauss(rng)) * sigma * 0.6);
      close[i] = price;
    }
    // RSI(14) on closes.
    const rsi = new Float64Array(BARS).fill(50);
    let gain = 0;
    let loss = 0;
    for (let i = 1; i < BARS; i++) {
      const change = close[i] - close[i - 1];
      const up = Math.max(change, 0);
      const down = Math.max(-change, 0);
      if (i <= 14) {
        gain += up / 14;
        loss += down / 14;
      } else {
        gain = (gain * 13 + up) / 14;
        loss = (loss * 13 + down) / 14;
      }
      rsi[i] = loss < 1e-12 ? 100 : 100 - 100 / (1 + gain / loss);
    }
    const series = { close, high, low, rsi };
    seriesCache.set(symbol, series);
    return series;
  };

  const passes = (value, operation, threshold) => {
    switch (operation) {
      case 'LESS':
      case 'LESS_OR_EQUAL':
        return value <= threshold;
      case 'GREATER':
      case 'GREATER_OR_EQUAL':
        return value >= threshold;
      default:
        return true;
    }
  };

  /** Runs a DCA grid (LONG: buys below, SHORT: sells above) over the payload's symbol and period. */
  const backtest = (payload) => {
    const short = payload.algorithm === 'SHORT';
    const symbol = payload.symbol ?? payload.symbols?.[0];
    const series = getSeries(symbol);
    const from = Date.parse(payload.from);
    const to = Date.parse(payload.to);
    const deposit = payload.deposit?.amount ?? 10;
    const leverage = payload.deposit?.leverage ?? 1;
    const notional = deposit * leverage;
    const settings = payload.settings ?? {};
    const orders = [settings.baseOrder, ...(settings.orders ?? [])].filter(Boolean);
    const tp = payload.profit?.checkPnl ?? 1;
    const pullUp = typeof payload.pullUp === 'number' ? payload.pullUp : null;
    const stop = typeof payload.stopLoss?.indent === 'number' && payload.stopLoss.indent > 0 ? payload.stopLoss.indent : null;
    const entry = (payload.conditions ?? []).find((condition) => condition.indicator === 'RSI');
    const fee = 0.0004;
    const first = Math.max(15, Math.ceil((from - START) / BAR_MS));
    const last = Math.min(BARS - 1, Math.floor((to - START) / BAR_MS));
    const cycles = [];
    let deal = null;
    let cooldown = 0;
    for (let i = first; i <= last; i++) {
      const time = START + i * BAR_MS;
      if (!deal) {
        if (cooldown > 0) {
          cooldown -= 1;
          continue;
        }
        if (entry && !passes(series.rsi[i], entry.operation, entry.value)) {
          continue;
        }
        const price = series.close[i];
        const volume = ((orders[0]?.volume ?? 10) / 100) * notional;
        deal = {
          start: time,
          anchor: price,
          qty: volume / price,
          cost: volume,
          fees: volume * fee,
          filled: 1,
          mae: 0,
          maePercent: 0,
          executions: [],
        };
        continue;
      }
      if (pullUp !== null && deal.filled === 1) {
        deal.anchor = short
          ? Math.min(deal.anchor, series.close[i] * (1 + pullUp / 100))
          : Math.max(deal.anchor, series.close[i] / (1 + pullUp / 100));
      }
      while (deal.filled < orders.length) {
        const order = orders[deal.filled];
        const level = deal.anchor * (short ? 1 + (order.indent ?? 0) / 100 : 1 - (order.indent ?? 0) / 100);
        if (short ? series.high[i] < level : series.low[i] > level) {
          break;
        }
        const volume = ((order.volume ?? 0) / 100) * notional;
        deal.qty += volume / level;
        deal.cost += volume;
        deal.fees += volume * fee;
        deal.executions.push({ position: deal.filled, at: time });
        deal.filled += 1;
      }
      const average = deal.cost / deal.qty;
      const worst = short ? series.high[i] : series.low[i];
      const adverse = deal.qty * (short ? worst - average : average - worst);
      deal.mae = Math.max(deal.mae, adverse);
      deal.maePercent = Math.max(deal.maePercent, ((short ? worst - average : average - worst) / average) * 100);
      const stopLevel = stop === null ? null : average * (short ? 1 + stop / 100 : 1 - stop / 100);
      if (stopLevel !== null && (short ? series.high[i] >= stopLevel : series.low[i] <= stopLevel)) {
        const gross = deal.qty * (short ? average - stopLevel : stopLevel - average);
        const fees = deal.fees + deal.qty * stopLevel * fee;
        cycles.push({ ...deal, end: time, status: 'FINISHED', gross, fees, net: gross - fees });
        deal = null;
        cooldown = 1;
        continue;
      }
      const target = average * (short ? 1 - tp / 100 : 1 + tp / 100);
      if (short ? series.low[i] <= target : series.high[i] >= target) {
        const gross = deal.qty * (short ? average - target : target - average);
        const fees = deal.fees + deal.qty * target * fee;
        cycles.push({ ...deal, end: time, status: 'FINISHED', gross, fees, net: gross - fees });
        deal = null;
        cooldown = 1;
      }
    }
    if (deal) {
      const average = deal.cost / deal.qty;
      const current = deal.qty * Math.max(0, short ? series.close[last] - average : average - series.close[last]);
      cycles.push({ ...deal, end: Math.min(to, START + last * BAR_MS), status: 'STARTED', gross: 0, fees: 0, net: 0, active: current });
    }
    const finished = cycles.filter((cycle) => cycle.status === 'FINISHED');
    const open = cycles.find((cycle) => cycle.status === 'STARTED');
    const profit = finished.reduce((sum, cycle) => sum + cycle.gross, 0);
    const commission = finished.reduce((sum, cycle) => sum + cycle.fees, 0);
    const wins = finished.filter((cycle) => cycle.net >= 0).length;
    const maxMae = cycles.reduce((max, cycle) => Math.max(max, cycle.mae), 0);
    const stats = {
      totalDeals: cycles.length,
      profits: wins,
      losses: finished.length - wins,
      breakevens: 0,
      winRateProfits: wins,
      winRateLosses: finished.length - wins,
      profitQuote: profit,
      commissionQuote: commission,
      maeAbsolute: maxMae,
      maePercent: cycles.reduce((max, cycle) => Math.max(max, cycle.maePercent), 0),
      maeDepositPercent: (maxMae / Math.max(deposit, 1e-9)) * 100,
      profitFactor: null,
      avgDuration:
        finished.length > 0 ? finished.reduce((sum, cycle) => sum + (cycle.end - cycle.start) / 1000, 0) / finished.length : 0,
      activeMaeAbsolute: open ? open.mae : 0,
    };
    const apiCycles = cycles.map((cycle) => ({
      date: new Date(cycle.end).toISOString(),
      createdAt: new Date(cycle.start).toISOString(),
      status: cycle.status,
      netQuote: cycle.net,
      maeAbsolute: cycle.mae,
      orders: cycle.executions.map((execution) => ({
        category: 'GRID',
        side: short ? 'SELL' : 'BUY',
        status: 'EXECUTED',
        position: execution.position,
        executedAt: new Date(execution.at).toISOString(),
        createdAt: new Date(execution.at).toISOString(),
      })),
    }));
    return { stats, cycles: apiCycles };
  };

  /** A strategy shaped like a Veles LONG grid bot. */
  const makeStrategy = () => ({
    id: 4242,
    name: 'LONG TEST GRID',
    exchange: 'BYBIT_FUTURES',
    algorithm: 'LONG',
    symbols: ['ADA/USDT'],
    pair: { exchange: 'BYBIT_FUTURES', type: 'FUTURES', from: 'ADA', to: 'USDT', symbol: 'ADAUSDT' },
    deposit: { amount: 10.5, leverage: 20, marginType: 'CROSS', currency: 'USDT' },
    conditions: [
      {
        type: 'INDICATOR',
        indicator: 'RSI',
        interval: 'FOUR_HOUR',
        basic: false,
        value: 30,
        operation: 'LESS',
        closed: true,
        reverse: false,
      },
    ],
    settings: {
      type: 'CUSTOM',
      indentType: 'ENTRY',
      includePosition: true,
      baseOrder: { indent: 0, volume: 6, conditions: [] },
      orders: [
        { indent: 1.5, volume: 6, conditions: [] },
        { indent: 3, volume: 8, conditions: [] },
        { indent: 5, volume: 10, conditions: [] },
        { indent: 7.5, volume: 12, conditions: [] },
        { indent: 10.5, volume: 15, conditions: [] },
        { indent: 14, volume: 19, conditions: [] },
        { indent: 18, volume: 24, conditions: [] },
      ],
    },
    profit: { type: 'PERCENT', currency: 'USDT', checkPnl: 0.5, conditions: [] },
    stopLoss: null,
    pullUp: 0.4,
    portion: null,
    termination: null,
  });

  window.VelesSim = { backtest, getSeries, makeStrategy };
})();
