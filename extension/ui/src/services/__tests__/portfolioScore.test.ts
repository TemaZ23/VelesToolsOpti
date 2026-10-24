import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { BacktestConfigDto } from '../../api/backtests.dtos';
import { MS_IN_DAY, MS_IN_HOUR } from '../../lib/dateTime';
import type { AggregatedBacktestsMetrics } from '../../types/backtestAggregations';
import type { BacktestInfo, BacktestInfoDeal } from '../../types/backtestInfos';
import type { BacktestCycle, BacktestDetail, BacktestOrder, BacktestStatistics } from '../../types/backtests';
import { aggregateBacktestsMetrics } from '../backtestAggregations';
import { buildBacktestInfo } from '../backtestInfos';
import {
  buildFoldWindows,
  computeFoldMetrics,
  computePortfolioMetrics,
  evaluateBasket,
  loadBacktestInfo,
  sanitizeCycles,
  toPortfolioMetrics,
} from '../portfolioScore';

const { getBacktestDetailMock, getBacktestCyclesMock } = vi.hoisted(() => ({
  getBacktestDetailMock: vi.fn<(id: number, options?: { ignoreCache?: boolean }) => Promise<BacktestDetail>>(),
  getBacktestCyclesMock: vi.fn<(id: number, params?: { ignoreCache?: boolean }) => Promise<BacktestCycle[]>>(),
}));

vi.mock('../backtests', () => ({
  backtestsService: {
    getBacktestDetail: getBacktestDetailMock,
    getBacktestCycles: getBacktestCyclesMock,
  },
}));

// ═══════════════════════════════════════════════════════════════════
// ФИКСТУРЫ: СДЕЛКИ И BacktestInfo
// ═══════════════════════════════════════════════════════════════════

const PERIOD_FROM = '2024-01-01T00:00:00.000Z';
const PERIOD_TO = '2024-01-31T00:00:00.000Z';
const T0 = Date.parse(PERIOD_FROM);
const T_END = Date.parse(PERIOD_TO);

const at = (days: number, hours = 0): number => T0 + days * MS_IN_DAY + hours * MS_IN_HOUR;

let dealCounter = 0;

const buildDeal = (overrides: Partial<BacktestInfoDeal> = {}): BacktestInfoDeal => {
  const start = overrides.start ?? at(0);
  const end = overrides.end ?? start + MS_IN_HOUR;
  const startDay = new Date(start);
  startDay.setHours(0, 0, 0, 0);
  const endDay = new Date(end);
  endDay.setHours(0, 0, 0, 0);
  dealCounter += 1;
  return {
    id: overrides.id ?? `deal-${dealCounter}`,
    start,
    end,
    startDay,
    endDay,
    status: overrides.status ?? 'FINISHED',
    net: overrides.net ?? 0,
    maeAbsolute: overrides.maeAbsolute ?? 0,
    mfeAbsolute: overrides.mfeAbsolute ?? 0,
    durationInDays: (end - start) / MS_IN_DAY,
    backtestId: overrides.backtestId ?? 0,
    backtestName: overrides.backtestName ?? 'Backtest',
    quoteCurrency: overrides.quoteCurrency ?? 'USDT',
  };
};

const buildInfo = (id: number, deals: BacktestInfoDeal[], overrides: Partial<BacktestInfo> = {}): BacktestInfo => ({
  id,
  name: `Backtest ${id}`,
  exchange: 'BYBIT_FUTURES',
  symbol: `COIN${id}USDT`,
  algorithm: 'LONG',
  base: `COIN${id}`,
  quote: 'USDT',
  from: PERIOD_FROM,
  to: PERIOD_TO,
  depositAmount: 100,
  depositCurrency: 'USDT',
  leverage: 10,
  winRatePercent: null,
  profitableDeals: 0,
  losingDeals: 0,
  profitNet: 0,
  netQuotePerDay: 0,
  activeMaeAbsolute: null,
  averageDurationDays: 0,
  maxDurationSeconds: 0,
  tradingDays: 0,
  maxMaeAbsolute: 0,
  maxMfeAbsolute: 0,
  avgMaeAbsolute: 0,
  avgMfeAbsolute: 0,
  maxDrawdownQuote: 0,
  pnlMaeRatio: null,
  ...overrides,
  deals,
});

/**
 * Две монеты на 30 днях, 3 фолда по 10 дней (границы — день 10 и день 20):
 * фолд 0: A (+10, MAE 5) и D (+6, MAE 4) пересекаются → МПУ 9;
 * фолд 1: B (−4, MAE 8) начинается ровно на границе дня 10;
 * фолд 2: C (+20, MAE 3) и E (+1) с входом ровно в конце периода;
 * F (+100) вошла до начала периода и не попадает ни в один фолд.
 */
const buildFoldScenario = (): BacktestInfo[] => [
  buildInfo(1, [
    buildDeal({ id: 'A', start: at(1), end: at(2), net: 10, maeAbsolute: 5 }),
    buildDeal({ id: 'B', start: at(10), end: at(11), net: -4, maeAbsolute: 8 }),
    buildDeal({ id: 'C', start: at(25), end: at(26), net: 20, maeAbsolute: 3 }),
  ]),
  buildInfo(2, [
    buildDeal({ id: 'F', start: at(-1), end: at(-1, 6), net: 100, maeAbsolute: 1 }),
    buildDeal({ id: 'D', start: at(1, 1), end: at(3), net: 6, maeAbsolute: 4 }),
    buildDeal({ id: 'E', start: T_END, end: T_END + MS_IN_HOUR, net: 1, maeAbsolute: 0 }),
  ]),
];

// ═══════════════════════════════════════════════════════════════════
// ФИКСТУРЫ: DTO ДЛЯ ЗАГРУЗКИ
// ═══════════════════════════════════════════════════════════════════

const baseConfig: BacktestConfigDto = {
  id: 1,
  name: 'Config',
  symbol: 'BTCUSDT',
  exchange: 'BYBIT_FUTURES',
  algorithm: 'LONG',
  pullUp: 0,
  portion: 0,
  profit: null,
  deposit: { amount: 100, leverage: 10, marginType: 'CROSS', currency: 'USDT' },
  stopLoss: null,
  settings: { type: 'GRID', includePosition: null },
  conditions: [],
  from: PERIOD_FROM,
  to: PERIOD_TO,
  status: 'FINISHED',
  commissions: { maker: 0.02, taker: 0.055 },
  public: false,
  useWicks: true,
  cursor: 'cursor',
};

const baseStatistics: BacktestStatistics = {
  id: 1,
  name: 'Backtest',
  date: PERIOD_TO,
  from: PERIOD_FROM,
  to: PERIOD_TO,
  algorithm: 'LONG',
  exchange: 'BYBIT_FUTURES',
  symbol: 'BTCUSDT',
  base: 'BTC',
  quote: 'USDT',
  duration: 0,
  profitBase: 0,
  profitQuote: 0,
  netBase: 0,
  netQuote: 0,
  netBasePerDay: 0,
  netQuotePerDay: 0,
  minProfitBase: 0,
  maxProfitBase: 0,
  avgProfitBase: 0,
  minProfitQuote: 0,
  maxProfitQuote: 0,
  avgProfitQuote: 0,
  volume: 0,
  minDuration: 0,
  maxDuration: 0,
  avgDuration: 0,
  profits: 0,
  losses: 0,
  breakevens: 0,
  pullUps: 0,
  winRateProfits: 0,
  winRateLosses: 0,
  totalDeals: 0,
  minGrid: 0,
  maxGrid: 0,
  avgGrid: 0,
  minProfit: 0,
  maxProfit: 0,
  avgProfit: 0,
  mfePercent: 0,
  mfeAbsolute: 0,
  maePercent: 0,
  maeAbsolute: 0,
  commissionBase: 0,
  commissionQuote: 0,
};

const createDetail = (id: number, base: string): BacktestDetail => ({
  statistics: { ...baseStatistics, id, name: `Backtest ${id}`, base, symbol: `${base}USDT` },
  config: { ...baseConfig, id },
});

const createOrder = (createdAt: string): BacktestOrder => ({
  category: 'GRID',
  side: 'BUY',
  type: 'MARKET',
  position: 1,
  quantity: 1,
  price: 1,
  status: 'EXECUTED',
  createdAt,
  executedAt: createdAt,
  commissionAmount: 0,
  commissionAsset: 'USDT',
});

const createCycle = (overrides: Partial<BacktestCycle> & { openedAt?: string | null } = {}): BacktestCycle => {
  const { openedAt, ...rest } = overrides;
  const openTime = openedAt === undefined ? '2024-01-02T00:00:00.000Z' : openedAt;
  return {
    date: '2024-01-02T06:00:00.000Z',
    status: 'FINISHED',
    substatus: 'TAKE_PROFIT',
    exchange: 'BYBIT_FUTURES',
    symbol: 'BTCUSDT',
    base: 'BTC',
    quote: 'USDT',
    profitQuote: 0,
    profitBase: 0,
    netQuote: 0,
    netBase: 0,
    pnl: 0,
    duration: 0,
    grid: 0,
    executedGrid: 0,
    profits: 0,
    executedProfits: 0,
    volume: 0,
    mfePercent: 0,
    mfeAbsolute: 0,
    maePercent: 0,
    maeAbsolute: 0,
    commissionBase: 0,
    commissionQuote: 0,
    orders: openTime === null ? [] : [createOrder(openTime)],
    ...rest,
  };
};

beforeEach(() => {
  getBacktestDetailMock.mockReset();
  getBacktestCyclesMock.mockReset();
});

// ═══════════════════════════════════════════════════════════════════
// sanitizeCycles
// ═══════════════════════════════════════════════════════════════════

describe('sanitizeCycles', () => {
  it('drops cycles without orders but keeps cancelled cycles with orders (as the group page does)', () => {
    const finished = createCycle({ date: '2024-01-03T00:00:00.000Z', netQuote: 5 });
    const cancelledWithOrders = createCycle({ status: 'CANCELLED', substatus: 'PULL_UP', netQuote: null });
    const cancelledEmpty = createCycle({ status: 'CANCELLED', substatus: 'PULL_UP', openedAt: null });
    const finishedEmpty = createCycle({ openedAt: null });
    const started = createCycle({ status: 'STARTED', date: '2024-01-04T00:00:00.000Z' });

    const input = [finished, cancelledWithOrders, cancelledEmpty, finishedEmpty, started];
    const result = sanitizeCycles(input);

    expect(result).toEqual([finished, cancelledWithOrders, started]);
    expect(result[0]).toBe(finished);
    expect(result[1]).toBe(cancelledWithOrders);
    expect(input).toHaveLength(5);
  });

  it('a cancelled pull-up cycle with orders counts as a deal with net 0 and occupies a block slot', () => {
    const detail = createDetail(1, 'BTC');
    const cancelled = createCycle({
      status: 'CANCELLED',
      substatus: 'PULL_UP',
      netQuote: null,
      openedAt: '2024-01-02T00:00:00.000Z',
      date: '2024-01-02T10:00:00.000Z',
    });
    const blocked = createCycle({
      openedAt: '2024-01-02T01:00:00.000Z',
      date: '2024-01-02T05:00:00.000Z',
      netQuote: 7,
    });
    const cycles = [cancelled, blocked];

    const info = buildBacktestInfo(detail, sanitizeCycles(cycles));
    expect(info.deals).toHaveLength(2);
    expect(info.deals[0].net).toBe(0);
    // Как на странице группы: при блокировке 1 отменённый цикл занимает место, следующая сделка не входит.
    expect(computePortfolioMetrics([info], 1)).toMatchObject({ totalDeals: 1, totalProfit: 0 });
    expect(computePortfolioMetrics([buildBacktestInfo(detail, cycles)], 1)).toEqual(computePortfolioMetrics([info], 1));
  });

  it('drops cycles whose orders field is not an array or whose dates cannot be parsed', () => {
    const malformedOrders = { ...createCycle(), orders: null } as unknown as BacktestCycle;
    const badOpenDate = createCycle({ openedAt: 'not-a-date' });
    const badCloseDate = createCycle({ date: '' });
    const valid = createCycle();

    expect(sanitizeCycles([malformedOrders, badOpenDate, badCloseDate, valid])).toEqual([valid]);
  });

  it('returns an empty array for empty input', () => {
    expect(sanitizeCycles([])).toEqual([]);
  });

  it('makes cycles safe for buildBacktestInfo without changing profit or MAE', () => {
    const detail = createDetail(1, 'BTC');
    const cycles = [
      createCycle({ netQuote: 12, maeAbsolute: -3 }),
      createCycle({ status: 'CANCELLED', substatus: 'PULL_UP', netQuote: null, openedAt: null }),
      createCycle({
        openedAt: '2024-01-05T00:00:00.000Z',
        date: '2024-01-05T05:00:00.000Z',
        netQuote: -2,
        maeAbsolute: -7,
      }),
    ];

    expect(() => buildBacktestInfo(detail, cycles)).toThrow();

    const info = buildBacktestInfo(detail, sanitizeCycles(cycles));
    expect(info.deals).toHaveLength(2);
    expect(info.deals.reduce((sum, deal) => sum + deal.net, 0)).toBe(10);
    expect(info.maxMaeAbsolute).toBe(7);
  });
});

// ═══════════════════════════════════════════════════════════════════
// toPortfolioMetrics / computePortfolioMetrics
// ═══════════════════════════════════════════════════════════════════

const buildAggregated = (overrides: Partial<AggregatedBacktestsMetrics>): AggregatedBacktestsMetrics => ({
  ...aggregateBacktestsMetrics([], { maxConcurrentPositions: 1 }),
  ...overrides,
});

describe('toPortfolioMetrics', () => {
  it('maps aggregated group metrics into portfolio metrics', () => {
    const agg = buildAggregated({
      totalBacktests: 99,
      totalProfitQuote: 1234.5,
      maxConcurrentMae: 300,
      maxAggregatedDrawdown: 120,
      pnlToRisk: 4.115,
      totalProfitableDeals: 40,
      totalLosingDeals: 5,
      openDeals: 2,
      aggregatedActiveMae: 17,
      averageNetPerDay: 3.2,
      maxConcurrentPositions: 6,
    });

    expect(toPortfolioMetrics(agg, 12)).toEqual({
      totalProfit: 1234.5,
      maxConcurrentMae: 300,
      maxDrawdown: 120,
      pnlToRisk: 4.115,
      totalDeals: 47,
      profitableDeals: 40,
      losingDeals: 5,
      openDeals: 2,
      activeMae: 17,
      netPerDay: 3.2,
      maxConcurrentPositions: 6,
      backtests: 12,
    });
  });

  it('replaces non-finite numbers with zero so scores stay comparable', () => {
    const agg = buildAggregated({
      totalProfitQuote: Number.NaN,
      pnlToRisk: Number.POSITIVE_INFINITY,
      averageNetPerDay: Number.NaN,
      totalProfitableDeals: 3,
      openDeals: Number.NaN,
    });

    const metrics = toPortfolioMetrics(agg, Number.NaN);

    expect(metrics.totalProfit).toBe(0);
    expect(metrics.pnlToRisk).toBe(0);
    expect(metrics.netPerDay).toBe(0);
    expect(metrics.openDeals).toBe(0);
    expect(metrics.totalDeals).toBe(3);
    expect(metrics.backtests).toBe(0);
  });
});

describe('computePortfolioMetrics', () => {
  const overlappingInfos = (): BacktestInfo[] => [
    buildInfo(1, [buildDeal({ start: at(1), end: at(3), net: 50, maeAbsolute: 20 })]),
    buildInfo(2, [buildDeal({ start: at(2), end: at(4), net: -10, maeAbsolute: 30 })]),
    buildInfo(3, [buildDeal({ start: at(2, 1), end: at(5), net: 25, maeAbsolute: 10 })]),
  ];

  it('matches aggregateBacktestsMetrics of the group page with the same block', () => {
    const infos = overlappingInfos();
    const expected = toPortfolioMetrics(aggregateBacktestsMetrics(infos, { maxConcurrentPositions: 2 }), 3);

    expect(computePortfolioMetrics(infos, 2)).toEqual(expected);
  });

  it('applies the block (concurrency limit) to the basket', () => {
    const infos = overlappingInfos();

    const blocked = computePortfolioMetrics(infos, 1);
    expect(blocked.totalProfit).toBe(50);
    expect(blocked.totalDeals).toBe(1);
    expect(blocked.maxConcurrentMae).toBe(20);
    expect(blocked.pnlToRisk).toBeCloseTo(2.5);

    const unblocked = computePortfolioMetrics(infos, 3);
    expect(unblocked.totalProfit).toBe(65);
    expect(unblocked.totalDeals).toBe(3);
    expect(unblocked.profitableDeals).toBe(2);
    expect(unblocked.losingDeals).toBe(1);
    expect(unblocked.maxConcurrentMae).toBe(60);
    expect(unblocked.maxConcurrentPositions).toBe(3);
    expect(unblocked.backtests).toBe(3);
  });

  it('treats invalid or fractional blocks as an integer limit of at least one', () => {
    const infos = overlappingInfos();
    const single = computePortfolioMetrics(infos, 1);

    expect(computePortfolioMetrics(infos, 0)).toEqual(single);
    expect(computePortfolioMetrics(infos, -5)).toEqual(single);
    expect(computePortfolioMetrics(infos, Number.NaN)).toEqual(single);
    expect(computePortfolioMetrics(infos, 2.9)).toEqual(computePortfolioMetrics(infos, 2));
  });

  it('counts open (STARTED) deals and their MAE', () => {
    const infos = [
      buildInfo(1, [
        buildDeal({ start: at(1), end: at(2), net: 8, maeAbsolute: 2 }),
        buildDeal({ start: at(28), end: T_END, status: 'STARTED', net: 0, maeAbsolute: 15 }),
      ]),
    ];

    const metrics = computePortfolioMetrics(infos, 5);

    expect(metrics.totalProfit).toBe(8);
    expect(metrics.openDeals).toBe(1);
    expect(metrics.totalDeals).toBe(2);
    expect(metrics.activeMae).toBe(15);
    expect(metrics.maxConcurrentMae).toBe(15);
  });

  it('returns zero metrics for an empty basket', () => {
    expect(computePortfolioMetrics([], 3)).toEqual({
      totalProfit: 0,
      maxConcurrentMae: 0,
      maxDrawdown: 0,
      pnlToRisk: 0,
      totalDeals: 0,
      profitableDeals: 0,
      losingDeals: 0,
      openDeals: 0,
      activeMae: 0,
      netPerDay: 0,
      maxConcurrentPositions: 0,
      backtests: 0,
    });
  });
});

// ═══════════════════════════════════════════════════════════════════
// buildFoldWindows / computeFoldMetrics
// ═══════════════════════════════════════════════════════════════════

describe('buildFoldWindows', () => {
  it('splits the period into contiguous equal windows', () => {
    const windows = buildFoldWindows(T0, T_END, 3);

    expect(windows).toEqual([
      { from: T0, to: at(10) },
      { from: at(10), to: at(20) },
      { from: at(20), to: T_END },
    ]);
  });

  it('rounds inner boundaries to whole milliseconds and keeps exact outer bounds', () => {
    const windows = buildFoldWindows(0, 10, 3);

    expect(windows).toEqual([
      { from: 0, to: 3 },
      { from: 3, to: 7 },
      { from: 7, to: 10 },
    ]);
  });

  it('floors fractional fold counts', () => {
    expect(buildFoldWindows(T0, T_END, 2.7)).toHaveLength(2);
  });

  it('returns no windows for fewer than two folds or an invalid period', () => {
    expect(buildFoldWindows(T0, T_END, 1)).toEqual([]);
    expect(buildFoldWindows(T0, T_END, 0)).toEqual([]);
    expect(buildFoldWindows(T0, T_END, Number.NaN)).toEqual([]);
    expect(buildFoldWindows(T_END, T0, 3)).toEqual([]);
    expect(buildFoldWindows(T0, T0, 3)).toEqual([]);
    expect(buildFoldWindows(Number.NaN, T_END, 3)).toEqual([]);
  });
});

describe('computeFoldMetrics', () => {
  it('returns an empty list when folds < 2', () => {
    const infos = buildFoldScenario();

    expect(computeFoldMetrics(infos, 5, PERIOD_FROM, PERIOD_TO, 1)).toEqual([]);
    expect(computeFoldMetrics(infos, 5, PERIOD_FROM, PERIOD_TO, 0)).toEqual([]);
    expect(computeFoldMetrics(infos, 5, PERIOD_FROM, PERIOD_TO, -3)).toEqual([]);
  });

  it('returns an empty list for an unparsable or empty period', () => {
    const infos = buildFoldScenario();

    expect(computeFoldMetrics(infos, 5, 'garbage', PERIOD_TO, 3)).toEqual([]);
    expect(computeFoldMetrics(infos, 5, PERIOD_FROM, '', 3)).toEqual([]);
    expect(computeFoldMetrics(infos, 5, PERIOD_TO, PERIOD_FROM, 3)).toEqual([]);
  });

  it('assigns deals to folds by entry time and computes profit, risk and deals per fold', () => {
    const folds = computeFoldMetrics(buildFoldScenario(), 5, PERIOD_FROM, PERIOD_TO, 3);

    expect(folds).toHaveLength(3);

    expect(folds[0]).toEqual({ from: T0, to: at(10), profit: 16, risk: 9, pnlToRisk: 16 / 9, deals: 2 });
    // Сделка на границе (день 10) относится к следующему фолду; просадка от первой точки серии — 0.
    expect(folds[1]).toEqual({ from: at(10), to: at(20), profit: -4, risk: 8, pnlToRisk: -0.5, deals: 1 });
    // Сделка ровно в конце периода входит в последний фолд.
    expect(folds[2]).toEqual({ from: at(20), to: T_END, profit: 21, risk: 3, pnlToRisk: 7, deals: 2 });
  });

  it('ignores deals outside the period, so fold profits add up to the in-period profit', () => {
    const infos = buildFoldScenario();
    const folds = computeFoldMetrics(infos, 5, PERIOD_FROM, PERIOD_TO, 3);
    const total = computePortfolioMetrics(infos, 5);

    const foldProfit = folds.reduce((sum, fold) => sum + fold.profit, 0);
    expect(total.totalProfit).toBe(133);
    expect(foldProfit).toBe(total.totalProfit - 100);
    expect(folds.reduce((sum, fold) => sum + fold.deals, 0)).toBe(total.totalDeals - 1);
  });

  it('applies the block inside every fold', () => {
    const folds = computeFoldMetrics(buildFoldScenario(), 1, PERIOD_FROM, PERIOD_TO, 3);

    // D пересекается с A и отсекается блокировкой 1.
    expect(folds[0]).toMatchObject({ profit: 10, risk: 5, pnlToRisk: 2, deals: 1 });
  });

  it('divides by at least one when risk is below one quote unit', () => {
    const infos = [
      buildInfo(1, [
        buildDeal({ start: at(1), end: at(2), net: 0.5, maeAbsolute: 0.2 }),
        buildDeal({ start: at(16), end: at(17), net: 0.3, maeAbsolute: 0 }),
      ]),
    ];

    const folds = computeFoldMetrics(infos, 1, PERIOD_FROM, PERIOD_TO, 2);

    expect(folds[0]).toMatchObject({ profit: 0.5, risk: 0.2, pnlToRisk: 0.5, deals: 1 });
    expect(folds[1]).toMatchObject({ profit: 0.3, risk: 0, pnlToRisk: 0.3, deals: 1 });
  });

  it('uses drawdown as risk when it exceeds the concurrent MAE', () => {
    const infos = [
      buildInfo(1, [
        buildDeal({ start: at(1), end: at(2), net: 30, maeAbsolute: 1 }),
        buildDeal({ start: at(3), end: at(4), net: -25, maeAbsolute: 2 }),
      ]),
    ];

    const [first] = computeFoldMetrics(infos, 1, PERIOD_FROM, PERIOD_TO, 2);

    expect(first).toMatchObject({ profit: 5, risk: 25, pnlToRisk: 0.2, deals: 2 });
  });

  it('counts open deals in the fold where they were entered', () => {
    const infos = [
      buildInfo(1, [buildDeal({ start: at(25), end: T_END, status: 'STARTED', net: 0, maeAbsolute: 11 })]),
    ];

    const folds = computeFoldMetrics(infos, 3, PERIOD_FROM, PERIOD_TO, 2);

    expect(folds[0]).toMatchObject({ profit: 0, risk: 0, deals: 0 });
    expect(folds[1]).toMatchObject({ profit: 0, risk: 11, pnlToRisk: 0, deals: 1 });
  });

  it('returns zero folds for backtests without deals in a window', () => {
    const infos = [buildInfo(1, [buildDeal({ start: at(2), end: at(3), net: 4, maeAbsolute: 1 })])];

    const folds = computeFoldMetrics(infos, 3, PERIOD_FROM, PERIOD_TO, 3);

    expect(folds.map((fold) => fold.deals)).toEqual([1, 0, 0]);
    expect(folds[2]).toMatchObject({ profit: 0, risk: 0, pnlToRisk: 0 });
  });

  it('does not mutate the input infos', () => {
    const infos = buildFoldScenario();
    const snapshot = infos.map((info) => ({ from: info.from, to: info.to, deals: [...info.deals] }));

    computeFoldMetrics(infos, 2, PERIOD_FROM, PERIOD_TO, 3);

    infos.forEach((info, index) => {
      expect(info.from).toBe(snapshot[index]?.from);
      expect(info.to).toBe(snapshot[index]?.to);
      expect(info.deals).toEqual(snapshot[index]?.deals);
    });
  });
});

// ═══════════════════════════════════════════════════════════════════
// loadBacktestInfo / evaluateBasket (сервисы замоканы)
// ═══════════════════════════════════════════════════════════════════

const cyclesById: Record<number, BacktestCycle[]> = {
  11: [
    createCycle({
      openedAt: '2024-01-02T00:00:00.000Z',
      date: '2024-01-02T06:00:00.000Z',
      netQuote: 40,
      maeAbsolute: -10,
    }),
    createCycle({ status: 'CANCELLED', substatus: 'PULL_UP', netQuote: null, openedAt: null }),
    createCycle({
      openedAt: '2024-01-25T00:00:00.000Z',
      date: '2024-01-25T08:00:00.000Z',
      netQuote: -15,
      maeAbsolute: -20,
    }),
  ],
  22: [
    createCycle({
      openedAt: '2024-01-03T00:00:00.000Z',
      date: '2024-01-04T00:00:00.000Z',
      netQuote: 25,
      maeAbsolute: -5,
    }),
    createCycle({
      status: 'STARTED',
      openedAt: '2024-01-29T00:00:00.000Z',
      date: PERIOD_TO,
      netQuote: -3,
      maeAbsolute: -12,
    }),
  ],
};

const mockServicesById = (): void => {
  getBacktestDetailMock.mockImplementation(async (id) => createDetail(id, `C${id}`));
  getBacktestCyclesMock.mockImplementation(async (id) => cyclesById[id] ?? []);
};

describe('loadBacktestInfo', () => {
  it('loads detail and cycles with default caching and builds a sanitized info', async () => {
    mockServicesById();

    const info = await loadBacktestInfo(11);

    expect(getBacktestDetailMock).toHaveBeenCalledTimes(1);
    expect(getBacktestDetailMock).toHaveBeenCalledWith(11);
    expect(getBacktestCyclesMock).toHaveBeenCalledTimes(1);
    expect(getBacktestCyclesMock).toHaveBeenCalledWith(11);
    expect(info.id).toBe(11);
    expect(info.base).toBe('C11');
    expect(info.deals).toHaveLength(2);
    expect(info.deals.map((deal) => deal.net).sort((left, right) => left - right)).toEqual([-15, 40]);
  });

  it('wraps load errors with the backtest id and keeps the original message and cause', async () => {
    const original = new Error('HTTP 401 Unauthorized');
    getBacktestDetailMock.mockImplementation(async (id) => createDetail(id, 'BTC'));
    getBacktestCyclesMock.mockRejectedValue(original);

    const failure = await loadBacktestInfo(7).catch((error: unknown) => error);

    expect(failure).toBeInstanceOf(Error);
    expect((failure as Error).message).toBe('Не удалось загрузить результаты бэктеста 7: HTTP 401 Unauthorized');
    expect((failure as Error).cause).toBe(original);
  });

  it('describes non-Error rejections', async () => {
    getBacktestDetailMock.mockRejectedValue('сеть недоступна');
    getBacktestCyclesMock.mockResolvedValue([]);

    await expect(loadBacktestInfo(3)).rejects.toThrow('Не удалось загрузить результаты бэктеста 3: сеть недоступна');
  });
});

describe('evaluateBasket', () => {
  const run = { periodFrom: PERIOD_FROM, periodTo: PERIOD_TO };

  it('loads every backtest in the given order and computes metrics and folds', async () => {
    mockServicesById();

    const result = await evaluateBasket([22, 11], 5, run, 2);

    expect(result.infos.map((info) => info.id)).toEqual([22, 11]);
    expect(result.metrics).toEqual(computePortfolioMetrics(result.infos, 5));
    expect(result.metrics.backtests).toBe(2);
    expect(result.metrics.totalProfit).toBe(50);
    expect(result.metrics.totalDeals).toBe(4);
    expect(result.metrics.openDeals).toBe(1);
    expect(result.folds).toEqual(computeFoldMetrics(result.infos, 5, PERIOD_FROM, PERIOD_TO, 2));
    expect(result.folds).toHaveLength(2);
    expect(result.folds[0]).toMatchObject({ profit: 65, deals: 2 });
    expect(result.folds[1]).toMatchObject({ profit: -15, deals: 2 });
  });

  it('skips fold metrics when folds < 2', async () => {
    mockServicesById();

    const result = await evaluateBasket([11], 3, run, 0);

    expect(result.folds).toEqual([]);
    expect(result.metrics.totalProfit).toBe(25);
  });

  it('rejects when any backtest of the basket fails to load', async () => {
    getBacktestDetailMock.mockImplementation(async (id) => {
      if (id === 22) {
        throw new Error('HTTP 500');
      }
      return createDetail(id, `C${id}`);
    });
    getBacktestCyclesMock.mockImplementation(async (id) => cyclesById[id] ?? []);

    await expect(evaluateBasket([11, 22], 3, run, 3)).rejects.toThrow('бэктеста 22: HTTP 500');
  });

  it('returns zero metrics for an empty basket without loading anything', async () => {
    const result = await evaluateBasket([], 3, run, 3);

    expect(getBacktestDetailMock).not.toHaveBeenCalled();
    expect(result.infos).toEqual([]);
    expect(result.metrics.totalDeals).toBe(0);
    expect(result.folds.every((fold) => fold.deals === 0)).toBe(true);
  });
});
