import { describe, expect, it } from 'vitest';
import {
  VM_GOALS,
  VM_STABLE_METRIC,
  vmAbsoluteVerdict,
  vmBuildTemplate,
  vmDealStats,
  vmGrid,
  vmIsTemplateGoal,
  vmMirrorCondition,
  vmMirrorToShort,
  vmMirrorValue,
  vmModeRanges,
  vmNormalizeTemplate,
  vmPrepareBase,
  vmScope,
  vmSelectCoins,
  vmSimpleForm,
  vmStableScore,
  vmStyleViolations,
} from '../src/modes.mjs';

/** A LONG grid bot shaped like the user's (signal take-profit, signal stop, pull-up). */
const makeShell = (patch = {}) => ({
  id: 7,
  name: 'LONG 22',
  exchange: 'BYBIT_FUTURES',
  algorithm: 'LONG',
  symbols: ['XTZ/USDT'],
  pair: { exchange: 'BYBIT_FUTURES', type: 'FUTURES', from: 'XTZ', to: 'USDT', symbol: 'XTZUSDT' },
  deposit: { amount: 7.5, leverage: 20, marginType: 'CROSS', currency: 'USDT' },
  apiKey: 11,
  conditions: [
    { type: 'INDICATOR', indicator: 'RSI', interval: 'ONE_HOUR', basic: false, value: 30, operation: 'LESS', closed: true, reverse: false },
    { type: 'INDICATOR', indicator: 'BOLLINGER_BANDS', interval: 'ONE_HOUR', basic: true, value: null, operation: null, closed: true, reverse: false },
    { type: 'INDICATOR', indicator: 'ADX', interval: 'ONE_HOUR', basic: false, value: 25, operation: 'GREATER', closed: true, reverse: false },
  ],
  settings: {
    type: 'CUSTOM',
    indentType: 'ENTRY',
    includePosition: true,
    baseOrder: { indent: 0, volume: 10, conditions: [] },
    orders: [
      {
        indent: 2,
        volume: 30,
        conditions: [
          { type: 'INDICATOR', indicator: 'CCI', interval: 'FIFTEEN_MINUTES', basic: false, value: -150, operation: 'LESS', closed: true, reverse: false },
        ],
      },
      { indent: 5, volume: 60, conditions: [] },
    ],
  },
  profit: {
    type: 'SIGNAL',
    currency: 'USDT',
    checkPnl: 0.8,
    conditions: [
      { type: 'INDICATOR', indicator: 'RSI', interval: 'ONE_HOUR', basic: false, value: 70, operation: 'GREATER', closed: true, reverse: false },
    ],
  },
  stopLoss: {
    indent: 20,
    termination: false,
    conditionalIndent: 5,
    conditionalIndentType: 'AVERAGE',
    conditions: [
      { type: 'INDICATOR', indicator: 'WILLIAMS_R', interval: 'ONE_HOUR', basic: false, value: -95, operation: 'LESS', closed: true, reverse: false },
    ],
  },
  pullUp: 0.2,
  portion: null,
  ...patch,
});

describe('modes registry', () => {
  it('lists the five tabs with improve first and template flags', () => {
    expect(VM_GOALS.map((goal) => goal.id)).toEqual(['improve', 'scalper', 'short', 'long', 'coins']);
    expect(vmIsTemplateGoal('improve')).toBe(false);
    expect(vmIsTemplateGoal('coins')).toBe(false);
    expect(vmIsTemplateGoal('scalper')).toBe(true);
    expect(vmIsTemplateGoal('short')).toBe(true);
    expect(vmIsTemplateGoal('long')).toBe(true);
    expect(vmIsTemplateGoal('unknown')).toBe(false);
  });

  it('normalises template options per mode', () => {
    expect(vmNormalizeTemplate('scalper', { direction: 'SHORT', timeframe: 'ONE_MINUTE', profile: 'single' })).toEqual({
      direction: 'SHORT',
      timeframe: 'ONE_MINUTE',
      profile: 'single',
      stopLoss: true,
    });
    expect(vmNormalizeTemplate('scalper', { timeframe: 'ONE_DAY', profile: 'nope' })).toMatchObject({
      timeframe: 'FIVE_MINUTES',
      profile: 'grid2',
      direction: 'LONG',
    });
    expect(vmNormalizeTemplate('short', { direction: 'LONG' }).direction).toBe('SHORT');
    expect(vmNormalizeTemplate('long', { direction: 'SHORT' }).direction).toBe('LONG');
    expect(vmNormalizeTemplate('improve', {})).toBeNull();
  });
});

describe('SHORT mirror', () => {
  it('mirrors thresholds of directional indicators around their centre', () => {
    expect(vmMirrorValue('RSI', 30)).toBe(70);
    expect(vmMirrorValue('CCI', -150)).toBe(150);
    expect(vmMirrorValue('WILLIAMS_R', -95)).toBe(-5);
    expect(vmMirrorValue('CMO', -50)).toBe(50);
    expect(vmMirrorValue('PRICE_CHANGE_PERIOD_10', -3)).toBe(3);
    expect(vmMirrorValue('ADX', 25)).toBeNull();
    expect(vmMirrorValue('ATR_PERCENT', 1)).toBeNull();
  });

  it('flips the comparison only for mirrored conditions and keeps signal indicators', () => {
    const shell = makeShell();
    expect(vmMirrorCondition(shell.conditions[0])).toMatchObject({
      changed: true,
      condition: { indicator: 'RSI', value: 70, operation: 'GREATER' },
    });
    expect(vmMirrorCondition(shell.conditions[1])).toMatchObject({ changed: false, signal: true });
    expect(vmMirrorCondition(shell.conditions[2])).toMatchObject({
      changed: false,
      condition: { indicator: 'ADX', value: 25, operation: 'GREATER' },
    });
  });

  it('mirrors entry, grid, take-profit and stop-loss filters and keeps the grid numbers', () => {
    const shell = makeShell();
    const { strategy, notes } = vmMirrorToShort(shell);
    expect(strategy.algorithm).toBe('SHORT');
    expect(strategy.conditions[0]).toMatchObject({ value: 70, operation: 'GREATER' });
    expect(strategy.settings.orders[0].conditions[0]).toMatchObject({ value: 150, operation: 'GREATER' });
    expect(strategy.profit.conditions[0]).toMatchObject({ value: 30, operation: 'LESS' });
    expect(strategy.stopLoss.conditions[0]).toMatchObject({ value: -5, operation: 'GREATER' });
    expect(strategy.settings.orders.map((order) => order.indent)).toEqual([2, 5]);
    expect(strategy.profit.checkPnl).toBe(0.8);
    expect(notes).toMatchObject({ changed: 4, signals: 1, alreadyShort: false });
    // The input is not modified.
    expect(shell.algorithm).toBe('LONG');
    expect(shell.conditions[0].value).toBe(30);
  });

  it('mirrors tree-format comparisons and leaves a SHORT bot unchanged', () => {
    const tree = makeShell({
      conditions: null,
      conditionGroups: [
        [
          {
            type: 'OPERATION',
            left: { type: 'INDICATOR', indicator: 'RSI', timeFrame: 60, method: 'CLOSE', shift: 0 },
            operator: 'LESS',
            right: { type: 'CONSTANT', value: 25 },
          },
          {
            type: 'OPERATION',
            left: { type: 'INDICATOR', indicator: 'PRICE', timeFrame: 60 },
            operator: 'GREATER',
            right: { type: 'INDICATOR', indicator: 'EMA', timeFrame: 60, period: 200 },
          },
        ],
      ],
    });
    const { strategy: mirrored, notes: treeNotes } = vmMirrorToShort(tree);
    expect(mirrored.conditionGroups[0][0]).toMatchObject({ operator: 'GREATER', right: { value: 75 } });
    // Indicator vs indicator is kept as is and reported for a manual check.
    expect(mirrored.conditionGroups[0][1].operator).toBe('GREATER');
    expect(treeNotes.kept).toBe(1);
    const short = makeShell({ algorithm: 'SHORT' });
    const again = vmMirrorToShort(short);
    expect(again.notes.alreadyShort).toBe(true);
    expect(again.strategy.conditions[0].value).toBe(30);
  });
});

describe('templates', () => {
  it('generates a grid with volumes summing to 100 and increasing indents', () => {
    const grid = vmGrid({ count: 8, lastIndent: 18, power: 1.25, ratio: 1.3 });
    const orders = [grid.baseOrder, ...grid.orders];
    expect(orders).toHaveLength(8);
    expect(orders.reduce((sum, order) => sum + order.volume, 0)).toBe(100);
    expect(orders[0].indent).toBe(0);
    expect(orders.at(-1).indent).toBe(18);
    for (let index = 1; index < orders.length; index += 1) {
      expect(orders[index].indent).toBeGreaterThan(orders[index - 1].indent);
    }
    const steps = vmGrid({ count: 4, lastIndent: 6, power: 1 }, 'PREVIOUS');
    expect([steps.baseOrder, ...steps.orders].map((order) => order.indent)).toEqual([0, 2, 2, 2]);
  });

  it('builds a scalper on the shell: keeps exchange, deposit and enums, generates grid, filters, TP and SL', () => {
    const shell = makeShell();
    const scalper = vmBuildTemplate('scalper', shell, { timeframe: 'ONE_MINUTE', profile: 'grid2' }, ['ADA', 'xrp/usdt']);
    expect(scalper.algorithm).toBe('LONG');
    expect(scalper.exchange).toBe('BYBIT_FUTURES');
    expect(scalper.deposit).toEqual(shell.deposit);
    expect(scalper.settings).toMatchObject({ type: 'CUSTOM', indentType: 'ENTRY', includePosition: true });
    expect([scalper.settings.baseOrder, ...scalper.settings.orders].map((order) => order.volume)).toEqual([40, 30, 30]);
    expect(scalper.conditions.map((condition) => [condition.indicator, condition.interval, condition.operation])).toEqual([
      ['RSI', 'ONE_MINUTE', 'LESS'],
      ['CCI', 'ONE_MINUTE', 'LESS'],
    ]);
    expect(scalper.profit).toMatchObject({ type: 'SIGNAL', currency: 'USDT', checkPnl: 0.5 });
    expect(scalper.profit.conditions).toHaveLength(1);
    expect(scalper.stopLoss).toMatchObject({ indent: 3, conditionalIndent: null, conditionalIndentType: 'AVERAGE', conditions: [] });
    expect(scalper.pullUp).toBeNull();
    expect(scalper.symbols).toEqual(['ADA/USDT', 'XRP/USDT']);
    expect(shell.settings.orders).toHaveLength(2);
  });

  it('builds SHORT templates by mirroring the LONG template', () => {
    const shell = makeShell();
    const shortScalper = vmBuildTemplate('scalper', shell, { direction: 'SHORT', profile: 'single', stopLoss: false });
    expect(shortScalper.algorithm).toBe('SHORT');
    expect(shortScalper.conditions[0]).toMatchObject({ indicator: 'RSI', operation: 'GREATER', value: 70 });
    expect(shortScalper.settings.orders).toEqual([]);
    expect(shortScalper.settings.baseOrder.volume).toBe(100);
    expect(shortScalper.stopLoss).toBeNull();
    const shortDca = vmBuildTemplate('short', shell, { profile: 'classic' });
    expect(shortDca.algorithm).toBe('SHORT');
    expect(shortDca.settings.orders).toHaveLength(7);
    expect(shortDca.pullUp).toBe(0.3);
  });

  it('uses the shell take-profit type and drops signal exits for non-signal take-profit', () => {
    const percent = makeShell({ profit: { type: 'PERCENT', currency: 'USDT', checkPnl: 1, conditions: [] }, stopLoss: null });
    const long = vmBuildTemplate('long', percent, { profile: 'tight' });
    expect(long.profit).toEqual({ type: 'PERCENT', currency: 'USDT', checkPnl: 0.7, conditions: [] });
    expect(long.stopLoss).toBeNull();
    // A ticked «стоп-лосс» gives DCA templates a stop beyond the grid.
    expect(vmBuildTemplate('long', percent, { profile: 'classic', stopLoss: true }).stopLoss.indent).toBe(25);
    expect(vmBuildTemplate('short', percent, { profile: 'tight', stopLoss: true }).stopLoss.indent).toBe(15);
    const scalper = vmBuildTemplate('scalper', percent, { profile: 'grid2', stopLoss: true });
    expect(scalper.stopLoss).toEqual({ indent: 3, termination: false, conditionalIndent: null, conditionalIndentType: null, conditions: null });
  });

  it('prepares the base of a run: bot as is, mirrored bot or template', () => {
    const shell = makeShell();
    expect(vmPrepareBase({ goal: 'improve' }, shell).strategy).toBe(shell);
    expect(vmPrepareBase({ goal: 'coins' }, shell).strategy).toBe(shell);
    const mirror = vmPrepareBase({ goal: 'short', template: { profile: 'mirror' }, symbols: ['ADA'] }, shell);
    expect(mirror.strategy.algorithm).toBe('SHORT');
    expect(mirror.strategy.symbols).toEqual(['ADA/USDT']);
    expect(mirror.strategy.settings.orders).toHaveLength(2);
    expect(mirror.notes).toMatchObject({ kind: 'mirror', changed: 4 });
    const long = vmPrepareBase({ goal: 'long', template: { profile: 'classic' }, symbols: [] }, shell);
    expect(long.strategy.settings.orders).toHaveLength(7);
    expect(long.notes).toEqual({ kind: 'template', profile: 'classic' });
  });
});

describe('search space', () => {
  it('opens the scope for global search with a valued indicator pool', () => {
    const scope = vmScope('scalper', { dcaIndents: false, rangeWidthPercent: 25 }, 'LONG');
    expect(scope).toMatchObject({ timeframes: true, pullUp: false, rangeWidthPercent: 100, structureSwap: true, structureAdd: 1 });
    expect(scope.structurePlaces).toEqual({ entry: true, order: false, takeProfit: false, stopLoss: false });
    expect(scope.structurePool).toContain('RSI');
    expect(vmScope('long', {}, 'LONG').structurePool).toBeNull();
    expect(vmScope('short', {}, 'SHORT').structurePool).toContain('CCI');
  });

  it('sets absolute ranges for TP, SL, indents and directional thresholds', () => {
    const strategy = vmBuildTemplate('scalper', makeShell(), { profile: 'grid2' });
    const drafts = [
      { id: 'tp', kind: 'takeProfit', baseValue: 0.5, hardMin: 0.01, hardMax: 100 },
      { id: 'sl', kind: 'stopLoss', baseValue: 3, hardMin: 0.1, hardMax: 100 },
      { id: 'grid:indent:1', kind: 'orderIndent', orderIndex: 1, baseValue: 0.8, hardMin: 0, hardMax: 100 },
      { id: 'grid:indent:0', kind: 'orderIndent', orderIndex: 0, baseValue: 0, hardMin: 0, hardMax: 100 },
      { id: 'cond:v:rsi', kind: 'conditionValue', baseValue: 30, hardMin: 0, hardMax: 100, targets: [{ path: ['conditions', 0], field: 'value' }] },
      { id: 'cond:v:cci', kind: 'conditionValue', baseValue: -100, hardMin: -400, hardMax: 400, targets: [{ path: ['conditions', 1], field: 'value' }] },
    ];
    const ranges = vmModeRanges('scalper', drafts, strategy);
    expect(ranges.tp).toEqual({ enabled: true, min: 0.2, max: 1.5 });
    expect(ranges.sl).toEqual({ enabled: true, min: 0.5, max: 6 });
    expect(ranges['grid:indent:1']).toEqual({ enabled: true, min: 0.32, max: 1.76 });
    expect(ranges['grid:indent:0']).toBeUndefined();
    expect(ranges['cond:v:rsi']).toEqual({ enabled: true, min: 8, max: 45 });
    expect(ranges['cond:v:cci']).toEqual({ enabled: true, min: -350, max: -30 });
    const short = vmBuildTemplate('scalper', makeShell(), { profile: 'grid2', direction: 'SHORT' });
    const shortDraft = { ...drafts[4], baseValue: 70 };
    expect(vmModeRanges('scalper', [shortDraft], short)['cond:v:rsi']).toEqual({ enabled: true, min: 55, max: 92 });
    expect(vmModeRanges('improve', drafts, strategy)).toEqual({});
  });

  it('builds the one-click form of a mode', () => {
    const form = {
      symbolsText: '',
      periodPreset: 'since2020',
      constraints: { minDealsSharePercent: 60, maxPortfolioDrawdownPercent: null },
      portfolio: { screeningCount: 20 },
      genetic: { maxParallel: 6 },
      robustness: { neighbors: 6 },
      scope: {},
      template: { profile: 'single' },
    };
    const context = {
      presets: { balanced: { searchBudget: 110, earlyStopGenerations: 3 } },
      periodOf: (preset) => ({ from: preset, to: '2026-10-15' }),
      symbols: ['ADA', 'XRP'],
    };
    const scalper = vmSimpleForm('scalper', form, context);
    expect(scalper).toMatchObject({
      goal: 'scalper',
      mode: 'portfolio',
      portfolioTarget: VM_STABLE_METRIC,
      symbolsText: 'ADA, XRP',
      periodFrom: 'since2020',
      genetic: { maxParallel: 6, searchBudget: 140, searchMethod: 'pattern' },
      constraints: { minDealsSharePercent: 0, maxPortfolioDrawdownPercent: 35, vtMaxAvgHoldHours: 24, vtMinDealsPerCoinMonth: 2 },
      portfolio: { screeningCount: 12, maxDepositChangePercent: 0, foldConsistency: true },
      robustness: { enabled: true, neighbors: 6 },
    });
    expect(scalper.template.profile).toBe('single');
    expect(scalper.scope.timeframes).toBe(true);
    const custom = vmSimpleForm('scalper', { ...form, periodPreset: 'custom' }, context);
    expect(custom.periodPreset).toBe('1y');
    const coins = vmSimpleForm('coins', form, context);
    expect(coins).toMatchObject({ goal: 'coins', template: null, robustness: { enabled: false } });
    expect(coins.constraints.vtMaxAvgHoldHours).toBeNull();
  });
});

describe('deal style, constraints and score', () => {
  const H = 3600 * 1000;
  const entries = [
    {
      symbol: 'ADA/USDT',
      from: '2024-01-01T00:00:00.000Z',
      to: '2024-03-01T00:00:00.000Z',
      deals: [
        { s: 0, e: 2 * H, n: 1, m: 0.5, st: 'F', o: [] },
        { s: 10 * H, e: 16 * H, n: 1, m: 1, st: 'F', o: [12 * H] },
        { s: 20 * H, e: 20 * H, n: 0, m: 2, st: 'S', o: [21 * H, 22 * H] },
      ],
    },
  ];

  it('measures holding time, deals per coin-month and grid usage', () => {
    const stats = vmDealStats(entries);
    expect(stats).toMatchObject({
      deals: 3,
      closedDeals: 2,
      avgHoldHours: 4,
      medianHoldHours: 4,
      gridShare: 0.6667,
      avgGridOrders: 1.5,
    });
    expect(stats.dealsPerCoinMonth).toBeCloseTo(3 / (60 / 30.4375), 2);
  });

  it('turns style limits into violations', () => {
    const stats = vmDealStats(entries);
    expect(vmStyleViolations(stats, { vtMaxAvgHoldHours: 2, vtMinDealsPerCoinMonth: 5 }).map((item) => item.message)).toEqual([
      'Средняя сделка 4.0 ч — дольше 2 ч',
      'Сделок 1.52 в месяц на монету — меньше 5',
    ]);
    expect(vmStyleViolations(stats, { vtMaxAvgHoldHours: 24, vtMinDealsPerCoinMonth: 1 })).toEqual([]);
    expect(vmStyleViolations(stats, {})).toEqual([]);
  });

  it('discounts profit-to-drawdown by the share of profitable folds', () => {
    const raw = { adjustedProfit: 300, risk: 100 };
    expect(vmStableScore(raw, null)).toBe(3);
    expect(vmStableScore(raw, [{ profit: 1 }, { profit: 2 }])).toBe(3);
    expect(vmStableScore(raw, [{ profit: 1 }, { profit: -2 }])).toBe(2.25);
    expect(vmStableScore(raw, [{ profit: -1 }, { profit: -2 }])).toBe(1.5);
    expect(vmStableScore({ adjustedProfit: -50, risk: 100 }, [{ profit: 1 }, { profit: 1 }])).toBe(-0.5);
    expect(vmStableScore({ adjustedProfit: 10, risk: 0 }, null)).toBe(100);
    // Above 50 the ratio is compressed, but stays ordered and below 100.
    const high = vmStableScore({ adjustedProfit: 120, risk: 1 }, null);
    const higher = vmStableScore({ adjustedProfit: 200, risk: 1 }, null);
    expect(high).toBeGreaterThan(50);
    expect(higher).toBeGreaterThan(high);
    expect(higher).toBeLessThan(100);
    expect(vmStableScore({ adjustedProfit: 40, risk: 1 }, null)).toBe(40);
  });
});

describe('absolute robustness verdict', () => {
  const segment = (kind, label, profit) => ({ kind, label, profit, baseProfit: 0, risk: 1, baseRisk: 1 });

  it('is robust when the variant earns in most years and outside the search', () => {
    const report = {
      segments: [
        segment('year', '2021', 5),
        segment('year', '2022', 2),
        segment('year', '2023', 1),
        segment('year', '2024', 3),
        segment('year', '2025', 4),
        segment('coins', 'Монеты вне поиска (20)', 10),
      ],
      plateau: { tested: 4, beatBase: 4, medianRetention: 0.8 },
      verdict: 'overfit',
      reasons: ['на всех монетах за весь период не лучше исходной'],
    };
    const verdict = vmAbsoluteVerdict(report);
    expect(verdict).toMatchObject({ verdict: 'robust', reasons: [], yearsNotWorse: 5, absolute: true });
  });

  it('is partial or overfit with losing years, losing segments or a sharp peak', () => {
    const partial = vmAbsoluteVerdict({
      segments: [segment('year', '2021', -1), segment('year', '2022', 2), segment('coins', 'Монеты вне поиска (5)', 1)],
      plateau: { tested: 4, beatBase: 2, medianRetention: 0.5 },
    });
    expect(partial.verdict).toBe('partial');
    expect(partial.reasons[0]).toBe('убыток в 1 из 2 лет: 2021');
    const overfit = vmAbsoluteVerdict({
      segments: [segment('year', '2021', -1), segment('year', '2022', -2), segment('coins', 'Монеты вне поиска (5)', -1)],
      plateau: null,
    });
    expect(overfit.verdict).toBe('overfit');
    expect(overfit.reasons).toContain('Монеты вне поиска (5): убыток');
    // Profitable segments but a loss on all coins over the full period: never robust.
    const losing = vmAbsoluteVerdict(
      { segments: [segment('year', '2021', 10), segment('year', '2022', -200), segment('year', '2023', 5), segment('year', '2024', 5), segment('year', '2025', 5), segment('coins', 'Монеты вне поиска (5)', 3)], plateau: { tested: 4, medianRetention: 0.9 } },
      { finalProfit: -177 },
    );
    expect(losing.verdict).toBe('overfit');
    expect(losing.reasons[0]).toBe('на всех монетах за весь период — убыток');
    // Shorter than a year: cannot be robust.
    const short = vmAbsoluteVerdict({ segments: [segment('coins', 'Монеты вне поиска (5)', 3)], plateau: null }, { finalProfit: 10 });
    expect(short.verdict).toBe('partial');
  });
});

describe('coin selection', () => {
  /** Portfolio aggregation stand-in: profit adds up, drawdown is the sum of per-coin drawdowns. */
  const evaluate = (subset) => {
    const profit = subset.reduce((sum, entry) => sum + entry.deals.reduce((acc, deal) => acc + deal.n, 0), 0);
    const risk = subset.reduce((sum, entry) => sum + Math.max(...entry.deals.map((deal) => deal.m)), 0);
    return { profit, risk, monthlyPercent: profit, drawdownPercent: risk, deals: subset.length };
  };
  const coin = (symbol, net, mae) => ({ symbol, deals: [{ s: 0, e: 1, n: net, m: mae, st: 'F', o: [] }] });

  it('drops coins that hurt profit-to-drawdown and keeps a minimum basket', async () => {
    const entries = [
      coin('A', 10, 2),
      coin('B', 8, 2),
      coin('C', 6, 2),
      coin('D', 5, 2),
      coin('E', 4, 2),
      coin('F', -6, 4),
      coin('G', 1, 8),
      { symbol: 'H', deals: [] },
    ];
    const result = await vmSelectCoins(entries, evaluate, { minCoins: 3, maxRemoveShare: 0.6 });
    expect(result.selected).toEqual(['A', 'B', 'C']);
    // G (small profit, deep drawdown) hurts profit-to-drawdown more than the losing F.
    expect(result.removed.map((item) => item.symbol)).toEqual(['G', 'F', 'E', 'D']);
    expect(result.noDeals).toEqual(['H']);
    expect(result.best.score).toBeGreaterThan(result.all.score);
    expect(result.perCoin[0].symbol).toBe('G');
    expect(result.perCoin[0].contribution).toBeLessThan(0);
  });

  it('keeps everything when no removal helps', async () => {
    const entries = [coin('A', 5, 1), coin('B', 5, 1), coin('C', 5, 1)];
    const result = await vmSelectCoins(entries, evaluate, { minCoins: 1 });
    expect(result.selected).toEqual(['A', 'B', 'C']);
    expect(result.removed).toEqual([]);
  });

  it('is bounded, reports progress, yields and stops', async () => {
    const entries = Array.from({ length: 40 }, (_, index) => coin(`C${index}`, index % 7 === 0 ? -3 : 5, 1 + (index % 5)));
    let calls = 0;
    const counting = (subset) => {
      calls += 1;
      return evaluate(subset);
    };
    const progress = [];
    let yields = 0;
    const bounded = await vmSelectCoins(entries, counting, {
      minCoins: 3,
      maxEvaluations: 30,
      onProgress: (done, total) => progress.push([done, total]),
      yieldNow: async () => {
        yields += 1;
      },
    });
    expect(calls).toBeLessThanOrEqual(30);
    expect(bounded.evaluations).toBe(calls);
    expect(bounded.truncated).toBe(true);
    expect(progress.at(-1)).toEqual([30, 30]);
    // Unmeasured coins have no contribution and are listed after the measured ones.
    expect(bounded.perCoin.some((item) => item.contribution === null)).toBe(true);
    expect(bounded.perCoin[0].contribution).not.toBeNull();
    // Losing coins (net -3) are measured first and removed; the set improves.
    expect(bounded.removed.some((item) => Number(item.symbol.slice(1)) % 7 === 0)).toBe(true);
    expect(bounded.best.score).toBeGreaterThan(bounded.all.score);
    // Deterministic: the same inputs pick the same set.
    const again = await vmSelectCoins(entries, evaluate, { minCoins: 3, maxEvaluations: 30, yieldNow: async () => {} });
    expect(again.selected).toEqual(bounded.selected);

    let checks = 0;
    const stoppedEarly = await vmSelectCoins(entries, evaluate, {
      minCoins: 3,
      shouldStop: () => {
        checks += 1;
        return checks > 5;
      },
      yieldNow: async () => {},
    });
    expect(stoppedEarly).toBeNull();
  });

  it('keeps searching past helpful coins and removes many losers in batches', async () => {
    // Coins whose removal helps but whose own net / drawdown look middling (review case).
    const ratio = (subset) => {
      const profit = subset.reduce((sum, entry) => sum + entry.deals[0].n, 0);
      const risk = subset.reduce((sum, entry) => sum + entry.deals[0].m, 0);
      return { profit, risk, monthlyPercent: profit, drawdownPercent: risk, deals: subset.length };
    };
    const mixed = [
      ...Array.from({ length: 12 }, (_, i) => coin(`HI${i}`, 30, 5)),
      ...Array.from({ length: 12 }, (_, i) => coin(`LO${i}`, 1, 0.1)),
      ...Array.from({ length: 20 }, (_, i) => coin(`BAD${i}`, 3, 2)),
      ...Array.from({ length: 16 }, (_, i) => coin(`MID${i}`, 20, 4)),
    ];
    const picked = await vmSelectCoins(mixed, ratio, { minCoins: 5, maxRemoveShare: 0.9, yieldNow: async () => {} });
    expect(picked.removed.filter((item) => item.symbol.startsWith('BAD')).length).toBe(20);
    expect(picked.selected.some((symbol) => symbol.startsWith('BAD'))).toBe(false);

    // 150 coins with 110 losers: the batch removal clears them within the default budget.
    const many = [
      ...Array.from({ length: 110 }, (_, i) => coin(`L${i}`, -5, 4)),
      ...Array.from({ length: 40 }, (_, i) => coin(`W${i}`, 10, 2)),
    ];
    const cleaned = await vmSelectCoins(many, ratio, { minCoins: 5, maxRemoveShare: 0.9, yieldNow: async () => {} });
    expect(cleaned.selected.every((symbol) => symbol.startsWith('W'))).toBe(true);
    expect(cleaned.best.profit).toBeGreaterThan(0);
    expect(cleaned.evaluations).toBeLessThanOrEqual(200);
  });
});
