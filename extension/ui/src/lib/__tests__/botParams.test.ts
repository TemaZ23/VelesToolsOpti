import { describe, expect, it } from 'vitest';
import type { BotStrategy } from '../../api/backtestRunner';
import type { BotOrderDto, StrategyConditionDto } from '../../api/bots.dtos';
import type { ParamSpec, ParamValues } from '../../types/portfolioOptimizer';
import {
  baseValues,
  buildBotCreatePayloadFromStrategy,
  clampAndRound,
  defaultParamBounds,
  describeParamChange,
  diffFromBase,
  enforceGridConstraints,
  extractParams,
  formatParamValue,
  getAtPath,
  normalizeVolumes,
  paramsHash,
  patchStrategy,
  pathToId,
  prepareValues,
  roundTo,
  setAtPath,
} from '../botParams';

// ═══════════════════════════════════════════════════════════════════
// ФИКСТУРЫ
// ═══════════════════════════════════════════════════════════════════

const cond = (
  indicator: string,
  interval: string,
  operation: string | null,
  value: number | null,
  basic = false,
): StrategyConditionDto => ({
  type: 'INDICATOR',
  indicator,
  interval,
  basic,
  value,
  operation,
  closed: true,
  reverse: false,
});

const signal = (indicator: string, interval: string): StrategyConditionDto =>
  cond(indicator, interval, null, null, true);

const cmo90 = (): StrategyConditionDto => cond('CMO', 'THIRTY_MINUTES', 'GREATER', -90);
const volume1h = (): StrategyConditionDto => cond('VOLUME', 'ONE_HOUR', 'GREATER', 100000);

const ORDER_INDENTS = [0.3, 5, 10, 15, 20, 30, 40, 50, 65, 80];
const ORDER_VOLUMES = [2, 3, 4, 6, 8, 10, 12, 15, 18, 22];

const ORDER_CONDITIONS: Array<StrategyConditionDto[] | null> = [
  null,
  [cond('ADX', 'FIFTEEN_MINUTES', 'GREATER', 40), cmo90(), volume1h()],
  [cond('MFI', 'FIVE_MINUTES', 'LESS', 19), volume1h()],
  [signal('STOCHASTIC_LEVELS', 'ONE_MINUTE'), cmo90()],
  [cond('MFI', 'FIVE_MINUTES', 'LESS', 19)],
  [signal('KELTNER_CHANNEL', 'FIVE_MINUTES'), cmo90()],
  [signal('CCI_LEVELS', 'ONE_MINUTE'), cmo90()],
  [signal('CCI_LEVELS', 'ONE_MINUTE'), cmo90()],
  [cmo90(), volume1h()],
  [cmo90()],
];

/** Условие с неизвестным полем DTO, которое должно пережить патч без изменений. */
const withExtraField = (condition: StrategyConditionDto): StrategyConditionDto =>
  Object.assign(condition, { extraFlag: 'keep-me', nested: { a: 1 } });

/** Реалистичный DTO в стиле LONG_STRATEGY 22.X. */
const buildLongStrategy = (): BotStrategy => {
  const orders: BotOrderDto[] = ORDER_INDENTS.map((indent, index) => ({
    indent,
    volume: ORDER_VOLUMES[index],
    conditions: ORDER_CONDITIONS[index],
  }));

  const strategy: BotStrategy = {
    id: 777,
    name: 'LONG_STRATEGY 22.X',
    symbol: 'ADA/USDT',
    symbols: ['ADA/USDT'],
    pair: { exchange: 'BYBIT_FUTURES', type: 'FUTURES', from: 'ADA', to: 'USDT', symbol: 'ADAUSDT' },
    exchange: 'BYBIT_FUTURES',
    status: 'RUNNING',
    algorithm: 'LONG',
    conditions: [
      signal('RSI_LEVELS', 'ONE_MINUTE'),
      withExtraField(cond('CCI', 'ONE_HOUR', 'LESS', 80)),
      cond('CMO', 'THIRTY_MINUTES', 'GREATER', -80),
      cond('WILLIAMS_R', 'THIRTY_MINUTES', 'LESS', -80),
      signal('KELTNER_CHANNEL', 'THIRTY_MINUTES'),
      cond('ATR_PERCENT', 'FIVE_MINUTES', 'GREATER', 0.5),
      volume1h(),
    ],
    settings: {
      type: 'GRID',
      includePosition: true,
      indentType: 'PERCENT',
      baseOrder: null,
      orders,
    },
    profit: {
      type: 'PERCENT',
      currency: 'USDT',
      checkPnl: 0.3,
      conditions: [signal('BOLLINGER_BANDS', 'ONE_MINUTE'), cond('RSI', 'FIVE_MINUTES', 'GREATER', 70)],
    },
    stopLoss: {
      indent: null,
      termination: true,
      conditionalIndent: 2.5,
      conditionalIndentType: 'AVERAGE',
      conditions: [signal('MACD', 'FOUR_HOURS')],
    },
    deposit: { amount: 100, leverage: 10, marginType: 'CROSS', currency: 'USDT' },
    pullUp: 0.2,
    portion: null,
  };
  return Object.assign(strategy, { apiKey: 4242 });
};

const findParam = (params: ParamSpec[], id: string): ParamSpec => {
  const param = params.find((candidate) => candidate.id === id);
  if (!param) {
    throw new Error(`Параметр ${id} не найден`);
  }
  return param;
};

/** Плоская карта «путь → примитив» для сравнения DTO до и после патча. */
const flattenLeaves = (value: unknown, prefix = ''): Map<string, unknown> => {
  const leaves = new Map<string, unknown>();
  const walk = (current: unknown, path: string): void => {
    if (typeof current === 'object' && current !== null) {
      Object.entries(current).forEach(([key, child]) => {
        walk(child, path ? `${path}.${key}` : key);
      });
      return;
    }
    leaves.set(path, current);
  };
  walk(value, prefix);
  return leaves;
};

const changedLeaves = (before: unknown, after: unknown): string[] => {
  const left = flattenLeaves(before);
  const right = flattenLeaves(after);
  const keys = new Set([...left.keys(), ...right.keys()]);
  return [...keys].filter((key) => !Object.is(left.get(key), right.get(key))).sort();
};

const sumVolumes = (params: ParamSpec[], values: ParamValues): number =>
  roundTo(
    params.filter((param) => param.kind === 'orderVolume').reduce((total, param) => total + values[param.id], 0),
    2,
  );

const makeParam = (overrides: Partial<ParamSpec> & Pick<ParamSpec, 'id' | 'kind' | 'base'>): ParamSpec => ({
  path: [overrides.id],
  label: overrides.id,
  group: 'Тест',
  min: overrides.base - 10,
  max: overrides.base + 10,
  step: 1,
  decimals: 2,
  enabled: true,
  ...overrides,
});

// ═══════════════════════════════════════════════════════════════════
// ПУТИ И ОКРУГЛЕНИЕ
// ═══════════════════════════════════════════════════════════════════

describe('pathToId', () => {
  it('serializes string and numeric segments', () => {
    expect(pathToId(['settings', 'orders', 3, 'indent'])).toBe('settings.orders[3].indent');
    expect(pathToId(['conditions', 0, 'value'])).toBe('conditions[0].value');
    expect(pathToId(['pullUp'])).toBe('pullUp');
    expect(pathToId([])).toBe('');
  });
});

describe('getAtPath / setAtPath', () => {
  it('reads nested values and returns undefined for missing paths', () => {
    const strategy = buildLongStrategy();
    expect(getAtPath(strategy, ['settings', 'orders', 1, 'indent'])).toBe(5);
    expect(getAtPath(strategy, ['conditions', 1, 'value'])).toBe(80);
    expect(getAtPath(strategy, ['settings', 'baseOrder', 'indent'])).toBeUndefined();
    expect(getAtPath(strategy, ['settings', 'orders', 99, 'indent'])).toBeUndefined();
    expect(getAtPath(null, ['a'])).toBeUndefined();
  });

  it('mutates the existing field in place', () => {
    const strategy = buildLongStrategy();
    setAtPath(strategy, ['settings', 'orders', 2, 'volume'], 7.5);
    expect(strategy.settings?.orders?.[2].volume).toBe(7.5);
  });

  it('throws when the path does not exist', () => {
    const strategy = buildLongStrategy();
    expect(() => setAtPath(strategy, ['settings', 'baseOrder', 'indent'], 1)).toThrow(/settings\.baseOrder\.indent/);
    expect(() => setAtPath(strategy, ['settings', 'orders', 10, 'indent'], 1)).toThrow(/не найдено/);
    expect(() => setAtPath(strategy, ['unknownField'], 1)).toThrow(/unknownField/);
    expect(() => setAtPath(strategy, [], 1)).toThrow();
  });
});

describe('roundTo / formatParamValue', () => {
  it('rounds half away from zero without binary artefacts', () => {
    expect(roundTo(1.005, 2)).toBe(1.01);
    expect(roundTo(-1.005, 2)).toBe(-1.01);
    expect(roundTo(0.1 + 0.2, 2)).toBe(0.3);
    expect(roundTo(2.5, 0)).toBe(3);
    expect(roundTo(-2.5, 0)).toBe(-3);
    expect(roundTo(46.4999, 0)).toBe(46);
    expect(roundTo(123456.789, 1)).toBe(123456.8);
  });

  it('normalizes negative zero and keeps non-finite values', () => {
    expect(Object.is(roundTo(-0.001, 2), 0)).toBe(true);
    expect(roundTo(Number.NaN, 2)).toBeNaN();
    expect(roundTo(Number.POSITIVE_INFINITY, 2)).toBe(Number.POSITIVE_INFINITY);
  });

  it('formats values without trailing zeros', () => {
    expect(formatParamValue(0.30000000000000004, 2)).toBe('0.3');
    expect(formatParamValue(100000, 0)).toBe('100000');
    expect(formatParamValue(-90, 0)).toBe('-90');
  });
});

// ═══════════════════════════════════════════════════════════════════
// ГРАНИЦЫ
// ═══════════════════════════════════════════════════════════════════

describe('defaultParamBounds', () => {
  it('uses indicator-specific ranges clamped to the natural domain', () => {
    expect(defaultParamBounds('entryValue', 30, 'RSI')).toEqual({
      min: 20,
      max: 40,
      step: 2,
      decimals: 0,
      enabled: true,
    });
    expect(defaultParamBounds('entryValue', 95, 'RSI')).toMatchObject({ min: 85, max: 100 });
    expect(defaultParamBounds('orderConditionValue', 19, 'MFI')).toMatchObject({ min: 9, max: 29, step: 2 });
    expect(defaultParamBounds('tpConditionValue', 5, 'STOCHASTIC')).toMatchObject({ min: 0, max: 15, step: 2 });
    expect(defaultParamBounds('entryValue', 80, 'CCI')).toEqual({
      min: 20,
      max: 140,
      step: 10,
      decimals: 0,
      enabled: true,
    });
    expect(defaultParamBounds('entryValue', -480, 'CCI')).toMatchObject({ min: -500, max: -420 });
    expect(defaultParamBounds('entryValue', -80, 'CMO')).toEqual({
      min: -100,
      max: -50,
      step: 5,
      decimals: 0,
      enabled: true,
    });
    expect(defaultParamBounds('entryValue', -80, 'WILLIAMS_R')).toEqual({
      min: -95,
      max: -65,
      step: 3,
      decimals: 0,
      enabled: true,
    });
    expect(defaultParamBounds('entryValue', -5, 'WILLIAMS_R')).toMatchObject({ min: -20, max: 0 });
    expect(defaultParamBounds('orderConditionValue', 40, 'ADX')).toEqual({
      min: 28,
      max: 52,
      step: 2,
      decimals: 0,
      enabled: true,
    });
    expect(defaultParamBounds('entryValue', 0.5, 'ATR_PERCENT')).toEqual({
      min: 0.2,
      max: 0.8,
      step: 0.05,
      decimals: 2,
      enabled: true,
    });
    expect(defaultParamBounds('entryValue', 0.1, 'ATR_PERCENT')).toMatchObject({ min: 0, max: 0.4 });
  });

  it('disables volume thresholds by default', () => {
    expect(defaultParamBounds('entryValue', 100000, 'VOLUME')).toEqual({
      min: 25000,
      max: 400000,
      step: 25000,
      decimals: 0,
      enabled: false,
    });
    expect(defaultParamBounds('entryValue', 2, 'VOLUME')).toMatchObject({ step: 1, min: 1, max: 8 });
  });

  it('falls back to relative bounds for unknown indicators', () => {
    expect(defaultParamBounds('slConditionValue', 0.5, 'MACD')).toEqual({
      min: 0.25,
      max: 0.75,
      step: 0.05,
      decimals: 2,
      enabled: true,
    });
    expect(defaultParamBounds('entryValue', 0, 'SOMETHING')).toEqual({
      min: -1,
      max: 1,
      step: 0.01,
      decimals: 2,
      enabled: true,
    });
    expect(defaultParamBounds('entryValue', -20, null)).toMatchObject({ min: -30, max: -10, step: 2 });
  });

  it('computes grid, take-profit, stop-loss and pull-up bounds', () => {
    expect(defaultParamBounds('orderIndent', 5)).toEqual({ min: 3, max: 7, step: 0.4, decimals: 2, enabled: true });
    expect(defaultParamBounds('orderIndent', 0.3)).toEqual({
      min: 0,
      max: 0.92,
      step: 0.1,
      decimals: 2,
      enabled: true,
    });
    expect(defaultParamBounds('orderIndent', 0)).toMatchObject({ min: 0, max: 0.5, step: 0.1 });
    expect(defaultParamBounds('orderVolume', 2)).toEqual({ min: 1, max: 4, step: 0.2, decimals: 2, enabled: true });
    expect(defaultParamBounds('orderVolume', 22)).toMatchObject({ min: 11, max: 44, step: 2.2 });
    expect(defaultParamBounds('tpMinPnl', 0.3)).toEqual({ min: 0.1, max: 2, step: 0.1, decimals: 2, enabled: true });
    expect(defaultParamBounds('tpMinPnl', 1.5)).toMatchObject({ max: 4.5 });
    expect(defaultParamBounds('slConditionalIndent', 2.5)).toEqual({
      min: 0,
      max: 12.5,
      step: 0.5,
      decimals: 2,
      enabled: true,
    });
    expect(defaultParamBounds('slIndent', 30)).toEqual({ min: 15, max: 45, step: 3, decimals: 2, enabled: true });
    expect(defaultParamBounds('slIndent', 2)).toMatchObject({ min: 1, max: 3, step: 0.5 });
    expect(defaultParamBounds('pullUp', 0.2)).toEqual({ min: 0.1, max: 2, step: 0.1, decimals: 2, enabled: true });
    expect(defaultParamBounds('pullUp', 3)).toMatchObject({ max: 6 });
  });

  it('always keeps the base value inside [min, max] and representable', () => {
    expect(defaultParamBounds('entryValue', 105, 'RSI')).toMatchObject({ min: 95, max: 105 });
    expect(defaultParamBounds('tpMinPnl', 0.05)).toMatchObject({ min: 0.05 });
    const fractionalRsi = defaultParamBounds('entryValue', 46.5, 'RSI');
    expect(fractionalRsi.decimals).toBe(1);
    expect(fractionalRsi.min).toBe(36.5);
    expect(fractionalRsi.max).toBe(56.5);
    expect(defaultParamBounds('orderVolume', 2.125).decimals).toBe(3);
  });
});

// ═══════════════════════════════════════════════════════════════════
// ИЗВЛЕЧЕНИЕ
// ═══════════════════════════════════════════════════════════════════

describe('extractParams', () => {
  const strategy = buildLongStrategy();
  const params = extractParams(strategy);

  it('extracts every finite numeric field in the documented order', () => {
    expect(params).toHaveLength(42);
    expect(params.slice(0, 5).map((param) => param.id)).toEqual([
      'conditions[1].value',
      'conditions[2].value',
      'conditions[3].value',
      'conditions[5].value',
      'conditions[6].value',
    ]);
    expect(params.slice(5, 7).map((param) => param.id)).toEqual([
      'settings.orders[0].indent',
      'settings.orders[0].volume',
    ]);
    expect(params.slice(7, 12).map((param) => param.id)).toEqual([
      'settings.orders[1].indent',
      'settings.orders[1].volume',
      'settings.orders[1].conditions[0].value',
      'settings.orders[1].conditions[1].value',
      'settings.orders[1].conditions[2].value',
    ]);
    expect(params.slice(-4).map((param) => param.id)).toEqual([
      'profit.checkPnl',
      'profit.conditions[1].value',
      'stopLoss.conditionalIndent',
      'pullUp',
    ]);
    expect(new Set(params.map((param) => param.id)).size).toBe(params.length);
  });

  it('skips basic signals without values and null scalar fields', () => {
    const ids = params.map((param) => param.id);
    expect(ids).not.toContain('conditions[0].value');
    expect(ids).not.toContain('conditions[4].value');
    expect(ids).not.toContain('settings.orders[3].conditions[0].value');
    expect(ids).not.toContain('profit.conditions[0].value');
    expect(ids).not.toContain('stopLoss.indent');
    expect(ids).not.toContain('stopLoss.conditions[0].value');
  });

  it('assigns kinds, groups, order indexes, indicators and bases', () => {
    expect(findParam(params, 'conditions[1].value')).toMatchObject({
      kind: 'entryValue',
      group: 'Вход',
      base: 80,
      indicator: 'CCI',
      label: 'CCI 1 час < 80',
      enabled: true,
    });
    expect(findParam(params, 'settings.orders[0].indent')).toMatchObject({
      kind: 'orderIndent',
      group: 'Ордер 1',
      base: 0.3,
      orderIndex: 0,
      label: 'Ордер 1: отступ',
    });
    expect(findParam(params, 'settings.orders[9].volume')).toMatchObject({
      kind: 'orderVolume',
      group: 'Ордер 10',
      base: 22,
      orderIndex: 9,
      label: 'Ордер 10: объём',
    });
    const orderCondition = findParam(params, 'settings.orders[1].conditions[0].value');
    expect(orderCondition).toMatchObject({
      kind: 'orderConditionValue',
      group: 'Ордер 2',
      label: 'Ордер 2: ADX 15 мин > 40',
      indicator: 'ADX',
    });
    expect(orderCondition.orderIndex).toBeUndefined();
    expect(findParam(params, 'profit.checkPnl')).toMatchObject({
      kind: 'tpMinPnl',
      group: 'Тейк',
      base: 0.3,
      label: 'Тейк: мин. P&L',
    });
    expect(findParam(params, 'profit.conditions[1].value')).toMatchObject({
      kind: 'tpConditionValue',
      label: 'Тейк: RSI 5 мин > 70',
    });
    expect(findParam(params, 'stopLoss.conditionalIndent')).toMatchObject({
      kind: 'slConditionalIndent',
      group: 'Стоп',
      base: 2.5,
      min: 0,
      max: 12.5,
    });
    expect(findParam(params, 'pullUp')).toMatchObject({ kind: 'pullUp', group: 'Сетка', base: 0.2 });
    expect(findParam(params, 'settings.orders[1].indent').path).toEqual(['settings', 'orders', 1, 'indent']);
  });

  it('builds Russian condition labels and disables volume filters', () => {
    expect(findParam(params, 'conditions[2].value').label).toBe('CMO 30 мин > -80');
    expect(findParam(params, 'conditions[3].value').label).toBe('Williams %R 30 мин < -80');
    expect(findParam(params, 'conditions[5].value').label).toBe('ATR% 5 мин > 0.5');
    expect(findParam(params, 'conditions[6].value')).toMatchObject({ label: 'Объём 1 час > 100000', enabled: false });
    expect(findParam(params, 'settings.orders[2].conditions[0].value').label).toBe('Ордер 3: MFI 5 мин < 19');
  });

  it('handles operations, unknown names and a missing operation in labels', () => {
    const custom: BotStrategy = {
      id: 1,
      conditions: [
        cond('RSI', 'ONE_DAY', 'GREATER_OR_EQUAL', 55),
        cond('CMO', 'FOUR_HOURS', 'LESS_OR_EQUAL', -10),
        cond('MFI', 'ONE_MINUTE', 'EQUAL', 50),
        cond('SUPER_TREND', 'TWO_HOURS', 'CROSS_UP', 1.5),
        cond('ADX', 'FIFTEEN_MINUTES', null, 25),
      ],
    };
    expect(extractParams(custom).map((param) => param.label)).toEqual([
      'RSI 1 день ≥ 55',
      'CMO 4 часа ≤ -10',
      'MFI 1 мин = 50',
      'SUPER_TREND TWO_HOURS CROSS_UP 1.5',
      'ADX 15 мин 25',
    ]);
  });

  it('keeps basic conditions with numeric values but disables them', () => {
    const custom: BotStrategy = {
      id: 1,
      conditions: [cond('RSI_LEVELS', 'ONE_MINUTE', null, 0.5, true)],
    };
    const [param] = extractParams(custom);
    expect(param).toMatchObject({
      id: 'conditions[0].value',
      base: 0.5,
      enabled: false,
      label: 'Уровни RSI 1 мин 0.5',
    });
  });

  it('reads settings.conditions and a non-null baseOrder as order 1', () => {
    const custom: BotStrategy = {
      id: 1,
      conditions: [cond('CCI', 'ONE_HOUR', 'LESS', -100)],
      settings: {
        type: 'GRID',
        baseOrder: { indent: 0, volume: 40, conditions: [cond('RSI', 'ONE_HOUR', 'LESS', 30)] },
        orders: [
          { indent: 2, volume: 60, conditions: null },
          { indent: null, volume: null },
        ],
      },
    };
    // settings.conditions задаём не через литерал: в DTO это поле есть не всегда.
    if (custom.settings) {
      Object.assign(custom.settings, { conditions: [cond('ADX', 'ONE_HOUR', 'GREATER', 20)] });
    }
    const params = extractParams(custom);
    expect(params.map((param) => param.id)).toEqual([
      'conditions[0].value',
      'settings.conditions[0].value',
      'settings.baseOrder.indent',
      'settings.baseOrder.volume',
      'settings.baseOrder.conditions[0].value',
      'settings.orders[0].indent',
      'settings.orders[0].volume',
    ]);
    expect(findParam(params, 'settings.conditions[0].value')).toMatchObject({ kind: 'entryValue', group: 'Вход' });
    expect(findParam(params, 'settings.baseOrder.indent')).toMatchObject({ orderIndex: 0, group: 'Ордер 1' });
    expect(findParam(params, 'settings.orders[0].indent')).toMatchObject({ orderIndex: 1, group: 'Ордер 2' });
  });

  it('does not duplicate entry conditions when settings.conditions is the same array', () => {
    const shared = [cond('CCI', 'ONE_HOUR', 'LESS', -100)];
    const custom: BotStrategy = { id: 1, conditions: shared, settings: { type: 'GRID' } };
    if (custom.settings) {
      Object.assign(custom.settings, { conditions: shared });
    }
    expect(extractParams(custom).map((param) => param.id)).toEqual(['conditions[0].value']);
  });

  it('labels a fixed stop, a simple take-profit and tolerates missing sections', () => {
    const custom: BotStrategy = {
      id: 1,
      profit: { type: 'PERCENT', currency: 'USDT', checkPnl: 1.2, conditions: [] },
      stopLoss: {
        indent: 30,
        termination: null,
        conditionalIndent: null,
        conditions: null,
        conditionalIndentType: null,
      },
    };
    expect(extractParams(custom).map((param) => [param.id, param.label])).toEqual([
      ['profit.checkPnl', 'Тейк: тейк-профит'],
      ['stopLoss.indent', 'Стоп: отступ'],
    ]);
    expect(extractParams({ id: null })).toEqual([]);
  });

  it('skips non-finite numbers', () => {
    const custom: BotStrategy = {
      id: 1,
      pullUp: Number.NaN,
      conditions: [cond('RSI', 'ONE_HOUR', 'LESS', Number.POSITIVE_INFINITY)],
    };
    expect(extractParams(custom)).toEqual([]);
  });
});

// ═══════════════════════════════════════════════════════════════════
// ЗНАЧЕНИЯ
// ═══════════════════════════════════════════════════════════════════

describe('baseValues / clampAndRound', () => {
  const params = extractParams(buildLongStrategy());

  it('maps every param id to its base value', () => {
    const values = baseValues(params);
    expect(Object.keys(values)).toHaveLength(params.length);
    expect(values['conditions[1].value']).toBe(80);
    expect(values['settings.orders[9].indent']).toBe(80);
    expect(values.pullUp).toBe(0.2);
  });

  it('clamps to [min, max], rounds and fills missing values with base', () => {
    const values = baseValues(params);
    const input: ParamValues = {
      ...values,
      'conditions[1].value': 1000,
      'conditions[2].value': -79.6,
      'conditions[5].value': 0.12345,
      'settings.orders[1].indent': Number.NaN,
      'unknown.id': 5,
    };
    delete input.pullUp;
    const result = clampAndRound(params, input);
    expect(result).not.toBe(input);
    expect(result['conditions[1].value']).toBe(140);
    expect(result['conditions[2].value']).toBe(-80);
    expect(result['conditions[5].value']).toBe(0.2);
    expect(result['settings.orders[1].indent']).toBe(5);
    expect(result.pullUp).toBe(0.2);
    expect(result['unknown.id']).toBeUndefined();
    expect(Object.keys(result)).toHaveLength(params.length);
  });

  it('keeps rounded values inside bounds that are not on the decimal grid', () => {
    const param = makeParam({ id: 'x', kind: 'tpMinPnl', base: 0.5, min: 0.125, max: 0.925, decimals: 2 });
    expect(clampAndRound([param], { x: 5 }).x).toBe(0.92);
    expect(clampAndRound([param], { x: 0 }).x).toBe(0.13);
  });
});

describe('enforceGridConstraints', () => {
  const indent = (orderIndex: number, base: number, step = 1): ParamSpec =>
    makeParam({ id: `indent${orderIndex}`, kind: 'orderIndent', base, step, orderIndex, min: 0, max: 100 });

  it('makes indents strictly increasing and never moves the first order', () => {
    const params = [indent(0, 1), indent(1, 5, 0.4), indent(2, 10, 0.8)];
    const values: ParamValues = { indent0: 6, indent1: 5.5, indent2: 5.6 };
    const result = enforceGridConstraints(params, values);
    expect(result).not.toBe(values);
    expect(result.indent0).toBe(6);
    expect(result.indent1).toBe(6.4);
    expect(result.indent2).toBe(7.2);
    expect(values).toEqual({ indent0: 6, indent1: 5.5, indent2: 5.6 });
  });

  it('respects orderIndex rather than array order and leaves valid grids unchanged', () => {
    const params = [indent(2, 10), indent(0, 1), indent(1, 5)];
    expect(enforceGridConstraints(params, { indent0: 1, indent1: 5, indent2: 10 })).toEqual({
      indent0: 1,
      indent1: 5,
      indent2: 10,
    });
    expect(enforceGridConstraints(params, { indent0: 1, indent1: 5, indent2: 5 }).indent2).toBe(6);
  });

  it('accepts a gap of exactly 0.01 and does not constrain pairs already violated in the base bot', () => {
    expect(enforceGridConstraints([indent(0, 1), indent(1, 2)], { indent0: 1, indent1: 1.01 }).indent1).toBe(1.01);
    const equalBase = [indent(0, 3), indent(1, 3)];
    expect(enforceGridConstraints(equalBase, { indent0: 3, indent1: 3 })).toEqual({ indent0: 3, indent1: 3 });
  });

  it('keeps the grid monotonic in the realistic strategy', () => {
    const params = extractParams(buildLongStrategy());
    const values = { ...baseValues(params), 'settings.orders[1].indent': 0.2, 'settings.orders[3].indent': 9 };
    const result = enforceGridConstraints(params, values);
    const indents = params.filter((param) => param.kind === 'orderIndent').map((param) => result[param.id]);
    indents.slice(1).forEach((value, index) => {
      expect(value).toBeGreaterThanOrEqual(roundTo(indents[index] + 0.01, 2));
    });
    expect(result['settings.orders[1].indent']).toBe(0.7);
    expect(result['settings.orders[3].indent']).toBe(11.2);
  });
});

describe('normalizeVolumes', () => {
  const volume = (orderIndex: number, base: number, enabled = true): ParamSpec =>
    makeParam({ id: `v${orderIndex}`, kind: 'orderVolume', base, orderIndex, enabled, min: 0.01, max: 100 });

  it('scales enabled volumes so that the total equals the base total exactly', () => {
    const params = [volume(0, 33.33), volume(1, 33.33), volume(2, 33.34)];
    const result = normalizeVolumes(params, { v0: 40, v1: 33.33, v2: 33.34 });
    expect(sumVolumes(params, result)).toBe(100);
    expect(result.v0).toBeGreaterThan(result.v1);
    Object.values(result).forEach((value) => {
      expect(roundTo(value, 2)).toBe(value);
    });
  });

  it('puts the rounding remainder on the last enabled volume', () => {
    const params = [volume(0, 50), volume(1, 30), volume(2, 20)];
    const result = normalizeVolumes(params, { v0: 1, v1: 1, v2: 1 });
    expect(result.v0).toBe(33.33);
    expect(result.v1).toBe(33.33);
    expect(result.v2).toBe(33.34);
  });

  it('keeps disabled volumes and scales only enabled ones', () => {
    const params = [volume(0, 20, false), volume(1, 30), volume(2, 50)];
    const result = normalizeVolumes(params, { v0: 20, v1: 60, v2: 60 });
    expect(result.v0).toBe(20);
    expect(result.v1).toBe(40);
    expect(result.v2).toBe(40);
    expect(sumVolumes(params, result)).toBe(100);
  });

  it('never lets a volume drop below 0.01', () => {
    const params = [volume(0, 50), volume(1, 50)];
    const result = normalizeVolumes(params, { v0: 0.000001, v1: 100000 });
    expect(result.v0).toBe(0.01);
    expect(result.v1).toBe(99.99);
    expect(sumVolumes(params, result)).toBe(100);
  });

  it('returns an unchanged copy when nothing can be normalized', () => {
    const disabled = [volume(0, 50, false), volume(1, 50, false)];
    const input = { v0: 10, v1: 10 };
    const unchanged = normalizeVolumes(disabled, input);
    expect(unchanged).toEqual(input);
    expect(unchanged).not.toBe(input);

    const zero = [volume(0, 50), volume(1, 50)];
    expect(normalizeVolumes(zero, { v0: 0, v1: 0 })).toEqual({ v0: 0, v1: 0 });
    const disabledTakesAll = [volume(0, 50, false), volume(1, 50)];
    expect(normalizeVolumes(disabledTakesAll, { v0: 100, v1: 5 })).toEqual({ v0: 100, v1: 5 });
  });

  it('keeps base volumes of the realistic strategy intact', () => {
    const params = extractParams(buildLongStrategy());
    const base = baseValues(params);
    expect(normalizeVolumes(params, base)).toEqual(base);
  });
});

describe('prepareValues', () => {
  const params = extractParams(buildLongStrategy());

  it('is a no-op for base values of a consistent strategy', () => {
    const base = baseValues(params);
    expect(prepareValues(params, base)).toEqual(base);
  });

  it('clamps, fixes the grid and normalizes volumes in one pass', () => {
    const values: ParamValues = {
      ...baseValues(params),
      'settings.orders[2].indent': 1,
      'settings.orders[4].volume': 16,
      'conditions[1].value': 81.4,
    };
    const result = prepareValues(params, values);
    expect(result['conditions[1].value']).toBe(81);
    // Отступ сначала ограничен min = 6, затем подтянут выше предыдущего (5) — 6 уже корректно.
    expect(result['settings.orders[2].indent']).toBe(6);
    expect(sumVolumes(params, result)).toBe(100);
    expect(result['settings.orders[4].volume']).toBeGreaterThan(result['settings.orders[3].volume']);
  });
});

// ═══════════════════════════════════════════════════════════════════
// ХЕШ, ПАТЧ, РАЗНИЦА
// ═══════════════════════════════════════════════════════════════════

describe('paramsHash', () => {
  const params = extractParams(buildLongStrategy());
  const base = baseValues(params);

  it('has a stable p-prefixed FNV-1a format', () => {
    const hash = paramsHash(params, base);
    expect(hash).toMatch(/^p[0-9a-f]{8}$/);
    expect(paramsHash(params, base)).toBe(hash);
  });

  it('does not depend on the order of params or value keys', () => {
    const reversedParams = [...params].reverse();
    const reversedValues = Object.fromEntries(Object.entries(base).reverse());
    expect(paramsHash(reversedParams, reversedValues)).toBe(paramsHash(params, base));
  });

  it('changes when a value changes and ignores sub-decimal noise', () => {
    const changed = { ...base, 'conditions[1].value': 70 };
    expect(paramsHash(params, changed)).not.toBe(paramsHash(params, base));
    const noisy = { ...base, 'conditions[1].value': 80.0000001 };
    expect(paramsHash(params, noisy)).toBe(paramsHash(params, base));
  });

  it('treats a missing value as the base value', () => {
    const partial: ParamValues = { ...base };
    delete partial.pullUp;
    expect(paramsHash(params, partial)).toBe(paramsHash(params, base));
  });
});

describe('patchStrategy', () => {
  it('is idempotent for base values and does not mutate the input', () => {
    const strategy = buildLongStrategy();
    const snapshot = structuredClone(strategy);
    const params = extractParams(strategy);
    const patched = patchStrategy(strategy, params, baseValues(params));
    expect(patched).not.toBe(strategy);
    expect(patched).toEqual(snapshot);
    expect(changedLeaves(snapshot, patched)).toEqual([]);
    expect(strategy).toEqual(snapshot);
  });

  it('touches exactly one path when a single non-grid value changes', () => {
    const strategy = buildLongStrategy();
    const params = extractParams(strategy);
    const values = { ...baseValues(params), 'conditions[1].value': 60 };
    const patched = patchStrategy(strategy, params, values);
    expect(changedLeaves(strategy, patched)).toEqual(['conditions.1.value']);
    expect(patched.conditions?.[1].value).toBe(60);
    expect(getAtPath(patched, ['conditions', 1, 'extraFlag'])).toBe('keep-me');
    expect(getAtPath(patched, ['conditions', 1, 'nested', 'a'])).toBe(1);
  });

  it('writes clamped and normalized values for grid changes only into grid fields', () => {
    const strategy = buildLongStrategy();
    const params = extractParams(strategy);
    const values = { ...baseValues(params), 'settings.orders[9].volume': 30, pullUp: 50 };
    const patched = patchStrategy(strategy, params, values);
    const changed = changedLeaves(strategy, patched);
    expect(changed).toContain('pullUp');
    expect(changed.every((path) => path === 'pullUp' || /^settings\.orders\.\d+\.volume$/.test(path))).toBe(true);
    expect(patched.pullUp).toBe(2);
    const volumes = (patched.settings?.orders ?? []).map((order) => order.volume ?? 0);
    expect(
      roundTo(
        volumes.reduce((total, value) => total + value, 0),
        2,
      ),
    ).toBe(100);
  });

  it('leaves base-equal values untouched even when the strategy holds a different value', () => {
    const strategy = buildLongStrategy();
    const params = extractParams(strategy);
    setAtPath(strategy, ['pullUp'], 0.25);
    const patched = patchStrategy(strategy, params, baseValues(params));
    expect(patched.pullUp).toBe(0.25);
  });

  it('throws when a param path is missing in the strategy', () => {
    const strategy = buildLongStrategy();
    const params = extractParams(strategy);
    const foreign = makeParam({ id: 'settings.baseOrder.indent', kind: 'tpMinPnl', base: 1 });
    foreign.path = ['settings', 'baseOrder', 'indent'];
    expect(() => patchStrategy(strategy, [...params, foreign], { [foreign.id]: 2 })).toThrow(/settings\.baseOrder/);
  });
});

describe('diffFromBase', () => {
  const params = extractParams(buildLongStrategy());

  it('lists ids whose rounded value differs from base, in params order', () => {
    const values = {
      ...baseValues(params),
      pullUp: 0.4,
      'conditions[1].value': 80.2,
      'conditions[3].value': -86,
    };
    expect(diffFromBase(params, values)).toEqual(['conditions[3].value', 'pullUp']);
    expect(diffFromBase(params, baseValues(params))).toEqual([]);
  });
});

describe('describeParamChange', () => {
  const params = extractParams(buildLongStrategy());

  it('describes condition and percentage changes', () => {
    expect(describeParamChange(findParam(params, 'conditions[1].value'), 60)).toBe('CCI 1 час < 80 → 60');
    expect(describeParamChange(findParam(params, 'settings.orders[3].indent'), 13.8)).toBe(
      'Ордер 4: отступ 15% → 13.8%',
    );
    expect(describeParamChange(findParam(params, 'conditions[5].value'), 0.55)).toBe('ATR% 5 мин > 0.5 → 0.55');
    expect(describeParamChange(findParam(params, 'pullUp'), 0.30000000000000004)).toBe('Подтяжка сетки 0.2% → 0.3%');
    expect(describeParamChange(findParam(params, 'settings.orders[1].conditions[1].value'), -85)).toBe(
      'Ордер 2: CMO 30 мин > -90 → -85',
    );
  });

  it('marks unchanged values', () => {
    expect(describeParamChange(findParam(params, 'conditions[1].value'), 80)).toBe('CCI 1 час < 80 (без изменений)');
    expect(describeParamChange(findParam(params, 'profit.checkPnl'), 0.3)).toBe('Тейк: мин. P&L 0.3% (без изменений)');
  });
});

// ═══════════════════════════════════════════════════════════════════
// СОЗДАНИЕ БОТА
// ═══════════════════════════════════════════════════════════════════

describe('buildBotCreatePayloadFromStrategy', () => {
  it('builds a create payload from the patched strategy with deep copies', () => {
    const strategy = buildLongStrategy();
    const params = extractParams(strategy);
    const patched = patchStrategy(strategy, params, { ...baseValues(params), 'conditions[1].value': 60 });
    const payload = buildBotCreatePayloadFromStrategy(patched, '  LONG OPT abc123  ');

    expect(payload).toEqual({
      algorithm: 'LONG',
      apiKey: 4242,
      conditions: patched.conditions,
      deposit: patched.deposit,
      exchange: 'BYBIT_FUTURES',
      id: null,
      name: 'LONG OPT abc123',
      portion: null,
      profit: patched.profit,
      pullUp: 0.2,
      settings: patched.settings,
      stopLoss: patched.stopLoss,
      symbols: ['ADA/USDT'],
      termination: null,
    });
    expect(payload.conditions[1].value).toBe(60);
    expect(payload.settings).not.toBe(patched.settings);
    expect(payload.conditions).not.toBe(patched.conditions);
    expect(payload.deposit).not.toBe(patched.deposit);

    payload.settings?.orders?.splice(0, 1);
    expect(patched.settings?.orders).toHaveLength(10);
  });

  it('falls back to pair exchange, single symbol and empty conditions', () => {
    const strategy = buildLongStrategy();
    strategy.exchange = null;
    strategy.symbols = null;
    strategy.conditions = null;
    strategy.profit = null;
    strategy.stopLoss = undefined;
    const payload = buildBotCreatePayloadFromStrategy(strategy, 'Bot');
    expect(payload.exchange).toBe('BYBIT_FUTURES');
    expect(payload.symbols).toEqual(['ADA/USDT']);
    expect(payload.conditions).toEqual([]);
    expect(payload.profit).toBeNull();
    expect(payload.stopLoss).toBeNull();
  });

  it('accepts a numeric string apiKey', () => {
    const strategy = Object.assign(buildLongStrategy(), { apiKey: '15' });
    expect(buildBotCreatePayloadFromStrategy(strategy, 'Bot').apiKey).toBe(15);
  });

  it('throws Russian errors for missing required fields', () => {
    const withoutKey: BotStrategy = { ...buildLongStrategy() };
    Reflect.deleteProperty(withoutKey, 'apiKey');
    expect(() => buildBotCreatePayloadFromStrategy(withoutKey, 'Bot')).toThrow(/API-ключ/);
    expect(() =>
      buildBotCreatePayloadFromStrategy(Object.assign(buildLongStrategy(), { apiKey: null }), 'Bot'),
    ).toThrow(/API-ключ/);
    expect(() => buildBotCreatePayloadFromStrategy(buildLongStrategy(), '   ')).toThrow(/название/);

    const noAlgorithm = buildLongStrategy();
    noAlgorithm.algorithm = null;
    expect(() => buildBotCreatePayloadFromStrategy(noAlgorithm, 'Bot')).toThrow(/алгоритм/);

    const noExchange = buildLongStrategy();
    noExchange.exchange = null;
    noExchange.pair = null;
    expect(() => buildBotCreatePayloadFromStrategy(noExchange, 'Bot')).toThrow(/биржа/);

    const noDeposit = buildLongStrategy();
    noDeposit.deposit = null;
    expect(() => buildBotCreatePayloadFromStrategy(noDeposit, 'Bot')).toThrow(/депозит/);
  });
});
