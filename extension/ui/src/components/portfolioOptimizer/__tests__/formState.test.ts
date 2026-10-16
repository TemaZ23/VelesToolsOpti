import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { BotStrategy } from '../../../api/backtestRunner';
import type { StrategyConditionDto } from '../../../api/bots.dtos';
import { extractParams } from '../../../lib/botParams';
import {
  DEFAULT_OBJECTIVE,
  DEFAULT_SCHEDULER_CONFIG,
  DEFAULT_SEARCH_CONFIG,
  normalizePortfolioConfig,
  pickScreenBasket,
} from '../../../services/portfolioOptimizer';
import type { ParamSpec } from '../../../types/portfolioOptimizer';
import {
  applyParamOverrides,
  basketFromSymbols,
  buildPortfolioConfig,
  buildRunSettings,
  collectParamOverrides,
  countParamValues,
  createDefaultFormState,
  DEFAULT_BLOCK,
  DEFAULT_MAKER_COMMISSION,
  DEFAULT_TAKER_COMMISSION,
  defaultPeriod,
  disableGrid,
  enableAllThresholds,
  FORM_LIMITS,
  isDateInput,
  minParamStep,
  PORTFOLIO_FORM_STORAGE_KEY,
  type PortfolioOptimizerFormState,
  parseBasket,
  readPortfolioFormState,
  resolveScreenBasket,
  resolveScreenSize,
  sameParamOverrides,
  sanitizeFormState,
  setParamEnabled,
  setParamsEnabledWhere,
  toRunPeriod,
  updateParamBound,
  validateRunDraft,
  writePortfolioFormState,
} from '../formState';

const NOW = new Date('2026-10-12T10:00:00.000Z');

const condition = (indicator: string, value: number | null): StrategyConditionDto => ({
  type: 'INDICATOR',
  indicator,
  interval: 'ONE_HOUR',
  basic: false,
  value,
  operation: 'LESS',
  closed: true,
  reverse: false,
});

const createStrategy = (): BotStrategy => ({
  id: 77,
  name: 'LONG 22',
  symbols: ['ADA/USDT', 'XRP/USDT'],
  exchange: 'BYBIT_FUTURES',
  algorithm: 'LONG',
  conditions: [condition('RSI', 40), condition('CCI', -100), condition('VOLUME', 1000)],
  settings: {
    type: 'CUSTOM',
    baseOrder: null,
    orders: [
      { indent: 0, volume: 40, conditions: [] },
      { indent: 2, volume: 60, conditions: [condition('MFI', 20)] },
    ],
  },
  profit: { type: 'ALL', currency: 'USDT', checkPnl: 1, conditions: [condition('RSI', 70)] },
  deposit: { amount: 100, leverage: 10, marginType: 'CROSS', currency: 'USDT' },
  stopLoss: null,
  pullUp: 0.2,
  portion: null,
});

const createParams = (): ParamSpec[] => extractParams(createStrategy());

const paramById = (params: ParamSpec[], id: string): ParamSpec => {
  const param = params.find((item) => item.id === id);
  if (!param) {
    throw new Error(`Нет параметра ${id}`);
  }
  return param;
};

const RSI_ID = 'conditions[0].value';
const VOLUME_ID = 'conditions[2].value';
const INDENT_ID = 'settings.orders[1].indent';

const formWith = (patch: Partial<PortfolioOptimizerFormState> = {}): PortfolioOptimizerFormState => ({
  ...createDefaultFormState(NOW),
  ...patch,
});

describe('formState: даты и период', () => {
  it('по умолчанию берёт последние 5 лет до сегодняшней даты (UTC)', () => {
    expect(defaultPeriod(NOW)).toEqual({ periodFrom: '2021-10-12', periodTo: '2026-10-12' });
  });

  it('проверяет формат и существование даты', () => {
    expect(isDateInput('2024-02-29')).toBe(true);
    expect(isDateInput('2023-02-29')).toBe(false);
    expect(isDateInput('2024-13-01')).toBe(false);
    expect(isDateInput('12.10.2026')).toBe(false);
    expect(isDateInput('')).toBe(false);
    expect(isDateInput(20240101)).toBe(false);
  });

  it('строит ISO-период с начала первого по конец последнего дня', () => {
    expect(toRunPeriod('2021-12-06', '2026-10-12')).toEqual({
      periodFrom: '2021-12-06T00:00:00.000Z',
      periodTo: '2026-10-12T23:59:59.999Z',
    });
    expect(toRunPeriod('2024-05-01', '2024-05-01')).not.toBeNull();
    expect(toRunPeriod('2024-05-02', '2024-05-01')).toBeNull();
    expect(toRunPeriod('bad', '2024-05-01')).toBeNull();
  });

  it('buildRunSettings оставляет пустой период при некорректных датах', () => {
    const settings = buildRunSettings(formWith({ periodFrom: '2025-01-02', periodTo: '2025-01-01' }));
    expect(settings.periodFrom).toBe('');
    expect(settings.periodTo).toBe('');
    expect(settings.makerCommission).toBe(DEFAULT_MAKER_COMMISSION);
    expect(settings.takerCommission).toBe(DEFAULT_TAKER_COMMISSION);
    expect(settings.includeWicks).toBe(true);
  });
});

describe('formState: корзина', () => {
  it('разбирает тикеры с разными разделителями, убирает котировку и повторы', () => {
    expect(parseBasket('ada, XRP  sol/usdt;\nDOGE,ada , 1000PEPE/USDT')).toEqual([
      'ADA',
      'XRP',
      'SOL',
      'DOGE',
      '1000PEPE',
    ]);
    expect(parseBasket('   ,, ; ')).toEqual([]);
  });

  it('берёт базовые монеты из символов бота', () => {
    expect(basketFromSymbols(['BTC/USDT', null, 'ETH/USDT', undefined, 'BTC/USDT'])).toEqual(['BTC', 'ETH']);
  });

  it('размер отборочной корзины: по умолчанию 10, в пределах корзины', () => {
    expect(resolveScreenSize(0, null)).toBe(0);
    expect(resolveScreenSize(25, null)).toBe(10);
    expect(resolveScreenSize(6, null)).toBe(6);
    expect(resolveScreenSize(25, 4)).toBe(4);
    expect(resolveScreenSize(25, 0)).toBe(1);
    expect(resolveScreenSize(5, 40)).toBe(5);
    expect(resolveScreenSize(25, 3.7)).toBe(3);
  });

  it('отборочная корзина детерминирована зерном и совпадает с pickScreenBasket', () => {
    const basket = Array.from({ length: 30 }, (_, index) => `C${index}`);
    const first = resolveScreenBasket({ screenSize: 5, screenSeed: 3 }, basket);
    expect(first).toEqual(pickScreenBasket(basket, 5, 3));
    expect(first).toHaveLength(5);
    expect(resolveScreenBasket({ screenSize: 5, screenSeed: 3 }, basket)).toEqual(first);
    expect(resolveScreenBasket({ screenSize: null, screenSeed: 1 }, basket)).toHaveLength(10);
  });
});

describe('formState: значения по умолчанию и сохранение', () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it('значения по умолчанию соответствуют спецификации', () => {
    const form = createDefaultFormState(NOW);
    expect(form.botId).toBeNull();
    expect(form.block).toBe(DEFAULT_BLOCK);
    expect(form.makerCommission).toBe(0.02);
    expect(form.takerCommission).toBe(0.055);
    expect(form.includeWicks).toBe(true);
    expect(form.objective).toEqual(DEFAULT_OBJECTIVE);
    expect(form.search).toEqual(DEFAULT_SEARCH_CONFIG);
    expect(form.scheduler).toEqual(DEFAULT_SCHEDULER_CONFIG);
    expect(form.periodFrom).toBe('2021-10-12');
    expect(form.periodTo).toBe('2026-10-12');
    expect(form.backtestDeposit).toBeNull();
  });

  it('депозит в бэктестах: положительное число сохраняется, остальное — «как в боте»', () => {
    expect(sanitizeFormState({ backtestDeposit: 37.5 }, NOW).backtestDeposit).toBe(37.5);
    expect(sanitizeFormState({ backtestDeposit: 0 }, NOW).backtestDeposit).toBeNull();
    expect(sanitizeFormState({ backtestDeposit: -5 }, NOW).backtestDeposit).toBeNull();
    expect(sanitizeFormState({ backtestDeposit: '37.5' }, NOW).backtestDeposit).toBeNull();
    expect(sanitizeFormState({}, NOW).backtestDeposit).toBeNull();
  });

  it('сохраняет и читает форму под ключом veles_portfolio_optimizer_form', () => {
    const form = formWith({
      botId: 42,
      basketText: 'ADA, XRP',
      screenSize: 2,
      screenSeed: 5,
      baseGroupId: 'group-1',
      periodFrom: '2022-01-01',
      periodTo: '2024-01-01',
      block: 12,
      objective: { kind: 'profitCapped', folds: 4, maxMaeCap: 500, minDeals: 10 },
      search: { ...DEFAULT_SEARCH_CONFIG, generations: 3, screenTolerance: 0.1 },
      scheduler: { ...DEFAULT_SCHEDULER_CONFIG, maxInFlight: 5 },
      paramOverrides: { botId: 42, params: { [RSI_ID]: { enabled: false, min: 30, max: 46, step: 1 } } },
    });
    expect(writePortfolioFormState(form)).toBe(true);
    const raw = window.localStorage.getItem(PORTFOLIO_FORM_STORAGE_KEY);
    expect(raw).not.toBeNull();
    expect(JSON.parse(raw ?? '{}').version).toBe(1);
    expect(readPortfolioFormState(NOW)).toEqual(form);
  });

  it('без сохранённых настроек, с повреждённым JSON или другой версией — значения по умолчанию', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    expect(readPortfolioFormState(NOW)).toEqual(createDefaultFormState(NOW));
    window.localStorage.setItem(PORTFOLIO_FORM_STORAGE_KEY, '{oops');
    expect(readPortfolioFormState(NOW)).toEqual(createDefaultFormState(NOW));
    expect(warn).toHaveBeenCalledTimes(1);
    window.localStorage.setItem(PORTFOLIO_FORM_STORAGE_KEY, JSON.stringify({ version: 99, botId: 5 }));
    expect(readPortfolioFormState(NOW)).toEqual(createDefaultFormState(NOW));
  });

  it('sanitizeFormState заменяет некорректные поля и ограничивает диапазоны', () => {
    const form = sanitizeFormState(
      {
        botId: -3,
        basketText: 7,
        screenSize: 0,
        screenSeed: 'x',
        baseGroupId: '   ',
        periodFrom: '2025-05-01',
        periodTo: '2024-01-01',
        makerCommission: -1,
        takerCommission: 'a',
        includeWicks: 'yes',
        block: 5000,
        objective: { kind: 'unknown', folds: 99, maxMaeCap: 'none', minDeals: -5 },
        search: { generations: 0, mutationsMin: 4, mutationsMax: 2, screenTolerance: 3, maxBacktests: 1e9 },
        scheduler: { maxInFlight: 50, submitDelaySeconds: 1, pollIntervalSeconds: null },
        paramOverrides: { botId: 1, params: { a: { enabled: true, min: 5, max: 1, step: 1 }, b: 'x' } },
      },
      NOW,
    );
    const defaults = createDefaultFormState(NOW);
    expect(form.botId).toBeNull();
    expect(form.basketText).toBe('');
    expect(form.screenSize).toBe(FORM_LIMITS.screenSize.min);
    expect(form.screenSeed).toBe(defaults.screenSeed);
    expect(form.baseGroupId).toBeNull();
    // Начало позже конца — берётся период по умолчанию целиком.
    expect(form.periodFrom).toBe(defaults.periodFrom);
    expect(form.periodTo).toBe(defaults.periodTo);
    expect(form.makerCommission).toBe(0);
    expect(form.takerCommission).toBe(defaults.takerCommission);
    expect(form.includeWicks).toBe(true);
    expect(form.block).toBe(FORM_LIMITS.block.max);
    expect(form.objective).toEqual({
      kind: DEFAULT_OBJECTIVE.kind,
      folds: FORM_LIMITS.folds.max,
      maxMaeCap: null,
      minDeals: 0,
    });
    expect(form.search.generations).toBe(FORM_LIMITS.generations.min);
    expect(form.search.mutationsMin).toBe(4);
    expect(form.search.mutationsMax).toBe(4);
    expect(form.search.screenTolerance).toBe(FORM_LIMITS.screenTolerance.max);
    expect(form.search.maxBacktests).toBe(FORM_LIMITS.maxBacktests.max);
    expect(form.search.candidatesPerGeneration).toBe(DEFAULT_SEARCH_CONFIG.candidatesPerGeneration);
    expect(form.scheduler.maxInFlight).toBe(FORM_LIMITS.maxInFlight.max);
    expect(form.scheduler.submitDelaySeconds).toBe(FORM_LIMITS.submitDelaySeconds.min);
    expect(form.scheduler.pollIntervalSeconds).toBe(DEFAULT_SCHEDULER_CONFIG.pollIntervalSeconds);
    expect(form.paramOverrides).toEqual({ botId: 1, params: {} });
  });

  it('sanitizeFormState не-объекта возвращает значения по умолчанию', () => {
    expect(sanitizeFormState(null, NOW)).toEqual(createDefaultFormState(NOW));
    expect(sanitizeFormState('form', NOW)).toEqual(createDefaultFormState(NOW));
  });
});

describe('formState: параметры', () => {
  it('правки сохраняются только для отличающихся параметров и применяются к тому же боту', () => {
    const defaults = createParams();
    const edited = updateParamBound(setParamEnabled(defaults, RSI_ID, false), RSI_ID, 'min', 34);
    const overrides = collectParamOverrides(77, edited, defaults);
    expect(Object.keys(overrides.params)).toEqual([RSI_ID]);
    expect(overrides.params[RSI_ID]).toMatchObject({ enabled: false, min: 34 });

    const restored = applyParamOverrides(defaults, overrides, 77);
    expect(paramById(restored, RSI_ID)).toMatchObject({ enabled: false, min: 34 });
    expect(restored).not.toBe(defaults);
    // Другой бот — правки не применяются.
    expect(applyParamOverrides(defaults, overrides, 78)).toEqual(defaults);
    expect(applyParamOverrides(defaults, null, 77)).toEqual(defaults);
  });

  it('правка не применяется, если базовое значение вышло за сохранённый диапазон', () => {
    const defaults = createParams();
    const overrides = { botId: 77, params: { [RSI_ID]: { enabled: true, min: 41, max: 50, step: 2 } } };
    expect(paramById(applyParamOverrides(defaults, overrides, 77), RSI_ID)).toEqual(paramById(defaults, RSI_ID));
  });

  it('sameParamOverrides сравнивает по содержимому', () => {
    const a = { botId: 1, params: { x: { enabled: true, min: 1, max: 3, step: 1 } } };
    expect(sameParamOverrides(a, { botId: 1, params: { x: { enabled: true, min: 1, max: 3, step: 1 } } })).toBe(true);
    expect(sameParamOverrides(a, { botId: 2, params: a.params })).toBe(false);
    expect(sameParamOverrides(a, { botId: 1, params: { x: { enabled: false, min: 1, max: 3, step: 1 } } })).toBe(false);
    expect(sameParamOverrides(a, { botId: 1, params: { y: a.params.x } })).toBe(false);
    expect(sameParamOverrides(a, { botId: 1, params: {} })).toBe(false);
    expect(sameParamOverrides(null, null)).toBe(true);
    expect(sameParamOverrides(a, null)).toBe(false);
  });

  it('границы: минимум не выше базы, максимум не ниже, шаг не меньше единицы знака', () => {
    const params = createParams();
    const rsi = paramById(params, RSI_ID);
    expect(paramById(updateParamBound(params, RSI_ID, 'min', 55), RSI_ID).min).toBe(rsi.base);
    expect(paramById(updateParamBound(params, RSI_ID, 'min', 31.4), RSI_ID).min).toBe(31);
    expect(paramById(updateParamBound(params, RSI_ID, 'max', 10), RSI_ID).max).toBe(rsi.base);
    expect(paramById(updateParamBound(params, RSI_ID, 'max', 60), RSI_ID).max).toBe(60);
    expect(paramById(updateParamBound(params, RSI_ID, 'step', 0), RSI_ID).step).toBe(1);
    const indent = paramById(params, INDENT_ID);
    expect(paramById(updateParamBound(params, INDENT_ID, 'step', 0.001), INDENT_ID).step).toBe(
      minParamStep(indent.decimals),
    );
    // null и нечисловые значения ничего не меняют.
    expect(updateParamBound(params, RSI_ID, 'min', null)).toBe(params);
    expect(updateParamBound(params, RSI_ID, 'min', Number.NaN)).toBe(params);
    // Остальные параметры не затрагиваются.
    const changed = updateParamBound(params, RSI_ID, 'max', 60);
    expect(changed.filter((param, index) => param !== params[index]).map((param) => param.id)).toEqual([RSI_ID]);
  });

  it('включение/выключение параметров и групповые кнопки', () => {
    const params = createParams();
    expect(paramById(params, VOLUME_ID).enabled).toBe(false);

    const thresholds = enableAllThresholds(params);
    for (const param of thresholds) {
      if (['entryValue', 'orderConditionValue', 'tpConditionValue', 'slConditionValue'].includes(param.kind)) {
        expect(param.enabled).toBe(true);
      }
    }
    expect(paramById(thresholds, VOLUME_ID).enabled).toBe(true);

    const noGrid = disableGrid(params);
    for (const param of noGrid) {
      if (['orderIndent', 'orderVolume', 'pullUp'].includes(param.kind)) {
        expect(param.enabled).toBe(false);
      } else {
        expect(param.enabled).toBe(paramById(params, param.id).enabled);
      }
    }

    const groupOff = setParamsEnabledWhere(params, (param) => param.group === 'Вход', false);
    expect(groupOff.filter((param) => param.group === 'Вход').every((param) => !param.enabled)).toBe(true);
    // Неизменённые параметры сохраняют идентичность объектов.
    expect(setParamEnabled(params, RSI_ID, true)).toEqual(params);
    expect(setParamEnabled(params, RSI_ID, true)[0]).toBe(params[0]);
  });

  it('countParamValues — число точек сетки поиска', () => {
    expect(countParamValues({ min: 30, max: 50, step: 2 })).toBe(11);
    expect(countParamValues({ min: 0.1, max: 0.3, step: 0.1 })).toBe(3);
    expect(countParamValues({ min: 5, max: 5, step: 1 })).toBe(1);
    expect(countParamValues({ min: 5, max: 1, step: 1 })).toBe(1);
    expect(countParamValues({ min: 0, max: 1, step: 0 })).toBe(1);
  });
});

describe('formState: конфигурация запуска', () => {
  it('собирает конфигурацию, которую принимает normalizePortfolioConfig', () => {
    const params = createParams();
    const form = formWith({
      botId: 77,
      basketText: 'ada xrp sol doge link',
      screenSize: 3,
      screenSeed: 2,
      periodFrom: '2022-01-01',
      periodTo: '2024-01-01',
      baseGroupId: 'g1',
    });
    const config = buildPortfolioConfig({ form, botId: 77, botName: 'LONG 22', params });
    expect(config.basket).toEqual(['ADA', 'XRP', 'SOL', 'DOGE', 'LINK']);
    expect(config.screenBasket).toEqual(pickScreenBasket(config.basket, 3, 2));
    expect(config.run).toEqual({
      periodFrom: '2022-01-01T00:00:00.000Z',
      periodTo: '2024-01-01T23:59:59.999Z',
      makerCommission: 0.02,
      takerCommission: 0.055,
      includeWicks: true,
    });
    expect(config.block).toBe(DEFAULT_BLOCK);
    expect(config.baseGroupId).toBe('g1');
    expect(config.params).toEqual(params);
    expect(config.params[0]).not.toBe(params[0]);
    expect(config.search).not.toBe(form.search);

    expect(config.backtestDeposit).toBeNull();

    const normalized = normalizePortfolioConfig(config);
    expect(normalized.basket).toEqual(config.basket);
    expect(normalized.screenBasket).toEqual(config.screenBasket);
    expect(normalized.backtestDeposit).toBeNull();

    const withDeposit = buildPortfolioConfig({
      form: { ...form, backtestDeposit: 37.456 },
      botId: 77,
      botName: 'LONG 22',
      params,
    });
    expect(normalizePortfolioConfig(withDeposit).backtestDeposit).toBe(37.46);
  });

  it('validateRunDraft перечисляет причины, по которым запуск невозможен', () => {
    const params = createParams();
    const ready = formWith({ botId: 77, basketText: 'ADA' });
    expect(validateRunDraft({ form: ready, loadedBotId: 77, params })).toEqual([]);

    expect(validateRunDraft({ form: formWith({ basketText: 'ADA' }), loadedBotId: null, params })).toEqual([
      'Выберите базового бота.',
    ]);
    expect(validateRunDraft({ form: ready, loadedBotId: 76, params })[0]).toContain('Загрузите стратегию');
    expect(validateRunDraft({ form: ready, loadedBotId: 77, params: [] })[0]).toContain('не найдено числовых');
    const disabled = params.map((param) => ({ ...param, enabled: false }));
    expect(validateRunDraft({ form: ready, loadedBotId: 77, params: disabled })).toEqual([
      'Включите хотя бы один параметр для поиска.',
    ]);

    const broken = formWith({ botId: 77, basketText: ' , ', periodFrom: '2025-01-02', periodTo: '2025-01-01' });
    const problems = validateRunDraft({ form: broken, loadedBotId: 77, params });
    expect(problems).toHaveLength(2);
    expect(problems[0]).toContain('Корзина монет пуста');
    expect(problems[1]).toContain('Некорректный период');
  });

  it('validateRunDraft не пускает запуск, если лимита бэктестов не хватит на базу и одного кандидата', () => {
    const params = createParams();
    const basketText = Array.from({ length: 40 }, (_, index) => `C${index}`).join(', ');
    const withLimit = (maxBacktests: number, patch: Partial<PortfolioOptimizerFormState> = {}) =>
      formWith({ botId: 77, basketText, screenSize: 10, ...patch, search: { ...DEFAULT_SEARCH_CONFIG, maxBacktests } });

    // Без группы база — все 40 монет.
    const [problem] = validateRunDraft({ form: withLimit(30), loadedBotId: 77, params });
    expect(problem).toContain('Лимит бэктестов (30) меньше, чем нужно для оценки базы (40)');
    expect(problem).toContain('готовые результаты базы');
    expect(validateRunDraft({ form: withLimit(45), loadedBotId: 77, params })[0]).toContain(
      'ни для одного кандидата: база — 40, отбор одного кандидата — 10',
    );
    expect(validateRunDraft({ form: withLimit(50), loadedBotId: 77, params })).toEqual([]);

    // С проверенной группой база — только недостающие монеты; с непроверенной — не известно.
    const grouped = withLimit(12, { baseGroupId: 'g1' });
    expect(validateRunDraft({ form: grouped, loadedBotId: 77, params, baseBacktests: 2 })).toEqual([]);
    expect(validateRunDraft({ form: grouped, loadedBotId: 77, params, baseBacktests: 5 })[0]).toContain(
      'не оставляет места',
    );
    expect(validateRunDraft({ form: grouped, loadedBotId: 77, params, baseBacktests: 13 })[0]).toContain(
      'Увеличьте лимит.',
    );
    expect(validateRunDraft({ form: grouped, loadedBotId: 77, params })).toEqual([]);
    expect(validateRunDraft({ form: grouped, loadedBotId: 77, params, baseBacktests: null })).toEqual([]);
  });
});
