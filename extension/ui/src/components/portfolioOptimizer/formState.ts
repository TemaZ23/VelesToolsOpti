/**
 * Форма портфельного оптимизатора: значения по умолчанию, проверка и сохранение последних
 * настроек (localStorage), разбор корзины монет, период, правка параметров и сборка
 * конфигурации запуска. Модуль чистый, кроме чтения/записи настроек через safeStorage.
 */

import { roundTo } from '../../lib/botParams';
import { readStorageValue, writeStorageValue } from '../../lib/safeStorage';
import { parseSymbols } from '../../services/optimizer';
import {
  DEFAULT_OBJECTIVE,
  DEFAULT_SCHEDULER_CONFIG,
  DEFAULT_SCREEN_BASKET_SIZE,
  DEFAULT_SEARCH_CONFIG,
  pickScreenBasket,
} from '../../services/portfolioOptimizer';
import type {
  BacktestRunSettings,
  ObjectiveConfig,
  ObjectiveKind,
  ParamKind,
  ParamSpec,
  PortfolioOptimizerConfig,
  SchedulerConfig,
  SearchConfig,
} from '../../types/portfolioOptimizer';

// ═══════════════════════════════════════════════════════════════════
// КОНСТАНТЫ
// ═══════════════════════════════════════════════════════════════════

/** Ключ localStorage с последними настройками формы. */
export const PORTFOLIO_FORM_STORAGE_KEY = 'veles_portfolio_optimizer_form';

const FORM_STORAGE_VERSION = 1;

/** Период по умолчанию: последние 5 лет. */
export const DEFAULT_PERIOD_YEARS = 5;
export const DEFAULT_MAKER_COMMISSION = 0.02;
export const DEFAULT_TAKER_COMMISSION = 0.055;
/** Блокировка по ботам по умолчанию (как в группе бэктестов). */
export const DEFAULT_BLOCK = 60;
/** Валюта котировки, если её не удалось определить по боту. */
export const DEFAULT_QUOTE_CURRENCY = 'USDT';

interface NumberRange {
  min: number;
  max: number;
}

/** Допустимые значения полей формы — общие для UI и проверки сохранённых настроек. */
export const FORM_LIMITS = {
  screenSize: { min: 1, max: 500 },
  commission: { min: 0, max: 5 },
  block: { min: 1, max: 1000 },
  /** Депозит бота в бэктестах (в валюте котировки). */
  backtestDeposit: { min: 0.01, max: 1_000_000_000 },
  folds: { min: 2, max: 6 },
  maxMaeCap: { min: 0, max: 1_000_000_000 },
  minDeals: { min: 0, max: 1_000_000 },
  generations: { min: 1, max: 200 },
  candidatesPerGeneration: { min: 1, max: 50 },
  promotePerGeneration: { min: 0, max: 50 },
  mutations: { min: 1, max: 20 },
  /** Допуск отбора и минимальное улучшение — доли (0.05 = 5%). */
  screenTolerance: { min: 0, max: 1 },
  minImprovement: { min: 0, max: 1 },
  maxBacktests: { min: 1, max: 100_000 },
  seed: { min: 0, max: 2_147_483_647 },
  maxInFlight: { min: 1, max: 10 },
  submitDelaySeconds: { min: 3, max: 120 },
  pollIntervalSeconds: { min: 3, max: 300 },
  backtestTimeoutMinutes: { min: 5, max: 720 },
} as const satisfies Record<string, NumberRange>;

const OBJECTIVE_KINDS: readonly ObjectiveKind[] = ['robust', 'pnlToRisk', 'profitCapped'];

/** Пороги условий (индикаторов) — для кнопки «Включить все пороги». */
export const THRESHOLD_PARAM_KINDS: ReadonlySet<ParamKind> = new Set<ParamKind>([
  'entryValue',
  'orderConditionValue',
  'tpConditionValue',
  'slConditionValue',
]);

/** Параметры сетки — для кнопки «Выключить сетку». */
export const GRID_PARAM_KINDS: ReadonlySet<ParamKind> = new Set<ParamKind>(['orderIndent', 'orderVolume', 'pullUp']);

const DATE_INPUT_PATTERN = /^\d{4}-\d{2}-\d{2}$/;

// ═══════════════════════════════════════════════════════════════════
// ТИПЫ
// ═══════════════════════════════════════════════════════════════════

/** Пользовательские границы параметра, отличающиеся от значений по умолчанию. */
export interface ParamOverride {
  enabled: boolean;
  min: number;
  max: number;
  step: number;
}

/** Правки параметров последнего загруженного бота (id параметра → правка). */
export interface ParamOverrides {
  botId: number;
  params: Record<string, ParamOverride>;
}

export interface PortfolioOptimizerFormState {
  botId: number | null;
  /** Корзина монет в свободной форме: «ADA, XRP SOL/USDT …». */
  basketText: string;
  /** Размер отборочной корзины; null — по умолчанию min(10, размер корзины). */
  screenSize: number | null;
  /** Зерно выборки отборочной корзины (кнопка «Перемешать» увеличивает его). */
  screenSeed: number;
  baseGroupId: string | null;
  /** Даты периода в формате YYYY-MM-DD (UTC). */
  periodFrom: string;
  periodTo: string;
  makerCommission: number;
  takerCommission: number;
  includeWicks: boolean;
  block: number;
  /** Депозит бота в бэктестах; null — как в боте. */
  backtestDeposit: number | null;
  objective: ObjectiveConfig;
  search: SearchConfig;
  scheduler: SchedulerConfig;
  paramOverrides: ParamOverrides | null;
}

export interface RunDraftInput {
  form: PortfolioOptimizerFormState;
  botId: number;
  botName: string;
  params: ParamSpec[];
}

export interface RunValidationInput {
  form: PortfolioOptimizerFormState;
  /** id бота, стратегия которого загружена (null — не загружена). */
  loadedBotId: number | null;
  params: ParamSpec[];
  /**
   * Сколько бэктестов нужно на оценку базы: без группы базы — вся корзина, с проверенной группой —
   * недостающие в ней монеты, null — неизвестно (группа выбрана, но не проверена). По умолчанию —
   * вся корзина, если группа базы не выбрана.
   */
  baseBacktests?: number | null;
}

// ═══════════════════════════════════════════════════════════════════
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
// ═══════════════════════════════════════════════════════════════════

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null;

const isFiniteNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

const clamp = (value: number, range: NumberRange): number => Math.min(range.max, Math.max(range.min, value));

/** Число из сохранённых настроек в допустимом диапазоне, иначе значение по умолчанию. */
const readNumber = (value: unknown, fallback: number, range: NumberRange): number =>
  isFiniteNumber(value) ? clamp(value, range) : fallback;

const readInteger = (value: unknown, fallback: number, range: NumberRange): number =>
  isFiniteNumber(value) ? clamp(Math.floor(value), range) : fallback;

const readBoolean = (value: unknown, fallback: boolean): boolean => (typeof value === 'boolean' ? value : fallback);

const readNonEmptyString = (value: unknown): string | null =>
  typeof value === 'string' && value.trim().length > 0 ? value.trim() : null;

const readPositiveId = (value: unknown): number | null =>
  isFiniteNumber(value) && Number.isInteger(value) && value > 0 ? value : null;

/** Минимальный шаг параметра: одна единица последнего знака после запятой. */
export const minParamStep = (decimals: number): number => 10 ** -Math.max(0, Math.floor(decimals));

// ═══════════════════════════════════════════════════════════════════
// ДАТЫ И ПЕРИОД
// ═══════════════════════════════════════════════════════════════════

/** Дата в формате поля ввода (YYYY-MM-DD, UTC). */
export const formatDateInput = (date: Date): string => date.toISOString().slice(0, 10);

export const isDateInput = (value: unknown): value is string =>
  typeof value === 'string' &&
  DATE_INPUT_PATTERN.test(value) &&
  !Number.isNaN(Date.parse(`${value}T00:00:00.000Z`)) &&
  formatDateInput(new Date(`${value}T00:00:00.000Z`)) === value;

/** Период по умолчанию: с даты «5 лет назад» по сегодня. */
export const defaultPeriod = (now: Date): { periodFrom: string; periodTo: string } => {
  const from = new Date(now.getTime());
  from.setUTCFullYear(from.getUTCFullYear() - DEFAULT_PERIOD_YEARS);
  return { periodFrom: formatDateInput(from), periodTo: formatDateInput(now) };
};

/**
 * Период бэктестов в ISO: с начала первого дня по конец последнего (UTC).
 * null — дата некорректна или начало позже конца.
 */
export const toRunPeriod = (from: string, to: string): Pick<BacktestRunSettings, 'periodFrom' | 'periodTo'> | null => {
  if (!(isDateInput(from) && isDateInput(to))) {
    return null;
  }
  const periodFrom = `${from}T00:00:00.000Z`;
  const periodTo = `${to}T23:59:59.999Z`;
  return Date.parse(periodTo) > Date.parse(periodFrom) ? { periodFrom, periodTo } : null;
};

// ═══════════════════════════════════════════════════════════════════
// КОРЗИНА МОНЕТ
// ═══════════════════════════════════════════════════════════════════

/**
 * Тикеры корзины из текста: разделители — запятая, точка с запятой, пробелы и переводы строк;
 * котировка после «/» отбрасывается («ADA/USDT» → «ADA»), повторы убираются, порядок сохраняется.
 */
export const parseBasket = (text: string): string[] => {
  const result: string[] = [];
  const seen = new Set<string>();
  for (const symbol of parseSymbols(text.replace(/;/g, ','))) {
    const ticker = symbol.split('/')[0].replace(/[^A-Z0-9]/g, '');
    if (ticker.length > 0 && !seen.has(ticker)) {
      seen.add(ticker);
      result.push(ticker);
    }
  }
  return result;
};

/** Базовые монеты из символов бота («BTC/USDT» → «BTC»). */
export const basketFromSymbols = (symbols: ReadonlyArray<string | null | undefined>): string[] =>
  parseBasket(symbols.filter((symbol): symbol is string => typeof symbol === 'string').join(', '));

/** Размер отборочной корзины: заданный (или 10) в пределах [1, размер корзины]. */
export const resolveScreenSize = (basketLength: number, screenSize: number | null): number => {
  if (basketLength <= 0) {
    return 0;
  }
  const requested = isFiniteNumber(screenSize) ? Math.floor(screenSize) : DEFAULT_SCREEN_BASKET_SIZE;
  return Math.min(basketLength, Math.max(1, requested));
};

/** Отборочная корзина формы (детерминированная выборка по зерну). */
export const resolveScreenBasket = (
  form: Pick<PortfolioOptimizerFormState, 'screenSize' | 'screenSeed'>,
  basket: string[],
) => pickScreenBasket(basket, resolveScreenSize(basket.length, form.screenSize), form.screenSeed);

// ═══════════════════════════════════════════════════════════════════
// ЗНАЧЕНИЯ ПО УМОЛЧАНИЮ И СОХРАНЕНИЕ
// ═══════════════════════════════════════════════════════════════════

export const createDefaultFormState = (now: Date = new Date()): PortfolioOptimizerFormState => {
  const period = defaultPeriod(now);
  return {
    botId: null,
    basketText: '',
    screenSize: null,
    screenSeed: 1,
    baseGroupId: null,
    periodFrom: period.periodFrom,
    periodTo: period.periodTo,
    makerCommission: DEFAULT_MAKER_COMMISSION,
    takerCommission: DEFAULT_TAKER_COMMISSION,
    includeWicks: true,
    block: DEFAULT_BLOCK,
    backtestDeposit: null,
    objective: { ...DEFAULT_OBJECTIVE },
    search: { ...DEFAULT_SEARCH_CONFIG },
    scheduler: { ...DEFAULT_SCHEDULER_CONFIG },
    paramOverrides: null,
  };
};

const sanitizeObjective = (value: unknown): ObjectiveConfig => {
  const source = isRecord(value) ? value : {};
  const kind = OBJECTIVE_KINDS.find((item) => item === source.kind) ?? DEFAULT_OBJECTIVE.kind;
  return {
    kind,
    folds: readInteger(source.folds, DEFAULT_OBJECTIVE.folds, FORM_LIMITS.folds),
    maxMaeCap: isFiniteNumber(source.maxMaeCap) ? clamp(source.maxMaeCap, FORM_LIMITS.maxMaeCap) : null,
    minDeals: readInteger(source.minDeals, DEFAULT_OBJECTIVE.minDeals, FORM_LIMITS.minDeals),
  };
};

const sanitizeSearch = (value: unknown): SearchConfig => {
  const source = isRecord(value) ? value : {};
  const defaults = DEFAULT_SEARCH_CONFIG;
  const mutationsMin = readInteger(source.mutationsMin, defaults.mutationsMin, FORM_LIMITS.mutations);
  const mutationsMax = readInteger(source.mutationsMax, defaults.mutationsMax, FORM_LIMITS.mutations);
  return {
    generations: readInteger(source.generations, defaults.generations, FORM_LIMITS.generations),
    candidatesPerGeneration: readInteger(
      source.candidatesPerGeneration,
      defaults.candidatesPerGeneration,
      FORM_LIMITS.candidatesPerGeneration,
    ),
    promotePerGeneration: readInteger(
      source.promotePerGeneration,
      defaults.promotePerGeneration,
      FORM_LIMITS.promotePerGeneration,
    ),
    mutationsMin,
    mutationsMax: Math.max(mutationsMin, mutationsMax),
    screenTolerance: readNumber(source.screenTolerance, defaults.screenTolerance, FORM_LIMITS.screenTolerance),
    minImprovement: readNumber(source.minImprovement, defaults.minImprovement, FORM_LIMITS.minImprovement),
    maxBacktests: readInteger(source.maxBacktests, defaults.maxBacktests, FORM_LIMITS.maxBacktests),
    seed: readInteger(source.seed, defaults.seed, FORM_LIMITS.seed),
  };
};

const sanitizeScheduler = (value: unknown): SchedulerConfig => {
  const source = isRecord(value) ? value : {};
  const defaults = DEFAULT_SCHEDULER_CONFIG;
  return {
    maxInFlight: readInteger(source.maxInFlight, defaults.maxInFlight, FORM_LIMITS.maxInFlight),
    submitDelaySeconds: readNumber(
      source.submitDelaySeconds,
      defaults.submitDelaySeconds,
      FORM_LIMITS.submitDelaySeconds,
    ),
    pollIntervalSeconds: readNumber(
      source.pollIntervalSeconds,
      defaults.pollIntervalSeconds,
      FORM_LIMITS.pollIntervalSeconds,
    ),
    backtestTimeoutMinutes: readNumber(
      source.backtestTimeoutMinutes,
      defaults.backtestTimeoutMinutes,
      FORM_LIMITS.backtestTimeoutMinutes,
    ),
  };
};

const isParamOverride = (value: unknown): value is ParamOverride =>
  isRecord(value) &&
  typeof value.enabled === 'boolean' &&
  isFiniteNumber(value.min) &&
  isFiniteNumber(value.max) &&
  isFiniteNumber(value.step) &&
  value.min <= value.max &&
  value.step > 0;

const sanitizeParamOverrides = (value: unknown): ParamOverrides | null => {
  if (!(isRecord(value) && isRecord(value.params))) {
    return null;
  }
  const botId = readPositiveId(value.botId);
  if (botId === null) {
    return null;
  }
  const params: Record<string, ParamOverride> = {};
  for (const [id, override] of Object.entries(value.params)) {
    if (isParamOverride(override)) {
      params[id] = { enabled: override.enabled, min: override.min, max: override.max, step: override.step };
    }
  }
  return { botId, params };
};

/** Проверка сохранённых настроек: некорректные поля заменяются значениями по умолчанию. */
export const sanitizeFormState = (value: unknown, now: Date = new Date()): PortfolioOptimizerFormState => {
  const defaults = createDefaultFormState(now);
  if (!isRecord(value)) {
    return defaults;
  }
  const periodFrom = isDateInput(value.periodFrom) ? value.periodFrom : defaults.periodFrom;
  const periodTo = isDateInput(value.periodTo) ? value.periodTo : defaults.periodTo;
  const periodValid = toRunPeriod(periodFrom, periodTo) !== null;
  return {
    botId: readPositiveId(value.botId),
    basketText: typeof value.basketText === 'string' ? value.basketText : defaults.basketText,
    screenSize: isFiniteNumber(value.screenSize) ? readInteger(value.screenSize, 1, FORM_LIMITS.screenSize) : null,
    screenSeed: readInteger(value.screenSeed, defaults.screenSeed, FORM_LIMITS.seed),
    baseGroupId: readNonEmptyString(value.baseGroupId),
    periodFrom: periodValid ? periodFrom : defaults.periodFrom,
    periodTo: periodValid ? periodTo : defaults.periodTo,
    makerCommission: readNumber(value.makerCommission, defaults.makerCommission, FORM_LIMITS.commission),
    takerCommission: readNumber(value.takerCommission, defaults.takerCommission, FORM_LIMITS.commission),
    includeWicks: readBoolean(value.includeWicks, defaults.includeWicks),
    block: readInteger(value.block, defaults.block, FORM_LIMITS.block),
    backtestDeposit:
      isFiniteNumber(value.backtestDeposit) && value.backtestDeposit > 0
        ? readNumber(value.backtestDeposit, FORM_LIMITS.backtestDeposit.min, FORM_LIMITS.backtestDeposit)
        : null,
    objective: sanitizeObjective(value.objective),
    search: sanitizeSearch(value.search),
    scheduler: sanitizeScheduler(value.scheduler),
    paramOverrides: sanitizeParamOverrides(value.paramOverrides),
  };
};

/** Последние настройки формы (или значения по умолчанию). */
export const readPortfolioFormState = (now: Date = new Date()): PortfolioOptimizerFormState => {
  const raw = readStorageValue(PORTFOLIO_FORM_STORAGE_KEY);
  if (!raw) {
    return createDefaultFormState(now);
  }
  try {
    const parsed: unknown = JSON.parse(raw);
    if (isRecord(parsed) && parsed.version === FORM_STORAGE_VERSION) {
      return sanitizeFormState(parsed, now);
    }
  } catch (error) {
    console.warn('[PortfolioOptimizer] Не удалось прочитать сохранённые настройки формы', error);
  }
  return createDefaultFormState(now);
};

export const writePortfolioFormState = (form: PortfolioOptimizerFormState): boolean =>
  writeStorageValue(PORTFOLIO_FORM_STORAGE_KEY, JSON.stringify({ version: FORM_STORAGE_VERSION, ...form }));

// ═══════════════════════════════════════════════════════════════════
// ПАРАМЕТРЫ
// ═══════════════════════════════════════════════════════════════════

const sameOverride = (param: ParamSpec, defaults: ParamSpec): boolean =>
  param.enabled === defaults.enabled &&
  param.min === defaults.min &&
  param.max === defaults.max &&
  param.step === defaults.step;

/** Правка применима, если базовое значение бота по-прежнему внутри диапазона. */
const isApplicableOverride = (param: ParamSpec, override: ParamOverride): boolean =>
  override.min <= param.base && param.base <= override.max && override.step > 0;

/**
 * Параметры с сохранёнными правками пользователя — только для того же бота и только если
 * базовое значение не вышло за сохранённый диапазон (бот могли изменить в Veles).
 */
export const applyParamOverrides = (
  params: ParamSpec[],
  overrides: ParamOverrides | null,
  botId: number,
): ParamSpec[] =>
  params.map((param) => {
    const override = overrides && overrides.botId === botId ? overrides.params[param.id] : undefined;
    if (!(override && isApplicableOverride(param, override))) {
      return { ...param };
    }
    return { ...param, enabled: override.enabled, min: override.min, max: override.max, step: override.step };
  });

/** Правки пользователя относительно параметров по умолчанию (для сохранения формы). */
export const collectParamOverrides = (botId: number, params: ParamSpec[], defaults: ParamSpec[]): ParamOverrides => {
  const defaultsById = new Map(defaults.map((param) => [param.id, param]));
  const result: Record<string, ParamOverride> = {};
  for (const param of params) {
    const base = defaultsById.get(param.id);
    if (base && !sameOverride(param, base)) {
      result[param.id] = { enabled: param.enabled, min: param.min, max: param.max, step: param.step };
    }
  }
  return { botId, params: result };
};

export const sameParamOverrides = (left: ParamOverrides | null, right: ParamOverrides | null): boolean => {
  if (left === null || right === null) {
    return left === right;
  }
  if (left.botId !== right.botId) {
    return false;
  }
  const leftKeys = Object.keys(left.params);
  if (leftKeys.length !== Object.keys(right.params).length) {
    return false;
  }
  return leftKeys.every((id) => {
    const a = left.params[id];
    const b = Object.hasOwn(right.params, id) ? right.params[id] : undefined;
    return b !== undefined && a.enabled === b.enabled && a.min === b.min && a.max === b.max && a.step === b.step;
  });
};

export type ParamBoundField = 'min' | 'max' | 'step';

/**
 * Новая граница параметра: минимум не выше базового значения, максимум не ниже,
 * шаг не меньше единицы последнего знака. null/нечисловое значение — без изменений.
 */
export const updateParamBound = (
  params: ParamSpec[],
  id: string,
  field: ParamBoundField,
  value: number | null,
): ParamSpec[] => {
  if (!isFiniteNumber(value)) {
    return params;
  }
  return params.map((param) => {
    if (param.id !== id) {
      return param;
    }
    const rounded = roundTo(value, param.decimals);
    if (field === 'min') {
      return { ...param, min: Math.min(rounded, param.base) };
    }
    if (field === 'max') {
      return { ...param, max: Math.max(rounded, param.base) };
    }
    return { ...param, step: Math.max(rounded, minParamStep(param.decimals)) };
  });
};

export const setParamEnabled = (params: ParamSpec[], id: string, enabled: boolean): ParamSpec[] =>
  params.map((param) => (param.id === id && param.enabled !== enabled ? { ...param, enabled } : param));

export const setParamsEnabledWhere = (
  params: ParamSpec[],
  predicate: (param: ParamSpec) => boolean,
  enabled: boolean,
): ParamSpec[] =>
  params.map((param) => (predicate(param) && param.enabled !== enabled ? { ...param, enabled } : param));

/** «Включить все пороги»: все пороги условий (вход, ордера, тейк, стоп), включая объёмные. */
export const enableAllThresholds = (params: ParamSpec[]): ParamSpec[] =>
  setParamsEnabledWhere(params, (param) => THRESHOLD_PARAM_KINDS.has(param.kind), true);

/** «Выключить сетку»: отступы и объёмы ордеров, подтяжка. */
export const disableGrid = (params: ParamSpec[]): ParamSpec[] =>
  setParamsEnabledWhere(params, (param) => GRID_PARAM_KINDS.has(param.kind), false);

/** Количество значений параметра на сетке поиска [min, max] с шагом step. */
export const countParamValues = (param: Pick<ParamSpec, 'min' | 'max' | 'step'>): number => {
  if (!(param.step > 0) || param.max < param.min) {
    return 1;
  }
  return Math.floor((param.max - param.min) / param.step + 1e-9) + 1;
};

// ═══════════════════════════════════════════════════════════════════
// КОНФИГУРАЦИЯ ЗАПУСКА
// ═══════════════════════════════════════════════════════════════════

/** Настройки бэктестов формы (период — пустые строки, если даты некорректны). */
export const buildRunSettings = (form: PortfolioOptimizerFormState): BacktestRunSettings => {
  const period = toRunPeriod(form.periodFrom, form.periodTo);
  return {
    periodFrom: period?.periodFrom ?? '',
    periodTo: period?.periodTo ?? '',
    makerCommission: form.makerCommission,
    takerCommission: form.takerCommission,
    includeWicks: form.includeWicks,
  };
};

/** Конфигурация запуска из формы (окончательная проверка — normalizePortfolioConfig в сервисе). */
export const buildPortfolioConfig = ({ form, botId, botName, params }: RunDraftInput): PortfolioOptimizerConfig => {
  const basket = parseBasket(form.basketText);
  return {
    botId,
    botName,
    basket,
    screenBasket: resolveScreenBasket(form, basket),
    run: buildRunSettings(form),
    block: form.block,
    objective: { ...form.objective },
    search: { ...form.search },
    scheduler: { ...form.scheduler },
    params: params.map((param) => ({ ...param })),
    baseGroupId: form.baseGroupId,
    backtestDeposit: form.backtestDeposit,
  };
};

/** Проблемы, из-за которых запуск невозможен (пустой список — можно запускать). */
export const validateRunDraft = ({ form, loadedBotId, params, baseBacktests }: RunValidationInput): string[] => {
  const problems: string[] = [];
  if (form.botId === null) {
    problems.push('Выберите базового бота.');
  } else if (loadedBotId !== form.botId) {
    problems.push('Загрузите стратегию выбранного бота (кнопка «Загрузить»).');
  } else if (params.length === 0) {
    problems.push('У базового бота не найдено числовых параметров для оптимизации.');
  } else if (!params.some((param) => param.enabled)) {
    problems.push('Включите хотя бы один параметр для поиска.');
  }
  const basketSize = parseBasket(form.basketText).length;
  if (basketSize === 0) {
    problems.push('Корзина монет пуста — добавьте хотя бы одну монету.');
  }
  if (toRunPeriod(form.periodFrom, form.periodTo) === null) {
    problems.push('Некорректный период: дата начала должна быть не позже даты окончания.');
  }
  const baseCost = baseBacktests !== undefined ? baseBacktests : form.baseGroupId ? null : basketSize;
  if (basketSize > 0 && baseCost !== null) {
    const limit = form.search.maxBacktests;
    const screenCost = resolveScreenSize(basketSize, form.screenSize);
    if (limit < baseCost) {
      problems.push(
        `Лимит бэктестов (${limit}) меньше, чем нужно для оценки базы (${baseCost}): запуск остановится на базе. ` +
          `Увеличьте лимит${form.baseGroupId ? '' : ' или выберите готовые результаты базы из группы бэктестов'}.`,
      );
    } else if (limit < baseCost + screenCost) {
      problems.push(
        `Лимит бэктестов (${limit}) не оставляет места ни для одного кандидата: база — ${baseCost}, ` +
          `отбор одного кандидата — ${screenCost}.`,
      );
    }
  }
  return problems;
};
