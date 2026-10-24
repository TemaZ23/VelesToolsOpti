/**
 * Числовые параметры реального DTO бота Veles для портфельного оптимизатора.
 *
 * Модуль чистый (без сети и хранилищ): извлекает из стратегии пороги условий, отступы/объёмы
 * сетки, мин. P&L тейка, отступы стопа и подтяжку, задаёт им границы поиска и умеет точечно
 * записывать новые значения обратно в глубокую копию DTO. Структура условий никогда не
 * пересобирается — меняется только `.value` (и числовые поля ордеров/тейка/стопа), поэтому
 * неизвестные поля DTO сохраняются как есть.
 */

import type { BotStrategy } from '../api/backtestRunner';
import type { CreateBotPayload } from '../api/bots';
import type { BotOrderDto } from '../api/bots.dtos';
import type { ParamKind, ParamPath, ParamSpec, ParamValues } from '../types/portfolioOptimizer';

// ═══════════════════════════════════════════════════════════════════
// КОНСТАНТЫ
// ═══════════════════════════════════════════════════════════════════

/** Максимум знаков после запятой, который сохраняем у параметра. */
const MAX_DECIMALS = 6;
/** Минимальный зазор между отступами соседних ордеров, %. */
const MIN_INDENT_GAP = 0.01;
/** Минимальный объём ордера, %. */
const MIN_VOLUME = 0.01;
/** Допуск сравнения чисел с плавающей точкой. */
const EPSILON = 1e-9;

const GROUP_ENTRY = 'Вход';
const GROUP_TAKE_PROFIT = 'Тейк';
const GROUP_STOP_LOSS = 'Стоп';
const GROUP_GRID = 'Сетка';

const orderGroup = (orderIndex: number): string => `Ордер ${orderIndex + 1}`;

const CONDITION_KINDS: ReadonlySet<ParamKind> = new Set<ParamKind>([
  'entryValue',
  'orderConditionValue',
  'tpConditionValue',
  'slConditionValue',
]);

const INDICATOR_LABELS: Readonly<Record<string, string>> = {
  RSI: 'RSI',
  RSI_LEVELS: 'Уровни RSI',
  CCI: 'CCI',
  CCI_LEVELS: 'Уровни CCI',
  CMO: 'CMO',
  WILLIAMS_R: 'Williams %R',
  MFI: 'MFI',
  ADX: 'ADX',
  ATR: 'ATR',
  ATR_PERCENT: 'ATR%',
  VOLUME: 'Объём',
  STOCHASTIC: 'Стохастик',
  STOCHASTIC_LEVELS: 'Уровни стохастика',
  MACD: 'MACD',
  BOLLINGER_BANDS: 'Полосы Боллинджера',
  KELTNER_CHANNEL: 'Канал Кельтнера',
  TURTLE_ZONE: 'Turtle Zone',
};

const INTERVAL_LABELS: Readonly<Record<string, string>> = {
  ONE_MINUTE: '1 мин',
  FIVE_MINUTES: '5 мин',
  FIFTEEN_MINUTES: '15 мин',
  THIRTY_MINUTES: '30 мин',
  ONE_HOUR: '1 час',
  FOUR_HOURS: '4 часа',
  ONE_DAY: '1 день',
};

const OPERATION_SYMBOLS: Readonly<Record<string, string>> = {
  GREATER: '>',
  GREATER_OR_EQUAL: '≥',
  LESS: '<',
  LESS_OR_EQUAL: '≤',
  EQUAL: '=',
};

/** Правило границ для порогов индикаторов: совпадение по подстроке названия в верхнем регистре. */
interface IndicatorBoundsRule {
  match: string;
  step: number;
  /** Полуширина диапазона поиска вокруг базового значения. */
  spread: number;
  /** Естественная область значений индикатора. */
  domain: readonly [number, number];
  decimals: number;
}

/** Порядок важен: первое совпадение побеждает (VOLUME обрабатывается отдельно). */
const INDICATOR_BOUNDS_RULES: readonly IndicatorBoundsRule[] = [
  { match: 'WILLIAMS', step: 3, spread: 15, domain: [-100, 0], decimals: 0 },
  { match: 'ATR', step: 0.05, spread: 0.3, domain: [0, 10], decimals: 2 },
  { match: 'CCI', step: 10, spread: 60, domain: [-500, 500], decimals: 0 },
  { match: 'CMO', step: 5, spread: 30, domain: [-100, 100], decimals: 0 },
  { match: 'ADX', step: 2, spread: 12, domain: [0, 100], decimals: 0 },
  { match: 'RSI', step: 2, spread: 10, domain: [0, 100], decimals: 0 },
  { match: 'MFI', step: 2, spread: 10, domain: [0, 100], decimals: 0 },
  { match: 'STOCH', step: 2, spread: 10, domain: [0, 100], decimals: 0 },
];

/** Параметры, значения которых задаются в процентах (для подписей изменений). */
const PERCENT_KINDS: ReadonlySet<ParamKind> = new Set<ParamKind>([
  'orderIndent',
  'orderVolume',
  'tpMinPnl',
  'slIndent',
  'slConditionalIndent',
  'pullUp',
]);

// ═══════════════════════════════════════════════════════════════════
// БАЗОВЫЕ ПОМОЩНИКИ
// ═══════════════════════════════════════════════════════════════════

/**
 * Обход произвольного JSON-объекта DTO: в DTO Veles встречаются поля, которых нет в наших типах,
 * поэтому навигация по пути работает с `unknown` и сужает тип рантайм-проверками.
 */
const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null;

const isFiniteNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

const clamp = (value: number, min: number, max: number): number => Math.min(max, Math.max(min, value));

const sum = (values: readonly number[]): number => values.reduce((total, value) => total + value, 0);

const normalizeDecimals = (decimals: number): number => {
  if (!Number.isFinite(decimals)) {
    return 0;
  }
  return clamp(Math.trunc(decimals), 0, 12);
};

/** Количество значащих знаков после запятой у числа (не больше MAX_DECIMALS). */
const countDecimals = (value: number): number => {
  if (!Number.isFinite(value) || Number.isInteger(value)) {
    return 0;
  }
  const text = String(Math.abs(value));
  const exponentIndex = text.indexOf('e');
  if (exponentIndex >= 0) {
    const mantissa = text.slice(0, exponentIndex);
    const exponent = Number(text.slice(exponentIndex + 1));
    const mantissaDecimals = mantissa.includes('.') ? mantissa.length - mantissa.indexOf('.') - 1 : 0;
    return clamp(mantissaDecimals - exponent, 0, MAX_DECIMALS);
  }
  const dotIndex = text.indexOf('.');
  return dotIndex >= 0 ? Math.min(text.length - dotIndex - 1, MAX_DECIMALS) : 0;
};

const cloneDeep = <T>(value: T): T => {
  if (value === null || value === undefined) {
    return value;
  }
  if (typeof structuredClone === 'function') {
    try {
      return structuredClone(value);
    } catch {
      // значение не поддерживает structuredClone — используем JSON
    }
  }
  return JSON.parse(JSON.stringify(value)) as T;
};

const compareByOrderIndex = (left: ParamSpec, right: ParamSpec): number =>
  (left.orderIndex ?? 0) - (right.orderIndex ?? 0);

/** Значение параметра из набора; отсутствующее или нечисловое заменяется базовым. */
const resolveValue = (param: ParamSpec, values: ParamValues): number => {
  const value: number | undefined = values[param.id];
  return isFiniteNumber(value) ? value : param.base;
};

// ═══════════════════════════════════════════════════════════════════
// ПУТИ И ОКРУГЛЕНИЕ
// ═══════════════════════════════════════════════════════════════════

/** ['settings', 'orders', 3, 'indent'] → 'settings.orders[3].indent'. */
export const pathToId = (path: ParamPath): string =>
  path.reduce<string>((id, segment) => {
    if (typeof segment === 'number') {
      return `${id}[${segment}]`;
    }
    return id.length > 0 ? `${id}.${segment}` : segment;
  }, '');

/** Читает значение по пути; если по дороге встречается не-объект — возвращает undefined. */
export const getAtPath = (obj: unknown, path: ParamPath): unknown => {
  let current: unknown = obj;
  for (const segment of path) {
    if (!isRecord(current)) {
      return undefined;
    }
    current = current[String(segment)];
  }
  return current;
};

/** Записывает число по пути (мутирует объект). Бросает ошибку, если путь отсутствует. */
export const setAtPath = (obj: unknown, path: ParamPath, value: number): void => {
  if (path.length === 0) {
    throw new Error('Пустой путь параметра бота.');
  }
  let container: unknown = obj;
  path.forEach((segment, index) => {
    const key = String(segment);
    if (!(isRecord(container) && Object.hasOwn(container, key))) {
      throw new Error(`Поле «${pathToId(path)}» не найдено в настройках бота.`);
    }
    if (index === path.length - 1) {
      container[key] = value;
      return;
    }
    container = container[key];
  });
};

/**
 * Округление до `decimals` знаков «от нуля» без артефактов двоичной арифметики
 * (1.005 → 1.01, −1.005 → −1.01). Нечисловые значения возвращаются как есть, −0 → 0.
 */
export const roundTo = (value: number, decimals: number): number => {
  if (!Number.isFinite(value)) {
    return value;
  }
  const factor = 10 ** normalizeDecimals(decimals);
  const scaled = Number((Math.abs(value) * factor).toPrecision(15));
  const rounded = (Math.sign(value) * Math.round(scaled)) / factor;
  return rounded === 0 ? 0 : rounded;
};

/** Значение параметра для подписей: округление до `decimals` без лишних нулей. */
export const formatParamValue = (value: number, decimals: number): string => String(roundTo(value, decimals));

// ═══════════════════════════════════════════════════════════════════
// ГРАНИЦЫ ПО УМОЛЧАНИЮ
// ═══════════════════════════════════════════════════════════════════

interface ParamBounds {
  min: number;
  max: number;
  step: number;
  decimals: number;
  enabled: boolean;
}

interface RawBounds {
  min: number;
  max: number;
  step: number;
  decimals: number;
  enabled: boolean;
  domain: readonly [number, number];
}

const NO_DOMAIN: readonly [number, number] = [Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY];
const NON_NEGATIVE_DOMAIN: readonly [number, number] = [0, Number.POSITIVE_INFINITY];

const conditionRawBounds = (base: number, indicator: string | null | undefined): RawBounds => {
  const name = (indicator ?? '').toUpperCase();

  if (name.includes('VOLUME')) {
    return {
      min: base * 0.25,
      max: base * 4,
      step: Math.max(1, Math.round(base * 0.25)),
      decimals: 0,
      enabled: false,
      domain: NON_NEGATIVE_DOMAIN,
    };
  }

  const rule = INDICATOR_BOUNDS_RULES.find((candidate) => name.includes(candidate.match));
  if (rule) {
    return {
      min: base - rule.spread,
      max: base + rule.spread,
      step: rule.step,
      decimals: rule.decimals,
      enabled: true,
      domain: rule.domain,
    };
  }

  const spread = base === 0 ? 1 : Math.abs(base) * 0.5;
  return {
    min: base - spread,
    max: base + spread,
    step: Math.max(Math.abs(base) * 0.1, 0.01),
    decimals: 2,
    enabled: true,
    domain: NO_DOMAIN,
  };
};

const scalarRawBounds = (kind: ParamKind, base: number): RawBounds => {
  switch (kind) {
    case 'orderIndent': {
      const smallIndentPadding = base < 1 ? 0.5 : 0;
      return {
        min: base * 0.6 - smallIndentPadding,
        max: base * 1.4 + smallIndentPadding,
        step: Math.max(0.1, roundTo(base * 0.08, 1)),
        decimals: 2,
        enabled: true,
        domain: NON_NEGATIVE_DOMAIN,
      };
    }
    case 'orderVolume':
      return {
        min: base * 0.5,
        max: base * 2,
        step: Math.max(0.2, roundTo(base * 0.1, 1)),
        decimals: 2,
        enabled: true,
        domain: [MIN_VOLUME, Number.POSITIVE_INFINITY],
      };
    case 'tpMinPnl':
      return { min: 0.1, max: Math.max(2, base * 3), step: 0.1, decimals: 2, enabled: true, domain: NO_DOMAIN };
    case 'slConditionalIndent':
      return { min: 0, max: base + 10, step: 0.5, decimals: 2, enabled: true, domain: NON_NEGATIVE_DOMAIN };
    case 'slIndent':
      return {
        min: base * 0.5,
        max: base * 1.5,
        step: Math.max(0.5, base * 0.1),
        decimals: 2,
        enabled: true,
        domain: NON_NEGATIVE_DOMAIN,
      };
    case 'pullUp':
      return {
        min: 0.1,
        max: Math.max(2, base * 2),
        step: 0.1,
        decimals: 2,
        enabled: true,
        domain: NON_NEGATIVE_DOMAIN,
      };
    default:
      return conditionRawBounds(base, null);
  }
};

/**
 * Границы поиска по умолчанию для параметра.
 * min/max ограничиваются естественной областью значений, базовое значение всегда внутри [min, max];
 * точность не меньше точности базового значения (иначе база «поплывёт» при округлении).
 */
export const defaultParamBounds = (kind: ParamKind, base: number, indicator?: string | null): ParamBounds => {
  const raw = CONDITION_KINDS.has(kind) ? conditionRawBounds(base, indicator) : scalarRawBounds(kind, base);
  const decimals = Math.max(raw.decimals, countDecimals(base));
  const [domainMin, domainMax] = raw.domain;

  const lower = Math.min(raw.min, raw.max);
  const upper = Math.max(raw.min, raw.max);
  const min = Math.min(roundTo(clamp(lower, domainMin, domainMax), decimals), base);
  const max = Math.max(roundTo(clamp(upper, domainMin, domainMax), decimals), base);

  const smallestStep = 10 ** -decimals;
  const roundedStep = roundTo(Math.abs(raw.step), decimals);
  const step = roundedStep > 0 ? roundedStep : smallestStep;

  return { min, max, step, decimals, enabled: raw.enabled };
};

// ═══════════════════════════════════════════════════════════════════
// ПОДПИСИ
// ═══════════════════════════════════════════════════════════════════

const indicatorLabel = (indicator: string): string => INDICATOR_LABELS[indicator.toUpperCase()] ?? indicator;

const intervalLabel = (interval: string): string => INTERVAL_LABELS[interval.toUpperCase()] ?? interval;

const operationSymbol = (operation: string | null): string => {
  if (operation === null) {
    return '';
  }
  return OPERATION_SYMBOLS[operation.toUpperCase()] ?? operation;
};

const readString = (record: Record<string, unknown>, key: string): string | null => {
  const value = record[key];
  return typeof value === 'string' && value.trim().length > 0 ? value.trim() : null;
};

const buildConditionLabel = (condition: Record<string, unknown>, value: number, decimals: number): string => {
  const indicator = readString(condition, 'indicator');
  const interval = readString(condition, 'interval');
  const name = indicator ? indicatorLabel(indicator) : (readString(condition, 'type') ?? 'Условие');
  return [name, interval ? intervalLabel(interval) : '', operationSymbol(readString(condition, 'operation'))]
    .concat(formatParamValue(value, decimals))
    .filter((part) => part.length > 0)
    .join(' ');
};

// ═══════════════════════════════════════════════════════════════════
// ИЗВЛЕЧЕНИЕ ПАРАМЕТРОВ
// ═══════════════════════════════════════════════════════════════════

interface ParamDraft {
  path: ParamPath;
  kind: ParamKind;
  group: string;
  base: number;
  buildLabel: (decimals: number) => string;
  orderIndex?: number;
  indicator?: string | null;
  /** Принудительно выключить параметр по умолчанию. */
  disabled?: boolean;
}

const createParamSpec = (draft: ParamDraft): ParamSpec => {
  const bounds = defaultParamBounds(draft.kind, draft.base, draft.indicator);
  const spec: ParamSpec = {
    id: pathToId(draft.path),
    path: draft.path,
    kind: draft.kind,
    label: draft.buildLabel(bounds.decimals),
    group: draft.group,
    base: draft.base,
    min: bounds.min,
    max: bounds.max,
    step: bounds.step,
    decimals: bounds.decimals,
    enabled: bounds.enabled && !draft.disabled,
  };
  if (draft.orderIndex !== undefined) {
    spec.orderIndex = draft.orderIndex;
  }
  if (draft.indicator !== undefined) {
    spec.indicator = draft.indicator;
  }
  return spec;
};

const scalarParam = (
  path: ParamPath,
  value: unknown,
  kind: ParamKind,
  group: string,
  label: string,
  orderIndex?: number,
): ParamSpec[] => {
  if (!isFiniteNumber(value)) {
    return [];
  }
  return [createParamSpec({ path, kind, group, base: value, buildLabel: () => label, orderIndex })];
};

const conditionParams = (
  conditions: unknown,
  basePath: ParamPath,
  kind: ParamKind,
  group: string,
  labelPrefix: string,
): ParamSpec[] => {
  if (!Array.isArray(conditions)) {
    return [];
  }
  return conditions.flatMap((condition: unknown, index): ParamSpec[] => {
    if (!(isRecord(condition) && isFiniteNumber(condition.value))) {
      return [];
    }
    const base = condition.value;
    const indicator = readString(condition, 'indicator');
    return [
      createParamSpec({
        path: [...basePath, index, 'value'],
        kind,
        group,
        base,
        indicator,
        // Базовые сигналы (пересечения, уровни) обычно без значения; если значение есть, оно редко
        // влияет на сигнал — не тратим на него бэктесты, пока пользователь не включит явно.
        disabled: condition.basic === true,
        buildLabel: (decimals) => `${labelPrefix}${buildConditionLabel(condition, base, decimals)}`,
      }),
    ];
  });
};

const orderParams = (order: unknown, basePath: ParamPath, orderIndex: number): ParamSpec[] => {
  if (!isRecord(order)) {
    return [];
  }
  const group = orderGroup(orderIndex);
  return [
    ...scalarParam([...basePath, 'indent'], order.indent, 'orderIndent', group, `${group}: отступ`, orderIndex),
    ...scalarParam([...basePath, 'volume'], order.volume, 'orderVolume', group, `${group}: объём`, orderIndex),
    ...conditionParams(order.conditions, [...basePath, 'conditions'], 'orderConditionValue', group, `${group}: `),
  ];
};

const gridParams = (strategy: BotStrategy): ParamSpec[] => {
  const settings = strategy.settings;
  if (!settings) {
    return [];
  }
  const baseOrder: BotOrderDto | null | undefined = settings.baseOrder;
  const hasBaseOrder = isRecord(baseOrder);
  const result: ParamSpec[] = hasBaseOrder ? orderParams(baseOrder, ['settings', 'baseOrder'], 0) : [];
  const offset = hasBaseOrder ? 1 : 0;
  const orders: unknown = settings.orders;
  if (Array.isArray(orders)) {
    orders.forEach((order: unknown, index) => {
      result.push(...orderParams(order, ['settings', 'orders', index], index + offset));
    });
  }
  return result;
};

const entryParams = (strategy: BotStrategy): ParamSpec[] => {
  const rootConditions: unknown = strategy.conditions;
  const settingsConditions: unknown = strategy.settings?.conditions;
  const result = conditionParams(rootConditions, ['conditions'], 'entryValue', GROUP_ENTRY, '');
  if (Array.isArray(settingsConditions) && settingsConditions !== rootConditions) {
    result.push(...conditionParams(settingsConditions, ['settings', 'conditions'], 'entryValue', GROUP_ENTRY, ''));
  }
  return result;
};

const takeProfitParams = (strategy: BotStrategy): ParamSpec[] => {
  const profit = strategy.profit;
  if (!isRecord(profit)) {
    return [];
  }
  const hasSignals = Array.isArray(profit.conditions) && profit.conditions.length > 0;
  const pnlLabel = hasSignals ? 'Тейк: мин. P&L' : 'Тейк: тейк-профит';
  return [
    ...scalarParam(['profit', 'checkPnl'], profit.checkPnl, 'tpMinPnl', GROUP_TAKE_PROFIT, pnlLabel),
    ...conditionParams(profit.conditions, ['profit', 'conditions'], 'tpConditionValue', GROUP_TAKE_PROFIT, 'Тейк: '),
  ];
};

const stopLossParams = (strategy: BotStrategy): ParamSpec[] => {
  const stopLoss = strategy.stopLoss;
  if (!isRecord(stopLoss)) {
    return [];
  }
  return [
    ...scalarParam(['stopLoss', 'indent'], stopLoss.indent, 'slIndent', GROUP_STOP_LOSS, 'Стоп: отступ'),
    ...scalarParam(
      ['stopLoss', 'conditionalIndent'],
      stopLoss.conditionalIndent,
      'slConditionalIndent',
      GROUP_STOP_LOSS,
      'Стоп: отступ по сигналу',
    ),
    ...conditionParams(stopLoss.conditions, ['stopLoss', 'conditions'], 'slConditionValue', GROUP_STOP_LOSS, 'Стоп: '),
  ];
};

/**
 * Извлекает числовые параметры стратегии в фиксированном порядке:
 * вход → ордера сетки (отступ, объём, условия) → тейк → стоп → подтяжка.
 * Параметр создаётся только для конечных чисел; null/undefined/нечисловые поля пропускаются.
 */
export const extractParams = (strategy: BotStrategy): ParamSpec[] => [
  ...entryParams(strategy),
  ...gridParams(strategy),
  ...takeProfitParams(strategy),
  ...stopLossParams(strategy),
  ...scalarParam(['pullUp'], strategy.pullUp, 'pullUp', GROUP_GRID, 'Подтяжка сетки'),
];

// ═══════════════════════════════════════════════════════════════════
// ЗНАЧЕНИЯ И ОГРАНИЧЕНИЯ
// ═══════════════════════════════════════════════════════════════════

/** Значения базового бота: id → base. */
export const baseValues = (params: ParamSpec[]): ParamValues =>
  Object.fromEntries(params.map((param) => [param.id, param.base]));

/** Ограничивает число диапазоном и округляет, стараясь не выйти за границы после округления. */
const clampToGrid = (value: number, min: number, max: number, decimals: number): number => {
  const rounded = roundTo(clamp(value, min, max), decimals);
  const unit = 10 ** -normalizeDecimals(decimals);
  if (rounded > max + EPSILON) {
    const lowered = roundTo(rounded - unit, decimals);
    return lowered >= min - EPSILON ? lowered : rounded;
  }
  if (rounded < min - EPSILON) {
    const raised = roundTo(rounded + unit, decimals);
    return raised <= max + EPSILON ? raised : rounded;
  }
  return rounded;
};

/** Для каждого параметра: значение (или база) → clamp в [min, max] → округление. Новый объект. */
export const clampAndRound = (params: ParamSpec[], values: ParamValues): ParamValues =>
  Object.fromEntries(
    params.map((param) => [
      param.id,
      clampToGrid(
        resolveValue(param, values),
        Math.min(param.min, param.max),
        Math.max(param.min, param.max),
        param.decimals,
      ),
    ]),
  );

/**
 * Отступы ордеров строго возрастают: indent[i] ≥ indent[i−1] + 0.01, иначе indent[i] = indent[i−1] + step.
 * Первый ордер не меняется. Пары, которые нарушают правило уже в базовом боте, не трогаем —
 * базовая сетка должна оставаться ровно такой, как в Veles. Монотонность важнее верхней границы max.
 */
export const enforceGridConstraints = (params: ParamSpec[], values: ParamValues): ParamValues => {
  const result: ParamValues = { ...values };
  const indents = params.filter((param) => param.kind === 'orderIndent').sort(compareByOrderIndex);

  for (let index = 1; index < indents.length; index += 1) {
    const previous = indents[index - 1];
    const current = indents[index];
    const baseIsIncreasing = current.base - previous.base >= MIN_INDENT_GAP - EPSILON;
    if (!baseIsIncreasing) {
      continue;
    }
    const previousValue = resolveValue(previous, result);
    const currentValue = resolveValue(current, result);
    if (currentValue - previousValue < MIN_INDENT_GAP - EPSILON) {
      const increment = Math.max(current.step, MIN_INDENT_GAP, 10 ** -normalizeDecimals(current.decimals));
      result[current.id] = roundTo(previousValue + increment, current.decimals);
    }
  }

  return result;
};

/**
 * Масштабирует включённые объёмы так, чтобы сумма всех объёмов равнялась сумме базовых (обычно 100%).
 * Выключенные объёмы не меняются; остаток округления ложится на последний включённый ордер
 * (при нехватке — на предыдущие), каждый объём остаётся ≥ 0.01.
 */
export const normalizeVolumes = (params: ParamSpec[], values: ParamValues): ParamValues => {
  const result: ParamValues = { ...values };
  const volumes = params.filter((param) => param.kind === 'orderVolume').sort(compareByOrderIndex);
  const enabled = volumes.filter((param) => param.enabled);
  if (enabled.length === 0) {
    return result;
  }

  const target = sum(volumes.map((param) => param.base));
  const disabledSum = sum(volumes.filter((param) => !param.enabled).map((param) => resolveValue(param, values)));
  const enabledSum = sum(enabled.map((param) => resolveValue(param, values)));
  const enabledTarget = target - disabledSum;
  if (target <= 0 || enabledSum <= 0 || enabledTarget <= 0) {
    return result;
  }

  const factor = enabledTarget / enabledSum;
  enabled.forEach((param) => {
    result[param.id] = Math.max(MIN_VOLUME, roundTo(resolveValue(param, values) * factor, param.decimals));
  });

  const precision = Math.max(2, ...volumes.map((param) => normalizeDecimals(param.decimals)));
  let remainder = roundTo(target - sum(volumes.map((param) => resolveValue(param, result))), precision);
  for (let index = enabled.length - 1; index >= 0 && remainder !== 0; index -= 1) {
    const param = enabled[index];
    const current = resolveValue(param, result);
    const next = Math.max(MIN_VOLUME, roundTo(current + remainder, param.decimals));
    result[param.id] = next;
    remainder = roundTo(remainder - (next - current), precision);
  }

  return result;
};

/** Полная подготовка значений кандидата: clampAndRound → enforceGridConstraints → normalizeVolumes. */
export const prepareValues = (params: ParamSpec[], values: ParamValues): ParamValues =>
  normalizeVolumes(params, enforceGridConstraints(params, clampAndRound(params, values)));

// ═══════════════════════════════════════════════════════════════════
// ХЕШ, ПАТЧ, РАЗНИЦА
// ═══════════════════════════════════════════════════════════════════

const FNV_OFFSET_BASIS = 0x811c9dc5;
const FNV_PRIME = 0x01000193;

const fnv1a32 = (text: string): string => {
  let hash = FNV_OFFSET_BASIS;
  for (const byte of new TextEncoder().encode(text)) {
    hash ^= byte;
    hash = Math.imul(hash, FNV_PRIME) >>> 0;
  }
  return hash.toString(16).padStart(8, '0');
};

/** Стабильный хеш значений: не зависит от порядка параметров и ключей, значения округляются до decimals. */
export const paramsHash = (params: ParamSpec[], values: ParamValues): string => {
  const serialized = [...params]
    .sort((left, right) => (left.id < right.id ? -1 : left.id > right.id ? 1 : 0))
    .map((param) => `${param.id}=${roundTo(resolveValue(param, values), param.decimals)}`)
    .join(';');
  return `p${fnv1a32(serialized)}`;
};

/**
 * Глубокая копия стратегии с записанными значениями кандидата (после prepareValues).
 * Значения, равные базовым, не записываются — исходные поля остаются нетронутыми,
 * поэтому patchStrategy(s, params, baseValues(params)) совпадает с копией s.
 */
export const patchStrategy = (strategy: BotStrategy, params: ParamSpec[], values: ParamValues): BotStrategy => {
  const patched = cloneDeep(strategy);
  const prepared = prepareValues(params, values);
  params.forEach((param) => {
    const value = resolveValue(param, prepared);
    if (value !== param.base) {
      setAtPath(patched, param.path, value);
    }
  });
  return patched;
};

/** id параметров, значение которых (после округления) отличается от базового. */
export const diffFromBase = (params: ParamSpec[], values: ParamValues): string[] =>
  params
    .filter((param) => roundTo(resolveValue(param, values), param.decimals) !== roundTo(param.base, param.decimals))
    .map((param) => param.id);

/** Человекочитаемое изменение параметра: «CCI 1 час < 46 → 40», «Ордер 4: отступ 15% → 13.8%». */
export const describeParamChange = (param: ParamSpec, value: number): string => {
  const isCondition = CONDITION_KINDS.has(param.kind);
  const unit = PERCENT_KINDS.has(param.kind) ? '%' : '';
  const baseText = `${formatParamValue(param.base, param.decimals)}${unit}`;
  const head = isCondition ? param.label : `${param.label} ${baseText}`;
  if (roundTo(value, param.decimals) === roundTo(param.base, param.decimals)) {
    return `${head} (без изменений)`;
  }
  return `${head} → ${formatParamValue(value, param.decimals)}${unit}`;
};

// ═══════════════════════════════════════════════════════════════════
// СОЗДАНИЕ БОТА
// ═══════════════════════════════════════════════════════════════════

const resolveApiKey = (strategy: BotStrategy): number => {
  // apiKey приходит в полном DTO бота, но отсутствует в типе BotStrategy.
  const raw: unknown = 'apiKey' in strategy ? strategy.apiKey : undefined;
  const parsed = typeof raw === 'string' && raw.trim().length > 0 ? Number(raw) : raw;
  if (!isFiniteNumber(parsed)) {
    throw new Error('У базового бота не найден API-ключ (apiKey) — создать бота по этой стратегии невозможно.');
  }
  return parsed;
};

const firstNonEmptyString = (...values: Array<string | null | undefined>): string | null => {
  for (const value of values) {
    if (typeof value === 'string' && value.trim().length > 0) {
      return value.trim();
    }
  }
  return null;
};

const resolveSymbols = (strategy: BotStrategy): string[] => {
  if (Array.isArray(strategy.symbols) && strategy.symbols.length > 0) {
    return strategy.symbols.filter((symbol): symbol is string => typeof symbol === 'string' && symbol.length > 0);
  }
  const symbol = firstNonEmptyString(strategy.symbol);
  return symbol ? [symbol] : [];
};

/** Payload создания бота (POST /api/bots) из полной (уже пропатченной) стратегии. Все вложенные объекты копируются. */
export const buildBotCreatePayloadFromStrategy = (strategy: BotStrategy, name: string): CreateBotPayload => {
  const trimmedName = name.trim();
  if (trimmedName.length === 0) {
    throw new Error('Укажите название бота.');
  }
  const algorithm = firstNonEmptyString(strategy.algorithm);
  if (!algorithm) {
    throw new Error('В стратегии бота не указан алгоритм (LONG/SHORT).');
  }
  const exchange = firstNonEmptyString(strategy.exchange, strategy.pair?.exchange);
  if (!exchange) {
    throw new Error('В стратегии бота не указана биржа.');
  }
  if (!strategy.deposit) {
    throw new Error('В стратегии бота нет настроек депозита.');
  }

  return {
    algorithm,
    apiKey: resolveApiKey(strategy),
    conditions: cloneDeep(strategy.conditions ?? []),
    deposit: cloneDeep(strategy.deposit),
    exchange,
    id: null,
    name: trimmedName,
    portion: strategy.portion ?? null,
    profit: cloneDeep(strategy.profit ?? null),
    pullUp: strategy.pullUp ?? null,
    settings: cloneDeep(strategy.settings ?? null),
    stopLoss: cloneDeep(strategy.stopLoss ?? null),
    symbols: resolveSymbols(strategy),
    termination: null,
  };
};
