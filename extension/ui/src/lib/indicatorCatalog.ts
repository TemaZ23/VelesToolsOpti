/**
 * Каталог индикаторов Veles
 * Список известных индикаторов с их параметрами
 */

import type { ConditionOperation, IndicatorCategory, IndicatorDefinition } from '../types/optimizer';

const COMPARISON_OPS: ConditionOperation[] = ['>', '<', '>=', '<='];
const CROSS_OPS: ConditionOperation[] = ['CROSS_UP', 'CROSS_DOWN'];
const ALL_OPS: ConditionOperation[] = [...COMPARISON_OPS, ...CROSS_OPS];

export const INDICATOR_CATALOG: IndicatorDefinition[] = [
  // ═══════════════════════════════════════════════════════════════════
  // ТРЕНДОВЫЕ ИНДИКАТОРЫ
  // ═══════════════════════════════════════════════════════════════════
  {
    id: 'ADX',
    name: 'ADX',
    nameRu: 'ADX (Average Directional Index)',
    category: 'trend',
    hasValue: true,
    defaultValue: 25,
    minValue: 0,
    maxValue: 100,
    operations: COMPARISON_OPS,
  },
  {
    id: 'EMA',
    name: 'EMA',
    nameRu: 'EMA (Exponential Moving Average)',
    category: 'trend',
    hasValue: false,
    defaultValue: null,
    minValue: null,
    maxValue: null,
    operations: CROSS_OPS,
  },
  {
    id: 'SMA',
    name: 'SMA',
    nameRu: 'SMA (Simple Moving Average)',
    category: 'trend',
    hasValue: false,
    defaultValue: null,
    minValue: null,
    maxValue: null,
    operations: CROSS_OPS,
  },
  {
    id: 'MACD',
    name: 'MACD',
    nameRu: 'MACD',
    category: 'trend',
    hasValue: false,
    defaultValue: null,
    minValue: null,
    maxValue: null,
    operations: CROSS_OPS,
  },
  {
    id: 'PSAR',
    name: 'Parabolic SAR',
    nameRu: 'Parabolic SAR',
    category: 'trend',
    hasValue: false,
    defaultValue: null,
    minValue: null,
    maxValue: null,
    operations: CROSS_OPS,
  },

  // ═══════════════════════════════════════════════════════════════════
  // КАНАЛЬНЫЕ ИНДИКАТОРЫ
  // ═══════════════════════════════════════════════════════════════════
  {
    id: 'BB',
    name: 'Bollinger Bands',
    nameRu: 'Полосы Боллинджера',
    category: 'channel',
    hasValue: false,
    defaultValue: null,
    minValue: null,
    maxValue: null,
    operations: CROSS_OPS,
  },
  {
    id: 'KELTNER',
    name: 'Keltner Channel',
    nameRu: 'Канал Кельтнера',
    category: 'channel',
    hasValue: false,
    defaultValue: null,
    minValue: null,
    maxValue: null,
    operations: CROSS_OPS,
  },
  {
    id: 'TURTLE',
    name: 'Turtle Zone',
    nameRu: 'Turtle Zone',
    category: 'channel',
    hasValue: false,
    defaultValue: null,
    minValue: null,
    maxValue: null,
    operations: CROSS_OPS,
  },
  {
    id: 'DONCHIAN',
    name: 'Donchian Channel',
    nameRu: 'Канал Дончиана',
    category: 'channel',
    hasValue: false,
    defaultValue: null,
    minValue: null,
    maxValue: null,
    operations: CROSS_OPS,
  },

  // ═══════════════════════════════════════════════════════════════════
  // ОСЦИЛЛЯТОРЫ
  // ═══════════════════════════════════════════════════════════════════
  {
    id: 'RSI',
    name: 'RSI',
    nameRu: 'RSI (Relative Strength Index)',
    category: 'oscillator',
    hasValue: true,
    defaultValue: 30,
    minValue: 0,
    maxValue: 100,
    operations: COMPARISON_OPS,
  },
  {
    id: 'RSI_LEVELS',
    name: 'RSI Levels',
    nameRu: 'Уровни RSI',
    category: 'oscillator',
    hasValue: false,
    defaultValue: null,
    minValue: null,
    maxValue: null,
    operations: CROSS_OPS,
  },
  {
    id: 'CCI',
    name: 'CCI',
    nameRu: 'CCI (Commodity Channel Index)',
    category: 'oscillator',
    hasValue: true,
    defaultValue: 100,
    minValue: -300,
    maxValue: 300,
    operations: COMPARISON_OPS,
  },
  {
    id: 'CCI_LEVELS',
    name: 'CCI Levels',
    nameRu: 'Уровни CCI',
    category: 'oscillator',
    hasValue: false,
    defaultValue: null,
    minValue: null,
    maxValue: null,
    operations: CROSS_OPS,
  },
  {
    id: 'CMO',
    name: 'CMO',
    nameRu: 'CMO (Chande Momentum Oscillator)',
    category: 'oscillator',
    hasValue: true,
    defaultValue: -50,
    minValue: -100,
    maxValue: 100,
    operations: COMPARISON_OPS,
  },
  {
    id: 'WILLIAMS_R',
    name: 'Williams %R',
    nameRu: 'Williams %R',
    category: 'oscillator',
    hasValue: true,
    defaultValue: -80,
    minValue: -100,
    maxValue: 0,
    operations: COMPARISON_OPS,
  },
  {
    id: 'STOCHASTIC',
    name: 'Stochastic',
    nameRu: 'Стохастик',
    category: 'oscillator',
    hasValue: true,
    defaultValue: 20,
    minValue: 0,
    maxValue: 100,
    operations: COMPARISON_OPS,
  },
  {
    id: 'STOCHASTIC_LEVELS',
    name: 'Stochastic Levels',
    nameRu: 'Стохастик, уровни',
    category: 'oscillator',
    hasValue: false,
    defaultValue: null,
    minValue: null,
    maxValue: null,
    operations: CROSS_OPS,
  },
  {
    id: 'MFI',
    name: 'MFI',
    nameRu: 'MFI (Money Flow Index)',
    category: 'oscillator',
    hasValue: true,
    defaultValue: 20,
    minValue: 0,
    maxValue: 100,
    operations: COMPARISON_OPS,
  },
  {
    id: 'ROC',
    name: 'ROC',
    nameRu: 'ROC (Rate of Change)',
    category: 'oscillator',
    hasValue: true,
    defaultValue: 0,
    minValue: -100,
    maxValue: 100,
    operations: COMPARISON_OPS,
  },
  {
    id: 'MOMENTUM',
    name: 'Momentum',
    nameRu: 'Momentum',
    category: 'oscillator',
    hasValue: true,
    defaultValue: 0,
    minValue: -100,
    maxValue: 100,
    operations: COMPARISON_OPS,
  },

  // ═══════════════════════════════════════════════════════════════════
  // ВОЛАТИЛЬНОСТЬ
  // ═══════════════════════════════════════════════════════════════════
  {
    id: 'ATR',
    name: 'ATR',
    nameRu: 'ATR (Average True Range)',
    category: 'volatility',
    hasValue: true,
    defaultValue: 1,
    minValue: 0,
    maxValue: 100,
    operations: COMPARISON_OPS,
  },
  {
    id: 'ATR_PERCENT',
    name: 'ATR%',
    nameRu: 'ATR%',
    category: 'volatility',
    hasValue: true,
    defaultValue: 0.5,
    minValue: 0,
    maxValue: 10,
    operations: COMPARISON_OPS,
  },

  // ═══════════════════════════════════════════════════════════════════
  // ОБЪЁМ
  // ═══════════════════════════════════════════════════════════════════
  {
    id: 'VOLUME',
    name: 'Volume',
    nameRu: 'Объём (номинальный)',
    category: 'volume',
    hasValue: true,
    defaultValue: 100000,
    minValue: 0,
    maxValue: 100000000,
    operations: COMPARISON_OPS,
  },
  {
    id: 'OBV',
    name: 'OBV',
    nameRu: 'OBV (On Balance Volume)',
    category: 'volume',
    hasValue: false,
    defaultValue: null,
    minValue: null,
    maxValue: null,
    operations: CROSS_OPS,
  },
  {
    id: 'VWAP',
    name: 'VWAP',
    nameRu: 'VWAP',
    category: 'volume',
    hasValue: false,
    defaultValue: null,
    minValue: null,
    maxValue: null,
    operations: CROSS_OPS,
  },
];

// Индексы для быстрого поиска
export const INDICATORS_BY_ID = new Map<string, IndicatorDefinition>(
  INDICATOR_CATALOG.map((ind) => [ind.id, ind]),
);

export const INDICATORS_BY_CATEGORY = INDICATOR_CATALOG.reduce(
  (acc, ind) => {
    if (!acc[ind.category]) {
      acc[ind.category] = [];
    }
    acc[ind.category].push(ind);
    return acc;
  },
  {} as Record<IndicatorCategory, IndicatorDefinition[]>,
);

// Индикаторы с числовыми значениями (для оптимизации порогов)
export const INDICATORS_WITH_VALUES = INDICATOR_CATALOG.filter((ind) => ind.hasValue);

// Индикаторы без значений (канальные, кроссы)
export const INDICATORS_WITHOUT_VALUES = INDICATOR_CATALOG.filter((ind) => !ind.hasValue);

// Категории на русском
export const CATEGORY_LABELS: Record<IndicatorCategory, string> = {
  trend: 'Трендовые',
  channel: 'Канальные',
  oscillator: 'Осцилляторы',
  volatility: 'Волатильность',
  volume: 'Объём',
};

/**
 * Получить индикатор по ID
 */
export const getIndicatorById = (id: string): IndicatorDefinition | null => {
  return INDICATORS_BY_ID.get(id) ?? null;
};

/**
 * Получить случайный индикатор
 */
export const getRandomIndicator = (): IndicatorDefinition => {
  const index = Math.floor(Math.random() * INDICATOR_CATALOG.length);
  return INDICATOR_CATALOG[index];
};

/**
 * Получить случайный индикатор из категории
 */
export const getRandomIndicatorFromCategory = (category: IndicatorCategory): IndicatorDefinition => {
  const indicators = INDICATORS_BY_CATEGORY[category];
  const index = Math.floor(Math.random() * indicators.length);
  return indicators[index];
};

/**
 * Получить случайное значение для индикатора
 */
export const getRandomValueForIndicator = (indicator: IndicatorDefinition): number | null => {
  if (!indicator.hasValue || indicator.minValue === null || indicator.maxValue === null) {
    return null;
  }
  const range = indicator.maxValue - indicator.minValue;
  return indicator.minValue + Math.random() * range;
};

/**
 * Мутировать значение индикатора
 */
export const mutateIndicatorValue = (
  indicator: IndicatorDefinition,
  currentValue: number | null,
  mutationStrength = 0.2,
): number | null => {
  if (!indicator.hasValue || currentValue === null || indicator.minValue === null || indicator.maxValue === null) {
    return null;
  }

  const range = indicator.maxValue - indicator.minValue;
  const mutation = (Math.random() - 0.5) * 2 * mutationStrength * range;
  const newValue = currentValue + mutation;

  return Math.max(indicator.minValue, Math.min(indicator.maxValue, newValue));
};

/**
 * Получить случайную операцию для индикатора
 */
export const getRandomOperation = (indicator: IndicatorDefinition): string => {
  const ops = indicator.operations;
  const index = Math.floor(Math.random() * ops.length);
  return ops[index];
};
