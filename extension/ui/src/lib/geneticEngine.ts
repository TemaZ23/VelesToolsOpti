/**
 * Генетический движок оптимизатора
 * Реализует генетический алгоритм для поиска оптимальных параметров бота
 */

import {
  getIndicatorById,
  getRandomIndicator,
  getRandomOperation,
  getRandomValueForIndicator,
  INDICATOR_CATALOG,
  INDICATORS_WITH_VALUES,
  mutateIndicatorValue,
} from './indicatorCatalog';
import type {
  BotGenome,
  ConditionGene,
  EvaluatedGenome,
  GeneticConfig,
  GenomeFitness,
  GridOrderGene,
  OrderOptimizationConfig,
  OptimizationScope,
  OptimizationTarget,
  StopLossGene,
  TakeProfitGene,
  TimeInterval,
} from '../types/optimizer';
import { TIME_INTERVALS } from '../types/optimizer';

// ═══════════════════════════════════════════════════════════════════
// УТИЛИТЫ
// ═══════════════════════════════════════════════════════════════════

const generateId = (): string => {
  return `genome-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
};

const randomChoice = <T>(array: T[]): T => {
  return array[Math.floor(Math.random() * array.length)];
};

const randomInt = (min: number, max: number): number => {
  return Math.floor(Math.random() * (max - min + 1)) + min;
};

const randomFloat = (min: number, max: number): number => {
  return min + Math.random() * (max - min);
};

const clamp = (value: number, min: number, max: number): number => {
  return Math.max(min, Math.min(max, value));
};

/**
 * Нормализация объёмов чтобы сумма baseOrder + DCA = 100%
 * Сохраняет пропорции между ордерами.
 * Зафиксированные ордера (locked / !optimizeVolume) не масштабируются.
 */
const normalizeVolumes = (
  baseOrder: GridOrderGene,
  dcaOrders: GridOrderGene[],
  orderCfgs?: OrderOptimizationConfig[],
): void => {
  const isVolumeLocked = (idx: number): boolean => {
    if (!orderCfgs) return false;
    const cfg = orderCfgs.find((c) => c.index === idx);
    return cfg ? cfg.locked || !cfg.optimizeVolume : false;
  };

  // Считаем фиксированную и свободную суммы
  let lockedTotal = 0;
  let freeTotal = 0;

  if (isVolumeLocked(0)) {
    lockedTotal += baseOrder.volume;
  } else {
    freeTotal += baseOrder.volume;
  }

  for (let i = 0; i < dcaOrders.length; i++) {
    if (isVolumeLocked(i + 1)) {
      lockedTotal += dcaOrders[i].volume;
    } else {
      freeTotal += dcaOrders[i].volume;
    }
  }

  const freeTarget = 100 - lockedTotal;

  // Если нечего масштабировать или целевое значение неположительное — fallback
  if (freeTotal <= 0 || freeTarget <= 0) return;

  const scale = freeTarget / freeTotal;

  if (!isVolumeLocked(0)) {
    baseOrder.volume = Math.max(1, baseOrder.volume * scale);
  }
  for (let i = 0; i < dcaOrders.length; i++) {
    if (!isVolumeLocked(i + 1)) {
      dcaOrders[i].volume = Math.max(1, dcaOrders[i].volume * scale);
    }
  }

  // Корректируем чтобы сумма была ровно 100% (из-за округления)
  const newTotal = baseOrder.volume + dcaOrders.reduce((sum, o) => sum + o.volume, 0);
  if (Math.abs(newTotal - 100) > 0.01) {
    const diff = 100 - newTotal;
    // Ищем последний нефиксированный ордер для коррекции
    for (let i = dcaOrders.length - 1; i >= 0; i--) {
      if (!isVolumeLocked(i + 1)) {
        dcaOrders[i].volume = Math.max(1, dcaOrders[i].volume + diff);
        return;
      }
    }
    if (!isVolumeLocked(0)) {
      baseOrder.volume = Math.max(1, baseOrder.volume + diff);
    }
  }
};

// ═══════════════════════════════════════════════════════════════════
// ГЕНЕРАЦИЯ СЛУЧАЙНЫХ ГЕНОВ
// ═══════════════════════════════════════════════════════════════════

/**
 * Создать случайное условие (ген)
 */
export const createRandomCondition = (): ConditionGene => {
  const indicator = getRandomIndicator();
  const interval = randomChoice(TIME_INTERVALS);
  const operation = getRandomOperation(indicator);
  const value = getRandomValueForIndicator(indicator);
  
  // basic = true для сигнальных индикаторов (без числовых значений)
  const basic = !indicator.hasValue;

  return {
    indicator: indicator.id,
    interval,
    value,
    operation: operation as ConditionGene['operation'],
    closed: Math.random() > 0.5,
    reverse: false,
    basic,
  };
};

/**
 * Создать случайный ордер сетки
 */
export const createRandomGridOrder = (
  minIndent: number,
  maxIndent: number,
  minVolume: number,
  maxVolume: number,
  withConditions: boolean,
): GridOrderGene => {
  const conditions: ConditionGene[] = [];

  if (withConditions && Math.random() > 0.5) {
    const conditionCount = randomInt(1, 3);
    for (let i = 0; i < conditionCount; i++) {
      conditions.push(createRandomCondition());
    }
  }

  return {
    indent: randomFloat(minIndent, maxIndent),
    volume: randomFloat(minVolume, maxVolume),
    conditions,
  };
};

/**
 * Создать случайный тейк-профит
 */
export const createRandomTakeProfit = (withIndicator: boolean): TakeProfitGene => {
  const types: TakeProfitGene['type'][] = ['PERCENT', 'PNL'];
  const type = randomChoice(types);
  const value = type === 'PERCENT' ? randomFloat(0.5, 5) : randomFloat(0.5, 10);

  let indicator: ConditionGene | null = null;
  if (withIndicator && Math.random() > 0.3) {
    indicator = createRandomCondition();
  }

  return { type, value, indicator };
};

/**
 * Создать случайный стоп-лосс
 */
export const createRandomStopLoss = (): StopLossGene => {
  const conditions: ConditionGene[] = [];
  if (Math.random() > 0.5) {
    conditions.push(createRandomCondition());
  }

  return {
    indent: randomFloat(2, 20),
    termination: Math.random() > 0.5,
    conditionalIndent: Math.random() > 0.5 ? randomFloat(1, 10) : null,
    conditions,
  };
};

// ═══════════════════════════════════════════════════════════════════
// СОЗДАНИЕ ГЕНОМА
// ═══════════════════════════════════════════════════════════════════

/**
 * Создать случайный геном бота
 */
export const createRandomGenome = (generation = 0): BotGenome => {
  // Условия входа (2-5 индикаторов)
  const entryConditionCount = randomInt(2, 5);
  const entryConditions: ConditionGene[] = [];
  for (let i = 0; i < entryConditionCount; i++) {
    entryConditions.push(createRandomCondition());
  }

  // Базовый ордер (indent=0 для base order — стандарт Veles)
  const baseOrder = createRandomGridOrder(0, 0, 5, 15, false);

  // DCA ордера (3-7 штук) с реалистичными отступами (до ~25%)
  const dcaCount = randomInt(3, 7);
  const dcaOrders: GridOrderGene[] = [];
  let prevIndent = 0;

  for (let i = 0; i < dcaCount; i++) {
    const minIndent = prevIndent + 0.5;
    const maxIndent = Math.min(prevIndent + 5, 25);
    if (minIndent >= maxIndent) break;
    const order = createRandomGridOrder(minIndent, maxIndent, 5, 35, Math.random() > 0.7);
    dcaOrders.push(order);
    prevIndent = order.indent;
  }

  // Нормализуем объёмы чтобы сумма была 100%
  normalizeVolumes(baseOrder, dcaOrders);

  return {
    id: generateId(),
    generation,
    algorithm: Math.random() > 0.5 ? 'LONG' : 'SHORT',
    leverage: randomInt(3, 20),
    depositAmount: randomFloat(5, 50),
    entryConditions,
    baseOrder,
    dcaOrders,
    takeProfit: createRandomTakeProfit(true),
    stopLoss: Math.random() > 0.3 ? createRandomStopLoss() : null,
    pullUp: Math.random() > 0.5 ? randomFloat(0, 50) : null,
    portion: Math.random() > 0.7 ? randomFloat(10, 50) : null,
  };
};

/**
 * Создать геном на основе существующего бота
 */
export const createGenomeFromBot = (
  botConfig: Record<string, unknown>,
  generation = 0,
): BotGenome => {
  // TODO: Парсинг реальной конфигурации бота
  // Пока возвращаем случайный геном
  return createRandomGenome(generation);
};

// ═══════════════════════════════════════════════════════════════════
// ГЕНЕТИЧЕСКИЕ ОПЕРАЦИИ
// ═══════════════════════════════════════════════════════════════════

/**
 * Мутация условия
 */
export const mutateCondition = (condition: ConditionGene, mutationStrength = 0.3): ConditionGene => {
  const mutated = { ...condition };

  // Мутация индикатора (20% шанс)
  if (Math.random() < 0.2) {
    const newIndicator = getRandomIndicator();
    mutated.indicator = newIndicator.id;
    mutated.operation = getRandomOperation(newIndicator) as ConditionGene['operation'];
    mutated.value = getRandomValueForIndicator(newIndicator);
    mutated.basic = !newIndicator.hasValue; // Обновляем basic
  }
  // Мутация интервала (30% шанс)
  else if (Math.random() < 0.3) {
    mutated.interval = randomChoice(TIME_INTERVALS);
  }
  // Мутация значения (50% шанс) - только для не-basic индикаторов
  else if (mutated.value !== null && !mutated.basic) {
    const indicator = getIndicatorById(mutated.indicator);
    if (indicator) {
      mutated.value = mutateIndicatorValue(indicator, mutated.value, mutationStrength);
    }
  }

  return mutated;
};

/**
 * Мутация ордера сетки
 * @param order - ордер для мутации
 * @param mutationStrength - сила мутации (0.3 = ±30%, 0.5 = ±50%)
 * @param guaranteed - гарантировать мутацию (для начальной популяции)
 * @param orderCfg - per-order конфиг (если задан, диапазоны indent/volume ограничены)
 */
export const mutateGridOrder = (
  order: GridOrderGene,
  mutationStrength = 0.3,
  guaranteed = false,
  orderCfg?: OrderOptimizationConfig,
): GridOrderGene => {
  // Если ордер зафиксирован — возвращаем без изменений
  if (orderCfg?.locked) {
    return { ...order, conditions: [...order.conditions] };
  }

  const mutated = { ...order, conditions: [...order.conditions] };

  // Определяем, мутировать ли indent/volume (per-order конфиг имеет приоритет)
  const shouldMutateIndent = orderCfg ? orderCfg.optimizeIndent : true;
  const shouldMutateVolume = orderCfg ? orderCfg.optimizeVolume : true;

  // Мутация отступа - при guaranteed=true всегда мутируем
  if (shouldMutateIndent && (guaranteed || Math.random() < 0.6)) {
    const relativeChange = (Math.random() - 0.5) * 2 * mutationStrength * order.indent;
    const minAbsoluteChange = order.indent * 0.1; // минимум 10% изменение
    const delta = Math.abs(relativeChange) < minAbsoluteChange 
      ? (relativeChange >= 0 ? minAbsoluteChange : -minAbsoluteChange)
      : relativeChange;
    const minIndent = orderCfg?.indentRange?.[0] ?? 0.01;
    const maxIndent = orderCfg?.indentRange?.[1] ?? 50;
    mutated.indent = clamp(order.indent + delta, Math.max(0.01, minIndent), maxIndent);
  }

  // Мутация объёма - при guaranteed=true всегда мутируем
  if (shouldMutateVolume && (guaranteed || Math.random() < 0.6)) {
    const relativeChange = (Math.random() - 0.5) * 2 * mutationStrength * order.volume;
    const minAbsoluteChange = order.volume * 0.1;
    const delta = Math.abs(relativeChange) < minAbsoluteChange
      ? (relativeChange >= 0 ? minAbsoluteChange : -minAbsoluteChange)
      : relativeChange;
    const minVolume = orderCfg?.volumeRange?.[0] ?? 1;
    const maxVolume = orderCfg?.volumeRange?.[1] ?? 50;
    mutated.volume = clamp(order.volume + delta, Math.max(1, minVolume), maxVolume);
  }

  // Мутация условий
  if (Math.random() < 0.3 && mutated.conditions.length > 0) {
    const idx = randomInt(0, mutated.conditions.length - 1);
    mutated.conditions[idx] = mutateCondition(mutated.conditions[idx], mutationStrength);
  }

  // Добавление/удаление условия
  if (Math.random() < 0.2) {
    if (mutated.conditions.length > 0 && Math.random() < 0.5) {
      mutated.conditions.pop();
    } else if (mutated.conditions.length < 4) {
      mutated.conditions.push(createRandomCondition());
    }
  }

  return mutated;
};

/**
 * Мутация генома
 */
export const mutateGenome = (genome: BotGenome, scope: OptimizationScope, mutationRate = 0.3): BotGenome => {
  const mutated: BotGenome = {
    ...genome,
    id: generateId(),
    entryConditions: [...genome.entryConditions],
    baseOrder: { ...genome.baseOrder, conditions: [...genome.baseOrder.conditions] },
    dcaOrders: genome.dcaOrders.map((o) => ({ ...o, conditions: [...o.conditions] })),
    takeProfit: { ...genome.takeProfit },
    stopLoss: genome.stopLoss ? { ...genome.stopLoss, conditions: [...genome.stopLoss.conditions] } : null,
  };

  // Мутация условий входа
  if (scope.entryConditions && Math.random() < mutationRate) {
    if (scope.entryConditionIndicators) {
      // Полная мутация (можем менять индикаторы)
      const idx = randomInt(0, mutated.entryConditions.length - 1);
      mutated.entryConditions[idx] = mutateCondition(mutated.entryConditions[idx]);
    } else if (scope.entryConditionValues) {
      // Только значения
      const idx = randomInt(0, mutated.entryConditions.length - 1);
      const cond = mutated.entryConditions[idx];
      if (cond.value !== null) {
        const indicator = getIndicatorById(cond.indicator);
        if (indicator) {
          mutated.entryConditions[idx] = {
            ...cond,
            value: mutateIndicatorValue(indicator, cond.value),
          };
        }
      }
    }
  }

  // Добавление/удаление условия входа
  if (scope.entryConditionIndicators && Math.random() < mutationRate * 0.5) {
    if (mutated.entryConditions.length > 2 && Math.random() < 0.5) {
      const idx = randomInt(0, mutated.entryConditions.length - 1);
      mutated.entryConditions.splice(idx, 1);
    } else if (mutated.entryConditions.length < 10) {
      mutated.entryConditions.push(createRandomCondition());
    }
  }

  // Флаг гарантированной мутации (для высокого mutationRate - начальная популяция)
  const guaranteedMutation = mutationRate >= 0.7;

  // Per-order конфиги (если заданы)
  const orderCfgs = scope.orderConfigs;
  const getOrderCfg = (idx: number): OrderOptimizationConfig | undefined =>
    orderCfgs?.find((c) => c.index === idx);

  // Мутация базового ордера (indent + volume)
  if ((scope.dcaIndents || scope.dcaVolumes) && (guaranteedMutation || Math.random() < mutationRate)) {
    const baseCfg = getOrderCfg(0);
    // Пропускаем если базовый ордер зафиксирован
    if (!baseCfg?.locked) {
      mutated.baseOrder = mutateGridOrder(mutated.baseOrder, 0.5, guaranteedMutation, baseCfg);
    }
  }

  // Мутация сетки DCA - мутируем КАЖДЫЙ ордер
  if (scope.dcaIndents || scope.dcaVolumes || scope.dcaConditions) {
    for (let i = 0; i < mutated.dcaOrders.length; i++) {
      const dcaCfg = getOrderCfg(i + 1); // index=0 — base, DCA начинается с 1
      // Пропускаем зафиксированные ордера
      if (dcaCfg?.locked) continue;
      // При guaranteedMutation мутируем все ордера, иначе по вероятности
      if (guaranteedMutation || Math.random() < mutationRate) {
        mutated.dcaOrders[i] = mutateGridOrder(mutated.dcaOrders[i], 0.5, guaranteedMutation, dcaCfg);
      }
    }
  }

  // Мутация структуры сетки (добавление/удаление ордеров)
  if (scope.dcaStructure && Math.random() < mutationRate * 0.3) {
    if (mutated.dcaOrders.length > 3 && Math.random() < 0.5) {
      const idx = randomInt(0, mutated.dcaOrders.length - 1);
      mutated.dcaOrders.splice(idx, 1);
    } else if (mutated.dcaOrders.length < 10) {
      const lastIndent = mutated.dcaOrders[mutated.dcaOrders.length - 1]?.indent ?? 10;
      mutated.dcaOrders.push(createRandomGridOrder(lastIndent + 5, lastIndent + 30, 5, 25, true));
    }
  }

  // Мутация тейк-профита
  if (scope.takeProfit && (guaranteedMutation || Math.random() < mutationRate)) {
    const tpCfg = scope.takeProfitConfig;
    // Пропускаем если зафиксирован
    if (!tpCfg?.locked) {
      // Минимум ±10% изменение или ±0.2 абсолютно
      const relativeChange = (Math.random() - 0.5) * 2 * 0.5 * mutated.takeProfit.value;
      const minChange = Math.max(0.2, mutated.takeProfit.value * 0.1);
      const delta = Math.abs(relativeChange) < minChange
        ? (relativeChange >= 0 ? minChange : -minChange)
        : relativeChange;
      const tpMin = tpCfg?.valueRange?.[0] ?? 0.1;
      const tpMax = tpCfg?.valueRange?.[1] ?? 10;
      mutated.takeProfit.value = clamp(mutated.takeProfit.value + delta, tpMin, tpMax);
    }
  }

  if (scope.takeProfitIndicator && mutated.takeProfit.indicator && Math.random() < mutationRate) {
    mutated.takeProfit.indicator = mutateCondition(mutated.takeProfit.indicator);
  }

  // Мутация стоп-лосса (только если он есть в базовом боте)
  if (scope.stopLoss && mutated.stopLoss && (guaranteedMutation || Math.random() < mutationRate)) {
    // Минимум ±10% изменение или ±0.5 абсолютно
    const relativeChange = (Math.random() - 0.5) * 2 * 0.5 * mutated.stopLoss.indent;
    const minChange = Math.max(0.5, mutated.stopLoss.indent * 0.1);
    const delta = Math.abs(relativeChange) < minChange
      ? (relativeChange >= 0 ? minChange : -minChange)
      : relativeChange;
    mutated.stopLoss.indent = clamp(mutated.stopLoss.indent + delta, 1, 50); // Мин 1%, макс 50%
  }

  // Мутация плеча (только если включено в scope)
  if (scope.leverage && (guaranteedMutation || Math.random() < mutationRate)) {
    const delta = randomInt(-5, 5);
    mutated.leverage = clamp(mutated.leverage + delta, 1, 125);
  }

  // Депозит НЕ мутируется - это константа

  // Мутация pullUp
  if (mutated.pullUp !== null && Math.random() < mutationRate * 0.5) {
    const delta = (Math.random() - 0.5) * 20;
    mutated.pullUp = clamp(mutated.pullUp + delta, 0, 100);
  }

  // ВАЖНО: Нормализуем объёмы чтобы сумма была 100%
  if (scope.dcaVolumes) {
    normalizeVolumes(mutated.baseOrder, mutated.dcaOrders, scope.orderConfigs);
  }

  return mutated;
};

/**
 * Скрещивание двух геномов
 */
export const crossover = (parent1: BotGenome, parent2: BotGenome, scope: OptimizationScope): BotGenome => {
  const orderCfgs = scope.orderConfigs;
  const isOrderLocked = (idx: number): boolean =>
    orderCfgs?.find((c) => c.index === idx)?.locked ?? false;

  const child: BotGenome = {
    id: generateId(),
    generation: Math.max(parent1.generation, parent2.generation) + 1,
    algorithm: parent1.algorithm, // Сохраняем направление от первого родителя
    leverage: Math.random() > 0.5 ? parent1.leverage : parent2.leverage,
    depositAmount: Math.random() > 0.5 ? parent1.depositAmount : parent2.depositAmount,

    // Условия входа - берём случайную комбинацию
    entryConditions: scope.entryConditions
      ? mixConditions(parent1.entryConditions, parent2.entryConditions)
      : [...parent1.entryConditions],

    // Базовый ордер — если зафиксирован, всегда от parent1 (носитель оригинала)
    baseOrder: isOrderLocked(0)
      ? { ...parent1.baseOrder }
      : Math.random() > 0.5 ? { ...parent1.baseOrder } : { ...parent2.baseOrder },

    // DCA — зафиксированные ордера берём от parent1, остальные смешиваем
    dcaOrders: scope.dcaStructure
      ? mixDcaOrders(parent1.dcaOrders, parent2.dcaOrders)
      : parent1.dcaOrders.map((o, i) => {
          if (isOrderLocked(i + 1)) return { ...parent1.dcaOrders[i] };
          return Math.random() > 0.5
            ? { ...parent1.dcaOrders[i] }
            : (parent2.dcaOrders[i] ? { ...parent2.dcaOrders[i] } : { ...parent1.dcaOrders[i] });
        }),

    // Тейк-профит
    takeProfit: Math.random() > 0.5 ? { ...parent1.takeProfit } : { ...parent2.takeProfit },

    // Стоп-лосс
    stopLoss:
      Math.random() > 0.5
        ? parent1.stopLoss
          ? { ...parent1.stopLoss }
          : null
        : parent2.stopLoss
          ? { ...parent2.stopLoss }
          : null,

    pullUp: Math.random() > 0.5 ? parent1.pullUp : parent2.pullUp,
    portion: Math.random() > 0.5 ? parent1.portion : parent2.portion,
  };

  // Нормализуем объёмы после скрещивания
  normalizeVolumes(child.baseOrder, child.dcaOrders, orderCfgs);

  return child;
};

/**
 * Смешивание условий от двух родителей
 */
const mixConditions = (conds1: ConditionGene[], conds2: ConditionGene[]): ConditionGene[] => {
  const all = [...conds1, ...conds2];
  const count = randomInt(Math.min(conds1.length, conds2.length), Math.max(conds1.length, conds2.length));
  const result: ConditionGene[] = [];

  for (let i = 0; i < count && all.length > 0; i++) {
    const idx = randomInt(0, all.length - 1);
    result.push({ ...all[idx] });
    all.splice(idx, 1);
  }

  return result;
};

/**
 * Смешивание DCA ордеров от двух родителей
 */
const mixDcaOrders = (orders1: GridOrderGene[], orders2: GridOrderGene[]): GridOrderGene[] => {
  const count = randomInt(Math.min(orders1.length, orders2.length), Math.max(orders1.length, orders2.length));
  const result: GridOrderGene[] = [];

  for (let i = 0; i < count; i++) {
    const source = Math.random() > 0.5 ? orders1 : orders2;
    if (i < source.length) {
      result.push({ ...source[i], conditions: [...source[i].conditions] });
    } else {
      const other = source === orders1 ? orders2 : orders1;
      if (i < other.length) {
        result.push({ ...other[i], conditions: [...other[i].conditions] });
      }
    }
  }

  // Сортируем по отступу
  result.sort((a, b) => a.indent - b.indent);

  return result;
};

// ═══════════════════════════════════════════════════════════════════
// NSGA-II MULTI-OBJECTIVE RANKING
// ═══════════════════════════════════════════════════════════════════

/**
 * Объективы NSGA-II (все maximize):
 *   1) totalPnl       — прибыльность
 *   2) maxDrawdown    — риск (менее отрицательный = лучше)
 *   3) winRate        — стабильность
 */
const getObjectives = (f: GenomeFitness): [number, number, number] => [
  f.totalPnl,
  f.maxDrawdown,  // negative, so maximizing = less loss
  f.winRate,
];

/**
 * A доминирует B, если A >= B по всем объективам и строго > хотя бы по одному.
 */
const dominates = (a: [number, number, number], b: [number, number, number]): boolean => {
  let strictlyBetter = false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] < b[i]) return false;
    if (a[i] > b[i]) strictlyBetter = true;
  }
  return strictlyBetter;
};

/**
 * Non-dominated sorting (разбиение на Pareto-слои).
 * Возвращает массив фронтов: fronts[0] = rank 1 (Pareto front), fronts[1] = rank 2, ...
 */
const nonDominatedSort = (population: EvaluatedGenome[]): EvaluatedGenome[][] => {
  const n = population.length;
  const objectives = population.map((p) => getObjectives(p.fitness));
  const dominationCount = new Array<number>(n).fill(0);
  const dominated: number[][] = Array.from({ length: n }, () => []);
  const fronts: EvaluatedGenome[][] = [];
  let currentFront: number[] = [];

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (dominates(objectives[i], objectives[j])) {
        dominated[i].push(j);
        dominationCount[j]++;
      } else if (dominates(objectives[j], objectives[i])) {
        dominated[j].push(i);
        dominationCount[i]++;
      }
    }
    if (dominationCount[i] === 0) {
      currentFront.push(i);
    }
  }

  while (currentFront.length > 0) {
    const front = currentFront.map((i) => population[i]);
    fronts.push(front);
    const nextFront: number[] = [];
    for (const i of currentFront) {
      for (const j of dominated[i]) {
        dominationCount[j]--;
        if (dominationCount[j] === 0) {
          nextFront.push(j);
        }
      }
    }
    currentFront = nextFront;
  }

  return fronts;
};

/**
 * Crowding distance — насколько решение уникально в пространстве объективов.
 * Больше = лучше (сохраняет разнообразие).
 */
const computeCrowdingDistance = (front: EvaluatedGenome[]): void => {
  const n = front.length;
  if (n <= 2) {
    for (const ind of front) ind.fitness.crowdingDistance = Number.POSITIVE_INFINITY;
    return;
  }
  for (const ind of front) ind.fitness.crowdingDistance = 0;

  // Для каждого объектива
  for (let m = 0; m < 3; m++) {
    const sorted = [...front].sort((a, b) => {
      const objA = getObjectives(a.fitness);
      const objB = getObjectives(b.fitness);
      return objA[m] - objB[m];
    });

    const objValues = sorted.map((s) => getObjectives(s.fitness)[m]);
    const range = objValues[n - 1] - objValues[0];
    if (range === 0) continue;

    // Крайние = бесконечность
    sorted[0].fitness.crowdingDistance = Number.POSITIVE_INFINITY;
    sorted[n - 1].fitness.crowdingDistance = Number.POSITIVE_INFINITY;

    for (let i = 1; i < n - 1; i++) {
      sorted[i].fitness.crowdingDistance += (objValues[i + 1] - objValues[i - 1]) / range;
    }
  }
};

/**
 * Назначить NSGA-II ранги и crowding distance всей популяции.
 * Также помечает paretoOptimal для rank=1.
 */
export const assignNsgaRanking = (population: EvaluatedGenome[]): void => {
  if (population.length === 0) return;

  const fronts = nonDominatedSort(population);

  for (let rank = 0; rank < fronts.length; rank++) {
    const front = fronts[rank];
    computeCrowdingDistance(front);
    for (const ind of front) {
      ind.fitness.nsgaRank = rank + 1; // 1-based
      ind.paretoOptimal = rank === 0;
    }
  }
};

// ═══════════════════════════════════════════════════════════════════
// ОЦЕНКА FITNESS
// ═══════════════════════════════════════════════════════════════════

/**
 * Utility score для совместимости / дисплея / seeding.
 * НЕ используется в NSGA-II селекции (там rank + crowding distance).
 */
export const calculateScore = (fitness: Omit<GenomeFitness, 'score' | 'nsgaRank' | 'crowdingDistance'>, target: OptimizationTarget): number => {
  const normalizedPnl = Math.max(0, fitness.totalPnl) / 1000;
  const normalizedWinRate = fitness.winRate / 100;
  const normalizedDrawdown = 1 - Math.min(1, Math.abs(fitness.maxDrawdown) / 50);
  const normalizedRatio = Math.min(1, fitness.pnlToRisk / 3);

  switch (target.metric) {
    case 'pnl':
      return fitness.totalPnl;
    case 'pnlPerDay':
      return fitness.avgPnlPerDay;
    case 'winRate':
      return fitness.winRate;
    case 'pnlToRisk':
      return fitness.pnlToRisk;
    case 'composite':
    default: {
      const weights = target.weights ?? { pnl: 0.3, winRate: 0.2, maxDrawdown: 0.2, pnlToRisk: 0.3 };
      return (
        weights.pnl * normalizedPnl +
        weights.winRate * normalizedWinRate +
        weights.maxDrawdown * normalizedDrawdown +
        weights.pnlToRisk * normalizedRatio
      );
    }
  }
};

// ═══════════════════════════════════════════════════════════════════
// СЕЛЕКЦИЯ
// ═══════════════════════════════════════════════════════════════════

/**
 * Турнирная селекция по score (проверенный подход).
 * NSGA-II ранги используются только для информационной маркировки.
 */
export const tournamentSelection = (
  population: EvaluatedGenome[],
  tournamentSize: number,
): EvaluatedGenome => {
  const tournament: EvaluatedGenome[] = [];

  for (let i = 0; i < tournamentSize; i++) {
    const idx = randomInt(0, population.length - 1);
    tournament.push(population[idx]);
  }

  tournament.sort((a, b) => b.fitness.score - a.fitness.score);
  return tournament[0];
};

/**
 * Создать новое поколение.
 * Селекция по score, Pareto-ранги — только для отображения.
 */
export const createNextGeneration = (
  population: EvaluatedGenome[],
  config: GeneticConfig,
  scope: OptimizationScope,
  _target: OptimizationTarget,
): BotGenome[] => {
  const nextGen: BotGenome[] = [];

  // Сортируем по score (проверенная селекция)
  const sorted = [...population].sort((a, b) => b.fitness.score - a.fitness.score);

  // Элитизм — лучшие по score переходят без изменений
  for (let i = 0; i < config.elitismCount && i < sorted.length; i++) {
    const elite = { ...sorted[i].genome, id: generateId() };
    nextGen.push(elite);
  }

  // Заполняем остальную популяцию
  while (nextGen.length < config.populationSize) {
    const parent1 = tournamentSelection(population, config.tournamentSize);
    const parent2 = tournamentSelection(population, config.tournamentSize);

    let child: BotGenome;

    // Скрещивание
    if (Math.random() < config.crossoverRate) {
      child = crossover(parent1.genome, parent2.genome, scope);
    } else {
      child = { ...parent1.genome, id: generateId() };
    }

    // Мутация
    if (Math.random() < config.mutationRate) {
      child = mutateGenome(child, scope, config.mutationRate);
    }

    nextGen.push(child);
  }

  return nextGen;
};

// ═══════════════════════════════════════════════════════════════════
// ИНИЦИАЛИЗАЦИЯ ПОПУЛЯЦИИ
// ═══════════════════════════════════════════════════════════════════

/**
 * Создать начальную популяцию
 */
export const createInitialPopulation = (
  size: number,
  baseGenome: BotGenome | null,
  scope: OptimizationScope,
): BotGenome[] => {
  const population: BotGenome[] = [];

  if (baseGenome) {
    // Добавляем базовый геном (оригинал без изменений)
    population.push({ ...baseGenome, id: generateId(), generation: 0 });

    console.log('[GeneticEngine] Base genome DCA:', baseGenome.dcaOrders.map((o) => o.indent.toFixed(2)));

    // Создаём вариации базового генома с ВЫСОКОЙ мутацией для разнообразия
    for (let i = 1; i < size; i++) {
      const variant = mutateGenome(
        { ...baseGenome, id: generateId(), generation: 0 },
        scope,
        0.8, // Очень высокая мутация → guaranteedMutation=true
      );
      
      // Логируем мутации
      console.log(`[GeneticEngine] Mutated genome ${i} DCA:`, variant.dcaOrders.map((o) => o.indent.toFixed(2)));
      
      population.push(variant);
    }
  } else {
    // Полностью случайная популяция
    for (let i = 0; i < size; i++) {
      population.push(createRandomGenome(0));
    }
  }

  return population;
};
