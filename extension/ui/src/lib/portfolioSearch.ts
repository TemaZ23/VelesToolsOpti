/**
 * Поиск кандидатов портфельного оптимизатора (чистые функции, без сети и хранилищ).
 *
 * Схема поиска — локальная эволюция вокруг лучших найденных настроек: новый кандидат получается
 * из одного из лучших родителей сдвигом нескольких включённых параметров на 1–2 шага. Параметры,
 * изменения которых чаще давали улучшение, выбираются чаще (оценка Лапласа (wins + 1) / (tries + 2)).
 * Отбор двухэтапный: дешёвая отборочная корзина → полная корзина только для лучших по отбору.
 *
 * Вся случайность идёт через переданный генератор `Rng`, поэтому при одинаковом seed результаты
 * полностью воспроизводимы (важно для продолжения запуска после перезагрузки).
 */

import type {
  FoldMetrics,
  ObjectiveConfig,
  ParamSpec,
  ParamValues,
  PortfolioMetrics,
} from '../types/portfolioOptimizer';
import { clampAndRound, diffFromBase, paramsHash, prepareValues } from './botParams';

// ═══════════════════════════════════════════════════════════════════
// ТИПЫ
// ═══════════════════════════════════════════════════════════════════

/** Генератор псевдослучайных чисел в диапазоне [0, 1). */
export type Rng = () => number;

/** Статистика параметра: сколько раз его изменение проверялось и сколько раз дало улучшение. */
export interface ParamStat {
  tries: number;
  wins: number;
}

/** id параметра → статистика успеха его изменений. */
export type ParamStats = Record<string, ParamStat>;

/** Родитель для мутаций (список родителей отсортирован от лучшего к худшему). */
export interface ProposalParent {
  values: ParamValues;
  score: number;
}

export interface ProposeCandidatesInput {
  params: ParamSpec[];
  /** Отсортированы от лучшего к худшему, длина ≥ 1. */
  parents: ProposalParent[];
  /** Сколько кандидатов нужно предложить. */
  count: number;
  /** Сколько параметров меняется у кандидата (от и до, ограничивается числом включённых параметров). */
  mutationsMin: number;
  mutationsMax: number;
  /** Хеши уже проверенных или поставленных в очередь кандидатов (не изменяется). */
  seen: ReadonlySet<string>;
  paramStats: ParamStats;
  rng: Rng;
  /** Максимум попыток сгенерировать кандидата (по умолчанию count × 50). */
  maxAttempts?: number;
}

export interface CandidateProposal {
  /** Подготовленные значения всех параметров (после prepareValues). */
  values: ParamValues;
  hash: string;
  /** Индекс родителя во входном списке parents. */
  parentIndex: number;
  /** id параметров, отличающихся от базового бота. */
  changed: string[];
  /** id параметров, которые мутация изменила относительно родителя (без побочных правок сетки/объёмов). */
  mutated: string[];
}

// ═══════════════════════════════════════════════════════════════════
// КОНСТАНТЫ
// ═══════════════════════════════════════════════════════════════════

/** Вероятность взять родителем лучшего кандидата. */
const BEST_PARENT_PROBABILITY = 0.6;
/** Иначе родитель выбирается равновероятно среди стольких лучших. */
const PARENT_POOL_SIZE = 3;
/** Вероятность сдвига на один шаг (иначе — на два). */
const SINGLE_STEP_PROBABILITY = 0.75;
/** Попыток на одного запрошенного кандидата по умолчанию. */
const DEFAULT_ATTEMPTS_PER_CANDIDATE = 50;
/** Вес полного периода в устойчивой цели (остальное — худший под-период). */
const ROBUST_FULL_PERIOD_WEIGHT = 0.5;
/** Относительный допуск сравнения с порогом отбора (защита от двоичного шума). */
const THRESHOLD_EPSILON = 1e-12;

const MULBERRY32_INCREMENT = 0x6d2b79f5;
const UINT32_RANGE = 4294967296;
/** Наибольшее число double меньше 1. */
const MAX_UNIT = 1 - Number.EPSILON / 2;

// ═══════════════════════════════════════════════════════════════════
// БАЗОВЫЕ ПОМОЩНИКИ
// ═══════════════════════════════════════════════════════════════════

const isFiniteNumber = (value: number | null | undefined): value is number =>
  typeof value === 'number' && Number.isFinite(value);

const clamp = (value: number, min: number, max: number): number => Math.min(max, Math.max(min, value));

/** Неотрицательное целое; нечисловые и бесконечные значения → 0. */
const toNonNegativeInt = (value: number): number => (Number.isFinite(value) ? Math.max(0, Math.floor(value)) : 0);

/** Неотрицательное конечное число; всё остальное → 0. */
const toNonNegative = (value: number | null | undefined): number => (isFiniteNumber(value) ? Math.max(0, value) : 0);

/** Следующее число генератора, гарантированно в [0, 1) (на случай некорректного Rng). */
const nextUnit = (rng: Rng): number => {
  const value = rng();
  return Number.isFinite(value) ? clamp(value, 0, MAX_UNIT) : 0;
};

/** Равновероятный индекс в [0, length). */
const randomIndex = (length: number, rng: Rng): number => Math.min(length - 1, Math.floor(nextUnit(rng) * length));

/** Равновероятное целое в [min, max] (min ≤ max). */
const randomInt = (min: number, max: number, rng: Rng): number => min + randomIndex(max - min + 1, rng);

const sanitizeWeight = (weight: number | undefined): number => (isFiniteNumber(weight) && weight > 0 ? weight : 0);

const readStat = (stats: ParamStats, id: string): ParamStat | undefined =>
  Object.hasOwn(stats, id) ? stats[id] : undefined;

// ═══════════════════════════════════════════════════════════════════
// ГЕНЕРАТОР И ВЗВЕШЕННЫЙ ВЫБОР
// ═══════════════════════════════════════════════════════════════════

/**
 * Детерминированный генератор mulberry32: одинаковый seed → одинаковая последовательность.
 * seed приводится к uint32 (дробная часть отбрасывается, NaN/±∞ → 0).
 */
export const createRng = (seed: number): Rng => {
  let state = Number.isFinite(seed) ? seed >>> 0 : 0;
  return () => {
    state = (state + MULBERRY32_INCREMENT) >>> 0;
    let mixed = Math.imul(state ^ (state >>> 15), state | 1);
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
    return ((mixed ^ (mixed >>> 14)) >>> 0) / UINT32_RANGE;
  };
};

/**
 * Выбор элемента с вероятностью, пропорциональной весу. Некорректные веса (NaN, ±∞, ≤ 0,
 * отсутствующие) считаются нулевыми; если все веса нулевые — выбор равновероятный.
 * Бросает ошибку для пустого списка.
 */
export const pickWeighted = <T>(items: T[], weights: number[], rng: Rng): T => {
  if (items.length === 0) {
    throw new Error('Нельзя выбрать элемент из пустого списка.');
  }
  const safeWeights = items.map((_, index) => sanitizeWeight(weights[index]));
  const total = safeWeights.reduce((accumulator, weight) => accumulator + weight, 0);
  if (!(total > 0 && Number.isFinite(total))) {
    return items[randomIndex(items.length, rng)];
  }

  const target = nextUnit(rng) * total;
  let cumulative = 0;
  let lastPositiveIndex = 0;
  for (let index = 0; index < items.length; index += 1) {
    const weight = safeWeights[index];
    if (weight <= 0) {
      continue;
    }
    cumulative += weight;
    lastPositiveIndex = index;
    if (target < cumulative) {
      return items[index];
    }
  }
  // Сюда попадаем только из-за погрешности суммирования — берём последний элемент с весом.
  return items[lastPositiveIndex];
};

// ═══════════════════════════════════════════════════════════════════
// ГЕНЕРАЦИЯ КАНДИДАТОВ
// ═══════════════════════════════════════════════════════════════════

/** Параметр участвует в мутациях: включён, есть положительный шаг и ненулевой диапазон. */
const isMutable = (param: ParamSpec): boolean =>
  param.enabled &&
  isFiniteNumber(param.step) &&
  param.step > 0 &&
  isFiniteNumber(param.min) &&
  isFiniteNumber(param.max) &&
  param.min !== param.max;

/** Вес выбора параметра по Лапласу: (wins + 1) / (tries + 2), всегда в (0, 1]. */
const paramWeight = (stat: ParamStat | undefined): number => {
  const wins = toNonNegative(stat?.wins);
  const tries = Math.max(wins, toNonNegative(stat?.tries));
  return (wins + 1) / (tries + 2);
};

/** Значение параметра, ограниченное его диапазоном и округлённое так же, как в botParams. */
const clampParamValue = (param: ParamSpec, value: number): number =>
  clampAndRound([param], { [param.id]: value })[param.id];

/** Количество мутаций [от, до]: не меньше 1 и не больше числа доступных параметров. */
const resolveMutationRange = (from: number, to: number, available: number): [number, number] => {
  const toCount = (value: number): number => (Number.isNaN(value) ? 1 : Math.floor(value));
  const low = clamp(toCount(Math.min(from, to)), 1, available);
  const high = clamp(toCount(Math.max(from, to)), low, available);
  return [low, high];
};

/** Родитель: лучший с вероятностью 0.6, иначе равновероятно среди трёх лучших. */
const chooseParentIndex = (parentCount: number, rng: Rng): number => {
  if (nextUnit(rng) < BEST_PARENT_PROBABILITY) {
    return 0;
  }
  return randomIndex(Math.min(PARENT_POOL_SIZE, parentCount), rng);
};

/** k различных параметров, взвешенный выбор без возвращения. */
const chooseParams = (pool: ParamSpec[], count: number, stats: ParamStats, rng: Rng): ParamSpec[] => {
  const remaining = [...pool];
  const chosen: ParamSpec[] = [];
  while (chosen.length < count && remaining.length > 0) {
    const indices = remaining.map((_, index) => index);
    const weights = remaining.map((param) => paramWeight(readStat(stats, param.id)));
    const pickedIndex = pickWeighted(indices, weights, rng);
    chosen.push(remaining[pickedIndex]);
    remaining.splice(pickedIndex, 1);
  }
  return chosen;
};

/**
 * Сдвиг значения на ±шаг × m (m = 1 с вероятностью 0.75, иначе 2), знак случайный.
 * Если после ограничения диапазоном значение не изменилось — пробуем другой знак.
 * null — параметр сдвинуть нельзя.
 */
const mutateParam = (param: ParamSpec, current: number, rng: Rng): number | null => {
  const multiplier = nextUnit(rng) < SINGLE_STEP_PROBABILITY ? 1 : 2;
  const direction = nextUnit(rng) < 0.5 ? -1 : 1;
  const delta = param.step * multiplier;
  for (const sign of [direction, -direction]) {
    const next = clampParamValue(param, current + sign * delta);
    if (next !== current) {
      return next;
    }
  }
  return null;
};

interface PreparedParent {
  /** Значения родителя после clampAndRound — отправная точка мутаций. */
  start: ParamValues;
  /** Значения родителя после prepareValues — с ними сравнивается результат. */
  prepared: ParamValues;
  hash: string;
}

const prepareParent = (params: ParamSpec[], parent: ProposalParent): PreparedParent => {
  const prepared = prepareValues(params, parent.values);
  return {
    start: clampAndRound(params, parent.values),
    prepared,
    hash: paramsHash(params, prepared),
  };
};

/**
 * Предлагает до `count` новых кандидатов мутацией лучших родителей.
 * Меняются только включённые параметры (побочно могут поправиться отступы сетки и объёмы —
 * prepareValues). Кандидаты, совпадающие с родителем, с `seen` или друг с другом, отбрасываются.
 * Останавливается после `count` кандидатов или `maxAttempts` попыток; при пустых родителях,
 * count ≤ 0 или отсутствии изменяемых параметров возвращает пустой список.
 */
export const proposeCandidates = (input: ProposeCandidatesInput): CandidateProposal[] => {
  const { params, parents, seen, paramStats, rng } = input;
  const count = toNonNegativeInt(input.count);
  const pool = params.filter(isMutable);
  if (count === 0 || parents.length === 0 || pool.length === 0) {
    return [];
  }

  const maxAttempts =
    input.maxAttempts !== undefined && Number.isFinite(input.maxAttempts)
      ? toNonNegativeInt(input.maxAttempts)
      : count * DEFAULT_ATTEMPTS_PER_CANDIDATE;
  const [minMutations, maxMutations] = resolveMutationRange(input.mutationsMin, input.mutationsMax, pool.length);

  const preparedParents = new Map<number, PreparedParent>();
  const getParent = (index: number): PreparedParent => {
    const cached = preparedParents.get(index);
    if (cached) {
      return cached;
    }
    const prepared = prepareParent(params, parents[index]);
    preparedParents.set(index, prepared);
    return prepared;
  };

  const proposedHashes = new Set<string>();
  const proposals: CandidateProposal[] = [];

  for (let attempt = 0; attempt < maxAttempts && proposals.length < count; attempt += 1) {
    const parentIndex = chooseParentIndex(parents.length, rng);
    const parent = getParent(parentIndex);
    const mutationCount = randomInt(minMutations, maxMutations, rng);

    const draft: ParamValues = { ...parent.start };
    const moved: string[] = [];
    for (const param of chooseParams(pool, mutationCount, paramStats, rng)) {
      const next = mutateParam(param, draft[param.id], rng);
      if (next !== null) {
        draft[param.id] = next;
        moved.push(param.id);
      }
    }
    if (moved.length === 0) {
      continue;
    }

    const values = prepareValues(params, draft);
    const hash = paramsHash(params, values);
    if (hash === parent.hash || seen.has(hash) || proposedHashes.has(hash)) {
      continue;
    }

    proposedHashes.add(hash);
    proposals.push({
      values,
      hash,
      parentIndex,
      changed: diffFromBase(params, values),
      mutated: moved.filter((id) => values[id] !== parent.prepared[id]),
    });
  }

  return proposals;
};

// ═══════════════════════════════════════════════════════════════════
// ЦЕЛЕВАЯ ФУНКЦИЯ
// ═══════════════════════════════════════════════════════════════════

/** Худший P&L к риску среди под-периодов со сделками (null — таких под-периодов нет). */
const worstFoldPnlToRisk = (folds: FoldMetrics[]): number | null => {
  const values = folds.filter((fold) => fold.deals > 0 && isFiniteNumber(fold.pnlToRisk)).map((fold) => fold.pnlToRisk);
  return values.length > 0 ? Math.min(...values) : null;
};

/**
 * P&L со штрафом за превышение МПУ: profit × (cap / МПУ)². Штраф применяется только к прибыли —
 * убыток не уменьшается, иначе превышение ограничения «улучшало» бы убыточный результат.
 * Некорректное ограничение (null, NaN, ±∞, < 0) означает «без ограничения».
 */
const cappedProfit = (metrics: PortfolioMetrics, cap: number | null): number => {
  const profit = metrics.totalProfit;
  if (!isFiniteNumber(cap) || cap < 0) {
    return profit;
  }
  const mae = metrics.maxConcurrentMae;
  if (!(mae > cap) || profit <= 0) {
    return profit;
  }
  return profit * (cap / mae) ** 2;
};

const baseObjective = (metrics: PortfolioMetrics, folds: FoldMetrics[], cfg: ObjectiveConfig): number => {
  switch (cfg.kind) {
    case 'pnlToRisk':
      return metrics.pnlToRisk;
    case 'robust': {
      const worst = worstFoldPnlToRisk(folds);
      if (worst === null) {
        return metrics.pnlToRisk;
      }
      return ROBUST_FULL_PERIOD_WEIGHT * metrics.pnlToRisk + (1 - ROBUST_FULL_PERIOD_WEIGHT) * worst;
    }
    case 'profitCapped':
      return cappedProfit(metrics, cfg.maxMaeCap);
    default: {
      const unknownKind: never = cfg.kind;
      throw new Error(`Неизвестная цель оптимизации: ${String(unknownKind)}.`);
    }
  }
};

/** Штраф за малое число сделок: score × deals / minDeals (только для положительного score). */
const applyDealsPenalty = (score: number, deals: number, minDeals: number): number => {
  if (!(isFiniteNumber(minDeals) && minDeals > 0) || deals >= minDeals || score <= 0) {
    return score;
  }
  return score * (deals / minDeals);
};

/**
 * Score кандидата по портфельным метрикам: null — оценить нельзя (нет метрик или нечисловой результат),
 * 0 — нет сделок. 'pnlToRisk' — как в группе бэктестов; 'robust' — среднее P&L к риску за весь период
 * и худшего под-периода со сделками; 'profitCapped' — P&L со штрафом (cap / МПУ)² при МПУ > cap.
 * Затем штраф × deals / minDeals, если сделок меньше minDeals.
 */
export const computeObjective = (
  metrics: PortfolioMetrics | null,
  folds: FoldMetrics[],
  cfg: ObjectiveConfig,
): number | null => {
  if (!metrics) {
    return null;
  }
  if (!(metrics.totalDeals > 0)) {
    return 0;
  }
  const raw = baseObjective(metrics, folds, cfg);
  if (!Number.isFinite(raw)) {
    return null;
  }
  const score = applyDealsPenalty(raw, metrics.totalDeals, cfg.minDeals);
  return score === 0 ? 0 : score;
};

// ═══════════════════════════════════════════════════════════════════
// ОТБОР И СТАТИСТИКА
// ═══════════════════════════════════════════════════════════════════

/**
 * Нижняя граница score отбора: best − |best| × tolerance (с крошечным допуском на двоичный шум).
 * null — лучший score неизвестен, порога нет.
 */
const promotionThreshold = (bestScreenScore: number | null, tolerance: number): number | null => {
  if (!isFiniteNumber(bestScreenScore)) {
    return null;
  }
  const threshold = bestScreenScore - Math.abs(bestScreenScore) * toNonNegative(tolerance);
  return threshold - Math.max(1, Math.abs(threshold)) * THRESHOLD_EPSILON;
};

/**
 * Кандидаты для проверки на полной корзине: без score отбрасываются; если лучший score отбора
 * неизвестен — просто топ promoteCount; иначе только score ≥ best − |best| × tolerance.
 * Сортировка по убыванию score (при равенстве сохраняется исходный порядок). Вход не изменяется.
 */
export const selectForPromotion = <T extends { screenScore: number | null }>(
  candidates: T[],
  bestScreenScore: number | null,
  promoteCount: number,
  tolerance: number,
): T[] => {
  const limit = toNonNegativeInt(promoteCount);
  if (limit === 0) {
    return [];
  }
  const scored = candidates.filter((candidate): candidate is T & { screenScore: number } =>
    isFiniteNumber(candidate.screenScore),
  );
  const threshold = promotionThreshold(bestScreenScore, tolerance);
  const eligible = threshold === null ? scored : scored.filter((candidate) => candidate.screenScore >= threshold);
  return eligible.sort((left, right) => right.screenScore - left.screenScore).slice(0, limit);
};

/**
 * Новая статистика параметров: каждому id из `changed` (без повторов) +1 попытка и +1 победа,
 * если кандидат улучшил результат. Входной объект не изменяется.
 */
export const updateParamStats = (stats: ParamStats, changed: string[], improved: boolean): ParamStats => {
  const next: ParamStats = Object.fromEntries(
    Object.entries(stats).map(([id, stat]): [string, ParamStat] => [id, { tries: stat.tries, wins: stat.wins }]),
  );
  for (const id of new Set(changed)) {
    const current = readStat(next, id);
    next[id] = {
      tries: toNonNegative(current?.tries) + 1,
      wins: toNonNegative(current?.wins) + (improved ? 1 : 0),
    };
  }
  return next;
};

/**
 * Улучшение: score известен и (лучшего ещё нет или score > best + |best| × minImprovement).
 * Нечисловой лучший score считается отсутствующим, отрицательный minImprovement — нулём.
 */
export const isImprovement = (score: number | null, bestScore: number | null, minImprovement: number): boolean => {
  if (!isFiniteNumber(score)) {
    return false;
  }
  if (!isFiniteNumber(bestScore)) {
    return true;
  }
  return score > bestScore + Math.abs(bestScore) * toNonNegative(minImprovement);
};
