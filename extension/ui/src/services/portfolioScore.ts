/**
 * Оценка корзины бэктестов для портфельного оптимизатора.
 *
 * Метрики считаются ровно так же, как на странице «Группы бэктестов»: те же buildBacktestInfo и
 * aggregateBacktestsMetrics, те же циклы (включая отменённые подтяжкой), та же «блокировка по ботам»
 * (maxConcurrentPositions). Поэтому числа оптимизатора совпадают с тем, что пользователь увидит,
 * создав группу из бэктестов кандидата с той же блокировкой (при том же порядке бэктестов — он важен
 * для сделок с одинаковым временем входа; группа хранит id по возрастанию, оркестратор передаёт их так же).
 *
 * Дополнительно период делится на под-периоды (фолды) для устойчивой целевой функции:
 * кандидат, который хорош на всём периоде, но провалился в одном из под-периодов, ценится ниже.
 */

import { toTimestamp } from '../lib/dateTime';
import type { AggregatedBacktestsMetrics, AggregationConfig } from '../types/backtestAggregations';
import type { BacktestInfo, BacktestInfoDeal } from '../types/backtestInfos';
import type { BacktestCycle } from '../types/backtests';
import type { BacktestRunSettings, FoldMetrics, PortfolioMetrics } from '../types/portfolioOptimizer';
import { aggregateBacktestsMetrics } from './backtestAggregations';
import { buildBacktestInfo } from './backtestInfos';
import { backtestsService } from './backtests';

// ═══════════════════════════════════════════════════════════════════
// ТИПЫ
// ═══════════════════════════════════════════════════════════════════

/** Временное окно под-периода, мс epoch. Сделка попадает в окно по времени входа: [from, to). */
export interface FoldWindow {
  from: number;
  to: number;
}

/** Период запуска, по которому делятся фолды. */
export type BasketPeriod = Pick<BacktestRunSettings, 'periodFrom' | 'periodTo'>;

export interface BasketEvaluation {
  metrics: PortfolioMetrics;
  folds: FoldMetrics[];
  /** Загруженные бэктесты в порядке переданных id. */
  infos: BacktestInfo[];
}

// ═══════════════════════════════════════════════════════════════════
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
// ═══════════════════════════════════════════════════════════════════

/** NaN/Infinity в метриках ломают сравнение кандидатов, поэтому заменяем их нулём. */
const finiteOrZero = (value: number): number => (Number.isFinite(value) ? value : 0);

/**
 * Блокировка по ботам: целое число ≥ 1. Некорректное значение трактуем как 1
 * (иначе агрегатор с лимитом NaN не впустил бы ни одной сделки).
 */
const toAggregationConfig = (block: number): AggregationConfig => ({
  maxConcurrentPositions: Number.isFinite(block) ? Math.max(1, Math.floor(block)) : 1,
});

const countDeals = (agg: AggregatedBacktestsMetrics): number =>
  finiteOrZero(agg.totalProfitableDeals) + finiteOrZero(agg.totalLosingDeals) + finiteOrZero(agg.openDeals);

const describeError = (error: unknown): string => (error instanceof Error ? error.message : String(error));

/**
 * Цикл пригоден для buildBacktestInfo: есть хотя бы один ордер (время входа) и обе даты
 * разбираются. Цикл с нераспознаваемой датой дал бы сделку со start = NaN, которая «застревает»
 * в очереди агрегатора и отрезает все последующие сделки. Статус не проверяется: отменённый
 * подтяжкой цикл с ордерами страница группы тоже считает сделкой (net 0, занимает место в блокировке).
 */
const isUsableCycle = (cycle: BacktestCycle): boolean => {
  if (!Array.isArray(cycle.orders) || cycle.orders.length === 0) {
    return false;
  }
  return toTimestamp(cycle.orders.at(0)?.createdAt) !== null && toTimestamp(cycle.date) !== null;
};

const isDealInWindow = (deal: BacktestInfoDeal, window: FoldWindow, isLastWindow: boolean): boolean => {
  if (deal.start < window.from) {
    return false;
  }
  // Последний фолд включает правую границу, чтобы сделка ровно на конце периода не потерялась.
  return deal.start < window.to || (isLastWindow && deal.start === window.to);
};

/**
 * Поверхностные копии бэктестов только со сделками окна. from/to заменяются границами окна,
 * чтобы производные агрегатора (P&L в день) относились к под-периоду. Исходные объекты не меняются.
 */
const sliceInfosToWindow = (infos: BacktestInfo[], window: FoldWindow, isLastWindow: boolean): BacktestInfo[] => {
  const fromIso = new Date(window.from).toISOString();
  const toIso = new Date(window.to).toISOString();
  return infos.map((info) => ({
    ...info,
    from: fromIso,
    to: toIso,
    deals: info.deals.filter((deal) => isDealInWindow(deal, window, isLastWindow)),
  }));
};

const toFoldMetrics = (window: FoldWindow, agg: AggregatedBacktestsMetrics): FoldMetrics => {
  const profit = finiteOrZero(agg.totalProfitQuote);
  const risk = Math.max(finiteOrZero(agg.maxConcurrentMae), finiteOrZero(agg.maxAggregatedDrawdown));
  return {
    from: window.from,
    to: window.to,
    profit,
    risk,
    pnlToRisk: profit / Math.max(1, risk),
    deals: countDeals(agg),
  };
};

// ═══════════════════════════════════════════════════════════════════
// ПУБЛИЧНЫЕ ФУНКЦИИ
// ═══════════════════════════════════════════════════════════════════

/**
 * Убирает только циклы, которые нельзя превратить в сделку: без ордеров (buildBacktestInfo упал бы
 * на `orders.at(0)!`) и с нераспознаваемыми датами. Отменённые подтяжкой циклы с ордерами остаются —
 * как на странице группы бэктестов они дают сделку с net 0 и занимают место в блокировке по ботам,
 * поэтому число сделок, P&L и МПУ совпадают с группой. Порядок циклов сохраняется.
 */
export const sanitizeCycles = (cycles: BacktestCycle[]): BacktestCycle[] => cycles.filter(isUsableCycle);

/**
 * Загружает бэктест (детали + циклы, с обычным кешем IndexedDB) и строит BacktestInfo,
 * как страница группы бэктестов. Ошибка дополняется id бэктеста; исходный текст
 * (например, «HTTP 401 …») сохраняется в сообщении, а сама ошибка — в `cause`.
 */
export const loadBacktestInfo = async (backtestId: number): Promise<BacktestInfo> => {
  try {
    const [detail, cycles] = await Promise.all([
      backtestsService.getBacktestDetail(backtestId),
      backtestsService.getBacktestCycles(backtestId),
    ]);
    return buildBacktestInfo(detail, sanitizeCycles(cycles));
  } catch (error) {
    throw new Error(`Не удалось загрузить результаты бэктеста ${backtestId}: ${describeError(error)}`, {
      cause: error,
    });
  }
};

/** Переводит агрегированные метрики группы в портфельные метрики оптимизатора. */
export const toPortfolioMetrics = (agg: AggregatedBacktestsMetrics, backtests: number): PortfolioMetrics => ({
  totalProfit: finiteOrZero(agg.totalProfitQuote),
  maxConcurrentMae: finiteOrZero(agg.maxConcurrentMae),
  maxDrawdown: finiteOrZero(agg.maxAggregatedDrawdown),
  pnlToRisk: finiteOrZero(agg.pnlToRisk),
  totalDeals: countDeals(agg),
  profitableDeals: finiteOrZero(agg.totalProfitableDeals),
  losingDeals: finiteOrZero(agg.totalLosingDeals),
  openDeals: finiteOrZero(agg.openDeals),
  activeMae: finiteOrZero(agg.aggregatedActiveMae),
  netPerDay: finiteOrZero(agg.averageNetPerDay),
  maxConcurrentPositions: finiteOrZero(agg.maxConcurrentPositions),
  backtests: finiteOrZero(backtests),
});

/** Портфельные метрики корзины с блокировкой по ботам `block` (как в группе бэктестов). */
export const computePortfolioMetrics = (infos: BacktestInfo[], block: number): PortfolioMetrics =>
  toPortfolioMetrics(aggregateBacktestsMetrics(infos, toAggregationConfig(block)), infos.length);

/**
 * Делит период [fromMs, toMs] на `folds` равных смежных окон (границы округлены до мс).
 * Меньше двух фолдов, некорректные числа или пустой период → [].
 */
export const buildFoldWindows = (fromMs: number, toMs: number, folds: number): FoldWindow[] => {
  if (!(Number.isFinite(fromMs) && Number.isFinite(toMs) && Number.isFinite(folds)) || toMs <= fromMs) {
    return [];
  }
  const count = Math.floor(folds);
  if (count < 2) {
    return [];
  }
  const span = toMs - fromMs;
  const boundary = (index: number): number => (index >= count ? toMs : Math.round(fromMs + (span * index) / count));
  return Array.from({ length: count }, (_, index) => ({ from: boundary(index), to: boundary(index + 1) }));
};

/**
 * Метрики под-периодов: период делится на `folds` равных частей, в каждую попадают сделки
 * с временем входа в [начало, конец) части (последняя часть включает конец периода),
 * и для них заново считается агрегация с той же блокировкой по ботам.
 * risk = max(МПУ, просадка), pnlToRisk = profit / max(1, risk). folds < 2 → [].
 */
export const computeFoldMetrics = (
  infos: BacktestInfo[],
  block: number,
  periodFrom: string,
  periodTo: string,
  folds: number,
): FoldMetrics[] => {
  const fromMs = toTimestamp(periodFrom);
  const toMs = toTimestamp(periodTo);
  if (fromMs === null || toMs === null) {
    return [];
  }
  const windows = buildFoldWindows(fromMs, toMs, folds);
  const config = toAggregationConfig(block);
  return windows.map((window, index) => {
    const sliced = sliceInfosToWindow(infos, window, index === windows.length - 1);
    return toFoldMetrics(window, aggregateBacktestsMetrics(sliced, config));
  });
};

/**
 * Полная оценка корзины: загружает все бэктесты (порядок id сохраняется), считает портфельные
 * метрики и метрики под-периодов. Ошибка загрузки любого бэктеста отклоняет промис —
 * частичные корзины не оцениваются.
 */
export const evaluateBasket = async (
  backtestIds: number[],
  block: number,
  run: BasketPeriod,
  folds: number,
): Promise<BasketEvaluation> => {
  const infos = await Promise.all(backtestIds.map((backtestId) => loadBacktestInfo(backtestId)));
  return {
    metrics: computePortfolioMetrics(infos, block),
    folds: computeFoldMetrics(infos, block, run.periodFrom, run.periodTo, folds),
    infos,
  };
};
