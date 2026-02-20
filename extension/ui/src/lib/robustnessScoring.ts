/**
 * Robustness scoring для walk-forward валидации
 *
 * Формулы адаптированы из Python strategy_lab optimizer:
 *   robustness = median(test_scores) - std(test_scores)
 *   overfit_ratio = avg(test) / avg(train)
 */

import type { GenomeFitness } from '../types/optimizer';
import type { WalkForwardAggregation, WalkForwardResult } from '../types/autoOptimizer';

/**
 * Медиана массива чисел
 */
const median = (values: number[]): number => {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2;
};

/**
 * Стандартное отклонение
 */
const std = (values: number[]): number => {
  if (values.length < 2) return 0;
  const mean = values.reduce((s, v) => s + v, 0) / values.length;
  const variance = values.reduce((s, v) => s + (v - mean) ** 2, 0) / (values.length - 1);
  return Math.sqrt(variance);
};

/**
 * Агрегация результатов walk-forward окон.
 *
 * Ключевые метрики:
 * - medianTestScore: устойчивость к выбросам
 * - robustnessScore: median - std → стабильность
 * - overfitRatio: avg(test) / avg(train) → идеально ≈ 1.0, < 0.5 = оверфит
 */
export const aggregateWalkForwardResults = (results: WalkForwardResult[]): WalkForwardAggregation => {
  if (results.length === 0) {
    return {
      medianTestScore: 0,
      avgTestScore: 0,
      stdTestScore: 0,
      robustnessScore: 0,
      avgTestPnl: 0,
      avgTestWinRate: 0,
      worstDrawdown: 0,
      minDeals: 0,
      overfitRatio: 0,
    };
  }

  const testScores = results.map((r) => r.testScore);
  const trainScores = results.map((r) => r.trainScore);

  const medianTest = median(testScores);
  const avgTest = testScores.reduce((s, v) => s + v, 0) / testScores.length;
  const stdTest = std(testScores);
  const avgTrain = trainScores.reduce((s, v) => s + v, 0) / trainScores.length;

  const avgPnl = results.reduce((s, r) => s + r.testFitness.totalPnl, 0) / results.length;
  const avgWinRate = results.reduce((s, r) => s + r.testFitness.winRate, 0) / results.length;
  const worstDrawdown = Math.min(...results.map((r) => r.testFitness.maxDrawdown));
  const minDeals = Math.min(...results.map((r) => r.testFitness.totalDeals));

  return {
    medianTestScore: medianTest,
    avgTestScore: avgTest,
    stdTestScore: stdTest,
    robustnessScore: medianTest - stdTest,
    avgTestPnl: avgPnl,
    avgTestWinRate: avgWinRate,
    worstDrawdown,
    minDeals,
    overfitRatio: avgTrain > 0 ? avgTest / avgTrain : 0,
  };
};

/**
 * Composite score для автопереборщика — более строгий чем базовый.
 *
 * Формула:
 *   score = (pnlPerDay_normalized * 0.25)
 *         + (winRate_normalized * 0.20)
 *         + (pnlToRisk * 0.25)
 *         + (drawdown_penalty * 0.15)
 *         + (deals_bonus * 0.15)
 *
 * Штрафы:
 *   - Менее 10 сделок → score * 0.1
 *   - Отрицательный PnL → 0
 */
export const calculateRobustScore = (fitness: GenomeFitness): number => {
  if (fitness.totalPnl < 0) return 0;
  if (fitness.totalDeals < 5) return 0;

  const pnlPerDay = Math.max(0, fitness.avgPnlPerDay);
  const pnlNorm = Math.min(1, pnlPerDay / 50); // $50/day = 1.0
  const winRateNorm = Math.min(1, Math.max(0, fitness.winRate / 100));
  const pnlToRisk = Math.min(1, Math.max(0, fitness.pnlToRisk / 5));
  const drawdownPenalty = 1 - Math.min(1, Math.abs(fitness.maxDrawdown) / 30); // <30% DD = хорошо
  const dealsBonus = Math.min(1, fitness.totalDeals / 100); // 100+ сделок = 1.0

  let score =
    pnlNorm * 0.25 +
    winRateNorm * 0.20 +
    pnlToRisk * 0.25 +
    drawdownPenalty * 0.15 +
    dealsBonus * 0.15;

  // Штраф за мало сделок
  if (fitness.totalDeals < 10) {
    score *= 0.1;
  } else if (fitness.totalDeals < 20) {
    score *= 0.5;
  }

  return score;
};
