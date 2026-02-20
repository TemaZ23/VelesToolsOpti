/**
 * Отслеживание важности индикаторов
 *
 * Адаптировано из Python IndicatorImportanceTracker.
 * Ведёт статистику: какие индикаторы чаще встречаются в лучших стратегиях,
 * с каким средним скором, какие пары работают лучше всего.
 */

import { INDICATOR_CATALOG } from './indicatorCatalog';
import type { EvaluatedGenome } from '../types/optimizer';

const ALL_IDS = INDICATOR_CATALOG.map((ind) => ind.id);

interface PairKey {
  a: string;
  b: string;
}

const pairKey = (a: string, b: string): string => {
  return a < b ? `${a}|${b}` : `${b}|${a}`;
};

export class IndicatorImportanceTracker {
  /** Сколько раз индикатор был в лучших стратегиях */
  private usageCount = new Map<string, number>();
  /** Сумма score */
  private scoreSum = new Map<string, number>();
  /** Скоры пар */
  private pairScores = new Map<string, number[]>();
  /** Общее число обработанных стратегий */
  private totalStrategies = 0;

  constructor() {
    for (const id of ALL_IDS) {
      this.usageCount.set(id, 0);
      this.scoreSum.set(id, 0);
    }
  }

  /**
   * Обновить трекер на основе оценённых геномов (лучших).
   */
  update(genomes: EvaluatedGenome[]): void {
    for (const { genome, fitness } of genomes) {
      if (fitness.score <= 0) continue;

      const indicators = genome.entryConditions.map((c) => c.indicator);
      this.totalStrategies++;

      for (const id of indicators) {
        this.usageCount.set(id, (this.usageCount.get(id) ?? 0) + 1);
        this.scoreSum.set(id, (this.scoreSum.get(id) ?? 0) + fitness.score);
      }

      // Обновить парные скоры
      for (let i = 0; i < indicators.length; i++) {
        for (let j = i + 1; j < indicators.length; j++) {
          const key = pairKey(indicators[i], indicators[j]);
          const existing = this.pairScores.get(key) ?? [];
          existing.push(fitness.score);
          this.pairScores.set(key, existing);
        }
      }
    }
  }

  /**
   * Средний score для индикатора.
   */
  avgScore(id: string): number {
    const usage = this.usageCount.get(id) ?? 0;
    if (usage === 0) return 0;
    return (this.scoreSum.get(id) ?? 0) / usage;
  }

  /**
   * Топ-N самых полезных индикаторов.
   */
  getTopIndicators(n: number = 15, minUsage: number = 2): string[] {
    const candidates = ALL_IDS
      .filter((id) => (this.usageCount.get(id) ?? 0) >= minUsage)
      .map((id) => ({ id, score: this.avgScore(id) }))
      .sort((a, b) => b.score - a.score);
    return candidates.slice(0, n).map((c) => c.id);
  }

  /**
   * Наименее исследованные индикаторы.
   */
  getUnderexplored(n: number = 10): string[] {
    return [...ALL_IDS]
      .sort((a, b) => (this.usageCount.get(a) ?? 0) - (this.usageCount.get(b) ?? 0))
      .slice(0, n);
  }

  /**
   * Лучшие пары индикаторов.
   */
  getBestPairs(n: number = 10): Array<{ a: string; b: string; avgScore: number }> {
    const pairs: Array<{ a: string; b: string; avgScore: number }> = [];

    for (const [key, scores] of this.pairScores.entries()) {
      if (scores.length < 2) continue;
      const avg = scores.reduce((s, v) => s + v, 0) / scores.length;
      const [a, b] = key.split('|');
      pairs.push({ a, b, avgScore: avg });
    }

    pairs.sort((x, y) => y.avgScore - x.avgScore);
    return pairs.slice(0, n);
  }

  /**
   * Веса важности (вероятность выбора).
   * При малом кол-ве данных — равномерные веса.
   */
  getImportanceWeights(): Map<string, number> {
    const weights = new Map<string, number>();

    if (this.totalStrategies < 5) {
      const uniform = 1 / ALL_IDS.length;
      for (const id of ALL_IDS) {
        weights.set(id, uniform);
      }
      return weights;
    }

    const scores: number[] = [];
    for (const id of ALL_IDS) {
      const usage = this.usageCount.get(id) ?? 0;
      if (usage > 0) {
        scores.push(Math.max(this.avgScore(id) * Math.log(usage + 1), 0.01));
      } else {
        scores.push(0.5); // неисследованные — средний вес
      }
    }

    const total = scores.reduce((s, v) => s + v, 0);
    ALL_IDS.forEach((id, i) => {
      weights.set(id, scores[i] / total);
    });

    return weights;
  }

  /**
   * Текстовый отчёт.
   */
  summary(): string {
    const lines = [`📊 Indicator Importance (${this.totalStrategies} strategies):`];
    const ranked = [...ALL_IDS].sort((a, b) => this.avgScore(b) - this.avgScore(a));

    for (const id of ranked.slice(0, 15)) {
      const cnt = this.usageCount.get(id) ?? 0;
      const avg = this.avgScore(id);
      lines.push(`  ${id}: used=${cnt}, avg_score=${avg.toFixed(3)}`);
    }

    const bestPairs = this.getBestPairs(5);
    if (bestPairs.length > 0) {
      lines.push('🔗 Best pairs:');
      for (const { a, b, avgScore } of bestPairs) {
        lines.push(`  ${a} + ${b}: ${avgScore.toFixed(3)}`);
      }
    }

    return lines.join('\n');
  }
}
