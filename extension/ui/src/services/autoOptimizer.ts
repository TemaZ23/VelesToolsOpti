/**
 * Полный автопереборщик стратегий
 *
 * Оркестрирует:
 * 1. Каскадную генетическую оптимизацию (индикаторы → сетка → TP/SL)
 * 2. Walk-forward валидацию лучшего генома
 * 3. Автоматическое создание бота из лучшего результата
 *
 * Использует существующий GeneticOptimizer как движок для каждой фазы.
 */

import { buildBacktestPayload, fetchBotStrategy, postBacktest, resolveQuoteCurrency } from '../api/backtestRunner';
import type { BotStrategy } from '../api/backtestRunner';
import { fetchBacktestStatistics } from '../api/backtests';
import type { BacktestStatisticsDto } from '../api/backtests.dtos';
import { createBot, type CreateBotPayload } from '../api/bots';
import type { BotConfigCreateDto } from '../api/bots.dtos';
import {
  calculateScore,
  createInitialPopulation,
  createNextGeneration,
  createRandomGenome,
} from '../lib/geneticEngine';
import { type FullBotStrategy, applyGenomeToStrategy, genomeToStrategy, strategyToGenome } from '../lib/genomeConverter';
import { IndicatorImportanceTracker } from '../lib/indicatorImportance';
import { aggregateWalkForwardResults, calculateRobustScore } from '../lib/robustnessScoring';
import { readStorageValue, removeStorageValue, writeStorageValue } from '../lib/safeStorage';
import { buildWalkForwardWindows, formatWindow } from '../lib/walkForward';
import type {
  AutoOptimizerCallbacks,
  AutoOptimizerConfig,
  AutoOptimizerPhase,
  AutoOptimizerProgress,
  AutoOptimizerResult,
  WalkForwardResult,
  WalkForwardWindow,
} from '../types/autoOptimizer';
import type {
  BotGenome,
  EvaluatedGenome,
  GeneticConfig,
  GenomeFitness,
  OptimizationScope,
  OptimizationTarget,
} from '../types/optimizer';

// ═══════════════════════════════════════════════════════════════════
// КОНСТАНТЫ
// ═══════════════════════════════════════════════════════════════════

const BACKTEST_POLL_INTERVAL_MS = 5_000;
const BACKTEST_TIMEOUT_MS = 600_000;
const MAX_RETRIES = 5;
const INITIAL_RETRY_DELAY_MS = 10_000;
const MAX_RETRY_DELAY_MS = 120_000;
const TOP_GENOMES_LIMIT = 10;
const AUTO_OPTIMIZER_STATE_KEY = 'veles_auto_optimizer_state';

// ═══════════════════════════════════════════════════════════════════
// УТИЛИТЫ
// ═══════════════════════════════════════════════════════════════════

const delay = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

const retryWithBackoff = async <T>(
  fn: () => Promise<T>,
  opts: {
    maxRetries: number;
    initialDelay: number;
    maxDelay: number;
    onRetry?: (attempt: number, delayMs: number, error: Error) => void;
    shouldRetry?: (error: Error) => boolean;
  },
): Promise<T> => {
  let lastError: Error;
  let delayMs = opts.initialDelay;

  for (let attempt = 1; attempt <= opts.maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));
      if (opts.shouldRetry && !opts.shouldRetry(lastError)) throw lastError;
      if (attempt === opts.maxRetries) throw lastError;
      opts.onRetry?.(attempt, delayMs, lastError);
      await delay(delayMs);
      delayMs = Math.min(delayMs * 2, opts.maxDelay);
    }
  }
  throw lastError!;
};

const isRecoverableError = (error: Error): boolean => {
  const msg = error.message.toLowerCase();
  return (
    msg.includes('network') || msg.includes('fetch') || msg.includes('timeout') ||
    msg.includes('429') || msg.includes('too many') || msg.includes('502') ||
    msg.includes('503') || msg.includes('504') || msg.includes('econnreset') ||
    msg.includes('econnrefused') || msg.includes('socket') || msg.includes('заблокировано')
  );
};

const normalizeCommission = (value: number): string => {
  return (value / 100).toFixed(6);
};

// ═══════════════════════════════════════════════════════════════════
// Fitness extraction (скопировано из optimizer.ts для автономности)
// ═══════════════════════════════════════════════════════════════════

const extractFitnessFromBacktest = (result: BacktestStatisticsDto, genomeId: string): GenomeFitness => {
  const totalPnl = result.netQuote ?? 0;
  const totalDeals = result.totalDeals ?? 0;

  const winProfits = result.winRateProfits ?? 0;
  const winLosses = result.winRateLosses ?? 0;
  const winRateTotal = winProfits + winLosses;

  let winRate = 0;
  if (winRateTotal > 0) {
    winRate = (winProfits / winRateTotal) * 100;
  } else if (totalDeals > 0 && result.profits > 0) {
    winRate = (result.profits / totalDeals) * 100;
  }

  const maxDrawdown = Math.abs(result.maePercent ?? 0);
  const periodDays = result.duration > 0 ? result.duration / (24 * 60 * 60 * 1000) : 1;
  const avgPnlPerDay = result.netQuotePerDay ?? (periodDays > 0 ? totalPnl / periodDays : 0);
  const avgDealDuration = (result.avgDuration ?? 0) / (60 * 60 * 1000);
  const pnlToRisk = maxDrawdown > 0 ? totalPnl / maxDrawdown : totalPnl > 0 ? totalPnl : 0;

  return {
    genomeId,
    backtestIds: [result.id],
    totalPnl,
    avgPnlPerDay,
    winRate,
    maxDrawdown: -maxDrawdown,
    pnlToRisk,
    totalDeals,
    avgDealDuration,
    score: 0,
    nsgaRank: 0,
    crowdingDistance: 0,
  };
};

const aggregateFitness = (fitnesses: GenomeFitness[], genomeId: string): GenomeFitness => {
  if (fitnesses.length === 0) {
    return { genomeId, backtestIds: [], totalPnl: 0, avgPnlPerDay: 0, winRate: 0, maxDrawdown: 0, pnlToRisk: 0, totalDeals: 0, avgDealDuration: 0, score: 0, nsgaRank: 0, crowdingDistance: 0 };
  }
  if (fitnesses.length === 1) return fitnesses[0];

  const allBacktestIds = fitnesses.flatMap((f) => f.backtestIds);
  const totalPnl = fitnesses.reduce((s, f) => s + f.totalPnl, 0);
  const totalDeals = fitnesses.reduce((s, f) => s + f.totalDeals, 0);
  const avgPnlPerDay = fitnesses.reduce((s, f) => s + f.avgPnlPerDay, 0) / fitnesses.length;
  const winRate = fitnesses.reduce((s, f) => s + f.winRate, 0) / fitnesses.length;
  const maxDrawdown = Math.min(...fitnesses.map((f) => f.maxDrawdown));
  const pnlToRisk = maxDrawdown < 0 ? totalPnl / Math.abs(maxDrawdown) : totalPnl > 0 ? totalPnl : 0;
  const avgDealDuration = fitnesses.reduce((s, f) => s + f.avgDealDuration, 0) / fitnesses.length;

  return { genomeId, backtestIds: allBacktestIds, totalPnl, avgPnlPerDay, winRate, maxDrawdown, pnlToRisk, totalDeals, avgDealDuration, score: 0, nsgaRank: 0, crowdingDistance: 0 };
};

// ═══════════════════════════════════════════════════════════════════
// КЛАСС АВТОПЕРЕБОРЩИКА
// ═══════════════════════════════════════════════════════════════════

export class AutoOptimizer {
  private config: AutoOptimizerConfig;
  private callbacks: AutoOptimizerCallbacks;
  private baseStrategy: FullBotStrategy | null = null;
  private quoteCurrency = 'USDT';
  private exchange = 'BINANCE_FUTURES';
  private importanceTracker = new IndicatorImportanceTracker();

  private phase: AutoOptimizerPhase = 'idle';
  private isStopped = false;
  private isPaused = false;
  private startedAt: number | null = null;
  private totalBacktestsCompleted = 0;
  private totalBacktestsEstimated = 0;

  // Лучший геном (обновляется после каждой фазы)
  private bestGenome: EvaluatedGenome | null = null;

  constructor(config: AutoOptimizerConfig, callbacks: AutoOptimizerCallbacks) {
    this.config = config;
    this.callbacks = callbacks;
  }

  // ─────────────────────────────────────────────────────────────────
  // PUBLIC API
  // ─────────────────────────────────────────────────────────────────

  async start(): Promise<AutoOptimizerResult> {
    this.startedAt = Date.now();
    this.isStopped = false;
    this.isPaused = false;
    this.totalBacktestsCompleted = 0;

    this.estimateTotalBacktests();

    try {
      // ──── ЗАГРУЗКА / ГЕНЕРАЦИЯ СТРАТЕГИИ ────
      let baseGenome: BotGenome;

      if (this.config.botId) {
        this.setPhase('loading_strategy', '📥 Загрузка стратегии базового бота...');
        const strategy = (await fetchBotStrategy(this.config.botId)) as FullBotStrategy;
        this.baseStrategy = strategy;
        this.quoteCurrency = resolveQuoteCurrency(strategy) ?? 'USDT';
        this.exchange = strategy.exchange ?? 'BINANCE_FUTURES';
        this.log('info', `🤖 Бот загружен: ${strategy.name ?? this.config.botId}`);
        this.log('info', `💰 Quote: ${this.quoteCurrency}, Exchange: ${this.exchange}`);
        baseGenome = strategyToGenome(strategy, 0);
      } else {
        this.setPhase('loading_strategy', '🧬 Режим с нуля: генерация случайной стратегии...');
        this.quoteCurrency = this.config.quoteCurrency ?? 'USDT';
        this.exchange = this.config.exchange ?? 'BINANCE_FUTURES';
        baseGenome = createRandomGenome(0);
        // Создаём синтетическую baseStrategy из случайного генома
        const symbol = this.config.symbols[0];
        const fullSymbol = symbol.includes('/') ? symbol : `${symbol}/${this.quoteCurrency}`;
        this.baseStrategy = genomeToStrategy(baseGenome, {
          exchange: this.exchange,
          symbol: fullSymbol,
          quoteCurrency: this.quoteCurrency,
        });
        this.log('info', `🧬 Генерация с нуля: ${this.config.symbols.length} символов`);
        this.log('info', `💰 Quote: ${this.quoteCurrency}, Exchange: ${this.exchange}`);
      }

      let currentBest: EvaluatedGenome | null = null;

      // ──── ФАЗА 1: ПОРОГОВЫЕ ЗНАЧЕНИЯ ИНДИКАТОРОВ ────
      if (this.config.cascade.indicators.enabled && !this.isStopped) {
        this.setPhase('phase_indicators', '🔬 Фаза 1: Оптимизация пороговых значений индикаторов...');
        const scope: OptimizationScope = {
          entryConditions: true,
          entryConditionValues: true,
          entryConditionIndicators: false, // Не меняем сами индикаторы — только их значения
          dcaConditions: false,
          dcaStructure: false,
          dcaIndents: false,
          dcaVolumes: false,
          takeProfit: false,
          takeProfitIndicator: false,
          stopLoss: false,
          leverage: false,
        };
        // В режиме «с нуля» первая фаза получает null → полностью случайная популяция
        const phaseBase = this.config.botId ? baseGenome : null;
        currentBest = await this.runPhase(
          phaseBase,
          scope,
          this.config.cascade.indicators.generations,
          this.config.cascade.indicators.populationSize,
        );
        this.callbacks.onPhaseComplete('phase_indicators', currentBest);
      }

      // ──── ФАЗА 2: СЕТКА DCA ────
      if (this.config.cascade.grid.enabled && !this.isStopped) {
        this.setPhase('phase_grid', '📊 Фаза 2: Оптимизация сетки DCA...');
        const scope: OptimizationScope = {
          entryConditions: false,
          entryConditionValues: false,
          entryConditionIndicators: false,
          dcaConditions: false,
          dcaStructure: true,
          dcaIndents: true,
          dcaVolumes: true,
          takeProfit: false,
          takeProfitIndicator: false,
          stopLoss: false,
          leverage: false,
        };
        currentBest = await this.runPhase(
          currentBest?.genome ?? baseGenome,
          scope,
          this.config.cascade.grid.generations,
          this.config.cascade.grid.populationSize,
        );
        this.callbacks.onPhaseComplete('phase_grid', currentBest);
      }

      // ──── ФАЗА 3: TP/SL ────
      if (this.config.cascade.tpSl.enabled && !this.isStopped) {
        this.setPhase('phase_tp_sl', '🎯 Фаза 3: Оптимизация TP/SL...');
        const scope: OptimizationScope = {
          entryConditions: false,
          entryConditionValues: false,
          entryConditionIndicators: false,
          dcaConditions: false,
          dcaStructure: false,
          dcaIndents: false,
          dcaVolumes: false,
          takeProfit: true,
          takeProfitIndicator: true,
          stopLoss: true,
          leverage: true,
        };
        currentBest = await this.runPhase(
          currentBest?.genome ?? baseGenome,
          scope,
          this.config.cascade.tpSl.generations,
          this.config.cascade.tpSl.populationSize,
        );
        this.callbacks.onPhaseComplete('phase_tp_sl', currentBest);
      }

      this.bestGenome = currentBest;

      if (!currentBest) {
        throw new Error('Оптимизация не дала результатов. Убедитесь что бот имеет сделки на выбранном периоде.');
      }

      // ──── WALK-FORWARD ВАЛИДАЦИЯ ────
      this.setPhase('walk_forward_test', '🔄 Walk-forward валидация...');
      const wfResults = await this.runWalkForwardValidation(currentBest.genome);

      const aggregation = aggregateWalkForwardResults(wfResults);

      this.log('info', `\n📈 Walk-Forward результаты:`);
      this.log('info', `   Median test score: ${aggregation.medianTestScore.toFixed(3)}`);
      this.log('info', `   Robustness: ${aggregation.robustnessScore.toFixed(3)}`);
      this.log('info', `   Overfit ratio: ${aggregation.overfitRatio.toFixed(2)} (ideal ≈ 1.0)`);
      this.log('info', `   Avg test PnL: $${aggregation.avgTestPnl.toFixed(2)}`);
      this.log('info', `   Avg test WinRate: ${aggregation.avgTestWinRate.toFixed(1)}%`);
      this.log('info', `   Min deals per window: ${aggregation.minDeals}`);

      // ──── СОЗДАНИЕ БОТА ────
      let createdBotId: number | null = null;
      if (
        this.config.autoCreateBot &&
        !this.isStopped &&
        aggregation.robustnessScore >= this.config.minRobustnessScore &&
        aggregation.minDeals >= this.config.minDealsPerWindow
      ) {
        this.setPhase('creating_bot', '🤖 Создание бота из лучшей стратегии...');
        createdBotId = await this.createBotFromGenome(currentBest.genome);

        if (createdBotId !== null) {
          this.log('success', `✅ Бот создан! ID: ${createdBotId}`);
        }
      } else if (this.config.autoCreateBot && !this.isStopped) {
        this.log('warning', `⚠️ Бот НЕ создан: robustness=${aggregation.robustnessScore.toFixed(3)} (мин. ${this.config.minRobustnessScore}), minDeals=${aggregation.minDeals} (мин. ${this.config.minDealsPerWindow})`);
      }

      // ──── РЕЗУЛЬТАТ ────
      this.setPhase('completed', '✅ Автопереборщик завершён!');

      const result: AutoOptimizerResult = {
        bestGenome: currentBest,
        walkForwardResults: wfResults,
        aggregation,
        createdBotId,
        totalBacktests: this.totalBacktestsCompleted,
        totalTimeMs: Date.now() - this.startedAt,
      };

      this.callbacks.onComplete(result);
      return result;
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Неизвестная ошибка';
      this.setPhase('error', `❌ ${msg}`);
      this.log('error', `Ошибка: ${msg}`);
      throw error;
    }
  }

  stop(): void {
    this.isStopped = true;
    this.isPaused = false;
    this.log('warning', '⏹️ Автопереборщик останавливается...');
  }

  pause(): void {
    this.isPaused = true;
    this.log('warning', '⏸️ Пауза...');
  }

  unpause(): void {
    this.isPaused = false;
    this.log('info', '▶️ Продолжение...');
  }

  getProgress(): AutoOptimizerProgress {
    return {
      phase: this.phase,
      phaseLabel: '',
      currentWindow: 0,
      totalWindows: this.config.walkForward.windowCount,
      currentGeneration: 0,
      totalGenerations: 0,
      totalBacktests: this.totalBacktestsEstimated,
      completedBacktests: this.totalBacktestsCompleted,
      estimatedEndAt: this.estimateEndTime(),
      startedAt: this.startedAt,
    };
  }

  // ─────────────────────────────────────────────────────────────────
  // КАСКАДНАЯ ФАЗА
  // ─────────────────────────────────────────────────────────────────

  /**
   * Запуск одной фазы каскадной оптимизации.
   * Использует полный период (periodFrom → periodTo).
   */
  private async runPhase(
    baseGenome: BotGenome | null,
    scope: OptimizationScope,
    generations: number,
    populationSize: number,
  ): Promise<EvaluatedGenome | null> {
    const target = this.config.target;
    let population = createInitialPopulation(populationSize, baseGenome, scope);
    let allTimeBest: EvaluatedGenome | null = null;

    for (let gen = 0; gen < generations; gen++) {
      if (this.isStopped) break;
      await this.waitIfPaused();

      this.log('info', `   📊 Поколение ${gen + 1}/${generations} (${population.length} геномов)`);

      // Оценить каждый геном
      const evaluated = await this.evaluatePopulation(
        population,
        this.config.walkForward.periodFrom,
        this.config.walkForward.periodTo,
      );

      // Score
      for (const ev of evaluated) {
        ev.fitness.score = calculateRobustScore(ev.fitness);
      }
      evaluated.sort((a, b) => b.fitness.score - a.fitness.score);

      // Обновить importance tracker
      this.importanceTracker.update(evaluated.slice(0, 5));

      // Обновить лучший
      const best = evaluated[0];
      if (best && (!allTimeBest || best.fitness.score > allTimeBest.fitness.score)) {
        allTimeBest = best;
      }

      if (best) {
        this.log('success', `   🏆 Лучший: Score=${best.fitness.score.toFixed(4)}, PnL=$${best.fitness.totalPnl.toFixed(2)}, WR=${best.fitness.winRate.toFixed(1)}%, Deals=${best.fitness.totalDeals}`);
      }

      // Следующее поколение
      if (gen < generations - 1) {
        const genConfig: GeneticConfig = {
          populationSize,
          generations,
          mutationRate: this.config.genetic.mutationRate,
          crossoverRate: this.config.genetic.crossoverRate,
          elitismCount: this.config.genetic.elitismCount,
          tournamentSize: this.config.genetic.tournamentSize,
          backtestDelaySeconds: this.config.genetic.backtestDelaySeconds,
        };
        population = createNextGeneration(evaluated, genConfig, scope, target);
      }
    }

    return allTimeBest;
  }

  // ─────────────────────────────────────────────────────────────────
  // WALK-FORWARD ВАЛИДАЦИЯ
  // ─────────────────────────────────────────────────────────────────

  /**
   * Проводит walk-forward валидацию: для каждого WF-окна
   * оптимизирует на train, тестирует на test.
   */
  private async runWalkForwardValidation(genome: BotGenome): Promise<WalkForwardResult[]> {
    const windows = buildWalkForwardWindows(this.config.walkForward);
    const results: WalkForwardResult[] = [];

    this.log('info', `\n🔄 Walk-Forward: ${windows.length} окон`);
    for (const w of windows) {
      this.log('info', `   ${formatWindow(w)}`);
    }

    for (const window of windows) {
      if (this.isStopped) break;
      await this.waitIfPaused();

      this.log('info', `\n🪟 Walk-Forward окно #${window.index + 1}/${windows.length}`);

      // ─── TRAIN: мини-оптимизация на train-периоде ───
      this.log('info', `   🏋️ Train: ${window.trainFrom} → ${window.trainTo} (${window.trainDays}д)`);

      const trainScope: OptimizationScope = {
        entryConditions: true,
        entryConditionValues: true,
        entryConditionIndicators: false, // На WF не меняем сами индикаторы, только значения
        dcaConditions: false,
        dcaStructure: false,
        dcaIndents: true,
        dcaVolumes: true,
        takeProfit: true,
        takeProfitIndicator: false,
        stopLoss: false,
        leverage: false,
      };

      // Мини-оптимизация (меньше поколений=быстрее)
      const wfGenerations = Math.max(3, Math.floor(this.config.cascade.grid.generations / 2));
      const wfPopSize = Math.max(8, Math.floor(this.config.cascade.grid.populationSize / 2));

      let trainBest: EvaluatedGenome | null = null;
      let population = createInitialPopulation(wfPopSize, genome, trainScope);

      for (let gen = 0; gen < wfGenerations; gen++) {
        if (this.isStopped) break;
        await this.waitIfPaused();

        const evaluated = await this.evaluatePopulation(population, window.trainFrom, window.trainTo);
        for (const ev of evaluated) {
          ev.fitness.score = calculateRobustScore(ev.fitness);
        }
        evaluated.sort((a, b) => b.fitness.score - a.fitness.score);

        const best = evaluated[0];
        if (best && (!trainBest || best.fitness.score > trainBest.fitness.score)) {
          trainBest = best;
        }

        if (gen < wfGenerations - 1) {
          const genConfig: GeneticConfig = {
            populationSize: wfPopSize,
            generations: wfGenerations,
            mutationRate: this.config.genetic.mutationRate,
            crossoverRate: this.config.genetic.crossoverRate,
            elitismCount: Math.min(2, this.config.genetic.elitismCount),
            tournamentSize: this.config.genetic.tournamentSize,
            backtestDelaySeconds: this.config.genetic.backtestDelaySeconds,
          };
          population = createNextGeneration(evaluated, genConfig, trainScope, this.config.target);
        }
      }

      if (!trainBest) {
        this.log('warning', `   ⚠️ Нет результата на train-окне #${window.index + 1}`);
        continue;
      }

      this.log('info', `   Train лучший: Score=${trainBest.fitness.score.toFixed(4)}, PnL=$${trainBest.fitness.totalPnl.toFixed(2)}`);

      // ─── TEST: один прогон лучшего генома на test-периоде ───
      this.log('info', `   🧪 Test: ${window.testFrom} → ${window.testTo} (${window.testDays}д)`);

      const testFitness = await this.evaluateSingleGenome(
        trainBest.genome,
        window.testFrom,
        window.testTo,
      );

      if (testFitness) {
        testFitness.score = calculateRobustScore(testFitness);

        this.log('info', `   Test: Score=${testFitness.score.toFixed(4)}, PnL=$${testFitness.totalPnl.toFixed(2)}, WR=${testFitness.winRate.toFixed(1)}%, Deals=${testFitness.totalDeals}`);

        const wfResult: WalkForwardResult = {
          window,
          bestGenome: trainBest,
          testFitness,
          trainScore: trainBest.fitness.score,
          testScore: testFitness.score,
        };
        results.push(wfResult);
        this.callbacks.onWalkForwardResult(wfResult);
      } else {
        this.log('warning', `   ⚠️ Не удалось провести test бэктест`);
      }
    }

    return results;
  }

  // ─────────────────────────────────────────────────────────────────
  // СОЗДАНИЕ БОТА
  // ─────────────────────────────────────────────────────────────────

  /**
   * Создать реального бота на платформе Veles из генома.
   */
  private async createBotFromGenome(genome: BotGenome): Promise<number | null> {
    try {
      const symbol = this.config.symbols[0];
      const fullSymbol = symbol.includes('/') ? symbol : `${symbol}/${this.quoteCurrency}`;
      const [baseCurrency] = fullSymbol.split('/');

      // Строим стратегию из генома
      const strategy = this.config.botId && this.baseStrategy
        ? applyGenomeToStrategy(this.baseStrategy, genome, {
            symbol: fullSymbol,
            quoteCurrency: this.quoteCurrency,
            applyConditions: true,
          })
        : genomeToStrategy(genome, {
            exchange: this.exchange,
            symbol: fullSymbol,
            quoteCurrency: this.quoteCurrency,
          });

      // Строим payload для создания бота
      const payload: CreateBotPayload = {
        algorithm: strategy.algorithm ?? 'LONG',
        apiKey: this.config.apiKeyId,
        conditions: strategy.conditions ?? [],
        deposit: {
          amount: this.config.botDeposit,
          leverage: this.config.botLeverage,
          marginType: this.config.botMarginType,
          currency: this.quoteCurrency,
        },
        exchange: this.exchange,
        id: null,
        name: `AutoOpt_${new Date().toISOString().slice(0, 10)}_${genome.id.slice(-6)}`,
        portion: strategy.portion ?? null,
        profit: strategy.profit ?? null,
        pullUp: strategy.pullUp ?? null,
        settings: strategy.settings ?? null,
        stopLoss: strategy.stopLoss ?? null,
        symbols: [fullSymbol],
        termination: null,
      };

      this.log('info', `📤 Создание бота: ${payload.name}`);
      const response = await createBot(payload);
      return response.id;
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Ошибка создания бота';
      this.log('error', `❌ Не удалось создать бота: ${msg}`);
      return null;
    }
  }

  // ─────────────────────────────────────────────────────────────────
  // БЭКТЕСТ ИНФРАСТРУКТУРА
  // ─────────────────────────────────────────────────────────────────

  /**
   * Оценить популяцию геномов на заданном периоде.
   */
  private async evaluatePopulation(
    population: BotGenome[],
    periodFrom: string,
    periodTo: string,
  ): Promise<EvaluatedGenome[]> {
    const results: EvaluatedGenome[] = [];

    for (const genome of population) {
      if (this.isStopped) break;
      await this.waitIfPaused();

      const fitness = await this.evaluateSingleGenome(genome, periodFrom, periodTo);
      if (fitness) {
        results.push({ genome, fitness });
      }

      // Задержка между бэктестами
      if (!this.isStopped) {
        await delay(this.getDelayMs());
      }
    }

    return results;
  }

  /**
   * Оценить один геном на заданном периоде (по всем символам).
   */
  private async evaluateSingleGenome(
    genome: BotGenome,
    periodFrom: string,
    periodTo: string,
  ): Promise<GenomeFitness | null> {
    const fitnesses: GenomeFitness[] = [];

    for (const symbol of this.config.symbols) {
      if (this.isStopped) break;

      try {
        const fitness = await retryWithBackoff(
          () => this.runSingleBacktest(genome, symbol, periodFrom, periodTo),
          {
            maxRetries: MAX_RETRIES,
            initialDelay: INITIAL_RETRY_DELAY_MS,
            maxDelay: MAX_RETRY_DELAY_MS,
            shouldRetry: isRecoverableError,
            onRetry: (attempt, delayMs, error) => {
              this.log('warning', `⏳ Retry ${attempt}/${MAX_RETRIES}: ${error.message}`);
            },
          },
        );

        if (fitness) {
          fitnesses.push(fitness);
        }
      } catch (error) {
        const msg = error instanceof Error ? error.message : 'Ошибка';
        this.log('warning', `⚠️ Бэктест ${symbol}: ${msg}`);
      }

      // Задержка между символами
      if (!this.isStopped && this.config.symbols.length > 1) {
        await delay(this.getDelayMs());
      }
    }

    if (fitnesses.length === 0) return null;
    return aggregateFitness(fitnesses, genome.id);
  }

  /**
   * Один бэктест: отправить → дождаться результата → извлечь fitness.
   */
  private async runSingleBacktest(
    genome: BotGenome,
    symbol: string,
    periodFrom: string,
    periodTo: string,
  ): Promise<GenomeFitness | null> {
    if (!this.baseStrategy) return null;

    const fullSymbol = symbol.includes('/') ? symbol : `${symbol}/${this.quoteCurrency}`;
    const [baseCurrency] = fullSymbol.split('/');

    // Применяем геном
    // Не заменяем conditions бота — только числовые параметры (как в рабочем AI Оптимизаторе)
    const applyConditions = false;

    // В режиме «с нуля» строим стратегию полностью из генома,
    // при наличии базового бота — применяем геном к его копии
    const strategy = this.config.botId && this.baseStrategy
      ? applyGenomeToStrategy(this.baseStrategy, genome, {
          symbol: fullSymbol,
          quoteCurrency: this.quoteCurrency,
          applyConditions,
        })
      : genomeToStrategy(genome, {
          exchange: this.exchange,
          symbol: fullSymbol,
          quoteCurrency: this.quoteCurrency,
        });

    const symbolDescriptor = {
      base: baseCurrency,
      quote: this.quoteCurrency,
      display: fullSymbol,
      pairCode: `${baseCurrency}${this.quoteCurrency}`,
    };

    const payload = buildBacktestPayload(strategy, {
      name: `AO_${genome.id.slice(-8)}_${symbol}`,
      makerCommission: 0.02,
      takerCommission: 0.04,
      includeWicks: true,
      isPublic: false,
      periodStartISO: periodFrom,
      periodEndISO: periodTo,
      overrideSymbol: symbolDescriptor,
    });

    // Логируем первый payload для отладки 400 ошибок
    if (this.totalBacktestsCompleted < 2) {
      console.log('[AutoOptimizer] Backtest payload:', JSON.stringify(payload, null, 2));
    }

    const response = await postBacktest(payload);
    const backtestId = response.id;
    this.totalBacktestsCompleted++;
    this.updateProgress();

    // Дождаться результата
    const result = await this.waitForBacktestResult(backtestId);
    if (!result) {
      this.log('warning', `⚠️ Timeout бэктеста ID=${backtestId}`);
      return null;
    }

    const fitness = extractFitnessFromBacktest(result, genome.id);

    if (fitness.totalDeals === 0) {
      this.log('warning', `⚠️ ${symbol}: 0 сделок`);
    }

    return fitness;
  }

  /**
   * Ожидание завершения бэктеста (polling).
   */
  private async waitForBacktestResult(backtestId: number): Promise<BacktestStatisticsDto | null> {
    const startTime = Date.now();
    while (Date.now() - startTime < BACKTEST_TIMEOUT_MS) {
      if (this.isStopped) return null;
      try {
        const result = await fetchBacktestStatistics(backtestId);
        if (result?.id) return result;
      } catch {
        // Ещё не готов
      }
      await delay(BACKTEST_POLL_INTERVAL_MS);
    }
    return null;
  }

  // ─────────────────────────────────────────────────────────────────
  // ВСПОМОГАТЕЛЬНЫЕ
  // ─────────────────────────────────────────────────────────────────

  private log(level: 'info' | 'success' | 'warning' | 'error', message: string): void {
    this.callbacks.onLog(level, message);
  }

  private setPhase(phase: AutoOptimizerPhase, label: string): void {
    this.phase = phase;
    this.log('info', label);
    this.updateProgress();
  }

  private updateProgress(): void {
    this.callbacks.onProgress(this.getProgress());
  }

  private getDelayMs(): number {
    const seconds = this.config.genetic.backtestDelaySeconds;
    return Math.max(3, Math.min(60, seconds ?? 31)) * 1000;
  }

  private async waitIfPaused(): Promise<void> {
    while (this.isPaused && !this.isStopped) {
      await delay(1000);
    }
  }

  private estimateTotalBacktests(): void {
    const { cascade, walkForward, symbols } = this.config;
    const symbolCount = symbols.length;

    let total = 0;

    // Каскадные фазы
    if (cascade.indicators.enabled) {
      total += cascade.indicators.generations * cascade.indicators.populationSize * symbolCount;
    }
    if (cascade.grid.enabled) {
      total += cascade.grid.generations * cascade.grid.populationSize * symbolCount;
    }
    if (cascade.tpSl.enabled) {
      total += cascade.tpSl.generations * cascade.tpSl.populationSize * symbolCount;
    }

    // Walk-forward
    const wfGens = Math.max(3, Math.floor(cascade.grid.generations / 2));
    const wfPop = Math.max(8, Math.floor(cascade.grid.populationSize / 2));
    // Train + test для каждого окна
    total += walkForward.windowCount * (wfGens * wfPop * symbolCount + symbolCount);

    this.totalBacktestsEstimated = total;
  }

  private estimateEndTime(): number | null {
    if (!this.startedAt || this.totalBacktestsCompleted === 0) return null;
    const elapsed = Date.now() - this.startedAt;
    const avgPerBacktest = elapsed / this.totalBacktestsCompleted;
    const remaining = this.totalBacktestsEstimated - this.totalBacktestsCompleted;
    return Date.now() + remaining * avgPerBacktest;
  }
}
