/**
 * Сервис оркестрации генетического оптимизатора
 *
 * Координирует:
 * - Генерацию популяций геномов
 * - Запуск бэктестов через API
 * - Сбор результатов и расчёт fitness
 * - Эволюцию поколений
 */

import { buildBacktestPayload, fetchBotStrategy, postBacktest, resolveQuoteCurrency } from '../api/backtestRunner';
import { fetchBacktestStatistics } from '../api/backtests';
import type { BacktestStatisticsDto } from '../api/backtests.dtos';
import {
  assignNsgaRanking,
  calculateScore,
  createInitialPopulation,
  createNextGeneration,
} from '../lib/geneticEngine';
import { type FullBotStrategy, applyGenomeToStrategy, genomeToStrategy, strategyToGenome } from '../lib/genomeConverter';
import { readStorageValue, removeStorageValue, writeStorageValue } from '../lib/safeStorage';
import type {
  BotGenome,
  EvaluatedGenome,
  GeneticConfig,
  GenomeFitness,
  OptimizationLogEntry,
  OptimizationProgress,
  OptimizationScope,
  OptimizationTarget,
} from '../types/optimizer';
import type { BotIdentifier } from '../types/bots';

// ═══════════════════════════════════════════════════════════════════
// ТИПЫ
// ═══════════════════════════════════════════════════════════════════

/**
 * Конфигурация запуска оптимизации (runtime)
 */
export interface OptimizationRunConfig {
  botId: BotIdentifier;
  symbols: string[];
  periodFrom: string;
  periodTo: string;
  genetic: GeneticConfig;
  scope: OptimizationScope;
  target: OptimizationTarget;
}

export interface OptimizerCallbacks {
  onLog: (level: OptimizationLogEntry['level'], message: string) => void;
  onProgress: (progress: OptimizationProgress) => void;
  onGenomeEvaluated: (genome: EvaluatedGenome) => void;
  onGenerationComplete: (generation: number, topGenomes: EvaluatedGenome[]) => void;
}

export interface BacktestJob {
  genomeId: string;
  symbol: string;
  backtestId: number | null;
  status: 'pending' | 'running' | 'completed' | 'failed';
  result: BacktestStatisticsDto | null;
  error: string | null;
}

interface OptimizerState {
  config: OptimizationRunConfig;
  baseStrategy: FullBotStrategy | null;
  currentGeneration: number;
  population: BotGenome[];
  evaluatedPopulation: EvaluatedGenome[];
  allTimeTop: EvaluatedGenome[];
  backtestJobs: BacktestJob[];
  isPaused: boolean;
  isStopped: boolean;
  startedAt: number | null;
  // Для восстановления: какой геном/символ сейчас обрабатывается
  currentGenomeIndex: number;
  currentSymbolIndex: number;
  // Промежуточные fitness для текущего генома
  currentGenomeFitnesses: GenomeFitness[];
}

/**
 * Сериализуемое состояние для автосохранения
 */
interface SavedOptimizerState {
  config: OptimizationRunConfig;
  baseStrategy: FullBotStrategy;
  currentGeneration: number;
  population: BotGenome[];
  evaluatedPopulation: EvaluatedGenome[];
  allTimeTop: EvaluatedGenome[];
  currentGenomeIndex: number;
  currentSymbolIndex: number;
  currentGenomeFitnesses: GenomeFitness[];
  startedAt: number;
  savedAt: number;
  completedBacktests: number; // Счётчик завершённых бэктестов
}

// ═══════════════════════════════════════════════════════════════════
// КОНСТАНТЫ
// ═══════════════════════════════════════════════════════════════════

const DEFAULT_BACKTEST_DELAY_MS = 31_000; // 31 секунда по умолчанию
const BACKTEST_POLL_INTERVAL_MS = 5000; // Проверка статуса каждые 5 сек
const BACKTEST_TIMEOUT_MS = 600000; // Таймаут 10 минут
const TOP_GENOMES_LIMIT = 10;

// Retry конфигурация для устойчивости к обрывам связи
const MAX_RETRIES = 5;
const INITIAL_RETRY_DELAY_MS = 10_000; // 10 секунд
const MAX_RETRY_DELAY_MS = 120_000; // 2 минуты максимум

// Ключ для автосохранения в localStorage
const OPTIMIZER_STATE_KEY = 'veles_optimizer_state';

// ═══════════════════════════════════════════════════════════════════
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
// ═══════════════════════════════════════════════════════════════════

/**
 * Задержка выполнения
 */
const delay = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Retry с экспоненциальным откатом
 */
const retryWithBackoff = async <T>(
  fn: () => Promise<T>,
  options: {
    maxRetries: number;
    initialDelay: number;
    maxDelay: number;
    onRetry?: (attempt: number, delay: number, error: Error) => void;
    shouldRetry?: (error: Error) => boolean;
  },
): Promise<T> => {
  let lastError: Error;
  let delayMs = options.initialDelay;

  for (let attempt = 1; attempt <= options.maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      // Проверяем, нужно ли повторять
      if (options.shouldRetry && !options.shouldRetry(lastError)) {
        throw lastError;
      }

      if (attempt === options.maxRetries) {
        throw lastError;
      }

      // Вызываем callback
      options.onRetry?.(attempt, delayMs, lastError);

      // Ждём
      await delay(delayMs);

      // Увеличиваем задержку экспоненциально
      delayMs = Math.min(delayMs * 2, options.maxDelay);
    }
  }

  throw lastError!;
};

/**
 * Проверка, является ли ошибка recoverable (сетевая, rate limit и т.д.)
 */
const isRecoverableError = (error: Error): boolean => {
  const message = error.message.toLowerCase();
  return (
    message.includes('network') ||
    message.includes('fetch') ||
    message.includes('timeout') ||
    message.includes('429') ||
    message.includes('too many') ||
    message.includes('заблокировано') ||
    message.includes('502') ||
    message.includes('503') ||
    message.includes('504') ||
    message.includes('econnreset') ||
    message.includes('econnrefused') ||
    message.includes('socket')
  );
};

/**
 * Сохранение состояния оптимизатора
 */
const saveOptimizerState = (state: SavedOptimizerState): boolean => {
  try {
    const json = JSON.stringify(state);
    return writeStorageValue(OPTIMIZER_STATE_KEY, json);
  } catch (error) {
    console.warn('[Optimizer] Не удалось сохранить состояние:', error);
    return false;
  }
};

/**
 * Загрузка сохранённого состояния
 */
const loadOptimizerState = (): SavedOptimizerState | null => {
  try {
    const json = readStorageValue(OPTIMIZER_STATE_KEY);
    if (!json) return null;
    return JSON.parse(json) as SavedOptimizerState;
  } catch (error) {
    console.warn('[Optimizer] Не удалось загрузить состояние:', error);
    return null;
  }
};

/**
 * Удаление сохранённого состояния
 */
const clearOptimizerState = (): void => {
  removeStorageValue(OPTIMIZER_STATE_KEY);
};

/**
 * Проверка наличия сохранённого состояния
 */
export const hasSavedOptimizerState = (): boolean => {
  return loadOptimizerState() !== null;
};

/**
 * Получение информации о сохранённом состоянии (для UI)
 */
export const getSavedOptimizerInfo = (): { 
  botId: BotIdentifier; 
  generation: number; 
  totalGenerations: number;
  evaluatedGenomes: number;
  savedAt: Date;
} | null => {
  const state = loadOptimizerState();
  if (!state) return null;
  
  return {
    botId: state.config.botId,
    generation: state.currentGeneration + 1,
    totalGenerations: state.config.genetic.generations,
    evaluatedGenomes: state.evaluatedPopulation.length,
    savedAt: new Date(state.savedAt),
  };
};

/**
 * Получение сохранённого списка лучших геномов (allTimeTop) для отображения при загрузке страницы.
 */
export const getSavedTopGenomes = (): EvaluatedGenome[] => {
  const state = loadOptimizerState();
  return state?.allTimeTop ?? [];
};

/**
 * Парсинг символов из строки
 */
export const parseSymbols = (input: string): string[] => {
  return input
    .split(/[,\s]+/)
    .map((s) => s.trim().toUpperCase())
    .filter((s) => s.length > 0);
};

/**
 * Извлечение метрик из результата бэктеста
 */
const extractFitnessFromBacktest = (result: BacktestStatisticsDto, genomeId: string): GenomeFitness => {
  // DEBUG: Смотрим что приходит от API
  console.log('[Optimizer] API result:', {
    totalDeals: result.totalDeals,
    profits: result.profits,
    losses: result.losses,
    winRateProfits: result.winRateProfits,
    winRateLosses: result.winRateLosses,
    netQuote: result.netQuote,
  });

  // Базовые метрики из BacktestStatisticsDto
  const totalPnl = result.netQuote ?? 0;
  const totalDeals = result.totalDeals ?? 0;
  
  // Win Rate - API возвращает КОЛИЧЕСТВО сделок (winRateProfits, winRateLosses)
  // winRateProfits = количество прибыльных сделок, учитываемых в win rate
  // winRateLosses = количество убыточных сделок, учитываемых в win rate
  const winProfits = result.winRateProfits ?? 0;
  const winLosses = result.winRateLosses ?? 0;
  const winRateTotal = winProfits + winLosses;
  
  let winRate = 0;
  if (winRateTotal > 0) {
    winRate = (winProfits / winRateTotal) * 100;
  } else if (totalDeals > 0 && result.profits > 0) {
    // Fallback если winRateProfits/winRateLosses не заполнены
    winRate = (result.profits / totalDeals) * 100;
  }

  console.log('[Optimizer] Calculated winRate:', winRate.toFixed(1) + '%', `(${winProfits}/${winRateTotal} deals)`);

  // MAE как максимальная просадка (в %)
  const maxDrawdown = Math.abs(result.maePercent ?? 0);

  // Расчёт PnL в день
  const periodDays = result.duration > 0 ? result.duration / (24 * 60 * 60 * 1000) : 1;
  const avgPnlPerDay = result.netQuotePerDay ?? (periodDays > 0 ? totalPnl / periodDays : 0);

  // Средняя длительность сделки (в часах)
  const avgDealDuration = (result.avgDuration ?? 0) / (60 * 60 * 1000);

  // Соотношение прибыли к риску
  const pnlToRisk = maxDrawdown > 0 ? totalPnl / maxDrawdown : totalPnl > 0 ? totalPnl : 0;

  return {
    genomeId,
    backtestIds: [result.id],
    totalPnl,
    avgPnlPerDay,
    winRate,
    maxDrawdown: -maxDrawdown, // Отрицательное значение
    pnlToRisk,
    totalDeals,
    avgDealDuration,
    score: 0, // Utility score — рассчитывается позже
    nsgaRank: 0, // Назначается assignNsgaRanking
    crowdingDistance: 0,
  };
};

/**
 * Агрегация fitness по нескольким бэктестам (разные символы)
 */
const aggregateFitness = (fitnesses: GenomeFitness[], genomeId: string): GenomeFitness => {
  if (fitnesses.length === 0) {
    return {
      genomeId,
      backtestIds: [],
      totalPnl: 0,
      avgPnlPerDay: 0,
      winRate: 0,
      maxDrawdown: 0,
      pnlToRisk: 0,
      totalDeals: 0,
      avgDealDuration: 0,
      score: 0,
      nsgaRank: 0,
      crowdingDistance: 0,
    };
  }

  if (fitnesses.length === 1) {
    return fitnesses[0];
  }

  const allBacktestIds = fitnesses.flatMap((f) => f.backtestIds);
  const totalPnl = fitnesses.reduce((sum, f) => sum + f.totalPnl, 0);
  const totalDeals = fitnesses.reduce((sum, f) => sum + f.totalDeals, 0);
  const avgPnlPerDay = fitnesses.reduce((sum, f) => sum + f.avgPnlPerDay, 0) / fitnesses.length;
  const winRate = fitnesses.reduce((sum, f) => sum + f.winRate, 0) / fitnesses.length;
  const maxDrawdown = Math.min(...fitnesses.map((f) => f.maxDrawdown)); // Худшая просадка
  const pnlToRisk = maxDrawdown < 0 ? totalPnl / Math.abs(maxDrawdown) : totalPnl > 0 ? totalPnl : 0;
  const avgDealDuration = fitnesses.reduce((sum, f) => sum + f.avgDealDuration, 0) / fitnesses.length;

  return {
    genomeId,
    backtestIds: allBacktestIds,
    totalPnl,
    avgPnlPerDay,
    winRate,
    maxDrawdown,
    pnlToRisk,
    totalDeals,
    avgDealDuration,
    score: 0,
    nsgaRank: 0,
    crowdingDistance: 0,
  };
};

// ═══════════════════════════════════════════════════════════════════
// КЛАСС ОПТИМИЗАТОРА
// ═══════════════════════════════════════════════════════════════════

export class GeneticOptimizer {
  private state: OptimizerState;
  private callbacks: OptimizerCallbacks;

  constructor(config: OptimizationRunConfig, callbacks: OptimizerCallbacks) {
    this.callbacks = callbacks;
    this.state = {
      config,
      baseStrategy: null,
      currentGeneration: 0,
      population: [],
      evaluatedPopulation: [],
      allTimeTop: [],
      backtestJobs: [],
      isPaused: false,
      isStopped: false,
      startedAt: null,
      currentGenomeIndex: 0,
      currentSymbolIndex: 0,
      currentGenomeFitnesses: [],
    };
  }

  /**
   * Восстановление из сохранённого состояния
   */
  static canResume(): boolean {
    return hasSavedOptimizerState();
  }

  /**
   * Создание оптимизатора из сохранённого состояния
   */
  static fromSavedState(callbacks: OptimizerCallbacks): GeneticOptimizer | null {
    const saved = loadOptimizerState();
    if (!saved) return null;

    const optimizer = new GeneticOptimizer(saved.config, callbacks);
    optimizer.state.baseStrategy = saved.baseStrategy;
    optimizer.state.currentGeneration = saved.currentGeneration;
    optimizer.state.population = saved.population;
    optimizer.state.evaluatedPopulation = saved.evaluatedPopulation;
    optimizer.state.allTimeTop = saved.allTimeTop;
    optimizer.state.currentGenomeIndex = saved.currentGenomeIndex;
    optimizer.state.currentSymbolIndex = saved.currentSymbolIndex;
    optimizer.state.currentGenomeFitnesses = saved.currentGenomeFitnesses;
    optimizer.state.startedAt = saved.startedAt;

    // Восстанавливаем счётчик бэктестов как плейсхолдер-jobs
    const completedCount = saved.completedBacktests ?? 0;
    for (let i = 0; i < completedCount; i++) {
      optimizer.state.backtestJobs.push({
        genomeId: `restored-${i}`,
        symbol: 'restored',
        backtestId: null,
        status: 'completed',
        result: null,
        error: null,
      });
    }

    return optimizer;
  }

  /**
   * Получить текущий топ геномов (для отображения сразу после восстановления)
   */
  getAllTimeTop(): EvaluatedGenome[] {
    return [...this.state.allTimeTop];
  }

  /**
   * Очистка сохранённого состояния
   */
  static clearSavedState(): void {
    clearOptimizerState();
  }

  /**
   * Сохранение текущего состояния
   */
  private saveState(): void {
    if (!this.state.baseStrategy || !this.state.startedAt) return;

    const saved: SavedOptimizerState = {
      config: this.state.config,
      baseStrategy: this.state.baseStrategy,
      currentGeneration: this.state.currentGeneration,
      population: this.state.population,
      evaluatedPopulation: this.state.evaluatedPopulation,
      allTimeTop: this.state.allTimeTop,
      currentGenomeIndex: this.state.currentGenomeIndex,
      currentSymbolIndex: this.state.currentSymbolIndex,
      currentGenomeFitnesses: this.state.currentGenomeFitnesses,
      startedAt: this.state.startedAt,
      savedAt: Date.now(),
      completedBacktests: this.state.backtestJobs.filter((j) => j.status === 'completed').length,
    };

    if (saveOptimizerState(saved)) {
      console.log('[Optimizer] Состояние сохранено');
    }
  }

  /**
   * Запуск оптимизации
   */
  async start(): Promise<EvaluatedGenome[]> {
    const { config } = this.state;
    this.state.startedAt = Date.now();
    this.state.isStopped = false;
    this.state.isPaused = false;

    this.log('info', `🚀 Запуск оптимизации для бота ID: ${config.botId}`);

    try {
      // Загружаем стратегию бота
      this.log('info', '📥 Загрузка стратегии бота...');
      const strategy = (await fetchBotStrategy(config.botId)) as FullBotStrategy;
      this.state.baseStrategy = strategy;

      // Определяем quote currency
      const quoteCurrency = resolveQuoteCurrency(strategy) ?? 'USDT';
      this.log('info', `💰 Валюта котировки: ${quoteCurrency}`);

      // Создаём начальную популяцию
      this.log('info', `🧬 Создание начальной популяции (${config.genetic.populationSize} особей)...`);

      // Если есть базовая стратегия, используем её как основу для части популяции
      const baseGenome = strategyToGenome(strategy, 0);
      const initialPopulation = createInitialPopulation(
        config.genetic.populationSize,
        baseGenome,
        config.scope,
      );

      this.state.population = initialPopulation;
      
      // Сохраняем начальное состояние
      this.saveState();

      // Основной цикл поколений
      await this.runGenerationLoop(quoteCurrency);

      this.log('success', '✅ Оптимизация завершена!');
      // Очищаем сохранённое состояние при успешном завершении
      clearOptimizerState();
      return this.state.allTimeTop;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Неизвестная ошибка';
      this.log('error', `❌ Ошибка: ${message}`);
      // Состояние уже сохранено, можно восстановить позже
      this.log('info', '💾 Прогресс сохранён. Можно продолжить после восстановления связи.');
      throw error;
    }
  }

  /**
   * Продолжение оптимизации из сохранённого состояния
   */
  async resume(): Promise<EvaluatedGenome[]> {
    const { config, baseStrategy } = this.state;
    
    if (!baseStrategy) {
      throw new Error('Нет сохранённой стратегии для восстановления');
    }

    this.state.isStopped = false;
    this.state.isPaused = false;

    const savedInfo = getSavedOptimizerInfo();
    this.log('info', `🔄 Восстановление оптимизации с поколения ${savedInfo?.generation ?? 1}`);
    this.log('info', `📊 Уже оценено геномов: ${this.state.evaluatedPopulation.length}`);

    const quoteCurrency = resolveQuoteCurrency(baseStrategy) ?? 'USDT';

    try {
      await this.runGenerationLoop(quoteCurrency);

      this.log('success', '✅ Оптимизация завершена!');
      clearOptimizerState();
      return this.state.allTimeTop;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Неизвестная ошибка';
      this.log('error', `❌ Ошибка: ${message}`);
      this.log('info', '💾 Прогресс сохранён. Можно продолжить после восстановления связи.');
      throw error;
    }
  }

  /**
   * Основной цикл поколений
   */
  private async runGenerationLoop(quoteCurrency: string): Promise<void> {
    const { config, baseStrategy } = this.state;
    
    if (!baseStrategy) {
      throw new Error('Базовая стратегия не загружена');
    }

    for (let gen = this.state.currentGeneration; gen < config.genetic.generations; gen++) {
      if (this.state.isStopped) {
        this.log('warning', '⏹️ Оптимизация остановлена пользователем');
        break;
      }

      while (this.state.isPaused) {
        await delay(1000);
        if (this.state.isStopped) break;
      }

      this.state.currentGeneration = gen;
      this.log('info', `\n📊 === Поколение ${gen + 1}/${config.genetic.generations} ===`);

      // Оцениваем текущую популяцию
      const evaluated = await this.evaluatePopulation(
        this.state.population,
        config.symbols,
        baseStrategy.exchange ?? 'BINANCE_FUTURES',
        quoteCurrency,
      );

      // Рассчитываем score для каждого генома
      for (const ev of evaluated) {
        ev.fitness.score = calculateScore(ev.fitness, config.target);
      }

      // NSGA-II: назначаем ранги для Pareto-меток (информационно)
      assignNsgaRanking(evaluated);

      // Сортируем по score (основная метрика селекции)
      evaluated.sort((a, b) => b.fitness.score - a.fitness.score);
      this.state.evaluatedPopulation = evaluated;

      // Обновляем топ за все время
      this.updateAllTimeTop(evaluated);

      // Сохраняем состояние после каждого поколения
      this.saveState();

      // Логируем лучшего в поколении
      const paretoCount = evaluated.filter((ev) => ev.paretoOptimal).length;
      const best = evaluated[0];
      if (best) {
        this.log(
          'success',
          `🏆 Лучший: Score=${best.fitness.score.toFixed(3)}, PnL=$${best.fitness.totalPnl.toFixed(2)}, WinRate=${best.fitness.winRate.toFixed(1)}%, DD=${best.fitness.maxDrawdown.toFixed(1)}%` +
            `${best.paretoOptimal ? ' ★' : ''} | Pareto: ${paretoCount}`,
        );
      }

      this.callbacks.onGenerationComplete(gen, this.state.allTimeTop);

      // Сбрасываем индексы для следующего поколения
      this.state.currentGenomeIndex = 0;
      this.state.currentSymbolIndex = 0;
      this.state.currentGenomeFitnesses = [];

      // Создаём следующее поколение (кроме последнего)
      if (gen < config.genetic.generations - 1) {
        this.log('info', '🔄 Создание следующего поколения...');
        this.state.population = createNextGeneration(
          evaluated,
          config.genetic,
          config.scope,
          config.target,
        );
        
        // Сохраняем после создания нового поколения
        this.saveState();
      }
    }
  }

  /**
   * Пауза оптимизации
   */
  pause(): void {
    this.state.isPaused = true;
    this.saveState();
    this.log('warning', '⏸️ Оптимизация приостановлена');
  }

  /**
   * Возобновление из паузы (в рамках текущей сессии)
   */
  unpause(): void {
    this.state.isPaused = false;
    this.log('info', '▶️ Оптимизация возобновлена');
  }

  /**
   * Остановка оптимизации
   */
  stop(): void {
    this.state.isStopped = true;
    this.state.isPaused = false;
    this.saveState();
    this.log('warning', '⏹️ Оптимизация остановлена');
  }

  /**
   * Получение текущего прогресса
   */
  getProgress(): OptimizationProgress {
    const { config, currentGeneration, backtestJobs, startedAt } = this.state;
    const totalBacktests = config.genetic.populationSize * config.genetic.generations * config.symbols.length;
    const completedBacktests = backtestJobs.filter((j) => j.status === 'completed').length;
    const delayMs = this.getBacktestDelayMs();

    return {
      status: this.state.isStopped ? 'idle' : this.state.isPaused ? 'paused' : 'running',
      currentGeneration: currentGeneration + 1,
      totalGenerations: config.genetic.generations,
      evaluatedGenomes: this.state.evaluatedPopulation.length,
      totalBacktests,
      completedBacktests,
      startedAt,
      estimatedEndAt: startedAt
        ? startedAt + (totalBacktests - completedBacktests) * delayMs
        : null,
      error: null,
    };
  }

  /**
   * Получить задержку между бэктестами в мс
   */
  private getBacktestDelayMs(): number {
    const seconds = this.state.config.genetic.backtestDelaySeconds;
    // Ограничиваем 1-60 сек (по умолчанию 31)
    const clamped = Math.max(1, Math.min(60, seconds ?? 31));
    return clamped * 1000;
  }

  // ═══════════════════════════════════════════════════════════════════
  // ПРИВАТНЫЕ МЕТОДЫ
  // ═══════════════════════════════════════════════════════════════════

  /**
   * Логирование с callback
   */
  private log(level: OptimizationLogEntry['level'], message: string): void {
    this.callbacks.onLog(level, message);
  }

  /**
   * Обновление прогресса
   */
  private updateProgress(): void {
    this.callbacks.onProgress(this.getProgress());
  }

  /**
   * Обновление топа за все время.
   * Сортировка по score, Pareto-метки пересчитываются для UI.
   */
  private updateAllTimeTop(evaluated: EvaluatedGenome[]): void {
    const combined = [...this.state.allTimeTop, ...evaluated];

    // Удаляем дубликаты по ID генома
    const unique = new Map<string, EvaluatedGenome>();
    for (const ev of combined) {
      const existing = unique.get(ev.genome.id);
      if (!existing || ev.fitness.score > existing.fitness.score) {
        unique.set(ev.genome.id, ev);
      }
    }

    // Сортируем по score и обрезаем
    const pool = Array.from(unique.values())
      .sort((a, b) => b.fitness.score - a.fitness.score)
      .slice(0, TOP_GENOMES_LIMIT);

    // Пересчитываем Pareto-метки для UI
    assignNsgaRanking(pool);

    this.state.allTimeTop = pool;
  }

  /**
   * Оценка популяции через бэктесты
   */
  private async evaluatePopulation(
    population: BotGenome[],
    symbols: string[],
    exchange: string,
    quoteCurrency: string,
  ): Promise<EvaluatedGenome[]> {
    const results: EvaluatedGenome[] = [...this.state.evaluatedPopulation];

    // Продолжаем с того места, где остановились
    const startGenomeIndex = this.state.currentGenomeIndex;
    
    for (let gi = startGenomeIndex; gi < population.length; gi++) {
      const genome = population[gi];
      this.state.currentGenomeIndex = gi;
      
      if (this.state.isStopped) break;

      while (this.state.isPaused) {
        await delay(1000);
        if (this.state.isStopped) break;
      }

      // Используем сохранённые fitness или начинаем заново
      const fitnesses: GenomeFitness[] = gi === startGenomeIndex 
        ? [...this.state.currentGenomeFitnesses]
        : [];
      
      // Продолжаем с того символа, где остановились
      const startSymbolIndex = gi === startGenomeIndex ? this.state.currentSymbolIndex : 0;

      for (let si = startSymbolIndex; si < symbols.length; si++) {
        const symbol = symbols[si];
        this.state.currentSymbolIndex = si;
        
        if (this.state.isStopped) break;

        try {
          // Retry с экспоненциальным откатом для устойчивости к сетевым ошибкам
          const fitness = await retryWithBackoff(
            () => this.runBacktestForGenome(genome, symbol, exchange, quoteCurrency),
            {
              maxRetries: MAX_RETRIES,
              initialDelay: INITIAL_RETRY_DELAY_MS,
              maxDelay: MAX_RETRY_DELAY_MS,
              shouldRetry: isRecoverableError,
              onRetry: (attempt, delayMs, error) => {
                this.log('warning', `⏳ Попытка ${attempt}/${MAX_RETRIES}: ${error.message}. Ждём ${Math.round(delayMs / 1000)}с...`);
                // Сохраняем состояние перед ожиданием
                this.state.currentGenomeFitnesses = fitnesses;
                this.saveState();
              },
            },
          );
          
          if (fitness) {
            fitnesses.push(fitness);
            // Сохраняем промежуточный результат
            this.state.currentGenomeFitnesses = fitnesses;
            this.saveState();
          }
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Ошибка бэктеста';
          this.log('warning', `⚠️ Ошибка бэктеста ${symbol} после ${MAX_RETRIES} попыток: ${message}`);
          
          // Если это критическая ошибка после всех retry - сохраняем и выбрасываем
          if (isRecoverableError(error instanceof Error ? error : new Error(message))) {
            this.state.currentGenomeFitnesses = fitnesses;
            this.saveState();
            throw error; // Позволяет восстановить позже
          }
        }

        // Задержка между бэктестами
        if (!this.state.isStopped && si < symbols.length - 1) {
          await delay(this.getBacktestDelayMs());
        }
      }

      // Агрегируем результаты для генома
      if (fitnesses.length > 0) {
        const aggregated = aggregateFitness(fitnesses, genome.id);
        // Рассчитываем score сразу, чтобы UI получал актуальное значение
        aggregated.score = calculateScore(aggregated, this.state.config.target);
        const evaluated: EvaluatedGenome = { genome, fitness: aggregated };
        results.push(evaluated);
        // Обновляем evaluatedPopulation сразу, чтобы счётчик «Оценено геномов» рос в реальном времени
        this.state.evaluatedPopulation = [...results];
        this.callbacks.onGenomeEvaluated(evaluated);
      }

      // Сбрасываем для следующего генома
      this.state.currentSymbolIndex = 0;
      this.state.currentGenomeFitnesses = [];

      this.updateProgress();
      
      // Задержка перед следующим геномом
      if (!this.state.isStopped && gi < population.length - 1) {
        await delay(this.getBacktestDelayMs());
      }
    }

    return results;
  }

  /**
   * Запуск бэктеста для одного генома и символа
   */
  private async runBacktestForGenome(
    genome: BotGenome,
    symbol: string,
    exchange: string,
    quoteCurrency: string,
  ): Promise<GenomeFitness | null> {
    const { config, baseStrategy } = this.state;

    if (!baseStrategy) {
      throw new Error('Базовая стратегия не загружена');
    }

    // Формируем полный символ
    const fullSymbol = symbol.includes('/') ? symbol : `${symbol}/${quoteCurrency}`;

    this.log('info', `🔬 Бэктест: ${fullSymbol} (геном ${genome.id.slice(-6)})`);

    // DEBUG: Логируем параметры генома для проверки мутаций
    console.log('[Optimizer] Genome DCA orders:', genome.dcaOrders.map((o, i) => 
      `DCA${i+1}: indent=${o.indent.toFixed(2)}%, volume=${o.volume.toFixed(2)}%`
    ));
    console.log('[Optimizer] Genome baseOrder:', `indent=${genome.baseOrder.indent.toFixed(2)}%, volume=${genome.baseOrder.volume.toFixed(2)}%`);
    console.log('[Optimizer] Genome TP:', genome.takeProfit.value, 'Deposit:', genome.depositAmount);

    // Применяем условия ТОЛЬКО если включены соответствующие КОНКРЕТНЫЕ галочки
    // entryConditions/entryConditionValues - это общий флаг, не значит что нужно применять мутированные условия
    // Применяем только если включены индикаторы (не просто значения)
    const applyConditions = config.scope.entryConditionIndicators || 
                           config.scope.dcaConditions || 
                           config.scope.takeProfitIndicator;

    // Применяем геном к копии базовой стратегии
    const strategy = applyGenomeToStrategy(baseStrategy, genome, {
      symbol: fullSymbol,
      quoteCurrency,
      applyConditions,
    });

    // DEBUG: Проверяем что изменения применились к strategy
    console.log('[Optimizer] Strategy after applyGenome - settings.orders:', 
      strategy.settings?.orders?.map((o, i) => `DCA${i+1}: indent=${o.indent}%, volume=${o.volume}%`)
    );
    console.log('[Optimizer] Strategy after applyGenome - baseOrder:', 
      `indent=${strategy.settings?.baseOrder?.indent}%, volume=${strategy.settings?.baseOrder?.volume}%`
    );
    console.log('[Optimizer] Strategy after applyGenome - deposit:', 
      `amount=${strategy.deposit?.amount}, leverage=${strategy.deposit?.leverage}`
    );
    console.log('[Optimizer] Strategy after applyGenome - profit.checkPnl:', strategy.profit?.checkPnl);
    
    if (applyConditions) {
      console.log('[Optimizer] Entry conditions applied:', strategy.conditions?.length ?? 0);
    }

    // Строим дескриптор символа для buildBacktestPayload
    const [baseCurrency] = fullSymbol.split('/');
    const symbolDescriptor = {
      base: baseCurrency,
      quote: quoteCurrency,
      display: fullSymbol,
      pairCode: `${baseCurrency}${quoteCurrency}`,
    };

    // Строим payload для бэктеста (используем overrideSymbol для корректной подстановки)
    const payload = buildBacktestPayload(strategy, {
      name: `Optimizer_${genome.id.slice(-8)}_${symbol}`,
      makerCommission: 0.02,
      takerCommission: 0.04,
      includeWicks: true,
      isPublic: false,
      periodStartISO: config.periodFrom,
      periodEndISO: config.periodTo,
      overrideSymbol: symbolDescriptor,
    });

    // DEBUG: Логируем payload для отладки
    console.log('[Optimizer] Backtest payload:', JSON.stringify(payload, null, 2));

    // Отправляем бэктест
    const response = await postBacktest(payload);
    const backtestId = response.id;

    // Добавляем job в список для отслеживания прогресса
    const job: BacktestJob = {
      genomeId: genome.id,
      symbol,
      backtestId,
      status: 'running',
      result: null,
      error: null,
    };
    this.state.backtestJobs.push(job);
    this.updateProgress();

    this.log('info', `📤 Бэктест отправлен: ID=${backtestId}`);

    // Ждём результат
    const result = await this.waitForBacktestResult(backtestId);

    if (!result) {
      job.status = 'failed';
      job.error = 'Timeout';
      this.updateProgress();
      this.log('warning', `⚠️ Бэктест ${backtestId} не завершился вовремя`);
      return null;
    }

    // Обновляем job
    job.status = 'completed';
    job.result = result;
    this.updateProgress();

    // Извлекаем метрики
    const fitness = extractFitnessFromBacktest(result, genome.id);
    
    // Логируем с количеством сделок для диагностики
    if (fitness.totalDeals === 0) {
      this.log('warning', `⚠️ ${symbol}: 0 сделок! Проверьте условия входа и период.`);
    } else {
      this.log(
        'info',
        `✓ Результат ${symbol}: PnL=$${fitness.totalPnl.toFixed(2)}, WinRate=${fitness.winRate.toFixed(1)}%, Сделок=${fitness.totalDeals}`,
      );
    }

    return fitness;
  }

  /**
   * Ожидание результата бэктеста (опрос статуса)
   */
  private async waitForBacktestResult(backtestId: number): Promise<BacktestStatisticsDto | null> {
    const startTime = Date.now();

    while (Date.now() - startTime < BACKTEST_TIMEOUT_MS) {
      if (this.state.isStopped) return null;

      try {
        const result = await fetchBacktestStatistics(backtestId);

        // BacktestStatisticsDto не имеет поля status напрямую,
        // но если запрос успешен - значит бэктест завершён
        if (result && result.id) {
          return result;
        }
      } catch {
        // Бэктест ещё не готов, продолжаем ждать
      }

      await delay(BACKTEST_POLL_INTERVAL_MS);
    }

    return null;
  }
}

// ═══════════════════════════════════════════════════════════════════
// ФАБРИКА
// ═══════════════════════════════════════════════════════════════════

/**
 * Создание конфигурации оптимизатора
 */
export const createOptimizerConfig = (params: {
  botId: BotIdentifier;
  symbols: string[];
  periodFrom: string;
  periodTo: string;
  genetic: GeneticConfig;
  scope: OptimizationScope;
  target: OptimizationTarget;
}): OptimizationRunConfig => {
  return {
    botId: params.botId,
    symbols: params.symbols,
    periodFrom: params.periodFrom,
    periodTo: params.periodTo,
    genetic: params.genetic,
    scope: params.scope,
    target: params.target,
  };
};
