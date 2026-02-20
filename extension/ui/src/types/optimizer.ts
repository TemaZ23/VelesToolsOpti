/**
 * Типы для генетического оптимизатора стратегий
 */

// ═══════════════════════════════════════════════════════════════════
// КАТАЛОГ ИНДИКАТОРОВ
// ═══════════════════════════════════════════════════════════════════

export type IndicatorCategory = 'trend' | 'channel' | 'oscillator' | 'volatility' | 'volume';

export interface IndicatorDefinition {
  id: string;
  name: string;
  nameRu: string;
  category: IndicatorCategory;
  hasValue: boolean;
  defaultValue: number | null;
  minValue: number | null;
  maxValue: number | null;
  operations: ConditionOperation[];
}

export type ConditionOperation = '>' | '<' | '>=' | '<=' | '==' | 'CROSS_UP' | 'CROSS_DOWN';

// Короткий формат интервалов (для UI)
export type TimeIntervalShort = '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d';

// Формат интервалов API
export type TimeIntervalApi = 'ONE_MINUTE' | 'FIVE_MINUTES' | 'FIFTEEN_MINUTES' | 'THIRTY_MINUTES' | 'ONE_HOUR' | 'FOUR_HOURS' | 'ONE_DAY';

// Объединённый тип (поддержка обоих форматов)
export type TimeInterval = TimeIntervalShort | TimeIntervalApi;

export const TIME_INTERVALS: TimeIntervalShort[] = ['1m', '5m', '15m', '30m', '1h', '4h', '1d'];

export const TIME_INTERVAL_LABELS: Record<TimeIntervalShort, string> = {
  '1m': '1 минута',
  '5m': '5 минут',
  '15m': '15 минут',
  '30m': '30 минут',
  '1h': '1 час',
  '4h': '4 часа',
  '1d': '1 день',
};

// ═══════════════════════════════════════════════════════════════════
// ГЕНОМ БОТА (генетическое представление)
// ═══════════════════════════════════════════════════════════════════

export interface ConditionGene {
  indicator: string;
  interval: TimeInterval;
  value: number | null;
  operation: ConditionOperation;
  closed: boolean;
  reverse: boolean;
  basic: boolean; // Важно для индикаторов-сигналов (BB, RSI_LEVELS, etc.)
}

export interface GridOrderGene {
  indent: number;
  volume: number;
  conditions: ConditionGene[];
}

export interface TakeProfitGene {
  type: 'PERCENT' | 'ABSOLUTE' | 'PNL';
  value: number;
  indicator: ConditionGene | null;
}

export interface StopLossGene {
  indent: number;
  termination: boolean;
  conditionalIndent: number | null;
  conditions: ConditionGene[];
}

export interface BotGenome {
  id: string;
  generation: number;
  
  // Основные параметры
  algorithm: 'LONG' | 'SHORT';
  leverage: number;
  depositAmount: number;
  
  // Условия входа
  entryConditions: ConditionGene[];
  
  // Сетка ордеров
  baseOrder: GridOrderGene;
  dcaOrders: GridOrderGene[];
  
  // Тейк-профит
  takeProfit: TakeProfitGene;
  
  // Стоп-лосс (опционально)
  stopLoss: StopLossGene | null;
  
  // Дополнительные параметры
  pullUp: number | null;
  portion: number | null;
}

// ═══════════════════════════════════════════════════════════════════
// РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ
// ═══════════════════════════════════════════════════════════════════

export interface GenomeFitness {
  genomeId: string;
  backtestIds: number[];
  
  // Метрики
  totalPnl: number;
  avgPnlPerDay: number;
  winRate: number;
  maxDrawdown: number;
  pnlToRisk: number;
  totalDeals: number;
  avgDealDuration: number;
  
  // NSGA-II ранжирование
  /** Итоговый utility score (для совместимости и сортировки) */
  score: number;
  /** NSGA-II ранг Pareto-слоя (1 = фронт, 2 = второй слой, ...) */
  nsgaRank: number;
  /** Crowding distance (бесконечность = граничный элемент) */
  crowdingDistance: number;
}

export interface EvaluatedGenome {
  genome: BotGenome;
  fitness: GenomeFitness;
  /** Принадлежит ли первому Pareto-фронту */
  paretoOptimal?: boolean;
}

// ═══════════════════════════════════════════════════════════════════
// КОНФИГУРАЦИЯ ОПТИМИЗАТОРА
// ═══════════════════════════════════════════════════════════════════

export interface OptimizationTarget {
  metric: 'pnl' | 'pnlPerDay' | 'winRate' | 'pnlToRisk' | 'composite';
  weights?: {
    pnl: number;
    winRate: number;
    maxDrawdown: number;
    pnlToRisk: number;
  };
}

export interface OptimizationScope {
  // Что оптимизируем
  entryConditions: boolean;
  entryConditionValues: boolean;
  entryConditionIndicators: boolean;
  
  dcaConditions: boolean;
  dcaStructure: boolean;
  dcaIndents: boolean;
  dcaVolumes: boolean;
  
  takeProfit: boolean;
  takeProfitIndicator: boolean;
  
  stopLoss: boolean;
  
  leverage: boolean;
}

export interface GeneticConfig {
  populationSize: number;
  generations: number;
  mutationRate: number;
  crossoverRate: number;
  elitismCount: number;
  tournamentSize: number;
  /** Задержка между бэктестами в секундах (3-60) */
  backtestDelaySeconds: number;
}

export interface OptimizerConfig {
  // Базовый бот для оптимизации
  baseBotId: number | null;
  baseGenome: BotGenome | null;
  
  // Период тестирования
  periodFrom: string;
  periodTo: string;
  
  // Монеты для тестирования
  symbols: string[];
  
  // Целевая метрика
  target: OptimizationTarget;
  
  // Область оптимизации
  scope: OptimizationScope;
  
  // Настройки генетического алгоритма
  genetic: GeneticConfig;
  
  // Комиссии
  makerCommission: number;
  takerCommission: number;
}

// ═══════════════════════════════════════════════════════════════════
// СОСТОЯНИЕ ОПТИМИЗАЦИИ
// ═══════════════════════════════════════════════════════════════════

export type OptimizationStatus = 'idle' | 'running' | 'paused' | 'completed' | 'error';

export interface OptimizationProgress {
  status: OptimizationStatus;
  currentGeneration: number;
  totalGenerations: number;
  evaluatedGenomes: number;
  totalBacktests: number;
  completedBacktests: number;
  startedAt: number | null;
  estimatedEndAt: number | null;
  error: string | null;
}

export interface OptimizationState {
  config: OptimizerConfig;
  progress: OptimizationProgress;
  population: EvaluatedGenome[];
  bestGenome: EvaluatedGenome | null;
  history: EvaluatedGenome[][];
  logs: OptimizationLogEntry[];
}

export interface OptimizationLogEntry {
  id: string;
  timestamp: number;
  level: 'info' | 'success' | 'warning' | 'error';
  message: string;
  details?: Record<string, unknown>;
}

// ═══════════════════════════════════════════════════════════════════
// РЕЗУЛЬТАТ ОПТИМИЗАЦИИ
// ═══════════════════════════════════════════════════════════════════

export interface OptimizationResult {
  config: OptimizerConfig;
  bestGenome: EvaluatedGenome;
  topGenomes: EvaluatedGenome[];
  totalBacktests: number;
  totalTime: number;
  generationsCompleted: number;
}
