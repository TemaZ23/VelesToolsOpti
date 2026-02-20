/**
 * Типы для полного автопереборщика стратегий
 *
 * Walk-forward validation + генетическая оптимизация + автосоздание ботов
 */

import type { BotGenome, EvaluatedGenome, GeneticConfig, GenomeFitness, OptimizationScope, OptimizationTarget } from './optimizer';
import type { BotIdentifier } from './bots';

// ═══════════════════════════════════════════════════════════════════
// WALK-FORWARD КОНФИГУРАЦИЯ
// ═══════════════════════════════════════════════════════════════════

/** Один сегмент walk-forward: train + test (out-of-sample) */
export interface WalkForwardWindow {
  /** Порядковый номер окна (0-based) */
  index: number;
  /** Начало train-периода (ISO) */
  trainFrom: string;
  /** Конец train-периода (ISO) */
  trainTo: string;
  /** Начало test-периода (ISO) */
  testFrom: string;
  /** Конец test-периода (ISO) */
  testTo: string;
  /** Длительность train в днях */
  trainDays: number;
  /** Длительность test в днях */
  testDays: number;
}

/** Конфигурация walk-forward разбиения */
export interface WalkForwardConfig {
  /** Общий период: начало (ISO) */
  periodFrom: string;
  /** Общий период: конец (ISO) */
  periodTo: string;
  /** Кол-во WF окон */
  windowCount: number;
  /** Доля train от каждого окна (0.6–0.9) */
  trainRatio: number;
  /** Перекрытие окон: скольжение (true) или нет (false) */
  sliding: boolean;
}

// ═══════════════════════════════════════════════════════════════════
// КАСКАДНАЯ ОПТИМИЗАЦИЯ
// ═══════════════════════════════════════════════════════════════════

/** Фаза автопереборщика */
export type AutoOptimizerPhase =
  | 'idle'
  | 'loading_strategy'
  | 'phase_indicators'   // Фаза 1: перебор условий входа
  | 'phase_grid'         // Фаза 2: перебор сетки DCA
  | 'phase_tp_sl'        // Фаза 3: перебор TP/SL
  | 'walk_forward_test'  // Финал: WF валидация лучшего генома
  | 'creating_bot'       // Создание бота на платформе
  | 'completed'
  | 'error';

/** Конфигурация каскадной оптимизации */
export interface CascadeConfig {
  /** Фаза 1: оптимизация индикаторов */
  indicators: {
    enabled: boolean;
    generations: number;
    populationSize: number;
  };
  /** Фаза 2: оптимизация сетки DCA */
  grid: {
    enabled: boolean;
    generations: number;
    populationSize: number;
  };
  /** Фаза 3: оптимизация TP/SL */
  tpSl: {
    enabled: boolean;
    generations: number;
    populationSize: number;
  };
}

// ═══════════════════════════════════════════════════════════════════
// РЕЗУЛЬТАТЫ
// ═══════════════════════════════════════════════════════════════════

/** Результат одного WF-окна */
export interface WalkForwardResult {
  window: WalkForwardWindow;
  /** Лучший геном, найденный на train */
  bestGenome: EvaluatedGenome;
  /** Результат на test (out-of-sample) */
  testFitness: GenomeFitness;
  /** Score на train */
  trainScore: number;
  /** Score на test */
  testScore: number;
}

/** Агрегированные метрики по всем WF-окнам */
export interface WalkForwardAggregation {
  /** Медианный score на test */
  medianTestScore: number;
  /** Средний score на test */
  avgTestScore: number;
  /** Стд. отклонение scores на test */
  stdTestScore: number;
  /** Robustness: median - std (стабильность) */
  robustnessScore: number;
  /** Средний total PnL на test */
  avgTestPnl: number;
  /** Средний win rate на test */
  avgTestWinRate: number;
  /** Худшая просадка среди всех test окон */
  worstDrawdown: number;
  /** Мин. кол-во сделок на test окне */
  minDeals: number;
  /** Соотношение: avgTestScore / avgTrainScore (идеально ≈ 1.0) */
  overfitRatio: number;
}

/** Полный результат автопереборщика */
export interface AutoOptimizerResult {
  /** Лучший геном из каскадной оптимизации */
  bestGenome: EvaluatedGenome;
  /** Результаты walk-forward валидации */
  walkForwardResults: WalkForwardResult[];
  /** Агрегация по WF */
  aggregation: WalkForwardAggregation;
  /** ID созданного бота (null если бот не создан) */
  createdBotId: number | null;
  /** Общее кол-во бэктестов */
  totalBacktests: number;
  /** Общее время работы (мс) */
  totalTimeMs: number;
}

// ═══════════════════════════════════════════════════════════════════
// КОНФИГУРАЦИЯ ЗАПУСКА
// ═══════════════════════════════════════════════════════════════════

/** Полная конфигурация автопереборщика */
export interface AutoOptimizerConfig {
  /** Базовый бот (источник стратегии). Если не указан — генерация стратегий с нуля */
  botId?: BotIdentifier | null;
  /** API-ключ для создания нового бота */
  apiKeyId: number;
  /** Символы для тестирования */
  symbols: string[];
  /** Биржа (обязательно при работе без базового бота) */
  exchange?: string;
  /** Quote-валюта (обязательно при работе без базового бота) */
  quoteCurrency?: string;
  /** Walk-forward настройки */
  walkForward: WalkForwardConfig;
  /** Каскадная оптимизация */
  cascade: CascadeConfig;
  /** Генетический алгоритм (общие параметры) */
  genetic: Omit<GeneticConfig, 'generations' | 'populationSize'> & {
    backtestDelaySeconds: number;
  };
  /** Целевая метрика */
  target: OptimizationTarget;
  /** Авто-создание бота из лучшего результата */
  autoCreateBot: boolean;
  /** Мин. robustness score для создания бота */
  minRobustnessScore: number;
  /** Мин. кол-во сделок на каждом test-окне */
  minDealsPerWindow: number;
  /** Депозит для создаваемого бота */
  botDeposit: number;
  /** Плечо для создаваемого бота */
  botLeverage: number;
  /** Тип маржи */
  botMarginType: 'ISOLATED' | 'CROSS';
}

// ═══════════════════════════════════════════════════════════════════
// ПРОГРЕСС
// ═══════════════════════════════════════════════════════════════════

export interface AutoOptimizerProgress {
  phase: AutoOptimizerPhase;
  /** Подробное описание текущего действия */
  phaseLabel: string;
  /** Текущее WF-окно (при walk_forward_test) */
  currentWindow: number;
  totalWindows: number;
  /** Текущее поколение в активной фазе */
  currentGeneration: number;
  totalGenerations: number;
  /** Всего бэктестов запланировано / завершено */
  totalBacktests: number;
  completedBacktests: number;
  /** ETA (мс timestamp) */
  estimatedEndAt: number | null;
  /** Время старта */
  startedAt: number | null;
}

export interface AutoOptimizerLogEntry {
  id: string;
  timestamp: number;
  level: 'info' | 'success' | 'warning' | 'error';
  message: string;
}

/** Callbacks для UI */
export interface AutoOptimizerCallbacks {
  onLog: (level: AutoOptimizerLogEntry['level'], message: string) => void;
  onProgress: (progress: AutoOptimizerProgress) => void;
  onPhaseComplete: (phase: AutoOptimizerPhase, bestGenome: EvaluatedGenome | null) => void;
  onWalkForwardResult: (result: WalkForwardResult) => void;
  onComplete: (result: AutoOptimizerResult) => void;
}
