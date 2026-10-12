/**
 * Типы портфельного оптимизатора («уточнение базового бота»).
 *
 * Идея режима: берём реальный DTO бота из Veles, извлекаем из него числовые параметры
 * (пороги условий, отступы/объёмы сетки, мин. P&L тейка, отступ стопа, подтяжку) и меняем
 * только их, не трогая структуру условий. Каждый кандидат прогоняется бэктестами Veles на
 * корзине монет, результат считается как в «Группах бэктестов» (P&L к риску по МПУ).
 */

import type { BotStrategy } from '../api/backtestRunner';

// ═══════════════════════════════════════════════════════════════════
// ПАРАМЕТРЫ БОТА
// ═══════════════════════════════════════════════════════════════════

/** Путь до числового поля внутри DTO стратегии, например ['settings', 'orders', 3, 'indent']. */
export type ParamPath = Array<string | number>;

export type ParamKind =
  | 'entryValue' // conditions[i].value (условия входа)
  | 'orderConditionValue' // settings.orders[i].conditions[j].value / settings.baseOrder.conditions[j].value
  | 'orderIndent' // settings.baseOrder.indent / settings.orders[i].indent
  | 'orderVolume' // settings.baseOrder.volume / settings.orders[i].volume
  | 'tpMinPnl' // profit.checkPnl
  | 'tpConditionValue' // profit.conditions[i].value
  | 'slIndent' // stopLoss.indent
  | 'slConditionalIndent' // stopLoss.conditionalIndent
  | 'slConditionValue' // stopLoss.conditions[i].value
  | 'pullUp'; // pullUp

export interface ParamSpec {
  /** Стабильный ключ = сериализованный путь, например "settings.orders[3].conditions[1].value". */
  id: string;
  path: ParamPath;
  kind: ParamKind;
  /** Подпись для UI, например "CCI 1 час < 46" или "Ордер 4: отступ". */
  label: string;
  /** Группа для UI: "Вход", "Ордер 1", …, "Тейк", "Стоп", "Сетка". */
  group: string;
  /** Значение в базовом боте. */
  base: number;
  min: number;
  max: number;
  /** Шаг мутации (одно «деление»). */
  step: number;
  /** Количество знаков после запятой при записи в DTO. */
  decimals: number;
  /** Участвует ли параметр в поиске. */
  enabled: boolean;
  /** Порядковый номер ордера в сетке (0 = первый ордер), только для orderIndent / orderVolume. */
  orderIndex?: number;
  /** Индикатор условия (как в DTO), для подписей и шага по умолчанию. */
  indicator?: string | null;
}

/** Значения параметров кандидата: id параметра → число. Содержит ВСЕ параметры (включая выключенные). */
export type ParamValues = Record<string, number>;

// ═══════════════════════════════════════════════════════════════════
// МЕТРИКИ
// ═══════════════════════════════════════════════════════════════════

/** Портфельные метрики корзины бэктестов (как на странице группы бэктестов). */
export interface PortfolioMetrics {
  totalProfit: number;
  /** МПУ: максимум суммы MAE одновременно открытых сделок (как «Макс. МПУ» в группе). */
  maxConcurrentMae: number;
  /** Максимальная просадка накопленного P&L закрытых сделок (как «Макс. просадка» в группе). */
  maxDrawdown: number;
  /** P&L к риску = totalProfit / max(1, max(МПУ, просадка)) — как в группе бэктестов. */
  pnlToRisk: number;
  totalDeals: number;
  profitableDeals: number;
  losingDeals: number;
  openDeals: number;
  /** Сумма MAE сделок, открытых на конец периода. */
  activeMae: number;
  netPerDay: number;
  maxConcurrentPositions: number;
  backtests: number;
}

/** Метрики одного под-периода (фолда), посчитанные по сделкам тех же бэктестов. */
export interface FoldMetrics {
  from: number;
  to: number;
  profit: number;
  risk: number;
  pnlToRisk: number;
  deals: number;
}

export type ObjectiveKind = 'pnlToRisk' | 'robust' | 'profitCapped';

export interface ObjectiveConfig {
  kind: ObjectiveKind;
  /** Количество под-периодов для 'robust' (2–6). */
  folds: number;
  /** Ограничение МПУ в валюте котировки для 'profitCapped' (null — без ограничения). */
  maxMaeCap: number | null;
  /** Минимум сделок на корзине; меньше — штраф score × deals / minDeals. */
  minDeals: number;
}

// ═══════════════════════════════════════════════════════════════════
// КАНДИДАТЫ
// ═══════════════════════════════════════════════════════════════════

export type EvaluationStage = 'screen' | 'full';
export type EvaluationStatus = 'pending' | 'running' | 'done' | 'failed';

export interface CandidateEvaluation {
  stage: EvaluationStage;
  symbols: string[];
  /** symbol → id бэктеста Veles. */
  backtestIds: Record<string, number>;
  failedSymbols: string[];
  status: EvaluationStatus;
  metrics: PortfolioMetrics | null;
  folds: FoldMetrics[];
  score: number | null;
  error: string | null;
}

export interface Candidate {
  id: string;
  /** Хеш значений параметров (дедупликация и кеш результатов). */
  hash: string;
  values: ParamValues;
  parentId: string | null;
  generation: number;
  /** id параметров, отличающихся от базового бота. */
  changed: string[];
  screen: CandidateEvaluation | null;
  full: CandidateEvaluation | null;
  createdAt: number;
}

// ═══════════════════════════════════════════════════════════════════
// КОНФИГУРАЦИЯ
// ═══════════════════════════════════════════════════════════════════

export interface BacktestRunSettings {
  /** ISO, например 2021-12-06T00:00:00.000Z */
  periodFrom: string;
  /** ISO, например 2026-10-12T23:59:59.999Z */
  periodTo: string;
  /** Комиссии в процентах, как в модалке бэктестов (0.02 / 0.055). */
  makerCommission: number;
  takerCommission: number;
  includeWicks: boolean;
}

export interface SchedulerConfig {
  /** Максимум одновременно выполняющихся бэктестов (потоки тарифа Veles). */
  maxInFlight: number;
  /** Минимальная пауза между созданием бэктестов, сек. */
  submitDelaySeconds: number;
  /** Интервал опроса статуса, сек. */
  pollIntervalSeconds: number;
  /** Таймаут одного бэктеста, мин. */
  backtestTimeoutMinutes: number;
}

export interface SearchConfig {
  generations: number;
  /** Кандидатов на отборочной корзине за поколение. */
  candidatesPerGeneration: number;
  /** Сколько лучших по отбору проверять на полной корзине. */
  promotePerGeneration: number;
  /** Сколько параметров меняется у кандидата (от и до). */
  mutationsMin: number;
  mutationsMax: number;
  /** Кандидат идёт на полную корзину, если score_отбор ≥ лучший_отбор × (1 − tolerance). */
  screenTolerance: number;
  /** Новый лучший принимается, если score_полный > лучший × (1 + minImprovement). */
  minImprovement: number;
  /** Лимит созданных бэктестов на весь запуск (кеш не считается). */
  maxBacktests: number;
  seed: number;
}

export interface PortfolioOptimizerConfig {
  botId: number;
  botName: string;
  /** Полная корзина монет (тикеры без котировки, например "ADA"). */
  basket: string[];
  /** Отборочная корзина — подмножество полной. */
  screenBasket: string[];
  run: BacktestRunSettings;
  /** Блокировка по ботам для агрегации (maxConcurrentPositions), как в группе бэктестов. */
  block: number;
  objective: ObjectiveConfig;
  search: SearchConfig;
  scheduler: SchedulerConfig;
  params: ParamSpec[];
  /** Группа бэктестов с готовыми результатами базового бота (экономит бэктесты базы). */
  baseGroupId: string | null;
}

// ═══════════════════════════════════════════════════════════════════
// СОСТОЯНИЕ ЗАПУСКА
// ═══════════════════════════════════════════════════════════════════

export type PortfolioRunStatus = 'idle' | 'running' | 'paused' | 'stopped' | 'completed' | 'error';

export type ScheduledJobState = 'queued' | 'submitted' | 'done' | 'failed';

/** Задача планировщика: один бэктест Veles. */
export interface ScheduledJob {
  /** Уникальный ключ: `${candidateHash}|${symbol}` (период и комиссии общие на запуск). */
  key: string;
  symbol: string;
  backtestId: number | null;
  state: ScheduledJobState;
  attempts: number;
  submittedAt: number | null;
  error: string | null;
}

export interface PortfolioLogEntry {
  timestamp: number;
  level: 'info' | 'success' | 'warning' | 'error';
  message: string;
}

export interface PortfolioOptimizerProgress {
  status: PortfolioRunStatus;
  generation: number;
  totalGenerations: number;
  backtestsSubmitted: number;
  backtestsLimit: number;
  inFlight: number;
  queued: number;
  candidatesEvaluated: number;
  startedAt: number | null;
  /** Оценка окончания по средней скорости, мс epoch. */
  estimatedEndAt: number | null;
}

export interface PortfolioOptimizerState {
  version: 1;
  runId: string;
  config: PortfolioOptimizerConfig;
  baseStrategy: BotStrategy;
  quoteCurrency: string;
  baseCandidateId: string;
  bestCandidateId: string | null;
  candidates: Candidate[];
  /** Кеш: `${hash}|${symbol}` → id готового бэктеста. */
  backtestCache: Record<string, number>;
  /** Задачи, созданные в Veles и ещё не завершённые (для продолжения после перезагрузки). */
  inFlightJobs: ScheduledJob[];
  /** Статистика успеха изменений по параметрам: id → {tries, wins}. */
  paramStats: Record<string, { tries: number; wins: number }>;
  generation: number;
  backtestsSubmitted: number;
  status: PortfolioRunStatus;
  startedAt: number;
  updatedAt: number;
  logs: PortfolioLogEntry[];
}

export interface PortfolioOptimizerCallbacks {
  onLog: (entry: PortfolioLogEntry) => void;
  onProgress: (progress: PortfolioOptimizerProgress) => void;
  /** Вызывается после каждого изменения кандидатов (оценка, новый лучший). */
  onCandidatesChanged: (candidates: Candidate[], bestCandidateId: string | null) => void;
}
