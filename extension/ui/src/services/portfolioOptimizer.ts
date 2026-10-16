/**
 * Оркестратор портфельного оптимизатора («уточнение базового бота»).
 *
 * Схема запуска:
 * 1. База: бэктесты базового бота на полной корзине (или готовые результаты из группы бэктестов);
 *    отборочная оценка базы — подмножество тех же бэктестов, без новых запусков.
 * 2. Поколения: мутации лучших найденных настроек → дешёвая отборочная корзина → лучшие по отбору
 *    проверяются на полной корзине (бэктесты отборочной корзины переиспользуются) → новый лучший
 *    принимается только по полной корзине.
 * 3. Корзина оценивается только целиком: если хоть одна монета не посчиталась, у этапа нет score.
 * 4. Сбои Veles и сети (исчерпаны повторы создания бэктеста, таймаут бэктеста, не загрузились результаты
 *    готовой корзины) не считаются неудачей кандидата: запуск останавливается так, чтобы его можно было
 *    продолжить, а незавершённые этапы пересчитываются (уже готовые бэктесты берутся из кеша).
 *
 * Всё состояние (кандидаты, кеш бэктестов, созданные и ещё не завершённые бэктесты, статистика
 * параметров, журнал) сохраняется после каждого изменения, поэтому запуск можно продолжить после
 * перезагрузки страницы: незавершённые бэктесты подключаются заново, незаконченные этапы
 * пересчитываются с использованием кеша, лимит бэктестов учитывает только реально созданные.
 *
 * Все внешние зависимости (планировщик, оценка корзины, загрузка группы, сохранение, время)
 * внедряются, поэтому оркестратор тестируется без сети и таймеров.
 */

import type { BotStrategy } from '../api/backtestRunner';
import { buildBacktestPayload, composeSymbol } from '../api/backtestRunner';
import type { BacktestConfigDto } from '../api/backtests.dtos';
import {
  baseValues,
  describeParamChange,
  formatParamValue,
  getAtPath,
  paramsHash,
  patchStrategy,
  roundTo,
} from '../lib/botParams';
import { MS_IN_DAY, toTimestamp } from '../lib/dateTime';
import {
  computeObjective,
  createRng,
  hasEnoughDeals,
  isImprovement,
  proposeCandidates,
  scaleObjective,
  screenStageScale,
  selectForPromotion,
  updateParamStats,
} from '../lib/portfolioSearch';
import { readStorageValue, removeStorageValue, writeStorageValue } from '../lib/safeStorage';
import { readBacktestGroups } from '../storage/backtestGroupsStore';
import type {
  BacktestRunSettings,
  Candidate,
  CandidateEvaluation,
  EvaluationStage,
  ExcludedSymbol,
  ObjectiveConfig,
  ObjectiveKind,
  ParamSpec,
  PortfolioLogEntry,
  PortfolioMetrics,
  PortfolioOptimizerCallbacks,
  PortfolioOptimizerConfig,
  PortfolioOptimizerProgress,
  PortfolioOptimizerState,
  PortfolioRunStatus,
  ScheduledJob,
  SchedulerConfig,
  SearchConfig,
} from '../types/portfolioOptimizer';
import {
  abortableSleep,
  BacktestScheduler,
  isAbortError,
  isPairRejectedError,
  isAuthError as isSchedulerAuthError,
  isTransientBacktestError,
  mergeSchedulerConfig,
  SchedulerAbortError,
  type SchedulerEvents,
  type SchedulerJobRequest,
} from './backtestScheduler';
import { backtestsService } from './backtests';
import { loadListingDates, pairCodeFor } from './pairListing';
import { type BasketEvaluation, type BasketPeriod, evaluateBasket } from './portfolioScore';

// ═══════════════════════════════════════════════════════════════════
// КОНСТАНТЫ И НАСТРОЙКИ ПО УМОЛЧАНИЮ
// ═══════════════════════════════════════════════════════════════════

/** Ключ localStorage с состоянием запуска. */
export const PORTFOLIO_OPTIMIZER_STATE_KEY = 'veles_portfolio_optimizer_state';

/** Сколько последних записей журнала хранится в состоянии. */
export const MAX_PORTFOLIO_LOG_ENTRIES = 300;

/** Размер отборочной корзины, если она не задана. */
export const DEFAULT_SCREEN_BASKET_SIZE = 10;

/** Максимальная длина названия бэктеста. */
export const MAX_BACKTEST_NAME_LENGTH = 80;

/** Допустимое расхождение периода бэктестов группы с периодом запуска (часовые пояса, «конец дня»). */
export const GROUP_PERIOD_TOLERANCE_MS = 2 * MS_IN_DAY;

export const DEFAULT_SEARCH_CONFIG: SearchConfig = {
  generations: 8,
  candidatesPerGeneration: 6,
  promotePerGeneration: 2,
  mutationsMin: 1,
  mutationsMax: 3,
  screenTolerance: 0.05,
  minImprovement: 0.01,
  maxBacktests: 1500,
  seed: 1,
};

export const DEFAULT_SCHEDULER_CONFIG: SchedulerConfig = {
  maxInFlight: 3,
  submitDelaySeconds: 31,
  pollIntervalSeconds: 10,
  backtestTimeoutMinutes: 45,
};

export const DEFAULT_OBJECTIVE: ObjectiveConfig = {
  kind: 'robust',
  folds: 3,
  maxMaeCap: null,
  minDeals: 200,
};

/** Попыток посчитать метрики корзины (загрузка результатов может упасть из-за сети). */
const EVALUATION_MAX_ATTEMPTS = 4;
const EVALUATION_RETRY_DELAYS_MS = [5_000, 30_000, 120_000];
/**
 * После скольких сеансов подряд с неудачным расчётом метрик готовой корзины этап считается неудачным
 * (в первый раз запуск останавливается с возможностью продолжить — обычно это сбой сети или Veles).
 */
const METRICS_MAX_FAILED_SESSIONS = 2;

/**
 * Наибольшая доля корзины, которую можно исключить из-за отказа Veles принять настройки бота для пары
 * (HTTP 412). Если отказов больше, дело не в отдельных монетах (депозит, объёмы ордеров) — запуск останавливается.
 */
const MAX_EXCLUDED_BASKET_SHARE = 0.5;

/** Ожидание ответа на POST, отправленный до остановки (чтобы не потерять id созданного бэктеста). */
const DRAIN_POLL_INTERVAL_MS = 250;
const DRAIN_MAX_POLLS = 120;

const MAX_LISTED_ITEMS = 5;
const MAX_DESCRIBED_CHANGES = 8;

/** Метка базового кандидата в названиях бэктестов (ASCII) и в журнале. */
const BASE_BACKTEST_LABEL = 'base';
const BASE_LOG_LABEL = 'база';
const BACKTEST_NAME_PREFIX = 'PO';

const OBJECTIVE_KINDS: ReadonlySet<string> = new Set<ObjectiveKind>(['pnlToRisk', 'robust', 'profitCapped']);
const RUN_STATUSES: ReadonlySet<string> = new Set<PortfolioRunStatus>([
  'idle',
  'running',
  'paused',
  'stopped',
  'completed',
  'error',
]);

const STAGE_TITLES: Readonly<Record<EvaluationStage, string>> = {
  screen: 'отбор',
  full: 'полная корзина',
};

// ═══════════════════════════════════════════════════════════════════
// ПУБЛИЧНЫЕ ТИПЫ
// ═══════════════════════════════════════════════════════════════════

/** Планировщик бэктестов, которым пользуется оркестратор (BacktestScheduler или тестовая подмена). */
export interface PortfolioScheduler {
  run(job: SchedulerJobRequest, signal: AbortSignal): Promise<number>;
  /** submittedAt — когда бэктест был создан (таймаут отсчитывается от него). */
  attach(key: string, backtestId: number, signal: AbortSignal, submittedAt?: number | null): Promise<number>;
  setPaused(paused: boolean): void;
  updateConfig(config: Partial<SchedulerConfig>): void;
  readonly inFlight: number;
  readonly queued: number;
}

/**
 * Фабрика планировщика: события (созданный бэктест, журнал) принадлежат оркестратору,
 * поэтому планировщик создаётся им самим на каждый сеанс запуска.
 */
export type PortfolioSchedulerFactory = (config: SchedulerConfig, events: SchedulerEvents) => PortfolioScheduler;

/** Оценка корзины бэктестов (evaluateBasket из portfolioScore или тестовая подмена). */
export type PortfolioBasketEvaluator = (
  backtestIds: number[],
  block: number,
  run: BasketPeriod,
  folds: number,
) => Promise<Pick<BasketEvaluation, 'metrics' | 'folds'>>;

/** Комиссии бэктеста группы, % (null — неизвестно). */
export interface GroupBacktestCommissions {
  maker: number | null;
  taker: number | null;
}

/** Депозит бэктеста группы (null — неизвестно). */
export interface GroupBacktestDeposit {
  amount: number | null;
  leverage: number | null;
}

/** Настройки стратегии бэктеста группы — те же пути, что у параметров базового бота (ParamSpec.path). */
export type GroupBacktestStrategy = Pick<
  BacktestConfigDto,
  'conditions' | 'settings' | 'profit' | 'stopLoss' | 'pullUp'
>;

/** Краткие сведения о бэктесте группы, нужные для подстановки результатов базы. */
export interface GroupBacktestSummary {
  backtestId: number;
  /** Базовая монета бэктеста (statistics.base), например "ADA". */
  base: string;
  quote: string | null;
  from: string | null;
  to: string | null;
  useWicks: boolean | null;
  algorithm: string | null;
  /** Необязательные сведения ниже: отсутствующие не проверяются. */
  commissions?: GroupBacktestCommissions | null;
  deposit?: GroupBacktestDeposit | null;
  exchange?: string | null;
  strategy?: GroupBacktestStrategy | null;
}

export interface GroupBacktestsLoadResult {
  groupName: string;
  backtests: GroupBacktestSummary[];
  /** Бэктесты группы, которые не удалось загрузить. */
  failed: Array<{ backtestId: number; error: string }>;
}

export interface PortfolioOptimizerDeps {
  createScheduler: PortfolioSchedulerFactory;
  evaluateBasket: PortfolioBasketEvaluator;
  /** Бэктесты группы с готовыми результатами базового бота. */
  loadGroupBacktests: (groupId: string) => Promise<GroupBacktestsLoadResult>;
  /** Сохранение состояния (сериализовать сразу: объект продолжает меняться). false — не сохранилось. */
  saveState: (state: PortfolioOptimizerState) => boolean;
  /** Сохранённое состояние (для бэктеста, созданного уже после окончания сеанса). */
  loadSavedState: () => PortfolioOptimizerState | null;
  /** Удаление сохранённого состояния: устаревшую копию нельзя предлагать к продолжению. */
  clearSavedState: () => void;
  /** Даты листинга пар биржи: код пары («AAVEUSDT») → мс epoch. */
  loadListingDates: (exchange: string) => Promise<Record<string, number>>;
  sleep: (ms: number, signal?: AbortSignal) => Promise<void>;
  now: () => number;
}

export interface PortfolioOptimizerInput {
  config: PortfolioOptimizerConfig;
  baseStrategy: BotStrategy;
  quoteCurrency: string;
}

export interface BacktestsEstimate {
  /** Максимум новых бэктестов за поколение: отбор всех кандидатов + досчёт полной корзины для продвинутых. */
  perGenerationMax: number;
  /** Бэктестов базы (0 — результаты берутся из группы; недостающие монеты добавятся при запуске). */
  baseBacktests: number;
  /** Максимум бэктестов на весь запуск с учётом лимита. */
  totalMax: number;
  /** Часов при интервале между созданием бэктестов (нижняя оценка длительности). */
  hoursAtDelay: number;
}

export interface GroupBaseMatchInput {
  backtests: GroupBacktestSummary[];
  basket: string[];
  quoteCurrency: string;
  run: BacktestRunSettings;
  /** Алгоритм базового бота (LONG/SHORT) для проверки. */
  algorithm?: string | null;
  /** Базовый бот: биржа и депозит бэктестов группы должны совпадать с ним. */
  baseStrategy?: Pick<BotStrategy, 'exchange' | 'pair' | 'deposit'> | null;
  /** Параметры базового бота: их значения в бэктестах группы должны совпадать с базовыми. */
  params?: ParamSpec[];
  /** Даты листинга (код пары → мс): бэктест монеты, листинг которой позже начала периода, начинается с листинга. */
  listingDates?: Record<string, number>;
  toleranceMs?: number;
}

export interface GroupBaseMatch {
  /** Тикер корзины → id бэктеста группы (в порядке корзины). */
  backtestIds: Record<string, number>;
  /** Тикеры корзины, для которых в группе нет подходящего бэктеста. */
  missing: string[];
  /** Бэктестов группы с монетами вне корзины. */
  outsideBasket: number;
  warnings: string[];
}

// ═══════════════════════════════════════════════════════════════════
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
// ═══════════════════════════════════════════════════════════════════

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null;

const isFiniteNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

const isPositiveInteger = (value: unknown): value is number =>
  typeof value === 'number' && Number.isInteger(value) && value > 0;

const isStringArray = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every((item) => typeof item === 'string');

const describeError = (error: unknown): string => (error instanceof Error ? error.message : String(error));

const cloneDeep = <T>(value: T): T => {
  if (typeof structuredClone === 'function') {
    try {
      return structuredClone(value);
    } catch {
      // значение не поддерживает structuredClone — используем JSON
    }
  }
  return JSON.parse(JSON.stringify(value)) as T;
};

const normalizeTicker = (value: string): string => value.trim().toUpperCase();

/** Тикеры без пустых значений и повторов, в верхнем регистре, порядок сохраняется. */
const uniqueTickers = (values: readonly string[]): string[] => {
  const result: string[] = [];
  const seen = new Set<string>();
  for (const value of values) {
    if (typeof value !== 'string') {
      continue;
    }
    const ticker = normalizeTicker(value);
    if (ticker.length > 0 && !seen.has(ticker)) {
      seen.add(ticker);
      result.push(ticker);
    }
  }
  return result;
};

const formatList = (items: readonly string[], limit = MAX_LISTED_ITEMS, separator = ', '): string => {
  if (items.length <= limit) {
    return items.join(separator);
  }
  return `${items.slice(0, limit).join(separator)} и ещё ${items.length - limit}`;
};

const formatScore = (score: number | null | undefined): string => (isFiniteNumber(score) ? score.toFixed(3) : '—');

const formatAmount = (value: number): string => value.toFixed(2);

const formatDate = (timestamp: number): string => new Date(timestamp).toISOString().slice(0, 10);

const formatSeconds = (ms: number): string => String(Math.round(ms / 100) / 10);

const jobKey = (hash: string, symbol: string): string => `${hash}|${symbol}`;

const symbolFromKey = (key: string): string => {
  const separator = key.lastIndexOf('|');
  return separator >= 0 ? key.slice(separator + 1) : key;
};

/** Отдельный поток случайных чисел на каждое поколение — воспроизводимо и при продолжении запуска. */
const generationSeed = (seed: number, generation: number): number =>
  ((seed >>> 0) ^ Math.imul(generation, 0x9e3779b1)) >>> 0;

const intAtLeast = (value: number | undefined, fallback: number, min: number): number =>
  Math.max(min, Math.floor(isFiniteNumber(value) ? value : fallback));

const numberAtLeast = (value: number | undefined, fallback: number, min: number): number =>
  Math.max(min, isFiniteNumber(value) ? value : fallback);

const createEvaluation = (stage: EvaluationStage, symbols: string[]): CandidateEvaluation => ({
  stage,
  symbols: [...symbols],
  backtestIds: {},
  failedSymbols: [],
  status: 'pending',
  metrics: null,
  folds: [],
  score: null,
  error: null,
});

const isStageDone = (evaluation: CandidateEvaluation | null): boolean => evaluation?.status === 'done';

const isStageFinished = (evaluation: CandidateEvaluation | null): boolean =>
  evaluation?.status === 'done' || evaluation?.status === 'failed';

/** Score этапа, если он посчитан. */
const stageScore = (evaluation: CandidateEvaluation | null): number | null =>
  evaluation?.status === 'done' && isFiniteNumber(evaluation.score) ? evaluation.score : null;

const describeEvaluationResult = (evaluation: CandidateEvaluation): string => {
  const metrics = evaluation.metrics;
  if (!metrics) {
    return `score ${formatScore(evaluation.score)}`;
  }
  return (
    `score ${formatScore(evaluation.score)} · P&L ${formatAmount(metrics.totalProfit)} · ` +
    `МПУ ${formatAmount(metrics.maxConcurrentMae)} · P&L/риск ${formatAmount(metrics.pnlToRisk)} · ` +
    `сделок ${metrics.totalDeals}`
  );
};

/** Ошибка авторизации Veles (401/403): продолжать бессмысленно, пока пользователь не войдёт заново. */
export const isAuthError = (error: unknown): boolean => isSchedulerAuthError(error);

const authErrorMessage = (error: unknown): string =>
  `Нет доступа к Veles: ${describeError(error)}. Войдите в аккаунт Veles и продолжите запуск — ` +
  'созданные бэктесты сохранены.';

/** Сбой Veles или сети: кандидат ни при чём, запуск останавливается с возможностью продолжить. */
const infrastructureErrorMessage = (detail: string): string =>
  `Veles или сеть недоступны: ${detail}. Запуск остановлен, чтобы не списывать кандидатов из-за сбоя: ` +
  'созданные бэктесты сохранены, незавершённые этапы будут пересчитаны — продолжите запуск, когда Veles снова доступен.';

/**
 * Стратегия с депозитом для бэктестов: amount заменяется на backtestDeposit (null — как в боте).
 * Вход не изменяется.
 */
export const withBacktestDeposit = <T extends Pick<BotStrategy, 'deposit'>>(
  strategy: T,
  backtestDeposit: number | null | undefined,
): T =>
  strategy.deposit && isFiniteNumber(backtestDeposit) && backtestDeposit > 0
    ? { ...strategy, deposit: { ...strategy.deposit, amount: backtestDeposit } }
    : strategy;

/**
 * Payload бэктеста без API-ключа бота: бэктест — симуляция, ключ биржи ему не нужен (Veles Tools 2.15 тоже
 * убирает его из бэктестов оптимизатора). С ключом Veles может проверять объём ордера по требованиям биржи
 * к паре и отказывать (HTTP 412 «Объём ордера не соответствует требованиям по паре»).
 */
export const withoutApiKey = (payload: BotStrategy): BotStrategy => {
  if ('apiKey' in payload) {
    const copy: BotStrategy = { ...payload };
    Reflect.deleteProperty(copy, 'apiKey');
    return copy;
  }
  return payload;
};

/** Метрики посчитанного этапа (null — этап не посчитан). */
const doneMetrics = (evaluation: CandidateEvaluation | null): PortfolioMetrics | null =>
  evaluation?.status === 'done' ? evaluation.metrics : null;

// ═══════════════════════════════════════════════════════════════════
// КОНФИГУРАЦИЯ
// ═══════════════════════════════════════════════════════════════════

/** Настройки поиска с подстановкой значений по умолчанию и приведением к допустимым диапазонам. */
export const normalizeSearchConfig = (search: Partial<SearchConfig>): SearchConfig => {
  const mutationsMin = intAtLeast(search.mutationsMin, DEFAULT_SEARCH_CONFIG.mutationsMin, 1);
  return {
    generations: intAtLeast(search.generations, DEFAULT_SEARCH_CONFIG.generations, 0),
    candidatesPerGeneration: intAtLeast(
      search.candidatesPerGeneration,
      DEFAULT_SEARCH_CONFIG.candidatesPerGeneration,
      1,
    ),
    promotePerGeneration: intAtLeast(search.promotePerGeneration, DEFAULT_SEARCH_CONFIG.promotePerGeneration, 0),
    mutationsMin,
    mutationsMax: intAtLeast(search.mutationsMax, DEFAULT_SEARCH_CONFIG.mutationsMax, mutationsMin),
    screenTolerance: numberAtLeast(search.screenTolerance, DEFAULT_SEARCH_CONFIG.screenTolerance, 0),
    minImprovement: numberAtLeast(search.minImprovement, DEFAULT_SEARCH_CONFIG.minImprovement, 0),
    maxBacktests: intAtLeast(search.maxBacktests, DEFAULT_SEARCH_CONFIG.maxBacktests, 0),
    seed: Math.floor(isFiniteNumber(search.seed) ? search.seed : DEFAULT_SEARCH_CONFIG.seed),
  };
};

const normalizeObjective = (objective: Partial<ObjectiveConfig>): ObjectiveConfig => {
  const kind = objective.kind !== undefined && OBJECTIVE_KINDS.has(objective.kind) ? objective.kind : undefined;
  return {
    kind: kind ?? DEFAULT_OBJECTIVE.kind,
    folds: intAtLeast(objective.folds, DEFAULT_OBJECTIVE.folds, 0),
    maxMaeCap: isFiniteNumber(objective.maxMaeCap) && objective.maxMaeCap >= 0 ? objective.maxMaeCap : null,
    minDeals: numberAtLeast(objective.minDeals, DEFAULT_OBJECTIVE.minDeals, 0),
  };
};

/**
 * Детерминированная «равномерная» выборка отборочной корзины: корзина делится на `size` равных
 * подряд идущих частей, из каждой берётся одна монета (позиция внутри части — по seed).
 * Тикеры приводятся к верхнему регистру, повторы убираются, порядок корзины сохраняется.
 */
export const pickScreenBasket = (basket: string[], size: number, seed: number): string[] => {
  const unique = uniqueTickers(basket);
  const count = isFiniteNumber(size) ? Math.min(unique.length, Math.max(0, Math.floor(size))) : 0;
  if (count >= unique.length) {
    return unique;
  }
  if (count === 0) {
    return [];
  }
  const rng = createRng(seed);
  const picked: string[] = [];
  for (let stratum = 0; stratum < count; stratum += 1) {
    const start = Math.floor((stratum * unique.length) / count);
    const end = Math.floor(((stratum + 1) * unique.length) / count);
    const width = end - start;
    picked.push(unique[start + Math.min(width - 1, Math.floor(rng() * width))]);
  }
  return picked;
};

/**
 * Проверяет и нормализует конфигурацию запуска: тикеры в верхнем регистре без повторов,
 * отборочная корзина — подмножество полной (пустая → pickScreenBasket), числовые настройки
 * в допустимых диапазонах. Бросает ошибку с русским текстом, если запускать нечего.
 */
export const normalizePortfolioConfig = (config: PortfolioOptimizerConfig): PortfolioOptimizerConfig => {
  const basket = uniqueTickers(config.basket);
  if (basket.length === 0) {
    throw new Error('Корзина монет пуста — добавьте хотя бы одну монету.');
  }
  if (config.params.length === 0) {
    throw new Error('У базового бота не найдено числовых параметров для оптимизации.');
  }
  const fromMs = toTimestamp(config.run.periodFrom);
  const toMs = toTimestamp(config.run.periodTo);
  if (fromMs === null || toMs === null || toMs <= fromMs) {
    throw new Error('Некорректный период бэктестов: начало должно быть раньше конца.');
  }

  const search = normalizeSearchConfig(config.search);
  const inBasket = new Set(basket);
  const screen = uniqueTickers(config.screenBasket).filter((ticker) => inBasket.has(ticker));
  const baseGroupId = typeof config.baseGroupId === 'string' ? config.baseGroupId.trim() : '';

  return {
    ...config,
    botName: config.botName.trim(),
    basket,
    screenBasket: screen.length > 0 ? screen : pickScreenBasket(basket, DEFAULT_SCREEN_BASKET_SIZE, search.seed),
    run: { ...config.run },
    block: isFiniteNumber(config.block) ? Math.max(1, Math.floor(config.block)) : 1,
    objective: normalizeObjective(config.objective),
    search,
    scheduler: mergeSchedulerConfig(DEFAULT_SCHEDULER_CONFIG, config.scheduler),
    params: cloneDeep(config.params),
    baseGroupId: baseGroupId.length > 0 ? baseGroupId : null,
    backtestDeposit:
      isFiniteNumber(config.backtestDeposit) && config.backtestDeposit > 0 ? roundTo(config.backtestDeposit, 2) : null,
  };
};

/**
 * Оценка объёма запуска: максимум бэктестов за поколение (отбор всех кандидатов + досчёт полной
 * корзины для продвинутых), на весь запуск (с лимитом) и длительность при интервале создания.
 */
export const estimateBacktests = (config: PortfolioOptimizerConfig): BacktestsEstimate => {
  const basket = uniqueTickers(config.basket);
  const inBasket = new Set(basket);
  const screen = uniqueTickers(config.screenBasket).filter((ticker) => inBasket.has(ticker));
  const screenSize = screen.length > 0 ? screen.length : Math.min(DEFAULT_SCREEN_BASKET_SIZE, basket.length);
  const search = normalizeSearchConfig(config.search);
  const promoted = Math.min(search.promotePerGeneration, search.candidatesPerGeneration);
  const perGenerationMax =
    search.candidatesPerGeneration * screenSize + promoted * Math.max(0, basket.length - screenSize);
  const baseBacktests = config.baseGroupId ? 0 : basket.length;
  const totalMax = Math.min(search.maxBacktests, baseBacktests + search.generations * perGenerationMax);
  const delaySeconds = numberAtLeast(
    config.scheduler?.submitDelaySeconds,
    DEFAULT_SCHEDULER_CONFIG.submitDelaySeconds,
    0,
  );
  return {
    perGenerationMax,
    baseBacktests,
    totalMax,
    hoursAtDelay: (totalMax * delaySeconds) / 3600,
  };
};

// ═══════════════════════════════════════════════════════════════════
// ГОТОВЫЕ РЕЗУЛЬТАТЫ БАЗЫ ИЗ ГРУППЫ БЭКТЕСТОВ
// ═══════════════════════════════════════════════════════════════════

const nonEmptyString = (value: unknown): string | null =>
  typeof value === 'string' && value.trim().length > 0 ? value.trim() : null;

/** Базовая монета бэктеста: statistics.base, иначе часть символа до «/». */
const resolveBaseTicker = (base: unknown, symbol: unknown): string => {
  const direct = nonEmptyString(base);
  if (direct) {
    return direct;
  }
  const pair = nonEmptyString(symbol);
  return pair ? (nonEmptyString(pair.split('/')[0]) ?? '') : '';
};

/** Число из DTO: конечное число или числовая строка ("0.02"), иначе null. */
const readNumeric = (value: unknown): number | null => {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : null;
  }
  if (typeof value === 'string' && value.trim().length > 0) {
    const parsed = Number(value.trim());
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
};

/** Сравнение чисел так, как они уходят в Veles (комиссии — 6 знаков, как в buildBacktestPayload). */
const sameRounded = (left: number, right: number, decimals = 6): boolean =>
  roundTo(left, decimals) === roundTo(right, decimals);

type GroupMismatchKind = 'commissions' | 'deposit' | 'exchange' | 'strategy';

const GROUP_MISMATCH_KINDS: readonly GroupMismatchKind[] = ['commissions', 'deposit', 'exchange', 'strategy'];

interface GroupMismatch {
  kind: GroupMismatchKind;
  detail: string;
}

/**
 * Почему бэктест группы нельзя взять результатом базы: другие комиссии, депозит/плечо, биржа или
 * значения параметров бота. null — совместим (неизвестные сведения не проверяются).
 */
const findGroupMismatch = (summary: GroupBacktestSummary, input: GroupBaseMatchInput): GroupMismatch | null => {
  const { maker, taker } = summary.commissions ?? { maker: null, taker: null };
  if (
    (maker !== null && !sameRounded(maker, input.run.makerCommission)) ||
    (taker !== null && !sameRounded(taker, input.run.takerCommission))
  ) {
    return { kind: 'commissions', detail: `${maker ?? '—'}/${taker ?? '—'}%` };
  }

  const baseDeposit = input.baseStrategy?.deposit ?? null;
  const deposit = summary.deposit ?? null;
  if (baseDeposit && deposit) {
    const amountDiffers =
      deposit.amount !== null && isFiniteNumber(baseDeposit.amount) && !sameRounded(deposit.amount, baseDeposit.amount);
    const leverageDiffers =
      deposit.leverage !== null &&
      isFiniteNumber(baseDeposit.leverage) &&
      !sameRounded(deposit.leverage, baseDeposit.leverage);
    if (amountDiffers || leverageDiffers) {
      return { kind: 'deposit', detail: `${deposit.amount ?? '—'} × ${deposit.leverage ?? '—'}` };
    }
  }

  const baseExchange =
    nonEmptyString(input.baseStrategy?.exchange)?.toUpperCase() ??
    nonEmptyString(input.baseStrategy?.pair?.exchange)?.toUpperCase() ??
    null;
  const exchange = nonEmptyString(summary.exchange)?.toUpperCase() ?? null;
  if (baseExchange && exchange && exchange !== baseExchange) {
    return { kind: 'exchange', detail: exchange };
  }

  if (summary.strategy && input.params) {
    for (const param of input.params) {
      const value = readNumeric(getAtPath(summary.strategy, param.path));
      if (value === null || roundTo(value, param.decimals) !== roundTo(param.base, param.decimals)) {
        const shown = value === null ? 'нет' : formatParamValue(value, param.decimals);
        return {
          kind: 'strategy',
          detail: `${param.label}: ${shown} вместо ${formatParamValue(param.base, param.decimals)}`,
        };
      }
    }
  }
  return null;
};

const GROUP_MISMATCH_WARNINGS: Readonly<
  Record<GroupMismatchKind, (list: string, input: GroupBaseMatchInput) => string>
> = {
  commissions: (list, input) =>
    `Комиссии в бэктестах группы отличаются от запуска (${input.run.makerCommission}/${input.run.takerCommission}%): ` +
    `${list}. Эти монеты будут посчитаны бэктестами заново.`,
  deposit: (list) =>
    `Депозит или плечо в бэктестах группы отличаются от базового бота: ${list}. ` +
    'Эти монеты будут посчитаны бэктестами заново.',
  exchange: (list) =>
    `Биржа бэктестов группы отличается от базового бота: ${list}. Эти монеты будут посчитаны бэктестами заново.`,
  strategy: (list) =>
    `Настройки бота в бэктестах группы отличаются от текущих (группа посчитана для другой версии бота?): ${list}. ` +
    'Эти монеты будут посчитаны бэктестами заново.',
};

/**
 * Сопоставляет бэктесты группы с корзиной по базовой монете. Бэктесты с другой котировкой
 * пропускаются; при нескольких бэктестах одной монеты берётся самый новый совместимый (больший id).
 * Несовместимые бэктесты — другие комиссии, депозит/плечо, биржа или значения параметров бота — не
 * берутся: такие монеты считаются недостающими и будут посчитаны бэктестами (с предупреждением).
 * Период, учёт теней и алгоритм проверяются — расхождения попадают в предупреждения,
 * но результаты всё равно используются (решение за пользователем).
 */
export const matchGroupBacktestsToBasket = (input: GroupBaseMatchInput): GroupBaseMatch => {
  const basket = uniqueTickers(input.basket);
  const inBasket = new Set(basket);
  const quote = normalizeTicker(input.quoteCurrency);
  const byTicker = new Map<string, GroupBacktestSummary[]>();
  const wrongQuote: string[] = [];
  let outsideBasket = 0;

  for (const summary of input.backtests) {
    const ticker = normalizeTicker(summary.base);
    if (!inBasket.has(ticker)) {
      outsideBasket += 1;
      continue;
    }
    const summaryQuote = summary.quote ? normalizeTicker(summary.quote) : null;
    if (summaryQuote && summaryQuote !== quote) {
      wrongQuote.push(`${ticker}/${summaryQuote}`);
      continue;
    }
    const options = byTicker.get(ticker) ?? [];
    options.push(summary);
    byTicker.set(ticker, options);
  }

  const tolerance = isFiniteNumber(input.toleranceMs) ? Math.max(0, input.toleranceMs) : GROUP_PERIOD_TOLERANCE_MS;
  const runFrom = toTimestamp(input.run.periodFrom);
  const runTo = toTimestamp(input.run.periodTo);
  const algorithm = nonEmptyString(input.algorithm)?.toUpperCase() ?? null;

  const backtestIds: Record<string, number> = {};
  const duplicates: string[] = [];
  const periodMismatch: string[] = [];
  const wicksMismatch: string[] = [];
  const algorithmMismatch: string[] = [];
  const excluded: Record<GroupMismatchKind, string[]> = { commissions: [], deposit: [], exchange: [], strategy: [] };

  for (const ticker of basket) {
    const options = [...(byTicker.get(ticker) ?? [])].sort((left, right) => right.backtestId - left.backtestId);
    if (options.length === 0) {
      continue;
    }
    if (options.length > 1) {
      duplicates.push(ticker);
    }
    let summary: GroupBacktestSummary | null = null;
    let firstMismatch: GroupMismatch | null = null;
    for (const option of options) {
      const mismatch = findGroupMismatch(option, input);
      if (mismatch === null) {
        summary = option;
        break;
      }
      firstMismatch ??= mismatch;
    }
    if (!summary) {
      if (firstMismatch) {
        excluded[firstMismatch.kind].push(`${ticker} (${firstMismatch.detail})`);
      }
      continue;
    }

    backtestIds[ticker] = summary.backtestId;
    const from = toTimestamp(summary.from);
    const to = toTimestamp(summary.to);
    const listedAt = input.listingDates?.[pairCodeFor(ticker, quote)];
    const expectedFrom = runFrom !== null && isFiniteNumber(listedAt) && listedAt > runFrom ? listedAt : runFrom;
    if (
      expectedFrom !== null &&
      runTo !== null &&
      from !== null &&
      to !== null &&
      (Math.abs(from - expectedFrom) > tolerance || Math.abs(to - runTo) > tolerance)
    ) {
      periodMismatch.push(`${ticker} (${formatDate(from)} – ${formatDate(to)})`);
    }
    if (summary.useWicks !== null && summary.useWicks !== input.run.includeWicks) {
      wicksMismatch.push(ticker);
    }
    const summaryAlgorithm = nonEmptyString(summary.algorithm)?.toUpperCase() ?? null;
    if (algorithm && summaryAlgorithm && summaryAlgorithm !== algorithm) {
      algorithmMismatch.push(`${ticker} (${summaryAlgorithm})`);
    }
  }

  const warnings: string[] = [];
  if (periodMismatch.length > 0) {
    const runPeriod = runFrom !== null && runTo !== null ? ` (${formatDate(runFrom)} – ${formatDate(runTo)})` : '';
    warnings.push(
      `Период бэктестов группы не совпадает с периодом запуска${runPeriod}: ${formatList(periodMismatch)}. ` +
        'Результаты базы могут быть несопоставимы с кандидатами.',
    );
  }
  if (wicksMismatch.length > 0) {
    warnings.push(
      `В бэктестах группы другая настройка учёта теней (${formatList(wicksMismatch)}): ` +
        'результаты базы могут быть несопоставимы с кандидатами.',
    );
  }
  if (algorithmMismatch.length > 0) {
    warnings.push(`Алгоритм бэктестов группы отличается от базового бота: ${formatList(algorithmMismatch)}.`);
  }
  for (const kind of GROUP_MISMATCH_KINDS) {
    if (excluded[kind].length > 0) {
      warnings.push(GROUP_MISMATCH_WARNINGS[kind](formatList(excluded[kind]), input));
    }
  }
  if (duplicates.length > 0) {
    warnings.push(`В группе несколько бэктестов для ${formatList(duplicates)} — взяты самые новые подходящие.`);
  }
  if (wrongQuote.length > 0) {
    warnings.push(`Бэктесты группы с другой котировкой пропущены: ${formatList(wrongQuote)}.`);
  }

  return {
    backtestIds,
    missing: basket.filter((ticker) => !Object.hasOwn(backtestIds, ticker)),
    outsideBasket,
    warnings,
  };
};

/** Загрузка бэктестов группы из локального хранилища групп и кеша деталей бэктестов. */
export const loadGroupBacktestsFromStore = async (groupId: string): Promise<GroupBacktestsLoadResult> => {
  const group = readBacktestGroups().find((item) => item.id === groupId);
  if (!group) {
    throw new Error('Группа бэктестов с результатами базы не найдена (возможно, она удалена).');
  }
  const settled = await Promise.allSettled(group.backtestIds.map((id) => backtestsService.getBacktestDetail(id)));
  const backtests: GroupBacktestSummary[] = [];
  const failed: GroupBacktestsLoadResult['failed'] = [];

  settled.forEach((result, index) => {
    const backtestId = group.backtestIds[index];
    if (result.status === 'rejected') {
      failed.push({ backtestId, error: describeError(result.reason) });
      return;
    }
    const { statistics, config } = result.value;
    const base = resolveBaseTicker(statistics.base, statistics.symbol);
    if (!base) {
      failed.push({ backtestId, error: 'не удалось определить монету бэктеста' });
      return;
    }
    const commissions: Partial<BacktestConfigDto['commissions']> | null = isRecord(config.commissions)
      ? config.commissions
      : null;
    const deposit: Partial<BacktestConfigDto['deposit']> | null = isRecord(config.deposit) ? config.deposit : null;
    backtests.push({
      backtestId,
      base,
      quote: nonEmptyString(statistics.quote),
      from: nonEmptyString(statistics.from) ?? nonEmptyString(config.from),
      to: nonEmptyString(statistics.to) ?? nonEmptyString(config.to),
      useWicks: typeof config.useWicks === 'boolean' ? config.useWicks : null,
      algorithm: nonEmptyString(statistics.algorithm) ?? nonEmptyString(config.algorithm),
      commissions: commissions
        ? { maker: readNumeric(commissions.maker), taker: readNumeric(commissions.taker) }
        : null,
      deposit: deposit ? { amount: readNumeric(deposit.amount), leverage: readNumeric(deposit.leverage) } : null,
      exchange: nonEmptyString(config.exchange) ?? nonEmptyString(statistics.exchange),
      strategy: {
        conditions: config.conditions,
        settings: config.settings,
        profit: config.profit,
        stopLoss: config.stopLoss,
        pullUp: config.pullUp,
      },
    });
  });

  return { groupName: group.name, backtests, failed };
};

// ═══════════════════════════════════════════════════════════════════
// СОХРАНЁННОЕ СОСТОЯНИЕ
// ═══════════════════════════════════════════════════════════════════

const isConfigLike = (value: unknown): boolean =>
  isRecord(value) &&
  typeof value.botName === 'string' &&
  isStringArray(value.basket) &&
  isStringArray(value.screenBasket) &&
  Array.isArray(value.params) &&
  isRecord(value.run) &&
  isRecord(value.search) &&
  isRecord(value.scheduler) &&
  isRecord(value.objective);

const isCandidateLike = (value: unknown): boolean =>
  isRecord(value) &&
  typeof value.id === 'string' &&
  typeof value.hash === 'string' &&
  isRecord(value.values) &&
  isFiniteNumber(value.generation) &&
  isStringArray(value.changed);

/** Структурная проверка сохранённого состояния (версия 1). */
export const isPortfolioOptimizerState = (value: unknown): value is PortfolioOptimizerState =>
  isRecord(value) &&
  value.version === 1 &&
  typeof value.runId === 'string' &&
  isConfigLike(value.config) &&
  isRecord(value.baseStrategy) &&
  typeof value.quoteCurrency === 'string' &&
  typeof value.baseCandidateId === 'string' &&
  (value.bestCandidateId === null || typeof value.bestCandidateId === 'string') &&
  Array.isArray(value.candidates) &&
  value.candidates.every(isCandidateLike) &&
  isRecord(value.backtestCache) &&
  Array.isArray(value.inFlightJobs) &&
  isRecord(value.paramStats) &&
  isFiniteNumber(value.generation) &&
  isFiniteNumber(value.backtestsSubmitted) &&
  typeof value.status === 'string' &&
  RUN_STATUSES.has(value.status) &&
  isFiniteNumber(value.startedAt) &&
  Array.isArray(value.logs);

/** Сохраняет состояние в localStorage (JSON). false — не удалось (квота, недоступное хранилище). */
export const savePortfolioState = (state: PortfolioOptimizerState): boolean => {
  try {
    return writeStorageValue(PORTFOLIO_OPTIMIZER_STATE_KEY, JSON.stringify(state));
  } catch (error) {
    console.warn('[PortfolioOptimizer] Не удалось сериализовать состояние', error);
    return false;
  }
};

/** Сохранённый запуск или null (нет, повреждён или другой версии). */
export const loadSavedPortfolioState = (): PortfolioOptimizerState | null => {
  const raw = readStorageValue(PORTFOLIO_OPTIMIZER_STATE_KEY);
  if (!raw) {
    return null;
  }
  try {
    const parsed: unknown = JSON.parse(raw);
    return isPortfolioOptimizerState(parsed) ? parsed : null;
  } catch (error) {
    console.warn('[PortfolioOptimizer] Не удалось прочитать сохранённое состояние', error);
    return null;
  }
};

export const clearSavedPortfolioState = (): void => {
  removeStorageValue(PORTFOLIO_OPTIMIZER_STATE_KEY);
};

/**
 * Учитывает в состоянии бэктест, созданный POST, ответ на который пришёл уже после окончания сеанса:
 * +1 к созданным и запись о незавершённом бэктесте (к нему подключатся при продолжении). Мутирует state.
 * false — этот бэктест уже учтён (тот же id в кеше или среди незавершённых).
 */
export const applyLateSubmission = (state: PortfolioOptimizerState, job: ScheduledJob): boolean => {
  if (!isPositiveInteger(job.backtestId)) {
    return false;
  }
  const known =
    Object.values(state.backtestCache).includes(job.backtestId) ||
    state.inFlightJobs.some((item) => item.backtestId === job.backtestId);
  if (known) {
    return false;
  }
  state.backtestsSubmitted += 1;
  if (!Object.hasOwn(state.backtestCache, job.key)) {
    state.inFlightJobs = [...state.inFlightJobs.filter((item) => item.key !== job.key), { ...job }];
  }
  return true;
};

/**
 * Состояние для продолжения: конфигурация — нормализованная, активный статус (running/paused —
 * страница закрылась во время запуска) становится 'stopped', в журнале не больше 300 записей,
 * из незавершённых задач остаются только созданные бэктесты с корректным id.
 */
const restoreSavedState = (
  saved: PortfolioOptimizerState,
  config: PortfolioOptimizerConfig,
  quoteCurrency: string,
): PortfolioOptimizerState => {
  const state = cloneDeep(saved);
  if (!state.candidates.some((candidate) => candidate.id === state.baseCandidateId)) {
    throw new Error('Сохранённое состояние портфельного оптимизатора повреждено: нет базового кандидата.');
  }
  const bestExists =
    state.bestCandidateId !== null && state.candidates.some((candidate) => candidate.id === state.bestCandidateId);
  return {
    ...state,
    config,
    quoteCurrency,
    bestCandidateId: bestExists ? state.bestCandidateId : null,
    backtestCache: Object.fromEntries(Object.entries(state.backtestCache).filter(([, id]) => isPositiveInteger(id))),
    inFlightJobs: state.inFlightJobs.filter(
      (job) =>
        isRecord(job) && typeof job.key === 'string' && job.state === 'submitted' && isPositiveInteger(job.backtestId),
    ),
    status: state.status === 'running' || state.status === 'paused' ? 'stopped' : state.status,
    logs: state.logs.slice(-MAX_PORTFOLIO_LOG_ENTRIES),
  };
};

// ═══════════════════════════════════════════════════════════════════
// ПОДПИСИ И НАЗВАНИЯ
// ═══════════════════════════════════════════════════════════════════

/** Короткая метка кандидата для журнала и UI: «база» или «g3-2» (поколение 3, кандидат 2). */
export const candidateShortLabel = (
  state: Pick<PortfolioOptimizerState, 'runId' | 'baseCandidateId'>,
  candidate: Pick<Candidate, 'id'>,
): string => {
  if (candidate.id === state.baseCandidateId) {
    return BASE_LOG_LABEL;
  }
  const prefix = `${state.runId}-`;
  return candidate.id.startsWith(prefix) ? candidate.id.slice(prefix.length) : candidate.id.slice(-6);
};

/** Название бэктеста кандидата: «PO <бот> <метка> <МОНЕТА>», не длиннее 80 символов (имя бота обрезается). */
export const buildCandidateBacktestName = (botName: string, label: string, symbol: string): string => {
  const suffix = ` ${label} ${symbol.toUpperCase()}`;
  const name = botName.trim().replace(/\s+/g, ' ');
  const room = MAX_BACKTEST_NAME_LENGTH - BACKTEST_NAME_PREFIX.length - 1 - suffix.length;
  if (room <= 0 || name.length === 0) {
    return `${BACKTEST_NAME_PREFIX}${suffix}`.slice(0, MAX_BACKTEST_NAME_LENGTH);
  }
  const fitted = name.length > room ? name.slice(0, room).trimEnd() : name;
  return `${BACKTEST_NAME_PREFIX} ${fitted}${suffix}`;
};

// ═══════════════════════════════════════════════════════════════════
// ЗАВИСИМОСТИ ПО УМОЛЧАНИЮ
// ═══════════════════════════════════════════════════════════════════

/** Зависимости на реальном API Veles, планировщике, localStorage и таймерах. */
export const createDefaultPortfolioOptimizerDeps = (): PortfolioOptimizerDeps => ({
  createScheduler: (config, events) => new BacktestScheduler(config, events),
  evaluateBasket,
  loadGroupBacktests: loadGroupBacktestsFromStore,
  saveState: savePortfolioState,
  loadSavedState: () => loadSavedPortfolioState(),
  clearSavedState: () => clearSavedPortfolioState(),
  loadListingDates,
  sleep: abortableSleep,
  now: () => Date.now(),
});

/** Подмена части зависимостей; явные undefined не затирают значения по умолчанию. */
export const resolvePortfolioOptimizerDeps = (
  overrides: Partial<PortfolioOptimizerDeps> = {},
): PortfolioOptimizerDeps => {
  const defaults = createDefaultPortfolioOptimizerDeps();
  return {
    createScheduler: overrides.createScheduler ?? defaults.createScheduler,
    evaluateBasket: overrides.evaluateBasket ?? defaults.evaluateBasket,
    loadGroupBacktests: overrides.loadGroupBacktests ?? defaults.loadGroupBacktests,
    saveState: overrides.saveState ?? defaults.saveState,
    loadSavedState: overrides.loadSavedState ?? defaults.loadSavedState,
    clearSavedState: overrides.clearSavedState ?? defaults.clearSavedState,
    loadListingDates: overrides.loadListingDates ?? defaults.loadListingDates,
    sleep: overrides.sleep ?? defaults.sleep,
    now: overrides.now ?? defaults.now,
  };
};

// ═══════════════════════════════════════════════════════════════════
// ОРКЕСТРАТОР
// ═══════════════════════════════════════════════════════════════════

type RunPhase = 'idle' | 'base' | 'screen' | 'full';
type LoopOutcome = 'done' | 'budget' | 'exhausted' | 'halted';
type HaltReason = 'stop' | 'error';

/** Бэктест, который оркестратор ждёт в текущем сеансе (создаётся или опрашивается планировщиком). */
interface JobHandle {
  key: string;
  /** Отмена именно этой задачи; связана с общей остановкой запуска. */
  controller: AbortController;
  /** Бэктест уже создан в Veles. */
  submitted: boolean;
  /** POST уже отправлялся (бэктест может создаваться прямо сейчас) — отменять задачу нельзя. */
  posting: boolean;
  /** Этап отменён, пока шёл POST: задача отменяется перед следующей попыткой, если та понадобится. */
  cancelPending: boolean;
  /** Задача занимает место в лимите бэктестов, пока ещё не создана. */
  reserved: boolean;
  promise: Promise<number>;
}

/** Задачи одного этапа оценки: при первой неудаче ещё не созданные бэктесты этапа отменяются. */
interface StageScope {
  cancelled: boolean;
  handles: JobHandle[];
}

interface StageFailure {
  symbol: string;
  error: string;
}

interface RankedParent {
  candidate: Candidate;
  score: number;
}

export class PortfolioOptimizer {
  private state: PortfolioOptimizerState;
  private readonly callbacks: PortfolioOptimizerCallbacks;
  private readonly deps: PortfolioOptimizerDeps;
  private readonly paramsById: Map<string, ParamSpec>;

  private scheduler: PortfolioScheduler | null = null;
  private controller: AbortController | null = null;
  private runPromise: Promise<void> | null = null;
  private paused = false;
  private pauseWaiters: Array<() => void> = [];
  private haltReason: HaltReason | null = null;
  private readonly jobs = new Map<string, JobHandle>();
  private reservedSubmissions = 0;
  private phase: RunPhase = 'idle';
  private activeGeneration = 0;
  private sessionStartedAt: number | null = null;
  private sessionSubmittedAtStart = 0;
  private readonly patchedStrategies = new Map<string, BotStrategy>();
  private saveFailureReported = false;
  /** Сеанс закончен: состояние больше не сохраняется (запуск мог продолжиться в новом экземпляре). */
  private disposed = false;

  constructor(
    input: PortfolioOptimizerInput,
    callbacks: PortfolioOptimizerCallbacks,
    deps?: Partial<PortfolioOptimizerDeps>,
  ) {
    this.callbacks = callbacks;
    this.deps = resolvePortfolioOptimizerDeps(deps);

    const config = normalizePortfolioConfig(input.config);
    const quoteCurrency = normalizeTicker(input.quoteCurrency);
    if (!quoteCurrency) {
      throw new Error('Не указана валюта котировки для бэктестов.');
    }
    const now = this.deps.now();
    const runId = `po${Math.max(0, Math.floor(now)).toString(36)}`;
    const values = baseValues(config.params);
    const base: Candidate = {
      id: `${runId}-base`,
      hash: paramsHash(config.params, values),
      values,
      parentId: null,
      generation: 0,
      changed: [],
      mutated: [],
      screen: null,
      full: null,
      createdAt: now,
    };

    this.state = {
      version: 1,
      runId,
      config,
      baseStrategy: cloneDeep(input.baseStrategy),
      quoteCurrency,
      baseCandidateId: base.id,
      bestCandidateId: null,
      candidates: [base],
      backtestCache: {},
      inFlightJobs: [],
      paramStats: {},
      generation: 0,
      backtestsSubmitted: 0,
      status: 'idle',
      startedAt: now,
      updatedAt: now,
      logs: [],
    };
    this.paramsById = new Map(config.params.map((param) => [param.id, param]));
  }

  /** Продолжение сохранённого запуска (после перезагрузки или остановки). */
  static fromSaved(
    state: PortfolioOptimizerState,
    callbacks: PortfolioOptimizerCallbacks,
    deps?: Partial<PortfolioOptimizerDeps>,
  ): PortfolioOptimizer {
    if (!isPortfolioOptimizerState(state)) {
      throw new Error('Сохранённое состояние портфельного оптимизатора повреждено или устарело.');
    }
    const optimizer = new PortfolioOptimizer(
      { config: state.config, baseStrategy: state.baseStrategy, quoteCurrency: state.quoteCurrency },
      callbacks,
      deps,
    );
    optimizer.state = restoreSavedState(state, optimizer.state.config, optimizer.state.quoteCurrency);
    return optimizer;
  }

  // ─── Публичное управление ────────────────────────────────────────

  /** Запуск (или продолжение) до завершения, остановки или ошибки. Промис не отклоняется. */
  start(): Promise<void> {
    if (!this.runPromise) {
      this.runPromise = this.execute().finally(() => {
        this.runPromise = null;
      });
    }
    return this.runPromise;
  }

  /** Пауза: новые кандидаты и бэктесты не создаются, уже запущенные бэктесты досчитываются. */
  pause(): void {
    if (!this.runPromise || this.haltReason || this.paused) {
      return;
    }
    this.paused = true;
    this.scheduler?.setPaused(true);
    this.state.status = 'paused';
    this.log('info', 'Пауза: новые бэктесты не создаются, запущенные досчитываются.');
    this.persist();
  }

  resume(): void {
    if (!this.runPromise || this.haltReason || !this.paused) {
      return;
    }
    this.paused = false;
    this.scheduler?.setPaused(false);
    this.state.status = 'running';
    this.log('info', 'Продолжаю после паузы.');
    this.persist();
    this.releasePauseWaiters();
  }

  /** Остановка: опрос и очередь прерываются, состояние сохраняется для fromSaved().start(). */
  stop(): void {
    if (!this.runPromise || this.haltReason) {
      return;
    }
    this.haltReason = 'stop';
    this.paused = false;
    this.state.status = 'stopped';
    this.log('warning', 'Запуск остановлен. Созданные бэктесты сохранены — запуск можно продолжить позже.');
    this.persist();
    this.controller?.abort();
    this.releasePauseWaiters();
  }

  get isRunning(): boolean {
    return this.runPromise !== null;
  }

  /** Копия текущего состояния. */
  getState(): PortfolioOptimizerState {
    return cloneDeep(this.state);
  }

  getProgress(): PortfolioOptimizerProgress {
    const { search } = this.state.config;
    return {
      status: this.state.status,
      generation: this.runPromise ? Math.max(this.activeGeneration, this.state.generation) : this.state.generation,
      totalGenerations: search.generations,
      backtestsSubmitted: this.state.backtestsSubmitted,
      backtestsLimit: search.maxBacktests,
      inFlight: this.scheduler?.inFlight ?? 0,
      queued: this.scheduler?.queued ?? 0,
      candidatesEvaluated: this.state.candidates.filter(
        (candidate) =>
          candidate.id !== this.state.baseCandidateId &&
          (isStageFinished(candidate.screen) || isStageFinished(candidate.full)),
      ).length,
      startedAt: this.state.status === 'idle' ? null : this.state.startedAt,
      estimatedEndAt: this.estimateEndAt(this.deps.now()),
    };
  }

  /** Стратегия базового бота со значениями параметров кандидата. */
  patchCandidate(candidateId: string): BotStrategy {
    const candidate = this.state.candidates.find((item) => item.id === candidateId);
    if (!candidate) {
      throw new Error(`Кандидат ${candidateId} не найден.`);
    }
    return patchStrategy(this.state.baseStrategy, this.state.config.params, candidate.values);
  }

  /** Изменение настроек планировщика на ходу (потоки, интервал, опрос, таймаут). */
  updateSchedulerConfig(patch: Partial<SchedulerConfig>): void {
    this.state.config.scheduler = mergeSchedulerConfig(this.state.config.scheduler, patch);
    this.scheduler?.updateConfig(this.state.config.scheduler);
    this.persist();
  }

  /**
   * Бэктест, созданный предыдущим (уже остановленным) экземпляром этого же запуска: ответ на его POST
   * пришёл позже. Учитывается в лимите, к нему подключимся вместо повторного создания.
   * false — это другой запуск (или оптимизатор уже не работает).
   */
  adoptLateSubmission(runId: string, job: ScheduledJob): boolean {
    if (this.disposed || !this.runPromise || runId !== this.state.runId) {
      return false;
    }
    if (applyLateSubmission(this.state, job)) {
      this.log(
        'info',
        `Бэктест ${job.symbol} (#${job.backtestId}) создан ещё до продолжения запуска — учтён, повторно не создаётся.`,
      );
      this.persist();
    }
    return true;
  }

  // ─── Сеанс запуска ───────────────────────────────────────────────

  private async execute(): Promise<void> {
    const resumed = this.state.status !== 'idle';
    this.beginSession();
    if (!resumed) {
      this.state.startedAt = this.deps.now();
    }
    this.state.status = 'running';
    this.log('info', resumed ? this.describeResume() : this.describeStart());
    this.persist();
    this.notifyCandidates();

    let outcome: LoopOutcome = 'halted';
    try {
      this.attachInFlightJobs();
      outcome = await this.runLoop();
    } catch (error) {
      this.fail(`Сбой портфельного оптимизатора: ${describeError(error)}`);
    }
    if (!this.haltReason) {
      this.complete(outcome);
    }
    await this.endSession();
  }

  private beginSession(): void {
    this.disposed = false;
    this.haltReason = null;
    this.paused = false;
    this.pauseWaiters = [];
    this.jobs.clear();
    this.reservedSubmissions = 0;
    this.phase = 'idle';
    this.activeGeneration = 0;
    this.controller = new AbortController();
    this.sessionStartedAt = this.deps.now();
    this.sessionSubmittedAtStart = this.state.backtestsSubmitted;
    this.scheduler = this.deps.createScheduler(this.state.config.scheduler, {
      onSubmitted: (key, backtestId) => this.handleSubmitted(key, backtestId),
      onLog: (level, message) => this.log(level, message),
      beforeSubmit: (key) => this.handleBeforeSubmit(key),
    });
    this.scheduler.setPaused(false);
  }

  private async endSession(): Promise<void> {
    const controller = this.controller;
    if (controller && !controller.signal.aborted) {
      controller.abort();
    }
    await this.drainScheduler();
    for (const handle of this.jobs.values()) {
      handle.reserved = false;
    }
    this.jobs.clear();
    this.reservedSubmissions = 0;
    this.scheduler = null;
    this.controller = null;
    this.paused = false;
    this.phase = 'idle';
    this.sessionStartedAt = null;
    this.patchedStrategies.clear();
    this.releasePauseWaiters();
    this.persist();
    // Этапы, прерванные остановкой, остаются «в работе» — UI показывает их по сохранённому состоянию.
    this.notifyCandidates();
    this.disposed = true;
  }

  /**
   * Ждём ответа на POST, отправленный до остановки, чтобы id созданного бэктеста попал в состояние.
   * Если ответ придёт позже (вкладка Veles отключена), его примет handleSubmitted после окончания сеанса.
   */
  private async drainScheduler(): Promise<void> {
    const scheduler = this.scheduler;
    if (!scheduler) {
      return;
    }
    for (let poll = 0; poll < DRAIN_MAX_POLLS && scheduler.queued > 0; poll += 1) {
      try {
        await this.deps.sleep(DRAIN_POLL_INTERVAL_MS);
      } catch {
        return;
      }
    }
  }

  private describeStart(): string {
    const { botName, basket, screenBasket, search } = this.state.config;
    return (
      `Старт: бот «${botName}», корзина ${basket.length} монет, отборочная ${screenBasket.length}, ` +
      `поколений ${search.generations} × ${search.candidatesPerGeneration} кандидатов ` +
      `(на полную корзину — до ${search.promotePerGeneration}), лимит бэктестов ${search.maxBacktests}.`
    );
  }

  private describeResume(): string {
    const { search } = this.state.config;
    return (
      `Продолжение запуска: завершено поколений ${this.state.generation} из ${search.generations}, ` +
      `создано бэктестов ${this.state.backtestsSubmitted} из ${search.maxBacktests}.`
    );
  }

  /** Подключение к бэктестам, созданным до перезагрузки (результаты попадут в кеш). */
  private attachInFlightJobs(): void {
    const scheduler = this.requireScheduler();
    const pending: Array<{ key: string; backtestId: number; submittedAt: number | null }> = [];
    for (const job of this.state.inFlightJobs) {
      if (this.cachedBacktestId(job.key) !== null) {
        this.removeInFlightJob(job.key);
        continue;
      }
      if (job.backtestId !== null && !this.jobs.has(job.key)) {
        pending.push({ key: job.key, backtestId: job.backtestId, submittedAt: job.submittedAt });
      }
    }
    if (pending.length === 0) {
      return;
    }
    this.log('info', `Подключаюсь к незавершённым бэктестам: ${pending.length}.`);
    for (const { key, backtestId, submittedAt } of pending) {
      this.trackJob(key, true, (signal) => scheduler.attach(key, backtestId, signal, submittedAt));
    }
  }

  // ─── Основной цикл ───────────────────────────────────────────────

  private async runLoop(): Promise<LoopOutcome> {
    await this.ensureListingDates();
    if (this.isHalted()) {
      return 'halted';
    }
    const baseReady = await this.evaluateBaseCandidate();
    if (!baseReady) {
      return 'halted';
    }
    const { generations } = this.state.config.search;
    for (let generation = this.state.generation + 1; generation <= generations; generation += 1) {
      if (this.isHalted()) {
        return 'halted';
      }
      this.activeGeneration = generation;
      const outcome = await this.runGeneration(generation);
      if (outcome !== 'done') {
        return outcome;
      }
    }
    return 'done';
  }

  /** Полная и отборочная оценка базового бота. false — запуск продолжать нельзя. */
  private async evaluateBaseCandidate(): Promise<boolean> {
    const base = this.baseCandidate();
    this.phase = 'base';
    this.activeGeneration = 0;

    let full = base.full;
    if (full === null) {
      if (!this.excludeUnlistedSymbols()) {
        return false;
      }
      await this.applyBaseGroup(base);
      if (this.isHalted()) {
        return false;
      }
      full = createEvaluation('full', this.state.config.basket);
      base.full = full;
      this.persist();
    }
    if (!isStageDone(full)) {
      await this.evaluateStage(base, full);
      if (this.isHalted()) {
        return false;
      }
      if (!isStageDone(full)) {
        if (this.budgetLeft() <= 0) {
          const { maxBacktests } = this.state.config.search;
          this.fail(
            `Базовый бот не оценён на полной корзине: не хватило лимита бэктестов (создано ` +
              `${this.state.backtestsSubmitted} из ${maxBacktests}). Продолжение этого запуска не поможет — ` +
              'начните новый запуск с лимитом не меньше числа монет корзины, без готовых результатов базы.',
          );
          return false;
        }
        const groupHint = this.state.config.baseGroupId ? ', либо запустите оптимизацию без результатов группы' : '';
        this.fail(
          `Базовый бот не оценён на полной корзине: ${full.error ?? 'неизвестная ошибка'}. ` +
            `Без оценки базы сравнивать кандидатов не с чем — проверьте корзину и период и продолжите запуск${groupHint}.`,
        );
        return false;
      }
    }

    if (!isStageDone(base.screen)) {
      // Отборочная корзина — подмножество полной: все бэктесты уже в кеше, новых запусков нет.
      // Конфигурация читается заново: при оценке полной корзины из неё могли исключить монеты.
      const screen = createEvaluation('screen', this.state.config.screenBasket);
      base.screen = screen;
      await this.evaluateStage(base, screen);
      if (this.isHalted()) {
        return false;
      }
      if (!isStageDone(screen)) {
        this.fail(`Базовый бот не оценён на отборочной корзине: ${screen.error ?? 'неизвестная ошибка'}.`);
        return false;
      }
    }

    if (this.state.bestCandidateId === null) {
      this.state.bestCandidateId = base.id;
      this.persist();
      this.notifyCandidates();
    }
    return true;
  }

  /** Подстановка готовых бэктестов базы из группы в кеш (недостающие монеты посчитаются обычно). */
  private async applyBaseGroup(base: Candidate): Promise<void> {
    const { baseGroupId, basket, run } = this.state.config;
    if (!baseGroupId) {
      return;
    }
    let loaded: GroupBacktestsLoadResult;
    try {
      loaded = await this.deps.loadGroupBacktests(baseGroupId);
    } catch (error) {
      if (this.isHalted()) {
        return;
      }
      if (isAuthError(error)) {
        this.fail(authErrorMessage(error));
        return;
      }
      this.log(
        'warning',
        `Не удалось загрузить группу бэктестов базы: ${describeError(error)}. Бэктесты базы будут созданы заново.`,
      );
      return;
    }
    if (this.isHalted()) {
      return;
    }

    if (loaded.failed.length > 0) {
      const ids = loaded.failed.map((item) => String(item.backtestId));
      this.log(
        'warning',
        `Не удалось загрузить бэктесты группы (${formatList(ids)}): ${loaded.failed[0].error}. ` +
          'Эти монеты будут посчитаны заново.',
      );
    }
    const match = matchGroupBacktestsToBasket({
      backtests: loaded.backtests,
      basket,
      quoteCurrency: this.state.quoteCurrency,
      run,
      algorithm: this.state.baseStrategy.algorithm ?? null,
      baseStrategy: withBacktestDeposit(this.state.baseStrategy, this.state.config.backtestDeposit),
      params: this.state.config.params,
      listingDates: this.state.listingDates,
    });
    for (const warning of match.warnings) {
      this.log('warning', warning);
    }
    for (const [symbol, backtestId] of Object.entries(match.backtestIds)) {
      this.state.backtestCache[jobKey(base.hash, symbol)] = backtestId;
    }

    const matched = basket.length - match.missing.length;
    if (matched === 0) {
      this.log('info', `В группе «${loaded.groupName}» нет бэктестов монет корзины — база будет посчитана бэктестами.`);
    } else {
      const missingNote =
        match.missing.length > 0
          ? ` Недостающие монеты (${formatList(match.missing)}) будут посчитаны бэктестами.`
          : '';
      this.log(
        'info',
        `Результаты базы из группы «${loaded.groupName}»: ${matched} из ${basket.length} монет.${missingNote}`,
      );
    }
    this.persist();
  }

  private async runGeneration(generation: number): Promise<LoopOutcome> {
    const { search, screenBasket } = this.state.config;
    let candidates = this.state.candidates.filter(
      (candidate) => candidate.generation === generation && candidate.id !== this.state.baseCandidateId,
    );

    if (candidates.length === 0) {
      if (this.budgetLeft() <= 0) {
        return 'budget';
      }
      await this.waitWhilePaused();
      if (this.isHalted()) {
        return 'halted';
      }
      const created = this.createGenerationCandidates(generation);
      if (!Array.isArray(created)) {
        return created;
      }
      candidates = created;
      this.log(
        'info',
        `Поколение ${generation}/${search.generations}: ${candidates.length} кандидатов на отборочной корзине ` +
          `(${screenBasket.length} монет).`,
      );
    } else {
      this.log('info', `Поколение ${generation}/${search.generations}: продолжаю (${candidates.length} кандидатов).`);
    }

    this.phase = 'screen';
    await Promise.all(
      candidates.flatMap((candidate) => {
        const screen = candidate.screen;
        return screen && !isStageFinished(screen) ? [this.evaluateStage(candidate, screen)] : [];
      }),
    );
    if (this.isHalted()) {
      return 'halted';
    }

    this.phase = 'full';
    let promoted = candidates.filter((candidate) => candidate.full !== null);
    if (promoted.length === 0) {
      await this.waitWhilePaused();
      if (this.isHalted()) {
        return 'halted';
      }
      promoted = this.promoteCandidates(generation, candidates);
    }
    await Promise.all(
      promoted.flatMap((candidate) => {
        const full = candidate.full;
        return full && !isStageFinished(full) ? [this.evaluateStage(candidate, full)] : [];
      }),
    );
    if (this.isHalted()) {
      return 'halted';
    }

    this.finalizeGeneration(generation, candidates);
    return 'done';
  }

  /** Новые кандидаты поколения (в пределах лимита бэктестов). */
  private createGenerationCandidates(generation: number): Candidate[] | 'budget' | 'exhausted' {
    const { params, search, screenBasket } = this.state.config;
    const parents = this.rankedParents();
    const proposals = proposeCandidates({
      params,
      parents: parents.map((parent) => ({ values: parent.candidate.values, score: parent.score })),
      count: search.candidatesPerGeneration,
      mutationsMin: search.mutationsMin,
      mutationsMax: search.mutationsMax,
      seen: new Set(this.state.candidates.map((candidate) => candidate.hash)),
      paramStats: this.state.paramStats,
      rng: createRng(generationSeed(search.seed, generation)),
    });
    if (proposals.length === 0) {
      return 'exhausted';
    }

    // Хеши новых кандидатов не встречались, поэтому в кеше их бэктестов нет: отбор стоит |screen| бэктестов.
    const cost = screenBasket.length;
    const affordable = cost > 0 ? Math.min(proposals.length, Math.floor(this.budgetLeft() / cost)) : proposals.length;
    if (affordable <= 0) {
      return 'budget';
    }
    if (affordable < proposals.length) {
      this.log(
        'warning',
        `Лимит бэктестов: в поколении ${generation} проверяю ${affordable} из ${proposals.length} кандидатов.`,
      );
    }

    const now = this.deps.now();
    const baseId = this.state.baseCandidateId;
    const created = proposals.slice(0, affordable).map(
      (proposal, index): Candidate => ({
        id: `${this.state.runId}-g${generation}-${index + 1}`,
        hash: proposal.hash,
        values: proposal.values,
        parentId: parents.at(proposal.parentIndex)?.candidate.id ?? baseId,
        generation,
        changed: proposal.changed,
        mutated: proposal.mutated,
        screen: createEvaluation('screen', screenBasket),
        full: null,
        createdAt: now,
      }),
    );
    this.state.candidates.push(...created);
    this.persist();
    this.notifyCandidates();
    return created;
  }

  /**
   * Родители мутаций: кандидаты с посчитанной полной корзиной и достаточным числом сделок, от лучшего
   * к худшему (иначе база).
   */
  private rankedParents(): RankedParent[] {
    const ranked = this.state.candidates.flatMap((candidate): RankedParent[] => {
      const score = this.eligibleScore(candidate, 'full');
      return score === null ? [] : [{ candidate, score }];
    });
    ranked.sort((left, right) => right.score - left.score);
    return ranked.length > 0 ? ranked : [{ candidate: this.baseCandidate(), score: 0 }];
  }

  /** Отбор на полную корзину по score отбора относительно лучшего кандидата (в пределах лимита). */
  private promoteCandidates(generation: number, candidates: Candidate[]): Candidate[] {
    const { search, basket } = this.state.config;
    const bestScreenScore = stageScore(this.bestCandidate()?.screen ?? null);
    const fewDeals = candidates.filter(
      (candidate) => stageScore(candidate.screen) !== null && this.eligibleScore(candidate, 'screen') === null,
    );
    if (fewDeals.length > 0) {
      this.log(
        'info',
        `Поколение ${generation}: слишком мало сделок на отборе по сравнению с базой — ` +
          `${formatList(fewDeals.map((item) => `${this.label(item)} (${this.describeDeals(item, 'screen')})`))}.`,
      );
    }
    const selected = selectForPromotion(
      candidates.map((candidate) => ({ candidate, screenScore: this.eligibleScore(candidate, 'screen') })),
      bestScreenScore,
      search.promotePerGeneration,
      search.screenTolerance,
    );

    let budget = this.budgetLeft();
    const promoted: Candidate[] = [];
    const skipped: Candidate[] = [];
    for (const { candidate } of selected) {
      const cost = basket.filter((symbol) => this.cachedBacktestId(jobKey(candidate.hash, symbol)) === null).length;
      if (cost > budget) {
        skipped.push(candidate);
        continue;
      }
      budget -= cost;
      candidate.full = createEvaluation('full', basket);
      promoted.push(candidate);
    }

    if (skipped.length > 0) {
      this.log(
        'warning',
        `Лимит бэктестов: полная корзина пропущена для ${formatList(skipped.map((item) => this.label(item)))}.`,
      );
    }
    if (promoted.length > 0) {
      this.log(
        'info',
        `Поколение ${generation}: на полную корзину — ${formatList(promoted.map((item) => this.label(item)))}.`,
      );
      this.persist();
      this.notifyCandidates();
    } else if (skipped.length === 0) {
      this.log(
        'info',
        `Поколение ${generation}: никто не прошёл отбор (лучший результат отбора ${formatScore(bestScreenScore)}).`,
      );
    }
    return promoted;
  }

  /**
   * Итоги поколения одной синхронной операцией (статистика параметров, новый лучший, номер
   * поколения) — после перезагрузки поколение либо пересчитывается целиком, либо уже учтено.
   */
  private finalizeGeneration(generation: number, candidates: Candidate[]): void {
    const { search } = this.state.config;
    const previousBest = this.bestCandidate();
    const referenceScore = stageScore(previousBest?.full ?? null);

    let stats = this.state.paramStats;
    let newBest: RankedParent | null = null;
    for (const candidate of candidates) {
      const mutated = candidate.mutated ?? candidate.changed;
      const rawScore = stageScore(candidate.full);
      const score = this.eligibleScore(candidate, 'full');
      if (score === null && rawScore !== null && isImprovement(rawScore, referenceScore, search.minImprovement)) {
        this.log(
          'warning',
          `${this.label(candidate)}: score ${formatScore(rawScore)} выше лучшего, но сделок слишком мало ` +
            `(${this.describeDeals(candidate, 'full')}) — лучшим не становится.`,
        );
      }
      const improved = score !== null && isImprovement(score, referenceScore, search.minImprovement);
      stats = updateParamStats(stats, mutated, improved);
      if (improved && score !== null && (newBest === null || score > newBest.score)) {
        newBest = { candidate, score };
      }
    }
    this.state.paramStats = stats;

    if (newBest) {
      this.state.bestCandidateId = newBest.candidate.id;
      this.log('success', this.describeNewBest(newBest.candidate, previousBest));
    } else {
      const bestLabel = previousBest ? this.label(previousBest) : BASE_LOG_LABEL;
      this.log(
        'info',
        `Поколение ${generation}/${search.generations} без улучшения. Лучший — ${bestLabel} ` +
          `(score ${formatScore(referenceScore)}).`,
      );
    }
    this.state.generation = generation;
    this.persist();
    this.notifyCandidates();
  }

  private complete(outcome: LoopOutcome): void {
    const { search } = this.state.config;
    if (outcome === 'budget') {
      this.log(
        'warning',
        `Достигнут лимит бэктестов: создано ${this.state.backtestsSubmitted} из ${search.maxBacktests}.`,
      );
    } else if (outcome === 'exhausted') {
      this.log('info', 'Новых кандидатов нет: все близкие варианты включённых параметров уже проверены.');
    }
    const base = this.baseCandidate();
    const best = this.bestCandidate();
    const summary =
      best && best.id !== base.id
        ? `Лучший — ${this.label(best)}: score ${formatScore(stageScore(best.full))} против ` +
          `${formatScore(stageScore(base.full))} у базы. Изменения: ${this.describeChanges(best)}.`
        : 'Улучшений относительно базового бота не найдено.';
    this.log(
      'success',
      `Оптимизация завершена (поколений ${this.state.generation} из ${search.generations}, ` +
        `создано бэктестов ${this.state.backtestsSubmitted}). ${summary}`,
    );
    this.state.status = 'completed';
    this.persist();
  }

  /** Ошибка, после которой продолжать нельзя: статус 'error', всё прерывается, состояние сохраняется. */
  private fail(message: string): void {
    if (this.haltReason) {
      return;
    }
    this.haltReason = 'error';
    this.paused = false;
    this.state.status = 'error';
    this.log('error', message);
    this.persist();
    this.controller?.abort();
    this.releasePauseWaiters();
  }

  /**
   * Сбой Veles или сети (исчерпаны повторы, таймаут, не загрузились результаты): этап не помечается
   * неудачным, статистика параметров не обновляется — запуск останавливается с возможностью продолжить.
   */
  private haltForInfrastructure(detail: string): void {
    this.fail(infrastructureErrorMessage(detail));
  }

  // ─── Оценка этапа ────────────────────────────────────────────────

  /** Оценка этапа кандидата. Никогда не отклоняется: ошибки кандидата не прерывают запуск. */
  private async evaluateStage(candidate: Candidate, evaluation: CandidateEvaluation): Promise<void> {
    try {
      await this.runStage(candidate, evaluation);
    } catch (error) {
      if (this.isHalted()) {
        return;
      }
      if (isAuthError(error)) {
        this.fail(authErrorMessage(error));
        return;
      }
      this.markStageFailed(candidate, evaluation, [], describeError(error));
    } finally {
      this.patchedStrategies.delete(candidate.id);
    }
  }

  private async runStage(candidate: Candidate, evaluation: CandidateEvaluation): Promise<void> {
    const previousMetricsFailures = evaluation.metricsFailures ?? 0;
    evaluation.status = 'running';
    evaluation.failedSymbols = [];
    evaluation.metrics = null;
    evaluation.folds = [];
    evaluation.score = null;
    evaluation.error = null;
    this.persist();
    this.notifyCandidates();

    const scope: StageScope = { cancelled: false, handles: [] };
    const failures: StageFailure[] = [];
    // На полной корзине базы монеты, для которых Veles не принимает настройки бота (HTTP 412), исключаются
    // из корзины запуска; у кандидатов такой отказ — неудача этапа (их настройки на этой паре неисполнимы).
    const canExclude = candidate.id === this.state.baseCandidateId && evaluation.stage === 'full';
    const rejected: ExcludedSymbol[] = [];
    await Promise.all(
      evaluation.symbols.map(async (symbol) => {
        try {
          evaluation.backtestIds[symbol] = await this.ensureBacktest(candidate, symbol, scope);
        } catch (error) {
          if (this.isHalted() || (scope.cancelled && isAbortError(error))) {
            return;
          }
          if (isAuthError(error)) {
            this.fail(authErrorMessage(error));
            return;
          }
          if (isTransientBacktestError(error)) {
            // Монета не посчиталась из-за Veles или сети, а не из-за кандидата: этап досчитается после продолжения.
            evaluation.error = `${symbol}: ${describeError(error)}`;
            this.haltForInfrastructure(describeError(error));
            return;
          }
          if (canExclude && isPairRejectedError(error)) {
            rejected.push({ symbol, reason: describeError(error) });
            return;
          }
          failures.push({ symbol, error: describeError(error) });
          // Частичная корзина не оценивается: ещё не созданные бэктесты этапа больше не нужны.
          this.cancelStage(scope);
        }
      }),
    );
    if (this.isHalted()) {
      return;
    }
    if (rejected.length > 0 && !this.excludeRejectedSymbols(evaluation, rejected)) {
      return;
    }
    if (failures.length > 0) {
      this.markStageFailed(candidate, evaluation, failures, null);
      return;
    }

    const ids: number[] = [];
    for (const symbol of evaluation.symbols) {
      const backtestId = Object.hasOwn(evaluation.backtestIds, symbol) ? evaluation.backtestIds[symbol] : null;
      if (!isPositiveInteger(backtestId)) {
        this.markStageFailed(candidate, evaluation, [{ symbol, error: 'нет результата бэктеста' }], null);
        return;
      }
      ids.push(backtestId);
    }
    // Как в группе бэктестов (она хранит id по возрастанию): порядок важен для сделок с одинаковым входом.
    ids.sort((left, right) => left - right);

    let result: Pick<BasketEvaluation, 'metrics' | 'folds'> | null;
    try {
      result = await this.evaluateWithRetry(ids, candidate);
    } catch (error) {
      if (this.isHalted()) {
        return;
      }
      if (isAuthError(error)) {
        this.fail(authErrorMessage(error));
        return;
      }
      // Все бэктесты готовы и закешированы — не загрузились результаты. Обычно это сбой сети или Veles:
      // первый раз запуск останавливается (после продолжения этап пересчитается без новых бэктестов).
      evaluation.metricsFailures = previousMetricsFailures + 1;
      if (evaluation.metricsFailures < METRICS_MAX_FAILED_SESSIONS) {
        evaluation.error = `метрики не посчитаны: ${describeError(error)}`;
        this.haltForInfrastructure(`${this.label(candidate)}: не удалось посчитать метрики — ${describeError(error)}`);
        return;
      }
      this.markStageFailed(candidate, evaluation, [], describeError(error));
      return;
    }
    if (result === null || this.isHalted()) {
      return;
    }
    evaluation.metrics = result.metrics;
    evaluation.folds = result.folds;
    evaluation.score = computeObjective(result.metrics, result.folds, this.stageObjective(evaluation));
    evaluation.metricsFailures = undefined;
    evaluation.status = 'done';
    this.persist();
    this.notifyCandidates();
    this.log(
      'info',
      `${this.label(candidate)}: ${STAGE_TITLES[evaluation.stage]} — ${describeEvaluationResult(evaluation)}.`,
    );
  }

  /**
   * Цель этапа. Минимум сделок и ограничение МПУ заданы для полной корзины, поэтому на отборочной
   * масштабируются долями отбора по базе (для самой базы — по её же отбору): ранжирование на отборе
   * согласуется с полной корзиной.
   */
  private stageObjective(evaluation: CandidateEvaluation): ObjectiveConfig {
    const { objective, basket, screenBasket } = this.state.config;
    if (evaluation.stage !== 'screen') {
      return objective;
    }
    const base = this.baseCandidate();
    const baseScreen = base.screen === evaluation ? evaluation.metrics : doneMetrics(base.screen);
    return scaleObjective(
      objective,
      screenStageScale({
        baseFull: doneMetrics(base.full),
        baseScreen,
        screenSize: screenBasket.length,
        basketSize: basket.length,
      }),
    );
  }

  /**
   * Score этапа кандидата, если у него достаточно сделок по сравнению с базой на той же корзине
   * (hasEnoughDeals): почти не торгующий кандидат не проходит отбор и не становится лучшим. У базы порога нет.
   */
  private eligibleScore(candidate: Candidate, stage: EvaluationStage): number | null {
    const evaluation = stage === 'full' ? candidate.full : candidate.screen;
    const score = stageScore(evaluation);
    if (score === null || candidate.id === this.state.baseCandidateId) {
      return score;
    }
    const base = this.baseCandidate();
    const reference = doneMetrics(stage === 'full' ? base.full : base.screen)?.totalDeals;
    return hasEnoughDeals(doneMetrics(evaluation)?.totalDeals, reference) ? score : null;
  }

  /** «12 сделок при 1500 у базы» для журнала. */
  private describeDeals(candidate: Candidate, stage: EvaluationStage): string {
    const base = this.baseCandidate();
    const deals = doneMetrics(stage === 'full' ? candidate.full : candidate.screen)?.totalDeals;
    const baseDeals = doneMetrics(stage === 'full' ? base.full : base.screen)?.totalDeals;
    return `${deals ?? '—'} сделок при ${baseDeals ?? '—'} у базы`;
  }

  /**
   * Исключает из корзины запуска монеты, для которых Veles не принимает настройки базового бота (HTTP 412,
   * например объём ордера меньше минимального для пары): ни база, ни кандидаты на них не считаются.
   * false — отказов слишком много (больше MAX_EXCLUDED_BASKET_SHARE корзины): запуск остановлен.
   */
  private excludeRejectedSymbols(evaluation: CandidateEvaluation, rejected: ExcludedSymbol[]): boolean {
    return this.excludeFromBasket(rejected, evaluation, {
      warning: (described) => `Исключены из корзины — Veles не принимает для них настройки бота: ${described}.`,
      advice: 'Чтобы учитывать эти монеты, увеличьте депозит в бэктестах (или депозит бота) либо объём первых ордеров.',
      failure: (excludedTotal, originalSize, described) =>
        `Veles не принимает настройки бота для ${excludedTotal} из ${originalSize} монет корзины: ${described}. ` +
        'Дело, скорее всего, не в отдельных монетах — проверьте депозит, плечо и объёмы ордеров бота ' +
        '(или задайте депозит в бэктестах).',
    });
  }

  /**
   * Монеты, которые в выбранный период ещё не торговались (листинг не раньше конца периода), исключаются
   * из корзины до первых бэктестов — как в Veles Tools 2.15. false — исключать пришлось бы слишком много.
   */
  private excludeUnlistedSymbols(): boolean {
    const periodTo = toTimestamp(this.state.config.run.periodTo);
    if (periodTo === null) {
      return true;
    }
    const unlisted = this.state.config.basket.flatMap((symbol): ExcludedSymbol[] => {
      const listedAt = this.listingFor(symbol);
      return listedAt !== null && listedAt >= periodTo
        ? [{ symbol, reason: `листинг ${formatDate(listedAt)} — после окончания периода` }]
        : [];
    });
    if (unlisted.length === 0) {
      return true;
    }
    return this.excludeFromBasket(unlisted, null, {
      warning: (described) => `Исключены из корзины — не торговались в выбранный период: ${described}.`,
      advice: 'Чтобы учитывать эти монеты, сдвиньте окончание периода позже их листинга.',
      failure: (excludedTotal, originalSize, described) =>
        `В выбранный период не торговались ${excludedTotal} из ${originalSize} монет корзины: ${described}. ` +
        'Выберите период позже листинга монет или другую корзину.',
    });
  }

  /**
   * Исключение монет из корзины запуска: config.basket и config.screenBasket обновляются (отборочная
   * дополняется другими монетами полной корзины до прежнего размера), монеты записываются в
   * state.excludedSymbols, этап evaluation (если передан) их больше не ждёт.
   * false — исключать пришлось бы больше MAX_EXCLUDED_BASKET_SHARE корзины: запуск остановлен.
   */
  private excludeFromBasket(
    items: ExcludedSymbol[],
    evaluation: CandidateEvaluation | null,
    messages: {
      warning: (described: string) => string;
      advice: string;
      failure: (excludedTotal: number, originalSize: number, described: string) => string;
    },
  ): boolean {
    const { config } = this.state;
    const previous = this.state.excludedSymbols ?? [];
    const removed = new Set(items.map((item) => item.symbol));
    const remaining = config.basket.filter((symbol) => !removed.has(symbol));
    const originalSize = config.basket.length + previous.length;
    const excludedTotal = previous.length + removed.size;
    const described = formatList(items.map((item) => `${item.symbol} (${item.reason})`));
    if (remaining.length === 0 || excludedTotal > originalSize * MAX_EXCLUDED_BASKET_SHARE) {
      this.fail(messages.failure(excludedTotal, originalSize, described));
      return false;
    }

    const keptScreen = config.screenBasket.filter((symbol) => !removed.has(symbol));
    const kept = new Set(keptScreen);
    const refill = pickScreenBasket(
      remaining.filter((symbol) => !kept.has(symbol)),
      config.screenBasket.length - keptScreen.length,
      config.search.seed,
    );
    const updated: PortfolioOptimizerConfig = {
      ...config,
      basket: remaining,
      screenBasket: [...keptScreen, ...refill],
    };
    const excluded = [...previous, ...items];
    this.state.config = updated;
    this.state.excludedSymbols = excluded;
    if (evaluation) {
      evaluation.symbols = evaluation.symbols.filter((symbol) => !removed.has(symbol));
      evaluation.backtestIds = Object.fromEntries(
        Object.entries(evaluation.backtestIds).filter(([symbol]) => !removed.has(symbol)),
      );
    }
    const refillNote = refill.length > 0 ? `, в отборочную добавлены ${formatList(refill)}` : '';
    this.log(
      'warning',
      `${messages.warning(described)} Корзина: ${remaining.length} монет, отборочная: ` +
        `${updated.screenBasket.length}${refillNote}. ${messages.advice}`,
    );
    this.persist();
    this.invokeCallback(() => this.callbacks.onBasketChanged?.(updated, excluded));
    return true;
  }

  /**
   * Даты листинга пар биржи бота — один раз за запуск (сохраняются в состоянии, чтобы периоды бэктестов
   * не менялись между сеансами). Не загрузились — период для всех монет как в форме.
   */
  private async ensureListingDates(): Promise<void> {
    if (this.state.listingDates) {
      return;
    }
    const exchange =
      nonEmptyString(this.state.baseStrategy.pair?.exchange) ?? nonEmptyString(this.state.baseStrategy.exchange);
    let listingDates: Record<string, number> = {};
    if (exchange) {
      try {
        listingDates = await this.deps.loadListingDates(exchange);
      } catch (error) {
        if (this.isHalted()) {
          return;
        }
        if (isAuthError(error)) {
          this.fail(authErrorMessage(error));
          return;
        }
        this.log(
          'warning',
          `Не удалось загрузить даты листинга монет (${describeError(error)}): период бэктестов для всех монет — как в форме.`,
        );
      }
    }
    this.state.listingDates = listingDates;
    const runFrom = toTimestamp(this.state.config.run.periodFrom);
    const runTo = toTimestamp(this.state.config.run.periodTo);
    const late = this.state.config.basket.flatMap((symbol) => {
      const listedAt = this.listingFor(symbol);
      return runFrom !== null && runTo !== null && listedAt !== null && listedAt > runFrom && listedAt < runTo
        ? [`${symbol} (с ${formatDate(listedAt)})`]
        : [];
    });
    if (late.length > 0) {
      this.log(
        'info',
        `Листинг позже начала периода — бэктесты этих монет начинаются с даты листинга: ${formatList(late)}.`,
      );
    }
    this.persist();
  }

  /** Дата листинга монеты корзины (мс) или null — неизвестна. */
  private listingFor(symbol: string): number | null {
    const listedAt = this.state.listingDates?.[pairCodeFor(symbol, this.state.quoteCurrency)];
    return isFiniteNumber(listedAt) ? listedAt : null;
  }

  /** Начало периода бэктеста монеты: не раньше её листинга (иначе Veles не посчитает бэктест). */
  private symbolPeriodStart(symbol: string): string {
    const { periodFrom } = this.state.config.run;
    const runFrom = toTimestamp(periodFrom);
    const listedAt = this.listingFor(symbol);
    return runFrom !== null && listedAt !== null && listedAt > runFrom ? new Date(listedAt).toISOString() : periodFrom;
  }

  private markStageFailed(
    candidate: Candidate,
    evaluation: CandidateEvaluation,
    failures: StageFailure[],
    error: string | null,
  ): void {
    const failedSymbols = evaluation.symbols.filter((symbol) => failures.some((failure) => failure.symbol === symbol));
    const firstFailure = failures.at(0);
    const message =
      error ??
      (firstFailure ? `не посчитаны ${formatList(failedSymbols)}: ${firstFailure.error}` : 'неизвестная ошибка');
    evaluation.status = 'failed';
    evaluation.failedSymbols = failedSymbols;
    evaluation.metrics = null;
    evaluation.folds = [];
    evaluation.score = null;
    evaluation.error = message;
    this.persist();
    this.notifyCandidates();
    this.log('warning', `${this.label(candidate)}: этап «${STAGE_TITLES[evaluation.stage]}» не оценён — ${message}`);
  }

  /** Метрики корзины с повтором при сбоях загрузки результатов. null — запуск остановлен. */
  private async evaluateWithRetry(
    ids: number[],
    candidate: Candidate,
  ): Promise<Pick<BasketEvaluation, 'metrics' | 'folds'> | null> {
    const { block, run, objective } = this.state.config;
    const folds = objective.kind === 'robust' ? objective.folds : 0;
    const period: BasketPeriod = { periodFrom: run.periodFrom, periodTo: run.periodTo };
    for (let attempt = 1; ; attempt += 1) {
      try {
        return await this.deps.evaluateBasket(ids, block, period, folds);
      } catch (error) {
        if (this.isHalted()) {
          return null;
        }
        if (isAuthError(error) || attempt >= EVALUATION_MAX_ATTEMPTS) {
          throw error;
        }
        const delayMs = EVALUATION_RETRY_DELAYS_MS.at(attempt - 1) ?? EVALUATION_RETRY_DELAYS_MS[0];
        this.log(
          'warning',
          `${this.label(candidate)}: не удалось посчитать метрики (попытка ${attempt}/${EVALUATION_MAX_ATTEMPTS}): ` +
            `${describeError(error)}. Повтор через ${formatSeconds(delayMs)} с.`,
        );
        try {
          await this.deps.sleep(delayMs, this.controller?.signal);
        } catch {
          if (this.isHalted()) {
            return null;
          }
        }
      }
    }
  }

  // ─── Бэктесты ────────────────────────────────────────────────────

  /**
   * id готового бэктеста кандидата по монете: из кеша, из уже ожидаемой задачи, подключением к
   * бэктесту, созданному до перезагрузки, или созданием нового (в пределах лимита).
   */
  private ensureBacktest(candidate: Candidate, symbol: string, scope: StageScope): Promise<number> {
    const key = jobKey(candidate.hash, symbol);
    const cached = this.cachedBacktestId(key);
    if (cached !== null) {
      return Promise.resolve(cached);
    }
    const existing = this.jobs.get(key);
    if (existing) {
      scope.handles.push(existing);
      return existing.promise;
    }
    if (scope.cancelled) {
      return Promise.reject(new SchedulerAbortError());
    }

    const scheduler = this.requireScheduler();
    const resumed = this.state.inFlightJobs.find((job) => job.key === key && job.backtestId !== null);
    let handle: JobHandle;
    if (resumed?.backtestId) {
      const { backtestId, submittedAt } = resumed;
      handle = this.trackJob(key, true, (signal) => scheduler.attach(key, backtestId, signal, submittedAt));
    } else {
      if (this.budgetLeft() <= 0) {
        return Promise.reject(new Error(`исчерпан лимит бэктестов (${this.state.config.search.maxBacktests})`));
      }
      let payload: BotStrategy;
      try {
        payload = this.buildPayload(candidate, symbol);
      } catch (error) {
        return Promise.reject(error);
      }
      handle = this.trackJob(key, false, (signal) => scheduler.run({ key, symbol, payload }, signal));
    }
    scope.handles.push(handle);
    return handle.promise;
  }

  /**
   * Регистрирует задачу сеанса: результат попадает в кеш, запись о незавершённом бэктесте удаляется.
   * Запись сохраняется (после продолжения к бэктесту подключимся заново) при общей остановке, ошибке
   * авторизации и сбое Veles/сети, в том числе при таймауте только что созданного бэктеста. Удаляется она
   * только при окончательной неудаче: бэктест завершился ошибкой или не завершился и после продолжения.
   */
  private trackJob(key: string, alreadySubmitted: boolean, start: (signal: AbortSignal) => Promise<number>): JobHandle {
    const sessionSignal = this.requireController().signal;
    const controller = new AbortController();
    const forwardAbort = (): void => controller.abort();
    if (sessionSignal.aborted) {
      controller.abort();
    } else {
      sessionSignal.addEventListener('abort', forwardAbort, { once: true });
    }

    const handle: JobHandle = {
      key,
      controller,
      submitted: alreadySubmitted,
      posting: false,
      cancelPending: false,
      reserved: !alreadySubmitted,
      // Заменяется ниже: задача должна быть зарегистрирована до старта (onSubmitted может прийти сразу).
      promise: Promise.resolve(0),
    };
    if (handle.reserved) {
      this.reservedSubmissions += 1;
    }
    this.jobs.set(key, handle);

    const track = async (): Promise<number> => {
      try {
        const backtestId = await start(controller.signal);
        this.state.backtestCache[key] = backtestId;
        this.removeInFlightJob(key);
        return backtestId;
      } catch (error) {
        if (sessionSignal.aborted || isAuthError(error)) {
          throw error;
        }
        if (isTransientBacktestError(error) && error.reason === 'timeout' && alreadySubmitted) {
          // Бэктест, к которому подключились после продолжения, так и не завершился (таймаут от его создания).
          this.removeInFlightJob(key);
          throw new Error(`${error.message} Бэктест не завершился и после продолжения запуска.`, { cause: error });
        }
        if (!isTransientBacktestError(error)) {
          // Бэктест завершился ошибкой или больше не нужен — продолжать его не будем.
          this.removeInFlightJob(key);
        }
        throw error;
      } finally {
        sessionSignal.removeEventListener('abort', forwardAbort);
        this.releaseReservation(handle);
        if (this.jobs.get(key) === handle) {
          this.jobs.delete(key);
        }
        this.persist();
      }
    };
    handle.promise = track();
    // Ошибка обрабатывается этапом; для «осиротевших» подключений — гасим необработанное отклонение.
    handle.promise.catch(() => undefined);
    return handle;
  }

  /** Бэктест создан в Veles: учитываем в лимите и запоминаем для продолжения после перезагрузки. */
  private handleSubmitted(key: string, backtestId: number): void {
    if (this.disposed) {
      this.recordLateSubmission(key, backtestId);
      return;
    }
    const handle = this.jobs.get(key);
    if (handle) {
      handle.submitted = true;
      this.releaseReservation(handle);
    }
    this.state.backtestsSubmitted += 1;
    const sessionAborted = this.controller === null || this.controller.signal.aborted;
    const jobAlive = handle !== undefined && !handle.controller.signal.aborted;
    if (jobAlive || sessionAborted) {
      this.upsertInFlightJob({
        key,
        symbol: symbolFromKey(key),
        backtestId,
        state: 'submitted',
        attempts: 1,
        submittedAt: this.deps.now(),
        error: null,
      });
    }
    this.persist();
  }

  /**
   * Ответ на POST пришёл после окончания сеанса. Состояние этого экземпляра больше не сохраняется (запуск
   * мог продолжиться в новом экземпляре), поэтому бэктест передаётся живому оптимизатору того же запуска,
   * а если его нет — дописывается в сохранённое состояние. Иначе id был бы потерян, а бэктест создан повторно.
   */
  private recordLateSubmission(key: string, backtestId: number): void {
    const job: ScheduledJob = {
      key,
      symbol: symbolFromKey(key),
      backtestId,
      state: 'submitted',
      attempts: 1,
      submittedAt: this.deps.now(),
      error: null,
    };
    let adopted = false;
    try {
      adopted = this.callbacks.onLateSubmission?.(this.state.runId, job) ?? false;
    } catch (error) {
      console.warn('[PortfolioOptimizer] Ошибка в обработчике событий', error);
    }
    if (adopted) {
      return;
    }
    try {
      const saved = this.deps.loadSavedState();
      if (saved && saved.runId === this.state.runId && applyLateSubmission(saved, job)) {
        this.deps.saveState(saved);
      }
    } catch (error) {
      console.warn('[PortfolioOptimizer] Не удалось записать бэктест, созданный после остановки', error);
    }
  }

  /**
   * Перед отправкой POST: отменённый этап отменяет задачу (бэктест ещё не создан); бэктест, который уже
   * создан для этой задачи (ответ остановленного сеанса пришёл позже), подключается без повторного создания.
   */
  private handleBeforeSubmit(key: string): number | null {
    const handle = this.jobs.get(key);
    if (handle?.cancelPending && !handle.submitted) {
      handle.controller.abort();
      return null;
    }
    const existing = this.state.inFlightJobs.find((job) => job.key === key && isPositiveInteger(job.backtestId));
    if (existing?.backtestId) {
      if (handle) {
        handle.submitted = true;
        this.releaseReservation(handle);
      }
      return existing.backtestId;
    }
    if (handle) {
      handle.posting = true;
    }
    return null;
  }

  /**
   * Отмена этапа: ещё не созданные бэктесты отменяются. Задачи, чей POST уже отправлялся, не прерываются —
   * бэктест может уже создаваться в Veles; созданный досчитается и попадёт в кеш, а не созданный
   * отменится перед повторной попыткой.
   */
  private cancelStage(scope: StageScope): void {
    scope.cancelled = true;
    for (const handle of scope.handles) {
      if (handle.submitted) {
        continue;
      }
      if (handle.posting) {
        handle.cancelPending = true;
        continue;
      }
      handle.controller.abort();
    }
  }

  private releaseReservation(handle: JobHandle): void {
    if (handle.reserved) {
      handle.reserved = false;
      this.reservedSubmissions = Math.max(0, this.reservedSubmissions - 1);
    }
  }

  /** Сколько ещё бэктестов можно создать (с учётом поставленных в очередь). */
  private budgetLeft(): number {
    return Math.max(
      0,
      this.state.config.search.maxBacktests - this.state.backtestsSubmitted - this.reservedSubmissions,
    );
  }

  private cachedBacktestId(key: string): number | null {
    const cache = this.state.backtestCache;
    if (!Object.hasOwn(cache, key)) {
      return null;
    }
    const backtestId = cache[key];
    return isPositiveInteger(backtestId) ? backtestId : null;
  }

  private upsertInFlightJob(job: ScheduledJob): void {
    const index = this.state.inFlightJobs.findIndex((item) => item.key === job.key);
    if (index >= 0) {
      this.state.inFlightJobs[index] = job;
    } else {
      this.state.inFlightJobs.push(job);
    }
  }

  private removeInFlightJob(key: string): void {
    if (this.state.inFlightJobs.some((job) => job.key === key)) {
      this.state.inFlightJobs = this.state.inFlightJobs.filter((job) => job.key !== key);
    }
  }

  private buildPayload(candidate: Candidate, symbol: string): BotStrategy {
    const { botName, params, run } = this.state.config;
    let patched = this.patchedStrategies.get(candidate.id);
    if (!patched) {
      patched = patchStrategy(this.state.baseStrategy, params, candidate.values);
      this.patchedStrategies.set(candidate.id, patched);
    }
    const label = candidate.id === this.state.baseCandidateId ? BASE_BACKTEST_LABEL : this.label(candidate);
    const payload = buildBacktestPayload(withBacktestDeposit(patched, this.state.config.backtestDeposit), {
      name: buildCandidateBacktestName(botName, label, symbol),
      makerCommission: run.makerCommission,
      takerCommission: run.takerCommission,
      includeWicks: run.includeWicks,
      isPublic: false,
      periodStartISO: this.symbolPeriodStart(symbol),
      periodEndISO: run.periodTo,
      overrideSymbol: composeSymbol(symbol, this.state.quoteCurrency),
    });
    return withoutApiKey(payload);
  }

  // ─── Пауза, остановка ────────────────────────────────────────────

  private isHalted(): boolean {
    return this.haltReason !== null;
  }

  private async waitWhilePaused(): Promise<void> {
    while (this.paused && !this.isHalted()) {
      await new Promise<void>((resolve) => {
        this.pauseWaiters.push(resolve);
      });
    }
  }

  private releasePauseWaiters(): void {
    const waiters = this.pauseWaiters;
    this.pauseWaiters = [];
    for (const resolve of waiters) {
      resolve();
    }
  }

  private requireScheduler(): PortfolioScheduler {
    if (!this.scheduler) {
      throw new Error('Планировщик бэктестов не запущен.');
    }
    return this.scheduler;
  }

  private requireController(): AbortController {
    if (!this.controller) {
      throw new Error('Запуск оптимизатора не активен.');
    }
    return this.controller;
  }

  // ─── Кандидаты и подписи ─────────────────────────────────────────

  private baseCandidate(): Candidate {
    const base = this.state.candidates.find((candidate) => candidate.id === this.state.baseCandidateId);
    if (!base) {
      throw new Error('Базовый кандидат не найден в состоянии оптимизатора.');
    }
    return base;
  }

  private bestCandidate(): Candidate | null {
    const bestId = this.state.bestCandidateId;
    return bestId === null ? null : (this.state.candidates.find((candidate) => candidate.id === bestId) ?? null);
  }

  private label(candidate: Candidate): string {
    return candidateShortLabel(this.state, candidate);
  }

  private describeChanges(candidate: Candidate): string {
    const parts = candidate.changed.flatMap((id) => {
      const param = this.paramsById.get(id);
      const value = Object.hasOwn(candidate.values, id) ? candidate.values[id] : undefined;
      return param && isFiniteNumber(value) ? [describeParamChange(param, value)] : [];
    });
    return parts.length > 0 ? formatList(parts, MAX_DESCRIBED_CHANGES, '; ') : 'без изменений параметров';
  }

  private describeNewBest(candidate: Candidate, previous: Candidate | null): string {
    const full = candidate.full;
    const previousScore = stageScore(previous?.full ?? null);
    const previousLabel = previous ? this.label(previous) : BASE_LOG_LABEL;
    const result = full ? describeEvaluationResult(full) : `score ${formatScore(null)}`;
    return (
      `Новый лучший — ${this.label(candidate)}: ${result} (было ${formatScore(previousScore)} у ${previousLabel}). ` +
      `Изменения: ${this.describeChanges(candidate)}.`
    );
  }

  // ─── Прогресс, журнал, сохранение ────────────────────────────────

  private estimateEndAt(now: number): number | null {
    const status = this.state.status;
    if ((status !== 'running' && status !== 'paused') || this.sessionStartedAt === null) {
      return null;
    }
    const submitted = this.state.backtestsSubmitted - this.sessionSubmittedAtStart;
    if (submitted <= 0) {
      return null;
    }
    const msPerBacktest = Math.max(0, now - this.sessionStartedAt) / submitted;
    return Math.round(now + msPerBacktest * this.estimateRemainingBacktests());
  }

  /** Оценка оставшихся созданий бэктестов: очередь, досчёт текущего поколения и следующие поколения. */
  private estimateRemainingBacktests(): number {
    const { search, basket, screenBasket } = this.state.config;
    const { perGenerationMax } = estimateBacktests(this.state.config);
    const current = Math.max(this.activeGeneration, this.state.generation);
    let pending = this.scheduler?.queued ?? 0;
    if (this.phase === 'screen') {
      pending +=
        Math.min(search.promotePerGeneration, search.candidatesPerGeneration) *
        Math.max(0, basket.length - screenBasket.length);
    }
    const later = Math.max(0, search.generations - current) * perGenerationMax;
    const cap = Math.max(0, search.maxBacktests - this.state.backtestsSubmitted);
    return Math.min(cap, pending + later);
  }

  private log(level: PortfolioLogEntry['level'], message: string): void {
    const entry: PortfolioLogEntry = { timestamp: this.deps.now(), level, message };
    const logs = this.state.logs;
    logs.push(entry);
    if (logs.length > MAX_PORTFOLIO_LOG_ENTRIES) {
      logs.splice(0, logs.length - MAX_PORTFOLIO_LOG_ENTRIES);
    }
    this.invokeCallback(() => this.callbacks.onLog({ ...entry }));
  }

  private persist(): void {
    if (this.disposed) {
      return;
    }
    this.state.updatedAt = this.deps.now();
    let saved: boolean;
    try {
      saved = this.deps.saveState(this.state);
    } catch (error) {
      console.warn('[PortfolioOptimizer] Ошибка сохранения состояния', error);
      saved = false;
    }
    if (!saved) {
      // Старая сохранённая копия устарела: продолжение с неё создало бы бэктесты повторно и превысило лимит.
      try {
        this.deps.clearSavedState();
      } catch (error) {
        console.warn('[PortfolioOptimizer] Не удалось удалить устаревшее состояние', error);
      }
      if (!this.saveFailureReported) {
        this.saveFailureReported = true;
        this.log(
          'warning',
          'Не удалось сохранить состояние запуска (хранилище недоступно или переполнено): устаревшая ' +
            'сохранённая копия удалена, после перезагрузки страницы продолжить запуск не получится.',
        );
      }
    }
    this.invokeCallback(() => this.callbacks.onProgress(this.getProgress()));
  }

  private notifyCandidates(): void {
    this.invokeCallback(() =>
      this.callbacks.onCandidatesChanged(cloneDeep(this.state.candidates), this.state.bestCandidateId),
    );
  }

  private invokeCallback(callback: () => void): void {
    try {
      callback();
    } catch (error) {
      console.warn('[PortfolioOptimizer] Ошибка в обработчике событий', error);
    }
  }
}
