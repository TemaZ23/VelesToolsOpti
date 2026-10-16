import { beforeEach, describe, expect, it, type Mock, vi } from 'vitest';

const { getBacktestDetailMock } = vi.hoisted(() => ({
  getBacktestDetailMock: vi.fn<(id: number) => Promise<BacktestDetail>>(),
}));

vi.mock('../backtests', () => ({
  backtestsService: {
    getBacktestDetail: getBacktestDetailMock,
  },
}));

import type { BotStrategy } from '../../api/backtestRunner';
import type { BacktestConfigDto } from '../../api/backtests.dtos';
import type { StrategyConditionDto } from '../../api/bots.dtos';
import { baseValues, extractParams, paramsHash } from '../../lib/botParams';
import { writeBacktestGroups } from '../../storage/backtestGroupsStore';
import type { BacktestDetail, BacktestStatistics } from '../../types/backtests';
import type {
  BacktestRunSettings,
  Candidate,
  CandidateEvaluation,
  ObjectiveConfig,
  ParamSpec,
  PortfolioLogEntry,
  PortfolioMetrics,
  PortfolioOptimizerCallbacks,
  PortfolioOptimizerConfig,
  PortfolioOptimizerProgress,
  PortfolioOptimizerState,
  ScheduledJob,
  SchedulerConfig,
  SearchConfig,
} from '../../types/portfolioOptimizer';
import {
  BacktestSubmitError,
  SchedulerAbortError,
  type SchedulerEvents,
  type SchedulerJobRequest,
  TransientBacktestError,
} from '../backtestScheduler';
import {
  applyLateSubmission,
  buildCandidateBacktestName,
  candidateShortLabel,
  clearSavedPortfolioState,
  DEFAULT_OBJECTIVE,
  DEFAULT_SCHEDULER_CONFIG,
  DEFAULT_SEARCH_CONFIG,
  estimateBacktests,
  GROUP_PERIOD_TOLERANCE_MS,
  type GroupBacktestStrategy,
  type GroupBacktestSummary,
  type GroupBacktestsLoadResult,
  isAuthError,
  isPortfolioOptimizerState,
  loadGroupBacktestsFromStore,
  loadSavedPortfolioState,
  MAX_BACKTEST_NAME_LENGTH,
  MAX_PORTFOLIO_LOG_ENTRIES,
  matchGroupBacktestsToBasket,
  normalizePortfolioConfig,
  normalizeSearchConfig,
  PORTFOLIO_OPTIMIZER_STATE_KEY,
  type PortfolioBasketEvaluator,
  PortfolioOptimizer,
  type PortfolioOptimizerDeps,
  type PortfolioScheduler,
  pickScreenBasket,
  savePortfolioState,
} from '../portfolioOptimizer';
import type { BasketPeriod } from '../portfolioScore';

// ═══════════════════════════════════════════════════════════════════
// ФИКСТУРЫ: СТРАТЕГИЯ И КОНФИГУРАЦИЯ
// ═══════════════════════════════════════════════════════════════════

const RSI_ID = 'conditions[0].value';
const BASKET = ['ADA', 'XRP', 'SOL', 'DOGE', 'LINK', 'AVAX'];
const SCREEN = ['ADA', 'SOL', 'LINK'];
const RUN: BacktestRunSettings = {
  periodFrom: '2022-01-01T00:00:00.000Z',
  periodTo: '2024-01-01T00:00:00.000Z',
  makerCommission: 0.02,
  takerCommission: 0.055,
  includeWicks: true,
};
const START_TIME = Date.parse('2026-10-12T08:00:00.000Z');

const condition = (indicator: string, value: number | null): StrategyConditionDto => ({
  type: 'INDICATOR',
  indicator,
  interval: 'ONE_HOUR',
  basic: false,
  value,
  operation: 'LESS',
  closed: true,
  reverse: false,
});

const createStrategy = (): BotStrategy => ({
  id: 77,
  name: 'LONG 22',
  symbol: 'BTC/USDT',
  symbols: ['BTC/USDT'],
  pair: { exchange: 'BYBIT_FUTURES', type: 'FUTURES', from: 'BTC', to: 'USDT', symbol: 'BTCUSDT' },
  exchange: 'BYBIT_FUTURES',
  algorithm: 'LONG',
  status: 'RUNNING',
  substatus: 'WAITING',
  conditions: [condition('RSI', 40), condition('CCI', -100)],
  settings: {
    type: 'CUSTOM',
    baseOrder: null,
    orders: [
      { indent: 0, volume: 40, conditions: [] },
      { indent: 2, volume: 60, conditions: [] },
    ],
  },
  profit: { type: 'ALL', currency: 'USDT', checkPnl: 1, conditions: [] },
  deposit: { amount: 100, leverage: 10, marginType: 'CROSS', currency: 'USDT' },
  stopLoss: null,
  pullUp: null,
  portion: null,
});

/** Параметры стратегии, в поиске участвует только порог RSI (детерминированное пространство поиска). */
const createParams = (patch: (param: ParamSpec) => ParamSpec = (param) => param): ParamSpec[] =>
  extractParams(createStrategy()).map((param) => patch({ ...param, enabled: param.id === RSI_ID }));

const createConfig = (
  overrides: Partial<PortfolioOptimizerConfig> = {},
  search: Partial<SearchConfig> = {},
): PortfolioOptimizerConfig => ({
  botId: 77,
  botName: 'LONG 22',
  basket: [...BASKET],
  screenBasket: [...SCREEN],
  run: { ...RUN },
  block: 60,
  objective: { kind: 'pnlToRisk', folds: 3, maxMaeCap: null, minDeals: 0 },
  search: {
    ...DEFAULT_SEARCH_CONFIG,
    generations: 2,
    candidatesPerGeneration: 3,
    promotePerGeneration: 1,
    mutationsMin: 1,
    mutationsMax: 1,
    maxBacktests: 500,
    seed: 7,
    ...search,
  },
  scheduler: { ...DEFAULT_SCHEDULER_CONFIG },
  params: createParams(),
  baseGroupId: null,
  ...overrides,
});

const createInput = (overrides: Partial<PortfolioOptimizerConfig> = {}, search: Partial<SearchConfig> = {}) => ({
  config: createConfig(overrides, search),
  baseStrategy: createStrategy(),
  quoteCurrency: 'usdt',
});

const BASE_HASH = paramsHash(createParams(), baseValues(createParams()));

/** Score одного бэктеста: максимум при RSI = 30, база (RSI 40) даёт 80. */
const scoreForRsi = (rsi: number): number => 100 - 2 * Math.abs(rsi - 30);

const metricsWith = (pnlToRisk: number, backtests: number): PortfolioMetrics => ({
  totalProfit: pnlToRisk * 10,
  maxConcurrentMae: 10,
  maxDrawdown: 5,
  pnlToRisk,
  totalDeals: 50 * backtests,
  profitableDeals: 50 * backtests,
  losingDeals: 0,
  openDeals: 0,
  activeMae: 0,
  netPerDay: 1,
  maxConcurrentPositions: 3,
  backtests,
});

// ═══════════════════════════════════════════════════════════════════
// ФЕЙКОВЫЙ VELES И ПЛАНИРОВЩИК
// ═══════════════════════════════════════════════════════════════════

interface VelesBacktest {
  id: number;
  key: string;
  symbol: string;
  rsi: number;
}

/** Реестр «созданных в Veles» бэктестов: по id оценщик узнаёт RSI кандидата и монету. */
class FakeVeles {
  private nextId = 1000;
  readonly backtests = new Map<number, VelesBacktest>();

  create(key: string, symbol: string, rsi: number): number {
    const id = this.nextId;
    this.nextId += 1;
    this.backtests.set(id, { id, key, symbol, rsi });
    return id;
  }

  register(id: number, symbol: string, rsi: number): void {
    this.backtests.set(id, { id, key: `group|${symbol}`, symbol, rsi });
  }

  hashOf(id: number): string {
    const backtest = this.backtests.get(id);
    if (!backtest) {
      throw new Error(`Неизвестный бэктест ${id}`);
    }
    return backtest.key.split('|')[0];
  }
}

type FakeJobState = 'queued' | 'posting' | 'submitted' | 'done' | 'failed' | 'aborted';

interface FakeJob {
  kind: 'run' | 'attach';
  key: string;
  symbol: string;
  payload: BotStrategy | null;
  backtestId: number | null;
  /** attach: время создания бэктеста, переданное оркестратором. */
  submittedAt: number | null;
  state: FakeJobState;
  signal: AbortSignal;
  resolve: (backtestId: number) => void;
  reject: (error: Error) => void;
}

interface FakeSchedulerBehavior {
  /** Сколько бэктестов одновременно «считает Veles». */
  maxInFlight: number;
  /** true — задачи создаются только по вызову pump() из теста. */
  manualPump: boolean;
  /** Не завершать созданный бэктест автоматически. */
  hold: (job: FakeJob) => boolean;
  /** Ошибка POST: бэктест не создан. */
  submitError: (job: FakeJob) => Error | null;
  /** Ошибка выполнения: бэктест создан, но завершился неудачно. */
  runError: (job: FakeJob) => Error | null;
  /** POST «зависает» до вызова completePost / failPost из теста. */
  deferPost: (job: FakeJob) => boolean;
}

const createBehavior = (patch: Partial<FakeSchedulerBehavior> = {}): FakeSchedulerBehavior => ({
  maxInFlight: Number.POSITIVE_INFINITY,
  manualPump: false,
  hold: () => false,
  submitError: () => null,
  runError: () => null,
  deferPost: () => false,
  ...patch,
});

const readRsi = (payload: BotStrategy | null): number => {
  const value = payload?.conditions?.[0]?.value;
  if (typeof value !== 'number') {
    throw new Error('В payload нет порога RSI');
  }
  return value;
};

class FakeScheduler implements PortfolioScheduler {
  readonly jobs: FakeJob[] = [];
  readonly pausedHistory: boolean[] = [];
  readonly configUpdates: Array<Partial<SchedulerConfig>> = [];
  paused = false;

  constructor(
    readonly config: SchedulerConfig,
    readonly events: SchedulerEvents,
    private readonly veles: FakeVeles,
    private readonly behavior: FakeSchedulerBehavior,
  ) {}

  get inFlight(): number {
    return this.jobs.filter((job) => job.state === 'submitted').length;
  }

  get queued(): number {
    return this.jobs.filter((job) => job.state === 'queued').length;
  }

  run(job: SchedulerJobRequest, signal: AbortSignal): Promise<number> {
    return this.track('run', job.key, job.symbol, job.payload, null, signal);
  }

  attach(key: string, backtestId: number, signal: AbortSignal, submittedAt: number | null = null): Promise<number> {
    return this.track('attach', key, key.split('|')[1] ?? key, null, backtestId, signal, submittedAt);
  }

  setPaused(paused: boolean): void {
    this.paused = paused;
    this.pausedHistory.push(paused);
    if (!paused) {
      this.schedulePump();
    }
  }

  updateConfig(config: Partial<SchedulerConfig>): void {
    this.configUpdates.push(config);
  }

  /** Создание очередных бэктестов (в пределах «потоков»). */
  pump(): void {
    if (this.paused) {
      return;
    }
    for (const job of this.jobs) {
      if (job.state !== 'queued') {
        continue;
      }
      if (this.inFlight >= this.behavior.maxInFlight) {
        break;
      }
      this.submit(job);
    }
  }

  /** Завершить созданный бэктест (успехом или переданной ошибкой). */
  finish(job: FakeJob, error: Error | null = null): void {
    if (job.state !== 'submitted') {
      return;
    }
    const failure = error ?? this.behavior.runError(job);
    if (failure) {
      job.state = 'failed';
      job.reject(failure);
    } else {
      job.state = 'done';
      job.resolve(job.backtestId ?? 0);
    }
    this.schedulePump();
  }

  completeAll(): void {
    for (const job of this.jobs.filter((item) => item.state === 'submitted')) {
      this.finish(job);
    }
  }

  /** Ответ на «зависший» POST: бэктест создан. */
  completePost(job: FakeJob): void {
    if (job.state !== 'posting' && job.state !== 'aborted') {
      return;
    }
    const wasAborted = job.state === 'aborted';
    job.backtestId = this.veles.create(job.key, job.symbol, readRsi(job.payload));
    this.events.onSubmitted(job.key, job.backtestId);
    if (wasAborted || job.signal.aborted) {
      job.state = 'aborted';
      return;
    }
    job.state = 'submitted';
    this.scheduleCompletion(job);
  }

  /** «Зависший» POST не удался (восстановимо): задача снова в очереди. */
  failPost(job: FakeJob): void {
    if (job.state === 'posting') {
      job.state = 'queued';
      this.schedulePump();
    }
  }

  private track(
    kind: FakeJob['kind'],
    key: string,
    symbol: string,
    payload: BotStrategy | null,
    backtestId: number | null,
    signal: AbortSignal,
    submittedAt: number | null = null,
  ): Promise<number> {
    if (signal.aborted) {
      return Promise.reject(new SchedulerAbortError());
    }
    return new Promise<number>((resolve, reject) => {
      const job: FakeJob = {
        kind,
        key,
        symbol,
        payload,
        backtestId,
        submittedAt,
        state: kind === 'attach' ? 'submitted' : 'queued',
        signal,
        resolve,
        reject,
      };
      signal.addEventListener(
        'abort',
        () => {
          if (job.state === 'queued' || job.state === 'submitted' || job.state === 'posting') {
            job.state = 'aborted';
            reject(new SchedulerAbortError());
          }
        },
        { once: true },
      );
      this.jobs.push(job);
      if (kind === 'attach') {
        this.scheduleCompletion(job);
      }
      this.schedulePump();
    });
  }

  private submit(job: FakeJob): void {
    const existing = this.events.beforeSubmit?.(job.key) ?? null;
    if (job.signal.aborted) {
      return;
    }
    if (existing !== null) {
      job.backtestId = existing;
      job.state = 'submitted';
      this.scheduleCompletion(job);
      return;
    }
    if (this.behavior.deferPost(job)) {
      job.state = 'posting';
      return;
    }
    const error = this.behavior.submitError(job);
    if (error) {
      job.state = 'failed';
      job.reject(error);
      return;
    }
    job.backtestId = this.veles.create(job.key, job.symbol, readRsi(job.payload));
    job.state = 'submitted';
    this.events.onSubmitted(job.key, job.backtestId);
    this.scheduleCompletion(job);
  }

  private schedulePump(): void {
    if (!this.behavior.manualPump) {
      queueMicrotask(() => this.pump());
    }
  }

  private scheduleCompletion(job: FakeJob): void {
    if (!this.behavior.hold(job)) {
      queueMicrotask(() => this.finish(job));
    }
  }
}

// ═══════════════════════════════════════════════════════════════════
// ФЕЙКОВАЯ ОЦЕНКА КОРЗИНЫ И ОКРУЖЕНИЕ ТЕСТА
// ═══════════════════════════════════════════════════════════════════

interface EvaluatorCall {
  ids: number[];
  block: number;
  run: BasketPeriod;
  folds: number;
}

interface FakeEvaluator {
  evaluate: PortfolioBasketEvaluator;
  calls: EvaluatorCall[];
  /** Ошибка для вызова (по умолчанию — успех). */
  failWhen: (call: EvaluatorCall) => Error | null;
  /** Правка метрик корзины (по умолчанию — без изменений); rsi — пороги RSI бэктестов корзины. */
  adjust: (metrics: PortfolioMetrics, rsi: number[]) => PortfolioMetrics;
}

const createEvaluator = (veles: FakeVeles): FakeEvaluator => {
  const evaluator: FakeEvaluator = {
    calls: [],
    failWhen: () => null,
    adjust: (metrics) => metrics,
    evaluate: async (ids, block, run, folds) => {
      const call: EvaluatorCall = { ids: [...ids], block, run: { ...run }, folds };
      evaluator.calls.push(call);
      const error = evaluator.failWhen(call);
      if (error) {
        throw error;
      }
      const rsi = ids.map((id) => {
        const backtest = veles.backtests.get(id);
        if (!backtest) {
          throw new Error(`Неизвестный бэктест ${id}`);
        }
        return backtest.rsi;
      });
      const mean = rsi.reduce((total, value) => total + scoreForRsi(value), 0) / rsi.length;
      return { metrics: evaluator.adjust(metricsWith(mean, ids.length), rsi), folds: [] };
    },
  };
  return evaluator;
};

interface Harness {
  veles: FakeVeles;
  behavior: FakeSchedulerBehavior;
  schedulers: FakeScheduler[];
  evaluator: FakeEvaluator;
  saves: string[];
  logs: PortfolioLogEntry[];
  progress: PortfolioOptimizerProgress[];
  candidateUpdates: Array<{ candidates: Candidate[]; bestCandidateId: string | null }>;
  sleep: Mock<(ms: number, signal?: AbortSignal) => Promise<void>>;
  loadGroupBacktests: Mock<(groupId: string) => Promise<GroupBacktestsLoadResult>>;
  clearSavedState: Mock<() => void>;
  loadListingDates: Mock<(exchange: string) => Promise<Record<string, number>>>;
  callbacks: PortfolioOptimizerCallbacks;
  deps: Partial<PortfolioOptimizerDeps>;
  scheduler: () => FakeScheduler;
  lastSaved: () => PortfolioOptimizerState;
}

interface HarnessOptions {
  veles?: FakeVeles;
  behavior?: Partial<FakeSchedulerBehavior>;
  startTime?: number;
}

const createHarness = (options: HarnessOptions = {}): Harness => {
  const veles = options.veles ?? new FakeVeles();
  const behavior = createBehavior(options.behavior);
  const schedulers: FakeScheduler[] = [];
  const evaluator = createEvaluator(veles);
  const saves: string[] = [];
  const logs: PortfolioLogEntry[] = [];
  const progress: PortfolioOptimizerProgress[] = [];
  const candidateUpdates: Harness['candidateUpdates'] = [];
  let time = options.startTime ?? START_TIME;

  const sleep = vi.fn<(ms: number, signal?: AbortSignal) => Promise<void>>(async () => undefined);
  const loadGroupBacktests = vi.fn<(groupId: string) => Promise<GroupBacktestsLoadResult>>(async () => ({
    groupName: 'Пусто',
    backtests: [],
    failed: [],
  }));
  const clearSavedState = vi.fn<() => void>();
  const loadListingDates = vi.fn<(exchange: string) => Promise<Record<string, number>>>(async () => ({}));
  const readSaved = (): PortfolioOptimizerState | null => {
    const raw = saves.at(-1);
    if (!raw) {
      return null;
    }
    const parsed: unknown = JSON.parse(raw);
    return isPortfolioOptimizerState(parsed) ? parsed : null;
  };

  return {
    veles,
    behavior,
    schedulers,
    evaluator,
    saves,
    logs,
    progress,
    candidateUpdates,
    sleep,
    loadGroupBacktests,
    clearSavedState,
    loadListingDates,
    callbacks: {
      onLog: (entry) => logs.push(entry),
      onProgress: (snapshot) => progress.push(snapshot),
      onCandidatesChanged: (candidates, bestCandidateId) => candidateUpdates.push({ candidates, bestCandidateId }),
    },
    deps: {
      createScheduler: (config, events) => {
        const scheduler = new FakeScheduler(config, events, veles, behavior);
        schedulers.push(scheduler);
        return scheduler;
      },
      evaluateBasket: evaluator.evaluate,
      loadGroupBacktests,
      saveState: (state) => {
        saves.push(JSON.stringify(state));
        return true;
      },
      loadSavedState: readSaved,
      clearSavedState,
      loadListingDates,
      sleep,
      now: () => {
        time += 1000;
        return time;
      },
    },
    scheduler: () => {
      const scheduler = schedulers.at(-1);
      if (!scheduler) {
        throw new Error('Планировщик ещё не создан');
      }
      return scheduler;
    },
    lastSaved: () => {
      const raw = saves.at(-1);
      if (!raw) {
        throw new Error('Состояние ещё не сохранялось');
      }
      const parsed: unknown = JSON.parse(raw);
      if (!isPortfolioOptimizerState(parsed)) {
        throw new Error('Сохранено некорректное состояние');
      }
      return parsed;
    },
  };
};

const flushMicrotasks = async (rounds = 50): Promise<void> => {
  for (let round = 0; round < rounds; round += 1) {
    await Promise.resolve();
  }
};

const waitFor = async (condition: () => boolean, description: string, maxRounds = 20_000): Promise<void> => {
  for (let round = 0; round < maxRounds; round += 1) {
    if (condition()) {
      return;
    }
    await Promise.resolve();
  }
  throw new Error(`Не дождались: ${description}`);
};

/** Ручной режим планировщика: по одному «тику» создания бэктестов, пока запуск не завершится. */
const drive = async (harness: Harness, done: Promise<void>, maxSteps = 2_000): Promise<void> => {
  let finished = false;
  void done.then(() => {
    finished = true;
  });
  for (let step = 0; step < maxSteps && !finished; step += 1) {
    harness.scheduler().pump();
    await flushMicrotasks(80);
  }
  if (!finished) {
    throw new Error('Запуск не завершился');
  }
};

const findBase = (state: PortfolioOptimizerState): Candidate => {
  const base = state.candidates.find((candidate) => candidate.id === state.baseCandidateId);
  if (!base) {
    throw new Error('Нет базового кандидата');
  }
  return base;
};

const findBest = (state: PortfolioOptimizerState): Candidate => {
  const best = state.candidates.find((candidate) => candidate.id === state.bestCandidateId);
  if (!best) {
    throw new Error('Нет лучшего кандидата');
  }
  return best;
};

const nonBase = (state: PortfolioOptimizerState): Candidate[] =>
  state.candidates.filter((candidate) => candidate.id !== state.baseCandidateId);

const requireEvaluation = (evaluation: CandidateEvaluation | null): CandidateEvaluation => {
  if (!evaluation) {
    throw new Error('Этап не создан');
  }
  return evaluation;
};

const requireScore = (evaluation: CandidateEvaluation | null): number => {
  const score = requireEvaluation(evaluation).score;
  if (score === null) {
    throw new Error('Нет score');
  }
  return score;
};

const runsFor = (harness: Harness, hash: string): FakeJob[] =>
  harness.schedulers.flatMap((scheduler) =>
    scheduler.jobs.filter((job) => job.kind === 'run' && job.key.startsWith(`${hash}|`)),
  );

const submittedJobs = (harness: Harness): FakeJob[] =>
  harness.schedulers.flatMap((scheduler) =>
    scheduler.jobs.filter((job) => job.kind === 'run' && job.backtestId !== null),
  );

const idsInOrder = (evaluation: CandidateEvaluation | null, symbols: string[]): number[] =>
  symbols.map((symbol) => requireEvaluation(evaluation).backtestIds[symbol]);

const hasLog = (harness: Harness, level: PortfolioLogEntry['level'], fragment: string): boolean =>
  harness.logs.some((entry) => entry.level === level && entry.message.includes(fragment));

beforeEach(() => {
  window.localStorage.clear();
  getBacktestDetailMock.mockReset();
});

// ═══════════════════════════════════════════════════════════════════
// ЧИСТЫЕ ПОМОЩНИКИ
// ═══════════════════════════════════════════════════════════════════

describe('pickScreenBasket', () => {
  const basket = ['ada', 'XRP', 'sol', 'DOGE', 'link', 'AVAX', 'DOT', 'TRX', 'NEAR', 'ATOM'];

  it('возвращает всю корзину (без повторов, в верхнем регистре), если размер не меньше корзины', () => {
    expect(pickScreenBasket(['ada', 'ADA', ' xrp ', ''], 5, 1)).toEqual(['ADA', 'XRP']);
    expect(pickScreenBasket(basket, 10, 1)).toHaveLength(10);
  });

  it('возвращает пустой список для нулевого, отрицательного и нечислового размера', () => {
    expect(pickScreenBasket(basket, 0, 1)).toEqual([]);
    expect(pickScreenBasket(basket, -3, 1)).toEqual([]);
    expect(pickScreenBasket(basket, Number.NaN, 1)).toEqual([]);
  });

  it('берёт по одной монете из каждой равной части корзины и сохраняет порядок', () => {
    const upper = basket.map((ticker) => ticker.toUpperCase());
    const picked = pickScreenBasket(basket, 4, 3);
    expect(picked).toHaveLength(4);
    const positions = picked.map((ticker) => upper.indexOf(ticker));
    expect(positions.every((position) => position >= 0)).toBe(true);
    expect([...positions].sort((left, right) => left - right)).toEqual(positions);
    const strata = [
      [0, 2],
      [2, 5],
      [5, 7],
      [7, 10],
    ];
    positions.forEach((position, index) => {
      expect(position).toBeGreaterThanOrEqual(strata[index][0]);
      expect(position).toBeLessThan(strata[index][1]);
    });
  });

  it('детерминирована по seed, а разные seed дают разные выборки', () => {
    expect(pickScreenBasket(basket, 4, 11)).toEqual(pickScreenBasket(basket, 4, 11));
    const samples = new Set(Array.from({ length: 10 }, (_, seed) => pickScreenBasket(basket, 4, seed).join(',')));
    expect(samples.size).toBeGreaterThan(1);
  });
});

describe('estimateBacktests', () => {
  it('считает максимум за поколение, базу, итог и длительность при интервале', () => {
    const estimate = estimateBacktests(createConfig({}, { generations: 4, candidatesPerGeneration: 5 }));
    // 5 кандидатов × 3 монеты отбора + 1 продвинутый × (6 − 3) монет досчёта
    expect(estimate.perGenerationMax).toBe(18);
    expect(estimate.baseBacktests).toBe(6);
    expect(estimate.totalMax).toBe(6 + 4 * 18);
    expect(estimate.hoursAtDelay).toBeCloseTo((78 * 31) / 3600, 10);
  });

  it('не считает бэктесты базы, если её результаты берутся из группы', () => {
    expect(estimateBacktests(createConfig({ baseGroupId: 'group-1' })).baseBacktests).toBe(0);
  });

  it('ограничивает итог лимитом бэктестов и число продвигаемых — числом кандидатов', () => {
    const capped = estimateBacktests(createConfig({}, { generations: 100, maxBacktests: 50 }));
    expect(capped.totalMax).toBe(50);
    const promoteAll = estimateBacktests(createConfig({}, { candidatesPerGeneration: 2, promotePerGeneration: 9 }));
    expect(promoteAll.perGenerationMax).toBe(2 * 3 + 2 * 3);
  });
});

describe('normalizePortfolioConfig', () => {
  it('нормализует корзины и настройки поиска', () => {
    const config = normalizePortfolioConfig(
      createConfig(
        { basket: ['ada', 'ADA', 'xrp', 'sol'], screenBasket: ['sol', 'btc', 'ada'], block: 7.9, baseGroupId: '  ' },
        { generations: -2, candidatesPerGeneration: 0, mutationsMin: 3, mutationsMax: 1, maxBacktests: Number.NaN },
      ),
    );
    expect(config.basket).toEqual(['ADA', 'XRP', 'SOL']);
    expect(config.screenBasket).toEqual(['SOL', 'ADA']);
    expect(config.block).toBe(7);
    expect(config.baseGroupId).toBeNull();
    expect(config.search.generations).toBe(0);
    expect(config.search.candidatesPerGeneration).toBe(1);
    expect(config.search.mutationsMax).toBe(3);
    expect(config.search.maxBacktests).toBe(DEFAULT_SEARCH_CONFIG.maxBacktests);
  });

  it('подбирает отборочную корзину, если подходящих монет в ней нет', () => {
    const config = normalizePortfolioConfig(createConfig({ screenBasket: ['BTC'] }, { seed: 5 }));
    expect(config.screenBasket).toEqual(pickScreenBasket(BASKET, 10, 5));
  });

  it('бросает понятные ошибки для пустой корзины, отсутствия параметров и неверного периода', () => {
    expect(() => normalizePortfolioConfig(createConfig({ basket: [' '] }))).toThrow('Корзина монет пуста');
    expect(() => normalizePortfolioConfig(createConfig({ params: [] }))).toThrow('числовых параметров');
    expect(() =>
      normalizePortfolioConfig(createConfig({ run: { ...RUN, periodFrom: RUN.periodTo, periodTo: RUN.periodFrom } })),
    ).toThrow('Некорректный период');
  });

  it('normalizeSearchConfig подставляет значения по умолчанию', () => {
    expect(normalizeSearchConfig({})).toEqual(DEFAULT_SEARCH_CONFIG);
  });
});

/** Настройки стратегии бэктеста группы: базовый бот с заданным порогом RSI. */
const groupStrategy = (rsi: number): GroupBacktestStrategy => {
  const strategy = createStrategy();
  return {
    conditions: (strategy.conditions ?? []).map((item, index) => (index === 0 ? { ...item, value: rsi } : item)),
    settings: strategy.settings ?? { type: 'CUSTOM' },
    profit: strategy.profit ?? null,
    stopLoss: null,
    pullUp: 0,
  };
};

describe('matchGroupBacktestsToBasket', () => {
  const summary = (
    backtestId: number,
    base: string,
    patch: Partial<GroupBacktestSummary> = {},
  ): GroupBacktestSummary => ({
    backtestId,
    base,
    quote: 'USDT',
    from: RUN.periodFrom,
    to: RUN.periodTo,
    useWicks: true,
    algorithm: 'LONG',
    ...patch,
  });

  it('сопоставляет бэктесты по базовой монете в порядке корзины', () => {
    const match = matchGroupBacktestsToBasket({
      backtests: [summary(3, 'sol'), summary(1, 'ADA'), summary(9, 'BTC')],
      basket: BASKET,
      quoteCurrency: 'usdt',
      run: RUN,
      algorithm: 'long',
    });
    expect(match.backtestIds).toEqual({ ADA: 1, SOL: 3 });
    expect(Object.keys(match.backtestIds)).toEqual(['ADA', 'SOL']);
    expect(match.missing).toEqual(['XRP', 'DOGE', 'LINK', 'AVAX']);
    expect(match.outsideBasket).toBe(1);
    expect(match.warnings).toEqual([]);
  });

  it('бэктест группы с началом от даты листинга не считается бэктестом за другой период', () => {
    const listedAt = '2023-01-01T00:00:00.000Z';
    const input = {
      backtests: [summary(1, 'ADA'), summary(2, 'DOGE', { from: listedAt })],
      basket: BASKET,
      quoteCurrency: 'USDT',
      run: RUN,
    };
    expect(matchGroupBacktestsToBasket(input).warnings.some((warning) => warning.includes('DOGE'))).toBe(true);
    const withListing = matchGroupBacktestsToBasket({ ...input, listingDates: { DOGEUSDT: Date.parse(listedAt) } });
    expect(withListing.backtestIds).toEqual({ ADA: 1, DOGE: 2 });
    expect(withListing.warnings).toEqual([]);
  });

  it('пропускает бэктесты с другой котировкой и берёт самый новый из повторов', () => {
    const match = matchGroupBacktestsToBasket({
      backtests: [summary(1, 'ADA'), summary(5, 'ADA'), summary(2, 'XRP', { quote: 'USDC' })],
      basket: BASKET,
      quoteCurrency: 'USDT',
      run: RUN,
    });
    expect(match.backtestIds).toEqual({ ADA: 5 });
    expect(
      match.warnings.some((warning) => warning.includes('другой котировкой') && warning.includes('XRP/USDC')),
    ).toBe(true);
    expect(match.warnings.some((warning) => warning.includes('несколько бэктестов для ADA'))).toBe(true);
  });

  it('предупреждает о другом периоде, учёте теней и алгоритме, но результаты использует', () => {
    const shiftedFrom = new Date(Date.parse(RUN.periodFrom) + 30 * 24 * 3600 * 1000).toISOString();
    const match = matchGroupBacktestsToBasket({
      backtests: [
        summary(1, 'ADA', { from: shiftedFrom }),
        summary(2, 'SOL', { useWicks: false }),
        summary(3, 'LINK', { algorithm: 'SHORT' }),
      ],
      basket: BASKET,
      quoteCurrency: 'USDT',
      run: RUN,
      algorithm: 'LONG',
    });
    expect(match.backtestIds).toEqual({ ADA: 1, SOL: 2, LINK: 3 });
    expect(match.warnings).toHaveLength(3);
    expect(match.warnings[0]).toContain('Период бэктестов группы не совпадает');
    expect(match.warnings[0]).toContain('ADA (2022-01-31 – 2024-01-01)');
    expect(match.warnings[1]).toContain('учёта теней (SOL)');
    expect(match.warnings[2]).toContain('LINK (SHORT)');
  });

  it('не берёт бэктесты с другими комиссиями, депозитом, биржей или настройками бота — монеты считаются недостающими', () => {
    const params = createParams();
    const match = matchGroupBacktestsToBasket({
      backtests: [
        summary(1, 'ADA', { commissions: { maker: 0, taker: 0 } }),
        summary(2, 'XRP', { deposit: { amount: 10, leverage: 10 } }),
        summary(3, 'SOL', { exchange: 'BINANCE_FUTURES' }),
        summary(4, 'DOGE', { strategy: groupStrategy(35) }),
        summary(5, 'LINK', {
          commissions: { maker: 0.02, taker: 0.055 },
          deposit: { amount: 100, leverage: 10 },
          exchange: 'bybit_futures',
          strategy: groupStrategy(40),
        }),
      ],
      basket: BASKET,
      quoteCurrency: 'USDT',
      run: RUN,
      algorithm: 'LONG',
      baseStrategy: createStrategy(),
      params,
    });
    expect(match.backtestIds).toEqual({ LINK: 5 });
    expect(match.missing).toEqual(['ADA', 'XRP', 'SOL', 'DOGE', 'AVAX']);
    expect(match.warnings.find((warning) => warning.includes('Комиссии'))).toContain('ADA (0/0%)');
    expect(match.warnings.find((warning) => warning.includes('Депозит'))).toContain('XRP (10 × 10)');
    expect(match.warnings.find((warning) => warning.includes('Биржа'))).toContain('SOL (BINANCE_FUTURES)');
    expect(match.warnings.find((warning) => warning.includes('Настройки бота'))).toContain('DOGE (RSI');
    expect(match.warnings.find((warning) => warning.includes('Настройки бота'))).toContain('35 вместо 40');
    for (const warning of match.warnings) {
      expect(warning).toContain('будут посчитаны бэктестами заново');
    }
  });

  it('из нескольких бэктестов монеты берёт самый новый совместимый', () => {
    const match = matchGroupBacktestsToBasket({
      backtests: [
        summary(1, 'ADA', { commissions: { maker: 0.02, taker: 0.055 } }),
        summary(9, 'ADA', { commissions: { maker: 0, taker: 0 } }),
        summary(4, 'ADA', { commissions: { maker: 0.02, taker: 0.055 } }),
      ],
      basket: BASKET,
      quoteCurrency: 'USDT',
      run: RUN,
    });
    expect(match.backtestIds).toEqual({ ADA: 4 });
    expect(match.warnings).toEqual([expect.stringContaining('несколько бэктестов для ADA')]);
  });

  it('неизвестные комиссии, депозит, биржа и настройки не проверяются', () => {
    const match = matchGroupBacktestsToBasket({
      backtests: [
        summary(1, 'ADA', { commissions: null, deposit: null, exchange: null, strategy: null }),
        summary(2, 'SOL', { commissions: { maker: null, taker: null }, deposit: { amount: null, leverage: null } }),
      ],
      basket: BASKET,
      quoteCurrency: 'USDT',
      run: RUN,
      baseStrategy: createStrategy(),
      params: createParams(),
    });
    expect(match.backtestIds).toEqual({ ADA: 1, SOL: 2 });
    expect(match.warnings).toEqual([]);
  });

  it('не предупреждает о расхождении периода в пределах допуска', () => {
    const almost = new Date(Date.parse(RUN.periodTo) - GROUP_PERIOD_TOLERANCE_MS + 1).toISOString();
    const match = matchGroupBacktestsToBasket({
      backtests: [summary(1, 'ADA', { to: almost }), summary(2, 'SOL', { from: null })],
      basket: BASKET,
      quoteCurrency: 'USDT',
      run: RUN,
    });
    expect(match.warnings).toEqual([]);
  });
});

describe('loadGroupBacktestsFromStore', () => {
  const baseConfig: BacktestConfigDto = {
    id: 1,
    name: 'Config',
    symbol: 'ADA/USDT',
    exchange: 'BYBIT_FUTURES',
    algorithm: 'LONG',
    pullUp: 0,
    portion: 0,
    profit: null,
    deposit: { amount: 100, leverage: 10, marginType: 'CROSS', currency: 'USDT' },
    stopLoss: null,
    settings: { type: 'CUSTOM' },
    conditions: [],
    from: RUN.periodFrom,
    to: RUN.periodTo,
    status: 'FINISHED',
    commissions: { maker: 0.02, taker: 0.055 },
    public: false,
    useWicks: false,
    cursor: 'cursor',
  };
  const baseStatistics: BacktestStatistics = {
    id: 1,
    name: 'Backtest',
    date: RUN.periodTo,
    from: RUN.periodFrom,
    to: RUN.periodTo,
    algorithm: 'LONG',
    exchange: 'BYBIT_FUTURES',
    symbol: 'ADA/USDT',
    base: 'ADA',
    quote: 'USDT',
    duration: 0,
    profitBase: 0,
    profitQuote: 0,
    netBase: 0,
    netQuote: 0,
    netBasePerDay: 0,
    netQuotePerDay: 0,
    minProfitBase: 0,
    maxProfitBase: 0,
    avgProfitBase: 0,
    minProfitQuote: 0,
    maxProfitQuote: 0,
    avgProfitQuote: 0,
    volume: 0,
    minDuration: 0,
    maxDuration: 0,
    avgDuration: 0,
    profits: 0,
    losses: 0,
    breakevens: 0,
    pullUps: 0,
    winRateProfits: 0,
    winRateLosses: 0,
    totalDeals: 0,
    minGrid: 0,
    maxGrid: 0,
    avgGrid: 0,
    minProfit: 0,
    maxProfit: 0,
    avgProfit: 0,
    mfePercent: 0,
    mfeAbsolute: 0,
    maePercent: 0,
    maeAbsolute: 0,
    commissionBase: 0,
    commissionQuote: 0,
  };

  it('бросает ошибку, если группы нет', async () => {
    await expect(loadGroupBacktestsFromStore('missing')).rejects.toThrow('не найдена');
  });

  it('читает детали бэктестов группы: монету, период, тени; ошибки — в failed', async () => {
    writeBacktestGroups([{ id: 'g1', name: ' База LONG ', backtestIds: [13, 11, 12], createdAt: 1, updatedAt: 2 }]);
    getBacktestDetailMock.mockImplementation(async (id) => {
      if (id === 11) {
        return { statistics: { ...baseStatistics, id }, config: { ...baseConfig, id } };
      }
      if (id === 12) {
        return {
          statistics: { ...baseStatistics, id, base: '', symbol: 'SOL/USDT', from: '' },
          config: { ...baseConfig, id, from: '2022-02-01T00:00:00.000Z' },
        };
      }
      throw new Error('HTTP 500 Internal Server Error');
    });

    const result = await loadGroupBacktestsFromStore('g1');
    expect(result.groupName).toBe('База LONG');
    const strategy = {
      conditions: [],
      settings: { type: 'CUSTOM' },
      profit: null,
      stopLoss: null,
      pullUp: 0,
    };
    expect(result.backtests).toEqual([
      {
        backtestId: 11,
        base: 'ADA',
        quote: 'USDT',
        from: RUN.periodFrom,
        to: RUN.periodTo,
        useWicks: false,
        algorithm: 'LONG',
        commissions: { maker: 0.02, taker: 0.055 },
        deposit: { amount: 100, leverage: 10 },
        exchange: 'BYBIT_FUTURES',
        strategy,
      },
      {
        backtestId: 12,
        base: 'SOL',
        quote: 'USDT',
        from: '2022-02-01T00:00:00.000Z',
        to: RUN.periodTo,
        useWicks: false,
        algorithm: 'LONG',
        commissions: { maker: 0.02, taker: 0.055 },
        deposit: { amount: 100, leverage: 10 },
        exchange: 'BYBIT_FUTURES',
        strategy,
      },
    ]);
    expect(result.failed).toEqual([{ backtestId: 13, error: 'HTTP 500 Internal Server Error' }]);
  });
});

describe('сохранённое состояние', () => {
  const createSavedState = async (): Promise<PortfolioOptimizerState> => {
    const harness = createHarness();
    const optimizer = new PortfolioOptimizer(createInput({}, { generations: 1 }), harness.callbacks, harness.deps);
    await optimizer.start();
    return optimizer.getState();
  };

  it('сохраняет, читает и удаляет состояние в localStorage', async () => {
    const state = await createSavedState();
    expect(loadSavedPortfolioState()).toBeNull();
    expect(savePortfolioState(state)).toBe(true);
    expect(loadSavedPortfolioState()).toEqual(state);
    clearSavedPortfolioState();
    expect(loadSavedPortfolioState()).toBeNull();
  });

  it('игнорирует повреждённое состояние и состояние другой версии', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    window.localStorage.setItem(PORTFOLIO_OPTIMIZER_STATE_KEY, '{oops');
    expect(loadSavedPortfolioState()).toBeNull();
    expect(warn).toHaveBeenCalledTimes(1);
    const state = await createSavedState();
    window.localStorage.setItem(PORTFOLIO_OPTIMIZER_STATE_KEY, JSON.stringify({ ...state, version: 2 }));
    expect(loadSavedPortfolioState()).toBeNull();
    window.localStorage.setItem(PORTFOLIO_OPTIMIZER_STATE_KEY, JSON.stringify({ ...state, status: 'flying' }));
    expect(loadSavedPortfolioState()).toBeNull();
  });
});

describe('подписи, названия и ошибки авторизации', () => {
  it('candidateShortLabel', () => {
    const state = { runId: 'po123', baseCandidateId: 'po123-base' };
    expect(candidateShortLabel(state, { id: 'po123-base' })).toBe('база');
    expect(candidateShortLabel(state, { id: 'po123-g3-2' })).toBe('g3-2');
    expect(candidateShortLabel(state, { id: 'foreign-candidate-42' })).toBe('ate-42');
  });

  it('buildCandidateBacktestName укладывается в 80 символов и сохраняет метку и монету', () => {
    expect(buildCandidateBacktestName('LONG  22', 'g1-2', 'ada')).toBe('PO LONG 22 g1-2 ADA');
    expect(buildCandidateBacktestName('  ', 'base', 'SOL')).toBe('PO base SOL');
    const long = buildCandidateBacktestName('Очень длинное название бота '.repeat(6), 'g12-6', 'DOGE');
    expect(long.length).toBeLessThanOrEqual(MAX_BACKTEST_NAME_LENGTH);
    expect(long.startsWith('PO Очень длинное')).toBe(true);
    expect(long.endsWith(' g12-6 DOGE')).toBe(true);
  });

  it('isAuthError распознаёт 401/403, но не номера бэктестов', () => {
    expect(isAuthError(new Error('HTTP 401 Unauthorized'))).toBe(true);
    expect(isAuthError(new BacktestSubmitError('Доступ запрещён', 403, null))).toBe(true);
    expect(isAuthError(new Error('Forbidden'))).toBe(true);
    expect(isAuthError(new Error('Бэктест 401 (ADA) завершился с ошибкой: статус FAILED'))).toBe(false);
    expect(isAuthError(new Error('HTTP 500 Internal Server Error'))).toBe(false);
  });
});

// ═══════════════════════════════════════════════════════════════════
// ОРКЕСТРАТОР
// ═══════════════════════════════════════════════════════════════════

describe('PortfolioOptimizer: основной сценарий', () => {
  it('оценивает базу на полной корзине, отбор базы берёт из тех же бэктестов и проводит поколения', async () => {
    const harness = createHarness();
    const optimizer = new PortfolioOptimizer(createInput(), harness.callbacks, harness.deps);
    await optimizer.start();
    const state = optimizer.getState();
    const base = findBase(state);

    expect(state.status).toBe('completed');
    expect(state.quoteCurrency).toBe('USDT');
    expect(base.hash).toBe(BASE_HASH);
    expect(runsFor(harness, base.hash).map((job) => job.symbol)).toEqual(BASKET);
    expect(base.full?.status).toBe('done');
    expect(base.screen?.status).toBe('done');
    expect(idsInOrder(base.screen, SCREEN)).toEqual(
      SCREEN.map((symbol) => requireEvaluation(base.full).backtestIds[symbol]),
    );
    expect(requireScore(base.full)).toBe(80);

    // Первые вызовы оценки: полная корзина базы, затем её отборочная корзина — в порядке корзин.
    expect(harness.evaluator.calls[0]).toEqual({
      ids: idsInOrder(base.full, BASKET),
      block: 60,
      run: { periodFrom: RUN.periodFrom, periodTo: RUN.periodTo },
      folds: 0,
    });
    expect(harness.evaluator.calls[1].ids).toEqual(idsInOrder(base.full, SCREEN));

    const candidates = nonBase(state);
    expect(candidates.map((candidate) => candidate.id)).toEqual([
      `${state.runId}-g1-1`,
      `${state.runId}-g1-2`,
      `${state.runId}-g1-3`,
      `${state.runId}-g2-1`,
      `${state.runId}-g2-2`,
      `${state.runId}-g2-3`,
    ]);
    for (const candidate of candidates) {
      expect(candidate.changed).toEqual([RSI_ID]);
      expect(candidate.mutated).toEqual([RSI_ID]);
      expect(candidate.screen?.status).toBe('done');
      expect(state.candidates.some((item) => item.id === candidate.parentId)).toBe(true);
    }

    const best = findBest(state);
    expect(best.id).not.toBe(base.id);
    expect(requireScore(best.full)).toBeGreaterThan(requireScore(base.full) * 1.01);
    expect(state.generation).toBe(2);
    expect(state.paramStats[RSI_ID].tries).toBe(candidates.length);
    expect(state.paramStats[RSI_ID].wins).toBeGreaterThanOrEqual(1);
    expect(state.inFlightJobs).toEqual([]);
    expect(hasLog(harness, 'success', 'Новый лучший')).toBe(true);
    expect(hasLog(harness, 'success', 'Оптимизация завершена')).toBe(true);
  });

  it('полная корзина переиспользует бэктесты отбора, а лимит учитывает только реальные создания', async () => {
    const harness = createHarness();
    const optimizer = new PortfolioOptimizer(createInput({}, { generations: 3 }), harness.callbacks, harness.deps);
    await optimizer.start();
    const state = optimizer.getState();
    const candidates = nonBase(state);

    let promoted = 0;
    for (const candidate of candidates) {
      const runs = runsFor(harness, candidate.hash).map((job) => job.symbol);
      if (candidate.full) {
        promoted += 1;
        expect([...runs].sort()).toEqual([...BASKET].sort());
        for (const symbol of SCREEN) {
          expect(candidate.full.backtestIds[symbol]).toBe(candidate.screen?.backtestIds[symbol]);
        }
        // В оценку id идут по возрастанию — как их хранит группа бэктестов.
        const fullIds = idsInOrder(candidate.full, BASKET).sort((left, right) => left - right);
        expect(harness.evaluator.calls.some((call) => call.ids.join() === fullIds.join())).toBe(true);
      } else {
        expect(runs).toEqual(SCREEN);
      }
    }

    const jobs = submittedJobs(harness);
    expect(new Set(jobs.map((job) => job.key)).size).toBe(jobs.length);
    expect(promoted).toBeGreaterThanOrEqual(1);
    expect(promoted).toBeLessThanOrEqual(3);
    expect(state.backtestsSubmitted).toBe(jobs.length);
    expect(state.backtestsSubmitted).toBe(
      BASKET.length + candidates.length * SCREEN.length + promoted * (BASKET.length - SCREEN.length),
    );
    for (const call of harness.evaluator.calls) {
      expect([SCREEN.length, BASKET.length]).toContain(call.ids.length);
    }
  });

  it('создаёт payload бэктеста для каждой монеты из пропатченной стратегии', async () => {
    const harness = createHarness();
    const input = createInput({}, { generations: 1 });
    const optimizer = new PortfolioOptimizer(input, harness.callbacks, harness.deps);
    await optimizer.start();
    const state = optimizer.getState();

    const baseAda = runsFor(harness, BASE_HASH).find((job) => job.symbol === 'ADA');
    expect(baseAda?.payload).toMatchObject({
      id: null,
      name: 'PO LONG 22 base ADA',
      symbol: 'ADA/USDT',
      symbols: ['ADA/USDT'],
      pair: { exchange: 'BYBIT_FUTURES', type: 'FUTURES', from: 'ADA', to: 'USDT', symbol: 'ADAUSDT' },
      commissions: { maker: '0.02', taker: '0.055' },
      useWicks: true,
      from: RUN.periodFrom,
      to: RUN.periodTo,
      public: false,
      status: 'FINISHED',
    });
    expect(baseAda?.payload?.substatus).toBeUndefined();
    expect(readRsi(baseAda?.payload ?? null)).toBe(40);

    const candidate = nonBase(state)[0];
    const job = runsFor(harness, candidate.hash).find((item) => item.symbol === 'SOL');
    expect(job?.payload?.name).toBe('PO LONG 22 g1-1 SOL');
    expect(readRsi(job?.payload ?? null)).toBe(candidate.values[RSI_ID]);
    expect(job?.payload?.conditions?.[1]?.value).toBe(-100);
    // Исходная стратегия не меняется.
    expect(input.baseStrategy).toEqual(createStrategy());
  });

  it('передаёт число под-периодов для устойчивой цели', async () => {
    const harness = createHarness();
    const optimizer = new PortfolioOptimizer(
      createInput({ objective: { ...DEFAULT_OBJECTIVE, kind: 'robust', folds: 4, minDeals: 0 } }, { generations: 1 }),
      harness.callbacks,
      harness.deps,
    );
    await optimizer.start();
    expect(harness.evaluator.calls.length).toBeGreaterThan(0);
    expect(harness.evaluator.calls.every((call) => call.folds === 4)).toBe(true);
  });

  it('детерминирован при одинаковом seed', async () => {
    const runOnce = async (): Promise<PortfolioOptimizerState> => {
      const harness = createHarness();
      const optimizer = new PortfolioOptimizer(createInput({}, { generations: 3 }), harness.callbacks, harness.deps);
      await optimizer.start();
      return optimizer.getState();
    };
    const first = await runOnce();
    const second = await runOnce();
    expect(second.candidates.map((candidate) => candidate.hash)).toEqual(
      first.candidates.map((candidate) => candidate.hash),
    );
    expect(second.bestCandidateId).toBe(first.bestCandidateId);
  });

  it('завершает запуск, когда новых кандидатов не осталось', async () => {
    const harness = createHarness();
    const params = createParams((param) => (param.id === RSI_ID ? { ...param, min: 38, max: 42 } : param));
    const optimizer = new PortfolioOptimizer(
      createInput({ params }, { generations: 5, candidatesPerGeneration: 3 }),
      harness.callbacks,
      harness.deps,
    );
    await optimizer.start();
    const state = optimizer.getState();
    expect(state.status).toBe('completed');
    expect(
      nonBase(state)
        .map((candidate) => candidate.values[RSI_ID])
        .sort(),
    ).toEqual([38, 42]);
    expect(state.generation).toBe(1);
    expect(hasLog(harness, 'info', 'Новых кандидатов нет')).toBe(true);
  });

  it('сообщает о прогрессе, оценке окончания и изменениях кандидатов (копиями)', async () => {
    const harness = createHarness();
    const optimizer = new PortfolioOptimizer(createInput(), harness.callbacks, harness.deps);
    expect(optimizer.getProgress()).toMatchObject({ status: 'idle', startedAt: null, estimatedEndAt: null });
    await optimizer.start();

    const running = harness.progress.filter((snapshot) => snapshot.status === 'running');
    expect(running.length).toBeGreaterThan(0);
    expect(running.some((snapshot) => snapshot.estimatedEndAt !== null)).toBe(true);
    expect(running.some((snapshot) => snapshot.generation === 2)).toBe(true);
    const last = harness.progress.at(-1);
    expect(last).toMatchObject({
      status: 'completed',
      generation: 2,
      totalGenerations: 2,
      backtestsLimit: 500,
      inFlight: 0,
      queued: 0,
      candidatesEvaluated: 6,
      estimatedEndAt: null,
    });
    expect(last?.backtestsSubmitted).toBe(optimizer.getState().backtestsSubmitted);

    const update = harness.candidateUpdates.at(-1);
    expect(update?.bestCandidateId).toBe(optimizer.getState().bestCandidateId);
    update?.candidates.splice(0);
    expect(optimizer.getState().candidates).toHaveLength(7);
    const snapshot = optimizer.getState();
    snapshot.candidates.splice(0);
    expect(optimizer.getState().candidates).toHaveLength(7);
  });

  it('patchCandidate возвращает стратегию с параметрами кандидата', async () => {
    const harness = createHarness();
    const optimizer = new PortfolioOptimizer(createInput(), harness.callbacks, harness.deps);
    await optimizer.start();
    const state = optimizer.getState();
    const best = findBest(state);
    const patched = optimizer.patchCandidate(best.id);
    expect(patched.conditions?.[0]?.value).toBe(best.values[RSI_ID]);
    expect(optimizer.patchCandidate(state.baseCandidateId)).toEqual(createStrategy());
    expect(() => optimizer.patchCandidate('nope')).toThrow('не найден');
  });

  it('повторный start() во время запуска возвращает тот же промис; журнал планировщика попадает в журнал', async () => {
    const harness = createHarness();
    const optimizer = new PortfolioOptimizer(createInput({}, { generations: 1 }), harness.callbacks, harness.deps);
    const first = optimizer.start();
    expect(optimizer.start()).toBe(first);
    expect(optimizer.isRunning).toBe(true);
    harness.scheduler().events.onLog('warning', 'Veles ограничил частоту создания бэктестов');
    await first;
    expect(optimizer.isRunning).toBe(false);
    expect(hasLog(harness, 'warning', 'Veles ограничил частоту')).toBe(true);
  });

  it('updateSchedulerConfig меняет настройки и передаёт их планировщику', async () => {
    const harness = createHarness({ behavior: { hold: () => true } });
    const optimizer = new PortfolioOptimizer(createInput({}, { generations: 0 }), harness.callbacks, harness.deps);
    optimizer.updateSchedulerConfig({ maxInFlight: 5 });
    expect(optimizer.getState().config.scheduler.maxInFlight).toBe(5);
    const done = optimizer.start();
    expect(harness.scheduler().config.maxInFlight).toBe(5);
    optimizer.updateSchedulerConfig({ submitDelaySeconds: 45 });
    expect(harness.scheduler().configUpdates.at(-1)).toMatchObject({ maxInFlight: 5, submitDelaySeconds: 45 });
    optimizer.stop();
    await done;
  });

  it('один раз предупреждает, если состояние не сохраняется, и продолжает работу', async () => {
    const harness = createHarness();
    const optimizer = new PortfolioOptimizer(createInput({}, { generations: 1 }), harness.callbacks, {
      ...harness.deps,
      saveState: () => false,
    });
    await optimizer.start();
    expect(harness.logs.filter((entry) => entry.message.includes('Не удалось сохранить состояние'))).toHaveLength(1);
    expect(hasLog(harness, 'warning', 'устаревшая сохранённая копия удалена')).toBe(true);
    // Устаревшая копия удаляется при каждой неудаче: её нельзя продолжить с неверным числом бэктестов.
    expect(harness.clearSavedState).toHaveBeenCalled();
    expect(harness.clearSavedState.mock.calls.length).toBeGreaterThan(1);
    expect(optimizer.getState().status).toBe('completed');
  });

  it('ошибки в обработчиках UI не прерывают запуск', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const harness = createHarness();
    const throwing: PortfolioOptimizerCallbacks = {
      onLog: () => {
        throw new Error('UI упал');
      },
      onProgress: () => {
        throw new Error('UI упал');
      },
      onCandidatesChanged: () => {
        throw new Error('UI упал');
      },
    };
    const optimizer = new PortfolioOptimizer(createInput({}, { generations: 1 }), throwing, harness.deps);
    await optimizer.start();
    expect(optimizer.getState().status).toBe('completed');
    expect(warn).toHaveBeenCalled();
  });

  it('пауза, продолжение и остановка без активного запуска ничего не делают', () => {
    const harness = createHarness();
    const optimizer = new PortfolioOptimizer(createInput(), harness.callbacks, harness.deps);
    optimizer.pause();
    optimizer.resume();
    optimizer.stop();
    expect(optimizer.getState().status).toBe('idle');
    expect(optimizer.isRunning).toBe(false);
    expect(harness.saves).toHaveLength(0);
  });

  it('проверяет входные данные', () => {
    const harness = createHarness();
    expect(
      () =>
        new PortfolioOptimizer(
          { config: createConfig({ basket: [] }), baseStrategy: createStrategy(), quoteCurrency: 'USDT' },
          harness.callbacks,
          harness.deps,
        ),
    ).toThrow('Корзина монет пуста');
    expect(
      () =>
        new PortfolioOptimizer(
          { config: createConfig(), baseStrategy: createStrategy(), quoteCurrency: ' ' },
          harness.callbacks,
          harness.deps,
        ),
    ).toThrow('валюта котировки');
  });
});

describe('PortfolioOptimizer: результаты базы из группы бэктестов', () => {
  const groupSummary = (backtestId: number, base: string, patch: Partial<GroupBacktestSummary> = {}) => ({
    backtestId,
    base,
    quote: 'USDT',
    from: RUN.periodFrom,
    to: RUN.periodTo,
    useWicks: true,
    algorithm: 'LONG',
    ...patch,
  });

  it('подставляет бэктесты группы по монете, досчитывает недостающие и предупреждает о другом периоде', async () => {
    const harness = createHarness();
    const shiftedFrom = new Date(Date.parse(RUN.periodFrom) + 20 * 24 * 3600 * 1000).toISOString();
    harness.veles.register(501, 'ADA', 40);
    harness.veles.register(503, 'SOL', 40);
    harness.veles.register(505, 'LINK', 40);
    harness.veles.register(507, 'XRP', 40);
    harness.loadGroupBacktests.mockResolvedValue({
      groupName: 'База LONG 22',
      backtests: [
        groupSummary(501, 'ADA'),
        groupSummary(503, 'SOL', { from: shiftedFrom }),
        groupSummary(505, 'LINK'),
        groupSummary(507, 'XRP', { quote: 'USDC' }),
        groupSummary(509, 'BTC'),
      ],
      failed: [{ backtestId: 511, error: 'HTTP 404 Not Found' }],
    });

    const optimizer = new PortfolioOptimizer(
      createInput({ baseGroupId: 'group-7' }, { generations: 1 }),
      harness.callbacks,
      harness.deps,
    );
    await optimizer.start();
    const state = optimizer.getState();
    const base = findBase(state);

    expect(harness.loadGroupBacktests).toHaveBeenCalledWith('group-7');
    expect(runsFor(harness, base.hash).map((job) => job.symbol)).toEqual(['XRP', 'DOGE', 'AVAX']);
    const full = requireEvaluation(base.full);
    expect(full.status).toBe('done');
    expect(full.backtestIds).toMatchObject({ ADA: 501, SOL: 503, LINK: 505 });
    // id корзины передаются в оценку по возрастанию — как их хранит и складывает группа бэктестов.
    expect(harness.evaluator.calls[0].ids).toEqual([501, 503, 505, 1000, 1001, 1002]);
    expect(harness.evaluator.calls[1].ids).toEqual([501, 503, 505]);
    expect(state.backtestCache[`${base.hash}|ADA`]).toBe(501);

    expect(hasLog(harness, 'warning', 'Период бэктестов группы не совпадает')).toBe(true);
    expect(hasLog(harness, 'warning', 'другой котировкой')).toBe(true);
    expect(hasLog(harness, 'warning', 'Не удалось загрузить бэктесты группы (511)')).toBe(true);
    expect(hasLog(harness, 'info', 'Результаты базы из группы «База LONG 22»: 3 из 6 монет')).toBe(true);
    expect(state.status).toBe('completed');
  });

  it('создаёт бэктесты базы заново, если группу загрузить не удалось', async () => {
    const harness = createHarness();
    harness.loadGroupBacktests.mockRejectedValue(new Error('Группа бэктестов с результатами базы не найдена'));
    const optimizer = new PortfolioOptimizer(
      createInput({ baseGroupId: 'gone' }, { generations: 0 }),
      harness.callbacks,
      harness.deps,
    );
    await optimizer.start();
    expect(runsFor(harness, BASE_HASH)).toHaveLength(BASKET.length);
    expect(hasLog(harness, 'warning', 'Бэктесты базы будут созданы заново')).toBe(true);
    expect(optimizer.getState().status).toBe('completed');
  });

  it('ошибка авторизации при загрузке группы останавливает запуск до создания бэктестов', async () => {
    const harness = createHarness();
    harness.loadGroupBacktests.mockRejectedValue(new Error('HTTP 403 Forbidden'));
    const optimizer = new PortfolioOptimizer(createInput({ baseGroupId: 'group-1' }), harness.callbacks, harness.deps);
    await optimizer.start();
    const state = optimizer.getState();
    expect(state.status).toBe('error');
    expect(findBase(state).full).toBeNull();
    expect(submittedJobs(harness)).toHaveLength(0);
    expect(hasLog(harness, 'error', 'Нет доступа к Veles')).toBe(true);
  });
});

describe('PortfolioOptimizer: ошибки и неполные корзины', () => {
  it('не оценивает частичную корзину: монета с ошибкой → этап failed без score, запуск продолжается', async () => {
    let failed = false;
    const harness = createHarness({
      behavior: {
        runError: (job) => {
          if (!(failed || job.key.startsWith(BASE_HASH)) && job.symbol === 'SOL') {
            failed = true;
            return new Error('Бэктест 1010 (SOL) завершился с ошибкой: статус FAILED');
          }
          return null;
        },
      },
    });
    const optimizer = new PortfolioOptimizer(createInput(), harness.callbacks, harness.deps);
    await optimizer.start();
    const state = optimizer.getState();

    const broken = nonBase(state).filter((candidate) => candidate.screen?.status === 'failed');
    expect(broken).toHaveLength(1);
    const screen = requireEvaluation(broken[0].screen);
    expect(screen.failedSymbols).toEqual(['SOL']);
    expect(screen.score).toBeNull();
    expect(screen.metrics).toBeNull();
    expect(screen.error).toContain('статус FAILED');
    expect(broken[0].full).toBeNull();
    const brokenIds = Object.values(screen.backtestIds);
    expect(harness.evaluator.calls.some((call) => call.ids.some((id) => brokenIds.includes(id)))).toBe(false);
    for (const call of harness.evaluator.calls) {
      expect([SCREEN.length, BASKET.length]).toContain(call.ids.length);
    }
    expect(state.status).toBe('completed');
    expect(state.paramStats[RSI_ID].tries).toBe(nonBase(state).length);
    expect(hasLog(harness, 'warning', 'не оценён — не посчитаны SOL')).toBe(true);
  });

  it('отменяет ещё не созданные бэктесты этапа после первой ошибки', async () => {
    let failedKey: string | null = null;
    const harness = createHarness({
      behavior: {
        maxInFlight: 1,
        manualPump: true,
        runError: (job) => {
          if (failedKey === null && !job.key.startsWith(BASE_HASH)) {
            failedKey = job.key;
            return new Error('Таймаут бэктеста');
          }
          return null;
        },
      },
    });
    const optimizer = new PortfolioOptimizer(createInput({}, { generations: 1 }), harness.callbacks, harness.deps);
    await drive(harness, optimizer.start());
    const state = optimizer.getState();

    const brokenHash = (failedKey ?? '').split('|')[0];
    const brokenJobs = runsFor(harness, brokenHash);
    expect(brokenJobs.map((job) => job.state)).toEqual(['failed', 'aborted', 'aborted']);
    expect(brokenJobs.filter((job) => job.backtestId !== null)).toHaveLength(1);
    const broken = state.candidates.find((candidate) => candidate.hash === brokenHash);
    expect(broken?.screen?.failedSymbols).toEqual(['ADA']);
    expect(state.backtestsSubmitted).toBe(submittedJobs(harness).length);
    expect(state.status).toBe('completed');
  });

  it('не учитывает в лимите бэктесты, которые не удалось создать', async () => {
    let rejected = false;
    const harness = createHarness({
      behavior: {
        submitError: (job) => {
          if (!(rejected || job.key.startsWith(BASE_HASH))) {
            rejected = true;
            return new BacktestSubmitError('HTTP 400 Bad Request: неверные параметры', 400, null);
          }
          return null;
        },
      },
    });
    const optimizer = new PortfolioOptimizer(createInput({}, { generations: 1 }), harness.callbacks, harness.deps);
    await optimizer.start();
    const state = optimizer.getState();
    expect(state.backtestsSubmitted).toBe(submittedJobs(harness).length);
    expect(nonBase(state).filter((candidate) => candidate.screen?.status === 'failed')).toHaveLength(1);
    expect(state.status).toBe('completed');
  });

  it('останавливает запуск с ошибкой, если базу не удалось оценить', async () => {
    const harness = createHarness({
      behavior: {
        runError: (job) => (job.symbol === 'DOGE' ? new Error('Бэктест 1003 (DOGE) завершился с ошибкой') : null),
      },
    });
    const optimizer = new PortfolioOptimizer(createInput(), harness.callbacks, harness.deps);
    await optimizer.start();
    const state = optimizer.getState();
    expect(state.status).toBe('error');
    expect(findBase(state).full?.failedSymbols).toEqual(['DOGE']);
    expect(nonBase(state)).toHaveLength(0);
    expect(hasLog(harness, 'error', 'Базовый бот не оценён на полной корзине')).toBe(true);
    expect(harness.lastSaved().status).toBe('error');
  });

  it('исключает из корзины монеты, для которых Veles не принимает настройки базового бота (HTTP 412)', async () => {
    const pairRejected = (symbol: string) =>
      new BacktestSubmitError(
        `HTTP 412: Объём ордера не соответствует требованиям по паре ${symbol}/USDT (Precondition Failed)`,
        412,
        null,
      );
    const harness = createHarness({
      behavior: {
        submitError: (job) => (job.symbol === 'SOL' ? pairRejected('SOL') : null),
      },
    });
    const basketChanges: Array<{ basket: string[]; screenBasket: string[]; excluded: string[] }> = [];
    const callbacks: PortfolioOptimizerCallbacks = {
      ...harness.callbacks,
      onBasketChanged: (config, excluded) =>
        basketChanges.push({
          basket: config.basket,
          screenBasket: config.screenBasket,
          excluded: excluded.map((item) => item.symbol),
        }),
    };
    const optimizer = new PortfolioOptimizer(createInput({}, { generations: 2 }), callbacks, harness.deps);
    await optimizer.start();
    const state = optimizer.getState();

    expect(state.status).toBe('completed');
    expect(state.config.basket).toEqual(['ADA', 'XRP', 'DOGE', 'LINK', 'AVAX']);
    // SOL был в отборочной корзине — её размер сохранён за счёт другой монеты полной корзины.
    expect(state.config.screenBasket).toHaveLength(SCREEN.length);
    expect(state.config.screenBasket).not.toContain('SOL');
    expect(state.config.screenBasket.slice(0, 2)).toEqual(['ADA', 'LINK']);
    expect(state.config.basket).toEqual(expect.arrayContaining(state.config.screenBasket));
    expect(state.excludedSymbols?.map((item) => item.symbol)).toEqual(['SOL']);
    expect(state.excludedSymbols?.[0].reason).toContain('HTTP 412');

    const base = findBase(state);
    expect(base.full?.status).toBe('done');
    expect(base.full?.symbols).not.toContain('SOL');
    expect(Object.keys(base.full?.backtestIds ?? {})).not.toContain('SOL');
    expect(base.screen?.symbols).toEqual(state.config.screenBasket);
    for (const candidate of nonBase(state)) {
      expect(candidate.screen?.symbols ?? []).not.toContain('SOL');
      expect(candidate.full?.symbols ?? []).not.toContain('SOL');
    }
    expect(nonBase(state).some((candidate) => candidate.full?.status === 'done')).toBe(true);
    expect(hasLog(harness, 'warning', 'Исключены из корзины')).toBe(true);
    expect(basketChanges).toEqual([
      { basket: state.config.basket, screenBasket: state.config.screenBasket, excluded: ['SOL'] },
    ]);
    expect(harness.lastSaved().excludedSymbols?.map((item) => item.symbol)).toEqual(['SOL']);
  });

  it('останавливает запуск, если Veles отказал в настройках бота для большей части корзины', async () => {
    const rejectedSymbols = new Set(['ADA', 'XRP', 'SOL', 'DOGE']);
    const harness = createHarness({
      behavior: {
        submitError: (job) =>
          rejectedSymbols.has(job.symbol)
            ? new BacktestSubmitError('HTTP 412: Объём ордера не соответствует требованиям по паре', 412, null)
            : null,
      },
    });
    const optimizer = new PortfolioOptimizer(createInput(), harness.callbacks, harness.deps);
    await optimizer.start();
    const state = optimizer.getState();

    expect(state.status).toBe('error');
    expect(hasLog(harness, 'error', 'Veles не принимает настройки бота для 4 из 6 монет')).toBe(true);
    expect(state.config.basket).toEqual(BASKET);
    expect(state.excludedSymbols ?? []).toEqual([]);
    expect(nonBase(state)).toHaveLength(0);
  });

  it('отказ Veles (HTTP 412) для кандидата — неудача его этапа, корзина не меняется', async () => {
    let rejected = false;
    const harness = createHarness({
      behavior: {
        submitError: (job) => {
          if (!(rejected || job.key.startsWith(BASE_HASH))) {
            rejected = true;
            return new BacktestSubmitError('HTTP 412: Объём ордера не соответствует требованиям по паре', 412, null);
          }
          return null;
        },
      },
    });
    const optimizer = new PortfolioOptimizer(createInput({}, { generations: 1 }), harness.callbacks, harness.deps);
    await optimizer.start();
    const state = optimizer.getState();

    expect(state.status).toBe('completed');
    expect(state.config.basket).toEqual(BASKET);
    expect(state.excludedSymbols ?? []).toEqual([]);
    expect(nonBase(state).filter((candidate) => candidate.screen?.status === 'failed')).toHaveLength(1);
  });

  it('отправляет бэктесты без API-ключа бота и с депозитом для бэктестов, не меняя сам бот', async () => {
    const harness = createHarness();
    const baseStrategy = Object.assign(createStrategy(), { apiKey: 5 });
    const optimizer = new PortfolioOptimizer(
      { ...createInput({ backtestDeposit: 37.5 }, { generations: 1 }), baseStrategy },
      harness.callbacks,
      harness.deps,
    );
    await optimizer.start();
    const payloads = submittedJobs(harness).map((job) => job.payload);

    expect(payloads.length).toBeGreaterThan(0);
    for (const payload of payloads) {
      expect(payload && 'apiKey' in payload).toBe(false);
      expect(payload?.deposit?.amount).toBe(37.5);
      expect(payload?.deposit?.leverage).toBe(10);
    }
    const state = optimizer.getState();
    expect(state.config.backtestDeposit).toBe(37.5);
    expect(state.baseStrategy.deposit?.amount).toBe(100);
    expect('apiKey' in state.baseStrategy).toBe(true);
  });

  it('без депозита для бэктестов отправляет депозит бота', async () => {
    const harness = createHarness();
    const optimizer = new PortfolioOptimizer(createInput({}, { generations: 1 }), harness.callbacks, harness.deps);
    await optimizer.start();
    expect(optimizer.getState().config.backtestDeposit).toBeNull();
    expect(submittedJobs(harness).every((job) => job.payload?.deposit?.amount === 100)).toBe(true);
  });

  it('начинает бэктест монеты с даты листинга и исключает монеты, не торговавшиеся в период', async () => {
    const harness = createHarness();
    const dogeListing = Date.parse('2023-01-01T00:00:00.000Z');
    harness.loadListingDates.mockResolvedValue({
      DOGEUSDT: dogeListing,
      AVAXUSDT: Date.parse('2025-01-01T00:00:00.000Z'),
      ADAUSDT: Date.parse('2018-01-01T00:00:00.000Z'),
    });
    const optimizer = new PortfolioOptimizer(createInput({}, { generations: 1 }), harness.callbacks, harness.deps);
    await optimizer.start();
    const state = optimizer.getState();

    expect(harness.loadListingDates).toHaveBeenCalledTimes(1);
    expect(harness.loadListingDates).toHaveBeenCalledWith('BYBIT_FUTURES');
    expect(state.status).toBe('completed');
    expect(state.config.basket).toEqual(['ADA', 'XRP', 'SOL', 'DOGE', 'LINK']);
    expect(state.excludedSymbols?.map((item) => item.symbol)).toEqual(['AVAX']);
    expect(state.excludedSymbols?.[0].reason).toContain('2025-01-01');
    expect(state.listingDates?.DOGEUSDT).toBe(dogeListing);
    const jobs = submittedJobs(harness);
    expect(jobs.some((job) => job.symbol === 'AVAX')).toBe(false);
    for (const job of jobs) {
      expect(job.payload?.from).toBe(job.symbol === 'DOGE' ? '2023-01-01T00:00:00.000Z' : RUN.periodFrom);
    }
    expect(hasLog(harness, 'info', 'DOGE (с 2023-01-01)')).toBe(true);
    expect(hasLog(harness, 'warning', 'не торговались в выбранный период')).toBe(true);
  });

  it('если даты листинга не загрузились, считает все монеты за период формы', async () => {
    const harness = createHarness();
    harness.loadListingDates.mockRejectedValue(new Error('HTTP 500 Internal Server Error'));
    const optimizer = new PortfolioOptimizer(createInput({}, { generations: 1 }), harness.callbacks, harness.deps);
    await optimizer.start();
    const state = optimizer.getState();

    expect(state.status).toBe('completed');
    expect(state.config.basket).toEqual(BASKET);
    expect(state.listingDates).toEqual({});
    expect(submittedJobs(harness).every((job) => job.payload?.from === RUN.periodFrom)).toBe(true);
    expect(hasLog(harness, 'warning', 'Не удалось загрузить даты листинга')).toBe(true);
  });

  it('ошибка авторизации переводит запуск в error, этап остаётся незавершённым и досчитывается после продолжения', async () => {
    const veles = new FakeVeles();
    let unauthorized = false;
    const first = createHarness({
      veles,
      behavior: {
        runError: (job) => {
          if (!(unauthorized || job.key.startsWith(BASE_HASH))) {
            unauthorized = true;
            return new Error('HTTP 401 Unauthorized');
          }
          return null;
        },
      },
    });
    const optimizer = new PortfolioOptimizer(createInput({}, { generations: 1 }), first.callbacks, first.deps);
    await optimizer.start();
    const stopped = optimizer.getState();
    expect(stopped.status).toBe('error');
    expect(hasLog(first, 'error', 'Нет доступа к Veles')).toBe(true);
    expect(nonBase(stopped).some((candidate) => candidate.screen?.status === 'failed')).toBe(false);
    expect(nonBase(stopped).some((candidate) => candidate.screen?.status === 'running')).toBe(true);

    const second = createHarness({ veles, startTime: START_TIME + 3_600_000 });
    const resumed = PortfolioOptimizer.fromSaved(first.lastSaved(), second.callbacks, second.deps);
    await resumed.start();
    const state = resumed.getState();
    expect(state.status).toBe('completed');
    expect(nonBase(state).map((candidate) => candidate.id)).toEqual(nonBase(stopped).map((candidate) => candidate.id));
    expect(nonBase(state).every((candidate) => candidate.screen?.status === 'done')).toBe(true);
  });

  it('повторяет расчёт метрик при временном сбое загрузки результатов', async () => {
    const harness = createHarness();
    let target: string | null = null;
    let attempts = 0;
    harness.evaluator.failWhen = (call) => {
      const hash = harness.veles.hashOf(call.ids[0]);
      if (hash === BASE_HASH) {
        return null;
      }
      target ??= hash;
      if (hash === target && attempts === 0) {
        attempts += 1;
        return new Error('Таймаут ответа от вкладки');
      }
      return null;
    };
    const optimizer = new PortfolioOptimizer(createInput({}, { generations: 1 }), harness.callbacks, harness.deps);
    await optimizer.start();
    const candidate = optimizer.getState().candidates.find((item) => item.hash === target);
    expect(candidate?.screen?.status).toBe('done');
    expect(harness.sleep).toHaveBeenCalledWith(5_000, expect.any(AbortSignal));
    expect(hasLog(harness, 'warning', 'не удалось посчитать метрики (попытка 1/4)')).toBe(true);
  });

  it('после исчерпания попыток расчёта метрик запуск останавливается, а после продолжения этап пересчитывается из кеша', async () => {
    const veles = new FakeVeles();
    const first = createHarness({ veles });
    let target: string | null = null;
    first.evaluator.failWhen = (call) => {
      const hash = veles.hashOf(call.ids[0]);
      if (hash === BASE_HASH) {
        return null;
      }
      target ??= hash;
      return hash === target ? new Error('Не удалось загрузить результаты бэктеста: сеть недоступна') : null;
    };
    const optimizer = new PortfolioOptimizer(createInput({}, { generations: 1 }), first.callbacks, first.deps);
    await optimizer.start();
    const stopped = optimizer.getState();
    const candidate = stopped.candidates.find((item) => item.hash === target);

    // Этап не помечен неудачным: бэктесты готовы, не загрузились только результаты.
    expect(stopped.status).toBe('error');
    expect(candidate?.screen?.status).toBe('running');
    expect(candidate?.screen?.metricsFailures).toBe(1);
    expect(candidate?.screen?.error).toContain('сеть недоступна');
    expect(first.evaluator.calls.filter((call) => veles.hashOf(call.ids[0]) === target)).toHaveLength(4);
    expect(first.sleep.mock.calls.map(([ms]) => ms)).toEqual(expect.arrayContaining([5_000, 30_000, 120_000]));
    expect(hasLog(first, 'error', 'Veles или сеть недоступны')).toBe(true);
    expect(hasLog(first, 'error', 'продолжите запуск')).toBe(true);
    // Поколение не подведено — статистика параметров не испорчена сбоем.
    expect(stopped.generation).toBe(0);
    expect(stopped.paramStats).toEqual({});

    const second = createHarness({ veles, startTime: START_TIME + 3_600_000 });
    const resumed = PortfolioOptimizer.fromSaved(first.lastSaved(), second.callbacks, second.deps);
    await resumed.start();
    const state = resumed.getState();
    const rescored = state.candidates.find((item) => item.hash === target);
    expect(state.status).toBe('completed');
    expect(rescored?.screen?.status).toBe('done');
    expect(rescored?.screen?.metricsFailures).toBeUndefined();
    // Бэктесты отбора этого кандидата не создавались заново.
    expect(runsFor(second, target ?? '').filter((job) => SCREEN.includes(job.symbol))).toHaveLength(0);
  });

  it('если метрики не считаются и после продолжения, этап получает статус failed и запуск идёт дальше', async () => {
    const veles = new FakeVeles();
    const failing = (harness: Harness, target: { hash: string | null }) => {
      harness.evaluator.failWhen = (call) => {
        const hash = veles.hashOf(call.ids[0]);
        if (hash === BASE_HASH) {
          return null;
        }
        target.hash ??= hash;
        return hash === target.hash ? new Error('Не удалось загрузить результаты бэктеста: битые данные') : null;
      };
    };
    const target: { hash: string | null } = { hash: null };
    const first = createHarness({ veles });
    failing(first, target);
    const optimizer = new PortfolioOptimizer(createInput({}, { generations: 1 }), first.callbacks, first.deps);
    await optimizer.start();
    expect(optimizer.getState().status).toBe('error');

    const second = createHarness({ veles, startTime: START_TIME + 3_600_000 });
    failing(second, target);
    const resumed = PortfolioOptimizer.fromSaved(first.lastSaved(), second.callbacks, second.deps);
    await resumed.start();
    const state = resumed.getState();
    const candidate = state.candidates.find((item) => item.hash === target.hash);
    expect(candidate?.screen?.status).toBe('failed');
    expect(candidate?.screen?.error).toContain('битые данные');
    expect(state.status).toBe('completed');
  });
});

describe('PortfolioOptimizer: лимит бэктестов', () => {
  it('пропускает полную корзину, на которую не хватает лимита, и завершает запуск по лимиту', async () => {
    const harness = createHarness();
    const optimizer = new PortfolioOptimizer(
      createInput({}, { generations: 5, maxBacktests: 17 }),
      harness.callbacks,
      harness.deps,
    );
    await optimizer.start();
    const state = optimizer.getState();
    expect(state.status).toBe('completed');
    expect(state.backtestsSubmitted).toBe(6 + 3 * 3);
    expect(state.backtestsSubmitted).toBeLessThanOrEqual(17);
    expect(nonBase(state)).toHaveLength(3);
    expect(nonBase(state).every((candidate) => candidate.full === null)).toBe(true);
    expect(state.generation).toBe(1);
    expect(hasLog(harness, 'warning', 'полная корзина пропущена')).toBe(true);
    expect(hasLog(harness, 'warning', 'Достигнут лимит бэктестов: создано 15 из 17')).toBe(true);
  });

  it('уменьшает число кандидатов поколения до доступного лимита', async () => {
    const harness = createHarness();
    const optimizer = new PortfolioOptimizer(
      createInput({}, { generations: 5, maxBacktests: 12 }),
      harness.callbacks,
      harness.deps,
    );
    await optimizer.start();
    const state = optimizer.getState();
    expect(state.backtestsSubmitted).toBe(12);
    expect(nonBase(state)).toHaveLength(2);
    expect(hasLog(harness, 'warning', 'в поколении 1 проверяю 2 из 3 кандидатов')).toBe(true);
  });

  it('никогда не превышает лимит, даже если его не хватает на базу', async () => {
    const harness = createHarness();
    const optimizer = new PortfolioOptimizer(createInput({}, { maxBacktests: 4 }), harness.callbacks, harness.deps);
    await optimizer.start();
    const state = optimizer.getState();
    expect(state.backtestsSubmitted).toBeLessThanOrEqual(4);
    expect(state.status).toBe('error');
    expect(findBase(state).full?.error).toContain('исчерпан лимит бэктестов');
    // Причина — лимит: продолжать этот запуск бессмысленно, сообщение не предлагает этого.
    expect(hasLog(harness, 'error', 'не хватило лимита бэктестов (создано 4 из 4)')).toBe(true);
    expect(hasLog(harness, 'error', 'начните новый запуск')).toBe(true);
  });
});

describe('PortfolioOptimizer: пауза, остановка и продолжение', () => {
  it('сохраняет состояние после каждого созданного бэктеста', async () => {
    const harness = createHarness({ behavior: { hold: () => true } });
    const optimizer = new PortfolioOptimizer(createInput(), harness.callbacks, harness.deps);
    const done = optimizer.start();
    await waitFor(() => harness.scheduler().inFlight === BASKET.length, 'созданы бэктесты базы');

    const saved = harness.lastSaved();
    expect(saved.status).toBe('running');
    expect(saved.backtestsSubmitted).toBe(BASKET.length);
    expect(saved.inFlightJobs.map((job) => job.symbol)).toEqual(BASKET);
    expect(saved.inFlightJobs.every((job) => job.state === 'submitted' && job.backtestId !== null)).toBe(true);
    const counts = harness.saves.map((raw) => {
      const parsed: unknown = JSON.parse(raw);
      return isPortfolioOptimizerState(parsed) ? parsed.backtestsSubmitted : -1;
    });
    for (let submitted = 1; submitted <= BASKET.length; submitted += 1) {
      expect(counts).toContain(submitted);
    }
    optimizer.stop();
    await done;
  });

  it('пауза не даёт создавать новых кандидатов, запущенные бэктесты досчитываются', async () => {
    const harness = createHarness({ behavior: { hold: () => true } });
    const optimizer = new PortfolioOptimizer(createInput(), harness.callbacks, harness.deps);
    const done = optimizer.start();
    await waitFor(() => harness.scheduler().inFlight === BASKET.length, 'созданы бэктесты базы');

    optimizer.pause();
    expect(optimizer.getState().status).toBe('paused');
    expect(optimizer.getProgress().status).toBe('paused');
    expect(harness.scheduler().paused).toBe(true);
    harness.behavior.hold = () => false;
    harness.scheduler().completeAll();
    await waitFor(() => findBase(optimizer.getState()).screen?.status === 'done', 'база оценена');
    await flushMicrotasks(500);
    expect(optimizer.getState().candidates).toHaveLength(1);
    expect(harness.scheduler().jobs).toHaveLength(BASKET.length);

    optimizer.resume();
    await done;
    const state = optimizer.getState();
    expect(state.status).toBe('completed');
    expect(nonBase(state)).toHaveLength(6);
    expect(harness.scheduler().pausedHistory).toEqual([false, true, false]);
  });

  it('остановка сохраняет созданные бэктесты, а fromSaved подключается к ним без повторного создания', async () => {
    const veles = new FakeVeles();
    const first = createHarness({ veles, behavior: { hold: () => true } });
    const optimizer = new PortfolioOptimizer(createInput(), first.callbacks, first.deps);
    const done = optimizer.start();
    await waitFor(() => first.scheduler().inFlight === BASKET.length, 'созданы бэктесты базы');
    optimizer.stop();
    await done;

    const stopped = first.lastSaved();
    expect(stopped.status).toBe('stopped');
    expect(stopped.backtestsSubmitted).toBe(BASKET.length);
    expect(stopped.inFlightJobs).toHaveLength(BASKET.length);
    expect(findBase(stopped).full?.status).toBe('running');
    expect(first.scheduler().jobs.every((job) => job.state === 'aborted')).toBe(true);
    expect(hasLog(first, 'warning', 'Запуск остановлен')).toBe(true);
    // После окончания сеанса UI получает итоговые кандидаты (этапы, прерванные остановкой, — «в работе»).
    expect(first.candidateUpdates.at(-1)?.candidates).toEqual(stopped.candidates);
    const submittedIds = first.scheduler().jobs.map((job) => job.backtestId);

    const second = createHarness({ veles, startTime: START_TIME + 3_600_000 });
    const resumed = PortfolioOptimizer.fromSaved(stopped, second.callbacks, second.deps);
    expect(resumed.getState().status).toBe('stopped');
    await resumed.start();
    const state = resumed.getState();

    const attaches = second.scheduler().jobs.filter((job) => job.kind === 'attach');
    expect(attaches.map((job) => job.backtestId)).toEqual(submittedIds);
    expect(runsFor(second, BASE_HASH)).toHaveLength(0);
    expect(idsInOrder(findBase(state).full, BASKET)).toEqual(submittedIds);
    expect(state.status).toBe('completed');
    expect(state.inFlightJobs).toEqual([]);
    expect(state.runId).toBe(stopped.runId);
    expect(state.startedAt).toBe(stopped.startedAt);
    expect(state.backtestsSubmitted).toBe(BASKET.length + submittedJobs(second).length);
    expect(hasLog(second, 'info', 'Продолжение запуска')).toBe(true);
    expect(hasLog(second, 'info', 'Подключаюсь к незавершённым бэктестам: 6')).toBe(true);
  });

  it('продолжает незаконченное поколение теми же кандидатами', async () => {
    const veles = new FakeVeles();
    const first = createHarness({ veles, behavior: { hold: (job) => !job.key.startsWith(BASE_HASH) } });
    const optimizer = new PortfolioOptimizer(createInput(), first.callbacks, first.deps);
    const done = optimizer.start();
    await waitFor(() => first.scheduler().inFlight === 3 * SCREEN.length, 'созданы бэктесты отбора поколения 1');
    optimizer.stop();
    await done;

    const stopped = first.lastSaved();
    expect(stopped.generation).toBe(0);
    expect(nonBase(stopped)).toHaveLength(3);
    expect(nonBase(stopped).every((candidate) => candidate.screen?.status === 'running')).toBe(true);
    expect(stopped.inFlightJobs).toHaveLength(3 * SCREEN.length);

    const second = createHarness({ veles, startTime: START_TIME + 3_600_000 });
    const resumed = PortfolioOptimizer.fromSaved(stopped, second.callbacks, second.deps);
    await resumed.start();
    const state = resumed.getState();

    expect(state.status).toBe('completed');
    expect(state.generation).toBe(2);
    const generationOne = state.candidates.filter((candidate) => candidate.generation === 1);
    expect(generationOne.map((candidate) => candidate.id)).toEqual(nonBase(stopped).map((candidate) => candidate.id));
    expect(second.scheduler().jobs.filter((job) => job.kind === 'attach')).toHaveLength(3 * SCREEN.length);
    for (const candidate of generationOne) {
      expect(runsFor(second, candidate.hash).filter((job) => SCREEN.includes(job.symbol))).toHaveLength(0);
    }
    expect(state.backtestsSubmitted).toBe(stopped.backtestsSubmitted + submittedJobs(second).length);
    expect(state.paramStats[RSI_ID].tries).toBe(nonBase(state).length);
  });

  it('после продолжения создаёт бэктесты, которые до остановки стояли в очереди', async () => {
    const veles = new FakeVeles();
    const first = createHarness({ veles, behavior: { hold: () => true, maxInFlight: 2 } });
    const optimizer = new PortfolioOptimizer(createInput({}, { generations: 1 }), first.callbacks, first.deps);
    const done = optimizer.start();
    await waitFor(() => first.scheduler().inFlight === 2, 'созданы два бэктеста базы');
    expect(first.scheduler().queued).toBe(BASKET.length - 2);
    optimizer.stop();
    await done;

    const stopped = first.lastSaved();
    expect(stopped.backtestsSubmitted).toBe(2);
    expect(stopped.inFlightJobs.map((job) => job.symbol)).toEqual(['ADA', 'XRP']);

    const second = createHarness({ veles, startTime: START_TIME + 3_600_000 });
    const resumed = PortfolioOptimizer.fromSaved(stopped, second.callbacks, second.deps);
    await resumed.start();
    const state = resumed.getState();
    expect(
      second
        .scheduler()
        .jobs.filter((job) => job.kind === 'attach')
        .map((job) => job.symbol),
    ).toEqual(['ADA', 'XRP']);
    expect(runsFor(second, BASE_HASH).map((job) => job.symbol)).toEqual(['SOL', 'DOGE', 'LINK', 'AVAX']);
    expect(state.backtestsSubmitted).toBe(2 + submittedJobs(second).length);
    expect(state.status).toBe('completed');
  });

  it('fromSaved нормализует состояние и отклоняет повреждённое', async () => {
    const harness = createHarness();
    const optimizer = new PortfolioOptimizer(createInput({}, { generations: 1 }), harness.callbacks, harness.deps);
    await optimizer.start();
    const saved = optimizer.getState();

    const logs: PortfolioLogEntry[] = Array.from({ length: 400 }, (_, index) => ({
      timestamp: index,
      level: 'info',
      message: `запись ${index}`,
    }));
    const restored = PortfolioOptimizer.fromSaved(
      {
        ...saved,
        status: 'running',
        logs,
        bestCandidateId: 'missing',
        inFlightJobs: [
          {
            key: 'x|ADA',
            symbol: 'ADA',
            backtestId: null,
            state: 'queued',
            attempts: 0,
            submittedAt: null,
            error: null,
          },
        ],
      },
      harness.callbacks,
      harness.deps,
    ).getState();
    expect(restored.status).toBe('stopped');
    expect(restored.logs).toHaveLength(MAX_PORTFOLIO_LOG_ENTRIES);
    expect(restored.logs[0].message).toBe('запись 100');
    expect(restored.bestCandidateId).toBeNull();
    expect(restored.inFlightJobs).toEqual([]);

    expect(() =>
      PortfolioOptimizer.fromSaved({ ...saved, candidates: nonBase(saved) }, harness.callbacks, harness.deps),
    ).toThrow('нет базового кандидата');
    const broken: unknown = { ...saved, version: 3 };
    expect(() =>
      PortfolioOptimizer.fromSaved(broken as PortfolioOptimizerState, harness.callbacks, harness.deps),
    ).toThrow('повреждено или устарело');
  });

  it('журнал не превышает 300 записей во время работы', async () => {
    const harness = createHarness();
    const optimizer = new PortfolioOptimizer(createInput({}, { generations: 1 }), harness.callbacks, harness.deps);
    const done = optimizer.start();
    for (let index = 0; index < 400; index += 1) {
      harness.scheduler().events.onLog('info', `шум ${index}`);
    }
    await done;
    const state = optimizer.getState();
    expect(state.logs.length).toBeLessThanOrEqual(MAX_PORTFOLIO_LOG_ENTRIES);
    expect(state.logs.at(-1)?.message).toContain('Оптимизация завершена');
  });
});

describe('PortfolioOptimizer: точность сравнения с базой', () => {
  it('на отборе минимум сделок масштабируется долей отборочной корзины (по базе)', async () => {
    const harness = createHarness();
    const objective: ObjectiveConfig = { kind: 'pnlToRisk', folds: 3, maxMaeCap: null, minDeals: 300 };
    const optimizer = new PortfolioOptimizer(createInput({ objective }, { generations: 1 }), harness.callbacks, {
      ...harness.deps,
    });
    await optimizer.start();
    const state = optimizer.getState();
    const base = findBase(state);
    // Полная корзина: 6 монет × 50 сделок = 300 — без штрафа. Отбор: 150 сделок при минимуме 300 × 150/300.
    expect(requireScore(base.full)).toBe(80);
    expect(requireScore(base.screen)).toBe(80);
    for (const candidate of nonBase(state)) {
      expect(requireScore(candidate.screen)).toBe(scoreForRsi(candidate.values[RSI_ID]));
    }
  });

  it('кандидат с малым числом сделок на отборе не продвигается на полную корзину', async () => {
    const harness = createHarness();
    // Кандидаты с RSI < 40 лучше по score, но почти не торгуют.
    harness.evaluator.adjust = (metrics, rsi) =>
      rsi.every((value) => value < 40) ? { ...metrics, totalDeals: rsi.length } : metrics;
    const optimizer = new PortfolioOptimizer(
      createInput({}, { generations: 2, promotePerGeneration: 3 }),
      harness.callbacks,
      harness.deps,
    );
    await optimizer.start();
    const state = optimizer.getState();
    const quiet = nonBase(state).filter((candidate) => candidate.values[RSI_ID] < 40);
    expect(quiet.length).toBeGreaterThan(0);
    for (const candidate of quiet) {
      expect(requireScore(candidate.screen)).toBeGreaterThan(requireScore(findBase(state).screen));
      expect(candidate.full).toBeNull();
    }
    expect(quiet.some((candidate) => candidate.id === state.bestCandidateId)).toBe(false);
    expect(hasLog(harness, 'info', 'слишком мало сделок на отборе')).toBe(true);
  });

  it('кандидат с малым числом сделок на полной корзине не становится лучшим и родителем', async () => {
    const harness = createHarness();
    harness.evaluator.adjust = (metrics, rsi) =>
      rsi.length === BASKET.length && rsi.every((value) => value < 40) ? { ...metrics, totalDeals: 6 } : metrics;
    const optimizer = new PortfolioOptimizer(
      createInput({}, { generations: 1, promotePerGeneration: 3 }),
      harness.callbacks,
      harness.deps,
    );
    await optimizer.start();
    const state = optimizer.getState();
    const quiet = nonBase(state).filter(
      (candidate) => candidate.full?.status === 'done' && candidate.values[RSI_ID] < 40,
    );
    expect(quiet.length).toBeGreaterThan(0);
    const base = findBase(state);
    for (const candidate of quiet) {
      expect(requireScore(candidate.full)).toBeGreaterThan(requireScore(base.full) * 1.01);
      expect(state.bestCandidateId).not.toBe(candidate.id);
    }
    expect(hasLog(harness, 'warning', 'лучшим не становится')).toBe(true);
    expect(hasLog(harness, 'warning', '6 сделок при 300 у базы')).toBe(true);
    for (const candidate of quiet) {
      for (const id of candidate.mutated ?? []) {
        expect(state.paramStats[id].wins).toBeLessThan(state.paramStats[id].tries);
      }
    }
  });

  it('бэктесты группы с другими комиссиями не берутся результатами базы и считаются заново', async () => {
    const harness = createHarness();
    harness.veles.register(501, 'ADA', 40);
    harness.veles.register(503, 'SOL', 40);
    harness.loadGroupBacktests.mockResolvedValue({
      groupName: 'Старая группа',
      backtests: [
        {
          backtestId: 501,
          base: 'ADA',
          quote: 'USDT',
          from: RUN.periodFrom,
          to: RUN.periodTo,
          useWicks: true,
          algorithm: 'LONG',
          commissions: { maker: 0, taker: 0 },
        },
        {
          backtestId: 503,
          base: 'SOL',
          quote: 'USDT',
          from: RUN.periodFrom,
          to: RUN.periodTo,
          useWicks: true,
          algorithm: 'LONG',
          commissions: { maker: 0.02, taker: 0.055 },
          strategy: groupStrategy(40),
        },
      ],
      failed: [],
    });
    const optimizer = new PortfolioOptimizer(
      createInput({ baseGroupId: 'group-old' }, { generations: 0 }),
      harness.callbacks,
      harness.deps,
    );
    await optimizer.start();
    const base = findBase(optimizer.getState());
    expect(runsFor(harness, base.hash).map((job) => job.symbol)).toEqual(['ADA', 'XRP', 'DOGE', 'LINK', 'AVAX']);
    expect(requireEvaluation(base.full).backtestIds.SOL).toBe(503);
    expect(requireEvaluation(base.full).backtestIds.ADA).not.toBe(501);
    expect(hasLog(harness, 'warning', 'Комиссии в бэктестах группы отличаются от запуска')).toBe(true);
  });
});

describe('PortfolioOptimizer: сбои Veles и сети', () => {
  it('таймаут нового бэктеста останавливает запуск, запись о бэктесте сохраняется и он подключается после продолжения', async () => {
    const veles = new FakeVeles();
    let timedOut: string | null = null;
    const first = createHarness({
      veles,
      behavior: {
        runError: (job) => {
          if (timedOut === null && !job.key.startsWith(BASE_HASH)) {
            timedOut = job.key;
            return new TransientBacktestError(
              `Таймаут бэктеста ${job.backtestId} (${job.symbol}): нет результата за 45 мин.`,
              'timeout',
              job.backtestId,
            );
          }
          return null;
        },
      },
    });
    const optimizer = new PortfolioOptimizer(createInput({}, { generations: 1 }), first.callbacks, first.deps);
    await optimizer.start();
    const stopped = first.lastSaved();

    expect(stopped.status).toBe('error');
    expect(hasLog(first, 'error', 'Veles или сеть недоступны: Таймаут бэктеста')).toBe(true);
    expect(nonBase(stopped).some((candidate) => candidate.screen?.status === 'failed')).toBe(false);
    const entry = stopped.inFlightJobs.find((job) => job.key === timedOut);
    expect(entry?.backtestId).not.toBeNull();
    expect(stopped.paramStats).toEqual({});

    const second = createHarness({ veles, startTime: START_TIME + 3_600_000 });
    const resumed = PortfolioOptimizer.fromSaved(stopped, second.callbacks, second.deps);
    await resumed.start();
    const state = resumed.getState();
    const attach = second.scheduler().jobs.find((job) => job.kind === 'attach' && job.key === timedOut);
    // Таймаут после продолжения отсчитывается от создания бэктеста.
    expect(attach?.submittedAt).toBe(entry?.submittedAt);
    expect(runsFor(second, (timedOut ?? '').split('|')[0]).some((job) => job.key === timedOut)).toBe(false);
    expect(state.status).toBe('completed');
    expect(state.backtestCache[timedOut ?? '']).toBe(entry?.backtestId);
  });

  it('бэктест, который не завершился и после продолжения, считается неудачным', async () => {
    const veles = new FakeVeles();
    const first = createHarness({ veles, behavior: { hold: () => true } });
    const optimizer = new PortfolioOptimizer(createInput(), first.callbacks, first.deps);
    const done = optimizer.start();
    await waitFor(() => first.scheduler().inFlight === BASKET.length, 'созданы бэктесты базы');
    optimizer.stop();
    await done;

    const second = createHarness({
      veles,
      startTime: START_TIME + 3_600_000,
      behavior: {
        runError: (job) =>
          job.kind === 'attach' && job.symbol === 'ADA'
            ? new TransientBacktestError('Таймаут бэктеста 1000 (ADA): нет результата за 45 мин.', 'timeout', 1000)
            : null,
      },
    });
    const resumed = PortfolioOptimizer.fromSaved(first.lastSaved(), second.callbacks, second.deps);
    await resumed.start();
    const state = resumed.getState();
    const full = requireEvaluation(findBase(state).full);
    expect(full.status).toBe('failed');
    expect(full.failedSymbols).toEqual(['ADA']);
    expect(full.error).toContain('не завершился и после продолжения');
    expect(state.inFlightJobs.some((job) => job.symbol === 'ADA')).toBe(false);
    expect(hasLog(second, 'error', 'Базовый бот не оценён')).toBe(true);
  });

  it('исчерпанные повторы создания бэктеста останавливают запуск без «провала» кандидата', async () => {
    const veles = new FakeVeles();
    let failedKey: string | null = null;
    const first = createHarness({
      veles,
      behavior: {
        submitError: (job) => {
          if (failedKey === null && !job.key.startsWith(BASE_HASH)) {
            failedKey = job.key;
            return new TransientBacktestError(
              `Не удалось создать бэктест ${job.symbol} за 5 попыток: Failed to fetch`,
              'submit',
            );
          }
          return null;
        },
      },
    });
    const optimizer = new PortfolioOptimizer(createInput({}, { generations: 1 }), first.callbacks, first.deps);
    await optimizer.start();
    const stopped = first.lastSaved();
    expect(stopped.status).toBe('error');
    expect(nonBase(stopped).every((candidate) => candidate.screen?.status !== 'failed')).toBe(true);
    expect(stopped.generation).toBe(0);
    expect(stopped.paramStats).toEqual({});
    expect(hasLog(first, 'error', 'Failed to fetch')).toBe(true);

    const second = createHarness({ veles, startTime: START_TIME + 3_600_000 });
    const resumed = PortfolioOptimizer.fromSaved(stopped, second.callbacks, second.deps);
    await resumed.start();
    const state = resumed.getState();
    expect(state.status).toBe('completed');
    expect(state.candidates.find((candidate) => failedKey?.startsWith(candidate.hash))?.screen?.status).toBe('done');
  });

  it('ошибка авторизации у созданного бэктеста сохраняет запись о нём для продолжения', async () => {
    const veles = new FakeVeles();
    let unauthorizedKey: string | null = null;
    const first = createHarness({
      veles,
      behavior: {
        runError: (job) => {
          if (unauthorizedKey === null && !job.key.startsWith(BASE_HASH)) {
            unauthorizedKey = job.key;
            return new Error('HTTP 401 Unauthorized');
          }
          return null;
        },
      },
    });
    const optimizer = new PortfolioOptimizer(createInput({}, { generations: 1 }), first.callbacks, first.deps);
    await optimizer.start();
    const stopped = first.lastSaved();
    const entry = stopped.inFlightJobs.find((job) => job.key === unauthorizedKey);
    expect(entry?.backtestId).not.toBeNull();

    const second = createHarness({ veles, startTime: START_TIME + 3_600_000 });
    const resumed = PortfolioOptimizer.fromSaved(stopped, second.callbacks, second.deps);
    await resumed.start();
    expect(second.scheduler().jobs.some((job) => job.kind === 'attach' && job.key === unauthorizedKey)).toBe(true);
    expect(resumed.getState().backtestCache[unauthorizedKey ?? '']).toBe(entry?.backtestId);
  });
});

describe('PortfolioOptimizer: отмена этапа и поздние ответы на POST', () => {
  const deferSolOfFirstCandidate = () => {
    const target: { hash: string | null } = { hash: null };
    const behavior: Partial<FakeSchedulerBehavior> = {
      deferPost: (job) => {
        if (job.key.startsWith(BASE_HASH)) {
          return false;
        }
        target.hash ??= job.key.split('|')[0];
        return job.key === `${target.hash}|SOL`;
      },
      runError: (job) =>
        job.key === `${target.hash}|ADA` ? new Error('Бэктест (ADA) завершился с ошибкой: статус FAILED') : null,
    };
    return { target, behavior };
  };

  it('не прерывает задачу, чей POST уже отправлен: созданный бэктест досчитывается и попадает в кеш', async () => {
    const { target, behavior } = deferSolOfFirstCandidate();
    const harness = createHarness({ behavior });
    const optimizer = new PortfolioOptimizer(createInput({}, { generations: 1 }), harness.callbacks, harness.deps);
    const done = optimizer.start();
    const sol = () => harness.scheduler().jobs.find((job) => job.key === `${target.hash}|SOL`);
    await waitFor(
      () =>
        sol()?.state === 'posting' &&
        harness.scheduler().jobs.some((job) => job.key === `${target.hash}|ADA` && job.state === 'failed'),
      'ADA завершился с ошибкой, POST SOL ещё в пути',
    );
    await flushMicrotasks(200);
    expect(sol()?.state).toBe('posting');

    const job = sol();
    if (!job) {
      throw new Error('нет задачи SOL');
    }
    harness.scheduler().completePost(job);
    await done;
    const state = optimizer.getState();
    expect(job.state).toBe('done');
    expect(state.backtestCache[`${target.hash}|SOL`]).toBe(job.backtestId);
    const broken = state.candidates.find((candidate) => candidate.hash === target.hash);
    expect(broken?.screen?.status).toBe('failed');
    expect(broken?.screen?.failedSymbols).toEqual(['ADA']);
    expect(state.backtestsSubmitted).toBe(submittedJobs(harness).length);
    expect(state.inFlightJobs).toEqual([]);
    expect(state.status).toBe('completed');
  });

  it('отменяет такую задачу перед повторной отправкой, если POST не удался', async () => {
    const { target, behavior } = deferSolOfFirstCandidate();
    const harness = createHarness({ behavior });
    const optimizer = new PortfolioOptimizer(createInput({}, { generations: 1 }), harness.callbacks, harness.deps);
    const done = optimizer.start();
    const sol = () => harness.scheduler().jobs.find((job) => job.key === `${target.hash}|SOL`);
    await waitFor(
      () =>
        sol()?.state === 'posting' &&
        harness.scheduler().jobs.some((job) => job.key === `${target.hash}|ADA` && job.state === 'failed'),
      'ADA завершился с ошибкой, POST SOL ещё в пути',
    );
    const job = sol();
    if (!job) {
      throw new Error('нет задачи SOL');
    }
    harness.scheduler().failPost(job);
    await done;
    expect(job.state).toBe('aborted');
    expect(job.backtestId).toBeNull();
    expect(optimizer.getState().status).toBe('completed');
    expect(optimizer.getState().backtestsSubmitted).toBe(submittedJobs(harness).length);
  });

  it('ответ на POST после остановки дописывается в сохранённое состояние, и после продолжения бэктест не создаётся повторно', async () => {
    const veles = new FakeVeles();
    const first = createHarness({
      veles,
      behavior: { hold: () => true, deferPost: (job) => job.key === `${BASE_HASH}|SOL` },
    });
    const optimizer = new PortfolioOptimizer(createInput({}, { generations: 0 }), first.callbacks, first.deps);
    const done = optimizer.start();
    await waitFor(() => first.scheduler().inFlight === BASKET.length - 1, 'созданы бэктесты базы, кроме SOL');
    optimizer.stop();
    await done;
    const stopped = first.lastSaved();
    expect(stopped.inFlightJobs.some((job) => job.symbol === 'SOL')).toBe(false);
    const savesAfterStop = first.saves.length;

    const sol = first.scheduler().jobs.find((job) => job.key === `${BASE_HASH}|SOL`);
    if (!sol) {
      throw new Error('нет задачи SOL');
    }
    first.scheduler().completePost(sol);
    expect(first.saves.length).toBe(savesAfterStop + 1);
    const patched = first.lastSaved();
    expect(patched.backtestsSubmitted).toBe(stopped.backtestsSubmitted + 1);
    expect(patched.inFlightJobs.find((job) => job.symbol === 'SOL')?.backtestId).toBe(sol.backtestId);
    // Остальное состояние не перезаписано устаревшей копией экземпляра.
    expect(patched.updatedAt).toBe(stopped.updatedAt);

    const second = createHarness({ veles, startTime: START_TIME + 3_600_000 });
    const resumed = PortfolioOptimizer.fromSaved(patched, second.callbacks, second.deps);
    await resumed.start();
    expect(runsFor(second, BASE_HASH)).toHaveLength(0);
    expect(resumed.getState().backtestCache[`${BASE_HASH}|SOL`]).toBe(sol.backtestId);
  });

  it('поздний ответ передаётся живому оптимизатору того же запуска, и его задача подключается без POST', async () => {
    const veles = new FakeVeles();
    let resumed: PortfolioOptimizer | null = null;
    const lateJobs: ScheduledJob[] = [];
    const first = createHarness({
      veles,
      behavior: { hold: () => true, deferPost: (job) => job.key === `${BASE_HASH}|SOL` },
    });
    const callbacks: PortfolioOptimizerCallbacks = {
      ...first.callbacks,
      onLateSubmission: (runId, job) => {
        lateJobs.push(job);
        return resumed?.adoptLateSubmission(runId, job) ?? false;
      },
    };
    const optimizer = new PortfolioOptimizer(createInput({}, { generations: 0 }), callbacks, first.deps);
    const done = optimizer.start();
    await waitFor(() => first.scheduler().inFlight === BASKET.length - 1, 'созданы бэктесты базы, кроме SOL');
    optimizer.stop();
    await done;
    const stopped = first.lastSaved();
    const savesAfterStop = first.saves.length;

    const second = createHarness({ veles, startTime: START_TIME + 3_600_000, behavior: { manualPump: true } });
    resumed = PortfolioOptimizer.fromSaved(stopped, second.callbacks, second.deps);
    const resumedDone = resumed.start();
    await waitFor(() => second.scheduler().queued === 1, 'SOL базы снова в очереди');

    const sol = first.scheduler().jobs.find((job) => job.key === `${BASE_HASH}|SOL`);
    if (!sol) {
      throw new Error('нет задачи SOL');
    }
    first.scheduler().completePost(sol);
    expect(lateJobs.map((job) => job.backtestId)).toEqual([sol.backtestId]);
    expect(first.saves.length).toBe(savesAfterStop);
    expect(hasLog(second, 'info', 'создан ещё до продолжения запуска')).toBe(true);

    await drive(second, resumedDone);
    const state = resumed.getState();
    const queuedSol = second.scheduler().jobs.find((job) => job.key === `${BASE_HASH}|SOL`);
    expect(queuedSol?.backtestId).toBe(sol.backtestId);
    // В Veles ровно один бэктест SOL базы — тот, что создал остановленный экземпляр.
    expect([...veles.backtests.values()].filter((backtest) => backtest.key === `${BASE_HASH}|SOL`)).toHaveLength(1);
    expect(state.backtestsSubmitted).toBe(stopped.backtestsSubmitted + 1);
    expect(state.backtestCache[`${BASE_HASH}|SOL`]).toBe(sol.backtestId);
    expect(state.status).toBe('completed');
  });

  it('applyLateSubmission не учитывает один бэктест дважды', () => {
    const state: PortfolioOptimizerState = {
      version: 1,
      runId: 'po1',
      config: createConfig(),
      baseStrategy: createStrategy(),
      quoteCurrency: 'USDT',
      baseCandidateId: 'po1-base',
      bestCandidateId: null,
      candidates: [],
      backtestCache: { 'h|ADA': 7 },
      inFlightJobs: [],
      paramStats: {},
      generation: 0,
      backtestsSubmitted: 3,
      status: 'stopped',
      startedAt: 1,
      updatedAt: 2,
      logs: [],
    };
    const job = (key: string, backtestId: number): ScheduledJob => ({
      key,
      symbol: key.split('|')[1],
      backtestId,
      state: 'submitted',
      attempts: 1,
      submittedAt: 5,
      error: null,
    });
    expect(applyLateSubmission(state, job('h|ADA', 7))).toBe(false);
    expect(applyLateSubmission(state, job('h|SOL', 9))).toBe(true);
    expect(applyLateSubmission(state, job('h|SOL', 9))).toBe(false);
    expect(state.backtestsSubmitted).toBe(4);
    expect(state.inFlightJobs.map((item) => item.key)).toEqual(['h|SOL']);
    // Дубликат уже посчитанной монеты: учитывается в лимите, но кеш не меняется.
    expect(applyLateSubmission(state, job('h|ADA', 11))).toBe(true);
    expect(state.backtestsSubmitted).toBe(5);
    expect(state.backtestCache['h|ADA']).toBe(7);
  });
});
