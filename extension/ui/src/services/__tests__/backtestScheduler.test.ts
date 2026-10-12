import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('../../api/backtestRunner', () => ({
  postBacktest: vi.fn(),
}));

vi.mock('../../api/backtests', () => ({
  fetchBacktestConfig: vi.fn(),
  fetchBacktestStatistics: vi.fn(),
  fetchBacktests: vi.fn(),
}));

import type { BotStrategy } from '../../api/backtestRunner';
import { postBacktest } from '../../api/backtestRunner';
import { fetchBacktestConfig, fetchBacktestStatistics, fetchBacktests } from '../../api/backtests';
import type { BacktestConfigDto, BacktestStatisticsDto, BacktestStatisticsListDto } from '../../api/backtests.dtos';
import type { SchedulerConfig } from '../../types/portfolioOptimizer';
import {
  abortableSleep,
  BacktestScheduler,
  BacktestSubmitError,
  classifyBacktestStatus,
  classifySubmitError,
  computeRetryBackoffMs,
  computeSubmitWaitMs,
  createDefaultSchedulerDeps,
  decaySubmitDelay,
  escalateSubmitDelay,
  extractHttpStatus,
  extractRetryAfterSeconds,
  isAbortError,
  isAuthError,
  isTransientBacktestError,
  mergeSchedulerConfig,
  parseBacktestId,
  parseRetryAfterFromMessage,
  RATE_LIMIT_DELAY_CAP_SECONDS,
  rescaleSubmitDelay,
  resolveSchedulerDeps,
  SCHEDULER_ABORT_MESSAGE,
  SchedulerAbortError,
  type SchedulerJobRequest,
  type SubmitResponse,
  type SubmitWaitInput,
  TransientBacktestError,
  toBacktestSubmitError,
} from '../backtestScheduler';

// ═══════════════════════════════════════════════════════════════════
// ВИРТУАЛЬНОЕ ВРЕМЯ И ФЕЙКОВЫЙ VELES
// ═══════════════════════════════════════════════════════════════════

interface VirtualTimer {
  at: number;
  order: number;
  fire: () => void;
}

interface VirtualClock {
  now: () => number;
  sleep: (ms: number, signal?: AbortSignal) => Promise<void>;
  /** Продвигает время по таймерам, пока не выполнится условие (или не кончатся таймеры / не превышен limit). */
  runUntil: (condition: () => boolean, limit?: number) => Promise<void>;
  /** Выполняет все таймеры до target включительно и выставляет время = target. */
  advanceTo: (target: number) => Promise<void>;
  pendingTimers: () => number;
}

const flushTasks = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));

const createVirtualClock = (): VirtualClock => {
  let current = 0;
  let order = 0;
  const timers: VirtualTimer[] = [];

  const sleep = (ms: number, signal?: AbortSignal): Promise<void> =>
    new Promise<void>((resolve, reject) => {
      if (signal?.aborted) {
        reject(new SchedulerAbortError());
        return;
      }
      const timer: VirtualTimer = { at: current + Math.max(0, ms), order, fire: () => undefined };
      order += 1;
      const onAbort = (): void => {
        const index = timers.indexOf(timer);
        if (index >= 0) {
          timers.splice(index, 1);
        }
        reject(new SchedulerAbortError());
      };
      timer.fire = () => {
        signal?.removeEventListener('abort', onAbort);
        resolve();
      };
      signal?.addEventListener('abort', onAbort, { once: true });
      timers.push(timer);
    });

  const runUntil = async (condition: () => boolean, limit = Number.POSITIVE_INFINITY): Promise<void> => {
    for (let step = 0; step < 20_000; step += 1) {
      await flushTasks();
      if (condition()) {
        return;
      }
      timers.sort((left, right) => left.at - right.at || left.order - right.order);
      const next = timers.at(0);
      if (!next || next.at > limit) {
        return;
      }
      timers.shift();
      current = Math.max(current, next.at);
      next.fire();
    }
    throw new Error('Виртуальные часы: слишком много шагов');
  };

  const advanceTo = async (target: number): Promise<void> => {
    await runUntil(() => false, target);
    current = Math.max(current, target);
    await flushTasks();
  };

  return { now: () => current, sleep, runUntil, advanceTo, pendingTimers: () => timers.length };
};

interface FakeBacktest {
  id: number;
  createdAt: number;
  name: string | null;
}

interface FakePost {
  at: number;
  name: string | null | undefined;
  id: number | null;
  inFlight: number;
}

interface FakeVelesOptions {
  durationMs?: number;
  status?: (backtest: FakeBacktest, now: number) => string | null;
  statisticsExists?: (backtest: FakeBacktest, now: number) => boolean;
}

const createFakeVeles = (clock: VirtualClock, options: FakeVelesOptions = {}) => {
  const durationMs = options.durationMs ?? 20_000;
  const statusFor =
    options.status ??
    ((backtest: FakeBacktest, now: number) => (now - backtest.createdAt >= durationMs ? 'FINISHED' : 'RUNNING'));
  const backtests = new Map<number, FakeBacktest>();
  const posts: FakePost[] = [];
  /** Ошибки POST, при которых бэктест не создан. */
  const postFailures: Error[] = [];
  /** Ошибки POST, при которых бэктест создан, но ответ потерялся (таймаут вкладки, 502 после записи). */
  const lostResponses: Error[] = [];
  /** Сколько следующих POST создают бэктест, но отвечают 2xx без id. */
  let idlessResponses = 0;
  let nextId = 1000;
  let inFlightProbe: () => number = () => 0;

  const create = (payload: BotStrategy): number => {
    const id = nextId;
    nextId += 1;
    backtests.set(id, { id, createdAt: clock.now(), name: payload.name ?? null });
    posts.push({ at: clock.now(), name: payload.name, id, inFlight: inFlightProbe() });
    return id;
  };

  const postBacktest = vi.fn(async (payload: BotStrategy): Promise<SubmitResponse> => {
    const failure = postFailures.shift();
    if (failure) {
      posts.push({ at: clock.now(), name: payload.name, id: null, inFlight: inFlightProbe() });
      throw failure;
    }
    const id = create(payload);
    const lost = lostResponses.shift();
    if (lost) {
      throw lost;
    }
    if (idlessResponses > 0) {
      idlessResponses -= 1;
      return { id: undefined };
    }
    return { id };
  });

  const findBacktestsByName = vi.fn(
    async (name: string, _symbol: string | null, createdAfter: number): Promise<number[]> =>
      [...backtests.values()]
        .filter((backtest) => backtest.name === name && backtest.createdAt >= createdAfter)
        .map((backtest) => backtest.id)
        .sort((left, right) => right - left),
  );

  const fetchStatus = vi.fn(async (id: number): Promise<string | null> => {
    const backtest = backtests.get(id);
    if (!backtest) {
      throw new Error(`HTTP 404: бэктест ${id} не найден`);
    }
    return statusFor(backtest, clock.now());
  });

  const fetchStatisticsExists = vi.fn(async (id: number): Promise<boolean> => {
    const backtest = backtests.get(id);
    return backtest ? (options.statisticsExists?.(backtest, clock.now()) ?? false) : false;
  });

  return {
    postBacktest,
    fetchStatus,
    fetchStatisticsExists,
    findBacktestsByName,
    posts,
    postFailures,
    lostResponses,
    addExisting: (id: number, createdAt: number) => backtests.set(id, { id, createdAt, name: null }),
    respondWithoutId: (times: number) => {
      idlessResponses = times;
    },
    setInFlightProbe: (probe: () => number) => {
      inFlightProbe = probe;
    },
  };
};

const BASE_CONFIG: SchedulerConfig = {
  maxInFlight: 3,
  submitDelaySeconds: 31,
  pollIntervalSeconds: 10,
  backtestTimeoutMinutes: 45,
};

interface LogRecord {
  level: 'info' | 'warning' | 'error';
  message: string;
}

const createHarness = (config: Partial<SchedulerConfig> = {}, velesOptions: FakeVelesOptions = {}) => {
  const clock = createVirtualClock();
  const veles = createFakeVeles(clock, velesOptions);
  const logs: LogRecord[] = [];
  const submitted: Array<{ key: string; id: number }> = [];
  const scheduler = new BacktestScheduler(
    { ...BASE_CONFIG, ...config },
    {
      onSubmitted: (key, id) => submitted.push({ key, id }),
      onLog: (level, message) => logs.push({ level, message }),
    },
    {
      postBacktest: veles.postBacktest,
      fetchStatus: veles.fetchStatus,
      fetchStatisticsExists: veles.fetchStatisticsExists,
      findBacktestsByName: veles.findBacktestsByName,
      sleep: clock.sleep,
      now: clock.now,
    },
  );
  veles.setInFlightProbe(() => scheduler.inFlight);
  return { clock, veles, logs, submitted, scheduler };
};

const makeJob = (symbol: string, hash = 'h1'): SchedulerJobRequest => ({
  key: `${hash}|${symbol}`,
  symbol,
  payload: { id: null, name: `PO ${symbol}` },
});

interface Tracked<T> {
  settled: boolean;
  value: T | undefined;
  error: unknown;
  settledAt: number | null;
}

const track = <T>(promise: Promise<T>, clock: VirtualClock): Tracked<T> => {
  const state: Tracked<T> = { settled: false, value: undefined, error: undefined, settledAt: null };
  promise.then(
    (value) => {
      state.settled = true;
      state.value = value;
      state.settledAt = clock.now();
    },
    (error: unknown) => {
      state.settled = true;
      state.error = error;
      state.settledAt = clock.now();
    },
  );
  return state;
};

const allSettled = (items: Array<Tracked<unknown>>): (() => boolean) => {
  return () => items.every((item) => item.settled);
};

const errorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));

afterEach(() => {
  vi.useRealTimers();
});

// ═══════════════════════════════════════════════════════════════════
// ЧИСТЫЕ ПОМОЩНИКИ
// ═══════════════════════════════════════════════════════════════════

describe('classifyBacktestStatus', () => {
  it.each(['FINISHED', 'COMPLETED', 'DONE', 'SUCCESS', 'SUCCEEDED'])('treats %s as done', (status) => {
    expect(classifyBacktestStatus(status)).toBe('done');
  });

  it.each(['FAILED', 'ERROR', 'CANCELED', 'CANCELLED', 'TERMINATED', 'REJECTED'])('treats %s as failed', (status) => {
    expect(classifyBacktestStatus(status)).toBe('failed');
  });

  it.each([
    'NEW',
    'CREATED',
    'QUEUED',
    'PENDING',
    'WAITING',
    'STARTED',
    'RUNNING',
    'IN_PROGRESS',
    'PROCESSING',
    'CALCULATING',
  ])('treats %s as running', (status) => {
    expect(classifyBacktestStatus(status)).toBe('running');
  });

  it('is case-insensitive and tolerant to spaces and hyphens', () => {
    expect(classifyBacktestStatus('finished')).toBe('done');
    expect(classifyBacktestStatus('  Failed ')).toBe('failed');
    expect(classifyBacktestStatus('in progress')).toBe('running');
    expect(classifyBacktestStatus('in-progress')).toBe('running');
  });

  it('returns unknown for empty, missing and unexpected values', () => {
    expect(classifyBacktestStatus(null)).toBe('unknown');
    expect(classifyBacktestStatus(undefined)).toBe('unknown');
    expect(classifyBacktestStatus('')).toBe('unknown');
    expect(classifyBacktestStatus('   ')).toBe('unknown');
    expect(classifyBacktestStatus('MYSTERY')).toBe('unknown');
  });
});

describe('extractHttpStatus', () => {
  it('reads a numeric status property', () => {
    expect(extractHttpStatus(Object.assign(new Error('boom'), { status: 503 }))).toBe(503);
    expect(extractHttpStatus({ status: 429, message: 'x' })).toBe(429);
  });

  it('parses the "HTTP NNN" label produced by resolveProxyErrorMessage', () => {
    expect(extractHttpStatus(new Error('HTTP 429: Действие временно заблокировано (Too Many Requests)'))).toBe(429);
    expect(extractHttpStatus(new Error('HTTP 502 Bad Gateway'))).toBe(502);
  });

  it('returns null when no status is available', () => {
    expect(extractHttpStatus(new Error('Таймаут ответа от вкладки'))).toBeNull();
    expect(extractHttpStatus('plain string')).toBeNull();
    expect(extractHttpStatus(Object.assign(new Error('x'), { status: 'oops' }))).toBeNull();
    expect(extractHttpStatus(Object.assign(new Error('x'), { status: 42 }))).toBeNull();
  });
});

describe('parseBacktestId', () => {
  it('accepts positive integer ids as numbers or digit strings', () => {
    expect(parseBacktestId({ id: 42 })).toBe(42);
    expect(parseBacktestId({ id: '1234' })).toBe(1234);
    expect(parseBacktestId({ id: ' 77 ' })).toBe(77);
  });

  it('rejects missing, malformed and non-positive ids', () => {
    expect(parseBacktestId({ id: Number.NaN })).toBeNull();
    expect(parseBacktestId({ id: 0 })).toBeNull();
    expect(parseBacktestId({ id: 1.5 })).toBeNull();
    expect(parseBacktestId({ id: '12a' })).toBeNull();
    expect(parseBacktestId({ id: null })).toBeNull();
    expect(parseBacktestId('created')).toBeNull();
    expect(parseBacktestId(null)).toBeNull();
  });
});

describe('isAuthError', () => {
  it('recognises 401/403 and auth wording but not backtest numbers', () => {
    expect(isAuthError(new Error('HTTP 401 Unauthorized'))).toBe(true);
    expect(isAuthError(new BacktestSubmitError('Доступ запрещён', 403, null))).toBe(true);
    expect(isAuthError(new Error('Forbidden'))).toBe(true);
    expect(isAuthError(new Error('Бэктест 401 (ADA) завершился с ошибкой: статус FAILED'))).toBe(false);
    expect(isAuthError(new Error('HTTP 500 Internal Server Error'))).toBe(false);
  });
});

describe('Retry-After parsing', () => {
  it('parses header-like and Russian phrasing', () => {
    expect(parseRetryAfterFromMessage('HTTP 429: Retry-After: 30')).toBe(30);
    expect(parseRetryAfterFromMessage('retry after 12 seconds')).toBe(12);
    expect(parseRetryAfterFromMessage('Попробуйте через 45 сек')).toBe(45);
    expect(parseRetryAfterFromMessage('Повторите через 2 мин')).toBe(120);
    expect(parseRetryAfterFromMessage('Повторите через 1,5 минуты')).toBe(90);
  });

  it('clamps to a sane range and ignores unrelated text', () => {
    expect(parseRetryAfterFromMessage('Retry-After: 0')).toBe(1);
    expect(parseRetryAfterFromMessage('Retry-After: 100000')).toBe(600);
    expect(parseRetryAfterFromMessage('HTTP 429 Too Many Requests')).toBeNull();
    expect(parseRetryAfterFromMessage('через 5 дней')).toBeNull();
  });

  it('prefers a numeric retryAfterSeconds property', () => {
    expect(extractRetryAfterSeconds(Object.assign(new Error('Retry-After: 5'), { retryAfterSeconds: 90 }))).toBe(90);
    expect(extractRetryAfterSeconds(new BacktestSubmitError('HTTP 429', 429, 15))).toBe(15);
    expect(extractRetryAfterSeconds(new Error('Retry-After: 5'))).toBe(5);
    expect(extractRetryAfterSeconds(new Error('HTTP 429'))).toBeNull();
  });
});

describe('toBacktestSubmitError', () => {
  it('keeps the message and attaches status and Retry-After', () => {
    const wrapped = toBacktestSubmitError(new Error('HTTP 429 Too Many Requests: Retry-After: 20'));
    expect(wrapped).toBeInstanceOf(BacktestSubmitError);
    expect(wrapped.message).toBe('HTTP 429 Too Many Requests: Retry-After: 20');
    expect(wrapped.status).toBe(429);
    expect(wrapped.retryAfterSeconds).toBe(20);
  });

  it('handles errors without status and non-Error values', () => {
    const wrapped = toBacktestSubmitError('Не найдена вкладка veles.finance');
    expect(wrapped.message).toBe('Не найдена вкладка veles.finance');
    expect(wrapped.status).toBeNull();
    expect(wrapped.retryAfterSeconds).toBeNull();
  });

  it('returns BacktestSubmitError instances unchanged', () => {
    const original = new BacktestSubmitError('HTTP 400', 400, null);
    expect(toBacktestSubmitError(original)).toBe(original);
  });
});

describe('classifySubmitError', () => {
  it.each([
    new Error('HTTP 429: Действие временно заблокировано (Too Many Requests)'),
    new Error('Too Many Requests'),
    new Error('Слишком много запросов, попробуйте позже'),
    new Error('Действие временно заблокировано'),
    new BacktestSubmitError('Лимит', 429, null),
  ])('detects rate limiting: %s', (error) => {
    expect(classifySubmitError(error)).toBe('rateLimit');
  });

  it.each([
    new Error('HTTP 500 Internal Server Error'),
    new Error('HTTP 502 Bad Gateway'),
    new Error('HTTP 503: Service Unavailable'),
    new Error('HTTP 408 Request Timeout'),
    new Error('Таймаут ответа от вкладки'),
    new Error('Не найдена вкладка veles.finance'),
    new Error('Failed to fetch'),
    new Error('NetworkError when attempting to fetch resource.'),
    new Error('Связь с расширением потеряна. Пожалуйста, перезагрузите страницу (F5).'),
    new Error('Сервер не вернул данные о созданном бэктесте.'),
  ])('treats transport and 5xx errors as recoverable: %s', (error) => {
    expect(classifySubmitError(error)).toBe('recoverable');
  });

  it.each([
    new Error('HTTP 400 Bad Request: Некорректные параметры'),
    new Error('HTTP 401 Unauthorized'),
    new Error('HTTP 403: Forbidden'),
    new Error('HTTP 404 Not Found'),
    new BacktestSubmitError('Неверный payload', 422, null),
  ])('treats other 4xx as fatal: %s', (error) => {
    expect(classifySubmitError(error)).toBe('fatal');
  });
});

describe('delay helpers', () => {
  it('computes exponential backoff capped at 120 s', () => {
    expect([1, 2, 3, 4, 5, 6].map(computeRetryBackoffMs)).toEqual([10_000, 20_000, 40_000, 80_000, 120_000, 120_000]);
    expect(computeRetryBackoffMs(0)).toBe(10_000);
  });

  it('escalates the submit delay by 1.5 with a cap', () => {
    expect(escalateSubmitDelay(31, 31)).toBeCloseTo(46.5);
    expect(escalateSubmitDelay(100, 31)).toBe(RATE_LIMIT_DELAY_CAP_SECONDS);
    expect(escalateSubmitDelay(0, 0)).toBeCloseTo(1.5);
    expect(escalateSubmitDelay(150, 150)).toBe(150);
  });

  it('decays the submit delay towards the configured value', () => {
    expect(decaySubmitDelay(46.5, 31)).toBeCloseTo(41.85);
    expect(decaySubmitDelay(32, 31)).toBe(31);
    expect(decaySubmitDelay(31, 31)).toBe(31);
  });

  it('rescales a penalised delay when the configured value changes', () => {
    expect(rescaleSubmitDelay(31, 31, 10)).toBe(10);
    expect(rescaleSubmitDelay(46.5, 31, 20)).toBeCloseTo(30);
    expect(rescaleSubmitDelay(100, 20, 60)).toBe(RATE_LIMIT_DELAY_CAP_SECONDS);
    expect(rescaleSubmitDelay(5, 0, 31)).toBe(31);
  });
});

describe('computeSubmitWaitMs', () => {
  const input = (overrides: Partial<SubmitWaitInput> = {}): SubmitWaitInput => ({
    paused: false,
    inFlight: 0,
    maxInFlight: 3,
    now: 100_000,
    lastPostAt: null,
    effectiveDelayMs: 31_000,
    cooldownUntil: 0,
    notBefore: 0,
    ...overrides,
  });

  it('allows the very first submission immediately', () => {
    expect(computeSubmitWaitMs(input())).toBe(0);
  });

  it('waits for an event when paused or when all slots are busy', () => {
    expect(computeSubmitWaitMs(input({ paused: true }))).toBeNull();
    expect(computeSubmitWaitMs(input({ inFlight: 3 }))).toBeNull();
  });

  it('respects spacing, cooldown and per-job backoff (the latest wins)', () => {
    expect(computeSubmitWaitMs(input({ lastPostAt: 90_000 }))).toBe(21_000);
    expect(computeSubmitWaitMs(input({ lastPostAt: 90_000, cooldownUntil: 150_000 }))).toBe(50_000);
    expect(computeSubmitWaitMs(input({ lastPostAt: 90_000, notBefore: 125_000 }))).toBe(25_000);
    expect(computeSubmitWaitMs(input({ lastPostAt: 60_000 }))).toBe(0);
  });
});

describe('mergeSchedulerConfig', () => {
  it('applies valid values and clamps them', () => {
    expect(mergeSchedulerConfig(BASE_CONFIG, { maxInFlight: 5.7, submitDelaySeconds: -3 })).toEqual({
      ...BASE_CONFIG,
      maxInFlight: 5,
      submitDelaySeconds: 0,
    });
    expect(
      mergeSchedulerConfig(BASE_CONFIG, { maxInFlight: 0, pollIntervalSeconds: 0, backtestTimeoutMinutes: 0 }),
    ).toEqual({ ...BASE_CONFIG, maxInFlight: 1, pollIntervalSeconds: 1, backtestTimeoutMinutes: 1 });
  });

  it('keeps current values for missing or non-finite fields', () => {
    expect(mergeSchedulerConfig(BASE_CONFIG, { maxInFlight: Number.NaN, submitDelaySeconds: undefined })).toEqual(
      BASE_CONFIG,
    );
  });
});

describe('isAbortError', () => {
  it('recognises scheduler and DOM abort errors', () => {
    expect(isAbortError(new SchedulerAbortError())).toBe(true);
    expect(new SchedulerAbortError().message).toBe(SCHEDULER_ABORT_MESSAGE);
    expect(isAbortError(new DOMException('aborted', 'AbortError'))).toBe(true);
    expect(isAbortError(new Error('Остановлено'))).toBe(false);
    expect(isAbortError('AbortError')).toBe(false);
  });
});

// ═══════════════════════════════════════════════════════════════════
// ПЛАНИРОВЩИК
// ═══════════════════════════════════════════════════════════════════

describe('BacktestScheduler: submission spacing and limits', () => {
  it('spaces POSTs by submitDelaySeconds in FIFO order and reports submissions', async () => {
    const { clock, veles, submitted, scheduler } = createHarness({ maxInFlight: 5 });
    const symbols = ['ADA', 'BTC', 'ETH', 'SOL'];
    const controller = new AbortController();
    const results = symbols.map((symbol) => track(scheduler.run(makeJob(symbol), controller.signal), clock));

    // Первый POST уже отправлен, но ещё не ответил — задача считается в очереди, а не в работе.
    expect(scheduler.queued).toBe(4);
    expect(scheduler.inFlight).toBe(0);
    await flushTasks();
    expect(scheduler.queued).toBe(3);
    expect(scheduler.inFlight).toBe(1);
    await clock.runUntil(allSettled(results));

    expect(veles.posts.map((post) => post.at)).toEqual([0, 31_000, 62_000, 93_000]);
    expect(veles.posts.map((post) => post.name)).toEqual(['PO ADA', 'PO BTC', 'PO ETH', 'PO SOL']);
    expect(results.map((result) => result.value)).toEqual([1000, 1001, 1002, 1003]);
    expect(submitted).toEqual([
      { key: 'h1|ADA', id: 1000 },
      { key: 'h1|BTC', id: 1001 },
      { key: 'h1|ETH', id: 1002 },
      { key: 'h1|SOL', id: 1003 },
    ]);
    expect(scheduler.inFlight).toBe(0);
    expect(scheduler.queued).toBe(0);
    expect(clock.pendingTimers()).toBe(0);
  });

  it('first poll happens one interval after submission and resolves when FINISHED', async () => {
    const { clock, veles, scheduler } = createHarness({}, { durationMs: 25_000 });
    const result = track(scheduler.run(makeJob('ADA'), new AbortController().signal), clock);

    await clock.runUntil(() => result.settled);

    expect(result.value).toBe(1000);
    expect(result.settledAt).toBe(30_000);
    expect(veles.fetchStatus).toHaveBeenCalledTimes(3);
  });

  it('never exceeds maxInFlight and refills a slot as soon as a backtest finishes', async () => {
    const { clock, veles, scheduler } = createHarness(
      { maxInFlight: 2, submitDelaySeconds: 1 },
      { durationMs: 100_000 },
    );
    const controller = new AbortController();
    const results = ['A', 'B', 'C', 'D', 'E'].map((symbol) =>
      track(scheduler.run(makeJob(symbol), controller.signal), clock),
    );

    await clock.advanceTo(2_000);
    expect(scheduler.inFlight).toBe(2);
    expect(scheduler.queued).toBe(3);

    await clock.runUntil(allSettled(results));

    expect(veles.posts.map((post) => post.at)).toEqual([0, 1_000, 100_000, 101_000, 200_000]);
    expect(veles.posts.every((post) => post.inFlight < 2)).toBe(true);
    expect(results.every((result) => typeof result.value === 'number')).toBe(true);
  });

  it('applies the limits globally across independent run() callers', async () => {
    const { clock, veles, scheduler } = createHarness(
      { maxInFlight: 1, submitDelaySeconds: 5 },
      { durationMs: 10_000 },
    );
    const first = new AbortController();
    const second = new AbortController();
    const results = [
      track(scheduler.run(makeJob('ADA', 'h1'), first.signal), clock),
      track(scheduler.run(makeJob('ADA', 'h2'), second.signal), clock),
      track(scheduler.run(makeJob('BTC', 'h1'), first.signal), clock),
    ];

    await clock.runUntil(allSettled(results));

    expect(veles.posts.map((post) => post.at)).toEqual([0, 10_000, 20_000]);
    expect(veles.posts.every((post) => post.inFlight === 0)).toBe(true);
  });

  it('updateConfig raising maxInFlight wakes the queue immediately', async () => {
    const { clock, veles, scheduler } = createHarness(
      { maxInFlight: 1, submitDelaySeconds: 1 },
      { durationMs: 1_000_000 },
    );
    const controller = new AbortController();
    track(scheduler.run(makeJob('ADA'), controller.signal), clock);
    track(scheduler.run(makeJob('BTC'), controller.signal), clock);

    await clock.advanceTo(5_000);
    expect(veles.posts).toHaveLength(1);

    scheduler.updateConfig({ maxInFlight: 2 });
    await flushTasks();

    expect(veles.posts.map((post) => post.at)).toEqual([0, 5_000]);
    controller.abort();
    await flushTasks();
  });

  it('updateConfig with a new delay is used for the next submission', async () => {
    const { clock, veles, scheduler } = createHarness({ maxInFlight: 5 }, { durationMs: 1_000 });
    const controller = new AbortController();
    const results = [
      track(scheduler.run(makeJob('ADA'), controller.signal), clock),
      track(scheduler.run(makeJob('BTC'), controller.signal), clock),
    ];

    await flushTasks();
    scheduler.updateConfig({ submitDelaySeconds: 10 });
    await clock.runUntil(allSettled(results));

    expect(veles.posts.map((post) => post.at)).toEqual([0, 10_000]);
    expect(scheduler.effectiveSubmitDelaySeconds).toBe(10);
  });
});

describe('BacktestScheduler: POST errors', () => {
  it('on 429 pauses globally for 60 s, escalates the delay ×1.5 and decays it ×0.9 after successes', async () => {
    const { clock, veles, logs, scheduler } = createHarness({ maxInFlight: 5 });
    veles.postFailures.push(new Error('HTTP 429: Действие временно заблокировано (Too Many Requests)'));
    const controller = new AbortController();
    const results = ['ADA', 'BTC', 'ETH'].map((symbol) =>
      track(scheduler.run(makeJob(symbol), controller.signal), clock),
    );

    await clock.advanceTo(1_000);
    expect(scheduler.effectiveSubmitDelaySeconds).toBeCloseTo(46.5);

    await clock.runUntil(allSettled(results));

    const times = veles.posts.map((post) => post.at);
    expect(times[0]).toBe(0);
    expect(times[1]).toBe(60_000);
    expect(times[2]).toBeCloseTo(60_000 + 41_850, 0);
    expect(times[3]).toBeCloseTo(60_000 + 41_850 + 37_665, 0);
    expect(veles.posts.map((post) => post.name)).toEqual(['PO ADA', 'PO ADA', 'PO BTC', 'PO ETH']);
    expect(results.every((result) => typeof result.value === 'number')).toBe(true);
    expect(scheduler.effectiveSubmitDelaySeconds).toBeCloseTo(46.5 * 0.9 ** 3);

    const warning = logs.find((log) => log.level === 'warning');
    expect(warning?.message).toContain('пауза 60 с');
    expect(warning?.message).toContain('46.5 с');
  });

  it('honors retryAfterSeconds from the error', async () => {
    const { clock, veles, scheduler } = createHarness({ maxInFlight: 5 });
    veles.postFailures.push(Object.assign(new Error('Too Many Requests'), { retryAfterSeconds: 90 }));
    const result = track(scheduler.run(makeJob('ADA'), new AbortController().signal), clock);

    await clock.runUntil(() => result.settled);

    expect(veles.posts.map((post) => post.at)).toEqual([0, 90_000]);
    expect(result.value).toBe(1000);
  });

  it('keeps retrying through repeated 429 without consuming recoverable attempts', async () => {
    const { clock, veles, scheduler } = createHarness({ maxInFlight: 5, submitDelaySeconds: 0 });
    for (let index = 0; index < 7; index += 1) {
      veles.postFailures.push(new BacktestSubmitError('HTTP 429', 429, 1));
    }
    const result = track(scheduler.run(makeJob('ADA'), new AbortController().signal), clock);

    await clock.runUntil(() => result.settled);

    expect(veles.postBacktest).toHaveBeenCalledTimes(8);
    expect(result.value).toBe(1000);
  });

  it('retries recoverable errors with exponential backoff keeping the job at the head of the queue', async () => {
    const { clock, veles, logs, scheduler } = createHarness({ maxInFlight: 5, submitDelaySeconds: 0 });
    veles.postFailures.push(new Error('Таймаут ответа от вкладки'), new Error('HTTP 502 Bad Gateway'));
    const controller = new AbortController();
    const results = [
      track(scheduler.run(makeJob('ADA'), controller.signal), clock),
      track(scheduler.run(makeJob('BTC'), controller.signal), clock),
    ];

    await clock.runUntil(allSettled(results));

    expect(veles.posts.map((post) => [post.at, post.name])).toEqual([
      [0, 'PO ADA'],
      [10_000, 'PO ADA'],
      [30_000, 'PO ADA'],
      [30_000, 'PO BTC'],
    ]);
    expect(results.map((result) => result.value)).toEqual([1000, 1001]);
    const warnings = logs.filter((log) => log.level === 'warning').map((log) => log.message);
    expect(warnings).toHaveLength(2);
    expect(warnings[0]).toContain('попытка 1/5');
    expect(warnings[0]).toContain('Повтор через 10 с');
    expect(warnings[1]).toContain('попытка 2/5');
  });

  it('rejects after 5 failed attempts and moves on to the next job', async () => {
    const { clock, veles, logs, scheduler } = createHarness({ maxInFlight: 5, submitDelaySeconds: 0 });
    for (let index = 0; index < 5; index += 1) {
      veles.postFailures.push(new Error('Failed to fetch'));
    }
    const controller = new AbortController();
    const failing = track(scheduler.run(makeJob('ADA'), controller.signal), clock);
    const next = track(scheduler.run(makeJob('BTC'), controller.signal), clock);

    await clock.runUntil(allSettled([failing, next]));

    expect(veles.posts.slice(0, 5).map((post) => post.at)).toEqual([0, 10_000, 30_000, 70_000, 150_000]);
    expect(errorMessage(failing.error)).toContain('за 5 попыток');
    expect(errorMessage(failing.error)).toContain('Failed to fetch');
    expect(isAbortError(failing.error)).toBe(false);
    // Сбой инфраструктуры, а не кандидата: оркестратор остановит запуск с возможностью продолжить.
    expect(isTransientBacktestError(failing.error)).toBe(true);
    expect(failing.error).toMatchObject({ reason: 'submit' });
    expect(next.value).toBe(1000);
    expect(veles.posts[5]).toMatchObject({ at: 150_000, name: 'PO BTC' });
    expect(logs.some((log) => log.level === 'error')).toBe(true);
  });

  it('rejects non-recoverable 4xx immediately with the original error', async () => {
    const { clock, veles, logs, scheduler } = createHarness({ maxInFlight: 5 });
    const badRequest = new Error('HTTP 400 Bad Request: Некорректные параметры');
    veles.postFailures.push(badRequest);
    const controller = new AbortController();
    const failing = track(scheduler.run(makeJob('ADA'), controller.signal), clock);
    const next = track(scheduler.run(makeJob('BTC'), controller.signal), clock);

    await clock.runUntil(allSettled([failing, next]));

    expect(failing.error).toBe(badRequest);
    expect(failing.settledAt).toBe(0);
    expect(veles.posts.map((post) => [post.at, post.name])).toEqual([
      [0, 'PO ADA'],
      [31_000, 'PO BTC'],
    ]);
    expect(next.value).toBe(1000);
    expect(logs.find((log) => log.level === 'error')?.message).toContain('Некорректные параметры');
  });

  it('accepts a numeric-string id from the response', async () => {
    const { clock, veles, submitted, scheduler } = createHarness({ maxInFlight: 5, submitDelaySeconds: 0 });
    veles.postBacktest.mockResolvedValueOnce({ id: '1000' });
    veles.addExisting(1000, 0);
    const result = track(scheduler.run(makeJob('ADA'), new AbortController().signal), clock);

    await clock.runUntil(() => result.settled);

    expect(veles.postBacktest).toHaveBeenCalledTimes(1);
    expect(result.value).toBe(1000);
    expect(submitted).toEqual([{ key: 'h1|ADA', id: 1000 }]);
  });

  it('never re-POSTs a 2xx response without an id: fails without a duplicate when nothing is found', async () => {
    const { clock, veles, logs, submitted, scheduler } = createHarness({ maxInFlight: 5, submitDelaySeconds: 0 });
    veles.postBacktest.mockResolvedValueOnce({ id: Number.NaN });
    const result = track(scheduler.run(makeJob('ADA'), new AbortController().signal), clock);

    await clock.runUntil(() => result.settled);

    expect(veles.postBacktest).toHaveBeenCalledTimes(1);
    expect(veles.findBacktestsByName).toHaveBeenCalledWith('PO ADA', null, 0);
    expect(result.error).toBeInstanceOf(BacktestSubmitError);
    expect(errorMessage(result.error)).toContain('не вернул идентификатор');
    expect(isTransientBacktestError(result.error)).toBe(false);
    expect(submitted).toEqual([]);
    expect(logs.some((log) => log.level === 'error')).toBe(true);
  });

  it('adopts the backtest created by a 2xx response without an id when it is found by name', async () => {
    const { clock, veles, submitted, scheduler } = createHarness({ maxInFlight: 5, submitDelaySeconds: 0 });
    veles.respondWithoutId(1);
    const result = track(scheduler.run(makeJob('ADA'), new AbortController().signal), clock);

    await clock.runUntil(() => result.settled);

    expect(veles.posts).toHaveLength(1);
    expect(result.value).toBe(1000);
    expect(submitted).toEqual([{ key: 'h1|ADA', id: 1000 }]);
  });

  it('after a lost response looks the backtest up by name instead of POSTing a duplicate', async () => {
    const { clock, veles, logs, submitted, scheduler } = createHarness({ maxInFlight: 5, submitDelaySeconds: 0 });
    veles.lostResponses.push(new Error('Таймаут ответа от вкладки'));
    const result = track(scheduler.run(makeJob('ADA'), new AbortController().signal), clock);

    await clock.runUntil(() => result.settled);

    expect(veles.posts).toHaveLength(1);
    expect(veles.findBacktestsByName).toHaveBeenCalledTimes(1);
    expect(result.value).toBe(1000);
    expect(submitted).toEqual([{ key: 'h1|ADA', id: 1000 }]);
    expect(logs.some((log) => log.message.includes('уже создан в Veles (#1000)'))).toBe(true);
  });

  it('re-POSTs after a recoverable error when the lookup finds nothing', async () => {
    const { clock, veles, submitted, scheduler } = createHarness({ maxInFlight: 5, submitDelaySeconds: 0 });
    veles.postFailures.push(new Error('HTTP 502 Bad Gateway'));
    const result = track(scheduler.run(makeJob('ADA'), new AbortController().signal), clock);

    await clock.runUntil(() => result.settled);

    expect(veles.findBacktestsByName).toHaveBeenCalledTimes(1);
    expect(veles.postBacktest).toHaveBeenCalledTimes(2);
    expect(result.value).toBe(1000);
    expect(submitted).toEqual([{ key: 'h1|ADA', id: 1000 }]);
  });

  it('does not POST again while the lookup itself fails; the failed lookup counts as an attempt', async () => {
    const { clock, veles, scheduler } = createHarness({ maxInFlight: 5, submitDelaySeconds: 0 });
    veles.postFailures.push(new Error('Таймаут ответа от вкладки'));
    veles.findBacktestsByName.mockRejectedValueOnce(new Error('Failed to fetch'));
    const result = track(scheduler.run(makeJob('ADA'), new AbortController().signal), clock);

    await clock.runUntil(() => result.settled);

    expect(veles.findBacktestsByName).toHaveBeenCalledTimes(2);
    expect(veles.posts.map((post) => post.at)).toEqual([0, 30_000]);
    expect(result.value).toBe(1000);
  });

  it('does not adopt backtests it already knows (another job with the same name)', async () => {
    const { clock, veles, scheduler } = createHarness({ maxInFlight: 5, submitDelaySeconds: 0 });
    const controller = new AbortController();
    const first = track(scheduler.run(makeJob('ADA', 'h1'), controller.signal), clock);
    await clock.runUntil(() => first.settled);
    veles.postFailures.push(new Error('HTTP 503 Service Unavailable'));
    const second = track(scheduler.run(makeJob('ADA', 'h2'), controller.signal), clock);

    await clock.runUntil(() => second.settled);

    expect(first.value).toBe(1000);
    expect(second.value).toBe(1001);
  });
});

describe('BacktestScheduler: beforeSubmit', () => {
  it('lets the caller cancel a job right before its POST', async () => {
    const clock = createVirtualClock();
    const veles = createFakeVeles(clock);
    const controller = new AbortController();
    const scheduler = new BacktestScheduler(
      BASE_CONFIG,
      {
        onSubmitted: () => undefined,
        onLog: () => undefined,
        beforeSubmit: () => {
          controller.abort();
          return null;
        },
      },
      { ...veles, sleep: clock.sleep, now: clock.now },
    );
    const result = track(scheduler.run(makeJob('ADA'), controller.signal), clock);

    await clock.runUntil(() => result.settled);

    expect(isAbortError(result.error)).toBe(true);
    expect(veles.postBacktest).not.toHaveBeenCalled();
    expect(scheduler.queued).toBe(0);
  });

  it('polls an already created backtest returned by beforeSubmit instead of POSTing', async () => {
    const clock = createVirtualClock();
    const veles = createFakeVeles(clock);
    veles.addExisting(555, -100_000);
    const submitted: number[] = [];
    const scheduler = new BacktestScheduler(
      BASE_CONFIG,
      { onSubmitted: (_key, id) => submitted.push(id), onLog: () => undefined, beforeSubmit: () => 555 },
      { ...veles, sleep: clock.sleep, now: clock.now },
    );
    const result = track(scheduler.run(makeJob('ADA'), new AbortController().signal), clock);

    await clock.runUntil(() => result.settled);

    expect(result.value).toBe(555);
    expect(veles.postBacktest).not.toHaveBeenCalled();
    expect(submitted).toEqual([]);
  });
});

describe('BacktestScheduler: polling', () => {
  it('rejects with the status text when the backtest failed', async () => {
    const { clock, scheduler } = createHarness({}, { status: () => 'FAILED' });
    const result = track(scheduler.run(makeJob('ADA'), new AbortController().signal), clock);

    await clock.runUntil(() => result.settled);

    expect(errorMessage(result.error)).toContain('FAILED');
    expect(errorMessage(result.error)).toContain('1000');
    expect(scheduler.inFlight).toBe(0);
  });

  it('treats unknown statuses as done when statistics exist and warns once per distinct status', async () => {
    const { clock, logs, scheduler } = createHarness(
      { maxInFlight: 5, submitDelaySeconds: 1 },
      {
        status: (backtest) => (backtest.id === 1002 ? 'OTHER' : 'MYSTERY'),
        statisticsExists: () => true,
      },
    );
    const controller = new AbortController();
    const results = ['ADA', 'BTC', 'ETH'].map((symbol) =>
      track(scheduler.run(makeJob(symbol), controller.signal), clock),
    );

    await clock.runUntil(allSettled(results));

    expect(results.map((result) => result.value)).toEqual([1000, 1001, 1002]);
    const warnings = logs.filter((log) => log.level === 'warning').map((log) => log.message);
    expect(warnings).toHaveLength(2);
    expect(warnings[0]).toContain('Неизвестный статус бэктеста «MYSTERY»');
    expect(warnings[0]).toContain('считаю завершённым');
    expect(warnings[1]).toContain('«OTHER»');
  });

  it('keeps waiting on unknown/null status until statistics appear', async () => {
    const { clock, veles, logs, scheduler } = createHarness(
      {},
      { status: () => null, statisticsExists: (backtest, now) => now - backtest.createdAt >= 50_000 },
    );
    const result = track(scheduler.run(makeJob('ADA'), new AbortController().signal), clock);

    await clock.runUntil(() => result.settled);

    expect(result.value).toBe(1000);
    expect(result.settledAt).toBe(50_000);
    expect(veles.fetchStatisticsExists).toHaveBeenCalledTimes(5);
    expect(logs.filter((log) => log.level === 'warning')).toHaveLength(1);
    expect(logs[0].message).toContain('«пусто»');
  });

  it('does not consult statistics while the status says running', async () => {
    const { clock, veles, scheduler } = createHarness({}, { durationMs: 30_000, statisticsExists: () => true });
    const result = track(scheduler.run(makeJob('ADA'), new AbortController().signal), clock);

    await clock.runUntil(() => result.settled);

    expect(result.settledAt).toBe(30_000);
    expect(veles.fetchStatisticsExists).not.toHaveBeenCalled();
  });

  it('logs poll errors sparsely and keeps polling', async () => {
    const clock = createVirtualClock();
    let calls = 0;
    const logs: LogRecord[] = [];
    const scheduler = new BacktestScheduler(
      BASE_CONFIG,
      { onSubmitted: () => undefined, onLog: (level, message) => logs.push({ level, message }) },
      {
        postBacktest: async () => ({ id: 7 }),
        fetchStatus: async () => {
          calls += 1;
          if (calls <= 12) {
            throw new Error('Таймаут ответа от вкладки');
          }
          return 'FINISHED';
        },
        fetchStatisticsExists: async () => false,
        sleep: clock.sleep,
        now: clock.now,
      },
    );
    const result = track(scheduler.run(makeJob('ADA'), new AbortController().signal), clock);

    await clock.runUntil(() => result.settled);

    expect(result.value).toBe(7);
    expect(calls).toBe(13);
    const warnings = logs.filter((log) => log.level === 'warning').map((log) => log.message);
    expect(warnings).toHaveLength(2);
    expect(warnings[0]).toContain('попытка 1');
    expect(warnings[1]).toContain('попытка 10');
  });

  it('rejects with a timeout after backtestTimeoutMinutes from submission', async () => {
    const { clock, scheduler } = createHarness({ backtestTimeoutMinutes: 1 }, { status: () => 'RUNNING' });
    const result = track(scheduler.run(makeJob('ADA'), new AbortController().signal), clock);

    await clock.runUntil(() => result.settled);

    expect(errorMessage(result.error)).toContain('Таймаут бэктеста 1000');
    expect(result.settledAt).toBe(60_000);
    expect(scheduler.inFlight).toBe(0);
  });

  it('reports the timeout as a transient error with the backtest id', async () => {
    const { clock, scheduler } = createHarness({ backtestTimeoutMinutes: 1 }, { status: () => 'RUNNING' });
    const result = track(scheduler.run(makeJob('ADA'), new AbortController().signal), clock);

    await clock.runUntil(() => result.settled);

    expect(result.error).toBeInstanceOf(TransientBacktestError);
    expect(result.error).toMatchObject({ reason: 'timeout', backtestId: 1000 });
  });

  it('does not time out while polls fail; times out on the first successful read past the deadline', async () => {
    const clock = createVirtualClock();
    let calls = 0;
    const scheduler = new BacktestScheduler(
      { ...BASE_CONFIG, backtestTimeoutMinutes: 1 },
      { onSubmitted: () => undefined, onLog: () => undefined },
      {
        postBacktest: async () => ({ id: 7 }),
        fetchStatus: async () => {
          calls += 1;
          if (calls <= 12) {
            throw new Error('Failed to fetch');
          }
          return 'RUNNING';
        },
        fetchStatisticsExists: async () => false,
        findBacktestsByName: async () => [],
        sleep: clock.sleep,
        now: clock.now,
      },
    );
    const result = track(scheduler.run(makeJob('ADA'), new AbortController().signal), clock);

    await clock.runUntil(() => result.settled);

    expect(errorMessage(result.error)).toContain('Таймаут');
    expect(calls).toBe(13);
    expect(result.settledAt).toBe(130_000);
  });

  it('a poll auth error rejects at once instead of counting towards the timeout', async () => {
    const clock = createVirtualClock();
    const scheduler = new BacktestScheduler(
      BASE_CONFIG,
      { onSubmitted: () => undefined, onLog: () => undefined },
      {
        postBacktest: async () => ({ id: 7 }),
        fetchStatus: async () => {
          throw new Error('HTTP 401 Unauthorized');
        },
        fetchStatisticsExists: async () => false,
        findBacktestsByName: async () => [],
        sleep: clock.sleep,
        now: clock.now,
      },
    );
    const result = track(scheduler.run(makeJob('ADA'), new AbortController().signal), clock);

    await clock.runUntil(() => result.settled);

    expect(errorMessage(result.error)).toContain('HTTP 401');
    expect(isAuthError(result.error)).toBe(true);
    expect(isTransientBacktestError(result.error)).toBe(false);
    expect(result.settledAt).toBe(10_000);
    expect(scheduler.inFlight).toBe(0);
  });

  it('ignores a wall-clock gap (laptop sleep) much larger than the poll interval', async () => {
    const clock = createVirtualClock();
    let jump = 0;
    const scheduler = new BacktestScheduler(
      { ...BASE_CONFIG, backtestTimeoutMinutes: 45 },
      { onSubmitted: () => undefined, onLog: () => undefined },
      {
        postBacktest: async () => ({ id: 7 }),
        fetchStatus: async () => 'RUNNING',
        fetchStatisticsExists: async () => false,
        findBacktestsByName: async () => [],
        sleep: clock.sleep,
        now: () => clock.now() + jump,
      },
    );
    const result = track(scheduler.run(makeJob('ADA'), new AbortController().signal), clock);
    await clock.advanceTo(20_000);

    // Ноутбук «проспал» 2 часа между опросами: скачок часов не засчитывается целиком.
    jump = 2 * 3_600_000;
    await clock.advanceTo(40_000);
    expect(result.settled).toBe(false);

    await clock.runUntil(() => result.settled);
    expect(isTransientBacktestError(result.error)).toBe(true);
    // 45 мин ожидания = виртуальное время + не больше (интервал + 2 мин) за шаг со скачком.
    expect(result.settledAt).toBe(45 * 60_000 - 120_000);
  });
});

describe('BacktestScheduler: abort and pause', () => {
  it('rejects immediately when the signal is already aborted', async () => {
    const { clock, veles, scheduler } = createHarness();
    const controller = new AbortController();
    controller.abort();

    const runResult = track(scheduler.run(makeJob('ADA'), controller.signal), clock);
    const attachResult = track(scheduler.attach('h1|BTC', 5, controller.signal), clock);
    await flushTasks();

    expect(isAbortError(runResult.error)).toBe(true);
    expect(isAbortError(attachResult.error)).toBe(true);
    expect(veles.postBacktest).not.toHaveBeenCalled();
    expect(veles.fetchStatus).not.toHaveBeenCalled();
  });

  it('on abort rejects queued and in-flight jobs, stops polling and sends no further POSTs', async () => {
    const { clock, veles, scheduler } = createHarness({ maxInFlight: 1 }, { durationMs: 100_000 });
    const controller = new AbortController();
    const results = ['ADA', 'BTC', 'ETH'].map((symbol) =>
      track(scheduler.run(makeJob(symbol), controller.signal), clock),
    );

    await clock.advanceTo(15_000);
    expect(veles.posts).toHaveLength(1);
    const pollsBeforeAbort = veles.fetchStatus.mock.calls.length;

    controller.abort();
    await clock.runUntil(allSettled(results));

    for (const result of results) {
      expect(isAbortError(result.error)).toBe(true);
      expect(errorMessage(result.error)).toBe(SCHEDULER_ABORT_MESSAGE);
    }
    expect(scheduler.inFlight).toBe(0);
    expect(scheduler.queued).toBe(0);

    await clock.advanceTo(1_000_000);
    expect(veles.posts).toHaveLength(1);
    expect(veles.fetchStatus.mock.calls.length).toBe(pollsBeforeAbort);
    expect(clock.pendingTimers()).toBe(0);
  });

  it('only aborts jobs bound to the aborted signal', async () => {
    const { clock, veles, scheduler } = createHarness(
      { maxInFlight: 1, submitDelaySeconds: 1 },
      { durationMs: 10_000 },
    );
    const stopped = new AbortController();
    const alive = new AbortController();
    const first = track(scheduler.run(makeJob('ADA', 'h1'), stopped.signal), clock);
    const second = track(scheduler.run(makeJob('ADA', 'h2'), alive.signal), clock);

    await clock.advanceTo(1_000);
    stopped.abort();
    await clock.runUntil(allSettled([first, second]));

    expect(isAbortError(first.error)).toBe(true);
    expect(second.value).toBe(1001);
    expect(veles.posts.map((post) => post.at)).toEqual([0, 1_000]);
  });

  it('aborting during a POST still reports the created backtest but does not poll it', async () => {
    const clock = createVirtualClock();
    const submitted: Array<{ key: string; id: number }> = [];
    let releasePost: (value: { id: number }) => void = () => undefined;
    const fetchStatus = vi.fn(async () => 'FINISHED');
    const scheduler = new BacktestScheduler(
      BASE_CONFIG,
      { onSubmitted: (key, id) => submitted.push({ key, id }), onLog: () => undefined },
      {
        postBacktest: () =>
          new Promise<{ id: number }>((resolve) => {
            releasePost = resolve;
          }),
        fetchStatus,
        fetchStatisticsExists: async () => false,
        sleep: clock.sleep,
        now: clock.now,
      },
    );
    const controller = new AbortController();
    const result = track(scheduler.run(makeJob('ADA'), controller.signal), clock);
    await flushTasks();
    expect(scheduler.queued).toBe(1);

    controller.abort();
    await flushTasks();
    expect(isAbortError(result.error)).toBe(true);

    releasePost({ id: 77 });
    await clock.runUntil(() => false);

    expect(submitted).toEqual([{ key: 'h1|ADA', id: 77 }]);
    expect(fetchStatus).not.toHaveBeenCalled();
    expect(scheduler.inFlight).toBe(0);
    expect(scheduler.queued).toBe(0);
  });

  it('a 429 answered after abort still applies the global cooldown to other callers', async () => {
    const clock = createVirtualClock();
    let rejectPost: (error: Error) => void = () => undefined;
    let postCalls = 0;
    const postTimes: number[] = [];
    const scheduler = new BacktestScheduler(
      { ...BASE_CONFIG, submitDelaySeconds: 1 },
      { onSubmitted: () => undefined, onLog: () => undefined },
      {
        postBacktest: (): Promise<{ id: number }> => {
          postCalls += 1;
          postTimes.push(clock.now());
          if (postCalls === 1) {
            return new Promise<{ id: number }>((_resolve, reject) => {
              rejectPost = reject;
            });
          }
          return Promise.resolve({ id: 100 + postCalls });
        },
        fetchStatus: async () => 'FINISHED',
        fetchStatisticsExists: async () => false,
        sleep: clock.sleep,
        now: clock.now,
      },
    );
    const stopped = new AbortController();
    const first = track(scheduler.run(makeJob('ADA', 'h1'), stopped.signal), clock);
    const second = track(scheduler.run(makeJob('ADA', 'h2'), new AbortController().signal), clock);
    await flushTasks();

    stopped.abort();
    rejectPost(new Error('HTTP 429 Too Many Requests'));
    await clock.runUntil(allSettled([first, second]));

    expect(isAbortError(first.error)).toBe(true);
    expect(second.value).toBe(102);
    expect(postTimes).toEqual([0, 60_000]);
    expect(scheduler.effectiveSubmitDelaySeconds).toBeCloseTo(1.35);
  });

  it('aborting during retry backoff rejects without another POST', async () => {
    const { clock, veles, scheduler } = createHarness({ submitDelaySeconds: 0 });
    veles.postFailures.push(new Error('HTTP 503 Service Unavailable'));
    const controller = new AbortController();
    const result = track(scheduler.run(makeJob('ADA'), controller.signal), clock);

    await clock.advanceTo(5_000);
    controller.abort();
    await clock.runUntil(() => result.settled);
    await clock.advanceTo(100_000);

    expect(isAbortError(result.error)).toBe(true);
    expect(veles.postBacktest).toHaveBeenCalledTimes(1);
    expect(clock.pendingTimers()).toBe(0);
  });

  it('pause blocks new submissions while in-flight polling continues', async () => {
    const { clock, veles, scheduler } = createHarness({ submitDelaySeconds: 1 }, { durationMs: 100_000 });
    const controller = new AbortController();
    const first = track(scheduler.run(makeJob('ADA'), controller.signal), clock);
    await flushTasks();
    scheduler.setPaused(true);
    expect(scheduler.isPaused).toBe(true);
    const second = track(scheduler.run(makeJob('BTC'), controller.signal), clock);

    await clock.advanceTo(200_000);
    expect(first.value).toBe(1000);
    expect(first.settledAt).toBe(100_000);
    expect(second.settled).toBe(false);
    expect(veles.posts).toHaveLength(1);
    expect(scheduler.queued).toBe(1);

    scheduler.setPaused(false);
    await clock.runUntil(() => second.settled);

    expect(veles.posts.map((post) => post.at)).toEqual([0, 200_000]);
    expect(second.value).toBe(1001);
  });
});

describe('BacktestScheduler: attach', () => {
  it('polls an existing backtest immediately and occupies an in-flight slot', async () => {
    const { clock, veles, submitted, scheduler } = createHarness(
      { maxInFlight: 1, submitDelaySeconds: 1 },
      { durationMs: 100_000 },
    );
    veles.addExisting(555, -50_000);
    const controller = new AbortController();
    const attached = track(scheduler.attach('h0|ADA', 555, controller.signal), clock);
    const queued = track(scheduler.run(makeJob('BTC'), controller.signal), clock);

    await flushTasks();
    expect(veles.fetchStatus).toHaveBeenCalledWith(555);
    expect(scheduler.inFlight).toBe(1);
    expect(veles.postBacktest).not.toHaveBeenCalled();

    await clock.runUntil(allSettled([attached, queued]));

    expect(attached.value).toBe(555);
    expect(attached.settledAt).toBe(50_000);
    expect(veles.posts.map((post) => post.at)).toEqual([50_000]);
    expect(queued.value).toBe(1000);
    expect(submitted).toEqual([{ key: 'h1|BTC', id: 1000 }]);
  });

  it('resolves at once when the attached backtest is already finished', async () => {
    const { clock, veles, scheduler } = createHarness();
    veles.addExisting(555, -1_000_000);
    const attached = track(scheduler.attach('h0|ADA', 555, new AbortController().signal), clock);

    await clock.runUntil(() => attached.settled);

    expect(attached.value).toBe(555);
    expect(attached.settledAt).toBe(0);
  });

  it('measures the timeout from the original submission time', async () => {
    const { clock, veles, scheduler } = createHarness({ backtestTimeoutMinutes: 45 }, { status: () => 'RUNNING' });
    veles.addExisting(555, 0);
    await clock.advanceTo(50 * 60_000);
    const attached = track(scheduler.attach('h0|ADA', 555, new AbortController().signal, 40 * 60_000), clock);

    await clock.runUntil(() => attached.settled);

    // Создан 10 минут назад: до таймаута осталось 35 минут, а не 45 заново.
    expect(isTransientBacktestError(attached.error)).toBe(true);
    expect(attached.settledAt).toBe(50 * 60_000 + 35 * 60_000);
  });

  it('rejects invalid ids and can be aborted', async () => {
    const { clock, veles, scheduler } = createHarness({}, { durationMs: 1_000_000 });
    const invalid = track(scheduler.attach('h0|ADA', 0, new AbortController().signal), clock);
    veles.addExisting(555, 0);
    const controller = new AbortController();
    const attached = track(scheduler.attach('h0|ETH', 555, controller.signal), clock);

    await clock.advanceTo(25_000);
    controller.abort();
    await clock.runUntil(allSettled([invalid, attached]));

    expect(errorMessage(invalid.error)).toContain('Некорректный идентификатор бэктеста');
    expect(isAbortError(attached.error)).toBe(true);
    expect(scheduler.inFlight).toBe(0);
  });
});

describe('BacktestScheduler: robustness', () => {
  it('survives throwing event callbacks', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const clock = createVirtualClock();
    const scheduler = new BacktestScheduler(
      BASE_CONFIG,
      {
        onSubmitted: () => {
          throw new Error('submitted boom');
        },
        onLog: () => {
          throw new Error('log boom');
        },
      },
      {
        postBacktest: async () => ({ id: 3 }),
        fetchStatus: async () => 'MYSTERY',
        fetchStatisticsExists: async () => true,
        sleep: clock.sleep,
        now: clock.now,
      },
    );
    const result = track(scheduler.run(makeJob('ADA'), new AbortController().signal), clock);

    await clock.runUntil(() => result.settled);

    expect(result.value).toBe(3);
    expect(warn).toHaveBeenCalledTimes(2);
  });
});

// ═══════════════════════════════════════════════════════════════════
// ЗАВИСИМОСТИ ПО УМОЛЧАНИЮ
// ═══════════════════════════════════════════════════════════════════

describe('abortableSleep', () => {
  it('resolves after the given time', async () => {
    vi.useFakeTimers();
    let done = false;
    const promise = abortableSleep(1_000).then(() => {
      done = true;
    });

    await vi.advanceTimersByTimeAsync(999);
    expect(done).toBe(false);
    await vi.advanceTimersByTimeAsync(1);
    await promise;
    expect(done).toBe(true);
  });

  it('rejects with an abort error for an already aborted signal', async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(abortableSleep(1_000, controller.signal)).rejects.toBeInstanceOf(SchedulerAbortError);
  });

  it('rejects on abort and clears its timer', async () => {
    vi.useFakeTimers();
    const controller = new AbortController();
    const promise = abortableSleep(60_000, controller.signal);
    expect(vi.getTimerCount()).toBe(1);

    controller.abort();

    await expect(promise).rejects.toThrow(SCHEDULER_ABORT_MESSAGE);
    expect(vi.getTimerCount()).toBe(0);
  });
});

describe('resolveSchedulerDeps', () => {
  it('uses overrides when given and defaults for missing or undefined entries', () => {
    const now = (): number => 123;
    const deps = resolveSchedulerDeps({ now, sleep: undefined });

    expect(deps.now).toBe(now);
    expect(deps.sleep).toBe(abortableSleep);
    expect(typeof deps.postBacktest).toBe('function');
    expect(typeof deps.fetchStatus).toBe('function');
    expect(typeof deps.fetchStatisticsExists).toBe('function');
  });
});

describe('createDefaultSchedulerDeps', () => {
  const statisticsWith = (totalDeals: unknown): BacktestStatisticsDto =>
    ({ id: 1, totalDeals }) as unknown as BacktestStatisticsDto;

  it('postBacktest wraps the API call and returns only the id', async () => {
    vi.mocked(postBacktest).mockResolvedValueOnce({ id: 42, name: 'PO ADA', status: 'NEW' });
    const deps = createDefaultSchedulerDeps();
    const payload: BotStrategy = { id: null, name: 'PO ADA' };

    await expect(deps.postBacktest(payload)).resolves.toEqual({ id: 42 });
    expect(postBacktest).toHaveBeenCalledWith(payload);
  });

  it('postBacktest converts proxy errors into BacktestSubmitError with HTTP status', async () => {
    vi.mocked(postBacktest).mockRejectedValueOnce(
      new Error('HTTP 429 Too Many Requests: Действие временно заблокировано'),
    );
    const deps = createDefaultSchedulerDeps();

    const error = await deps.postBacktest({ id: null }).catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(BacktestSubmitError);
    expect(error).toMatchObject({
      message: 'HTTP 429 Too Many Requests: Действие временно заблокировано',
      status: 429,
      retryAfterSeconds: null,
    });
    expect(classifySubmitError(error)).toBe('rateLimit');
  });

  it('fetchStatus returns the config status or null', async () => {
    vi.mocked(fetchBacktestConfig).mockResolvedValueOnce({ id: 5, status: 'FINISHED' } as BacktestConfigDto);
    vi.mocked(fetchBacktestConfig).mockResolvedValueOnce({ id: 5 } as BacktestConfigDto);
    const deps = createDefaultSchedulerDeps();

    await expect(deps.fetchStatus(5)).resolves.toBe('FINISHED');
    await expect(deps.fetchStatus(5)).resolves.toBeNull();
    expect(fetchBacktestConfig).toHaveBeenCalledWith(5);
  });

  it('fetchStatus propagates transport errors (handled as poll errors by the scheduler)', async () => {
    vi.mocked(fetchBacktestConfig).mockRejectedValueOnce(new Error('HTTP 502 Bad Gateway'));
    const deps = createDefaultSchedulerDeps();

    await expect(deps.fetchStatus(5)).rejects.toThrow('HTTP 502');
  });

  it('fetchStatisticsExists requires a numeric totalDeals and swallows errors', async () => {
    vi.mocked(fetchBacktestStatistics).mockResolvedValueOnce(statisticsWith(12));
    vi.mocked(fetchBacktestStatistics).mockResolvedValueOnce(statisticsWith(0));
    vi.mocked(fetchBacktestStatistics).mockResolvedValueOnce(statisticsWith(null));
    vi.mocked(fetchBacktestStatistics).mockResolvedValueOnce(statisticsWith(Number.NaN));
    vi.mocked(fetchBacktestStatistics).mockRejectedValueOnce(new Error('HTTP 404 Not Found'));
    const deps = createDefaultSchedulerDeps();

    await expect(deps.fetchStatisticsExists(1)).resolves.toBe(true);
    await expect(deps.fetchStatisticsExists(1)).resolves.toBe(true);
    await expect(deps.fetchStatisticsExists(1)).resolves.toBe(false);
    await expect(deps.fetchStatisticsExists(1)).resolves.toBe(false);
    await expect(deps.fetchStatisticsExists(1)).resolves.toBe(false);
  });

  it('findBacktestsByName looks through the newest backtests by exact name, symbol and creation time', async () => {
    const item = (id: number, name: string, date: string, symbol = 'ADA/USDT'): BacktestStatisticsDto =>
      ({ id, name, date, symbol }) as BacktestStatisticsDto;
    const list: BacktestStatisticsListDto = {
      totalElements: 5,
      totalPages: 1,
      pageNumber: 0,
      content: [
        item(15, 'PO ADA', '2026-10-12T10:00:00.000Z'),
        item(14, 'PO ADA', '2026-10-12T10:00:00.000Z', 'XRP/USDT'),
        item(13, 'PO ADA ', '2026-10-12T10:00:00.000Z'),
        item(12, 'PO ADA', 'not-a-date'),
        item(11, 'PO ADA', '2026-10-11T10:00:00.000Z'),
      ],
    };
    vi.mocked(fetchBacktests).mockResolvedValueOnce(list);
    const deps = createDefaultSchedulerDeps();

    await expect(
      deps.findBacktestsByName('PO ADA', 'ADA/USDT', Date.parse('2026-10-12T10:02:00.000Z')),
    ).resolves.toEqual([15]);
    expect(fetchBacktests).toHaveBeenCalledWith({ page: 0, size: 20, sort: 'date,desc' });
  });

  it('a scheduler without injected deps runs on the real API wrappers and timers', async () => {
    vi.useFakeTimers();
    vi.mocked(postBacktest).mockResolvedValueOnce({ id: 9 });
    vi.mocked(fetchBacktestConfig).mockResolvedValueOnce({ id: 9, status: 'RUNNING' } as BacktestConfigDto);
    vi.mocked(fetchBacktestConfig).mockResolvedValueOnce({ id: 9, status: 'FINISHED' } as BacktestConfigDto);
    const submitted: number[] = [];
    const scheduler = new BacktestScheduler(
      { ...BASE_CONFIG, pollIntervalSeconds: 2 },
      { onSubmitted: (_key, id) => submitted.push(id), onLog: () => undefined },
    );

    const promise = scheduler.run(makeJob('ADA'), new AbortController().signal);
    await vi.advanceTimersByTimeAsync(4_000);

    await expect(promise).resolves.toBe(9);
    expect(submitted).toEqual([9]);
    expect(fetchBacktestConfig).toHaveBeenCalledTimes(2);
    expect(vi.getTimerCount()).toBe(0);
  });
});
