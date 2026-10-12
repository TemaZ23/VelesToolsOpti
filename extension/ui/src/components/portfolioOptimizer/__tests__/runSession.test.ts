import { describe, expect, it, vi } from 'vitest';
import type { BotStrategy } from '../../../api/backtestRunner';
import type { StrategyConditionDto } from '../../../api/bots.dtos';
import { extractParams } from '../../../lib/botParams';
import { SchedulerAbortError, type SchedulerEvents } from '../../../services/backtestScheduler';
import {
  DEFAULT_SCHEDULER_CONFIG,
  DEFAULT_SEARCH_CONFIG,
  MAX_PORTFOLIO_LOG_ENTRIES,
  type PortfolioBasketEvaluator,
  type PortfolioOptimizerDeps,
  type PortfolioOptimizerInput,
  type PortfolioScheduler,
} from '../../../services/portfolioOptimizer';
import type {
  PortfolioLogEntry,
  PortfolioMetrics,
  PortfolioOptimizerConfig,
  PortfolioOptimizerState,
  SchedulerConfig,
} from '../../../types/portfolioOptimizer';
import { PORTFOLIO_RUN_HEARTBEAT_KEY, type RunLockProvider, readRunHeartbeat } from '../runLock';
import { OTHER_TAB_RUN_MESSAGE, PortfolioRunSession } from '../runSession';

// ═══════════════════════════════════════════════════════════════════
// ФИКСТУРЫ
// ═══════════════════════════════════════════════════════════════════

const RSI_ID = 'conditions[0].value';

const condition = (indicator: string, value: number): StrategyConditionDto => ({
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
  symbols: ['BTC/USDT'],
  pair: { exchange: 'BYBIT_FUTURES', from: 'BTC', to: 'USDT', symbol: 'BTCUSDT' },
  exchange: 'BYBIT_FUTURES',
  algorithm: 'LONG',
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

const createConfig = (patch: Partial<PortfolioOptimizerConfig> = {}): PortfolioOptimizerConfig => ({
  botId: 77,
  botName: 'LONG 22',
  basket: ['ADA', 'XRP', 'SOL', 'DOGE'],
  screenBasket: ['ADA', 'SOL'],
  run: {
    periodFrom: '2022-01-01T00:00:00.000Z',
    periodTo: '2024-01-01T23:59:59.999Z',
    makerCommission: 0.02,
    takerCommission: 0.055,
    includeWicks: true,
  },
  block: 60,
  objective: { kind: 'pnlToRisk', folds: 3, maxMaeCap: null, minDeals: 0 },
  search: {
    ...DEFAULT_SEARCH_CONFIG,
    generations: 2,
    candidatesPerGeneration: 2,
    promotePerGeneration: 1,
    mutationsMin: 1,
    mutationsMax: 1,
    maxBacktests: 100,
    seed: 3,
  },
  scheduler: { ...DEFAULT_SCHEDULER_CONFIG },
  params: extractParams(createStrategy()).map((param) => ({ ...param, enabled: param.id === RSI_ID })),
  baseGroupId: null,
  ...patch,
});

const createInput = (patch: Partial<PortfolioOptimizerConfig> = {}): PortfolioOptimizerInput => ({
  config: createConfig(patch),
  baseStrategy: createStrategy(),
  quoteCurrency: 'USDT',
});

const METRICS: PortfolioMetrics = {
  totalProfit: 100,
  maxConcurrentMae: 50,
  maxDrawdown: 20,
  pnlToRisk: 2,
  totalDeals: 500,
  profitableDeals: 490,
  losingDeals: 10,
  openDeals: 0,
  activeMae: 0,
  netPerDay: 1,
  maxConcurrentPositions: 4,
  backtests: 2,
};

// ═══════════════════════════════════════════════════════════════════
// ФЕЙКОВЫЙ ПЛАНИРОВЩИК И ЗАВИСИМОСТИ
// ═══════════════════════════════════════════════════════════════════

interface FakeControl {
  /** true — бэктесты «считаются», пока тест не вызовет release(). */
  hold: boolean;
  submitted: number;
  attached: number[];
  pending: Array<() => void>;
  updates: Array<Partial<SchedulerConfig>>;
  pausedCalls: boolean[];
  saved: PortfolioOptimizerState | null;
  /** Шаг фейковых часов на каждый вызов now(), мс (0 — время стоит). */
  clockStep: number;
  /** События каждого созданного планировщика (по порядку сеансов). */
  events: SchedulerEvents[];
}

const createControl = (hold = false): FakeControl => ({
  hold,
  submitted: 0,
  attached: [],
  pending: [],
  updates: [],
  pausedCalls: [],
  saved: null,
  clockStep: 1_000,
  events: [],
});

const waitForBacktest = (control: FakeControl, backtestId: number, signal: AbortSignal): Promise<number> =>
  new Promise<number>((resolve, reject) => {
    if (signal.aborted) {
      reject(new SchedulerAbortError());
      return;
    }
    const finish = () => resolve(backtestId);
    if (control.hold) {
      control.pending.push(finish);
      signal.addEventListener('abort', () => reject(new SchedulerAbortError()), { once: true });
    } else {
      queueMicrotask(finish);
    }
  });

const createFakeScheduler =
  (control: FakeControl) =>
  (_config: SchedulerConfig, events: SchedulerEvents): PortfolioScheduler => {
    control.events.push(events);
    return {
      run: (job, signal) => {
        if (signal.aborted) {
          return Promise.reject(new SchedulerAbortError());
        }
        control.submitted += 1;
        const backtestId = 1000 + control.submitted;
        events.onSubmitted(job.key, backtestId);
        return waitForBacktest(control, backtestId, signal);
      },
      attach: (_key, backtestId, signal) => {
        control.attached.push(backtestId);
        return waitForBacktest(control, backtestId, signal);
      },
      setPaused: (paused) => {
        control.pausedCalls.push(paused);
      },
      updateConfig: (patch) => {
        control.updates.push(patch);
      },
      inFlight: 0,
      queued: 0,
    };
  };

const createDeps = (control: FakeControl): Partial<PortfolioOptimizerDeps> => {
  let clock = Date.parse('2026-10-12T08:00:00.000Z');
  const evaluateBasket: PortfolioBasketEvaluator = async (ids) => ({
    metrics: { ...METRICS, backtests: ids.length },
    folds: [],
  });
  return {
    createScheduler: createFakeScheduler(control),
    evaluateBasket,
    loadGroupBacktests: async () => ({ groupName: 'нет', backtests: [], failed: [] }),
    saveState: (state) => {
      control.saved = JSON.parse(JSON.stringify(state)) as PortfolioOptimizerState;
      return true;
    },
    loadSavedState: () => (control.saved ? JSON.parse(JSON.stringify(control.saved)) : null),
    clearSavedState: () => undefined,
    sleep: async () => undefined,
    now: () => {
      clock += control.clockStep;
      return clock;
    },
  };
};

const createSession = (control: FakeControl, lock: RunLockProvider | null = null) =>
  new PortfolioRunSession({ deps: createDeps(control), refreshIntervalMs: 0, lock });

interface FakeLock {
  provider: RunLockProvider;
  /** Блокировку держит другая вкладка. */
  heldElsewhere: boolean;
  acquired: number;
  released: number;
}

const createFakeLock = (heldElsewhere = false): FakeLock => {
  const lock: FakeLock = {
    heldElsewhere,
    acquired: 0,
    released: 0,
    provider: {
      acquire: async () => {
        if (lock.heldElsewhere) {
          return null;
        }
        lock.acquired += 1;
        return {
          release: () => {
            lock.released += 1;
          },
        };
      },
      isHeld: async () => lock.heldElsewhere || lock.acquired > lock.released,
    },
  };
  return lock;
};

const requireSaved = (control: FakeControl): PortfolioOptimizerState => {
  if (!control.saved) {
    throw new Error('Состояние не сохранено');
  }
  return control.saved;
};

// ═══════════════════════════════════════════════════════════════════
// ТЕСТЫ
// ═══════════════════════════════════════════════════════════════════

describe('PortfolioRunSession', () => {
  it('выполняет запуск: снимок активен во время работы и неактивен после', async () => {
    const control = createControl();
    const session = createSession(control);
    expect(session.getSnapshot()).toBeNull();

    const done = session.start(createInput());
    const running = session.getSnapshot();
    expect(running?.active).toBe(true);
    expect(session.isActive).toBe(true);
    expect(running?.meta.config.botName).toBe('LONG 22');
    expect(running?.meta.quoteCurrency).toBe('USDT');
    expect(running?.meta.baseCandidateId).toBe(`${running?.meta.runId}-base`);

    await done;
    const finished = session.getSnapshot();
    expect(session.isActive).toBe(false);
    expect(finished?.active).toBe(false);
    expect(finished?.progress.status).toBe('completed');
    expect(finished?.bestCandidateId).not.toBeNull();
    // База + до 2 поколений × 2 кандидата.
    expect(finished?.candidates.length).toBeGreaterThan(1);
    expect(finished?.candidates.length).toBeLessThanOrEqual(5);
    expect(control.submitted).toBeGreaterThan(0);
    expect(finished?.progress.backtestsSubmitted).toBe(control.submitted);

    const logIds = finished?.logs.map((entry) => entry.id) ?? [];
    expect(logIds.length).toBeGreaterThan(0);
    expect(new Set(logIds).size).toBe(logIds.length);
    expect(finished?.logs.at(-1)?.message).toContain('Оптимизация завершена');
  });

  it('уведомляет подписчиков неизменяемыми снимками и прекращает после отписки', async () => {
    const control = createControl();
    const session = createSession(control);
    const snapshots: unknown[] = [];
    const unsubscribe = session.subscribe(() => snapshots.push(session.getSnapshot()));

    const done = session.start(createInput());
    const countAfterStart = snapshots.length;
    expect(countAfterStart).toBeGreaterThan(0);
    await done;
    expect(snapshots.length).toBeGreaterThan(countAfterStart);
    // Каждое изменение — новый объект снимка.
    expect(new Set(snapshots).size).toBe(snapshots.length);

    unsubscribe();
    const total = snapshots.length;
    session.clear();
    expect(snapshots.length).toBe(total);
  });

  it('не запускает второй оптимизатор, пока идёт первый', async () => {
    const control = createControl(true);
    const session = createSession(control);
    const done = session.start(createInput());
    expect(() => session.start(createInput())).toThrow('Оптимизация уже выполняется');
    expect(() => session.resume(requireSaved(control))).toThrow('Оптимизация уже выполняется');
    session.stop();
    await done;
  });

  it('некорректная конфигурация — ошибка сразу, снимок не меняется', () => {
    const session = createSession(createControl());
    expect(() => session.start(createInput({ basket: [] }))).toThrow('Корзина монет пуста');
    expect(session.getSnapshot()).toBeNull();
    expect(session.isActive).toBe(false);
  });

  it('пауза, продолжение и остановка; остановленный запуск продолжается из сохранённого состояния', async () => {
    const control = createControl(true);
    const session = createSession(control);
    const done = session.start(createInput());

    await vi.waitFor(() => expect(control.pending.length).toBeGreaterThan(0));
    session.pause();
    expect(session.getSnapshot()?.progress.status).toBe('paused');
    expect(control.pausedCalls.at(-1)).toBe(true);
    session.unpause();
    expect(session.getSnapshot()?.progress.status).toBe('running');
    expect(control.pausedCalls.at(-1)).toBe(false);

    session.stop();
    await done;
    expect(session.getSnapshot()?.active).toBe(false);
    expect(session.getSnapshot()?.progress.status).toBe('stopped');

    const saved = requireSaved(control);
    expect(saved.status).toBe('stopped');
    expect(saved.inFlightJobs.length).toBeGreaterThan(0);

    control.hold = false;
    await session.resume(saved);
    expect(control.attached.length).toBe(saved.inFlightJobs.length);
    expect(session.getSnapshot()?.progress.status).toBe('completed');
    expect(session.getSnapshot()?.meta.runId).toBe(saved.runId);
  });

  it('меняет настройки планировщика на ходу и отражает их в снимке', async () => {
    const control = createControl(true);
    const session = createSession(control);
    const done = session.start(createInput());

    session.updateSchedulerConfig({ maxInFlight: 5, submitDelaySeconds: 45 });
    expect(control.updates).toContainEqual(expect.objectContaining({ maxInFlight: 5, submitDelaySeconds: 45 }));
    expect(session.getSnapshot()?.meta.config.scheduler).toMatchObject({ maxInFlight: 5, submitDelaySeconds: 45 });

    session.stop();
    await done;
    const updatesBefore = control.updates.length;
    // Без активного запуска изменение игнорируется.
    session.updateSchedulerConfig({ maxInFlight: 2 });
    expect(control.updates.length).toBe(updatesBefore);
  });

  it('показывает сохранённый запуск без выполнения и сбрасывает его', async () => {
    const control = createControl();
    const runner = createSession(control);
    await runner.start(createInput());
    const saved = requireSaved(control);

    const session = createSession(createControl());
    session.show(saved);
    const shown = session.getSnapshot();
    expect(shown?.active).toBe(false);
    expect(shown?.meta.runId).toBe(saved.runId);
    expect(shown?.candidates).toHaveLength(saved.candidates.length);
    expect(shown?.progress).toMatchObject({
      status: 'completed',
      generation: saved.generation,
      backtestsSubmitted: saved.backtestsSubmitted,
      inFlight: 0,
      queued: 0,
      estimatedEndAt: null,
    });
    expect(shown?.logs).toHaveLength(saved.logs.length);

    expect(session.clear()).toBe(true);
    expect(session.getSnapshot()).toBeNull();
  });

  it('пока запуск идёт, show и clear не трогают его снимок', async () => {
    const control = createControl(true);
    const session = createSession(control);
    const done = session.start(createInput());
    const runId = session.getSnapshot()?.meta.runId;

    session.show({ ...requireSaved(control), runId: 'foreign' });
    expect(session.getSnapshot()?.meta.runId).toBe(runId);
    expect(session.clear()).toBe(false);
    expect(session.getSnapshot()).not.toBeNull();

    session.stop();
    await done;
    expect(session.clear()).toBe(true);
  });

  it('журнал сохранённого запуска ограничен 300 записями с уникальными ключами', () => {
    const control = createControl();
    const session = createSession(control);
    const logs: PortfolioLogEntry[] = Array.from({ length: MAX_PORTFOLIO_LOG_ENTRIES + 50 }, (_, index) => ({
      timestamp: index,
      level: 'info',
      message: `запись ${index}`,
    }));
    const base: PortfolioOptimizerState = {
      version: 1,
      runId: 'posaved',
      config: createConfig(),
      baseStrategy: createStrategy(),
      quoteCurrency: 'USDT',
      baseCandidateId: 'posaved-base',
      bestCandidateId: null,
      candidates: [],
      backtestCache: {},
      inFlightJobs: [],
      paramStats: {},
      generation: 0,
      backtestsSubmitted: 0,
      status: 'stopped',
      startedAt: 1,
      updatedAt: 2,
      logs,
    };
    session.show(base);
    const shown = session.getSnapshot()?.logs ?? [];
    expect(shown).toHaveLength(MAX_PORTFOLIO_LOG_ENTRIES);
    expect(shown[0].message).toBe('запись 50');
    expect(new Set(shown.map((entry) => entry.id)).size).toBe(MAX_PORTFOLIO_LOG_ENTRIES);
  });

  it('refreshProgress уведомляет только при изменении прогресса', async () => {
    const control = createControl(true);
    const session = createSession(control);
    const done = session.start(createInput());
    await vi.waitFor(() => expect(control.pending.length).toBeGreaterThan(0));

    // Время остановлено: прогресс (включая ETA) больше не меняется.
    control.clockStep = 0;
    const listener = vi.fn();
    const unsubscribe = session.subscribe(listener);
    session.refreshProgress();
    const callsAfterFirst = listener.mock.calls.length;
    expect(callsAfterFirst).toBeLessThanOrEqual(1);
    session.refreshProgress();
    session.refreshProgress();
    expect(listener).toHaveBeenCalledTimes(callsAfterFirst);

    // Время пошло — ETA изменился, подписчик уведомлён.
    control.clockStep = 60_000;
    session.refreshProgress();
    expect(listener).toHaveBeenCalledTimes(callsAfterFirst + 1);
    unsubscribe();

    session.stop();
    await done;
    session.refreshProgress();
    expect(session.getSnapshot()?.progress.status).toBe('stopped');
  });

  it('по умолчанию обновляет прогресс таймером, пока запуск идёт', async () => {
    vi.useFakeTimers();
    try {
      const control = createControl(true);
      const session = new PortfolioRunSession({ deps: createDeps(control), refreshIntervalMs: 1_000 });
      const refresh = vi.spyOn(session, 'refreshProgress');
      const done = session.start(createInput());
      await vi.advanceTimersByTimeAsync(3_000);
      expect(refresh).toHaveBeenCalledTimes(3);
      session.stop();
      await vi.runAllTimersAsync();
      await done;
      refresh.mockClear();
      await vi.advanceTimersByTimeAsync(3_000);
      expect(refresh).not.toHaveBeenCalled();
    } finally {
      vi.useRealTimers();
    }
  });

  it('держит блокировку между вкладками всё время работы, включая паузу, и отпускает её в конце', async () => {
    const control = createControl(true);
    const lock = createFakeLock();
    const session = createSession(control, lock.provider);
    const done = session.start(createInput());
    expect(session.isActive).toBe(true);
    await vi.waitFor(() => expect(control.pending.length).toBeGreaterThan(0));
    expect(lock.acquired).toBe(1);

    session.pause();
    expect(lock.released).toBe(0);
    session.stop();
    await done;
    expect(lock.released).toBe(1);
    expect(session.isActive).toBe(false);
  });

  it('не запускает оптимизатор, если блокировку держит другая вкладка, и показывает прежний снимок', async () => {
    const control = createControl();
    const runner = createSession(control);
    await runner.start(createInput());
    const saved = requireSaved(control);
    const savedBefore = JSON.stringify(saved);

    const lock = createFakeLock(true);
    const session = createSession(control, lock.provider);
    session.show(saved);
    await session.resume(saved);

    expect(session.isActive).toBe(false);
    expect(control.events).toHaveLength(1);
    expect(JSON.stringify(control.saved)).toBe(savedBefore);
    const snapshot = session.getSnapshot();
    expect(snapshot?.active).toBe(false);
    expect(snapshot?.meta.runId).toBe(saved.runId);
    expect(snapshot?.logs.at(-1)?.message).toBe(OTHER_TAB_RUN_MESSAGE);
    await expect(session.detectOtherTabRun(saved, Date.now())).resolves.toBe('locked');
  });

  it('остановка до получения блокировки отменяет запуск', async () => {
    const control = createControl();
    const lock = createFakeLock();
    const session = createSession(control, lock.provider);
    const done = session.start(createInput());
    session.stop();
    await done;
    expect(control.events).toHaveLength(0);
    expect(control.submitted).toBe(0);
    expect(session.isActive).toBe(false);
    expect(lock.released).toBe(lock.acquired);
  });

  it('без Web Locks пишет «пульс» во время работы и паузы и убирает его после', async () => {
    vi.useFakeTimers();
    try {
      window.localStorage.removeItem(PORTFOLIO_RUN_HEARTBEAT_KEY);
      const control = createControl(true);
      const session = new PortfolioRunSession({
        deps: createDeps(control),
        refreshIntervalMs: 0,
        lock: null,
        heartbeatIntervalMs: 1_000,
      });
      const done = session.start(createInput());
      const runId = session.getSnapshot()?.meta.runId;
      expect(readRunHeartbeat()).toMatchObject({ runId, tabId: session.tabId });
      session.pause();
      const first = readRunHeartbeat()?.at ?? 0;
      await vi.advanceTimersByTimeAsync(2_000);
      expect(readRunHeartbeat()?.at).toBeGreaterThan(first);

      // Другая вкладка (другой сеанс) видит запуск на паузе как выполняющийся.
      const other = createSession(createControl(), null);
      const saved = requireSaved(control);
      await expect(other.detectOtherTabRun(saved, Date.now() + 10 * 60_000)).resolves.toBe('none');
      await expect(other.detectOtherTabRun(saved, Date.now())).resolves.toBe('probable');
      // Своя вкладка свой «пульс» не учитывает.
      await expect(session.detectOtherTabRun(saved, Date.now())).resolves.toBe('none');

      session.stop();
      await vi.runAllTimersAsync();
      await done;
      expect(readRunHeartbeat()).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it('ответ на POST остановленного экземпляра передаётся продолженному запуску', async () => {
    const control = createControl(true);
    const session = createSession(control);
    const done = session.start(createInput());
    await vi.waitFor(() => expect(control.pending.length).toBeGreaterThan(0));
    session.stop();
    await done;
    const stopped = requireSaved(control);
    const stoppedEvents = control.events[0];

    const resumed = session.resume(stopped);
    expect(session.isActive).toBe(true);
    stoppedEvents.onSubmitted(`${stopped.candidates[0].hash}|LTC`, 9_999);
    const progress = session.getSnapshot()?.progress;
    expect(progress?.backtestsSubmitted).toBeGreaterThan(stopped.backtestsSubmitted);
    expect(session.getSnapshot()?.logs.some((entry) => entry.message.includes('#9999'))).toBe(true);
    expect(requireSaved(control).inFlightJobs.some((job) => job.backtestId === 9_999)).toBe(true);

    session.stop();
    await resumed;
  });
});
