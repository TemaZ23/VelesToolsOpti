/**
 * Планировщик бэктестов Veles для портфельного оптимизатора.
 *
 * Отвечает только за «транспортную дисциплину» поверх API:
 * - глобальный лимит одновременно выполняющихся бэктестов (потоки тарифа Veles);
 * - минимальная пауза между созданием бэктестов (очередь FIFO);
 * - реакция на 429 (глобальная пауза + временное увеличение интервала);
 * - повторы при сетевых сбоях и ошибках 5xx с экспоненциальной задержкой;
 * - опрос статуса до завершения, таймаут и остановка по AbortSignal.
 *
 * Сам планировщик ничего не знает о кандидатах и метриках — это задача оркестратора.
 */

import type { BotStrategy } from '../api/backtestRunner';
import { postBacktest } from '../api/backtestRunner';
import { fetchBacktestConfig, fetchBacktestStatistics } from '../api/backtests';
import type { SchedulerConfig } from '../types/portfolioOptimizer';

// ═══════════════════════════════════════════════════════════════════
// ПУБЛИЧНЫЕ ТИПЫ
// ═══════════════════════════════════════════════════════════════════

export type BacktestStatusClass = 'running' | 'done' | 'failed' | 'unknown';

export interface SchedulerJobRequest {
  /** Уникальный ключ задачи, обычно `${candidateHash}|${symbol}`. */
  key: string;
  /** Тикер монеты (для сообщений в журнале). */
  symbol: string;
  /** Готовый payload бэктеста (результат buildBacktestPayload). */
  payload: BotStrategy;
}

export interface SchedulerEvents {
  /** Бэктест создан в Veles (вызывается и тогда, когда запуск успели остановить во время POST). */
  onSubmitted: (key: string, backtestId: number) => void;
  onLog: (level: 'info' | 'warning' | 'error', message: string) => void;
}

export interface SchedulerDeps {
  postBacktest: (payload: BotStrategy) => Promise<{ id: number }>;
  /** Статус бэктеста: fetchBacktestConfig(id).status. */
  fetchStatus: (id: number) => Promise<string | null>;
  /** Есть ли у бэктеста статистика (fetchBacktestStatistics успешен и totalDeals — число). */
  fetchStatisticsExists: (id: number) => Promise<boolean>;
  sleep: (ms: number, signal?: AbortSignal) => Promise<void>;
  now: () => number;
}

export type SubmitErrorKind = 'rateLimit' | 'recoverable' | 'fatal';

// ═══════════════════════════════════════════════════════════════════
// КОНСТАНТЫ
// ═══════════════════════════════════════════════════════════════════

/** Сообщение ошибки остановки (как AbortError). */
export const SCHEDULER_ABORT_MESSAGE = 'Остановлено';

/** Максимум попыток создать бэктест при восстановимых ошибках (включая первую). */
export const SUBMIT_MAX_ATTEMPTS = 5;
const RETRY_BASE_DELAY_MS = 10_000;
const RETRY_MAX_DELAY_MS = 120_000;

/** Пауза после 429, если сервер не сообщил Retry-After. */
export const RATE_LIMIT_DEFAULT_COOLDOWN_SECONDS = 60;
const RATE_LIMIT_DELAY_FACTOR = 1.5;
/** Верхняя граница интервала между бэктестами после штрафов за 429. */
export const RATE_LIMIT_DELAY_CAP_SECONDS = 120;
const DELAY_DECAY_FACTOR = 0.9;
const RETRY_AFTER_MIN_SECONDS = 1;
const RETRY_AFTER_MAX_SECONDS = 600;

/** Ошибки опроса пишутся в журнал на 1-й и каждой N-й подряд неудаче (чтобы не забивать журнал). */
const POLL_ERROR_LOG_EVERY = 10;

const FALLBACK_SCHEDULER_CONFIG: SchedulerConfig = {
  maxInFlight: 3,
  submitDelaySeconds: 31,
  pollIntervalSeconds: 10,
  backtestTimeoutMinutes: 45,
};

const MIN_POLL_INTERVAL_SECONDS = 1;
const MIN_TIMEOUT_MINUTES = 1;

const DONE_STATUSES = new Set(['FINISHED', 'COMPLETED', 'DONE', 'SUCCESS', 'SUCCEEDED']);
const FAILED_STATUSES = new Set(['FAILED', 'ERROR', 'CANCELED', 'CANCELLED', 'TERMINATED', 'REJECTED']);
const RUNNING_STATUSES = new Set([
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
]);

const RATE_LIMIT_PATTERNS = [/\b429\b/, /too many/, /слишком много/, /временно заблокировано/];

// ═══════════════════════════════════════════════════════════════════
// ОШИБКИ
// ═══════════════════════════════════════════════════════════════════

/** Ошибка остановки планировщика (name = 'AbortError', message = 'Остановлено'). */
export class SchedulerAbortError extends Error {
  constructor() {
    super(SCHEDULER_ABORT_MESSAGE);
    this.name = 'AbortError';
  }
}

/** Ошибка создания бэктеста с HTTP-статусом и Retry-After (если их удалось определить). */
export class BacktestSubmitError extends Error {
  readonly status: number | null;
  readonly retryAfterSeconds: number | null;

  constructor(message: string, status: number | null, retryAfterSeconds: number | null) {
    super(message);
    this.name = 'BacktestSubmitError';
    this.status = status;
    this.retryAfterSeconds = retryAfterSeconds;
  }
}

const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === 'object' && value !== null;
};

/** Ошибка остановки: SchedulerAbortError или любая ошибка с name = 'AbortError' (в т.ч. DOMException). */
export const isAbortError = (error: unknown): boolean => {
  return isRecord(error) && error.name === 'AbortError';
};

const toError = (error: unknown): Error => {
  return error instanceof Error ? error : new Error(String(error));
};

const readFiniteNumber = (value: unknown, key: string): number | null => {
  if (!isRecord(value)) {
    return null;
  }
  const candidate = value[key];
  return typeof candidate === 'number' && Number.isFinite(candidate) ? candidate : null;
};

const readMessage = (error: unknown): string => {
  if (error instanceof Error) {
    return error.message;
  }
  return typeof error === 'string' ? error : String(error);
};

// ═══════════════════════════════════════════════════════════════════
// ЧИСТЫЕ ПОМОЩНИКИ
// ═══════════════════════════════════════════════════════════════════

/** Классификация статуса бэктеста Veles (регистр, пробелы и дефисы не важны). */
export const classifyBacktestStatus = (status: string | null | undefined): BacktestStatusClass => {
  if (typeof status !== 'string') {
    return 'unknown';
  }
  const normalized = status
    .trim()
    .toUpperCase()
    .replace(/[\s-]+/g, '_');
  if (!normalized) {
    return 'unknown';
  }
  if (DONE_STATUSES.has(normalized)) {
    return 'done';
  }
  if (FAILED_STATUSES.has(normalized)) {
    return 'failed';
  }
  if (RUNNING_STATUSES.has(normalized)) {
    return 'running';
  }
  return 'unknown';
};

/**
 * HTTP-статус ошибки: числовое свойство `status`, иначе метка «HTTP 429 …»,
 * которую добавляет resolveProxyErrorMessage.
 */
export const extractHttpStatus = (error: unknown): number | null => {
  const direct = readFiniteNumber(error, 'status');
  if (direct !== null && direct >= 100 && direct <= 599) {
    return Math.trunc(direct);
  }
  const match = /\bHTTP\s+(\d{3})\b/i.exec(readMessage(error));
  return match ? Number(match[1]) : null;
};

const clampRetryAfter = (seconds: number): number => {
  return Math.min(RETRY_AFTER_MAX_SECONDS, Math.max(RETRY_AFTER_MIN_SECONDS, seconds));
};

/** Retry-After из текста ошибки: «Retry-After: 30», «retry after 30», «через 30 сек», «через 2 мин». */
export const parseRetryAfterFromMessage = (message: string): number | null => {
  const header = /retry[-\s]?after\D{0,3}(\d+(?:\.\d+)?)/i.exec(message);
  if (header) {
    return clampRetryAfter(Number(header[1]));
  }
  const russian = /через\s+(\d+(?:[.,]\d+)?)\s*(сек[а-яё]*|с|мин[а-яё]*)(?![а-яё])/i.exec(message);
  if (russian) {
    const amount = Number(russian[1].replace(',', '.'));
    const unit = russian[2].toLowerCase();
    const seconds = unit.startsWith('мин') ? amount * 60 : amount;
    return clampRetryAfter(seconds);
  }
  return null;
};

/** Retry-After в секундах: числовое свойство `retryAfterSeconds`, иначе разбор текста ошибки. */
export const extractRetryAfterSeconds = (error: unknown): number | null => {
  const direct = readFiniteNumber(error, 'retryAfterSeconds');
  if (direct !== null && direct > 0) {
    return clampRetryAfter(direct);
  }
  return parseRetryAfterFromMessage(readMessage(error));
};

/** Обёртка ошибки postBacktest: сохраняет текст, добавляет статус и Retry-After. */
export const toBacktestSubmitError = (error: unknown): BacktestSubmitError => {
  if (error instanceof BacktestSubmitError) {
    return error;
  }
  const submitError = new BacktestSubmitError(
    readMessage(error),
    extractHttpStatus(error),
    extractRetryAfterSeconds(error),
  );
  if (error instanceof Error && error.stack) {
    submitError.stack = error.stack;
  }
  return submitError;
};

/**
 * Классификация ошибки создания бэктеста:
 * - 'rateLimit' — 429 / «Too Many» / «слишком много» / «временно заблокировано»;
 * - 'fatal' — прочие 4xx (кроме 408): повтор бессмыслен (ошибка payload, авторизации);
 * - 'recoverable' — 5xx, 408, сетевые сбои, таймауты, потеря вкладки и ошибки без HTTP-статуса
 *   (все HTTP-ошибки прокси содержат метку «HTTP NNN», так что без статуса — это сбой транспорта).
 */
export const classifySubmitError = (error: unknown): SubmitErrorKind => {
  const status = extractHttpStatus(error);
  const message = readMessage(error).toLowerCase();

  if (status === 429 || RATE_LIMIT_PATTERNS.some((pattern) => pattern.test(message))) {
    return 'rateLimit';
  }
  if (status !== null && status >= 400 && status < 500 && status !== 408) {
    return 'fatal';
  }
  // 5xx, 408 и ошибки без HTTP-статуса: сеть, «Таймаут ответа от вкладки», «Не найдена вкладка», потеря связи.
  return 'recoverable';
};

/** Задержка перед повтором после n-й неудачи (n ≥ 1): 10 с × 2^(n−1), не больше 120 с. */
export const computeRetryBackoffMs = (failedAttempts: number): number => {
  const exponent = Math.max(0, Math.floor(failedAttempts) - 1);
  return Math.min(RETRY_MAX_DELAY_MS, RETRY_BASE_DELAY_MS * 2 ** exponent);
};

/** Интервал после 429: ×1.5, не больше max(120 с, настроенный). */
export const escalateSubmitDelay = (effectiveSeconds: number, configuredSeconds: number): number => {
  const cap = Math.max(RATE_LIMIT_DELAY_CAP_SECONDS, configuredSeconds);
  const escalated = Math.max(effectiveSeconds, RETRY_AFTER_MIN_SECONDS) * RATE_LIMIT_DELAY_FACTOR;
  return Math.min(cap, Math.max(configuredSeconds, escalated));
};

/** Возврат интервала к настроенному после успешного POST: ×0.9, не меньше настроенного. */
export const decaySubmitDelay = (effectiveSeconds: number, configuredSeconds: number): number => {
  return Math.max(configuredSeconds, effectiveSeconds * DELAY_DECAY_FACTOR);
};

/** Пересчёт текущего интервала при смене настройки: штраф за 429 сохраняется пропорционально. */
export const rescaleSubmitDelay = (
  effectiveSeconds: number,
  previousConfiguredSeconds: number,
  nextConfiguredSeconds: number,
): number => {
  if (previousConfiguredSeconds <= 0 || effectiveSeconds <= previousConfiguredSeconds) {
    return nextConfiguredSeconds;
  }
  const penalty = effectiveSeconds / previousConfiguredSeconds;
  const cap = Math.max(RATE_LIMIT_DELAY_CAP_SECONDS, nextConfiguredSeconds);
  return Math.min(cap, Math.max(nextConfiguredSeconds, nextConfiguredSeconds * penalty));
};

export interface SubmitWaitInput {
  paused: boolean;
  inFlight: number;
  maxInFlight: number;
  now: number;
  /** Время завершения последнего POST (мс epoch) или null, если POST ещё не было. */
  lastPostAt: number | null;
  effectiveDelayMs: number;
  cooldownUntil: number;
  /** Не раньше этого времени можно повторить задачу (backoff), мс epoch. */
  notBefore: number;
}

/**
 * Сколько ждать до следующего POST: null — ждать события (пауза или все потоки заняты),
 * 0 — можно отправлять сейчас, иначе — миллисекунды.
 */
export const computeSubmitWaitMs = (input: SubmitWaitInput): number | null => {
  if (input.paused || input.inFlight >= input.maxInFlight) {
    return null;
  }
  const spacingReadyAt =
    input.lastPostAt === null ? Number.NEGATIVE_INFINITY : input.lastPostAt + input.effectiveDelayMs;
  const readyAt = Math.max(spacingReadyAt, input.cooldownUntil, input.notBefore);
  return readyAt > input.now ? readyAt - input.now : 0;
};

const pickNumber = (value: number | undefined, fallback: number): number => {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
};

/** Слияние и нормализация настроек: нечисловые значения берутся из `current`. */
export const mergeSchedulerConfig = (current: SchedulerConfig, patch: Partial<SchedulerConfig>): SchedulerConfig => {
  return {
    maxInFlight: Math.max(1, Math.floor(pickNumber(patch.maxInFlight, current.maxInFlight))),
    submitDelaySeconds: Math.max(0, pickNumber(patch.submitDelaySeconds, current.submitDelaySeconds)),
    pollIntervalSeconds: Math.max(
      MIN_POLL_INTERVAL_SECONDS,
      pickNumber(patch.pollIntervalSeconds, current.pollIntervalSeconds),
    ),
    backtestTimeoutMinutes: Math.max(
      MIN_TIMEOUT_MINUTES,
      pickNumber(patch.backtestTimeoutMinutes, current.backtestTimeoutMinutes),
    ),
  };
};

const formatSeconds = (seconds: number): string => {
  const rounded = Math.round(seconds * 10) / 10;
  return Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1);
};

const labelFromKey = (key: string): string => {
  const separator = key.lastIndexOf('|');
  return separator >= 0 && separator < key.length - 1 ? key.slice(separator + 1) : key;
};

// ═══════════════════════════════════════════════════════════════════
// ЗАВИСИМОСТИ ПО УМОЛЧАНИЮ
// ═══════════════════════════════════════════════════════════════════

/** setTimeout-пауза, прерываемая AbortSignal (отклоняется SchedulerAbortError). */
export const abortableSleep = (ms: number, signal?: AbortSignal): Promise<void> => {
  return new Promise<void>((resolve, reject) => {
    if (signal?.aborted) {
      reject(new SchedulerAbortError());
      return;
    }
    const onAbort = (): void => {
      clearTimeout(timer);
      reject(new SchedulerAbortError());
    };
    const timer = setTimeout(
      () => {
        signal?.removeEventListener('abort', onAbort);
        resolve();
      },
      Math.max(0, ms),
    );
    signal?.addEventListener('abort', onAbort, { once: true });
  });
};

const submitBacktestViaApi = async (payload: BotStrategy): Promise<{ id: number }> => {
  try {
    const response = await postBacktest(payload);
    return { id: response.id };
  } catch (error) {
    throw toBacktestSubmitError(error);
  }
};

const fetchStatusViaApi = async (id: number): Promise<string | null> => {
  const config = await fetchBacktestConfig(id);
  return typeof config.status === 'string' ? config.status : null;
};

const fetchStatisticsExistsViaApi = async (id: number): Promise<boolean> => {
  try {
    const statistics = await fetchBacktestStatistics(id);
    return typeof statistics.totalDeals === 'number' && Number.isFinite(statistics.totalDeals);
  } catch {
    return false;
  }
};

/** Зависимости на реальном API Veles и таймерах. */
export const createDefaultSchedulerDeps = (): SchedulerDeps => ({
  postBacktest: submitBacktestViaApi,
  fetchStatus: fetchStatusViaApi,
  fetchStatisticsExists: fetchStatisticsExistsViaApi,
  sleep: abortableSleep,
  now: () => Date.now(),
});

/** Подмена части зависимостей; явные undefined не затирают значения по умолчанию. */
export const resolveSchedulerDeps = (overrides: Partial<SchedulerDeps> = {}): SchedulerDeps => {
  const defaults = createDefaultSchedulerDeps();
  return {
    postBacktest: overrides.postBacktest ?? defaults.postBacktest,
    fetchStatus: overrides.fetchStatus ?? defaults.fetchStatus,
    fetchStatisticsExists: overrides.fetchStatisticsExists ?? defaults.fetchStatisticsExists,
    sleep: overrides.sleep ?? defaults.sleep,
    now: overrides.now ?? defaults.now,
  };
};

// ═══════════════════════════════════════════════════════════════════
// ПЛАНИРОВЩИК
// ═══════════════════════════════════════════════════════════════════

interface Deferred<T> {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (error: Error) => void;
}

const createDeferred = <T>(): Deferred<T> => {
  let resolve: (value: T) => void = () => undefined;
  let reject: (error: Error) => void = () => undefined;
  const promise = new Promise<T>((onResolve, onReject) => {
    resolve = onResolve;
    reject = onReject;
  });
  return { promise, resolve, reject };
};

type EntryPhase = 'queued' | 'submitting' | 'polling';

interface TrackedEntry {
  key: string;
  label: string;
  signal: AbortSignal;
  phase: EntryPhase;
  settled: boolean;
  deferred: Deferred<number>;
  detach: () => void;
}

interface QueuedEntry extends TrackedEntry {
  payload: BotStrategy;
  failedAttempts: number;
  notBefore: number;
}

type PollOutcome = { kind: 'done' } | { kind: 'running' } | { kind: 'failed'; status: string };

const throwIfAborted = (signal: AbortSignal): void => {
  if (signal.aborted) {
    throw new SchedulerAbortError();
  }
};

export class BacktestScheduler {
  private config: SchedulerConfig;
  private readonly events: SchedulerEvents;
  private readonly deps: SchedulerDeps;

  private readonly queue: QueuedEntry[] = [];
  private submittingEntry: QueuedEntry | null = null;
  private inFlightCount = 0;
  private paused = false;
  private pumping = false;
  private wakeGate: Deferred<void> = createDeferred<void>();

  private lastPostAt: number | null = null;
  private cooldownUntil = 0;
  private effectiveDelaySeconds: number;
  private readonly warnedStatuses = new Set<string>();

  constructor(config: SchedulerConfig, events: SchedulerEvents, deps?: Partial<SchedulerDeps>) {
    this.config = mergeSchedulerConfig(FALLBACK_SCHEDULER_CONFIG, config);
    this.events = events;
    this.deps = resolveSchedulerDeps(deps);
    this.effectiveDelaySeconds = this.config.submitDelaySeconds;
  }

  /** Бэктестов создано в Veles и ещё не завершено (включая подключённые через attach). */
  get inFlight(): number {
    return this.inFlightCount;
  }

  /** Задач ждут создания (включая ту, чей POST выполняется сейчас, и ожидающие повтора). */
  get queued(): number {
    return this.queue.length + (this.submittingEntry ? 1 : 0);
  }

  get isPaused(): boolean {
    return this.paused;
  }

  /** Текущий интервал между созданием бэктестов, с учётом штрафа за 429. */
  get effectiveSubmitDelaySeconds(): number {
    return this.effectiveDelaySeconds;
  }

  /** Создать бэктест (в порядке очереди) и дождаться завершения. Возвращает id бэктеста. */
  run(job: SchedulerJobRequest, signal: AbortSignal): Promise<number> {
    if (signal.aborted) {
      return Promise.reject(new SchedulerAbortError());
    }
    const entry: QueuedEntry = {
      ...this.createTrackedEntry(job.key, job.symbol, signal, 'queued'),
      payload: job.payload,
      failedAttempts: 0,
      notBefore: 0,
    };
    this.listenAbort(entry);
    this.queue.push(entry);
    this.wake();
    void this.pump();
    return entry.deferred.promise;
  }

  /** Продолжение после перезагрузки: только опрос уже созданного бэктеста (занимает поток). */
  attach(key: string, backtestId: number, signal: AbortSignal): Promise<number> {
    if (signal.aborted) {
      return Promise.reject(new SchedulerAbortError());
    }
    if (!Number.isInteger(backtestId) || backtestId <= 0) {
      return Promise.reject(new Error(`Некорректный идентификатор бэктеста: ${backtestId}`));
    }
    const entry = this.createTrackedEntry(key, labelFromKey(key), signal, 'polling');
    this.listenAbort(entry);
    void this.track(entry, backtestId, true);
    return entry.deferred.promise;
  }

  updateConfig(config: Partial<SchedulerConfig>): void {
    const next = mergeSchedulerConfig(this.config, config);
    this.effectiveDelaySeconds = rescaleSubmitDelay(
      this.effectiveDelaySeconds,
      this.config.submitDelaySeconds,
      next.submitDelaySeconds,
    );
    this.config = next;
    this.wake();
  }

  /** Пауза: новые бэктесты не создаются, опрос уже запущенных продолжается. */
  setPaused(paused: boolean): void {
    if (this.paused === paused) {
      return;
    }
    this.paused = paused;
    this.wake();
  }

  // ─── Внутреннее: задачи ──────────────────────────────────────────

  private createTrackedEntry(key: string, label: string, signal: AbortSignal, phase: EntryPhase): TrackedEntry {
    return {
      key,
      label,
      signal,
      phase,
      settled: false,
      deferred: createDeferred<number>(),
      detach: () => undefined,
    };
  }

  private listenAbort(entry: TrackedEntry): void {
    const onAbort = (): void => this.handleAbort(entry);
    entry.signal.addEventListener('abort', onAbort, { once: true });
    entry.detach = () => entry.signal.removeEventListener('abort', onAbort);
  }

  private handleAbort(entry: TrackedEntry): void {
    if (entry.settled) {
      return;
    }
    if (entry.phase === 'queued') {
      // Безопасное расширение типа элементов (только чтение) — без приведения типов.
      const queuedView: readonly TrackedEntry[] = this.queue;
      const index = queuedView.indexOf(entry);
      if (index >= 0) {
        this.queue.splice(index, 1);
      }
    }
    this.fail(entry, new SchedulerAbortError());
    this.wake();
  }

  private succeed(entry: TrackedEntry, backtestId: number): void {
    if (entry.settled) {
      return;
    }
    entry.settled = true;
    entry.detach();
    entry.deferred.resolve(backtestId);
  }

  private fail(entry: TrackedEntry, error: Error): void {
    if (entry.settled) {
      return;
    }
    entry.settled = true;
    entry.detach();
    entry.deferred.reject(error);
  }

  // ─── Внутреннее: события ─────────────────────────────────────────

  private log(level: 'info' | 'warning' | 'error', message: string): void {
    try {
      this.events.onLog(level, message);
    } catch (error) {
      console.warn('[BacktestScheduler] onLog завершился с ошибкой:', error);
    }
  }

  private notifySubmitted(key: string, backtestId: number): void {
    try {
      this.events.onSubmitted(key, backtestId);
    } catch (error) {
      console.warn('[BacktestScheduler] onSubmitted завершился с ошибкой:', error);
    }
  }

  // ─── Внутреннее: очередь отправки ────────────────────────────────

  private wake(): void {
    const gate = this.wakeGate;
    this.wakeGate = createDeferred<void>();
    gate.resolve();
  }

  /** Ждать `ms` (или события при null); любое изменение состояния (wake) прерывает ожидание. */
  private async waitForChange(ms: number | null): Promise<void> {
    const gate = this.wakeGate.promise;
    if (ms === null) {
      await gate;
      return;
    }
    const controller = new AbortController();
    const timer = this.deps.sleep(ms, controller.signal).catch(() => undefined);
    await Promise.race([timer, gate]);
    controller.abort();
  }

  private async pump(): Promise<void> {
    if (this.pumping) {
      return;
    }
    this.pumping = true;
    try {
      for (;;) {
        const head = this.queue.at(0);
        if (!head) {
          break;
        }
        const waitMs = computeSubmitWaitMs({
          paused: this.paused,
          inFlight: this.inFlightCount,
          maxInFlight: this.config.maxInFlight,
          now: this.deps.now(),
          lastPostAt: this.lastPostAt,
          effectiveDelayMs: this.effectiveDelaySeconds * 1000,
          cooldownUntil: this.cooldownUntil,
          notBefore: head.notBefore,
        });
        if (waitMs !== 0) {
          await this.waitForChange(waitMs);
          continue;
        }
        this.queue.shift();
        await this.submit(head);
      }
    } catch (error) {
      this.log('error', `Сбой очереди бэктестов: ${readMessage(error)}`);
    } finally {
      this.pumping = false;
    }
  }

  private async submit(entry: QueuedEntry): Promise<void> {
    if (entry.settled) {
      return;
    }
    entry.phase = 'submitting';
    this.submittingEntry = entry;

    let backtestId: number;
    try {
      const response = await this.deps.postBacktest(entry.payload);
      const id = readFiniteNumber(response, 'id');
      if (id === null || !Number.isInteger(id) || id <= 0) {
        throw new Error('Сервер не вернул идентификатор созданного бэктеста.');
      }
      backtestId = id;
    } catch (error) {
      this.submittingEntry = null;
      // Интервал считается от ответа сервера: так запросы гарантированно разнесены и на стороне Veles.
      this.lastPostAt = this.deps.now();
      this.handleSubmitError(entry, toError(error));
      return;
    }

    this.submittingEntry = null;
    this.lastPostAt = this.deps.now();
    this.relaxDelayAfterSuccess();
    // Бэктест уже создан в Veles — сообщаем всегда, чтобы оркестратор мог сохранить id для продолжения.
    this.notifySubmitted(entry.key, backtestId);

    if (entry.settled || entry.signal.aborted) {
      this.fail(entry, new SchedulerAbortError());
      return;
    }
    void this.track(entry, backtestId, false);
  }

  private relaxDelayAfterSuccess(): void {
    const configured = this.config.submitDelaySeconds;
    if (this.effectiveDelaySeconds <= configured) {
      this.effectiveDelaySeconds = configured;
      return;
    }
    this.effectiveDelaySeconds = decaySubmitDelay(this.effectiveDelaySeconds, configured);
    if (this.effectiveDelaySeconds <= configured) {
      this.log('info', `Интервал между бэктестами вернулся к ${formatSeconds(configured)} с.`);
    }
  }

  private requeueAtHead(entry: QueuedEntry): void {
    entry.phase = 'queued';
    this.queue.unshift(entry);
  }

  private handleSubmitError(entry: QueuedEntry, error: Error): void {
    const now = this.deps.now();
    const kind = classifySubmitError(error);

    if (kind === 'rateLimit') {
      // Ограничение частоты глобальное: учитываем его даже если эту задачу уже остановили.
      const cooldownSeconds = extractRetryAfterSeconds(error) ?? RATE_LIMIT_DEFAULT_COOLDOWN_SECONDS;
      this.cooldownUntil = Math.max(this.cooldownUntil, now + cooldownSeconds * 1000);
      this.effectiveDelaySeconds = escalateSubmitDelay(this.effectiveDelaySeconds, this.config.submitDelaySeconds);
      if (entry.settled || entry.signal.aborted) {
        this.fail(entry, new SchedulerAbortError());
        return;
      }
      this.requeueAtHead(entry);
      this.log(
        'warning',
        `Veles ограничил частоту создания бэктестов (${entry.label}): пауза ${formatSeconds(cooldownSeconds)} с, ` +
          `интервал увеличен до ${formatSeconds(this.effectiveDelaySeconds)} с.`,
      );
      return;
    }

    if (entry.settled || entry.signal.aborted) {
      this.fail(entry, new SchedulerAbortError());
      return;
    }

    if (kind === 'recoverable') {
      entry.failedAttempts += 1;
      if (entry.failedAttempts >= SUBMIT_MAX_ATTEMPTS) {
        this.log(
          'error',
          `Не удалось создать бэктест ${entry.label} за ${SUBMIT_MAX_ATTEMPTS} попыток: ${error.message}`,
        );
        this.fail(
          entry,
          new Error(`Не удалось создать бэктест ${entry.label} за ${SUBMIT_MAX_ATTEMPTS} попыток: ${error.message}`),
        );
        return;
      }
      const backoffMs = computeRetryBackoffMs(entry.failedAttempts);
      entry.notBefore = now + backoffMs;
      this.requeueAtHead(entry);
      this.log(
        'warning',
        `Ошибка создания бэктеста ${entry.label} (попытка ${entry.failedAttempts}/${SUBMIT_MAX_ATTEMPTS}): ` +
          `${error.message}. Повтор через ${formatSeconds(backoffMs / 1000)} с.`,
      );
      return;
    }

    this.log('error', `Бэктест ${entry.label} не создан: ${error.message}`);
    this.fail(entry, error);
  }

  // ─── Внутреннее: опрос ───────────────────────────────────────────

  private async track(entry: TrackedEntry, backtestId: number, checkImmediately: boolean): Promise<void> {
    entry.phase = 'polling';
    this.inFlightCount += 1;
    try {
      const finishedId = await this.pollUntilFinished(entry, backtestId, checkImmediately);
      this.succeed(entry, finishedId);
    } catch (error) {
      this.fail(entry, toError(error));
    } finally {
      this.inFlightCount -= 1;
      this.wake();
    }
  }

  private async pollUntilFinished(entry: TrackedEntry, backtestId: number, checkImmediately: boolean): Promise<number> {
    const startedAt = this.deps.now();
    let consecutiveErrors = 0;
    let skipSleep = checkImmediately;

    for (;;) {
      if (!skipSleep) {
        await this.deps.sleep(this.config.pollIntervalSeconds * 1000, entry.signal);
      }
      skipSleep = false;
      throwIfAborted(entry.signal);

      let outcome: PollOutcome;
      try {
        outcome = await this.checkBacktest(backtestId);
        consecutiveErrors = 0;
      } catch (error) {
        throwIfAborted(entry.signal);
        consecutiveErrors += 1;
        if (consecutiveErrors === 1 || consecutiveErrors % POLL_ERROR_LOG_EVERY === 0) {
          this.log(
            'warning',
            `Ошибка опроса бэктеста ${backtestId} (${entry.label}), попытка ${consecutiveErrors}: ${readMessage(error)}`,
          );
        }
        outcome = { kind: 'running' };
      }
      throwIfAborted(entry.signal);

      if (outcome.kind === 'done') {
        return backtestId;
      }
      if (outcome.kind === 'failed') {
        throw new Error(`Бэктест ${backtestId} (${entry.label}) завершился с ошибкой: статус ${outcome.status}`);
      }

      const timeoutMinutes = this.config.backtestTimeoutMinutes;
      if (this.deps.now() - startedAt >= timeoutMinutes * 60_000) {
        throw new Error(
          `Таймаут бэктеста ${backtestId} (${entry.label}): нет результата за ${formatSeconds(timeoutMinutes)} мин.`,
        );
      }
    }
  }

  private async checkBacktest(backtestId: number): Promise<PollOutcome> {
    const status = await this.deps.fetchStatus(backtestId);
    const statusClass = classifyBacktestStatus(status);
    if (statusClass === 'done') {
      return { kind: 'done' };
    }
    if (statusClass === 'failed') {
      return { kind: 'failed', status: status ?? '' };
    }
    if (statusClass === 'running') {
      return { kind: 'running' };
    }
    const hasStatistics = await this.deps.fetchStatisticsExists(backtestId);
    if (!hasStatistics) {
      return { kind: 'running' };
    }
    this.warnUnknownStatus(status);
    return { kind: 'done' };
  }

  private warnUnknownStatus(status: string | null): void {
    const label = status?.trim() ? status.trim() : 'пусто';
    if (this.warnedStatuses.has(label)) {
      return;
    }
    this.warnedStatuses.add(label);
    this.log('warning', `Неизвестный статус бэктеста «${label}», считаю завершённым (статистика уже доступна).`);
  }
}
