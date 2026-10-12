/**
 * Сеанс портфельного оптимизатора вне React.
 *
 * Экземпляр PortfolioOptimizer живёт в модуле, а не в компоненте страницы: при переходе на
 * другую страницу расширения (например, в созданную группу бэктестов) запуск продолжается, а
 * вернувшись, страница снова подписывается на него — без повторного «продолжения» из
 * сохранённого состояния, которое запустило бы второй оптимизатор параллельно.
 *
 * Снимок (snapshot) неизменяемый: каждое событие оптимизатора создаёт новый объект, поэтому
 * сеанс подходит для useSyncExternalStore.
 */

import type { BotStrategy } from '../../api/backtestRunner';
import { mergeSchedulerConfig } from '../../services/backtestScheduler';
import {
  MAX_PORTFOLIO_LOG_ENTRIES,
  PortfolioOptimizer,
  type PortfolioOptimizerDeps,
  type PortfolioOptimizerInput,
} from '../../services/portfolioOptimizer';
import type {
  Candidate,
  PortfolioLogEntry,
  PortfolioOptimizerCallbacks,
  PortfolioOptimizerConfig,
  PortfolioOptimizerProgress,
  PortfolioOptimizerState,
  SchedulerConfig,
} from '../../types/portfolioOptimizer';
import { progressFromState } from './runView';

/** Как часто обновлять прогресс (в работе, очередь, ETA) между событиями оптимизатора. */
export const PROGRESS_REFRESH_INTERVAL_MS = 5_000;

/** Запись журнала с ключом для списка React. */
export interface RunLogEntry extends PortfolioLogEntry {
  id: number;
}

/** Неизменные сведения о запуске (настройки планировщика могут меняться на ходу). */
export interface PortfolioRunMeta {
  runId: string;
  baseCandidateId: string;
  config: PortfolioOptimizerConfig;
  baseStrategy: BotStrategy;
  quoteCurrency: string;
  startedAt: number;
}

export interface PortfolioRunSnapshot {
  meta: PortfolioRunMeta;
  candidates: Candidate[];
  bestCandidateId: string | null;
  progress: PortfolioOptimizerProgress;
  logs: RunLogEntry[];
  /** Запуск выполняется в этой вкладке (экземпляр оптимизатора жив). */
  active: boolean;
}

export interface PortfolioRunSessionOptions {
  /** Зависимости оптимизатора (для тестов); по умолчанию — реальный API Veles. */
  deps?: Partial<PortfolioOptimizerDeps>;
  /** Интервал обновления прогресса, мс; 0 — без таймера. */
  refreshIntervalMs?: number;
}

type Listener = () => void;

const PROGRESS_KEYS: ReadonlyArray<keyof PortfolioOptimizerProgress> = [
  'status',
  'generation',
  'totalGenerations',
  'backtestsSubmitted',
  'backtestsLimit',
  'inFlight',
  'queued',
  'candidatesEvaluated',
  'startedAt',
  'estimatedEndAt',
];

const sameProgress = (left: PortfolioOptimizerProgress, right: PortfolioOptimizerProgress): boolean =>
  PROGRESS_KEYS.every((key) => left[key] === right[key]);

const toMeta = (state: PortfolioOptimizerState): PortfolioRunMeta => ({
  runId: state.runId,
  baseCandidateId: state.baseCandidateId,
  config: state.config,
  baseStrategy: state.baseStrategy,
  quoteCurrency: state.quoteCurrency,
  startedAt: state.startedAt,
});

export class PortfolioRunSession {
  private optimizer: PortfolioOptimizer | null = null;
  private snapshot: PortfolioRunSnapshot | null = null;
  private readonly listeners = new Set<Listener>();
  private readonly options: PortfolioRunSessionOptions;
  private nextLogId = 1;
  private refreshTimer: ReturnType<typeof setInterval> | null = null;

  constructor(options: PortfolioRunSessionOptions = {}) {
    this.options = options;
  }

  /** Подписка на изменения снимка (для useSyncExternalStore). */
  readonly subscribe = (listener: Listener): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  readonly getSnapshot = (): PortfolioRunSnapshot | null => this.snapshot;

  /** Запуск выполняется в этой вкладке. */
  get isActive(): boolean {
    return this.optimizer !== null;
  }

  /**
   * Новый запуск. Бросает ошибку сразу (с русским текстом), если конфигурация некорректна или
   * запуск уже идёт; возвращаемый промис завершения не отклоняется.
   */
  start(input: PortfolioOptimizerInput): Promise<void> {
    return this.launch((callbacks) => new PortfolioOptimizer(input, callbacks, this.options.deps));
  }

  /** Продолжение сохранённого запуска (ошибки — как у start). */
  resume(state: PortfolioOptimizerState): Promise<void> {
    return this.launch((callbacks) => PortfolioOptimizer.fromSaved(state, callbacks, this.options.deps));
  }

  /** Показ сохранённого запуска без выполнения (игнорируется, пока запуск идёт в этой вкладке). */
  show(state: PortfolioOptimizerState): void {
    if (this.optimizer) {
      return;
    }
    this.snapshot = {
      meta: toMeta(state),
      candidates: state.candidates,
      bestCandidateId: state.bestCandidateId,
      progress: progressFromState(state),
      logs: this.withLogIds(state.logs),
      active: false,
    };
    this.emit();
  }

  /** Сброс показанного запуска. false — запуск выполняется, сбрасывать нельзя. */
  clear(): boolean {
    if (this.optimizer) {
      return false;
    }
    if (this.snapshot !== null) {
      this.snapshot = null;
      this.emit();
    }
    return true;
  }

  pause(): void {
    this.optimizer?.pause();
  }

  unpause(): void {
    this.optimizer?.resume();
  }

  stop(): void {
    this.optimizer?.stop();
  }

  /** Потоки, интервал и таймауты планировщика на ходу. */
  updateSchedulerConfig(patch: Partial<SchedulerConfig>): void {
    const optimizer = this.optimizer;
    if (!optimizer) {
      return;
    }
    optimizer.updateSchedulerConfig(patch);
    this.update((snapshot) => ({
      ...snapshot,
      meta: {
        ...snapshot.meta,
        config: { ...snapshot.meta.config, scheduler: mergeSchedulerConfig(snapshot.meta.config.scheduler, patch) },
      },
    }));
  }

  /** Обновление прогресса (в работе, очередь, ETA) — вызывается таймером сеанса. */
  refreshProgress(): void {
    const optimizer = this.optimizer;
    const snapshot = this.snapshot;
    if (!(optimizer && snapshot)) {
      return;
    }
    const progress = optimizer.getProgress();
    if (!sameProgress(progress, snapshot.progress)) {
      this.update((current) => ({ ...current, progress }));
    }
  }

  // ─── Внутреннее ──────────────────────────────────────────────────

  private launch(create: (callbacks: PortfolioOptimizerCallbacks) => PortfolioOptimizer): Promise<void> {
    if (this.optimizer) {
      throw new Error('Оптимизация уже выполняется — дождитесь окончания или остановите текущий запуск.');
    }
    let instance: PortfolioOptimizer | null = null;
    const isCurrent = (): boolean => instance !== null && this.optimizer === instance;
    const callbacks: PortfolioOptimizerCallbacks = {
      onLog: (entry) => {
        if (isCurrent()) {
          this.update((snapshot) => ({ ...snapshot, logs: this.appendLog(snapshot.logs, entry) }));
        }
      },
      onProgress: (progress) => {
        if (isCurrent()) {
          this.update((snapshot) => ({ ...snapshot, progress }));
        }
      },
      onCandidatesChanged: (candidates, bestCandidateId) => {
        if (isCurrent()) {
          this.update((snapshot) => ({ ...snapshot, candidates, bestCandidateId }));
        }
      },
    };

    const optimizer = create(callbacks);
    instance = optimizer;
    const state = optimizer.getState();
    this.optimizer = optimizer;
    this.snapshot = {
      meta: toMeta(state),
      candidates: state.candidates,
      bestCandidateId: state.bestCandidateId,
      progress: optimizer.getProgress(),
      logs: this.withLogIds(state.logs),
      active: true,
    };
    this.emit();
    this.startRefreshTimer();

    return optimizer.start().finally(() => {
      if (this.optimizer !== optimizer) {
        return;
      }
      this.stopRefreshTimer();
      this.optimizer = null;
      const progress = optimizer.getProgress();
      this.update((snapshot) => ({ ...snapshot, progress, active: false }));
    });
  }

  private update(mutator: (snapshot: PortfolioRunSnapshot) => PortfolioRunSnapshot): void {
    if (!this.snapshot) {
      return;
    }
    this.snapshot = mutator(this.snapshot);
    this.emit();
  }

  private emit(): void {
    for (const listener of [...this.listeners]) {
      try {
        listener();
      } catch (error) {
        console.warn('[PortfolioOptimizer] Ошибка подписчика сеанса', error);
      }
    }
  }

  private withLogIds(entries: PortfolioLogEntry[]): RunLogEntry[] {
    return entries.slice(-MAX_PORTFOLIO_LOG_ENTRIES).map((entry) => this.toLogEntry(entry));
  }

  private toLogEntry(entry: PortfolioLogEntry): RunLogEntry {
    const id = this.nextLogId;
    this.nextLogId += 1;
    return { ...entry, id };
  }

  private appendLog(logs: RunLogEntry[], entry: PortfolioLogEntry): RunLogEntry[] {
    const next = [...logs, this.toLogEntry(entry)];
    return next.length > MAX_PORTFOLIO_LOG_ENTRIES ? next.slice(next.length - MAX_PORTFOLIO_LOG_ENTRIES) : next;
  }

  private startRefreshTimer(): void {
    const interval = this.options.refreshIntervalMs ?? PROGRESS_REFRESH_INTERVAL_MS;
    this.stopRefreshTimer();
    if (interval > 0) {
      this.refreshTimer = setInterval(() => this.refreshProgress(), interval);
    }
  }

  private stopRefreshTimer(): void {
    if (this.refreshTimer !== null) {
      clearInterval(this.refreshTimer);
      this.refreshTimer = null;
    }
  }
}

/** Общий сеанс страницы портфельного оптимизатора. */
export const portfolioRunSession = new PortfolioRunSession();
