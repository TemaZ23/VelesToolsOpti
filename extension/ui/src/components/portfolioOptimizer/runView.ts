/**
 * Представление запуска портфельного оптимизатора для UI: строки таблицы кандидатов,
 * статусы, сравнение лучшего кандидата с базой, прогресс по сохранённому состоянию и
 * форматирование чисел/длительностей. Модуль чистый.
 */

import { describeParamChange } from '../../lib/botParams';
import { type BacktestsEstimate, candidateShortLabel } from '../../services/portfolioOptimizer';
import type {
  Candidate,
  CandidateEvaluation,
  EvaluationStage,
  FoldMetrics,
  ParamSpec,
  PortfolioMetrics,
  PortfolioOptimizerProgress,
  PortfolioOptimizerState,
  PortfolioRunStatus,
} from '../../types/portfolioOptimizer';

// ═══════════════════════════════════════════════════════════════════
// ФОРМАТИРОВАНИЕ
// ═══════════════════════════════════════════════════════════════════

const MS_IN_MINUTE = 60_000;
const MS_IN_HOUR = 60 * MS_IN_MINUTE;
const MS_IN_DAY = 24 * MS_IN_HOUR;

const isFiniteNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

/** Число с фиксированным количеством знаков; «—» для отсутствующего значения. */
export const formatNumber = (value: number | null | undefined, digits = 2): string =>
  isFiniteNumber(value) ? value.toFixed(digits) : '—';

/** Число со знаком: «+1.50», «−0.25», «0.00». */
export const formatSigned = (value: number | null | undefined, digits = 2): string => {
  if (!isFiniteNumber(value)) {
    return '—';
  }
  const text = Math.abs(value).toFixed(digits);
  if (Number(text) === 0) {
    return text;
  }
  return value > 0 ? `+${text}` : `−${text}`;
};

/** Длительность: «2 д 4 ч», «3 ч 15 мин», «45 мин», «< 1 мин». */
export const formatDuration = (ms: number): string => {
  if (!isFiniteNumber(ms) || ms < MS_IN_MINUTE) {
    return '< 1 мин';
  }
  const days = Math.floor(ms / MS_IN_DAY);
  const hours = Math.floor((ms % MS_IN_DAY) / MS_IN_HOUR);
  const minutes = Math.floor((ms % MS_IN_HOUR) / MS_IN_MINUTE);
  if (days > 0) {
    return hours > 0 ? `${days} д ${hours} ч` : `${days} д`;
  }
  if (hours > 0) {
    return minutes > 0 ? `${hours} ч ${minutes} мин` : `${hours} ч`;
  }
  return `${minutes} мин`;
};

export const formatDateTime = (timestamp: number | null | undefined): string => {
  if (!isFiniteNumber(timestamp)) {
    return '—';
  }
  return new Date(timestamp).toLocaleString('ru-RU');
};

/** Оценка окончания: «12.10.2026, 18:30:00 (через 3 ч 10 мин)»; null — оценки нет. */
export const formatEta = (estimatedEndAt: number | null, now: number): string | null => {
  if (!isFiniteNumber(estimatedEndAt)) {
    return null;
  }
  return `${formatDateTime(estimatedEndAt)} (через ${formatDuration(estimatedEndAt - now)})`;
};

// ═══════════════════════════════════════════════════════════════════
// СТАТУСЫ
// ═══════════════════════════════════════════════════════════════════

/** Цвет тега antd. */
export type StatusTone = 'default' | 'processing' | 'success' | 'warning' | 'error';

export interface StatusLabel {
  text: string;
  tone: StatusTone;
}

export const RUN_STATUS_LABELS: Readonly<Record<PortfolioRunStatus, StatusLabel>> = {
  idle: { text: 'Не запущен', tone: 'default' },
  running: { text: 'Выполняется', tone: 'processing' },
  paused: { text: 'Пауза', tone: 'warning' },
  stopped: { text: 'Остановлен', tone: 'default' },
  completed: { text: 'Завершён', tone: 'success' },
  error: { text: 'Ошибка', tone: 'error' },
};

export type CandidatePhase =
  | 'base'
  | 'queued'
  | 'screenRunning'
  | 'screenDone'
  | 'screenFailed'
  | 'rejected'
  | 'fullPending'
  | 'fullRunning'
  | 'fullDone'
  | 'fullFailed';

export const CANDIDATE_PHASE_LABELS: Readonly<Record<CandidatePhase, StatusLabel>> = {
  base: { text: 'База', tone: 'default' },
  queued: { text: 'Ожидает отбора', tone: 'default' },
  screenRunning: { text: 'Отбор…', tone: 'processing' },
  screenDone: { text: 'Отбор посчитан', tone: 'default' },
  screenFailed: { text: 'Ошибка отбора', tone: 'error' },
  rejected: { text: 'Не прошёл отбор', tone: 'default' },
  fullPending: { text: 'Ждёт полной корзины', tone: 'default' },
  fullRunning: { text: 'Полная корзина…', tone: 'processing' },
  fullDone: { text: 'Полная корзина', tone: 'success' },
  fullFailed: { text: 'Ошибка полной корзины', tone: 'error' },
};

export interface CandidatePhaseContext {
  isBase: boolean;
  /** Текущее поколение запуска (активное при выполнении, завершённое — иначе). */
  currentGeneration: number;
  /** Запуск выполняется в этой вкладке. */
  active: boolean;
}

/**
 * Этап кандидата для таблицы. Кандидат без полной оценки «не прошёл отбор», только если его
 * поколение уже подведено: оно раньше текущего или запуск не активен и поколение завершено.
 */
export const resolveCandidatePhase = (candidate: Candidate, context: CandidatePhaseContext): CandidatePhase => {
  const { full, screen } = candidate;
  if (full) {
    switch (full.status) {
      case 'running':
        return 'fullRunning';
      case 'done':
        return context.isBase ? 'base' : 'fullDone';
      case 'failed':
        return 'fullFailed';
      default:
        return 'fullPending';
    }
  }
  if (context.isBase) {
    return 'base';
  }
  if (!screen || screen.status === 'pending') {
    return 'queued';
  }
  if (screen.status === 'running') {
    return 'screenRunning';
  }
  if (screen.status === 'failed') {
    return 'screenFailed';
  }
  const finalized =
    candidate.generation < context.currentGeneration ||
    (!context.active && candidate.generation <= context.currentGeneration);
  return finalized ? 'rejected' : 'screenDone';
};

// ═══════════════════════════════════════════════════════════════════
// КАНДИДАТЫ
// ═══════════════════════════════════════════════════════════════════

/** Score этапа, если этап посчитан. */
export const stageScore = (evaluation: CandidateEvaluation | null): number | null =>
  evaluation?.status === 'done' && isFiniteNumber(evaluation.score) ? evaluation.score : null;

const doneMetrics = (evaluation: CandidateEvaluation | null): PortfolioMetrics | null =>
  evaluation?.status === 'done' ? evaluation.metrics : null;

/** Описания изменённых параметров кандидата: «CCI 1 час < 46 → 40». */
export const describeCandidateChanges = (params: ParamSpec[], candidate: Pick<Candidate, 'changed' | 'values'>) => {
  const byId = new Map(params.map((param) => [param.id, param]));
  return candidate.changed.flatMap((id) => {
    const param = byId.get(id);
    const value = Object.hasOwn(candidate.values, id) ? candidate.values[id] : undefined;
    return param && isFiniteNumber(value) ? [describeParamChange(param, value)] : [];
  });
};

export interface CandidateBacktests {
  stage: EvaluationStage;
  /** id бэктестов в порядке монет этапа. */
  ids: number[];
}

const evaluationBacktestIds = (evaluation: CandidateEvaluation | null): number[] => {
  if (!evaluation) {
    return [];
  }
  return evaluation.symbols.flatMap((symbol) => {
    const id = Object.hasOwn(evaluation.backtestIds, symbol) ? evaluation.backtestIds[symbol] : undefined;
    return isFiniteNumber(id) && id > 0 ? [id] : [];
  });
};

/** Бэктесты кандидата: полной корзины, если они есть, иначе отборочной. */
export const candidateBacktests = (candidate: Pick<Candidate, 'screen' | 'full'>): CandidateBacktests | null => {
  const fullIds = evaluationBacktestIds(candidate.full);
  if (fullIds.length > 0) {
    return { stage: 'full', ids: fullIds };
  }
  const screenIds = evaluationBacktestIds(candidate.screen);
  return screenIds.length > 0 ? { stage: 'screen', ids: screenIds } : null;
};

/** Кандидат, над которым выполняется действие (группа бэктестов, JSON, создание бота). */
export interface CandidateActionTarget {
  candidate: Candidate;
  /** «база» или «g3-2». */
  label: string;
  isBase: boolean;
  backtests: CandidateBacktests | null;
}

/** Название бота, созданного из кандидата: «<бот> OPT <метка>». */
export const buildOptimizedBotName = (botName: string, label: string): string =>
  `${botName.trim().replace(/\s+/g, ' ')} OPT ${label}`.trim();

/** Название группы бэктестов кандидата: «PO <бот> <метка>» (+ «(отбор)» для отборочной корзины). */
export const buildCandidateGroupName = (botName: string, target: Pick<CandidateActionTarget, 'label' | 'backtests'>) =>
  `PO ${botName.trim().replace(/\s+/g, ' ')} ${target.label}${target.backtests?.stage === 'screen' ? ' (отбор)' : ''}`;

export interface CandidateRow extends CandidateActionTarget {
  key: string;
  isBest: boolean;
  generation: number;
  changes: string[];
  screenScore: number | null;
  fullScore: number | null;
  /** Метрики полной корзины, если посчитаны, иначе отборочной. */
  metrics: PortfolioMetrics | null;
  metricsStage: EvaluationStage | null;
  phase: CandidatePhase;
  error: string | null;
}

export interface CandidateRowsInput {
  runId: string;
  baseCandidateId: string;
  params: ParamSpec[];
  candidates: Candidate[];
  bestCandidateId: string | null;
  currentGeneration: number;
  active: boolean;
}

/** null в конце при сортировке по убыванию. */
const compareNullableDesc = (left: number | null, right: number | null): number => {
  if (left === null || right === null) {
    return left === right ? 0 : left === null ? 1 : -1;
  }
  return right - left;
};

/** Строки таблицы: по score полной корзины, затем отбора (оба по убыванию), затем по порядку создания. */
export const buildCandidateRows = (input: CandidateRowsInput): CandidateRow[] => {
  const state = { runId: input.runId, baseCandidateId: input.baseCandidateId };
  const rows = input.candidates.map((candidate, index) => {
    const isBase = candidate.id === input.baseCandidateId;
    const fullMetrics = doneMetrics(candidate.full);
    const screenMetrics = doneMetrics(candidate.screen);
    const metricsStage: EvaluationStage | null = fullMetrics ? 'full' : screenMetrics ? 'screen' : null;
    const row: CandidateRow = {
      key: candidate.id,
      candidate,
      label: candidateShortLabel(state, candidate),
      isBase,
      isBest: candidate.id === input.bestCandidateId,
      generation: candidate.generation,
      changes: describeCandidateChanges(input.params, candidate),
      screenScore: stageScore(candidate.screen),
      fullScore: stageScore(candidate.full),
      metrics: fullMetrics ?? screenMetrics,
      metricsStage,
      phase: resolveCandidatePhase(candidate, {
        isBase,
        currentGeneration: input.currentGeneration,
        active: input.active,
      }),
      error: candidate.full?.error ?? candidate.screen?.error ?? null,
      backtests: candidateBacktests(candidate),
    };
    return { row, index };
  });
  rows.sort(
    (left, right) =>
      compareNullableDesc(left.row.fullScore, right.row.fullScore) ||
      compareNullableDesc(left.row.screenScore, right.row.screenScore) ||
      left.index - right.index,
  );
  return rows.map(({ row }) => row);
};

// ═══════════════════════════════════════════════════════════════════
// ЛУЧШИЙ КАНДИДАТ
// ═══════════════════════════════════════════════════════════════════

export interface MetricComparison {
  value: number | null;
  base: number | null;
  delta: number | null;
  /** Изменение в процентах от |base|; null, если база 0 или неизвестна. */
  deltaPercent: number | null;
  /** true — лучше базы, false — хуже, null — без изменений или нечего сравнивать. */
  better: boolean | null;
}

export const compareMetric = (
  value: number | null | undefined,
  base: number | null | undefined,
  higherIsBetter: boolean,
): MetricComparison => {
  const current = isFiniteNumber(value) ? value : null;
  const reference = isFiniteNumber(base) ? base : null;
  if (current === null || reference === null) {
    return { value: current, base: reference, delta: null, deltaPercent: null, better: null };
  }
  const delta = current - reference;
  const deltaPercent = reference !== 0 ? (delta / Math.abs(reference)) * 100 : null;
  const better = Math.abs(delta) < 1e-9 ? null : higherIsBetter ? delta > 0 : delta < 0;
  return { value: current, base: reference, delta, deltaPercent, better };
};

export interface BestSummary {
  best: Candidate;
  base: Candidate;
  label: string;
  /** Лучший — сам базовый бот (улучшений нет). */
  isBase: boolean;
  score: MetricComparison;
  profit: MetricComparison;
  mae: MetricComparison;
  drawdown: MetricComparison;
  pnlToRisk: MetricComparison;
  deals: MetricComparison;
  changes: string[];
  folds: FoldMetrics[];
  baseFolds: FoldMetrics[];
  backtests: CandidateBacktests | null;
}

export interface BestSummaryInput {
  runId: string;
  baseCandidateId: string;
  params: ParamSpec[];
  candidates: Candidate[];
  bestCandidateId: string | null;
}

/** Сравнение лучшего кандидата с базой по полной корзине; null — база ещё не посчитана. */
export const buildBestSummary = (input: BestSummaryInput): BestSummary | null => {
  const base = input.candidates.find((candidate) => candidate.id === input.baseCandidateId);
  if (!base || stageScore(base.full) === null) {
    return null;
  }
  const best = input.candidates.find((candidate) => candidate.id === input.bestCandidateId) ?? base;
  const bestMetrics = doneMetrics(best.full);
  const baseMetrics = doneMetrics(base.full);
  return {
    best,
    base,
    label: candidateShortLabel({ runId: input.runId, baseCandidateId: input.baseCandidateId }, best),
    isBase: best.id === base.id,
    score: compareMetric(stageScore(best.full), stageScore(base.full), true),
    profit: compareMetric(bestMetrics?.totalProfit, baseMetrics?.totalProfit, true),
    mae: compareMetric(bestMetrics?.maxConcurrentMae, baseMetrics?.maxConcurrentMae, false),
    drawdown: compareMetric(bestMetrics?.maxDrawdown, baseMetrics?.maxDrawdown, false),
    pnlToRisk: compareMetric(bestMetrics?.pnlToRisk, baseMetrics?.pnlToRisk, true),
    deals: compareMetric(bestMetrics?.totalDeals, baseMetrics?.totalDeals, true),
    changes: describeCandidateChanges(input.params, best),
    folds: best.full?.status === 'done' ? best.full.folds : [],
    baseFolds: base.full?.status === 'done' ? base.full.folds : [],
    backtests: candidateBacktests(best),
  };
};

// ═══════════════════════════════════════════════════════════════════
// ПРОГРЕСС И СОХРАНЁННЫЙ ЗАПУСК
// ═══════════════════════════════════════════════════════════════════

const isStageFinished = (evaluation: CandidateEvaluation | null): boolean =>
  evaluation?.status === 'done' || evaluation?.status === 'failed';

/**
 * Статус сохранённого запуска, который не выполняется в этой вкладке: «выполняется» и «пауза»
 * означают, что страница закрылась во время запуска, — он прерван (как при продолжении через
 * PortfolioOptimizer.fromSaved, где такой статус становится «остановлен»).
 */
export const savedRunStatus = (status: PortfolioRunStatus): PortfolioRunStatus =>
  status === 'running' || status === 'paused' ? 'stopped' : status;

/** Подпись статуса сохранённого запуска: прерванный выделяется отдельно. */
export const savedRunStatusLabel = (status: PortfolioRunStatus): StatusLabel =>
  status === 'running' || status === 'paused' ? { text: 'Прерван', tone: 'warning' } : RUN_STATUS_LABELS[status];

/** Прогресс по сохранённому состоянию (запуск не выполняется в этой вкладке). */
export const progressFromState = (state: PortfolioOptimizerState): PortfolioOptimizerProgress => ({
  status: savedRunStatus(state.status),
  generation: state.generation,
  totalGenerations: state.config.search.generations,
  backtestsSubmitted: state.backtestsSubmitted,
  backtestsLimit: state.config.search.maxBacktests,
  inFlight: 0,
  queued: 0,
  candidatesEvaluated: state.candidates.filter(
    (candidate) =>
      candidate.id !== state.baseCandidateId && (isStageFinished(candidate.screen) || isStageFinished(candidate.full)),
  ).length,
  startedAt: state.status === 'idle' ? null : state.startedAt,
  estimatedEndAt: null,
});

/** Сохранённый запуск можно продолжить (завершённый — только просмотр). */
export const canResumeSavedRun = (state: Pick<PortfolioOptimizerState, 'status'>): boolean =>
  state.status !== 'completed';

/** Окно, в течение которого «выполняющийся» сохранённый запуск может принадлежать другой вкладке. */
export const OTHER_TAB_ACTIVITY_WINDOW_MS = 3 * MS_IN_MINUTE;

/**
 * Сохранённый запуск помечен выполняющимся и недавно обновлялся — вероятно, он идёт в другой
 * вкладке расширения (в этой вкладке его нет).
 */
export const isProbablyRunningElsewhere = (
  state: Pick<PortfolioOptimizerState, 'status' | 'updatedAt'>,
  now: number,
): boolean =>
  (state.status === 'running' || state.status === 'paused') &&
  isFiniteNumber(state.updatedAt) &&
  now - state.updatedAt < OTHER_TAB_ACTIVITY_WINDOW_MS;

/**
 * Прогресс запуска в процентах: созданные бэктесты относительно оценки их общего числа
 * (оценка — максимум, поэтому до завершения не больше 99%, после — 100%).
 */
export const runProgressPercent = (
  progress: Pick<PortfolioOptimizerProgress, 'status' | 'backtestsSubmitted'>,
  expectedTotal: number,
): number => {
  if (progress.status === 'completed') {
    return 100;
  }
  if (!(expectedTotal > 0)) {
    return 0;
  }
  return Math.min(99, Math.max(0, Math.floor((progress.backtestsSubmitted / expectedTotal) * 100)));
};

/**
 * Оценка объёма с учётом проверенной группы бэктестов базы: недостающие в группе монеты
 * базы будут посчитаны бэктестами (estimateBacktests считает базу из группы бесплатной).
 */
export const withGroupMissing = (
  estimate: BacktestsEstimate,
  missing: number,
  maxBacktests: number,
  submitDelaySeconds: number,
): BacktestsEstimate => {
  const baseBacktests = Math.max(0, Math.floor(missing));
  const totalMax = Math.min(Math.max(0, maxBacktests), estimate.totalMax + baseBacktests);
  return {
    ...estimate,
    baseBacktests,
    totalMax,
    hoursAtDelay: (totalMax * Math.max(0, submitDelaySeconds)) / 3600,
  };
};
