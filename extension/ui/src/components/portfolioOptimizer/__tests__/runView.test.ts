import { describe, expect, it } from 'vitest';
import type { BotStrategy } from '../../../api/backtestRunner';
import type { StrategyConditionDto } from '../../../api/bots.dtos';
import { describeParamChange, extractParams } from '../../../lib/botParams';
import {
  DEFAULT_OBJECTIVE,
  DEFAULT_SCHEDULER_CONFIG,
  DEFAULT_SEARCH_CONFIG,
} from '../../../services/portfolioOptimizer';
import type {
  Candidate,
  CandidateEvaluation,
  EvaluationStage,
  FoldMetrics,
  ParamSpec,
  PortfolioMetrics,
  PortfolioOptimizerState,
} from '../../../types/portfolioOptimizer';
import {
  buildBestSummary,
  buildCandidateGroupName,
  buildCandidateRows,
  buildOptimizedBotName,
  CANDIDATE_PHASE_LABELS,
  type CandidateRow,
  candidateBacktests,
  canResumeSavedRun,
  compareMetric,
  compareNullable,
  compareRowMetric,
  describeCandidateChanges,
  formatDuration,
  formatEta,
  formatNumber,
  formatSigned,
  isProbablyRunningElsewhere,
  metricsStageRank,
  OTHER_TAB_ACTIVITY_WINDOW_MS,
  progressFromState,
  RUN_STATUS_LABELS,
  resolveCandidatePhase,
  runProgressPercent,
  savedRunStatus,
  savedRunStatusLabel,
  stageScore,
  withGroupMissing,
  withSchedulerConfig,
} from '../runView';

// ═══════════════════════════════════════════════════════════════════
// ФИКСТУРЫ
// ═══════════════════════════════════════════════════════════════════

const RUN_ID = 'potest';
const BASE_ID = `${RUN_ID}-base`;
const RSI_ID = 'conditions[0].value';

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

const strategy: BotStrategy = {
  id: 1,
  name: 'LONG',
  algorithm: 'LONG',
  conditions: [condition('RSI', 40), condition('CCI', -100)],
  settings: {
    type: 'CUSTOM',
    baseOrder: null,
    orders: [
      { indent: 0, volume: 50, conditions: [] },
      { indent: 2, volume: 50, conditions: [] },
    ],
  },
  profit: { type: 'ALL', currency: 'USDT', checkPnl: 1, conditions: [] },
  pullUp: null,
};

const PARAMS: ParamSpec[] = extractParams(strategy);

const metrics = (pnlToRisk: number, patch: Partial<PortfolioMetrics> = {}): PortfolioMetrics => ({
  totalProfit: pnlToRisk * 100,
  maxConcurrentMae: 100,
  maxDrawdown: 50,
  pnlToRisk,
  totalDeals: 400,
  profitableDeals: 390,
  losingDeals: 10,
  openDeals: 0,
  activeMae: 0,
  netPerDay: 1,
  maxConcurrentPositions: 5,
  backtests: 3,
  ...patch,
});

const evaluation = (
  stage: EvaluationStage,
  status: CandidateEvaluation['status'],
  score: number | null,
  patch: Partial<CandidateEvaluation> = {},
): CandidateEvaluation => {
  const symbols = stage === 'screen' ? ['ADA', 'SOL'] : ['ADA', 'SOL', 'XRP'];
  return {
    stage,
    symbols,
    backtestIds: status === 'done' ? Object.fromEntries(symbols.map((symbol, index) => [symbol, 100 + index])) : {},
    failedSymbols: [],
    status,
    metrics: status === 'done' && score !== null ? metrics(score) : null,
    folds: [],
    score: status === 'done' ? score : null,
    error: status === 'failed' ? 'не посчитаны XRP: ошибка' : null,
    ...patch,
  };
};

const candidate = (id: string, patch: Partial<Candidate> = {}): Candidate => ({
  id,
  hash: `h-${id}`,
  values: Object.fromEntries(PARAMS.map((param) => [param.id, param.base])),
  parentId: null,
  generation: 1,
  changed: [],
  mutated: [],
  screen: null,
  full: null,
  createdAt: 0,
  ...patch,
});

const baseCandidate = (fullScore = 2): Candidate =>
  candidate(BASE_ID, {
    generation: 0,
    screen: evaluation('screen', 'done', fullScore),
    full: evaluation('full', 'done', fullScore),
  });

const createState = (patch: Partial<PortfolioOptimizerState> = {}): PortfolioOptimizerState => ({
  version: 1,
  runId: RUN_ID,
  config: {
    botId: 1,
    botName: 'LONG',
    basket: ['ADA', 'SOL', 'XRP'],
    screenBasket: ['ADA', 'SOL'],
    run: {
      periodFrom: '2022-01-01T00:00:00.000Z',
      periodTo: '2024-01-01T23:59:59.999Z',
      makerCommission: 0.02,
      takerCommission: 0.055,
      includeWicks: true,
    },
    block: 60,
    objective: { ...DEFAULT_OBJECTIVE },
    search: { ...DEFAULT_SEARCH_CONFIG, generations: 4, maxBacktests: 200 },
    scheduler: { ...DEFAULT_SCHEDULER_CONFIG },
    params: PARAMS,
    baseGroupId: null,
  },
  baseStrategy: strategy,
  quoteCurrency: 'USDT',
  baseCandidateId: BASE_ID,
  bestCandidateId: BASE_ID,
  candidates: [baseCandidate()],
  backtestCache: {},
  inFlightJobs: [],
  paramStats: {},
  generation: 2,
  backtestsSubmitted: 40,
  status: 'stopped',
  startedAt: 1_000,
  updatedAt: 2_000,
  logs: [],
  ...patch,
});

// ═══════════════════════════════════════════════════════════════════
// ТЕСТЫ
// ═══════════════════════════════════════════════════════════════════

describe('runView: форматирование', () => {
  it('formatNumber и formatSigned', () => {
    expect(formatNumber(1.23456)).toBe('1.23');
    expect(formatNumber(1.23456, 3)).toBe('1.235');
    expect(formatNumber(null)).toBe('—');
    expect(formatNumber(Number.NaN)).toBe('—');
    expect(formatSigned(1.5)).toBe('+1.50');
    expect(formatSigned(-0.25)).toBe('−0.25');
    expect(formatSigned(0.001)).toBe('0.00');
    expect(formatSigned(undefined)).toBe('—');
  });

  it('formatDuration', () => {
    expect(formatDuration(30_000)).toBe('< 1 мин');
    expect(formatDuration(Number.NaN)).toBe('< 1 мин');
    expect(formatDuration(45 * 60_000)).toBe('45 мин');
    expect(formatDuration(3 * 3_600_000)).toBe('3 ч');
    expect(formatDuration(3 * 3_600_000 + 15 * 60_000)).toBe('3 ч 15 мин');
    expect(formatDuration(2 * 86_400_000)).toBe('2 д');
    expect(formatDuration(2 * 86_400_000 + 4 * 3_600_000)).toBe('2 д 4 ч');
  });

  it('formatEta возвращает null без оценки и добавляет оставшееся время', () => {
    expect(formatEta(null, 0)).toBeNull();
    const eta = formatEta(Date.parse('2026-10-12T12:00:00.000Z'), Date.parse('2026-10-12T10:30:00.000Z'));
    expect(eta).toContain('через 1 ч 30 мин');
  });

  it('названия бота и группы кандидата', () => {
    expect(buildOptimizedBotName('  LONG   22 ', 'g3-2')).toBe('LONG 22 OPT g3-2');
    expect(buildCandidateGroupName('LONG 22', { label: 'g1-1', backtests: { stage: 'full', ids: [1] } })).toBe(
      'PO LONG 22 g1-1',
    );
    expect(buildCandidateGroupName('LONG 22', { label: 'g1-1', backtests: { stage: 'screen', ids: [1] } })).toBe(
      'PO LONG 22 g1-1 (отбор)',
    );
  });

  it('у всех статусов есть подписи', () => {
    expect(RUN_STATUS_LABELS.running.text).toBe('Выполняется');
    expect(Object.keys(CANDIDATE_PHASE_LABELS)).toHaveLength(11);
  });
});

describe('runView: этапы кандидатов', () => {
  const context = { isBase: false, currentGeneration: 2, active: true };

  it('этап по полной корзине', () => {
    expect(resolveCandidatePhase(candidate('a', { full: evaluation('full', 'running', null) }), context)).toBe(
      'fullRunning',
    );
    expect(resolveCandidatePhase(candidate('a', { full: evaluation('full', 'done', 2) }), context)).toBe('fullDone');
    expect(resolveCandidatePhase(candidate('a', { full: evaluation('full', 'failed', null) }), context)).toBe(
      'fullFailed',
    );
    expect(resolveCandidatePhase(candidate('a', { full: evaluation('full', 'pending', null) }), context)).toBe(
      'fullPending',
    );
  });

  it('этап по отбору', () => {
    expect(resolveCandidatePhase(candidate('a'), context)).toBe('queued');
    expect(resolveCandidatePhase(candidate('a', { screen: evaluation('screen', 'pending', null) }), context)).toBe(
      'queued',
    );
    expect(resolveCandidatePhase(candidate('a', { screen: evaluation('screen', 'running', null) }), context)).toBe(
      'screenRunning',
    );
    expect(resolveCandidatePhase(candidate('a', { screen: evaluation('screen', 'failed', null) }), context)).toBe(
      'screenFailed',
    );
  });

  it('«не прошёл отбор» — только для подведённых поколений', () => {
    const screened = (generation: number) => candidate('a', { generation, screen: evaluation('screen', 'done', 1.5) });
    expect(resolveCandidatePhase(screened(2), context)).toBe('screenDone');
    expect(resolveCandidatePhase(screened(1), context)).toBe('rejected');
    expect(resolveCandidatePhase(screened(2), { ...context, active: false })).toBe('rejected');
    expect(resolveCandidatePhase(screened(3), { ...context, active: false })).toBe('screenDone');
  });

  it('без живого оптимизатора считающийся этап — «прерван», а не «в работе»', () => {
    const inactive = { ...context, active: false };
    expect(resolveCandidatePhase(candidate('a', { screen: evaluation('screen', 'running', null) }), inactive)).toBe(
      'interrupted',
    );
    expect(resolveCandidatePhase(candidate('a', { full: evaluation('full', 'running', null) }), inactive)).toBe(
      'interrupted',
    );
    expect(resolveCandidatePhase(candidate('a', { full: evaluation('full', 'pending', null) }), inactive)).toBe(
      'interrupted',
    );
    expect(
      resolveCandidatePhase(candidate(BASE_ID, { full: evaluation('full', 'running', null) }), {
        ...inactive,
        isBase: true,
      }),
    ).toBe('interrupted');
    // Посчитанные и неудачные этапы показываются как есть.
    expect(resolveCandidatePhase(candidate('a', { full: evaluation('full', 'done', 2) }), inactive)).toBe('fullDone');
    expect(resolveCandidatePhase(candidate('a', { screen: evaluation('screen', 'failed', null) }), inactive)).toBe(
      'screenFailed',
    );
    expect(CANDIDATE_PHASE_LABELS.interrupted).toEqual({ text: 'Прерван', tone: 'warning' });
  });

  it('база', () => {
    expect(resolveCandidatePhase(baseCandidate(), { ...context, isBase: true })).toBe('base');
    expect(resolveCandidatePhase(candidate(BASE_ID), { ...context, isBase: true })).toBe('base');
    expect(
      resolveCandidatePhase(candidate(BASE_ID, { full: evaluation('full', 'running', null) }), {
        ...context,
        isBase: true,
      }),
    ).toBe('fullRunning');
  });
});

describe('runView: строки кандидатов', () => {
  const rsi = PARAMS.find((param) => param.id === RSI_ID);
  if (!rsi) {
    throw new Error('нет параметра RSI');
  }

  it('описывает изменения и игнорирует неизвестные параметры', () => {
    const changed = candidate('a', { changed: [RSI_ID, 'unknown'], values: { [RSI_ID]: 34 } });
    expect(describeCandidateChanges(PARAMS, changed)).toEqual([describeParamChange(rsi, 34)]);
  });

  it('бэктесты: полной корзины, иначе отбора, в порядке монет', () => {
    expect(candidateBacktests(candidate('a'))).toBeNull();
    expect(candidateBacktests(candidate('a', { screen: evaluation('screen', 'done', 1) }))).toEqual({
      stage: 'screen',
      ids: [100, 101],
    });
    const both = candidate('a', {
      screen: evaluation('screen', 'done', 1),
      full: evaluation('full', 'running', null, { backtestIds: { XRP: 7, ADA: 5, SOL: 0 } }),
    });
    expect(candidateBacktests(both)).toEqual({ stage: 'full', ids: [5, 7] });
  });

  it('stageScore — только у посчитанного этапа', () => {
    expect(stageScore(null)).toBeNull();
    expect(stageScore(evaluation('screen', 'done', 1.2))).toBe(1.2);
    expect(stageScore(evaluation('screen', 'failed', null))).toBeNull();
    expect(stageScore({ ...evaluation('screen', 'done', 1), score: Number.NaN })).toBeNull();
  });

  it('сортирует по полной корзине, затем по отбору и порядку создания', () => {
    const candidates = [
      baseCandidate(2),
      candidate(`${RUN_ID}-g1-1`, { screen: evaluation('screen', 'done', 1.1) }),
      candidate(`${RUN_ID}-g1-2`, {
        screen: evaluation('screen', 'done', 2.5),
        full: evaluation('full', 'done', 2.4),
        changed: [RSI_ID],
        values: { [RSI_ID]: 34 },
      }),
      candidate(`${RUN_ID}-g1-3`, { screen: evaluation('screen', 'done', 3) }),
      candidate(`${RUN_ID}-g2-1`),
      candidate(`${RUN_ID}-g2-2`, { generation: 2, screen: evaluation('screen', 'failed', null) }),
    ];
    const rows = buildCandidateRows({
      runId: RUN_ID,
      baseCandidateId: BASE_ID,
      params: PARAMS,
      candidates,
      bestCandidateId: `${RUN_ID}-g1-2`,
      currentGeneration: 2,
      active: true,
    });
    expect(rows.map((row) => row.label)).toEqual(['g1-2', 'база', 'g1-3', 'g1-1', 'g2-1', 'g2-2']);

    const best = rows[0];
    expect(best.isBest).toBe(true);
    expect(best.fullScore).toBe(2.4);
    expect(best.metricsStage).toBe('full');
    expect(best.metrics?.pnlToRisk).toBe(2.4);
    expect(best.changes).toEqual([describeParamChange(rsi, 34)]);
    expect(best.backtests?.stage).toBe('full');

    const base = rows[1];
    expect(base.isBase).toBe(true);
    expect(base.phase).toBe('base');

    const screenOnly = rows[2];
    expect(screenOnly.metricsStage).toBe('screen');
    expect(screenOnly.fullScore).toBeNull();
    expect(screenOnly.phase).toBe('rejected');

    const failed = rows[5];
    expect(failed.phase).toBe('screenFailed');
    expect(failed.error).toContain('не посчитаны');
    expect(failed.metrics).toBeNull();
  });
});

describe('runView: сортировка метрик в таблице', () => {
  const row = (stage: CandidateRow['metricsStage'], pnlToRisk: number | null) => ({
    metricsStage: stage,
    metrics: pnlToRisk === null ? null : metrics(pnlToRisk),
  });

  it('compareNullable ставит отсутствующие значения в начало', () => {
    expect([3, null, 1, Number.NaN, 2].sort(compareNullable)).toEqual([null, Number.NaN, 1, 2, 3]);
  });

  it('не смешивает метрики отборочной и полной корзины', () => {
    expect(metricsStageRank({ metricsStage: 'full' })).toBe(2);
    expect(metricsStageRank({ metricsStage: 'screen' })).toBe(1);
    expect(metricsStageRank({ metricsStage: null })).toBe(0);
    // Отбор 2.4 (меньше монет) не «обгоняет» полную корзину 1.9.
    const rows = [row('screen', 2.4), row('full', 1.9), row(null, null), row('full', 1.2), row('screen', 0.5)];
    const ascending = [...rows].sort((left, right) => compareRowMetric(left, right, (item) => item.pnlToRisk));
    expect(ascending.map((item) => [item.metricsStage, item.metrics?.pnlToRisk ?? null])).toEqual([
      [null, null],
      ['screen', 0.5],
      ['screen', 2.4],
      ['full', 1.2],
      ['full', 1.9],
    ]);
  });
});

describe('runView: лучший кандидат', () => {
  it('compareMetric учитывает направление «лучше»', () => {
    expect(compareMetric(12, 10, true)).toEqual({ value: 12, base: 10, delta: 2, deltaPercent: 20, better: true });
    expect(compareMetric(12, 10, false).better).toBe(false);
    expect(compareMetric(8, 10, false).better).toBe(true);
    expect(compareMetric(10, 10, true).better).toBeNull();
    expect(compareMetric(5, 0, true).deltaPercent).toBeNull();
    expect(compareMetric(-5, -10, true).deltaPercent).toBe(50);
    expect(compareMetric(null, 10, true)).toEqual({
      value: null,
      base: 10,
      delta: null,
      deltaPercent: null,
      better: null,
    });
  });

  it('null, пока база не посчитана на полной корзине', () => {
    const input = {
      runId: RUN_ID,
      baseCandidateId: BASE_ID,
      params: PARAMS,
      bestCandidateId: null,
    };
    expect(buildBestSummary({ ...input, candidates: [] })).toBeNull();
    expect(buildBestSummary({ ...input, candidates: [candidate(BASE_ID, { generation: 0 })] })).toBeNull();
  });

  it('сравнивает лучшего с базой и отдаёт под-периоды', () => {
    const folds: FoldMetrics[] = [
      { from: 0, to: 10, profit: 5, risk: 2, pnlToRisk: 2.5, deals: 10 },
      { from: 10, to: 20, profit: 3, risk: 2, pnlToRisk: 1.5, deals: 8 },
    ];
    const best = candidate(`${RUN_ID}-g2-1`, {
      generation: 2,
      changed: [RSI_ID],
      values: { [RSI_ID]: 34 },
      screen: evaluation('screen', 'done', 3),
      full: evaluation('full', 'done', 3, {
        metrics: metrics(3, { maxConcurrentMae: 80, totalDeals: 380 }),
        folds,
      }),
    });
    const summary = buildBestSummary({
      runId: RUN_ID,
      baseCandidateId: BASE_ID,
      params: PARAMS,
      candidates: [baseCandidate(2), best],
      bestCandidateId: best.id,
    });
    expect(summary).not.toBeNull();
    expect(summary?.label).toBe('g2-1');
    expect(summary?.isBase).toBe(false);
    expect(summary?.score).toMatchObject({ value: 3, base: 2, delta: 1, better: true });
    expect(summary?.profit).toMatchObject({ value: 300, base: 200, better: true });
    expect(summary?.mae).toMatchObject({ value: 80, base: 100, better: true });
    expect(summary?.deals).toMatchObject({ value: 380, base: 400, better: false });
    expect(summary?.changes).toHaveLength(1);
    expect(summary?.folds).toEqual(folds);
    expect(summary?.baseFolds).toEqual([]);
    expect(summary?.backtests).toEqual({ stage: 'full', ids: [100, 101, 102] });
  });

  it('без лучшего кандидата лучший — база', () => {
    const summary = buildBestSummary({
      runId: RUN_ID,
      baseCandidateId: BASE_ID,
      params: PARAMS,
      candidates: [baseCandidate(2)],
      bestCandidateId: 'missing',
    });
    expect(summary?.isBase).toBe(true);
    expect(summary?.label).toBe('база');
    expect(summary?.score.better).toBeNull();
  });
});

describe('runView: прогресс и сохранённый запуск', () => {
  it('progressFromState считает оценённых кандидатов без базы', () => {
    const state = createState({
      candidates: [
        baseCandidate(),
        candidate(`${RUN_ID}-g1-1`, { screen: evaluation('screen', 'done', 1) }),
        candidate(`${RUN_ID}-g1-2`, { screen: evaluation('screen', 'failed', null) }),
        candidate(`${RUN_ID}-g1-3`, { screen: evaluation('screen', 'running', null) }),
      ],
    });
    expect(progressFromState(state)).toEqual({
      status: 'stopped',
      generation: 2,
      totalGenerations: 4,
      backtestsSubmitted: 40,
      backtestsLimit: 200,
      inFlight: 0,
      queued: 0,
      candidatesEvaluated: 2,
      startedAt: 1_000,
      estimatedEndAt: null,
    });
    expect(progressFromState(createState({ status: 'idle' })).startedAt).toBeNull();
  });

  it('«выполняется» и «пауза» без живого оптимизатора показываются как прерванный запуск', () => {
    expect(progressFromState(createState({ status: 'running' })).status).toBe('stopped');
    expect(progressFromState(createState({ status: 'paused' })).status).toBe('stopped');
    expect(progressFromState(createState({ status: 'completed' })).status).toBe('completed');
    expect(progressFromState(createState({ status: 'error' })).status).toBe('error');
    expect(savedRunStatus('running')).toBe('stopped');
    expect(savedRunStatus('idle')).toBe('idle');
    expect(savedRunStatusLabel('paused')).toEqual({ text: 'Прерван', tone: 'warning' });
    expect(savedRunStatusLabel('completed')).toEqual(RUN_STATUS_LABELS.completed);
    // Запуск, который держит блокировку другой вкладки, — не «прерван».
    expect(savedRunStatusLabel('paused', true)).toEqual({ text: 'Выполняется в другой вкладке', tone: 'processing' });
  });

  it('продолжить можно всё, кроме завершённого запуска', () => {
    expect(canResumeSavedRun({ status: 'stopped' })).toBe(true);
    expect(canResumeSavedRun({ status: 'error' })).toBe(true);
    expect(canResumeSavedRun({ status: 'completed' })).toBe(false);
  });

  it('недавно обновлявшийся «выполняющийся» запуск — вероятно, в другой вкладке', () => {
    const now = 10 * OTHER_TAB_ACTIVITY_WINDOW_MS;
    const run = (status: PortfolioOptimizerState['status'], updatedAt: number) => ({
      status,
      updatedAt,
      runId: RUN_ID,
    });
    expect(isProbablyRunningElsewhere(run('running', now - 1_000), now)).toBe(true);
    expect(isProbablyRunningElsewhere(run('paused', now - 1_000), now)).toBe(true);
    expect(isProbablyRunningElsewhere(run('running', now - OTHER_TAB_ACTIVITY_WINDOW_MS), now)).toBe(false);
    expect(isProbablyRunningElsewhere(run('stopped', now), now)).toBe(false);
  });

  it('«пульс» другой вкладки показывает запуск на паузе, хотя состояние давно не сохранялось', () => {
    const now = 10 * OTHER_TAB_ACTIVITY_WINDOW_MS;
    const paused = { status: 'paused' as const, updatedAt: now - 5 * OTHER_TAB_ACTIVITY_WINDOW_MS, runId: RUN_ID };
    const beat = { runId: RUN_ID, tabId: 'other', at: now - 20_000 };
    expect(isProbablyRunningElsewhere(paused, now)).toBe(false);
    expect(isProbablyRunningElsewhere(paused, now, beat, 'mine')).toBe(true);
    // Свой «пульс», чужой запуск и устаревший «пульс» не считаются.
    expect(isProbablyRunningElsewhere(paused, now, beat, 'other')).toBe(false);
    expect(isProbablyRunningElsewhere(paused, now, { ...beat, runId: 'po-other' }, 'mine')).toBe(false);
    expect(isProbablyRunningElsewhere(paused, now, { ...beat, at: now - OTHER_TAB_ACTIVITY_WINDOW_MS }, 'mine')).toBe(
      false,
    );
    expect(isProbablyRunningElsewhere({ ...paused, status: 'stopped' }, now, beat, 'mine')).toBe(false);
  });

  it('продолжение берёт настройки планировщика из формы, остальное — из сохранённого запуска', () => {
    const state = createState();
    const scheduler = { maxInFlight: 2, submitDelaySeconds: 45, pollIntervalSeconds: 15, backtestTimeoutMinutes: 90 };
    const patched = withSchedulerConfig(state, scheduler);
    expect(patched.config.scheduler).toEqual(scheduler);
    expect(patched.config.scheduler).not.toBe(scheduler);
    expect({ ...patched.config, scheduler: state.config.scheduler }).toEqual(state.config);
    expect(state.config.scheduler).toEqual(DEFAULT_SCHEDULER_CONFIG);
  });

  it('проценты прогресса', () => {
    expect(runProgressPercent({ status: 'running', backtestsSubmitted: 50 }, 100)).toBe(50);
    expect(runProgressPercent({ status: 'running', backtestsSubmitted: 150 }, 100)).toBe(99);
    expect(runProgressPercent({ status: 'completed', backtestsSubmitted: 10 }, 100)).toBe(100);
    expect(runProgressPercent({ status: 'running', backtestsSubmitted: 10 }, 0)).toBe(0);
  });

  it('оценка с недостающими в группе монетами базы', () => {
    const estimate = { perGenerationMax: 50, baseBacktests: 0, totalMax: 400, hoursAtDelay: 1 };
    expect(withGroupMissing(estimate, 3, 1500, 36)).toEqual({
      perGenerationMax: 50,
      baseBacktests: 3,
      totalMax: 403,
      hoursAtDelay: (403 * 36) / 3600,
    });
    expect(withGroupMissing({ ...estimate, totalMax: 1500 }, 3, 1500, 36).totalMax).toBe(1500);
    expect(withGroupMissing(estimate, -2, 1500, 36).baseBacktests).toBe(0);
  });
});
