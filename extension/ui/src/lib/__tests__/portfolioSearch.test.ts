import { describe, expect, it } from 'vitest';
import type { BotStrategy } from '../../api/backtestRunner';
import type { StrategyConditionDto } from '../../api/bots.dtos';
import type {
  FoldMetrics,
  ObjectiveConfig,
  ParamSpec,
  ParamValues,
  PortfolioMetrics,
} from '../../types/portfolioOptimizer';
import { baseValues, diffFromBase, extractParams, paramsHash, prepareValues, roundTo } from '../botParams';
import {
  type CandidateProposal,
  computeObjective,
  createRng,
  isImprovement,
  type ParamStats,
  type ProposeCandidatesInput,
  pickWeighted,
  proposeCandidates,
  selectForPromotion,
  updateParamStats,
} from '../portfolioSearch';

// ═══════════════════════════════════════════════════════════════════
// ФИКСТУРЫ
// ═══════════════════════════════════════════════════════════════════

const makeParam = (overrides: Partial<ParamSpec> & Pick<ParamSpec, 'id'>): ParamSpec => ({
  path: [overrides.id],
  kind: 'entryValue',
  label: overrides.id,
  group: 'Вход',
  base: 50,
  min: 0,
  max: 100,
  step: 2,
  decimals: 0,
  enabled: true,
  ...overrides,
});

/** Пороги условий без побочных правок (сетка/объёмы) — удобно проверять, что именно изменилось. */
const conditionParams = (): ParamSpec[] => [
  makeParam({ id: 'conditions[0].value', base: 50, step: 2, min: 30, max: 70 }),
  makeParam({ id: 'conditions[1].value', base: -80, step: 10, min: -200, max: 40, indicator: 'CCI' }),
  makeParam({ id: 'conditions[2].value', base: -40, step: 5, min: -70, max: -10, indicator: 'CMO' }),
  makeParam({ id: 'conditions[3].value', base: 0.5, step: 0.05, min: 0.2, max: 0.8, decimals: 2 }),
  makeParam({ id: 'conditions[4].value', base: 25, step: 2, min: 13, max: 37, indicator: 'ADX' }),
  makeParam({ id: 'conditions[5].value', base: 100000, step: 25000, min: 25000, max: 400000, enabled: false }),
  makeParam({ id: 'conditions[6].value', base: 70, step: 2, min: 60, max: 80, enabled: false }),
];

const cond = (
  indicator: string,
  interval: string,
  operation: string | null,
  value: number | null,
): StrategyConditionDto => ({
  type: 'INDICATOR',
  indicator,
  interval,
  basic: value === null,
  value,
  operation,
  closed: true,
  reverse: false,
});

/** Компактный реалистичный DTO: условия входа, сетка из 6 ордеров (объёмы = 100%), тейк, стоп, подтяжка. */
const buildStrategy = (): BotStrategy => ({
  id: 1,
  name: 'TEST',
  algorithm: 'LONG',
  exchange: 'BYBIT_FUTURES',
  symbols: ['ADA/USDT'],
  conditions: [
    cond('RSI_LEVELS', 'ONE_MINUTE', null, null),
    cond('CCI', 'ONE_HOUR', 'LESS', 80),
    cond('CMO', 'THIRTY_MINUTES', 'GREATER', -80),
    cond('ATR_PERCENT', 'FIVE_MINUTES', 'GREATER', 0.5),
  ],
  settings: {
    type: 'GRID',
    baseOrder: null,
    orders: [
      { indent: 0.3, volume: 5, conditions: null },
      { indent: 5, volume: 10, conditions: [cond('ADX', 'FIFTEEN_MINUTES', 'GREATER', 40)] },
      { indent: 10, volume: 15, conditions: [cond('MFI', 'FIVE_MINUTES', 'LESS', 19)] },
      { indent: 20, volume: 20, conditions: null },
      { indent: 35, volume: 22, conditions: null },
      { indent: 50, volume: 28, conditions: [cond('CMO', 'THIRTY_MINUTES', 'GREATER', -90)] },
    ],
  },
  profit: {
    type: 'PERCENT',
    currency: 'USDT',
    checkPnl: 0.3,
    conditions: [cond('RSI', 'FIVE_MINUTES', 'GREATER', 70)],
  },
  stopLoss: {
    indent: null,
    termination: true,
    conditionalIndent: 2.5,
    conditionalIndentType: 'AVERAGE',
    conditions: [cond('MACD', 'FOUR_HOURS', null, null)],
  },
  pullUp: 0.2,
  portion: null,
});

const makeInput = (overrides: Partial<ProposeCandidatesInput> = {}): ProposeCandidatesInput => {
  const params = overrides.params ?? conditionParams();
  return {
    params,
    parents: [{ values: baseValues(params), score: 1 }],
    count: 6,
    mutationsMin: 1,
    mutationsMax: 3,
    seen: new Set<string>(),
    paramStats: {},
    rng: createRng(1),
    ...overrides,
  };
};

/** id параметров, значения которых отличаются между двумя наборами. */
const diffIds = (params: ParamSpec[], left: ParamValues, right: ParamValues): string[] =>
  params.filter((param) => left[param.id] !== right[param.id]).map((param) => param.id);

const makeMetrics = (overrides: Partial<PortfolioMetrics> = {}): PortfolioMetrics => ({
  totalProfit: 1000,
  maxConcurrentMae: 400,
  maxDrawdown: 300,
  pnlToRisk: 2.5,
  totalDeals: 500,
  profitableDeals: 480,
  losingDeals: 20,
  openDeals: 2,
  activeMae: 10,
  netPerDay: 1.2,
  maxConcurrentPositions: 60,
  backtests: 10,
  ...overrides,
});

const makeFold = (pnlToRisk: number, deals: number): FoldMetrics => ({
  from: 0,
  to: 1,
  profit: pnlToRisk * 100,
  risk: 100,
  pnlToRisk,
  deals,
});

const objective = (overrides: Partial<ObjectiveConfig> = {}): ObjectiveConfig => ({
  kind: 'pnlToRisk',
  folds: 3,
  maxMaeCap: null,
  minDeals: 0,
  ...overrides,
});

// ═══════════════════════════════════════════════════════════════════
// ГЕНЕРАТОР И ВЗВЕШЕННЫЙ ВЫБОР
// ═══════════════════════════════════════════════════════════════════

describe('createRng', () => {
  it('matches the reference mulberry32 sequence', () => {
    const rng = createRng(1);
    expect(rng()).toBe(0.6270739405881613);
    expect(rng()).toBe(0.002735721180215478);
    expect(rng()).toBe(0.5274470399599522);
    expect(createRng(42)()).toBe(0.6011037519201636);
  });

  it('is deterministic per seed and differs between seeds', () => {
    const draw = (seed: number): number[] => {
      const rng = createRng(seed);
      return Array.from({ length: 20 }, () => rng());
    };
    expect(draw(7)).toEqual(draw(7));
    expect(draw(7)).not.toEqual(draw(8));
  });

  it('produces values in [0, 1) with a roughly uniform mean', () => {
    const rng = createRng(123);
    const values = Array.from({ length: 20000 }, () => rng());
    expect(values.every((value) => value >= 0 && value < 1)).toBe(true);
    const mean = values.reduce((total, value) => total + value, 0) / values.length;
    expect(mean).toBeGreaterThan(0.49);
    expect(mean).toBeLessThan(0.51);
  });

  it('normalizes seeds to uint32', () => {
    expect(createRng(-1)()).toBe(createRng(4294967295)());
    expect(createRng(1.9)()).toBe(createRng(1)());
    expect(createRng(Number.NaN)()).toBe(createRng(0)());
    expect(createRng(Number.POSITIVE_INFINITY)()).toBe(createRng(0)());
  });
});

describe('pickWeighted', () => {
  it('throws on an empty list', () => {
    expect(() => pickWeighted([], [], createRng(1))).toThrow('пустого списка');
  });

  it('maps the roll onto cumulative weights', () => {
    const items = ['a', 'b', 'c'];
    const weights = [1, 2, 1];
    expect(pickWeighted(items, weights, () => 0)).toBe('a');
    expect(pickWeighted(items, weights, () => 0.24)).toBe('a');
    expect(pickWeighted(items, weights, () => 0.25)).toBe('b');
    expect(pickWeighted(items, weights, () => 0.74)).toBe('b');
    expect(pickWeighted(items, weights, () => 0.75)).toBe('c');
    expect(pickWeighted(items, weights, () => 0.999999)).toBe('c');
  });

  it('ignores zero, negative, non-finite and missing weights', () => {
    const items = ['zero', 'neg', 'nan', 'inf', 'ok', 'missing'];
    const weights = [0, -5, Number.NaN, Number.POSITIVE_INFINITY, 3];
    const rng = createRng(5);
    for (let index = 0; index < 200; index += 1) {
      expect(pickWeighted(items, weights, rng)).toBe('ok');
    }
  });

  it('falls back to a uniform choice when all weights are zero', () => {
    const items = ['a', 'b', 'c', 'd'];
    expect(pickWeighted(items, [0, 0, 0, 0], () => 0)).toBe('a');
    expect(pickWeighted(items, [0, 0, 0, 0], () => 0.5)).toBe('c');
    expect(pickWeighted(items, [], () => 0.99)).toBe('d');
  });

  it('survives an out-of-range generator', () => {
    const items = ['a', 'b', 'c'];
    expect(pickWeighted(items, [1, 1, 0], () => 1)).toBe('b');
    expect(pickWeighted(items, [1, 1, 0], () => 7)).toBe('b');
    expect(pickWeighted(items, [1, 1, 0], () => -3)).toBe('a');
    expect(pickWeighted(items, [1, 1, 0], () => Number.NaN)).toBe('a');
    expect(pickWeighted(items, [0, 0, 0], () => 1)).toBe('c');
  });

  it('picks proportionally to the weights', () => {
    const rng = createRng(99);
    const counts = { a: 0, b: 0 };
    for (let index = 0; index < 10000; index += 1) {
      counts[pickWeighted(['a', 'b'] as const, [1, 3], rng)] += 1;
    }
    expect(counts.b / 10000).toBeGreaterThan(0.72);
    expect(counts.b / 10000).toBeLessThan(0.78);
  });
});

// ═══════════════════════════════════════════════════════════════════
// ГЕНЕРАЦИЯ КАНДИДАТОВ
// ═══════════════════════════════════════════════════════════════════

describe('proposeCandidates', () => {
  it('is deterministic for the same seed', () => {
    const first = proposeCandidates(makeInput({ rng: createRng(17) }));
    const second = proposeCandidates(makeInput({ rng: createRng(17) }));
    const other = proposeCandidates(makeInput({ rng: createRng(18) }));
    expect(first).toHaveLength(6);
    expect(second).toEqual(first);
    expect(other.map((proposal) => proposal.hash)).not.toEqual(first.map((proposal) => proposal.hash));
  });

  it('never repeats seen hashes or its own proposals and does not mutate seen', () => {
    const params = conditionParams();
    const base = baseValues(params);
    const previous = proposeCandidates(makeInput({ params, count: 10, rng: createRng(3) }));
    const seen = new Set<string>([paramsHash(params, base), ...previous.map((proposal) => proposal.hash)]);
    const seenBefore = [...seen];

    const proposals = proposeCandidates(makeInput({ params, count: 30, seen, rng: createRng(3) }));

    expect(proposals).toHaveLength(30);
    expect(proposals.some((proposal) => seen.has(proposal.hash))).toBe(false);
    expect(new Set(proposals.map((proposal) => proposal.hash)).size).toBe(proposals.length);
    expect([...seen]).toEqual(seenBefore);
  });

  it('changes only enabled params and keeps every param id in values', () => {
    const params = conditionParams();
    const parent = baseValues(params);
    const enabledIds = new Set(params.filter((param) => param.enabled).map((param) => param.id));

    const proposals = proposeCandidates(makeInput({ params, count: 40, mutationsMax: 5, rng: createRng(11) }));

    expect(proposals.length).toBe(40);
    proposals.forEach((proposal) => {
      expect(Object.keys(proposal.values).sort()).toEqual(params.map((param) => param.id).sort());
      const changedIds = diffIds(params, proposal.values, parent);
      expect(changedIds.length).toBeGreaterThan(0);
      expect(changedIds.every((id) => enabledIds.has(id))).toBe(true);
      // порядок `mutated` — порядок выбора параметров, поэтому сравниваем множества
      expect(new Set(proposal.mutated)).toEqual(new Set(changedIds));
    });
  });

  it('reports hash and changed consistently with botParams', () => {
    const params = conditionParams();
    const proposals = proposeCandidates(makeInput({ params, rng: createRng(21) }));
    proposals.forEach((proposal) => {
      expect(proposal.values).toEqual(prepareValues(params, proposal.values));
      expect(proposal.hash).toBe(paramsHash(params, proposal.values));
      expect(proposal.changed).toEqual(diffFromBase(params, proposal.values));
    });
  });

  it('moves a param by one or two steps', () => {
    const params = [
      makeParam({ id: 'a', base: 50, step: 2, min: 0, max: 100 }),
      makeParam({ id: 'b', base: 0.5, step: 0.05, min: 0, max: 1, decimals: 2 }),
    ];
    const proposals = proposeCandidates(
      makeInput({ params, count: 8, mutationsMin: 1, mutationsMax: 1, rng: createRng(4) }),
    );
    expect(proposals).toHaveLength(8);
    const distances = new Set<number>();
    proposals.forEach((proposal) => {
      expect(proposal.mutated).toHaveLength(1);
      const param = params.find((candidate) => candidate.id === proposal.mutated[0]);
      if (!param) {
        throw new Error('Параметр не найден');
      }
      const steps = roundTo(Math.abs(proposal.values[param.id] - param.base) / param.step, 6);
      distances.add(steps);
      expect([1, 2]).toContain(steps);
    });
    // 2 параметра × 2 знака × 2 длины шага = 8 соседей — все должны найтись.
    expect(distances).toEqual(new Set([1, 2]));
  });

  it('respects the mutation count range and clamps it to the enabled params', () => {
    const params = conditionParams();
    const parent = baseValues(params);
    const ranged = proposeCandidates(
      makeInput({ params, count: 30, mutationsMin: 2, mutationsMax: 3, rng: createRng(8) }),
    );
    const counts = new Set(ranged.map((proposal) => diffIds(params, proposal.values, parent).length));
    expect(counts).toEqual(new Set([2, 3]));

    const twoParams = [makeParam({ id: 'x', base: 50 }), makeParam({ id: 'y', base: 50 })];
    const clamped = proposeCandidates(
      makeInput({ params: twoParams, count: 10, mutationsMin: 5, mutationsMax: 10, rng: createRng(8) }),
    );
    expect(clamped.length).toBeGreaterThan(0);
    clamped.forEach((proposal) => {
      expect(proposal.mutated).toHaveLength(2);
    });

    const swapped = proposeCandidates(
      makeInput({ params, count: 10, mutationsMin: 1, mutationsMax: 1, rng: createRng(9) }),
    );
    swapped.forEach((proposal) => {
      expect(diffIds(params, proposal.values, parent)).toHaveLength(1);
    });
  });

  it('tries the other direction when a move is blocked by a bound', () => {
    const params = [makeParam({ id: 'top', base: 100, min: 80, max: 100, step: 2 })];
    const proposals = proposeCandidates(
      makeInput({ params, count: 2, mutationsMin: 1, mutationsMax: 1, rng: createRng(2) }),
    );
    expect(proposals.map((proposal) => proposal.values.top).sort()).toEqual([96, 98]);
  });

  it('picks the best parent most of the time and only among the top three', () => {
    const params = conditionParams();
    const base = baseValues(params);
    const parents = [0, 1, 2, 3, 4].map((offset) => ({
      values: { ...base, 'conditions[0].value': 40 + offset * 4 },
      score: 10 - offset,
    }));
    const counts = [0, 0, 0, 0, 0];
    for (let seed = 1; seed <= 300; seed += 1) {
      const [proposal] = proposeCandidates(makeInput({ params, parents, count: 1, rng: createRng(seed) }));
      counts[proposal.parentIndex] += 1;
    }
    expect(counts[3]).toBe(0);
    expect(counts[4]).toBe(0);
    // Ожидание: 0.6 + 0.4 / 3 ≈ 0.73 для лучшего родителя, ≈ 0.13 для 2-го и 3-го.
    expect(counts[0] / 300).toBeGreaterThan(0.65);
    expect(counts[0] / 300).toBeLessThan(0.81);
    expect(counts[1]).toBeGreaterThan(15);
    expect(counts[2]).toBeGreaterThan(15);
  });

  it('builds proposals from the chosen parent values', () => {
    const params = conditionParams();
    const base = baseValues(params);
    const secondParent: ParamValues = { ...base, 'conditions[1].value': -120, 'conditions[2].value': -60 };
    const parents = [
      { values: base, score: 3 },
      { values: secondParent, score: 2 },
    ];
    const proposals = proposeCandidates(makeInput({ params, parents, count: 40, rng: createRng(31) }));
    const fromSecond = proposals.filter((proposal) => proposal.parentIndex === 1);
    expect(fromSecond.length).toBeGreaterThan(0);
    fromSecond.forEach((proposal) => {
      const untouched = params.filter((param) => !proposal.mutated.includes(param.id));
      untouched.forEach((param) => {
        expect(proposal.values[param.id]).toBe(secondParent[param.id]);
      });
      expect(proposal.changed).toEqual(diffFromBase(params, proposal.values));
      expect(proposal.changed).toEqual(
        expect.arrayContaining(
          ['conditions[1].value', 'conditions[2].value'].filter((id) => proposal.values[id] !== base[id]),
        ),
      );
    });
  });

  it('preserves the total volume and monotonic indents on a real strategy', () => {
    const params = extractParams(buildStrategy());
    const volumes = params.filter((param) => param.kind === 'orderVolume');
    const indents = params.filter((param) => param.kind === 'orderIndent');
    expect(volumes).toHaveLength(6);
    const baseVolume = roundTo(
      volumes.reduce((total, param) => total + param.base, 0),
      2,
    );

    const proposals = proposeCandidates(
      makeInput({ params, count: 40, mutationsMin: 2, mutationsMax: 4, rng: createRng(77) }),
    );

    expect(proposals).toHaveLength(40);
    expect(proposals.some((proposal) => proposal.changed.some((id) => id.endsWith('.volume')))).toBe(true);
    proposals.forEach((proposal) => {
      const total = roundTo(
        volumes.reduce((sum, param) => sum + proposal.values[param.id], 0),
        2,
      );
      expect(total).toBe(baseVolume);
      for (let index = 1; index < indents.length; index += 1) {
        expect(proposal.values[indents[index].id]).toBeGreaterThan(proposal.values[indents[index - 1].id]);
      }
      params.forEach((param) => {
        if (param.kind !== 'orderVolume') {
          expect(proposal.values[param.id]).toBeGreaterThanOrEqual(param.min);
          expect(proposal.values[param.id]).toBeLessThanOrEqual(param.max);
        }
      });
    });
  });

  it('prefers params whose changes improved the score', () => {
    const params = [makeParam({ id: 'lucky', base: 50 }), makeParam({ id: 'unlucky', base: 50 })];
    const paramStats: ParamStats = { lucky: { tries: 20, wins: 20 }, unlucky: { tries: 20, wins: 0 } };
    let lucky = 0;
    for (let seed = 1; seed <= 200; seed += 1) {
      const [proposal] = proposeCandidates(
        makeInput({ params, paramStats, count: 1, mutationsMin: 1, mutationsMax: 1, rng: createRng(seed) }),
      );
      if (proposal.mutated[0] === 'lucky') {
        lucky += 1;
      }
    }
    // Вес lucky = 21/22, unlucky = 1/22 → ожидаемая доля ≈ 0.95.
    expect(lucky).toBeGreaterThan(170);
    expect(lucky).toBeLessThan(200);
  });

  it('stops when the neighbourhood is exhausted or attempts run out', () => {
    const params = [makeParam({ id: 'tiny', base: 1, min: 0, max: 2, step: 1 })];
    const seen = new Set([paramsHash(params, baseValues(params))]);
    const exhausted = proposeCandidates(makeInput({ params, seen, count: 10, rng: createRng(1) }));
    expect(exhausted.map((proposal) => proposal.values.tiny).sort()).toEqual([0, 2]);

    expect(proposeCandidates(makeInput({ maxAttempts: 0 }))).toEqual([]);
    expect(proposeCandidates(makeInput({ maxAttempts: 1 })).length).toBeLessThanOrEqual(1);
    expect(proposeCandidates(makeInput({ maxAttempts: Number.POSITIVE_INFINITY }))).toHaveLength(6);
  });

  it('skips proposals that collapse back onto the parent', () => {
    // Единственный включённый объём нормализуется обратно к 100 − выключенный объём.
    const params = [
      makeParam({ id: 'v0', kind: 'orderVolume', base: 50, min: 25, max: 100, step: 5, decimals: 2, orderIndex: 0 }),
      makeParam({
        id: 'v1',
        kind: 'orderVolume',
        base: 50,
        min: 25,
        max: 100,
        step: 5,
        decimals: 2,
        orderIndex: 1,
        enabled: false,
      }),
    ];
    expect(proposeCandidates(makeInput({ params, count: 3, rng: createRng(1) }))).toEqual([]);
  });

  it('returns nothing for degenerate inputs', () => {
    const params = conditionParams();
    expect(proposeCandidates(makeInput({ count: 0 }))).toEqual([]);
    expect(proposeCandidates(makeInput({ count: Number.NaN }))).toEqual([]);
    expect(proposeCandidates(makeInput({ parents: [] }))).toEqual([]);
    expect(proposeCandidates(makeInput({ params: params.map((param) => ({ ...param, enabled: false })) }))).toEqual([]);
    const frozen = params.map((param) => ({ ...param, min: param.base, max: param.base }));
    expect(proposeCandidates(makeInput({ params: frozen }))).toEqual([]);
    const noStep = params.map((param) => ({ ...param, step: 0 }));
    expect(proposeCandidates(makeInput({ params: noStep }))).toEqual([]);
  });

  it('treats missing parent values as base values', () => {
    const params = conditionParams();
    const proposals: CandidateProposal[] = proposeCandidates(
      makeInput({ params, parents: [{ values: {}, score: 0 }], count: 5, rng: createRng(6) }),
    );
    expect(proposals).toHaveLength(5);
    proposals.forEach((proposal) => {
      expect(proposal.changed.length).toBeGreaterThan(0);
      expect(new Set(proposal.changed)).toEqual(new Set(proposal.mutated));
    });
  });
});

// ═══════════════════════════════════════════════════════════════════
// ЦЕЛЕВАЯ ФУНКЦИЯ
// ═══════════════════════════════════════════════════════════════════

describe('computeObjective', () => {
  it('returns null without metrics and 0 without deals', () => {
    expect(computeObjective(null, [], objective())).toBeNull();
    expect(computeObjective(makeMetrics({ totalDeals: 0, pnlToRisk: 5 }), [], objective())).toBe(0);
    expect(computeObjective(makeMetrics({ totalDeals: 0 }), [], objective({ kind: 'profitCapped' }))).toBe(0);
  });

  it('uses the group P&L to risk for pnlToRisk', () => {
    expect(computeObjective(makeMetrics({ pnlToRisk: 2.5 }), [makeFold(-1, 10)], objective())).toBe(2.5);
  });

  it('blends the full period with the worst fold for robust', () => {
    const metrics = makeMetrics({ pnlToRisk: 3 });
    const folds = [makeFold(4, 100), makeFold(1, 50), makeFold(-5, 0), makeFold(2, 80)];
    expect(computeObjective(metrics, folds, objective({ kind: 'robust' }))).toBe(2);
    expect(computeObjective(metrics, [makeFold(-1, 5)], objective({ kind: 'robust' }))).toBe(1);
  });

  it('falls back to the full period when no fold has deals', () => {
    const metrics = makeMetrics({ pnlToRisk: 3 });
    expect(computeObjective(metrics, [], objective({ kind: 'robust' }))).toBe(3);
    expect(computeObjective(metrics, [makeFold(-2, 0), makeFold(-3, 0)], objective({ kind: 'robust' }))).toBe(3);
  });

  it('penalizes profit above the MAE cap quadratically', () => {
    const cfg = (maxMaeCap: number | null): ObjectiveConfig => objective({ kind: 'profitCapped', maxMaeCap });
    expect(computeObjective(makeMetrics({ totalProfit: 1000, maxConcurrentMae: 200 }), [], cfg(null))).toBe(1000);
    expect(computeObjective(makeMetrics({ totalProfit: 1000, maxConcurrentMae: 200 }), [], cfg(200))).toBe(1000);
    expect(computeObjective(makeMetrics({ totalProfit: 1000, maxConcurrentMae: 200 }), [], cfg(100))).toBe(250);
    expect(computeObjective(makeMetrics({ totalProfit: 1000, maxConcurrentMae: 200 }), [], cfg(0))).toBe(0);
  });

  it('never improves a loss by exceeding the MAE cap', () => {
    const cfg = objective({ kind: 'profitCapped', maxMaeCap: 100 });
    expect(computeObjective(makeMetrics({ totalProfit: -400, maxConcurrentMae: 200 }), [], cfg)).toBe(-400);
  });

  it('ignores an invalid MAE cap', () => {
    const metrics = makeMetrics({ totalProfit: 1000, maxConcurrentMae: 200 });
    expect(computeObjective(metrics, [], objective({ kind: 'profitCapped', maxMaeCap: -5 }))).toBe(1000);
    expect(computeObjective(metrics, [], objective({ kind: 'profitCapped', maxMaeCap: Number.NaN }))).toBe(1000);
  });

  it('applies the minimum deals penalty only to positive scores', () => {
    const metrics = makeMetrics({ totalDeals: 100, pnlToRisk: 2 });
    expect(computeObjective(metrics, [], objective({ minDeals: 200 }))).toBe(1);
    expect(computeObjective(metrics, [], objective({ minDeals: 100 }))).toBe(2);
    expect(computeObjective(metrics, [], objective({ minDeals: 0 }))).toBe(2);
    expect(computeObjective(makeMetrics({ totalDeals: 100, pnlToRisk: -2 }), [], objective({ minDeals: 200 }))).toBe(
      -2,
    );
    const capped = objective({ kind: 'profitCapped', maxMaeCap: 100, minDeals: 400 });
    expect(
      computeObjective(makeMetrics({ totalDeals: 100, totalProfit: 1000, maxConcurrentMae: 200 }), [], capped),
    ).toBe(62.5);
  });

  it('returns null for a non-finite score and normalizes negative zero', () => {
    expect(computeObjective(makeMetrics({ pnlToRisk: Number.NaN }), [], objective())).toBeNull();
    expect(computeObjective(makeMetrics({ pnlToRisk: Number.POSITIVE_INFINITY }), [], objective())).toBeNull();
    expect(Object.is(computeObjective(makeMetrics({ pnlToRisk: -0 }), [], objective()), 0)).toBe(true);
  });
});

// ═══════════════════════════════════════════════════════════════════
// ОТБОР И СТАТИСТИКА
// ═══════════════════════════════════════════════════════════════════

interface ScreenRow {
  id: string;
  screenScore: number | null;
}

const rows = (...scores: Array<number | null>): ScreenRow[] =>
  scores.map((screenScore, index) => ({ id: `c${index}`, screenScore }));

const ids = (selected: ScreenRow[]): string[] => selected.map((row) => row.id);

describe('selectForPromotion', () => {
  it('takes the top scores when the best screen score is unknown', () => {
    expect(ids(selectForPromotion(rows(1, null, 3, 2, Number.NaN), null, 2, 0.05))).toEqual(['c2', 'c3']);
    expect(ids(selectForPromotion(rows(1, 3), Number.NaN, 5, 0.05))).toEqual(['c1', 'c0']);
  });

  it('keeps only candidates within the tolerance of the best score', () => {
    expect(ids(selectForPromotion(rows(1.89, 2.1, 1.9, 1.95), 2, 10, 0.05))).toEqual(['c1', 'c3', 'c2']);
    expect(ids(selectForPromotion(rows(1.89, 2.1, 1.9, 1.95), 2, 2, 0.05))).toEqual(['c1', 'c3']);
    expect(ids(selectForPromotion(rows(-0.1, 0, 0.5), 0, 5, 0.5))).toEqual(['c2', 'c1']);
  });

  it('uses the absolute value of a negative best score', () => {
    expect(ids(selectForPromotion(rows(-12, -10.5, -11), -10, 5, 0.1))).toEqual(['c1', 'c2']);
  });

  it('is robust to binary noise at the threshold', () => {
    // 1.1 − 1.1 × 0.1 = 0.9900000000000001 в double.
    expect(ids(selectForPromotion(rows(0.99, 0.98), 1.1, 5, 0.1))).toEqual(['c0']);
  });

  it('treats an invalid tolerance as zero', () => {
    expect(ids(selectForPromotion(rows(1.9, 2, 2.2), 2, 5, -1))).toEqual(['c2', 'c1']);
    expect(ids(selectForPromotion(rows(1.9, 2, 2.2), 2, 5, Number.NaN))).toEqual(['c2', 'c1']);
  });

  it('handles the promote count edge cases', () => {
    expect(selectForPromotion(rows(1, 2), null, 0, 0.05)).toEqual([]);
    expect(selectForPromotion(rows(1, 2), null, -3, 0.05)).toEqual([]);
    expect(ids(selectForPromotion(rows(1, 2, 3), null, 2.9, 0.05))).toEqual(['c2', 'c1']);
    expect(selectForPromotion(rows(null, null), null, 3, 0.05)).toEqual([]);
    expect(selectForPromotion([], 1, 3, 0.05)).toEqual([]);
  });

  it('keeps ties stable, returns the same objects and does not mutate the input', () => {
    const input = rows(2, 1, 2, 2);
    const snapshot = input.map((row) => ({ ...row }));
    const selected = selectForPromotion(input, null, 3, 0);
    expect(ids(selected)).toEqual(['c0', 'c2', 'c3']);
    expect(selected[0]).toBe(input[0]);
    expect(input).toEqual(snapshot);
  });
});

describe('updateParamStats', () => {
  it('counts tries and wins in a new object', () => {
    const stats: ParamStats = { a: { tries: 2, wins: 1 }, untouched: { tries: 5, wins: 3 } };
    const snapshot = structuredClone(stats);

    const improved = updateParamStats(stats, ['a', 'b'], true);
    expect(improved).toEqual({ a: { tries: 3, wins: 2 }, b: { tries: 1, wins: 1 }, untouched: { tries: 5, wins: 3 } });
    expect(stats).toEqual(snapshot);
    expect(improved).not.toBe(stats);
    expect(improved.untouched).not.toBe(stats.untouched);

    const failed = updateParamStats(improved, ['a'], false);
    expect(failed.a).toEqual({ tries: 4, wins: 2 });
  });

  it('counts each param once per candidate', () => {
    expect(updateParamStats({}, ['a', 'a', 'b'], true)).toEqual({ a: { tries: 1, wins: 1 }, b: { tries: 1, wins: 1 } });
    expect(updateParamStats({ a: { tries: 1, wins: 0 } }, [], true)).toEqual({ a: { tries: 1, wins: 0 } });
  });

  it('repairs corrupted counters of updated params', () => {
    const stats: ParamStats = { a: { tries: Number.NaN, wins: -2 } };
    expect(updateParamStats(stats, ['a'], false)).toEqual({ a: { tries: 1, wins: 0 } });
  });
});

describe('isImprovement', () => {
  it('requires a known score', () => {
    expect(isImprovement(null, null, 0.01)).toBe(false);
    expect(isImprovement(Number.NaN, 1, 0.01)).toBe(false);
  });

  it('accepts any score when there is no best yet', () => {
    expect(isImprovement(-5, null, 0.01)).toBe(true);
    expect(isImprovement(0, Number.NaN, 0.01)).toBe(true);
  });

  it('requires beating the best by the relative margin', () => {
    expect(isImprovement(1.01, 1, 0.01)).toBe(false);
    expect(isImprovement(1.0101, 1, 0.01)).toBe(true);
    expect(isImprovement(1, 1, 0)).toBe(false);
    expect(isImprovement(1.0000001, 1, 0)).toBe(true);
    expect(isImprovement(0.1, 0, 0.5)).toBe(true);
    expect(isImprovement(0, 0, 0.5)).toBe(false);
    expect(isImprovement(1.0000001, 1, -1)).toBe(true);
  });

  it('uses the absolute value of a negative best score', () => {
    expect(isImprovement(-0.95, -1, 0.1)).toBe(false);
    expect(isImprovement(-0.89, -1, 0.1)).toBe(true);
  });
});
