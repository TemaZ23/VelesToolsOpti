import { describe, expect, it } from 'vitest';
import {
  psApplyMove,
  psBuildMoves,
  psCertify,
  psCertifyCost,
  psChoiceMoves,
  psCoinGroups,
  psCreateRng,
  psFoldMetrics,
  psFoldScore,
  psFoldScores,
  psGridSize,
  psGridStride,
  psGridValues,
  psObjective,
  psPatternPoint,
  psPickFinalists,
  psPickScreeningSymbols,
  psRescale,
  psRunGrid,
  psRunSearch,
  psScaleEntries,
  psSliceEntries,
  psTimeFolds,
} from '../src/patternSearch.mjs';

const roundStep = (value, step) => Math.round(value / step) * step;

/** A space shaped like a real 8-order DCA bot. */
const makeSpace = () => {
  const dimensions = [];
  const indents = [0, 1.2, 2.5, 4, 6, 8.5, 11, 14];
  const volumes = [5, 5, 7, 9, 12, 15, 20, 27];
  indents.forEach((indent, index) => {
    if (index === 0) {
      return;
    }
    dimensions.push({
      id: `grid:indent:${index}`,
      kind: 'orderIndent',
      group: 'grid',
      label: `DCA #${index}: отступ, %`,
      min: indent * 0.5,
      max: indent * 1.5,
      step: 0.05,
      baseValue: indent,
      orderIndex: index,
    });
  });
  volumes.forEach((volume, index) => {
    dimensions.push({
      id: `grid:volume:${index}`,
      kind: 'orderVolume',
      group: 'grid',
      label: `Ордер ${index}: объём, %`,
      min: volume * 0.5,
      max: volume * 1.5,
      step: 0.5,
      baseValue: volume,
      orderIndex: index,
    });
  });
  dimensions.push({ id: 'tp', kind: 'takeProfit', group: 'exit', label: 'Тейк-профит, %', min: 0.25, max: 1, step: 0.05, baseValue: 0.5 });
  dimensions.push({ id: 'pullUp', kind: 'pullUp', group: 'grid', label: 'Подтяжка сетки, %', min: 0.1, max: 1, step: 0.05, baseValue: 0.4 });
  dimensions.push({ id: 'cond:v:entry:0:RSI', kind: 'conditionValue', group: 'filters', label: 'Вход: RSI 15м < 30', min: 10, max: 50, step: 1, baseValue: 30 });
  dimensions.push({ id: 'cond:v:order:CMO', kind: 'conditionValue', group: 'filters', label: 'Сетка: CMO < -50', min: -90, max: -10, step: 1, baseValue: -50 });
  dimensions.push({
    id: 'struct:swap:entry:0',
    kind: 'conditionSwap',
    group: 'structure',
    label: 'Вход: RSI — заменить/выкл.',
    shortLabel: 'вход RSI',
    min: 0,
    max: 4,
    step: 1,
    baseValue: 0,
    integer: true,
    choices: [
      { kind: 'keep', label: 'как есть' },
      { kind: 'off', label: 'выключить' },
      { kind: 'set', label: 'CCI < -100' },
      { kind: 'set', label: 'MFI < 20' },
      { kind: 'set', label: 'CMO < -50' },
    ],
  });
  const volumeTotal = volumes.reduce((sum, value) => sum + value, 0);
  const space = { dimensions, volumeTotal };
  const repair = (genome) => {
    const out = {};
    for (const dim of dimensions) {
      const raw = typeof genome[dim.id] === 'number' ? genome[dim.id] : dim.baseValue;
      const value = Math.min(dim.max, Math.max(dim.min, raw));
      out[dim.id] = dim.integer ? Math.round(value) : Number(roundStep(value, dim.step).toFixed(4));
    }
    const volumeDims = dimensions.filter((dim) => dim.kind === 'orderVolume');
    const sum = volumeDims.reduce((total, dim) => total + out[dim.id], 0);
    for (const dim of volumeDims) {
      out[dim.id] = Number(roundStep((out[dim.id] * volumeTotal) / sum, 0.01).toFixed(2));
    }
    return out;
  };
  const baseGenome = Object.fromEntries(dimensions.map((dim) => [dim.id, dim.baseValue]));
  return { space, repair, baseGenome };
};

const keyOf = (genome) =>
  Object.keys(genome)
    .sort()
    .map((key) => `${key}=${genome[key]}`)
    .join('|');

/**
 * Synthetic objective: TP should be ~0.65, grid ~15% wider, pull-up lower, RSI looser,
 * swapping the entry filter to MFI helps a bit; everything else is a bit worse when moved.
 */
const makeObjective = (space) => {
  const dims = new Map(space.dimensions.map((dim) => [dim.id, dim]));
  const norm = (genome, id) => {
    const dim = dims.get(id);
    return (genome[id] - dim.baseValue) / (dim.max - dim.min);
  };
  return (genome) => {
    let score = 1;
    score -= 2.5 * (norm(genome, 'tp') - 0.2) ** 2 - 2.5 * 0.2 ** 2;
    score -= 2 * (norm(genome, 'pullUp') + 0.1) ** 2 - 2 * 0.1 ** 2;
    score -= 1.5 * (norm(genome, 'cond:v:entry:0:RSI') - 0.12) ** 2 - 1.5 * 0.12 ** 2;
    let depth = 0;
    for (let index = 1; index <= 7; index++) {
      depth += norm(genome, `grid:indent:${index}`);
    }
    depth /= 7;
    score -= 3 * (depth - 0.15) ** 2 - 3 * 0.15 ** 2;
    for (let index = 1; index <= 7; index++) {
      score -= 0.3 * (norm(genome, `grid:indent:${index}`) - depth) ** 2;
    }
    for (let index = 0; index < 8; index++) {
      score -= 0.5 * norm(genome, `grid:volume:${index}`) ** 2;
    }
    score -= 0.8 * norm(genome, 'cond:v:order:CMO') ** 2;
    if (genome['struct:swap:entry:0'] === 3) {
      score += 0.02;
    } else if (genome['struct:swap:entry:0'] !== 0) {
      score -= 0.05;
    }
    return Number(score.toFixed(6));
  };
};

describe('psRescale', () => {
  it('reduces to min(profit ratio, drawdown ratio) without deposit change', () => {
    const result = psRescale({ profit: 120, risk: 110 }, { profit: 100, risk: 100 }, 0);
    expect(result.scale).toBe(1);
    expect(result.score).toBeCloseTo(100 / 110, 4);
  });

  it('balances profit and drawdown by the deposit multiplier', () => {
    // 20% more profit for 10% more drawdown: better profit/drawdown ratio.
    const result = psRescale({ profit: 120, risk: 110 }, { profit: 100, risk: 100 }, 0.3);
    expect(result.scale).toBeLessThan(1);
    expect(result.score).toBeGreaterThan(1);
    expect(result.profitGain).toBeGreaterThan(1);
    expect(result.riskGain).toBeGreaterThan(1);
  });

  it('keeps the original at exactly 1 and rejects a worse ratio', () => {
    expect(psRescale({ profit: 100, risk: 100 }, { profit: 100, risk: 100 }, 0.3).score).toBe(1);
    expect(psRescale({ profit: 100, risk: 120 }, { profit: 100, risk: 100 }, 0.3).score).toBeLessThan(1);
  });

  it('limits the deposit change', () => {
    // Half the profit, a quarter of the drawdown: would need ×1.41, only ×1.3 allowed.
    const result = psRescale({ profit: 50, risk: 25 }, { profit: 100, risk: 100 }, 0.3);
    expect(result.scale).toBe(1.3);
    expect(result.profitGain).toBeCloseTo(0.65, 4);
    expect(result.score).toBeCloseTo(0.65, 4);
  });

  it('has no reference for an unprofitable original', () => {
    expect(psRescale({ profit: 10, risk: 5 }, { profit: -1, risk: 10 }, 0.3)).toBeNull();
  });
});

describe('folds', () => {
  it('splits three years into six contiguous half-years', () => {
    const folds = psTimeFolds('2023-10-11T00:00:00.000Z', '2026-10-10T23:59:59.999Z');
    expect(folds).toHaveLength(6);
    for (let i = 1; i < folds.length; i++) {
      expect(folds[i].from).toBe(folds[i - 1].to);
    }
    expect(folds[0].from).toBe(Date.parse('2023-10-11T00:00:00.000Z'));
    expect(folds[5].to).toBeGreaterThan(Date.parse('2026-10-10T23:59:59.999Z'));
  });

  it('has no time folds for short periods', () => {
    expect(psTimeFolds('2026-04-01T00:00:00.000Z', '2026-10-01T00:00:00.000Z')).toEqual([]);
  });

  it('builds balanced coin groups', () => {
    const symbols = Array.from({ length: 12 }, (_, i) => `C${i}/USDT`);
    const rank = new Map(symbols.map((symbol, i) => [symbol, 100 - i]));
    const groups = psCoinGroups(symbols, rank);
    expect(groups).toHaveLength(3);
    expect(groups.map((group) => group.symbols.length)).toEqual([4, 4, 4]);
    // snake order: the strongest coin and the 6th strongest are in the same group
    expect(groups[0].symbols).toContain('C0/USDT');
    expect(groups[0].symbols).toContain('C5/USDT');
    expect(psCoinGroups(['A/USDT', 'B/USDT', 'C/USDT'])).toEqual([]);
  });

  it('computes fold metrics on sliced and filtered entries', () => {
    const t0 = Date.parse('2024-01-01T00:00:00.000Z');
    const day = 86400000;
    const entries = [
      { symbol: 'A/USDT', from: '2024-01-01T00:00:00.000Z', to: '2024-12-31T00:00:00.000Z', depositAmount: 10, deals: [
        { s: t0 + day, e: t0 + 2 * day, n: 5, m: 2, st: 'F' },
        { s: t0 + 200 * day, e: t0 + 201 * day, n: 7, m: 3, st: 'F' },
      ] },
      { symbol: 'B/USDT', from: '2024-01-01T00:00:00.000Z', to: '2024-12-31T00:00:00.000Z', depositAmount: 10, deals: [
        { s: t0 + 10 * day, e: t0 + 12 * day, n: -1, m: 4, st: 'F' },
      ] },
    ];
    const metricsOf = (subset) => ({
      profit: subset.reduce((sum, entry) => sum + entry.deals.reduce((s, deal) => s + deal.n, 0), 0),
      risk: subset.reduce((sum, entry) => sum + entry.deals.reduce((s, deal) => Math.max(s, deal.m), 0), 0),
    });
    const folds = [
      { key: 't0', kind: 'time', from: t0, to: t0 + 100 * day },
      { key: 't1', kind: 'time', from: t0 + 100 * day, to: t0 + 366 * day },
      { key: 'c0', kind: 'coins', symbols: ['B/USDT'] },
    ];
    expect(psFoldMetrics(entries, folds, metricsOf)).toEqual([
      { key: 't0', profit: 4, risk: 6 },
      { key: 't1', profit: 7, risk: 3 },
      { key: 'c0', profit: -1, risk: 4 },
    ]);
    expect(psSliceEntries(entries, t0 + 100 * day, t0 + 366 * day)[1].deals).toEqual([]);
    const scaled = psScaleEntries(entries, 2);
    expect(scaled[0].deals[0].n).toBe(10);
    expect(scaled[0].deals[0].m).toBe(4);
    expect(scaled[0].depositAmount).toBe(20);
    expect(psScaleEntries(entries, 1)).toBe(entries);
  });

  it('scores folds against the original and caps the objective by the median', () => {
    expect(psFoldScore({ profit: 110, risk: 90 }, { profit: 100, risk: 100 }, 1, 5000)).toBeCloseTo(1.1, 4);
    // the original lost money in this fold: compare by difference relative to its drawdown
    expect(psFoldScore({ profit: -5, risk: 100 }, { profit: -10, risk: 100 }, 1, 5000)).toBeCloseTo(1, 4);
    expect(psFoldScore({ profit: 0, risk: 50 }, { profit: -10, risk: 100 }, 1, 5000)).toBeCloseTo(1.1, 4);
    const scores = psFoldScores(
      [
        { key: 'a', profit: 110, risk: 100 },
        { key: 'b', profit: 90, risk: 100 },
        { key: 'x', profit: 1, risk: 1 },
      ],
      [
        { key: 'a', profit: 100, risk: 100 },
        { key: 'b', profit: 100, risk: 100 },
      ],
      1,
      5000,
    );
    expect(scores.map((item) => item.key)).toEqual(['a', 'b']);
    expect(psObjective(1.2, [1.1, 0.9, 1.05])).toBeCloseTo(1.05, 4);
    expect(psObjective(1.02, [1.1, 1.3, 1.05])).toBeCloseTo(1.02, 4);
    expect(psObjective(1.2, [0.9])).toBe(1.2);
  });
});

describe('moves', () => {
  it('builds single and macro moves in priority order', () => {
    const { space } = makeSpace();
    const moves = psBuildMoves(space);
    const ids = moves.map((move) => move.id);
    expect(ids[0]).toBe('tp:+');
    expect(ids).toContain('macro:depth:+');
    expect(ids).toContain('macro:shape:-');
    expect(ids).toContain('macro:martingale:+');
    expect(ids.indexOf('macro:depth:+')).toBeLessThan(ids.indexOf('grid:indent:1:+'));
    expect(ids.indexOf('cond:v:entry:0:RSI:+')).toBeLessThan(ids.indexOf('grid:volume:3:+'));
    expect(moves.every((move) => move.kind === 'step')).toBe(true);
    const depth = moves.find((move) => move.id === 'macro:depth:+');
    expect(depth.deltas).toHaveLength(7);
  });

  it('applies moves within bounds and reports no-ops', () => {
    const { space, repair, baseGenome } = makeSpace();
    const tpUp = psBuildMoves(space).find((move) => move.id === 'tp:+');
    const next = psApplyMove(space, baseGenome, tpUp, 0.125, repair);
    expect(next.tp).toBeCloseTo(0.6, 6);
    const atMax = { ...baseGenome, tp: 1 };
    expect(psApplyMove(space, atMax, tpUp, 0.125, repair)).toBeNull();
    // tiny steps still move by one grid step
    expect(psApplyMove(space, baseGenome, tpUp, 0.001, repair).tp).toBeCloseTo(0.55, 6);
  });

  it('samples filter swaps deterministically and always offers revert/off', () => {
    const { space, baseGenome } = makeSpace();
    const a = psChoiceMoves(space, baseGenome, psCreateRng(7), 2);
    const b = psChoiceMoves(space, baseGenome, psCreateRng(7), 2);
    expect(a.map((move) => move.id)).toEqual(b.map((move) => move.id));
    expect(a.map((move) => move.choice)).toContain(1);
    expect(a).toHaveLength(3);
    const swapped = psChoiceMoves(space, { ...baseGenome, 'struct:swap:entry:0': 3 }, psCreateRng(1), 0);
    expect(swapped.map((move) => move.choice)).toEqual([0, 1]);
  });

  it('repeats the round move as a pattern move', () => {
    const { space, repair, baseGenome } = makeSpace();
    const to = repair({ ...baseGenome, tp: 0.6 });
    const pattern = psPatternPoint(space, baseGenome, to, repair);
    expect(pattern.tp).toBeCloseTo(0.7, 6);
  });
});

describe('psPickScreeningSymbols', () => {
  it('returns distinct coins in the original order, including the worst-moment contributor', () => {
    const t0 = Date.parse('2024-01-01T00:00:00.000Z');
    const hour = 3600000;
    const entries = Array.from({ length: 20 }, (_, i) => ({
      symbol: `C${i}/USDT`,
      from: '2024-01-01T00:00:00.000Z',
      to: '2025-01-01T00:00:00.000Z',
      depositAmount: 10,
      deals: Array.from({ length: 3 + (i % 5) }, (__, j) => ({
        s: t0 + (j * 50 + i) * hour,
        e: t0 + (j * 50 + i + 10) * hour,
        n: i % 4 === 0 ? -1 : i,
        m: i === 13 ? 500 : 1 + j,
        st: 'F',
      })),
    }));
    const picked = psPickScreeningSymbols(entries, 9);
    expect(picked).toHaveLength(9);
    expect(new Set(picked).size).toBe(9);
    expect(picked).toContain('C13/USDT');
    const order = picked.map((symbol) => Number(symbol.slice(1, symbol.indexOf('/'))));
    expect([...order].sort((a, b) => a - b)).toEqual(order);
    // losing coins are represented too
    expect(picked.some((symbol) => Number(symbol.slice(1, symbol.indexOf('/'))) % 4 === 0)).toBe(true);
  });
});

describe('psPickFinalists', () => {
  it('picks the best and the simplest strong variant', () => {
    const items = [
      { key: 'base', score: 1, valid: true, changed: 0, isBase: true },
      { key: 'best', score: 1.1, valid: true, changed: 9 },
      { key: 'second', score: 1.09, valid: true, changed: 8 },
      { key: 'simple', score: 1.06, valid: true, changed: 1 },
      { key: 'weak', score: 1.01, valid: true, changed: 1 },
      { key: 'invalid', score: 1.5, valid: false, changed: 1 },
    ];
    const { keys, improved } = psPickFinalists(items, 3, 1);
    expect(improved).toBe(true);
    expect(keys).toEqual(['best', 'simple', 'second']);
  });

  it('falls back to the best tries when nothing beats the original', () => {
    const { keys, improved } = psPickFinalists(
      [
        { key: 'a', score: 0.99, valid: true, changed: 1 },
        { key: 'b', score: 0.97, valid: true, changed: 1 },
      ],
      3,
      1,
    );
    expect(improved).toBe(false);
    expect(keys).toEqual(['a', 'b']);
  });
});

describe('psRunSearch', () => {
  const run = async (budget, cache = new Map(), calls = { count: 0 }, extra = {}) => {
    const { space, repair, baseGenome } = makeSpace();
    const objective = makeObjective(space);
    const events = [];
    const outcome = await psRunSearch({
      space,
      repair,
      keyOf,
      baseGenome,
      budget,
      rng: psCreateRng(42),
      evaluate: async (genomes) =>
        genomes.map((genome) => {
          const key = keyOf(genome);
          if (!cache.has(key)) {
            calls.count += 1;
            cache.set(key, { score: objective(genome), valid: true, violation: 0 });
          }
          return cache.get(key);
        }),
      onEvent: (event) => events.push(event),
      ...extra,
    });
    return { outcome, events, space, objective };
  };

  it('finds a clearly better point within a realistic budget', async () => {
    const calls = { count: 0 };
    const { outcome } = await run(110, new Map(), calls);
    expect(outcome.base.score).toBe(1);
    expect(outcome.best.score).toBeGreaterThan(1.12);
    expect(outcome.best.genome.tp).toBeGreaterThan(0.55);
    expect(outcome.best.genome.pullUp).toBeLessThan(0.4);
    expect(outcome.evaluations).toBeLessThanOrEqual(110);
    expect(calls.count).toBe(outcome.evaluations);
    expect(outcome.sensitivity.length).toBeGreaterThan(30);
    expect(outcome.rounds.length).toBeGreaterThanOrEqual(2);
  });

  it('replays an interrupted search for free and continues identically', async () => {
    const cache = new Map();
    const first = await run(110, cache);
    const calls = { count: 0 };
    const second = await run(110, cache, calls);
    expect(calls.count).toBe(0);
    expect(second.outcome.best.key).toBe(first.outcome.best.key);
    expect(second.outcome.path.map((point) => point.key)).toEqual(first.outcome.path.map((point) => point.key));
  });

  it('stops when asked and never exceeds the budget', async () => {
    let evaluations = 0;
    const { outcome } = await run(200, new Map(), { count: 0 }, {
      shouldStop: () => evaluations > 20,
      onRound: () => {},
      onEvent: (event) => {
        if (event.type === 'scanned') {
          evaluations += event.scan.length;
        }
      },
    });
    expect(outcome.stoppedBy).toBe('stop');
    const small = await run(15);
    expect(small.outcome.evaluations).toBeLessThanOrEqual(15);
  });

  it('keeps the original when every move is worse', async () => {
    const { space, repair, baseGenome } = makeSpace();
    const outcome = await psRunSearch({
      space,
      repair,
      keyOf,
      baseGenome,
      budget: 80,
      evaluate: async (genomes) =>
        genomes.map((genome) => {
          const distance = space.dimensions.reduce(
            (sum, dim) => sum + Math.abs((genome[dim.id] ?? dim.baseValue) - dim.baseValue) / (dim.max - dim.min),
            0,
          );
          return { score: 1 - distance, valid: true, violation: 0 };
        }),
    });
    expect(outcome.best.key).toBe(outcome.base.key);
    expect(outcome.path).toHaveLength(1);
    expect(outcome.stoppedBy).toBe('converged');
  });
});

describe('psDescribeMove', () => {
  it('lists the moved parameter first and counts re-balanced ones', async () => {
    const { psDescribeMove } = await import('../src/patternSearch.mjs');
    const { space, repair, baseGenome } = makeSpace();
    const move = psBuildMoves(space).find((item) => item.id === 'grid:volume:6:+');
    const next = psApplyMove(space, baseGenome, move, 0.125, repair);
    const text = psDescribeMove(space, baseGenome, next, move, (dim, value) => String(value));
    expect(text.startsWith('Ордер 6: объём 20→')).toBe(true);
    expect(text).toMatch(/пересчитано связанных: \d+/);
  });
});

describe('exhaustive grid', () => {
  /** Three numeric parameters and a filter swap: 8 × 5 × 4 × 3 = 480 combinations. */
  const smallSpace = () => ({
    dimensions: [
      { id: 'tp', kind: 'takeProfit', group: 'exit', label: 'Тейк, %', min: 0.4, max: 0.75, step: 0.05, baseValue: 0.5 },
      { id: 'pull', kind: 'pullUp', group: 'grid', label: 'Подтяжка, %', min: 0.2, max: 0.6, step: 0.1, baseValue: 0.4 },
      { id: 'n', kind: 'conditionInterval', group: 'timeframes', label: 'ТФ', min: 0, max: 3, step: 1, integer: true, baseValue: 1 },
      {
        id: 'sw',
        kind: 'conditionSwap',
        group: 'structure',
        label: 'Фильтр',
        min: 0,
        max: 2,
        step: 1,
        baseValue: 0,
        choices: [{ kind: 'set', label: 'RSI' }, { kind: 'set', label: 'CCI' }, { kind: 'off', label: 'выкл.' }],
      },
    ],
  });
  const key = (genome) => JSON.stringify(Object.keys(genome).sort().map((id) => [id, Math.round(genome[id] * 1e6) / 1e6]));
  const identity = (genome) => ({ ...genome });
  const base = (space) => Object.fromEntries(space.dimensions.map((dim) => [dim.id, dim.baseValue]));
  // A bumpy objective: the global optimum is not where a one-parameter-at-a-time search lands first.
  const objective = (g) => 1 + Math.sin(g.tp * 17) * 0.3 + Math.cos(g.pull * 9 + g.n) * 0.2 + (g.sw === 1 ? 0.15 : 0) - 0.05 * g.n * g.sw;
  const evaluator = (cache, calls) => async (genomes) =>
    genomes.map((genome) => {
      const k = key(genome);
      if (!cache.has(k)) {
        calls.count += 1;
        cache.set(k, { score: objective(genome), valid: true, violation: 0 });
      }
      return cache.get(k);
    });

  it('lists every value of a dimension, including an off-grid maximum', () => {
    expect(psGridValues({ min: 0.4, max: 0.75, step: 0.05 })).toEqual([0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]);
    expect(psGridValues({ min: 1, max: 2.1, step: 0.5 })).toEqual([1, 1.5, 2, 2.1]);
    expect(psGridValues({ min: 0, max: 3, step: 1, integer: true })).toEqual([0, 1, 2, 3]);
    expect(psGridValues({ kind: 'conditionSwap', choices: [{}, {}, {}], min: 0, max: 2, step: 1 })).toEqual([0, 1, 2]);
    expect(psGridValues({ min: 5, max: 5, step: 1 })).toEqual([5]);
    const space = smallSpace();
    expect(psGridSize(space).total).toBe(8 * 5 * 4 * 3);
    expect(psCertifyCost(space)).toBe(7 + 4 + 3 + 2);
  });

  it('visits a permutation of the grid', () => {
    for (const total of [1, 2, 7, 12, 480, 1001]) {
      const stride = psGridStride(total);
      const seen = new Set();
      for (let k = 0; k < total; k++) {
        seen.add((k * stride) % total);
      }
      expect(seen.size).toBe(total);
    }
  });

  it('checks every combination exactly once and returns the global best', async () => {
    const space = smallSpace();
    const calls = { count: 0 };
    const cache = new Map();
    const out = await psRunGrid({ space, repair: identity, keyOf: key, baseGenome: base(space), budget: 1000, batch: 7, evaluate: evaluator(cache, calls) });
    expect(out.stoppedBy).toBe('complete');
    expect(out.grid.visited).toBe(480);
    expect(calls.count).toBe(480); // the base genome is a grid point, so nothing extra
    let bestScore = -Infinity;
    for (const tp of psGridValues(space.dimensions[0]))
      for (const pull of psGridValues(space.dimensions[1]))
        for (const n of [0, 1, 2, 3])
          for (const sw of [0, 1, 2]) bestScore = Math.max(bestScore, objective({ tp, pull, n, sw }));
    expect(out.best.score).toBeCloseTo(bestScore, 12);
    // Resumed from the cache: nothing is evaluated again and the answer is the same.
    const again = { count: 0 };
    const second = await psRunGrid({ space, repair: identity, keyOf: key, baseGenome: base(space), budget: 1000, evaluate: evaluator(cache, again) });
    expect(again.count).toBe(0);
    expect(second.best.key).toBe(out.best.key);
  });

  it('evaluates genomes made equal by the repair once', async () => {
    const space = smallSpace();
    const calls = { count: 0 };
    // The filter swap does not matter when the timeframe index is 0: collapse it.
    const repair = (genome) => (genome.n === 0 ? { ...genome, sw: 0 } : { ...genome });
    const out = await psRunGrid({ space, repair, keyOf: key, baseGenome: base(space), budget: 1000, evaluate: evaluator(new Map(), calls) });
    expect(out.stoppedBy).toBe('complete');
    expect(calls.count).toBe(480 - 8 * 5 * 2);
    expect(out.grid.unique).toBe(calls.count);
  });

  it('does not start a grid above the limit and resumes after a stop', async () => {
    const space = smallSpace();
    const tooLarge = await psRunGrid({ space, repair: identity, keyOf: key, baseGenome: base(space), budget: 100, evaluate: evaluator(new Map(), { count: 0 }) });
    expect(tooLarge.stoppedBy).toBe('too-large');
    expect(tooLarge.evaluations).toBe(1);
    const cache = new Map();
    const first = { count: 0 };
    let batches = 0;
    const stopped = await psRunGrid({
      space,
      repair: identity,
      keyOf: key,
      baseGenome: base(space),
      budget: 1000,
      batch: 50,
      evaluate: evaluator(cache, first),
      shouldStop: () => batches >= 3,
      onEvent: (event) => {
        if (event.type === 'grid') {
          batches += 1;
        }
      },
    });
    expect(stopped.stoppedBy).toBe('stop');
    expect(stopped.grid.visited).toBe(150);
    const rest = { count: 0 };
    const resumed = await psRunGrid({ space, repair: identity, keyOf: key, baseGenome: base(space), budget: 1000, evaluate: evaluator(cache, rest) });
    expect(resumed.stoppedBy).toBe('complete');
    expect(first.count + rest.count).toBe(480);
  });

  it('certifies a point no single-parameter change improves', async () => {
    const space = smallSpace();
    const cache = new Map();
    const calls = { count: 0 };
    const evaluate = evaluator(cache, calls);
    const start = { ...(await evaluate([base(space)]))[0], key: key(base(space)), genome: base(space) };
    const out = await psCertify({ space, repair: identity, keyOf: key, start, evaluate, maxPasses: 6 });
    expect(out.certified).toBe(true);
    expect(out.best.score).toBeGreaterThan(start.score);
    for (const dim of space.dimensions) {
      for (const value of psGridValues(dim)) {
        const neighbour = { ...out.best.genome, [dim.id]: value };
        expect(objective(neighbour)).toBeLessThanOrEqual(out.best.score + 0.0025 + 1e-12);
      }
    }
    // Starting from the certified point it is certified again in one pass, all from the cache.
    const again = { count: 0 };
    const second = await psCertify({ space, repair: identity, keyOf: key, start: out.best, evaluate: evaluator(cache, again) });
    expect(second.certified).toBe(true);
    expect(second.passes).toBe(1);
    expect(again.count).toBe(0);
  });

  it('reports an unfinished certificate when the budget runs out', async () => {
    const space = smallSpace();
    const evaluate = evaluator(new Map(), { count: 0 });
    const start = { ...(await evaluate([base(space)]))[0], key: key(base(space)), genome: base(space) };
    const out = await psCertify({ space, repair: identity, keyOf: key, start, evaluate, budget: 5 });
    expect(out.certified).toBe(false);
    expect(out.stoppedBy).toBe('budget');
    expect(out.evaluations).toBeLessThanOrEqual(5);
  });
});
